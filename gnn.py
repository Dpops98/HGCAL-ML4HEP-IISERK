import torch
from torch_geometric.data import Dataset, Data
import numpy as np
import h5py
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class HGCalGraphDataset(Dataset):
    def __init__(self, h5_path, num_events=None, k=3, max_hits=1600, use_xyz=True):
        super().__init__()
        self.k = k
        self.max_hits = max_hits
        self.use_xyz = use_xyz  # if True: build graph in x-y-z space; else x-y
        self.data_list = []

        with h5py.File(h5_path, "r") as f:
            nhits = f["nhits"][:].astype(int)
            target = f["target"][:]
            rechit_energy = f["rechit_energy"][:]
            rechit_x = f["rechit_x"][:]
            rechit_y = f["rechit_y"][:]
            rechit_z = f["rechit_z"][:]

        if num_events is not None:
            nhits = nhits[:num_events]
            target = target[:num_events]

        rechit_event_indices = np.zeros(len(nhits) + 1, dtype=np.int64)
        rechit_event_indices[1:] = np.cumsum(nhits)

        for i in range(len(nhits)):
            start, end = rechit_event_indices[i], rechit_event_indices[i + 1]
            if nhits[i] < 2 or nhits[i] > max_hits:
                continue  # skip tiny or massive events

            e = rechit_energy[start:end]
            x = rechit_x[start:end]
            y = rechit_y[start:end]
            z = rechit_z[start:end]

            # node features
            x_node = np.stack([e, x, y, z], axis=1).astype(np.float32)

            # position space for KNN
            pos = np.stack([x, y, z], axis=1) if self.use_xyz else np.stack([x, y], axis=1)

            tree = cKDTree(pos)
            edge_index_list = []
            for idx in range(len(pos)):
                dists, neighbors = tree.query(pos[idx], k=min(self.k + 1, len(pos)))
                for n in neighbors[1:]:
                    edge_index_list.append([idx, n])

            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            x_tensor = torch.tensor(x_node, dtype=torch.float)
            y_tensor = torch.tensor([target[i]], dtype=torch.float)

            data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
            self.data_list.append(data)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
        

from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_mean_pool
from torch_geometric.nn.norm import BatchNorm

# Simplified smaller model
class SimpleEdgeNetRegression(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, k=3):  # Reduced hidden_dim from 64 to 32
        super().__init__()
        self.k = k

        # Single EdgeConv layer instead of 3
        self.conv1 = EdgeConv(nn=nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))

        self.bn1 = BatchNorm(hidden_dim)

        # Simpler regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 16),  # Reduced from 64 to 16
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        # Pooling over all nodes in a graph to get one vector per graph
        x = global_mean_pool(x, batch)

        # Regress scalar target energy
        out = self.regressor(x)
        return out.view(-1)

# Test with much smaller dataset
print("Loading dataset...")
data_set = HGCalGraphDataset("/data/docs/deba.s/data/hgcal_electron_data_0001.h5", 
                           num_events=600000, k=3)  # Reduced from 100000 to 1000
print(f"Dataset loaded: {len(data_set)} events")

train_len = int(len(data_set)*0.7)    
from torch.utils.data import random_split  
train_set, test_set = random_split(data_set, [train_len, len(data_set)-train_len])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)  # Reduced batch size
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

print(f"Train set: {len(train_set)} events")
print(f"Test set: {len(test_set)} events")

# Smaller model
model = SimpleEdgeNetRegression(input_dim=4, hidden_dim=32, k=3)

# Model Summary Functions
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model):
    print("=" * 60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)
    
    total_params = 0
    trainable_params = 0
    
    print(f"{'Layer':<25} {'Parameters':<15} {'Shape':<20}")
    print("-" * 60)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        
        print(f"{name:<25} {param_count:<15} {str(list(param.shape)):<20}")
    
    print("-" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 60)
    
    return total_params, trainable_params

def detailed_model_structure(model):
    print("\nDETAILED MODEL STRUCTURE:")
    print("=" * 60)
    
    def print_layer_info(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            print(f"{full_name}:")
            
            if hasattr(child, 'weight') and child.weight is not None:
                print(f"  Weight: {child.weight.shape}")
            if hasattr(child, 'bias') and child.bias is not None:
                print(f"  Bias: {child.bias.shape}")
            
            if len(list(child.children())) > 0:
                print_layer_info(child, full_name)
            else:
                print(f"  Type: {type(child).__name__}")
    
    print_layer_info(model)
    print("=" * 60)

# Display model information
print("\n" + "="*60)
print("MODEL INFORMATION")
print("="*60)

print(f"Model: {model.__class__.__name__}")
print(f"Input dimension: 4 (energy, x, y, z)")
print(f"Hidden dimension: 32")
print(f"K-nearest neighbors: 3")

# Show model architecture
model_summary(model)

# Show detailed structure
detailed_model_structure(model)

# Alternative: Print the model directly
print("\nMODEL STRUCTURE (PyTorch default):")
print("-" * 40)
print(model)
print("-" * 40)

# Loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

def evaluate(model, dataloader, device):
    model.eval()
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            pred = model(batch)
            y_pred_all.extend(pred.cpu().numpy())
            y_true_all.extend(batch.y.view(-1).cpu().numpy())

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    return y_true_all, y_pred_all

# Quick training loop - just 2 epochs for testing
print("Starting training...")
best_loss = float("inf")
for epoch in range(50):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        pred = model(batch)
        loss = loss_fn(pred, batch.y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_gnn_model.pt")

model.load_state_dict(torch.load("best_gnn_model.pt"))
y_true, y_pred = evaluate(model, test_loader, device)

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
mre = np.mean(np.abs((y_pred - y_true) / y_true))

print(f"MAE: {mae:.4f} GeV")
print(f"MSE: {mse:.4f} GeV²")
print(f"MRE: {mre:.4f}")

ratio = y_pred / y_true
bins = np.linspace(min(y_true), max(y_true), 20)
bin_centers = 0.5 * (bins[1:] + bins[:-1])

response_mean = []
resolution = []

for i in range(len(bins)-1):
    idx = (y_true >= bins[i]) & (y_true < bins[i+1])
    if np.sum(idx) < 5:
        response_mean.append(np.nan)
        resolution.append(np.nan)
        continue
    R = y_pred[idx] / y_true[idx]
    response_mean.append(np.mean(R))
    resolution.append(np.std(R) / np.mean(R))  # σ_R / <R>

plt.figure()
plt.plot(bin_centers, np.array(response_mean) * bin_centers, marker='o', label='⟨R⟩ × E_true')
plt.xlabel("E_true [GeV]")
plt.ylabel("⟨R⟩ × E_true [GeV]")
plt.title("Response vs E_true")
plt.grid(True)
plt.legend()
plt.savefig("response_vs_Etrue.png")
#plt.show()

plt.figure()
plt.plot(bin_centers, resolution, marker='o', label='σ_R / ⟨R⟩')
plt.xlabel("E_true [GeV]")
plt.ylabel("σ_R / ⟨R⟩")
plt.title("Resolution vs E_true")
plt.grid(True)
plt.legend()
plt.savefig("resolution_vs_Etrue.png")
#plt.show()

def resolution_model(E, s, c):
    return np.sqrt((s / np.sqrt(E))**2 + c**2)

# Remove NaNs
valid = ~np.isnan(bin_centers) & ~np.isnan(resolution)
popt, pcov = curve_fit(resolution_model, bin_centers[valid], np.array(resolution)[valid], p0=[0.2, 0.01])

s, c = popt
print(f"Stochastic term (s): {s:.4f}")
print(f"Constant term (c): {c:.4f}")

plt.figure()
plt.plot(bin_centers, resolution, 'o', label='Data')
plt.plot(bin_centers, resolution_model(bin_centers, *popt), label=f'Fit: s={s:.3f}, c={c:.3f}')
plt.xlabel("E_true [GeV]")
plt.ylabel("σ_R / ⟨R⟩")
plt.title("Resolution Fit")
plt.grid(True)
plt.legend()
plt.savefig("resolution_fit.png")
#plt.show()

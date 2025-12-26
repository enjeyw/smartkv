import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file, save_file
import glob
import os
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import time

class BatchedKeyDataset(Dataset):
    def __init__(self, data_dir, layer_idx=None):
        """
        Loads data for a specific layer's heads.
        """
        self.inputs = [] # List of [N, Dim] tensors per head
        self.targets = [] # List of [N] tensors per head
        self.head_ids = []
        
        # Find directories for this layer
        if layer_idx is not None:
            pattern = os.path.join(data_dir, f"l{layer_idx}_h*")
        else:
            raise ValueError("layer_idx must be provided")
            
        head_dirs = sorted(glob.glob(pattern))
        if not head_dirs:
            raise ValueError(f"No data found matching {pattern}")
            
        print(f"Loading data for {len(head_dirs)} heads (Layer {layer_idx})...")
        
        first_head_shards = sorted(glob.glob(os.path.join(head_dirs[0], "*.safetensors")))
        num_shards = len(first_head_shards)
        
        # Pre-allocate lists for each head
        head_data_inputs = [[] for _ in head_dirs]
        head_data_targets = [[] for _ in head_dirs]
        
        for shard_idx in range(num_shards):
            # Load shard_i for all heads
            for h_idx, h_dir in enumerate(head_dirs):
                # Using glob to be safe about shard naming
                shards = sorted(glob.glob(os.path.join(h_dir, "*.safetensors")))
                current_shard = shards[shard_idx]
                
                data = load_file(current_shard)
                head_data_inputs[h_idx].append(data["inputs"].float())
                head_data_targets[h_idx].append(data["targets"].float())
                
                if shard_idx == 0:
                    self.head_ids.append(os.path.basename(h_dir))

        # Concatenate shards per head
        full_inputs_per_head = [torch.cat(inputs) for inputs in head_data_inputs] # List of [N, D]
        full_targets_per_head = [torch.cat(targets) for targets in head_data_targets] # List of [N]
        
        # Stack heads: [N, Num_Heads, D]
        self.inputs = torch.stack(full_inputs_per_head, dim=1)
        self.targets = torch.stack(full_targets_per_head, dim=1).unsqueeze(-1) # [N, Num_Heads, 1]
        
        print(f"Dataset shape: {self.inputs.shape}")
        
    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class BatchedGateMLP(nn.Module):
    def __init__(self, num_heads, input_dim, hidden_dim=256):
        super().__init__()
        self.num_heads = num_heads
        
        # Layer 1: [Num_Heads, Input_Dim, Hidden_Dim]
        self.W1 = nn.Parameter(torch.randn(num_heads, input_dim, hidden_dim) / np.sqrt(input_dim))
        self.b1 = nn.Parameter(torch.zeros(num_heads, hidden_dim))
        
        # Layer 2: [Num_Heads, Hidden_Dim, Hidden_Dim]
        self.W2 = nn.Parameter(torch.randn(num_heads, hidden_dim, hidden_dim) / np.sqrt(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(num_heads, hidden_dim))
        
        # Layer 3: [Num_Heads, Hidden_Dim, Hidden_Dim]
        self.W3 = nn.Parameter(torch.randn(num_heads, hidden_dim, hidden_dim) / np.sqrt(hidden_dim))
        self.b3 = nn.Parameter(torch.zeros(num_heads, hidden_dim))
        
        # Layer 4: [Num_Heads, Hidden_Dim, 1]
        self.W4 = nn.Parameter(torch.randn(num_heads, hidden_dim, 1) / np.sqrt(hidden_dim))
        self.b4 = nn.Parameter(torch.zeros(num_heads, 1))
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [Batch, Num_Heads, Input_Dim]
        
        # Transpose for bmm: [Num_Heads, Batch, Input_Dim]
        x_t = x.transpose(0, 1) 
        
        # Layer 1
        h = torch.bmm(x_t, self.W1) + self.b1.unsqueeze(1)
        h = self.relu(h)
        
        # Layer 2
        h = torch.bmm(h, self.W2) + self.b2.unsqueeze(1)
        h = self.relu(h)
        
        # Layer 3
        h = torch.bmm(h, self.W3) + self.b3.unsqueeze(1)
        h = self.relu(h)
        
        # Layer 4
        out = torch.bmm(h, self.W4) + self.b4.unsqueeze(1)
        
        # Return: [Batch, Num_Heads, 1]
        return out.transpose(0, 1)

def train_layer(args, layer_idx):
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    try:
        dataset = BatchedKeyDataset(args.data_dir, layer_idx=layer_idx)
    except ValueError as e:
        print(f"Skipping Layer {layer_idx}: {e}")
        return

    # Split Train/Val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    num_heads = dataset.inputs.shape[1]
    input_dim = dataset.inputs.shape[2]
    model = BatchedGateMLP(num_heads, input_dim).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print(f"Training Layer {layer_idx} ({num_heads} heads)...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            pred = model(x) # [B, H, 1]
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
        
    # Validation
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
            
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    avg_r2 = 0
    for h in range(num_heads):
        r2 = r2_score(all_targets[:, h, 0], all_preds[:, h, 0])
        avg_r2 += r2
    avg_r2 /= num_heads
    print(f"  Layer {layer_idx} Avg R²: {avg_r2:.4f}")

    # Save ONE model file for the whole layer
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"layer_{layer_idx}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"  Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/train_qwen3_regression")
    parser.add_argument("--output_dir", type=str, default="models/gates_qwen3_regression")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()
    
    # Train all 28 layers
    for i in range(28):
        train_layer(args, i)


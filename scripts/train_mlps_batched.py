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
        Loads data for multiple heads.
        If layer_idx is provided, loads only heads for that layer.
        Otherwise loads ALL heads found.
        
        Assumption: All heads have aligned samples (generated from same text in same order).
        We verified generate_dataset.py produces synchronized shards.
        """
        self.inputs = [] # List of [N, Dim] tensors per head
        self.targets = [] # List of [N] tensors per head
        self.head_ids = []
        
        # Find directories
        if layer_idx is not None:
            pattern = os.path.join(data_dir, f"l{layer_idx}_h*")
        else:
            pattern = os.path.join(data_dir, "l*_h*")
            
        head_dirs = sorted(glob.glob(pattern))
        if not head_dirs:
            raise ValueError(f"No data found matching {pattern}")
            
        print(f"Loading data for {len(head_dirs)} heads...")
        
        # Check consistency
        # We assume all heads have same number of shards and samples
        # We load them all into memory.
        
        # Optimization: Process one shard at a time across all heads to avoid huge peaks?
        # Or just load all. H100 has 80GB VRAM, CPU RAM is likely > 100GB.
        # 200 samples * 1024 tokens * 224 heads * 128 dim * 4 bytes = ~22 GB.
        # It fits in CPU RAM easily.
        
        # Structure: final inputs should be [Total_Samples, Num_Heads, Dim]
        
        first_head_shards = sorted(glob.glob(os.path.join(head_dirs[0], "*.safetensors")))
        num_shards = len(first_head_shards)
        
        # Pre-allocate lists for each head
        head_data_inputs = [[] for _ in head_dirs]
        head_data_targets = [[] for _ in head_dirs]
        
        for shard_idx in range(num_shards):
            # Load shard_i for all heads
            for h_idx, h_dir in enumerate(head_dirs):
                shard_path = os.path.join(h_dir, f"shard_{100 * (shard_idx+1)}.safetensors") 
                # Note: shard naming might vary, let's rely on sorted list
                # Actually, generate_dataset uses specific step numbers.
                # Let's just glob and sort per head.
                shards = sorted(glob.glob(os.path.join(h_dir, "*.safetensors")))
                current_shard = shards[shard_idx]
                
                data = load_file(current_shard)
                head_data_inputs[h_idx].append(data["inputs"].float())
                head_data_targets[h_idx].append(data["targets"].float())
                
                if shard_idx == 0:
                    self.head_ids.append(os.path.basename(h_dir))

        # Concatenate shards per head
        print("Concatenating shards...")
        full_inputs_per_head = [torch.cat(inputs) for inputs in head_data_inputs] # List of [N, D]
        full_targets_per_head = [torch.cat(targets) for targets in head_data_targets] # List of [N]
        
        # Stack heads: [N, Num_Heads, D]
        print("Stacking heads...")
        self.inputs = torch.stack(full_inputs_per_head, dim=1)
        self.targets = torch.stack(full_targets_per_head, dim=1).unsqueeze(-1) # [N, Num_Heads, 1]
        
        print(f"Dataset shape: {self.inputs.shape}")
        print(f"Targets shape: {self.targets.shape}")
        
    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class BatchedGateMLP(nn.Module):
    def __init__(self, num_heads, input_dim, hidden_dim=64):
        super().__init__()
        self.num_heads = num_heads
        
        # Weights: [Num_Heads, Input_Dim, Hidden_Dim]
        self.W1 = nn.Parameter(torch.randn(num_heads, input_dim, hidden_dim) / np.sqrt(input_dim))
        self.b1 = nn.Parameter(torch.zeros(num_heads, hidden_dim))
        
        # Weights: [Num_Heads, Hidden_Dim, 1]
        self.W2 = nn.Parameter(torch.randn(num_heads, hidden_dim, 1) / np.sqrt(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(num_heads, 1))
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [Batch, Num_Heads, Input_Dim]
        
        # Transpose for bmm: [Num_Heads, Batch, Input_Dim]
        x_t = x.transpose(0, 1) 
        
        # Layer 1: [H, B, I] @ [H, I, Hidden] -> [H, B, Hidden]
        h = torch.bmm(x_t, self.W1) + self.b1.unsqueeze(1)
        h = self.relu(h)
        
        # Layer 2: [H, B, Hidden] @ [H, Hidden, 1] -> [H, B, 1]
        out = torch.bmm(h, self.W2) + self.b2.unsqueeze(1)
        
        # Return: [Batch, Num_Heads, 1]
        return out.transpose(0, 1)

def train_batched(args):
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load Data
    print("Loading dataset into RAM...")
    # If loading all is too much, filter by layer. For now, try all.
    try:
        dataset = BatchedKeyDataset(args.data_dir)
    except MemoryError:
        print("OOM loading all data. Switching to Per-Layer training.")
        # Fallback not implemented yet in this snippet, assume it fits
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
    
    print(f"\nStarting Batched Training for {num_heads} heads...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        # Progress bar for batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            pred = model(x) # [B, H, 1]
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
        
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds.")
    
    # Validation & R2 Per Head
    print("Evaluating...")
    model.eval()
    
    # Accumulators for all validation data
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
            
    all_preds = torch.cat(all_preds, dim=0).numpy() # [N, H, 1]
    all_targets = torch.cat(all_targets, dim=0).numpy() # [N, H, 1]
    
    # Calculate R2 per head
    print("\nTop 10 Best Calibrated Heads:")
    r2_scores = []
    for h_idx in range(num_heads):
        head_name = dataset.head_ids[h_idx]
        p = all_preds[:, h_idx, 0]
        t = all_targets[:, h_idx, 0]
        score = r2_score(t, p)
        r2_scores.append((head_name, score))
        
    # Sort by R2
    r2_scores.sort(key=lambda x: x[1], reverse=True)
    
    for name, score in r2_scores[:10]:
        print(f"{name}: R² = {score:.4f}")
        
    # Save Per-Head Checkpoints (for compatibility with inference script)
    print(f"\nSaving {num_heads} individual models to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    state_dict = model.state_dict()
    # Batched weights: W1 is [H, I, Hidden]
    W1 = state_dict["W1"].cpu()
    b1 = state_dict["b1"].cpu()
    W2 = state_dict["W2"].cpu()
    b2 = state_dict["b2"].cpu()
    
    for h_idx in range(num_heads):
        head_name = dataset.head_ids[h_idx]
        # Create individual state dict
        single_sd = {
            "net.0.weight": W1[h_idx].transpose(0, 1), # Linear weight is [Out, In]
            "net.0.bias": b1[h_idx],
            "net.2.weight": W2[h_idx].transpose(0, 1),
            "net.2.bias": b2[h_idx]
        }
        
        torch.save(single_sd, os.path.join(args.output_dir, f"{head_name}.pt"))
        
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/train_qwen3_t05")
    parser.add_argument("--output_dir", type=str, default="models/gates_qwen3_batched")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8192) # Larger batch size for efficiency
    args = parser.parse_args()
    
    train_batched(args)


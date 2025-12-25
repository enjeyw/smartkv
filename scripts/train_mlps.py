import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file, save_file
import glob
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

class KeyDataset(Dataset):
    def __init__(self, shard_paths):
        self.inputs = []
        self.targets = []
        
        print(f"Loading {len(shard_paths)} shards...")
        for path in tqdm(shard_paths):
            data = load_file(path)
            self.inputs.append(data["inputs"].float()) # Convert Half -> Float
            self.targets.append(data["targets"].float()) # [N]
            
        self.inputs = torch.cat(self.inputs)
        self.targets = torch.cat(self.targets).unsqueeze(1) # [Total, 1]
        
        print(f"Total samples: {len(self.inputs)}")
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class GateMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )
        
    def forward(self, x):
        return self.net(x)

def train_head(head_dir, output_dir, epochs=3, batch_size=4096, lr=1e-3, plot=False):
    head_id = os.path.basename(head_dir)
    print(f"Training {head_id}...")
    
    shards = glob.glob(os.path.join(head_dir, "*.safetensors"))
    if not shards:
        print(f"No data for {head_id}, skipping.")
        return

    dataset = KeyDataset(shards)
    # Split into train/val for R2 calculation (simple 90/10 split)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = dataset.inputs.shape[1]
    model = GateMLP(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
    # Evaluation & R2
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    r2 = r2_score(all_targets, all_preds)
    print(f"Validation R²: {r2:.4f}")

    # Save Model
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{head_id}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved to {save_path}")
    
    # Plotting
    if plot:
        print(f"Generating plot for {head_id}...")
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(8, 8))
        plt.scatter(all_targets, all_preds, alpha=0.1, s=1)
        
        # Ideal line
        min_val = min(all_targets.min(), all_preds.min())
        max_val = max(all_targets.max(), all_preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel("Actual Log-Lifespan")
        plt.ylabel("Predicted Log-Lifespan")
        plt.title(f"Calibration: {head_id} (R²={r2:.3f})")
        plt.savefig(f"plots/calibration_{head_id}.png")
        plt.close()

def train_all(args):
    # Find all head directories
    head_dirs = glob.glob(os.path.join(args.data_dir, "l0_h*")) # FOCUS ON LAYER 0 ONLY
    head_dirs = sorted(head_dirs)
    
    print(f"Found {len(head_dirs)} heads to train (Filtered for Layer 0).")
    
    for head_dir in head_dirs:
        head_id = os.path.basename(head_dir)
        should_plot = True 
                 
        train_head(head_dir, args.output_dir, args.epochs, args.batch_size, plot=should_plot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/train_qwen3_t05")
    parser.add_argument("--output_dir", type=str, default="models/gates_qwen3_t05")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()
    
    train_all(args)

import torch
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import load_file
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

# Reuse dataset logic
class BatchedKeyDataset(Dataset):
    def __init__(self, data_dir):
        # Just load everything (reused from train_mlps_batched.py)
        # Simplified for visualization script
        pattern = os.path.join(data_dir, "l*_h*")
        head_dirs = sorted(glob.glob(pattern))
        
        if not head_dirs:
            raise ValueError(f"No data in {data_dir}")
            
        print(f"Loading data for {len(head_dirs)} heads...")
        
        first_head_shards = sorted(glob.glob(os.path.join(head_dirs[0], "*.safetensors")))
        num_shards = len(first_head_shards)
        
        head_data_inputs = [[] for _ in head_dirs]
        head_data_targets = [[] for _ in head_dirs]
        self.head_ids = []
        
        for shard_idx in range(num_shards):
            for h_idx, h_dir in enumerate(head_dirs):
                if shard_idx == 0: self.head_ids.append(os.path.basename(h_dir))
                
                shards = sorted(glob.glob(os.path.join(h_dir, "*.safetensors")))
                current_shard = shards[shard_idx]
                data = load_file(current_shard)
                head_data_inputs[h_idx].append(data["inputs"].float())
                head_data_targets[h_idx].append(data["targets"].float())

        full_inputs_per_head = [torch.cat(inputs) for inputs in head_data_inputs]
        full_targets_per_head = [torch.cat(targets) for targets in head_data_targets]
        
        self.inputs = torch.stack(full_inputs_per_head, dim=1) # [N, H, D]
        self.targets = torch.stack(full_targets_per_head, dim=1).unsqueeze(-1) # [N, H, 1]
        
    def __len__(self): return self.inputs.shape[0]
    def __getitem__(self, idx): return self.inputs[idx], self.targets[idx]

# Model Wrapper
class GateMLP(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

def visualize_all(data_dir="data/train_qwen3_t05", model_dir="models/gates_qwen3_batched"):
    dataset = BatchedKeyDataset(data_dir)
    
    # Use only a subset for visualization to save time/RAM
    subset_size = min(5000, len(dataset))
    indices = torch.randperm(len(dataset))[:subset_size]
    inputs = dataset.inputs[indices]   # [N, H, D]
    targets = dataset.targets[indices] # [N, H, 1]
    
    print(f"Visualizing {subset_size} samples...")
    
    # Load all models
    models = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for h_id in dataset.head_ids:
        path = os.path.join(model_dir, f"{h_id}.pt")
        model = GateMLP(input_dim=inputs.shape[2]).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
        
    # Run Inference per head
    inputs = inputs.to(device)
    preds = []
    
    with torch.no_grad():
        for i, model in enumerate(tqdm(models, desc="Predicting")):
            head_input = inputs[:, i, :] # [N, D]
            head_pred = model(head_input) # [N, 1]
            preds.append(head_pred.cpu())
            
    preds = torch.stack(preds, dim=1) # [N, H, 1]
    targets = targets.cpu()
    
    # Plotting
    # We want a grid. Qwen3 has 28 layers, 8 heads.
    # 28 rows, 8 columns.
    
    num_layers = 28
    num_heads = 8
    
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(24, 80))
    fig.suptitle("Calibration: Predicted vs Actual Log-Lifespan", fontsize=20, y=1.005)
    
    # Map head_ids (lX_hY) to grid
    head_map = {hid: i for i, hid in enumerate(dataset.head_ids)}
    
    for l in range(num_layers):
        for h in range(num_heads):
            hid = f"l{l}_h{h}"
            ax = axes[l, h]
            
            if hid in head_map:
                idx = head_map[hid]
                p = preds[:, idx, 0].numpy()
                t = targets[:, idx, 0].numpy()
                
                ax.scatter(t, p, alpha=0.05, s=1, c='blue')
                
                # Ideal line
                min_v, max_v = 0, 6 # Log4 space 0-6
                ax.plot([min_v, max_v], [min_v, max_v], 'r--', alpha=0.5)
                
                ax.set_xlim(min_v, max_v)
                ax.set_ylim(min_v, max_v)
                
                if l == num_layers - 1: ax.set_xlabel("Actual")
                if h == 0: ax.set_ylabel("Pred")
                
                ax.set_title(hid, fontsize=8)
                ax.tick_params(labelsize=6)
            else:
                ax.axis('off')
                
    plt.tight_layout()
    os.makedirs("plots/grid", exist_ok=True)
    plt.savefig("plots/grid/all_heads_calibration.png", dpi=150, bbox_inches='tight')
    print("Saved plots/grid/all_heads_calibration.png")

if __name__ == "__main__":
    visualize_all()


import torch
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import load_file
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from binary_gate_io import GateLoader

# Reuse dataset logic
class BatchedKeyDataset(Dataset):
    def __init__(self, data_dir):
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

class BinaryGateMLP(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # V3 Architecture (256 hidden, 2 layers)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
    def forward(self, x): return self.net(x)

def visualize_binary_grid(data_dir="data/train_binary_256", model_dir="models/gates_binary_256_v3"):
    dataset = BatchedKeyDataset(data_dir)
    loader = GateLoader(model_dir)
    
    # Use subset
    subset_size = min(5000, len(dataset))
    indices = torch.randperm(len(dataset))[:subset_size]
    inputs = dataset.inputs[indices]
    targets = dataset.targets[indices]
    
    print(f"Visualizing {subset_size} samples...")
    
    models = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for h_id in dataset.head_ids:
        sd = loader.get_legacy_state_dict(h_id, device=device)
        if sd is None:
            print(f"Warning: {h_id} not found, using dummy")
            models.append(None)
            continue
        
        # Fix keys
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("net."):
                new_sd[k.replace("net.", "")] = v
            else:
                new_sd[k] = v
                
        model = BinaryGateMLP(input_dim=inputs.shape[2]).to(device)
        model.net.load_state_dict(new_sd)
        model.eval()
        models.append(model)
        
    inputs = inputs.to(device)
    probs = []
    
    with torch.no_grad():
        for i, model in enumerate(tqdm(models, desc="Predicting")):
            if model is None:
                probs.append(torch.zeros(inputs.shape[0], 1))
            else:
                head_input = inputs[:, i, :]
                logits = model(head_input)
                p = torch.sigmoid(logits) # Convert to probability
                probs.append(p.cpu())
            
    probs = torch.stack(probs, dim=1) # [N, H, 1]
    targets = targets.cpu()
    
    # Plotting Grid
    num_layers = 28
    num_heads = 8
    
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(24, 80))
    fig.suptitle("Binary Calibration: Predicted Probability vs Actual Outcome (Red=Evicted/Short, Blue=Keep/Long)", fontsize=20, y=1.005)
    
    head_map = {hid: i for i, hid in enumerate(dataset.head_ids)}
    
    for l in range(num_layers):
        for h in range(num_heads):
            hid = f"l{l}_h{h}"
            ax = axes[l, h]
            
            if hid in head_map:
                idx = head_map[hid]
                p = probs[:, idx, 0].numpy()
                t = targets[:, idx, 0].numpy()
                
                # Jitter for visualization
                jitter = np.random.normal(0, 0.02, size=p.shape)
                
                # Color by ground truth: 0 (Evict) = Red, 1 (Keep) = Blue
                colors = ['red' if x == 0 else 'blue' for x in t]
                
                ax.scatter(p, t + jitter, alpha=0.1, s=2, c=colors)
                
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.1, 1.1)
                
                if l == num_layers - 1: ax.set_xlabel("Pred Prob")
                if h == 0: ax.set_ylabel("Actual")
                
                ax.set_title(hid, fontsize=8)
                ax.tick_params(labelsize=6)
                
                # Add text for positive rate
                pos_rate = t.mean()
                ax.text(0.05, 0.9, f"Pos: {pos_rate:.3f}", transform=ax.transAxes, fontsize=6)
            else:
                ax.axis('off')
                
    plt.tight_layout()
    os.makedirs("plots/grid_binary", exist_ok=True)
    plt.savefig("plots/grid_binary/all_heads_binary_calibration.png", dpi=150, bbox_inches='tight')
    print("Saved plots/grid_binary/all_heads_binary_calibration.png")

if __name__ == "__main__":
    visualize_binary_grid()


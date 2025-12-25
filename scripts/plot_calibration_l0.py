import torch
import torch.nn as nn
from torch.utils.data import Dataset
from safetensors.torch import load_file
import glob
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse

from binary_gate_io import GateLoader

class BatchedKeyDataset(Dataset):
    def __init__(self, data_dir, layer=0):
        pattern = os.path.join(data_dir, f"l{layer}_h*")
        head_dirs = sorted(glob.glob(pattern))
        
        if not head_dirs:
            raise ValueError(f"No data for layer {layer} in {data_dir}")
            
        print(f"Loading data for {len(head_dirs)} heads...")
        
        head_data_inputs = [[] for _ in head_dirs]
        head_data_targets = [[] for _ in head_dirs]
        self.head_ids = []
        
        for h_idx, h_dir in enumerate(head_dirs):
            self.head_ids.append(os.path.basename(h_dir))
            shards = sorted(glob.glob(os.path.join(h_dir, "*.safetensors")))
            
            # Load subset for speed if needed, or all
            for shard in shards[:1]: # Loading just one shard for visualization is enough
                data = load_file(shard)
                head_data_inputs[h_idx].append(data["inputs"].float())
                head_data_targets[h_idx].append(data["targets"].float())

        full_inputs_per_head = [torch.cat(inputs) for inputs in head_data_inputs]
        full_targets_per_head = [torch.cat(targets) for targets in head_data_targets]
        
        self.inputs = torch.stack(full_inputs_per_head, dim=1)
        self.targets = torch.stack(full_targets_per_head, dim=1).unsqueeze(-1) 
        
    def __len__(self): return self.inputs.shape[0]

def plot_calibration(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = BatchedKeyDataset(args.data_dir, layer=args.layer)
    loader = GateLoader(args.model_dir)
    
    # Random subset
    subset_size = min(5000, len(dataset))
    indices = torch.randperm(len(dataset))[:subset_size]
    inputs = dataset.inputs[indices].to(device)
    targets = dataset.targets[indices]
    
    models = []
    
    for h_id in dataset.head_ids:
        sd = loader.get_legacy_state_dict(h_id, device=device)
        if sd is None:
            print(f"Warning: checkpoint missing for {h_id}")
            models.append(None)
            continue
        
        # Load V3 Model (Sequential)
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("net."):
                new_sd[k.replace("net.", "")] = v
            else:
                new_sd[k] = v
                
        model = nn.Sequential(
             nn.Linear(128, 256),
             nn.ReLU(),
             nn.Linear(256, 256),
             nn.ReLU(),
             nn.Linear(256, 1)
         ).to(device)
        model.load_state_dict(new_sd)
        model.eval()
        models.append(model)
        
    probs = []
    with torch.no_grad():
        for i, model in enumerate(models):
            if model is None:
                probs.append(torch.zeros(inputs.shape[0], 1))
                continue
            logits = model(inputs[:, i, :])
            p = torch.sigmoid(logits)
            probs.append(p.cpu())
            
    probs = torch.stack(probs, dim=1) # [N, H, 1]
    
    # Plotting
    num_heads = len(models)
    cols = 4
    rows = (num_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    fig.suptitle(f"Layer {args.layer} Binary Calibration (Red=Evicted, Blue=Keep)", fontsize=16)
    
    axes = axes.flatten()
    
    for i in range(len(axes)):
        ax = axes[i]
        if i < num_heads:
            h_id = dataset.head_ids[i]
            p = probs[:, i, 0].numpy()
            t = targets[:, i, 0].numpy()
            
            jitter = np.random.normal(0, 0.02, size=p.shape)
            colors = ['red' if x == 0 else 'blue' for x in t]
            
            ax.scatter(p, t + jitter, alpha=0.1, s=2, c=colors)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_title(h_id)
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Actual Label")
            
            # Accuracy / Pos Rate
            pos_rate = t.mean()
            ax.text(0.05, 0.9, f"Pos Rate: {pos_rate:.3f}", transform=ax.transAxes)
        else:
            ax.axis('off')
            
    plt.tight_layout()
    os.makedirs("plots/grid_binary", exist_ok=True)
    out_path = f"plots/grid_binary/layer{args.layer}_calibration.png"
    plt.savefig(out_path)
    print(f"Saved calibration plot to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="data/train_binary_256")
    parser.add_argument("--model_dir", type=str, default="models/gates_binary_256_v3")
    args = parser.parse_args()
    
    plot_calibration(args)


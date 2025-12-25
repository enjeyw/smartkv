import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import argparse

from binary_gate_io import GateLoader

class BatchedKeyDataset(Dataset):
    def __init__(self, data_dir, layer=0):
        pattern = os.path.join(data_dir, f"l{layer}_h*")
        head_dirs = sorted(glob.glob(pattern))
        
        if not head_dirs:
            raise ValueError(f"No data for layer {layer} in {data_dir}")
            
        print(f"Loading data for {len(head_dirs)} heads in Layer {layer}...")
        
        # Load one shard is enough for eval visualization
        head_data_inputs = [[] for _ in head_dirs]
        head_data_targets = [[] for _ in head_dirs]
        self.head_ids = []
        
        for h_idx, h_dir in enumerate(head_dirs):
            self.head_ids.append(os.path.basename(h_dir))
            shards = sorted(glob.glob(os.path.join(h_dir, "*.safetensors")))
            
            for shard in shards[:1]: 
                data = load_file(shard)
                head_data_inputs[h_idx].append(data["inputs"].float())
                head_data_targets[h_idx].append(data["targets"].float())

        full_inputs_per_head = [torch.cat(inputs) for inputs in head_data_inputs]
        full_targets_per_head = [torch.cat(targets) for targets in head_data_targets]
        
        self.inputs = torch.stack(full_inputs_per_head, dim=1)
        self.targets = torch.stack(full_targets_per_head, dim=1).unsqueeze(-1) 
        
    def __len__(self): return self.inputs.shape[0]

def plot_confusion_matrices(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = BatchedKeyDataset(args.data_dir, layer=args.layer)
    inputs = dataset.inputs.to(device)
    targets = dataset.targets
    loader = GateLoader(args.model_dir)
    
    # Setup plot grid
    num_heads = len(dataset.head_ids)
    cols = 4
    rows = (num_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    fig.suptitle(f"Confusion Matrices - Layer {args.layer} (Threshold={args.threshold})", fontsize=16)
    axes = axes.flatten()
    
    for i, h_id in enumerate(dataset.head_ids):
        print(f"Evaluating {h_id}...")
        sd = loader.get_legacy_state_dict(h_id, device=device)
        if sd is None:
            print(f"Model not found for {h_id}")
            continue
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
        
        # Inference
        head_inputs = inputs[:, i, :]
        with torch.no_grad():
            logits = model(head_inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
        y_true = targets[:, i, 0].numpy().flatten()
        y_pred = (probs > args.threshold).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        # Add Normalized values as small text
        # (Optional, but 'd' is integer count which is good for absolute numbers)
        
        ax.set_title(f"{h_id}\nAcc: {(y_true==y_pred).mean():.3f}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticklabels(['Drop', 'Keep'])
        ax.set_yticklabels(['Drop', 'Keep'])
        
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    os.makedirs("plots/confusion_matrices", exist_ok=True)
    out_path = f"plots/confusion_matrices/layer{args.layer}_cm_t{args.threshold}.png"
    plt.savefig(out_path)
    print(f"\nSaved Confusion Matrices to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0, help="Layer to evaluate")
    parser.add_argument("--data_dir", type=str, default="data/train_binary_256")
    parser.add_argument("--model_dir", type=str, default="models/gates_binary_256_v3")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    args = parser.parse_args()
    
    plot_confusion_matrices(args)


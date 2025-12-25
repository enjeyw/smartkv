import torch
import torch.nn as nn
from torch.utils.data import Dataset
from safetensors.torch import load_file
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import argparse

from binary_gate_io import GateLoader

class BatchedKeyDataset(Dataset):
    def __init__(self, data_dir):
        pattern = os.path.join(data_dir, f"l*_h*")
        head_dirs = sorted(glob.glob(pattern))
        
        if not head_dirs:
            raise ValueError(f"No data in {data_dir}")
            
        print(f"Loading data for {len(head_dirs)} heads...")
        
        head_data_inputs = [[] for _ in head_dirs]
        head_data_targets = [[] for _ in head_dirs]
        self.head_ids = []
        
        for h_idx, h_dir in enumerate(head_dirs):
            self.head_ids.append(os.path.basename(h_dir))
            shards = sorted(glob.glob(os.path.join(h_dir, "*.safetensors")))
            
            # Load just one shard for speed
            for shard in shards[:1]:
                data = load_file(shard)
                head_data_inputs[h_idx].append(data["inputs"].float())
                head_data_targets[h_idx].append(data["targets"].float())

        full_inputs_per_head = [torch.cat(inputs) for inputs in head_data_inputs]
        full_targets_per_head = [torch.cat(targets) for targets in head_data_targets]
        
        self.inputs = torch.stack(full_inputs_per_head, dim=1)
        self.targets = torch.stack(full_targets_per_head, dim=1).unsqueeze(-1) 
        
    def __len__(self): return self.inputs.shape[0]

def plot_roc_all(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = BatchedKeyDataset(args.data_dir)
    inputs = dataset.inputs.to(device)
    targets = dataset.targets
    loader = GateLoader(args.model_dir)
    
    num_layers = 28
    num_heads = 8
    
    fig, axes = plt.subplots(7, 4, figsize=(20, 30))
    fig.suptitle("ROC Curves by Layer", fontsize=16)
    axes = axes.flatten()
    
    head_map = {hid: i for i, hid in enumerate(dataset.head_ids)}
    
    for l in range(num_layers):
        if l >= len(axes): break
        ax = axes[l]
        
        for h in range(num_heads):
            hid = f"l{l}_h{h}"
            if hid not in head_map: continue
            
            idx = head_map[hid]
            sd = loader.get_legacy_state_dict(hid, device=device)
            if sd is None:
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
            
            head_input = inputs[:, idx, :]
            with torch.no_grad():
                logits = model(head_input)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            y_true = targets[:, idx, 0].numpy().flatten()
            
            if len(np.unique(y_true)) < 2: continue
            
            fpr, tpr, _ = roc_curve(y_true, probs)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, label=f'H{h} ({roc_auc:.2f})')
            
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_title(f"Layer {l}")
        if l >= 24: ax.set_xlabel("FPR")
        if l % 4 == 0: ax.set_ylabel("TPR")
        ax.legend(fontsize='xx-small')
        
    plt.tight_layout()
    os.makedirs("plots/grid_roc", exist_ok=True)
    plt.savefig("plots/grid_roc/all_layers_roc.png")
    print("Saved plots/grid_roc/all_layers_roc.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/train_binary_256")
    parser.add_argument("--model_dir", type=str, default="models/gates_binary_256_v3")
    args = parser.parse_args()
    plot_roc_all(args)


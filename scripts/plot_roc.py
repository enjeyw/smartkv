import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import argparse

from binary_gate_io import GateLoader

class BatchedKeyDataset(Dataset):
    def __init__(self, data_dir, layer=0):
        # Specific layer filter
        pattern = os.path.join(data_dir, f"l{layer}_h*")
        head_dirs = sorted(glob.glob(pattern))
        
        if not head_dirs:
            raise ValueError(f"No data for layer {layer} in {data_dir}")
            
        print(f"Loading data for {len(head_dirs)} heads in Layer {layer}...")
        
        # Load just the first shard for quick eval if dataset is huge, 
        # or all shards for accuracy. Let's load all.
        head_data_inputs = [[] for _ in head_dirs]
        head_data_targets = [[] for _ in head_dirs]
        self.head_ids = []
        
        for h_idx, h_dir in enumerate(head_dirs):
            self.head_ids.append(os.path.basename(h_dir))
            shards = sorted(glob.glob(os.path.join(h_dir, "*.safetensors")))
            
            for shard in shards:
                data = load_file(shard)
                head_data_inputs[h_idx].append(data["inputs"].float())
                head_data_targets[h_idx].append(data["targets"].float())

        full_inputs_per_head = [torch.cat(inputs) for inputs in head_data_inputs]
        full_targets_per_head = [torch.cat(targets) for targets in head_data_targets]
        
        self.inputs = torch.stack(full_inputs_per_head, dim=1)
        self.targets = torch.stack(full_targets_per_head, dim=1).unsqueeze(-1) 
        
    def __len__(self): return self.inputs.shape[0]
    def __getitem__(self, idx): return self.inputs[idx], self.targets[idx]

class LegacyBinaryGateMLP(nn.Module):
    def __init__(self, num_heads, input_dim, hidden_dim=64):
        super().__init__()
        # Matches old architecture: Linear -> ReLU -> Linear
        # Weights were stored as net.0 and net.2
        self.W1 = nn.Parameter(torch.randn(num_heads, input_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(num_heads, hidden_dim))
        self.W2 = nn.Parameter(torch.randn(num_heads, hidden_dim, 1))
        self.b2 = nn.Parameter(torch.zeros(num_heads, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x_t = x.transpose(0, 1) 
        h = torch.bmm(x_t, self.W1) + self.b1.unsqueeze(1)
        h = self.relu(h)
        out = torch.bmm(h, self.W2) + self.b2.unsqueeze(1)
        return out.transpose(0, 1)

    def load_from_dict(self, state_dict, h_idx):
        # Map flat state_dict to per-head parameters
        # Old dict keys: net.0.weight (64, 128), net.0.bias (64), net.2.weight (1, 64), net.2.bias (1)
        # We need to manually assign specific head slice if we were loading a big batch model,
        # but here we load individual head checkpoints into a batched model shell?
        # Actually, simpler to just implement a single head model for eval if we load individual files.
        pass

class SingleHeadLegacyMLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x): return self.net(x)

def plot_roc(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = BatchedKeyDataset(args.data_dir, layer=args.layer)
    inputs = dataset.inputs.to(device) # [N, Heads, Dim]
    targets = dataset.targets # [N, Heads, 1]
    loader = GateLoader(args.model_dir)
    
    plt.figure(figsize=(10, 8))
    
    for i, h_id in enumerate(dataset.head_ids):
        print(f"Evaluating {h_id}...")
        sd = loader.get_legacy_state_dict(h_id, device=device)
        if sd is None:
            print(f"Model not found: {h_id}")
            continue
        
        if "net.0.weight" in sd and sd["net.0.weight"].shape[0] == 64:
            # Legacy Small
            model = SingleHeadLegacyMLP(input_dim=128, hidden_dim=64).to(device)
            model.load_state_dict(sd)
        elif "net.0.weight" in sd and sd["net.0.weight"].shape[0] == 256:
             # New Big (assuming 2 layers? Wait, new model has 3 layers: net.0, net.2, net.4)
             # Let's check keys
             if "net.4.weight" in sd:
                 # Big V3
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
             else:
                 # Maybe intermediate?
                 pass
        else:
            # Fallback for V3 (Manual reconstruction if needed)
            # The saved V3 model was saved as keys "net.0.weight" etc.
            # We need to strip the "net." prefix if loading into nn.Sequential(0, 1, 2, 3, 4)
            # Or rename keys in SD.
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
        
        # Check if we have both classes
        if len(np.unique(y_true)) < 2:
            print(f"Skipping {h_id}: Only one class present.")
            continue
            
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{h_id} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - Layer {args.layer}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    out_path = f"plots/roc_layer{args.layer}.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(out_path)
    print(f"\nSaved ROC plot to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0, help="Layer to evaluate")
    parser.add_argument("--data_dir", type=str, default="data/train_binary_256")
    parser.add_argument("--model_dir", type=str, default="models/gates_binary_256")
    args = parser.parse_args()
    
    plot_roc(args)


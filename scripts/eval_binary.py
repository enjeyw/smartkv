import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
import glob
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from binary_gate_io import GateLoader

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
        
        self.inputs = torch.stack(full_inputs_per_head, dim=1)
        self.targets = torch.stack(full_targets_per_head, dim=1).unsqueeze(-1) 
        
    def __len__(self): return self.inputs.shape[0]
    def __getitem__(self, idx): return self.inputs[idx], self.targets[idx]

class BinaryGateMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

def evaluate_binary():
    data_dir = "data/train_binary_256"
    model_dir = "models/gates_binary_256"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading Dataset...")
    dataset = BatchedKeyDataset(data_dir)
    
    # Use subset for eval speed
    eval_size = 10000
    indices = torch.randperm(len(dataset))[:eval_size]
    inputs = dataset.inputs[indices].to(device) # [N, H, D]
    targets = dataset.targets[indices].cpu().numpy() # [N, H, 1]
    
    input_dim = inputs.shape[2]
    
    print("Loading Models...")
    loader = GateLoader(model_dir)
    model_entries = []
    for idx, h_id in enumerate(dataset.head_ids):
        sd = loader.get_legacy_state_dict(h_id, device=device)
        if sd is None:
            print(f"Skipping {h_id}: checkpoint not found.")
            continue
        m = BinaryGateMLP(input_dim).to(device)
        m.load_state_dict(sd)
        m.eval()
        model_entries.append((idx, h_id, m))
    
    if not model_entries:
        raise RuntimeError("No heads could be loaded for evaluation.")
        
    print("Running Inference...")
    all_preds = [] # Logits
    
    with torch.no_grad():
        for dataset_idx, _, model in tqdm(model_entries):
            head_input = inputs[:, dataset_idx, :] # [N, D]
            logits = model(head_input)   # [N, 1]
            all_preds.append(logits.cpu().numpy())
            
    all_preds = np.stack(all_preds, axis=1) # [N, H, 1]
    
    # Metrics
    results = []
    print("\nCalculating Metrics per Head...")
    
    for col_idx, (dataset_idx, h_id, _) in enumerate(model_entries):
        y_true = targets[:, dataset_idx, 0]
        y_scores = all_preds[:, col_idx, 0] # Logits
        y_probs = 1 / (1 + np.exp(-y_scores)) # Sigmoid
        y_pred = (y_probs > 0.5).astype(int)
        
        # Skip if only one class present (ROC AUC undefined)
        if len(np.unique(y_true)) < 2:
            roc = 0.5
        else:
            roc = roc_auc_score(y_true, y_probs)
            
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        pos_rate = y_true.mean()
        
        results.append({
            "head": h_id,
            "roc_auc": roc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "pos_rate": pos_rate
        })
        
    # Sorting
    results.sort(key=lambda x: x['f1'], reverse=True)
    
    print("\n=== Top 20 Heads for Long-Term Memory (Sorted by F1) ===")
    print(f"{'Head':<10} | {'F1':<6} | {'ROC':<6} | {'Prec':<6} | {'Rec':<6} | {'PosRate':<6}")
    print("-" * 60)
    for r in results[:20]:
        print(f"{r['head']:<10} | {r['f1']:.4f} | {r['roc_auc']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['pos_rate']:.4f}")
        
    print("\n=== Bottom 5 Heads ===")
    for r in results[-5:]:
        print(f"{r['head']:<10} | {r['f1']:.4f} | {r['roc_auc']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['pos_rate']:.4f}")

    # Visualization
    f1s = [r['f1'] for r in results]
    rocs = [r['roc_auc'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(rocs, f1s, alpha=0.5)
    plt.xlabel("ROC AUC")
    plt.ylabel("F1 Score")
    plt.title("Head Performance: Long-Term Prediction (Threshold > 256)")
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/binary_head_performance.png")
    print("\nSaved plot to plots/binary_head_performance.png")

if __name__ == "__main__":
    evaluate_binary()


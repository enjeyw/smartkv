import torch
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import load_file
import glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

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
        
        self.inputs = torch.stack(full_inputs_per_head, dim=1) # [N, H, D]
        self.targets = torch.stack(full_targets_per_head, dim=1).unsqueeze(-1) # [N, H, 1]
        
    def __len__(self): return self.inputs.shape[0]
    def __getitem__(self, idx): return self.inputs[idx], self.targets[idx]

class BinaryGateMLP(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

def calc_sensitivity_specificity():
    data_dir = "data/train_binary_256"
    model_dir = "models/gates_binary_256"
    out_file = "metrics/sensitivity_specificity.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading Dataset...")
    dataset = BatchedKeyDataset(data_dir)
    
    # Use a reasonable subset for evaluation
    eval_size = 20000
    indices = torch.randperm(len(dataset))[:eval_size]
    inputs = dataset.inputs[indices].to(device)
    targets = dataset.targets[indices].cpu().numpy()
    
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
        raise RuntimeError("No heads available for metric computation.")
        
    print("Running Inference...")
    all_probs = []
    
    with torch.no_grad():
        for dataset_idx, _, model in tqdm(model_entries):
            head_input = inputs[:, dataset_idx, :]
            logits = model(head_input)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            
    all_probs = np.stack(all_probs, axis=1) # [N, H, 1]
    
    print("\nCalculating Metrics...")
    thresholds = [0.1, 0.2, 0.5]
    results = []
    
    for col_idx, (dataset_idx, h_id, _) in enumerate(model_entries):
        y_true = targets[:, dataset_idx, 0]
        y_prob = all_probs[:, col_idx, 0]
        
        row = {"head": h_id}
        
        # Calculate positive rate (% retained)
        pos_rate = y_true.mean()
        row["pct_retained"] = pos_rate
        
        # Sensitivity (Recall) = TP / (TP + FN)
        # Specificity (True Negative Rate) = TN / (TN + FP)
        
        for p in thresholds:
            y_pred = (y_prob >= p).astype(int)
            
            # Confusion Matrix elements
            TP = ((y_pred == 1) & (y_true == 1)).sum()
            TN = ((y_pred == 0) & (y_true == 0)).sum()
            FP = ((y_pred == 1) & (y_true == 0)).sum()
            FN = ((y_pred == 0) & (y_true == 1)).sum()
            
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            
            row[f"sens_p{p}"] = sensitivity
            row[f"spec_p{p}"] = specificity
            
        results.append(row)
        
    # Save to CSV
    os.makedirs("metrics", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_file, index=False)
    print(f"\nSaved metrics to {out_file}")
    
    # Display top 5 by sensitivity at p=0.5
    print("\nTop 5 Heads by Sensitivity (p=0.5):")
    print(df.sort_values(by="sens_p0.5", ascending=False).head(5))

if __name__ == "__main__":
    calc_sensitivity_specificity()


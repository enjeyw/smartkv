import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
import glob
import os
from tqdm import tqdm
import argparse
import numpy as np

from binary_gate_io import save_combined_gate_checkpoint, COMBINED_FILENAME

class BatchedKeyDataset(Dataset):
    def __init__(self, data_dir, layers=None, balanced=False):
        # layers: list of int, e.g. [1] or [0, 1]
        pattern = os.path.join(data_dir, "l*_h*")
        head_dirs = sorted(glob.glob(pattern))
        
        if not head_dirs:
            raise ValueError(f"No data in {data_dir}")
        
        # Filter by layers if specified
        if layers is not None:
            filtered_dirs = []
            for d in head_dirs:
                # directory name format: l{layer}_h{head}
                dirname = os.path.basename(d)
                try:
                    # Extract layer number. Assumes format lX_hY
                    l_part = dirname.split('_')[0]
                    layer_num = int(l_part[1:])
                    if layer_num in layers:
                        filtered_dirs.append(d)
                except:
                    pass
            head_dirs = filtered_dirs
            
        if not head_dirs:
            raise ValueError(f"No data found for layers {layers} in {data_dir}")
            
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
        
        self.inputs = torch.stack(full_inputs_per_head, dim=1) # [N, Heads, Dim]
        self.targets = torch.stack(full_targets_per_head, dim=1).unsqueeze(-1) # [N, Heads, 1]
        
        print(f"Dataset: {self.inputs.shape}")
        
        if balanced:
            print("Applying dataset balancing (oversampling positives)...")
            # Identify samples that are positive for AT LEAST ONE head
            # [N, Heads, 1] -> [N]
            any_positive = (self.targets > 0.5).any(dim=1).any(dim=1)
            
            pos_indices = torch.where(any_positive)[0]
            neg_indices = torch.where(~any_positive)[0]
            
            num_pos = len(pos_indices)
            num_neg = len(neg_indices)
            
            print(f"Original stats: Pos (any head)={num_pos}, Neg={num_neg}, Ratio={num_pos/len(self.inputs):.4f}")
            
            # Upsample positives to match negatives (or reasonable ratio)
            # Let's aim for 50/50 split of (Any-Positive vs All-Negative)
            if num_pos > 0:
                upsample_indices = torch.randint(0, num_pos, (num_neg,))
                pos_indices_upsampled = pos_indices[upsample_indices]
                
                # Combine
                all_indices = torch.cat([neg_indices, pos_indices_upsampled])
                # Shuffle
                perm = torch.randperm(len(all_indices))
                all_indices = all_indices[perm]
                
                self.inputs = self.inputs[all_indices]
                self.targets = self.targets[all_indices]
                
                print(f"Balanced Dataset: {self.inputs.shape}")
            else:
                print("Warning: No positive samples found! Skipping balancing.")
        
    def __len__(self): return self.inputs.shape[0]
    def __getitem__(self, idx): return self.inputs[idx], self.targets[idx]

class BinaryGateMLP(nn.Module):
    def __init__(self, num_heads, input_dim, hidden_dim=256):
        super().__init__()
        self.num_heads = num_heads
        
        # Layer 1: [Num_Heads, Input_Dim, Hidden_Dim]
        self.W1 = nn.Parameter(torch.randn(num_heads, input_dim, hidden_dim) / np.sqrt(input_dim))
        self.b1 = nn.Parameter(torch.zeros(num_heads, hidden_dim))
        
        # Layer 2: [Num_Heads, Hidden_Dim, Hidden_Dim]
        self.W2 = nn.Parameter(torch.randn(num_heads, hidden_dim, hidden_dim) / np.sqrt(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(num_heads, hidden_dim))
        
        # Layer 3: [Num_Heads, Hidden_Dim, 1]
        self.W3 = nn.Parameter(torch.randn(num_heads, hidden_dim, 1) / np.sqrt(hidden_dim))
        self.b3 = nn.Parameter(torch.zeros(num_heads, 1))
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [Batch, Num_Heads, Input_Dim]
        x_t = x.transpose(0, 1) # [Num_Heads, Batch, Input_Dim]
        
        # Layer 1
        h = torch.bmm(x_t, self.W1) + self.b1.unsqueeze(1)
        h = self.relu(h)
        
        # Layer 2
        h = torch.bmm(h, self.W2) + self.b2.unsqueeze(1)
        h = self.relu(h)
        
        # Layer 3
        out = torch.bmm(h, self.W3) + self.b3.unsqueeze(1)
        
        return out.transpose(0, 1) # [Batch, Num_Heads, 1]

def train_batched(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    layers = None
    if args.layers:
        layers = [int(l) for l in args.layers.split(',')]
        print(f"Filtering for layers: {layers}")
    
    dataset = BatchedKeyDataset(args.data_dir, layers=layers, balanced=args.balanced)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    num_heads = dataset.inputs.shape[1]
    input_dim = dataset.inputs.shape[2]
    model = BinaryGateMLP(num_heads, input_dim, hidden_dim=args.hidden_dim).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Standard BCE Loss without pos_weight
    criterion = nn.BCEWithLogitsLoss() 
    
    print(f"\nTraining {num_heads} binary classifiers (Hidden={args.hidden_dim})...")
    
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_since_improve = 0
    
    for epoch in range(args.max_epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            pred = model(x) 
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss + args.min_delta < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(
                    f"Early stopping triggered after epoch {epoch+1} "
                    f"(patience={args.patience})."
                )
                break
            
    if best_state is None:
        print("Warning: validation never improved; saving final epoch weights.")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
    
    print(
        f"Best validation loss {best_val_loss:.4f} achieved at epoch {best_epoch}. "
        f"Saving checkpoints to {args.output_dir}..."
    )
    os.makedirs(args.output_dir, exist_ok=True)
    
    metadata = {
        "best_epoch": str(best_epoch),
        "best_val_loss": f"{best_val_loss:.6f}",
        "hidden_dim": str(args.hidden_dim),
        "balanced": str(args.balanced),
    }
    combined_path = save_combined_gate_checkpoint(
        best_state,
        dataset.head_ids,
        args.output_dir,
        filename=args.combined_filename,
        metadata=metadata,
    )
    print(f"Wrote combined weights to {combined_path}")
    
    if args.save_legacy_heads:
        print("Saving legacy per-head checkpoints for backward compatibility...")
        W1, b1 = best_state["W1"], best_state["b1"]
        W2, b2 = best_state["W2"], best_state["b2"]
        W3, b3 = best_state["W3"], best_state["b3"]
        
        for h_idx in range(num_heads):
            head_name = dataset.head_ids[h_idx]
            single_sd = {
                "net.0.weight": W1[h_idx].transpose(0, 1),
                "net.0.bias": b1[h_idx],
                "net.2.weight": W2[h_idx].transpose(0, 1),
                "net.2.bias": b2[h_idx],
                "net.4.weight": W3[h_idx].transpose(0, 1),
                "net.4.bias": b3[h_idx]
            }
            torch.save(single_sd, os.path.join(args.output_dir, f"{head_name}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/train_binary_256")
    parser.add_argument("--output_dir", type=str, default="models/gates_binary_256")
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--layers", type=str, default=None, help="Comma separated list of layers to train on, e.g. '1' or '0,1'")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--balanced", action="store_true", help="Balance dataset by oversampling positives")
    parser.add_argument("--patience", type=int, default=5, help="Number of unimproved epochs before early stopping")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Minimum validation improvement to reset patience")
    parser.add_argument("--combined_filename", type=str, default=COMBINED_FILENAME, help="Filename for consolidated safetensors checkpoint")
    parser.add_argument("--save_legacy_heads", action="store_true", help="Also emit per-head .pt files for backwards compatibility")
    args = parser.parse_args()
    
    train_batched(args)

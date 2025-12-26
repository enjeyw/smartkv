import torch
import numpy as np
from safetensors.torch import load_file
import glob
import os
import matplotlib.pyplot as plt

def analyze_lifespans(data_dir):
    # Select a few representative heads
    # Early (Layer 0), Middle (Layer 14), Deep (Layer 27)
    heads_to_check = ["l0_h0", "l14_h0", "l27_h0"]
    
    print(f"Analyzing lifespans in {data_dir}...")
    
    for head_id in heads_to_check:
        print(f"\n--- Analysis for {head_id} ---")
        head_dir = os.path.join(data_dir, head_id)
        
        if not os.path.exists(head_dir):
            print(f"Directory {head_dir} not found.")
            continue
            
        shards = sorted(glob.glob(os.path.join(head_dir, "*.safetensors")))
        if not shards:
            print("No shards found.")
            continue
            
        # Load all targets for this head
        all_targets = []
        for shard in shards:
            data = load_file(shard)
            all_targets.append(data["targets"].float().numpy())
            
        if not all_targets:
            continue
            
        targets = np.concatenate(all_targets)
        
        # Convert log4(lifespan) back to lifespan
        # target = log4(lifespan) -> lifespan = 4^target
        lifespans = np.power(4, targets)
        
        # Statistics
        print(f"Total Samples: {len(lifespans)}")
        print(f"Min Lifespan: {np.min(lifespans):.2f}")
        print(f"Max Lifespan: {np.max(lifespans):.2f}")
        print(f"Mean Lifespan: {np.mean(lifespans):.2f}")
        print(f"Median Lifespan: {np.median(lifespans):.2f}")
        
        # Percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        p_vals = np.percentile(lifespans, percentiles)
        print("Percentiles:")
        for p, val in zip(percentiles, p_vals):
            print(f"  {p}th: {val:.2f}")
            
        # Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(targets, bins=50, alpha=0.7, label=head_id)
        plt.title(f"Log4 Lifespan Distribution ({head_id})")
        plt.xlabel("Log4(Lifespan)")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"lifespan_dist_{head_id}.png")
        plt.close()
        print(f"Saved histogram to lifespan_dist_{head_id}.png")

if __name__ == "__main__":
    analyze_lifespans("data/train_qwen3_regression")


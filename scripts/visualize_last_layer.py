import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_last_layer_attention():
    model_id = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.float16, 
        trust_remote_code=True,
        attn_implementation="eager" # Needed to get attn weights easily
    )

    text = "The quick brown fox jumps over the lazy dog. " * 3
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        
    # Get last layer attention
    # outputs.attentions is a tuple of (Batch, Num_Heads, Seq, Seq)
    # Last layer is index -1
    last_layer_attn = outputs.attentions[-1][0].float().cpu().numpy() # [Num_Heads, Seq, Seq]
    
    num_heads = last_layer_attn.shape[0]
    seq_len = last_layer_attn.shape[1]
    
    print(f"Layer 27 Attention Shape: {last_layer_attn.shape}")
    
    # Plot Grid of Heads
    rows = 4
    cols = num_heads // rows
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
    axes = axes.flatten()
    
    for h in range(num_heads):
        ax = axes[h]
        # Log scale for visibility
        sns.heatmap(last_layer_attn[h], ax=ax, cmap="viridis", cbar=False)
        ax.set_title(f"Head {h}")
        ax.axis('off')
        
    plt.suptitle("Layer 27 Attention Patterns", fontsize=16)
    plt.tight_layout()
    plt.savefig("layer27_attention_vis.png")
    print("Saved layer27_attention_vis.png")
    
    # Check "Self-Attention Only" Hypothesis
    # We look at the average attention distance
    # Diag = 0 distance.
    
    print("\n--- Attention Statistics ---")
    for h in range(num_heads):
        matrix = last_layer_attn[h]
        # Mask lower triangle
        mask = np.tril(np.ones_like(matrix))
        matrix = matrix * mask
        
        # Calculate center of mass of attention for each query
        # For query i, what is the average key position j?
        # We want to see if it's mostly attending to j=i (diagonal)
        
        diag_mass = 0
        total_mass = 0
        
        for i in range(seq_len):
            row = matrix[i]
            diag_val = row[i]
            row_sum = np.sum(row)
            if row_sum > 0:
                diag_mass += diag_val
                total_mass += row_sum
                
        ratio = diag_mass / total_mass if total_mass > 0 else 0
        print(f"Head {h}: Diagonal Attention Ratio = {ratio:.4f}")

if __name__ == "__main__":
    visualize_last_layer_attention()


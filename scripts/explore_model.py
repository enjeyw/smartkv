import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import os

def explore_model():
    # Updating to Qwen3-0.6B as requested
    target_model = "Qwen/Qwen3-0.6B" 
    # Fallback if Qwen3 doesn't exist (since it might be hypothetical)
    fallback_model = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Loading {target_model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            target_model, 
            device_map="auto", 
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager" 
        )
    except Exception as e:
        print(f"Error loading {target_model}: {e}")
        print(f"Falling back to {fallback_model}...")
        target_model = fallback_model
        tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            target_model, 
            device_map="auto", 
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager" 
        )

    print("\nModel Architecture:")
    print(model)
    
    # Input text
    text = "The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. " * 10
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    print(f"\nInput shape: {inputs.input_ids.shape}")
    
    # Lists to capture data
    attention_scores = []
    
    def attn_hook(module, input, output):
        if isinstance(output, tuple):
            if len(output) > 1 and output[1] is not None:
                 attention_scores.append(output[1].detach().cpu())
    
    # Register hooks ONLY on the Attention Module
    hooks = []
    for name, module in model.named_modules():
        if name.endswith("self_attn"): 
            hooks.append(module.register_forward_hook(attn_hook))
            print(f"Hooked {name}")

    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    print(f"\nCaptured {len(attention_scores)} attention maps.")
    
    if len(attention_scores) == 0:
        print("WARNING: No attention scores captured. Check hooks.")
        return

    # Analyze distribution
    layer_idx = 5
    if layer_idx < len(attention_scores):
        attn_map = attention_scores[layer_idx][0] # Batch 0
        
        seq_len = attn_map.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len))
        
        all_valid_scores = []
        for head_idx in range(attn_map.shape[0]):
            head_attn = attn_map[head_idx]
            valid_scores = head_attn[mask == 1]
            all_valid_scores.append(valid_scores)
            
        all_valid_scores = torch.cat(all_valid_scores)
        valid_scores_np = all_valid_scores.float().numpy()
        
        print(f"\nLayer {layer_idx} Stats (All Heads):")
        print(f"Min: {np.min(valid_scores_np)}")
        print(f"Max: {np.max(valid_scores_np)}")
        print(f"Mean: {np.mean(valid_scores_np)}")
        print(f"Median: {np.median(valid_scores_np)}")
        
        percentiles = [50, 75, 90, 95, 99, 99.9]
        results = np.percentile(valid_scores_np, percentiles)
        
        print("\nPercentiles:")
        for p, val in zip(percentiles, results):
            print(f"{p}th: {val:.6f}")
            
        suggested_T = results[4] # 99th percentile
        print(f"\nSuggested Threshold (99th percentile): {suggested_T}")
        
        # Visualization
        print("\nGenerating histogram...")
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.hist(valid_scores_np, bins=100, log=True, alpha=0.7, color='blue')
        plt.axvline(suggested_T, color='red', linestyle='dashed', linewidth=1, label=f'99th % ({suggested_T:.4f})')
        plt.title(f"Attention Score Distribution (Layer {layer_idx})")
        plt.xlabel("Attention Score")
        plt.ylabel("Frequency (Log Scale)")
        plt.legend()
        plt.savefig("plots/attention_dist.png")
        print("Saved plots/attention_dist.png")
        
    # Verify Key Extraction Point
    print("\nVerifying Key Projection...")
    for name, module in model.named_modules():
        if "k_proj" in name:
            print(f"Found Key Projection: {name} -> {module}")
            break

if __name__ == "__main__":
    explore_model()

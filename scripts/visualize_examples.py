import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file

# --- Definitions ---
class GateMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

class SmartKVVisualizer:
    def __init__(self, model_path="models/gates_qwen3_batched", device="cuda"):
        self.device = device
        self.gates = {}
        
        # Load model
        self.model_id = "Qwen/Qwen3-0.6B"
        try:
            print(f"Loading {self.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, attn_implementation="eager"
            )
        except:
            print("Fallback to Qwen2.5")
            self.model_id = "Qwen/Qwen2.5-0.5B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)

        # Load Gates
        gate_files = glob.glob(os.path.join(model_path, "*.pt"))
        print(f"Loading {len(gate_files)} gates...")
        for f in gate_files:
            name = os.path.basename(f).replace(".pt", "")
            parts = name.split("_")
            layer = int(parts[0][1:])
            head = int(parts[1][1:])
            
            gate = GateMLP(input_dim=128).to(device) # Hardcoded 128 for Qwen3-0.6B
            gate.load_state_dict(torch.load(f, map_location=device))
            gate.eval()
            self.gates[(layer, head)] = gate

    def analyze_text(self, text, target_layer=0, target_head=6):
        # We focus on one specific head to make the viz readable.
        # l0_h6 was our "Best Predictor" in findings.
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        seq_len = inputs.input_ids.shape[1]
        
        print(f"Analyzing {seq_len} tokens...")
        
        # Hooks
        captured_keys = {}
        captured_attn = {}
        
        def k_hook(m, i, o):
            # o: [B, S, Hidden] -> [B, S, Heads, Dim]
            B, S, H_dim = o.shape
            num_heads = 8
            head_dim = H_dim // num_heads
            keys = o.view(B, S, num_heads, head_dim).transpose(1, 2)
            captured_keys['keys'] = keys.detach() # [B, H, S, D]
            
        def a_hook(m, i, o):
            # o[1]: [B, H, S, S]
            if o[1] is not None:
                captured_attn['attn'] = o[1].detach()
                
        h1 = self.model.model.layers[target_layer].self_attn.k_proj.register_forward_hook(k_hook)
        h2 = self.model.model.layers[target_layer].self_attn.register_forward_hook(a_hook)
        
        with torch.no_grad():
            self.model(inputs.input_ids, output_attentions=True)
            
        h1.remove()
        h2.remove()
        
        # Get Data
        keys = captured_keys['keys'][0, target_head] # [S, D]
        attn = captured_attn['attn'][0, target_head] # [S, S]
        
        # 1. Calculate Ground Truth Lifespan
        # For each token i, find max(j) where j > i and attn[j, i] > threshold
        threshold = 0.05
        lifespans = []
        
        attn_cpu = attn.cpu().float()
        
        for i in range(seq_len):
            future_mask = attn_cpu[i+1:, i] # Query pos > i
            hits = (future_mask > threshold).nonzero(as_tuple=True)[0]
            
            if len(hits) > 0:
                last_hit_relative = hits[-1].item() # relative to i+1
                # if hits[0] is index 0 of future_mask, that is pos i+1.
                # lifespan = (i + 1 + relative) - i = 1 + relative
                lifespan = last_hit_relative + 1
            else:
                lifespan = 0 # or 1 (minimum)
            
            lifespan = max(1, lifespan)
            lifespans.append(lifespan)
            
        # 2. Get Predictions
        gate = self.gates.get((target_layer, target_head))
        if not gate:
            print("Gate not found!")
            return
            
        with torch.no_grad():
            log_preds = gate(keys.float()).squeeze().cpu().numpy()
            
        # Convert log4 to actual
        preds = 4.0 ** log_preds
        actuals = np.array(lifespans)
        
        # 3. Visualize
        # We want to show tokens where error is low vs high
        # Error metric: abs(log_pred - log_actual)
        log_actuals = np.log(actuals) / np.log(4.0)
        errors = np.abs(log_preds - log_actuals)
        
        # Filter out boring tokens (short lifespan, short prediction)
        # We want interesting cases.
        
        results = []
        for i in range(seq_len):
            results.append({
                "token": tokens[i].replace("Ġ", ""), # Clean token
                "pos": i,
                "pred": preds[i],
                "actual": actuals[i],
                "log_pred": log_preds[i],
                "log_actual": log_actuals[i],
                "error": errors[i],
                "context": self.get_context(tokens, i)
            })
            
        return results

    def get_context(self, tokens, idx, window=3):
        start = max(0, idx - window)
        end = min(len(tokens), idx + window + 1)
        
        ctx = []
        for i in range(start, end):
            t = tokens[i].replace("Ġ", "")
            if i == idx: t = f"**{t}**"
            ctx.append(t)
        return " ".join(ctx)

def visualize_examples():
    viz = SmartKVVisualizer()
    
    # Sample Text: Mix of narrative and code to trigger different patterns
    text = """
    Once upon a time in a digital kingdom, there lived a young coder named Alice. 
    She loved Python because it was simple yet powerful. 
    One day, she wrote a function called 'process_data'. 
    def process_data(data):
        return [x * 2 for x in data]
    Alice realized that 'process_data' was inefficient for large lists.
    So she refactored it. The new version was faster.
    """
    
    # Analyze Layer 0 Head 6 (The "Good" Head)
    print("\n=== Analyzing Layer 0 Head 6 ===")
    results = viz.analyze_text(text, target_layer=0, target_head=6)
    
    # Sort by Error
    results_sorted = sorted(results, key=lambda x: x['error'])
    
    # Top 5 Correct
    print("\n--- Top 5 Correct Predictions ---")
    print(f"{'Token':<15} | {'Pred':<8} | {'Actual':<8} | {'Context'}")
    print("-" * 60)
    for r in results_sorted[:5]:
        print(f"{r['token']:<15} | {r['pred']:<8.1f} | {r['actual']:<8.1f} | {r['context']}")
        
    # Top 5 Incorrect (Overestimates or Underestimates)
    print("\n--- Top 5 Incorrect Predictions ---")
    print(f"{'Token':<15} | {'Pred':<8} | {'Actual':<8} | {'Context'}")
    print("-" * 60)
    for r in results_sorted[-5:]:
        print(f"{r['token']:<15} | {r['pred']:<8.1f} | {r['actual']:<8.1f} | {r['context']}")

    # Create Plot
    tokens = [r['token'] for r in results]
    pred_vals = [r['pred'] for r in results]
    act_vals = [r['actual'] for r in results]
    
    plt.figure(figsize=(15, 6))
    x = range(len(tokens))
    plt.plot(x, act_vals, label="Actual Lifespan", marker='o', markersize=4, alpha=0.6)
    plt.plot(x, pred_vals, label="Predicted Lifespan", marker='x', markersize=4, alpha=0.8)
    
    plt.yscale('log', base=4)
    plt.xticks(x[::2], tokens[::2], rotation=90, fontsize=8)
    plt.title(f"Lifespan Prediction: Layer 0 Head 6\nText: 'Alice & Python Code'")
    plt.ylabel("Lifespan (Log4 Scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/example_viz_l0_h6.png")
    print("\nSaved plot to plots/example_viz_l0_h6.png")

if __name__ == "__main__":
    visualize_examples()


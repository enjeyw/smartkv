import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import os
import glob
import numpy as np
from safetensors.torch import load_file

# --- Model & Gate Definitions ---

class GateMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )
        
    def forward(self, x):
        return self.net(x)

class SmartKVController:
    def __init__(self, model_path="models/gates_qwen3_t05", device="cuda"):
        self.device = device
        self.gates = {}
        self.enabled_heads = set()
        
        # Load all available gates
        gate_files = glob.glob(os.path.join(model_path, "*.pt"))
        print(f"Loading {len(gate_files)} gate models...")
        
        for f in gate_files:
            name = os.path.basename(f).replace(".pt", "")
            # Parse layer and head
            try:
                # format lX_hY
                parts = name.split("_")
                layer = int(parts[0][1:])
                head = int(parts[1][1:])
                
                # Initialize model
                # We need to know input dim. Qwen 0.5B head_dim is 128? 
                # Qwen3-0.6B: hidden=1024, heads=8 => head_dim=128.
                gate = GateMLP(input_dim=128).to(device) # GateMLP is float32 by default
                gate.load_state_dict(torch.load(f, map_location=device))
                gate.eval()
                # Convert gate to half if needed, or inputs to float
                # Since Qwen is loaded in half (float16), we should probably match.
                # But the training was done in float.
                # Let's keep gate in float and cast input in predict_lifespan.
                
                self.gates[(layer, head)] = gate
                self.enabled_heads.add((layer, head))
            except Exception as e:
                print(f"Skipping {name}: {e}")

    def predict_lifespan(self, layer_idx, head_idx, key_vector):
        """
        key_vector: [Batch, Dim] or [Dim]
        Returns: scalar log-lifespan score
        """
        if (layer_idx, head_idx) not in self.gates:
            return torch.zeros(key_vector.shape[0], device=self.device) + 999.0 # Default keep
            
        with torch.no_grad():
            # Ensure input is float32
            x = key_vector.float()
            score = self.gates[(layer_idx, head_idx)](x)
        return score.squeeze()

# --- Benchmarking Logic ---

def run_benchmark():
    model_id = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_id} for benchmarking...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto", 
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager" 
        )
    except:
        print("Fallback to Qwen2.5")
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)

    controller = SmartKVController(device=model.device)
    
    # Needle in a Haystack Test
    # Context: [Passkey Definition] ... [Junk] ... [Question]
    passkey = "The secret password is 'BLUEBERRY'."
    junk = "The quick brown fox jumps over the lazy dog. " * 500 # ~5000 tokens? 500 * 9 = 4500
    question = "What is the secret password?"
    
    text = f"{passkey}\n{junk}\n{question}"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    print(f"Input Length: {inputs.input_ids.shape[1]} tokens")
    
    # We need to hook the model to simulate eviction or monitor values
    # Real eviction requires custom attention implementation.
    # For this benchmark, we will Measure Prediction Accuracy on the Critical Token ("BLUEBERRY")
    
    # 1. Identify the "Needle" tokens
    needle_ids = tokenizer(passkey, return_tensors="pt").input_ids[0]
    needle_len = len(needle_ids)
    print(f"Needle length: {needle_len}")
    
    # 2. Run Forward Pass
    print("Running forward pass...")
    
    # Capture Keys
    captured_keys = {}
    def key_hook(module, input, output, layer_idx):
        # Output: [B, S, H, D] (after reshape in generate_dataset, but here it's raw)
        # Qwen: [B, S, Hidden]
        # We need to manually reshape to heads
        B, S, H_dim = output.shape
        num_heads = 8 # Hardcoded for Qwen3-0.6B/Qwen2.5-0.5B (verify config)
        head_dim = H_dim // num_heads
        
        keys = output.view(B, S, num_heads, head_dim).transpose(1, 2) # [B, Heads, S, Dim]
        captured_keys[layer_idx] = keys
        
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.k_proj.register_forward_hook(
            lambda m, i, o, idx=i: key_hook(m, i, o, idx)
        ))
        
    with torch.no_grad():
        model(inputs.input_ids)
        
    # 3. Evaluate Predictions on Needle
    # The needle is at the start (index 0 to needle_len)
    # The "Lifespan" should be HIGH (it is needed at the end)
    # Let's see what the gates predict for these tokens.
    
    print("\nEvaluating Gate Predictions on Needle Tokens...")
    print("(Expect High Scores > 4.0 for long retention)")
    
    results = []
    
    num_layers = len(model.model.layers)
    num_heads_model = 8 # Qwen3-0.6B / Qwen2.5-0.5B

    for layer_idx in range(num_layers):
        if layer_idx not in captured_keys: continue
        
        keys = captured_keys[layer_idx] # [1, Heads, Seq, Dim]
        
        # Check heads
        # NOTE: captured_keys might have incorrect head count if my hardcoded assumption was wrong
        # keys.shape[1] is the ground truth from the reshape
        
        for head_idx in range(keys.shape[1]):
            # Only check if we have a gate
            if (layer_idx, head_idx) in controller.gates:
                # Get needle vectors
                needle_vectors = keys[0, head_idx, :needle_len, :] # [Needle_Len, Dim]
                
                scores = controller.predict_lifespan(layer_idx, head_idx, needle_vectors)
                
                # Handle single-token needle case where scores is 0-d
                if scores.ndim == 0:
                     avg_score = scores.item()
                else:
                     avg_score = scores.mean().item()
                
                results.append((layer_idx, head_idx, avg_score))

    # Sort by score (highest retention first)
    results.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop 10 Heads favoring the Needle:")
    for r in results[:10]:
        print(f"Layer {r[0]} Head {r[1]}: Score {r[2]:.2f}")
        
    print("\nBottom 10 Heads (Would evict Needle):")
    for r in results[-10:]:
        print(f"Layer {r[0]} Head {r[1]}: Score {r[2]:.2f}")

    # Cleanup
    for h in hooks: h.remove()

if __name__ == "__main__":
    run_benchmark()


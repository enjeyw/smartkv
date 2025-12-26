import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from SmartKVCache import SmartKVDynamicCache, GateLoader
import time

def run_inference():
    model_id = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="cuda", 
        torch_dtype=torch.float16, 
        trust_remote_code=True,
        attn_implementation="eager" # Required for custom cache usually
    )
    
    # Load Gates
    gate_loader = GateLoader("models/gates_qwen3_regression", device="cuda")
    
    # Params
    WINDOW_SIZE = 32
    SINK_SIZE = 4
    BUDGET_SIZE = 128 # Very aggressive compression for testing
    
    cache = SmartKVDynamicCache(
        window_size=WINDOW_SIZE,
        sink_size=SINK_SIZE,
        budget_size=BUDGET_SIZE,
        gate_loader=gate_loader,
        num_layers=len(model.model.layers),
        num_kv_heads=model.config.num_key_value_heads,
        device=model.device
    )
    
    # Text
    prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
    long_prompt = prompt * 20 # ~800 tokens
    
    print(f"Prompt length: ~{len(tokenizer.encode(long_prompt))} tokens")
    
    inputs = tokenizer(long_prompt, return_tensors="pt").to("cuda")
    
    print("\n--- Phase 1: Prefill (No Pruning) ---")
    start = time.time()
    with torch.no_grad():
        # We manually prefill to inject our cache
        # Actually model.generate will do prefill.
        # But we need to intercept the transition to prune.
        
        # Method: Run forward pass on full prompt
        out = model(inputs.input_ids, use_cache=True, past_key_values=cache)
        logits = out.logits
        
    print(f"Prefill done in {time.time() - start:.2f}s")
    print(f"Cache size (Layer 0) before prune: {cache.get_seq_length(0)}")
    
    print("\n--- Phase 2: Bulk Prune ---")
    start = time.time()
    cache.bulk_prune()
    print(f"Prune done in {time.time() - start:.2f}s")
    print(f"Cache size (Layer 0) after prune: {cache.get_seq_length(0)}")
    print(f"Expected size: {SINK_SIZE + BUDGET_SIZE + WINDOW_SIZE}")
    
    print("\n--- Phase 3: Generation (Streaming Prune) ---")
    # We need to pass the *next* token and the *pruned* cache
    # The last token from prefill output is the input for generation
    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
    
    generated_ids = []
    
    for _ in range(50):
        with torch.no_grad():
            out = model(next_token, use_cache=True, past_key_values=cache)
            logits = out.logits
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            generated_ids.append(next_token.item())
            
    print("\nGenerated Output:")
    print(tokenizer.decode(generated_ids))
    
    # Check final sizes
    print(f"\nFinal Cache Size (Layer 0): {cache.get_seq_length(0)}")
    print(f"Final Cache Size (Layer 14): {cache.get_seq_length(14)}")

if __name__ == "__main__":
    run_inference()

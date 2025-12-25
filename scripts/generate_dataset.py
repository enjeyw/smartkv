import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, interleave_datasets
import numpy as np
from safetensors.torch import save_file
import os
from tqdm import tqdm
import argparse

def get_dataset(num_samples=10000):
    """
    Loads and interleaves PG19 (Books) and GSM8K (Math).
    """
    print("Loading datasets...")
    
    # Load PG19 (subset)
    try:
        pg19 = load_dataset("deepmind/pg19", split="train", streaming=True)
    except:
        print("PG19 load failed, trying fallback...")
        # Fallback if PG19 is restricted/slow, use C4 or similar for demo
        pg19 = load_dataset("allenai/c4", "en", split="train", streaming=True)

    # Load GSM8K
    gsm8k = load_dataset("gsm8k", "main", split="train", streaming=True)
    
    # Format GSM8K to plain text
    def format_gsm8k(example):
        return {"text": example["question"] + "\n" + example["answer"]}
    
    gsm8k = gsm8k.map(format_gsm8k)
    
    # Interleave: 70% PG19, 30% Math
    mixed_dataset = interleave_datasets([pg19, gsm8k], probabilities=[0.7, 0.3], seed=42)
    
    return mixed_dataset.take(num_samples)

def generate_data(args):
    model_id = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto", 
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager"
        )
    except Exception as e:
        print(f"Failed to load {model_id}: {e}")
        print("Using fallback Qwen/Qwen2.5-0.5B-Instruct")
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto", 
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager"
        )
    
    # Storage for current batch
    # Structure: keys_storage[layer_idx][head_idx] -> list of tensors
    keys_storage = {} 
    labels_storage = {}
    
    # Initialize storage
    num_layers = len(model.model.layers)
    # Qwen2 config usually has num_key_value_heads in config, but module might not expose it directly as attribute
    num_heads = model.config.num_key_value_heads
    
    print(f"Model has {num_layers} layers and {num_heads} KV heads.")
    
    # Hooks
    captured_keys = {} # Per forward pass
    captured_attns = {} # Per forward pass
    
    def key_hook(module, input, output, layer_idx):
        # output of k_proj is [Batch, Seq, Num_KV_Heads * Head_Dim]
        # We need to reshape to [Batch, Seq, Num_KV_Heads, Head_Dim]
        # and verify this matches the model structure
        
        # Qwen2 implementation:
        # k_proj output is [batch, seq_len, hidden_size] (actually projected size)
        # It gets reshaped later.
        
        B, L, D = output.shape
        head_dim = D // num_heads
        
        # Save as [Batch, Num_KV_Heads, Seq, Head_Dim] for easier indexing
        keys = output.view(B, L, num_heads, head_dim).transpose(1, 2)
        captured_keys[layer_idx] = keys.detach().cpu() # Move to CPU immediately

    def attn_hook(module, input, output, layer_idx):
        # output[1] is attn_weights: [Batch, Num_Q_Heads, Seq, Seq]
        # NOTE: Qwen uses GQA. There are more Q heads than KV heads.
        # We need to map Q heads to KV heads to know "who attended to this key".
        # But for "Usage prediction", if ANY Q head in the group attends, the key is useful.
        
        if output[1] is not None:
            captured_attns[layer_idx] = output[1].detach().cpu()

    # Register Hooks
    for i, layer in enumerate(model.model.layers):
        # Hook Key Projection (Pre-RoPE)
        layer.self_attn.k_proj.register_forward_hook(
            lambda m, i, o, idx=i: key_hook(m, i, o, idx)
        )
        # Hook Attention (to get ground truth)
        layer.self_attn.register_forward_hook(
            lambda m, i, o, idx=i: attn_hook(m, i, o, idx)
        )

    # Processing Loop
    dataset = get_dataset(args.num_samples)
    
    SEQ_LEN = 1024 # Context Window for training
    THRESHOLD = 0.05 # Attention Threshold (Adjusted lower for better sensitivity)
    
    sample_count = 0
    
    for sample in tqdm(dataset):
        text = sample["text"]
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
        
        if tokens.input_ids.shape[1] < 128: # Skip very short samples
            continue
            
        input_ids = tokens.input_ids.to(model.device)
        
        # Clear captures
        captured_keys = {}
        captured_attns = {}
        
        with torch.no_grad():
            model(input_ids, output_attentions=True)
            
        # Process captured data
        # We iterate over layers
        for layer_idx in range(num_layers):
            keys = captured_keys[layer_idx] # [1, KV_Heads, Seq, Dim]
            attns = captured_attns[layer_idx] # [1, Q_Heads, Seq, Seq]
            
            # Handle GQA: Map Q heads to KV heads
            num_q_heads = attns.shape[1]
            q_per_kv = num_q_heads // num_heads
            
            # Max pool attention across the query group sharing this key
            # Reshape attns to [1, KV_Heads, Q_Per_KV, Seq, Seq]
            attns_grouped = attns.view(1, num_heads, q_per_kv, attns.shape[2], attns.shape[3])
            # Max over Q_Per_KV dimension -> [1, KV_Heads, Seq, Seq]
            attns_reduced, _ = attns_grouped.max(dim=2)
            
            seq_len = keys.shape[2]
            
            # For each KV Head
            for head_idx in range(num_heads):
                k_vectors = keys[0, head_idx] # [Seq, Dim]
                attn_matrix = attns_reduced[0, head_idx] # [Seq, Seq] (Rows=Queries, Cols=Keys)
                
                # Vectorized Label Generation
                # We want: For each key k_i, what is log4(lifespan)?
                # Lifespan = (last_significant_query_pos - key_pos)
                
                # 1. Threshold the attention matrix
                mask = (attn_matrix > THRESHOLD).float()
                
                # 2. Find indices of hits
                # We only care about future attention (Lower triangular is causal, so Q_idx > K_idx is standard)
                # attn_matrix[q, k]. q is query index (time), k is key index (time).
                # We need max(q) for each k where mask[q, k] == 1.
                
                # Create a matrix of Query Indices
                q_indices = torch.arange(seq_len).unsqueeze(1).expand(seq_len, seq_len) # [Seq, Seq]
                
                # Apply mask (Zero out non-hits)
                # We multiply by mask. If no hit, we get 0.
                # But 0 is a valid index. So we usually fill non-hits with -1.
                hits_q_indices = q_indices.clone()
                hits_q_indices[mask == 0] = -1
                
                # Max over Query dimension (dim 0) to find last usage time for each key column
                last_usage_pos, _ = hits_q_indices.max(dim=0) # [Seq] (one per key)
                
                # Current position of the key is its index
                key_pos = torch.arange(seq_len)
                
                # Calculate Lifespan
                # If last_usage_pos <= key_pos, it means no *future* attention (or only self-attention).
                lifespan = last_usage_pos - key_pos
                lifespan = torch.clamp(lifespan, min=1) # Minimum lifespan 1
                
                # Log4 transform
                targets = torch.log(lifespan.float()) / torch.log(torch.tensor(4.0))
                targets = torch.clamp(targets, max=6.0) # Cap at 4096 tokens
                
                # Store
                k_vectors_np = k_vectors.numpy()
                targets_np = targets.numpy()
                
                key_id = f"l{layer_idx}_h{head_idx}"
                if key_id not in keys_storage:
                    keys_storage[key_id] = []
                    labels_storage[key_id] = []
                    
                keys_storage[key_id].append(k_vectors_np)
                labels_storage[key_id].append(targets_np)
        
        sample_count += 1
        
        # Save periodically
        if sample_count % 100 == 0:
            save_shards(keys_storage, labels_storage, args.output_dir, sample_count)
            # Reset storage to free RAM
            keys_storage = {}
            labels_storage = {}

    # Final Save
    if any(keys_storage):
         save_shards(keys_storage, labels_storage, args.output_dir, sample_count)

def save_shards(keys, labels, out_dir, step):
    print(f"Saving checkpoint at step {step}...")
    for key_id in keys:
        # Concatenate all samples for this head
        k_data = np.concatenate(keys[key_id], axis=0) # [Total_Seq, Dim]
        l_data = np.concatenate(labels[key_id], axis=0) # [Total_Seq]
        
        # Convert to tensor
        save_dict = {
            "inputs": torch.from_numpy(k_data),
            "targets": torch.from_numpy(l_data)
        }
        
        # File path: data/train/layer_X_head_Y/part_Z.safetensors
        head_dir = os.path.join(out_dir, key_id)
        os.makedirs(head_dir, exist_ok=True)
        
        fname = os.path.join(head_dir, f"shard_{step}.safetensors")
        save_file(save_dict, fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="data/train_qwen3_t05")
    args = parser.parse_args()
    
    generate_data(args)


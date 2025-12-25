import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, interleave_datasets
import numpy as np
from safetensors.torch import save_file
import os
from tqdm import tqdm
import argparse

def get_dataset(num_samples=10000):
    print("Loading datasets...")
    
    try:
        pg19 = load_dataset("deepmind/pg19", split="train", streaming=True)
    except:
        print("PG19 load failed, trying fallback...")
        pg19 = load_dataset("allenai/c4", "en", split="train", streaming=True)

    gsm8k = load_dataset("gsm8k", "main", split="train", streaming=True)
    
    def format_gsm8k(example):
        return {"text": example["question"] + "\n" + example["answer"]}
    
    gsm8k = gsm8k.map(format_gsm8k)
    
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
    
    keys_storage = {} 
    labels_storage = {}
    
    num_layers = len(model.model.layers)
    num_heads = model.config.num_key_value_heads
    
    print(f"Model has {num_layers} layers and {num_heads} KV heads.")
    
    captured_keys = {}
    captured_attns = {}
    
    def key_hook(module, input, output, layer_idx):
        B, L, D = output.shape
        head_dim = D // num_heads
        keys = output.view(B, L, num_heads, head_dim).transpose(1, 2)
        captured_keys[layer_idx] = keys.detach().cpu()

    def attn_hook(module, input, output, layer_idx):
        if output[1] is not None:
            captured_attns[layer_idx] = output[1].detach().cpu()

    for i, layer in enumerate(model.model.layers):
        layer.self_attn.k_proj.register_forward_hook(
            lambda m, i, o, idx=i: key_hook(m, i, o, idx)
        )
        layer.self_attn.register_forward_hook(
            lambda m, i, o, idx=i: attn_hook(m, i, o, idx)
        )

    dataset = get_dataset(args.num_samples)
    
    SEQ_LEN = 1024
    ATTN_THRESHOLD = 0.01 # Significant attention threshold
    LONG_TERM_THRESHOLD = 256 # Lifespan threshold
    
    sample_count = 0
    
    for sample in tqdm(dataset):
        text = sample["text"]
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
        
        if tokens.input_ids.shape[1] < 128:
            continue
            
        input_ids = tokens.input_ids.to(model.device)
        
        captured_keys = {}
        captured_attns = {}
        
        with torch.no_grad():
            model(input_ids, output_attentions=True)
            
        for layer_idx in range(num_layers):
            keys = captured_keys[layer_idx]
            attns = captured_attns[layer_idx]
            
            num_q_heads = attns.shape[1]
            q_per_kv = num_q_heads // num_heads
            
            attns_grouped = attns.view(1, num_heads, q_per_kv, attns.shape[2], attns.shape[3])
            attns_reduced, _ = attns_grouped.max(dim=2)
            
            seq_len = keys.shape[2]
            
            for head_idx in range(num_heads):
                k_vectors = keys[0, head_idx]
                attn_matrix = attns_reduced[0, head_idx]
                
                # Binary Label Generation
                # Label = 1 if ANY query j > i + 256 has attn(j, i) > 0.01
                
                mask = (attn_matrix > ATTN_THRESHOLD).float()
                
                q_indices = torch.arange(seq_len).unsqueeze(1).expand(seq_len, seq_len)
                hits_q_indices = q_indices.clone()
                hits_q_indices[mask == 0] = -1
                
                last_usage_pos, _ = hits_q_indices.max(dim=0)
                
                key_pos = torch.arange(seq_len)
                lifespan = last_usage_pos - key_pos
                
                # Binary Target: 1 if lifespan > 256, else 0
                targets = (lifespan > LONG_TERM_THRESHOLD).float()
                
                k_vectors_np = k_vectors.numpy()
                targets_np = targets.numpy()
                
                key_id = f"l{layer_idx}_h{head_idx}"
                if key_id not in keys_storage:
                    keys_storage[key_id] = []
                    labels_storage[key_id] = []
                    
                keys_storage[key_id].append(k_vectors_np)
                labels_storage[key_id].append(targets_np)
        
        sample_count += 1
        
        if sample_count % 100 == 0:
            save_shards(keys_storage, labels_storage, args.output_dir, sample_count)
            keys_storage = {}
            labels_storage = {}

    if any(keys_storage):
         save_shards(keys_storage, labels_storage, args.output_dir, sample_count)

def save_shards(keys, labels, out_dir, step):
    print(f"Saving checkpoint at step {step}...")
    for key_id in keys:
        k_data = np.concatenate(keys[key_id], axis=0)
        l_data = np.concatenate(labels[key_id], axis=0)
        
        save_dict = {
            "inputs": torch.from_numpy(k_data),
            "targets": torch.from_numpy(l_data)
        }
        
        head_dir = os.path.join(out_dir, key_id)
        os.makedirs(head_dir, exist_ok=True)
        
        fname = os.path.join(head_dir, f"shard_{step}.safetensors")
        save_file(save_dict, fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="data/train_binary_256")
    args = parser.parse_args()
    
    generate_data(args)


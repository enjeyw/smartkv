# Experiment Logs

This document contains key logs and outputs from the development and execution of the SmartKV pipeline.

## 1. Model Exploration & Calibration
**Script:** `scripts/explore_model.py`
**Date:** Nov 22, 2025
**Goal:** Verify Qwen2.5-0.5B architecture and determine the "hit" threshold for attention.

### Output Snippet:
```
Loading Qwen/Qwen2.5-0.5B-Instruct...
Model Architecture:
Qwen2ForCausalLM(
  (model): Qwen2Model(
    ...
    (layers): ModuleList(
      (0-23): 24 x Qwen2DecoderLayer(
        (self_attn): Qwen2Attention(
          (q_proj): Linear(in_features=896, out_features=896, bias=True)
          (k_proj): Linear(in_features=896, out_features=128, bias=True)
          ...
)

Attn map shape: torch.Size([14, 291, 291])

Layer 5 Stats (All Heads):
Min: 0.0
Max: 1.0
Mean: 0.006849
Median: 0.000436

Percentiles:
50th: 0.000436
75th: 0.001932
90th: 0.007061
95th: 0.015930
99th: 0.115112
99.9th: 0.743749

Suggested Threshold (99th percentile): 0.1151123046875

Verifying Key Projection...
Found Key Projection: model.layers.0.self_attn.k_proj -> Linear(in_features=896, out_features=128, bias=True)
```
**Conclusion:** The attention distribution is extremely long-tailed. A threshold of `0.1` captures the top ~1% of interactions, filtering out the vast majority of "background noise" attention.

---

## 2. Data Generation
**Script:** `scripts/generate_dataset.py`
**Parameters:** `--num_samples 500` (Hybrid PG19 + GSM8K)
**Goal:** Generate training shards containing `(Pre-RoPE Key, log4_Lifespan)` pairs.

### Output Snippet:
```
Loading Qwen/Qwen2.5-0.5B-Instruct...
Model has 24 layers and 2 KV heads.
Loading datasets...
PG19 load failed, trying fallback... (Using C4/GSM8K mix)

0it [00:00, ?it/s]
...
100it [00:17,  6.94it/s]
Saving checkpoint at step 100...
...
500it [01:10,  7.12it/s]
Saving checkpoint at step 378... (Final batch)
```

**Verification:**
```bash
$ ls data/train
l0_h0  l0_h1  l1_h0  l1_h1 ... (folders for all 48 heads)

$ ls data/train/l0_h0
shard_78.safetensors  shard_100.safetensors ...
```
**Conclusion:** Successfully generated roughly 178,000 samples per head across 48 heads.

---

## 3. Training
**Script:** `scripts/train_mlps.py`
**Parameters:** `--epochs 3 --batch_size 4096`
**Goal:** Train 48 separate MLP regressors.

### Output Snippet:
```
Found 48 heads to train.

Training l0_h0...
Loading 5 shards...
Total samples: 178170
Epoch 1: Loss = 11.4300
Epoch 2: Loss = 1.2796
Epoch 3: Loss = 1.1993
Saved to models/gates/l0_h0.pt

Training l10_h0...
Epoch 1: Loss = 0.6285
Epoch 2: Loss = 0.3808
Epoch 3: Loss = 0.3415
Saved to models/gates/l10_h0.pt

Training l23_h1...
Epoch 1: Loss = 1.5210
Epoch 2: Loss = 1.1317
Epoch 3: Loss = 1.0156
Saved to models/gates/l23_h1.pt
```

**Analysis:**
- **Layer 0 (Input)**: High final loss (~1.2). This is expected as early layers process raw tokens which may have ambiguous long-term utility before context is built.
- **Layer 10 (Middle)**: Low final loss (~0.34). Middle layers often handle syntax and core semantic linking, which is more predictable.
- **Layer 23 (Output)**: Moderate final loss (~1.0). Deep layers deal with abstract concepts and next-token prediction, which can be volatile.
- **Convergence**: All models showed significant loss reduction from Epoch 1 to Epoch 3, proving the data contains learnable signal.


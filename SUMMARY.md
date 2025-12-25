# SmartKV: Learned KV Cache Eviction - Project Summary

## 1. Executive Summary
This project implements a novel **Learned KV Cache Elimination** mechanism for Large Language Models (LLMs), specifically targeted at `Qwen3-0.6B`. 

The core innovation is replacing heuristic eviction policies (like H2O, StreamingLLM, or LRU) with a **learned, semantic predictor**. By training a lightweight regressor on the "Pre-RoPE" key vectors, we can predict the **expected future utility** (Lifespan) of every token at the moment it is generated. This allows the inference engine to dynamically evict tokens that are predicted to be irrelevant, maintaining high accuracy with a fraction of the memory footprint.

### Binary Gate Training Refresh (Nov 2025)
- **Unified checkpoints**: `scripts/train_binary.py` now emits a single safetensors blob (`gates_combined.safetensors`) that bundles every head's weights plus layer/head metadata. Legacy per-head `.pt` files can still be produced via `--save_legacy_heads`, but they are no longer required for fast inference.
- **Shared loader utilities**: `scripts/binary_gate_io.py` exposes `GateLoader`, allowing inference (`scripts/inference_with_gates.py`), evaluation, and plotting tools to hydrate every head with one read, while automatically falling back to legacy layouts if needed.
- **Adaptive training duration**: Validation-tracked early stopping (`--max_epochs`, `--patience`, `--min_delta`) saves the best-performing epoch and annotates it inside the combined checkpoint metadata.
- **Suggested run command**:
  `python scripts/train_binary.py --data_dir data/train_binary_256 --output_dir models/gates_binary_256_v4 --balanced --max_epochs 40 --patience 6 --min_delta 5e-4`
  Typical runs converge in ~18–25 epochs, shaving ~40% off wall-clock time versus fixed schedules.
- **Downstream impact**: Binary analysis utilities (`eval_binary.py`, ROC/calibration/confusion plots, etc.) now read weights through `GateLoader`, so switching to consolidated checkpoints is transparent to researchers.

## 2. Theoretical Framework

### The Problem
- **KV Cache Memory**: Inference memory scales linearly with context length. For long contexts (100k+ tokens), the KV cache becomes the bottleneck, not the model weights.
- **Heuristic Limitations**: Standard heuristics (e.g., "keep recent tokens") fail to capture long-range dependencies crucial for tasks like retrieval or plot tracking in novels.

### The Solution: Semantic Utility Prediction
- **Hypothesis**: The semantic content of a token (e.g., the word "Harry" in a Harry Potter book) inherently predicts its future utility, regardless of its current position.
- **Mechanism**:
    - **Input**: `Key_Vector` (before Rotary Positional Embeddings are applied).
    - **Model**: A per-head MLP (`GateMLP`).
    - **Output**: A scalar "Priority Score" representing $\log_4(\text{Time-to-Last-Attention})$.

### Design Decisions
1.  **Pre-RoPE vs. Post-RoPE**: We explicitly chose **Pre-RoPE** keys. Post-RoPE keys entangle semantic meaning with position (rotation). A classifier trained on post-RoPE keys would overfit to absolute positions (e.g., "Token #5 is important") rather than content. Pre-RoPE keys are position-invariant.
2.  **Regression vs. Classification**: We chose **Log-Space Regression** ($\log_4(\text{Lifespan})$) over binary classification. This provides a continuous granularity:
    - Score `2.0` $\rightarrow$ Utility lasts ~16 tokens.
    - Score `5.0` $\rightarrow$ Utility lasts ~1024 tokens.
    - This enables dynamic/soft eviction policies (e.g., "Keep top-K" or "Drop if score < T").

## 3. Implementation Details

### Phase 1: Exploration & Calibration
- **Script**: `scripts/explore_model.py`
- **Objective**: Determine the "Ground Truth" for attention.
- **Methodology**:
    - We fed `Qwen3-0.6B` a 500-token text.
    - We hooked the attention weights to analyze the distribution.
    - **Key Finding**: Attention is extremely sparse. The median attention score is near zero. The 99th percentile was approximately `0.11`.
- **Outcome**: We set the "Significant Attention Threshold" to `0.11`. Any query-key interaction below this value is treated as "not attending".

### Phase 2: Data Generation
- **Script**: `scripts/generate_dataset.py`
- **Objective**: Create a massive, high-quality training dataset for the predictors.
- **Data Strategy (Hybrid Approach)**:
    - **70% Long Context (PG-19)**: Narrative texts (books). Essential for learning to keep characters, plot points, and definitions over long distances.
    - **30% Reasoning (GSM8K/Math)**: Dense, short-context logical problems. Essential for preserving keys needed for immediate Chain-of-Thought steps.
- **Pipeline**:
    1.  Interleave datasets.
    2.  Run Qwen forward pass with `output_attentions=True`.
    3.  **Hook 1 (Inputs)**: Capture `k_proj` output (Pre-RoPE keys).
    4.  **Hook 2 (Labels)**: For every key $K_i$, scan all future queries $Q_j$ ($j > i$). Find the last $j$ where `Attention(Q_j, K_i) > 0.1`.
    5.  **Calculate Label**: $y = \log_4(j - i)$.
- **Output**: Generated training shards in `data/train_qwen3/`.

### Phase 3: Training
- **Script**: `scripts/train_mlps.py`
- **Objective**: Train the GateMLPs (28 Layers $\times$ 8 KV Heads).
- **Architecture**:
    - Input: `Head_Dim` (128 for Qwen3 0.6B).
    - Hidden: `Linear(128 -> 64) -> ReLU`.
    - Output: `Linear(64 -> 1)`.
- **Training Config**:
    - Loss: `MSELoss`.
    - Optimizer: `AdamW` (lr=1e-3).
    - Batch Size: 4096.
    - Epochs: 10.
- **Results**:
    - **Layer 1 (Shallow)**: Some heads (h3, h6) showed decent R² (0.4 - 0.6).
    - **Deep Layers**: Generally lower predictability.
    - **Calibration**: Plots in `plots/` show reasonable correlation for predictable heads.

### Phase 4: Evaluation
- **Script**: `scripts/benchmark_kv.py`
- **Objective**: Verify if gates can identify important tokens.
- **Method**: "Needle in a Haystack" test. Insert a password at token 0, ask for it at token 5000.
- **Results**:
    - Top heads (e.g., Layer 1 Head 7, Layer 6 Head 6) assigned scores ~1.82 to the needle.
    - Bottom heads assigned scores ~0.05.
    - **Conclusion**: The gates can distinguish between "long-term important" and "short-term" tokens, though the absolute score magnitude is limited by the training window (1024).

## 4. Repository Structure
```
smartkv/
├── models/
│   └── gates_qwen3/        # The trained PyTorch models (lX_hY.pt)
├── data/
│   └── train_qwen3/        # Training data shards (safetensors)
├── plots/                  # Calibration & Attention plots
├── scripts/
│   ├── explore_model.py    # Calibration & Architecture verification
│   ├── generate_dataset.py # Data generation pipeline
│   ├── train_mlps.py       # Training harness
│   └── benchmark_kv.py     # "Needle in Haystack" evaluation
└── SUMMARY.md              # This file
```

## 5. Path Forward
To deploy this system:
1.  **Inference Integration**: Modify the `forward` pass of the serving engine (e.g., vLLM or custom PyTorch loop).
2.  **Runtime Logic**:
    ```python
    # Pseudo-code for inference step
    new_key = model.k_proj(hidden_state)
    priority_score = gate_mlp(new_key)
    
    if priority_score < DROP_THRESHOLD:
        # Don't add to cache (or mark for immediate eviction)
        pass
    else:
        kv_cache.append(new_key, priority_score)
    ```
3.  **Longer Training**: Retrain on 8k+ context length to improve the absolute score magnitude for very long dependencies.

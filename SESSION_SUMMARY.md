# Learned Gated KV Cache: Session Summary

## 1. Overview
The goal of this session was to implement a **Learned Gated KV Cache** eviction strategy for the Qwen3-0.6B model. Instead of heuristic eviction (like H2O or standard sliding window), we trained binary classifiers ("gates") for every attention head to predict whether a token needs to be retained in the Key-Value (KV) cache based on its attention scores.

## 2. Training Pipeline

### Architecture Evolution
We moved from a small legacy architecture to a more robust MLP to improve predictive capacity:
- **Input:** Head Key Vector (Dim: 128)
- **Model:** 3-Layer MLP
  - Linear (128 $\to$ 256) $\to$ ReLU
  - Linear (256 $\to$ 256) $\to$ ReLU
  - Linear (256 $\to$ 1) $\to$ Sigmoid
- **Total Gates:** 224 separate models (28 layers $\times$ 8 KV heads).

### Data & Balancing
- **Challenge:** The dataset was highly imbalanced (~2% of tokens are "important" enough to keep).
- **Solution:** Implemented **Oversampling** in `BatchedKeyDataset`.
  - We ensure each batch contains roughly 50/50 split of positive and negative samples.
  - Removed `pos_weight` from Loss function in favor of balanced sampling to stabilize gradients.

### Results
- **Training:** Successfully trained all 224 heads (Layers 0-27).
- **Evaluation:**
  - **Loss:** Validation loss tracks training loss closely, indicating minimal overfitting.
  - **ROC/AUC:** Most heads achieve AUC > 0.95, showing excellent discrimination ability.
  - **Calibration:** Models are well-calibrated (predicted probability matches actual retention rate).
  - **Confusion Matrices:** At threshold 0.1, we achieve high Recall (keeping most important tokens) with reasonable Precision.

## 3. Inference Architecture

The core of the implementation is the `SmartKVInferencer`, designed to run efficiently in Python without custom CUDA kernels.

### Strategy: Sink + Window + Learned Gates
The eviction policy for the KV cache consists of three safe-guards:
1.  **Attention Sink:** Always keep the first $N$ tokens (e.g., 4) to preserve anchoring.
2.  **Sliding Window:** Always keep the most recent $W$ tokens (e.g., 256) to maintain local context.
3.  **Learned Gated Retention:**
    - As a token "slides out" of the window (at index `current_len - window_size`), it is not immediately discarded.
    - We feed its Key vector into the learned gates.
    - If the gates deem it important, it is moved to the "long-term" cache (retained). Otherwise, it is evicted.

### Optimization: Batched Gate Inference
Running 224 small MLPs sequentially in a Python loop is prohibitively slow. We implemented a **Batched Gate MLP**:

- **Tensorized Weights:** All 224 models are stacked into single tensors:
  - `W1`: shape `[224, 128, 256]`
  - `W2`: shape `[224, 256, 256]`
  - `W3`: shape `[224, 256, 1]`
- **Single Pass Execution:**
  - We extract the candidate token's Key vector from all layers/heads: Input shape `[1, 224, 128]`.
  - We use Batch Matrix Multiplication (`torch.bmm`) to compute scores for all 224 heads simultaneously.
  - **Result:** Inference overhead for gating is negligible compared to the model forward pass.

### Policy: Global Union (Uniform Pruning)
Standard HuggingFace models (and FlashAttention) expect the KV cache to be rectangular (same sequence length across all layers). Pruning different tokens in different layers ("Ragged Cache") breaks compatibility.

To solve this, we implemented a **Global Union Policy**:
1.  Compute retention scores for all 224 heads.
2.  **Aggregation:** If **ANY** head in **ANY** layer wants to keep the token (score > threshold), it is marked as "globally important".
3.  **Action:**
    - **Keep:** The token is retained in **ALL** layers.
    - **Drop:** The token is evicted from **ALL** layers.

This guarantees a valid, rectangular cache structure while still allowing compression for tokens that are universally regarded as noise.

## 4. Current Status
- **Scripts:**
  - `train_binary.py`: Fully functional training script with balancing.
  - `inference_with_gates.py`: Functional inference script with Batched Gates and Qwen3 compatibility fixes.
  - `plot_*.py`: Suite of visualization tools (ROC, Calibration, Confusion Matrix).
- **Performance:**
  - Inference runs without crashing.
  - Generation quality is coherent on real text inputs.
  - Gate loading is efficient.
  - **Compression:** The "Global Union Policy" is currently very conservative (keeps nearly all tokens) to ensure correctness. Achieving high compression rates with this strict policy will require significantly higher thresholds (e.g., > 0.9) or a different model architecture that supports ragged caches.

## 5. Future Directions
- **Threshold Tuning:** Experiment with higher thresholds (0.5 - 0.95) to encourage more aggressive pruning.
- **Alternative Policies:** Investigate "Per-Layer Union" (requires modifying the model to handle ragged caches or using masking).
- **Long-Context Benchmarks:** Run "Needle in a Haystack" to verify retention of distant critical information.

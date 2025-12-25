Based on the discussion, I will implement the KV Cache Elimination system for Qwen2.5-0.5B.

### Goal
Train a per-head MLP to predict the **log-lifespan** of a KV cache entry.

### Implementation Steps

#### 1. Exploration & Calibration
- **File**: `scripts/explore_model.py`
- **Action**:
    - Load `Qwen/Qwen2.5-0.5B-Instruct`.
    - Feed a sample text (~500 tokens).
    - **Identify Hook Points**: Locate exactly where to intercept `keys` *before* RoPE.
    - **Calibrate Threshold**: Analyze the distribution of attention scores to determine what counts as a "hit".

#### 2. Data Generation
- **File**: `scripts/generate_dataset.py`
- **Action**:
    - **Input**: Hybrid dataset (PG19 + Math).
    - **Process**:
        1. Interleave samples.
        2. Run model.
        3. **Capture**: `Pre-RoPE Keys` & `Attention Matrix`.
        4. **Labeling Logic**:
            - Calculate `Lifespan` = `Last_Future_Attn_Pos` - `Current_Pos`.
            - **Regression Target**: $y = \log_4(\max(\text{Lifespan}, 1))$.
            - Example: Lifespan 16 $\rightarrow$ 2.0. Lifespan 1024 $\rightarrow$ 5.0.
        5. **Storage**: Save `(Pre_RoPE_Key, Scalar_Target)` pairs.

#### 3. Training
- **File**: `scripts/train_mlps.py`
- **Action**:
    - Define `GateMLP`: `Linear(HeadDim -> 64) -> ReLU -> Linear(64 -> 1)`.
    - Load data shards.
    - Train with **MSELoss** or **HuberLoss**.
    - Save checkpoints.

### To-dos
- [ ] Explore Qwen2.5-0.5B structure and attention values to determine thresholds and hook points
- [ ] Create data generation script capturing Pre-RoPE keys and log-lifespan targets
- [ ] Generate training dataset from Hybrid sources
- [ ] Implement and train MLP regressors for each head

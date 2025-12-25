# SmartKV: Learned KV Cache Eviction

## Objective
The goal of this project is to develop a learned mechanism for Key-Value (KV) Cache eviction to reduce the memory footprint of Large Language Models (LLMs) during inference.

## Core Hypothesis
Semantic information contained in the Key vector itself is sufficient to predict the **expected lifespan** of that key. By training a small regressor on the "Pre-RoPE" (semantic) keys, we can predict how long a key will remain relevant and evict those with short predicted lifespans.

## Implementation Strategy

### 1. Model Architecture
**Qwen2.5-0.5B** (GQA + RoPE).

### 2. The Predictor (GateMLP)
For each attention head:
- **Input**: The `Key` vector (before RoPE application).
- **Architecture**: `Linear(HeadDim -> 64) -> ReLU -> Linear(64 -> 1)`.
- **Output**: Scalar prediction of $\log_4(\text{Lifespan})$.
    - Prediction `2.0` means $4^2 = 16$ tokens of lifespan.
    - Prediction `5.0` means $4^5 = 1024$ tokens of lifespan.

### 3. Data Strategy (The Hybrid Approach)
- **PG-19 (70%)**: Long-context narratives.
- **Math/Reasoning (30%)**: GSM8K, MATH-500, AIME24.

### 4. Trade-offs & Decisions

#### Regression vs. Classification
- **Decision**: We regress on $\log_4(\text{Lifespan})$.
- **Reasoning**: 
    - Provides a continuous "priority score" for eviction.
    - Captures the ordinal nature of time (100 is closer to 200 than 1).
    - Simpler architecture (scalar output).

#### Why Pre-RoPE Keys?
- **Decision**: Feed Key vector *before* Rotary Embeddings.
- **Reasoning**: Ensures the predictor learns semantic importance ("Python is important") rather than positional importance ("Token #5 is important").

## Execution Plan

### Phase 1: Exploration & Calibration
- **Script**: `scripts/explore_model.py`
- **Status**: In Progress.

### Phase 2: Data Generation
- **Script**: `scripts/generate_dataset.py`
- **Target**: `y = log4(lifespan)`.

### Phase 3: Training
- **Script**: `scripts/train_mlps.py`
- **Loss**: MSE / Huber.

### Phase 4: Evaluation
- **Script**: `scripts/benchmark_kv.py`

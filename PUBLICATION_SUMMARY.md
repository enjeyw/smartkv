# Learned Gated KV Cache: From Heuristics to Learned Sparsity

## 1. Motivation

Large Language Models (LLMs) are memory-bound during inference, primarily due to the Key-Value (KV) cache which grows linearly with sequence length. For a 7B model with a long context, the KV cache can easily exceed the size of the model weights themselves.

Standard approaches to KV cache compression (like H2O or Sliding Window) rely on **heuristics**:
- **Sliding Window:** "Only recent tokens matter." (Fails for long-range dependencies).
- **H2O (Heavy Hitters):** "Tokens that were important in the past will be important in the future." (Reactive, not predictive).

**Hypothesis:** Can we train a lightweight, per-head classifier to *predict* if a token currently in the cache will be needed for future attention, based solely on its Key vector?

## 2. Overall Concept

The architecture involves attaching a "Gate" (a small Multi-Layer Perceptron) to every attention head in the LLM.

- **Input:** The Key vector of a token (size $D_{head} = 128$).
- **Output:** A retention score.
- **Inference Policy:** 
    1.  **Attention Sink:** Always keep the first few tokens.
    2.  **Local Window:** Always keep the most recent $N$ tokens.
    3.  **Learned Gate:** For tokens sliding out of the local window, query the Gate. If `Score > Threshold`, move to long-term storage; otherwise, evict.

---

## 3. Phase 1: Predicting Retention Lifetime ($4^x$)

Our initial approach attempted to regress the "Time-To-Live" (TTL) of a token. We quantized the remaining useful life of a token into logarithmic buckets ($4^0, 4^1, 4^2, \dots$ steps).

### Why this didn't work
While conceptually appealing, this approach proved impractical:
1.  **Rigidity:** It forced the model to predict *exactly* how long a token would be needed, which is a harder task than simply knowing *if* it is needed.
2.  **Tuning Difficulty:** The output was a specific duration class. This made it impossible to tune the **Sensitivity vs. Specificity** trade-off. We couldn't easily say "be 10% more aggressive" without retraining or complex heuristic mapping of classes to probabilities.
3.  **Optimization:** The loss landscape for regression/multi-class classification on noisy attention data was difficult to converge.

---

## 4. Phase 2: Swapping to Binary Predictor

We pivoted to a **Binary Classification** formulation:
> "Will this token receive significant attention mass in the future? (Yes/No)"

### Architecture & Training
- **Model:** 3-Layer MLP (`128 -> 256 -> 256 -> 1`) with ReLU activations.
- **Loss:** `BCEWithLogitsLoss`.
- **Data Balancing:** The dataset is extremely imbalanced (~98% of tokens are noise). We implemented **Oversampling** to ensure batches were 50/50 balanced, removing the need for fragile `pos_weight` tuning.

### Training Results
The binary classifiers converged rapidly and showed excellent discrimination.

![Training Loss](plots/train_all_layers_loss.png)
*Figure 1: Training and Validation Loss for Layers 1-27. The consistent drop indicates the model effectively learned the sparsity pattern without overfitting.*

### ROC Analysis
We evaluated the Area Under the Curve (AUC) for all 224 heads. The results were remarkably strong, with most heads achieving AUC > 0.90.

![ROC Curves](plots/grid_roc/all_layers_roc.png)
*Figure 2: ROC Curves for all layers. The curves hug the top-left corner, indicating high True Positive Rate (Recall) with low False Positive Rate.*

### Calibration
It is critical that the model's output probability corresponds to the actual likelihood of retention.

![Calibration Plot](plots/grid_binary/all_heads_binary_calibration.png)
*Figure 3: Calibration plots for all heads. The alignment with the diagonal indicates the models are well-calibrated probabilities, not just uncalibrated scores.*

---

## 5. Confusion Matrices & Threshold Tuning

With a binary predictor, we gain the ability to tune the **Threshold**. This allows us to choose our operating point on the ROC curve.

### Standard Threshold (0.5)
At a standard 50% probability threshold, the model is conservative about predicting "Keep".

![Confusion Matrix 0.5](plots/confusion_matrices/layer0_cm.png)
*Figure 4: Confusion Matrix at Threshold 0.5. Note the high specificity but potentially lower recall for the minority "Keep" class.*

### Aggressive Threshold (0.1)
For KV caching, **Recall is paramount**. Dropping an important token breaks the generation, while keeping a useless one just wastes a bit of RAM. By lowering the threshold to 0.1, we drastically improve Recall.

![Confusion Matrix 0.1](plots/confusion_matrices/layer0_cm_t0.1.png)
*Figure 5: Confusion Matrix at Threshold 0.1. We successfully capture the vast majority of "Important" tokens (bottom-right quadrant is minimized) at the cost of keeping more "Trash" tokens (top-right).*

---

## 6. Inference: Challenges & Solutions

Deploying 224 learned gates into a running LLM inference loop presented unique engineering challenges.

### Challenge A: Computational Overhead
Running 224 small MLPs sequentially in Python (looping over layers and heads) creates massive CPU overhead, slowing generation to a crawl.

**Solution: Batched Gate Execution**
We stacked the weights of all 224 gates into single large tensors. This allows us to score retention for *every head in every layer* in a single `torch.bmm` (Batch Matrix Multiply) operation. The overhead became negligible (< 1ms).

### Challenge B: The "Ragged Cache" Problem
Standard attention implementations (HuggingFace `transformers`, FlashAttention) expect the KV cache to be a rectangular tensor `[Batch, Heads, Seq_Len, Dim]`.
If Layer 1 decides to keep 50 tokens, but Layer 2 decides to keep 100, the sequence lengths differ. This "Ragged Cache" is incompatible with optimized kernels.

**Solution: The Global Union Policy**
We implemented a consensus mechanism:
1.  Compute scores for all heads.
2.  If **ANY** head in **ANY** layer votes to keep a token (Score > Threshold), that token is kept in **ALL** layers.
3.  This enforces a uniform cache length across the model, satisfying implementation constraints.

While this policy is conservative (limiting compression), it guarantees safety and correctness.

### Demonstration
We verified the system using a summarization task. The model successfully retained critical information (summarizing the "Global Union Policy" correctly) even after the tokens had passed through the eviction gate.

```text
Generated Response:
The Global Union Policy is a strategy that marks a token as "globally important" if any of the heads in any layer has a retention score greater than a threshold. This ensures that the token is retained in all layers or dropped from all layers...
```

---

## 7. Conclusion

We have demonstrated that **learned sparsity** is viable. By moving from complex regression ($4^x$) to robust binary classification, we achieved high-accuracy prediction of token importance. The integration of Batched Gates and the Global Union Policy allows this to run efficiently within standard LLM pipelines.

Future work will focus on relaxing the Global Union Policy (e.g., via block-sparse masking) to unlock higher compression rates.


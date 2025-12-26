# SmartKV Cache Architecture: "Equal Size, Different Content"

## 1. Core Concept
The system maintains a fixed memory budget per layer (e.g., 512 tokens) but allows each layer to choose *which* tokens to keep based on its own learned utility function. This preserves the "rectangular" shape required for efficient batching while maximizing semantic retention.

## 2. Components

### A. The Gate Models (Per-Layer)
- **Input**: Key vector of a single token ($1 \times 128$).
- **Model**: 4-Layer MLP (256 hidden dims).
- **Output**: Predicted `log4(lifespan)` scalar.
- **Role**: Provides an "Importance Score" for every token as it enters the cache. Higher score = longer expected utility.

### B. The Smart Layer Cache (`SmartKVDynamicLayer`)
Each layer maintains its own independent cache state:

1.  **Storage**:
    - `keys`: `[Batch, Heads, Capacity, Dim]`
    - `values`: `[Batch, Heads, Capacity, Dim]`
    - *Note*: `Capacity` is fixed (Sink + Budget + Window).

2.  **Metadata**:
    - `sink_indices`: Fixed set of indices for the first $N$ tokens (Attention Sink).
    - `window_deque`: FIFO queue for the recent sliding window.
    - `high_value_indices`: Sorted list of `(score, index)` tuples, tracking the "Long Term Memory".

3.  **State Modes**:
    - **Prefill Mode (Pruning OFF)**:
        - All tokens are accepted.
        - Cache grows linearly.
        - **Why**: Causal masking relies on strict positional ordering `0..N`. Removing tokens mid-sequence breaks the triangular mask structure `(i > j)`.
    - **Generation Mode (Pruning ON)**:
        - Tokens enter one-by-one.
        - **Streaming Prune**: As a token leaves the sliding window, it competes for a spot in the "High Value" cache (unless it is a Sink token).
        - **Bulk Prune** (Transition): When switching from Prefill $\to$ Generation, we take the massive linear cache and compress it down to the Budget size *once*.

## 3. The Pruning Logic

### Bulk Prune (Post-Prefill)
Executed once after the prompt is processed.
1.  **Score**: Run Gate MLP on all $N$ prompt tokens.
2.  **Select**:
    - **Sink**: Keep first $S$ tokens unconditionally.
    - **Window**: Keep last $W$ tokens unconditionally.
    - **Budget**: From the remaining $N - (S + W)$ tokens, select top $B$ based on Gate Score.
3.  **Compact**:
    - Create new `keys/values` tensors of size $S + B + W$.
    - Copy selected tokens.
    - **Crucial**: We must update the Position IDs or attention mask to reflect that tokens are missing, OR we rely on the fact that standard attention simply ignores missing keys (as long as they are removed from the *key* tensor).

### Streaming Prune (Token-by-Token)
Executed during `layer.update()` in generation.
1.  **New Token Arrives**:
    - Calculate Score $S_{new}$.
    - Add to `Window`.
2.  **Window Overflow**:
    - Token $T_{old}$ falls out of Window (size $W$).
    - **Check Sink**: If $T_{old}$ is part of the Sink (index $< S$), it is permanently safe. Do nothing.
    - **Competition**:
        - If $S_{old} > \min(\text{HighValueScores})$ and `HighValue` is full:
            - Evict lowest scoring HighValue token.
            - Move $T_{old}$ to HighValue.
        - Else:
            - Discard $T_{old}$.
3.  **Memory Management**:
    - Since we have a fixed tensor size, "Evict" means overwriting the slot of the discarded token with the new data, or using a pointer system.
    - *Simplification*: We might just append to a buffer and periodically compact, or use a ring buffer.

## 4. Integration Point
The logic sits inside a custom `Cache` subclass (e.g., `SmartKVDynamicCache`) compatible with `model.generate()`.
- **`update()`**: Handles the streaming insertion and logic.
- **`bulk_prune()`**: New method to be called explicitly between `prefill` and `generate`.


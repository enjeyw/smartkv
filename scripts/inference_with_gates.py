import time
from typing import Optional, List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from binary_gate_io import GateLoader


# --- Batched Model Definition ---
class BatchedGateMLP(nn.Module):
    def __init__(self, num_heads, input_dim=128, hidden_dim=256):
        super().__init__()
        self.num_heads = num_heads

        # Weights: [Num_Total_Heads, Input, Hidden]
        self.W1 = nn.Parameter(torch.zeros(num_heads, input_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(num_heads, hidden_dim))
        self.W2 = nn.Parameter(torch.zeros(num_heads, hidden_dim, hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(num_heads, hidden_dim))
        self.W3 = nn.Parameter(torch.zeros(num_heads, hidden_dim, 1))
        self.b3 = nn.Parameter(torch.zeros(num_heads, 1))
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, Num_Total_Heads, Input]
        returns: [Batch, Num_Total_Heads, 1]
        """
        # Permute to [Num_Total_Heads, Batch, Input] for bmm
        x_t = x.transpose(0, 1)  # [NH, B, In]

        # Layer 1
        h = torch.bmm(x_t, self.W1) + self.b1.unsqueeze(1)
        h = self.relu(h)

        # Layer 2
        h = torch.bmm(h, self.W2) + self.b2.unsqueeze(1)
        h = self.relu(h)

        # Layer 3
        out = torch.bmm(h, self.W3) + self.b3.unsqueeze(1)

        return out.transpose(0, 1)  # [B, NH, 1]


# --- Custom Cache with gating + sinks + window ---
class SmartKVDynamicCache(DynamicCache):
    """
    DynamicCache subclass that:
      - keeps a set of sink tokens at the start
      - maintains a sliding window at the right
      - uses a learned gate to decide which tokens may be dropped
      - evicts only when tokens are outside the window
    Batch size 1 only.
    """

    def __init__(
        self,
        window_size: int,
        sink_size: int,
        gate_mlp: BatchedGateMLP,
        num_layers: int,
        num_kv_heads: int,
        threshold: float,
        device: torch.device,
    ):
        super().__init__()
        self.window_size = window_size
        self.sink_size = sink_size
        self.threshold = threshold
        self.device = device

        self.gate_mlp = gate_mlp
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.total_heads = num_layers * num_kv_heads

        # Per-token flags aligned with cached positions.
        # keep_flags[i] in {None, True, False}
        #   None  -> gate not run yet
        #   True  -> important, never drop (even outside window)
        #   False -> unimportant, can be dropped when outside window
        self.keep_flags: List[Optional[bool]] = []

    # --- helper methods ---

    def _ensure_keep_flags_len(self, seq_len: int):
        """Extend keep_flags to match current seq length."""
        while len(self.keep_flags) < seq_len:
            idx = len(self.keep_flags)
            if idx < self.sink_size:
                # sinks are always logically "keep"
                self.keep_flags.append(True)
            else:
                self.keep_flags.append(None)

        # In case seq_len shrank due to pruning (should not happen here usually)
        if len(self.keep_flags) > seq_len:
            self.keep_flags = self.keep_flags[:seq_len]

    def _run_gate_for_index(self, idx: int) -> bool:
        """Run gate on token at index idx if needed, return keep decision."""
        if self.keep_flags[idx] is not None:
            return self.keep_flags[idx]

        # We assume key_cache is fully populated at this point
        if not self.key_cache or self.key_cache[0] is None:
            # No cache yet – conservative keep
            self.keep_flags[idx] = True
            return True

        # Collect K across all layers at this index
        all_k_vecs = []
        for layer_idx in range(self.num_layers):
            k_layer = self.key_cache[layer_idx]  # [B, H, L, D]
            if k_layer is None:
                continue
            # batch size assumed 1
            k_vec = k_layer[:, :, idx, :]  # [1, H, D]
            all_k_vecs.append(k_vec)

        if not all_k_vecs:
            self.keep_flags[idx] = True
            return True

        batch_input = torch.cat(all_k_vecs, dim=1)  # [1, Total_Heads, D]
        with torch.no_grad():
            scores = torch.sigmoid(self.gate_mlp(batch_input.float()))  # [1, Total_Heads, 1]

        mean_score = scores.squeeze().mean().item()
        keep = bool(mean_score > self.threshold)
        self.keep_flags[idx] = keep
        # Optional debug print
        # print(f"[Gate] idx={idx}, mean_score={mean_score:.4f}, keep={keep}")
        return keep

    def maybe_prune(self):
        """
        Apply delayed eviction:
          - assign keep_flags to tokens entering the window
          - while cache too long, drop unimportant tokens
            before the window (but after sinks)
        """
        if not self.key_cache or self.key_cache[0] is None:
            return

        k0 = self.key_cache[0]  # [B, H, L, D]
        batch, _, L, _ = k0.shape
        assert batch == 1, "SmartKVDynamicCache currently supports batch_size=1 only"

        self._ensure_keep_flags_len(L)

        # Define window
        window_start = max(self.sink_size, L - self.window_size)

        # 1. Assign gates for tokens that have just entered the window region
        for idx in range(window_start, L):
            if self.keep_flags[idx] is None:
                self._run_gate_for_index(idx)

        # 2. Prune only when longer than sinks + window
        max_len = self.sink_size + self.window_size

        # While too long and there is a region between sinks and window
        while L > max_len and window_start > self.sink_size:
            drop_idx = None

            # Look for the oldest non-sink token outside the window that is unimportant
            for i in range(self.sink_size, window_start):
                if self.keep_flags[i] is False:
                    drop_idx = i
                    break

            if drop_idx is None:
                # No unimportant tokens to drop. Either accept longer cache
                # or break. We break here to avoid deleting "important" tokens.
                break

            # Drop token at drop_idx from all layers
            for layer_idx in range(self.num_layers):
                k_layer = self.key_cache[layer_idx]
                v_layer = self.value_cache[layer_idx]
                if k_layer is None or v_layer is None:
                    continue

                # [B, H, L, D]
                k_new = torch.cat(
                    [k_layer[:, :, :drop_idx, :], k_layer[:, :, drop_idx + 1 :, :]],
                    dim=2,
                )
                v_new = torch.cat(
                    [v_layer[:, :, :drop_idx, :], v_layer[:, :, drop_idx + 1 :, :]],
                    dim=2,
                )
                self.key_cache[layer_idx] = k_new
                self.value_cache[layer_idx] = v_new

            # Drop flag and recompute lengths
            self.keep_flags.pop(drop_idx)
            L -= 1
            window_start = max(self.sink_size, L - self.window_size)

        # Final sanity alignment
        self._ensure_keep_flags_len(L)


# --- High level inferencer ---
class SmartKVInferencer:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-0.6B",
        gate_dir: str = "models/gates_binary_256_v4",
        window_size: int = 100,
        sink_size: int = 4,
        threshold: float = 0.5,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.window_size = window_size
        self.sink_size = sink_size
        self.threshold = threshold

        print(f"Loading Qwen model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        self.model.eval()

        self.num_layers = len(self.model.model.layers)
        self.num_heads_per_layer = self.model.config.num_key_value_heads
        self.head_dim = (
            self.model.config.hidden_size // self.model.config.num_attention_heads
        )
        self.total_heads = self.num_layers * self.num_heads_per_layer

        print(
            f"Model Config: Layers={self.num_layers}, "
            f"KV Heads={self.num_heads_per_layer}, HeadDim={self.head_dim}"
        )

        # Gate MLP
        print(f"Loading {self.total_heads} gate heads into Batched MLP...")
        self.batched_gate = BatchedGateMLP(
            self.total_heads, input_dim=128
        ).to(self.device)

        loader = GateLoader(gate_dir)
        loaded_count, missing = loader.fill_batched_model(
            self.batched_gate,
            self.num_layers,
            self.num_heads_per_layer,
            device=self.device,
        )
        source = "combined bundle" if loader.bundle else "per-head checkpoints"
        print(f"Loaded {loaded_count} heads from {source}.")
        if missing:
            print(f"Missing {len(missing)} heads: {', '.join(missing[:5])}...")

        self.batched_gate.eval()

    def _init_cache(self) -> SmartKVDynamicCache:
        return SmartKVDynamicCache(
            window_size=self.window_size,
            sink_size=self.sink_size,
            gate_mlp=self.batched_gate,
            num_layers=self.num_layers,
            num_kv_heads=self.num_heads_per_layer,
            threshold=self.threshold,
            device=self.device,
        )

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]  # [1, L0]

        # Initialise our custom cache
        cache = self._init_cache()

        print("Running prefill...")
        with torch.no_grad():
            # For prefill we let the model handle positions normally (no pruning yet)
            outputs = self.model(
                input_ids,
                use_cache=True,
                past_key_values=cache,
            )

        # Cache is filled now
        cache.maybe_prune()  # probably no-op right after prefill
        logits = outputs.logits
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]

        generated_ids = [input_ids, next_token]

        # Absolute position of the next token in the *original* sequence.
        # We don't let this shrink when we prune.
        abs_pos = input_ids.shape[1]

        for step in range(max_new_tokens):
            # Position for this new token: shape [1, 1]
            cache_position = torch.full(
                (1, 1),
                abs_pos,
                dtype=torch.long,
                device=self.device,
            )

            with torch.no_grad():
                outputs = self.model(
                    next_token,
                    past_key_values=cache,
                    cache_position=cache_position,
                    use_cache=True,
                )

            abs_pos += 1

            # Model has appended one step to K/V; now maybe prune
            cache.maybe_prune()

            logits = outputs.logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids.append(next_token)

            # Debugging cache length
            if step % 10 == 0:
                if cache.key_cache and cache.key_cache[0] is not None:
                    L = cache.key_cache[0].shape[2]
                    print(
                        f"Step {step}: cache_seq_len={L}, "
                        f"abs_pos={abs_pos}"
                    )

        full_seq = torch.cat(generated_ids, dim=1)[0]
        return self.tokenizer.decode(full_seq, skip_special_tokens=True)


if __name__ == "__main__":
    inferencer = SmartKVInferencer(
        threshold=0.8,
        window_size=100,
        sink_size=4,
    )

    context = """
# Learned Gated KV Cache: Session Summary

## 1. Overview
The goal of this session was to implement a **Learned Gated KV Cache** eviction strategy for the Qwen3-0.6B model. Instead of heuristic eviction (like H2O or standard sliding window), we trained binary classifiers ("gates") for every attention head to predict whether a token needs to be retained in the Key-Value (KV) cache based on its attention scores.

## 2. Training Pipeline

### Architecture Evolution
We moved from a small legacy architecture to a more robust MLP to improve predictive capacity:
- **Input:** Head Key Vector (Dim: 128)
- **Model:** 3-Layer MLP
  - Linear (128 to 256) -> ReLU
  - Linear (256 to 256) -> ReLU
  - Linear (256 to 1) -> Sigmoid
- **Total Gates:** 224 separate models (28 layers x 8 KV heads).

### Results
- **Training:** Successfully trained all 224 heads.
- **Evaluation:**
  - **Loss:** Validation loss tracks training loss closely, indicating minimal overfitting.
  - **ROC/AUC:** Most heads achieve AUC > 0.95.

## 3. Inference Architecture
The core of the implementation is the `SmartKVInferencer`.

### Policy: Global Union (Uniform Pruning)
Standard HuggingFace models expect the KV cache to be rectangular.
To solve this, we implemented a **Global Union Policy**:
1.  Compute retention scores for all 224 heads.
2.  **Aggregation:** If **ANY** head in **ANY** layer wants to keep the token (score > threshold), it is marked as "globally important".
3.  **Action:**
    - **Keep:** The token is retained in **ALL** layers.
    - **Drop:** The token is evicted from **ALL** layers.
This guarantees a valid, rectangular cache structure.
"""

    prompt = (
        f"{context}\n\nQ: Summarize the Global Union Policy described above. "
        f"And describe potential future enhancements.\nA:"
    )

    print("Starting Generation...")
    print(f"Input Length: {len(inferencer.tokenizer.encode(prompt))} tokens")

    start_t = time.time()
    output = inferencer.generate(prompt, max_new_tokens=100)
    print(f"Generation took {time.time() - start_t:.2f}s")

    if "A:" in output:
        gen_text = output.split("A:", 1)[1].strip()
    else:
        gen_text = output

    print("\nGenerated Response:\n", gen_text)

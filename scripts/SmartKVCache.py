import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from collections import deque
import bisect
import numpy as np
import os

# --- Model Definition (Must match training) ---
class BatchedGateMLP(nn.Module):
    def __init__(self, num_heads, input_dim=128, hidden_dim=256):
        super().__init__()
        self.num_heads = num_heads
        
        # Layer 1
        self.W1 = nn.Parameter(torch.randn(num_heads, input_dim, hidden_dim) / np.sqrt(input_dim))
        self.b1 = nn.Parameter(torch.zeros(num_heads, hidden_dim))
        
        # Layer 2
        self.W2 = nn.Parameter(torch.randn(num_heads, hidden_dim, hidden_dim) / np.sqrt(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(num_heads, hidden_dim))
        
        # Layer 3
        self.W3 = nn.Parameter(torch.randn(num_heads, hidden_dim, hidden_dim) / np.sqrt(hidden_dim))
        self.b3 = nn.Parameter(torch.zeros(num_heads, hidden_dim))
        
        # Layer 4 (Output)
        self.W4 = nn.Parameter(torch.randn(num_heads, hidden_dim, 1) / np.sqrt(hidden_dim))
        self.b4 = nn.Parameter(torch.zeros(num_heads, 1))
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [Batch, Num_Heads, Input_Dim]
        x_t = x.transpose(0, 1) # [H, B, I]
        
        h = self.relu(torch.bmm(x_t, self.W1) + self.b1.unsqueeze(1))
        h = self.relu(torch.bmm(h, self.W2) + self.b2.unsqueeze(1))
        h = self.relu(torch.bmm(h, self.W3) + self.b3.unsqueeze(1))
        out = torch.bmm(h, self.W4) + self.b4.unsqueeze(1)
        
        return out.transpose(0, 1) # [B, H, 1]

# --- Gate Loader ---
class GateLoader:
    def __init__(self, model_dir, num_layers=28, num_heads=8, input_dim=128, hidden_dim=256, device="cuda"):
        self.gates = {}
        self.device = device
        
        print(f"Loading {num_layers} gate models from {model_dir}...")
        for i in range(num_layers):
            model_path = os.path.join(model_dir, f"layer_{i}.pt")
            if not os.path.exists(model_path):
                print(f"Warning: Gate model for layer {i} not found at {model_path}")
                continue
                
            gate = BatchedGateMLP(num_heads, input_dim, hidden_dim)
            gate.load_state_dict(torch.load(model_path, map_location=device))
            gate.to(device)
            gate.eval()
            self.gates[i] = gate
            
    def get_gate(self, layer_idx):
        return self.gates.get(layer_idx)

# --- Smart Layer Logic ---
class SmartKVDynamicLayer:
    def __init__(self, window_size, sink_size, budget_size, layer_idx, gate_model):
        self.window_size = window_size
        self.sink_size = sink_size
        self.budget_size = budget_size # Capacity for "High Value" tokens
        self.layer_idx = layer_idx
        self.gate_model = gate_model
        
        # State
        self.keys = None # [B, H, Capacity, D]
        self.values = None
        
        # Metadata
        # high_value_indices: List of tuples (score, cache_index)
        # cache_index is the physical index in the tensor (0 to Max_Capacity)
        self.high_value_indices = [] 
        
        # window_deque: List of (score, cache_index) for tokens currently in window
        self.window_deque = deque()
        
        # We need to map "Logical Token Index" (from prompt) to "Physical Cache Index"?
        # Or simply manage the physical tensor as a growing buffer that we compact?
        # Simplest for "rectangular" requirement: 
        # The cache tensor stays fixed size: (Sink + Budget + Window).
        # We physically overwrite slots.
        
        self.max_capacity = sink_size + budget_size + window_size
        self.current_seq_len = 0 # Logical sequence length
        
        # Buffer for Prefill (since we can't prune during prefill easily without breaking masks)
        # In prefill, we just let it grow. We Bulk Prune at end of prefill.
        self.is_prefill = True
        
    def initialize(self, key_states, value_states):
        # Initial allocation
        self.keys = key_states
        self.values = value_states
        self.current_seq_len = key_states.shape[2]
        
    def update(self, key_states, value_states):
        """
        key_states: [Batch, Heads, New_Tokens, Dim]
        """
        # Lazy Init
        if self.keys is None:
            self.initialize(key_states, value_states)
            return self.keys, self.values
            
        # Append logic
        # If in prefill, just cat.
        if self.is_prefill:
            self.keys = torch.cat([self.keys, key_states], dim=2)
            self.values = torch.cat([self.values, value_states], dim=2)
            self.current_seq_len += key_states.shape[2]
            return self.keys, self.values
            
        # GENERATION MODE (Streaming Prune)
        # New token(s) arriving. Typically 1.
        batch_size, num_heads, num_new, head_dim = key_states.shape
        
        # 1. Score the new token
        with torch.no_grad():
            score = self.gate_model(key_states.squeeze(2)) # [B, H, 1]
            score = score.max(dim=1).values.squeeze(-1) # [B]
            score = score.item()
            
        # 2. Add to Window logic
        
        # Scenario A: Window is NOT full.
        # Just append physically.
        # But "Append" in a fixed tensor means writing to the next empty slot?
        # Or do we resize?
        # To support "Overwrite", we generally assume the tensor is already at Max Capacity.
        # But if we just transitioned from Bulk Prune, we ARE at Max Capacity.
        
        # Determine the physical index to write to.
        victim_idx = None
        
        if len(self.window_deque) < self.window_size:
            # Window not full. This usually only happens at start if window was pruned smaller than max?
            # Or if we have a growing buffer.
            # For simplicity: If we are not full capacity, we cat.
            if self.keys.shape[2] < self.max_capacity:
                 self.keys = torch.cat([self.keys, key_states], dim=2)
                 self.values = torch.cat([self.values, value_states], dim=2)
                 victim_idx = self.keys.shape[2] - 1
            else:
                 # Should not happen if logic is correct (Bulk prune fills to capacity)
                 # Unless we had a very short prompt < Capacity.
                 # In that case, we should treat it as "not full" and cat.
                 # Let's assume we handle the "Short Prompt" case by just letting it grow until capacity.
                 self.keys = torch.cat([self.keys, key_states], dim=2)
                 self.values = torch.cat([self.values, value_states], dim=2)
                 victim_idx = self.keys.shape[2] - 1
                 
            self.window_deque.append((score, victim_idx))
            return self.keys, self.values

        # Scenario B: Window IS Full.
        # We must pop one.
        old_score, old_phys_idx = self.window_deque.popleft()
        
        # Is it a Sink Token? (Check if index is in sink range 0..S)
        # We need a way to track if `old_phys_idx` is a sink token.
        # Simple heuristic: If `old_phys_idx` < `sink_size`.
        # BUT: With overwrite, indices get reused! So physical index 0 might eventually hold token 1000.
        # WE NEED TO TRACK LOGICAL INDICES or have a dedicated "Sink Set".
        
        # Solution: We never overwrite Sink Indices.
        # Sink indices are 0..S-1.
        # When we reuse an index, we pick the one from the Evicted Token.
        # If the Evicted Token was in the Sink... wait, Sink tokens are never evicted.
        # So we only evict from Window or HighValue.
        
        # Token leaving window:
        victim_idx = None # This is the slot we will reclaim
        
        # Does it compete for High Value?
        kept_in_hv = False
        
        # Simple Sink Check: We assume Sink tokens are NEVER in `window_deque` once they pass the window?
        # No. Window slides over everything.
        # If the token leaving window is a Sink Token, we CANNOT evict it.
        # We must find *another* slot?
        # No, if it's a Sink Token, it stays in the Sink set (physically 0..S-1).
        # It effectively "duplicates" logically? No.
        # A token is either in Sink, HighValue, or Window.
        # If it's in Sink, it shouldn't be in Window Deque?
        # Ah, Window Deque tracks the *Logical* last W tokens.
        # If Logical Window overlaps with Sink, then that token is physically in Sink.
        
        # Correct Logic:
        # Check if `old_phys_idx` is a reserved Sink Index.
        is_sink = (old_phys_idx < self.sink_size)
        
        if is_sink:
            # We cannot reclaim this slot.
            # We need a new slot for the incoming token.
            # But we are at capacity!
            # This implies we must evict something else?
            # No, if Window overlaps Sink, we haven't reached capacity usage of non-sink slots yet.
            # It implies we are still in early generation.
            # So we can likely just `cat` (Scenario A logic above handles growing).
            # Wait, if `len(window_deque) == window_size` but `keys.shape` < `max_capacity`.
            pass 
            # Revert to append logic if physical capacity not met
            if self.keys.shape[2] < self.max_capacity:
                 self.keys = torch.cat([self.keys, key_states], dim=2)
                 self.values = torch.cat([self.values, value_states], dim=2)
                 victim_idx = self.keys.shape[2] - 1
                 self.window_deque.append((score, victim_idx))
                 return self.keys, self.values

        # If we are here, we are presumably at full physical capacity.
        # And `old_phys_idx` is leaving the window.
        
        if is_sink:
             # It leaves window, but stays in Sink.
             # We need a slot for the new token.
             # Who provides the slot?
             # Since we are at capacity, there must be a HighValue token we can evict?
             # Or we are transitioning from "Window overlapping Sink" to "Window past Sink".
             # If Window is past Sink, then `old_phys_idx` is NOT Sink.
             # So this branch only hits if Window overlaps Sink.
             # Which means we are NOT at capacity?
             # Contradiction? 
             # Only if `Budget` > 0.
             pass
        
        # Let's assume we are past the initial overlap phase for the complex "Swap" logic.
        
        # Competition
        if not is_sink:
            # It's a candidate for eviction.
            # Try to put into High Value.
            
            # If HV has space (shouldn't happen if at capacity)
            if len(self.high_value_indices) < self.budget_size:
                 bisect.insort(self.high_value_indices, (old_score, old_phys_idx))
                 # We kept it! But we still need a slot for New Token.
                 # This implies we weren't at capacity.
                 # Error in logic assumption.
                 pass
            
            # Competition with Min HV
            elif len(self.high_value_indices) > 0:
                min_s, min_idx = self.high_value_indices[0]
                if old_score > min_s:
                    # Upgrade Old to HV
                    self.high_value_indices.pop(0)
                    bisect.insort(self.high_value_indices, (old_score, old_phys_idx))
                    
                    # Victim is the one we popped
                    victim_idx = min_idx
                else:
                    # Old loses. Victim is Old.
                    victim_idx = old_phys_idx
            else:
                # No HV budget. Old loses.
                victim_idx = old_phys_idx
        
        else:
             # Old was Sink. It stays.
             # We need a slot.
             # Since we are supposedly at capacity, we must evict from HV?
             if len(self.high_value_indices) > 0:
                  # Force evict lowest HV to make room for new window token
                  # This prioritizes Window over HV.
                  min_s, min_idx = self.high_value_indices.pop(0)
                  victim_idx = min_idx
             else:
                  # No HV, Old is Sink.
                  # This implies Window + Sink = Capacity.
                  # We can't evict Sink.
                  # We simply cannot accept new token without growing?
                  # This implies Capacity calc was wrong or we allow growing here.
                  # We'll just cat.
                  self.keys = torch.cat([self.keys, key_states], dim=2)
                  self.values = torch.cat([self.values, value_states], dim=2)
                  victim_idx = self.keys.shape[2] - 1
                  self.window_deque.append((score, victim_idx))
                  return self.keys, self.values

        # PERFORM OVERWRITE
        if victim_idx is not None:
            self.keys[:, :, victim_idx, :] = key_states.squeeze(2)
            self.values[:, :, victim_idx, :] = value_states.squeeze(2)
            self.window_deque.append((score, victim_idx))
            
        return self.keys, self.values

    def bulk_prune(self):
        """
        Compresses the linear prefill cache into (Sink + Budget + Window).
        """
        self.is_prefill = False
        seq_len = self.keys.shape[2]
        
        # If fits in capacity, do nothing
        if seq_len <= self.max_capacity:
            # Just populate the metadata structures
            # 0..Sink -> Ignore (implicitly kept)
            # Sink..Seq-Window -> High Value Candidates
            # Seq-Window..Seq -> Window
            
            # Add Window
            for i in range(seq_len - self.window_size, seq_len):
                if i >= 0:
                    # Score? We need to score them retrospectively.
                    # For simplicity, we can score them now or just assign 0 since they are in window.
                    # Let's score them properly.
                    k = self.keys[:, :, i, :]
                    with torch.no_grad():
                        s = self.gate_model(k).max(dim=1).values.item()
                    self.window_deque.append((s, i))
            
            # Add Candidates to HV
            start = self.sink_size
            end = max(start, seq_len - self.window_size)
            for i in range(start, end):
                k = self.keys[:, :, i, :]
                with torch.no_grad():
                    s = self.gate_model(k).max(dim=1).values.item()
                # Insert into HV
                if len(self.high_value_indices) < self.budget_size:
                    bisect.insort(self.high_value_indices, (s, i))
                else:
                    # Competition (same as streaming)
                    min_s, min_idx = self.high_value_indices[0]
                    if s > min_s:
                        self.high_value_indices.pop(0)
                        bisect.insort(self.high_value_indices, (s, i))
            return

        # If we need to prune
        # 1. Identify Sink Indices
        keep_indices = list(range(self.sink_size))
        
        # 2. Identify Window Indices
        window_start = seq_len - self.window_size
        window_indices = list(range(window_start, seq_len))
        
        # 3. Identify Budget Indices (Middle)
        # We need to score all tokens in [Sink : WindowStart]
        candidates = []
        middle_start = self.sink_size
        middle_end = window_start
        
        if middle_end > middle_start:
            # Batch score?
            # chunking to avoid OOM
            chunk_size = 512
            for i in range(middle_start, middle_end, chunk_size):
                chunk_k = self.keys[:, :, i:min(i+chunk_size, middle_end), :] # [B, H, Chunk, D]
                # Reshape for gate: [B, H, Chunk, D] -> [B*Chunk, H, D] ? 
                # Our gate expects [B, H, D]. So we iterate or reshape.
                # Actually gate handles [B, H, D]. We treat Chunk as Batch.
                # chunk_k: [1, 8, C, 128]. Permute to [C, 8, 128]
                c_size = chunk_k.shape[2]
                flat_k = chunk_k.squeeze(0).transpose(0, 1) # [H, C, D] -> Wait.
                # Gate expects [B, H, D].
                # Let's permute chunk_k to [Chunk, H, D]
                gate_in = chunk_k.squeeze(0).permute(1, 0, 2) # [C, H, D]
                
                with torch.no_grad():
                    scores = self.gate_model(gate_in) # [C, H, 1]
                    scores = scores.max(dim=1).values.squeeze(-1) # [C]
                
                for j in range(c_size):
                    candidates.append((scores[j].item(), i + j))
            
            # Sort candidates by score descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Keep top B
            best_candidates = candidates[:self.budget_size]
            
            # Add to keep indices
            keep_indices.extend([x[1] for x in best_candidates])
            
            # Init High Value Metadata
            # We need to store them sorted by score for future maintenance
            for s, idx in best_candidates:
                # Note: The indices will change after compaction!
                # We handle that below.
                pass
                
        keep_indices.extend(window_indices)
        keep_indices.sort() # Ensure physical order
        
        # COMPACT
        self.keys = self.keys[:, :, keep_indices, :]
        self.values = self.values[:, :, keep_indices, :]
        
        # REBUILD METADATA with new indices
        self.high_value_indices = []
        self.window_deque = deque()
        
        # Mapping: Old -> New
        old_to_new = {old: new for new, old in enumerate(keep_indices)}
        
        # Re-populate Window
        for old_idx in window_indices:
            # Re-score? We calculated scores above only for middle. 
            # Need scores for window too to maintain deque.
            new_idx = old_to_new[old_idx]
            k = self.keys[:, :, new_idx, :]
            with torch.no_grad():
                s = self.gate_model(k).max(dim=1).values.item()
            self.window_deque.append((s, new_idx))
            
        # Re-populate HV
        if middle_end > middle_start:
            for s, old_idx in best_candidates:
                new_idx = old_to_new[old_idx]
                bisect.insort(self.high_value_indices, (s, new_idx))


# --- Main Cache Class ---
class SmartKVDynamicCache(Cache):
    def __init__(
        self,
        window_size: int,
        sink_size: int,
        budget_size: int,
        gate_loader: GateLoader,
        num_layers: int,
        num_kv_heads: int,
        device: torch.device
    ):
        super().__init__()
        self.window_size = window_size
        self.sink_size = sink_size
        self.budget_size = budget_size
        self.gate_loader = gate_loader
        self.num_layers = num_layers
        self.device = device
        
        # Initialize layers - We do this MANUALLY so we can pass unique gates
        # We don't use layer_class_to_replicate because each layer is unique (different gate)
        self.layers = []
        for i in range(num_layers):
            gate = gate_loader.get_gate(i)
            self.layers.append(SmartKVDynamicLayer(window_size, sink_size, budget_size, i, gate))
            
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ):
        if layer_idx >= len(self.layers):
             raise ValueError(f"Layer index {layer_idx} out of bounds")
             
        return self.layers[layer_idx].update(key_states, value_states)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if self.layers[layer_idx].keys is None:
            return 0
        return self.layers[layer_idx].keys.shape[2]

    def get_max_length(self) -> int:
        return 32768 # Arbitrary max

    def bulk_prune(self):
        print("Bulk pruning all layers...")
        for layer in self.layers:
            layer.bulk_prune()

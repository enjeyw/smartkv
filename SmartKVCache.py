import time
from typing import Optional, List, Iterable, Any

import bisect
from collections import deque
import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicLayer
import numpy as np
import os
import heapq


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
        x_t = x.transpose(0, 1)  # [H, B, I]

        h = self.relu(torch.bmm(x_t, self.W1) + self.b1.unsqueeze(1))
        h = self.relu(torch.bmm(h, self.W2) + self.b2.unsqueeze(1))
        h = self.relu(torch.bmm(h, self.W3) + self.b3.unsqueeze(1))
        out = torch.bmm(h, self.W4) + self.b4.unsqueeze(1)

        return out.transpose(0, 1)  # [B, H, 1]

class GateLoader:
    def __init__(self, model_dir, num_cache_layers=28, num_heads=8, input_dim=128, hidden_dim=256, device="mps"):
        self.gates = {}
        self.device = device

        print(f"Loading {num_cache_layers} gate models from {model_dir}...")
        for i in range(num_cache_layers):
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

class SmartKVDynamicLayer(DynamicLayer):
    def __init__(
            self,
            window_size: int,
            sink_size: int,
            cache_budget: int,
            gate_model: BatchedGateMLP
    ):
        self.window_size = window_size
        self.sink_size = sink_size
        self.cache_budget = cache_budget
        self.gate_model = gate_model

        self.high_value_indices = []
        self.window_deque = deque()
        self.prune_cache_on_update = False

        super().__init__()

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if not self.is_initialized:
            self.lazy_initialization(key_states)

        if self.prune_cache_on_update:
            num_new_tokens = key_states.shape[2]

            for idx in range(num_new_tokens):
                new_token_keys = key_states[:, :, idx, :]
                new_token_values = value_states[:, :, idx, :]
                self.process_single_token(new_token_keys, new_token_values)

            return self.keys, self.values

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)

        keys = self.keys
        values = self.values

        self.initialize_pruned_cache()

        return keys, values

    def get_importance_for_keys(self, keys):
        """
        Gets the importance for a set of keys (multiple heads)
        """
        with torch.no_grad():
            scores = self.gate_model(keys.squeeze(2))  # [B, H, 1]
        import random
        # return random.random()
        return float(scores.sum())

    def get_high_value_cache_destination_index(self, item):
        """
        heap: list[(importance, key)]
        item: (importance, key)
        """
        if len(self.high_value_indices) < self.cache_budget:
            heapq.heappush(self.high_value_indices, item)
            return None

        if item[0] > self.high_value_indices[0][0]:
            #i nsert new item into cache, drop the least important
            _, popped_index = heapq.heapreplace(self.high_value_indices, item)
            return popped_index

        # new item was least important, so don't modify cache, replace this item
        return item[1]

    def initialize_pruned_cache(self):

        self.prune_cache_on_update = True

        if len(self.window_deque) > 0:
            raise Exception("Cache already initialised")

        importances = []

        for token_index in range(self.sink_size, self.keys.shape[2]):
            keys = self.keys[:, :, token_index, :]
            importance = self.get_importance_for_keys(keys)
            importances.append((importance, token_index))


        if len(importances) < self.window_size:
            self.window_deque = deque(importances[-self.window_size:])
            return

        self.high_value_indices = heapq.nlargest(self.cache_budget, importances[:-self.window_size])

        sink_indices_to_keep = [index for index in range(0,self.sink_size)]
        hv_indices_to_keep = [index for _, index in self.high_value_indices]
        window_indices_to_keep = [index for _, index in importances[-self.window_size:]]

        all_indices_to_keep = sink_indices_to_keep + hv_indices_to_keep + window_indices_to_keep

        self.keys = self.keys[:, :, all_indices_to_keep, :]
        self.values = self.values[:, :, all_indices_to_keep, :]

        first_window_idx = len(sink_indices_to_keep) + len(hv_indices_to_keep)

        self.high_value_indices = [(score, enum_idx + self.sink_size) for enum_idx, (score, _) in enumerate(self.high_value_indices)]

        heapq.heapify(self.high_value_indices)

        self.window_deque = deque(
            [(score, enum_idx + first_window_idx) for enum_idx, (score, _) in enumerate(importances[-self.window_size:])]
        )

    def process_single_token(
            self,
            new_token_keys,
            new_token_values,
    ):
        """
        ~~~ overall process ~~~
        1. identify token that is leaving window
        2. determine whether its importance puts it inside the high value cache
        3a If so:
              - insert the index of the token into HV cache list
              - return index of token evicted from bottom of cache
              - insert new token entering window at that index
        3b else:
              - insert new token entering window at the leaving window index

        ~~~ bookkeeping datastructures ~~~
        ==> hv_cache_list
        list of tuples (gate_val, index), sorted from lowest gate_val to highest
        where
         - gate_val is how important the token is according to the gate
         - index is where the token is located in the actual KV cache

                   |--high val cache--|
        gate_val    0.4, 0.5, 0.6, 0.9
           index      5,   9,  10,   4

        ==> window_deque
        list of tuples (gate_val, index) (same as above)
        """

        new_token_keys = new_token_keys.unsqueeze(2)
        new_token_values = new_token_values.unsqueeze(2)

        if self.keys.shape[0] == 0:
            # Cache is empty
            self.keys = new_token_keys
            self.values = new_token_values
            return

        if self.keys.shape[2] > 400:
            tt = 3

        num_tokens_in_cache = self.keys.shape[2]
        if num_tokens_in_cache < self.sink_size:
            # sink isn't full yet, add straight to cache
            self.keys = torch.cat([self.keys, new_token_keys], dim=-2)
            self.values = torch.cat([self.values, new_token_values], dim=-2)
            return

        new_token_importance = self.get_importance_for_keys(new_token_keys)

        num_tokens_in_window = len(self.window_deque)

        if num_tokens_in_window < self.window_size:
            # context window isn't full yet, insert new token, with index being the position in the window
            self.keys = torch.cat([self.keys, new_token_keys], dim=-2)
            self.values = torch.cat([self.values, new_token_values], dim=-2)

            cache_index_of_new_token = self.keys.shape[2] - 1
            self.window_deque.append((new_token_importance, cache_index_of_new_token))
            return

        # remove token from window, and determine whether to put it in the hv cache, or overwrite it
        token_leaving_window = self.window_deque.popleft()

        cache_index_of_new_token = self.get_high_value_cache_destination_index(token_leaving_window)

        if cache_index_of_new_token is None:
            # budget isn't yet filled, append
            self.keys = torch.cat([self.keys, new_token_keys], dim=-2)
            self.values = torch.cat([self.values, new_token_values], dim=-2)

            cache_index_of_new_token = self.keys.shape[2] - 1
            self.window_deque.append((new_token_importance, cache_index_of_new_token))
            return

        self.window_deque.append((new_token_importance, cache_index_of_new_token))

        # [Batch, Head, Length/Index, Embedding Dim]
        self.keys[:, :, cache_index_of_new_token, :] = new_token_keys.squeeze(2)
        self.values[:, :, cache_index_of_new_token, :] = new_token_values.squeeze(2)

        return

# --- Custom Cache with gating + sinks + window ---
class SmartKVDynamicCache(Cache):
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
        cache_budget: int,
        gate_loader: GateLoader,
        num_layers: int,
        num_kv_heads: int,
        device: torch.device,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):

        super().__init__(
            layer_class_to_replicate=lambda: SmartKVDynamicLayer(window_size, sink_size, cache_budget),  # type: ignore[arg-type]
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )

        self.device = device
        self.gate_loader = gate_loader
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads

        self.layers = []
        for i in range(num_layers):
            gate = gate_loader.get_gate(i)
            self.layers.append(SmartKVDynamicLayer(window_size, sink_size, cache_budget, gate))

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # if layer_idx == 0:
        #     try:
        #         print(self.layers[layer_idx].keys.shape)
        #     except:
        #         pass

        keys, values = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

        return keys, values

    def initialize_pruned_cache(self):
        for layer in self.layers:
            layer.initialize_pruned_cache()

    def set_cache_should_prune(self, should_prune: bool):
        for layer in self.layers:
            layer.prune_cache_on_update = should_prune

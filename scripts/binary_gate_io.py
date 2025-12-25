import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file

COMBINED_FILENAME = "gates_combined.safetensors"
_WEIGHT_KEYS = ("W1", "b1", "W2", "b2", "W3", "b3")


def _parse_head_id(head_id: str) -> Tuple[int, int]:
    """Extract (layer, head) from identifiers formatted as lX_hY."""
    parts = head_id.split("_")
    if len(parts) != 2:
        raise ValueError(f"Unexpected head identifier format: {head_id}")
    layer = int(parts[0][1:])
    head = int(parts[1][1:])
    return layer, head


def _head_metadata(head_ids: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    layers, heads = zip(*(_parse_head_id(hid) for hid in head_ids))
    return (
        torch.tensor(layers, dtype=torch.int16),
        torch.tensor(heads, dtype=torch.int16),
    )


@dataclass
class GateBundle:
    weights: Dict[str, torch.Tensor]
    head_ids: List[str]
    head_layers: torch.Tensor
    head_heads: torch.Tensor
    source: str
    path: str
    _index_cache: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self._index_cache:
            self._index_cache = {hid: idx for idx, hid in enumerate(self.head_ids)}

    def head_index(self, head_id: str) -> Optional[int]:
        return self._index_cache.get(head_id)

    def iter_heads(self):
        for hid, layer, head in zip(self.head_ids, self.head_layers, self.head_heads):
            yield hid, int(layer.item()), int(head.item())

    def get_head_tensors(self, head_id: str) -> Optional[Dict[str, torch.Tensor]]:
        idx = self.head_index(head_id)
        if idx is None:
            return None
        return {k: v[idx] for k, v in self.weights.items()}


class GateLoader:
    """Loads binary gate checkpoints, preferring the consolidated bundle."""

    def __init__(
        self,
        gate_dir: str,
        prefer_combined: bool = True,
        combined_filename: str = COMBINED_FILENAME,
        device: str = "cpu",
    ):
        self.gate_dir = gate_dir
        self.device = device
        self.combined_path = os.path.join(gate_dir, combined_filename)
        self.bundle: Optional[GateBundle] = None
        self.legacy_paths: Dict[str, str] = {}
        self._legacy_cache: Dict[str, Dict[str, torch.Tensor]] = {}

        if prefer_combined and os.path.exists(self.combined_path):
            self.bundle = _load_combined_bundle(self.combined_path, device=device)

        if self.bundle is None:
            self.legacy_paths = _discover_legacy_paths(gate_dir)
            if not self.legacy_paths:
                raise FileNotFoundError(
                    f"No gate checkpoints found in {gate_dir}. "
                    f"Expected {combined_filename} or per-head .pt files."
                )

    @property
    def available_heads(self) -> List[str]:
        if self.bundle:
            return list(self.bundle.head_ids)
        return sorted(self.legacy_paths.keys())

    def has_head(self, head_id: str) -> bool:
        if self.bundle:
            return self.bundle.head_index(head_id) is not None
        return head_id in self.legacy_paths

    def get_legacy_state_dict(
        self, head_id: str, device: Optional[torch.device] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Return a state_dict with legacy keys (net.0.weight, etc.)."""
        target_device = device if device is not None else torch.device(self.device)

        if self.bundle:
            if head_id not in self.available_heads:
                return None
            cached = self._legacy_cache.get(head_id)
            if cached is None:
                head_tensors = self.bundle.get_head_tensors(head_id)
                if head_tensors is None:
                    return None
                cached = _batched_to_legacy_state(head_tensors)
                self._legacy_cache[head_id] = cached
            return {k: v.to(target_device) for k, v in cached.items()}

        path = self.legacy_paths.get(head_id)
        if path is None:
            return None
        return torch.load(path, map_location=target_device)

    def fill_batched_model(
        self,
        batched_gate: torch.nn.Module,
        num_layers: int,
        num_heads_per_layer: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[int, List[str]]:
        """Populate BatchedGateMLP parameters. Returns loaded count & missing head IDs."""
        target_device = device if device is not None else batched_gate.W1.device
        target_dtype = batched_gate.W1.dtype
        missing: List[str] = []
        loaded = 0

        with torch.no_grad():
            if self.bundle:
                for head_id, layer, head in self.bundle.iter_heads():
                    global_idx = layer * num_heads_per_layer + head
                    if layer >= num_layers or head >= num_heads_per_layer:
                        continue
                    head_tensors = self.bundle.get_head_tensors(head_id)
                    if head_tensors is None:
                        missing.append(head_id)
                        continue
                    _assign_head_slice(
                        batched_gate,
                        global_idx,
                        head_tensors,
                        target_device,
                        target_dtype,
                    )
                    loaded += 1
            else:
                for layer in range(num_layers):
                    for head in range(num_heads_per_layer):
                        head_id = f"l{layer}_h{head}"
                        state = self.get_legacy_state_dict(head_id, device=torch.device("cpu"))
                        if state is None:
                            missing.append(head_id)
                            continue
                        head_tensors = _legacy_to_batched_tensors(state)
                        _assign_head_slice(
                            batched_gate,
                            layer * num_heads_per_layer + head,
                            head_tensors,
                            target_device,
                            target_dtype,
                        )
                        loaded += 1

        return loaded, missing


def save_combined_gate_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    head_ids: List[str],
    output_dir: str,
    filename: str = COMBINED_FILENAME,
    metadata: Optional[Dict[str, str]] = None,
) -> str:
    """Persist stacked gate tensors plus metadata as a safetensors file."""
    os.makedirs(output_dir, exist_ok=True)
    payload = {}
    for key in _WEIGHT_KEYS:
        if key not in state_dict:
            raise KeyError(f"Missing key '{key}' in model state_dict")
        payload[key] = state_dict[key].detach().cpu()

    head_layers, head_heads = _head_metadata(head_ids)
    payload["head_layers"] = head_layers
    payload["head_heads"] = head_heads

    meta = metadata.copy() if metadata else {}
    meta.update(
        {
            "num_heads": str(len(head_ids)),
            "input_dim": str(payload["W1"].shape[1]),
            "hidden_dim": str(payload["W1"].shape[2]),
        }
    )

    out_path = os.path.join(output_dir, filename)
    save_file(payload, out_path, metadata=meta)
    return out_path


def _discover_legacy_paths(gate_dir: str) -> Dict[str, str]:
    files = glob.glob(os.path.join(gate_dir, "l*_h*.pt"))
    return {os.path.splitext(os.path.basename(f))[0]: f for f in sorted(files)}


def _load_combined_bundle(path: str, device: str) -> GateBundle:
    data = load_file(path, device=device)
    try:
        head_layers = data.pop("head_layers")
        head_heads = data.pop("head_heads")
    except KeyError as exc:
        raise KeyError(f"Combined checkpoint missing metadata tensors: {exc}") from exc

    head_ids = [f"l{int(l.item())}_h{int(h.item())}" for l, h in zip(head_layers, head_heads)]
    weights = {k: data[k] for k in _WEIGHT_KEYS}

    return GateBundle(
        weights=weights,
        head_ids=head_ids,
        head_layers=head_layers,
        head_heads=head_heads,
        source="combined",
        path=path,
    )


def _batched_to_legacy_state(head_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert batched tensor slices to the original per-head state_dict format."""
    legacy = {
        "net.0.weight": head_tensors["W1"].transpose(0, 1).contiguous(),
        "net.0.bias": head_tensors["b1"].contiguous(),
        "net.2.weight": head_tensors["W2"].transpose(0, 1).contiguous(),
        "net.2.bias": head_tensors["b2"].contiguous(),
        "net.4.weight": head_tensors["W3"].transpose(0, 1).contiguous(),
        "net.4.bias": head_tensors["b3"].contiguous(),
    }
    return legacy


def _legacy_to_batched_tensors(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert legacy per-head checkpoint to batched tensor shapes."""
    return {
        "W1": state_dict["net.0.weight"].transpose(0, 1),
        "b1": state_dict["net.0.bias"],
        "W2": state_dict["net.2.weight"].transpose(0, 1),
        "b2": state_dict["net.2.bias"],
        "W3": state_dict["net.4.weight"].transpose(0, 1),
        "b3": state_dict["net.4.bias"],
    }


def _assign_head_slice(
    batched_gate: torch.nn.Module,
    idx: int,
    head_tensors: Dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Copy a single head's tensors into the batched model."""
    for key, target in (
        ("W1", batched_gate.W1),
        ("b1", batched_gate.b1),
        ("W2", batched_gate.W2),
        ("b2", batched_gate.b2),
        ("W3", batched_gate.W3),
        ("b3", batched_gate.b3),
    ):
        value = head_tensors[key].to(device=device, dtype=dtype)
        target[idx].copy_(value)



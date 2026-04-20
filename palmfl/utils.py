from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int, device: torch.device | str | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    use_cuda = False
    if device is not None:
        use_cuda = torch.device(device).type == "cuda"
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic mode can slow things down, but is helpful for research reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _probe_cuda_device(device: torch.device) -> tuple[bool, str | None]:
    if device.type != "cuda":
        return False, f"Expected a CUDA device, got {device}"
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"

    try:
        if device.index is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        # A tiny Conv+BatchNorm forward pass catches many driver / architecture mismatches
        # that are not detected by torch.cuda.is_available() alone.
        with torch.no_grad():
            probe_x = torch.randn(2, 1, 8, 8, device=device)
            probe_conv = torch.nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1).to(device)
            probe_bn = torch.nn.BatchNorm2d(2).to(device)
            probe_y = probe_bn(probe_conv(probe_x))
            _ = float(probe_y.mean().item())
        torch.cuda.synchronize(device)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def resolve_device(device_str: str, fallback_to_cpu: bool = True, verbose: bool = True) -> torch.device:
    requested = str(device_str).strip()
    if requested == "auto":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        cuda_device = torch.device("cuda")
        ok, reason = _probe_cuda_device(cuda_device)
        if ok:
            return cuda_device
        if verbose:
            print("[PALM-FL] CUDA is visible but unusable with the current PyTorch build. "
                  f"Falling back to CPU. Reason: {reason}")
        return torch.device("cpu")

    resolved = torch.device(requested)
    if resolved.type != "cuda":
        return resolved

    ok, reason = _probe_cuda_device(resolved)
    if ok:
        return resolved
    if fallback_to_cpu:
        if verbose:
            print("[PALM-FL] Requested CUDA device is unusable in this environment. "
                  f"Falling back to CPU. Reason: {reason}")
        return torch.device("cpu")
    raise RuntimeError(f"Requested CUDA device {resolved} is unusable: {reason}")


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def nested_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    cur: Any = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def nested_set(d: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur = d
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def parse_override(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "none":
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_override(item.strip()) for item in inner.split(",")]
    return value


def load_config(path: str | os.PathLike[str], overrides: list[str] | None = None) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    overrides = overrides or []
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item}")
        key, value = item.split("=", 1)
        nested_set(cfg, key, parse_override(value))
    return cfg


def save_yaml(path: str | os.PathLike[str], data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(path: str | os.PathLike[str], data: Any, indent: int = 2) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def append_jsonl(path: str | os.PathLike[str], record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def to_cpu_detached_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu()


def bytes_from_tensors(*tensors: torch.Tensor) -> int:
    total = 0
    for tensor in tensors:
        total += tensor.numel() * tensor.element_size()
    return total


def pretty_num_params(num_params: int) -> str:
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.2f}K"
    return str(num_params)

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf


def resolve_project_path(path: str | Path, project_root: str | Path) -> Path:
    """Resolve a path against an explicit project root."""
    path = Path(path)
    return path if path.is_absolute() else Path(project_root) / path


def load_resolved_config(path: str | Path) -> DictConfig:
    """Load an OmegaConf file and resolve all interpolations."""
    config = OmegaConf.load(path)
    OmegaConf.resolve(config)
    return config


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch and request deterministic Torch kernels."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, Torch, and available CUDA random states."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore random states captured by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])


def write_csv_atomic(frame: pd.DataFrame, path: str | Path) -> None:
    """Write a CSV through a temporary sibling before replacing its target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def write_json_atomic(payload: dict[str, Any], path: str | Path) -> None:
    """Write a JSON artifact atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def save_torch_atomic(payload: Any, path: str | Path) -> None:
    """Write a Torch artifact atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def save_omegaconf_atomic(config: DictConfig, path: str | Path) -> None:
    """Write a resolved experiment configuration atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    OmegaConf.save(config, tmp_path, resolve=True)
    tmp_path.replace(path)


def tensor_mapping_to_lists(values_by_name: dict[str, Any]) -> dict[str, list[Any]]:
    """Convert a mapping of tensor-like values to CPU lists for metadata."""
    return {
        name: torch.as_tensor(values).detach().cpu().tolist()
        for name, values in values_by_name.items()
    }


def format_mean_sd(mean: Any, sd: Any, digits: int = 3) -> str:
    """Format a scalar estimate as ``mean ± sd`` for result tables."""
    if pd.isna(mean):
        return ""
    if pd.isna(sd):
        sd = 0.0
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"

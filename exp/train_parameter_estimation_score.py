#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.experiment_io import resolve_project_path, write_csv_atomic
from utils.noise_geometry import spd_2d_from_uv
from utils.score_matching import (
    ConditionalScoreMLP,
    save_score_matching_checkpoint,
    set_score_matching_seed,
    train_conditional_score_model,
)


def load_score_training_config(
    path: str | Path,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load a score-training config and apply OmegaConf dot-list overrides."""
    config_path = resolve_project_path(path, PROJECT_ROOT)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = OmegaConf.load(config_path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    validate_score_training_config(config)
    return config


def validate_score_training_config(config: DictConfig) -> None:
    """Validate the fields consumed by the score-training entry point."""
    if str(config.training.objective) != "exact_hyvarinen":
        raise ValueError("training.objective must be exact_hyvarinen")
    for name in ("n_steps", "batch_size", "log_every"):
        if int(config.training[name]) <= 0:
            raise ValueError(f"training.{name} must be positive")
    for name in ("lr", "grad_clip"):
        if float(config.training[name]) <= 0:
            raise ValueError(f"training.{name} must be positive")
    if float(config.data.std_clean) <= 0:
        raise ValueError("data.std_clean must be positive")

    sigmas = list(config.noise.sigmas)
    if not sigmas or any(float(sigma) <= 0 for sigma in sigmas):
        raise ValueError("noise.sigmas must contain positive values")

    dim = int(config.model.dim)
    covariance = config.generalized.covariance
    if str(covariance.type) != "fixed_spd_2d":
        raise ValueError("generalized.covariance.type must be fixed_spd_2d")
    if dim != 2:
        raise ValueError("fixed_spd_2d covariance requires model.dim=2")
    spd_2d_from_uv(torch.as_tensor(list(covariance.uv), dtype=torch.float64))


def train_score_model(config: DictConfig, result_dir: str | Path) -> Path:
    """Train one score network and save all training artifacts."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, result_dir / "config.yaml")

    device = torch.device(str(config.system.device))
    dtype_name = str(config.system.dtype)
    if dtype_name != "float32":
        raise ValueError("Score training requires system.dtype=float32")
    dtype = torch.float32

    set_score_matching_seed(int(config.training.seed))
    model = ConditionalScoreMLP(
        dim=int(config.model.dim),
        width=int(config.model.width),
        depth=int(config.model.depth),
        activation=str(config.model.activation),
    ).to(device=device, dtype=dtype)
    Sigma = spd_2d_from_uv(
        torch.tensor(list(config.generalized.covariance.uv), device=device, dtype=dtype)
    )

    history = train_conditional_score_model(
        model,
        sigmas=list(config.noise.sigmas),
        beta=float(config.generalized.beta_noise),
        lam=float(config.generalized.lam),
        Sigma=Sigma,
        n_steps=int(config.training.n_steps),
        batch_size=int(config.training.batch_size),
        lr=float(config.training.lr),
        weight_decay=float(config.training.weight_decay),
        grad_clip=float(config.training.grad_clip),
        radius=float(config.data.radius),
        std=float(config.data.std_clean),
        log_every=int(config.training.log_every),
        progress=True,
        progress_desc=f"score seed {int(config.training.seed)}",
        seed=int(config.training.seed),
        device=device,
        dtype=dtype,
    )

    history_path = result_dir / "training_history.csv"
    write_csv_atomic(pd.DataFrame(history), history_path)
    checkpoint_path = result_dir / "model.pt"
    resolved_config = OmegaConf.to_container(config, resolve=True)
    save_score_matching_checkpoint(
        checkpoint_path,
        model,
        history=history,
        config=resolved_config,
        metadata={
            "artifact": "matched conditional exact Hyvarinen score estimator",
            "training_seed": int(config.training.seed),
        },
    )

    print(f"Resolved config: {result_dir / 'config.yaml'}", flush=True)
    print(f"Training history: {history_path}", flush=True)
    print(f"Score checkpoint: {checkpoint_path}", flush=True)
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a matched conditional score model for parameter estimation"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Config path relative to the repository root or an absolute path",
    )
    parser.add_argument("--result_dir", required=True)
    args, dotlist = parser.parse_known_args()

    config = load_score_training_config(args.config, dotlist)
    print(OmegaConf.to_yaml(config, resolve=True), flush=True)
    train_score_model(config, args.result_dir)


if __name__ == "__main__":
    main()

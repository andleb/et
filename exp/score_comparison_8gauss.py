#!/usr/bin/env python3
"""Evaluate and train the Eight-Gaussians direct-score models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.eight_gauss import eight_gaussians
from utils.experiment_io import (
    save_torch_atomic,
    set_reproducible_seed,
    write_csv_atomic,
    write_json_atomic,
)
from utils.noise_geometry import (
    generalized_gaussian_actual_rms,
    normalize_full_spd_condition,
    reference_rho_to_sigma,
    sample_generalized_gaussian_noise,
    spd_2d_from_uv,
)
from utils.sampling import generalized_importance_noisy_score
from utils.score_matching import (
    ParameterConditionedScoreMLP,
    full_spd_validation_loss_floors,
    train_full_spd_score_model,
)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_training_config(path: Path, device: str | None = None) -> dict:
    config = OmegaConf.load(path)
    posterior = OmegaConf.load(project_path(str(config.posterior_config)))
    config.data = {
        "radius": float(posterior.data.radius),
        "std_clean": float(posterior.data.std_clean),
    }
    config.noise = posterior.noise
    config.et_training = {"model": posterior.model, "training": posterior.training}
    if device is not None:
        config.system.device = device
    if config.system.dtype != "float32":
        raise ValueError("The score comparison uses float32")
    if config.training.loss_weight != "actual_rms_squared":
        raise ValueError("The score comparison uses actual-RMS-squared loss weights")
    if int(config.model.dim) != 2 or int(config.model.condition_dim) != 5:
        raise ValueError("The comparison requires 2D data and five noise coordinates")
    for key in (
        "n_steps",
        "batch_size",
        "log_every",
        "checkpoint_every",
        "validation_every",
        "validation_batches",
        "validation_batch_size",
    ):
        if int(config.training[key]) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if not set(config.objectives) <= {"et_dsm", "hyvarinen"}:
        raise ValueError("Unknown training objective")
    if not 0 <= float(config.training.lr_min) <= float(config.training.lr):
        raise ValueError("Require 0 <= training.lr_min <= training.lr")
    return OmegaConf.to_container(config, resolve=True)


def load_run_config(config: dict, objective: str, seed: int) -> dict:
    """Combine a model's saved configuration with the evaluation settings."""
    path = (
        project_path(config["model_root"])
        / "score_comparison"
        / f"{objective}_seed{seed}"
        / "config.json"
    )
    saved = json.loads(path.read_text())
    for key in (
        "output_root",
        "reference_cache",
        "et_results",
        "evaluation_seed",
        "model_root",
    ):
        saved[key] = config[key]
    saved["system"]["device"] = config["system"]["device"]
    return saved


def scientific_config(config: dict) -> dict:
    return {key: config[key] for key in ("data", "noise", "model", "training")} | {
        "dtype": config["system"]["dtype"]
    }


def load_evaluation_protocol(config: dict) -> dict:
    """Read the score notebook's generated evaluation settings."""
    path = project_path(config["et_results"]) / "quantitative_config.json"
    return json.loads(path.read_text())


def prepare_evaluation_inputs(config: dict, protocol: dict) -> list[Path]:
    """Generate or reload a separate test set shared by the direct estimators.

    Match the posterior evaluation's conditions and sample counts, using an
    independent seed for the observations and the split reference bank.
    """
    directory = project_path(config["reference_cache"]) / "evaluation_inputs"
    directory.mkdir(parents=True, exist_ok=True)
    paths = [
        directory / f"{condition['condition_id']}.pt"
        for condition in protocol["conditions"]
    ]
    device = torch.device(config["system"]["device"])
    seed = int(config["evaluation_seed"])
    prior = None
    rows = []
    for index, (condition, path) in enumerate(zip(protocol["conditions"], paths)):
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=False)
        else:
            if prior is None:
                set_reproducible_seed(seed)
                prior = eight_gaussians(
                    int(protocol["n_prior"]),
                    radius=float(config["data"]["radius"]),
                    std=float(config["data"]["std_clean"]),
                    device=device,
                )
            set_reproducible_seed(seed + 1000 * (index + 1))
            clean = eight_gaussians(
                int(protocol["n_y_per_condition"]),
                radius=float(config["data"]["radius"]),
                std=float(config["data"]["std_clean"]),
                device=device,
            )
            beta, lam, rho = (float(condition[key]) for key in ("beta", "lam", "rho"))
            shape = spd_2d_from_uv(
                torch.tensor([condition["u"], condition["v"]], device=device)
            )
            sigma = float(reference_rho_to_sigma(rho, beta, 2))
            noise_kwargs = dict(beta=beta, lam=lam, Sigma=shape, sigma=sigma)
            y = clean + sample_generalized_gaussian_noise(
                len(clean), **noise_kwargs, device=device, dtype=clean.dtype
            )
            first, second = [
                generalized_importance_noisy_score(
                    y,
                    bank,
                    **noise_kwargs,
                    chunk_size=int(protocol["importance_chunk"]),
                ).cpu()
                for bank in prior.chunk(2)
            ]
            payload = {
                "condition": condition,
                "sigma": sigma,
                "actual_rms": float(generalized_gaussian_actual_rms(rho, beta, lam)),
                "y": y.cpu(),
                "reference": 0.5 * (first + second),
                "reference_half_1": first,
                "reference_half_2": second,
            }
            save_torch_atomic(payload, path)
            print(f"Direct-score references: {index + 1}/{len(paths)}", flush=True)
        reference = payload["reference"]
        difference = payload["reference_half_1"] - payload["reference_half_2"]
        rows.append(
            {
                **{key: value for key, value in condition.items() if key != "lam"},
                "lambda": condition["lam"],
                "sigma": payload["sigma"],
                "actual_rms": payload["actual_rms"],
                "direct_score_split_nmse": float(
                    difference.square().sum(-1).mean()
                    / reference.square().sum(-1).mean().clamp_min(1e-12)
                ),
            }
        )
    write_csv_atomic(
        pd.DataFrame(rows),
        project_path(config["reference_cache"]) / "score_reference_quality.csv",
    )
    return paths


def load_run_checkpoint(config: dict, objective: str, seed: int) -> dict:
    """Load the requested direct-score model from the selected model root."""
    checkpoint = (
        project_path(config["model_root"])
        / "score_comparison"
        / f"{objective}_seed{seed}"
        / "model.pt"
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity = payload["identity"]
    if identity["objective"] != objective or int(identity["seed"]) != seed:
        raise ValueError(f"Checkpoint objective or seed mismatch: {checkpoint}")
    if scientific_config(identity["config"]) != scientific_config(config):
        raise ValueError(f"Checkpoint scientific configuration mismatch: {checkpoint}")
    if payload["model_config"] != config["model"]:
        raise ValueError(f"Checkpoint architecture mismatch: {checkpoint}")
    if int(payload["step"]) != int(config["training"]["n_steps"]):
        raise ValueError(f"Training is incomplete: {checkpoint}")
    return payload


def evaluate_run(
    config: dict, output: Path, objective: str, seed: int, inputs: list[Path]
) -> None:
    run = output / objective / f"seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    payload = load_run_checkpoint(config, objective, seed)
    write_csv_atomic(pd.DataFrame(payload["history"]), run / "training_history.csv")
    device = torch.device(config["system"]["device"])
    metrics_path = run / "score_metrics.csv"
    model = ParameterConditionedScoreMLP(**payload["model_config"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    noise_config = OmegaConf.create(config["noise"])
    rows, estimates = [], {}
    for path in inputs:
        data = torch.load(path, map_location="cpu", weights_only=False)
        condition = data["condition"]
        y, reference = data["y"].to(device), data["reference"].to(device)
        encoded = normalize_full_spd_condition(
            condition["beta"],
            condition["lam"],
            condition["rho"],
            torch.tensor([condition["u"], condition["v"]], device=device),
            noise_config,
            batch_size=len(y),
        )
        with torch.no_grad():
            score = model(y, encoded)
            power = reference.square().sum(-1).mean().clamp_min(1e-12)
            cosine = (score * reference).sum(-1) / (
                score.norm(dim=-1) * reference.norm(dim=-1)
            ).clamp_min(1e-12)
            rows.append(
                {
                    "method": objective,
                    "model_seed": seed,
                    **{key: value for key, value in condition.items() if key != "lam"},
                    "lambda": condition["lam"],
                    "sigma": data["sigma"],
                    "actual_rms": data["actual_rms"],
                    "score_nmse": float(
                        (score - reference).square().sum(-1).mean() / power
                    ),
                    "score_cosine": float(cosine.mean()),
                    "score_norm_ratio": float(
                        score.norm(dim=-1).mean()
                        / reference.norm(dim=-1).mean().clamp_min(1e-12)
                    ),
                }
            )
            estimates[condition["condition_id"]] = score.cpu()
    write_csv_atomic(pd.DataFrame(rows), metrics_path)
    save_torch_atomic(estimates, run / "score_predictions.pt")
    print(f"Saved {metrics_path}", flush=True)


def prepare_validation_floors(config: dict, seeds: list[int], objective: str) -> None:
    """Generate validation batches and estimate their population-loss floors."""
    prior = torch.load(
        project_path(config["et_results"]) / "evaluation_inputs/prior_bank.pt",
        map_location="cpu",
        weights_only=False,
    )
    for seed in seeds:
        run_config = load_run_config(config, objective, seed)
        full_spd_validation_loss_floors(
            run_config,
            OmegaConf.create(run_config["noise"]),
            seed=seed,
            prior_bank=prior,
            result_dir=project_path(config["reference_cache"])
            / "validation_loss_floors"
            / f"seed{seed}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--stage",
        choices=["all", "train", "evaluate", "floors", "histories"],
        default="all",
    )
    parser.add_argument("--objective", choices=["et_dsm", "hyvarinen"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument(
        "--model-root",
        type=Path,
        help="Model root for evaluation and saved training losses",
    )
    args = parser.parse_args()
    if args.stage == "train":
        config = load_training_config(
            project_path(
                args.config or "exp/configs/train_8gauss_score_comparison.yaml"
            ),
            args.device,
        )
    else:
        config = OmegaConf.to_container(
            OmegaConf.load(
                project_path(args.config or "exp/configs/8gauss-score-comparison.yaml")
            ),
            resolve=True,
        )
        if args.device is not None:
            config["system"]["device"] = args.device
    if args.model_root is not None:
        config["model_root"] = str(args.model_root)
    torch.set_default_dtype(torch.float32)
    output = project_path(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    write_json_atomic(config, config_path)
    objectives = [args.objective] if args.objective else config["objectives"]
    seeds = [args.seed] if args.seed is not None else config["seeds"]
    if not set(seeds) <= set(config["seeds"]):
        raise ValueError("Select seeds from the configuration")
    if not set(objectives) <= set(config["objectives"]):
        raise ValueError("Select objectives from the configuration")
    if args.stage == "train":
        for seed in seeds:
            for objective in objectives:
                train_full_spd_score_model(
                    config,
                    OmegaConf.create(config["noise"]),
                    objective=objective,
                    seed=seed,
                    result_dir=PROJECT_ROOT
                    / "res/models/score_comparison"
                    / f"{objective}_seed{seed}",
                )
        return
    if args.stage == "histories":
        for seed in seeds:
            for objective in objectives:
                run_config = load_run_config(config, objective, seed)
                payload = load_run_checkpoint(run_config, objective, seed)
                write_csv_atomic(
                    pd.DataFrame(payload["history"]),
                    output / objective / f"seed{seed}" / "training_history.csv",
                )
        return
    if args.stage in {"all", "evaluate"}:
        protocol = load_evaluation_protocol(config)
        reference_config = load_run_config(config, objectives[0], seeds[0])
        inputs = prepare_evaluation_inputs(reference_config, protocol)
        for seed in seeds:
            for objective in objectives:
                run_config = load_run_config(config, objective, seed)
                evaluate_run(run_config, output, objective, seed, inputs)
    if args.stage in {"all", "floors"}:
        prepare_validation_floors(config, seeds, objectives[0])


if __name__ == "__main__":
    main()

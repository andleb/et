"""Generate the five MNIST paper paths from one conditional posterior checkpoint.

Run from the repository root with ``python -m exp.sample_diffusion_mnist --config
exp/configs/diffusion-mnist-gaussian-isotropic.yaml``. Each seed produces two
128-image chunks. Existing chunks with matching metadata are reused.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from filelock import FileLock
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ENGRESSION_PATH = ROOT / "src" / "engression"
if str(ENGRESSION_PATH) not in sys.path:
    sys.path.insert(1, str(ENGRESSION_PATH))
from utils.experiment_io import (
    save_torch_atomic,
    set_reproducible_seed,
    tensor_mapping_to_lists,
    write_csv_atomic,
    write_json_atomic,
)
from utils.mnist import (
    MNIST_DIM,
    build_mnist_engressor,
    build_mnist_sampling_schedule,
)
from utils.noise_geometry import normalize_full_spd_condition
from utils.sampling import (
    ParameterConditionedEngressorSampler,
    annealed_langevin_generate_parameter_path,
)

ALLOWED_PATHS = {
    "gaussian",
    "gaussian_non_diagonal",
    "generalized_isotropic",
    "fixed_non_diagonal",
    "joint",
}


def load_model_sampler(spec, device, model_root):
    model_dir = model_root / spec["run_name"]
    checkpoint_path = model_dir / spec["checkpoint"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Download final.pt from https://github.com/andleb/et/releases/tag/MNIST_model "
            f"and place it at {checkpoint_path}."
        )
    model_config = OmegaConf.load(model_dir / "config.yaml")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    engressor = build_mnist_engressor(model_config, device)
    engressor.model.load_state_dict(checkpoint["model_state_dict"])
    sampler = ParameterConditionedEngressorSampler(
        engressor,
        device,
        condition_encoder=partial(
            normalize_full_spd_condition, noise_config=model_config.noise
        ),
        compile_model=False,
    )
    return {
        "config": model_config,
        "checkpoint_path": checkpoint_path,
        "engressor": engressor,
        "sampler": sampler,
    }


def generate_cached_bundle(
    model_info,
    schedule,
    metadata,
    generation_seed,
    output_path,
    device,
    sampler_kwargs,
    progress_desc,
):
    payload = None
    if output_path.is_file():
        candidate = torch.load(output_path, map_location="cpu", weights_only=False)
        if candidate.get("metadata") == metadata:
            payload = candidate
            print(f"Reusing cached bundle {output_path}")
    if payload is None:
        set_reproducible_seed(int(generation_seed))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.monotonic()
        generated = annealed_langevin_generate_parameter_path(
            model_info["sampler"].sample,
            dim=MNIST_DIM,
            schedule=schedule,
            device=device,
            verbose=False,
            show_progress=True,
            progress_desc=progress_desc,
            **sampler_kwargs,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        payload = {
            "metadata": metadata,
            "elapsed_seconds": time.monotonic() - started,
            "final": generated["final"],
            "last_noisy": generated["last_noisy"],
            "trajectory": generated["trajectory"],
            "diagnostics": generated["diagnostics"],
        }
        save_torch_atomic(payload, output_path)
    return payload


def run_configured_production_path(config_path, device, seeds=None):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)
    spec = OmegaConf.to_container(config.model, resolve=True)
    path_name = str(config.path.name)
    if path_name not in ALLOWED_PATHS:
        raise ValueError(f"Unknown MNIST paper path: {path_name}")
    configured_seeds = [int(seed) for seed in config.production.seeds]
    selected_seeds = (
        configured_seeds if seeds is None else [int(seed) for seed in seeds]
    )
    unknown_seeds = sorted(set(selected_seeds) - set(configured_seeds))
    if unknown_seeds:
        raise ValueError(f"Seeds {unknown_seeds} are not listed in {config_path}")
    chunks_per_seed = int(config.production.chunks_per_seed)
    chunk_samples = int(config.production.chunk_samples)
    if chunks_per_seed <= 0 or chunk_samples <= 0:
        raise ValueError("Production chunk counts and sizes must be positive")
    sampler_config = {
        **OmegaConf.to_container(config.sampler, resolve=True),
        "n_samples": chunk_samples,
    }
    production_root = ROOT / str(config.output_root)
    production_dir = production_root / path_name
    production_run_dir = production_dir / "runs"
    production_run_dir.mkdir(parents=True, exist_ok=True)
    config_output_path = production_dir / "config.json"
    with FileLock(str(config_output_path) + ".lock"):
        write_json_atomic(
            OmegaConf.to_container(config, resolve=True), config_output_path
        )
    model_info = load_model_sampler(spec, device, ROOT / str(config.model_root))
    records = []
    try:
        schedule = build_mnist_sampling_schedule(model_info["config"], config, device)
        schedule_meta = tensor_mapping_to_lists(schedule)
        generation_kwargs = {
            key: value for key, value in sampler_config.items() if key != "n_levels"
        }
        for seed in selected_seeds:
            for chunk_index in range(chunks_per_seed):
                chunk_seed = int(seed) * 100 + chunk_index
                output_path = (
                    production_run_dir
                    / f"{spec['run_name']}__{path_name}__s{int(seed)}_chunk{chunk_index}.pt"
                )
                metadata = {
                    "model_directory": str(model_info["checkpoint_path"].parent),
                    "run_name": spec["run_name"],
                    "model_label": spec["label"],
                    "path": path_name,
                    "generation_seed": int(seed),
                    "chunk_index": chunk_index,
                    "chunk_seed": chunk_seed,
                    "sampler": sampler_config,
                    "schedule": schedule_meta,
                }
                payload = generate_cached_bundle(
                    model_info,
                    schedule,
                    metadata,
                    chunk_seed,
                    output_path,
                    device,
                    generation_kwargs,
                    f"{path_name} s{seed} chunk {chunk_index}",
                )
                records.append(
                    {
                        "run_name": spec["run_name"],
                        "model_label": spec["label"],
                        "path": path_name,
                        "generation_seed": int(seed),
                        "chunk_index": chunk_index,
                        "chunk_seed": chunk_seed,
                        "n_samples": chunk_samples,
                        "bundle_path": str(output_path),
                        "elapsed_seconds": float(payload["elapsed_seconds"]),
                    }
                )
                records_name = (
                    "records.csv"
                    if selected_seeds == configured_seeds
                    else "records_s"
                    + "_".join((str(value) for value in selected_seeds))
                    + ".csv"
                )
                write_csv_atomic(pd.DataFrame(records), production_dir / records_name)
    finally:
        del model_info
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated subset of the three production seeds",
    )
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA requested but unavailable")
    seeds = (
        [int(value) for value in arguments.seeds.split(",")]
        if arguments.seeds
        else None
    )
    run_configured_production_path(arguments.config, device, seeds)


if __name__ == "__main__":
    main()

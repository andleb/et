#!/usr/bin/env python3
"""Train a five-parameter conditional Engressor on dequantized MNIST."""

from __future__ import annotations

import argparse
import math
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import OmegaConf

from engression.loss_func import energy_loss_two_sample
from utils.experiment_io import (
    capture_rng_state,
    restore_rng_state,
    save_omegaconf_atomic,
    save_torch_atomic,
    set_reproducible_seed,
    write_csv_atomic,
    write_json_atomic,
)
from utils.mnist import (
    MNIST_DIM,
    build_mnist_engressor,
    build_mnist_preview_schedule,
    build_mnist_scheduler,
    dequantize_mnist,
    make_mnist_loader,
    save_mnist_image_grid,
    save_mnist_progression_grid,
)
from utils.noise_geometry import (
    FullSPDNoiseBatch,
    FullSPDNoiseParameterSampler,
    normalize_full_spd_condition,
    repeated_block_energy_score,
)
from utils.sampling import (
    ParameterConditionedEngressorSampler,
    annealed_langevin_generate_parameter_path,
)


class ParameterCoverage:

    def __init__(self, device: torch.device):
        self.count = torch.zeros((), device=device, dtype=torch.long)
        self.anchor_count = torch.zeros((), device=device, dtype=torch.long)
        self.minimum = torch.full((5,), math.inf, device=device)
        self.maximum = torch.full((5,), -math.inf, device=device)
        self.total = torch.zeros(5, device=device)

    def update(self, parameters: FullSPDNoiseBatch) -> None:
        values = parameters.parameter_vector().detach()
        self.count += 1
        self.anchor_count += parameters.used_anchor.detach().long()
        self.minimum = torch.minimum(self.minimum, values)
        self.maximum = torch.maximum(self.maximum, values)
        self.total += values

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "count": self.count.detach().cpu(),
            "anchor_count": self.anchor_count.detach().cpu(),
            "minimum": self.minimum.detach().cpu(),
            "maximum": self.maximum.detach().cpu(),
            "total": self.total.detach().cpu(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name, value in state.items():
            setattr(self, name, value.to(self.count.device))

    def summary(self) -> dict[str, Any]:
        count = int(self.count.detach().cpu())
        if count == 0:
            return {"batches": 0, "anchor_batches": 0, "parameters": {}}
        minimum = self.minimum.detach().cpu().tolist()
        maximum = self.maximum.detach().cpu().tolist()
        mean = (self.total / self.count).detach().cpu().tolist()
        return {
            "batches": count,
            "anchor_batches": int(self.anchor_count.detach().cpu()),
            "anchor_fraction": float(self.anchor_count.float() / self.count),
            "parameters": {
                name: {"min": low, "max": high, "mean": average}
                for name, low, high, average in zip(
                    ("beta", "lam", "rho", "u", "v"), minimum, maximum, mean
                )
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint path or run directory containing latest.pt",
    )
    parser.add_argument("overrides", nargs="*", help="OmegaConf dot-list overrides")
    return parser.parse_args()


def resolve_config(path: str | Path, overrides: list[str]):
    config = OmegaConf.load(path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(config)
    config.training.energy_score = str(config.training.get("energy_score", "euclidean"))
    if config.training.energy_score not in {"euclidean", "noise_matched"}:
        raise ValueError("training.energy_score must be euclidean or noise_matched")
    return config


def resolve_resume_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "latest.pt"
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    return path


def move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def checkpoint_payload(
    *,
    engressor,
    scheduler,
    step: int,
    epoch: int,
    coverage: ParameterCoverage,
    loader_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "checkpoint_kind": "model_training_state",
        "step": int(step),
        "epoch": int(epoch),
        "model_state_dict": engressor.model.state_dict(),
        "optimizer_state_dict": engressor.optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
        "loader_generator_state": loader_generator.get_state(),
        "coverage_state": coverage.state_dict(),
    }


def save_training_checkpoint(
    result_dir: Path,
    snapshot_dir: Path,
    *,
    engressor,
    scheduler,
    step: int,
    epoch: int,
    coverage: ParameterCoverage,
    loader_generator: torch.Generator,
    archive: bool,
    final: bool = False,
) -> None:
    latest_path = result_dir / "latest.pt"
    if final:
        save_torch_atomic(
            {"model_state_dict": engressor.model.state_dict()}, result_dir / "final.pt"
        )
        latest_path.unlink(missing_ok=True)
        return
    payload = checkpoint_payload(
        engressor=engressor,
        scheduler=scheduler,
        step=step,
        epoch=epoch,
        coverage=coverage,
        loader_generator=loader_generator,
    )
    if archive:
        save_torch_atomic(payload, snapshot_dir / f"epoch_{epoch:05d}.pt")
    save_torch_atomic(payload, latest_path)


def save_training_history(
    result_dir: Path,
    *,
    history: list[dict[str, float]],
    coverage: ParameterCoverage,
) -> None:
    if history:
        write_csv_atomic(pd.DataFrame(history), result_dir / "losses.csv")
    write_json_atomic(coverage.summary(), result_dir / "parameter_coverage.json")


def save_generation_preview(
    *,
    config,
    engressor,
    device: torch.device,
    result_dir: Path,
    epoch: int,
    step: int,
) -> None:
    generation = config.generation
    schedule = build_mnist_preview_schedule(
        config.noise,
        generation,
        device=device,
    )
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    was_training = engressor.model.training
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(generation.seed))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(generation.seed))
            sampler = ParameterConditionedEngressorSampler(
                engressor,
                device,
                condition_encoder=partial(
                    normalize_full_spd_condition,
                    noise_config=config.noise,
                ),
                compile_model=False,
            )
            print(
                f"Generating fixed-path samples at epoch {epoch}, step {step}",
                flush=True,
            )
            generated = annealed_langevin_generate_parameter_path(
                sampler.sample,
                n_samples=int(generation.n_samples),
                dim=MNIST_DIM,
                schedule=schedule,
                n_steps_each=int(generation.n_steps_each),
                first_level_steps=int(generation.first_level_steps),
                n_mc=int(generation.n_mc),
                base_lr=(
                    None if generation.base_lr is None else float(generation.base_lr)
                ),
                alpha_cap=float(generation.alpha_cap),
                update_clip=float(generation.update_clip),
                state_clip=(
                    None
                    if generation.state_clip is None
                    else tuple(float(value) for value in generation.state_clip)
                ),
                device=device,
                verbose=False,
            )
    finally:
        if was_training:
            engressor.train_mode()

    trajectory = generated["trajectory"]
    progression_rows = min(int(generation.progression_rows), len(trajectory))
    level_indices = (
        torch.linspace(0, len(trajectory) - 1, progression_rows)
        .round()
        .to(torch.int64)
        .tolist()
    )
    schedule_cpu = {name: values.detach().cpu() for name, values in schedule.items()}
    stem = f"epoch_{epoch:05d}_step_{step:07d}"
    artifact = {
        "epoch": int(epoch),
        "step": int(step),
        "seed": int(generation.seed),
        "schedule": schedule_cpu,
        "sampler": {
            "initialization": "schedule_noise",
            "finalization": "posterior_draw",
            "n_samples": int(generation.n_samples),
            "n_steps_each": int(generation.n_steps_each),
            "first_level_steps": int(generation.first_level_steps),
            "n_mc": int(generation.n_mc),
            "base_lr": float(generated["base_lr"]),
            "alpha_cap": float(generation.alpha_cap),
            "update_clip": float(generation.update_clip),
            "state_clip": (
                None
                if generation.state_clip is None
                else [float(value) for value in generation.state_clip]
            ),
        },
        "final": generated["final"],
        "last_noisy": generated["last_noisy"],
        "level_indices": level_indices,
        "trajectory": [trajectory[index] for index in level_indices],
        "diagnostics": generated["diagnostics"],
    }
    sample_dir = result_dir / "samples"
    save_torch_atomic(artifact, sample_dir / f"{stem}.pt")
    save_mnist_image_grid(
        artifact["final"],
        sample_dir / f"{stem}.png",
        nrow=max(1, int(math.ceil(math.sqrt(int(generation.n_samples))))),
    )
    save_mnist_progression_grid(
        trajectory,
        level_indices=level_indices,
        n_chains=int(generation.progression_chains),
        path=sample_dir / f"{stem}_path.png",
    )
    print(
        f"Saved generation samples to {sample_dir / f'{stem}.png'}",
        flush=True,
    )


def maybe_save_initial_preview(
    *,
    config,
    engressor,
    device: torch.device,
    result_dir: Path,
    epoch: int,
    step: int,
) -> None:
    """Save the startup preview only when explicitly enabled in the config."""
    if not bool(config.training.preview_at_start):
        return
    save_generation_preview(
        config=config,
        engressor=engressor,
        device=device,
        result_dir=result_dir,
        epoch=epoch,
        step=step,
    )


def main() -> None:
    args = parse_args()
    config = resolve_config(args.config, args.overrides)
    result_dir = Path(args.result_dir).expanduser().resolve()
    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    resume_path = resolve_resume_path(args.resume)
    result_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    existing_checkpoint = next(
        (
            path
            for path in (result_dir / "latest.pt", result_dir / "final.pt")
            if path.exists()
        ),
        None,
    )
    if resume_path is None and existing_checkpoint is not None:
        raise FileExistsError(
            f"{result_dir} already contains {existing_checkpoint.name}; "
            "use a new run name or resume the unfinished run"
        )

    device = torch.device(str(config.system.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    set_reproducible_seed(int(config.training.seed))
    checkpoint = (
        torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_path is not None
        else None
    )
    if (
        checkpoint is not None
        and checkpoint.get("checkpoint_kind") != "model_training_state"
    ):
        raise ValueError(
            "Resume requires latest.pt or an epoch snapshot with training state"
        )
    if checkpoint is not None:
        resume_run_dir = (
            result_dir if resume_path.parent == snapshot_dir else resume_path.parent
        )
        resume_config = OmegaConf.load(resume_run_dir / "config.yaml")
        if resume_config.training.energy_score != config.training.energy_score:
            raise ValueError(
                "Resume requires the same training.energy_score as the saved run config"
            )
    save_omegaconf_atomic(config, result_dir / "config.yaml")
    engressor = build_mnist_engressor(config, device)
    scheduler = build_mnist_scheduler(engressor.optimizer, config.training)
    coverage = ParameterCoverage(device)
    history: list[dict[str, float]] = []
    step = 0
    completed_epoch = 0
    loader_state = None
    if checkpoint is not None:
        engressor.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            engressor.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            move_optimizer_state(engressor.optimizer, device)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        coverage.load_state_dict(checkpoint["coverage_state"])
        step = int(checkpoint["step"])
        completed_epoch = int(checkpoint["epoch"])
        loader_state = checkpoint.get("loader_generator_state")
        history_path = resume_run_dir / "losses.csv"
        if history_path.is_file():
            saved_history = pd.read_csv(history_path)
            history = saved_history.loc[saved_history["step"] <= step].to_dict(
                orient="records"
            )

    loader, loader_generator = make_mnist_loader(
        config.data.root,
        train=True,
        batch_size=int(config.training.batch_size),
        seed=int(config.data.loader_seed),
        num_workers=int(config.data.num_workers),
        download=bool(config.data.download),
        generator_state=loader_state,
    )
    parameter_sampler = FullSPDNoiseParameterSampler(
        config.noise, dim=MNIST_DIM, device=device
    )
    if checkpoint is not None:
        restore_rng_state(checkpoint["rng_state"])

    parameter_count = sum(
        parameter.numel() for parameter in engressor.model.parameters()
    )
    print(f"Result directory: {result_dir}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Model parameters: {parameter_count:,}", flush=True)
    print(f"Batches per epoch: {len(loader)}", flush=True)
    print(
        f"Starting at step {step:,}, completed epoch {completed_epoch:,}; "
        f"target step {int(config.training.max_steps):,}",
        flush=True,
    )

    maybe_save_initial_preview(
        config=config,
        engressor=engressor,
        device=device,
        result_dir=result_dir,
        epoch=completed_epoch,
        step=step,
    )

    max_steps = int(config.training.max_steps)
    noise_matched_energy = config.training.energy_score == "noise_matched"
    log_every = int(config.training.log_every_steps)
    history_every = int(config.training.history_every_steps)
    history_save_every = int(config.training.history_save_every_epochs)
    preview_every = int(config.training.preview_every_epochs)
    checkpoint_every = int(config.training.checkpoint_every_epochs)
    archive_every = int(config.training.archive_every_epochs)
    early_preview_epochs = {
        int(value) for value in config.training.preview_early_epochs
    }
    epoch = completed_epoch
    engressor.model.train()

    while step < max_steps:
        epoch += 1
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_totals = torch.zeros(3, device=device)
        epoch_batches = 0
        for uint8_images, _ in loader:
            if step >= max_steps:
                break
            clean = dequantize_mnist(uint8_images, device=device)
            parameters = parameter_sampler.sample(clean.shape[0])
            noisy = parameter_sampler.corrupt(clean, parameters)

            engressor.optimizer.zero_grad(set_to_none=True)
            posterior_1 = engressor.model(noisy, parameters.condition)
            posterior_2 = engressor.model(noisy, parameters.condition)
            if noise_matched_energy:
                loss, prediction_term, dispersion_term = repeated_block_energy_score(
                    clean,
                    posterior_1,
                    posterior_2,
                    beta=parameters.beta,
                    sigma=parameters.sigma,
                    uv=parameters.uv,
                )
            else:
                loss, prediction_term, dispersion_term = energy_loss_two_sample(
                    clean,
                    posterior_1,
                    posterior_2,
                    beta=float(config.model.energy_beta),
                    verbose=True,
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss at step {step + 1}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                engressor.model.parameters(), float(config.training.grad_clip_norm)
            )
            engressor.optimizer.step()
            scheduler.step()

            step += 1
            epoch_batches += 1
            metrics = torch.stack(
                (loss.detach(), prediction_term.detach(), dispersion_term.detach())
            )
            epoch_totals += metrics
            coverage.update(parameters)

            if step % history_every == 0 or step == 1:
                values = parameters.parameter_vector().detach().cpu().tolist()
                history.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "energy_loss": float(metrics[0]),
                        "prediction_term": float(metrics[1]),
                        "dispersion_term": float(metrics[2]),
                        "learning_rate": float(
                            engressor.optimizer.param_groups[0]["lr"]
                        ),
                        "grad_norm_before_clip": float(grad_norm),
                        "beta": values[0],
                        "lam": values[1],
                        "rho": values[2],
                        "u": values[3],
                        "v": values[4],
                        "actual_rms": float(parameters.actual_rms),
                        "anchor_batch": int(parameters.used_anchor),
                    }
                )
            if step % log_every == 0 or step == 1:
                print(
                    f"step={step:07d} epoch={epoch:05d} "
                    f"loss={float(loss):.5f} pred={float(prediction_term):.5f} "
                    f"disp={float(dispersion_term):.5f} "
                    f"lr={engressor.optimizer.param_groups[0]['lr']:.2e} "
                    f"rms={float(parameters.actual_rms):.4f}",
                    flush=True,
                )

        if epoch_batches == 0:
            break
        epoch_means = epoch_totals / epoch_batches
        memory_message = ""
        if device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
            peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
            memory_message = (
                f" peak_gpu_allocated={peak_allocated:.2f}GiB"
                f" peak_gpu_reserved={peak_reserved:.2f}GiB"
            )
        print(
            f"epoch={epoch:05d} complete step={step:07d} "
            f"mean_loss={float(epoch_means[0]):.5f}{memory_message}",
            flush=True,
        )
        should_preview = epoch in early_preview_epochs or epoch % preview_every == 0
        if should_preview:
            save_generation_preview(
                config=config,
                engressor=engressor,
                device=device,
                result_dir=result_dir,
                epoch=epoch,
                step=step,
            )
        if epoch % history_save_every == 0 or epoch % checkpoint_every == 0:
            save_training_history(
                result_dir,
                history=history,
                coverage=coverage,
            )
        if epoch % checkpoint_every == 0:
            save_training_checkpoint(
                result_dir,
                snapshot_dir,
                engressor=engressor,
                scheduler=scheduler,
                step=step,
                epoch=epoch,
                coverage=coverage,
                loader_generator=loader_generator,
                archive=epoch % archive_every == 0,
            )
        engressor.model.train()

    if epoch not in early_preview_epochs and epoch % preview_every != 0:
        save_generation_preview(
            config=config,
            engressor=engressor,
            device=device,
            result_dir=result_dir,
            epoch=epoch,
            step=step,
        )
    save_training_history(
        result_dir,
        history=history,
        coverage=coverage,
    )
    save_training_checkpoint(
        result_dir,
        snapshot_dir,
        engressor=engressor,
        scheduler=scheduler,
        step=step,
        epoch=epoch,
        coverage=coverage,
        loader_generator=loader_generator,
        archive=False,
        final=True,
    )
    print(f"Finished at step {step:,}, epoch {epoch:,}", flush=True)


if __name__ == "__main__":
    main()

"""Generate the quantitative Eight-Gaussians bundles.

``--all`` processes all experiment arms in one resumable process and loads
each trained model once. All arms are generated directly on the 49-level
schedule. Every endpoint uses the same deterministic finalization rule
and is saved with its sampler settings. ``--task-index`` selects one task.
Evaluate the results in ``exp/reproduce_diffusion_8gauss.ipynb``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
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
    load_resolved_config,
    resolve_project_path,
    save_torch_atomic,
    set_reproducible_seed,
    tensor_mapping_to_lists,
    write_csv_atomic,
    write_json_atomic,
)
from utils.noise_geometry import (
    build_fixed_shape_actual_rms_schedule,
)
from utils.sampling import (
    annealed_langevin_generate_parameter_path,
    load_parameter_conditioned_engressor,
)

DEFAULT_CONFIG_PATH = ROOT / "exp" / "configs" / "diffusion-8gauss-controls.yaml"
REQUIRED_ARM_COLUMNS = {
    "arm_id",
    "parameter",
    "value",
    "value_label",
    "beta",
    "lam",
    "u",
    "v",
    "plot_order",
}


def relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_arms(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Arm file has no header: {path}")
        missing_columns = REQUIRED_ARM_COLUMNS - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"Arm file is missing columns {sorted(missing_columns)}: {path}"
            )
        arms = []
        for arm_index, raw in enumerate(reader):
            arm = {
                "arm_index": arm_index,
                "arm_id": raw["arm_id"].strip(),
                "parameter": raw["parameter"].strip(),
                "value": raw["value"].strip(),
                "value_label": raw["value_label"].strip(),
                "beta": float(raw["beta"]),
                "lam": float(raw["lam"]),
                "u": float(raw["u"]),
                "v": float(raw["v"]),
                "plot_order": int(raw["plot_order"]),
            }
            arms.append(arm)
    return arms


def model_specs(config) -> list[dict[str, object]]:
    return [
        {
            "training_seed": int(spec.training_seed),
            "run_name": str(spec.run_name),
            "checkpoint": str(spec.checkpoint),
        }
        for spec in config.models
    ]


def build_schedule(arm: dict[str, object], config) -> dict[str, torch.Tensor]:
    source_grid = build_source_grid(arm, config)
    schedule = {
        name: values[: int(config.schedule.n_levels)].clone()
        for name, values in source_grid.items()
    }
    return schedule


def build_source_grid(arm: dict[str, object], config) -> dict[str, torch.Tensor]:
    return build_fixed_shape_actual_rms_schedule(
        int(config.schedule_grid.n_levels),
        float(config.schedule_grid.actual_rms_start),
        float(config.schedule_grid.actual_rms_end),
        beta=float(arm["beta"]),
        lam=float(arm["lam"]),
        uv=[float(arm["u"]), float(arm["v"])],
        dim=int(config.data.dim),
        dtype=torch.float64,
    )


def validate_model_coverage(model_config, arm: dict[str, object]) -> None:
    noise = model_config.noise
    if str(noise.condition) != "full_spd":
        raise ValueError("Ceteris-paribus generation requires a full-SPD model")
    if not float(noise.beta_min) <= float(arm["beta"]) <= float(noise.beta_max):
        raise ValueError(f"Arm {arm['arm_id']} lies outside the beta range")
    if not float(noise.lambda_min) <= float(arm["lam"]) <= float(noise.lambda_max):
        raise ValueError(f"Arm {arm['arm_id']} lies outside the lambda range")
    radius = (float(arm["u"]) ** 2 + float(arm["v"]) ** 2) ** 0.5
    if radius > float(noise.spd.radius_max) + 1e-12:
        raise ValueError(f"Arm {arm['arm_id']} lies outside the SPD disk")


def validate_schedule_coverage(
    model_config, arm: dict[str, object], schedule: dict[str, torch.Tensor]
) -> None:
    rho_min = float(schedule["rho"].min())
    rho_max = float(schedule["rho"].max())
    trained_rho_min = float(model_config.noise.rho_min)
    trained_rho_max = float(model_config.noise.rho_max)
    if rho_min < trained_rho_min - 1e-12 or rho_max > trained_rho_max + 1e-12:
        raise ValueError(
            f"Arm {arm['arm_id']} uses rho in [{rho_min:.8g}, {rho_max:.8g}], outside the trained range [{trained_rho_min:.8g}, {trained_rho_max:.8g}]"
        )


def make_bundle_metadata(
    *,
    task: dict[str, object],
    sampling_seed: int,
    sampler_config: dict[str, object],
    schedule: dict[str, torch.Tensor],
    endpoint_seed: int,
) -> dict[str, object]:
    metadata = {
        "task_index": int(task["task_index"]),
        "arm": task["arm"],
        "model": task["model"],
        "sampling_seed": int(sampling_seed),
        "sampler": sampler_config,
        "schedule": tensor_mapping_to_lists(schedule),
    }
    metadata["finalization_seed"] = int(endpoint_seed)
    return metadata


@torch.inference_mode()
def finalize_endpoint(
    sampler,
    last_noisy: torch.Tensor,
    schedule: dict[str, torch.Tensor],
    endpoint_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    set_reproducible_seed(endpoint_seed)
    state = torch.as_tensor(last_noisy).to(device)
    uv = schedule["uv"][-1].to(device=device, dtype=state.dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.monotonic()
    final = sampler.sample(
        state,
        float(schedule["beta"][-1]),
        float(schedule["lam"][-1]),
        float(schedule["rho"][-1]),
        uv,
        1,
    ).squeeze(0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (final.detach().cpu(), time.monotonic() - started)


def write_resolved_experiment_config(
    config,
    config_path: Path,
    arms: list[dict[str, object]],
    models: list[dict[str, object]],
    output_root: Path,
) -> None:
    payload = {
        "source_config": relative_to_root(config_path),
        "resolved_config": OmegaConf.to_container(config, resolve=True),
        "arms": arms,
        "models": models,
        "task_count": len(arms) * len(models),
    }
    output_path = output_root / "config.json"
    output_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_path) + ".lock"):
        write_json_atomic(payload, output_path)


def task_grid(
    arms: list[dict[str, object]], models: list[dict[str, object]]
) -> list[dict[str, object]]:
    return [
        {
            "task_index": arm_index * len(models) + model_index,
            "arm": arm,
            "model": model,
        }
        for arm_index, arm in enumerate(arms)
        for model_index, model in enumerate(models)
    ]


def run_task(
    task: dict[str, object],
    config,
    config_path: Path,
    arms: list[dict[str, object]],
    models: list[dict[str, object]],
    device: torch.device,
    *,
    model_info=None,
    write_config: bool = True,
) -> None:
    arm = task["arm"]
    spec = task["model"]
    model_root = resolve_project_path(config.model_root, ROOT)
    output_root = resolve_project_path(config.output_root, ROOT)
    run_dir = (
        output_root
        / "runs"
        / str(arm["arm_id"])
        / f"model_s{int(spec['training_seed'])}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if write_config:
        write_resolved_experiment_config(config, config_path, arms, models, output_root)
    owns_model_info = model_info is None
    if owns_model_info:
        model_info = load_parameter_conditioned_engressor(
            model_root / str(spec["run_name"]),
            checkpoint=str(spec["checkpoint"]),
            device=device,
        )
    try:
        validate_model_coverage(model_info["config"], arm)
        schedule_cpu = build_schedule(arm, config)
        validate_schedule_coverage(model_info["config"], arm, schedule_cpu)
        schedule_device = {
            name: values.to(device=device, dtype=torch.get_default_dtype())
            for name, values in schedule_cpu.items()
            if name in {"beta", "lam", "rho", "actual_rms", "uv"}
        }
        sampler_config = OmegaConf.to_container(config.sampler, resolve=True)
        records = []
        for sampling_seed_value in config.sampling_seeds:
            sampling_seed = int(sampling_seed_value)
            endpoint_seed = int(config.finalization.seed_offset) + sampling_seed
            bundle_path = run_dir / f"sampling_s{sampling_seed}.pt"
            metadata = make_bundle_metadata(
                task=task,
                sampling_seed=sampling_seed,
                sampler_config=sampler_config,
                schedule=schedule_cpu,
                endpoint_seed=endpoint_seed,
            )
            payload = None
            if bundle_path.is_file():
                candidate = torch.load(
                    bundle_path, map_location="cpu", weights_only=False
                )
                if candidate["metadata"] != metadata:
                    raise ValueError(f"Sampling settings differ from {bundle_path}")
                payload = candidate
                print(f"Reusing {relative_to_root(bundle_path)}", flush=True)
            if payload is None:
                set_reproducible_seed(sampling_seed)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.monotonic()
                generation = annealed_langevin_generate_parameter_path(
                    model_info["sampler"].sample,
                    n_samples=int(config.sampler.n_samples),
                    dim=int(config.data.dim),
                    schedule=schedule_device,
                    n_steps_each=int(config.sampler.n_steps_each),
                    n_mc=int(config.sampler.n_mc),
                    alpha_cap=float(config.sampler.alpha_cap),
                    update_clip=float(config.sampler.update_clip),
                    state_clip=(
                        None
                        if config.sampler.state_clip is None
                        else tuple(
                            (float(value) for value in config.sampler.state_clip)
                        )
                    ),
                    device=device,
                    verbose=False,
                    show_progress=True,
                    progress_desc=f"{arm['arm_id']} model s{spec['training_seed']} sampling s{sampling_seed}",
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                sampler_seconds = time.monotonic() - started
                final, refinalization_seconds = finalize_endpoint(
                    model_info["sampler"],
                    generation["last_noisy"],
                    schedule_cpu,
                    endpoint_seed,
                    device,
                )
                payload = {
                    "metadata": metadata,
                    "elapsed_seconds": sampler_seconds + refinalization_seconds,
                    "final": final,
                    "last_noisy": generation["last_noisy"],
                    "trajectory": generation["trajectory"],
                    "diagnostics": generation["diagnostics"],
                    "base_lr": generation["base_lr"],
                }
                save_torch_atomic(payload, bundle_path)
                print(f"Generated {relative_to_root(bundle_path)}", flush=True)
            records.append(
                {
                    "task_index": int(task["task_index"]),
                    "arm_id": arm["arm_id"],
                    "parameter": arm["parameter"],
                    "value": arm["value"],
                    "model_seed": int(spec["training_seed"]),
                    "run_name": spec["run_name"],
                    "sampling_seed": sampling_seed,
                    "n_samples": int(config.sampler.n_samples),
                    "bundle_path": relative_to_root(bundle_path),
                    "elapsed_seconds": float(payload["elapsed_seconds"]),
                }
            )
            write_csv_atomic(pd.DataFrame(records), run_dir / "records.csv")
    finally:
        if owns_model_info:
            del model_info
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_all_tasks(
    tasks: list[dict[str, object]],
    config,
    config_path: Path,
    arms: list[dict[str, object]],
    models: list[dict[str, object]],
    device: torch.device,
) -> None:
    model_root = resolve_project_path(config.model_root, ROOT)
    output_root = resolve_project_path(config.output_root, ROOT)
    write_resolved_experiment_config(config, config_path, arms, models, output_root)
    for spec in models:
        model_seed = int(spec["training_seed"])
        model_tasks = [
            task for task in tasks if int(task["model"]["training_seed"]) == model_seed
        ]
        if len(model_tasks) != len(arms):
            raise ValueError(
                f"Expected {len(arms)} tasks for model seed {model_seed}, found {len(model_tasks)}"
            )
        print(
            f"Loading model seed {model_seed} for {len(model_tasks)} arms", flush=True
        )
        model_info = load_parameter_conditioned_engressor(
            model_root / str(spec["run_name"]),
            checkpoint=str(spec["checkpoint"]),
            device=device,
        )
        try:
            for task_number, task in enumerate(model_tasks, start=1):
                print(
                    f"Model seed {model_seed}: arm {task_number}/{len(model_tasks)} ({task['arm']['arm_id']})",
                    flush=True,
                )
                run_task(
                    task,
                    config,
                    config_path,
                    arms,
                    models,
                    device,
                    model_info=model_info,
                    write_config=False,
                )
        finally:
            del model_info
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(
        f"Completed {len(tasks)} tasks and {len(tasks) * len(config.sampling_seeds)} bundles.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", dest="run_all", action="store_true")
    selection.add_argument("--task-index", type=int)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    config_path = resolve_project_path(arguments.config, ROOT)
    config = load_resolved_config(config_path)
    arms_path = resolve_project_path(config.arms_file, ROOT)
    arms = load_arms(arms_path)
    models = model_specs(config)
    tasks = task_grid(arms, models)
    device = torch.device(arguments.device)
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA requested but unavailable")
    if arguments.run_all:
        run_all_tasks(tasks, config, config_path, arms, models, device)
        return
    if arguments.task_index is None or not 0 <= arguments.task_index < len(tasks):
        raise ValueError(
            f"Task index must be between 0 and {len(tasks) - 1}, got {arguments.task_index}"
        )
    run_task(tasks[arguments.task_index], config, config_path, arms, models, device)


if __name__ == "__main__":
    main()

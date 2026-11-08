#!/usr/bin/env python3

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_sched
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(project_root / "src" / "engression"))

from engression.engression import Engressor
from engression.loss_func import energy_loss_two_sample
from utils.eight_gauss import eight_gaussians
from utils.noise_geometry import (
    FullSPDNoiseParameterSampler,
    log_sigma_spd_condition,
    repeated_block_energy_score,
    sample_generalized_gaussian_noise,
    sample_spd_2d_uv,
    spd_2d_from_uv,
)
from utils.sampling import get_noise_schedule


def _to_plain_container(value):
    """Convert OmegaConf containers before tensor construction."""
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def random_spd_2d_options_from_config(generalized_cfg) -> dict:
    """Read the random-SPD training distribution from the covariance config."""
    covariance_cfg = getattr(generalized_cfg, "covariance", None)
    if (
        covariance_cfg is None
        or str(getattr(covariance_cfg, "type", "isotropic")) == "isotropic"
    ):
        return {}
    if str(covariance_cfg.type) != "random_spd_2d":
        raise ValueError(
            "generalized.covariance.type must be isotropic or random_spd_2d"
        )

    anchors_cfg = getattr(covariance_cfg, "anchors", None)
    anchors = None if anchors_cfg is None else _to_plain_container(anchors_cfg)
    radius_max = float(getattr(covariance_cfg, "radius_max", 0.7))
    radial_distribution = str(
        getattr(covariance_cfg, "radial_distribution", "uniform_area")
    )
    anchor_probability = float(getattr(covariance_cfg, "anchor_probability", 0.0))

    return {
        "random_spd_radius_max": radius_max,
        "random_spd_radial_distribution": radial_distribution,
        "random_spd_anchors": anchors,
        "random_spd_anchor_probability": anchor_probability,
    }


def condition_dimension(condition: str) -> int:
    """Return the conditioning width used by the Engressor."""
    if condition in {"sigma", "log_sigma"}:
        return 1
    if condition == "log_sigma_spd":
        return 3
    if condition == "full_spd":
        return 5
    raise ValueError(f"Unknown condition={condition}")


def noise_schedule_from_config(noise_cfg) -> np.ndarray:
    """Read an explicit sigma list when present, else build the configured schedule."""
    explicit_sigmas = getattr(noise_cfg, "sigmas", None)
    if explicit_sigmas is not None:
        sigmas = np.asarray(_to_plain_container(explicit_sigmas), dtype=np.float32)
        if sigmas.ndim != 1 or sigmas.size == 0:
            raise ValueError("noise.sigmas must be a non-empty 1D list")
        if np.any(sigmas <= 0):
            raise ValueError("noise.sigmas entries must be positive")
        return sigmas

    sigmas = get_noise_schedule(
        int(noise_cfg.n_sigma_levels),
        float(noise_cfg.sigma_min),
        float(noise_cfg.sigma_max),
        str(noise_cfg.schedule),
    )
    anchor_sigmas = getattr(noise_cfg, "anchor_sigmas", None)
    if anchor_sigmas is not None:
        anchors = np.asarray(_to_plain_container(anchor_sigmas), dtype=np.float32)
        sigmas = np.unique(np.concatenate([sigmas, anchors])).astype(np.float32)
    return sigmas


def print_gpu_memory(device, prefix=""):
    """Print current GPU memory usage if using CUDA."""
    if torch.cuda.is_available() and device != "cpu":
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        print(
            f"{prefix}GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB",
            flush=True,
        )


class GPUBatchGenerator:

    def __init__(
        self,
        n_samples: int,
        batch_size: int,
        radius: float = 2.0,
        std_clean: float = 0.2,
        sigmas: np.ndarray = None,
        mode: str = "gaussian",
        beta_noise: float = 1.4,
        lam: float = 1.8,
        condition: str = "log_sigma",
        random_spd_radius_max: float | None = None,
        random_spd_radial_distribution: str = "uniform_area",
        random_spd_anchors=None,
        random_spd_anchor_probability: float = 0.0,
        full_spd_noise_config=None,
        sigma_sampling: str = "discrete",
        anchor_sigmas: np.ndarray | None = None,
        anchor_probability: float = 0.0,
        device: str = "cuda",
        return_noise_parameters: bool = False,
    ):
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.radius = radius
        self.std_clean = std_clean
        self.mode = mode
        self.beta_noise = beta_noise
        self.lam = lam
        self.condition = condition
        self.random_spd_radius_max = random_spd_radius_max
        self.random_spd_radial_distribution = random_spd_radial_distribution
        self.random_spd_anchors = random_spd_anchors
        self.random_spd_anchor_probability = float(random_spd_anchor_probability)
        self.full_spd_parameter_sampler = (
            FullSPDNoiseParameterSampler(
                full_spd_noise_config,
                dim=2,
                device=device,
                dtype=torch.float32,
            )
            if full_spd_noise_config is not None
            else None
        )
        self.sigma_sampling = sigma_sampling
        self.anchor_probability = float(anchor_probability)
        self.device = device
        self.return_noise_parameters = bool(return_noise_parameters)

        self.sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device)
        self.anchor_sigmas = (
            torch.tensor(anchor_sigmas, dtype=torch.float32, device=device)
            if anchor_sigmas is not None
            else None
        )
        self.log_sigma_min = float(np.log(sigmas.min()))
        self.log_sigma_max = float(np.log(sigmas.max()))
        self.n_sigmas = len(sigmas)
        self.n_anchor_sigmas = (
            0 if self.anchor_sigmas is None else len(self.anchor_sigmas)
        )

        angles = torch.linspace(0, 2 * math.pi * 7 / 8, 8, device=self.device)
        self.means = (
            torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * self.radius
        )

        self.n_batches = n_samples // batch_size

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            yield self._generate_batch()

    def _generate_batch(self):
        B = self.batch_size
        D = 2

        x = eight_gaussians(
            n=B,
            radius=self.radius,
            std=self.std_clean,
            device=self.device,
            means=self.means,
        )

        if self.full_spd_parameter_sampler is not None:
            parameters = self.full_spd_parameter_sampler.sample(B)
            y = self.full_spd_parameter_sampler.corrupt(x, parameters)
            if self.return_noise_parameters:
                return (
                    y,
                    x,
                    parameters.condition,
                    parameters.sigma,
                    parameters.beta,
                    parameters.uv,
                )
            return y, x, parameters.condition, parameters.sigma

        if self.sigma_sampling == "discrete":
            idx = torch.randint(0, self.n_sigmas, (1,), device=self.device)
            sigma = self.sigmas[idx]
            log_sigma = sigma.log()
        elif self.sigma_sampling == "continuous_log_uniform":
            log_sigma = (
                torch.rand((), device=self.device)
                * (self.log_sigma_max - self.log_sigma_min)
                + self.log_sigma_min
            )
            sigma = log_sigma.exp()
        elif self.sigma_sampling == "continuous_log_uniform_with_anchors":
            use_anchor = (
                self.anchor_sigmas is not None
                and torch.rand((), device=self.device).item() < self.anchor_probability
            )
            if use_anchor:
                idx = torch.randint(0, self.n_anchor_sigmas, (1,), device=self.device)
                sigma = self.anchor_sigmas[idx]
                log_sigma = sigma.log()
            else:
                log_sigma = (
                    torch.rand((), device=self.device)
                    * (self.log_sigma_max - self.log_sigma_min)
                    + self.log_sigma_min
                )
                sigma = log_sigma.exp()
        else:
            raise ValueError(f"Unknown sigma_sampling={self.sigma_sampling}")

        shape_uv = None
        batch_noise_shape = None
        if self.random_spd_radius_max is not None:
            shape_uv = sample_spd_2d_uv(
                1,
                radius_max=self.random_spd_radius_max,
                radial_distribution=self.random_spd_radial_distribution,
                anchors=self.random_spd_anchors,
                anchor_probability=self.random_spd_anchor_probability,
                device=self.device,
                dtype=x.dtype,
            )[0]
            batch_noise_shape = spd_2d_from_uv(shape_uv)

        if self.mode == "gaussian":
            eps = torch.randn(B, D, device=self.device) * sigma
        elif self.mode == "generalized":
            eps = sample_generalized_gaussian_noise(
                n=B,
                beta=self.beta_noise,
                lam=self.lam,
                Sigma=batch_noise_shape,
                sigma=sigma,
                dim=D,
                device=self.device,
                dtype=x.dtype,
            )
        else:
            raise ValueError(f"Unknown mode={self.mode}")

        y = x + eps

        if self.condition == "sigma":
            cond = sigma.expand(B, 1)
        elif self.condition == "log_sigma":
            cond = log_sigma.expand(B, 1)
        elif self.condition == "log_sigma_spd":
            if shape_uv is None:
                raise ValueError("log_sigma_spd requires random_spd_2d covariance")
            cond = log_sigma_spd_condition(log_sigma, shape_uv, B)
        else:
            raise ValueError(f"Unknown condition={self.condition}")

        if self.return_noise_parameters:
            return (
                y,
                x,
                cond,
                sigma.reshape(()),
                2.0 if self.mode == "gaussian" else self.beta_noise,
                (
                    shape_uv
                    if self.mode == "generalized" and shape_uv is not None
                    else x.new_zeros(2)
                ),
            )
        return y, x, cond


class CleanEightGaussDataset(Dataset):

    def __init__(self, n_samples: int, radius: float = 2.0, std_clean: float = 0.2):
        self.x_clean = eight_gaussians(
            n_samples, radius=radius, std=std_clean, device="cpu"
        )

    def __len__(self):
        return self.x_clean.shape[0]

    def __getitem__(self, idx):
        return self.x_clean[idx]


def make_noisy_collate_fn(
    sigmas: np.ndarray,
    mode: str = "gaussian",
    beta_noise: float = 1.4,
    lam: float = 1.8,
    condition: str = "log_sigma",
    random_spd_radius_max: float | None = None,
    random_spd_radial_distribution: str = "uniform_area",
    random_spd_anchors=None,
    random_spd_anchor_probability: float = 0.0,
    full_spd_noise_config=None,
    sigma_sampling: str = "discrete",
    anchor_sigmas: np.ndarray | None = None,
    anchor_probability: float = 0.0,
    return_noise_parameters: bool = False,
):
    sigmas = np.asarray(sigmas, dtype=np.float64)
    anchor_sigmas = (
        np.asarray(anchor_sigmas, dtype=np.float64)
        if anchor_sigmas is not None
        else None
    )
    anchor_probability = float(anchor_probability)
    log_sigma_min = float(np.log(sigmas.min()))
    log_sigma_max = float(np.log(sigmas.max()))
    full_spd_parameter_sampler = (
        FullSPDNoiseParameterSampler(
            full_spd_noise_config,
            dim=2,
            device="cpu",
            dtype=torch.float32,
        )
        if full_spd_noise_config is not None
        else None
    )

    def collate(batch):
        x = torch.stack(batch, dim=0)
        B, D = x.shape

        if full_spd_parameter_sampler is not None:
            parameters = full_spd_parameter_sampler.sample(B)
            y = full_spd_parameter_sampler.corrupt(x, parameters)
            if return_noise_parameters:
                return (
                    y,
                    x,
                    parameters.condition,
                    parameters.sigma,
                    parameters.beta,
                    parameters.uv,
                )
            return y, x, parameters.condition, parameters.sigma

        if sigma_sampling == "discrete":
            sigma = float(np.random.choice(sigmas))
        elif sigma_sampling == "continuous_log_uniform":
            log_sigma = float(np.random.uniform(log_sigma_min, log_sigma_max))
            sigma = float(np.exp(log_sigma))
        elif sigma_sampling == "continuous_log_uniform_with_anchors":
            if anchor_sigmas is not None and np.random.rand() < anchor_probability:
                sigma = float(np.random.choice(anchor_sigmas))
            else:
                log_sigma = float(np.random.uniform(log_sigma_min, log_sigma_max))
                sigma = float(np.exp(log_sigma))
        else:
            raise ValueError(f"Unknown sigma_sampling={sigma_sampling}")

        shape_uv = None
        batch_noise_shape = None
        if random_spd_radius_max is not None:
            shape_uv = sample_spd_2d_uv(
                1,
                radius_max=random_spd_radius_max,
                radial_distribution=random_spd_radial_distribution,
                anchors=random_spd_anchors,
                anchor_probability=random_spd_anchor_probability,
                device=x.device,
                dtype=x.dtype,
            )[0]
            batch_noise_shape = spd_2d_from_uv(shape_uv)

        if mode == "gaussian":
            eps = torch.randn_like(x) * sigma
        elif mode == "generalized":
            eps = sample_generalized_gaussian_noise(
                n=B,
                beta=beta_noise,
                lam=lam,
                Sigma=batch_noise_shape,
                sigma=sigma,
                dim=D,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            raise ValueError(f"Unknown mode={mode}")

        y = x + eps

        if condition == "sigma":
            cond = torch.full((B, 1), sigma, device=x.device)
        elif condition == "log_sigma":
            cond = torch.full((B, 1), np.log(sigma), device=x.device)
        elif condition == "log_sigma_spd":
            if shape_uv is None:
                raise ValueError("log_sigma_spd requires random_spd_2d covariance")
            cond = log_sigma_spd_condition(math.log(sigma), shape_uv, B)
        else:
            raise ValueError(f"Unknown condition={condition}")

        if return_noise_parameters:
            return (
                y,
                x,
                cond,
                torch.as_tensor(sigma, device=x.device, dtype=x.dtype),
                2.0 if mode == "gaussian" else beta_noise,
                (
                    shape_uv
                    if mode == "generalized" and shape_uv is not None
                    else x.new_zeros(2)
                ),
            )
        return y, x, cond

    return collate


def create_run_directory(
    base_dir, mode, use_timestamp=True, timestamp_format="%Y%m%d_%H%M%S"
):
    """Create a versioned run directory."""
    base_dir = Path(base_dir)

    if use_timestamp:
        timestamp = datetime.now().strftime(timestamp_format)
        run_name = f"{mode}_{timestamp}"
    else:

        existing_runs = [d for d in base_dir.glob(f"{mode}_run_*") if d.is_dir()]
        run_numbers = [
            int(d.name.split("_")[-1])
            for d in existing_runs
            if d.name.split("_")[-1].isdigit()
        ]
        next_num = max(run_numbers, default=0) + 1
        run_name = f"{mode}_run_{next_num:03d}"

    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def save_config(config, run_dir):
    """Save the config to the run directory."""
    config_path = run_dir / "config.yaml"
    OmegaConf.save(config, config_path)
    print(f"Config saved to: {config_path}", flush=True)


def train_conditional_engression(
    config,
    mode: str = "gaussian",
    resume_dir: str = None,
    resume_epoch: int = None,
    result_dir: str = None,
):
    """
    Train conditional engression model for denoising.

    The model learns P(X_clean | X_noisy, noise condition).

    Args:
        config: OmegaConf config object
        mode: "gaussian" or "generalized"
        resume_dir: Directory containing checkpoint to resume from
        resume_epoch: Epoch number to resume from
        result_dir: Directory to save results (overrides config if provided)
    """
    print(f"\n{'='*60}", flush=True)
    print(f"Starting training: {mode}", flush=True)
    if resume_dir:
        print(f"Resuming from: {resume_dir}, epoch {resume_epoch}", flush=True)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'='*60}\n", flush=True)

    n_samples = config.data.n_samples
    radius = config.data.radius
    std_clean = config.data.std_clean

    sigma_min = config.noise.sigma_min
    sigma_max = config.noise.sigma_max
    schedule = config.noise.schedule
    condition = config.noise.condition
    full_spd_condition = condition == "full_spd"

    num_epochs = config.training.num_epochs
    batch_size = config.training.batch_size
    lr = config.training.lr
    accum_steps = config.training.accum_steps
    print_every = config.training.print_every
    seed = config.training.seed
    grad_clip = getattr(config.training, "grad_clip", False)
    grad_clip_norm = getattr(config.training, "grad_clip_norm", 1.0)
    snapshot_every = int(config.training.snapshot_every)
    energy_score = str(config.training.get("energy_score", "euclidean"))
    if energy_score not in {"euclidean", "noise_matched"}:
        raise ValueError("training.energy_score must be euclidean or noise_matched")
    config.training.energy_score = energy_score
    noise_matched_energy = energy_score == "noise_matched"

    num_layers = config.model.num_layers
    hidden_dim = config.model.hidden_dim
    noise_dim = config.model.noise_dim
    noise_floor = config.model.noise_floor
    beta = config.model.beta
    vary_stochasticity = config.model.vary_stochasticity
    film_mlp_hidden = config.model.get("film_mlp_hidden", None)

    device = config.system.device

    if mode == "generalized" and full_spd_condition:
        beta_noise = None
        lam = None
        random_spd_options = {}
        FullSPDNoiseParameterSampler(
            config.noise,
            dim=2,
            device="cpu",
            dtype=torch.float32,
        )
    elif mode == "generalized":
        beta_noise = config.generalized.beta_noise
        lam = config.generalized.lam
        random_spd_options = random_spd_2d_options_from_config(config.generalized)
    else:
        beta_noise = 2.0
        lam = 1.0
        random_spd_options = {}

    if full_spd_condition and mode != "generalized":
        raise ValueError("noise.condition=full_spd requires generalized noise")
    if random_spd_options and condition != "log_sigma_spd":
        raise ValueError(
            "random_spd_2d covariance requires noise.condition=log_sigma_spd"
        )
    if condition == "log_sigma_spd" and not random_spd_options:
        raise ValueError(
            "noise.condition=log_sigma_spd requires generalized.covariance.type=random_spd_2d"
        )
    condition_dim = condition_dimension(condition)

    sigma_weight_enabled = getattr(
        config.training, "sigma_weight", None
    ) is not None and getattr(
        getattr(config.training, "sigma_weight", None), "enabled", False
    )
    if sigma_weight_enabled:
        sigma_weight_mode = getattr(config.training.sigma_weight, "mode", "none")
        sigma_weight_power = float(getattr(config.training.sigma_weight, "power", 2.0))
    else:
        sigma_weight_mode = "none"
        sigma_weight_power = 2.0

    print(f"Setting random seed: {seed}", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Creating run directory...", flush=True)

    if resume_dir:

        run_dir = Path(resume_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {run_dir}")
        print(f"Resuming in existing directory: {run_dir}", flush=True)
    elif result_dir:

        run_dir = Path(result_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using provided result directory: {run_dir}", flush=True)

        save_config(config, run_dir)
    else:

        base_dir = project_root / config.system.run_dir
        run_dir = create_run_directory(
            base_dir,
            mode,
            use_timestamp=config.versioning.use_timestamp,
            timestamp_format=config.versioning.timestamp_format,
        )

        save_config(config, run_dir)

    print(f"\n{'='*60}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    print(f"Generating noise schedule...", flush=True)
    sigmas = noise_schedule_from_config(config.noise)
    sigma_min = float(sigmas.min())
    sigma_max = float(sigmas.max())
    anchor_sigmas_cfg = getattr(config.noise, "anchor_sigmas", None)
    anchor_sigmas = (
        np.asarray(_to_plain_container(anchor_sigmas_cfg), dtype=np.float32)
        if anchor_sigmas_cfg is not None
        else None
    )
    anchor_probability = float(getattr(config.noise, "anchor_probability", 0.0))
    sigma_sampling = getattr(config.noise, "sigma_sampling", "discrete")
    if anchor_sigmas is not None:
        if anchor_sigmas.ndim != 1 or anchor_sigmas.size == 0:
            raise ValueError("noise.anchor_sigmas must be a non-empty 1D list")
        if np.any(anchor_sigmas <= 0):
            raise ValueError("noise.anchor_sigmas entries must be positive")
        if np.any(anchor_sigmas < sigma_min) or np.any(anchor_sigmas > sigma_max):
            raise ValueError(
                "noise.anchor_sigmas must lie inside the training sigma range"
            )
    if sigma_sampling == "continuous_log_uniform_with_anchors":
        if anchor_sigmas is None:
            raise ValueError(
                "noise.anchor_sigmas is required for continuous_log_uniform_with_anchors"
            )
        if not (0.0 < anchor_probability <= 1.0):
            raise ValueError("noise.anchor_probability must be in (0, 1]")

    log_sigmas = np.log(sigmas)
    bin_edges = np.concatenate(
        [[-np.inf], 0.5 * (log_sigmas[:-1] + log_sigmas[1:]), [np.inf]]
    )

    schedule_label = (
        "explicit" if getattr(config.noise, "sigmas", None) is not None else schedule
    )
    print(f"\nNoise schedule ({schedule_label}):", flush=True)
    print(f"  sigma_min: {sigmas.min():.4f}", flush=True)
    print(f"  sigma_max: {sigmas.max():.4f}", flush=True)
    print(f"  n_levels: {len(sigmas)}", flush=True)
    if anchor_sigmas is not None:
        print(f"  anchor_sigmas: {anchor_sigmas.tolist()}", flush=True)
        print(f"  anchor_probability: {anchor_probability}", flush=True)

    use_gpu_batches = getattr(config.data, "gpu_batch_gen", False)

    if use_gpu_batches and device != "cpu":

        print(f"\nUsing GPU batch generator (bypassing DataLoader)...", flush=True)
        batch_generator = GPUBatchGenerator(
            n_samples=n_samples,
            batch_size=batch_size,
            radius=radius,
            std_clean=std_clean,
            sigmas=sigmas,
            mode=mode,
            beta_noise=beta_noise,
            lam=lam,
            condition=condition,
            **random_spd_options,
            full_spd_noise_config=config.noise if full_spd_condition else None,
            sigma_sampling=sigma_sampling,
            anchor_sigmas=anchor_sigmas,
            anchor_probability=anchor_probability,
            device=device,
            return_noise_parameters=noise_matched_energy,
        )
        n_batches_per_epoch = len(batch_generator)
        print(
            f"GPU batch generator created: {n_batches_per_epoch} batches per epoch",
            flush=True,
        )

        dataloader = None

    else:

        print(f"\nCreating dataset...", flush=True)
        dataset = CleanEightGaussDataset(
            n_samples=n_samples,
            radius=radius,
            std_clean=std_clean,
        )
        print(f"Dataset created with {len(dataset)} samples", flush=True)

        collate_fn = make_noisy_collate_fn(
            sigmas=sigmas,
            mode=mode,
            beta_noise=beta_noise,
            lam=lam,
            condition=condition,
            **random_spd_options,
            full_spd_noise_config=config.noise if full_spd_condition else None,
            sigma_sampling=sigma_sampling,
            anchor_sigmas=anchor_sigmas,
            anchor_probability=anchor_probability,
            return_noise_parameters=noise_matched_energy,
        )

        configured_workers = config.system.get("num_workers", None)
        if configured_workers is not None:
            num_workers = int(configured_workers)
        elif device == "cpu":
            num_workers = 0
        else:
            num_workers = min(os.cpu_count(), 8) if os.cpu_count() is not None else 2
        print(f"Creating dataloader with {num_workers} workers...", flush=True)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device != "cpu"),
            collate_fn=collate_fn,
        )
        n_batches_per_epoch = len(dataloader)
        batch_generator = None
        print(
            f"Dataloader created: {n_batches_per_epoch} batches per epoch", flush=True
        )

    if accum_steps < 1:
        raise ValueError("training.accum_steps must be at least 1")
    if n_batches_per_epoch < 1:
        raise ValueError(
            "No training batches were created; reduce batch_size or increase n_samples"
        )
    full_groups, remainder = divmod(n_batches_per_epoch, accum_steps)

    print(f"\nTraining configuration:", flush=True)
    print(f"  Mode: {mode}", flush=True)
    print(f"  N samples: {n_samples}", flush=True)
    print(f"  Batch size: {batch_size}", flush=True)
    print(f"  Batches per epoch: {n_batches_per_epoch}", flush=True)
    print(f"  GPU batch generation: {use_gpu_batches and device != 'cpu'}", flush=True)
    print(f"  Epochs: {num_epochs}", flush=True)
    print(f"  Learning rate: {lr}", flush=True)
    print(f"  Gradient accumulation steps: {accum_steps}", flush=True)
    print(
        f"  Gradient clipping: {grad_clip} (norm={grad_clip_norm if grad_clip else 'N/A'})",
        flush=True,
    )
    print(f"  Vary stochasticity: {vary_stochasticity}", flush=True)
    if vary_stochasticity:
        print(f"    Noise floor: {noise_floor}", flush=True)
    print(f"  Sigma weighting: {sigma_weight_enabled}", flush=True)
    if sigma_weight_enabled:
        print(f"    Mode: {sigma_weight_mode}", flush=True)
        print(f"    Power: {sigma_weight_power}", flush=True)
    if full_spd_condition:
        print(
            "  Parameter sampling: continuous in beta, log lambda, log rho, "
            "and the SPD disk, mixed with exact tuple anchors",
            flush=True,
        )
    else:
        print(
            f"  Sigma sampling: {sigma_sampling}",
            flush=True,
        )
    print(f"  Condition type: {condition}", flush=True)
    print(f"  Device: {device}", flush=True)
    if mode == "generalized":
        if full_spd_condition:
            print(
                f"  Beta range: [{config.noise.beta_min}, {config.noise.beta_max}]",
                flush=True,
            )
            print(
                "  Lambda range: "
                f"[{config.noise.lambda_min}, {config.noise.lambda_max}]",
                flush=True,
            )
            print(
                f"  Rho range: [{config.noise.rho_min}, {config.noise.rho_max}]",
                flush=True,
            )
            print(
                f"  SPD radius: {config.noise.spd.radius_max}",
                flush=True,
            )
            print(
                f"  Full-tuple anchor probability: {config.noise.anchor_probability}",
                flush=True,
            )
        else:
            print(f"  Beta (noise): {beta_noise}", flush=True)
            print(f"  Lambda: {lam}", flush=True)
        if random_spd_options:
            print(
                f"  Random-SPD radius: {random_spd_options['random_spd_radius_max']}",
                flush=True,
            )
            print(
                "  Random-SPD radial distribution: "
                f"{random_spd_options['random_spd_radial_distribution']}",
                flush=True,
            )
    print(f"", flush=True)

    if resume_dir and resume_epoch:

        print(f"Loading checkpoint from epoch {resume_epoch}...", flush=True)
        ckpt_path = run_dir / f"epoch{resume_epoch}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        engressor = torch.load(ckpt_path, map_location=device, weights_only=False)
        print(f"Checkpoint loaded from: {ckpt_path}", flush=True)
        if int(getattr(engressor, "condition_dim", -1)) != condition_dim:
            raise ValueError(
                "Checkpoint condition dimension does not match the configured condition"
            )

        engressor.lr = lr
        engressor.num_epochs = num_epochs
        engressor.batch_size = batch_size

        start_epoch = resume_epoch
        print(f"Resuming training from epoch {start_epoch + 1}", flush=True)
    else:

        print(f"Creating Engressor model...", flush=True)
        engressor = Engressor(
            in_dim=2,
            out_dim=2,
            condition_dim=condition_dim,
            classification=False,
            pre_anm=False,
            noise_std=1,
            standardize=config.model.standardize,
            noise_all_layer=config.model.noise_all_layer,
            num_layer=num_layers,
            hidden_dim=hidden_dim,
            noise_dim=noise_dim,
            out_act=config.model.out_act,
            resblock=config.model.resblock,
            add_bn=config.model.add_bn,
            film_pos=config.model.film_pos,
            film_level=config.model.film_level,
            film_mlp_hidden=film_mlp_hidden,
            standardize_condition=config.model.standardize_condition,
            beta=beta,
            lr=lr,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device,
            check_device=True,
            verbose=False,
        )
        start_epoch = 0

    print(f"\nModel architecture:", flush=True)
    print(f"  Input dim: 2", flush=True)
    print(f"  Output dim: 2", flush=True)
    print(f"  Condition dim: {condition_dim}", flush=True)
    print(f"  Layers: {num_layers}", flush=True)
    print(f"  Hidden dim: {hidden_dim}", flush=True)
    print(f"  Noise dim: {noise_dim}", flush=True)
    print(f"  Beta (energy): {beta}", flush=True)
    print(f"  FiLM position: {config.model.film_pos}", flush=True)
    print(f"  FiLM level: {config.model.film_level}", flush=True)
    print(f"  Standardize inputs: {config.model.standardize}", flush=True)
    print(f"  Standardize condition: {config.model.standardize_condition}", flush=True)

    n_params = sum(p.numel() for p in engressor.model.parameters() if p.requires_grad)
    print(f"  Total parameters: {n_params:,}", flush=True)

    print(f"", flush=True)
    print_gpu_memory(device, "Initial ")

    print(f"\n{'='*60}", flush=True)
    print("Starting training loop...", flush=True)
    print(f"{'='*60}\n", flush=True)

    scheduler = None
    scheduler_cfg = config.training.get("scheduler", None)
    if scheduler_cfg is not None and getattr(scheduler_cfg, "type", None) is not None:

        for g in engressor.optimizer.param_groups:
            g["lr"] = config.training.lr

        if scheduler_cfg.type == "multistep":

            milestones = list(scheduler_cfg.milestones)
            gamma = float(scheduler_cfg.gamma)

            scheduler = lr_sched.MultiStepLR(
                engressor.optimizer,
                milestones=milestones,
                gamma=gamma,
                last_epoch=start_epoch - 1,
            )
            print(
                f"Using MultiStepLR scheduler: milestones={milestones}, gamma={gamma}",
                flush=True,
            )
        elif scheduler_cfg.type == "cosine":
            eta_min = float(getattr(scheduler_cfg, "eta_min", 1e-9))
            scheduler = lr_sched.CosineAnnealingLR(
                engressor.optimizer,
                T_max=config.training.num_epochs,
                eta_min=eta_min,
                last_epoch=start_epoch - 1,
            )
            print(
                f"Using CosineAnnealingLR scheduler: T_max={config.training.num_epochs}, eta_min={eta_min}",
                flush=True,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_cfg.type}")

    engressor.train_mode()
    losses = []

    start_time = datetime.now()
    print(
        f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True
    )
    print(f"", flush=True)

    engressor.model.train()

    prev_lr = engressor.optimizer.param_groups[0]["lr"]

    for epoch in range(start_epoch, num_epochs):

        epoch_start = datetime.now()

        epoch_loss = torch.tensor(0.0, device=device)
        epoch_loss1 = torch.tensor(0.0, device=device)
        epoch_loss2 = torch.tensor(0.0, device=device)

        batch_log_sigmas = []
        batch_losses = []

        engressor.optimizer.zero_grad(set_to_none=True)
        batch_idx = 0

        data_source = batch_generator if batch_generator is not None else dataloader

        for batch_idx, batch in enumerate(data_source):
            if noise_matched_energy:
                (
                    y_batch,
                    x_batch,
                    cond_batch,
                    sampled_sigma,
                    sampled_beta,
                    sampled_uv,
                ) = batch
            elif len(batch) == 4:
                y_batch, x_batch, cond_batch, sampled_sigma = batch
            else:
                y_batch, x_batch, cond_batch = batch
                sampled_sigma = None

            if dataloader is not None:
                y_batch = y_batch.to(device, non_blocking=(device != "cpu"))
                x_batch = x_batch.to(device, non_blocking=(device != "cpu"))
                cond_batch = cond_batch.to(device, non_blocking=(device != "cpu"))
                if sampled_sigma is not None:
                    sampled_sigma = sampled_sigma.to(
                        device,
                        non_blocking=(device != "cpu"),
                    )

            if sampled_sigma is not None:
                sigma = sampled_sigma
                log_sigma = sigma.log()
            elif condition in {"log_sigma", "log_sigma_spd"}:
                log_sigma = cond_batch[0, 0]
                sigma = log_sigma.exp()
            else:
                sigma = cond_batch[0, 0]
                log_sigma = sigma.log()

            if vary_stochasticity:
                noise_std = torch.clamp(
                    (sigma / sigma_max).sqrt(), min=float(noise_floor)
                ).item()
                engressor.model.set_noise_std(noise_std)

            x1 = engressor.model(y_batch, cond_batch)
            x2 = engressor.model(y_batch, cond_batch)

            if noise_matched_energy:
                loss, loss1, loss2 = repeated_block_energy_score(
                    x_batch,
                    x1,
                    x2,
                    beta=sampled_beta,
                    sigma=sampled_sigma,
                    uv=sampled_uv,
                )
            else:
                loss, loss1, loss2 = energy_loss_two_sample(
                    x0=x_batch, x=x1, xp=x2, beta=engressor.beta, verbose=True
                )

            if sigma_weight_enabled:
                if sigma_weight_mode == "sigma2":
                    w = sigma**sigma_weight_power
                elif sigma_weight_mode == "sigma2_norm":
                    w = (sigma / sigma_max) ** sigma_weight_power
                elif sigma_weight_mode == "inv_sigma2":
                    w = (1.0 / sigma) ** sigma_weight_power
                elif sigma_weight_mode == "inv_sigma2_norm":
                    w = (sigma_min / sigma) ** sigma_weight_power
                elif sigma_weight_mode == "none":
                    w = 1.0
                else:
                    raise ValueError(f"Unknown sigma_weight.mode: {sigma_weight_mode}")

                loss = loss * w

            group_id = batch_idx // accum_steps

            if group_id < full_groups:
                group_size = accum_steps
            else:

                group_size = remainder if remainder > 0 else accum_steps

            (loss / group_size).backward()

            epoch_loss += loss.detach()
            epoch_loss1 += loss1.detach()
            epoch_loss2 += loss2.detach()

            batch_log_sigmas.append(log_sigma.detach())
            batch_losses.append(loss.detach())

            is_group_end = ((batch_idx + 1) % accum_steps == 0) or (
                batch_idx + 1 == n_batches_per_epoch
            )
            if is_group_end:
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        engressor.model.parameters(), max_norm=grad_clip_norm
                    )
                engressor.optimizer.step()
                engressor.optimizer.zero_grad(set_to_none=True)

        num_batches = batch_idx + 1
        avg_loss = (epoch_loss / num_batches).item()
        avg_loss1 = (epoch_loss1 / num_batches).item()
        avg_loss2 = (epoch_loss2 / num_batches).item()
        losses.append(avg_loss)

        if scheduler is not None:
            scheduler.step()
            new_lr = engressor.optimizer.param_groups[0]["lr"]
            if np.abs(new_lr - prev_lr) > 1e-6:
                print(
                    f"[Epoch {epoch+1}] Learning rate changed: {prev_lr:.2e} -> {new_lr:.2e}",
                    flush=True,
                )
                prev_lr = new_lr

        if (epoch + 1) % print_every == 0 or epoch == start_epoch:

            epoch_time = (datetime.now() - epoch_start).total_seconds()
            elapsed_time = (datetime.now() - start_time).total_seconds()
            print(
                f"[Epoch {epoch+1:4d}/{num_epochs}] "
                f"Loss: {avg_loss:.6f}, E|X-X̂|: {avg_loss1:.6f}, E|X̂-X̂'|: {avg_loss2:.6f} "
                f"Current LR: {engressor.optimizer.param_groups[0]["lr"]:.3e} "
                f"| Time: {epoch_time:.2f}s, Elapsed: {elapsed_time:.1f}s",
                flush=True,
            )

            print_gpu_memory(device, "  ")

            remaining_epochs = num_epochs - (epoch + 1)
            est_remaining_time = (
                elapsed_time / (epoch + 1 - start_epoch)
            ) * remaining_epochs
            print(
                f"  -> Estimated remaining time: {est_remaining_time/60:.1f} minutes",
                flush=True,
            )

            if len(batch_log_sigmas) > 0:

                log_sigma_tensor = torch.stack(batch_log_sigmas)
                loss_tensor = torch.stack(batch_losses)

                bin_edges_gpu = torch.tensor(
                    bin_edges, device=device, dtype=torch.float32
                )
                bin_indices = torch.searchsorted(bin_edges_gpu, log_sigma_tensor) - 1
                bin_indices = torch.clamp(bin_indices, 0, len(sigmas) - 1)

                loss_sum_by_level = torch.zeros(len(sigmas), device=device)
                count_by_level = torch.zeros(len(sigmas), device=device)

                loss_sum_by_level.scatter_add_(0, bin_indices, loss_tensor)
                count_by_level.scatter_add_(
                    0, bin_indices, torch.ones_like(loss_tensor)
                )

                loss_sum_np = loss_sum_by_level.cpu().numpy()
                count_np = count_by_level.cpu().numpy()
                avg_by_level = loss_sum_np / np.maximum(count_np, 1.0)

                nonzero = np.flatnonzero(count_np > 0)
                if nonzero.size <= 10:
                    idxs = nonzero
                else:
                    pick = np.linspace(0, nonzero.size - 1, 10)
                    idxs = nonzero[np.round(pick).astype(int)]
                    idxs = np.unique(idxs)

                sigma_centers = sigmas

                print(
                    f"  Loss by sigma level: total_batches={int(count_np.sum())}",
                    flush=True,
                )
                for ii in idxs:
                    print(
                        f"    sigma≈{sigma_centers[ii]:.4g}  avg_loss={avg_by_level[ii]:.6g}  n={int(count_np[ii])}",
                        flush=True,
                    )

            batch_log_sigmas = []
            batch_losses = []

        if (epoch + 1) % snapshot_every == 0:
            ckpt_path = run_dir / f"epoch{epoch+1}.pt"
            try:
                torch.save(engressor, ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}", flush=True)
            except Exception as e:
                print(f"  Failed saving checkpoint at epoch {epoch+1}: {e}", flush=True)

    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*60}", flush=True)
    print(f"Training completed!", flush=True)
    print(
        f"Total training time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)",
        flush=True,
    )
    print(f"Average time per epoch: {total_time/num_epochs:.2f} seconds", flush=True)
    print_gpu_memory(device, "Final ")
    print(f"{'='*60}\n", flush=True)

    print(f"Saving model...", flush=True)
    model_path = run_dir / "model.pt"
    torch.save(engressor, model_path)
    print(f"Engressor saved to: {model_path}", flush=True)

    losses_path = run_dir / "losses.npy"
    np.save(losses_path, np.array(losses))
    print(f"Training losses saved to: {losses_path}", flush=True)
    print(f"Final loss: {losses[-1]:.6f}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Run completed successfully!", flush=True)
    print(f"Results saved in: {run_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    return engressor, run_dir


def main():

    parser = argparse.ArgumentParser(
        description="Train conditional engression for denoising"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the training config"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["gaussian", "generalized"],
        help="Noise type",
    )

    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sigma_min", type=float, default=None)
    parser.add_argument("--sigma_max", type=float, default=None)

    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help="Directory containing checkpoint to resume from",
    )
    parser.add_argument(
        "--resume_epoch", type=int, default=None, help="Epoch number to resume from"
    )

    parser.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help="Directory to save results (overrides config)",
    )

    args, dotlist = parser.parse_known_args()

    print("\n" + "=" * 60, flush=True)
    print("8-Gaussians Conditional Engression Training", flush=True)
    print("=" * 60 + "\n", flush=True)

    config_path = project_root / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loading config from: {config_path}", flush=True)

    config = OmegaConf.load(config_path)

    if args.num_epochs is not None:
        config.training.num_epochs = args.num_epochs
        print(f"Override: num_epochs = {args.num_epochs}", flush=True)
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
        print(f"Override: batch_size = {args.batch_size}", flush=True)
    if args.lr is not None:
        config.training.lr = args.lr
        print(f"Override: lr = {args.lr}", flush=True)
    if args.device is not None:
        config.system.device = args.device
        print(f"Override: device = {args.device}", flush=True)
    if args.seed is not None:
        config.training.seed = args.seed
        print(f"Override: seed = {args.seed}", flush=True)
    if args.sigma_min is not None:
        config.noise.sigma_min = args.sigma_min
        print(f"Override: sigma_min = {args.sigma_min}", flush=True)
    if args.sigma_max is not None:
        config.noise.sigma_max = args.sigma_max
        print(f"Override: sigma_max = {args.sigma_max}", flush=True)

    if len(dotlist) > 0:
        print("Dotlist overrides:", dotlist, flush=True)
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(dotlist))

    print("\n" + "=" * 60, flush=True)
    print("Configuration:", flush=True)
    print("=" * 60, flush=True)
    print(OmegaConf.to_yaml(config), flush=True)
    print("=" * 60 + "\n", flush=True)

    train_conditional_engression(
        config,
        mode=args.mode,
        resume_dir=args.resume_dir,
        resume_epoch=args.resume_epoch,
        result_dir=args.result_dir,
    )


if __name__ == "__main__":
    main()

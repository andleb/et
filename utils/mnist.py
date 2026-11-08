from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from filelock import FileLock
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.utils import make_grid, save_image

from engression.engression import Engressor

from .noise_geometry import (
    generalized_gaussian_actual_rms,
    normalize_full_spd_condition,
    reference_rho_to_sigma,
    spd_2d_from_uv,
)

MNIST_SIDE = 28


MNIST_DIM = MNIST_SIDE * MNIST_SIDE


class MNISTUint8Dataset(Dataset):

    def __init__(self, root: str | Path, *, train: bool, download: bool):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        if download:
            with FileLock(str(root / ".mnist-download.lock"), timeout=1800):
                dataset = MNIST(root=str(root), train=bool(train), download=True)
        else:
            dataset = MNIST(root=str(root), train=bool(train), download=False)
        self.data = dataset.data
        self.targets = dataset.targets

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.targets[index]


def seed_mnist_worker(worker_id: int) -> None:
    """Seed NumPy and Python from Torch's deterministic DataLoader worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_mnist_loader(
    root: str | Path,
    *,
    train: bool,
    batch_size: int,
    seed: int,
    num_workers: int,
    download: bool,
    shuffle: bool | None = None,
    drop_last: bool | None = None,
    generator_state: torch.Tensor | None = None,
) -> tuple[DataLoader, torch.Generator]:
    """Build a seeded MNIST loader and return its explicit CPU generator."""
    dataset = MNISTUint8Dataset(root, train=train, download=download)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    if generator_state is not None:
        generator.set_state(generator_state.cpu())
    if shuffle is None:
        shuffle = bool(train)
    if drop_last is None:
        drop_last = bool(train)
    workers = int(num_workers)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=bool(drop_last),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=seed_mnist_worker,
        generator=generator,
    )
    return loader, generator


def dequantize_mnist(
    images: torch.Tensor,
    *,
    device: str | torch.device,
) -> torch.Tensor:
    """Uniformly dequantize uint8 images and flatten them to 784-vectors."""
    images = images.to(device=device, dtype=torch.float32, non_blocking=True)
    images = (images + torch.rand_like(images)) / 256.0
    return images.reshape(images.shape[0], MNIST_DIM)


def mnist_vectors_to_images(vectors: torch.Tensor) -> torch.Tensor:
    """Reshape flattened MNIST vectors to ``(N, 1, 28, 28)``."""
    if vectors.ndim != 2 or vectors.shape[1] != MNIST_DIM:
        raise ValueError(f"vectors must have shape (N, {MNIST_DIM})")
    return vectors.reshape(vectors.shape[0], 1, MNIST_SIDE, MNIST_SIDE)


def make_mnist_image_grid(
    vectors: torch.Tensor,
    *,
    nrow: int,
    padding: int = 2,
    pad_value: float = 1.0,
) -> torch.Tensor:
    """Return a clamped TorchVision grid from flattened MNIST vectors."""
    images = mnist_vectors_to_images(vectors.detach().cpu().clamp(0.0, 1.0))
    return make_grid(
        images,
        nrow=int(nrow),
        padding=int(padding),
        pad_value=float(pad_value),
    )


def save_mnist_image_grid(
    vectors: torch.Tensor,
    path: str | Path,
    *,
    nrow: int,
) -> None:
    """Save a simple clamped image grid from flattened MNIST vectors."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = mnist_vectors_to_images(vectors.detach().cpu().clamp(0.0, 1.0))
    save_image(images, path, nrow=int(nrow), padding=2, pad_value=1.0)


def save_mnist_progression_grid(
    trajectory: list[torch.Tensor],
    *,
    level_indices: list[int],
    n_chains: int,
    path: str | Path,
) -> None:
    """Save selected sampler levels as rows for the same fixed chains."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in level_indices:
        state = trajectory[int(index)][: int(n_chains)].clamp(0.0, 1.0)
        rows.append(mnist_vectors_to_images(state))
    save_image(
        torch.cat(rows, dim=0),
        path,
        nrow=int(n_chains),
        padding=2,
        pad_value=1.0,
    )


def build_mnist_engressor(config: Any, device: str | torch.device) -> Engressor:
    """Construct the configured conditional stochastic residual MLP."""
    model_config = config.model
    return Engressor(
        in_dim=MNIST_DIM,
        out_dim=MNIST_DIM,
        condition_dim=5,
        pre_anm=True,
        noise_std=float(model_config.noise_std),
        noise_all_layer=bool(model_config.noise_all_layer),
        num_layer=int(model_config.num_layers),
        hidden_dim=int(model_config.hidden_dim),
        noise_dim=int(model_config.noise_dim),
        out_act=str(model_config.out_act),
        add_bn=bool(model_config.add_bn),
        resblock=bool(model_config.resblock),
        film_pos=str(model_config.film_pos),
        film_level=int(model_config.film_level),
        film_mlp_hidden=int(model_config.film_mlp_hidden),
        standardize=False,
        standardize_condition=False,
        beta=float(model_config.energy_beta),
        lr=float(config.training.lr),
        device=device,
        check_device=False,
        verbose=False,
    )


def build_mnist_scheduler(optimizer: torch.optim.Optimizer, training_config: Any):
    """Build linear warmup followed by conservative cosine decay."""
    max_steps = int(training_config.max_steps)
    warmup_steps = int(training_config.warmup_steps)
    min_lr = float(training_config.min_lr)
    if max_steps <= 0 or not 0 <= warmup_steps < max_steps:
        raise ValueError("training requires 0 <= warmup_steps < max_steps")
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, max_steps - warmup_steps),
        eta_min=min_lr,
    )
    if warmup_steps == 0:
        return cosine
    warmup = LinearLR(
        optimizer,
        start_factor=float(training_config.warmup_start_factor),
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def build_mnist_preview_schedule(
    noise_config: Any,
    generation_config: Any,
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Construct the training-preview schedule."""
    levels = int(generation_config.n_levels)
    if levels <= 1:
        raise ValueError("generation.n_levels must exceed one")
    progress = torch.linspace(0.0, 1.0, levels, device=device, dtype=dtype)
    preview = generation_config.preview_path
    beta = torch.full_like(progress, float(preview.beta))
    lam = torch.full_like(progress, float(preview.lam))
    u = torch.full_like(progress, float(preview.u))
    v = torch.full_like(progress, float(preview.v))
    rho = (
        float(preview.rho_start)
        * (float(preview.rho_end) / float(preview.rho_start)) ** progress
    )
    uv = torch.stack((u, v), dim=-1)
    sigma = reference_rho_to_sigma(rho, beta, MNIST_DIM)
    actual_rms = generalized_gaussian_actual_rms(rho, beta, lam)
    schedule = {
        "progress": progress,
        "beta": beta,
        "lam": lam,
        "rho": rho,
        "u": u,
        "v": v,
        "uv": uv,
        "sigma": sigma,
        "actual_rms": actual_rms,
    }
    validate_mnist_schedule(schedule, noise_config)
    return schedule


def validate_mnist_schedule(
    schedule: dict[str, torch.Tensor],
    noise_config: Any,
) -> None:
    """Reject schedules outside training support or with increasing RMS noise."""
    normalize_full_spd_condition(
        schedule["beta"],
        schedule["lam"],
        schedule["rho"],
        schedule["uv"],
        noise_config,
    )
    spd_2d_from_uv(schedule["uv"])
    differences = schedule["actual_rms"][1:] - schedule["actual_rms"][:-1]
    if torch.any(differences > 1e-7):
        raise ValueError("actual RMS noise must decrease along the generation path")


def build_mnist_sampling_schedule(model_config: Any, path_config: Any, device: Any):
    """Construct a fixed or joint sampling path at matched actual RMS."""
    progress = torch.linspace(
        0.0, 1.0, int(path_config.sampler.n_levels), device=device
    )
    if str(path_config.path.name) == "joint":
        beta = 2.0 - 0.6 * progress
        lam = 1.0 + 0.8 * progress
        u = 0.3 * torch.sin(math.pi * progress)
        v = 0.6 * progress
    else:
        beta = torch.full_like(progress, float(path_config.path.beta))
        lam = torch.full_like(progress, float(path_config.path.lam))
        u = torch.full_like(progress, float(path_config.path.u))
        v = torch.full_like(progress, float(path_config.path.v))

    actual_rms = (
        float(path_config.schedule.actual_rms_start)
        * (
            float(path_config.schedule.actual_rms_end)
            / float(path_config.schedule.actual_rms_start)
        )
        ** progress
    )
    rho = actual_rms * lam.pow(1.0 / beta)
    uv = torch.stack((u, v), dim=-1)
    schedule = {
        "progress": progress,
        "beta": beta,
        "lam": lam,
        "rho": rho,
        "u": u,
        "v": v,
        "uv": uv,
        "sigma": reference_rho_to_sigma(rho, beta, MNIST_DIM),
        "actual_rms": generalized_gaussian_actual_rms(rho, beta, lam),
    }
    validate_mnist_schedule(schedule, model_config.noise)
    if not torch.allclose(schedule["actual_rms"], actual_rms, atol=1e-6, rtol=1e-6):
        raise RuntimeError("The constructed schedule does not preserve actual RMS")
    return schedule

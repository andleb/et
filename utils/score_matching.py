from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from .eight_gauss import eight_gaussians, eight_gaussians_noise_convolution_score
from .experiment_io import (
    capture_rng_state,
    restore_rng_state,
    save_torch_atomic,
    set_reproducible_seed,
    write_csv_atomic,
    write_json_atomic,
)
from .noise_geometry import (
    FullSPDNoiseParameterSampler,
    generalized_gaussian_conditional_score,
    generalized_gaussian_score_second_moment,
    sample_generalized_gaussian_noise,
    spd_2d_from_uv,
)
from .sampling import generalized_importance_noisy_score


def set_score_matching_seed(seed: int | None) -> None:
    """Set Python, NumPy, and Torch seeds for reproducible notebook runs."""
    if seed is None:
        return
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_i)


def _activation(name: str) -> nn.Module:
    """Construct a ReLU, GELU, tanh, or SiLU activation from its name."""
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"unknown activation: {name}")


def sigma_feature(
    sigma: torch.Tensor | float,
    *,
    batch_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a ``(batch_size, 1)`` log-sigma feature."""
    sig = torch.as_tensor(sigma, device=device, dtype=dtype)
    if sig.ndim == 0:
        sig = sig.expand(int(batch_size))
    elif sig.ndim == 2 and sig.shape[1] == 1:
        sig = sig.reshape(-1)
    elif sig.ndim != 1:
        raise ValueError(
            f"sigma must be scalar, (B,), or (B, 1), got {tuple(sig.shape)}"
        )
    if sig.shape[0] != int(batch_size):
        raise ValueError(
            f"sigma length {sig.shape[0]} does not match batch {batch_size}"
        )
    return sig.clamp_min(1e-12).log().reshape(-1, 1)


def score_features(y: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
    """Concatenate ``y`` and ``log(sigma)`` for a conditional score network."""
    if y.ndim != 2:
        raise ValueError(f"y must have shape (B, D), got {tuple(y.shape)}")
    log_sigma = sigma_feature(
        sigma,
        batch_size=y.shape[0],
        device=y.device,
        dtype=y.dtype,
    )
    return torch.cat([y, log_sigma], dim=1)


class ConditionalScoreMLP(nn.Module):
    """Estimate noisy marginal scores conditioned on the noise scale.

    Concatenates each observation with ``log(sigma)`` and predicts a score
    vector of dimension ``dim``. The noise scale may be shared across the
    batch or supplied per observation.
    """

    def __init__(
        self,
        *,
        dim: int = 2,
        width: int = 128,
        depth: int = 3,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if width <= 0:
            raise ValueError("width must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")

        self.dim = int(dim)
        self.width = int(width)
        self.depth = int(depth)
        self.activation = str(activation)

        layers: list[nn.Module] = []
        in_dim = self.dim + 1
        for _ in range(self.depth):
            layers.append(nn.Linear(in_dim, self.width))
            layers.append(_activation(self.activation))
            in_dim = self.width
        layers.append(nn.Linear(in_dim, self.dim))
        self.net = nn.Sequential(*layers)

    @property
    def config(self) -> dict[str, Any]:
        """Return the architecture settings needed to reconstruct the network."""
        return {
            "dim": self.dim,
            "width": self.width,
            "depth": self.depth,
            "activation": self.activation,
        }

    def forward(
        self,
        y: torch.Tensor,
        sigma: torch.Tensor | float,
    ) -> torch.Tensor:
        """Predict one score vector per observation using its log noise scale."""
        return self.net(score_features(y, sigma))


class ParameterConditionedScoreMLP(nn.Module):
    """Estimate noisy marginal scores conditioned on the entire noise parameter vector.

    Concatenates observations with ``condition_dim`` features encoded by the
    caller. A shared condition vector is broadcast across the batch; separate
    conditions may also be supplied per observation. Returns one score vector
    of dimension ``dim`` per observation.
    """

    def __init__(
        self,
        dim: int = 2,
        condition_dim: int = 5,
        width: int = 256,
        depth: int = 3,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if min(dim, condition_dim, width, depth) <= 0:
            raise ValueError("Model dimensions must be positive")
        self.config = dict(
            dim=dim,
            condition_dim=condition_dim,
            width=width,
            depth=depth,
            activation=activation,
        )
        layers = []
        in_dim = dim + condition_dim
        for _ in range(depth):
            layers.extend([nn.Linear(in_dim, width), _activation(activation)])
            in_dim = width
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Predict scores for the batch, broadcasting a shared condition when supplied."""
        condition = condition.to(device=y.device, dtype=y.dtype)
        if condition.ndim == 1:
            condition = condition.unsqueeze(0)
        condition = condition.expand(y.shape[0], -1)
        return self.net(torch.cat([y, condition], dim=-1))


def et_dsm_loss(
    model: nn.Module,
    y: torch.Tensor,
    x_clean: torch.Tensor,
    condition: torch.Tensor,
    *,
    beta: torch.Tensor | float,
    lam: torch.Tensor | float,
    Sigma: torch.Tensor,
    sigma: torch.Tensor | float,
) -> torch.Tensor:
    """Per-observation Energy-Tweedie denoising score-matching loss."""
    target = generalized_gaussian_conditional_score(
        y - x_clean,
        beta=beta,
        lam=lam,
        Sigma=Sigma,
        sigma=sigma,
    ).detach()
    return 0.5 * (model(y, condition) - target).square().sum(-1)


def exact_score_matching_loss(
    model: nn.Module,
    y: torch.Tensor,
    sigma: torch.Tensor | float,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Hyvarinen score-matching loss with exact divergence.

    In the current 2D problem this costs only two vector-Jacobian products, so
    it is preferable to a sliced estimator.
    """
    if y.ndim != 2:
        raise ValueError(f"y must have shape (B, D), got {tuple(y.shape)}")
    y_req = y.detach().requires_grad_(True)
    score = model(y_req, sigma)
    if score.shape != y_req.shape:
        raise ValueError(
            f"model must return shape {tuple(y_req.shape)}, got {tuple(score.shape)}"
        )

    div = torch.zeros(y_req.shape[0], device=y_req.device, dtype=y_req.dtype)
    for j in range(y_req.shape[1]):
        grad_j = torch.autograd.grad(
            score[:, j].sum(),
            y_req,
            create_graph=True,
            retain_graph=True,
        )[0][:, j]
        div = div + grad_j

    loss_vec = 0.5 * score.square().sum(dim=1) + div
    if reduction == "mean":
        return loss_vec.mean()
    if reduction == "sum":
        return loss_vec.sum()
    if reduction == "none":
        return loss_vec
    raise ValueError(f"unknown reduction: {reduction}")


@torch.no_grad()
def sample_noisy_eight_gaussians(
    n: int,
    *,
    sigmas: Iterable[float] | torch.Tensor,
    beta: float,
    lam: float,
    Sigma: torch.Tensor,
    radius: float = 2.0,
    std: float = 0.1,
    balanced: bool = True,
    shuffle: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw noisy 8-Gaussians samples for score matching.

    Returns ``(y, sigma_values, x_clean)``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if torch.is_tensor(sigmas):
        sig_t = sigmas.to(device=device, dtype=dtype).reshape(-1)
    else:
        sig_t = torch.as_tensor(list(sigmas), device=device, dtype=dtype).reshape(-1)
    if sig_t.numel() == 0:
        raise ValueError("sigmas must be non-empty")

    y_parts: list[torch.Tensor] = []
    sigma_parts: list[torch.Tensor] = []
    x_parts: list[torch.Tensor] = []

    if balanced:
        n_sig = int(sig_t.numel())
        base = int(n) // n_sig
        rem = int(n) - base * n_sig
        counts = [base + (1 if i < rem else 0) for i in range(n_sig)]
    else:
        idx = torch.randint(sig_t.numel(), (int(n),), device=sig_t.device)
        counts = [(idx == i).sum().item() for i in range(sig_t.numel())]

    for sig, count in zip(sig_t, counts):
        count_i = int(count)
        if count_i == 0:
            continue
        x_clean = eight_gaussians(
            count_i,
            radius=float(radius),
            std=float(std),
            device=device,
        ).to(dtype=dtype)
        eps = sample_generalized_gaussian_noise(
            count_i,
            beta=float(beta),
            lam=float(lam),
            Sigma=Sigma,
            sigma=float(sig),
            device=device,
            dtype=dtype,
        )
        y_parts.append(x_clean + eps)
        sigma_parts.append(
            torch.full((count_i,), float(sig), device=device, dtype=dtype)
        )
        x_parts.append(x_clean)

    y = torch.cat(y_parts, dim=0)
    sigma_values = torch.cat(sigma_parts, dim=0)
    x_clean = torch.cat(x_parts, dim=0)
    if shuffle:
        perm = torch.randperm(y.shape[0], device=y.device)
        y = y[perm]
        sigma_values = sigma_values[perm]
        x_clean = x_clean[perm]
    return y, sigma_values, x_clean


def train_conditional_score_model(
    model: ConditionalScoreMLP,
    *,
    sigmas: Iterable[float] | torch.Tensor,
    beta: float,
    lam: float,
    Sigma: torch.Tensor,
    n_steps: int = 20_000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip: float | None = 10.0,
    radius: float = 2.0,
    std: float = 0.1,
    log_every: int = 100,
    progress: bool = True,
    progress_desc: str = "score matching",
    seed: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> list[dict[str, float]]:
    """Train a conditional score network by exact score matching."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    set_score_matching_seed(seed)

    first_param = next(model.parameters())
    target_device = torch.device(device) if device is not None else first_param.device
    target_dtype = dtype if dtype is not None else first_param.dtype
    model.to(device=target_device, dtype=target_dtype)
    Sigma = Sigma.to(device=target_device, dtype=target_dtype)

    opt = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    history: list[dict[str, float]] = []
    model.train()
    steps = range(1, int(n_steps) + 1)
    progress_bar = None
    if progress:
        progress_bar = tqdm(steps, total=int(n_steps), desc=str(progress_desc))
        steps = progress_bar

    for step in steps:
        y_b, sig_b, _ = sample_noisy_eight_gaussians(
            int(batch_size),
            sigmas=sigmas,
            beta=beta,
            lam=lam,
            Sigma=Sigma,
            radius=radius,
            std=std,
            device=target_device,
            dtype=target_dtype,
        )
        loss = exact_score_matching_loss(model, y_b, sig_b)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = (
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            if grad_clip is not None
            else None
        )
        opt.step()

        if log_every > 0 and (
            step == 1 or step % int(log_every) == 0 or step == int(n_steps)
        ):
            row = {"step": float(step), "loss": float(loss.detach().cpu())}
            if grad_norm is not None:
                row["grad_norm"] = float(torch.as_tensor(grad_norm).detach().cpu())
            history.append(row)
            if progress_bar is not None:
                progress_bar.set_postfix(
                    loss=f"{row['loss']:.4g}",
                    grad_norm=f"{row['grad_norm']:.4g}" if "grad_norm" in row else "",
                )
    return history


@torch.no_grad()
def predict_conditional_score(
    model: nn.Module,
    y: torch.Tensor,
    sigma: torch.Tensor | float,
    *,
    batch_size: int = 8192,
) -> torch.Tensor:
    """Evaluate a conditional score model in batches."""
    if y.ndim != 2:
        raise ValueError(f"y must have shape (B, D), got {tuple(y.shape)}")
    was_training = model.training
    model.eval()
    outs = []
    for start in range(0, y.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), y.shape[0])
        sig_b = sigma
        if torch.is_tensor(sigma) and sigma.ndim > 0:
            sig_b = sigma[start:stop]
        outs.append(model(y[start:stop], sig_b))
    if was_training:
        model.train()
    return torch.cat(outs, dim=0)


@torch.no_grad()
def score_diagnostics_by_sigma(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    sigma_values: torch.Tensor,
    *,
    sigmas: Iterable[float] | torch.Tensor | None = None,
    eps: float = 1e-12,
) -> list[dict[str, float]]:
    """Return MSE, cosine, norm-ratio, and projection slopes by sigma."""
    if estimate.shape != reference.shape:
        raise ValueError(
            f"estimate and reference must match, got {tuple(estimate.shape)} and {tuple(reference.shape)}"
        )
    if estimate.ndim != 2:
        raise ValueError(f"scores must have shape (B, D), got {tuple(estimate.shape)}")
    sigma_values = sigma_values.to(
        device=estimate.device, dtype=estimate.dtype
    ).reshape(-1)
    if sigma_values.shape[0] != estimate.shape[0]:
        raise ValueError("sigma_values length must match score batch")
    sig_t = (
        torch.unique(sigma_values).sort().values
        if sigmas is None
        else (
            sigmas.to(device=estimate.device, dtype=estimate.dtype).reshape(-1)
            if torch.is_tensor(sigmas)
            else torch.as_tensor(
                list(sigmas), device=estimate.device, dtype=estimate.dtype
            ).reshape(-1)
        )
    )

    rows = []
    for sig in sig_t:
        mask = torch.isclose(sigma_values, sig)
        if not bool(mask.any()):
            continue
        est = estimate[mask]
        ref = reference[mask]
        dot = (est * ref).sum()
        ref_sq = ref.square().sum().clamp_min(float(eps))
        est_sq = est.square().sum().clamp_min(float(eps))
        rows.append(
            {
                "sigma": float(sig.detach().cpu()),
                "n": float(est.shape[0]),
                "mse": float((est - ref).square().sum(dim=1).mean().detach().cpu()),
                "cosine": float(
                    F.cosine_similarity(est, ref, dim=1, eps=float(eps))
                    .mean()
                    .detach()
                    .cpu()
                ),
                "norm_ratio": float(
                    (
                        est.norm(dim=1).mean()
                        / ref.norm(dim=1).mean().clamp_min(float(eps))
                    )
                    .detach()
                    .cpu()
                ),
                "estimate_to_reference_slope": float((dot / ref_sq).detach().cpu()),
                "reference_from_estimate_slope": float((dot / est_sq).detach().cpu()),
            }
        )
    return rows


def save_score_matching_checkpoint(
    path: str | Path,
    model: ConditionalScoreMLP,
    *,
    history: list[dict[str, float]] | None = None,
    config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a conditional score model with enough metadata to reload it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_config": model.config,
            "state_dict": model.state_dict(),
            "history": history or [],
            "config": config or {},
            "metadata": metadata or {},
        },
        temporary_path,
    )
    temporary_path.replace(path)


def load_score_matching_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> tuple[ConditionalScoreMLP, dict[str, Any]]:
    """Load a checkpoint saved by :func:`save_score_matching_checkpoint`."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model = ConditionalScoreMLP(**payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    return model, payload


def full_spd_score_training_loss(
    model: nn.Module,
    sampler: FullSPDNoiseParameterSampler,
    objective: str,
    *,
    batch_size: int,
    radius: float,
    std_clean: float,
) -> torch.Tensor:
    """Draw an Eight-Gaussians batch and compute an RMS-weighted score loss."""
    clean = eight_gaussians(
        batch_size,
        radius=radius,
        std=std_clean,
        device=sampler.device,
    ).to(dtype=sampler.dtype)
    parameters = sampler.sample(batch_size)
    noisy = sampler.corrupt(clean, parameters)
    if objective == "et_dsm":
        loss = et_dsm_loss(
            model,
            noisy,
            clean,
            parameters.condition,
            beta=parameters.beta,
            lam=parameters.lam,
            Sigma=spd_2d_from_uv(parameters.uv),
            sigma=parameters.sigma,
        )
    elif objective == "hyvarinen":
        loss = exact_score_matching_loss(
            model,
            noisy,
            parameters.condition,
            reduction="none",
        )
    else:
        raise ValueError(f"Unknown score objective: {objective}")
    return parameters.actual_rms.square() * loss.mean()


@torch.no_grad()
def full_spd_validation_loss_floors(
    config: dict[str, Any],
    noise_config: Any,
    *,
    seed: int,
    prior_bank: torch.Tensor,
    result_dir: str | Path,
    query_batch_size: int = 32,
    reference_chunk_size: int = 4096,
) -> pd.DataFrame:
    """Cache population-loss estimates for the fixed validation conditions.

    Replays the validation sampler, without loading or evaluating learned models.
    The Hyvarinen floor is minus half the marginal score energy; the DSM floor
    adds half the analytically known noise-score energy.
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = result_dir / "floors.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)

    device = torch.device(config["system"]["device"])
    training, data = config["training"], config["data"]
    rng_state = capture_rng_state()
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        inputs_path = result_dir / "validation_inputs.pt"
        if inputs_path.exists():
            inputs = torch.load(inputs_path, map_location="cpu", weights_only=False)
        else:
            sampler = FullSPDNoiseParameterSampler(
                noise_config, dim=2, device=device, dtype=torch.float32
            )
            set_reproducible_seed(int(training["validation_seed_offset"]) + seed)
            inputs = []
            for _ in range(int(training["validation_batches"])):
                clean = eight_gaussians(
                    int(training["validation_batch_size"]),
                    radius=float(data["radius"]),
                    std=float(data["std_clean"]),
                    device=device,
                ).to(dtype=sampler.dtype)
                parameters = sampler.sample(len(clean))
                noisy = sampler.corrupt(clean, parameters)
                inputs.append(
                    {
                        "y": noisy.cpu(),
                        "clean": clean.cpu(),
                        "beta": float(parameters.beta),
                        "lam": float(parameters.lam),
                        "sigma": float(parameters.sigma),
                        "actual_rms": float(parameters.actual_rms),
                        "shape": spd_2d_from_uv(parameters.uv).cpu(),
                    }
                )
            save_torch_atomic(inputs, inputs_path)
        write_json_atomic(
            {
                "config": config,
                "seed": seed,
                "reference_particles": len(prior_bank),
                "query_batch_size": query_batch_size,
                "reference_chunk_size": reference_chunk_size,
                "torch_version": str(torch.__version__),
            },
            result_dir / "config.json",
        )

        prior_bank = prior_bank.to(device=device, dtype=torch.float32)
        batch_rows = []
        for index, batch in enumerate(inputs):
            reference_path = result_dir / f"reference_{index:02d}.pt"
            if reference_path.exists():
                reference = torch.load(
                    reference_path, map_location="cpu", weights_only=False
                )
            else:
                y = batch["y"].to(device)
                shape = batch["shape"].to(device)
                noise_kwargs = dict(
                    beta=batch["beta"],
                    lam=batch["lam"],
                    Sigma=shape,
                    sigma=batch["sigma"],
                )
                use_noise = batch["actual_rms"] < float(data["std_clean"])
                if use_noise:
                    set_reproducible_seed(3_000_000 + 100 * seed + index)
                    particles = sample_generalized_gaussian_noise(
                        len(prior_bank), **noise_kwargs, device=device, dtype=y.dtype
                    )
                else:
                    particles = prior_bank
                halves = []
                for bank in particles.chunk(2):
                    chunks = []
                    for queries in y.split(query_batch_size):
                        if use_noise:
                            score = eight_gaussians_noise_convolution_score(
                                queries,
                                bank,
                                radius=float(data["radius"]),
                                std_clean=float(data["std_clean"]),
                                chunk_size=reference_chunk_size,
                            )
                        else:
                            score = generalized_importance_noisy_score(
                                queries,
                                bank,
                                **noise_kwargs,
                                chunk_size=reference_chunk_size,
                            )
                        chunks.append(score.cpu())
                    halves.append(torch.cat(chunks))
                reference = {
                    "half_1": halves[0],
                    "half_2": halves[1],
                    "proposal": "noise" if use_noise else "clean_prior",
                }
                save_torch_atomic(reference, reference_path)

            first, second = reference["half_1"].double(), reference["half_2"].double()
            weight = batch["actual_rms"] ** 2
            h_floor_queries = -0.5 * weight * (first * second).sum(dim=1)
            noise_constant = (
                0.5
                * weight
                * generalized_gaussian_score_second_moment(
                    beta=batch["beta"],
                    lam=batch["lam"],
                    Sigma=batch["shape"],
                    sigma=batch["sigma"],
                )
            )
            score_power = ((first + second) / 2).square().sum(dim=1).mean()
            batch_rows.append(
                {
                    "batch": index,
                    "beta": batch["beta"],
                    "lambda": batch["lam"],
                    "sigma": batch["sigma"],
                    "actual_rms": batch["actual_rms"],
                    "u": float(batch["shape"][0, 0] - 1),
                    "v": float(batch["shape"][0, 1]),
                    "hyvarinen_floor": float(h_floor_queries.mean()),
                    "et_dsm_floor": float(h_floor_queries.mean()) + noise_constant,
                    "query_variance_of_mean": float(h_floor_queries.var() / len(first)),
                    "half_1_floor": float(
                        -0.5 * weight * first.square().sum(dim=1).mean()
                    ),
                    "half_2_floor": float(
                        -0.5 * weight * second.square().sum(dim=1).mean()
                    ),
                    "reference_split_nmse": float(
                        (first - second).square().sum(dim=1).mean()
                        / score_power.clamp_min(1e-12)
                    ),
                    "proposal": reference["proposal"],
                }
            )
            print(
                f"Validation floors, seed {seed}: {index + 1}/{len(inputs)}", flush=True
            )

        batches = pd.DataFrame(batch_rows)
        write_csv_atomic(batches, result_dir / "per_condition.csv")
        query_se = float(
            np.sqrt(batches["query_variance_of_mean"].sum()) / len(batches)
        )
        half_difference = abs(
            float(batches["half_1_floor"].mean() - batches["half_2_floor"].mean())
        )
        summary = pd.DataFrame(
            [
                {
                    "method": method,
                    "model_seed": seed,
                    "floor": float(batches[f"{method}_floor"].mean()),
                    "query_se": query_se,
                    "half_bank_floor_difference": half_difference,
                    "max_reference_split_nmse": float(
                        batches["reference_split_nmse"].max()
                    ),
                    "n_conditions": len(batches),
                    "n_queries": sum(len(b["y"]) for b in inputs),
                }
                for method in ("et_dsm", "hyvarinen")
            ]
        )
        write_csv_atomic(summary, summary_path)
        return summary
    finally:
        torch.set_default_dtype(original_dtype)
        restore_rng_state(rng_state)


def train_full_spd_score_model(
    config: dict[str, Any],
    noise_config: Any,
    *,
    objective: str,
    seed: int,
    result_dir: str | Path,
) -> Path:
    """Train/resume a conditional score network with cosine LR decay.

    ``config`` contains resolved data, model, training, system and noise
    settings. Noise parameters are supplied as known conditions; the model learns
    the noisy score. Runs with different objectives use the same initial weights
    and training batches for a given seed, with isolated validation RNG.
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = result_dir / "model.pt"
    identity = {"config": config, "objective": objective, "seed": int(seed)}
    payload = None
    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload["identity"] != identity:
            raise ValueError(
                f"Configuration differs from existing run: {checkpoint_path}"
            )
        if int(payload["step"]) == int(config["training"]["n_steps"]):
            write_csv_atomic(
                pd.DataFrame(payload["history"]), result_dir / "training_history.csv"
            )
            print(f"Already complete: {objective}, seed {seed}", flush=True)
            return checkpoint_path

    write_json_atomic(config, result_dir / "config.json")
    training = config["training"]
    n_steps = int(training["n_steps"])
    device = torch.device(config["system"]["device"])
    set_reproducible_seed(seed)
    model = ParameterConditionedScoreMLP(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["lr"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_steps,
        eta_min=float(training["lr_min"]),
    )
    sampler = FullSPDNoiseParameterSampler(
        noise_config,
        dim=2,
        device=device,
        dtype=torch.float32,
    )
    set_reproducible_seed(int(training["data_seed_offset"]) + seed)
    start_step, elapsed, previous_peak = 0, 0.0, 0
    history = []
    if payload is not None:
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        restore_rng_state(payload["rng_state"])
        start_step = int(payload["step"])
        history = payload["history"]
        elapsed = float(payload["elapsed_seconds"])
        previous_peak = int(payload["peak_gpu_bytes"])
    del payload
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    batch_kwargs = dict(
        batch_size=int(training["batch_size"]),
        radius=float(config["data"]["radius"]),
        std_clean=float(config["data"]["std_clean"]),
    )
    model.train()
    for step in range(start_step + 1, n_steps + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad(set_to_none=True)
        loss = full_spd_score_training_loss(model, sampler, objective, **batch_kwargs)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"Nonfinite loss: {objective}, seed {seed}, step {step}"
            )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(training["grad_clip"]),
            error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler.step()
        validate = step % int(training["validation_every"]) == 0 or step == n_steps
        log = step == 1 or step % int(training["log_every"]) == 0 or validate
        validation_loss = None
        if validate:
            train_rng = capture_rng_state()
            try:
                set_reproducible_seed(int(training["validation_seed_offset"]) + seed)
                model.eval()
                values = []
                for _ in range(int(training["validation_batches"])):
                    value = full_spd_score_training_loss(
                        model,
                        sampler,
                        objective,
                        **{
                            **batch_kwargs,
                            "batch_size": int(training["validation_batch_size"]),
                        },
                    )
                    values.append(float(value.detach()))
                validation_loss = float(np.mean(values))
            finally:
                model.train()
                restore_rng_state(train_rng)
        if log:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            peak = max(
                previous_peak,
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            )
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "learning_rate": learning_rate,
                    "grad_norm": float(grad_norm),
                    "validation_loss": validation_loss,
                    "elapsed_seconds": elapsed + time.perf_counter() - started,
                    "peak_gpu_bytes": peak,
                }
            )
            print(
                f"{objective} seed={seed} step={step}/{n_steps} "
                f"loss={history[-1]['loss']:.6g} validation={validation_loss} "
                f"lr={learning_rate:.6g} "
                f"elapsed={history[-1]['elapsed_seconds']:.0f}s",
                flush=True,
            )
        if step % int(training["checkpoint_every"]) == 0 or step == n_steps:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            save_torch_atomic(
                {
                    "identity": identity,
                    "model_config": model.config,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "rng_state": capture_rng_state(),
                    "step": step,
                    "history": history,
                    "elapsed_seconds": elapsed + time.perf_counter() - started,
                    "peak_gpu_bytes": max(
                        previous_peak,
                        (
                            torch.cuda.max_memory_allocated(device)
                            if device.type == "cuda"
                            else 0
                        ),
                    ),
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "torch_version": str(torch.__version__),
                    "device_name": (
                        torch.cuda.get_device_name(device)
                        if device.type == "cuda"
                        else "cpu"
                    ),
                },
                checkpoint_path,
            )
            write_csv_atomic(pd.DataFrame(history), result_dir / "training_history.csv")
    return checkpoint_path

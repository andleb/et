import math
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from .experiment_io import load_resolved_config
from .noise_geometry import (
    RepeatedBlockSPD,
    log_sigma_spd_condition,
    normalize_full_spd_condition,
    reference_rho_to_sigma,
    sample_repeated_block_generalized_gaussian,
)

SampleLayout = Literal["auto", "kbd", "bkd", "bdk"]


def coerce_posterior_samples(
    x_samples: torch.Tensor,
    y: torch.Tensor,
    *,
    layout: SampleLayout = "auto",
) -> torch.Tensor:
    """Return posterior samples with shape ``(K, B, D)``.

    Supported layouts:
    - ``kbd``: already ``(K, B, D)``.
    - ``bkd``: ``(B, K, D)``.
    - ``bdk``: engression-style ``(B, D, K)``.
    - ``auto``: infer from dimensions.
    """
    if x_samples.ndim != 3:
        raise ValueError(
            f"posterior samples must be rank 3, got shape {tuple(x_samples.shape)}"
        )
    if y.ndim != 2:
        raise ValueError(f"y must have shape (B, D), got {tuple(y.shape)}")

    bsz, dim = y.shape
    if layout == "kbd":
        out = x_samples
    elif layout == "bkd":
        out = x_samples.permute(1, 0, 2)
    elif layout == "bdk":
        out = x_samples.permute(2, 0, 1)
    elif layout == "auto":
        if x_samples.shape[1] == bsz and x_samples.shape[2] == dim:
            out = x_samples
        elif x_samples.shape[0] == bsz and x_samples.shape[2] == dim:
            out = x_samples.permute(1, 0, 2)
        elif x_samples.shape[0] == bsz and x_samples.shape[1] == dim:
            out = x_samples.permute(2, 0, 1)
        else:
            raise ValueError(
                "could not infer posterior-sample layout from "
                f"samples={tuple(x_samples.shape)} and y={tuple(y.shape)}"
            )
    else:
        raise ValueError(f"unknown sample layout: {layout}")

    if out.shape[1:] != y.shape:
        raise ValueError(
            f"posterior samples must coerce to (K, B, D), got {tuple(out.shape)} "
            f"for y={tuple(y.shape)}"
        )
    return out.contiguous()


class EngressorSampler:
    """Expose a noise-level-conditioned Engressor for posterior score estimation.
    For observations ``y`` of shape ``(B, D)``, it returns Engression's draws arranged
    as ``(n_samples, B, D)``.
    """

    def __init__(
        self,
        engressor: Any,
        device,
        log_sigma: bool = True,
        compile_model: bool = True,
    ):
        self.engressor = engressor
        self.device = torch.device(device)
        self.log_sigma = log_sigma
        model = self.engressor.model.to(self.device)
        if compile_model:
            model = torch.compile(model)
        self.engressor.model = model
        self.engressor.eval_mode()

    @torch.inference_mode()
    def sample(self, y: torch.Tensor, sigma: float, n_samples: int) -> torch.Tensor:
        """Return ``(n_samples, B, D)`` posterior draws at the given noise scale."""
        y = y.to(self.device)
        sig = math.log(sigma) if self.log_sigma else sigma
        cond = torch.full((y.shape[0], 1), sig, device=self.device, dtype=y.dtype)

        return (
            self.engressor.model.sample_onebatch(
                y, condition=cond, sample_size=n_samples, expand_dim=True
            )
            .permute(2, 0, 1)
            .contiguous()
        )


class SPDConditionedEngressorSampler:
    """Expose an Engressor conditioned jointly on noise scale and SPD shape.
    Builds ``(log(sigma), u, v)`` conditions, where ``(u, v)`` parameterizes a
    2D SPD matrix of trace 2. For observations ``y`` of shape ``(B, D)``, it
    returns Engression's draws arranged as ``(n_samples, B, D)``.
    """

    def __init__(
        self,
        engressor: Any,
        device,
        compile_model: bool = True,
    ):
        self.engressor = engressor
        self.device = torch.device(device)
        model = self.engressor.model.to(self.device)
        if compile_model:
            model = torch.compile(model)
        self.engressor.model = model
        self.engressor.eval_mode()

    @torch.inference_mode()
    def sample(
        self,
        y: torch.Tensor,
        sigma: float,
        uv: torch.Tensor | list[float] | tuple[float, float],
        n_samples: int,
    ) -> torch.Tensor:
        """Draw from the posterior conditioned on ``(log(sigma), u, v)``.

        Return ``(n_samples, B, D)`` draws for observations ``y`` of shape
        ``(B, D)``.
        """
        y = y.to(self.device)
        uv_tensor = torch.as_tensor(uv, device=self.device, dtype=y.dtype)
        condition = log_sigma_spd_condition(
            math.log(float(sigma)), uv_tensor, y.shape[0]
        )
        return (
            self.engressor.model.sample_onebatch(
                y,
                condition=condition,
                sample_size=int(n_samples),
                expand_dim=True,
            )
            .permute(2, 0, 1)
            .contiguous()
        )


class ParameterConditionedEngressorSampler:
    """Expose an Engressor conditioned on generalized-Gaussian noise parameters.
    A supplied ``condition_encoder`` encodes the exponent ``beta``, radial
    coefficient ``lam``, reference RMS ``rho``, and SPD shape coordinates
    ``(u, v)``. For observations ``y`` of shape ``(B, D)``, it returns
    Engression's draws arranged as ``(n_samples, B, D)``.
    """

    def __init__(
        self,
        engressor: Any,
        device,
        condition_encoder: Callable[..., torch.Tensor],
        compile_model: bool = False,
    ):
        self.engressor = engressor
        self.device = torch.device(device)
        self.condition_encoder = condition_encoder
        model = self.engressor.model.to(self.device)
        if compile_model:
            model = torch.compile(model)
        self.engressor.model = model
        self.engressor.eval_mode()

    @torch.inference_mode()
    def sample(
        self,
        y: torch.Tensor,
        beta: float,
        lam: float,
        rho: float,
        uv: torch.Tensor | list[float] | tuple[float, float],
        n_samples: int,
    ) -> torch.Tensor:
        """Encode the supplied noise parameters and sample the conditional posterior.

        Return ``(n_samples, B, D)`` draws for observations ``y`` of shape
        ``(B, D)``.
        """
        y = y.to(self.device)
        uv_tensor = torch.as_tensor(uv, device=self.device, dtype=y.dtype)
        condition = self.condition_encoder(
            beta,
            lam,
            rho,
            uv_tensor,
            batch_size=y.shape[0],
        )
        return (
            self.engressor.model.sample_onebatch(
                y,
                condition=condition,
                sample_size=int(n_samples),
                expand_dim=True,
            )
            .permute(2, 0, 1)
            .contiguous()
        )


def load_parameter_conditioned_engressor(
    model_dir: str | Path,
    *,
    checkpoint: str = "model.pt",
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load an Engressor checkpoint with five-parameter noise conditioning."""
    model_dir = Path(model_dir)
    config = load_resolved_config(model_dir / "config.yaml")
    checkpoint_path = model_dir / checkpoint
    engressor = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sampler = ParameterConditionedEngressorSampler(
        engressor,
        device,
        condition_encoder=partial(
            normalize_full_spd_condition, noise_config=config.noise
        ),
        compile_model=False,
    )
    return {
        "config": config,
        "checkpoint_path": checkpoint_path,
        "engressor": engressor,
        "sampler": sampler,
    }


@torch.inference_mode()
def sample_grouped_engressor_posterior(
    sampler: Any,
    y: torch.Tensor,
    sigma_values: torch.Tensor,
    n_samples: int,
    *,
    batch_size: int | None = None,
    sample_layout: SampleLayout = "auto",
    condition_args: tuple[Any, ...] = (),
) -> torch.Tensor:
    """Sample a sigma-conditioned Engressor for grouped multiscale queries.

    Args:
        sampler: Object exposing ``sample(y_batch, sigma, n_samples)``.
        y: Query tensor with shape ``(B, D)``.
        sigma_values: One scalar noise level per query.
        n_samples: Number of posterior samples per query.
        batch_size: Optional maximum number of same-sigma queries per call.
        sample_layout: Layout returned by ``sampler.sample``.
        condition_args: Fixed positional condition arguments inserted between
            ``sigma`` and ``n_samples`` when calling the sampler.

    Returns:
        Posterior samples with shape ``(B, n_samples, D)``.
    """
    if y.ndim != 2:
        raise ValueError(f"y must have shape (B, D), got {tuple(y.shape)}")
    if int(n_samples) <= 0:
        raise ValueError("n_samples must be positive")
    if batch_size is not None and int(batch_size) <= 0:
        raise ValueError("batch_size must be positive when provided")

    sigma_values = sigma_values.to(device=y.device, dtype=y.dtype).reshape(-1)
    if sigma_values.shape[0] != y.shape[0]:
        raise ValueError("sigma_values must have one entry per row of y")

    effective_batch = y.shape[0] if batch_size is None else int(batch_size)
    condition_args = tuple(condition_args)
    posterior = torch.empty(
        (y.shape[0], int(n_samples), y.shape[1]),
        device=y.device,
        dtype=y.dtype,
    )
    for sigma in torch.unique(sigma_values).sort().values:
        indices = torch.nonzero(
            torch.isclose(sigma_values, sigma),
            as_tuple=False,
        ).reshape(-1)
        for index_chunk in torch.split(indices, effective_batch):
            raw = sampler.sample(
                y[index_chunk],
                float(sigma.detach().cpu()),
                *condition_args,
                n_samples=int(n_samples),
            )
            samples_kbd = coerce_posterior_samples(
                raw,
                y[index_chunk],
                layout=sample_layout,
            )
            posterior[index_chunk] = samples_kbd.permute(1, 0, 2)
    return posterior


def generalized_gaussian_log_weights(
    y: torch.Tensor,
    x_prior: torch.Tensor,
    *,
    beta: float,
    lam: float,
    Sigma: torch.Tensor,
    sigma: float = 1.0,
    chunk_size: int = 32768,
) -> torch.Tensor:
    """Unnormalized log likelihood weights ``log q(y - x_prior)``.

    Returns a dense ``(B, M)`` tensor.  Use this when the estimation batch is
    modest enough to materialize the full weight matrix.
    """
    beta_f = float(beta)
    lam_f = float(lam)
    sigma_f = float(sigma)
    Sigma = Sigma.to(device=y.device, dtype=y.dtype)
    inv_Sigma = torch.linalg.inv(Sigma)

    chunks = []
    for start in range(0, x_prior.shape[0], int(chunk_size)):
        x_ch = x_prior[start : start + int(chunk_size)]
        diff = y[:, None, :] - x_ch[None, :, :]
        maha = torch.einsum("bmd,de,bme->bm", diff, inv_Sigma, diff).clamp_min(0.0)
        r = maha.sqrt() / sigma_f
        chunks.append(-(lam_f / beta_f) * r.pow(beta_f))
    return torch.cat(chunks, dim=1)


@torch.no_grad()
def generalized_importance_noisy_score(
    y: torch.Tensor,
    x_prior: torch.Tensor,
    *,
    beta: float,
    lam: float,
    Sigma: torch.Tensor,
    sigma: float = 1.0,
    chunk_size: int = 32768,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Self-normalized importance estimate of the noisy marginal score."""
    beta_f = float(beta)
    lam_f = float(lam)
    sigma_f = float(sigma)
    Sigma = Sigma.to(device=y.device, dtype=y.dtype)
    inv_Sigma = torch.linalg.inv(Sigma)
    inv_cov = inv_Sigma / (sigma_f**2)

    max_logw = None
    for start in range(0, x_prior.shape[0], int(chunk_size)):
        x_ch = x_prior[start : start + int(chunk_size)]
        diff = y[:, None, :] - x_ch[None, :, :]
        maha = torch.einsum("bmd,de,bme->bm", diff, inv_Sigma, diff).clamp_min(0.0)
        r = maha.sqrt() / sigma_f
        logw = -(lam_f / beta_f) * r.pow(beta_f)
        cur = logw.max(dim=1).values
        max_logw = cur if max_logw is None else torch.maximum(max_logw, cur)

    denom = torch.zeros(y.shape[0], device=y.device, dtype=y.dtype)
    score_num = torch.zeros_like(y)
    for start in range(0, x_prior.shape[0], int(chunk_size)):
        x_ch = x_prior[start : start + int(chunk_size)]
        diff = y[:, None, :] - x_ch[None, :, :]
        maha = torch.einsum("bmd,de,bme->bm", diff, inv_Sigma, diff).clamp_min(0.0)
        r = (maha.sqrt() / sigma_f).clamp_min(float(eps))
        logw = -(lam_f / beta_f) * r.pow(beta_f)
        w = torch.exp(logw - max_logw[:, None])
        inv_diff = torch.einsum("de,bme->bmd", inv_cov, diff)
        likelihood_score = -lam_f * r.pow(beta_f - 2.0).unsqueeze(-1) * inv_diff
        denom += w.sum(dim=1)
        score_num += (w.unsqueeze(-1) * likelihood_score).sum(dim=1)

    return score_num / denom[:, None].clamp_min(1e-30)


@torch.no_grad()
def generalized_importance_posterior_samples(
    y: torch.Tensor,
    x_prior: torch.Tensor,
    *,
    beta: float,
    lam: float,
    Sigma: torch.Tensor,
    sigma: float = 1.0,
    n_samples: int = 128,
    chunk_size: int = 32768,
) -> torch.Tensor:
    """Self-normalized importance-resampling approximation to ``X | Y=y``.

    Returns samples with shape ``(B, n_samples, D)``.
    """
    logw = generalized_gaussian_log_weights(
        y,
        x_prior,
        beta=beta,
        lam=lam,
        Sigma=Sigma,
        sigma=sigma,
        chunk_size=chunk_size,
    )
    probs = torch.softmax(logw, dim=1)
    idx = torch.multinomial(probs, num_samples=int(n_samples), replacement=True)
    return x_prior[idx]


@torch.no_grad()
def generalized_importance_references(
    y: torch.Tensor,
    sigma_values: torch.Tensor,
    x_prior: torch.Tensor,
    *,
    posterior_prior: torch.Tensor | None = None,
    beta: float,
    lam: float,
    Sigma: torch.Tensor,
    n_samples: int = 128,
    chunk_size: int = 32768,
    query_batch_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build multiscale importance-score and posterior references.

    The score uses ``x_prior``. By default the posterior uses the same bank;
    pass ``posterior_prior`` to use an independent bank. Randomness in posterior
    resampling is controlled by the caller's Torch random state. Returned
    posterior samples have shape ``(B, n_samples, D)``.
    """
    if y.ndim != 2:
        raise ValueError(f"y must have shape (B, D), got {tuple(y.shape)}")
    if int(n_samples) <= 0:
        raise ValueError("n_samples must be positive")
    if int(chunk_size) <= 0 or int(query_batch_size) <= 0:
        raise ValueError("chunk_size and query_batch_size must be positive")

    sigma_values = sigma_values.to(device=y.device, dtype=y.dtype).reshape(-1)
    if sigma_values.shape[0] != y.shape[0]:
        raise ValueError("sigma_values must have one entry per row of y")
    x_prior = x_prior.to(device=y.device, dtype=y.dtype)
    posterior_prior = (
        x_prior
        if posterior_prior is None
        else posterior_prior.to(device=y.device, dtype=y.dtype)
    )
    Sigma = Sigma.to(device=y.device, dtype=y.dtype)

    score = torch.empty_like(y)
    posterior = torch.empty(
        (y.shape[0], int(n_samples), y.shape[1]),
        device=y.device,
        dtype=y.dtype,
    )
    for sigma in torch.unique(sigma_values).sort().values:
        indices = torch.nonzero(
            torch.isclose(sigma_values, sigma),
            as_tuple=False,
        ).reshape(-1)
        for index_chunk in torch.split(indices, int(query_batch_size)):
            y_chunk = y[index_chunk]
            sigma_value = float(sigma.detach().cpu())
            score[index_chunk] = generalized_importance_noisy_score(
                y_chunk,
                x_prior,
                beta=beta,
                lam=lam,
                Sigma=Sigma,
                sigma=sigma_value,
                chunk_size=chunk_size,
            )
            posterior[index_chunk] = generalized_importance_posterior_samples(
                y_chunk,
                posterior_prior,
                beta=beta,
                lam=lam,
                Sigma=Sigma,
                sigma=sigma_value,
                n_samples=n_samples,
                chunk_size=chunk_size,
            )
    return score, posterior


def energy_tweedie_field_from_samples(
    y: torch.Tensor,
    x_samples: torch.Tensor,
    *,
    beta: torch.Tensor | float,
    Sigma: torch.Tensor,
    eps: float = 1e-12,
    sample_layout: SampleLayout = "auto",
) -> torch.Tensor:
    """Monte Carlo field ``G_{beta,Sigma}(y)`` from posterior samples."""
    x_kbd = coerce_posterior_samples(x_samples, y, layout=sample_layout)
    diff = x_kbd - y.unsqueeze(0)
    k, bsz, dim = diff.shape
    flat = diff.reshape(k * bsz, dim)

    solved = torch.linalg.solve(Sigma, flat.T).T.reshape(k, bsz, dim)
    sq_norm = (diff * solved).sum(dim=-1)
    norm = sq_norm.clamp_min(float(eps) ** 2).sqrt()
    weights = norm.pow(torch.as_tensor(beta, device=y.device, dtype=y.dtype) - 2.0)
    mean_term = (weights.unsqueeze(-1) * diff).mean(dim=0)

    return torch.linalg.solve(Sigma, mean_term.T).T


def scaled_energy_tweedie_field_from_samples(
    y: torch.Tensor,
    x_samples: torch.Tensor,
    *,
    beta: torch.Tensor | float,
    Sigma: torch.Tensor,
    sigma_values: torch.Tensor | float,
    eps: float = 1e-12,
    sample_layout: SampleLayout = "bkd",
) -> torch.Tensor:
    """Energy-Tweedie field for ``sigma^2 Sigma`` using a shared shape matrix."""
    beta_t = torch.as_tensor(beta, device=y.device, dtype=y.dtype)
    field_shape = energy_tweedie_field_from_samples(
        y,
        x_samples,
        beta=beta_t,
        Sigma=Sigma.to(device=y.device, dtype=y.dtype),
        eps=eps,
        sample_layout=sample_layout,
    )
    scale = torch.as_tensor(sigma_values, device=y.device, dtype=y.dtype).reshape(-1, 1)
    return scale.pow(-beta_t) * field_shape


@torch.no_grad()
def mc_score_from_posterior_samples(
    y: torch.Tensor,
    x_samples: torch.Tensor,
    sigma: float,
    beta: float,
    lam: float,
    eps: float = 1e-12,
):
    """
    Returns score estimate s_hat(y) of the noisy marginal at scale sigma for the isotropic case.
    """
    sigma_inv = 1.0 / sigma

    d = y.unsqueeze(0) - x_samples

    r = torch.linalg.norm(d, dim=-1) * sigma_inv
    w = (r.clamp_min(eps)).pow(beta - 2.0)

    mean_term = (w.unsqueeze(-1) * d).mean(dim=0)
    score = -(lam / (sigma**2)) * mean_term

    return score


@torch.no_grad()
def mc_score_from_posterior_samples_spd(
    y: torch.Tensor,
    x_samples: torch.Tensor,
    sigma: float,
    beta: float,
    lam: float,
    Sigma: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Energy-Tweedie noisy-score estimate for ``sigma**2 * Sigma``."""
    field = scaled_energy_tweedie_field_from_samples(
        y,
        x_samples,
        beta=beta,
        Sigma=Sigma,
        sigma_values=sigma,
        eps=eps,
        sample_layout="kbd",
    )
    return float(lam) * field


@torch.no_grad()
def mc_score_from_posterior_samples_repeated_spd(
    y: torch.Tensor,
    x_samples: torch.Tensor,
    *,
    beta: float,
    lam: float,
    rho: float,
    uv: torch.Tensor | list[float] | tuple[float, float],
    eps: float = 1e-12,
) -> torch.Tensor:
    """Energy-Tweedie score for a repeated 2-by-2 SPD image operator."""
    if y.ndim != 2 or x_samples.ndim != 3:
        raise ValueError("expected y=(B,D) and x_samples=(K,B,D)")
    if x_samples.shape[1:] != y.shape:
        raise ValueError("posterior sample batch and dimension must match y")
    geometry = RepeatedBlockSPD(uv, y.shape[1], device=y.device, dtype=y.dtype)
    sigma = reference_rho_to_sigma(
        torch.as_tensor(rho, device=y.device, dtype=y.dtype),
        torch.as_tensor(beta, device=y.device, dtype=y.dtype),
        y.shape[1],
    )
    residual = y.unsqueeze(0) - x_samples
    inverse_residual = geometry.apply_inverse(residual)
    quadratic = (residual * inverse_residual).sum(dim=-1).clamp_min(0.0)
    radius = quadratic.sqrt() / sigma
    weights = radius.clamp_min(float(eps)).pow(float(beta) - 2.0)
    mean_term = (weights.unsqueeze(-1) * inverse_residual).mean(dim=0)
    return -(float(lam) / sigma.square()) * mean_term


@torch.no_grad()
def richardson_extrapolate(
    s1: torch.Tensor, s2: torch.Tensor, *, sigma1: float, sigma2: float
) -> torch.Tensor:
    """
    Richardson extrapolation to estimate clean score from two noisy score estimates.

    Args:
        s1: Score estimate at sigma1 (B, D)
        s2: Score estimate at sigma2 (B, D)
        sigma1: First noise level (should be smaller)
        sigma2: Second noise level (should be larger)

    Returns:
        Extrapolated clean score estimate (B, D)
    """
    sigma1 = float(sigma1)
    sigma2 = float(sigma2)
    return (sigma2**2 * s1 - sigma1**2 * s2) / (sigma2**2 - sigma1**2)


def get_noise_schedule(
    n_levels: int = 100,
    sigma_min: float = 0.01,
    sigma_max: float = 0.5,
    schedule: str = "cosine",
):
    """
    Generate noise schedule focused on small noise levels.
    """
    if schedule == "linear":
        sigmas = np.linspace(sigma_min, sigma_max, n_levels)
    elif schedule == "geometric":
        sigmas = np.geomspace(sigma_min, sigma_max, n_levels)
    elif schedule == "cosine":

        t = np.linspace(0, 1, n_levels)
        s = 1 - np.cos(t * np.pi / 2)
        sigmas = sigma_min + (sigma_max - sigma_min) * s
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    return sigmas.astype(np.float32)


@torch.inference_mode()
def annealed_langevin_generate_parameter_path(
    posterior_sampler_fn,
    *,
    n_samples: int,
    dim: int,
    schedule: dict[str, torch.Tensor],
    n_steps_each: int | Sequence[int] | torch.Tensor = 10,
    first_level_steps: int | None = None,
    n_mc: int | Sequence[int] | torch.Tensor = 8,
    base_lr: float | None = None,
    alpha_cap: float = 0.0025,
    update_clip: float = 5.0,
    state_clip: tuple[float, float] | None = (-2.0, 3.0),
    device: str | torch.device = "cpu",
    verbose: bool = True,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> dict[str, Any]:
    """Sample along a generalized-Gaussian parameter path in image space.

    The schedule must contain descending ``actual_rms`` values and aligned
    ``beta``, ``lam``, ``rho``, and ``uv`` tensors. `n_steps_each`` and ``n_mc`` may
    be scalars or one value per level. ``first_level_steps`` can give the
    initial pure-noise state additional mixing before the noise level descends.
    """
    n_samples = int(n_samples)
    dim = int(dim)
    if first_level_steps is not None:
        first_level_steps = int(first_level_steps)
    if min(n_samples, dim) <= 0:
        raise ValueError("sample counts and dimension must be positive")
    if first_level_steps is not None and first_level_steps <= 0:
        raise ValueError("first_level_steps must be positive when provided")
    if float(alpha_cap) <= 0 or float(update_clip) <= 0:
        raise ValueError("alpha_cap and update_clip must be positive")
    if state_clip is not None and float(state_clip[0]) >= float(state_clip[1]):
        raise ValueError("state_clip must have increasing lower and upper bounds")
    device = torch.device(device)
    beta_values = torch.as_tensor(schedule["beta"], device=device)
    lam_values = torch.as_tensor(schedule["lam"], device=device)
    rho_values = torch.as_tensor(schedule["rho"], device=device)
    uv_values = torch.as_tensor(schedule["uv"], device=device)
    rms_values = torch.as_tensor(schedule["actual_rms"], device=device)
    levels = int(beta_values.numel())
    if levels <= 1:
        raise ValueError("the path must contain at least two levels")

    def integer_level_schedule(values, name):
        """Return positive integer budgets, repeating a shared integer across levels.

        Per-level inputs must contain exactly ``levels`` entries. ``name``
        identifies the setting in validation errors.
        """
        if isinstance(values, int):
            level_values = [int(values)] * levels
        else:
            values_tensor = torch.as_tensor(values).reshape(-1)
            if values_tensor.numel() != levels:
                raise ValueError(f"{name} must be scalar or have one value per level")
            level_values = [int(value) for value in values_tensor.tolist()]
        if min(level_values) <= 0:
            raise ValueError(f"{name} values must be positive")
        return level_values

    steps_by_level = integer_level_schedule(n_steps_each, "n_steps_each")
    mc_by_level = integer_level_schedule(n_mc, "n_mc")
    if any(
        values.shape[0] != levels
        for values in (lam_values, rho_values, uv_values, rms_values)
    ):
        raise ValueError("all schedule entries must have the same number of levels")
    if uv_values.shape != (levels, 2):
        raise ValueError("schedule uv must have shape (levels, 2)")
    if torch.any(rms_values[1:] > rms_values[:-1] + 1e-7):
        raise ValueError("actual_rms must be ordered from largest to smallest")

    first_beta = float(beta_values[0])
    first_lam = float(lam_values[0])
    first_rho = float(rho_values[0])
    y = sample_repeated_block_generalized_gaussian(
        n_samples,
        dim,
        beta=first_beta,
        lam=first_lam,
        rho=first_rho,
        uv=uv_values[0],
        device=device,
        dtype=beta_values.dtype,
    )
    rms_max = float(rms_values[0])
    rms_min = float(rms_values[-1])
    if base_lr is None:
        base_lr = float(alpha_cap) * (rms_min / rms_max) ** 2

    trajectory = [y.detach().cpu()]
    diagnostics = []
    level_iterator = tqdm(
        range(levels),
        desc=progress_desc or "parameter-path levels",
        leave=False,
        disable=not show_progress,
    )
    for level in level_iterator:
        beta = float(beta_values[level])
        lam = float(lam_values[level])
        rho = float(rho_values[level])
        uv = uv_values[level]
        rms = float(rms_values[level])
        alpha = min(float(base_lr) * (rms / rms_min) ** 2, float(alpha_cap))
        noise_scale = math.sqrt(alpha)
        score_norm_sum = 0.0
        update_norm_sum = 0.0
        clipped_fraction_sum = 0.0
        level_steps = (
            first_level_steps
            if level == 0 and first_level_steps is not None
            else steps_by_level[level]
        )
        level_n_mc = mc_by_level[level]

        for _ in range(level_steps):
            posterior = posterior_sampler_fn(y, beta, lam, rho, uv, level_n_mc)
            score = mc_score_from_posterior_samples_repeated_spd(
                y,
                posterior,
                beta=beta,
                lam=lam,
                rho=rho,
                uv=uv,
            )
            update = 0.5 * alpha * score + noise_scale * torch.randn_like(y)
            update = torch.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)
            update_norm = torch.linalg.vector_norm(update, dim=-1, keepdim=True)
            clipped = update_norm > float(update_clip)
            update = update * (
                float(update_clip) / update_norm.clamp_min(1e-12)
            ).clamp_max(1.0)
            y = y + update
            if state_clip is not None:
                y = y.clamp(float(state_clip[0]), float(state_clip[1]))

            score_norm_sum += float(torch.linalg.vector_norm(score, dim=-1).mean())
            update_norm_sum += float(update_norm.mean())
            clipped_fraction_sum += float(clipped.float().mean())

        trajectory.append(y.detach().cpu())
        diagnostics.append(
            {
                "level": level,
                "beta": beta,
                "lam": lam,
                "rho": rho,
                "actual_rms": rms,
                "u": float(uv[0]),
                "v": float(uv[1]),
                "alpha": alpha,
                "n_steps": level_steps,
                "n_mc": level_n_mc,
                "state_clip_active": state_clip is not None,
                "mean_score_norm": score_norm_sum / level_steps,
                "mean_update_norm_before_clip": update_norm_sum / level_steps,
                "mean_clipped_fraction": clipped_fraction_sum / level_steps,
                "state_min": float(y.min()),
                "state_max": float(y.max()),
                "state_std": float(y.std()),
            }
        )
        if verbose:
            print(
                f"Completed path level {level + 1}/{levels}: "
                f"rho={rho:.4f}, beta={beta:.3f}, lambda={lam:.3f}",
                flush=True,
            )

    final = posterior_sampler_fn(
        y,
        float(beta_values[-1]),
        float(lam_values[-1]),
        float(rho_values[-1]),
        uv_values[-1],
        1,
    ).squeeze(0)
    return {
        "final": final.detach().cpu(),
        "last_noisy": y.detach().cpu(),
        "trajectory": trajectory,
        "diagnostics": diagnostics,
        "base_lr": float(base_lr),
    }

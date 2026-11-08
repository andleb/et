from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch


def spd_2d_from_uv(uv: torch.Tensor | Sequence[float]) -> torch.Tensor:
    """Map ``(..., 2)`` disk coordinates to trace-two SPD matrices."""
    uv_tensor = torch.as_tensor(uv)
    if uv_tensor.shape[-1:] != (2,):
        raise ValueError("uv must have final dimension 2")
    radius_squared = uv_tensor.square().sum(dim=-1)
    if torch.any(radius_squared >= 1.0):
        raise ValueError("u^2 + v^2 must be less than 1")

    u, v = uv_tensor.unbind(dim=-1)
    row_0 = torch.stack((1.0 + u, v), dim=-1)
    row_1 = torch.stack((v, 1.0 - u), dim=-1)
    return torch.stack((row_0, row_1), dim=-2)


def generalized_gaussian_rms_constant(
    beta: torch.Tensor | float,
    dim: int,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return the per-coordinate RMS at unit ``sigma`` and ``lambda=1``."""
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be positive")
    beta_tensor = torch.as_tensor(beta, device=device, dtype=dtype)
    if torch.any(beta_tensor <= 0):
        raise ValueError("beta must be positive")
    dim_tensor = torch.as_tensor(
        float(dim), device=beta_tensor.device, dtype=beta_tensor.dtype
    )
    log_constant_squared = (
        (2.0 / beta_tensor) * beta_tensor.log()
        - dim_tensor.log()
        + torch.lgamma((dim_tensor + 2.0) / beta_tensor)
        - torch.lgamma(dim_tensor / beta_tensor)
    )
    return (0.5 * log_constant_squared).exp()


def reference_rho_to_sigma(
    rho: torch.Tensor | float,
    beta: torch.Tensor | float,
    dim: int,
) -> torch.Tensor:
    """Convert reference per-pixel RMS ``rho`` to generalized scale ``sigma``."""
    rho_tensor = torch.as_tensor(rho)
    if torch.any(rho_tensor <= 0):
        raise ValueError("rho must be positive")
    beta_tensor = torch.as_tensor(
        beta, device=rho_tensor.device, dtype=rho_tensor.dtype
    )
    return rho_tensor / generalized_gaussian_rms_constant(
        beta_tensor,
        dim,
        device=rho_tensor.device,
        dtype=rho_tensor.dtype,
    )


def generalized_gaussian_actual_rms(
    rho: torch.Tensor | float,
    beta: torch.Tensor | float,
    lam: torch.Tensor | float,
) -> torch.Tensor:
    """Return actual per-pixel RMS for reference scale ``rho`` and ``lambda``."""
    rho_tensor = torch.as_tensor(rho)
    beta_tensor = torch.as_tensor(
        beta, device=rho_tensor.device, dtype=rho_tensor.dtype
    )
    lam_tensor = torch.as_tensor(lam, device=rho_tensor.device, dtype=rho_tensor.dtype)
    if torch.any(rho_tensor <= 0) or torch.any(beta_tensor <= 0):
        raise ValueError("rho and beta must be positive")
    if torch.any(lam_tensor <= 0):
        raise ValueError("lam must be positive")
    return rho_tensor * lam_tensor.pow(-1.0 / beta_tensor)


def generalized_gaussian_conditional_score(
    residual: torch.Tensor,
    *,
    beta: torch.Tensor | float,
    lam: torch.Tensor | float,
    Sigma: torch.Tensor,
    sigma: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Noise score at ``y - x`` for scale matrix ``sigma**2 * Sigma``."""
    shape = torch.as_tensor(Sigma, device=residual.device, dtype=residual.dtype)
    beta = torch.as_tensor(beta, device=residual.device, dtype=residual.dtype)
    lam = torch.as_tensor(lam, device=residual.device, dtype=residual.dtype)
    sigma = torch.as_tensor(sigma, device=residual.device, dtype=residual.dtype)
    if bool((beta <= 1).any()) or bool((lam <= 0).any()) or bool((sigma <= 0).any()):
        raise ValueError("This score helper requires beta > 1, lambda > 0, sigma > 0")
    inverse_residual = torch.linalg.solve(shape, residual.unsqueeze(-1)).squeeze(-1)
    radius = (residual * inverse_residual).sum(-1).clamp_min(0).sqrt() / sigma
    safe_radius = torch.where(radius > 0, radius, torch.ones_like(radius))
    return (
        -(lam / sigma.square())
        * safe_radius.pow(beta - 2).unsqueeze(-1)
        * inverse_residual
    )


def generalized_gaussian_score_second_moment(
    *, beta: float, lam: float, Sigma: torch.Tensor, sigma: float = 1.0
) -> float:
    """Exact E[||grad log q(epsilon)||^2] for scale matrix sigma**2 * Sigma."""
    dim = Sigma.shape[-1]
    beta, lam, sigma = float(beta), float(lam), float(sigma)
    if beta <= max(0, 1 - dim / 2) or lam <= 0 or sigma <= 0:
        raise ValueError("The noise score must have a finite second moment")
    exponent = 2 * beta - 2
    radial_moment = math.exp(
        (exponent / beta) * math.log(beta / lam)
        + math.lgamma((dim + exponent) / beta)
        - math.lgamma(dim / beta)
    )
    inverse_trace = float(torch.linalg.inv(Sigma.double()).trace())
    return lam**2 * radial_moment * inverse_trace / (dim * sigma**2)


def _sample_spherical_generalized_gaussian(
    n: int,
    dim: int,
    *,
    beta: torch.Tensor | float,
    lam: torch.Tensor | float,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample the unit-scale spherical part of a generalized Gaussian."""
    n = int(n)
    dim = int(dim)
    if n <= 0 or dim <= 0:
        raise ValueError("n and dim must be positive")
    beta_tensor = torch.as_tensor(beta, device=device, dtype=dtype)
    lam_tensor = torch.as_tensor(lam, device=device, dtype=dtype)
    if beta_tensor.ndim != 0 or lam_tensor.ndim != 0:
        raise ValueError("beta and lam must be scalar")
    if beta_tensor <= 0 or lam_tensor <= 0:
        raise ValueError("beta and lam must be positive")

    direction = torch.randn(n, dim, device=device, dtype=dtype)
    direction = direction / torch.linalg.vector_norm(
        direction, dim=-1, keepdim=True
    ).clamp_min(1e-12)
    concentration = (
        torch.as_tensor(float(dim), device=device, dtype=dtype) / beta_tensor
    )
    radial_gamma = torch.distributions.Gamma(
        concentration=concentration,
        rate=torch.ones((), device=device, dtype=dtype),
    ).sample((n,))
    radius = (beta_tensor / lam_tensor).pow(1.0 / beta_tensor) * radial_gamma.pow(
        1.0 / beta_tensor
    )
    return direction * radius[:, None]


def sample_generalized_gaussian_noise(
    n: int,
    *,
    beta: torch.Tensor | float,
    lam: torch.Tensor | float,
    Sigma: torch.Tensor | None = None,
    sigma: float = 1.0,
    dim: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample isotropic or dense-SPD generalized-Gaussian noise."""
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    cholesky = None
    if Sigma is None:
        if dim is None:
            raise ValueError("dim must be provided when Sigma is None")
        dim = int(dim)
    else:
        geometry = torch.as_tensor(Sigma, device=device, dtype=dtype)
        if geometry.ndim != 2 or geometry.shape[0] != geometry.shape[1]:
            raise ValueError(f"Sigma must be square, got shape {tuple(geometry.shape)}")
        cholesky = torch.linalg.cholesky(geometry)
        dim = int(geometry.shape[0])

    spherical = _sample_spherical_generalized_gaussian(
        n,
        dim,
        beta=beta,
        lam=lam,
        device=device,
        dtype=dtype,
    )
    if cholesky is None:
        return sigma * spherical
    return sigma * (spherical @ cholesky.T)


def build_fixed_shape_actual_rms_schedule(
    n_levels: int,
    actual_rms_start: float,
    actual_rms_end: float,
    *,
    beta: float,
    lam: float,
    uv: torch.Tensor | Sequence[float],
    dim: int,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Build a decreasing actual-RMS schedule at one fixed noise shape."""
    n_levels = int(n_levels)
    actual_rms_start = float(actual_rms_start)
    actual_rms_end = float(actual_rms_end)
    beta = float(beta)
    lam = float(lam)
    dim = int(dim)
    if n_levels < 2:
        raise ValueError("n_levels must be at least two")
    if not all(
        math.isfinite(value) for value in (actual_rms_start, actual_rms_end, beta, lam)
    ):
        raise ValueError("schedule parameters must be finite")
    if actual_rms_start <= actual_rms_end or actual_rms_end <= 0:
        raise ValueError("actual RMS endpoints must be positive and decreasing")
    if beta <= 0 or lam <= 0:
        raise ValueError("beta and lam must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")

    dtype = dtype or torch.get_default_dtype()
    progress = torch.linspace(
        0.0,
        1.0,
        n_levels,
        device=device,
        dtype=dtype,
    )
    actual_rms = actual_rms_start * (actual_rms_end / actual_rms_start) ** progress
    beta_schedule = torch.full_like(progress, beta)
    lam_schedule = torch.full_like(progress, lam)
    rho = actual_rms * lam_schedule.pow(1.0 / beta_schedule)
    uv_tensor = torch.as_tensor(uv, device=device, dtype=dtype)
    if uv_tensor.shape != (2,):
        raise ValueError("uv must contain one fixed pair")
    if not torch.isfinite(uv_tensor).all():
        raise ValueError("uv must be finite")
    spd_2d_from_uv(uv_tensor)
    uv_schedule = uv_tensor.reshape(1, 2).expand(n_levels, 2).clone()
    sigma = reference_rho_to_sigma(rho, beta_schedule, dim)
    torch.testing.assert_close(
        generalized_gaussian_actual_rms(rho, beta_schedule, lam_schedule),
        actual_rms,
    )
    return {
        "progress": progress,
        "beta": beta_schedule,
        "lam": lam_schedule,
        "rho": rho,
        "sigma": sigma,
        "actual_rms": actual_rms,
        "uv": uv_schedule,
    }


class RepeatedBlockSPD:

    def __init__(
        self,
        uv: torch.Tensor | Sequence[float],
        dim: int,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        dim = int(dim)
        if dim <= 0 or dim % 2 != 0:
            raise ValueError("dim must be a positive even integer")
        uv_tensor = torch.as_tensor(uv, device=device, dtype=dtype)
        if uv_tensor.shape != (2,):
            raise ValueError("RepeatedBlockSPD requires one shared uv pair")
        self.uv = uv_tensor
        self.dim = dim
        self.block = spd_2d_from_uv(uv_tensor)
        self.cholesky = torch.linalg.cholesky(self.block)

    @property
    def trace(self) -> torch.Tensor:
        """Trace of the implicit full operator."""
        return self.block.diagonal().sum() * (self.dim // 2)

    def _pairs(self, values: torch.Tensor) -> torch.Tensor:
        """View the final coordinate axis as ``dim // 2`` adjacent pairs."""
        if values.shape[-1] != self.dim:
            raise ValueError(
                f"values must have final dimension {self.dim}, got {values.shape[-1]}"
            )
        return values.reshape(*values.shape[:-1], self.dim // 2, 2)

    def apply_sqrt(self, values: torch.Tensor) -> torch.Tensor:
        """Right-apply the symmetric covariance shape through its Cholesky root."""
        pairs = self._pairs(values)
        transformed = pairs @ self.cholesky.transpose(-1, -2)
        return transformed.reshape_as(values)

    def apply_inverse(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the inverse of the repeated-block SPD operator."""
        pairs = self._pairs(values)
        solved = torch.linalg.solve(self.block, pairs.unsqueeze(-1)).squeeze(-1)
        return solved.reshape_as(values)


def repeated_block_energy_score(
    target: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    beta: torch.Tensor | float,
    sigma: torch.Tensor | float,
    uv: torch.Tensor | Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a two-draw energy score using repeated-block Mahalanobis distances.

    Whiten residuals using ``uv``, divide their norms by ``sigma``, and
    raise them to ``beta``. All observations and draws share these noise
    parameters. Return ``(loss, prediction, dispersion)``, where prediction
    averages the two target-to-draw terms and dispersion compares the draws.
    """
    if target.shape != first.shape or target.shape != second.shape:
        raise ValueError("target and posterior draws must have the same shape")
    geometry = RepeatedBlockSPD(
        uv, target.shape[-1], device=target.device, dtype=target.dtype
    )
    beta = torch.as_tensor(beta, device=target.device, dtype=target.dtype)
    sigma = torch.as_tensor(sigma, device=target.device, dtype=target.dtype)
    if beta.ndim != 0 or sigma.ndim != 0 or not 0 < beta <= 2 or sigma <= 0:
        raise ValueError("Require scalar 0 < beta <= 2 and sigma > 0")
    differences = torch.stack((first - target, second - target, first - second))
    pairs = differences.reshape(*differences.shape[:-1], -1, 2)
    whitened = (
        torch.linalg.solve_triangular(
            geometry.cholesky, pairs.unsqueeze(-1), upper=False
        )
        .squeeze(-1)
        .reshape_as(differences)
    )
    distances = (torch.linalg.vector_norm(whitened, dim=-1) / sigma).pow(beta)
    prediction = 0.5 * (distances[0].mean() + distances[1].mean())
    dispersion = distances[2].mean()

    return prediction - 0.5 * dispersion, prediction, dispersion


def sample_repeated_block_generalized_gaussian(
    n: int,
    dim: int,
    *,
    beta: torch.Tensor | float,
    lam: torch.Tensor | float,
    rho: torch.Tensor | float,
    uv: torch.Tensor | Sequence[float],
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample structured generalized-Gaussian noise at reference RMS ``rho``."""
    n = int(n)
    dim = int(dim)
    if n <= 0:
        raise ValueError("n must be positive")
    geometry = RepeatedBlockSPD(uv, dim, device=device, dtype=dtype)
    beta_tensor = torch.as_tensor(beta, device=device, dtype=dtype)
    lam_tensor = torch.as_tensor(lam, device=device, dtype=dtype)
    rho_tensor = torch.as_tensor(rho, device=device, dtype=dtype)
    if torch.any(beta_tensor <= 0) or torch.any(lam_tensor <= 0):
        raise ValueError("beta and lam must be positive")
    sigma = reference_rho_to_sigma(rho_tensor, beta_tensor, dim)

    spherical = _sample_spherical_generalized_gaussian(
        n,
        dim,
        beta=beta_tensor,
        lam=lam_tensor,
        device=device,
        dtype=dtype,
    )
    return sigma * geometry.apply_sqrt(spherical)


def uv_from_spd_2d(matrix: torch.Tensor | Sequence[Sequence[float]]) -> torch.Tensor:
    """Return ``(u, v)`` for a symmetric trace-two 2D SPD matrix."""
    matrix_tensor = torch.as_tensor(matrix)
    if matrix_tensor.shape[-2:] != (2, 2):
        raise ValueError("matrix must have final shape (2, 2)")
    if not torch.allclose(
        matrix_tensor,
        matrix_tensor.transpose(-1, -2),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("matrix must be symmetric")
    trace = matrix_tensor.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    if not torch.allclose(trace, torch.full_like(trace, 2.0), atol=1e-6, rtol=1e-6):
        raise ValueError("matrix must have trace 2")
    torch.linalg.cholesky(matrix_tensor)
    return torch.stack(
        (
            (matrix_tensor[..., 0, 0] - matrix_tensor[..., 1, 1]) / 2.0,
            matrix_tensor[..., 0, 1],
        ),
        dim=-1,
    )


def sample_spd_2d_uv(
    n: int,
    *,
    radius_max: float = 0.7,
    radial_distribution: str = "uniform_area",
    anchors: torch.Tensor | Sequence[Sequence[float]] | None = None,
    anchor_probability: float = 0.0,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample bounded SPD coordinates, optionally mixing in exact anchors.

    ``uniform_area`` is uniform over the disk. ``uniform_radius`` assigns equal
    probability to equal radial intervals and therefore visits nearly
    isotropic shapes more often.
    """
    n = int(n)
    radius_max = float(radius_max)
    anchor_probability = float(anchor_probability)
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 <= radius_max < 1.0:
        raise ValueError("radius_max must be in [0, 1)")
    if not 0.0 <= anchor_probability <= 1.0:
        raise ValueError("anchor_probability must be in [0, 1]")
    if radial_distribution not in {"uniform_area", "uniform_radius"}:
        raise ValueError(
            "radial_distribution must be 'uniform_area' or 'uniform_radius'"
        )

    dtype = dtype or torch.get_default_dtype()
    unit = torch.rand(n, device=device, dtype=dtype, generator=generator)
    radius = radius_max * (
        unit.sqrt() if radial_distribution == "uniform_area" else unit
    )
    angle = (
        2.0 * math.pi * torch.rand(n, device=device, dtype=dtype, generator=generator)
    )
    uv = torch.stack((radius * angle.cos(), radius * angle.sin()), dim=-1)

    if anchors is None:
        if anchor_probability != 0.0:
            raise ValueError("anchors are required when anchor_probability is positive")
        return uv

    anchor_tensor = torch.as_tensor(anchors, device=device, dtype=dtype)
    if (
        anchor_tensor.ndim != 2
        or anchor_tensor.shape[1] != 2
        or anchor_tensor.shape[0] == 0
    ):
        raise ValueError("anchors must have shape (n_anchors, 2)")
    anchor_radius = torch.linalg.vector_norm(anchor_tensor, dim=-1)
    if torch.any(anchor_radius > radius_max + 1e-7):
        raise ValueError("all anchors must lie inside radius_max")
    if torch.any(anchor_radius >= 1.0):
        raise ValueError("all anchors must define positive-definite matrices")

    use_anchor = (
        torch.rand(n, device=device, dtype=dtype, generator=generator)
        < anchor_probability
    )
    anchor_index = torch.randint(
        anchor_tensor.shape[0],
        (n,),
        device=device,
        generator=generator,
    )
    return torch.where(use_anchor[:, None], anchor_tensor[anchor_index], uv)


def _full_spd_condition_bounds(
    noise_config: Any,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bounds for encoding ``(beta, log(lam), log(rho), u, v)``.

    Read the ranges from ``noise_config`` and match the device and dtype
    of ``reference``.
    """
    radius = float(noise_config.spd.radius_max)
    lower = reference.new_tensor(
        [
            float(noise_config.beta_min),
            math.log(float(noise_config.lambda_min)),
            math.log(float(noise_config.rho_min)),
            -radius,
            -radius,
        ]
    )
    upper = reference.new_tensor(
        [
            float(noise_config.beta_max),
            math.log(float(noise_config.lambda_max)),
            math.log(float(noise_config.rho_max)),
            radius,
            radius,
        ]
    )
    return lower, upper


def normalize_full_spd_condition(
    beta: torch.Tensor | float,
    lam: torch.Tensor | float,
    rho: torch.Tensor | float,
    uv: torch.Tensor | Sequence[float],
    noise_config: Any,
    *,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Encode ``(beta, log lambda, log rho, u, v)`` affinely to ``[-1, 1]``."""
    uv_tensor = torch.as_tensor(uv)
    if uv_tensor.shape[-1:] != (2,):
        raise ValueError("uv must have final dimension 2")
    beta_tensor = torch.as_tensor(beta, device=uv_tensor.device, dtype=uv_tensor.dtype)
    lam_tensor = torch.as_tensor(lam, device=uv_tensor.device, dtype=uv_tensor.dtype)
    rho_tensor = torch.as_tensor(rho, device=uv_tensor.device, dtype=uv_tensor.dtype)
    beta_tensor, lam_tensor, rho_tensor, u_tensor, v_tensor = torch.broadcast_tensors(
        beta_tensor,
        lam_tensor,
        rho_tensor,
        uv_tensor[..., 0],
        uv_tensor[..., 1],
    )
    if torch.any(beta_tensor <= 0):
        raise ValueError("beta must be positive")
    if torch.any(lam_tensor <= 0) or torch.any(rho_tensor <= 0):
        raise ValueError("lam and rho must be positive")
    raw = torch.stack(
        (beta_tensor, lam_tensor.log(), rho_tensor.log(), u_tensor, v_tensor),
        dim=-1,
    )
    lower, upper = _full_spd_condition_bounds(noise_config, raw)
    if torch.any(upper <= lower):
        raise ValueError("all full-SPD conditioning ranges must have positive width")
    tolerance = 1e-6
    radius = torch.linalg.vector_norm(
        torch.stack((u_tensor, v_tensor), dim=-1),
        dim=-1,
    )
    if torch.any(radius > float(noise_config.spd.radius_max) + tolerance):
        raise ValueError("uv lies outside the configured SPD disk")
    if torch.any(raw < lower - tolerance) or torch.any(raw > upper + tolerance):
        raise ValueError("noise condition lies outside the configured bounds")
    normalized = 2.0 * (raw - lower) / (upper - lower) - 1.0
    if normalized.ndim == 1:
        normalized = normalized.unsqueeze(0)
    if batch_size is not None:
        if normalized.shape[0] not in {1, int(batch_size)}:
            raise ValueError("condition must be shared or have one row per example")
        normalized = normalized.expand(int(batch_size), -1)
    return normalized


@dataclass(frozen=True)
class FullSPDNoiseBatch:
    beta: torch.Tensor
    lam: torch.Tensor
    rho: torch.Tensor
    uv: torch.Tensor
    sigma: torch.Tensor
    actual_rms: torch.Tensor
    condition: torch.Tensor
    used_anchor: torch.Tensor

    def parameter_vector(self) -> torch.Tensor:
        return torch.stack(
            (self.beta, self.lam, self.rho, self.uv[0], self.uv[1]),
            dim=0,
        )


class FullSPDNoiseParameterSampler:

    def __init__(
        self,
        noise_config: Any,
        *,
        dim: int,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = noise_config
        self.dim = int(dim)
        self.device = torch.device(device)
        self.dtype = dtype
        if self.dim <= 0 or self.dim % 2 != 0:
            raise ValueError("dim must be a positive even integer")

        self.anchor_probability = float(noise_config.anchor_probability)
        if not 0.0 <= self.anchor_probability <= 1.0:
            raise ValueError("noise.anchor_probability must be in [0, 1]")

        anchors = []
        for anchor in noise_config.anchors:
            anchors.append(
                [
                    float(anchor.beta),
                    float(anchor.lam),
                    float(anchor.rho),
                    float(anchor.u),
                    float(anchor.v),
                ]
            )
        if not anchors and self.anchor_probability > 0.0:
            raise ValueError("noise.anchors cannot be empty when anchors are sampled")
        self.anchors = torch.as_tensor(anchors, device=self.device, dtype=self.dtype)
        if self.anchors.numel() > 0:
            normalize_full_spd_condition(
                self.anchors[:, 0],
                self.anchors[:, 1],
                self.anchors[:, 2],
                self.anchors[:, 3:5],
                self.config,
            )
            spd_2d_from_uv(self.anchors[:, 3:5])

    def sample(self, batch_size: int) -> FullSPDNoiseBatch:
        """Draw one noise-parameter tuple shared by a batch.

        Mix draws from the configured continuous parameter distribution with
        fixed anchors. Return the parameters, induced noise scales, and an
        encoded conditioning matrix of shape ``(batch_size, 5)``.
        """
        beta = torch.empty((), device=self.device, dtype=self.dtype).uniform_(
            float(self.config.beta_min),
            float(self.config.beta_max),
        )
        log_lam = torch.empty((), device=self.device, dtype=self.dtype).uniform_(
            math.log(float(self.config.lambda_min)),
            math.log(float(self.config.lambda_max)),
        )
        log_rho = torch.empty((), device=self.device, dtype=self.dtype).uniform_(
            math.log(float(self.config.rho_min)),
            math.log(float(self.config.rho_max)),
        )
        uv = sample_spd_2d_uv(
            1,
            radius_max=float(self.config.spd.radius_max),
            radial_distribution=str(self.config.spd.radial_distribution),
            device=self.device,
            dtype=self.dtype,
        )[0]
        random_parameters = torch.stack((beta, log_lam.exp(), log_rho.exp(), *uv))

        used_anchor = torch.zeros((), device=self.device, dtype=torch.bool)
        parameters = random_parameters
        if self.anchors.numel() > 0:
            used_anchor = torch.rand((), device=self.device) < self.anchor_probability
            index = torch.randint(self.anchors.shape[0], (), device=self.device)
            parameters = torch.where(
                used_anchor,
                self.anchors[index],
                random_parameters,
            )

        beta, lam, rho, u, v = parameters.unbind()
        uv = torch.stack((u, v))
        condition = normalize_full_spd_condition(
            beta,
            lam,
            rho,
            uv,
            self.config,
            batch_size=int(batch_size),
        )
        sigma = reference_rho_to_sigma(rho, beta, self.dim)
        actual_rms = generalized_gaussian_actual_rms(rho, beta, lam)
        return FullSPDNoiseBatch(
            beta=beta,
            lam=lam,
            rho=rho,
            uv=uv,
            sigma=sigma,
            actual_rms=actual_rms,
            condition=condition,
            used_anchor=used_anchor,
        )

    def corrupt(
        self,
        clean: torch.Tensor,
        parameters: FullSPDNoiseBatch,
    ) -> torch.Tensor:
        """Add independent generalized-Gaussian noise draws to the rows of ``clean``.

        All rows share ``parameters`` and its repeated-block SPD shape.
        Return noisy observations with the same ``(batch, dim)`` shape.
        """
        if clean.ndim != 2 or clean.shape[1] != self.dim:
            raise ValueError(f"clean must have shape (batch, {self.dim})")
        noise = sample_repeated_block_generalized_gaussian(
            clean.shape[0],
            self.dim,
            beta=parameters.beta,
            lam=parameters.lam,
            rho=parameters.rho,
            uv=parameters.uv,
            device=clean.device,
            dtype=clean.dtype,
        )
        return clean + noise


def log_sigma_spd_condition(
    log_sigma: torch.Tensor | float,
    uv: torch.Tensor | Sequence[float],
    batch_size: int | None = None,
) -> torch.Tensor:
    """Combine log-noise magnitude and SPD coordinates into ``(B, 3)``."""
    uv_tensor = torch.as_tensor(uv)
    if uv_tensor.shape[-1:] != (2,):
        raise ValueError("uv must have final dimension 2")
    if uv_tensor.ndim == 1:
        uv_tensor = uv_tensor.reshape(1, 2)
    elif uv_tensor.ndim != 2:
        raise ValueError("uv must have shape (2,) or (B, 2)")

    log_sigma_tensor = torch.as_tensor(
        log_sigma,
        device=uv_tensor.device,
        dtype=uv_tensor.dtype,
    ).reshape(-1, 1)
    inferred_batch = max(log_sigma_tensor.shape[0], uv_tensor.shape[0])
    target_batch = inferred_batch if batch_size is None else int(batch_size)
    if target_batch <= 0:
        raise ValueError("batch_size must be positive")
    if log_sigma_tensor.shape[0] not in {1, target_batch}:
        raise ValueError("log_sigma must be scalar or have one value per batch row")
    if uv_tensor.shape[0] not in {1, target_batch}:
        raise ValueError("uv must be one shared shape or have one shape per batch row")

    return torch.cat(
        (
            log_sigma_tensor.expand(target_batch, 1),
            uv_tensor.expand(target_batch, 2),
        ),
        dim=-1,
    )

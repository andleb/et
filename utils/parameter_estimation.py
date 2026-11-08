from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch
from torch import nn
from torch.nn import functional as F

from .sampling import (
    SampleLayout,
    coerce_posterior_samples,
    energy_tweedie_field_from_samples,
    scaled_energy_tweedie_field_from_samples,
)

CovarianceType = Literal["isotropic", "diagonal", "full"]


ScoreEstimator = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class ParameterEstimationResult:

    beta: float
    lam: float
    Sigma: torch.Tensor
    objective: float
    lambda_unclipped: float
    covariance_type: CovarianceType
    history: list[dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return estimates, fit diagnostics, and optimization history as a dictionary."""
        return {
            "beta": self.beta,
            "lambda": self.lam,
            "Sigma": self.Sigma,
            "objective": self.objective,
            "lambda_unclipped": self.lambda_unclipped,
            "covariance_type": self.covariance_type,
            "history": self.history,
        }


def profile_lambda(
    score: torch.Tensor,
    field: torch.Tensor,
    *,
    lambda_bounds: tuple[float, float] = (1e-8, float("inf")),
    ridge: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Least-squares profile for scalar lambda.

    Returns ``(lambda_clipped, lambda_unclipped, numerator, denominator)``.
    """
    numerator = (score * field).sum()
    denominator = (field.square()).sum().clamp_min(float(ridge))
    lam_unclipped = numerator / denominator
    lo, hi = lambda_bounds
    lam = torch.clamp(lam_unclipped, min=float(lo), max=float(hi))
    return lam, lam_unclipped, numerator, denominator


def profiled_score_reconstruction(
    score: torch.Tensor,
    field: torch.Tensor,
    *,
    lambda_bounds: tuple[float, float] = (1e-8, 1e8),
    ridge: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Profile lambda and return score reconstruction diagnostics."""
    lam, lam_raw, numerator, denominator = profile_lambda(
        score,
        field,
        lambda_bounds=lambda_bounds,
        ridge=ridge,
    )
    pred = lam * field
    objective = (score - pred).square().sum(dim=-1).mean()
    return {
        "lambda": lam,
        "lambda_unclipped": lam_raw,
        "objective": objective,
        "numerator": numerator,
        "denominator": denominator,
        "pred": pred,
    }


class _BetaParameter(nn.Module):
    def __init__(
        self,
        *,
        beta_min: float,
        beta_init: float,
        learn_beta: bool = True,
    ) -> None:
        super().__init__()
        if not (0.0 < beta_min < 2.0):
            raise ValueError("beta_min must lie in (0, 2)")
        self.beta_min = float(beta_min)
        self.learn_beta = bool(learn_beta)

        if learn_beta:
            beta_init = min(max(float(beta_init), beta_min + 1e-5), 2.0 - 1e-5)
            p = (beta_init - beta_min) / (2.0 - beta_min)
            rho = math.log(p / (1.0 - p))
            self.rho = nn.Parameter(torch.tensor(float(rho)))
            self.register_buffer("_fixed_beta", torch.empty(0))
        else:
            beta_init = min(max(float(beta_init), beta_min), 2.0)
            self.register_parameter("rho", None)
            self.register_buffer("_fixed_beta", torch.tensor(float(beta_init)))

    def forward(self) -> torch.Tensor:
        if self.rho is None:
            return self._fixed_beta
        return self.beta_min + (2.0 - self.beta_min) * torch.sigmoid(self.rho)


class TraceNormalizedCovariance(nn.Module):

    def __init__(
        self,
        dim: int,
        *,
        covariance_type: CovarianceType = "full",
        init_sigma: torch.Tensor | None = None,
        diag_jitter: float = 1e-5,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if covariance_type not in {"isotropic", "diagonal", "full"}:
            raise ValueError(f"unknown covariance_type: {covariance_type}")

        self.dim = int(dim)
        self.covariance_type = covariance_type
        self.diag_jitter = float(diag_jitter)

        if covariance_type == "isotropic":
            self.register_buffer("_eye", torch.eye(dim))
            return

        init_sigma = self._normalize_initial_sigma(init_sigma)
        if covariance_type == "diagonal":
            diag = torch.diag(init_sigma) if init_sigma is not None else torch.ones(dim)
            diag = self.dim * diag.clamp_min(diag_jitter) / diag.sum()
            self.raw_diag = nn.Parameter(torch.log(diag / self.dim))
            return

        rows, cols = torch.tril_indices(dim, dim)
        self.register_buffer("_tril_rows", rows)
        self.register_buffer("_tril_cols", cols)
        self.register_buffer("_tril_diag_mask", rows == cols)
        raw = torch.zeros(rows.numel())
        if init_sigma is not None:
            chol = torch.linalg.cholesky(init_sigma)
            vals = chol[rows, cols]
            diag_vals = vals[rows == cols].clamp_min(diag_jitter * 2.0)
            vals[rows == cols] = _inv_softplus(diag_vals - diag_jitter)
            raw = vals
        self.raw_tril = nn.Parameter(raw)

    def _normalize_initial_sigma(
        self, sigma: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Copy and symmetrize an initial shape, then rescale its trace to ``dim``."""
        if sigma is None:
            return None
        sigma = sigma.detach().clone().float()
        if sigma.shape != (self.dim, self.dim):
            raise ValueError(
                f"init_sigma must have shape {(self.dim, self.dim)}, got {tuple(sigma.shape)}"
            )
        sigma = 0.5 * (sigma + sigma.T)
        return self.dim * sigma / torch.trace(sigma)

    def forward(self) -> torch.Tensor:
        """Return the current SPD shape matrix, normalized to trace ``dim``."""
        if self.covariance_type == "isotropic":
            return self._eye

        if self.covariance_type == "diagonal":
            diag = self.dim * torch.softmax(self.raw_diag, dim=0)
            return torch.diag(diag)

        raw = self.raw_tril
        rows = self._tril_rows
        cols = self._tril_cols
        diag_mask = self._tril_diag_mask
        vals = raw.clone()
        vals[diag_mask] = F.softplus(vals[diag_mask]) + self.diag_jitter

        L = raw.new_zeros((self.dim, self.dim))
        L[rows, cols] = vals
        sigma = L @ L.T
        return self.dim * sigma / torch.trace(sigma)


class ProfiledNoiseParameterModel(nn.Module):

    def __init__(
        self,
        dim: int,
        *,
        beta_min: float = 0.25,
        beta_init: float = 1.5,
        learn_beta: bool = True,
        covariance_type: CovarianceType = "full",
        init_sigma: torch.Tensor | None = None,
        diag_jitter: float = 1e-5,
    ) -> None:
        super().__init__()
        self.beta_param = _BetaParameter(
            beta_min=beta_min,
            beta_init=beta_init,
            learn_beta=learn_beta,
        )
        self.covariance = TraceNormalizedCovariance(
            dim,
            covariance_type=covariance_type,
            init_sigma=init_sigma,
            diag_jitter=diag_jitter,
        )

    @property
    def covariance_type(self) -> CovarianceType:
        return self.covariance.covariance_type

    def beta(self) -> torch.Tensor:
        return self.beta_param()

    def Sigma(self) -> torch.Tensor:
        return self.covariance()

    def field(
        self,
        y: torch.Tensor,
        x_samples: torch.Tensor,
        *,
        eps: float = 1e-12,
        sample_layout: SampleLayout = "auto",
    ) -> torch.Tensor:
        """Evaluate the Energy-Tweedie field using the current ``beta`` and ``Sigma``.

        Use posterior draws in ``sample_layout`` and return one vector per
        observation in ``y``.
        """
        return energy_tweedie_field_from_samples(
            y,
            x_samples,
            beta=self.beta(),
            Sigma=self.Sigma(),
            eps=eps,
            sample_layout=sample_layout,
        )


def fit_multiscale_profiled_noise_parameters(
    y: torch.Tensor,
    sigma_values: torch.Tensor,
    x_samples: torch.Tensor,
    *,
    score_estimator: ScoreEstimator | torch.Tensor,
    n_steps: int = 1000,
    batch_size: int = 256,
    covariance_type: CovarianceType = "diagonal",
    beta_min: float = 0.25,
    beta_init: float = 1.5,
    learn_beta: bool = True,
    init_sigma: torch.Tensor | None = None,
    lambda_bounds: tuple[float, float] = (1e-8, float("inf")),
    lr: float = 1e-2,
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
    eps: float = 1e-12,
    ridge: float = 1e-12,
    detach_profile_lambda: bool = False,
    eval_batch_size: int | None = None,
    sample_layout: SampleLayout = "auto",
    grad_clip: float | None = None,
    log_every: int = 50,
    seed: int | None = None,
    device: str | torch.device | None = None,
) -> ParameterEstimationResult:
    """Fit the profiled objective with per-observation noise scales.

    This is the setting where the shape matrix ``Sigma`` is shared
    but each observation has covariance shape ``sigma_i^2 Sigma``.  Posterior
    samples are precomputed and supplied as ``x_samples``.
    """
    if y.ndim != 2:
        raise ValueError(f"y must have shape (n, d), got {tuple(y.shape)}")
    if sigma_values.reshape(-1).shape[0] != y.shape[0]:
        raise ValueError("sigma_values must have one entry per row of y")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if seed is not None:
        torch.manual_seed(int(seed))

    target_device = torch.device(device) if device is not None else y.device
    y = y.to(target_device)
    sigma_values = sigma_values.to(device=target_device, dtype=y.dtype).reshape(-1)
    x_kbd = coerce_posterior_samples(
        x_samples.to(device=target_device, dtype=y.dtype),
        y,
        layout=sample_layout,
    )
    x_bkd = x_kbd.permute(1, 0, 2).contiguous()
    n, dim = y.shape

    score_tensor: torch.Tensor | None
    if torch.is_tensor(score_estimator):
        score_tensor = score_estimator.to(device=target_device, dtype=y.dtype)
        if score_tensor.shape != y.shape:
            raise ValueError(
                f"score tensor must match y shape {tuple(y.shape)}, got {tuple(score_tensor.shape)}"
            )
    else:
        score_tensor = None

    model = ProfiledNoiseParameterModel(
        dim,
        beta_min=beta_min,
        beta_init=beta_init,
        learn_beta=learn_beta,
        covariance_type=covariance_type,
        init_sigma=init_sigma,
    ).to(device=target_device, dtype=y.dtype)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("there are no learnable parameters to optimize")
    optimizer = optimizer_cls(params, lr=lr)

    history: list[dict[str, float]] = []
    effective_batch = min(int(batch_size), n)

    for step in range(1, n_steps + 1):
        idx = torch.randint(n, (effective_batch,), device=target_device)
        y_b = y[idx]
        sig_b = sigma_values[idx]
        s_b = _score_batch(score_estimator, score_tensor, idx, y_b)
        x_b = x_bkd[idx]

        field_b = scaled_energy_tweedie_field_from_samples(
            y_b,
            x_b,
            beta=model.beta(),
            Sigma=model.Sigma(),
            sigma_values=sig_b,
            eps=eps,
            sample_layout="bkd",
        )
        lam, lam_raw, numerator, denominator = profile_lambda(
            s_b,
            field_b,
            lambda_bounds=lambda_bounds,
            ridge=ridge,
        )
        lam_for_loss = lam.detach() if detach_profile_lambda else lam
        loss = (s_b - lam_for_loss * field_b).square().sum(dim=-1).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
        optimizer.step()

        if log_every > 0 and (step == 1 or step % log_every == 0 or step == n_steps):
            history_row = {
                "step": float(step),
                "objective": float(loss.detach().cpu()),
                "beta": float(model.beta().detach().cpu()),
                "lambda": float(lam.detach().cpu()),
                "lambda_unclipped": float(lam_raw.detach().cpu()),
                "alignment_numerator": float(numerator.detach().cpu()),
                "field_sq_denominator": float(denominator.detach().cpu()),
            }
            Sigma_history = model.Sigma().detach().cpu()
            if dim == 2:
                history_row.update(
                    {
                        "sigma_00": float(Sigma_history[0, 0]),
                        "sigma_01": float(Sigma_history[0, 1]),
                        "sigma_11": float(Sigma_history[1, 1]),
                        "u": float(0.5 * (Sigma_history[0, 0] - Sigma_history[1, 1])),
                        "v": float(Sigma_history[0, 1]),
                    }
                )
            history.append(history_row)

    eval_stats = evaluate_multiscale_profiled_noise_parameters(
        y,
        sigma_values,
        x_bkd,
        score_estimator=score_estimator,
        beta=model.beta().detach(),
        Sigma=model.Sigma().detach(),
        lambda_bounds=lambda_bounds,
        batch_size=eval_batch_size or effective_batch,
        eps=eps,
        ridge=ridge,
        sample_layout="bkd",
    )

    return ParameterEstimationResult(
        beta=float(model.beta().detach().cpu()),
        lam=float(eval_stats["lambda"].detach().cpu()),
        Sigma=model.Sigma().detach().cpu(),
        objective=float(eval_stats["objective"].detach().cpu()),
        lambda_unclipped=float(eval_stats["lambda_unclipped"].detach().cpu()),
        covariance_type=covariance_type,
        history=history,
    )


def _multiscale_fields_and_scores(
    y,
    sigma_values,
    x_samples,
    *,
    score_estimator,
    beta,
    Sigma,
    batch_size,
    eps,
    sample_layout,
):
    """Evaluate Energy-Tweedie fields and target scores in observation batches."""
    if y.ndim != 2:
        raise ValueError(f"y must have shape (n, d), got {tuple(y.shape)}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    device = y.device
    dtype = y.dtype
    sigma_values = sigma_values.to(device=device, dtype=dtype).reshape(-1)
    if sigma_values.shape[0] != y.shape[0]:
        raise ValueError("sigma_values must have one entry per row of y")
    x_kbd = coerce_posterior_samples(
        x_samples.to(device=device, dtype=dtype),
        y,
        layout=sample_layout,
    )
    x_bkd = x_kbd.permute(1, 0, 2).contiguous()
    Sigma = Sigma.to(device=device, dtype=dtype)
    beta_t = torch.as_tensor(beta, device=device, dtype=dtype)

    score_tensor = None
    if torch.is_tensor(score_estimator):
        score_tensor = score_estimator.to(device=device, dtype=dtype)
        if score_tensor.shape != y.shape:
            raise ValueError(
                f"score tensor must match y shape {tuple(y.shape)}, got {tuple(score_tensor.shape)}"
            )

    fields = []
    scores = []
    n = y.shape[0]
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        y_b = y[start:stop]
        idx = torch.arange(start, stop, device=device)
        s_b = _score_batch(score_estimator, score_tensor, idx, y_b)
        g_b = scaled_energy_tweedie_field_from_samples(
            y_b,
            x_bkd[start:stop],
            beta=beta_t,
            Sigma=Sigma,
            sigma_values=sigma_values[start:stop],
            eps=eps,
            sample_layout="bkd",
        )
        fields.append(g_b)
        scores.append(s_b)

    field_all = torch.cat(fields, dim=0)
    score_all = torch.cat(scores, dim=0)
    return field_all, score_all


@torch.no_grad()
def evaluate_multiscale_profiled_noise_parameters(
    y: torch.Tensor,
    sigma_values: torch.Tensor,
    x_samples: torch.Tensor,
    *,
    score_estimator: ScoreEstimator | torch.Tensor,
    beta: torch.Tensor | float,
    Sigma: torch.Tensor,
    lambda_bounds: tuple[float, float] = (1e-8, float("inf")),
    batch_size: int = 1024,
    eps: float = 1e-12,
    ridge: float = 1e-12,
    sample_layout: SampleLayout = "auto",
) -> dict[str, torch.Tensor]:
    """Evaluate the multiscale profiled objective for fixed beta and Sigma."""
    field_all, score_all = _multiscale_fields_and_scores(
        y,
        sigma_values,
        x_samples,
        score_estimator=score_estimator,
        beta=beta,
        Sigma=Sigma,
        batch_size=batch_size,
        eps=eps,
        sample_layout=sample_layout,
    )
    lam, lam_raw, numerator, denominator = profile_lambda(
        score_all,
        field_all,
        lambda_bounds=lambda_bounds,
        ridge=ridge,
    )
    objective = (score_all - lam * field_all).square().sum(dim=-1).mean()
    return {
        "objective": objective,
        "lambda": lam,
        "lambda_unclipped": lam_raw,
        "alignment_numerator": numerator,
        "field_sq_denominator": denominator,
        "field": field_all,
        "score": score_all,
    }


@torch.no_grad()
def evaluate_multiscale_noise_parameters(
    y: torch.Tensor,
    sigma_values: torch.Tensor,
    x_samples: torch.Tensor,
    *,
    score_estimator: ScoreEstimator | torch.Tensor,
    beta: torch.Tensor | float,
    lam: torch.Tensor | float,
    Sigma: torch.Tensor,
    reference_score: torch.Tensor | None = None,
    batch_size: int = 1024,
    eps: float = 1e-12,
    sample_layout: SampleLayout = "auto",
) -> dict[str, torch.Tensor]:
    """Evaluate a fixed multiscale parameter fit on held-out queries.

    ``identity_error`` compares the reconstructed field with ``score_estimator``.
    When ``reference_score`` is supplied, ``reference_error`` compares the same
    reconstruction with that reference independently of the fitted score source.
    Both pointwise errors are returned to support grouped diagnostics.
    """
    field, score = _multiscale_fields_and_scores(
        y,
        sigma_values,
        x_samples,
        score_estimator=score_estimator,
        beta=beta,
        Sigma=Sigma,
        batch_size=int(batch_size),
        eps=eps,
        sample_layout=sample_layout,
    )
    lam_t = torch.as_tensor(lam, device=y.device, dtype=y.dtype)
    prediction = lam_t * field
    identity_error = (score - prediction).square().sum(dim=-1)
    result = {
        "identity_residual": identity_error.mean(),
        "identity_error": identity_error,
        "prediction": prediction,
        "field": field,
        "score": score,
    }
    if reference_score is not None:
        reference_score = reference_score.to(device=y.device, dtype=y.dtype)
        if reference_score.shape != y.shape:
            raise ValueError(
                "reference_score must match y shape "
                f"{tuple(y.shape)}, got {tuple(reference_score.shape)}"
            )
        reference_error = (reference_score - prediction).square().sum(dim=-1)
        result["reference_mse"] = reference_error.mean()
        result["reference_error"] = reference_error
    return result


def _score_batch(
    score_estimator: ScoreEstimator | torch.Tensor,
    score_tensor: torch.Tensor | None,
    idx: torch.Tensor,
    y_b: torch.Tensor,
) -> torch.Tensor:
    """Select cached scores or evaluate the score estimator for a query batch.

    ``idx`` indexes the cached score tensor when supplied; otherwise the
    estimator receives ``y_b``. Return detached scores matching ``y_b``.
    """
    if score_tensor is not None:
        return score_tensor[idx].detach()
    with torch.no_grad():
        score = score_estimator(y_b)
    if score.shape != y_b.shape:
        raise ValueError(
            f"score_estimator must return shape {tuple(y_b.shape)}, got {tuple(score.shape)}"
        )
    return score.detach()


def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(1e-12)
    return x + torch.log(-torch.expm1(-x))

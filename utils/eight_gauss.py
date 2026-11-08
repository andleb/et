import math

import torch

from .sampling import mc_score_from_posterior_samples


def eight_gaussians(
    n: int,
    radius: float = 2.0,
    std: float = 0.1,
    device: str = "cpu",
    means=None,
):
    """Sample from 8 Gaussians arranged in a circle."""
    if means is None:
        means = eight_gaussians_centers(radius, device=device)

    z = torch.randint(0, 8, (n,), device=device)
    x = means[z] + torch.randn(n, 2, device=device) * std

    return x


def eight_gaussians_centers(radius: float, device=None, dtype=None) -> torch.Tensor:
    """
    Returns centers mu_k on a ring of given radius: shape (8, 2).
    """
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    k = torch.arange(8, device=device, dtype=dtype)
    angles = 2.0 * math.pi * k / 8.0
    mus = radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    return mus


def eight_gaussians_score(
    x: torch.Tensor, radius: float = 2.0, std: float = 0.1, device=None
) -> torch.Tensor:
    """Gradient of the Eight-Gaussians log density."""
    device = x.device if device is None else device
    means = eight_gaussians_centers(radius, device=device, dtype=x.dtype)

    diff = x[:, None, :] - means[None, :, :]
    log_comp = -0.5 * (
        torch.sum(diff**2, dim=-1) / std**2 + 2 * math.log(std * math.sqrt(2 * math.pi))
    )

    w = torch.softmax(log_comp, dim=1)

    return -torch.sum(w[..., None] * diff, dim=1) / std**2


def eight_gaussians_noisy_score(
    y: torch.Tensor,
    sigma: float,
    std_clean: float = 0.1,
    radius: float = 2.0,
) -> torch.Tensor:
    """Exact noisy marginal score for Gaussian noise."""
    mus = eight_gaussians_centers(radius, device=y.device, dtype=y.dtype)
    v = std_clean**2 + sigma**2
    diff = y[:, None, :] - mus[None, :, :]
    log_w = -0.5 * (torch.sum(diff**2, dim=-1) / v)
    w = torch.softmax(log_w, dim=1)
    return torch.sum(w[..., None] * (mus[None, :, :] - y[:, None, :]), dim=1) / v


def empirical_mode_weights_nearest_center(
    samples: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """
    Nearest-center assignment => empirical mode weights.

    Args:
        samples: (n, S, 2) (recommended) OR (S, 2) (treated as n=1)
    Returns:
        w_hat: (n, 8)
    """
    if samples.dim() == 2:
        samples = samples.unsqueeze(0)

    n, S, d = samples.shape
    mus = eight_gaussians_centers(radius, device=samples.device, dtype=samples.dtype)

    flat = samples.reshape(-1, d)
    dist = torch.cdist(flat, mus, p=2)
    idx = dist.argmin(dim=1).view(n, S)

    w_hat = torch.stack([(idx == k).float().mean(dim=1) for k in range(8)], dim=1)
    return w_hat


def energy_distance_unbiased(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Unbiased finite-sample energy distance estimator:
        ED = 2 E||X-Y|| - E||X-X'|| - E||Y-Y'||
    using averages over all pairs, excluding diagonals for within terms.

    Args:
        x: (n, d)
        y: (m, d)
    Returns:
        scalar tensor
    """
    n = x.shape[0]
    m = y.shape[0]
    if n < 2 or m < 2:
        raise ValueError(
            "energy_distance_unbiased requires at least two samples per input"
        )

    dxy = torch.cdist(x, y, p=2).mean()
    dxx_mean = torch.pdist(x, p=2).mean()
    dyy_mean = torch.pdist(y, p=2).mean()

    return 2.0 * dxy - dxx_mean - dyy_mean


def make_evaluation_grid(
    radius: float = 2.0,
    n_grid: int = 50,
    padding: float = 1.5,
    device: str = "cpu",
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Create a 2D grid of points for score evaluation.

    Returns:
        grid: (n_grid^2, 2) tensor of grid points
    """
    extent = radius * padding
    x = torch.linspace(-extent, extent, n_grid, device=device, dtype=dtype)
    y = torch.linspace(-extent, extent, n_grid, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return grid


def compute_noisy_score_mse_grid(
    posterior_sampler_fn,
    sigma: float,
    std_clean: float = 0.1,
    radius: float = 2.0,
    n_grid: int = 50,
    n_mc: int = 256,
    device: str = "cpu",
) -> dict:
    """
    Compute score MSE between model's implied score and analytic noisy score on a grid.

    For Gaussian noise, the noisy score at sigma uses effective variance = std_clean^2 + sigma^2.

    Returns:
        dict with 'mse', 'grid', 's_hat', 's_true'
    """
    grid = make_evaluation_grid(
        radius=radius, n_grid=n_grid, padding=2.0, device=device
    )

    x_samples = posterior_sampler_fn(grid, sigma, n_mc)
    mu_hat = x_samples.mean(dim=0)

    s_hat = (mu_hat - grid) / (sigma**2)

    s_true = eight_gaussians_noisy_score(
        grid, sigma=sigma, std_clean=std_clean, radius=radius
    )

    mse = float(((s_hat - s_true) ** 2).mean())

    return {
        "mse": mse,
        "grid": grid.cpu(),
        "s_hat": s_hat.cpu(),
        "s_true": s_true.cpu(),
    }


def true_generalized_noisy_score(
    y: torch.Tensor,
    sigma: float,
    beta: float,
    lam: float,
    std_clean: float = 0.1,
    radius: float = 2.0,
    n_prior_mc: int = 8192,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Approximate the noisy marginal score for generalized-Gaussian noise,
    using Monte Carlo over the prior p(X).
    """
    y = y.to(device)
    x_prior = eight_gaussians(n_prior_mc, radius=radius, std=std_clean, device=device)

    sigma = float(sigma)
    sigma2 = sigma * sigma

    d_yx = y.unsqueeze(1) - x_prior.unsqueeze(0)
    r = torch.linalg.norm(d_yx, dim=-1) / sigma

    log_q = -(lam / beta) * (r.clamp_min(1e-12) ** beta)
    log_q_max, _ = log_q.max(dim=1, keepdim=True)
    w = torch.exp(log_q - log_q_max)

    grad_log_q = (
        -lam / sigma2 * (r.clamp_min(1e-12) ** (beta - 2.0)).unsqueeze(-1) * d_yx
    )

    num = (w.unsqueeze(-1) * grad_log_q).sum(dim=1)
    den = w.sum(dim=1, keepdim=True)

    sm_hat = num / den.clamp_min(1e-12)
    return sm_hat


def eight_gaussians_logpdf(
    x: torch.Tensor, radius: float = 2.0, std: float = 0.1
) -> torch.Tensor:
    """Log density of the Eight-Gaussians mixture."""
    means = eight_gaussians_centers(radius, device=x.device, dtype=x.dtype)

    diff = x[:, None, :] - means[None, :, :]
    log_comp = -0.5 * (
        torch.sum(diff**2, dim=-1) / std**2 + 2 * math.log(std * math.sqrt(2 * math.pi))
    )

    return torch.logsumexp(log_comp, dim=1) - math.log(8)


@torch.no_grad()
def eight_gaussians_noise_convolution_score(
    y: torch.Tensor,
    noise_draws: torch.Tensor,
    *,
    radius: float = 2.0,
    std_clean: float = 0.2,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Estimate grad log E_noise[p(y - noise)] using the analytic clean score.

    This representation is useful when noise is narrower than a clean mixture
    component, where clean-prior importance sampling has few nearby particles.
    Call on small query batches to bound memory use.
    """
    max_logw = torch.full((len(y),), -torch.inf, device=y.device, dtype=y.dtype)
    denominator = torch.zeros_like(max_logw)
    numerator = torch.zeros_like(y)
    for start in range(0, len(noise_draws), chunk_size):
        noise = noise_draws[start : start + chunk_size].to(y)
        clean = y[:, None, :] - noise[None, :, :]
        flat = clean.reshape(-1, 2)
        logw = eight_gaussians_logpdf(flat, radius=radius, std=std_clean).reshape(
            len(y), len(noise)
        )
        score = eight_gaussians_score(flat, radius=radius, std=std_clean).reshape(
            len(y), len(noise), 2
        )
        new_max = torch.maximum(max_logw, logw.max(dim=1).values)
        previous_scale = (max_logw - new_max).exp()
        weights = (logw - new_max[:, None]).exp()
        denominator = denominator * previous_scale + weights.sum(dim=1)
        numerator = numerator * previous_scale[:, None] + (
            weights[:, :, None] * score
        ).sum(dim=1)
        max_logw = new_max
    return numerator / denominator[:, None]


def compute_noisy_score_mse_grid_generalized(
    posterior_sampler_fn,
    sigma: float,
    std_clean: float,
    radius: float,
    beta: float,
    lam: float,
    n_grid: int = 50,
    n_mc: int = 256,
    n_prior_mc: int = 8192,
    device: str = "cpu",
) -> dict:
    """Compare posterior-implied scores with a prior Monte Carlo reference on a grid.

    The posterior sampler receives ``(grid, sigma, n_mc)`` and returns draws
    of shape ``(n_mc, B, 2)``. Return the coordinate-wise MSE and CPU tensors
    for the grid, estimated scores, and reference scores.
    """
    grid = make_evaluation_grid(
        radius=radius, n_grid=n_grid, padding=2.0, device=device
    )

    x_samples = posterior_sampler_fn(grid, sigma, n_mc)
    s_hat = mc_score_from_posterior_samples(
        y=grid, x_samples=x_samples, sigma=sigma, beta=beta, lam=lam
    )

    s_true = true_generalized_noisy_score(
        y=grid,
        sigma=sigma,
        beta=beta,
        lam=lam,
        std_clean=std_clean,
        radius=radius,
        n_prior_mc=n_prior_mc,
        device=device,
    )

    mse = float(((s_hat - s_true) ** 2).mean())

    return {
        "mse": mse,
        "grid": grid.cpu(),
        "s_hat": s_hat.cpu(),
        "s_true": s_true.cpu(),
    }

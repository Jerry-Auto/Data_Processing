"""Distribution metrics for the known conditional Gaussian-mixture toy target."""
from __future__ import annotations
import torch
from .toy_data import CENTERS

TARGET_STD = .25


@torch.no_grad()
def conditional_gaussian_metrics(samples: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Estimate per-condition Gaussian moments and compare them to the true target.

    Each target component is ``N(center_k, TARGET_STD**2 I)``.  ``w2`` is the
    exact 2-Wasserstein distance between that target Gaussian and the Gaussian
    fitted to each generated component, then averaged over conditions.  It thus
    penalizes both a shifted mode and collapsed/excessive covariance.
    """
    centers = torch.tensor(CENTERS, device=samples.device, dtype=samples.dtype)
    finite = torch.isfinite(samples).all(dim=-1)
    finite_fraction = finite.float().mean().item()
    if finite_fraction < 1.0:
        return dict(mean_error=float('inf'), covariance_error=float('inf'),
                    w2=float('inf'), component_std=float('inf'),
                    finite_fraction=finite_fraction)
    target_var = TARGET_STD ** 2
    target_cov = torch.eye(samples.shape[-1], device=samples.device, dtype=samples.dtype) * target_var
    mean_errors, cov_errors, w2_values, component_stds = [], [], [], []
    for class_id in range(len(CENTERS)):
        x = samples[labels == class_id]
        if len(x) < 2:
            # Random benchmark batches can omit a class when deliberately run
            # at tiny smoke-test sizes. A covariance is then undefined.
            return dict(mean_error=float('inf'), covariance_error=float('inf'),
                        w2=float('inf'), component_std=float('inf'),
                        finite_fraction=finite_fraction)
        mean = x.mean(dim=0)
        centered = x - mean
        covariance = centered.T @ centered / len(x)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.)
        # Since target covariance is s^2 I, the Gaussian W2 expression reduces to
        # ||mu-mu*||^2 + sum_i (sqrt(lambda_i(C)) - s)^2.
        w2_squared = (mean - centers[class_id]).square().sum() + (eigenvalues.sqrt() - TARGET_STD).square().sum()
        mean_errors.append((mean - centers[class_id]).norm())
        cov_errors.append(torch.linalg.matrix_norm(covariance - target_cov, ord='fro'))
        w2_values.append(w2_squared.sqrt())
        component_stds.append(eigenvalues.mean().sqrt())
    return dict(mean_error=torch.stack(mean_errors).mean().item(),
                covariance_error=torch.stack(cov_errors).mean().item(),
                w2=torch.stack(w2_values).mean().item(),
                component_std=torch.stack(component_stds).mean().item(),
                finite_fraction=finite_fraction)

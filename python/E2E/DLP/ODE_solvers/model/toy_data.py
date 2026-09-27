"""Synthetic conditional data used by the small DiT solver experiment."""

import torch
from torch.nn import functional as F

CENTERS = ((2., 0.), (-2., 0.), (0., 2.), (0., -2.))


def sample_data(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Gaussian-mixture targets and a one-hot scene condition."""
    centers = torch.tensor(CENTERS, device=device)
    labels = torch.randint(len(CENTERS), (batch_size,), device=device)
    data = centers[labels] + .25 * torch.randn(batch_size, 2, device=device)
    condition = F.one_hot(labels, num_classes=len(CENTERS)).float()
    return data, condition

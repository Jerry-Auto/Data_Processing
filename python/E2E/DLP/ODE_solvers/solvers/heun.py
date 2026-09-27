"""Heun / explicit trapezoidal second-order solver for CondOT flow matching."""

import torch
from torch import Tensor

from .base import FixedGridSolver


class HeunSolver(FixedGridSolver):
    """Predict with Euler, then average velocities at both endpoints."""

    name, order = "heun", 2

    @torch.no_grad()
    def sample(self, x_init: Tensor, times: Tensor) -> Tensor:
        times, x = self._validate_times(x_init, times), x_init
        self.stats.nfe = 0
        for t, s in zip(times[:-1], times[1:]):
            h = s - t
            k1 = self._velocity(x, t)
            k2 = self._velocity(x + h * k1, s)
            x = x + .5 * h * (k1 + k2)
        return x

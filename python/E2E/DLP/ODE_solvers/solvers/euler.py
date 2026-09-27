"""Forward Euler for the CondOT flow ODE dx/dt = v_theta(x, t)."""

import torch
from torch import Tensor

from .base import FixedGridSolver


class EulerSolver(FixedGridSolver):
    """First-order explicit Euler: ``x_{n+1}=x_n+(t_{n+1}-t_n)v_n``."""

    name, order = "euler", 1

    @torch.no_grad()
    def sample(self, x_init: Tensor, times: Tensor) -> Tensor:
        times, x = self._validate_times(x_init, times), x_init
        self.stats.nfe = 0
        for t, s in zip(times[:-1], times[1:]):
            x = x + (s - t) * self._velocity(x, t)
        return x

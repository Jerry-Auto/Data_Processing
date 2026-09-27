"""Classical fourth-order Runge--Kutta solver for CondOT flow matching."""

import torch
from torch import Tensor

from .base import FixedGridSolver


class RK4Solver(FixedGridSolver):
    """Classical RK4; four velocity evaluations per integration interval."""

    name, order = "rk4", 4

    @torch.no_grad()
    def sample(self, x_init: Tensor, times: Tensor) -> Tensor:
        times, x = self._validate_times(x_init, times), x_init
        self.stats.nfe = 0
        for t, s in zip(times[:-1], times[1:]):
            h = s - t
            k1 = self._velocity(x, t)
            k2 = self._velocity(x + .5 * h * k1, t + .5 * h)
            k3 = self._velocity(x + .5 * h * k2, t + .5 * h)
            k4 = self._velocity(x + h * k3, s)
            x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return x

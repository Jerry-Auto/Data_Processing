"""Explicit midpoint (second-order Runge--Kutta) for CondOT flow matching."""

import torch
from torch import Tensor

from .base import FixedGridSolver


class MidpointSolver(FixedGridSolver):
    """RK2 midpoint; evaluates velocity at the predicted half-step."""

    name, order = "midpoint", 2

    @torch.no_grad()
    def sample(self, x_init: Tensor, times: Tensor) -> Tensor:
        times, x = self._validate_times(x_init, times), x_init
        self.stats.nfe = 0
        for t, s in zip(times[:-1], times[1:]):
            h = s - t
            k1 = self._velocity(x, t)
            x = x + h * self._velocity(x + .5 * h * k1, t + .5 * h)
        return x

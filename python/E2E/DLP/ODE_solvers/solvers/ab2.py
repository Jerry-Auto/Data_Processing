"""Two-step Adams--Bashforth solver for CondOT flow matching."""

import torch
from torch import Tensor

from .base import FixedGridSolver


class AdamsBashforth2Solver(FixedGridSolver):
    """Euler warm-up, then variable-step AB2; one new NFE per step."""

    name, order = "ab2", 2

    @torch.no_grad()
    def sample(self, x_init: Tensor, times: Tensor) -> Tensor:
        times, x = self._validate_times(x_init, times), x_init
        self.stats.nfe = 0
        v_previous = None
        for index, (t, s) in enumerate(zip(times[:-1], times[1:])):
            h = s - t
            v_now = self._velocity(x, t)
            if index == 0:
                x = x + h * v_now
            else:
                # Integrate the linear interpolant through (t_{n-1}, v_{n-1})
                # and (t_n, v_n).  With r=h_n/h_{n-1}, this is
                # h_n[(1+r/2)v_n - (r/2)v_{n-1}], reducing to 3/2,-1/2 on
                # a uniform grid.
                h_previous = t - times[index - 1]
                ratio = h / h_previous
                x = x + h * ((1. + .5 * ratio) * v_now - .5 * ratio * v_previous)
            v_previous = v_now
        return x

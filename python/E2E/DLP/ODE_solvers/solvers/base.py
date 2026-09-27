"""Shared infrastructure for the hand-written CondOT ODE solver classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

VelocityFn = Callable[[Tensor, Tensor], Tensor]


@dataclass
class SolverStats:
    nfe: int = 0


class FixedGridSolver:
    """Base class for fixed, monotonically increasing grids on ``t in [0, 1]``."""

    name = "base"
    order = 0

    def __init__(self, velocity_model: VelocityFn):
        self.velocity_model = velocity_model
        self.stats = SolverStats()

    def _velocity(self, x: Tensor, t: Tensor) -> Tensor:
        self.stats.nfe += 1
        return self.velocity_model(x, t.to(device=x.device, dtype=x.dtype))

    @staticmethod
    def _validate_times(x_init: Tensor, times: Tensor) -> Tensor:
        if times.ndim != 1 or len(times) < 2:
            raise ValueError("times must be a 1-D tensor containing at least two values")
        if not bool(torch.all(times[1:] > times[:-1])):
            raise ValueError("times must be strictly increasing")
        return times.to(device=x_init.device, dtype=x_init.dtype)


"""Generic explicit probability-flow ODE solvers for a decreasing VP time grid."""
from __future__ import annotations
import torch


class VPExplicitSolver:
    """Euler, midpoint, Heun, RK4, and variable-step Adams--Bashforth 2."""
    def __init__(self, velocity, method): self.velocity, self.method, self.nfe = velocity, method, 0
    def f(self, x, t): self.nfe += 1; return self.velocity(x, t)
    @torch.no_grad()
    def sample(self, x, times):
        self.nfe = 0; previous = previous_h = None
        for t, s in zip(times[:-1], times[1:]):
            h = s - t; k1 = self.f(x, t)
            if self.method == 'euler': x = x + h*k1
            elif self.method == 'midpoint': x = x + h*self.f(x + .5*h*k1, t + .5*h)
            elif self.method == 'heun': x = x + .5*h*(k1 + self.f(x + h*k1, s))
            elif self.method == 'rk4':
                k2 = self.f(x + .5*h*k1, t + .5*h); k3 = self.f(x + .5*h*k2, t + .5*h); k4 = self.f(x + h*k3, s)
                x = x + h*(k1 + 2*k2 + 2*k3 + k4)/6
            elif self.method == 'ab2':
                if previous is None: x = x + h*k1
                else:
                    ratio = h/previous_h; x = x + h*((1 + .5*ratio)*k1 - .5*ratio*previous)
                previous, previous_h = k1, h
            else: raise ValueError('unknown method')
        return x

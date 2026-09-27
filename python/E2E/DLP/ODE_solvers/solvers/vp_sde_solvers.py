"""Symmetric VP-SDE solver facade, analogous to ``flow_matching_solvers.py``."""
from __future__ import annotations
from schedules import VPSDEParameterization
from .vp_explicit import VPExplicitSolver
from .vp_dpm_solver import VPDPM2MSolver


class VPSDESolver:
    """Wrap a raw network prediction as any supported VP-SDE ODE solver."""
    valid_methods = {'euler', 'midpoint', 'heun', 'rk4', 'ab2', 'dpmsolver_2m', 'dpmsolverpp_2m'}
    def __init__(self, predict, path, method='euler', prediction_type='noise'):
        if method not in self.valid_methods: raise ValueError(f'unknown VP-SDE method {method!r}')
        parameterization = VPSDEParameterization(predict, path, prediction_type)
        self.method = method
        self.impl = (VPDPM2MSolver(parameterization, path, 'dpmsolver++' if method == 'dpmsolverpp_2m' else 'dpmsolver')
                     if method.startswith('dpmsolver') else VPExplicitSolver(parameterization.velocity, method))
    @property
    def nfe(self): return self.impl.nfe
    def sample(self, x_init, times): return self.impl.sample(x_init, times)

"""Structure-aware CondOT two-step solvers in t-space and log-SNR space."""

import torch
from torch import Tensor

from .base import FixedGridSolver


class CondOT2MSolver(FixedGridSolver):
    """A CondOT analogue of a DPM++-2M-style exponential multistep update.

    It is not a mechanical VPSDE DPM++ port.  For velocity v, it caches the
    clean-endpoint estimate D=x+(1-t)v and integrates
    d[x/(1-t)]/dt = D/(1-t)^2 with a linear extrapolation of D.
    """

    name, order = "condot_2m", 2

    def __init__(self, velocity_model, lower_order_final: bool = True):
        super().__init__(velocity_model)
        self.lower_order_final = lower_order_final

    @staticmethod
    def _update(x: Tensor, d_previous: Tensor, d_now: Tensor,
                t_previous: Tensor, t_now: Tensor, t_next: Tensor) -> Tensor:
        u_now, u_next = 1. - t_now, 1. - t_next
        slope = (d_now - d_previous) / (t_now - t_previous)
        # lim_{u_next->0} u_next * log(u_next/u_now) = 0.  Avoid 0 * -inf.
        if bool(u_next == 0):
            return d_now + u_now * slope
        integral_linear_basis = u_now - u_next + u_next * torch.log(u_next / u_now)
        return ((u_next / u_now) * x + ((t_next - t_now) / u_now) * d_now
                + integral_linear_basis * slope)

    @torch.no_grad()
    def sample(self, x_init: Tensor, times: Tensor) -> Tensor:
        times, x = self._validate_times(x_init, times), x_init
        self.stats.nfe = 0
        d_previous = None
        final_index = len(times) - 2
        for index, (t, s) in enumerate(zip(times[:-1], times[1:])):
            v_now = self._velocity(x, t)
            d_now = x + (1. - t) * v_now
            if index == 0 or (self.lower_order_final and index == final_index):
                x = x + (s - t) * v_now  # Euler warm-up / robust endpoint fallback
            else:
                x = self._update(x, d_previous, d_now, times[index - 1], t, s)
            d_previous = d_now
        return x


class CondOTLambdaTwoStepSolver(FixedGridSolver):
    """CondOT data-prediction multistep solver on a uniform log-SNR grid.

    For CondOT, alpha=t and sigma=1-t.  With D=x+sigma*v, the ODE becomes
    ``d(x/sigma)/d(lambda) = exp(lambda) * D`` where
    ``lambda=log(alpha/sigma)``.  This class integrates that equation exactly
    after approximating D(lambda) by a constant (warm-up) or a linear
    polynomial (two-step update).

    ``variant='exact_linear'`` uses the exact integral of that linear
    polynomial.  ``variant='dpmpp'`` uses the DPM-Solver++-2M second-order
    coefficient, re-derived with CondOT alpha/sigma, for an apples-to-apples
    comparison with Diffusion-Planner's implementation.

    The lambda coordinate is infinite at t=0 and t=1.  Sampling therefore
    starts with a one-NFE Euler step from 0 to eps, solves on
    [eps, 1-eps], then returns the already-evaluated data prediction at
    1-eps.  No extra endpoint decoder call is used.
    """

    name, order = "condot_lambda_2m", 2

    def __init__(self, velocity_model, variant: str = "exact_linear"):
        super().__init__(velocity_model)
        if variant not in {"exact_linear", "dpmpp"}:
            raise ValueError("variant must be 'exact_linear' or 'dpmpp'")
        self.variant = variant

    @staticmethod
    def _lambda(t: Tensor) -> Tensor:
        return torch.log(t) - torch.log1p(-t)

    @staticmethod
    def _first_update(x: Tensor, d_now: Tensor, t_now: Tensor, t_next: Tensor,
                      lambda_now: Tensor, lambda_next: Tensor) -> Tensor:
        """Exact integral when D(lambda) is frozen at its current value."""
        h = lambda_next - lambda_now
        sigma_now, sigma_next = 1. - t_now, 1. - t_next
        # -alpha_next * (exp(-h)-1) == (t_next-t_now)/sigma_now.
        coefficient = -t_next * torch.expm1(-h)
        return (sigma_next / sigma_now) * x + coefficient * d_now

    def _second_update(self, x: Tensor, d_previous: Tensor, d_now: Tensor,
                       t_now: Tensor, t_next: Tensor,
                       lambda_previous: Tensor, lambda_now: Tensor,
                       lambda_next: Tensor) -> Tensor:
        """One-NFE second-order update derived by lambda-space interpolation."""
        h = lambda_next - lambda_now
        h0 = lambda_now - lambda_previous
        sigma_now, sigma_next = 1. - t_now, 1. - t_next
        first_coefficient = -t_next * torch.expm1(-h)
        difference = d_now - d_previous
        if self.variant == "exact_linear":
            # sigma_next * integral exp(lambda)*(lambda-lambda_now)dλ / h0
            second_coefficient = t_next * (torch.expm1(-h) + h) / h0
        else:  # DPM-Solver++-2M coefficient: 0.5*A0*(h/h0).
            second_coefficient = .5 * first_coefficient * h / h0
        return ((sigma_next / sigma_now) * x
                + first_coefficient * d_now
                + second_coefficient * difference)

    @torch.no_grad()
    def sample(self, x_init: Tensor, nfe: int, eps: float = 1e-3) -> Tensor:
        """Generate one endpoint sample using exactly ``nfe`` velocity calls."""
        if nfe < 2:
            raise ValueError("lambda-space two-step sampling requires nfe >= 2")
        if not 0. < eps < .5:
            raise ValueError("eps must lie strictly between 0 and 0.5")
        device, dtype = x_init.device, x_init.dtype
        eps_t = torch.tensor(eps, device=device, dtype=dtype)
        end_t = 1. - eps_t
        lambda_grid = torch.linspace(
            self._lambda(eps_t), self._lambda(end_t), nfe - 1,
            device=device, dtype=dtype,
        )
        time_grid = torch.sigmoid(lambda_grid)
        x, self.stats.nfe = x_init, 0

        # t=0 has no finite lambda: make a small first-order bridge to eps.
        v0 = self._velocity(x, torch.zeros((), device=device, dtype=dtype))
        x = x + eps_t * v0

        d_previous = lambda_previous = None
        for index, (t_now, lambda_now) in enumerate(zip(time_grid, lambda_grid)):
            v_now = self._velocity(x, t_now)
            d_now = x + (1. - t_now) * v_now
            is_last = index == len(time_grid) - 1
            if is_last:
                # D(x_{1-eps},1-eps) is the finite-endpoint data map.
                return d_now
            t_next, lambda_next = time_grid[index + 1], lambda_grid[index + 1]
            if index == 0:
                x = self._first_update(x, d_now, t_now, t_next, lambda_now, lambda_next)
            else:
                x = self._second_update(
                    x, d_previous, d_now, t_now, t_next,
                    lambda_previous, lambda_now, lambda_next,
                )
            d_previous, lambda_previous = d_now, lambda_now
        raise RuntimeError("unreachable")

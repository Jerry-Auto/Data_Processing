"""Compatibility facade for the individual CondOT solver modules."""

from .base import SolverStats, VelocityFn
from schedules import TimeSchedule, CondOTParameterization
from .euler import EulerSolver
from .midpoint import MidpointSolver
from .heun import HeunSolver
from .rk4 import RK4Solver
from .ab2 import AdamsBashforth2Solver
from .condot_2m import CondOT2MSolver
from .condot_2m import CondOTLambdaTwoStepSolver

SOLVER_CLASSES = {
    "euler": EulerSolver, "midpoint": MidpointSolver, "heun": HeunSolver,
           "rk4": RK4Solver, "ab2": AdamsBashforth2Solver, "condot_2m": CondOT2MSolver,
}


class CondOTSolver:
    """Backward-compatible factory delegating to an individual solver class."""

    valid_methods = set(SOLVER_CLASSES)

    def __init__(self, velocity_model: VelocityFn, method: str = "euler",
                 lower_order_final: bool = True, prediction_type: str = "velocity",
                 endpoint_eps: float = 1e-3):
        try:
            cls = SOLVER_CLASSES[method]
        except KeyError as exc:
            raise ValueError(f"Unknown method {method!r}; choose {sorted(SOLVER_CLASSES)}") from exc
        # Every solver receives velocity.  A direct-clean-data model is first
        # converted by the exact CondOT straight-path identity.
        flow_model = CondOTParameterization(velocity_model, prediction_type, endpoint_eps)
        self.impl = (cls(flow_model, lower_order_final=lower_order_final)
                     if method == "condot_2m" else cls(flow_model))

    @property
    def stats(self) -> SolverStats:
        return self.impl.stats

    def sample(self, x_init, times):
        return self.impl.sample(x_init, times)


def solve(velocity_model: VelocityFn, x_init, times, method: str = "euler",
          lower_order_final: bool = True, prediction_type: str = "velocity",
          endpoint_eps: float = 1e-3):
    solver = CondOTSolver(velocity_model, method, lower_order_final, prediction_type, endpoint_eps)
    return solver.sample(x_init, times), solver.stats.nfe


__all__ = ["TimeSchedule", "EulerSolver", "MidpointSolver", "HeunSolver", "RK4Solver",
           "AdamsBashforth2Solver", "CondOT2MSolver", "CondOTLambdaTwoStepSolver", "CondOTSolver", "SOLVER_CLASSES", "solve"]

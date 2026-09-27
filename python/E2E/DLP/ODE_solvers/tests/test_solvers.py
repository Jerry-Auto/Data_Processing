import torch

from solvers.flow_matching_solvers import CondOTSolver
from solvers.condot_2m import CondOTLambdaTwoStepSolver
from schedules import TimeSchedule, CondOTParameterization
from solvers import VPExplicitSolver


def test_constant_velocity_is_exact_for_all_methods():
    x0 = torch.randn(8, 2)
    velocity = lambda x, t: torch.ones_like(x) * 2.5
    target = x0 + 2.5
    for method in CondOTSolver.valid_methods:
        result = CondOTSolver(velocity, method).sample(x0, TimeSchedule.uniform(5))
        torch.testing.assert_close(result, target, atol=2e-6, rtol=0)


def test_nfe_counts():
    x0 = torch.zeros(2, 2)
    velocity = lambda x, t: torch.ones_like(x)
    expected = {"euler": 4, "midpoint": 8, "heun": 8, "rk4": 16, "ab2": 4, "condot_2m": 4}
    for method, nfe in expected.items():
        solver = CondOTSolver(velocity, method)
        solver.sample(x0, TimeSchedule.uniform(4))
        assert solver.stats.nfe == nfe


def test_all_schedules_are_valid_time_grids():
    for name in ("uniform", "linear", "shifted", "karras", "log_snr"):
        times = TimeSchedule.make(name, 8)
        assert times[0] == 0 and times[-1] == 1
        assert bool(torch.all(times[1:] > times[:-1]))


def test_condot_2m_full_order_endpoint_is_finite():
    solver = CondOTSolver(lambda x, t: torch.ones_like(x), "condot_2m", lower_order_final=False)
    output = solver.sample(torch.zeros(2, 2), TimeSchedule.uniform(4))
    assert bool(torch.isfinite(output).all())


def test_condot_lambda_two_step_constant_velocity():
    x0 = torch.zeros(2, 2)
    for variant in ("exact_linear", "dpmpp"):
        solver = CondOTLambdaTwoStepSolver(lambda x, t: torch.ones_like(x) * 2., variant)
        output = solver.sample(x0, nfe=8)
        torch.testing.assert_close(output, torch.full_like(x0, 2.), atol=2e-5, rtol=0)
        assert solver.stats.nfe == 8


def test_variable_step_ab2_is_exact_for_affine_time_velocity():
    # Later AB2 updates integrate this affine velocity exactly. The first
    # Euler warm-up on [0,.1] has the known error -0.01, so the final result is
    # 1.99 rather than the exact integral 2. Deliberately use a nonuniform grid.
    times = torch.tensor([0., .1, .35, .8, 1.])
    solver = CondOTSolver(lambda x, t: torch.ones_like(x) * (1. + 2. * t), "ab2")
    output = solver.sample(torch.zeros(3, 2), times)
    torch.testing.assert_close(output, torch.full_like(output, 1.99), atol=2e-6, rtol=0)


def test_condot_parameterizations_agree_away_from_singular_endpoints():
    # Choose v and construct its exactly equivalent D and score outputs.
    x = torch.tensor([[.3, -.7], [1.2, .4]])
    t = torch.tensor(.4)
    velocity = torch.tensor([[.5, -1.], [.2, .8]])
    data = x + (1. - t) * velocity
    score = (t * velocity - x) / (1. - t)
    velocity_from_data = CondOTParameterization(lambda _x, _t: data, "data")(x, t)
    velocity_from_score = CondOTParameterization(lambda _x, _t: score, "score")(x, t)
    torch.testing.assert_close(velocity_from_data, velocity, atol=2e-6, rtol=0)
    torch.testing.assert_close(velocity_from_score, velocity, atol=2e-6, rtol=0)



def test_vp_variable_step_ab2_constant_velocity():
    # A constant field is integrated exactly even when the VP grid is nonuniform.
    times = torch.tensor([1., .8, .35, .001])
    solver = VPExplicitSolver(lambda x, t: torch.ones_like(x), "ab2")
    output = solver.sample(torch.zeros(3, 2), times)
    torch.testing.assert_close(output, torch.full_like(output, times[-1] - times[0]), atol=2e-6, rtol=0)
    assert solver.nfe == len(times) - 1


if __name__ == "__main__":
    test_constant_velocity_is_exact_for_all_methods()
    test_nfe_counts()
    test_all_schedules_are_valid_time_grids()
    test_condot_2m_full_order_endpoint_is_finite()
    test_condot_lambda_two_step_constant_velocity()
    test_variable_step_ab2_is_exact_for_affine_time_velocity()
    test_vp_variable_step_ab2_constant_velocity()
    test_condot_parameterizations_agree_away_from_singular_endpoints()
    print("all solver checks passed")

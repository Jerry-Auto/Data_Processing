"""Benchmark solvers from a pre-trained small DiT checkpoint; never trains."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from solvers.flow_matching_solvers import CondOTSolver
from solvers.condot_2m import CondOTLambdaTwoStepSolver
from schedules import TimeSchedule, CondOTParameterization
from model import sample_data, ToyEncoderDecoderFlow, conditional_gaussian_metrics
from train.train import CHECKPOINT_FORMAT, PREVIOUS_CHECKPOINT_FORMAT, LEGACY_CHECKPOINT_FORMAT


def load_checkpoint(path: str, device: torch.device) -> tuple[ToyEncoderDecoderFlow, str]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    checkpoint_format = state.get("format")
    if checkpoint_format not in {CHECKPOINT_FORMAT, PREVIOUS_CHECKPOINT_FORMAT, LEGACY_CHECKPOINT_FORMAT}:
        raise ValueError("Unsupported checkpoint. Train a new DiT checkpoint with train.py.")
    model = ToyEncoderDecoderFlow(state["condition_dim"], state["hidden_dim"],
                                  state["depth"], state["num_heads"]).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.training_endpoint_eps = state.get("training_endpoint_eps", 1e-3)
    # v4 checkpoints were trained before the parameterization switch existed.
    prediction_type = state.get("prediction_type", "velocity")
    print(f"checkpoint loaded: {checkpoint_path.resolve()} (trained for {state['train_steps']} steps; "
          f"prediction={prediction_type})")
    return model, prediction_type


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_inference(model: ToyEncoderDecoderFlow, prediction_type: str, method: str, x_init: torch.Tensor,
                  condition: torch.Tensor, times: torch.Tensor,
                  data_velocity_eps: float) -> tuple[torch.Tensor, int, int, int]:
    """One complete inference: encoder once, decoder once at each solver NFE."""
    model.reset_inference_counters()
    encoded_scene = model.encode(condition)
    endpoint_eps = model.training_endpoint_eps if data_velocity_eps is None else data_velocity_eps
    velocity = CondOTParameterization(
        lambda x, t: model.decode(x, t.clamp(endpoint_eps, 1. - endpoint_eps), encoded_scene), prediction_type, endpoint_eps,
    )
    if method in {"condot_lambda_exact", "condot_lambda_dpmpp"}:
        variant = "exact_linear" if method == "condot_lambda_exact" else "dpmpp"
        solver = CondOTLambdaTwoStepSolver(
            velocity, variant=variant,
        )
        result = solver.sample(x_init, nfe=len(times) - 1)
        return result, solver.stats.nfe, model.encoder_calls, model.decoder_calls
    # Compare the production-safe Euler endpoint fallback with the analytic
    # second-order t=1 limit; both use the same number of decoder evaluations.
    base_method = "condot_2m" if method == "condot_2m_full_endpoint" else method
    solver = CondOTSolver(
        velocity, base_method,
        lower_order_final=(method != "condot_2m_full_endpoint"),
    )
    result = solver.sample(x_init, times)
    return result, solver.stats.nfe, model.encoder_calls, model.decoder_calls


@torch.no_grad()
def benchmark(model: ToyEncoderDecoderFlow, prediction_type: str, device: torch.device,
              integration_steps: int, repeats: int, batch_size: int, reference_steps: int,
              schedule_name: str, data_velocity_eps: float) -> None:
    model.eval()
    _, condition = sample_data(batch_size, device)
    labels = condition.argmax(dim=1)
    x_init = torch.randn(batch_size, 2, device=device)
    reference, _, _, _ = run_inference(
        model, prediction_type, "rk4", x_init, condition,
        TimeSchedule.make(schedule_name, reference_steps, device=device), data_velocity_eps,
    )
    if integration_steps < 2:
        raise ValueError("integration_steps must be an integer >= 2")
    print(f"Fixed-step benchmark: prediction={prediction_type}; every row below uses exactly "
          f"{integration_steps} integration intervals.")
    print(f"method       intervals NFE  enc/dec per sample  mean ms    MSE vs RK4({reference_steps})     Gaussian W2   mean/cov err")
    # This table fixes the number of intervals, not NFE.  Consequently the
    # measured decoder calls show each method's actual deployment cost:
    # Euler/AB2/2M use 1, Midpoint/Heun use 2, and classical RK4 uses 4 per interval.
    comparisons = ["euler", "midpoint", "heun", "rk4", "ab2", "condot_2m",
                   "condot_2m_full_endpoint", "condot_lambda_exact", "condot_lambda_dpmpp"]
    for method in comparisons:
        times = TimeSchedule.make(schedule_name, integration_steps, device=device)
        run_inference(model, prediction_type, method, x_init, condition, times, data_velocity_eps)
        synchronize(device)
        started = time.perf_counter()
        for _ in range(repeats):
            result, nfe, encoder_calls, decoder_calls = run_inference(
                model, prediction_type, method, x_init, condition, times, data_velocity_eps,
            )
        synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1e3 / repeats
        assert encoder_calls == 1 and decoder_calls == nfe
        mse = (result - reference).square().mean().item()
        quality = conditional_gaussian_metrics(result, labels)
        print(f"{method:25s} {integration_steps:5d} {nfe:4d} {encoder_calls:3d}/{decoder_calls:<3d}"
              f"          {elapsed_ms:7.3f} {mse:17.6e} {quality['w2']:13.6f}"
              f" {quality['mean_error']:.3f}/{quality['covariance_error']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="A checkpoint produced by train.py")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu", "cuda", "gpu"])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--steps", "--integration-steps", dest="integration_steps", type=int, default=30,
                        help="Number of integration intervals for every method in the comparison table.")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--reference-steps", type=int, default=512)
    parser.add_argument("--endpoint-eps", "--data-velocity-eps", dest="data_velocity_eps", type=float,
                        default=None, help="Default: checkpoint training_endpoint_eps; override only for an ablation.")
    parser.add_argument("--schedule", choices=["uniform", "linear", "shifted", "karras", "log_snr"],
                        default="uniform")
    args = parser.parse_args()
    device_name = "cuda" if args.device == "gpu" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda/gpu was requested, but this PyTorch environment has no usable CUDA GPU.")
    device = torch.device(device_name)
    torch.manual_seed(7)
    model, prediction_type = load_checkpoint(args.checkpoint, device)
    benchmark(model, prediction_type, device, args.integration_steps, args.repeats,
              args.batch_size, args.reference_steps, args.schedule, args.data_velocity_eps)


if __name__ == "__main__":
    main()

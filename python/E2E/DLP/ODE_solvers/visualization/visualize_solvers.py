"""Visualize samples from every solver for velocity, data, and score checkpoints.

This is a distribution-quality diagnostic, not a paired reconstruction test:
the initial Gaussian noise is independent of the target Gaussian-mixture draw.
All checkpoint/method panels receive the same initial noise and conditions.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from benchmark.benchmark import load_checkpoint, run_inference
from schedules import TimeSchedule
from model import CENTERS, conditional_gaussian_metrics


METHODS = (
    "euler", "midpoint", "heun", "rk4", "ab2", "condot_2m",
    "condot_2m_full_endpoint", "condot_lambda_exact", "condot_lambda_dpmpp",
)
METHOD_LABELS = {
    "euler": "Euler", "midpoint": "Explicit Midpoint", "heun": "Heun",
    "rk4": "Classical RK4", "ab2": "Adams-Bashforth 2",
    "condot_2m": "CondOT 2M", "condot_2m_full_endpoint": "CondOT 2M (full endpoint)",
    "condot_lambda_exact": "CondOT lambda 2M (exact linear)",
    "condot_lambda_dpmpp": "CondOT lambda 2M (DPM++ style)",
}
COLORS = ("#e15759", "#4e79a7", "#59a14f", "#b07aa1")


def balanced_conditions(num_samples: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return balanced four-class conditions and corresponding target-mixture draws."""
    num_classes = len(CENTERS)
    labels = torch.arange(num_samples, device=device) % num_classes
    permutation = torch.randperm(num_samples, device=device)
    labels = labels[permutation]
    centers = torch.tensor(CENTERS, device=device)
    target = centers[labels] + .25 * torch.randn(num_samples, 2, device=device)
    condition = F.one_hot(labels, num_classes=num_classes).float()
    return labels, condition, target


def scatter(ax, samples: torch.Tensor, labels: torch.Tensor, limit: float) -> None:
    values, labels = samples.detach().float().cpu(), labels.cpu()
    finite = torch.isfinite(values).all(dim=1)
    for class_id, color in enumerate(COLORS):
        selected = values[(labels == class_id) & finite]
        ax.scatter(selected[:, 0], selected[:, 1], s=3, alpha=.38, color=color, linewidths=0)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(())
    ax.set_yticks(())


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-checkpoint", default="checkpoints/toy_parameterization_velocity.pt")
    parser.add_argument("--data-checkpoint", default="checkpoints/toy_parameterization_data.pt")
    parser.add_argument("--score-checkpoint", default="checkpoints/toy_parameterization_score.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu", "cuda", "gpu"])
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--reference-steps", type=int, default=512)
    parser.add_argument("--schedule", choices=["uniform", "linear", "shifted", "karras", "log_snr"],
                        default="uniform")
    parser.add_argument("--endpoint-eps", type=float, default=None,
                        help="Default: each checkpoint's training_endpoint_eps.")
    parser.add_argument("--plot-limit", type=float, default=4.0)
    parser.add_argument("--output", default="outputs/solver_samples.png")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.samples < len(CENTERS):
        raise ValueError("--samples must be at least the number of mixture components")
    device_name = "cuda" if args.device == "gpu" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda/gpu was requested, but CUDA is unavailable")
    device = torch.device(device_name)
    torch.manual_seed(args.seed)
    labels, condition, target = balanced_conditions(args.samples, device)
    x_init = torch.randn(args.samples, 2, device=device)
    times = TimeSchedule.make(args.schedule, args.steps, device=device)
    reference_times = TimeSchedule.make(args.schedule, args.reference_steps, device=device)
    checkpoints = (
        ("velocity prediction", args.velocity_checkpoint),
        ("clean-data prediction", args.data_checkpoint),
        ("score prediction", args.score_checkpoint),
    )
    figure, axes = plt.subplots(len(checkpoints), len(METHODS) + 1,
                                figsize=(2.15 * (len(METHODS) + 1), 6.6), squeeze=False)
    for row, (row_name, checkpoint) in enumerate(checkpoints):
        model, prediction_type = load_checkpoint(checkpoint, device)
        model.eval()
        endpoint_eps = model.training_endpoint_eps if args.endpoint_eps is None else args.endpoint_eps
        reference, _, _, _ = run_inference(
            model, prediction_type, "rk4", x_init, condition, reference_times, endpoint_eps,
        )
        target_axis = axes[row, 0]
        scatter(target_axis, target, labels, args.plot_limit)
        target_axis.set_title("Target distribution", fontsize=8)
        target_axis.set_ylabel(row_name, fontsize=10)
        for col, method in enumerate(METHODS, start=1):
            samples, nfe, _, _ = run_inference(
                model, prediction_type, method, x_init, condition, times, endpoint_eps,
            )
            mse_reference = (samples - reference).square().mean().item()
            quality = conditional_gaussian_metrics(samples, labels)
            axis = axes[row, col]
            scatter(axis, samples, labels, args.plot_limit)
            title = (f"{METHOD_LABELS[method]}\nNFE={nfe}, ref={mse_reference:.1e}\n"
                     f"W2={quality['w2']:.2f}, std={quality['component_std']:.2f}")
            if quality['finite_fraction'] < 1.0:
                title += f", finite={quality['finite_fraction']:.2%}"
            axis.set_title(title, fontsize=7)
    figure.suptitle(
        f"CondOT solver samples: {args.steps} intervals, {args.schedule} schedule\n"
        "Colours denote the four conditions; target radial std is 0.25.", fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, .92))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    print(f"figure saved: {output.resolve()}")


if __name__ == "__main__":
    main()

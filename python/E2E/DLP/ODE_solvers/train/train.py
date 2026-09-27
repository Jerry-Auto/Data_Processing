"""Train and save the small encoder-once / DiT-decoder CondOT model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import sample_data, ToyEncoderDecoderFlow

CHECKPOINT_FORMAT = "toy-condot-dit-time-token-v6-ode-weighted"
PREVIOUS_CHECKPOINT_FORMAT = "toy-condot-dit-time-token-v5"
LEGACY_CHECKPOINT_FORMAT = "toy-condot-dit-time-token-v4"


def save_checkpoint(model: ToyEncoderDecoderFlow, path: str, train_steps: int,
                    prediction_type: str, endpoint_eps: float, weight_clip: float,
                    normalize_loss_weight: bool) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "model_state_dict": model.state_dict(),
        "hidden_dim": model.hidden_dim,
        "condition_dim": model.condition_dim,
        "depth": model.depth,
        "num_heads": model.num_heads,
        "train_steps": train_steps,
        "prediction_type": prediction_type,
        "loss_space": "velocity_equivalent",
        "training_endpoint_eps": endpoint_eps,
        "weight_clip": weight_clip,
        "normalize_loss_weight": normalize_loss_weight,
    }, checkpoint_path)
    print(f"checkpoint saved: {checkpoint_path.resolve()}")


def train(model: ToyEncoderDecoderFlow, steps: int, batch_size: int, device: torch.device,
          prediction_type: str, endpoint_eps: float, weight_clip: float,
          normalize_loss_weight: bool) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    for step in range(steps):
        data, condition = sample_data(batch_size, device)
        noise = torch.randn_like(data)
        t = endpoint_eps + (1. - 2. * endpoint_eps) * torch.rand(batch_size, 1, device=device)
        x_t = (1 - t) * noise + t * data
        # Every raw output is optimized in the common CondOT velocity space:
        # delta_v=delta_D/(1-t) and delta_v=(1-t)delta_score/t.
        if prediction_type == "velocity":
            target, weight = data - noise, 1.0
        elif prediction_type == "data":
            target, weight = data, (1. - t).reciprocal().square()
        else:
            target = -noise / (1. - t)
            weight = ((1. - t) / t).square()
        if not torch.is_tensor(weight):
            weight = torch.ones_like(t)
        weight = weight.clamp_max(weight_clip)
        if normalize_loss_weight:
            weight = weight / (weight.mean().detach() + 1e-8)
        loss = ((model(x_t, t, condition) - target).square() * weight).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % max(1, steps // 4) == 0:
            print(f"train {step + 1:5d}/{steps}: loss={loss.item():.4f}")


def checkpoint_for_prediction(base_path: str, prediction_type: str, train_all: bool) -> str:
    """Keep a single requested path for one run; derive stable names for all runs."""
    if not train_all:
        return base_path
    path = Path(base_path)
    suffix = path.suffix or ".pt"
    stem = path.stem if path.suffix else path.name
    return str(path.with_name(f"{stem}_{prediction_type}{suffix}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu", "cuda", "gpu"])
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--prediction-type", choices=["velocity", "data", "score", "all"], default="velocity",
                        help="Train one parameterization, or all three in one invocation.")
    parser.add_argument("--training-endpoint-eps", type=float, default=1e-2)
    parser.add_argument("--weight-clip", type=float, default=100.)
    parser.add_argument("--no-normalize-loss-weight", action="store_true")
    parser.add_argument("--checkpoint", default="checkpoints/toy_dit_time_token.pt")
    args = parser.parse_args()
    device_name = "cuda" if args.device == "gpu" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda/gpu was requested, but this PyTorch environment has no usable CUDA GPU.")
    if not 0. < args.training_endpoint_eps < .5:
        raise ValueError("--training-endpoint-eps must lie in (0, .5)")
    prediction_types = ("velocity", "data", "score") if args.prediction_type == "all" else (args.prediction_type,)
    for prediction_type in prediction_types:
        # Resetting the seed makes architecture initialization and the sequence
        # of sampled minibatches identical across the three objectives.
        torch.manual_seed(7)
        model = ToyEncoderDecoderFlow().to(device_name)
        print(f"\n=== training prediction_type={prediction_type} ===")
        train(model, args.train_steps, args.batch_size, torch.device(device_name), prediction_type,
              args.training_endpoint_eps, args.weight_clip, not args.no_normalize_loss_weight)
        checkpoint = checkpoint_for_prediction(args.checkpoint, prediction_type,
                                               args.prediction_type == "all")
        save_checkpoint(model, checkpoint, args.train_steps, prediction_type,
                        args.training_endpoint_eps, args.weight_clip,
                        not args.no_normalize_loss_weight)


if __name__ == "__main__":
    main()

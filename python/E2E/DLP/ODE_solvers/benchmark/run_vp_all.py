"""One-command VP-SDE experiment: train (when needed), benchmark, tables and figures.

All methods in a row use the same conditional batch and the same initial Gaussian
noise.  ``reference_mse`` therefore isolates numerical integration error for one
learned ODE, while ``mean_error`` and ``scale_error`` compare samples to the
known toy target distribution and can be compared between parameterizations.
"""
from __future__ import annotations
import argparse, csv, subprocess, sys, time
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from benchmark.benchmark_vp import load, run
from model import CENTERS, conditional_gaussian_metrics
from schedules import LinearVPSDE
from reports import save_table_image

KINDS = ('noise', 'x_start', 'pf_velocity', 'score')
METHODS = ('euler', 'midpoint', 'heun', 'rk4', 'ab2', 'dpmsolver_2m', 'dpmsolverpp_2m')
DISPLAY = {'euler':'Euler', 'midpoint':'Midpoint', 'heun':'Heun', 'rk4':'RK4',
           'ab2':'AB2', 'dpmsolver_2m':'DPM-Solver 2M', 'dpmsolverpp_2m':'DPM-Solver++ 2M'}


def checkpoint_paths(base: str):
    p = Path(base)
    return {k: p.with_name(f'{p.stem}_{k}{p.suffix or ".pt"}') for k in KINDS}


def make_batch(n, device):
    # Balanced modes prevent a random class imbalance from contaminating quality metrics.
    labels = (torch.arange(n, device=device) % 4)[torch.randperm(n, device=device)]
    cond = F.one_hot(labels, 4).float()
    target = torch.tensor(CENTERS, device=device)[labels] + .25 * torch.randn(n, 2, device=device)
    x_init = torch.randn(n, 2, device=device)
    return labels, cond, target, x_init


def sync(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


@torch.no_grad()
def collect(args, paths, out_dir):
    device = torch.device('cuda' if args.device == 'gpu' else args.device)
    torch.manual_seed(args.seed)
    labels, cond, _, x_init = make_batch(args.batch_size, device)
    sde = LinearVPSDE()
    times = sde.times(args.steps, device, eps=args.endpoint_eps, schedule=args.schedule)
    ref_times = sde.times(args.reference_steps, device, eps=args.endpoint_eps, schedule=args.schedule)
    rows = []
    for kind, path in paths.items():
        model, saved_kind = load(path, device)
        if kind != saved_kind: raise RuntimeError(f'{path}: expected {kind}, found {saved_kind}')
        model.eval()
        reference, _ = run(model, kind, 'rk4', x_init, cond, ref_times, sde)
        for method in METHODS:
            # Warm-up avoids reporting one-time CUDA allocator / kernel setup time.
            run(model, kind, method, x_init, cond, times, sde); sync(device)
            elapsed = []
            for _ in range(args.repeats):
                start = time.perf_counter(); sample, nfe = run(model, kind, method, x_init, cond, times, sde); sync(device)
                elapsed.append((time.perf_counter() - start) * 1e3)
            quality = conditional_gaussian_metrics(sample, labels)
            rows.append(dict(prediction=kind, method=method, intervals=args.steps,
                             nfe=nfe, latency_ms=sum(elapsed)/len(elapsed), latency_std_ms=torch.tensor(elapsed).std(unbiased=False).item(),
                             reference_mse=(sample-reference).square().mean().item(),
                             **quality))
            print(f'{kind:7s} {method:18s} NFE={nfe:3d}  {rows[-1]["latency_ms"]:8.3f} ms  ref={rows[-1]["reference_mse"]:.3e}')
    return rows


def write_tables(rows, path):
    fields = list(rows[0])
    with (path/'results.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    lines = ['# VP-SDE solver experiment', '',
             '| prediction | solver | intervals | NFE | latency (ms) | latency std (ms) | MSE vs RK4 reference | mean error | covariance error | Gaussian W2 | component std | finite |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["prediction"]} | {DISPLAY[r["method"]]} | {r["intervals"]} | {r["nfe"]} | {r["latency_ms"]:.3f} | {r["latency_std_ms"]:.3f} | {r["reference_mse"]:.3e} | {r["mean_error"]:.4f} | {r["covariance_error"]:.4f} | {r["w2"]:.4f} | {r["component_std"]:.4f} | {r["finite_fraction"]:.3f} |')
    (path/'results.md').write_text('\n'.join(lines)+'\n')


def plot_summary(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    symbols = {'noise':'o', 'x_start':'s', 'pf_velocity':'^', 'score':'D'}
    colors = plt.get_cmap('tab10').colors
    for i, method in enumerate(METHODS):
        subset = [r for r in rows if r['method'] == method]
        axes[0].scatter([r['latency_ms'] for r in subset], [r['reference_mse'] for r in subset], label=DISPLAY[method], color=colors[i], s=45)
    for r in rows:
        axes[0].scatter(r['latency_ms'], r['reference_mse'], marker=symbols[r['prediction']], color=colors[METHODS.index(r['method'])], s=45)
    axes[0].set(xlabel='Mean inference latency per batch (ms)', ylabel='MSE vs own RK4 reference', yscale='log', title='Numerical error / latency trade-off')
    axes[0].grid(alpha=.25); axes[0].legend(fontsize=7, ncol=2)
    for i, kind in enumerate(KINDS):
        subset = [r for r in rows if r['prediction'] == kind]
        xs = torch.arange(len(METHODS)).numpy() + (i-1.5)*.19
        axes[1].bar(xs, [r['w2'] for r in subset], width=.19, label=kind)
    axes[1].set_xticks(range(len(METHODS)), [DISPLAY[m].replace(' ', '\n') for m in METHODS], fontsize=8)
    axes[1].set(ylabel='Gaussian Wasserstein-2 distance to true conditional target', title='Sample-distribution quality', yscale='log')
    axes[1].grid(axis='y', alpha=.25); axes[1].legend()
    fig.tight_layout(); fig.savefig(path/'summary.png', dpi=180); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description='Run the complete four-parameterization VP-SDE solver experiment.')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--checkpoint-base', default='checkpoints/toy_vpsde.pt')
    p.add_argument('--output-dir', default='outputs/vp_all')
    p.add_argument('--train-steps', type=int, default=3200); p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--steps', type=int, default=30); p.add_argument('--reference-steps', type=int, default=512); p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--schedule', choices=['uniform','logsnr'], default='logsnr'); p.add_argument('--endpoint-eps', type=float, default=1e-2, help='Match the new VP checkpoint training endpoint by default.'); p.add_argument('--seed', type=int, default=7)
    p.add_argument('--retrain', action='store_true', help='Train the four checkpoints even if they already exist.')
    a = p.parse_args(); root = Path(__file__).resolve().parents[1]; out = (root/a.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    paths = checkpoint_paths(a.checkpoint_base)
    if a.retrain or not all(x.exists() for x in paths.values()):
        print('training four VP-SDE parameterizations...')
        subprocess.run([sys.executable, str(root/'train'/'train_vp.py'), '--device', a.device, '--train-steps', str(a.train_steps), '--batch-size', str(a.batch_size), '--checkpoint', a.checkpoint_base], check=True)
    else: print('all four checkpoints found; skipping training (pass --retrain to train again).')
    rows = collect(a, paths, out); write_tables(rows, out); plot_summary(rows, out); save_table_image(rows, out/'table.png', 'VP-SDE: solver cost and quality', METHODS, DISPLAY)
    visual_cmd = [sys.executable, str(root/'visualization'/'visualize_vp_solvers.py'), '--device', a.device, '--noise-checkpoint', str(paths['noise']), '--x-start-checkpoint', str(paths['x_start']), '--pf-velocity-checkpoint', str(paths['pf_velocity']), '--score-checkpoint', str(paths['score']), '--samples', str(a.batch_size), '--steps', str(a.steps), '--reference-steps', str(a.reference_steps), '--schedule', a.schedule, '--endpoint-eps', str(a.endpoint_eps), '--output', str(out/'samples.png')]
    subprocess.run(visual_cmd, check=True)
    print(f'\nComplete. Results: {out}\n  results.csv / results.md / table.png: detailed statistics\n  summary.png: latency-error and quality plots\n  samples.png: scatter samples for every model and solver')

if __name__ == '__main__': main()

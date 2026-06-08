"""
推理速度对比可视化

从 benchmark_results.json 读取结果，生成对比图表。

用法:
  python visualize.py
  python visualize.py --input benchmark_results.json --output benchmark_result.png
"""
import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_results(path):
    with open(path) as f:
        return json.load(f)


def plot_group(ax, data, title, ylabel):
    """为一个组（CPU 或 GPU）绘制柱状图"""
    if not data:
        ax.set_visible(False)
        return

    batch_sizes = [d["batch_size"] for d in data]
    # 收集所有方法名（按第一次出现的顺序）
    method_names = []
    seen = set()
    for d in data:
        for name in d["results"]:
            if name not in seen:
                method_names.append(name)
                seen.add(name)

    x = np.arange(len(batch_sizes))
    n = len(method_names)
    width = 0.8 / max(n, 1)

    # 颜色调色板
    palette = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52",
        "#8172B2", "#937860", "#4E8FBA", "#D4A04A",
    ]
    hatches = ["", "", "", "", "", "", "", ""]

    for i, name in enumerate(method_names):
        values = []
        for d in data:
            if name in d["results"]:
                values.append(d["results"][name]["mean"])
            else:
                values.append(0)
        offset = i * width - 0.4 + width / 2
        bars = ax.bar(
            x + offset, values, width,
            label=name,
            color=palette[i % len(palette)],
            edgecolor="white",
            linewidth=0.5,
        )
        # 在柱子顶部标注数值
        for bar, val in zip(bars, values):
            if val > 0:
                ax.annotate(
                    f"{val:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=6, rotation=45,
                )

    ax.set_xlabel("Batch Size")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)


def plot_speedup(ax, data, title):
    """绘制加速比图"""
    if not data:
        ax.set_visible(False)
        return

    batch_sizes = [d["batch_size"] for d in data]
    method_names = []
    seen = set()
    for d in data:
        for name in d["results"]:
            if name not in seen:
                method_names.append(name)
                seen.add(name)

    x = np.arange(len(batch_sizes))
    n = len(method_names)
    width = 0.8 / max(n, 1)

    palette = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52",
        "#8172B2", "#937860", "#4E8FBA", "#D4A04A",
    ]

    # 以第一个方法为基准
    baseline_name = method_names[0]

    for i, name in enumerate(method_names):
        speedups = []
        for d in data:
            base_val = d["results"].get(baseline_name, {}).get("mean", 0)
            cur_val = d["results"].get(name, {}).get("mean", 0)
            if cur_val > 0 and base_val > 0:
                speedups.append(base_val / cur_val)
            else:
                speedups.append(0)
        offset = i * width - 0.4 + width / 2
        bars = ax.bar(
            x + offset, speedups, width,
            label=name,
            color=palette[i % len(palette)],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, speedups):
            if val > 0:
                ax.annotate(
                    f"{val:.2f}x",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=6, rotation=45,
                )

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Baseline (1.0x)")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Speedup (vs first method)")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="推理速度对比可视化")
    parser.add_argument(
        "--input", default=os.path.join(ROOT, "benchmark_results.json"),
        help="benchmark 结果 JSON 文件路径"
    )
    parser.add_argument(
        "--output", default=os.path.join(ROOT, "benchmark_result.png"),
        help="输出图片路径"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"结果文件不存在: {args.input}")
        print("请先运行: python benchmark.py")
        return

    data = load_results(args.input)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 上排：延迟柱状图
    plot_group(axes[0, 0], data["cpu"], "CPU Inference Latency (lower is better)", "Latency (ms)")
    plot_group(axes[0, 1], data["gpu"], "GPU Inference Latency (lower is better)", "Latency (ms)")

    # 下排：加速比柱状图
    plot_speedup(axes[1, 0], data["cpu"], "CPU Speedup Ratio (higher is better)")
    plot_speedup(axes[1, 1], data["gpu"], "GPU Speedup Ratio (higher is better)")

    fig.suptitle("Inference Benchmark: PyTorch vs ONNX Runtime vs TensorRT\n(Python + C++)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"图表已保存: {args.output}")


if __name__ == "__main__":
    main()

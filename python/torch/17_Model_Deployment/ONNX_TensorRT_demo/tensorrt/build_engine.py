"""
TensorRT Engine 构建脚本
使用 trtexec 命令行工具从 ONNX 构建 TensorRT Engine

前置条件:
  1. 已安装 TensorRT（含 trtexec 工具）
  2. 已导出 ONNX 模型（先运行 onnx/export_onnx.py）

安装 TensorRT:
  pip install tensorrt pycuda
  或从 https://developer.nvidia.com/tensorrt 下载
"""
import os
import subprocess
import sys


def build_engine(
    onnx_path="onnx/model.onnx",
    engine_path="tensorrt/model.engine",
    precision="fp32",
    workspace=1024,
):
    """
    用 trtexec 构建 TensorRT Engine

    Args:
        onnx_path: 输入的 ONNX 模型路径
        engine_path: 输出的 Engine 文件路径
        precision: 精度，可选 "fp32" 或 "fp16"
        workspace: 最大工作显存 (MiB)
    """
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace}",
    ]

    if precision == "fp16":
        cmd.append("--fp16")

    print(f"构建命令: {' '.join(cmd)}")
    print("构建中（可能需要几十秒）...")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"构建失败:\n{result.stderr}")
        sys.exit(1)

    print(f"✓ Engine 已保存到: {engine_path}")
    # 打印性能摘要
    for line in result.stdout.splitlines():
        if "throughput" in line.lower() or "latency" in line.lower():
            print(f"  {line.strip()}")


def benchmark_engine(engine_path="tensorrt/model.engine", batch_size=1, iterations=100):
    """用 trtexec 对已有 Engine 做性能测试"""
    cmd = [
        "trtexec",
        f"--loadEngine={engine_path}",
        f"--shapes=input:{batch_size}x4",
        f"--iterations={iterations}",
    ]
    print(f"测速命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(__file__), "..")
    onnx_path = os.path.join(root, "onnx", "model.onnx")

    if not os.path.exists(onnx_path):
        print(f"ONNX 模型不存在: {onnx_path}")
        print("请先运行: python onnx/export_onnx.py")
        sys.exit(1)

    # 构建 FP32 Engine
    engine_fp32 = os.path.join(root, "tensorrt", "model.engine")
    build_engine(onnx_path, engine_fp32, precision="fp32")

    # 构建 FP16 Engine（可选）
    engine_fp16 = os.path.join(root, "tensorrt", "model_fp16.engine")
    build_engine(onnx_path, engine_fp16, precision="fp16")

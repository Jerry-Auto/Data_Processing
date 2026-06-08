"""
TensorRT Engine 构建脚本

支持两种方式：
  1. trtexec 命令行工具（优先）
  2. TensorRT Python API（备选）

会自动检测 trtexec 位置（系统 PATH / 项目 TensorRT/bin/），
如都不可用则回退到 Python API 构建。

前置条件:
  pip install tensorrt torch
  已导出 ONNX 模型（先运行 onnx/export_onnx.py）
"""
import os
import shutil
import subprocess
import sys


def find_trtexec():
    """查找 trtexec 可执行文件"""
    # 1. 系统 PATH
    path = shutil.which("trtexec")
    if path:
        return path

    # 2. 项目目录下的 third_party/TensorRT/bin/
    root = os.path.join(os.path.dirname(__file__), "..")
    project_trtexec = os.path.join(root, "third_party", "TensorRT", "bin", "trtexec")
    if os.path.isfile(project_trtexec) and os.access(project_trtexec, os.X_OK):
        return project_trtexec

    return None


def build_engine_trtexec(
    onnx_path="onnx/model.onnx",
    engine_path="tensorrt/model.engine",
    precision="fp32",
    trtexec_path=None,
):
    """
    用 trtexec 构建 TensorRT Engine

    Args:
        onnx_path: 输入的 ONNX 模型路径
        engine_path: 输出的 Engine 文件路径
        precision: 精度，可选 "fp32" 或 "fp16"
        trtexec_path: trtexec 可执行文件路径（None 则自动查找）
    """
    if trtexec_path is None:
        trtexec_path = find_trtexec()
    if trtexec_path is None:
        return False

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)

    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--minShapes=input:1x4",
        "--optShapes=input:8x4",
        "--maxShapes=input:64x4",
    ]

    if precision == "fp16":
        cmd.append("--fp16")

    print(f"构建命令: {' '.join(cmd)}")
    print("构建中（可能需要几十秒）...")

    env = os.environ.copy()
    # 确保 TensorRT 动态库可找到
    trt_lib = os.path.join(os.path.dirname(trtexec_path), "..", "lib")
    if os.path.isdir(trt_lib):
        env["LD_LIBRARY_PATH"] = trt_lib + ":" + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"构建失败:\n{result.stderr}")
        return False

    print(f"✓ Engine 已保存到: {engine_path}")
    # 打印性能摘要
    for line in result.stdout.splitlines():
        if "throughput" in line.lower() or "latency" in line.lower():
            print(f"  {line.strip()}")
    return True


def build_engine_python_api(
    onnx_path="onnx/model.onnx",
    engine_path="tensorrt/model.engine",
    precision="fp32",
):
    """
    用 TensorRT Python API 构建 Engine（无需 trtexec）

    Args:
        onnx_path: 输入的 ONNX 模型路径
        engine_path: 输出的 Engine 文件路径
        precision: 精度，可选 "fp32" 或 "fp16"
    """
    import tensorrt as trt

    major = int(trt.__version__.split('.')[0])
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, logger)

    print(f"解析 ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX 解析错误: {parser.get_error(i)}")
            return False
    print(f"  ONNX 解析成功")

    config = builder.create_builder_config()

    if major >= 11:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256)  # MiB
    else:
        config.max_workspace_size = 256 * 1024 * 1024  # 256 MiB

    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  启用 FP16 模式")
        else:
            print("  警告: 平台不支持 FP16，回退到 FP32")

    # 创建优化 profile（支持动态 batch size）
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, (1, 4), (8, 4), (64, 4))
    config.add_optimization_profile(profile)

    print(f"  正在构建 Engine（{precision}）...")

    if major >= 11:
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            print("  Engine 构建失败!")
            return False
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
        print(f"  ✓ Engine 已保存: {engine_path} ({len(engine_bytes) / 1024:.1f} KB)")
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            print("  Engine 构建失败!")
            return False
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        print(f"  ✓ Engine 已保存: {engine_path}")

    return True


def build_engine(
    onnx_path="onnx/model.onnx",
    engine_path="tensorrt/model.engine",
    precision="fp32",
):
    """
    构建 TensorRT Engine（自动选择方式）

    优先使用 trtexec，不可用则回退到 Python API。
    """
    # 尝试 trtexec
    trtexec_path = find_trtexec()
    if trtexec_path:
        print(f"使用 trtexec: {trtexec_path}")
        if build_engine_trtexec(onnx_path, engine_path, precision, trtexec_path):
            return True
        print("trtexec 构建失败，尝试 Python API...")

    # 回退到 Python API
    print("使用 TensorRT Python API 构建")
    return build_engine_python_api(onnx_path, engine_path, precision)


def benchmark_engine(engine_path="tensorrt/model.engine", batch_size=1, iterations=100):
    """用 trtexec 对已有 Engine 做性能测试"""
    trtexec_path = find_trtexec()
    if trtexec_path is None:
        print("trtexec 不可用，无法运行 benchmark_engine")
        print("可运行 python benchmark.py 做更完整的性能测试")
        return

    cmd = [
        trtexec_path,
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

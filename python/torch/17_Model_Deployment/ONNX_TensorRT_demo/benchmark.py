"""
推理速度对比基准测试

分两大类对比：
  CPU: PyTorch CPU / PyTorch C++ (LibTorch) / ONNX Runtime Python CPU / ONNX Runtime C++ CPU
  GPU: PyTorch GPU / PyTorch C++ (LibTorch) / ONNX Runtime Python GPU / TensorRT FP32/FP16 Python+C++

用法:
  cd ONNX_TensorRT_demo
  python benchmark.py

依赖:
  pip install torch onnx onnxruntime tensorrt
"""
import os
import sys
import json
import time
import ctypes
import subprocess
import warnings
import numpy as np

# ======================== 自动检测 cuDNN 并预加载 ========================
# onnxruntime-gpu 1.18+ 需要 cuDNN 9，可能不在默认 LD_LIBRARY_PATH 中
# 通过 ctypes 预加载 cuDNN 9 使其符号全局可用（必须在 import onnxruntime 之前）
def _setup_cudnn():
    """自动检测并预加载 cuDNN 9（在 import onnxruntime 之前调用）"""
    search_dirs = []

    # 1. conda 环境中 NVIDIA pip 包自带的 cuDNN
    # 使用 sys.prefix 而非 CONDA_PREFIX（后者仅在 conda activate 后才有值）
    conda_prefix = os.environ.get("CONDA_PREFIX", "") or sys.prefix
    if conda_prefix:
        for py_ver in ["python3.9", "python3.10", "python3.11", "python3.12"]:
            nvidia_dir = os.path.join(conda_prefix, "lib", py_ver,
                                      "site-packages", "nvidia")
            if os.path.isdir(nvidia_dir):
                cudnn_dir = os.path.join(nvidia_dir, "cudnn", "lib")
                if os.path.isdir(cudnn_dir):
                    search_dirs.append(cudnn_dir)

    # 2. 也设置 LD_LIBRARY_PATH（供子进程使用）
    extra_paths = []
    ort_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "onnxruntime", "lib")
    if os.path.isdir(ort_lib):
        extra_paths.append(ort_lib)
    trt_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "TensorRT", "lib")
    if os.path.isdir(trt_lib):
        extra_paths.append(trt_lib)
    extra_paths.extend(search_dirs)
    if extra_paths:
        current = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(extra_paths) + (":" + current if current else "")

    # 3. 预加载 cuDNN 9 共享库
    cudnn_libs = [
        "libcudnn.so.9",
        "libcudnn_ops.so.9",
        "libcudnn_cnn.so.9",
        "libcudnn_adv.so.9",
        "libcudnn_graph.so.9",
        "libcudnn_engines_runtime_compiled.so.9",
        "libcudnn_engines_precompiled.so.9",
        "libcudnn_heuristic.so.9",
    ]
    for d in search_dirs:
        loaded = False
        for lib in cudnn_libs:
            path = os.path.join(d, lib)
            if os.path.exists(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    loaded = True
                except OSError:
                    pass
        if loaded:
            break

_setup_cudnn()

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

import torch
from model.model import get_model

# ======================== 配置 ========================
ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(ROOT, "model", "weights.pth")
ONNX_PATH = os.path.join(ROOT, "onnx", "model.onnx")
ENGINE_FP32_PATH = os.path.join(ROOT, "tensorrt", "model.engine")
ENGINE_FP16_PATH = os.path.join(ROOT, "tensorrt", "model_fp16.engine")

ONNX_BENCH_EXE = os.path.join(ROOT, "onnx", "cpp", "build", "onnx_bench")
TRT_BENCH_EXE = os.path.join(ROOT, "tensorrt", "cpp", "build", "trt_bench")
LIBTORCH_BENCH_EXE = os.path.join(ROOT, "libtorch", "cpp", "build", "libtorch_bench")
TORCHSCRIPT_PATH = os.path.join(ROOT, "model", "model.pt")

WARMUP_ITERS = 20
BENCHMARK_ITERS = 200
BATCH_SIZES = [1, 4, 8, 16, 32]


# ======================== TensorRT Engine 构建 ========================

def get_trt_major_version():
    import tensorrt as trt
    return int(getattr(trt, '__version__', '0.0.0').split('.')[0])


def build_engine(onnx_path, engine_path, precision="fp32"):
    """根据 TRT 版本自动选择构建方式"""
    import tensorrt as trt

    major = get_trt_major_version()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX 解析错误: {parser.get_error(i)}")
            return False
    print(f"  ONNX 解析成功")

    config = builder.create_builder_config()

    if major >= 11:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256)
    else:
        config.max_workspace_size = 256 * 1024 * 1024

    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print(f"  启用 FP16 模式")

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


def ensure_engines():
    if not os.path.exists(ONNX_PATH):
        return False, False
    fp32_ok = os.path.exists(ENGINE_FP32_PATH)
    fp16_ok = os.path.exists(ENGINE_FP16_PATH)
    if not fp32_ok:
        print("\n自动构建 TensorRT FP32 Engine...")
        fp32_ok = build_engine(ONNX_PATH, ENGINE_FP32_PATH, "fp32")
    if not fp16_ok:
        print("\n自动构建 TensorRT FP16 Engine...")
        fp16_ok = build_engine(ONNX_PATH, ENGINE_FP16_PATH, "fp16")
    return fp32_ok, fp16_ok


# ======================== 工具函数 ========================

def time_func(func, warmup=WARMUP_ITERS, iters=BENCHMARK_ITERS):
    """对 func 做多次调用，返回耗时统计（毫秒）"""
    for _ in range(warmup):
        func()

    use_cuda_sync = torch.cuda.is_available()
    times = []
    for _ in range(iters):
        if use_cuda_sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        func()
        if use_cuda_sync:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = np.array(times)
    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "median": float(np.median(times)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
    }


def run_cpp_bench(exe_path, args, timeout=60):
    """运行 C++ benchmark 程序，解析 JSON 输出"""
    cmd = [exe_path] + [str(a) for a in args]
    env = os.environ.copy()
    # 收集所有可能的库路径
    lib_paths = [
        os.path.join(ROOT, "third_party", "onnxruntime", "lib"),
        os.path.join(ROOT, "third_party", "TensorRT", "lib"),
        os.path.join(sys.prefix, "lib", "python3.9", "site-packages", "torch", "lib"),
    ]
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(p for p in lib_paths if os.path.isdir(p)) + \
                              (":" + existing if existing else "")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env
        )
        if result.returncode != 0:
            print(f"  C++ 运行失败: {result.stderr.strip().splitlines()[-1] if result.stderr else 'unknown'}")
            return None
        # 解析 stdout 的 JSON
        return json.loads(result.stdout.strip())
    except Exception as e:
        print(f"  C++ 运行异常: {e}")
        return None


# ======================== Python 推理封装 ========================

def make_pytorch_bench(batch_size, device="cpu"):
    model = get_model(WEIGHTS_PATH).to(device)
    data = torch.randn(batch_size, 4, device=device)

    @torch.no_grad()
    def run():
        model(data)
    return run


def make_onnx_cpu_bench(batch_size):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 2
    session = ort.InferenceSession(
        ONNX_PATH, sess_options=opts, providers=["CPUExecutionProvider"]
    )
    data = np.random.randn(batch_size, 4).astype(np.float32)

    def run():
        session.run(None, {"input": data})
    return run


def make_onnx_gpu_bench(batch_size):
    import onnxruntime as ort
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        return None
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            session = ort.InferenceSession(
                ONNX_PATH, sess_options=opts, providers=["CUDAExecutionProvider"]
            )
        if "CUDAExecutionProvider" not in session.get_providers():
            return None
    except Exception:
        return None
    data = np.random.randn(batch_size, 4).astype(np.float32)

    def run():
        session.run(None, {"input": data})
    return run


def make_tensorrt_bench(batch_size, engine_path):
    import tensorrt as trt
    major = get_trt_major_version()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    if major >= 11:
        # TRT 11 API
        input_name = output_name = None
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(i) == trt.TensorIOMode.INPUT:
                input_name = name
            else:
                output_name = name
        input_np = np.random.randn(batch_size, 4).astype(np.float32)
        context.set_input_shape(input_name, input_np.shape)
        output_shape = context.get_tensor_shape(output_name)
    else:
        # TRT 8.x API
        input_name = output_name = None
        input_idx = output_idx = -1
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            if engine.binding_is_input(i):
                input_name = name
                input_idx = i
            else:
                output_name = name
                output_idx = i
        context.active_optimization_profile = 0
        context.set_binding_shape(input_idx, (batch_size, 4))
        output_shape = context.get_binding_shape(output_idx)
        input_np = np.random.randn(batch_size, 4).astype(np.float32)

    d_input = torch.empty(batch_size, 4, dtype=torch.float32, device="cuda")
    d_output = torch.empty(*output_shape, dtype=torch.float32, device="cuda")

    # 预拷贝输入数据到 GPU（不在计时范围内）
    input_tensor = torch.from_numpy(input_np).cuda()
    d_input.copy_(input_tensor)
    torch.cuda.synchronize()

    if major >= 11:
        context.set_tensor_address(input_name, d_input.data_ptr())
        context.set_tensor_address(output_name, d_output.data_ptr())
        stream = torch.cuda.Stream()

        def run():
            context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
    else:
        bindings = [None] * engine.num_bindings
        bindings[input_idx] = d_input.data_ptr()
        bindings[output_idx] = d_output.data_ptr()
        stream = torch.cuda.current_stream()

        def run():
            context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)
            stream.synchronize()

    return run


# ======================== 结果输出 ========================

def print_header():
    print("=" * 105)
    print("推理速度对比基准测试".center(95))
    print("=" * 105)
    print(f"预热: {WARMUP_ITERS} 次    测试: {BENCHMARK_ITERS} 次")
    print(f"PyTorch: {torch.__version__}    CUDA: {torch.cuda.is_available()}", end="")
    if torch.cuda.is_available():
        print(f"  ({torch.cuda.get_device_name(0)})")
    else:
        print()
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime: {ort.__version__}    Providers: {ort.get_available_providers()}")
    except ImportError:
        print("ONNX Runtime: 未安装")
    try:
        import tensorrt as trt
        print(f"TensorRT: {trt.__version__}")
    except ImportError:
        print("TensorRT: 未安装")

    # C++ 可执行文件检测
    onnx_cpp_ok = os.path.isfile(ONNX_BENCH_EXE)
    trt_cpp_ok = os.path.isfile(TRT_BENCH_EXE)
    print(f"C++ ONNX Bench: {'✓' if onnx_cpp_ok else '✗ (需编译: cd onnx/cpp/build && cmake .. && make)'}")
    print(f"C++ TRT  Bench: {'✓' if trt_cpp_ok else '✗ (需编译: cd tensorrt/cpp/build && cmake .. && make)'}")
    libtorch_ok = os.path.isfile(LIBTORCH_BENCH_EXE)
    print(f"C++ LibTorch:   {'✓' if libtorch_ok else '✗ (需编译: cd libtorch/cpp/build && cmake .. && make)'}")

    # trtexec 检测
    trtexec_path = os.path.join(ROOT, "TensorRT", "bin", "trtexec")
    if os.path.isfile(trtexec_path) and os.access(trtexec_path, os.X_OK):
        print(f"trtexec: ✓ ({trtexec_path})")
    else:
        import shutil
        which = shutil.which("trtexec")
        if which:
            print(f"trtexec: ✓ ({which})")
        else:
            print(f"trtexec: ✗ (未找到，Engine 构建将使用 Python API)")

    print("=" * 105)


def print_table(title, results, batch_size):
    """打印单个 batch_size 的结果表"""
    print(f"\n{'─' * 105}")
    print(f"  {title}  |  Batch Size = {batch_size}")
    print(f"{'─' * 105}")

    header = f"{'推理方式':<32} {'平均(ms)':>10} {'中位(ms)':>10} {'标准差':>10} {'最小':>10} {'最大':>10} {'P95':>10} {'P99':>10}"
    print(header)
    print("─" * 105)

    for name, stats in results:
        print(
            f"{name:<32} "
            f"{stats['mean']:>10.4f} "
            f"{stats['median']:>10.4f} "
            f"{stats['std']:>10.4f} "
            f"{stats['min']:>10.4f} "
            f"{stats['max']:>10.4f} "
            f"{stats['p95']:>10.4f} "
            f"{stats['p99']:>10.4f}"
        )

    if len(results) >= 2:
        baseline = results[0][1]["mean"]
        print(f"\n  {'加速比（以首行为基准）:':<32}", end="")
        for name, stats in results:
            speedup = baseline / stats["mean"] if stats["mean"] > 0 else 0
            print(f" {speedup:>9.2f}x", end="")
        print()


def print_summary(all_cpu, all_gpu):
    """打印汇总对比表"""
    print("\n\n" + "=" * 105)
    print("汇总".center(95))
    print("=" * 105)

    for title, data in [("CPU 推理对比", all_cpu), ("GPU 推理对比", all_gpu)]:
        if not data:
            continue
        method_names = [name for name, _ in data[0][1]]
        print(f"\n  ── {title} ──\n")

        header = f"{'Batch':>8}"
        for name in method_names:
            header += f" {name:>22}"
        print(header)
        print("─" * (8 + 23 * len(method_names)))

        for bs, results in data:
            row = f"{bs:>8}"
            for name, stats in results:
                row += f" {stats['mean']:>21.4f}"
            print(row)

        # 加速比
        print(f"\n  加速比（以首行为基准）:")
        print("─" * (8 + 23 * len(method_names)))
        header = f"{'Batch':>8}"
        for name in method_names:
            header += f" {name:>22}"
        print(header)
        print("─" * (8 + 23 * len(method_names)))

        for bs, results in data:
            baseline = results[0][1]["mean"]
            row = f"{bs:>8}"
            for name, stats in results:
                speedup = baseline / stats["mean"] if stats["mean"] > 0 else 0
                row += f" {speedup:>21.2f}x"
            print(row)


# ======================== 主流程 ========================

def main():
    print_header()

    # 检查模型文件
    if not os.path.exists(WEIGHTS_PATH):
        from model.model import save_weights
        save_weights(WEIGHTS_PATH)
    if not os.path.exists(ONNX_PATH):
        print("\nONNX 模型不存在，正在自动导出...")
        from onnx.export_onnx import export_onnx
        export_onnx(WEIGHTS_PATH, ONNX_PATH)

    # 检测 TensorRT
    has_tensorrt = False
    try:
        import tensorrt  # noqa: F401
        if hasattr(tensorrt, 'Builder'):
            has_tensorrt = True
    except ImportError:
        pass

    has_trt_fp32 = False
    has_trt_fp16 = False
    if has_tensorrt and torch.cuda.is_available():
        has_trt_fp32, has_trt_fp16 = ensure_engines()

    # 检测 C++ 可执行文件
    has_onnx_cpp = os.path.isfile(ONNX_BENCH_EXE)
    has_trt_cpp = os.path.isfile(TRT_BENCH_EXE)
    has_libtorch = os.path.isfile(LIBTORCH_BENCH_EXE)

    # 检测 ONNX Runtime GPU provider
    has_ort_gpu = False
    try:
        import onnxruntime as ort
        has_ort_gpu = "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        pass

    all_cpu_results = []
    all_gpu_results = []

    for bs in BATCH_SIZES:
        # ==================== CPU 推理对比 ====================
        cpu_results = []

        # PyTorch CPU
        try:
            bench = make_pytorch_bench(bs, device="cpu")
            stats = time_func(bench)
            cpu_results.append(("PyTorch CPU (Python)", stats))
        except Exception as e:
            print(f"  PyTorch CPU 失败: {e}")

        # ONNX Runtime CPU (Python)
        try:
            bench = make_onnx_cpu_bench(bs)
            stats = time_func(bench)
            cpu_results.append(("ONNX Runtime CPU (Python)", stats))
        except Exception as e:
            print(f"  ONNX Runtime CPU (Python) 失败: {e}")

        # ONNX Runtime CPU (C++)
        if has_onnx_cpp:
            stats = run_cpp_bench(ONNX_BENCH_EXE, [ONNX_PATH, bs, BENCHMARK_ITERS, WARMUP_ITERS, "cpu"])
            if stats:
                cpu_results.append(("ONNX Runtime CPU (C++)", stats))

        # LibTorch CPU (C++)
        if has_libtorch:
            stats = run_cpp_bench(LIBTORCH_BENCH_EXE, [TORCHSCRIPT_PATH, bs, BENCHMARK_ITERS, WARMUP_ITERS, "cpu"])
            if stats:
                cpu_results.append(("PyTorch CPU (C++)", stats))

        if cpu_results:
            print_table("CPU 推理对比", cpu_results, bs)
            all_cpu_results.append((bs, cpu_results))

        # ==================== GPU 推理对比 ====================
        gpu_results = []

        if not torch.cuda.is_available():
            continue

        # PyTorch GPU
        try:
            bench = make_pytorch_bench(bs, device="cuda")
            stats = time_func(bench)
            gpu_results.append(("PyTorch GPU (Python)", stats))
        except Exception as e:
            print(f"  PyTorch GPU 失败: {e}")

        # LibTorch GPU (C++)
        if has_libtorch:
            stats = run_cpp_bench(LIBTORCH_BENCH_EXE, [TORCHSCRIPT_PATH, bs, BENCHMARK_ITERS, WARMUP_ITERS, "cuda"])
            if stats:
                gpu_results.append(("PyTorch GPU (C++)", stats))

        # ONNX Runtime GPU (Python)
        if has_ort_gpu:
            try:
                bench = make_onnx_gpu_bench(bs)
                if bench is not None:
                    stats = time_func(bench)
                    gpu_results.append(("ONNX Runtime GPU (Python)", stats))
            except Exception:
                pass

        # ONNX Runtime GPU (C++)
        if has_onnx_cpp:
            stats = run_cpp_bench(ONNX_BENCH_EXE, [ONNX_PATH, bs, BENCHMARK_ITERS, WARMUP_ITERS, "cuda"])
            if stats:
                gpu_results.append(("ONNX Runtime GPU (C++)", stats))

        # TensorRT FP32 (Python)
        if has_trt_fp32:
            try:
                bench = make_tensorrt_bench(bs, ENGINE_FP32_PATH)
                stats = time_func(bench)
                gpu_results.append(("TensorRT FP32 (Python)", stats))
            except Exception as e:
                print(f"  TensorRT FP32 (Python) 失败: {e}")

        # TensorRT FP16 (Python)
        if has_trt_fp16:
            try:
                bench = make_tensorrt_bench(bs, ENGINE_FP16_PATH)
                stats = time_func(bench)
                gpu_results.append(("TensorRT FP16 (Python)", stats))
            except Exception as e:
                print(f"  TensorRT FP16 (Python) 失败: {e}")

        # TensorRT FP32 (C++)
        if has_trt_cpp and has_trt_fp32:
            stats = run_cpp_bench(TRT_BENCH_EXE, [ENGINE_FP32_PATH, bs, BENCHMARK_ITERS, WARMUP_ITERS])
            if stats:
                gpu_results.append(("TensorRT FP32 (C++)", stats))

        # TensorRT FP16 (C++)
        if has_trt_cpp and has_trt_fp16:
            stats = run_cpp_bench(TRT_BENCH_EXE, [ENGINE_FP16_PATH, bs, BENCHMARK_ITERS, WARMUP_ITERS])
            if stats:
                gpu_results.append(("TensorRT FP16 (C++)", stats))

        if gpu_results:
            print_table("GPU 推理对比", gpu_results, bs)
            all_gpu_results.append((bs, gpu_results))

    if all_cpu_results or all_gpu_results:
        print_summary(all_cpu_results, all_gpu_results)

    # 保存结果到 JSON 供可视化使用
    output = {"cpu": [], "gpu": []}
    for bs, results in all_cpu_results:
        output["cpu"].append({"batch_size": bs, "results": {n: s for n, s in results}})
    for bs, results in all_gpu_results:
        output["gpu"].append({"batch_size": bs, "results": {n: s for n, s in results}})

    result_path = os.path.join(ROOT, "benchmark_results.json")
    with open(result_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {result_path}")
    print(f"运行 python visualize.py 生成可视化图表")

    print("\n" + "=" * 105)
    print("测试完成！")
    print("=" * 105)


if __name__ == "__main__":
    main()

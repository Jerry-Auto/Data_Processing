"""
TensorRT Python 推理示例（使用 torch 管理 CUDA 内存）

前置条件:
  pip install tensorrt torch
  已构建 Engine（先运行 tensorrt/build_engine.py 或 benchmark.py 会自动构建）
"""
import os
import warnings
import numpy as np

# 抑制 TRT 8.x 的 deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def get_trt_major_version():
    """获取 TensorRT 主版本号"""
    return int(trt.__version__.split('.')[0])


def load_engine(engine_path):
    """加载 TensorRT Engine"""
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    print(f"Engine 加载成功: {engine_path}")
    print(f"  输入输出数量: {engine.num_io_tensors if get_trt_major_version() >= 11 else engine.num_bindings}")
    return engine


def infer(engine, input_np):
    """
    单次推理（使用 torch 管理 GPU 内存）

    Args:
        engine: TensorRT engine 对象
        input_np: numpy 输入数据 (float32)

    Returns:
        numpy 输出数据
    """
    major = get_trt_major_version()
    context = engine.create_execution_context()

    if major >= 11:
        # TRT 11.x API
        input_name = output_name = None
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(i) == trt.TensorIOMode.INPUT:
                input_name = name
            else:
                output_name = name

        context.set_input_shape(input_name, input_np.shape)
        output_shape = context.get_tensor_shape(output_name)

        d_input = torch.empty(*input_np.shape, dtype=torch.float32, device="cuda")
        d_output = torch.empty(*output_shape, dtype=torch.float32, device="cuda")

        context.set_tensor_address(input_name, d_input.data_ptr())
        context.set_tensor_address(output_name, d_output.data_ptr())

        stream = torch.cuda.Stream()
        d_input.copy_(torch.from_numpy(input_np).cuda())
        context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        return d_output.cpu().numpy()

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
        context.set_binding_shape(input_idx, input_np.shape)
        output_shape = context.get_binding_shape(output_idx)

        d_input = torch.empty(*input_np.shape, dtype=torch.float32, device="cuda")
        d_output = torch.empty(*output_shape, dtype=torch.float32, device="cuda")

        bindings = [None] * engine.num_bindings
        bindings[input_idx] = d_input.data_ptr()
        bindings[output_idx] = d_output.data_ptr()

        stream = torch.cuda.current_stream()
        d_input.copy_(torch.from_numpy(input_np).cuda())
        context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)
        stream.synchronize()

        return d_output.cpu().numpy()


def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    engine_path = os.path.join(root, "tensorrt", "model.engine")

    if not os.path.exists(engine_path):
        print(f"Engine 不存在: {engine_path}")
        print("请先运行: python tensorrt/build_engine.py")
        return

    engine = load_engine(engine_path)

    # ========== 基本推理 ==========
    print("\n--- 基本推理 ---")
    input_data = np.array([[1.0, 2.0, 3.0, 4.0],
                           [0.5, 1.5, 2.5, 3.5]], dtype=np.float32)
    print(f"输入:\n{input_data}")

    output = infer(engine, input_data)
    print(f"输出:\n{output}")
    print(f"输出 shape: {output.shape}")

    # ========== 动态 batch 推理 ==========
    print("\n--- 动态 batch 推理 ---")
    for bs in [1, 4, 8]:
        data = np.random.randn(bs, 4).astype(np.float32)
        out = infer(engine, data)
        print(f"  batch={bs}: 输入 {data.shape} → 输出 {out.shape}")

    # ========== 多次推理 ==========
    print("\n--- 连续 5 次推理 ---")
    for i in range(5):
        data = np.random.randn(1, 4).astype(np.float32)
        out = infer(engine, data)
        print(f"  iter {i}: 输出 {out.squeeze()}")


if __name__ == "__main__":
    main()

"""
TensorRT Python 推理示例

前置条件:
  pip install tensorrt pycuda
  已构建 Engine（先运行 tensorrt/build_engine.py）
"""
import os
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  自动初始化 pycuda


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(engine_path):
    """加载 TensorRT Engine"""
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    print(f"Engine 加载成功: {engine_path}")
    print(f"  输入数量: {engine.num_io_tensors}")
    return engine


def infer(engine, input_np):
    """
    单次推理

    Args:
        engine: TensorRT engine 对象
        input_np: numpy 输入数据 (float32)

    Returns:
        numpy 输出数据
    """
    context = engine.create_execution_context()

    # 获取输入输出名称和索引
    input_name = None
    output_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(i)
        if mode == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_name = name

    # 设置动态 shape（如果 Engine 支持）
    context.set_input_shape(input_name, input_np.shape)

    # 获取输出形状
    output_shape = context.get_tensor_shape(output_name)
    output_size = int(np.prod(output_shape))

    # 分配 GPU 显存
    d_input = cuda.mem_alloc(input_np.nbytes)
    d_output = cuda.mem_alloc(output_size * 4)  # float32 = 4 bytes

    # 拷贝输入到 GPU
    cuda.memcpy_htod(d_input, np.ascontiguousarray(input_np))

    # 设置 tensor 地址
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    # 执行推理
    context.execute_async_v3(cuda.Stream().handle)

    # 拷贝输出回 CPU
    output_np = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output_np, d_output)

    return output_np


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

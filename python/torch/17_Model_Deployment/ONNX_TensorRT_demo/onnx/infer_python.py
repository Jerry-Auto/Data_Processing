"""
ONNX Runtime Python 推理示例
演示基本推理流程、性能优化配置、多次推理
"""
import os
import numpy as np
import onnxruntime as ort


def create_session(model_path="onnx/model.onnx"):
    """创建推理会话（带性能优化配置）"""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 2

    session = ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    return session


def basic_inference(session):
    """基本推理"""
    # 查看模型信息
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    print(f"输入名: {inp.name}, shape: {inp.shape}, dtype: {inp.type}")
    print(f"输出名: {out.name}, shape: {out.shape}, dtype: {out.type}")

    # 构造输入
    data = np.array([[1.0, 2.0, 3.0, 4.0],
                     [0.5, 1.5, 2.5, 3.5]], dtype=np.float32)
    print(f"\n输入数据:\n{data}")

    # 推理
    outputs = session.run(None, {"input": data})
    result = outputs[0]
    print(f"\n输出结果:\n{result}")
    print(f"输出 shape: {result.shape}")
    return result


def dynamic_batch_inference(session):
    """动态 batch 推理演示"""
    print("\n--- 动态 batch 推理 ---")
    for batch_size in [1, 4, 8]:
        data = np.random.randn(batch_size, 4).astype(np.float32)
        result = session.run(None, {"input": data})[0]
        print(f"  batch_size={batch_size}: 输入 {data.shape} → 输出 {result.shape}")


def batch_inference_loop(session, num_iters=5):
    """多次推理模拟"""
    print(f"\n--- 连续 {num_iters} 次推理 ---")
    for i in range(num_iters):
        data = np.random.randn(1, 4).astype(np.float32)
        result = session.run(None, {"input": data})[0]
        print(f"  iter {i}: 输出 {result.squeeze()}")


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(__file__), "..")
    model_path = os.path.join(root, "onnx", "model.onnx")

    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先运行: python onnx/export_onnx.py")
        exit(1)

    session = create_session(model_path)
    basic_inference(session)
    dynamic_batch_inference(session)
    batch_inference_loop(session)

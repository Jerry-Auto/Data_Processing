"""
ONNX 导出脚本
1. 加载模型 → 导出 ONNX（支持动态 batch）
2. 结构检查
3. 数值精度对比（PyTorch vs ONNX Runtime）
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import onnx
import onnxruntime as ort
from model.model import get_model, save_weights


def export_onnx(weights_path="model/weights.pth", output_path="onnx/model.onnx"):
    """导出 ONNX 模型"""
    # 确保权重存在
    if not os.path.exists(weights_path):
        save_weights(weights_path)

    model = get_model(weights_path)
    dummy = torch.randn(1, 4)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"ONNX 模型已导出到: {output_path}")
    return output_path


def verify_onnx(onnx_path="onnx/model.onnx"):
    """结构检查"""
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    print("✓ ONNX 结构检查通过")

    # 打印输入输出信息
    print(f"\n模型输入:")
    for inp in model_onnx.graph.input:
        print(f"  {inp.name}: {[d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]}")
    print(f"模型输出:")
    for out in model_onnx.graph.output:
        print(f"  {out.name}: {[d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]}")


def compare_outputs(weights_path="model/weights.pth", onnx_path="onnx/model.onnx"):
    """PyTorch vs ONNX Runtime 数值精度对比"""
    model = get_model(weights_path)
    test_input = torch.randn(2, 4)

    # PyTorch 输出
    with torch.no_grad():
        ref = model(test_input).numpy()

    # ONNX Runtime 输出
    sess = ort.InferenceSession(onnx_path)
    onnx_out = sess.run(None, {"input": test_input.numpy()})[0]

    # 比较
    np.testing.assert_allclose(ref, onnx_out, rtol=1e-3, atol=1e-5)
    print("✓ 数值精度对比通过（rtol=1e-3, atol=1e-5）")
    print(f"\nPyTorch 输出:\n{ref}")
    print(f"ONNX Runtime 输出:\n{onnx_out}")
    print(f"最大绝对误差: {np.max(np.abs(ref - onnx_out)):.2e}")


if __name__ == "__main__":
    # 从项目根目录运行时的路径处理
    root = os.path.join(os.path.dirname(__file__), "..")
    weights = os.path.join(root, "model", "weights.pth")
    onnx_out = os.path.join(root, "onnx", "model.onnx")

    export_onnx(weights, onnx_out)
    verify_onnx(onnx_out)
    compare_outputs(weights, onnx_out)

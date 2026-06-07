"""
Python 端：定义一个简单的 MLP 模型，导出为 TorchScript (.pt)
供 C++ 端 (libtorch) 加载并推理
"""
import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """一个简单的 3 层 MLP"""

    def __init__(self, input_dim=4, hidden_dim=16, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    model = SimpleMLP()
    model.eval()  # 切到推理模式

    # 构造一个示例输入，用于 trace
    dummy_input = torch.randn(1, 4)

    # 用 trace 方式导出（简单模型推荐 trace）
    traced = torch.jit.trace(model, dummy_input)

    output_path = "model.pt"
    traced.save(output_path)
    print(f"模型已导出到: {output_path}")

    # 验证：用 Python 加载回来跑一下
    loaded = torch.jit.load(output_path)
    test_input = torch.randn(2, 4)
    with torch.no_grad():
        out = loaded(test_input)
    print(f"验证输入 shape: {test_input.shape}")
    print(f"验证输出 shape: {out.shape}")
    print(f"验证输出:\n{out}")


if __name__ == "__main__":
    main()

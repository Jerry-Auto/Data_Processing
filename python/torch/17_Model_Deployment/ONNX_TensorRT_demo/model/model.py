"""
共享模型定义：SimpleMLP
供 ONNX 和 TensorRT 导出共用
"""
import torch
import torch.nn as nn
import os


class SimpleMLP(nn.Module):
    """
    简单 3 层 MLP
    输入: (batch, 4) → 隐藏层: 16 → 输出: (batch, 3)
    """

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


def get_model(pretrained_path=None):
    """获取模型实例（eval 模式）"""
    model = SimpleMLP()
    if pretrained_path and os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, weights_only=True))
        print(f"已加载权重: {pretrained_path}")
    model.eval()
    return model


def save_weights(save_path="model/weights.pth"):
    """保存模型权重（用于演示加载预训练权重的流程）"""
    model = SimpleMLP()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"权重已保存到: {save_path}")
    return save_path


if __name__ == "__main__":
    save_weights()

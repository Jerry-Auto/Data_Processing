"""
共享模型定义：SimpleMLP
供 ONNX 和 TensorRT 导出共用

模型规模：~33M 参数
  首层:  Linear(4, 320)     = 1,600
  319层: Linear(320, 320)   = 102,720 × 319 = 32,767,680
  末层:  Linear(320, 3)     = 963
  合计:  32,770,243 (~33M)
"""
import torch
import torch.nn as nn
import os


class SimpleMLP(nn.Module):
    """
    正方形深层 MLP，约 33M 参数
    输入: (batch, 4) → 320层隐藏层(320) → 输出: (batch, 3)
    """

    def __init__(self, input_dim=4, hidden_dim=320, output_dim=3, num_hidden_layers=320):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

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

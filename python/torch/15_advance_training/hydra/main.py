import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from einops import pack, unpack

# ==========================================
# 1. 用 einsum 实现一个轻量级的高级自注意力层
# ==========================================
class SimpleEinsumAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_projection = nn.Linear(embed_dim, embed_dim * 3)
        self.out_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x 形状: [Batch, Seq_len, Dim]
        b, t, d = x.shape
        
        # 算出 Q, K, V 并重塑为多头形状 [B, T, Heads, Head_dim]
        qkv = self.qkv_projection(x).reshape(b, t, self.num_heads, 3, self.head_dim)
        q, k, v = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]

        # 【高级操作 1：einsum 矩阵乘法】
        # 计算 Attention Map: Q 和 K 在 head_dim (d) 维度点积 -> [B, H, T, T]
        attn_scores = torch.einsum('b i h d, b j h d -> b h i j', q, k)
        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)

        # 【高级操作 2：einsum 上下文聚合】
        # 权重与 V 相乘 -> [B, H, T, Head_dim]
        out = torch.einsum('b h i j, b j h d -> b i h d', attn_weights, v)
        
        # 拉平多头并输出
        return self.out_projection(out.reshape(b, t, d))

# ==========================================
# 2. 定义多模态网络（对应 YAML 中的 _target_）
# ==========================================
class MultimodalNetwork(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.attn = SimpleEinsumAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)  # 压缩序列中每个位置的特征
        )

    def forward(self, text_feat, img_feat):
        # 模拟文本特征 text_feat: [B, T, D], 图像特征 img_feat: [B, I, D]
        
        # 【高级操作 3：einops.pack 打包】
        # 把文本和图像在序列维度合并，并让 packed_shapes 帮我们记账
        packed_tensor, packed_shapes = pack([text_feat, img_feat], 'b * d')
        print(f"-> [Pack] 混合多模态特征打包后的形状: {packed_tensor.shape}")

        # 扔进高级自注意力机制里做模态交互
        context_tensor = self.attn(packed_tensor)

        # 【高级操作 4：einops.unpack 解包】
        # 交互完后，各回各家，各找各妈，还原成原本的文本和图像序列长度
        text_out, img_out = unpack(context_tensor, packed_shapes, 'b * d')
        print(f"-> [Unpack] 解包还原后 - 文本形状: {text_out.shape}, 图像形状: {img_out.shape}")

        # 简单聚合输出一个分类logits
        logits = self.mlp(text_out).squeeze(-1).sum(dim=-1) # [Batch]
        return logits


# ==========================================
# 3. Hydra 驱动的主程序入口
# ==========================================
@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== 1. 读取 YAML 配置 ===")
    print(f"当前选择的模型路径: {cfg.model._target_}")
    print(f"网络内部绑定的特征维度: {cfg.model.embed_dim} (成功同步自 global_dim!)")

    print("\n=== 2. 开始通过 Hydra 动态实例化模型 ===")
    # 一行代码，直接创建出整个复杂的网络实例
    model = hydra.utils.instantiate(cfg.model)

    print("\n=== 3. 构造模拟双模态数据 ===")
    # 根据 YAML 中配置的 batch_size 和长度自动生成随机张量
    B = cfg.train.batch_size
    D = cfg.global_dim
    mock_text = torch.randn(B, cfg.train.text_len, D)   # [2, 3, 16]
    mock_image = torch.randn(B, cfg.train.image_len, D)  # [2, 2, 16]
    print(f"输入文本特征形状: {mock_text.shape}")
    print(f"输入图像特征形状: {mock_image.shape}")

    print("\n=== 4. 前向传播演练 ===")
    logits = model(mock_text, mock_image)
    
    print("\n=== 5. 标签验证的高级索引操作 ===")
    # 模拟网络预测的标签结果：假设 Batch=2 对应的预测类别分别是 3 和 1
    pred_labels = torch.tensor([[3], [1]], dtype=torch.long)
    
    # 【高级操作 5：torch.scatter_ 动态刷入】
    # 创建一个全 0 的基准矩阵，利用 scatter_ 动态把预测位置点亮为 1.0 (生成 One-hot)
    one_hot_matrix = torch.zeros(B, cfg.num_classes)
    one_hot_matrix.scatter_(dim=1, index=pred_labels, value=1.0)
    print("通过 scatter_ 动态生成的预测 One-hot 矩阵:\n", one_hot_matrix)

if __name__ == "__main__":
    main()
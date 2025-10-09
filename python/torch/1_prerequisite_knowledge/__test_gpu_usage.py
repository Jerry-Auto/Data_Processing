# 运行的一瞬间gpu爆满
import torch
import time

# 确认 GPU 可用
if not torch.cuda.is_available():
    print("CUDA 不可用！")
    exit()

# 创建大矩阵并移动到 GPU
device = torch.device("cuda")
a = torch.randn(10000, 10000, device=device)
b = torch.randn(10000, 10000, device=device)

# 执行矩阵乘法（GPU 密集型任务）
start = time.time()
c = torch.matmul(a, b)
torch.cuda.synchronize()  # 等待 GPU 计算完成
end = time.time()

print(f"GPU 计算耗时: {end - start:.4f} 秒")
print(f"结果前 5 个元素: {c[0, :5]}")

""" GPU监控工具，推荐使用nvitop
pip install nvitop
nvitop -m full  #展示超完整的显卡信息
"""

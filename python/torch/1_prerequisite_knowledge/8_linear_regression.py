# %% [markdown]
# 使用pytorch框架实现简单的线性回归

# %% [markdown]
# 生成数据集

# %%
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)#返回两个tensor

# %% [markdown]
# 读取数据集  
# TensorDataset已实现了__len__() 和__getitem__() 方法  
# 输入参数是两个，一个是features,一个是labels  
# *data_arrays 操作符会将 data_arrays 中的元素拆开，作为单独的参数传递给函数  
# 
# Dataset、DataLoader：可迭代对象  
# 
# <pre>
# DataLoader 如何实现迭代？
#     内部通过 _DataLoaderIter 类（C++ 加速的迭代器）管理数据加载逻辑。
#     每次调用 next(iter(dataloader)) 或 for batch in dataloader 时，DataLoader 会：
#         根据 batch_size 和 shuffle 参数生成一批索引。
#         调用 dataset.__getitem__(idx) 获取样本。
#         返回批处理后的数据（如 (batch_features, batch_labels)）。
# <pre>
# 
# 

# %%
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)


# %%
next(iter(data_iter))

# %% [markdown]
# 定义模型  
# 首先定义一个模型变量net，它是一个Sequential类的实例。   Sequential类将多个层串联在一起。   
# 当给定输入数据时，Sequential实例将数据传入到第一层， 然后将第一层的输出作为第二层的输入，以此类推。  
# 全连接层在Linear类中定义，需要将两个参数传递到nn.Linear中。  
# 第一个指定输入特征形状，即2，第二个指定输出特征形状，输出特征形状为单个标量，因此为1

# %%
# nn是神经网络的缩写
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))

# %% [markdown]
# 初始化模型参数  
# 能直接访问定义好的模型的参数以设定它们的初始值  
# 通过net[0]选择网络中的第一个图层， 然后使用weight.data和bias.data方法访问参数  
# 还可以使用替换方法normal_和fill_来重写参数值

# %%
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)


# %% [markdown]
# 定义损失函数  
# 定义优化算法  
# 实例化一个SGD实例时，我们要指定优化的参数 （可通过net.parameters()从我们的模型中获得）以及优化算法所需的超参数字典
# 

# %%
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# %% [markdown]
# 训练  
# 在每个迭代周期里，我们将完整遍历一次数据集（train_data）， 不停地从中获取一个小批量的输入和相应的标签。  
# 对于每一个小批量，我们会进行以下步骤:
# <pre>
#     通过调用net(X)生成预测并计算损失l（前向传播）。  
#     通过进行反向传播来计算梯度。  
#     通过调用优化器来更新模型参数。
# <pre>
# 

# %%
num_epochs = 10
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')


# %%
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)


# %%




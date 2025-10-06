import random
import math
import time
import numpy as np
import torch
from d2l import torch as d2l

def synthetic_data(w, b, num_examples, device=torch.device('cuda')):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)), device=device)
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape, device=device)
    return X, y.reshape((-1, 1))

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)], device=features.device)
        yield features[batch_indices], labels[batch_indices]

def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 设置设备
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device='cpu'
print(device)

# 超参数
num_sample = 50000000
batch_size = int(num_sample / 1)
num_epochs = int(num_sample / batch_size)
lr = 0.03

# 生成数据
true_w = torch.tensor([2, -3.4], device=device)
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, num_sample, device=device)

# 初始化参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True, device=device)
b = torch.zeros(1, requires_grad=True, device=device)

# 训练
timer = d2l.Timer()
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = squared_loss(linreg(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = squared_loss(linreg(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'Total time: {timer.stop()}')
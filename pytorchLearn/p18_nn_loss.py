import torch
from torch import nn

# L1Loss / MSELoss → 回归任务（连续值预测）
# CrossEntropyLoss → 分类任务（离散类别预测）

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
labels = torch.tensor([1, 3, 5], dtype=torch.float32)
inputs = torch.reshape(inputs, (1, 1, 1, 3))
labels = torch.reshape(labels, (1, 1, 1, 3))

loss_sum = nn.L1Loss(reduction='sum')
res_sum = loss_sum(inputs, labels)

# 平均绝对误差
# L1Loss = (1/N) * Σ |xi - yi|
loss_sum = nn.L1Loss(reduction='mean')
res_mean = loss_sum(inputs, labels)

# 均方误差
# MSELoss = (1/N) * Σ (xi - yi)^2
loss_mse = nn.MSELoss()
res_mse = loss_mse(inputs, labels)

print(res_sum)
print(res_mean)
print(res_mse)

x = torch.tensor([0.1, 0.2, 0.3])
x = torch.reshape(x, (1, 3))
y = torch.tensor([1])

# 交叉熵损失函数
# CrossEntropyLoss = - (1/N) * Σ log( exp(xi[yi]) / Σ exp(xi[j]) )
loss_cross = nn.CrossEntropyLoss()
res_cross = loss_cross(x, y)
print(res_cross)

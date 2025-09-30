import torch
import torchvision
from torch.utils.data import DataLoader
from p24_model import *

train_data = torchvision.datasets.CIFAR10('./dataset', train=True,
                                           transform=torchvision.transforms.ToTensor(), download=True )
test_data = torchvision.datasets.CIFAR10('./dataset', train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True )

print("训练集大小：{}".format(len(train_data)))
print("测试集大小：{}".format(len(test_data)))

# 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络
net = Net()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learing_rate = 0.01
optimizer = torch.optim.SGD(net.parameters(), lr=learing_rate)

train_step = 0
train_epoch = 20
for i in range(train_epoch):
    print("=====================第{}轮训练=====================".format(i+1))
    for data in train_dataloader:
        imgs, targets = data

        outputs = net(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_step += 1



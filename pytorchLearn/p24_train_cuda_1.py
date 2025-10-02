import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
import time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, input):
        return self.model(input)

# 准备数据集
train_data = torchvision.datasets.CIFAR10('./dataset', train=True,
                                           transform=torchvision.transforms.ToTensor(), download=True )
test_data = torchvision.datasets.CIFAR10('./dataset', train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True )
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集大小：{}".format(train_data_size))
print("测试集大小：{}".format(test_data_size))
if torch.cuda.is_available():
    print("cuda可用")
else:
    print("cuda不可用")

# 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
test_dataloader_size = len(test_dataloader)

# 创建网络
net = Net()
# net.load_state_dict(torch.load('net_20.pth'))
if torch.cuda.is_available():
    net = net.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

train_step = 0
test_step = 0
total_test_loss = 0.0
train_epoch = 20

start_time = time.time()
writer = SummaryWriter("./logs_train")
for i in range(train_epoch):
    print("=====================第{}轮训练=====================".format(i+1))

    # 训练
    net.train()
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()

        outputs = net(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_step += 1
        if train_step % 100 == 0:
            end_time = time.time()
            print("已用时：{:.4f} s, 训练次数：{}, loss: {:.4f}".format(end_time-start_time, train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), train_step)

    # 测试
    net.eval()
    total_test_loss = 0.0
    accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()

            outputs = net(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy += (outputs.argmax(1)==targets).sum()

    avg_test_loss = total_test_loss / test_dataloader_size
    print("测试集平均loss： {:.4f}".format(avg_test_loss))
    print("测试集上的正确率：{:.4f}".format(accuracy/test_data_size))
    writer.add_scalar("test_loss", avg_test_loss, test_step)
    test_step += 1

    torch.save(net.state_dict(), "./compass_model/net_{}.pth".format(i+1))
    print("Model {} saved successfully.".format(i+1))

writer.close()

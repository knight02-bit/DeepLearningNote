import torch
import torchvision
from torch import nn, optim
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

# 定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = Sequential(
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

    def forward(self, x):
        x = self.model(x)
        return x


# 1. 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2. 数据集
dataset = torchvision.datasets.CIFAR10(
    root='./dataset', train=False,
    download=True, transform=torchvision.transforms.ToTensor()
)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)  # 批量大点更适合GPU

# 3. 模型/损失/优化器
net = Net().to(device)
loss_cross = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.02)

# 4. 训练
for epoch in range(100):
    running_loss = 0.0
    for data in dataloader:
        imgs, labels = data
        # 把数据迁移到 GPU
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(imgs)
        loss = loss_cross(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"epoch: {epoch}  running loss: {running_loss:.4f}")



'''
Using device: cuda
epoch: 0  running loss: 183.9587
epoch: 1  running loss: 177.4316
epoch: 2  running loss: 172.1884
epoch: 3  running loss: 168.4696
epoch: 4  running loss: 162.5563
epoch: 5  running loss: 157.3800
epoch: 6  running loss: 151.3617
epoch: 7  running loss: 145.7537
epoch: 8  running loss: 141.6249
epoch: 9  running loss: 136.8641
epoch: 10  running loss: 130.3825
......
epoch: 94  running loss: 0.1382
epoch: 95  running loss: 0.1346
epoch: 96  running loss: 0.1325
epoch: 97  running loss: 0.1298
epoch: 98  running loss: 0.1274
epoch: 99  running loss: 0.1255
'''
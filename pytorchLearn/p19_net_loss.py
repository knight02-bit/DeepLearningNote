import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader


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


dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False,
                                       download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=1)

loss_cross = nn.CrossEntropyLoss()
net = Net()
step = 0
for data in dataloader:
    imgs, labels = data
    outputs = net(imgs)
    loss = loss_cross(outputs, labels)
    loss.backward()
    # 通过反向传播,梯度grad会不断更新
    print("---------------------grad {}---------------------".format(step))
    print("net.model._modules['0'].weight.grad => {}"
          .format(net.model._modules['0'].weight.grad))
    step += 1
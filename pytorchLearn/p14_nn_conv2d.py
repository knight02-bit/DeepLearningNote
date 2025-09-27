import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor())
data_loader = DataLoader(dataset, batch_size=64)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


net = Net()
# print(net)
writer = SummaryWriter("./logs")

step = 0
for data in data_loader:
    imgs, labels = data
    output = net(imgs)
    # print(imgs.shape)   # torch.Size([64, 3, 32, 32])
    # print(output.shape) # torch.Size([64, 6, 30, 30])

    writer.add_images("input", imgs, step)

    # add_images 函数期望的输入张量形状是 (N, 3, H, W)，其中 3 表示 RGB 三个通道
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step += 1


writer.close()
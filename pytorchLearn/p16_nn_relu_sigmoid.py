import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d, ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False,
                                       download=True, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        # 图像三个通道的取值都是0 - 255,用relu等于没有变化,用sigmoid才能映射到0 - 1的区间
        output = self.sigmoid1(input)
        return output

net = Net()
writer = SummaryWriter("./logs_relu_sigmoid")

step = 0
for data in dataloader:
    imgs, labels = data
    writer.add_images("input", imgs, step)
    output = net(imgs)
    writer.add_images("output", output, step)

    step += 1

writer.close()
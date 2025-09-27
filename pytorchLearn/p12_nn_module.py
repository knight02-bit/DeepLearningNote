import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


net = Net()
x = torch.tensor(1.0)
output = net(x)
print(output)

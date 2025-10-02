import torch
import torchvision.transforms
from PIL import Image
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear


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


# 图片处理
img_path = "./images/cat_test2.png"
img = Image.open(img_path)
img = img.convert('RGB') # 确保图像是标准的RGB三通道格式
transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                             torchvision.transforms.ToTensor()])
img = transforms(img)
img = torch.reshape(img, (1, 3, 32, 32))

# 加载保存的模型（这里用的.pth存了参数）
model = Net()
state_dict = torch.load("net_20.pth", map_location=torch.device('cuda'))
model.load_state_dict(state_dict)

# 预测
model.eval()
with torch.no_grad():
    output = model(img)
print(output)
print("预测标签为：{}".format(output.argmax(1)))

print("ps： {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}")



'''
{'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
'''
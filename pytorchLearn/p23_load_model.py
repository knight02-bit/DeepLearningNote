import torch
import torchvision

# 方式1
model = torch.load('vgg16_1.pth')

# 方式2
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load('vgg16_state_dict.pth'))

print("test")
import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1, 模型结构 + 模型参数
torch.save(vgg16, 'vgg16_1.pth')

# 保存方式2， 模型参数
print(vgg16.state_dict())
torch.save(vgg16.state_dict(), 'vgg16_state_dict.pth')
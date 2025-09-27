from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms,datasets

dataset_transform = transforms.Compose([
    transforms.ToTensor()
])

train_set = datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
test_set = datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)
print(train_set[0])

writer = SummaryWriter("p10")
for i in range(10):
    img, target = train_set[i]
    writer.add_image("test_set", img, i)

writer.close()

# print(test_set)
# print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[1]
# print(img, target)
# print(test_set.classes[target])
#
# img.show()
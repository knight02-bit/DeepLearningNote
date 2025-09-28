import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10(root="./dataset", transform=torchvision.transforms.ToTensor())

# drop_last: True舍弃最后不满一批的， False不舍弃
# shuffle: True 不同的epoch顺序会打乱， False不打乱
tset_loader = DataLoader(dataset= test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

writer = SummaryWriter("logs_dataloader")

for epoch in range(2):
    step = 0
    for data in tset_loader:
        imgs, label = data
        # print(imgs.shape, label)
        writer.add_images("Epoch:{}".format(epoch), imgs, step)
        step += 1

writer.close()
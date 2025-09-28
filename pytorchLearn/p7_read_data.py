from torch.utils.data import Dataset
from PIL import Image
import os

class MyDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_dir = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_dir)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "dataset/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyDataset(root_dir, ants_label_dir)
bees_dataset = MyDataset(root_dir, bees_label_dir)

img, label = bees_dataset[3]
img.show()
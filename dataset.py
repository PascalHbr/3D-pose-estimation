import torch.nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, ColorJitter, RandomResizedCrop
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
from utils import get_R_from_angles, get_R_from_uv


class PoseDataset(Dataset):
    def __init__(self, euler=False, rgb=False):
        self.euler = euler
        self.rgb = rgb

        self.images = self.load_images()
        self.angles = self.load_angles()
        self.transforms = self.set_transforms()

    def load_images(self):
        image_path = "images/*.png"
        images = sorted(glob.glob(image_path))
        return images

    def load_angles(self):
        labels_path = "labels.txt"
        with open(labels_path, "r") as f:
            lines = f.readlines()
            labels = list(map(lambda x: list(map(float, x.split(" ")[1:4])), lines))
        return np.asarray(labels)

    def load_image(self, path):
        img = Image.open(path)
        return img

    def set_transforms(self):
        self.crop_size = 224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.p_jitter = 0.2

        transforms = Compose([
            RandomResizedCrop(size=self.crop_size, scale=(0.4, 1.0), interpolation=InterpolationMode.BICUBIC),
            ColorJitter(brightness=self.p_jitter, contrast=self.p_jitter, hue=self.p_jitter),
            ToTensor(),
            Normalize(mean=self.mean, std=self.std),
        ])

        return transforms

    def make_binary_image(self, img, threshold=254):
        img = img.point(lambda p: 255 if p > threshold else 0)
        return img

    def get_6D_representations(self, angles):
        R = get_R_from_angles(angles)
        u = R[:, 0]
        v = R[:, 1]
        rep_6D = np.concatenate([u, v])
        return rep_6D

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # Get items
        img_path = self.images[item]
        img_org = self.load_image(img_path)
        angles = self.angles[item]

        # Make transformation
        if not self.rgb:
            img_org = self.make_binary_image(img_org)
        img = self.transforms(img_org)
        label = angles if self.euler else self.get_6D_representations(angles)
        label = torch.Tensor(label)

        return img, label, np.asarray(img_org)


if __name__ == "__main__":
    dataset = PoseDataset(rgb=True)
    img, label, img_org = dataset[0]

    # plt.imshow(img_org)
    # plt.show()
    print(len(dataset))
    print(img.shape)
    print(label)

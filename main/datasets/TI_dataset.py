import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import blobfile as bf
import matplotlib.pyplot as plt
import torchvision.transforms as T

class TI(Dataset):
    def __init__(self, root, norm=False, subsample_size=None, transform=None, **kwargs):
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")
        self.root = root
        self.norm = norm

        self.images = []

        img_path = os.path.join(self.root, "???")
        image_filenames = os.listdir(img_path)
        sorted_image_filenames = sorted(image_filenames, key=lambda x: int(x[4:-4]))
        for img in tqdm(sorted_image_filenames):
            full_img_path = os.path.join(img_path, img)
            self.images.append(full_img_path)
        # Subsample the dataset (if enabled)
        if subsample_size is not None:
            self.images = self.images[:subsample_size]

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)

        transform = T.Compose(
            [
                T.Resize((128, 128)),
                T.Grayscale(num_output_channels=1),
                T.ToTensor(),
                T.Lambda(lambda x: (x > 0.5).float()),
                # T.Normalize([0.5], [0.5]),
            ]
        )
        img = transform(img)

        if self.norm:
            img = np.asarray(img).astype(np.float)
        else:
            img = np.asarray(img).astype(np.float)

        return torch.from_numpy(img).float()

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    root = r"???"
    dataset = CelebAMaskHQDataset(root, subsample_size=None)
    print(len(dataset))




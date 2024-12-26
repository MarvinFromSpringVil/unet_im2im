from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import os
from glob import glob

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DeepStainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.x_images = glob(os.path.join(os.path.join(root_dir, 'x'), '*.png'))
        self.y_images = glob(os.path.join(os.path.join(root_dir, 'y'), '*.png'))
        
        assert len(self.x_images) == len(self.y_images)

        self.transform = transform

    def __len__(self):
        return len(self.x_images)

    def __getitem__(self, idx):
        x = Image.open(self.x_images[idx])
        y = Image.open(self.y_images[idx])

        x = self.transform(x)
        y = self.transform(y)
        
        sample = {'x': x, 'y': y}
        return sample


def get_dataloader(root_dir, batch_size, transforms):
    dataset = DeepStainDataset(root_dir, transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

if __name__ == '__main__':
    ROOT_DIR = './DeepStain'
    TRANSFORMS = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = DeepStainDataset(ROOT_DIR, TRANSFORMS)

    print(len(dataset)) 

    for elem in dataset:
        print(elem['x'].shape)
        print(elem['y'].shape)
        break 
    
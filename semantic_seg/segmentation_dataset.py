import os

from torch.utils.data import DataLoader
from torchvision import datasets
import pytorch_lightning as pl
from utils import read_rgb_mask


class ImageMaskModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size, augmentations, num_workers=2):
        super().__init__()
        self.data_path = data_path

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

        self.aug_train = augmentations[0]
        self.aug_eval = augmentations[1]

    def setup(self, stage=None):
        if self.dataset_train is None:
            self.dataset_train = datasets.ImageFolder(os.path.join(self.data_path, "train", "img"),
                                                      transform=self.aug_train, loader=read_rgb_mask)
            self.dataset_val = datasets.ImageFolder(os.path.join(self.data_path, "val", "img"),
                                                    transform=self.aug_eval, loader=read_rgb_mask)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          drop_last=False, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=False, pin_memory=True)
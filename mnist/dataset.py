"""
This Dataloader makes it easier to gather up the
common steps that are required to feed data into the models
for training.

Why is this better than handling it yourself?

QUOTE
But now, as the complexity of your processing grows
(transforms, multiple-GPU training), you can let Lightning
handle those details for you while making this dataset reusable
so you can share with colleagues or use in different projects.

NB: you need to handle your own transforms from PIL to tensor

"""

from pytorch_lightning import LightningDataModule
from typing import Optional
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms


class MNISTDataModule(LightningDataModule):
    def __init__(self, batch_size, data_dir):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        # PIL -> Tensor
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def setup(self, stage: Optional[str] = None):
        self.mnist_test = MNIST(
            self.data_dir, train=False, download=True, transform=self.transform
        )
        self.mnist_predict = MNIST(
            self.data_dir, train=False, download=True, transform=self.transform
        )

        mnist_full = MNIST(
            self.data_dir, train=True, download=True, transform=self.transform
        )

        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)

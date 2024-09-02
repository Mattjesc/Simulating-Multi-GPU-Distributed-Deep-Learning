from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './data', batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        self.fashionmnist_train = None
        self.fashionmnist_test = None

    def prepare_data(self):
        datasets.FashionMNIST(self.data_dir, train=True, download=True)
        datasets.FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.fashionmnist_train = datasets.FashionMNIST(self.data_dir, train=True, transform=self.transform)
        if stage == 'validate' or stage is None:
            self.fashionmnist_test = datasets.FashionMNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.fashionmnist_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.fashionmnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

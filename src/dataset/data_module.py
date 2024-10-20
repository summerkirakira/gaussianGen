from .dataloader import PLYPointCloudDataset, custom_collate_fn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from lightning.pytorch import LightningDataModule
from ..config.base_config import BaseConfig


class DataModule(LightningDataModule):
    def __init__(self, dataset_config: BaseConfig.Dataset):
        super().__init__()
        self.config = dataset_config
        self.dataset_path = self.config.path
        self.batch_size = self.config.batch_size
        self.num_workers = self.config.num_workers

    def train_dataloader(self):
        dataset = PLYPointCloudDataset(directory=self.dataset_path)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        dataset = PLYPointCloudDataset(directory=self.dataset_path)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=self.num_workers)
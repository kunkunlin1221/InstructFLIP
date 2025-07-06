import lightning as L
from torch.utils.data import DataLoader

from .dataset import build_dataset


class DataModule(L.LightningDataModule):
    def __init__(self, data_cfg, dataloder_cfg):
        super().__init__()
        self.train_data_cfg = data_cfg.train
        self.valid_data_cfg = data_cfg.valid
        self.dataloder_cfg = dataloder_cfg

    def train_dataloader(self):
        dataloader = DataLoader(dataset=build_dataset(**self.train_data_cfg), **self.dataloder_cfg.train)
        return dataloader

    def val_dataloader(self):
        valid_datasets = [build_dataset(**cfg) for cfg in self.valid_data_cfg]
        dataloader = [DataLoader(dataset=dataset, **self.dataloder_cfg.valid) for dataset in valid_datasets]
        return tuple(dataloader)

import numpy as np
import torch
import pytorch_lightning as pl

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        return {'a': np.zeros((100, 3), np.float32), 'b': np.zeros((100, 5), np.float32)}

class OpenProtDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

    def train_dataloader(self):
        ds = DummyDataset()
        loader = torch.utils.data.DataLoader(ds, batch_size=3)
        return loader
        
    def val_dataloader(self):
        ds = DummyDataset()
        loader = torch.utils.data.DataLoader(ds, batch_size=3)
        return loader
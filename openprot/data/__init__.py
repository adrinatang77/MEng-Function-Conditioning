import numpy as np
import torch
import pytorch_lightning as pl
import importlib
import sys


from .. import tracks
from .dummy import DummyDataset


class OpenProtDataset(torch.utils.data.IterableDataset):
    def __init__(self, cfg, rank, world_size):
        super().__init__()
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size

        self.datasets = []
        for name in cfg.datasets:  # autoload the datasets
            module, name_ = name.rsplit(".", 1)
            ds = getattr(importlib.import_module(module), name_)(cfg.datasets[name])
            self.datasets.append(ds)

        self.tracks = []  # autoload the tracks
        for name in cfg.tracks:
            module, name_ = name.rsplit(".", 1)
            track = getattr(importlib.import_module(module), name_)(cfg.tracks[name])
            self.tracks.append(track)

    def crop_or_pad(self, data):
        # crop or pad the protein here
        new_data = {}
        L = len(data["seqres"])
        pad_mask = np.zeros(self.cfg.data.crop, dtype=np.float32)
        pad_mask[: min(self.cfg.data.crop, L)] = 1.0
        new_data["pad_mask"] = pad_mask

        if L >= self.cfg.data.crop:  # needs crop
            start = np.random.randint(0, L - self.cfg.data.crop + 1)
            end = start + self.cfg.data.crop
            for key in data:
                new_data[key] = data[key][start:end]
        elif L < self.cfg.data.crop:  # needs pad
            pad = self.cfg.data.crop - L
            for key in data:
                # unfortunately this is a string, unlike everything else
                if key == "seqres":
                    new_data[key] = data[key] + "X" * pad
                else:
                    shape = data[key].shape
                    dtype = data[key].dtype
                    new_data[key] = np.concatenate(
                        [data[key], np.zeros((pad, *shape[1:]), dtype=dtype)]
                    )
        

        return new_data

    def process(self, data):
        # tokenize the data
        data = self.crop_or_pad(data)
        
        data_tok = {"pad_mask": data["pad_mask"]}
        for track in self.tracks:
            track.tokenize(data, data_tok)
        return data_tok

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        i = 0
        while True:  # very temporary
            data = self.datasets[0][i]
            if not self.cfg.data.overfit:
                i += 1
            yield self.process(data)

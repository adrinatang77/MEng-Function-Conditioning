import sys

import numpy as np
import pytorch_lightning as pl
import torch

from .. import tracks
from .dummy import DummyDataset


class OpenProtDataset(torch.utils.data.IterableDataset):
    def __init__(self, cfg, rank, world_size):
        super().__init__()
        self.rank = rank
        self.world_size = world_size

        self.datasets = []
        for name in cfg.datasets:  # autoload the datasets
            ds = getattr(sys.modules[__name__], name)(cfg.datasets[name])
            self.datasets.append(ds)

        self.tracks = []  # autoload the tracks
        for name in cfg.tracks:
            track = getattr(tracks, name)(cfg.tracks[name])
            self.tracks.append(track)

    def process(self, data):
        # protein should be cropped here

        data_tok = {}
        for track in self.tracks:
            track.tokenize(data, data_tok)
        return data_tok

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        while True:
            data = self.datasets[0][0]
            yield self.process(data)

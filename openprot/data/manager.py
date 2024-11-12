import numpy as np
import torch
import pytorch_lightning as pl
import importlib
import sys

from .. import tracks
from ..tracks.manager import OpenProtTrackManager
from ..utils.misc_utils import autoimport
from .data import OpenProtData


class OpenProtDatasetManager(torch.utils.data.IterableDataset):
    def __init__(self, cfg, tracks: OpenProtTrackManager, rank=0, world_size=1):
        super().__init__()
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size

        self.datasets = {}
        for name in cfg.datasets:  # autoload the datasets
            type_ = cfg.datasets[name].type
            ds = autoimport(f"openprot.data.{type_}")(cfg.datasets[name], cfg.features)
            self.datasets[name] = ds

        self.tasks = {}
        for name in cfg.tasks:  # autoload the train tasks
            task = autoimport(f"openprot.tasks.{name}")(cfg.tasks[name], self.datasets)
            self.tasks[name] = task

        self.task_probs = np.array([cfg.tasks[name].fraction for name in cfg.tasks])

        self.tracks = tracks

    def process(self, data: OpenProtData):

        
        loss_keys = []
        for task in self.tasks.values():
            loss_keys.extend(task.register_loss_masks())
            
        for key in loss_keys:
            if key not in data:
                data[key] = np.zeros((), dtype=np.float32)
                                    
        data.pad(self.cfg.data.crop)

        self.tracks.tokenize(data)

        return data

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if not worker_info:
            world_size = self.world_size
            rank = self.rank

        else:
            world_size = self.world_size * worker_info.num_workers
            rank = self.rank * worker_info.num_workers + worker_info.id

        rng = np.random.default_rng(seed=self.cfg.data.seed)
        assert self.task_probs.sum() == 1

        i = 0
        while True:
            task = rng.choice(self.cfg.tasks, p=self.task_probs)
            task = self.tasks[task]
            if i % world_size == rank:
                yield self.process(task.yield_data(crop=self.cfg.data.crop))
            task.advance()
            i += 1

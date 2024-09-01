import numpy as np
import torch
import pytorch_lightning as pl
import importlib
import sys

from .. import tracks
from ..utils.misc_utils import autoimport


class OpenProtDatasetManager(torch.utils.data.IterableDataset):
    def __init__(self, cfg, rank, world_size):
        super().__init__()
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size

        self.datasets = {}
        for name in cfg.datasets:  # autoload the datasets
            ds = autoimport(f"openprot.data.{name}")(cfg.datasets[name], cfg.features)
            self.datasets[name] = ds

        self.tasks = {}
        for name in cfg.tasks:  # autoload the train tasks
            task = autoimport(f"openprot.tasks.{name}")(cfg.tasks[name], self.datasets)
            self.tasks[name] = task

        self.task_probs = np.array([cfg.tasks[name].fraction for name in cfg.tasks])
        assert self.task_probs.sum() == 1

        self.tracks = {}  # autoload the tracks
        for name in cfg.tracks:
            track = autoimport(f"openprot.tracks.{name}")(cfg.tracks[name])
            self.tracks[name] = track

    def process(self, data):

        data.pad(self.cfg.data.crop)

        for track in self.tracks:
            self.tracks[track].tokenize(data)

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

        i = 0
        while True:
            task = rng.choice(self.cfg.tasks, p=self.task_probs)
            task = self.tasks[task.split(".")[-1]]
            if i % world_size == rank:
                yield self.process(task.yield_data(crop=self.cfg.data.crop))
            task.advance()
            i += 1

import numpy as np
import torch
import pytorch_lightning as pl
import importlib
import sys

from .. import tracks

def autoimport(name):
    module, name_ = name.rsplit(".", 1)
    return getattr(importlib.import_module(module), name_)
    
class OpenProtDatasetManager(torch.utils.data.IterableDataset):
    def __init__(self, cfg, rank, world_size):
        super().__init__()
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size

        self.datasets = {}
        for name in cfg.datasets:  # autoload the datasets
            ds = autoimport(name)(cfg.datasets[name])
            self.datasets[name.split('.')[-1]] = ds

        self.tasks = {}
        for name in cfg.tasks: # autoload the train tasks
            task = autoimport(name)(cfg.tasks[name], self.datasets)
            task.shuffle_datasets()
            self.tasks[name.split('.')[-1]] = task

        self.task_probs = np.array([cfg.tasks[name].fraction for name in cfg.tasks])
        assert self.task_probs.sum() == 1
        
        self.tracks = {}  # autoload the tracks
        for name in cfg.tracks:
            track = autoimport(name)(cfg.tracks[name])
            self.tracks[name.split('.')[-1]] = track

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
            self.tracks[track].tokenize(data, data_tok)
        return data_tok

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
            task = self.tasks[task.split('.')[-1]]
            if i % world_size == rank:
                yield self.process(task.yield_data())
            task.advance()
            i += 1

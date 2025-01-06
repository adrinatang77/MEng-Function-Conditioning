import numpy as np
import torch
import pytorch_lightning as pl
import importlib
import sys
import tqdm

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
            ds.cfg.name = name
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

        self.tracks.tokenize(data)

        return data

    def form_batches(self, buf):
        lens = [len(data['seqres']) for data in buf]
        batches = []
        buf = [buf[i] for i in np.argsort(lens)]
        curr = []
        for data in buf:
            if (len(curr)+1) * len(data['seqres']) > self.cfg.data.max_toks:
                batches.append(curr)
                curr = []
            curr.append(data)
        if len(curr) > 0:
            batches.append(curr)
        return batches
        
    def __iter__(self):
        
        it = self.unbuffered_iter()
        if not self.cfg.buffer:
            return it
            
        batches = []
        buf = []
        for _ in range(self.cfg.data.buffer):
            buf.append(next(it))
        
        while True:
            if len(batches) == 0:
                assert len(buf) == self.cfg.data.buffer
                batches = self.form_batches(buf)
                buf = []
            idx = np.random.randint(0, len(batches))
            
            batch = batches[idx]
            batches = batches[:idx] + batches[idx+1:]
            yield batch
            
            for _ in range(len(batch)): buf.append(next(it))
            

            
    def unbuffered_iter(self):
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

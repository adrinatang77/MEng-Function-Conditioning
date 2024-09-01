from ..utils.misc_utils import autoimport
import torch
import numpy as np


class OpenProtEvalManager(torch.utils.data.IterableDataset):
    def __init__(self, cfg, rank=0, world_size=1):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size

        self.tracks = {}
        for name in cfg.tracks:  # autoload the tracks
            track = autoimport(f"openprot.tracks.{name}")(cfg.tracks[name])
            self.tracks[name] = track

        self.evals = {}
        for name in cfg.evals:  # autoload the eval tasks
            eval_ = autoimport(f"openprot.evals.{name}")(cfg.evals[name], cfg.features)
            self.evals[name] = eval_

    def process(self, data):
        data.pad(None)  # just to get the pad mask

        for track in self.tracks:
            self.tracks[track].tokenize(data)

    def __iter__(self):
        i = 0
        for name in self.cfg.evals:
            ev = self.evals[name]
            for j in range(len(ev)):
                if i % self.world_size == self.rank:
                    data = self.evals[name][j]
                    self.process(data)
                    data["eval"] = name  # :/
                    yield data
                i += 1

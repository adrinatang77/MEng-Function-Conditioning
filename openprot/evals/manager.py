from ..utils.misc_utils import autoimport
import torch
import numpy as np
from ..data.data import OpenProtData
from ..tracks.manager import OpenProtTrackManager


class OpenProtEvalManager(torch.utils.data.IterableDataset):
    def __init__(self, cfg, tracks: OpenProtTrackManager, rank=0, world_size=1):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size

        self.tracks = tracks

        self.evals = {}
        for name in cfg.evals:  # autoload the eval tasks
            type_ = cfg.evals[name].type
            eval_ = autoimport(f"openprot.evals.{type_}")(
                cfg.evals[name], cfg.features, tracks
            )
            eval_.cfg.name = name  # know its own name
            self.evals[name] = eval_

    def process(self, data: OpenProtData):
        data.pad(None)  # just to get the pad mask

        self.tracks.tokenize(data)

    def compute_metrics(self):
        breakpoint()

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

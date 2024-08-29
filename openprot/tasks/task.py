import numpy as np
from ..utils.misc_utils import temp_seed

class Task:
    def __init__(self, cfg, datasets):
        self.cfg = cfg
        self.datasets = datasets
        self.dataset_probs = np.array([cfg.datasets[ds].fraction for ds in cfg.datasets])
        assert self.dataset_probs.sum() == 1
        self.rng = np.random.default_rng(seed=cfg.seed)
        
    
    def shuffle_datasets(self):
        self.shuffled_idx = {}
        self.counter = {}

        for name in self.cfg.datasets:
            idx = np.arange(len(self.datasets[name]))
            with temp_seed(self.cfg.datasets[name].seed):
                np.random.shuffle(idx)
            self.shuffled_idx[name] = idx
            self.counter[name] = self.cfg.datasets[name].start

        self.curr_ds = self.rng.choice(self.cfg.datasets, p=self.dataset_probs)
        
    def advance(self):
        self.counter[self.curr_ds] += 1
        self.curr_ds = self.rng.choice(self.cfg.datasets, p=self.dataset_probs)
        

    def yield_data(self):
        
        name = self.curr_ds
        ds = self.datasets[name]
        order = self.shuffled_idx[name]
        idx = self.counter[name]

        # print(f"i={i} rank={rank} ds={name} idx={idx} actual={order[idx]}")
        return self.prep_data(ds[order[idx]])
            

        
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--yaml', type=str, required=True)
parser.add_argument('--prefix', type=str, required=True)
args = parser.parse_args()

import numpy as np
import os
from omegaconf import OmegaConf

class Sweeper:
    def __init__(self, grid=10):
        self.best = np.ones((grid, grid+1)) * -np.inf
        self.dirs = np.zeros((grid, grid+1)) * np.nan
        
    def get_path(self, i, j):
        path = np.zeros(11, dtype=int)
        path[i+1] = j
        for i in list(range(-1, i))[::-1]:
            path[i+1] = self.dirs[i+1,path[i+2]]
        return path
    
    def get_next(self, i):
        out = {}
        for j in range(11):
            for k in [-2, -1, 0, 1, 2]:
                jj = j+k
                if jj not in list(range(11)): continue
                key = f"{i}_{j}_{jj}"
                arr = self.get_path(i-1, jj)
                arr[i+1] = j
                out[key] = arr
        return out
    def update(self, data):
        for key, val in data.items():
            i, j, jj = key.split('_')
            i, j, jj = int(i), int(j), int(jj)
            if val > self.best[i,j]:
                self.dirs[i,j] = jj
                self.best[i,j] = val


sweeper = Sweeper()

os.makedirs(f"workdir/sweep_{args.prefix}", exist_ok=True)
def run_job(key, sched):
    np.save(f"workdir/sweep_{args.prefix}/{key}.npy", sched[::-1]/10)
    cfg = OmegaConf.load(args.yaml)
    cfg.logger.name = f'{args.prefix}/{key}'
    i, j, k = key.split('_')
    cfg.evals.codesign_1k.truncate = (int(i)+1)/10
    cfg.evals.codesign_1k.schedule['seq_temp'] = f"workdir/sweep_{args.prefix}/{key}.npy"
    with open(f"workdir/sweep_{args.prefix}/{key}.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"workdir/sweep_{args.prefix}/{key}.yaml")
    return np.random.rand()
for i in range(10):
    res = {}
    jobs = sweeper.get_next(i)
    linspace = np.linspace(0, 1, 1000)
    cos = 10*np.sin(np.pi*linspace)
    for key in jobs:
        path = np.interp(linspace, np.linspace(0, 1, 11), jobs[key])
        res[key] = run_job(key, jobs[key]) # -np.sum(np.abs(cos - path))
    sweeper.update(res)
print(sweeper.best)
print(sweeper.dirs)
    
            
print(sweeper.get_path(9, np.argmax(sweeper.best[-1])))
        

        
from ..tracks.sequence import MASK_IDX
from torch.distributions.categorical import Categorical
import torch
import tqdm
import numpy as np


class OpenProtSampler:
    def __init__(self, schedules, steppers):
        self.schedules = schedules
        self.steppers = steppers

    def sample(self, model, noisy_batch, steps=100, trunc=None):

        extra = {}
        steps = np.linspace(0, 1, steps+1)
        steps = list(zip(steps[:-1], steps[1:]))
        for t, s in tqdm.tqdm(steps):
            if trunc is not None and t < trunc: continue
            sched = {key: (sched(t), sched(s)) for key, sched in self.schedules.items()}
            noisy_batch = self.single_step(model, noisy_batch, sched, extra)

        return noisy_batch, extra
        
    def single_step(self, model, noisy_batch, sched, extra={}): # step from t to s

        for stepper in self.steppers:
            stepper.set_step(noisy_batch, sched, extra)
            _, out = model.forward(noisy_batch)
            stepper.advance(noisy_batch, sched, out, extra)

        return noisy_batch
        

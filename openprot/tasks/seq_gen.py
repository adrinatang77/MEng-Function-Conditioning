from .task import Task
import numpy as np


class SequenceGeneration(Task):
    def prep_data(self, data):

        data['seq_noise'] = (np.random.rand(len(data)) < self.cfg.mask_rate).astype(np.float32)
        data['seq_noise'] *= data['seq_mask']
        
        return data

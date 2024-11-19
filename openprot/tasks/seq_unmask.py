from .task import OpenProtTask
import numpy as np


class SequenceUnmasking(OpenProtTask):
    def prep_data(self, data, crop=None):

        if crop is not None:
            data.crop(crop)

        if np.random.rand() < self.cfg.uniform_prob:
            noise_level = np.random.rand()
        else:
            noise_level = np.random.beta(*self.cfg.beta)
        noise_level = 0.15
        L = len(data["seqres"])
        data["seq_noise"] = (np.random.rand(L) < noise_level).astype(np.float32)

        return data

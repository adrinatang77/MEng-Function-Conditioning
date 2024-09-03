from .task import OpenProtTask
import numpy as np


class SequenceGeneration(OpenProtTask):
    def prep_data(self, data):

        data["seq_noise"] = (
            np.random.rand(len(data["seqres"])) < self.cfg.mask_rate
        ).astype(np.float32)
        data["seq_noise"] *= data["seq_mask"]

        return data

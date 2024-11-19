from .task import OpenProtTask
import numpy as np


class SequenceGeneration(OpenProtTask):
    def prep_data(self, data, crop=None):

        if crop is not None:
            data.crop(crop)

        data["seq_noise"] = (
            np.random.rand(len(data["seqres"])) < self.cfg.mask_rate
        ).astype(np.float32)
        data["seq_noise"] *= data["seq_mask"]

        return data

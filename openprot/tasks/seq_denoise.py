from .task import OpenProtTask
import numpy as np

class SequenceDenoising(OpenProtTask):
    def prep_data(self, data, crop=None):

        if crop is not None:
            data.crop(crop)
        # random noise for each token, sampled from uniform distribution between 0 and 1
        seq_len = len(data["seqres"])
        if self.cfg.sampling == "uniform":
            noise = (
                np.random.uniform(0, 1, seq_len)
            ).astype(np.float32)
        elif self.cfg.sampling == "beta":
            noise = np.random.beta(*self.cfg.sampling_args, seq_len).astype(np.float32)
        data["seq_noise"] = noise
        data["seq_noise"] *= data["seq_mask"]

        return data

from .task import OpenProtTask
import numpy as np

class SequenceDenoising(OpenProtTask):
    def prep_data(self, data):
        # random noise for each token, sampled from uniform distribution between 0 and 1
        data["seq_noise"] = (
            np.random.rand(len(data["seqres"]))
        ).astype(np.float32)
        data["seq_noise"] *= data["seq_mask"]

        return data
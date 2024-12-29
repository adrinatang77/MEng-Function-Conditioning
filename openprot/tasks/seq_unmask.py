from .task import OpenProtTask
import numpy as np


class SequenceUnmasking(OpenProtTask):

    def register_loss_masks(self):
        return ["/seq_gen"]
        
    def prep_data(self, data, crop=None, eps=1e-5, inf=1e5):

        if crop is not None:
            data.crop(crop)

        if np.random.rand() < self.cfg.uniform_prob:
            noise_level = np.random.rand()
        else:
            noise_level = np.random.beta(*self.cfg.beta)

        L = len(data["seqres"])
        data["seq_noise"] = (np.random.rand(L) < noise_level).astype(np.float32)
        t = (data["seq_mask"] * data["seq_noise"]).sum() / (eps + data["seq_mask"].sum())
        data["seq_weight"] = np.ones(L, dtype=np.float32) / t * self.cfg.weight
        
        
        data["struct_noise"] = np.ones(L, dtype=np.float32) * inf
        
        data["/seq_gen"] = np.ones((), dtype=np.float32)
        
        return data


        # ours @ 15/0: 12, 17
        # dplm @ 15/0: 10, 2
        # ours @ beta/0: 13.3 / 18
        # dplm @ beta/0: 11.2 / 2
        
        # dplm @ beta/0.15: 12.5 / 3.0
from .task import OpenProtTask
import numpy as np


class SequenceUnmasking(OpenProtTask):

    def register_loss_masks(self):
        return ["/seq_gen"] + [f"/seq_gen_{i}" for i in range(20)]
        
    def prep_data(self, data, crop=None, eps=1e-5, inf=1e5):

        if crop is not None:
            data.crop(crop)

        if np.random.rand() < self.cfg.uniform_prob:
            noise_level = np.random.rand()
        else:
            noise_level = np.random.beta(*self.cfg.beta)

        L = len(data["seqres"])
        
        t_inv = min(100, 1/noise_level)
        t = noise_level

        ones = np.ones(L, dtype=np.float32)
        if self.cfg.reweight == 'linear':
            data["seq_weight"] = ones * (1-t) 
        elif self.cfg.reweight == 'inverse':
            data["seq_weight"] = ones * t_inv 
        else:
            assert not self.cfg.reweight, 'reweight type not recognized'
            data["seq_weight"] = ones 
        
        data["struct_noise"] = np.ones(L, dtype=np.float32) * inf
        
        data["/seq_gen"] = np.ones((), dtype=np.float32)
        i = int(20 * noise_level)
        data[f"/seq_gen_{i}"] = np.ones((), dtype=np.float32)
        
        return data


        # ours @ 15/0: 12, 17
        # dplm @ 15/0: 10, 2
        # ours @ beta/0: 13.3 / 18
        # dplm @ beta/0: 11.2 / 2
        
        # dplm @ beta/0.15: 12.5 / 3.0
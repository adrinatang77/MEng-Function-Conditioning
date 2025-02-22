from .task import OpenProtTask
import numpy as np
from ..utils import residue_constants as rc
from scipy.spatial.transform import Rotation as R


class CodesignTask(OpenProtTask):

    def center_random_rot(self, data, eps=1e-6):
        # center the structures
        pos = data["struct"]
        mask = data["struct_mask"][..., None]
        com = (pos * mask).sum(-2) / (mask.sum(-2) + eps)
        data["struct"] -= com

        randrot = R.random().as_matrix()
        data["struct"] @= randrot.T

        idx = np.unique(data['chain'])
        for i in idx: # note that multi-residue ligands will be wrong
            conf = data["ref_conf"][data['chain'] == i]
            conf -= conf.mean(0)
            randrot = R.random().as_matrix()
            conf @= randrot.T
            data["ref_conf"][data['chain'] == i] = conf
            
        
    def add_sequence_noise(self, data, noise_level=None, sup=False):
        
        def sample_noise_level():
            rand = np.random.rand()
            probs = [
                self.cfg.seq.get('zero_prob', 0),
                self.cfg.seq.get('max_prob', 0),
                self.cfg.seq.get('uniform_prob', 0),
            ]
            probs = np.cumsum(probs)
            
            if rand < probs[0]:
                noise_level = 0.0
            elif rand < probs[1]:
                noise_level = 1.0
            elif rand < probs[2]:
                noise_level = np.random.rand()
            else:
                noise_level = np.random.beta(*self.cfg.seq.beta)
            return noise_level
            
        # different noise level per chain
        L = len(data["seqres"])
        
        if noise_level is None:
            data["seq_noise"] = np.zeros(L, dtype=np.float32)
            idx = np.unique(data['chain'])
            for i in idx:
                data["seq_noise"][data['chain'] == i] = sample_noise_level()
        else:
            data["seq_noise"] = np.ones(L, dtype=np.float32) * noise_level

        
        if sup:
            data["seq_weight"] = np.ones(L, dtype=np.float32)

        
    def add_structure_noise(self, data, eps=1e-6, noise_level=None, sup=False):
        

        def sample_noise_level():
            rand = np.random.rand()
            
            probs = [
                self.cfg.struct.get('zero_prob', 0),
                self.cfg.struct.get('max_prob', 0),
                self.cfg.struct.get('uniform_prob', 0),
            ]
            probs = np.cumsum(probs)
            if rand < probs[0]:
                noise_level = 0.0
            elif rand < probs[1]:
                noise_level = 1.0
            elif rand < probs[2]:
                noise_level = np.random.rand()
            else:
                noise_level = np.random.beta(*self.cfg.struct.beta)

            #####
            p = self.cfg.edm.sched_p
            sigma = (
                self.cfg.edm.sigma_min ** (1 / p)
                + noise_level * (self.cfg.edm.sigma_max ** (1 / p) - self.cfg.edm.sigma_min ** (1 / p))
            ) ** p
            #####
        
            return sigma
        
        L = len(data["seqres"])
        if noise_level is None:
            data["struct_noise"] = np.zeros(L, dtype=np.float32)
            idx = np.unique(data['chain'])
            for i in idx:
                data["struct_noise"][data['chain'] == i] = sample_noise_level()
        else:
            data["struct_noise"] = np.ones(L, dtype=np.float32) * noise_level

        if sup:
            data["struct_weight"] = np.ones(L, dtype=np.float32)


        self.center_random_rot(data)

        return noise_level

class Codesign(CodesignTask):
    def register_loss_masks(self):
        return ["/codesign"]

    def prep_data(self, data, crop=None):

        if crop is not None:
            data.crop(crop)

        self.add_sequence_noise(data, sup=True)
        self.add_structure_noise(data, sup=True)
        data["/codesign"] = np.ones((), dtype=np.float32)
        
        return data
    
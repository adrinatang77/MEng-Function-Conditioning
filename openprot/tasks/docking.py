from .task import OpenProtTask
import numpy as np
from ..utils import residue_constants as rc
from scipy.spatial.transform import Rotation as R
from .codesign import CodesignTask

class Docking(CodesignTask):

    def register_loss_masks(self):
        return ["/docking"] # , "/struct_pred/t1"]

    def prep_data(self, data, crop=None, eps=1e-6):

        if crop is not None:
            data.crop(crop)

        mask = data['mol_type'] == 0
        data['ref_conf'][mask] = data['struct'][mask]
        data['ref_conf_mask'][mask] = data['struct_mask'][mask]
        noise_level = self.add_structure_noise(data, sup=True)
        
        data["/struct_pred"] = np.ones((), dtype=np.float32)
        # if noise_level > 1 - eps:
        #     data["/struct_pred/t1"] = np.ones((), dtype=np.float32)
        # else:
        #     data["/struct_pred/t1"] = np.zeros((), dtype=np.float32)

        return data

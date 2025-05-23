from .task import OpenProtTask
import numpy as np
from ..utils import residue_constants as rc
from scipy.spatial.transform import Rotation as R
from .codesign import CodesignTask

class StructurePrediction(CodesignTask):

    def register_loss_masks(self):
        return ["/struct_pred"] # , "/struct_pred/t1"]

    def prep_data(self, data, crop=None, eps=1e-6):

        if crop is not None:
            data.crop(crop)

        noise_level = self.add_structure_noise(data, sup=True)

        data["_cath_noise"] = np.ones((), dtype=np.float32)
        
        data["/struct_pred"] = np.ones((), dtype=np.float32)
        
        # if noise_level > 1 - eps:
        #     data["/struct_pred/t1"] = np.ones((), dtype=np.float32)
        # else:
        #     data["/struct_pred/t1"] = np.zeros((), dtype=np.float32)

        return data

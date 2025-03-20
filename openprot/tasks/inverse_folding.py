from .task import OpenProtTask
import numpy as np
from ..utils import residue_constants as rc
from scipy.spatial.transform import Rotation as R
from .codesign import CodesignTask

class InverseFolding(CodesignTask):

    def register_loss_masks(self):
        return ["/inv_fold"]
        
    def prep_data(self, data, crop=None, eps=1e-6):

        if crop is not None:
            data.crop(crop)

        self.add_sequence_noise(data, sup=True)
        self.add_structure_noise(data, noise_level=0)

        self.center_random_rot(data)
        
        data["/inv_fold"] = np.ones((), dtype=np.float32)
        return data
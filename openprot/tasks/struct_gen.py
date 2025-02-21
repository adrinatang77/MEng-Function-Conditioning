from .task import OpenProtTask
import numpy as np
from ..utils import residue_constants as rc
from scipy.spatial.transform import Rotation as R
from .codesign import CodesignTask

class StructureGeneration(CodesignTask):

    def register_loss_masks(self):
        return ["/struct_gen"]

    def prep_data(self, data, crop=None, eps=1e-6):

        if crop is not None:
            data.crop(crop)

        self.add_sequence_noise(data, noise_level=1.01)
        self.add_structure_noise(data, sup=True)
        
        data["/struct_gen"] = np.ones((), dtype=np.float32)

        return data

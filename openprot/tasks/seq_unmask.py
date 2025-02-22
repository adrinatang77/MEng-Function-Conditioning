from .task import OpenProtTask
import numpy as np
from .codesign import CodesignTask

class SequenceUnmasking(CodesignTask):

    def register_loss_masks(self):
        return ["/seq_gen"] + [f"/seq_gen_{i}" for i in range(20)]
        
    def prep_data(self, data, crop=None, eps=1e-5, inf=1e5):

        if crop is not None:
            data.crop(crop)

        noise_level = self.add_sequence_noise(data, sup=True)
        self.add_structure_noise(data, noise_level=1.01)
        
        data["/seq_gen"] = np.ones((), dtype=np.float32)
        # i = int(20 * noise_level)
        # data[f"/seq_gen_{i}"] = np.ones((), dtype=np.float32)
        
        return data

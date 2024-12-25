### Algorithm 1 Motif-scaffolding data augmentation
# Require: Protein backbone T; Min and max motif percent γmin = 0.05, γmax = 0.5.
# 1: s ∼ Uniform{⌊N · γmin⌋, . . . , ⌊N · γmax⌋} ▷ Sample maximum motif size.
# 2: m ∼ Uniform{1, . . . , s} ▷ Sample maximum number of motifs.
# 3: TM ← ∅
# 4: for i ∈ {1, . . . , m} do
# 5: j ∼ Uniform{1, . . . , N} \ TM ▷ Sample location for each motif
# 6: ℓ ∼ Uniform{1, . . . , s − m + i − |TM|} ▷ Sample length of each motif.
# 7: TM ← TM ∪ {Tj , . . . , Tmin(j+ℓ,N)} ▷ Append to existing motif.
# 8: end for
# 9: TS ← {T1, . . . , TN } \ TM ▷ Assign rest of residues as the scaffold
# 10: return TM, TS

from .task import OpenProtTask
import numpy as np
from ..utils import residue_constants as rc
from scipy.spatial.transform import Rotation as R


class MotifScaffolding(OpenProtTask):

    def register_loss_masks(self):
        return []

    def prep_data(self, data, crop=None, eps=1e-6):

        if crop is not None:
            data.crop(crop)
        
        L = len(data['seqres'])
        s = np.random.rand() * (self.cfg.ymax - self.cfg.ymin) + self.cfg.ymin
        s = int(s * L)
        m = np.random.randint(1, max(s, 1) + 1)

        is_motif = np.zeros(len(data['seqres']), dtype=bool)
        for i in range(m):
            j = np.random.randint(0, L)
            end = s - m + i - is_motif.sum()
            if end > 0:
                l = np.random.randint(1, end+1)
                is_motif[j:j+l] = True
            
        ## noise EVERYTHING
        if np.random.rand() < self.cfg.uniform_prob:
            noise_level = np.random.rand()
        else:
            noise_level = np.random.beta(*self.cfg.beta)

        L = len(data["seqres"])
        data["trans_noise"] = np.ones(L, dtype=np.float32) * noise_level
        data["rots_noise"] = np.ones(L, dtype=np.float32) * noise_level

        data['trans_noise'][is_motif] = 0
        data['rots_noise'][is_motif] = 0

        data["seq_noise"] = np.ones(L, dtype=np.float32)

        # data["torsion_noise"] = np.ones(len(data["seqres"]))

        # center the structures
        pos = data["atom37"][..., rc.atom_order["CA"], :]
        mask = data["atom37_mask"][..., rc.atom_order["CA"], None]
        com = (pos * mask).sum(-2) / (mask.sum(-2) + eps)
        data["atom37"] -= com

        if self.cfg.random_rot:
            randrot = R.random().as_matrix()
            data["atom37"] @= randrot.T

        return data

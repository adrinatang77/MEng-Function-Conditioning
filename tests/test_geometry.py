import numpy as np
import openprot.utils.residue_constants as rc
from openprot.utils.geometry import (
    atom14_to_atom37,
    atom37_to_atom14,
    atom37_to_frames,
    atom37_to_torsions,
    frames_torsions_to_atom37,
)
import torch


def test_atom14_to_37():
    np.random.seed(137)
    batch_dims = (2, 3)
    L = 128
    aatype = np.random.randint(0, 20, batch_dims + (L,))

    atom14 = np.random.randn(*(batch_dims + (L, 14, 3)))
    atom14 *= rc.RESTYPE_ATOM14_MASK[aatype, :, None]

    ## roundtrip without mask
    atom37 = atom14_to_atom37(atom14, aatype)
    atom14_ = atom37_to_atom14(atom37, aatype)

    assert np.allclose(atom14, atom14_)

    # create a nontrivial mask
    atom14_mask = np.random.randint(0, 2, batch_dims + (L, 14)).astype(float)
    atom14_mask *= rc.RESTYPE_ATOM14_MASK[aatype]  # make it a valid mask

    ## roundtrip with mask
    atom14 *= atom14_mask[..., None]

    atom37, atom37_mask = atom14_to_atom37(atom14, aatype, atom14_mask)
    assert atom37_mask.sum() == atom14_mask.sum()

    atom14_, atom14_mask_ = atom37_to_atom14(atom37, aatype, atom37_mask)
    assert np.allclose(atom14, atom14_)
    assert np.allclose(atom14_mask, atom14_mask_)


def test_atom37_to_torsions():
    np.random.seed(137)
    batch_dims = (2, 3)
    L = 128
    aatype = np.random.randint(0, 20, batch_dims + (L,))

    atom37 = np.random.randn(*(batch_dims + (L, 37, 3)))
    atom37 *= rc.RESTYPE_ATOM37_MASK[aatype, :, None]

    ## roundtrip to standardize
    frames = atom37_to_frames(atom37)
    torsions, torsion_mask = atom37_to_torsions(atom37, aatype)
    atom37 = frames_torsions_to_atom37(frames, torsions, aatype).numpy()

    ## roundtrip without a mask
    frames = atom37_to_frames(atom37)
    torsions, torsion_mask = atom37_to_torsions(atom37, aatype)
    atom37_ = frames_torsions_to_atom37(frames, torsions, aatype).numpy()

    assert np.allclose(atom37, atom37_, rtol=1e-4, atol=1e-4)

    # create a nontrivial mask
    atom37_mask = np.random.randint(0, 2, batch_dims + (L, 37)).astype(float)
    atom37_mask *= rc.RESTYPE_ATOM37_MASK[aatype]  # make it a valid mask

    ## check that the mask is propagated
    frames = atom37_to_frames(atom37)
    torsions_, torsion_mask_ = atom37_to_torsions(atom37, aatype, atom37_mask)
    assert torsion_mask_.sum() < torsion_mask.sum()

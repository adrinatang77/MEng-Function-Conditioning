import torch
import numpy as np

from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from openfold.data.data_transforms import get_chi_atom_indices
from .rigid_utils import Rigid, Rotation
from . import residue_constants as rc
from openfold.utils.tensor_utils import batched_gather

'''
TODO: optimize
'''
def atom14_to_atom37(atom14, aatype):
    L = len(aatype)
    atom37 = np.zeros(atom14.shape[:-2] + (37, 3))
    for i in range(L):
        atom37[...,i,:,:] = atom14[...,i,rc.RESTYPE_ATOM37_TO_ATOM14[aatype[i]],:]
        atom37[...,i,:,:] *= rc.RESTYPE_ATOM37_MASK[aatype[i]][:,None]
    return atom37

'''
TODO: optimize
'''
def atom14_to_atom37(atom14, aatype):
    L = len(aatype)
    atom37 = np.zeros(atom14.shape[:-2] + (37, 3))
    for i in range(L):
        atom37[...,i,:,:] = atom14[...,i,rc.RESTYPE_ATOM37_TO_ATOM14[aatype[i]],:]
        atom37[...,i,:,:] *= rc.RESTYPE_ATOM37_MASK[aatype[i]][:,None]
    return atom37

def atom37_to_frames(atom37):
    n_coords = atom37[:,rc.atom_order['N']]
    ca_coords = atom37[:,rc.atom_order['CA']]
    c_coords = atom37[:,rc.atom_order['C']]
    prot_frames = Rigid.from_3_points(
        c_coords,
        ca_coords,
        n_coords,
    )
    rots = torch.eye(3, device=atom37.device)#[None].repeat(atom14.shape[0], 1, 1)
    rots[0,0] = -1
    rots[2,2] = -1
    rots = Rotation(rot_mats=rots)
    return prot_frames.compose(Rigid(rots, None))


def frames_torsions_to_atom14(frames, torsions, aatype):
    default_frames = torch.from_numpy(rc.restype_rigid_group_default_frame).to(aatype.device)
    lit_positions = torch.from_numpy(rc.restype_atom14_rigid_group_positions).to(aatype.device)
    group_idx = torch.from_numpy(rc.restype_atom14_to_rigid_group).to(aatype.device)
    atom_mask = torch.from_numpy(rc.restype_atom14_mask).to(aatype.device)
    frames_out = torsion_angles_to_frames(frames, torsions, aatype, default_frames)
    return frames_and_literature_positions_to_atom14_pos(frames_out, aatype, default_frames, group_idx, atom_mask, lit_positions)


def frames_torsions_to_atom14(frames, torsions, aatype):
    default_frames = torch.from_numpy(rc.restype_rigid_group_default_frame).to(aatype.device)
    lit_positions = torch.from_numpy(rc.restype_atom14_rigid_group_positions).to(aatype.device)
    group_idx = torch.from_numpy(rc.restype_atom14_to_rigid_group).to(aatype.device)
    atom_mask = torch.from_numpy(rc.restype_atom14_mask).to(aatype.device)
    frames_out = torsion_angles_to_frames(frames, torsions, aatype, default_frames)
    return frames_and_literature_positions_to_atom14_pos(frames_out, aatype, default_frames, group_idx, atom_mask, lit_positions)


def atom37_to_torsions(all_atom_positions, aatype):
    
    all_atom_mask = torch.from_numpy(rc.RESTYPE_ATOM37_MASK[aatype])
    pad = all_atom_positions.new_zeros(
        [*all_atom_positions.shape[:-3], 1, 37, 3]
    )
    prev_all_atom_positions = torch.cat(
        [pad, all_atom_positions[..., :-1, :, :]], dim=-3
    )

    pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, 37])
    prev_all_atom_mask = torch.cat([pad, all_atom_mask[..., :-1, :]], dim=-2)

    pre_omega_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 1:3, :], all_atom_positions[..., :2, :]],
        dim=-2,
    )
    phi_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 2:3, :], all_atom_positions[..., :3, :]],
        dim=-2,
    )
    psi_atom_pos = torch.cat(
        [all_atom_positions[..., :3, :], all_atom_positions[..., 4:5, :]],
        dim=-2,
    )

    pre_omega_mask = torch.prod(
        prev_all_atom_mask[..., 1:3], dim=-1
    ) * torch.prod(all_atom_mask[..., :2], dim=-1)
    phi_mask = prev_all_atom_mask[..., 2] * torch.prod(
        all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype
    )
    psi_mask = (
        torch.prod(all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype)
        * all_atom_mask[..., 4]
    )

    chi_atom_indices = torch.as_tensor(
        get_chi_atom_indices(), device=aatype.device
    )
    
    atom_indices = chi_atom_indices[..., aatype, :, :]
    chis_atom_pos = batched_gather(
        all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2])
    )

    chi_angles_mask = list(rc.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)

    chis_mask = chi_angles_mask[aatype, :]

    chi_angle_atoms_mask = batched_gather(
        all_atom_mask,
        atom_indices,
        dim=-1,
        no_batch_dims=len(atom_indices.shape[:-2]),
    )
    chi_angle_atoms_mask = torch.prod(
        chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype
    )
    chis_mask = chis_mask * chi_angle_atoms_mask

    torsions_atom_pos = torch.cat(
        [
            pre_omega_atom_pos[..., None, :, :],
            phi_atom_pos[..., None, :, :],
            psi_atom_pos[..., None, :, :],
            chis_atom_pos,
        ],
        dim=-3,
    )

    torsion_angles_mask = torch.cat(
        [
            pre_omega_mask[..., None],
            phi_mask[..., None],
            psi_mask[..., None],
            chis_mask,
        ],
        dim=-1,
    )

    torsion_frames = Rigid.from_3_points(
        torsions_atom_pos[..., 1, :],
        torsions_atom_pos[..., 2, :],
        torsions_atom_pos[..., 0, :],
        eps=1e-8,
    )

    fourth_atom_rel_pos = torsion_frames.invert().apply(
        torsions_atom_pos[..., 3, :]
    )

    torsion_angles_sin_cos = torch.stack(
        [fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1
    )

    denom = torch.sqrt(
        torch.sum(
            torch.square(torsion_angles_sin_cos),
            dim=-1,
            dtype=torsion_angles_sin_cos.dtype,
            keepdims=True,
        )
        + 1e-8
    )
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom

    torsion_angles_sin_cos = torsion_angles_sin_cos * all_atom_mask.new_tensor(
        [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
    )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]

    return torsion_angles_sin_cos, torsion_angles_mask


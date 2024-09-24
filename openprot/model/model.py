import torch
import torch.nn as nn
import torch.nn.functional as F
from .gmha import GeometricMultiHeadAttention
from ..utils.rotation_conversions import axis_angle_to_matrix
from .trunk import FoldingTrunk
from ..utils.rigid_utils import Rigid

class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.lin_in = nn.Linear(dim, ff_dim)
        self.act = nn.ReLU()
        self.lin_out = nn.Linear(ff_dim, dim)

    def forward(self, x):
        x = self.lin_in(x)
        x = self.act(x)
        x = self.lin_out(x)
        return x


class OpenProtTransformerBlock(nn.Module):
    def __init__(self, dim, heads, ff_expand, geometric_attn=False, frame_update=False):
        super().__init__()
        self.mha = GeometricMultiHeadAttention(dim, heads, geometric=geometric_attn)
        self.mha_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_expand * dim)
        self.ff_norm = nn.LayerNorm(dim)

        if frame_update:
            self.ff_update = nn.Linear(dim, 6)
        else:
            self.ff_update = None

    def forward(self, x, mask=None, rots=None, trans=None):
        x = x + self.mha(self.mha_norm(x), mask, rots=rots, trans=trans)
        x = x + self.ff(self.ff_norm(x))
        if self.ff_update is not None:
            vec, rotvec = self.ff_update(x).split(3, dim=-1)
            trans = trans + torch.einsum("blij,blj->bli", rots, vec)
            rots = rots @ axis_angle_to_matrix(rotvec).mT
        return x, rots, trans


class OpenProtModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.trunk:
            self.trunk = FoldingTrunk(cfg.trunk)
        
        if cfg.in_norm:
            self.in_norm = nn.LayerNorm(cfg.dim)

        self.blocks = nn.ModuleList()
        for i in range(cfg.blocks):
            self.blocks.append(
                OpenProtTransformerBlock(
                    dim=cfg.dim,
                    heads=cfg.heads,
                    ff_expand=cfg.ff_expand,
                    geometric_attn=i in cfg.geometric_attn,
                    frame_update=i in cfg.frame_update,
                )
            )

    def forward(self, inp):

        
        if self.cfg.trunk:
            seq_feats = inp['x']
            B, L, _ = seq_feats.shape
            pair_feats = seq_feats.new_zeros(B, L, L, self.cfg.trunk.pairwise_state_dim)
            true_aa = torch.zeros(B, L, dtype=torch.long, device=seq_feats.device)
            residx = torch.arange(L, device=seq_feats.device)[None].expand(B, -1)
            mask = inp["pad_mask"]

            out = self.trunk(seq_feats, pair_feats, true_aa, residx, mask, no_recycles=0)

            frames = Rigid.from_tensor_7(out['sm']['frames'])
            return {
                'x': out['s_s'],
                'z': out['s_z'],
                'trans': frames._trans,
                'rots': frames._rots.get_rot_mats(),
            }
        x = inp["x"]
        mask = inp["pad_mask"]
        rots = inp.get("rots", None)
        trans = inp.get("trans", None)
        if self.cfg.in_norm:
            x = self.in_norm(x)
        for block in self.blocks:
            x, rots, trans = block(x, mask, rots=rots, trans=trans)
        return {
            "x": x,
            "rots": rots,
            "trans": trans,
        }

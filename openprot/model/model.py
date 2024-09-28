import torch
import torch.nn as nn
import torch.nn.functional as F
from .gmha import GeometricMultiHeadAttention
from ..utils.rotation_conversions import axis_angle_to_matrix
from .trunk import RelativePosition
from ..utils.rigid_utils import Rigid, Rotation
from .ipa import InvariantPointAttention
from .layers import (
    Dropout,
    PairToSequence,
    SequenceToPair,
)
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from openfold.utils.checkpointing import checkpoint_blocks


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
    def __init__(
        self,
        dim,
        heads,
        ff_expand,
        pairwise_dim=None,
        pairwise_heads=None,
        no_qk_points=None,
        no_v_points=None,
        dropout=0,
        pair=False,
        pair_in=False,
        ipa=False,
        frame_update=False,
    ):
        super().__init__()
        self.cfg = {
            "pair": pair,
            "pair_in": pair_in,
            "ipa": ipa,
            "frame_update": frame_update,
        }
        if ipa:
            self.ipa = InvariantPointAttention(
                c_s=dim,
                c_z=pairwise_dim,
                c_hidden=dim // heads,
                no_heads=heads,
                no_qk_points=no_qk_points,
                no_v_points=no_v_points,
            )

        else:
            self.mha = GeometricMultiHeadAttention(dim, heads, pair=pair_in)

        self.mha_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_expand * dim)
        self.ff_norm = nn.LayerNorm(dim)

        if frame_update:
            self.ff_update_norm = nn.LayerNorm(dim)
            self.ff_update = nn.Linear(dim, 6)
            torch.nn.init.zeros_(self.ff_update.weight)
            torch.nn.init.zeros_(self.ff_update.bias)

        if (pair_in or pair) and (not ipa):  # ipa has its own bias head
            self.pair_to_sequence = PairToSequence(pairwise_dim, heads)

        if pair:
            self.sequence_to_pair = SequenceToPair(dim, pairwise_dim // 2, pairwise_dim)
            self.pair_to_sequence = PairToSequence(pairwise_dim, heads)
            self.tri_mul_out = TriangleMultiplicationOutgoing(
                pairwise_dim, pairwise_dim
            )
            self.tri_mul_in = TriangleMultiplicationIncoming(pairwise_dim, pairwise_dim)
            self.mlp_pair = FeedForward(pairwise_dim, ff_expand * pairwise_dim)

            self.drop = nn.Dropout(dropout)
            self.row_drop = Dropout(dropout * 2, 2)
            self.col_drop = Dropout(dropout * 2, 1)

            torch.nn.init.zeros_(self.tri_mul_in.linear_z.weight)
            torch.nn.init.zeros_(self.tri_mul_in.linear_z.bias)
            torch.nn.init.zeros_(self.tri_mul_out.linear_z.weight)
            torch.nn.init.zeros_(self.tri_mul_out.linear_z.bias)

            torch.nn.init.zeros_(self.sequence_to_pair.o_proj.weight)
            torch.nn.init.zeros_(self.sequence_to_pair.o_proj.bias)

    def forward(self, x, z, mask, rots, trans):

        if (self.cfg["pair"] or self.cfg["pair_in"]) and not self.cfg["ipa"]:
            bias = self.pair_to_sequence(z)
        else:
            bias = torch.zeros_like(mask)

        if self.cfg["ipa"]:
            x = x + self.ipa(
                self.mha_norm(x),
                z,
                Rigid(trans=trans, rots=Rotation(rot_mats=rots)),
                mask,
            )
        else:
            x = x + self.mha(
                self.mha_norm(x),
                mask.bool(),
                z=z,
                rots=rots,
                trans=trans,
                bias=bias.permute(0, 3, 1, 2),
            )

        x = x + self.ff(self.ff_norm(x))

        if self.cfg["pair"]:
            z = z + self.sequence_to_pair(x)
            tri_mask = (
                mask.unsqueeze(2) * mask.unsqueeze(1) if mask is not None else None
            )
            z = z + self.row_drop(self.tri_mul_out(z, mask=tri_mask))
            z = z + self.col_drop(self.tri_mul_in(z, mask=tri_mask))
            z = z + self.mlp_pair(z)

        if self.cfg["frame_update"]:
            vec, rotvec = self.ff_update(self.ff_update_norm(x)).split(3, dim=-1)
            trans = trans + torch.einsum("blij,blj->bli", rots, vec)
            rots = rots @ axis_angle_to_matrix(rotvec).mT

        return x, z, mask, rots, trans


class OpenProtModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.pairwise_positional_embedding = RelativePosition(
            cfg.position_bins, cfg.pairwise_dim
        )

        self.blocks = nn.ModuleList()
        for i in range(cfg.blocks):
            self.blocks.append(
                OpenProtTransformerBlock(
                    dim=cfg.dim,
                    heads=cfg.heads,
                    ff_expand=cfg.ff_expand,
                    pairwise_dim=cfg.pairwise_dim,
                    pairwise_heads=cfg.pairwise_heads,
                    pair=True,
                    ipa=False,
                )
            )

        self.ipa_block = OpenProtTransformerBlock(
            dim=cfg.dim,
            heads=cfg.heads,
            ff_expand=cfg.ff_expand,
            pairwise_dim=cfg.pairwise_dim,
            no_qk_points=cfg.no_qk_points,
            no_v_points=cfg.no_v_points,
            ipa=True,
            frame_update=True,
        )

    def forward(self, inp):

        x = inp["x"]
        B, L, _ = x.shape
        residx = torch.arange(L, device=x.device)[None].expand(B, -1)
        mask = inp["pad_mask"]
        z = self.pairwise_positional_embedding(residx, mask=mask)

        rots = inp.get("rots", None)
        trans = inp.get("trans", None)

        for block in self.blocks:
            if block.cfg["pair"]:
                x, z, mask, rots, trans = torch.utils.checkpoint.checkpoint(
                    block, x, z, mask, rots, trans
                )
            else:
                x, z, mask, rots, trans = block(x, z, mask, rots, trans)

        all_rots = [rots]
        all_trans = [trans]
        for block in range(self.cfg.ipa_blocks):
            x, z, mask, rots, trans = self.ipa_block(x, z, mask, rots, trans)
            all_rots.append(rots)
            all_trans.append(trans)
            rots = rots.detach()
            trans = trans.detach()

        return {
            "x": x,
            "z": z,
            "rots": torch.stack(all_rots),
            "trans": torch.stack(all_trans),
        }

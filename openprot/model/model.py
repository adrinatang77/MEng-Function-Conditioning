import torch
import torch.nn as nn
import torch.nn.functional as F
from .gmha import GeometricMultiHeadAttention
from ..utils.rotation_conversions import axis_angle_to_matrix
from ..utils.rigid_utils import Rigid, Rotation
from .layers import (
    Dropout,
    PairToSequence,
    SequenceToPair,
)
from .tri_mul import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from ..utils.checkpointing import checkpoint_blocks


def modulate(x, shift, scale):
    if shift is not None:
        return x * (1 + scale) + shift
    else:
        return x


def gate(x, gate_):
    if gate_ is not None:
        return x * gate_
    else:
        return x


class RelativePosition(nn.Module):
    def __init__(self, bins, pairwise_state_dim):
        super().__init__()
        self.bins = bins

        # Note an additional offset is used so that the 0th position
        # is reserved for masked pairs.
        self.embedding = torch.nn.Embedding(2 * bins + 2, pairwise_state_dim)

    def forward(self, residue_index, mask=None):
        """
        Input:
          residue_index: B x L tensor of indices (dytpe=torch.long)
          mask: B x L tensor of booleans

        Output:
          pairwise_state: B x L x L x pairwise_state_dim tensor of embeddings
        """

        assert residue_index.dtype == torch.long
        if mask is not None:
            assert residue_index.shape == mask.shape

        diff = residue_index[:, None, :] - residue_index[:, :, None]
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1  # Add 1 to adjust for padding index.

        if mask is not None:
            mask = mask[:, None, :] * mask[:, :, None]
            diff[mask == False] = 0

        output = self.embedding(diff)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim, layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.LayerNorm(dim))
        self.layers.append(nn.Linear(dim, ff_dim))
        for i in range(layers - 2):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(ff_dim, ff_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(ff_dim, dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class OpenProtTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        ff_expand=4,
        ff_layers=2,
        pairwise_dim=128,
        pairwise_heads=4,
        rope=False,  # rotate scalar queries and keys
        pair_bias=False,  # use pairs to bias
        pair_updates=False,
        pair_ffn=False,
        pair_ff_expand=1,
        tri_mul=False,
        pair_values=False,  # aggregate values from pair reps
        ipa_attn=False,  # use point attention
        ipa_values=False,
        ipa_frames=False,  # use frames in point attention
        relpos_attn=False,  # instead use trans relpos
        relpos_values=False,
        embed_rots=False,
        no_qk_points=4,
        no_v_points=8,
        frame_update=False,
        update_rots=False,
        readout_rots=False,
        update_x=True,
        adaLN=False,
        rots_type="quat",
    ):
        super().__init__()
        self.mha = GeometricMultiHeadAttention(
            dim=dim,
            heads=heads,
            pairwise_dim=pairwise_dim,
            rope=rope,
            pair_bias=pair_bias,
            pair_values=pair_values,
            ipa_attn=ipa_attn,
            ipa_values=ipa_values,
            ipa_frames=ipa_frames,
            relpos_attn=relpos_attn,
            relpos_values=relpos_values,
            embed_rots=embed_rots,
            no_qk_points=no_qk_points,
            no_v_points=no_v_points,
        )
        self.ff = FeedForward(dim, ff_expand * dim, layers=ff_layers)

        self.pair_updates = pair_updates
        self.pair_ffn = pair_ffn
        self.frame_update = frame_update
        self.update_rots = update_rots
        self.rots_type = rots_type
        self.readout_rots = readout_rots
        self.tri_mul = tri_mul
        self.update_x = update_x
        self.adaLN = adaLN

        self.mha_norm = nn.LayerNorm(dim, elementwise_affine=not adaLN)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=not adaLN)

        if adaLN:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        if pair_bias or pair_values:
            self.pair_norm = nn.LayerNorm(pairwise_dim)

        rot_dim = {"quat": 4, "vec": 3}[rots_type]
        if frame_update:
            self.linear_frame_update = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 3 + rot_dim if update_rots else 3),
            )
            torch.nn.init.zeros_(self.linear_frame_update[-1].weight)
            torch.nn.init.zeros_(self.linear_frame_update[-1].bias)

        if readout_rots:
            self.linear_rots_out = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, rot_dim)
            )
        if pair_updates:
            self.sequence_to_pair = SequenceToPair(dim, pairwise_dim // 2, pairwise_dim)

            if tri_mul:
                self.tri_mul_out = TriangleMultiplicationOutgoing(
                    pairwise_dim, pairwise_dim
                )
                self.tri_mul_in = TriangleMultiplicationIncoming(
                    pairwise_dim, pairwise_dim
                )
                dropout = 0
                self.row_drop = Dropout(dropout * 2, 2)
                self.col_drop = Dropout(dropout * 2, 1)
                torch.nn.init.zeros_(self.tri_mul_in.linear_z.weight)
                torch.nn.init.zeros_(self.tri_mul_in.linear_z.bias)
                torch.nn.init.zeros_(self.tri_mul_out.linear_z.weight)
                torch.nn.init.zeros_(self.tri_mul_out.linear_z.bias)

            if pair_ffn:
                self.mlp_pair = FeedForward(pairwise_dim, pair_ff_expand * pairwise_dim)
                self.pair_ff_norm = nn.LayerNorm(pairwise_dim)

            torch.nn.init.zeros_(self.sequence_to_pair.o_proj.weight)
            torch.nn.init.zeros_(self.sequence_to_pair.o_proj.bias)

    def forward(self, x, z, rots, trans, mask, x_cond=None):

        ### no pair2sequence

        if self.adaLN:
            shift_mha, scale_mha, gate_mha, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(x_cond).chunk(6, dim=-1)
            )
        else:
            shift_mha, scale_mha, gate_mha, shift_mlp, scale_mlp, gate_mlp = [None] * 6

        x_in = x
        x = x + gate(
            self.mha(
                x=modulate(self.mha_norm(x), shift_mha, scale_mha),
                z=self.pair_norm(z) if hasattr(self, "pair_norm") else None,
                mask=mask.bool(),
                trans=trans,
                rots=rots,
            ),
            gate_mha,
        )

        x = x + gate(self.ff(modulate(self.ff_norm(x), shift_mlp, scale_mlp)), gate_mlp)

        if self.pair_updates:
            z = z + self.sequence_to_pair(x)
            if self.tri_mul:
                tri_mask = (
                    mask.unsqueeze(2) * mask.unsqueeze(1) if mask is not None else None
                )
                z = z + self.row_drop(self.tri_mul_out(z, mask=tri_mask))
                z = z + self.col_drop(self.tri_mul_in(z, mask=tri_mask))
            if self.pair_ffn:
                z = z + self.mlp_pair(self.pair_ff_norm(z))

        if self.frame_update:

            if self.update_rots:
                update = self.linear_frame_update(x)
                vec, rotvec = update[..., :3], update[..., 3:]
                trans = trans + torch.einsum("blij,blj->bli", rots, vec)
                if self.rots_type == "vec":
                    rot_update = axis_angle_to_matrix(rotvec)
                elif self.rots_type == "quat":
                    rot_update = Rotation(
                        quats=rotvec, normalize_quats=True
                    ).get_rot_mats()
                rots = rots @ rot_update.mT
            else:
                vec = self.linear_frame_update(x)
                trans = trans + vec

        if self.readout_rots:
            rotvec = self.linear_rots_out(x)
            if self.rots_type == "quat":
                rots = Rotation(quats=rotvec, normalize_quats=True).get_rot_mats()
            elif self.rots_type == "vec":
                rots = axis_angle_to_matrix(rotvec)

        if not self.update_x:
            x = x_in
        return x, z, rots, trans


class OpenProtModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.pairwise_pos_emb:
            self.pairwise_positional_embedding = RelativePosition(
                cfg.position_bins, cfg.pairwise_dim
            )

        ipa_args = dict(
            ipa_attn=cfg.ipa_attn,
            ipa_values=cfg.ipa_values,
            ipa_frames=cfg.ipa_frames,
            relpos_attn=cfg.relpos_attn,
            relpos_values=cfg.relpos_values,
            embed_rots=cfg.embed_rots,
            no_qk_points=cfg.no_qk_points,
            no_v_points=cfg.no_v_points,
            update_rots=cfg.update_rots,
            readout_rots=cfg.readout_rots,
            rots_type=cfg.rots_type,
            update_x=cfg.update_x,
        )
        self.blocks = nn.ModuleList()
        for i in range(cfg.blocks):
            is_pair = (i + 1) % cfg.pair_block_interval == 0
            is_ipa = (i + 1) % cfg.ipa_block_interval == 0
            self.blocks.append(
                OpenProtTransformerBlock(
                    dim=cfg.dim,
                    heads=cfg.heads,
                    ff_expand=cfg.ff_expand,
                    pairwise_dim=cfg.pairwise_dim,
                    pairwise_heads=cfg.pairwise_heads,
                    rope=cfg.rope,
                    pair_bias=is_pair,
                    pair_updates=is_pair,
                    pair_values=is_pair and cfg.pair_values,
                    pair_ffn=is_pair and cfg.pair_ffn,
                    pair_ff_expand=cfg.pair_ff_expand,
                    tri_mul=is_pair and cfg.tri_mul,
                    adaLN=cfg.trunk_adaLN,
                    **(ipa_args if is_ipa else {}),
                )
            )

        block_fn = lambda: OpenProtTransformerBlock(
            dim=cfg.dim,
            heads=cfg.heads,
            ff_expand=cfg.ff_expand,
            pairwise_dim=cfg.pairwise_dim,
            pair_bias=cfg.ipa_pair_bias,
            pair_values=cfg.ipa_pair_values,
            adaLN=cfg.sm_adaLN,
            frame_update=cfg.frame_update,
            **ipa_args,
        )
        if cfg.separate_ipa_blocks:
            self.ipa_blocks = nn.ModuleList()
            for i in range(cfg.ipa_blocks):
                self.ipa_blocks.append(block_fn())
        elif self.cfg.ipa_blocks > 0:
            self.ipa_block = block_fn()

        if cfg.embed_trans_before_ipa:
            self.trans_in = nn.Linear(3, cfg.dim)

    def forward(self, inp):

        x = inp["x"]
        B, L, _ = x.shape
        residx = torch.arange(L, device=x.device)[None].expand(B, -1)
        mask = inp["pad_mask"]
        z = inp.get("z", 0)
        if self.cfg.pairwise_pos_emb:
            self.pairwise_positional_embedding(residx, mask=mask)

        rots = inp.get("rots", None)
        trans = inp.get("trans", None)
        x_cond = inp.get("x_cond", None)

        for block in self.blocks:
            if block.pair_updates and self.cfg.checkpoint:
                x, z, rots, trans = torch.utils.checkpoint.checkpoint(
                    block, x, z, rots, trans, mask, x_cond
                )
            else:
                x, z, rots, trans = block(x, z, rots, trans, mask, x_cond)

        if self.cfg.detach_before_ipa:
            x = x.detach()

        if self.cfg.embed_trans_before_ipa:
            x = x + self.trans_in(trans)

        if self.cfg.augment_before_ipa:
            R, B, L, D = x.shape
            x = x.reshape(R * B, L, D)
            x_cond = x_cond[None].expand(R, -1, -1, -1).reshape(R * B, L, D)
            mask = mask[None].expand(R, -1, -1).reshape(R * B, L)
            z_ = z[None].expand(R, -1, -1, -1, -1).reshape(R * B, L, L, -1)
        else:
            z_ = z

        if self.cfg.zero_frames_before_ipa:
            trans = torch.zeros_like(trans)
            rots = torch.zeros_like(rots) + torch.eye(
                3, device=rots.device, dtype=rots.dtype
            )

        all_rots = [rots]
        all_trans = [trans]
        for i in range(self.cfg.ipa_blocks):
            if self.cfg.separate_ipa_blocks:
                block = self.ipa_blocks[i]
            else:
                block = self.ipa_block

            x, z_, rots, trans = block(x, z_, rots, trans, mask, x_cond)
            all_rots.append(rots)
            all_trans.append(trans)

            if self.cfg.detach_rots:
                rots = rots.detach()
            if self.cfg.detach_trans:
                trans = trans.detach()

        if self.cfg.augment_before_ipa:
            x = x.reshape(R, B, L, D)
        return {
            "x": x,
            "z": z,
            "rots": torch.stack(all_rots) if rots is not None else None,
            "trans": torch.stack(all_trans),
        }

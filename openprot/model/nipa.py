import torch
from torch import nn
import numpy as np
import math
from openfold.model.primitives import ipa_point_weights_init_
from openfold.utils.tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
)
def sinusoidal_embedding(pos, n_freqs, max_period, min_period):
    periods = torch.exp(torch.linspace(
        math.log(min_period), 
        math.log(max_period), 
        n_freqs, 
        device=pos.device
    ))
    freqs = 2 * np.pi / periods
    return torch.cat(
        [torch.cos(pos[..., None] * freqs), torch.sin(pos[..., None] * freqs)], -1
    )

class NonInvariantPointAttention(nn.Module):
    def __init__(self, dim, pairwise_dim, heads, no_qk_points=4, no_v_points=8):
        super().__init__()
        self.heads = heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.w_z = nn.Linear(heads * pairwise_dim, dim)
        self.linear_b = nn.Linear(pairwise_dim, heads)
        # self.embed_trans = nn.Sequential(nn.Linear(3, dim), nn.ReLU(), nn.Linear(dim, dim))
        # self.embed_rots = nn.Sequential(nn.Linear(9, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.w_o = nn.Linear(dim, dim, bias=False)
        
        # self.w_qr = nn.Linear(dim, heads * 3 * 64, bias=False)
        # self.w_r = nn.Linear(heads * 3 * 64, dim, bias=False)

        #######
        hpq = heads * no_qk_points * 3
        self.linear_q_points = nn.Linear(dim, hpq)

        hpkv = heads * (no_qk_points + no_v_points) * 3
        self.linear_kv_points = nn.Linear(dim, hpkv)

        hpv = heads * no_v_points * 3

        self.head_weights = nn.Parameter(torch.zeros((heads)))
        ipa_point_weights_init_(self.head_weights)
        self.softplus = nn.Softplus()
        self.w_pt = nn.Linear(heads * no_v_points * 4, dim)
        #####
                                     

    def forward(self, x, z, rigids, mask):
        B, L, D = x.shape

        trans = rigids._trans
        rots = rigids._rots._rot_mats
        # x = x + self.embed_rots(rots.view(B, L, -1))
        
        query = self.w_q(x).view(B, L, self.heads, -1).transpose(1, 2)  # B H L D
        key = self.w_k(x).view(B, L, self.heads, -1).transpose(1, 2)
        attn = query @ key.mT / math.sqrt(D)  # B H L L




        ###################

        
        ##############################
        q_pts = self.linear_q_points(x)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)

        q_pts = trans[:,:,None] + q_pts

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(q_pts.shape[:-2] + (self.heads, self.no_qk_points, 3))

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(x)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = trans[:,:,None] + kv_pts

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        # [*, N_res, N_res, H, P_q, 3]
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_att**2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )

        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        
        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))
        attn = attn + pt_att

        ################################
        
        """
        relpos_query = self.w_qr(x).view(B, L, self.heads, -1).transpose(1, 2)
        relpos = trans[:,None] - trans[:,:,None]
        relpos_emb = sinusoidal_embedding(relpos, n_freqs=32, max_period=100, min_period=1)
        relpos_emb = relpos_emb.view(B, L, L, -1)
        
        relpos_attn = torch.einsum('bhid,bijd->bhij', relpos_query, relpos_emb) / math.sqrt(3 * 64)
        attn = attn + relpos_attn
        
        
        """
        
        bias = self.linear_b(z).permute(0, 3, 1, 2)
        if mask is not None:
            mask = mask.view(B, 1, 1, -1)
        if bias is not None:
            if mask is not None:
                mask = torch.where(mask.bool(), 0, -float("inf")) + bias
            else:
                mask = bias
        if mask is not None:
            attn = attn + mask

        attn = torch.softmax(attn, dim=-1)

        ## scalar values
        value = self.w_v(x).view(B, L, self.heads, -1).transpose(1, 2)
        out = (attn @ value).transpose(1, 2).reshape(B, L, D)

        # relpos_out = torch.einsum('bhij,bijd->bihd', attn, relpos_emb).reshape(B, L, -1)
        ########
        o_pt = torch.sum(
            (
                attn[..., None, :, :, None]
                * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = o_pt - trans[:,:,None,None]

        # [*, N_res, H * P_v]
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt**2, dim=-1) + 1e-6), 2
        )

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)
        ########
        z_out = torch.einsum("bhij,bijd->bihd", attn, z).reshape(B, L, -1)
        return self.w_o(out) + self.w_z(z_out) + self.w_pt(
            torch.cat((*torch.unbind(o_pt, dim=-1), o_pt_norm), dim=-1)
        )
                    # + self.w_r(relpos_out)

        
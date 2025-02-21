import pytorch_lightning as pl
import torch, time, os
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import pickle
from ..utils.rigid_utils import Rigid, Rotation
from collections import defaultdict
from functools import partial
from multiflow.flow_model import FlowModel
from multiflow.data.interpolant import Interpolant
from multiflow.data import utils as du
from multiflow.data import so3_utils, all_atom
# from .utils import frames_to_pdb, trans_to_pdb, compute_lddt, compute_fape, compute_distogram_loss, upgrade_state_dict
# from proteinblobs.utils import atom37_to_pdb
# from proteinblobs.designability_utils import run_designability
# from .blobs import blob_likelihood, reblob_jsd, blob_accuracy, soft_blob_accuracy, blob_coverage, blob_occupancy, blob_misplacement
# from .wrapper import Wrapper

class MultiflowWrapper(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = FlowModel(cfg.model)
        self.interpolant = Interpolant(cfg.interpolant)
        self.aatype_pred_num_tokens = cfg.model.aatype_pred_num_tokens
        self._exp_cfg = cfg.experiment
        self.rng = np.random.default_rng(137)
        
    def general_step(self, batch):
        
        self.interpolant.set_device(self.device)
        
        ## translation
        batch['trans_1']  = batch['trans']
        batch['rotmats_1'] = batch['rots']
        batch['aatypes_1'] = batch['seqres']
        if 'diffuse_mask' not in batch:
            batch['diffuse_mask'] = torch.ones_like(batch['res_mask'])
        noisy_batch = self.interpolant.corrupt_batch(batch)
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['mask'] * noisy_batch['diffuse_mask']
        
        # if training_cfg.mask_plddt:
        #     loss_mask *= noisy_batch['plddt_mask']
        
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
            
        num_batch, num_res = loss_mask.shape
        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        gt_aatypes_1 = noisy_batch['aatypes_1']
        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = so3_utils.calc_rot_vf(
            rotmats_t, gt_rotmats_1.type(torch.float32))
        gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3] 
        
        # Timestep used for normalization.
        r3_t = noisy_batch['r3_t'] # (B, 1)
        so3_t = noisy_batch['so3_t'] # (B, 1)
        cat_t = noisy_batch['cat_t'] # (B, 1)
        r3_norm_scale = 1 - torch.min(
            r3_t[..., None], torch.tensor(training_cfg.t_normalize_clip)) # (B, 1, 1)
        so3_norm_scale = 1 - torch.min(
            so3_t[..., None], torch.tensor(training_cfg.t_normalize_clip)) # (B, 1, 1)
        if training_cfg.aatypes_loss_use_likelihood_weighting:
            cat_norm_scale = 1 - torch.min(
                cat_t, torch.tensor(training_cfg.t_normalize_clip)) # (B, 1)
            assert cat_norm_scale.shape == (num_batch, 1)
        else:
            cat_norm_scale = 1.0

        # if self.rng.random() > 0.5:
        #     with torch.no_grad():
        #         with torch.cuda.amp.autocast(False):
        #             model_sc = self.model(noisy_batch)
        #         noisy_batch['trans_sc'] = (
        #             model_sc['pred_trans'] * noisy_batch['diffuse_mask'][..., None]
        #             + noisy_batch['trans_1'] * (1 - noisy_batch['diffuse_mask'][..., None])
        #         )
        #         logits_1 = torch.nn.functional.one_hot(
        #             batch['aatypes_1'].long(), num_classes=self.aatype_pred_num_tokens).float()
        #         noisy_batch['aatypes_sc'] = (
        #             model_sc['pred_logits'] * noisy_batch['diffuse_mask'][..., None]
        #             + logits_1 * (1 - noisy_batch['diffuse_mask'][..., None])
        #         )
            
        model_output = self.model(noisy_batch)
        # baseline loss
        # 'train/mutliflow/aatypes_loss': 3.044921875,
        # 'train/mutliflow/trans_loss': 2.8262554204463957,
        # pred_trans_1 = noisy_batch['trans_t'] + 0.0 * model_output['pred_trans']
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_logits = model_output['pred_logits'] # (B, N, aatype_pred_num_tokens)
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        
        ce_loss = torch.nn.functional.cross_entropy(
            pred_logits.reshape(-1, self.aatype_pred_num_tokens),
            gt_aatypes_1.flatten().long(),
            reduction='none',
        ).reshape(num_batch, num_res) / cat_norm_scale
        
        aatypes_loss = ce_loss * loss_mask # torch.sum(ce_loss * loss_mask, dim=-1) / (loss_denom / 3)
        aatypes_loss *= training_cfg.aatypes_loss_weight
        
        # Translation VF loss
        trans_error = (gt_trans_1 - pred_trans_1) / r3_norm_scale * training_cfg.trans_scale
        
        trans_loss = (trans_error**2 * loss_mask[...,None]).sum(-1) / 3
        trans_loss *= training_cfg.translation_loss_weight
        
        trans_loss = torch.clamp(trans_loss, max=5)
        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / so3_norm_scale
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None],
            dim=(-1),# -2)
        ) / 3 # loss_denom

        
        # trans_loss = torch.nan_to_num(trans_loss, 0.0)
        # rots_vf_loss = torch.nan_to_num(rots_vf_loss, 0.0)
        # aatypes_loss = torch.nan_to_num(aatypes_loss, 0.0)
        
        loss = trans_loss + rots_vf_loss + aatypes_loss
        
        return {
            'trans_loss': trans_loss,
            'rots_vf_loss': rots_vf_loss,
            # 'auxiliary_loss': auxiliary_loss,
            'aatypes_loss': aatypes_loss
        }

  
    def inference(self, batch, trans_0=None, rotmats_0=None):
        interpolant = Interpolant(self.cfg.inference.interpolant) 
        interpolant.set_device(self.device)

        true_bb_pos = None
        trans_1 = rotmats_1 = diffuse_mask = aatypes_1 = true_aatypes = None
        num_batch, sample_length = batch['res_mask'].shape
        
        prot_traj, model_traj = interpolant.sample(
            batch['res_mask'],
            self.model,
            trans_0=trans_0,
            rotmats_0=rotmats_0,
            trans_1=trans_1,
            rotmats_1=rotmats_1,
            aatypes_1=aatypes_1,
            diffuse_mask=diffuse_mask,
            forward_folding=False,
            inverse_folding=False,
            separate_t=self.cfg.inference.interpolant.codesign_separate_t,
        )
        diffuse_mask = diffuse_mask if diffuse_mask is not None else torch.ones(1, sample_length)
        atom37_traj = [x[0] for x in prot_traj]
        atom37_model_traj = [x[0] for x in model_traj]

        bb_trajs = du.to_numpy(torch.stack(atom37_traj, dim=0).transpose(0, 1))
        noisy_traj_length = bb_trajs.shape[1]
        assert bb_trajs.shape == (num_batch, noisy_traj_length, sample_length, 37, 3)

        model_trajs = du.to_numpy(torch.stack(atom37_model_traj, dim=0).transpose(0, 1))
        clean_traj_length = model_trajs.shape[1]
        assert model_trajs.shape == (num_batch, clean_traj_length, sample_length, 37, 3)

        aa_traj = [x[1] for x in prot_traj]
        clean_aa_traj = [x[1] for x in model_traj]

        aa_trajs = du.to_numpy(torch.stack(aa_traj, dim=0).transpose(0, 1).long())
        assert aa_trajs.shape == (num_batch, noisy_traj_length, sample_length)

        for i in range(aa_trajs.shape[0]):
            for j in range(aa_trajs.shape[2]):
                if aa_trajs[i, -1, j] == du.MASK_TOKEN_INDEX:
                    print("WARNING mask in predicted AA")
                    aa_trajs[i, -1, j] = 0
        clean_aa_trajs = du.to_numpy(torch.stack(clean_aa_traj, dim=0).transpose(0, 1).long())
        assert clean_aa_trajs.shape == (num_batch, clean_traj_length, sample_length)

        return bb_trajs[:,-1], clean_aa_trajs[:,-1]


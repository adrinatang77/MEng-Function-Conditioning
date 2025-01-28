import pytorch_lightning as pl
from ..utils.logger import Logger
import torch
from .model import OpenProtModel
from abc import abstractmethod
from ..utils.misc_utils import autoimport
import os
import tqdm
from ..tracks.manager import OpenProtTrackManager
from ..utils import residue_constants as rc
from ..utils.tensor_utils import tensor_tree_map
from .multiflow_wrapper import MultiflowWrapper
from omegaconf import OmegaConf
from .ema import ExponentialMovingAverage
from multiflow.data import so3_utils
# from ..evals.manager import OpenProtEvalManager


class Wrapper(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters("cfg")
        self.cfg = cfg
        self._logger = Logger(cfg.logger)
        self.ema = None
        self.cached_weights = None
        
    def training_step(self, batch, batch_idx):
        self._logger.prefix = "train"
        out = self.general_step(batch)
        self._logger.step(self.trainer, "train")
        return out

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.cfg.model.ema:
            if self.cached_weights is None:
                self.load_ema_weights()
        self._logger.prefix = "val"
        # self.general_step(batch)
        self.validation_step_extra(batch, batch_idx)
        self._logger.step(self.trainer, "val")

    @abstractmethod
    def general_step(self, batch):
        NotImplemented

    def validation_step_extra(self, batch, batch_idx):
        pass

    def on_train_epoch_end(self):
        self._logger.epoch_end(self.trainer, prefix="train")

    def on_validation_epoch_end(self):
        self._logger.epoch_end(self.trainer, prefix="val")
        if self.cfg.model.ema:
            self.restore_cached_weights()

    def load_ema_weights(self):
        clone_param = lambda t: t.detach().clone()
        self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
        self.model.load_state_dict(self.ema.state_dict()["params"])

    def restore_cached_weights(self):
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None
        
    def on_before_zero_grad(self, *args, **kwargs):
        if self.cfg.model.ema:
            self.ema.update(self.model)
            
    # uncomment this to debug
    def on_before_optimizer_step(self, optimizer):
        quit = False
        for name, p in self.model.named_parameters():
            if p.requires_grad and p.grad is None:
                print(name, "has no grad")
                quit = True

        if quit:
            exit()

    def configure_optimizers(self):
        cls = getattr(torch.optim, self.cfg.optimizer.type)
        all_params = filter(lambda p: p.requires_grad, self.model.parameters())
        # TEMPORARY AND HACKY
        # post_params = self.model.blocks[12:].parameters()
        optimizer = cls(
            # [
            #     {"params": list(set(all_params) - set(post_params))},
            #     {"params": post_params, "lr": self.cfg.optimizer.lr*8},
            # ],
            all_params,
            lr=self.cfg.optimizer.lr,
        )

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-12,
            end_factor=1.0,
            total_iters=self.cfg.optimizer.scheduler.warmup_steps,
        )
        decay = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=self.cfg.optimizer.scheduler.end_factor,
            total_iters=int(0.9 * self.cfg.optimizer.scheduler.total_steps),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, decay],
            milestones=[self.cfg.optimizer.scheduler.start_decay],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

        return optimizer


class OpenProtWrapper(Wrapper):
    def __init__(self, cfg, tracks: OpenProtTrackManager, evals: dict):
        super().__init__(cfg)
        
        if cfg.model.multiflow:
            
            mf_cfg = OmegaConf.load(cfg.model.multiflow_cfg)
            self.model = MultiflowWrapper(mf_cfg)
            if cfg.model.multiflow_ckpt:
                ckpt = torch.load(cfg.model.multiflow_ckpt, map_location=self.device)
                self.model.load_state_dict(ckpt['state_dict'], strict=True)
        
            
        else:
            self.model = OpenProtModel(cfg.model)
            tracks.add_modules(self.model)

        self.tracks = tracks
        self.evals = evals
        if self.cfg.model.ema:
            self.ema = ExponentialMovingAverage(self.model, self.cfg.model.ema_decay)

        
    def on_save_checkpoint(self, checkpoint):
        esm_keys = {k for k in checkpoint['state_dict'].items() if "model.esm." in k}
        checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if k not in esm_keys}

        if self.cached_weights is not None:
            self.restore_cached_weights()
        if self.cfg.model.ema:
            checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        state_dict = self.state_dict()
        esm_keys = {k for k in state_dict.items() if "model.esm." in k}
        checkpoint['state_dict'] |= {k: state_dict[k] for k in esm_keys}
        if self.cfg.model.ema:
            ema = checkpoint["ema"]
            self.ema.load_state_dict(ema)
            
        
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if self.cfg.model.ema:
            if self.ema.device != device:
                self.ema.to(device)
        return batch.to(device)

    def get_lr(self):
        for param_group in self.optimizers().param_groups:
            return param_group["lr"]

    def forward(self, noisy_batch):
        
        ## embed the tracks into an input dict
        inp = self.tracks.embed(self.model, noisy_batch)

        ## run it thorugh the model
        out = self.model(inp)

        ## place the readouts in a dict
        readout = self.tracks.readout(self.model, inp, out)

        return out, readout

    def general_step(self, batch):

        
        # if self.cfg.model.multiflow:
        #     L = batch['pad_mask'].shape[1]
        #     mf_batch = {
        #         'trans': batch['struct'],
        #         'rots': batch['rots'],
        #         'seqres': batch['aatype'],
        #         'res_mask': batch['struct_mask'],
        #         'diffuse_mask': batch['struct_mask'],
        #         'mask': batch['struct_mask'],
        #         'chain_idx': torch.zeros_like(batch['pad_mask'], dtype=torch.long),
        #         'res_idx': batch['residx'],
        #     }
        #     losses = self.model.general_step(mf_batch)
        #     loss = 0
        #     for key in losses:
        #         loss += losses[key]
        #         self._logger.log(f"mutliflow/{key}", losses[key], batch['struct_mask'])
        #     self._logger.log("loss", loss, batch['struct_mask'])
        #     return (loss * batch["struct_mask"]).sum() / batch["struct_mask"].sum()

        batch['pad_mask'] = batch['seq_mask'] = batch['struct_mask'] # to make losses comparable
        self._logger.register_masks(batch)
        self._logger.masked_log("toks", batch["pad_mask"], sum=True)
    
        ## corrupt all the tracks
        noisy_batch, target = self.tracks.corrupt(batch, logger=self._logger)

        if self.cfg.model.multiflow:
            B = batch['pad_mask'].shape[0]
            L = batch['pad_mask'].shape[1]
            mf_batch = {
                # 'trans': batch['struct'],
                # 'rots': batch['rots'],
                # 'seqres': batch['aatype'],
                # 'mask': batch['struct_mask'],
                'res_mask': batch['struct_mask'],
                'diffuse_mask': batch['struct_mask'],
                'chain_idx': torch.zeros_like(batch['pad_mask'], dtype=torch.long),
                'res_idx': batch['residx']
            } | {
                # 'trans_t': noisy_batch['struct'],
                # 'rotmats_t': noisy_batch['rots'],
                'aatypes_t': noisy_batch['aatype'],
                # 'so3_t': noisy_batch['struct_noise'],
                # 'r3_t': noisy_batch['struct_noise'],
                'cat_t': torch.zeros_like(noisy_batch['struct_noise']),
                'trans_sc': torch.zeros_like(noisy_batch['struct']),
                'aatypes_sc': noisy_batch['struct'].new_zeros(B, L, 21),
            }
            
            interpolant_batch = {
                'trans_1': batch['struct'],
                'rotmats_1': batch['rots'],
                'aatypes_1': batch['aatype'],
                'res_mask': batch['struct_mask'],
                'diffuse_mask': batch['struct_mask'],
            }
            # breakpoint()
            self.model.interpolant._device = self.device
            interpolant_out = self.model.interpolant.corrupt_batch(interpolant_batch)
            
            mf_batch['trans_t'] = interpolant_out['trans_t']
            target['noisy_rots'] = mf_batch['rotmats_t'] = interpolant_out['rotmats_t']
            mf_batch['r3_t'] = mf_batch['so3_t'] = interpolant_out['r3_t']

            target['rots_noise'] = target['struct_noise'] = torch.ones_like(target['struct_noise']) - interpolant_out['r3_t']
            
            out = self.model.model(mf_batch)
            readout = {
                'trans': out['pred_trans'][None],
                'rots': out['pred_rotmats'],
                'aatype': out['pred_logits'],
            }

            loss_batch = {
                'gt_trans': batch['struct'],
                'gt_rots': batch['rots'],
                'gt_aatype': batch['aatype'],
                'noisy_rots': mf_batch['rotmats_t'],
                'pred_trans': readout['trans'][0],
                'pred_rots': readout['rots'],
                'pred_aatype': readout['aatype'],
                'r3_t': mf_batch['r3_t'],
                'so3_t': mf_batch['so3_t'],
                'cat_t': mf_batch['cat_t'],
                'mask': batch['struct_mask'],
            }

            losses = self.multiflow_loss(loss_batch)
            for key in losses:
                self._logger.log(f"mutliflow/{key}", losses[key], batch['struct_mask'])
             
        else:
            out, readout = self.forward(noisy_batch)

        ## compute the loss
        loss = self.tracks.compute_loss(
            readout, target, logger=self._logger, step=self.trainer.global_step
        )

        ## log some metrics
        self._logger.masked_log("loss", loss, batch["pad_mask"])
        self._logger.log("lr", self.get_lr())

        self._logger.clear_masks()

        return (loss * batch["pad_mask"]).sum() / batch["pad_mask"].sum()

    def multiflow_loss(self, batch):
        loss_mask = batch['mask']
        gt_trans_1 = batch['gt_trans']
        gt_rotmats_1 = batch['gt_rots']
        gt_aatypes_1 = batch['gt_aatype']
        rotmats_t = batch['noisy_rots']
        gt_rot_vf = so3_utils.calc_rot_vf(
            rotmats_t, gt_rotmats_1.type(torch.float32))
        
        # Timestep used for normalization.
        r3_t = batch['r3_t'] # (B, 1)
        so3_t = batch['so3_t'] # (B, 1)
        cat_t = batch['cat_t'] # (B, 1)
        r3_norm_scale = 1 - torch.min(
            r3_t[..., None], torch.tensor(0.9)) # (B, 1, 1)
        so3_norm_scale = 1 - torch.min(
            so3_t[..., None], torch.tensor(0.9)) # (B, 1, 1)
        cat_norm_scale = 1.0
            
        pred_trans_1 = batch['pred_trans']
        pred_rotmats_1 = batch['pred_rots']
        pred_logits = batch['pred_aatype'] # (B, N, aatype_pred_num_tokens)
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)

        num_batch, num_res = gt_aatypes_1.shape
        ce_loss = torch.nn.functional.cross_entropy(
            pred_logits.reshape(-1, 21),
            gt_aatypes_1.flatten().long(),
            reduction='none',
        ).reshape(num_batch, num_res) / cat_norm_scale
        
        aatypes_loss = ce_loss * loss_mask # torch.sum(ce_loss * loss_mask, dim=-1) / (loss_denom / 3)
        
        # Translation VF loss
        trans_error = (gt_trans_1 - pred_trans_1) / r3_norm_scale * 0.1
        
        trans_loss = (trans_error**2 * loss_mask[...,None]).sum(-1) / 3
        
        trans_loss = torch.clamp(trans_loss, max=5)
        # Rotation VF loss
        
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / so3_norm_scale
        rots_vf_loss = torch.sum(
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
        
    def on_validation_epoch_end(self):
        savedir = f'{os.environ["MODEL_DIR"]}/eval_step{self.trainer.global_step}'
        for name, eval_ in self.evals.items():
            eval_.compute_metrics(
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
                device=self.device,
                savedir=f"{savedir}/{name}",
                logger=self._logger,
            )
        super().on_validation_epoch_end()

    def validation_step_extra(self, batch, batch_idx):
        name = batch["dataset"][0]
        savedir = (
            f'{os.environ["MODEL_DIR"]}/eval_step{self.trainer.global_step}/{name}'
        )
        os.makedirs(savedir, exist_ok=True)
        noisy_batch = batch.copy("name", "pad_mask")
        for track in self.tracks.values():
            track.corrupt(batch, noisy_batch, {})
        self.evals[name].run_batch(
            self,
            batch,
            noisy_batch,
            savedir=savedir,
            logger=self._logger
        )

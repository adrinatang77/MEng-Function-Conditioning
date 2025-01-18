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
from .multiflow_wrapper import MultiflowWrapper
from omegaconf import OmegaConf
# from ..evals.manager import OpenProtEvalManager


class Wrapper(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters("cfg")
        self.cfg = cfg
        self._logger = Logger(cfg.logger)

    def training_step(self, batch, batch_idx):
        self._logger.prefix = "train"
        out = self.general_step(batch)
        self._logger.step(self.trainer, "train")
        return out

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
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
        
        if cfg.model.dplm_ckpt:
            from byprot.models.lm.dplm import DiffusionProteinLanguageModel
            
            self.model = DiffusionProteinLanguageModel.from_pretrained(
                self.cfg.model.dplm_ckpt
            ).train()

            if cfg.model.dplm_reinit:
                self.model.apply(self.model.net._init_weights)
            
            
            ours_to_dplm = [self.model.tokenizer._token_to_id[c] for c in rc.restypes] + [32]
            self.register_buffer("ours_to_dplm", torch.tensor(ours_to_dplm))
            del self.model.net.esm.embeddings.position_embeddings
            del self.model.net.esm.contact_head.regression

        elif cfg.model.multiflow:
            
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

        
    def on_save_checkpoint(self, checkpoint):
        esm_keys = {k for k in checkpoint['state_dict'].items() if "model.esm." in k}
        checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if k not in esm_keys}

    def on_load_checkpoint(self, checkpoint):
        state_dict = self.state_dict()
        esm_keys = {k for k in state_dict.items() if "model.esm." in k}
        checkpoint['state_dict'] |= {k: state_dict[k] for k in esm_keys}
        
            
        
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device)

    def get_lr(self):
        for param_group in self.optimizers().param_groups:
            return param_group["lr"]

    def forward(self, noisy_batch):
        
        
        if self.cfg.model.dplm_ckpt:
            
            inp = self.ours_to_dplm[noisy_batch['aatype']]
            inp = torch.where(noisy_batch['pad_mask'].bool(), inp, self.model.tokenizer._token_to_id["<pad>"])
            B, L = inp.shape
            inp_ = inp.new_zeros(B, L+2) + self.model.tokenizer._token_to_id["<pad>"]
            inp_[:,0] = self.model.tokenizer._token_to_id["<cls>"]
            inp_[:,1:-1] = inp
            inp_[torch.arange(B), noisy_batch['pad_mask'].sum(-1).long()+1] = self.model.tokenizer._token_to_id["<eos>"]
            
            out = None
            readout = {}
            
            logits = self.model.net(input_ids=inp_)['logits'][:,1:-1]
            readout['aatype'] = logits[:,:,self.ours_to_dplm]
            
            ##### 


            return out, readout
        
        ## embed the tracks into an input dict
        inp = self.tracks.embed(self.model, noisy_batch)

        ## run it thorugh the model
        out = self.model(inp)

        ## place the readouts in a dict
        readout = self.tracks.readout(self.model, inp, out)

        return out, readout

    def general_step(self, batch):

        
        if self.cfg.model.multiflow:
            L = batch['pad_mask'].shape[1]
            mf_batch = {
                'trans': batch['struct'],
                'rots': batch['rots'],
                'seqres': batch['aatype'],
                'res_mask': batch['struct_mask'],
                'diffuse_mask': batch['struct_mask'],
                'mask': batch['pad_mask'],
                'chain_idx': torch.zeros_like(batch['pad_mask'], dtype=torch.long),
                'res_idx': batch['residx'],
            }
            loss = self.model.general_step(mf_batch)
            self._logger.log("loss", loss.mean())
            return torch.nanmean(loss)
            
        self._logger.register_masks(batch)
        self._logger.masked_log("toks", batch["pad_mask"], sum=True)
    
        ## corrupt all the tracks
        noisy_batch, target = self.tracks.corrupt(batch, logger=self._logger)

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

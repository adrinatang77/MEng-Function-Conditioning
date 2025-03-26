import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--debug", action='store_true')
parser.add_argument("--gpus", type=int, default=None)
parser.add_argument("--workers", type=int, default=None)
parser.add_argument("--validate", action='store_true')
args = parser.parse_args()
from omegaconf import OmegaConf

cfg = OmegaConf.load(args.config)

if args.debug:
    cfg.logger.neptune = False
    cfg.logger.name = 'default'
    cfg.logger.logfile = None
    cfg.trainer.devices = 1
    cfg.trainer.enable_progress_bar = True

if args.gpus:
    cfg.trainer.devices = args.gpus

if args.workers:
    cfg.data.num_workers = args.workers

if args.validate:
    cfg.validate = True
    
os.environ["CONFIG"] = args.config
os.environ["MODEL_DIR"] = model_dir = os.path.join("workdir", cfg.logger.name)
os.makedirs(model_dir, exist_ok=True)

from openprot.utils.logger import setup_logging

setup_logging(cfg.logger)

import torch
import os
import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

from openprot.data.data import OpenProtData
from openprot.data.manager import OpenProtDatasetManager
from openprot.model.wrapper import OpenProtWrapper
from openprot.evals.manager import OpenProtEvalManager
from openprot.tracks.manager import OpenProtTrackManager

cfg.trainer.devices = int(os.environ.get("SLURM_NTASKS_PER_NODE", cfg.trainer.devices))

trainer = pl.Trainer(
    **cfg.trainer,
    default_root_dir=model_dir,
    callbacks=[
        ModelCheckpoint(
            dirpath=model_dir,
            save_top_k=-1,
            every_n_train_steps=cfg.logger.ckpt_freq,
            filename="{step}",
        ),
        ModelCheckpoint(
            dirpath=model_dir,
            every_n_train_steps=cfg.logger.save_freq,
            filename="last",
            enable_version_counter=False,
        ),
        ModelSummary(max_depth=4),
    ],
    num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
)
########## EVERYTHING BELOW NOW IN PARALLEL ######

tracks = OpenProtTrackManager(cfg.tracks)

dataset = OpenProtDatasetManager(cfg, tracks, trainer.global_rank, trainer.world_size)

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=cfg.data.batch,
    num_workers=cfg.data.num_workers,
    collate_fn=OpenProtData.batch,
)

evals = OpenProtEvalManager(cfg, tracks)

model = OpenProtWrapper(cfg, tracks, evals.evals)
if cfg.pretrained is not None:
    ckpt = torch.load(cfg.pretrained)
    model.load_state_dict(ckpt["state_dict"], strict=False)

if cfg.validate:
    trainer.validate(model, evals.loaders, ckpt_path=cfg.ckpt)
else:
    trainer.fit(model, train_loader, evals.loaders, ckpt_path=cfg.ckpt)

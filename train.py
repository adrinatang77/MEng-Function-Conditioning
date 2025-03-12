import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print('Starting...')

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
args = parser.parse_args()
from omegaconf import OmegaConf

cfg = OmegaConf.load(args.config)
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

print('Setting up trainer...')

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

print('Setting up tracks...')

tracks = OpenProtTrackManager(cfg.tracks)

print(tracks)

print('Setting up dataset...')

dataset = OpenProtDatasetManager(cfg, tracks, trainer.global_rank, trainer.world_size)

print(dataset)

print('Setting up train loader...')

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=cfg.data.batch,
    num_workers=cfg.data.num_workers,
    collate_fn=OpenProtData.batch,
)

print('Setting up evals...')

evals = OpenProtEvalManager(cfg, tracks)

print('Model...')

model = OpenProtWrapper(cfg, tracks, evals.evals)
if cfg.pretrained is not None:
    ckpt = torch.load(cfg.pretrained)
    model.load_state_dict(ckpt["state_dict"], strict=False)

print('Training...')

if cfg.validate:
    trainer.validate(model, evals.loaders, ckpt_path=cfg.ckpt)
else:
    print('Fitting...')
    trainer.fit(model, train_loader, evals.loaders, ckpt_path=cfg.ckpt)

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
args = parser.parse_args()
from omegaconf import OmegaConf

cfg = OmegaConf.load(args.config)

os.environ["MODEL_DIR"] = model_dir = os.path.join("workdir", cfg.logger.name)
os.makedirs(model_dir, exist_ok=True)

from openprot.utils.logger import setup_logging

setup_logging(cfg.logger)

import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

from openprot.data.manager import OpenProtDatasetManager
from openprot.model.wrapper import OpenProtWrapper
from openprot.evals.manager import OpenProtEvalManager

trainer = pl.Trainer(
    **cfg.trainer,
    default_root_dir=model_dir,
    callbacks=[
        ModelCheckpoint(
            dirpath=model_dir,
            save_top_k=-1,
        ),
        ModelSummary(max_depth=2),
    ],
)
########## EVERYTHING BELOW NOW IN PARALLEL ######

dataset = OpenProtDatasetManager(cfg, trainer.global_rank, trainer.world_size)

train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=cfg.data.batch, num_workers=cfg.data.num_workers
)

evals = OpenProtEvalManager(cfg, trainer.global_rank, trainer.world_size)
eval_loader = torch.utils.data.DataLoader(
    evals, batch_size=1, num_workers=0, shuffle=False
)
model = OpenProtWrapper(cfg)

if cfg.validate:
    trainer.validate(model, eval_loader, ckpt_path=cfg.ckpt)
else:
    trainer.fit(model, train_loader, eval_loader, ckpt_path=cfg.ckpt)

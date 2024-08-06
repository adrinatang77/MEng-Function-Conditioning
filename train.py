import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
args = parser.parse_args()
from omegaconf import OmegaConf

cfg = OmegaConf.load(args.config)

os.environ["MODEL_DIR"] = model_dir = os.path.join("workdir", cfg.logger.name)
os.makedirs(model_dir, exist_ok=True)

from openprot.utils.logging import setup_logging

setup_logging(cfg.logger)

import torch, os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

from openprot.data.datamodule import OpenProtDataModule
from openprot.model.wrapper import OpenProtWrapper


data_module = OpenProtDataModule(cfg.data_module)

model = OpenProtWrapper(cfg)

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

if cfg.validate:
    trainer.validate(model, data_module, ckpt_path=cfg.ckpt)
else:
    trainer.fit(model, data_module, ckpt_path=cfg.ckpt)

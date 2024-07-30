import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml')
parser.add_argument('--validate', action='store_true')
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--run_name', type=str, default='default')
args = parser.parse_args()

os.environ["MODEL_DIR"] = os.path.join("workdir", args.run_name)

import torch, os
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

from openprot.data.datamodule import OpenProtDataModule
from openprot.model.wrapper import OpenProtWrapper

cfg = OmegaConf.load(args.config)

data_module = OpenProtDataModule(cfg.data_module)

model = OpenProtWrapper(cfg)

trainer = pl.Trainer(
    **cfg.trainer,
    default_root_dir=os.environ["MODEL_DIR"],
    callbacks=[
        ModelCheckpoint(
            dirpath=os.environ["MODEL_DIR"],
            save_top_k=-1,
    #         every_n_epochs=args.ckpt_freq,
         ),
         ModelSummary(max_depth=2),
    ],
)

if args.validate:
    trainer.validate(model, data_module, ckpt_path=args.ckpt)
else:
    trainer.fit(model, data_module, ckpt_path=args.ckpt)
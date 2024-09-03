import numpy as np
import torch
from openprot.data.manager import OpenProtDatasetManager
from openprot.model.wrapper import OpenProtWrapper
import argparse
import os
from omegaconf import OmegaConf
from openprot.utils.logger import setup_logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from openprot.tracks.sequence import SequenceTrack

# def test_test():
#     dummy_var = True
#     print('Testing...')
#     assert dummy_var

# temp test - exploring sequence track/data
def test_sequence_data():
    # train.py code
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args([])

    cfg = OmegaConf.load(args.config)
    cfg['logger']['neptune'] = False
    cfg['datasets'] = {'openprot.data.uniref.UnirefDataset': {'path': '/Users/asapp/Documents/PhD/openprot_test_data/uniref50.fasta', 'index': '/Users/asapp/Documents/PhD/openprot_test_data/uniref50.index'}}
    cfg['tasks'] = {'openprot.tasks.seq_gen.SequenceGeneration': {'fraction': 1.0, 'mask_rate': 0.15, 'seed': 137, 'datasets': {'UnirefDataset': {'fraction': 1.0, 'start': 0, 'seed': 137}}}}
    # print(cfg)

    os.environ["MODEL_DIR"] = model_dir = os.path.join("workdir", cfg.logger.name)
    os.makedirs(model_dir, exist_ok=True)

    setup_logging(cfg.logger)

    model = OpenProtWrapper(cfg)

    # print(model)

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

    dataset = OpenProtDatasetManager(cfg, trainer.global_rank, trainer.world_size)
    dataset_batch = next(iter(dataset))
    # print('Dataset entry keys: ')
    # print(dataset_batch.keys())
    # print('seqres ' + str(dataset_batch['seqres']))
    # print('seq_mask ' + str(dataset_batch['seq_mask']))
    # print('seq_noise ' + str(dataset_batch['seq_noise']))

    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.data.batch)

    batch = next(iter(loader))

    print('Got batch of data. Initializing sequence track...')

    # initialize a sequence track 
    seq_tr = SequenceTrack(cfg=cfg)
    print(batch['aatype'].shape)
    print(batch.keys())

    print('Done.')

# TODO 
def test_seq_fm():
    pass

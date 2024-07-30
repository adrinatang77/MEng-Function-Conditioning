import logging
import time
import torch
import os
import numpy as np
import socket
from collections import defaultdict
import neptune
from pytorch_lightning.utilities.rank_zero import rank_zero_only


def get_logger(name):
    logger = logging.Logger(name)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    model_dir = os.environ["MODEL_DIR"]
    os.makedirs(model_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(model_dir, "log.out"))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        f"%(asctime)s [{socket.gethostname()}:%(process)d] [%(levelname)s] %(message)s"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

logger = get_logger(__name__)

def gather_log(log, world_size):
    if world_size == 1:
        return log
    log_list = [None] * world_size
    torch.distributed.all_gather_object(log_list, log)
    log = {key: sum([l[key] for l in log_list], []) for key in log}
    return log


def get_log_mean(log):
    out = {}
    for key in log:
        try:
            out[key] = np.nanmean(log[key])
        except:
            pass
    return out


class Logger:
    def __init__(self, cfg):
        self.cfg = cfg
        self._log = defaultdict(list)
        self.last_log_time = time.time()
        self.iter_step = defaultdict(int)
        self.prefix = 'train'
        if cfg.neptune:
            self.neptune_init()
        
    def step(self, trainer):
        self.iter_step[self.prefix] += 1
        self._log[self.prefix + '/dur'].append(time.time() - self.last_log_time)
        self.last_log_time = time.time()

        interval = {
            'train': self.cfg.train_log_freq, 
            'val': self.cfg.val_log_freq
        }[self.prefix]
        if interval is not None and self.iter_step[self.prefix] % interval == 0:
            self.print_log(trainer)

    def log(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.mean().item()
        self._log[self.prefix + "/" + key].append(data)

    def epoch_end(self, trainer):
        interval = {
            'train': self.cfg.train_log_freq, 
            'val': self.cfg.val_log_freq
        }[self.prefix]
        if interval is None:
            self.print_log(trainer)
    
    def print_log(self, trainer, save=False, extra_logs=None):
        log = self._log
        log = {key: log[key] for key in log if f"{self.prefix}/" in key}
        log = gather_log(log, trainer.world_size)
        mean_log = get_log_mean(log)

        mean_log.update(
            {
                self.prefix + "/epoch": trainer.current_epoch,
                self.prefix + "/global_step": trainer.global_step,
                self.prefix + "/iter_step": self.iter_step[self.prefix],
                self.prefix + "/count": len(log[next(iter(log))]),
            }
        )
        if extra_logs:
            mean_log.update(extra_logs)

        if trainer.is_global_zero:
            logger.info(str(mean_log))
            if self.cfg.neptune:
                 self.neptune_log(mean_log)
            if save:
                path = os.path.join(
                    os.environ["MODEL_DIR"],
                    f"{self.prefix}_{self.trainer.current_epoch}.csv",
                )
                pd.DataFrame(log).to_csv(path)
        for key in list(log.keys()):
            if f"{self.prefix}/" in key:
                del self._log[key]

    @rank_zero_only
    def neptune_init(self):
        self.run = neptune.init_run(
            project=self.cfg.project,
            name=os.environ["MODEL_DIR"].split('/')[1]
        )

    @rank_zero_only
    def neptune_log(self, log):
        for key in log:
            self.run[key].append(log[key])
import torch
import numpy as np
from abc import abstractmethod


class OpenProtDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, feats=None, logger=None):
        super().__init__()
        self.cfg = cfg
        self.feats = feats
        self.setup()

    @abstractmethod
    def setup(self):
        NotImplemented

    @abstractmethod
    def __len__(self):
        NotImplemented

    @abstractmethod
    def __getitem__(self, idx: int):
        NotImplemented

    def make_data(self, **kwargs):
        return OpenProtData(feats=self.feats, **kwargs)


class OpenProtData(dict):
    def __init__(self, *, feats, **kwargs):
        assert "seqres" in kwargs
        assert "name" in kwargs
        self["name"] = kwargs["name"]
        self["seqres"] = kwargs["seqres"]
        for feat, shape in feats.items():
            if feat in kwargs:
                self[feat] = kwargs[feat]
            else:
                self[feat] = np.zeros((len(self["seqres"]), *shape), dtype=np.float32)

    def keys_to_crop(self):
        return [key for key in self if key != "name"]

    def crop(self, crop_len: int):
        L = len(self["seqres"])

        if L >= crop_len:  # needs crop
            start = np.random.randint(0, L - crop_len + 1)
            end = start + crop_len
            for key in self.keys_to_crop():
                self[key] = self[key][start:end]
        return self

    def pad(self, pad_len: int):
        L = len(self["seqres"])
        if pad_len and L < pad_len:  # needs pad
            pad = pad_len - L
            for key in self.keys_to_crop():
                # unfortunately this is a string, unlike everything else
                if key == "seqres":
                    self[key] = self[key] + "X" * pad
                else:
                    shape = self[key].shape
                    dtype = self[key].dtype
                    self[key] = np.concatenate(
                        [self[key], np.zeros((pad, *shape[1:]), dtype=dtype)]
                    )

        pad_len = pad_len or L
        pad_mask = np.zeros(pad_len, dtype=np.float32)
        pad_mask[: min(pad_len, L)] = 1.0
        self["pad_mask"] = pad_mask
        return self

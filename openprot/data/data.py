import torch
import numpy as np

class OpenProtDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, feats):
        super().__init__()
        self.cfg = cfg
        self.feats = feats
        self.setup()
        
    def make_data(self, **kwargs):
        return OpenProtData(feats=self.feats, **kwargs)

class OpenProtData(dict):
    def __init__(self, *, feats, **kwargs):
        assert 'seqres' in kwargs
        self["seqres"] = kwargs['seqres']
        for feat, shape in feats.items():
            if feat in kwargs:
                self[feat] = kwargs[feat]
            else:
                self[feat] = np.zeros((len(self["seqres"]), *shape), dtype=np.float32)

    def crop_or_pad(self, crop_len):
        L = len(self["seqres"])
        
        if L >= crop_len:  # needs crop
            start = np.random.randint(0, L - crop_len + 1)
            end = start + crop_len
            for key in self:
                self[key] = self[key][start:end]
        elif L < crop_len:  # needs pad
            pad = crop_len - L
            for key in self:
                # unfortunately this is a string, unlike everything else
                if key == "seqres":
                    self[key] = self[key] + "X" * pad
                else:
                    shape = self[key].shape
                    dtype = self[key].dtype
                    self[key] = np.concatenate(
                        [self[key], np.zeros((pad, *shape[1:]), dtype=dtype)]
                    )
        
        pad_mask = np.zeros(crop_len, dtype=np.float32)
        pad_mask[: min(crop_len, L)] = 1.0
        self["pad_mask"] = pad_mask
        
import torch
import numpy as np
from abc import abstractmethod


class OpenProtDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, feats=None):
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
        assert 'name' in kwargs
        assert 'seqres' in kwargs

        data = OpenProtData()

        data["name"] = kwargs["name"]
        data["seqres"] = kwargs["seqres"]
        
        L = len(data["seqres"])
        for feat, shape in self.feats.items():
            shape = [(n if n > 0 else L) for n in shape]
            if feat in kwargs:
                data[feat] = kwargs[feat]
            else:
                data[feat] = np.zeros(shape, dtype=np.float32)

        return data


class OpenProtData(dict):
    def copy(self, *keys):
        data = OpenProtData()
        for key in keys:
            data[key] = self[key]
        return data
    
    def crop(self, crop_len: int):
        L = len(self["seqres"])

        ### todo support tensors!
        if L >= crop_len:  # needs crop
            start = np.random.randint(0, L - crop_len + 1)
            end = start + crop_len
            for key in self.keys():
                # special attribute
                if key == 'seqres':
                    self[key] = self[key][start:end]

                # non-array attribute
                elif type(self[key]) not in [torch.Tensor, np.ndarray]:
                    pass

                # pairwise attribute
                elif key[-1] == '_': 
                    self[key] = self[key][start:end,start:end]

                # regular attribute
                else:
                    self[key] = self[key][start:end]


        return self

    def pad(self, pad_len: int):
        L = len(self["seqres"])
        if pad_len and L < pad_len:  # needs pad
            pad = pad_len - L
            for key in self.keys():
                # special attribute
                if key == "seqres":
                    self[key] = self[key] + "X" * pad
                
                # non-array attribute
                elif type(self[key]) not in [torch.Tensor, np.ndarray]:
                    pass

                # pairwise attribute
                elif key[-1] == '_': 
                    shape = self[key].shape
                    dtype = self[key].dtype
                    padded = np.zeros((pad_len, pad_len, *shape[2:]), dtype=dtype)
                    padded[:L,:L] = self[key]
                    self[key] = padded
                    
                # regular attribute
                else:
                    shape = self[key].shape
                    dtype = self[key].dtype
                    padded = np.zeros((pad_len, *shape[1:]), dtype=dtype)
                    padded[:L] = self[key]
                    self[key] = padded

        pad_len = pad_len or L
        pad_mask = np.zeros(pad_len, dtype=np.float32)
        pad_mask[: min(pad_len, L)] = 1.0
        self["pad_mask"] = pad_mask
        return self

    def batch(datas):
        batch = OpenProtData()
        key_union = list(set(sum([list(data.keys()) for data in datas], [])))
        for key in key_union:
            try:
                batch[key] = [data[key] for data in datas]
            except:
                raise Exception(f"Key {key} not present in all batch elements.")
        for key in key_union:
            if type(batch[key][0]) is np.ndarray:
                batch[key] = torch.from_numpy(np.stack(batch[key]))
            elif type(batch[key][0]) is torch.Tensor:
                batch[key] = torch.stack(batch[key])

        return batch

    def to(self, device):
        for key in self.keys():
            if isinstance(self[key], torch.Tensor):
                self[key] = self[key].to(device)
        return self
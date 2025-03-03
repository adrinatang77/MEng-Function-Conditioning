import torch
import numpy as np
from abc import abstractmethod
RNA_LETTERS = {"A": 0, "G": 1, "C": 2, "U": 3}
DNA_LETTERS = {"A": 0, "G": 1, "C": 2, "T": 3}
from ..utils import residue_constants as rc
from ..utils.prot_utils import seqres_to_aatype

class OpenProtDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, feats=None, tracks=None):
        super().__init__()
        self.cfg = cfg
        self.feats = feats
        self.tracks = tracks
        self.setup()

    @abstractmethod
    def setup(self):
        NotImplemented

    @abstractmethod
    def __len__(self):
        NotImplemented

    @abstractmethod
    def __getitem__(self, idx: int):
        """
        Returns OpenProtData via self.make_data(...) with args
        (1) name and seqres (required)
        (2) any features that should be set to non-default values (np arrays only!)
        """
        NotImplemented

    def make_data(self, **kwargs):
        assert "name" in kwargs
        assert "seqres" in kwargs
        for key in kwargs:
            assert key in self.feats or key in ['name', 'seqres']

        data = OpenProtData()

        data["name"] = kwargs["name"]
        data["seqres"] = kwargs["seqres"]
        data["dataset"] = self.cfg.name 
        L = len(data["seqres"])
        for feat, shape in self.feats.items():
            shape = [(n if n > 0 else L) for n in shape]
            if feat in kwargs:
                assert type(kwargs[feat]) is np.ndarray, feat
                # assert kwargs[feat].dtype == np.float32, feat
                data[feat] = kwargs[feat].astype(np.float32)
            else:
                data[feat] = np.zeros(shape, dtype=np.float32)

        return data


class OpenProtData(dict):
    def copy(self, *keys):
        data = OpenProtData()
        for key in keys:
            data[key] = self[key]
        return data

    def update_seqres(self):
        
        seqres = []
        for aa, mt in zip(
            self['aatype'].cpu().long(), 
            self['mol_type'].cpu().long(), 
        ):
            if mt == 0:
                seqres += rc.restypes_with_x[aa]
            elif mt == 1:
                seqres += list(DNA_LETTERS.keys())[aa - 21]
            elif mt == 2:
                seqres += list(RNA_LETTERS.keys())[aa - 26]
            else:
                seqres += "*"
        self['seqres'] = ''.join(seqres)
        
    def get_contiguous_crop(self, crop_len):
        L = len(self["seqres"])
        start = np.random.randint(0, L - crop_len + 1)
        end = start + crop_len
        return list(range(start, end))

    def get_spatial_crop(self, crop_len):
        cmap = np.square(self['struct'][None] - self['struct'][:,None]).sum(-1)**0.5 < 15.0
        cmap &= self['chain'][None] != self['chain'][:,None]
        cmap &= self['struct_mask'][None].astype(bool) & self['struct_mask'][:,None].astype(bool)
        cmap = np.any(cmap, -1)

        if cmap.sum() == 0:
            return self.get_contiguous_crop(crop_len)
            
        i = np.random.choice(np.nonzero(cmap)[0])

        dist = np.square(self['struct'] - self['struct'][i]).sum(-1)**0.5
        dist[self['struct_mask']==0] = float('inf')
        idx = np.argsort(dist)[:crop_len]
        return sorted(idx)        
        
    def crop(self, crop_len: int):
        L = len(self["seqres"])
        if L >= crop_len:  # needs crop
            if np.unique(self['chain']).size > 1:
                idx = self.get_spatial_crop(crop_len)
            else:
                idx = self.get_contiguous_crop(crop_len)
            for key in self.keys():
                # special attribute
                if key == "seqres":
                    self[key] = [self[key][i] for i in idx]
                
                # non-array attribute
                elif type(self[key]) not in [torch.Tensor, np.ndarray]:
                    pass

                # global attribute
                elif key[0] == "/":
                    pass

                # pairwise attribute
                elif key[0] == "_":
                    self[key] = self[key][idx, idx]

                # regular attribute
                else:
                    self[key] = self[key][idx]

        return self

    def pad(self, pad_len: int):
        L = len(self["seqres"])
        if pad_len and L < pad_len:  # needs pad
            pad = pad_len - L
            for key in self.keys():
                # special attribute
                if key == "seqres":
                    self[key] = self[key] + " " * pad

                # non-array attribute
                elif type(self[key]) not in [torch.Tensor, np.ndarray]:
                    pass

                # global attribute
                elif key[0] == "/":
                    pass

                # pairwise attribute
                elif key[0] == "_":
                    shape = self[key].shape
                    dtype = self[key].dtype
                    padded = np.zeros((pad_len, pad_len, *shape[2:]), dtype=dtype)
                    padded[:L, :L] = self[key]
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

    def batch(datas, pad=True):
        if pad:
            lens = [len(data['seqres']) for data in datas]
            for data in datas:
                data.pad(max(lens))
                
        batch = OpenProtData()
        key_union = list(set(sum([list(data.keys()) for data in datas], [])))
        for key in key_union:
            try:
                batch[key] = [data[key] for data in datas]
            except:
                raise Exception(f"Key {key} not present in all batch elements.")
        for key in key_union:
            try:
                if type(batch[key][0]) is np.ndarray:
                    batch[key] = torch.from_numpy(np.stack(batch[key]))
                elif type(batch[key][0]) is torch.Tensor:
                    batch[key] = torch.stack(batch[key])
            except Exception as e:
                raise Exception(f"Key {key} exception: {e}")
        return batch

    def unbatch(self, trim=True):
        datas = [OpenProtData() for _ in self['seqres']]
        for key in self:
            for i in range(len(self['seqres'])):
                datas[i][key] = self[key][i]

        if trim:
            datas = [data.trim() for data in datas]
        return datas

    def trim(self):
        mask = self['pad_mask'].bool()
        for key in self:
            if key == "seqres":
             self[key] = ''.join([self[key][i] for i, m in enumerate(mask) if m])

            # non-array attribute
            elif type(self[key]) not in [torch.Tensor, np.ndarray]:
                pass

            # global attribute
            elif key[0] == "/":
                pass

            # pairwise attribute
            elif key[0] == "_":
                self[key] = self[key][mask,mask]

            # regular attribute
            else:
                self[key] = self[key][mask]
        return self

    def concat(datas):
        batch = OpenProtData()
        key_union = list(set(sum([list(data.keys()) for data in datas], [])))
        for key in key_union:
            try:
                batch[key] = [data[key] for data in datas]
            except:
                raise Exception(f"Key {key} not present in all batch elements.")
        for key in key_union:
            try:
                if type(batch[key][0]) is np.ndarray:
                    batch[key] = np.concatenate(batch[key], 0)
                elif type(batch[key][0]) is torch.Tensor:
                    batch[key] = torch.cat(batch[key], 0)
                elif type(batch[key][0]) is str:
                    batch[key] = "".join(batch[key])
            except Exception as e:
                raise Exception(f"Key {key} exception: {e}")
        return batch

    def to(self, device):
        for key in self.keys():
            if isinstance(self[key], torch.Tensor):
                self[key] = self[key].to(device)
        return self

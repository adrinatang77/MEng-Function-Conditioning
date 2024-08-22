import numpy as np


def batched_gather(data, inds, dim=0, no_batch_dims=0, np=np):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = np.arange(s)
        r = r.reshape(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[tuple(ranges)]

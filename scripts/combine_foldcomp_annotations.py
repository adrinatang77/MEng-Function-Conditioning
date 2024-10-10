import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='tmp')
parser.add_argument('--out', type=str, default=None)
parser.add_argument('--num_workers', type=int, default=1)
args = parser.parse_args()

import numpy as np
from collections import defaultdict

arrs = defaultdict(list)
for i in range(args.num_workers):
    npz = dict(np.load(f'{args.dir}/tmp_{i}.npz'))
    for key in npz:
        arrs[key].append(npz[key])

arrs = {key: np.stack(arrs[key]).sum(0) for key in arrs}
np.savez(args.out, **arrs)
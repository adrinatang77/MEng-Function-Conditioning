import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="tmp")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--num_workers", type=int, default=1)
args = parser.parse_args()

import numpy as np
import pandas as pd
from collections import defaultdict

dfs = []
for i in range(args.num_workers):
    df = pd.read_pickle(f"{args.dir}/{i}.pkl")
    dfs.append(df)

pd.concat(dfs).sort_index().to_pickle(args.out)

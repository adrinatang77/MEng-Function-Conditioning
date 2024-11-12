import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--workdir", type=str, required=True)
args = parser.parse_args()

import pandas as pd
import os

dirs = [path for path in os.listdir(args.workdir) if "eval" in path]

dfs = []
for dir_ in dirs:
    df = pd.read_csv(f"{args.workdir}/{dir_}/info.csv", index_col="domain")
    dfs.append(df)

df = pd.concat(dfs)
df["designable"] = df["scRMSD"] < 2.0

print(df.mean())

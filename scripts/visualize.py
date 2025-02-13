import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)
parser.add_argument('--out', type=str)
parser.add_argument('--width', type=int, default=5)
parser.add_argument('--height', type=int, default=5)
parser.add_argument('--annotate', action='store_true')

args = parser.parse_args()

import matplotlib.pyplot as plt
from pymol import cmd
import tqdm, os
import numpy as np
import pandas as pd
from PIL import Image

fig, axs = plt.subplots(args.width, args.height, figsize=(args.width, args.height), dpi=300)

if args.annotate:
    df = pd.read_csv(f'{args.dir}/info.csv', index_col=0)

def render(path):

    cmd.reinitialize()
    cmd.load(path, 'tmp')
    cmd.set('depth_cue', 0)
    cmd.set('ray_shadows', 0)
    cmd.spectrum('count', 'rainbow')

    
    cmd.png(f'{path}.png', 640, 640)
    im = np.array(Image.open(f'{path}.png'))
    os.remove(f'{path}.png')
    return im

for i, ax in tqdm.tqdm(enumerate(axs.flatten())):
    path = f"{args.dir}/sample{i}.pdb"
    im = render(path)
    
    if args.annotate:
        row = df.loc[f"sample{i}"]
        
        ax.text(0, 0, 
            f"{i} {row.scrmsd:.1f}A / {getattr(row, 'pmpnn_scrmsd', -1):.1f}A \n"
            f"{int(row.helix*100)}h:{int(row.sheet*100)}s:{int(row.loop*100)}l",
        size=4)
    ax.imshow(im)
    ax.set_axis_off()
fig.savefig(args.out, bbox_inches='tight', pad_inches=0)
    



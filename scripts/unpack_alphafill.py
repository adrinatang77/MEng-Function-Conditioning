import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=str, default='tmp/alphafill.idx')
parser.add_argument("--outdir", type=str, default="./data/alphafill_npz")
parser.add_argument("--alphafill", type=str, default="./data/alphafill")
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--worker_id", type=int, default=0)
args = parser.parse_args()

from openprot.utils.structure import Structure
import tqdm, os

lines = list(open(args.manifest))
for i in tqdm.trange(args.worker_id, len(lines), args.num_workers):
    line = lines[i]
    name = line.split()[0]
    path = f"{args.alphafill}/{name[3:5]}/{name}"
    outpath = f"{args.outdir}/{name[3:5]}/{name.replace('.cif.gz', '.npz')}"
    os.makedirs(f"{args.outdir}/{name[3:5]}", exist_ok=True)
    try:
        struct = Structure.from_mmcif(path)
        struct.to_npz(outpath)
    except:
        print(path, 'failure')
    
    # breakpoint()
    # except:
    #     print(path, flush=True)
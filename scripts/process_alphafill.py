import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--alphafill', default='data/alphafill')
parser.add_argument('--out', default='tmp/alphafill.idx')
args = parser.parse_args()
import tqdm, json

print('Loading AlphaFill', flush=True)
with open(args.alphafill + '/manifest') as f:
    with open(args.out, 'w') as g:
        for line in tqdm.tqdm(f, total=3531084):
            line = line.strip()
            if line[:3] == 'AF-' and line[-7:] == '.cif.gz':
                
                try:
                    _, upkb, _, _ = line.split('-')
                    js = f"{args.alphafill}/{upkb[:2]}/{line.replace('gz','json')}"
                    js = json.load(open(js))
                    if len(js['hits']) > 0:
                        g.write(f"{line} {len(js['hits'])}\n")
                
                except:
                    print(line, flush=True)
                    continue
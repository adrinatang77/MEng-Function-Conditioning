import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--xml", type=str, default='/data/cb/scratch/datasets/uniref100.xml.gz')
parser.add_argument("--out", type=str, default='tmp/uniref100.mapping')
args = parser.parse_args()

import tqdm
import numpy as np
import gzip
from collections import defaultdict

# https://stackoverflow.com/questions/2148119/how-to-convert-an-xml-string-to-a-dictionary
def etree_to_dict(t):
    # t.tag = t.tag.replace('{http://uniprot.org/uniref}', '')
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k:v[0] if len(v) == 1 else v for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v) for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
              d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d

import xml.etree.ElementTree as ET
i = 0
# Open the large XML file

def remove_namespace(elem):
    """Removes namespace from an element tag."""
    elem.tag = elem.tag.split('}', 1)[-1]  # Removes '{namespace}'
    for child in elem:
        remove_namespace(child)
    return elem

progress = tqdm.tqdm()

with open(args.out, 'w') as f:
    for _, element in ET.iterparse(gzip.open(args.xml, 'rt', encoding='utf-8')):
        
        if element.tag == '{http://uniprot.org/uniref}entry':
            tree = etree_to_dict(remove_namespace(element))
            
            ur100 = tree['entry']['@id']
            ur90 = None
            ur50 = None
            upkb = []
            for prop in tree['entry']['representativeMember']['dbReference']['property']:
                if prop['@type'] == 'UniRef90 ID':
                    ur90 = prop['@value']
                elif prop['@type'] == 'UniRef50 ID':
                    ur50 = prop['@value']
                elif prop['@type'] == 'UniProtKB accession':
                    upkb.append(prop['@value'])

            # seq = tree['entry']['representativeMember']['sequence']
            if ur90 and ur50: # skip those without cluster assignments
                f.write(" ".join([
                    ur100, ur90, ur50, *upkb, "\n"
                ]))
            
            element.clear()
            progress.update()
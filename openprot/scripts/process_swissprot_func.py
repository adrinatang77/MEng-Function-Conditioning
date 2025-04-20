import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
args = parser.parse_args()

import json
import jsonlines
import numpy as np
import os
import tqdm

dir = args.data_dir

# get jsonl metadata files 
files = []
for filename in os.listdir(dir):
    full_path = os.path.join(dir, filename)
    if os.path.isfile(full_path) and filename.endswith('.jsonl'):
        files.append(filename)

total_proteins = 0
go_terms_and_ec_numbers = {}

pbar = tqdm.tqdm()
# extract go terms and ec numbers 
for filename in files:
    with jsonlines.open(dir + filename) as reader:
        for line in reader:
            pbar.update()
            total_proteins += 1
            primaryAcc = line['primaryAccession']
            go_terms_and_ec_numbers[primaryAcc] = {'secondaryAccessions': [], 'GO terms': [], 'EC numbers': []} # create dictionary for each protein
            if 'secondaryAccessions' in line:
                secondaryAcc = line['secondaryAccessions']
                go_terms_and_ec_numbers[primaryAcc]['secondaryAccessions'] = secondaryAcc
            # GO terms
            if 'uniProtKBCrossReferences' in line:
                db_crossrefs = line['uniProtKBCrossReferences']
                for ref in db_crossrefs:
                    if ref['database'] == 'GO':
                        go_id = ref['id']
                        go_description = ref['properties'][0]['value']
                        go_evidence = ref['properties'][1]['value']
                        go_terms_and_ec_numbers[primaryAcc]['GO terms'].append({'GO ID': go_id, 'GO description': go_description, 'GO evidence': go_evidence})
            # EC numbers
            if 'proteinDescription' in line:
                if 'recommendedName' in line['proteinDescription']:
                    recommended_name = line['proteinDescription']['recommendedName']
                    if 'ecNumbers' in recommended_name:
                        for ec in recommended_name['ecNumbers']:
                            go_terms_and_ec_numbers[primaryAcc]['EC numbers'].append(ec['value'])
                if 'contains' in line['proteinDescription']:
                    for protein in line['proteinDescription']['contains']:
                        if 'recommendedName' in protein:
                            recommended_name = protein['recommendedName']
                            if 'ecNumbers' in recommended_name:
                                for ec in recommended_name['ecNumbers']:
                                    go_terms_and_ec_numbers[primaryAcc]['EC numbers'].append(ec['value'])
            if 'comments' in line:
                for entry in line['comments']:
                    if 'commentType' in entry and entry['commentType'] == 'CATALYTIC ACTIVITY':
                        if 'reaction' in entry and 'ecNumber' in entry['reaction']:
                            go_terms_and_ec_numbers[primaryAcc]['EC numbers'].append(entry['reaction']['ecNumber'])

with open(args.out, 'w') as json_file:
    json.dump(go_terms_and_ec_numbers, json_file, indent=4)
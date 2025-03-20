import json
import os
import sys
from go_term_utils import get_go_graph, get_associated_go_terms
from tqdm import tqdm
import concurrent.futures

# TODO: add filepaths as arguments
swissprot_file = '/data/cb/asapp/func_data/swissprot_func_data.json'
with open(swissprot_file) as f:
    swissprot_data = json.load(f)

map_file = '/data/cb/asapp/uniref_id_mapping.dat'
IDs_not_found = '/data/cb/asapp/func_data/not_found.txt'
map_dataset = 'UniRef50'
swissprot_seq_func_map = '/data/cb/asapp/func_data/swissprot_uniref50_seq_func_map.txt'

graph = get_go_graph()

# load ID mapping file (UniRef/UniprotKB)
ID_mapping = {}
with open(map_file, 'r') as file:
    for line in tqdm(file, desc="Processing lines"):
        columns = line.strip().split("\t")
        id = columns[0]
        category = columns[1]
        associated_id = columns[2]

        if id not in ID_mapping:
            ID_mapping[id] = {"UniProtKB-ID": [], "UniRef100": [], "UniRef50": []}

        if category == "UniProtKB-ID":
            ID_mapping[id]["UniProtKB-ID"].append(associated_id)
        elif category == "UniRef100":
            ID_mapping[id]["UniRef100"].append(associated_id)
        elif category == "UniRef50":
            ID_mapping[id]["UniRef50"].append(associated_id)

# Function to process each item in swissprot_data
def process_entry(seq_name, func_data):
    all_terms = None
    go_data = func_data['GO terms']
    
    if len(go_data) > 0:
        go_terms = [entry['GO ID'] for entry in go_data]
    else:
        go_terms = None

    if go_terms is not None:
        # get associated GO terms of specific relationship/type
        all_terms = get_associated_go_terms(graph, go_terms, ['molecular_function'], ['is_a'])

    uniref_id = None
    if seq_name in ID_mapping:
        if len(ID_mapping[seq_name][map_dataset]) > 0:
            uniref_id = ID_mapping[seq_name][map_dataset][0]
    else:
        with open(IDs_not_found, 'a') as f:
            f.write(seq_name + '\n')

    if uniref_id is not None:
        return (uniref_id, all_terms)
    else:
        return None

# To batch file writing, accumulate results and write once at the end
def write_results(results):
    with open(swissprot_seq_func_map, 'a') as f:
        for uniref_id, all_terms in results:
            f.write('>' + uniref_id + '\n')
            if all_terms is not None:
                terms = ','.join(all_terms)
                f.write(terms + '\n')

# Use ProcessPoolExecutor to parallelize the processing of each sequence
with concurrent.futures.ProcessPoolExecutor() as executor:
    # Prepare the data to pass to the executor
    futures = []
    for seq_name, func_data in swissprot_data.items():
        futures.append(executor.submit(process_entry, seq_name, func_data))

    # Collect results
    results = []
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing terms"):
        result = future.result()
        if result is not None:
            results.append(result)

    # Write results to file after all processing is done
    write_results(results)


# import json
# import os
# import sys
# from go_term_utils import get_go_graph, get_associated_go_terms
# from tqdm import tqdm
# import concurrent.futures

# # TODO: add filepaths as arguments
# swissprot_file = '/data/cb/asapp/func_data/swissprot_func_data.json'
# with open(swissprot_file) as f:
#     swissprot_data = json.load(f)

# map_file = '/data/cb/asapp/uniref_id_mapping.dat'
# IDs_not_found = '/data/cb/asapp/func_data/not_found.txt'
# map_dataset = 'UniRef50'
# swissprot_seq_func_map = '/data/cb/asapp/func_data/swissprot_uniref50_seq_func_map.txt'

# graph = get_go_graph()

# # load ID mapping file (UniRef/UniprotKB)

# ID_mapping = {}
# with open(map_file, 'r') as file:
#     for line in tqdm(file, desc="Processing lines"):
#         columns = line.strip().split("\t")
#         id = columns[0]
#         category = columns[1]
#         associated_id = columns[2]

#         # if the ID is not already in the dictionary, add it with an empty list
#         if id not in ID_mapping:
#             ID_mapping[id] = {"UniProtKB-ID": [], "UniRef100": [], "UniRef50": []}

#         # append the associated ID to the correct category
#         if category == "UniProtKB-ID":
#             ID_mapping[id]["UniProtKB-ID"].append(associated_id)
#         elif category == "UniRef100":
#             ID_mapping[id]["UniRef100"].append(associated_id)
#         elif category == "UniRef50":
#             ID_mapping[id]["UniRef50"].append(associated_id)

# # process the GO terms and make a text file from seq name
# # to swissprot function data

# for i, (seq_name, func_data) in tqdm(enumerate(list(swissprot_data.items())), desc="Processing terms"):

#     # first check for GO terms
#     go_data = func_data['GO terms']
#     if len(go_data) > 0:
#         go_terms = []
#         for entry in go_data:
#             go_terms.append(entry['GO ID'])
#     else:
#         go_terms = None
#     all_terms = None
#     if go_terms is not None:
#         # get associated GO terms of specific relationship/type
#         all_terms = get_associated_go_terms(graph, go_terms, ['molecular_function'], ['is_a'])
#     # ID mapping
#     uniref_id = None
#     if seq_name in ID_mapping:
#         if len(ID_mapping[seq_name][map_dataset]) > 0:
#           uniref_id = ID_mapping[seq_name][map_dataset][0]
#     else: 
#         with open(IDs_not_found, 'a') as f:
#             f.write(seq_name + '\n')

#     if uniref_id is not None: # is an entry in UniRef 
#         with open(swissprot_seq_func_map, 'a') as f:
#             f.write('>' + uniref_id + '\n')
#             if all_terms is not None:
#                 terms = ','.join(all_terms)
#                 f.write(terms + '\n')  



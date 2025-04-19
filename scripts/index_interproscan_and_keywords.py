# #!/usr/bin/env python3
# """
# Simple script to process InterProScan data and create:
# 1. protein_2_entry.txt - Contains proteins, sequences and their InterPro entries
# 2. positions.idx.npy - Index file marking where each protein starts
# 3. entry_2_keywords.txt - Maps InterPro entries to their keywords
# 4. keyword_idfs.txt - Contains keywords and their IDF values
# 5. hyperplanes.npy - Random hyperplanes for LSH

# Usage:
#     python index_interproscan.py /path/to/input_dir /path/to/output_dir

# Author: Created based on provided requirements
# """

# import argparse
# import gzip
# import json
# import os
# import string
# import sys
# from collections import Counter, defaultdict
# from pathlib import Path
# import numpy as np
# from tqdm import tqdm
# from math import log

# # Set of terms to exclude when extracting keywords
# EXCLUDED_TERMS = {
#     "binding domain",
#     "biological_process",
#     "biological process",
#     "biologicalprocess",
#     "c",
#     "cellular_component",
#     "cellular component",
#     "cellularcomponent",
#     "cellular_process",
#     "cellularprocess",
#     "cellular process",
#     "like domain",
#     "molecular function",
#     "molecular_function",
#     "molecularfunction", 
#     "n",
#     "protein",
#     "domain",
#     "family"
# }

# def clean_text(text):
#     """Clean text by replacing hyphens, removing punctuation, and converting to lowercase"""
#     text = text.replace("-", " ")
#     text = text.translate(str.maketrans("", "", string.punctuation))
#     text = text.lower()
#     return text

# def extract_keywords(text):
#     """Extract keywords and bigrams from text"""
#     elements = text.split(", ")
#     tokens = []
    
#     for element in elements:
#         element = clean_text(element)
#         words = element.split()
        
#         # Add individual words
#         tokens.extend(words)
        
#         # Add bigrams
#         for i in range(len(words) - 1):
#             bigram = words[i] + " " + words[i + 1]
#             tokens.append(bigram)
    
#     # Filter out short tokens and excluded terms
#     return [token for token in tokens if len(token) > 1 and token not in EXCLUDED_TERMS]

# def process_file(file_path):
#     """Process a single JSON.GZ file and extract data"""
#     try:
#         with gzip.open(file_path, 'rb') as f:
#             data = json.loads(f.read().decode('utf-8'))
        
#         results = []
        
#         for result in data.get('results', []):
#             # Skip if no matches or sequence
#             if not result.get('matches') or not result.get('sequence'):
#                 continue
            
#             # Get sequence ID
#             sequence_id = None
#             if result.get('xref') and len(result['xref']) > 0:
#                 sequence_id = result['xref'][0].get('id')
            
#             if not sequence_id:
#                 sequence_id = result.get('id')
#                 if not sequence_id:
#                     continue
            
#             sequence = result.get('sequence', '')
#             entries = []
            
#             # Process matches to find InterPro entries
#             for match in result.get('matches', []):
#                 # Skip if no locations
#                 if not match.get('locations'):
#                     continue
                
#                 # Get locations
#                 locations = []
#                 for loc in match.get('locations', []):
#                     start = loc.get('start')
#                     end = loc.get('end')
#                     if start is not None and end is not None:
#                         locations.append((start, end))
                
#                 # Skip if no valid locations
#                 if not locations:
#                     continue
                
#                 # Get entry from signature
#                 entry_id = None
#                 entry_info = None
                
#                 if match.get('signature') and match['signature'].get('entry'):
#                     entry = match['signature']['entry']
#                     entry_id = entry.get('accession')
                    
#                     if entry_id:
#                         description = entry.get('description', '')
#                         name = entry.get('name', '')
                        
#                         # Get GO terms
#                         go_terms = []
#                         if entry.get('goXRefs'):
#                             for go_ref in entry['goXRefs']:
#                                 go_term_name = go_ref.get('name', '')
#                                 if go_term_name:
#                                     go_terms.append(go_term_name)

#                         entry_info = {
#                             'id': entry_id,
#                             'description': description,
#                             'name': name,
#                             'go_terms': go_terms,
#                             'locations': locations
#                         }
                
#                 # If no entry from signature, try match directly
#                 if not entry_info and match.get('accession'):
#                     entry_id = match.get('accession')
#                     description = match.get('description', '')
#                     name = match.get('name', '')
                    
#                     # Get GO terms directly from match
#                     go_terms = []
#                     if match.get('goXRefs'):
#                         for go_ref in match['goXRefs']:
#                             go_term_name = go_ref.get('name', '')
#                             if go_term_name:
#                                 go_terms.append(go_term_name)
                    
#                     entry_info = {
#                         'id': entry_id,
#                         'description': description,
#                         'name': name,
#                         'go_terms': go_terms,
#                         'locations': locations
#                     }
                
#                 if entry_info:
#                     entries.append(entry_info)
            
#             # Only add proteins with entries
#             if entries:
#                 results.append({
#                     'id': sequence_id,
#                     'sequence': sequence,
#                     'entries': entries
#                 })
        
#         return results
    
#     except Exception as e:
#         print(f"Error processing {file_path}: {e}", file=sys.stderr)
#         return []

# def main():
#     # Parse arguments
#     parser = argparse.ArgumentParser(description='Process InterProScan data files')
#     parser.add_argument('input_dir', help='Directory containing InterProScan JSON.GZ files')
#     parser.add_argument('output_dir', help='Directory to store output files')
#     parser.add_argument('--num_hyperplanes', type=int, default=64, 
#                         help='Number of hyperplanes for LSH')
#     parser.add_argument('--max_chunks', type=int, default=None, help='maximum number of chunks to process')
#     args = parser.parse_args()
    
#     # Create output directory
#     os.makedirs(args.output_dir, exist_ok=True)
    
#     # Output file paths
#     protein_entry_path = os.path.join(args.output_dir, 'protein_2_entry.txt')
#     positions_path = os.path.join(args.output_dir, 'positions.idx.npy')
#     entry_keywords_path = os.path.join(args.output_dir, 'entry_2_keywords.txt')
#     keyword_idfs_path = os.path.join(args.output_dir, 'keyword_idfs.txt')
#     hyperplanes_path = os.path.join(args.output_dir, 'hyperplanes.npy')
    
#     # Find all JSON.GZ files
#     input_files = sorted(Path(args.input_dir).glob('*.json.gz'))
#     if not input_files:
#         print(f"No JSON.GZ files found in {args.input_dir}", file=sys.stderr)
#         return 1
    
#     print(f"Found {len(input_files)} JSON.GZ files")
    
#     if args.max_chunks is None:
#         args.max_chunks = len(input_files)
    
#     # Initialize variables
#     positions = [0]  # Start at position 0
#     current_position = 0
#     entry_to_keywords = defaultdict(set)
#     keyword_counter_entries = Counter()  # Count keywords per entry
#     keyword_counter_annotations = Counter()  # Count keywords per annotation
    
#     # Process files and write protein_2_entry.txt
#     with open(protein_entry_path, 'w') as f_out:
#         for file_path in tqdm(input_files[:args.max_chunks], desc="Processing files"):
#             results = process_file(file_path)
            
#             for protein in results:
#                 protein_id = protein['id']
#                 sequence = protein['sequence']
                
#                 # Write protein ID and sequence
#                 protein_text = f"{protein_id}\n>{sequence}\n"
#                 f_out.write(protein_text)
                
#                 # Update position
#                 current_position += len(protein_text)
                
#                 # Process entries
#                 for entry in protein['entries']:
#                     entry_id = entry['id']
                    
#                     # Write entry info with locations
#                     for start, end in entry['locations']:
#                         entry_line = f"{entry_id} {start} {end}\n"
#                         f_out.write(entry_line)
#                         current_position += len(entry_line)

#                     text_parts = []
                    
#                     # Collect keywords for entry_2_keywords
#                     if entry['description']:
#                         text_parts.extend(entry['description'])
#                     else:
#                         text_parts.extend(entry['name'])
#                     if entry['go_terms']:
#                         text_parts.extend(entry['go_terms'])
                    
#                     combined_text = ", ".join(text_parts)
#                     if combined_text:
#                         keywords = extract_keywords(combined_text)
                        
#                         # Add keywords to this entry
#                         entry_to_keywords[entry_id].update(keywords)
                        
#                         # Update keyword counters
#                         keyword_counter_entries.update(set(keywords))  # Count once per entry
#                         keyword_counter_annotations.update(keywords)  # Count every occurrence
                
#                 # Add blank line between proteins
#                 f_out.write("\n")
#                 current_position += 1
                
#                 # Record position for next protein
#                 positions.append(current_position)
    
#     # Save positions index
#     np.save(positions_path, np.array(positions, dtype=np.int64))
#     print(f"Saved position index to {positions_path}")
    
#     # Calculate IDFs
#     num_entries = len(entry_to_keywords)
#     total_annotations = sum(keyword_counter_annotations.values())
    
#     keyword_idfs = {}
#     for keyword, entry_count in keyword_counter_entries.items():
#         annotation_count = keyword_counter_annotations[keyword]
        
#         # IDF across entries
#         idf_entries = log(num_entries / entry_count) if entry_count > 0 else 0
        
#         # IDF across all annotations
#         idf_annotations = log(total_annotations / annotation_count) if annotation_count > 0 else 0
        
#         keyword_idfs[keyword] = (idf_entries, idf_annotations)
    
#     # Write entry_2_keywords.txt
#     with open(entry_keywords_path, 'w') as f:
#         for entry_id, keywords in sorted(entry_to_keywords.items()):
#             keywords_str = ",".join(sorted(keywords))
#             f.write(f"{entry_id} {keywords_str}\n")
    
#     print(f"Saved entry to keywords mapping to {entry_keywords_path}")
    
#     # Write keyword_idfs.txt
#     with open(keyword_idfs_path, 'w') as f:
#         for keyword, (idf_entries, idf_annotations) in sorted(keyword_idfs.items()):
#             f.write(f"{keyword} {idf_entries:.6f} {idf_annotations:.6f}\n")
    
#     print(f"Saved keyword IDF values to {keyword_idfs_path}")
    
#     # Generate hyperplanes
#     keywords_list = list(keyword_idfs.keys())
#     vocab_size = len(keywords_list)
    
#     hyperplanes = np.random.randn(args.num_hyperplanes, vocab_size)
#     hyperplanes /= np.linalg.norm(hyperplanes, axis=-1, keepdims=True)
    
#     np.save(hyperplanes_path, hyperplanes)
#     print(f"Saved {args.num_hyperplanes} hyperplanes to {hyperplanes_path}")
    
#     print(f"Successfully processed {len(positions)-1} proteins")
#     print(f"Found {len(entry_to_keywords)} unique InterPro entries")
#     print(f"Extracted {len(keyword_idfs)} unique keywords")
    
#     return 0

# if __name__ == "__main__":
#     sys.exit(main())
#!/usr/bin/env python3
"""
Simple script to process InterProScan data and create:
1. protein_2_entry.txt - Contains proteins, sequences and their InterPro entries
2. positions.idx.npy - Index file marking where each protein starts
3. entry_2_keywords.txt - Maps InterPro entries to their keywords
4. keyword_idfs.txt - Contains keywords and their IDF values
5. hyperplanes.npy - Random hyperplanes for LSH

Usage:
    python index_interproscan.py /path/to/input_dir /path/to/output_dir

Author: Created based on provided requirements
"""

import argparse
import gzip
import json
import os
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from tqdm import tqdm
from math import log

# Set of terms to exclude when extracting keywords
EXCLUDED_TERMS = {
    "binding domain", "biological_process", "biological process", 
    "biologicalprocess", "c", "cellular_component", "cellular component",
    "cellularcomponent", "cellular_process", "cellularprocess", "cellular process",
    "like domain", "molecular function", "molecular_function",
    "molecularfunction", "n", "protein", "domain", "family"
}

def clean_text(text):
    """Clean text by replacing hyphens, removing punctuation, and converting to lowercase"""
    text = text.replace("-", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    return text

def extract_keywords(text):
    """Extract keywords and bigrams from text"""
    elements = text.split(", ")
    tokens = []
    
    for element in elements:
        element = clean_text(element)
        words = element.split()
        
        # Add individual words
        tokens.extend(words)
        
        # Add bigrams
        for i in range(len(words) - 1):
            bigram = words[i] + " " + words[i + 1]
            tokens.append(bigram)
    
    # Filter out short tokens and excluded terms
    return [token for token in tokens if len(token) > 1 and token not in EXCLUDED_TERMS]

def process_file(file_path):
    """Process a single JSON.GZ file and extract data"""
    try:
        with gzip.open(file_path, 'rb') as f:
            data = json.loads(f.read().decode('utf-8'))
        
        results = []
        
        for result in data.get('results', []):
            # Skip if no matches or sequence
            if not result.get('matches') or not result.get('sequence'):
                continue
            
            # Get sequence ID
            sequence_id = None
            if result.get('xref') and len(result['xref']) > 0:
                sequence_id = result['xref'][0].get('id')
            
            if not sequence_id:
                sequence_id = result.get('id')
                if not sequence_id:
                    continue
            
            sequence = result.get('sequence', '')
            entries = []
            
            # Process matches to find InterPro entries
            for match in result.get('matches', []):
                # Skip if no locations
                if not match.get('locations'):
                    continue
                
                # Get locations
                locations = []
                for loc in match.get('locations', []):
                    start = loc.get('start')
                    end = loc.get('end')
                    if start is not None and end is not None:
                        locations.append((start, end))
                
                # Skip if no valid locations
                if not locations:
                    continue
                
                # Get entry from signature
                entry_id = None
                entry_info = None
                
                if match.get('signature') and match['signature'].get('entry'):
                    entry = match['signature']['entry']
                    entry_id = entry.get('accession')
                    
                    if entry_id:
                        description = entry.get('description', '')
                        name = entry.get('name', '')
                        
                        # Get GO terms
                        go_terms = []
                        if entry.get('goXRefs'):
                            for go_ref in entry['goXRefs']:
                                go_term_name = go_ref.get('name', '')
                                if go_term_name:
                                    go_terms.append(go_term_name)
                        
                        entry_info = {
                            'id': entry_id,
                            'description': description,
                            'name': name,
                            'go_terms': go_terms,
                            'locations': locations
                        }
                
                # If no entry from signature, try match directly
                if not entry_info and match.get('accession'):
                    entry_id = match.get('accession')
                    description = match.get('description', '')
                    name = match.get('name', '')
                    
                    # Get GO terms directly from match
                    go_terms = []
                    if match.get('goXRefs'):
                        for go_ref in match['goXRefs']:
                            go_term_name = go_ref.get('name', '')
                            if go_term_name:
                                go_terms.append(go_term_name)
                    
                    entry_info = {
                        'id': entry_id,
                        'description': description,
                        'name': name,
                        'go_terms': go_terms,
                        'locations': locations
                    }
                
                if entry_info:
                    entries.append(entry_info)
            
            # Only add proteins with entries
            if entries:
                results.append({
                    'id': sequence_id,
                    'sequence': sequence,
                    'entries': entries
                })
        
        return results
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return []

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Process InterProScan data files')
    parser.add_argument('input_dir', help='Directory containing InterProScan JSON.GZ files')
    parser.add_argument('output_dir', help='Directory to store output files')
    parser.add_argument('--num_hyperplanes', type=int, default=64, 
                        help='Number of hyperplanes for LSH')
    parser.add_argument('--max_chunks', type=int, default=None, help='maximum number of chunks to process')

    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Output file paths
    protein_entry_path = os.path.join(args.output_dir, 'protein_2_entry.txt')
    positions_path = os.path.join(args.output_dir, 'positions.idx.npy')
    entry_keywords_path = os.path.join(args.output_dir, 'entry_2_keywords.txt')
    keyword_idfs_path = os.path.join(args.output_dir, 'keyword_idfs.txt')
    hyperplanes_path = os.path.join(args.output_dir, 'hyperplanes.npy')
    
    # Find all JSON.GZ files
    input_files = sorted(Path(args.input_dir).glob('*.json.gz'))
    if not input_files:
        print(f"No JSON.GZ files found in {args.input_dir}", file=sys.stderr)
        return 1
    
    print(f"Found {len(input_files)} JSON.GZ files")
    if not args.max_chunks:
        args.max_chunks = len(input_files)
    
    # Initialize variables
    positions = [0]  # Start at position 0
    current_position = 0
    entry_to_keywords = defaultdict(set)
    keyword_counter_entries = Counter()  # Count keywords per entry
    keyword_counter_annotations = Counter()  # Count keywords per annotation
    
    # Process files and write protein_2_entry.txt
    with open(protein_entry_path, 'w') as f_out:
        for file_path in tqdm(input_files[:args.max_chunks], desc="Processing files"):
            results = process_file(file_path)
            
            for protein in results:
                protein_id = protein['id']
                sequence = protein['sequence']
                
                # Write protein ID and sequence
                protein_text = f"{protein_id}\n> {sequence}\n"
                f_out.write(protein_text)
                
                # Update position
                current_position += len(protein_text)
                
                # Process entries
                for entry in protein['entries']:
                    entry_id = entry['id']
                    
                    # Write entry info with locations
                    for start, end in entry['locations']:
                        entry_line = f"{entry_id} {start} {end}\n"
                        f_out.write(entry_line)
                        current_position += len(entry_line)
                    
                    # Collect keywords for entry_2_keywords
                    text_parts = []
                    if entry['name']:
                        text_parts.append(entry['name'])
                    if entry['description']:
                        text_parts.append(entry['description'])
                    text_parts.extend(entry['go_terms'])
                    
                    combined_text = ", ".join(text_parts)
                    if combined_text:
                        keywords = extract_keywords(combined_text)
                        
                        # Add keywords to this entry
                        entry_to_keywords[entry_id].update(keywords)
                        
                        # Update keyword counters
                        keyword_counter_entries.update(set(keywords))  # Count once per entry
                        keyword_counter_annotations.update(keywords)  # Count every occurrence
                
                # Add blank line between proteins
                f_out.write("\n")
                current_position += 1
                
                # Record position for next protein
                positions.append(current_position)
    
    # Save positions index
    np.save(positions_path, np.array(positions, dtype=np.int64))
    print(f"Saved position index to {positions_path}")
    
    # Calculate IDFs
    num_entries = len(entry_to_keywords)
    total_annotations = sum(keyword_counter_annotations.values())
    
    keyword_idfs = {}
    for keyword, entry_count in keyword_counter_entries.items():
        annotation_count = keyword_counter_annotations[keyword]
        
        # IDF across entries
        idf_entries = log(num_entries / entry_count) if entry_count > 0 else 0
        
        # IDF across all annotations
        idf_annotations = log(total_annotations / annotation_count) if annotation_count > 0 else 0
        
        keyword_idfs[keyword] = (idf_entries, idf_annotations)
    
    # Write entry_2_keywords.txt
    with open(entry_keywords_path, 'w') as f:
        for entry_id, keywords in sorted(entry_to_keywords.items()):
            keywords_str = ",".join(sorted(keywords))
            f.write(f"{entry_id} {keywords_str}\n")
    
    print(f"Saved entry to keywords mapping to {entry_keywords_path}")
    
    # Write keyword_idfs.txt
    with open(keyword_idfs_path, 'w') as f:
        for keyword, (idf_entries, idf_annotations) in sorted(keyword_idfs.items()):
            f.write(f"{keyword} {idf_entries:.6f} {idf_annotations:.6f}\n")
    
    print(f"Saved keyword IDF values to {keyword_idfs_path}")

    # Generate hyperplanes
    keywords_list = list(keyword_idfs.keys())
    vocab_size = len(keywords_list)
    
    hyperplanes = np.random.randn(args.num_hyperplanes, vocab_size)
    hyperplanes /= np.linalg.norm(hyperplanes, axis=-1, keepdims=True)
    
    np.save(hyperplanes_path, hyperplanes)
    print(f"Saved {args.num_hyperplanes} hyperplanes to {hyperplanes_path}")
    
    print(f"Successfully processed {len(positions)-1} proteins")
    print(f"Found {len(entry_to_keywords)} unique InterPro entries")
    print(f"Extracted {len(keyword_idfs)} unique keywords")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
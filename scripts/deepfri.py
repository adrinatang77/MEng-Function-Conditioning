import os
import tensorflow as tf
import argparse
import json
import glob

# Import DeepFRI modules
from deepfri.models.deepfrier import DeepFrier

parser = argparse.ArgumentParser()
parser.add_argument('--seq', type=str, default=None, help='Input protein sequence')
parser.add_argument('--outdir', type=str, default='/tmp', help='Output directory')
parser.add_argument('--fasta', type=str, default=None, help='Path to FASTA file')
parser.add_argument('--fasta_dir', type=str, default=None, help='Directory containing FASTA files (will process each file)')
parser.add_argument('--pdb', type=str, default=None, help='Path to PDB file')
parser.add_argument('--pdb_dir', type=str, default=None, help='Directory containing PDB files')
parser.add_argument('--model', type=str, default='fold', 
                    choices=['fold', 'sequence'], help='DeepFRI model type')
parser.add_argument('--ontology', type=str, default='mf', 
                    choices=['mf', 'bp', 'cc', 'ec'], 
                    help='Gene Ontology aspect')
parser.add_argument('--threshold', type=float, default=0.5, 
                    help='Probability threshold for GO term predictions')
args = parser.parse_args()

# Create output directory
os.makedirs(args.outdir, exist_ok=True)

# Initialize DeepFRI model
print(f"Initializing DeepFRI model: {args.model} for {args.ontology} ontology")
model = DeepFrier(saliency_type=None, 
                 model_type=args.model, 
                 ontology=args.ontology,
                 threshold=args.threshold)

# Single sequence
if args.seq:
    print(f"Processing direct sequence input")
    predictions = model.predict(args.seq)
    with open(os.path.join(args.outdir, "sequence_results.json"), "w") as f:
        json.dump(predictions, f, indent=2)

# Single FASTA file
elif args.fasta:
    print(f"Processing FASTA file: {args.fasta}")
    base_name = os.path.basename(args.fasta).split('.')[0]
    predictions = model.predict_from_fasta(args.fasta)
    with open(os.path.join(args.outdir, f"{base_name}_results.json"), "w") as f:
        json.dump(predictions, f, indent=2)

# Single PDB file
elif args.pdb:
    print(f"Processing PDB file: {args.pdb}")
    base_name = os.path.basename(args.pdb).split('.')[0]
    predictions = model.predict_from_pdb(args.pdb)
    with open(os.path.join(args.outdir, f"{base_name}_results.json"), "w") as f:
        json.dump(predictions, f, indent=2)

# Directory of PDB files
elif args.pdb_dir:
    pdb_files = glob.glob(os.path.join(args.pdb_dir, "*.pdb"))
    print(f"Found {len(pdb_files)} PDB files in {args.pdb_dir}")
    
    all_predictions = {}
    for pdb_file in pdb_files:
        name = os.path.basename(pdb_file).split('.')[0]
        print(f"Processing PDB file: {pdb_file}")
        try:
            predictions = model.predict_from_pdb(pdb_file)
            all_predictions[name] = predictions
            
            # Save individual result
            with open(os.path.join(args.outdir, f"{name}_results.json"), "w") as f:
                json.dump(predictions, f, indent=2)
        except Exception as e:
            print(f"Error processing PDB file {pdb_file}: {e}")
    
    # Save combined results
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(all_predictions, f, indent=2)

# Directory of FASTA files
elif args.fasta_dir:
    fasta_files = glob.glob(os.path.join(args.fasta_dir, "*.fasta"))
    print(f"Found {len(fasta_files)} FASTA files in {args.fasta_dir}")
    
    all_predictions = {}
    for fasta_file in fasta_files:
        name = os.path.basename(fasta_file).split('.')[0]
        print(f"Processing FASTA file: {fasta_file}")
        try:
            # Need to handle individually since DeepFRI doesn't have a directory function
            predictions = model.predict_from_fasta(fasta_file)
            all_predictions[name] = predictions
            
            # Save individual result
            with open(os.path.join(args.outdir, f"{name}_results.json"), "w") as f:
                json.dump(predictions, f, indent=2)
        except Exception as e:
            print(f"Error processing FASTA file {fasta_file}: {e}")
    
    # Save combined results
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(all_predictions, f, indent=2)

else:
    print("Error: No input provided. Please specify --seq, --fasta, --pdb, --fasta_dir, or --pdb_dir.")
    exit(1)

print("Processing complete. Results saved to:", args.outdir)
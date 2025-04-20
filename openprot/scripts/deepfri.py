import os
import argparse
import json
import glob
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--seq', type=str, default=None, help='Input protein sequence')
parser.add_argument('--outdir', type=str, default='/tmp', help='Output directory')
parser.add_argument('--fasta', type=str, default=None, help='Path to FASTA file')
parser.add_argument('--fasta_dir', type=str, default=None, help='Directory containing FASTA files')
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

# Path to DeepFRI predict.py script - update this to your path
DEEPFRI_PREDICT_PATH = "DeepFRI/predict.py"

# Function to run DeepFRI prediction
def run_deepfri_predict(output_path, model_type, ontology, **kwargs):
    """Run DeepFRI predict.py with specified parameters"""
    cmd = [
        "python", DEEPFRI_PREDICT_PATH,
        "-ont", ontology,
        "--output_fn_prefix", output_path,
        "--model_config",
        "DeepFRI/trained_models/model_config.json"
    ]
    
    # Add the appropriate input flag based on what was provided
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key}", value])
    
#     if model_type == "fold":
#         cmd.extend(["--model_config", "DeepFRI/trained_models/fold/model_config.json"])
#     else:  # sequence
#         cmd.extend(["--model_config", "DeepFRI/trained_models/sequence/model_config.json"])
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
    
    # The output files have specific naming conventions based on ontology
    json_output = f"{output_path}_{ontology.upper()}_pred_scores.json"
    csv_output = f"{output_path}_{ontology.upper()}_predictions.csv"
    
    # Check if JSON output exists and read it
    if os.path.exists(json_output):
        with open(json_output, 'r') as f:
            try:
                # Convert to our simplified format
                data = json.load(f)
                predictions = {}
                
                # Extract scores for each protein
                for i, prot in enumerate(data.get('pdb_chains', [])):
                    prot_preds = {}
                    scores = data.get('Y_hat', [])[i]
                    goterms = data.get('goterms', [])
                    
                    for j, term in enumerate(goterms):
                        if j < len(scores) and scores[j] >= args.threshold:
                            prot_preds[term] = float(scores[j])
                    
                    predictions[prot] = prot_preds
                
                # Save our simplified version
                with open(os.path.join(args.outdir, f"{os.path.basename(output_path)}_results.json"), 'w') as out_f:
                    json.dump(predictions, out_f, indent=2)
                
                return predictions
            except json.JSONDecodeError:
                print(f"Error: Could not parse JSON from {json_output}")
                return {}
    else:
        print(f"Error: Output file {json_output} was not created")
        return {}

# Single sequence
if args.seq:
    print(f"Processing direct sequence input")
    output_base = os.path.join(args.outdir, "sequence")
    predictions = run_deepfri_predict(output_base, args.model, args.ontology, seq=args.seq)

# Single FASTA file
elif args.fasta:
#     print(f"Processing FASTA file: {args.fasta}")
    base_name = os.path.basename(args.fasta).split('.')[0]
    output_base = os.path.join(args.outdir, base_name)
    predictions = run_deepfri_predict(output_base, args.model, args.ontology, fasta_fn=args.fasta)

# Single PDB file
elif args.pdb:
#     print(f"Processing PDB file: {args.pdb}")
    base_name = os.path.basename(args.pdb).split('.')[0]
    output_base = os.path.join(args.outdir, base_name)
    predictions = run_deepfri_predict(output_base, args.model, args.ontology, pdb=args.pdb)

# Directory of PDB files
elif args.pdb_dir:
#     print(f"Processing PDB directory: {args.pdb_dir}")
    output_base = os.path.join(args.outdir, "pdb_dir_results")
    all_predictions = run_deepfri_predict(output_base, args.model, args.ontology, pdb_dir=args.pdb_dir)
    
    # DeepFRI processes all PDBs in the directory at once
    # Save individual results for each protein for consistency
    for prot_name, preds in all_predictions.items():
        with open(os.path.join(args.outdir, f"{prot_name}_results.json"), "w") as f:
            json.dump({prot_name: preds}, f, indent=2)

# Directory of FASTA files
elif args.fasta_dir:
    # DeepFRI doesn't directly support a directory of FASTA files
    # We need to process each file individually
    fasta_files = glob.glob(os.path.join(args.fasta_dir, "*.fasta"))
    print(f"Found {len(fasta_files)} FASTA files in {args.fasta_dir}")
    
    all_predictions = {}
    for fasta_file in fasta_files:
        name = os.path.basename(fasta_file).split('.')[0]
#         print(f"Processing FASTA file: {fasta_file}")
        
        output_base = os.path.join(args.outdir, name)
        try:
            predictions = run_deepfri_predict(output_base, args.model, args.ontology, fasta_fn=fasta_file)
            if predictions:
                all_predictions.update(predictions)
        except Exception as e:
            print(f"Error processing FASTA file {fasta_file}: {e}")
    
    # Save combined results
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(all_predictions, f, indent=2)

else:
    print("Error: No input provided. Please specify --seq, --fasta, --pdb, --pdb_dir, or --fasta_dir.")
    exit(1)

print("Processing complete. Results saved to:", args.outdir)
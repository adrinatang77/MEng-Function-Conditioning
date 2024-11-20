import sys
from Bio import SeqIO
import torch
import esm
import numpy as np


def load_fasta(file_path):
    """Load sequences from a FASTA file."""
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences


def get_pLDDT(esmfold_model, sequence):
    """Run ESMFold to get pLDDT scores for a given sequence."""
    with torch.no_grad():
        results = esmfold_model.infer_pdb(sequence)
    pLDDT_scores = results["pLDDT"]
    return pLDDT_scores


def calculate_sequence_diversity(sequences):
    """Calculate diversity as the average pairwise sequence distance."""

    def hamming_distance(seq1, seq2):
        return sum(c1 != c2 for c1, c2 in zip(seq1, seq2)) / len(seq1)

    num_sequences = len(sequences)
    distances = []
    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            distances.append(hamming_distance(sequences[i], sequences[j]))
    return np.mean(distances) if distances else 0


def main(fasta_file):
    # Load sequences from FASTA file
    sequences = load_fasta(fasta_file)

    # Initialize ESMFold model
    esmfold_model, _ = esm.pretrained.esmfold_v1()
    esmfold_model = esmfold_model.eval().cuda()  # Run on GPU if available

    # Analyze pLDDTs and diversity
    pLDDTs = []
    for seq in sequences:
        pLDDT_scores = get_pLDDT(esmfold_model, seq)
        avg_pLDDT = np.mean(pLDDT_scores)
        pLDDTs.append(avg_pLDDT)
        print(f"Sequence pLDDT: {avg_pLDDT}")
    print(f"Overall average pLDDT {np.mean(np.array(plDDTs))}")

    # Calculate diversity
    diversity = calculate_sequence_diversity(sequences)
    print(f"Average Diversity: {diversity}")

    return pLDDTs, diversity


if __name__ == "__main__":
    # Check if the file path is provided
    if len(sys.argv) != 2:
        print("Usage: python3 esmfold_sequence_evaluation.py <path_to_fasta_file>")
        sys.exit(1)

    fasta_file = sys.argv[1]
    pLDDTs, diversity = main(fasta_file)

from collections import defaultdict
from .eval import OpenProtEval
from ..utils import protein
from ..utils.geometry import compute_lddt
from ..utils import residue_constants as rc
from ..tracks.sequence import MASK_IDX
from ..generate.sampler import OpenProtSampler
from ..generate.sequence import SequenceUnmaskingStepper
import numpy as np
import torch
import os
import math
import tqdm
import shutil
import torch.nn.functional as F
import subprocess
import json
import pandas as pd

from .seq_gen import SequenceGenerationEval

class FunctionConditioningEval(SequenceGenerationEval):
    def setup(self):
        super().setup()
        
        # Load GO vocabulary and ancestors
        with open(self.cfg.go_vocab, 'r') as f:
            self.go_vocab = json.load(f)
        with open(self.cfg.go_ancestors, 'r') as f:
            self.go_ancestors = json.load(f)
        
        # Load target GO terms for each sample
        self.sample_to_go_terms = {}
        
        if os.path.exists(self.cfg.target_go_terms):
            with open(self.cfg.target_go_terms, 'r') as f:
                lines = f.readlines()
            
            # Process each line and map to sample index
            for i, line in enumerate(lines):
                sample_name = f"sample{i}"
                direct_go_terms = line.strip().split()  # Support multiple GO terms per line
                go_terms = set()
                for term in direct_go_terms:
                    go_terms = go_terms | set(self.go_ancestors.get(term, [term]))
                self.sample_to_go_terms[sample_name] = go_terms
            
            print(f"Loaded GO terms for {len(self.sample_to_go_terms)} samples from {self.cfg.target_go_terms}")
        else:
            print("Warning: target_go_terms_file not specified or file not found")

    def __getitem__(self, idx):
        L = self.cfg.sample_length
        sample_name = f"sample{idx}"
        
        # Get GO terms for this sample
        go_terms = self.sample_to_go_terms.get(sample_name, [])
        
        # Create function conditioning array
        max_depth = self.cfg.max_depth
        func_cond = np.zeros((L, max_depth), dtype=int)

        go_term_indices = list(set(np.array([go_index for go_term in go_terms if (go_index := self.go_vocab.get(go_term)) is not None], dtype=int)))
        num_go_terms = len(go_term_indices)

        func_cond[:, :num_go_terms] = go_term_indices
        
        # Create data with function conditioning
        data = self.make_data(
            name=sample_name,
            seqres="A"*L,
            seq_mask=np.ones(L, dtype=np.float32),
            seq_noise=np.ones(L, dtype=np.float32),
            residx=np.arange(L, dtype=np.float32),
            func_cond=func_cond,  # Add function conditioning
        )
        
        return data
    
    def compute_metrics(self, rank=0, world_size=1, device=None, savedir=".", logger=None):
        # Call parent method to compute basic metrics
        super().compute_metrics(rank, world_size, device, savedir, logger)
        
        idx = list(range(rank, len(self), world_size))
        df = defaultdict(dict)
        self.run_deepfri_recall(idx, rank, world_size, savedir, logger, df)

        # Save DataFrame with DeepFRI metrics
        df_pd = pd.DataFrame(df).T
        df_pd.to_csv(f"{savedir}/deepfri_metrics_rank{rank}.csv")

        if world_size > 1:
            torch.distributed.barrier()

        if rank == 0:
            dfs = []
            for r in range(world_size):
                csv_path = f"{savedir}/deepfri_metrics_rank{r}.csv"
                if os.path.exists(csv_path):
                    dfs.append(pd.read_csv(csv_path, index_col=0))
                    os.remove(csv_path)

            if dfs:
                combined_df = pd.concat(dfs)
                combined_df.to_csv(f"{savedir}/deepfri_metrics.csv")
    
#     def run_deepfri_recall(self, idx, rank, world_size, savedir, logger, df):
#         """
#         Run DeepFRI evaluation on generated proteins and calculate metrics.
#         """
#         # Create directories
#         deepfri_dir = f"{savedir}/rank{rank}/deepfri"
#         os.makedirs(deepfri_dir, exist_ok=True)
        
#         # Create directory for FASTA files
#         fasta_dir = f"{deepfri_dir}/fastas"
#         os.makedirs(fasta_dir, exist_ok=True)
        
#         # Copy files to the directory
#         for i in idx:
#             sample_name = f"sample{i}"
#             fasta_file = f"{savedir}/{sample_name}.fasta"
            
#             if os.path.exists(fasta_file):
#                 cmd = ['cp', fasta_file, fasta_dir]
#                 subprocess.run(cmd)
        
#         # Run DeepFRI with sequence model
#         seq_output_dir = f"{deepfri_dir}/seq_results"
#         os.makedirs(seq_output_dir, exist_ok=True)
        
#         cmd = [
#             "bash",
#             "openprot/scripts/switch_conda_env.sh",
#             "deepfri",
#             "python",
#             "-m",
#             "openprot.scripts.deepfri",
#             "--fasta_dir", fasta_dir,
#             "--outdir", seq_output_dir,
#             "--model", "sequence",
#             "--ontology", "mf",
#             "--threshold", str(self.cfg.threshold),
#         ]
#         subprocess.run(cmd)
        
#         # Process results and calculate metrics
#         seq_correct = 0
#         seq_total = 0
#         seq_confidences = []
        
#         # Track per-sample metrics
#         results = {}
        
#         for i in idx:
#             sample_name = f"sample{i}"
            
#             # Get target GO terms for this sample
#             target_go_terms = self.sample_to_go_terms.get(sample_name, [])
#             if not target_go_terms:
#                 continue
                
#             # Process sequence results
#             seq_result_file = f"{seq_output_dir}/{sample_name}_results.json"
#             if os.path.exists(seq_result_file):
#                 with open(seq_result_file, 'r') as f:
#                     import json
#                     seq_result = json.load(f)
                    
#                     # Get predictions above threshold
#                     seq_predictions = seq_result.get(sample_name, {})
                    
#                     # Check if any target term was predicted
#                     seq_total += 1
#                     predicted_any = any(term in seq_predictions for term in target_go_terms)
#                     if predicted_any:
#                         seq_correct += 1
                    
# #                     # Track confidence for target terms
# #                     sample_confidences = []
# #                     for term in target_go_terms:
# #                         confidence = seq_predictions.get(term, 0)
# #                         seq_confidences.append(confidence)
# #                         sample_confidences.append(confidence)
#                     sample_confidences = list(seq_predictions.values())
#                     seq_confidences.extend(sample_confidences)
                    
                    
#                     # Store per-sample metrics
#                     results[sample_name] = {
#                         'target_terms': ','.join(target_go_terms),
#                         'predicted': 1 if predicted_any else 0,
#                         'avg_confidence': np.mean(sample_confidences) if sample_confidences else 0
#                     }
                    
#                     # Store top predictions for reference
#                     top_predictions = sorted(seq_predictions.items(), key=lambda x: x[1], reverse=True)[:5]
#                     for j, (term, conf) in enumerate(top_predictions):
#                         results[sample_name][f'top{j+1}_term'] = term
#                         results[sample_name][f'top{j+1}_conf'] = conf
        
#         # Calculate aggregate metrics
#         seq_recall = seq_correct / seq_total if seq_total > 0 else 0
#         seq_avg_conf = np.mean(seq_confidences) if seq_confidences else 0
        
#         # Add metrics to dataframe
#         df["deepfri_summary"] = {
#             "seq_recall": seq_recall,
#             "seq_avg_conf": seq_avg_conf,
#             "seq_correct": seq_correct,
#             "seq_total": seq_total
#         }
        
#         # Add per-sample results
#         for sample_name, result in results.items():
#             df[sample_name] = result
        
#         # Log metrics
#         if logger is not None:
#             logger.log(f"{self.cfg.name}/deepfri_seq_recall", seq_recall)
#             logger.log(f"{self.cfg.name}/deepfri_seq_avg_conf", seq_avg_conf)

    def run_deepfri_recall(self, idx, rank, world_size, savedir, logger, df):

        # Create directories
        deepfri_dir = f"{savedir}/rank{rank}/deepfri"
        os.makedirs(deepfri_dir, exist_ok=True)
        fasta_dir = f"{deepfri_dir}/fastas"
        os.makedirs(fasta_dir, exist_ok=True)

        for i in idx:
            sample_name = f"sample{i}"
            fasta_file = f"{savedir}/{sample_name}.fasta"
            if os.path.exists(fasta_file):
                subprocess.run(['cp', fasta_file, fasta_dir])

        seq_output_dir = f"{deepfri_dir}/seq_results"
        os.makedirs(seq_output_dir, exist_ok=True)

        subprocess.run([
            "bash", "openprot/scripts/switch_conda_env.sh", "deepfri",
            "python", "-m", "openprot.scripts.deepfri",
            "--fasta_dir", fasta_dir,
            "--outdir", seq_output_dir,
            "--model", "sequence",
            "--ontology", "mf",
            "--threshold", str(self.cfg.threshold),
        ])

        # Metrics
        seq_total, seq_correct = 0, 0
        total_pred_terms, total_target_terms, total_correct = 0, 0, 0
        sample_recalls = []
        sample_precisions = []
        seq_confidences = []
        results = {}

        for i in idx:
            sample_name = f"sample{i}"
            target_go_terms = self.sample_to_go_terms.get(sample_name, [])
            if not target_go_terms:
                continue

            # Expand target GO terms with ancestors
            expanded_target_terms = set()
            for term in target_go_terms:
                expanded_target_terms |= set(self.go_ancestors.get(term, [term]))

            seq_result_file = f"{seq_output_dir}/{sample_name}_results.json"
            if not os.path.exists(seq_result_file):
                continue

            with open(seq_result_file, 'r') as f:
                seq_result = json.load(f)
            seq_predictions = seq_result.get(sample_name, {})
            predicted_terms = set(seq_predictions.keys())

            correct_terms = predicted_terms.intersection(expanded_target_terms)

            # Binary per-sample: at least one matched
            seq_total += 1
            if correct_terms:
                seq_correct += 1

            # Per-term metrics
            total_pred_terms += len(predicted_terms)
            total_target_terms += len(expanded_target_terms)
            total_correct += len(correct_terms)

            # Sample-wise precision/recall
            recall_i = len(correct_terms) / len(expanded_target_terms) if expanded_target_terms else 0
            precision_i = len(correct_terms) / len(predicted_terms) if predicted_terms else 0
            sample_recalls.append(recall_i)
            sample_precisions.append(precision_i)

            # Confidence
            sample_confidences = list(seq_predictions.values())
            seq_confidences.extend(sample_confidences)

            # Results per sample
            results[sample_name] = {
                'target_terms': ','.join(target_go_terms),
                'predicted': 1 if correct_terms else 0,
                'avg_confidence': np.mean(sample_confidences) if sample_confidences else 0
            }

            top_predictions = sorted(seq_predictions.items(), key=lambda x: x[1], reverse=True)[:5]
            for j, (term, conf) in enumerate(top_predictions):
                results[sample_name][f'top{j+1}_term'] = term
                results[sample_name][f'top{j+1}_conf'] = conf

        # Final metrics
        recall = total_correct / total_target_terms if total_target_terms > 0 else 0
        precision = total_correct / total_pred_terms if total_pred_terms > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        coverage = seq_correct / seq_total if seq_total > 0 else 0
        avg_conf = np.mean(seq_confidences) if seq_confidences else 0

        sample_recall = np.mean(sample_recalls) if sample_recalls else 0
        sample_precision = np.mean(sample_precisions) if sample_precisions else 0
        sample_f1 = (
            2 * sample_precision * sample_recall / (sample_precision + sample_recall)
            if (sample_precision + sample_recall) > 0 else 0
        )

        # Add to dataframe
        df["deepfri_summary"] = {
            "seq_term_recall": recall,
            "seq_term_precision": precision,
            "seq_term_f1": f1,
            "seq_samplewise_recall": sample_recall,
            "seq_samplewise_precision": sample_precision,
            "seq_samplewise_f1": sample_f1,
            "seq_sample_coverage": coverage,
            "seq_avg_conf": avg_conf,
            "seq_correct_samples": seq_correct,
            "seq_total_samples": seq_total
        }

        for sample_name, result in results.items():
            df[sample_name] = result

        # Logging
        if logger is not None:
            logger.log(f"{self.cfg.name}/deepfri_seq_term_recall", recall)
            logger.log(f"{self.cfg.name}/deepfri_seq_term_precision", precision)
            logger.log(f"{self.cfg.name}/deepfri_seq_term_f1", f1)
            logger.log(f"{self.cfg.name}/deepfri_seq_samplewise_recall", sample_recall)
            logger.log(f"{self.cfg.name}/deepfri_seq_samplewise_precision", sample_precision)
            logger.log(f"{self.cfg.name}/deepfri_seq_samplewise_f1", sample_f1)
            logger.log(f"{self.cfg.name}/deepfri_seq_sample_coverage", coverage)
            logger.log(f"{self.cfg.name}/deepfri_seq_avg_conf", avg_conf)


    def run_batch(
        self,
        model,
        batch: dict,
        noisy_batch: dict,
        savedir=".", 
        device=None,
        logger=None
    ):


        sampler = OpenProtSampler(schedules={
            'sequence': lambda t: 1-t,
        }, steppers=[
            SequenceUnmaskingStepper(self.cfg)
        ])
        
        sample, extra = sampler.sample(model, noisy_batch, self.cfg.steps)
        
        B = len(sample['aatype'])
        for i in range(B):
            name = batch["name"][i]

            seq = "".join([rc.restypes_with_x[aa] for aa in sample["aatype"][i]])
            with open(f"{savedir}/{name}.fasta", "w") as f:
                f.write(f">{name}\n")  # FASTA format header
                f.write(seq + "\n")

            if logger is not None:
                logger.log(f"{self.cfg.name}/seqent", self.compute_sequence_entropy(seq))
                
            with open(f"{savedir}/{name}_traj.fasta", "w") as f:
                for seqs in extra['seq_traj']:
                    seq = "".join([rc.restypes_with_x[aa] for aa in seqs[i]])
                    seq = seq.replace('X', '-')
                    f.write(seq+'\n')
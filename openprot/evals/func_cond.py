# from .eval import OpenProtEval
# # import foldcomp
# from ..utils import protein
# from ..utils.prot_utils import make_ca_prot, write_ca_traj, compute_tmscore, aatype_to_seqres
# from ..utils.geometry import compute_lddt, rmsdalign, compute_rmsd
# from ..utils import residue_constants as rc
# from ..generate.sampler import OpenProtSampler
# from ..generate.structure import EDMDiffusionStepper, GaussianFMStepper
# from ..generate.sequence import SequenceUnmaskingStepper
# import numpy as np
# import torch
# import os, tqdm, math, subprocess
# import pandas as pd
# from biopandas.pdb import PandasPdb
# from ..utils.secondary import assign_secondary_structures
# from collections import defaultdict

# class FunctionConditioningEval(OpenProtEval):
#     def setup(self):
#         def t_skew_func(t, skew):
#             midpoint_y = 0.5 + skew / 2# [0, 1]
#             midpoint_x = 0.5 
#             if t < midpoint_x:
#                 return midpoint_y / midpoint_x * t
#             else:
#                 return midpoint_y + (1 - midpoint_y) / (1 - midpoint_x) * (t - midpoint_x)

#         self.seq_sched_fn = lambda t: 1-t_skew_func(t, -self.cfg.skew)
        
#     def run(self, model):
#         NotImplemented

#     def __len__(self):
#         return self.cfg.num_samples

#     def __getitem__(self, idx):
#         L = self.cfg.sample_length
        
#         if self.cfg.truncate:
#             seq_noise = self.seq_sched_fn(self.cfg.truncate)
#         else:
#             seq_noise = 1

#         if self.cfg.dir is not None:
#             with open(f"{self.cfg.dir}/sample{idx}.fasta") as f:
# #                 prot = protein.from_pdb_string(f.read())
            
# #             seqres = aatype_to_seqres(prot.aatype)
#             data = self.make_data(
#                 name=f"sample{idx}",
#                 seqres=seqres,
#                 seq_mask=np.ones(L),
#                 seq_noise=np.ones(L, dtype=np.float32) * seq_noise,
#                 residx=np.arange(L, dtype=np.float32),
#                 func_cond=None
#                 # prep func cond
#             )
#             return data
        
#         data = self.make_data(
#             name=f"sample{idx}",
#             seqres="A"*L,
#             seq_mask=np.ones(L, dtype=np.float32),
#             seq_noise=np.ones(L, dtype=np.float32) * seq_noise,
#             residx=np.arange(L, dtype=np.float32),
#             func_cond=None
#             # prep func cond
#         )
#         return data

#     def run_designability(self, idx, rank, world_size, savedir, logger, df):
#         for i in idx:
#             cmd = ['cp', f"{savedir}/sample{i}.fasta", f"{savedir}/rank{rank}"]
#             subprocess.run(cmd)
                
#         cmd = [
#             "bash",
#             "scripts/switch_conda_env.sh",
#             "eval",
#             "python",
#             "-m",
#             "scripts.esmfold",
#             "--outdir",
#             f"{savedir}/rank{rank}",
#             "--dir",
#             f"{savedir}/rank{rank}",
#             # "--print",
#             "--device",
#             str(torch.cuda.current_device())
#         ]
#         out = subprocess.run(cmd) 
        
#         for i in idx:
            
#             with open(f"{savedir}/sample{i}.pdb") as f:
#                 prot = protein.from_pdb_string(f.read())
#             with open(f"{savedir}/rank{rank}/sample{i}.pdb") as f:
#                 pred = protein.from_pdb_string(f.read())
#             lddt = compute_lddt(
#                 torch.from_numpy(pred.atom_positions[:,1]), 
#                 torch.from_numpy(prot.atom_positions[:,1]), 
#                 torch.from_numpy(prot.atom_mask[:,1])
#             )
#             rmsd = compute_rmsd(
#                 torch.from_numpy(pred.atom_positions[:,1]),  
#                 torch.from_numpy(prot.atom_positions[:,1])
#             )
#             tmscore = compute_tmscore(  # second is reference
#                 coords1=pred.atom_positions[:,1],
#                 coords2=prot.atom_positions[:,1],
#             )['tm']

#             plddt = PandasPdb().read_pdb(f"{savedir}/rank{rank}/sample{i}.pdb").df['ATOM']['b_factor'].mean()
            
#             if logger is not None:
#                 logger.log(f"{self.cfg.name}/sclddt", lddt)
#                 logger.log(f"{self.cfg.name}/scrmsd", rmsd)
#                 logger.log(f"{self.cfg.name}/scrmsd<2", (rmsd < 2).float())
#                 logger.log(f"{self.cfg.name}/scTM", tmscore)
#                 logger.log(f"{self.cfg.name}/sclddt>80", (lddt > 0.8).float())
#                 logger.log(f"{self.cfg.name}/scTM>80", tmscore > 0.8)
#                 logger.log(f"{self.cfg.name}/plddt", plddt)

#             df[f"sample{i}"]["plddt"] = plddt
#             df[f"sample{i}"]["scrmsd"] = float(rmsd)
#             df[f"sample{i}"]["sctm"] = tmscore
#             df[f"sample{i}"]["sclddt"] = float(lddt)

#     def make_plot(self, idx, rank, world_size, savedir, logger, df):
#         cmd = [
#             "bash",
#             "scripts/switch_conda_env.sh",
#             "pymol",
#             "python",
#             "-m",
#             "scripts.visualize",
#             "--dir",
#             savedir,
#             "--out",
#             f"{savedir}/out.png",
#             "--annotate",
#         ]
#         out = subprocess.run(cmd) 
    
#     def run_deepfri_recall(self, idx, rank, world_size, save_dir, logger, df):
#         # Create directories
#         deepfri_dir = f"{savedir}/rank{rank}/deepfri"
#         os.makedirs(deepfri_dir, exist_ok=True)

#         # Create a directory for PDB files and a directory for FASTA files
#         fasta_dir = f"{deepfri_dir}/fastas"
#         os.makedirs(pdb_dir, exist_ok=True)
#         os.makedirs(fasta_dir, exist_ok=True)

#         # Copy files to their respective directories
#         for i in idx:
#             sample_name = f"sample{i}"

#             # Copy PDB and FASTA files
#             pdb_file = f"{savedir}/{sample_name}.pdb"
#             fasta_file = f"{savedir}/{sample_name}.fasta"

#             if os.path.exists(pdb_file):
#                 cmd = ['cp', pdb_file, pdb_dir]
#                 subprocess.run(cmd)

#             if os.path.exists(fasta_file):
#                 cmd = ['cp', fasta_file, fasta_dir]
#                 subprocess.run(cmd)

#         # Run DeepFRI with fold model (structure-based)
#         fold_output_dir = f"{deepfri_dir}/fold_results"
#         os.makedirs(fold_output_dir, exist_ok=True)

#         cmd = [
#             "bash",
#             "scripts/switch_conda_env_with_CUDA.sh",
#             "deepfri",
#             str(torch.cuda.current_device()),
#             "python",
#             "-m",
#             "scripts.deepfri",
#             "--pdb_dir", pdb_dir,
#             "--outdir", fold_output_dir,
#             "--model", "fold",
#             "--ontology", "mf",
#             "--threshold", "0.5",
#         ]
#         subprocess.run(cmd)

#         # Run DeepFRI with sequence model
#         seq_output_dir = f"{deepfri_dir}/seq_results"
#         os.makedirs(seq_output_dir, exist_ok=True)

#         cmd = [
#             "bash",
#             "scripts/switch_conda_env_with_CUDA.sh",
#             "deepfri",
#             str(torch.cuda.current_device()),
#             "python",
#             "-m",
#             "scripts.deepfri",
#             "--fasta_dir", fasta_dir,
#             "--outdir", seq_output_dir,
#             "--model", "sequence",
#             "--ontology", "mf",
#             "--threshold", "0.5",
#             "--device", str(torch.cuda.current_device())
#         ]
#         subprocess.run(cmd)

#         # Process results and calculate metrics directly
#         fold_correct = 0
#         fold_total = 0
#         fold_confidences = []

#         seq_correct = 0
#         seq_total = 0
#         seq_confidences = []

#         for i in idx:
#             sample_name = f"sample{i}"

#             # TODO: Get target terms for this sample
#             # Implement based on how target terms are stored in your model
#             target_terms = self.get_target_go_terms(sample_name, savedir)

#             if not target_terms:
#                 continue

#             # Process fold results
#             fold_result_file = f"{fold_output_dir}/{sample_name}_results.json"
#             if os.path.exists(fold_result_file):
#                 with open(fold_result_file, 'r') as f:
#                     fold_result = json.load(f)

#                     # Get predictions above threshold
#                     fold_predictions = fold_result.get('predictions', {})

#                     # Check if any target term was predicted
#                     fold_total += 1
#                     predicted_any = any(term in fold_predictions for term in target_terms)
#                     if predicted_any:
#                         fold_correct += 1

#                     # Track confidence for target terms
#                     for term in target_terms:
#                         confidence = fold_predictions.get(term, 0)
#                         fold_confidences.append(confidence)

#             # Process sequence results
#             seq_result_file = f"{seq_output_dir}/{sample_name}_results.json"
#             if os.path.exists(seq_result_file):
#                 with open(seq_result_file, 'r') as f:
#                     seq_result = json.load(f)

#                     # Get predictions above threshold
#                     seq_predictions = seq_result.get('predictions', {})

#                     # Check if any target term was predicted
#                     seq_total += 1
#                     predicted_any = any(term in seq_predictions for term in target_terms)
#                     if predicted_any:
#                         seq_correct += 1

#                     # Track confidence for target terms
#                     for term in target_terms:
#                         confidence = seq_predictions.get(term, 0)
#                         seq_confidences.append(confidence)

#         # Calculate aggregate metrics
#         fold_recall = fold_correct / fold_total if fold_total > 0 else 0
#         fold_avg_conf = np.mean(fold_confidences) if fold_confidences else 0

#         seq_recall = seq_correct / seq_total if seq_total > 0 else 0
#         seq_avg_conf = np.mean(seq_confidences) if seq_confidences else 0

#         # Save metrics to a summary file
#         metrics = {
#             "fold_recall": fold_recall,
#             "fold_avg_confidence": fold_avg_conf,
#             "fold_correct": fold_correct,
#             "fold_total": fold_total,
#             "seq_recall": seq_recall,
#             "seq_avg_confidence": seq_avg_conf,
#             "seq_correct": seq_correct,
#             "seq_total": seq_total
#         }

# #         with open(f"{deepfri_dir}/metrics.json", 'w') as f:
# #             json.dump(metrics, f, indent=2)

#         # Add metrics to dataframe
#         df["deepfri_summary"] = {
#             "fold_recall": fold_recall,
#             "fold_avg_conf": fold_avg_conf,
#             "seq_recall": seq_recall,
#             "seq_avg_conf": seq_avg_conf
#         }

#         # Log metrics
#         if logger is not None:
#             logger.log(f"{self.cfg.name}/deepfri_fold_recall", fold_recall)
#             logger.log(f"{self.cfg.name}/deepfri_fold_avg_conf", fold_avg_conf)
#             logger.log(f"{self.cfg.name}/deepfri_seq_recall", seq_recall)
#             logger.log(f"{self.cfg.name}/deepfri_seq_avg_conf", seq_avg_conf)

        
#     def save_df(self, idx, rank, world_size, savedir, logger, df):
#         df = pd.DataFrame(df).T.astype(float)
#         df.to_csv(f"{savedir}/rank{rank}.csv")
    
#         if world_size > 1:
#             torch.distributed.barrier()
#         if rank == 0:
#             dfs = []
#             for r in range(world_size):
#                 dfs.append(pd.read_csv(f"{savedir}/rank{r}.csv", index_col=0))
#                 subprocess.run(["rm", f"{savedir}/rank{r}.csv"])
#             df = pd.concat(dfs).sort_index()
#             df.to_csv(f"{savedir}/info.csv")
        
#     def compute_metrics(
#         self, rank=0, world_size=1, device=None, savedir=".", logger=None
#     ):

#         idx = list(range(rank, len(self), world_size))
#         os.makedirs(f"{savedir}/rank{rank}", exist_ok=True)

#         df = defaultdict(dict) 
        
#         if self.cfg.run_designability:
#             torch.cuda.empty_cache()
#             self.run_designability(idx, rank, world_size, savedir, logger, df)

#         if self.cfg.run_deepfri_recall:
#             self.run_deepfri_crecall(idx, rank, world_size, savedir, logger, df)
        
#         # this has to be last
#         self.save_df(idx, rank, world_size, savedir, logger, df)
#         if self.cfg.run_plot and rank == 0:
#             self.make_plot(idx, rank, world_size, savedir, logger, df)
            
#     def run_batch(
#         self,
#         model,
#         batch: dict,
#         noisy_batch: dict,
#         savedir=".", 
#         device=None,
#         logger=None
#     ):
                    
#         sampler = OpenProtSampler(schedules={
#             'sequence': lambda t: 1-t, # self.seq_sched_fn,
#         }, steppers=[
#             SequenceUnmaskingStepper(self.cfg)
#         ])
        
#         sample, extra = sampler.sample(model, noisy_batch, self.cfg.steps)
#         B = len(sample['aatype'])
#         for i in range(B):
#             name = batch["name"][i]

#             seq = "".join([rc.restypes_with_x[aa] for aa in sample["aatype"][i]])
#             with open(f"{savedir}/{name}.fasta", "w") as f:
#                 f.write(f">{name}\n")  # FASTA format header
#                 f.write(seq + "\n")

#             if logger is not None:
#                 logger.log(f"{self.cfg.name}/seqent", self.compute_sequence_entropy(seq))

#             with open(f"{savedir}/{name}_traj.fasta", "w") as f:
#                 for seqs in extra['seq_traj']:
#                     seq = "".join([rc.restypes_with_x[aa] for aa in seqs[i]])
#                     seq = seq.replace('X', '-')
#                     f.write(seq+'\n')
       
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
    
    def compute_deepfri_recall(self, rank=0, world_size=1, device=None, savedir=".", logger=None):
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
    
    def run_deepfri_recall(self, idx, rank, world_size, savedir, logger, df):
        """
        Run DeepFRI evaluation on generated proteins and calculate metrics.
        """
        # Create directories
        deepfri_dir = f"{savedir}/rank{rank}/deepfri"
        os.makedirs(deepfri_dir, exist_ok=True)
        
        # Create directory for FASTA files
        fasta_dir = f"{deepfri_dir}/fastas"
        os.makedirs(fasta_dir, exist_ok=True)
        
        # Copy files to the directory
        for i in idx:
            sample_name = f"sample{i}"
            fasta_file = f"{savedir}/{sample_name}.fasta"
            
            if os.path.exists(fasta_file):
                cmd = ['cp', fasta_file, fasta_dir]
                subprocess.run(cmd)
        
        # Run DeepFRI with sequence model
        seq_output_dir = f"{deepfri_dir}/seq_results"
        os.makedirs(seq_output_dir, exist_ok=True)
        
        cmd = [
            "bash",
            "openprot/scripts/switch_conda_env.sh",
            "deepfri",
            "python",
            "-m",
            "openprot.scripts.deepfri",
            "--fasta_dir", fasta_dir,
            "--outdir", seq_output_dir,
            "--model", "sequence",
            "--ontology", "mf",
            "--threshold", str(self.cfg.threshold),
        ]
        subprocess.run(cmd)
        
        # Process results and calculate metrics
        seq_correct = 0
        seq_total = 0
        seq_confidences = []
        
        # Track per-sample metrics
        results = {}
        
        for i in idx:
            sample_name = f"sample{i}"
            
            # Get target GO terms for this sample
            target_go_terms = self.sample_to_go_terms.get(sample_name, [])
            if not target_go_terms:
                continue
                
            # Process sequence results
            seq_result_file = f"{seq_output_dir}/{sample_name}_results.json"
            if os.path.exists(seq_result_file):
                with open(seq_result_file, 'r') as f:
                    import json
                    seq_result = json.load(f)
                    
                    # Get predictions above threshold
                    seq_predictions = seq_result.get(sample_name, {})
                    
                    # Check if any target term was predicted
                    seq_total += 1
                    predicted_any = any(term in seq_predictions for term in target_go_terms)
                    if predicted_any:
                        seq_correct += 1
                    
#                     # Track confidence for target terms
#                     sample_confidences = []
#                     for term in target_go_terms:
#                         confidence = seq_predictions.get(term, 0)
#                         seq_confidences.append(confidence)
#                         sample_confidences.append(confidence)
                    sample_confidences = list(seq_predictions.values())
                    seq_confidences.extend(sample_confidences)
                    
                    
                    # Store per-sample metrics
                    results[sample_name] = {
                        'target_terms': ','.join(target_go_terms),
                        'predicted': 1 if predicted_any else 0,
                        'avg_confidence': np.mean(sample_confidences) if sample_confidences else 0
                    }
                    
                    # Store top predictions for reference
                    top_predictions = sorted(seq_predictions.items(), key=lambda x: x[1], reverse=True)[:5]
                    for j, (term, conf) in enumerate(top_predictions):
                        results[sample_name][f'top{j+1}_term'] = term
                        results[sample_name][f'top{j+1}_conf'] = conf
        
        # Calculate aggregate metrics
        seq_recall = seq_correct / seq_total if seq_total > 0 else 0
        seq_avg_conf = np.mean(seq_confidences) if seq_confidences else 0
        
        # Add metrics to dataframe
        df["deepfri_summary"] = {
            "seq_recall": seq_recall,
            "seq_avg_conf": seq_avg_conf,
            "seq_correct": seq_correct,
            "seq_total": seq_total
        }
        
        # Add per-sample results
        for sample_name, result in results.items():
            df[sample_name] = result
        
        # Log metrics
        if logger is not None:
            logger.log(f"{self.cfg.name}/deepfri_seq_recall", seq_recall)
            logger.log(f"{self.cfg.name}/deepfri_seq_avg_conf", seq_avg_conf)
            
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
        
        self.compute_deepfri_recall(logger=logger, savedir=savedir)

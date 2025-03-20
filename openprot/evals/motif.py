from .codesign import CodesignEval
import pandas as pd
import pickle, os
import torch
from ..utils.misc_utils import temp_seed
import numpy as np
from rdkit import Chem
from ..utils import residue_constants as rc
from ..data.data import OpenProtData
from ..generate.sampler import OpenProtSampler
from ..utils.prot_utils import make_ca_prot, aatype_to_seqres
from ..generate.structure import EDMDiffusionStepper, GaussianFMStepper
from ..generate.sequence import SequenceUnmaskingStepper
from ..utils.prot_utils import write_mmcif, aatype_to_seqres, compute_tmscore
from ..utils import protein 
from ..utils.geometry import rmsdalign, compute_rmsd, compute_lddt
from ..utils.motif_utils import load_motif_spec, sample_motif_mask
from ..utils.structure import Structure, Polypeptide, Ligand
from collections import defaultdict
from biopandas.pdb import PandasPdb
import subprocess

def masked_center(x, mask=None, eps=1e-5):
    if mask is None:
        return x - x.mean(-2, keepdims=True)
    mask = mask[..., None]
    com = (x * mask).sum(-2, keepdims=True) / (eps + mask.sum(-2, keepdims=True))
    return x - com

class MotifEval(CodesignEval):
    def __len__(self):
        return len(self.cfg.path) * self.cfg.num_samples
        
    def __getitem__(self, idx):

        path = self.cfg.path[idx % len(self.cfg.path)]
        spec = load_motif_spec(path)
        masks = sample_motif_mask(spec)
        motif_mask = masks['sequence']
        with open(path) as f:
            prot = protein.from_pdb_string(f.read())

        L = len(motif_mask)
        motif_ca = np.zeros((L, 3))
        motif_ca[motif_mask] = prot.atom_positions[:,1]
        motif_aatype = np.zeros(L, dtype=int)
        motif_aatype[motif_mask] = prot.aatype

        name = path.split('/')[-1].split('.')[0]
        data = self.make_data(
            name=f"{name}_sample{idx // len(self.cfg.path)}",
            seqres=aatype_to_seqres(motif_aatype),
            seq_mask=np.ones(L),
            #seq_noise=np.ones(L),
            seq_noise=~motif_mask,
            struct_noise=np.ones(L) * self.cfg.struct.edm.sigma_max,
            struct=np.zeros((L, 3)),
            struct_mask=np.ones(L),
            ref_conf=masked_center(motif_ca, motif_mask),
            ref_conf_mask=motif_mask,
            struct_align_mask=np.ones(L),
            residx=np.arange(L),
        )
        return data

    def compute_metrics(
        self, rank=0, world_size=1, device=None, savedir=".", logger=None
    ):
        idx = list(range(rank, len(self), world_size))
        os.makedirs(f"{savedir}/rank{rank}", exist_ok=True)

        df = defaultdict(dict) 
        
        if self.cfg.run_designability:
            torch.cuda.empty_cache()
            self.run_designability(idx, rank, world_size, savedir, logger, df)
                   
        # if self.cfg.run_secondary:
        #     self.run_secondary(idx, rank, world_size, savedir, logger, df)

        # if self.cfg.run_diversity:
        #     self.run_diversity(idx, rank, world_size, savedir, logger, df)

        if self.cfg.run_pmpnn_designability:
            torch.cuda.empty_cache()
            self.run_pmpnn_designability(idx, rank, world_size, savedir, logger, df)
        
        # this has to be last
        self.save_df(idx, rank, world_size, savedir, logger, df)
        # if self.cfg.run_plot and rank == 0:
        #     self.make_plot(idx, rank, world_size, savedir, logger, df)


    def run_designability(self, idx, rank, world_size, savedir, logger, df):
        for i in idx:
            name = self[i]['name']
            cmd = ['cp', f"{savedir}/{name}.fasta", f"{savedir}/rank{rank}"]
            subprocess.run(cmd)
                
        cmd = [
            "bash",
            "scripts/switch_conda_env.sh",
            "eval",
            "python",
            "-m",
            "scripts.esmfold",
            "--outdir",
            f"{savedir}/rank{rank}",
            "--dir",
            f"{savedir}/rank{rank}",
            # "--print",
            "--device",
            str(torch.cuda.current_device())
        ]
        out = subprocess.run(cmd) 
        
        for i in idx:
            name = self[i]['name']
            with open(f"{savedir}/{name}.pdb") as f:
                prot = protein.from_pdb_string(f.read())
            with open(f"{savedir}/rank{rank}/{name}.pdb") as f:
                pred = protein.from_pdb_string(f.read())
            with open(f"{savedir}/{name}_motif.pdb") as f:
                motif = protein.from_pdb_string(f.read())
            
            motif_rmsd = compute_rmsd(
                torch.from_numpy(pred.atom_positions[motif.residue_index-1,1]),
                torch.from_numpy(prot.atom_positions[motif.residue_index-1,1]),
            )
            
            lddt = compute_lddt(
                torch.from_numpy(pred.atom_positions[:,1]), 
                torch.from_numpy(prot.atom_positions[:,1]), 
                torch.from_numpy(prot.atom_mask[:,1])
            )
            rmsd = compute_rmsd(
                torch.from_numpy(pred.atom_positions[:,1]),  
                torch.from_numpy(prot.atom_positions[:,1])
            )
            tmscore = compute_tmscore(  # second is reference
                coords1=pred.atom_positions[:,1],
                coords2=prot.atom_positions[:,1],
            )['tm']

            plddt = PandasPdb().read_pdb(f"{savedir}/rank{rank}/{name}.pdb").df['ATOM']['b_factor'].mean()
            
            if logger is not None:
                logger.log(f"{self.cfg.name}/sclddt", lddt)
                logger.log(f"{self.cfg.name}/scrmsd", rmsd)
                logger.log(f"{self.cfg.name}/scrmsd<2", (rmsd < 2).float())
                logger.log(f"{self.cfg.name}/scTM", tmscore)
                logger.log(f"{self.cfg.name}/sclddt>80", (lddt > 0.8).float())
                logger.log(f"{self.cfg.name}/scTM>80", tmscore > 0.8)
                logger.log(f"{self.cfg.name}/plddt", plddt)
                logger.log(f"{self.cfg.name}/mRMSD", motif_rmsd)
                logger.log(
                    f"{self.cfg.name}/success", 
                    (rmsd < 2).float() * (motif_rmsd < 1).float()
                )

            df[f"sample{i}"]["plddt"] = plddt
            df[f"sample{i}"]["scrmsd"] = float(rmsd)
            df[f"sample{i}"]["sctm"] = tmscore
            df[f"sample{i}"]["sclddt"] = float(lddt)
            df[f"sample{i}"]["mRMSD"] = float(motif_rmsd)
    

    def run_pmpnn_designability(self, idx, rank, world_size, savedir, logger, df):


        os.makedirs(f"{savedir}/rank{rank}/pmpnn/pdbs", exist_ok=True)
        os.makedirs(f"{savedir}/rank{rank}/pmpnn/motif_pdbs", exist_ok=True)
        for i in idx:
            name = self[i]['name']
            
            cmd = [
                'cp',
                f"{savedir}/{name}.pdb",
                f"{savedir}/rank{rank}/pmpnn/pdbs"
            ]
            subprocess.run(cmd)
            cmd = [
                'cp',
                f"{savedir}/{name}_motif.pdb",
                f"{savedir}/rank{rank}/pmpnn/motif_pdbs/{name}.pdb"
            ]
            subprocess.run(cmd)
            
        cmd = [
            "bash",
            "scripts/run_genie_motif_pipeline.sh",
            f"{savedir}/rank{rank}/pmpnn",
        ]
        cvd = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if cvd:
            dev = cvd.split(',')[torch.cuda.current_device()]
        else:
            dev = torch.cuda.current_device()
        
        subprocess.run(cmd, env=os.environ | {
            'CUDA_VISIBLE_DEVICES': str(dev)
        })  

        pmpnn_df = pd.read_csv(
            f"{savedir}/rank{rank}/pmpnn/info.csv", index_col="domain"
        )
        pmpnn_df["designable"] = pmpnn_df["scRMSD"] < 2
        pmpnn_df["success"] = (pmpnn_df["scRMSD"] < 2) & (pmpnn_df.loc[name].motif_ca_rmsd < 1)
        if logger is not None:
            for col in pmpnn_df.columns:
                for val in pmpnn_df[col].tolist():
                    logger.log(f"{self.cfg.name}/pmpnn_{col}", val)
        for i in idx:
            name = self[i]['name']
            df[name]["pmpnn_scrmsd"] = pmpnn_df.loc[name].scRMSD
            df[name]["pmpnn_scTM"] = pmpnn_df.loc[name].scTM
            df[name]["pmpnn_pLDDT"] = pmpnn_df.loc[name].pLDDT
            df[name]["pmpnn_mRMSD"] = pmpnn_df.loc[name].motif_ca_rmsd
            
    def run_batch(
        self,
        model,
        batch: dict,
        noisy_batch: dict,
        savedir=".", 
        device=None,
        logger=None
    ):
        
        schedules = {
            'structure': self.struct_sched_fn,
            'sequence': self.seq_sched_fn,
        }

        mask = batch['seq_noise'].bool()
        sampler = OpenProtSampler(schedules, steppers=[
            EDMDiffusionStepper(self.cfg.struct),
            SequenceUnmaskingStepper(self.cfg.seq, mask=mask)
        ])

        sample, extra = sampler.sample(model, noisy_batch, self.cfg.steps)

        pred_traj = torch.stack(extra['preds'])
        samp_traj = torch.stack(extra['traj'])

        batch['struct'] = sample['struct']
        batch['aatype'] = sample['aatype']
        
        datas = batch.unbatch()
        
        for i, data in enumerate(datas):

            data.update_seqres()
            name = data["name"]

            mask = data['ref_conf_mask'].bool()
            
            ref_motif = data['ref_conf'][mask]
            samp_motif = data['struct'][mask]

            rmsd = compute_rmsd(ref_motif, samp_motif)
            if logger is not None:
                logger.log(f"{self.cfg.name}/samp_mRMSD", rmsd)

            prot = make_ca_prot(
                data['struct'].cpu().numpy(),
                data["aatype"].cpu().numpy(),
                data["struct_mask"].cpu().numpy(),
            )
            name = data["name"]
            with open(f"{savedir}/{name}.pdb", "w") as f:
                f.write(protein.to_pdb(prot))

            seq = aatype_to_seqres(data["aatype"])
            with open(f"{savedir}/{name}.fasta", "w") as f:
                f.write(f">{name}\n")  # FASTA format header
                f.write(seq + "\n")

            
            prot = make_ca_prot(
                data['ref_conf'].cpu().numpy(),
                data["aatype"].cpu().numpy(),
                data["ref_conf_mask"].cpu().numpy(),
            )
            with open(f"{savedir}/{name}_motif.pdb", "w") as f:
                f.write(protein.to_pdb(prot))

            


        
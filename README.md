# OpenProt

## Dependencies
* python==3.12
* biopython
* numpy==1.26.4
* `pip install torch==2.2.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html` on the csail machines
* `pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124` on vista
* pytorch-lightning==2.4.0
* dm-tree
* omegaconf
* fair-esm
* ruff (optional, see below)
* black
* neptune
* pytest
* foldcomp (must install with `pip install git+https://github.com/steineggerlab/foldcomp@f868b95` repo, see https://github.com/steineggerlab/foldcomp/issues/52)

Install the dependencies with `mamba env create -f environment.yml` and activate the environment with `conda activate openprot`. (Note: this is Sam's workflow, Bowen recommends installing things manually)

### Setting up datasets
Only need to do this once
```
aws s3 sync --no-sign-request s3://pdbsnapshots/20240101/pub/pdb/data/structures/divided/mmCIF pdb_mmcif
python -m scripts.unpack_mmcif --mmcif_dir ../data/pdb_mmcif --outdir ../data/pdb_npz --outcsv ../data/pdb_chains.csv --num_workers 100
python -c "import foldcomp; foldcomp.setup('afdb_swissprot_v4')" # foldcomp server might be down though
curl -O https://ftp.ebi.ac.uk/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz # then gunzip it
python -m scripts.process_uniref --fasta [PATH] --out [PATH]
python -m scripts.cluster_chains --mmseqs_path [PATH] --chains /scratch/projects/cgai/openprot/data/pdb_npz/pdb_chains.csv --out pdb_clusters
```

## Contributing

For sprint targets, make a new branch with `git branch [NAME]` and switch to it with `git checkout [NAME]`. Run `git push -u origin [NAME]` when pushing for the first time.

Prior to your first contribution, `cd` into the repository and run `pre-commit install`. This will install a pre-commit hook that will run formatting checks (`ruff`), as well as eventual tests for breaking changes, on all files in the repository. (Note: this is also Sam's workflow, Bowen thinks this is optional)

To run tests, add a module to `tests/` and run `pytest tests -s`. Or, to run a specific module, run `pytest tests/{module} -s`. (The `-s` flag allows normal stdout to be printed)

All PRs should pass the pre-commit checks before they are merged into `master`.

The repository skeleton has been set up so that `python train.py` will always run. The `config.yaml` file controls all aspects of the training script. To contribute, you will likely need to add a module to openprot/data or openprot/tracks and register it in `config.yaml`. Unless you are directly working on the training harness or workflow, changes to the template files should be avoided if possible to avoid merge conflicts.

Because wandb is extremely laggy, we will use [neptune.ai](https://neptune.ai/). Please make a new account and let Bowen know your email address so he can invite you to the openprot project.

Numerous other settings are controlled in `config.yaml` which have been commented there.

## Workflow Abstraction

We have five abstraction layers:
* The **dataset** layer corresponds to individual data sources, i.e., AFDB, PDB, etc
* The **task** layer corresponds to specific training objectives, i.e., inverse folding, structure prediction. Tasks and datasets are collectively grouped under and managed by the [`OpenProtDataManager`](openprot/data/manager.py) class.
* The **track** layer corresponds to each data modality, and is responsible for noising, embedding (into the model), predicting (from the model), and supervising data modalities. Tracks are grouped under the [`OpenProtTrackManager`](openprot/tracks/manager.py) class.
* The **model** is the model itself.
* The **evaluation** layer manages the data and inference procedures for each validation workflow. Evals are grouped under and managed by the [`OpenProtEvalManager`](openprot/evals/manager.py) class.


To understand the repo, let's walk through the workflow abstraction by following the lifecycle of a training example:

* Throughout the workflow, training examples are of class [`OpenProtData`](openprot/data/data.py), which subclasses `dict`. `OpenProtData` objects know how to `crop`, `pad`, `batch`, `unbatch`, and `to(device)` themselves. To do this, we impose some conventions on their attributes.
    * Regular attributes, which are (numpy or torch) arrays with (unbatched) shape `(L, ...)`
    * Pairwise attributes, which have shape `(L, L, ...)`. Their names must always END in `_`.
    * The special `seqres` attribute, which is the sole non-array attribute that will be cropped or padded. (This is because `seqres` is the only non-array raw data type.)
    * Other attributes, which is any attribute that is NOT an array (such as `name`).
      
* All training examples originate from some [`OpenProtDataset`](openprot/data/data.py) which map from an integer index to a training example via `__getitem__`. This map should be deterministic and is not shuffled. The dataset defines its own `setup` based on its fields specified in `config.yaml`.
    * To ensure uniformity in the resulting `OpenProtData` objects, datasets should call `self.make_data` instead of instantiating directly. This will fill in default values for any unspecified features with shapes specified in `config.yaml`.
    * Every track will have a `_mask` and `_noise` feature which indicate if the track is present and the noise level. `OpenProtDataset` objects should set the `_mask` for modalities that are present.

> &#x26A0; Be sure to define/use features in a way where default 0 makes sense.

> &#x26A0; All of our masks are 1 if the data is PRESENT because we use masks extensively to reduce loss tensors!

* [`Task`](openprot/tasks/task.py) objects are responsible for keeping a mix of datasets and sampling from them with probabilities specified in `config.yaml`. Upon fetching the data, they execute task-specific preprocessing in `prep_data(data, crop=None)`. The preprocessing should populate the appropriate `_noise` features so that we determine what we are going to mask, noise, and supervise. The task needs to ensure the maximum length of the returned data at most `crop`, probably by calling `data.crop(crop)`.

> &#x26A0; We ask task classes to crop the data themselves because for complex tasks, naive cropping might not work.

> &#x26A0; Any features touched by `OpenProtDataset` or `OpenProtTask` objects before this point must be logged to `config.yaml`; otherwise there will be issues when batching! Also, these features must be NUMPY ARRAYS, not TORCH TENSORS. (`OpenProtData` doesn't yet know how to crop or pad tensors).

* The [`OpenProtDataManager`](openprot/data/manager.py) samples from a mix of tasks with probabilities specified in `config.yaml`.  The `OpenProtDataManager` then calls on the `OpenProtTrackManager` to `tokenize` the data, which introduces additional numpy arrays to the data for each track. The data is finally padded to a fixed length.

* PyTorch Lightning now automatically batches our data and moves it the GPU. The rest of the training step happens in [`OpenProtWrapper`](openprot/model/wrapper.py).

* The batch is noised by calling `corrupt` on each track. In doing so, the track places tensors that will be embedded into the model in `noisy_batch`, and tensors needed to compute the loss in `target`. This must include a `_supervise` mask.

* The tracks `embed` the data from `noisy_batch` into an embedding tensor. Tracks should define a `null` and `mask` token for residues that are not present or at maximum noise, respectively. Each track must define `add_modules` to add the necessary embedding (and prediction) layers to the model.

> &#x26A0; Note that every track sees every data point at each step, regardless of whether the track is present. This is why the `_mask` is crucial!

* The batch passes through the model, masked by `pad_mask`. The tracks make a prediction and place them in the `readout`.

* The tasks then run `compute_loss` using the `readout` and `target`, which are weighted to obtain the final loss. The loss should be computed per-token, and logged with the `loss_mask`, so that we monitor a per-supervised-token `track/loss`. We also log `track/toks` and `track/sup_toks` as the sum of the `_mask` and `_supervise` masks. 

> &#x26A0; The return value of each track's `compute_loss` must be a **scalar mean** over all **non-pad tokens in the batch**. This is so that the loss is weighted by the fraction of tokens that are supervised in that track (to avoid spiky token weights if some tracks are rarely supervised). Thus, the weight in `config.yaml` should be regarded as a per-supervised-token weight.

> &#x26A0; We may add more systematic logging to have per-dataset, per-task metrics.

The **validation** life cycle is similar, but a bit simpler:

* Each [`OpenProtEval`](openprot/evals/eval.py) class is a subclass of `OpenProtDataset`, i.e., it is also a map from scalar index to `OpenProtData` object. The [`OpenProtEvalManager`](openprot/eval/manager.py) maintains a list of eval tasks and iterates through them sequentially in a parallizeable fashion (i.e., across ranks). The eval manager will `tokenize` the data object returned from the eval and label it with the source eval, but will not crop, pad, or otherwise preprocess it.

> &#x26A0; We intentionally keep the validation batch size 1 so that batch corresponds to one task.

* The validation example surfaces in the `OpenProtWrapper`'s validation step. There, the batch is handed to the `run_batch` method of the source eval. The source eval executes arbitrary sampling and inference logic, logs arbitrary metrics, and saves arbitrary artifacts.


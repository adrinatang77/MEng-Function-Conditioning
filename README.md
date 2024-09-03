# OpenProt

## Dependencies
* python==3.12
* biopython
* numpy==1.26.4
* pytorch==2.2.0 (`pip install torch==2.2.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html` on the csail machines)
* pytorch-lightning==2.4.0
* dm-tree
* omegaconf
* ruff (optional, see below)
* neptune
* pytest
* foldcomp (must install with `pip install git+https://github.com/steineggerlab/foldcomp@f868b95` repo, see https://github.com/steineggerlab/foldcomp/issues/52)

Install the dependencies with `mamba env create -f environment.yml` and activate the environment with `conda activate openprot`. (Note: this is Sam's workflow, Bowen recommends installing things manually)

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

* All training examples originate from some [`OpenProtDataset`](openprot/data/data.py) which map from an integer index to a training example via `__getitem__`. This map should be deterministic and is not shuffled. The dataset defines its own `setup` based on its fields specified in `config.yaml`.
* The training examples are of class [`OpenProtData`](openprot/data/data.py), whose constructor requires `seqres` (since all examples have it) but will fill in default values for all other unspecified features. The features will be zero-initialized with per-token shapes as specified in `config.yaml`. `OpenProtDataset` objects should override the approriate defaults when returning the data.

> &#x26A0; Be sure to define/use features in a way where default 0 makes sense.

> &#x26A0; All features must be float32 arrays (except for `seqres`) and per-token, but we may revisit this.

* Canonically, every track will have a `_mask` and `_noise` feature which indicate if the track is present and the noise level. `OpenProtDataset` objects should set the `_mask`.

> &#x26A0; All of our masks are 1 if the data is PRESENT because we use masks extensively to reduce loss tensors!

* [`Task`](openprot/tasks/task.py) objects are responsible for keeping a mix of datasets and sampling from them with probabilities specified in `config.yaml`. Upon fetching the data, they crop the data before executing task-specific preprocessing in `prep_data`. The preprocessing should populate the appropriate `_noise` features so that we determine what we are going to mask, noise, and supervise.

> &#x26A0; We crop here because the preprocessing may be crop-dependent. In the future, there may be more general data transforms as part of the task preprocessing.

* The [`OpenProtDataManager`](openprot/data/manager.py) samples from a mix of tasks with probabilities specified in `config.yaml`. The data is padded it to a fixed length which introduces a special `pad_mask` feature. The `OpenProtDataManager` then calls on the `OpenProtTrackManager` to `tokenize` the data, which introduces additional numpy arrays to the data.

* PyTorch Lightning now automatically batches our data and moves it the GPU. The rest of the training step happens in [`OpenProtWrapper`](openprot/model/wrapper.py).

> &#x26A0; We may consider moving the tokenization step here.

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


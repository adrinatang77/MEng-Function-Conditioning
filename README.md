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

## Key Abstractions

We are training a multimodal model with many datasets, many pretraining tasks, and many modalities/tasks. We want the ability to easily swap in and out any of these, make adjustments, and resume and reproduce runs. We want to do all this with minimal friction and technical debt. Thus, the abstractions can get complicated, and are worth going over in some detail here:

**At a high level**, we have a three key abstraction layers:
* The **dataset** layer corresponds to individual data sources, i.e., AFDB, PDB, etc
* The **task** layer corresponds to specific training objectives, i.e., inverse folding, structure prediction
* The **track** layer corresponds to each data modality, and is responsible for noising, embedding (into the model), predicting (from the model), and supervising data modalities.

Because each task can source from many datasets, and determines which tracks are noised/supervised, it is helpful to think of **tasks** as the central abstraction.

Each training batch will sample a mix of tasks and therefore datasets, and will be supervised on a mix of tracks. We need to serve these batches in a way that accounts for multiple ranks and multiple dataloaders per rank, but is reproducible and stateful (so that we can resume runs where we started, even if the configuration changes).

**Datasets** are objects that map from an integer index to a training example via `__getitem__`. This map should be DETERMINISTIC and should NOT be shuffled. (They do not have to be 100% deterministic---each entry may correspond to a sequence cluster for example---but should be semantically the same for each integer index).

**Tasks** are ITERABLE objects that yield a stream of training examples, sampled from their assigned datasets and labeled with information that will dictate the downstream noising and supervision masks. To do this in parallel, instantiations of task objects must know their global rank, which is obtained by passing in `rank` and `world_size` at construction and fetching the dataloader `worker_info` inside of `__iter__`. To do this statefully, tasks will maintain a shuffled **iteration order** for each dataset controlled by a deterministic `seed` and for which the iteration starts at the specific `start` for each dataset and advances according to the total world size. The dataset to take from upon each `yield` is sampled randomly.

> To be continued...

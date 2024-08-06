# OpenProt

## Dependencies

* pytorch
* pytorch-lightning
* omegaconf
* black
* neptune

## Contributing

For sprint targets, make a new branch with `git branch [NAME]` and switch to it with `git checkout [NAME]`. Run `git push -u origin [NAME]` when pushing for the first time.

We will ask that all PRs pass `black .` before they will be merged into `master`.

The repository skeleton has been set up so that `python train.py` will always run. The `config.yaml` file controls all aspects of the training script. To contribute, you will likely need to add a module to openprot/data or openprot/tracks and register it in `config.yaml`. Unless you are directly working on the training harness or workflow, changes to the template files should be avoided if possible to avoid merge conflicts.

Because wandb is extremely laggy, we will use neptune.ai. Please make a new account and let Bowen know your email address so he can invite you to the openprot project.

Numerous other settings are controlled in `config.yaml` which have been commented there.
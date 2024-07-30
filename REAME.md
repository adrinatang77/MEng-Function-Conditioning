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

## Testing

The repository skeleton has been set up so that `python train.py` will always run.

## Logging

Because wandb is extremely laggy, we will use neptune.ai.
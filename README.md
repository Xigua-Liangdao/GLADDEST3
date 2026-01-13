# GLADDEST

This repo contains code to train and evaluate the GLADDEST model with Hydra configs.

The project is set up to be anonymous and portable:
- Checkpoints can be provided via an environment variable (no hardcoded personal paths).
- Dataset root can be configured via environment variable or resolved from the project root.


## Data setup

In configs, `dataset.datadir` may be relative (recommended). If it's relative, the code resolves it as:
1) `${DATASET_ROOT}` environment variable if set; otherwise
2) Hydra's original working directory (the project root when you launched the run).

To point to your datasets explicitly:

```bash
export DATASET_ROOT=/abs/path/to/your/datasets
```

Or keep `dataset.datadir` relative to the project root.

## Train

Hydra is used for configuration. Example (bzr):

```bash
PYTHONPATH=. python src/main.py +experiment=bzr dataset=bzr
```

Notes:
- `+experiment=<name>` and `dataset=<name>` select the config group entries under `configs/`.
- Set `general.gpus=0` to run on CPU; otherwise the trainer will use available GPUs as configured.

### Example: Tox21_MMP (train)

Ensure `general.test_only: null` (default) in `configs/general/general_default.yaml`, then run:

```bash
PYTHONPATH=. python src/main.py +experiment=Tox21_MMP dataset=Tox21_MMP
```

### Example: Tox21_MMP (test)

Set `general.test_only` to the provided checkpoint path, then run the transition-based evaluation:

Option A (edit config):

1) Edit `configs/general/general_default.yaml`:

```yaml
general:
	# ... other fields
	test_only: checkpoints/graph-tf-model/epoch=999.ckpt  # <- put the provided ckpt path here
```

2) Run:

```bash
PYTHONPATH=. python src/transition_method.py +experiment=Tox21_MMP dataset=Tox21_MMP
```

Option B (without editing config):

```bash
PYTHONPATH=. python src/transition_method.py +experiment=Tox21_MMP dataset=Tox21_MMP \
	general.test_only=checkpoints/graph-tf-model/epoch=999.ckpt
```

## Provided checkpoint (example)

We provide a checkpoint for `Tox21_MMP` testing. Use one of the test-only methods above to point `general.test_only` to the file (either via `CKPT_PATH` or a relative path under `checkpoints/`).

## Optional: Transition-method evaluation

There is an evaluation script that computes anomaly scores over a time-interpolation process:

```bash
PYTHONPATH=. python src/transition_method.py +experiment=Tox21_MMP dataset=Tox21_MMP \
	general.test_only=checkpoints/graph-tf-model/epoch=999.ckpt
```

This script also respects `DATASET_ROOT` and `CKPT_PATH` the same way as above.

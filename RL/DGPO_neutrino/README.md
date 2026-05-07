# DGPO Neutrino RL

DGPO fine-tuning for EveNet neutrino diffusion.

This directory lives at `RL/DGPO_neutrino/` inside the `DGPO-NeurIPS-Addon`
repo. `config.yaml` references TT2L YAML defaults at `../../configs/tt2l/`.

## Runtime (default)

Use **`docker.io/avencast1994/evenet:1.5`** (canonical reference: `DGPO-NeurIPS-Addon/runtime-image.txt`). Mount your `DGPO-NeurIPS-Addon/` checkout — including the nested `EveNet-Full/` companion clone (see top-level README "Install") — into the container and run `dgpo_trainer.py` as below. **No extra `pip install` is needed inside the image.**

## Required Inputs

- **Repo layout**: `DGPO-NeurIPS-Addon/` (this add-on, providing `RL.*`, `shared.*`, `event_selection.*`) with a nested `DGPO-NeurIPS-Addon/EveNet-Full/` checkout (providing `evenet.*` core and `preprocessing.*`; the `evenet/` directory inside it is itself a git submodule of `EveNet-HEP/Core`). The trainer adds both `DGPO-NeurIPS-Addon/` and `DGPO-NeurIPS-Addon/EveNet-Full/` to `sys.path` and to the Ray worker `PYTHONPATH`.
- **TT2L Parquet** train (and optionally val) directories — already materialized `.parquet` files plus `shape_metadata.json` next to them (you do not need to run custom preprocess scripts here if those artifacts already exist).
- **`normalization.pt`** — path in `options.Dataset.normalization_file` (must match the Parquet you use).
- **Checkpoint** — set `options.Training.model_checkpoint_load_path` to your supervised / Phase‑1 `.ckpt` (weights load).
- **Writable** checkpoint and logger dirs from `config.yaml`.

## Quick Checks

```bash
python -m py_compile RL/DGPO_neutrino/dgpo_trainer.py
```

Smoke run (after editing paths in `config.yaml`):

```bash
python RL/DGPO_neutrino/dgpo_trainer.py RL/DGPO_neutrino/config.yaml --max-steps 2 --no-wandb
```

## Training

Edit `config.yaml` for data, normalization, checkpoint, and output paths.

For a small smoke run:

```bash
python RL/DGPO_neutrino/dgpo_trainer.py RL/DGPO_neutrino/config.yaml --max-steps 50 --no-wandb
```

The same Python entry point works for single-machine and multi-node Ray runs.
Use `platform.number_of_workers`, `platform.resources_per_worker`, and
`platform.use_gpu` to match the available hardware. If a Ray cluster is already
running, set `RAY_ADDRESS`; otherwise the trainer falls back to local Ray.

# Install DGPO Add-on Into EveNet

The DGPO add-on consumes EveNet's model, config loader, Ray data helpers, and
diffusion sampler from a **nested companion checkout** at
`DGPO-NeurIPS-Addon/EveNet-Full/` (see the top-level `README.md` "Install"
section for the exact `git clone --recurse-submodules` recipe). **Default
DGPO runtime:** `docker.io/avencast1994/evenet:1.5` (canonical copy in
`DGPO-NeurIPS-Addon/runtime-image.txt`). Everything below is meant to run
from inside that image (or inside an equivalent fallback environment).

## 1. Clone the add-on (with nested EveNet) and pull the image

```bash
git clone <anonymous-dgpo-addon-url> DGPO-NeurIPS-Addon
cd DGPO-NeurIPS-Addon

# Nested EveNet checkout (provides evenet/ + preprocessing/). The trailing
# --recurse-submodules also pulls evenet/ = EveNet-HEP/Core inside it.
git clone --recurse-submodules https://github.com/EveNet-HEP/EveNet-Full.git

# Pull once. Use the engine your site supports.
docker  pull docker.io/avencast1994/evenet:1.5
# or:
shifterimg pull docker.io/avencast1994/evenet:1.5
# or:
apptainer pull evenet_1.5.sif docker://avencast1994/evenet:1.5
```

See `docs/container.md` for full launch commands per engine.

## 2. Enter the runtime

The public image already provides every Python dependency. No extra
`pip install` is required, even though `evenet` lives at
`EveNet-Full/evenet/` — the trainer and unfolder bootstrap `sys.path` to
that nested location automatically.

```bash
docker run --rm -it --gpus all \
  -v "$PWD:/workspace/DGPO-NeurIPS-Addon" \
  -w /workspace/DGPO-NeurIPS-Addon \
  docker.io/avencast1994/evenet:1.5 bash
```

If the public image is not available, install from the add-on requirements
file (pulls **the same packages as `EveNet-Full/evenet/setup.py`** via
editable `evenet`, plus Ray):

```bash
pip install -r requirements-dgpo.txt   # run from DGPO-NeurIPS-Addon/
```

## 3. Run checks

Inside the runtime, from the `DGPO-NeurIPS-Addon/` root:

```bash
python -m py_compile RL/DGPO_neutrino/dgpo_trainer.py
```

## 4. Run DGPO

Edit `DGPO-NeurIPS-Addon/RL/DGPO_neutrino/config.yaml`:

- `platform.data_parquet_dir`
- `platform.data_parquet_val_dir`
- `options.Dataset.normalization_file`
- `options.Training.model_checkpoint_load_path`
- `options.Training.model_checkpoint_save_path`

Then:

```bash
python RL/DGPO_neutrino/dgpo_trainer.py \
    RL/DGPO_neutrino/config.yaml --max-steps 50 --no-wandb
```

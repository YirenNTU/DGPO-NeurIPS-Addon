# Container

**DGPO default / recommended image:** `docker.io/avencast1994/evenet:1.5` (also in `../runtime-image.txt`).

The recommended runtime for DGPO fine-tuning is the public EveNet image:

```text
docker.io/avencast1994/evenet:1.5
```

It already contains EveNet's Python stack (PyTorch + CUDA, Ray, Lightning,
NumPy, PyArrow, Matplotlib, W&B, Vector, Awkward, h5py, etc.). The DGPO add-on
relies entirely on this stack and adds no extra Python packages.

The image does **not** include PyROOT. Run RooUnfold-based SVD unfolding panels
in a separate ROOT environment; see `docs/roounfold.md`.

## Docker / Podman

```bash
docker pull docker.io/avencast1994/evenet:1.5

# Mount the DGPO-NeurIPS-Addon checkout that already has the nested
# EveNet-Full/ companion clone populated (see top-level README "Install").
docker run --rm -it --gpus all \
  -v "$PWD/DGPO-NeurIPS-Addon:/workspace/DGPO-NeurIPS-Addon" \
  -w /workspace/DGPO-NeurIPS-Addon \
  docker.io/avencast1994/evenet:1.5 \
  bash
```

Inside the container (no extra `pip install` needed):

```bash
python -m py_compile RL/DGPO_neutrino/dgpo_trainer.py
python RL/DGPO_neutrino/dgpo_trainer.py RL/DGPO_neutrino/config.yaml --no-wandb
```

## Shifter (HPC)

```bash
shifterimg pull docker.io/avencast1994/evenet:1.5
shifter --image=docker.io/avencast1994/evenet:1.5 bash -lc \
  'cd /path/to/DGPO-NeurIPS-Addon && python RL/DGPO_neutrino/test_rewards.py'
```

## Apptainer / Singularity

```bash
apptainer pull evenet_1.5.sif docker://avencast1994/evenet:1.5
apptainer exec --nv evenet_1.5.sif bash -lc \
  'cd /path/to/DGPO-NeurIPS-Addon && python RL/DGPO_neutrino/test_rewards.py'
```

## Multi-node Ray

For distributed runs, start the Ray head and workers according to the site
scheduler, set `RAY_ADDRESS` on the head if needed, and adjust
`platform.number_of_workers` and `resources_per_worker` in
`RL/DGPO_neutrino/config.yaml`.

## Fallback: Local virtualenv

If you cannot use the image, install from `DGPO-NeurIPS-Addon/requirements-dgpo.txt`:
editable **evenet** (resolved as `EveNet-Full/evenet/`, same `install_requires` as
upstream `evenet/setup.py`) plus **Ray** for DGPO. Run from the
`DGPO-NeurIPS-Addon/` repo root (which now also contains the nested
`EveNet-Full/` companion clone — see top-level README "Install"):

```bash
cd DGPO-NeurIPS-Addon
python -m pip install -r requirements-dgpo.txt
```

This path is unsupported for full multi-node runs; the public image is the
reference environment for reviewers.

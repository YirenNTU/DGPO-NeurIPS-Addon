# DGPO Neutrino Add-on

NeurIPS add-on for DGPO fine-tuning and neutrino unfolding. **Not** a
standalone repo — DGPO depends on the EveNet core network code (cloned as a
git submodule, see [Step 1](#1-clone-with-submodules))
and, optionally for SVD unfolding, RooUnfold (cloned next to it).

**Default DGPO runtime (container):** `docker.io/avencast1994/evenet:1.5` —
same single-line value as `runtime-image.txt`. DGPO adds **no** Python deps on
top of this image; everything in `requirements-dgpo.txt` is identical to
`EveNet-Full/requirements.txt`, exposed only as a bare-venv fallback.

## TL;DR

```bash
# Single recursive clone — pulls EveNet-Full (with its evenet/ submodule)
# and RooUnfold (optional, SVD only) at the exact pinned commits.
git config --global url."https://github.com/".insteadOf "git@github.com:"
git clone --recurse-submodules https://github.com/<you>/DGPO-NeurIPS-Addon.git
cd DGPO-NeurIPS-Addon

# DGPO training (inside the official EveNet image)
docker pull docker.io/avencast1994/evenet:1.5
docker run --rm -it --gpus all \
    -v "$PWD":/workspace/DGPO-NeurIPS-Addon \
    -w /workspace/DGPO-NeurIPS-Addon \
    docker.io/avencast1994/evenet:1.5 \
    python RL/DGPO_neutrino/dgpo_trainer.py \
        RL/DGPO_neutrino/config.yaml --max-steps 2 --no-wandb

# Prediction → .pt files (template at prediction/predict_TT2L.yaml, see Step 4)
python EveNet-Full/scripts/predict.py prediction/predict_TT2L.yaml

# (Optional) SVD unfolding overlays — needs ROOT + RooUnfold (Step 5)
python RL/Unfolding/run_neurips_neutrino_unfolding.py \
    --model nominal=/path/to/nominal.pt \
    --output_dir outputs/unfolding_neutrino
```

Tested end-to-end: NERSC Perlmutter via Shifter (training), macOS arm64 +
conda-forge ROOT (RooUnfold build + unfolding).

---

## 1. Clone (with submodules)

This repo is *not* self-contained. Entry-point scripts (`RL/DGPO_neutrino/dgpo_trainer.py`,
`RL/Unfolding/run_neurips_neutrino_unfolding.py`) bootstrap `sys.path` to **two**
roots:

- `DGPO-NeurIPS-Addon/` (this repo) — provides `RL.*`, `shared.*`, `event_selection.*`.
- `DGPO-NeurIPS-Addon/EveNet-Full/` (git submodule of this repo, which itself
  contains a nested submodule `evenet/` = `EveNet-HEP/Core`) — provides
  `evenet.*` and `preprocessing.*`.

`RooUnfold/` is also a (top-level) git submodule, used only for SVD-based
unfolding; the CMake build needs ROOT and is deferred to
[Step 5](#5-optional--roounfold-for-svd-unfolding-only). You can skip building
it entirely if you do not need SVD panels.

A single recursive clone pins all three repos to the exact commits we tested:

```bash
# Reviewers without SSH keys to EveNet-HEP/Core: tell git to use HTTPS for
# any github.com SSH URLs (the nested evenet submodule's upstream pin).
git config --global url."https://github.com/".insteadOf "git@github.com:"

git clone --recurse-submodules https://github.com/<you>/DGPO-NeurIPS-Addon.git
cd DGPO-NeurIPS-Addon

# If you forgot --recurse-submodules:
#   git submodule update --init --recursive
```

Verify the layout:

```bash
test -f EveNet-Full/evenet/network/evenet_model.py    && echo evenet_OK    || echo evenet_MISSING
test -f EveNet-Full/preprocessing/preprocess.py       && echo preproc_OK   || echo preproc_MISSING
test -f RooUnfold/CMakeLists.txt                      && echo roounfold_OK || echo roounfold_MISSING
```

Final on-disk layout (the `sys.path` bootstrap depends on these names):

```text
DGPO-NeurIPS-Addon/
├── RL/                         # DGPO trainer + unfolding (this repo)
│   ├── DGPO_neutrino/          # dgpo_trainer.py, rewards.py, config.yaml, ...
│   └── Unfolding/              # run_neurips_neutrino_unfolding.py
├── configs/  event_selection/  shared/  prediction/  docs/
├── requirements-dgpo.txt       # pin-equivalent to EveNet-Full/requirements.txt
├── runtime-image.txt           # docker.io/avencast1994/evenet:1.5
├── EveNet-Full/                # ← submodule (EveNet-HEP/EveNet-Full)
│   ├── evenet/                 # ← nested submodule (EveNet-HEP/Core)
│   ├── preprocessing/
│   └── requirements.txt        # source of truth for Python deps
└── RooUnfold/                  # ← submodule (gitlab.cern.ch/RooUnfold/RooUnfold)
    └── build/libRooUnfold.{so,dylib}    # produced in step 5 (not tracked)
```

## 2. Pick a runtime — image (recommended) or bare venv

DGPO is intended to run inside the **public EveNet image** (Docker / Podman /
Apptainer / Singularity / Shifter — full per-engine commands in
`docs/container.md`):

```text
docker.io/avencast1994/evenet:1.5
```

Mount the host `DGPO-NeurIPS-Addon/` directory (with the populated
`EveNet-Full/`) into the container and run the entry-point scripts directly —
**no `pip install` is needed inside the container**.

If you cannot use the image, install the same pins in a venv/conda env:

```bash
python -m pip install -r requirements-dgpo.txt   # from the DGPO-NeurIPS-Addon/ root
```

`requirements-dgpo.txt` is a verbatim copy of `EveNet-Full/requirements.txt`,
so the bare-venv path matches the image package set.

### Supported platforms

| Platform | Status | Notes |
|---|---|---|
| Linux + Docker / Podman / Apptainer / Singularity / Shifter | **Recommended** | Primary path. NERSC Perlmutter (Shifter) confirmed. |
| macOS (arm64/x86_64) | Local setup + RooUnfold build verified | Use conda-forge `python=3.12 root cmake compilers`. GPU training not supported. |
| Windows | Not tested | Use WSL2 + Docker, or run on Linux/HPC. |

### Verified environments

- **NERSC Perlmutter** — Shifter image `docker.io/avencast1994/evenet:1.5` on
  4 nodes × 4× NVIDIA A100; full DGPO training + prediction loop end-to-end.
- **macOS arm64 (Apple Silicon)** — conda-forge `python=3.12 root=6.38 cmake
  compilers`; RooUnfold CMake build + SVD unfolding overlays end-to-end on a
  saved DGPO prediction `.pt`.

### HPC clusters

The container is the portability contract: any cluster that can run an OCI
image (Apptainer / Singularity / Shifter / Podman / enroot) can run DGPO with
the same commands as Docker, just substituting the engine. **Site-specific job
schedulers (Slurm, PBS, LSF, …) are out of scope** — the container provides
the reproducible Python/CUDA stack regardless of how you submit it.

## 3. Run DGPO training inside the image (Docker, step-by-step)

These steps run anywhere Docker / Podman is available and the host has NVIDIA
GPUs visible to the engine (`nvidia-container-toolkit` for Docker; Podman
needs `--device nvidia.com/gpu=all` or equivalent). For Apptainer / Singularity
/ Shifter and HPC-specific notes, see `docs/container.md`.

**3a. Edit `RL/DGPO_neutrino/config.yaml`** so the YAML paths are valid **inside
the container** (i.e. they match the mount targets you will use in 3c, *not*
the host paths):

| YAML field | Example value (matches the mounts in 3c) |
|---|---|
| `platform.data_parquet_dir`, `platform.data_parquet_val_dir` | `/data/tt2l_parquet/train`, `/data/tt2l_parquet/val` |
| `options.Dataset.normalization_file` | `/data/tt2l_parquet/train/normalization.pt` |
| `options.Training.model_checkpoint_load_path` | `/checkpoints/pretrain.ckpt` |
| `options.Training.model_checkpoint_save_path`, `logger.save_dir` | `/checkpoints/dgpo_run/`, `/checkpoints/dgpo_run/logs/` |

**3b. Pull the image once** (skip if already pulled):

```bash
docker pull docker.io/avencast1994/evenet:1.5
```

**3c. Launch the container.** Three things to mount:

1. The `DGPO-NeurIPS-Addon/` checkout itself — the populated `EveNet-Full/`
   sits *inside* it, so this single mount provides both `RL.*` / `shared.*`
   and `evenet.*` / `preprocessing.*` to the `sys.path` bootstrap.
2. Your **TT2L parquet + normalization.pt** directory (read-only is fine).
3. Your **checkpoints** directory (read-write — the trainer writes here).

```bash
cd /path/to/DGPO-NeurIPS-Addon         # host repo root

docker run --rm -it --gpus all \
    -v "$PWD":/workspace/DGPO-NeurIPS-Addon \   # repo (with EveNet-Full/ inside)
    -v /host/abs/path/to/tt2l_parquet:/data \   # → matches /data/... in config.yaml
    -v /host/abs/path/to/checkpoints:/checkpoints \  # → matches /checkpoints/... in config.yaml
    -w /workspace/DGPO-NeurIPS-Addon \
    docker.io/avencast1994/evenet:1.5 \
    bash
```

> Note: the YAML paths in 3a (`/data/...`, `/checkpoints/...`) refer to the
> *container-side* mount targets above — not your host filesystem. Change the
> host-side paths to wherever your data and checkpoints actually live.
> `--gpus all` requires the NVIDIA Container Toolkit (Docker on Linux) or
> `--device nvidia.com/gpu=all` (Podman); HPC equivalents are in
> `docs/container.md`.

**3d. Inside the container**, no extra `pip install` is needed. Sanity-check
imports first, then run the trainer (smoke run with `--max-steps 2 --no-wandb`
before a long run):

```bash
# (inside the container, cwd = /workspace/DGPO-NeurIPS-Addon)
python -m py_compile RL/DGPO_neutrino/dgpo_trainer.py
python -c 'import evenet, preprocessing, RL.DGPO_neutrino.dgpo_trainer; print("OK")'

# smoke test
python RL/DGPO_neutrino/dgpo_trainer.py \
    RL/DGPO_neutrino/config.yaml --max-steps 2 --no-wandb

# real run
python RL/DGPO_neutrino/dgpo_trainer.py \
    RL/DGPO_neutrino/config.yaml --max-steps 50 --no-wandb
```

**3e. (Optional) Bare-venv equivalent** if Docker is not an option (uses the
exact same package pins as the image; same commands as 3d, just without
`docker run`):

```bash
cd /path/to/DGPO-NeurIPS-Addon
python -m pip install -r requirements-dgpo.txt
python RL/DGPO_neutrino/dgpo_trainer.py \
    RL/DGPO_neutrino/config.yaml --max-steps 2 --no-wandb
```

## 4. Run prediction → produce `.pt` files for unfolding

`prediction/predict_TT2L.yaml` is a template config for EveNet's prediction
script. It loads a DGPO-fine-tuned (or pretrained) checkpoint, runs inference
over a TT2L parquet directory, and writes a single `.pt` containing point-cloud
tensors plus the EXTRA branches (reconstructed / truth / LHE) that the
unfolding script consumes.

**4a. Edit the template** — every `/path/to/...` placeholder in
`prediction/predict_TT2L.yaml`:

- `platform.data_parquet_dir` — input parquet directory.
- `options.prediction.output_dir` and `options.prediction.filename` — where
  the `.pt` is written (final path = `output_dir/filename`).
- `options.Training.model_checkpoint_load_path` — checkpoint `.ckpt` (or a
  directory containing `*.ckpt`; the latest mtime is picked).
- `options.Dataset.normalization_file` — `normalization.pt` matching the
  input parquet.

**4b. Run prediction** (inside the EveNet image, same `docker run` invocation
as Step 3):

```bash
# (inside the container, cwd = /workspace/DGPO-NeurIPS-Addon)
python EveNet-Full/scripts/predict.py prediction/predict_TT2L.yaml
```

The output `.pt` is then fed to the unfolding script as `--model KEY=PATH`
(see Step 5). To produce multiple ablations, copy the template, change
`prediction.filename` and `model_checkpoint_load_path`, and rerun.

## 5. (Optional) RooUnfold — for SVD unfolding only

Overlay-only unfolding plots run in the container with no extra setup. The SVD
panels in `RL/Unfolding/run_neurips_neutrino_unfolding.py` additionally need
PyROOT + a built `libRooUnfold`. The DGPO container does **not** ship ROOT, so
build RooUnfold in a separate conda env (or any ROOT-enabled environment such
as a CVMFS LCG view).

The unfolding script imports only standard Python modules plus `matplotlib`,
`numpy`, `torch`, `vector`, and this repo's `event_selection.event_cuts` —
all already in the EveNet image and in `requirements-dgpo.txt`. The only extra
requirement for SVD panels is a working ROOT/PyROOT environment that can load
the locally built `libRooUnfold`.

```bash
# 4a. Conda env with ROOT + CMake + a working C/C++/Fortran toolchain
conda create -n MyEve -c conda-forge python=3.12 root cmake compilers
conda activate MyEve

# 4b. Build (RooUnfold/ submodule was checked out in Step 1)
cd /path/to/DGPO-NeurIPS-Addon
cmake -S RooUnfold -B RooUnfold/build
cmake --build RooUnfold/build -j 4

# 4c. Load the runtime env in every new shell before SVD unfolding
source RooUnfold/build/setup.sh
```

After the build you should have `RooUnfold/build/libRooUnfold.{so,dylib}` at
the `DGPO-NeurIPS-Addon/` root. The unfolding script auto-discovers it via the
loader paths listed in `docs/roounfold.md` (searched first under the
`DGPO-NeurIPS-Addon/` repo root, then inside `EveNet-Full/` as a fallback,
then system `libRooUnfold`). Override explicitly with:

```bash
export ROOUNFOLD_LIB_PATH=/abs/path/to/libRooUnfold   # no extension
```

Quick sanity check:

```bash
python - <<'PY'
import ROOT
ROOT.gSystem.Load("RooUnfold/build/libRooUnfold")
assert hasattr(ROOT, "RooUnfoldResponse") and hasattr(ROOT, "RooUnfoldSvd")
print("RooUnfold OK")
PY
```

Notes:
- Build RooUnfold on the same OS where you will run the unfolding. Linux
  produces `libRooUnfold.so`; macOS produces `libRooUnfold.dylib`. Do not copy
  a macOS `.dylib` to Linux (or vice versa).
- macOS arm64: install `compilers` from conda-forge (provides
  `arm64-apple-darwin*-clang`) before configuring, otherwise CMake may fall
  back to system Xcode and mismatch ROOT's ABI.
- Tested on macOS arm64 with Python 3.12 + ROOT 6.38; the same recipe works on
  Linux (incl. Perlmutter login nodes via conda or CVMFS).

### Run unfolding

```bash
python RL/Unfolding/run_neurips_neutrino_unfolding.py \
  --model nominal=/path/to/nominal.pt \
  --model ablation_no_anchor=/path/to/ablation_no_anchor.pt \
  --output_dir outputs/unfolding_neutrino
```

`--model KEY=PATH` accepts any number of saved DGPO prediction tensors; the
first model is used as the ratio baseline. Outputs include reconstructed-top /
W / neutrino kinematic overlays plus per-feature SVD unfolding panels
(`pt_nu`, `eta_nu`, `phi_nu`, `pt_nubar`, `eta_nubar`, `phi_nubar`).

---

## Layout reference

- `RL/DGPO_neutrino/` — DGPO trainer, rewards, anchors, `config.yaml`.
- `RL/Unfolding/` — turns saved `.pt` predictions into overlay/unfolding plots.
- `configs/tt2l/` — TT2L YAML defaults overlaid by `config.yaml`.
- `event_selection/event_cuts.py` — shared event selection.
- `docs/` — `install_into_evenet.md`, `container.md`, `roounfold.md`.
- `runtime-image.txt` — `docker.io/avencast1994/evenet:1.5` (single line).
- `requirements-dgpo.txt` — **pin-equivalent to `EveNet-Full/requirements.txt`**.
- `EveNet-Full/` — submodule, see [Step 1](#1-clone-with-submodules).
- `prediction/predict_TT2L.yaml` — template prediction config; see [Step 4](#4-run-prediction--produce-pt-files-for-unfolding).
- `RooUnfold/` — submodule, optional; checked out in [Step 1](#1-clone-with-submodules), built in [Step 5](#5-optional--roounfold-for-svd-unfolding-only).

## License

This add-on is released under the [MIT License](LICENSE). It clones, but does
not modify, two third-party companion repositories: **EveNet-Full** (MIT) and
**RooUnfold** (BSD-3-Clause). All Python runtime dependencies retain their
upstream open-source licenses (BSD / Apache-2.0 / MIT). See
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) for the per-asset table
and source URLs.

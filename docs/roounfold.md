# RooUnfold Integration

`RL/Unfolding` can make kinematic overlays without RooUnfold. SVD unfolding
panels require PyROOT plus a built RooUnfold library.

`RL/Unfolding/run_neurips_neutrino_unfolding.py` imports only standard Python
modules plus `matplotlib`, `numpy`, `torch`, `vector`, `ROOT` (PyROOT, for SVD
panels), and this repo's `event_selection.event_cuts`. The non-ROOT Python
packages are already included in `docker.io/avencast1994/evenet:1.5` and in
`requirements-dgpo.txt`. ROOT/PyROOT and a built `libRooUnfold` are the only
extra requirements for SVD unfolding.

## Build RooUnfold

Build RooUnfold directly under the `DGPO-NeurIPS-Addon/` repo root (cloned in
the README's Step 1c next to `EveNet-Full/`):

```bash
cd DGPO-NeurIPS-Addon

# Use an environment with ROOT/PyROOT, CMake, and compilers.
# Example:
#   conda create -n MyEve -c conda-forge python=3.12 root cmake compilers
#   conda activate MyEve

# (only needed if you skipped Step 1c)
git clone https://gitlab.cern.ch/RooUnfold/RooUnfold.git
cmake -S RooUnfold -B RooUnfold/build
cmake --build RooUnfold/build -j 4
source RooUnfold/build/setup.sh
```

Alternatively point the loader directly at the library prefix:

```bash
export ROOUNFOLD_LIB_PATH=/path/to/libRooUnfold
```

Build RooUnfold on the same OS where you will run the unfolding. Linux produces
`RooUnfold/build/libRooUnfold.so`; macOS produces
`RooUnfold/build/libRooUnfold.dylib`. Do not reuse a Linux `.so` on macOS or a
macOS `.dylib` on Linux. In each new shell/session, re-run:

```bash
source RooUnfold/build/setup.sh
```

If ROOT still cannot find the library, set `ROOUNFOLD_LIB_PATH` to the absolute
library prefix (no `.so` / `.dylib` suffix).

## Loader Search Order

The unfolding scripts try, in order (paths searched first under
`DGPO-NeurIPS-Addon/` itself, then inside `DGPO-NeurIPS-Addon/EveNet-Full/` as
a fallback for older layouts):

- `$ROOUNFOLD_LIB_PATH`
- `external/RooUnfold/build/libRooUnfold`
- `RooUnfold/build-conda/libRooUnfold`
- `RooUnfold/build/libRooUnfold`
- `EveNet-Full/external/RooUnfold/build/libRooUnfold`
- `EveNet-Full/RooUnfold/build-conda/libRooUnfold`
- `EveNet-Full/RooUnfold/build/libRooUnfold`
- `libRooUnfold` from the dynamic library path

## Check ROOT / RooUnfold

Run from the `DGPO-NeurIPS-Addon/` repo root:

```bash
python - <<'PY'
import os
import ROOT

candidates = []
extra = os.environ.get("ROOUNFOLD_LIB_PATH", "").strip()
if extra:
    candidates.append(extra)
for prefix in ("", "EveNet-Full/"):
    candidates += [
        f"{prefix}external/RooUnfold/build/libRooUnfold",
        f"{prefix}RooUnfold/build-conda/libRooUnfold",
        f"{prefix}RooUnfold/build/libRooUnfold",
    ]
candidates.append("libRooUnfold")

for candidate in candidates:
    if ROOT.gSystem.Load(candidate) >= 0:
        print("Loaded RooUnfold from:", candidate)
        break
else:
    raise SystemExit("Could not load RooUnfold")

assert hasattr(ROOT, "RooUnfoldResponse")
assert hasattr(ROOT, "RooUnfoldSvd")
print("OK")
PY
```

## Run Unfolding

```bash
python RL/Unfolding/run_neurips_neutrino_unfolding.py \
  --model nominal=/path/to/nominal.pt \
  --model ablation_no_anchor=/path/to/ablation_no_anchor.pt \
  --output_dir outputs/unfolding_neutrino
```

# Third-Party Licenses

`DGPO-NeurIPS-Addon` is released under the [MIT License](LICENSE). It depends
on, and ships build / clone instructions for, the following third-party assets.
Each retains its original license; nothing in this repository overrides those
upstream terms.

## Cloned companion repositories

| Asset | License | Source | Notes |
|---|---|---|---|
| **EveNet-Full** (incl. `evenet/` core submodule) | MIT | `https://github.com/EveNet-HEP/EveNet-Full` (submodule: `https://github.com/EveNet-HEP/Core`) | Cloned unmodified into `EveNet-Full/` (Step 1b). License file: `EveNet-Full/LICENSE`. |
| **RooUnfold** | BSD 3-Clause | `https://gitlab.cern.ch/RooUnfold/RooUnfold` | Cloned unmodified into `RooUnfold/` (Step 1c). License files: `RooUnfold/LICENSE`, `RooUnfold/LICENSE.spdx`. |

## Container image

| Asset | License | Source |
|---|---|---|
| `docker.io/avencast1994/evenet:1.5` | Built from EveNet-Full (MIT) on top of an NVIDIA CUDA / PyTorch base image | Pinned in `runtime-image.txt`. The base image and bundled PyPI packages retain their respective upstream licenses. |

## Python runtime dependencies

All Python packages installed via `requirements-dgpo.txt` (and equivalently via
the container image) are open-source releases distributed on PyPI under their
own licenses. Major packages and their licenses:

| Package | License |
|---|---|
| `torch` | BSD-3-Clause |
| `lightning` (PyTorch Lightning) | Apache-2.0 |
| `ray` (data, train, tune, serve) | Apache-2.0 |
| `numpy`, `scikit-learn`, `pyarrow`, `h5py` | BSD-3-Clause |
| `numba` | BSD-2-Clause |
| `transformers` | Apache-2.0 |
| `wandb` | MIT |
| `awkward`, `vector`, `uproot`, `hist`, `mplhep` | BSD-3-Clause |
| `lion-pytorch`, `torchJD`, `opt-einsum` | MIT |
| `pyyaml`, `rich`, `seaborn`, `plotly` | MIT |

For exact pins and a complete list, see `requirements-dgpo.txt` (verbatim copy
of `EveNet-Full/requirements.txt`). License metadata is also retrievable via
`pip show <package>` inside the container.

## Datasets

| Dataset | Source | Notes |
|---|---|---|
| TT2L (ν2-Flows benchmark) | Cited in the paper; underlying simulated samples are public per the original ν2-Flows publication | Not redistributed in this repository. Users obtain it from the upstream source. |

## Citing this work and its dependencies

If you use this add-on, please cite the accompanying NeurIPS paper as well as
the upstream EveNet, ν2-Flows, and (if SVD unfolding is used) RooUnfold
publications referenced in the manuscript.

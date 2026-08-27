# Galfind

[![workflow](https://github.com/duncanaustin98/galfind/actions/workflows/python-app.yml/badge.svg)](https://github.com/duncanaustin98/galfind/actions)
[![codecov](https://codecov.io/gh/duncanaustin98/galfind/branch/main/graph/badge.svg)](https://codecov.io/gh/duncanaustin98/galfind)
[![Documentation Status](https://github.com/duncanaustin98/galfind/actions/workflows/publish_docs.yml/badge.svg)](https://galfind.readthedocs.io/en/latest/index.html)
[![Apptainer](https://github.com/duncanaustin98/galfind/actions/workflows/apptainer.yml/badge.svg)](https://github.com/duncanaustin98/galfind/actions/workflows/apptainer.yml)
[![run with apptainer/singularity](https://img.shields.io/badge/run%20with-apptainer%2Fsingularity-1E95D3.svg?labelColor=000000&logo=data:image/svg%2bxml;base64,PHN2ZyB3aWR0aD0iMjQ1IiBoZWlnaHQ9IjI0MCIgdmlld0JveD0iNjAgMCAzMTAgMjUwIiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgk8cGF0aCBkPSJtIDI3MC4xOCwyNTMuOTggYyAtMS44LC0xLjIgLTMuNCwtMyAtNC40LC01LjIgbCAtNTIuNiwtMTE3LjQgYyAtMi4yLC00LjggLTMuOCwtOC42IC01LjIsLTExLjYgLTIuMiwtNC40IC0yLjIsLTUuNiAtMi4yLC02LjQgMCwtMi4yIDAuOCwtMy44IDIuNiwtNC44IHYgLTQuNCBoIC00My4yIHYgNC40IGMgMC44LDAuNCAxLjIsMS4yIDEuOCwxLjggMC40LDAuOCAwLjgsMS44IDAuOCwzIDAsMS4yIC0wLjQsMyAtMS44LDUuNiAtMS4yLDIuNiAtMi42LDUuNiAtNC40LDkuNCBsIC01MS44LDExNyBjIC0wLjgsMS44IC0yLjIsNC40IC0zLjgsNy40IC0xLjgsMyAtNC44LDQuNCAtOC4yLDQuOCB2IDMuOCBoIDQ5LjYgdiAtMy44IGMgLTUuNiwwIC04LjIsLTIuMiAtOC4yLC01LjYgMCwtMS44IDAuOCwtNC44IDMsLTkgMS44LC0zLjQgMy44LC03LjggNS42LC0xMiAyNC42LDkuNCA1Mi4yLDEwIDc2LjgsMC44IDIuMiw0LjQgMy44LDguMiA1LjIsMTEuMiAxLjgsMy40IDIuNiw2LjQgMi42LDguNiAwLDIuMiAtMC44LDMuOCAtMi4yLDQuOCAtMS4yLDAuNCAtMi4yLDAuOCAtMy40LDEuMiB2IDMuOCBoIDUwLjQgdiAtMy44IGMgLTIuOCwtMS44IC01LjQsLTIuOCAtNywtMy42IHogbSAtMTExLjQsLTQ3IDI3LjYsLTYxLjQgMjgsNjIuMiBjIC0xOCw2IC0zNy40LDYgLTU1LjYsLTAuOCB6IiBmaWxsPSJ3aGl0ZSIvPiA8cGF0aCBkPSJtIDg5Ljc4LDE0MC45OCBjIDAsLTkgMS4yLC0xNy42IDMuNCwtMjYuNCBsIC0yOCwtMTIuNiBjIC0zLjgsMTIgLTYsMjQuNiAtNiwzNy42IDAsMzUgMTQuMiw2OC42IDM5LjgsOTIuOCBsIDEuOCwtMy40IDExLjIsLTI1LjQgYyAtMTMuNiwtMTcuNCAtMjIuMiwtMzkgLTIyLjIsLTYyLjYgeiIgZmlsbD0iIzkzOTU5OCIvPiA8cGF0aCBkPSJtIDMxMC4xOCwxMDIuNTggLTI4LDEyLjYgYyAyLjIsOC4yIDMuNCwxNi44IDMuNCwyNS44IDAsMjMuOCAtOC42LDQ1LjggLTIyLjgsNjIuNiBsIDExLjYsMjUuNCAxLjgsMy40IGMgMjUuNCwtMjQuMiAzOS44LC01Ny44IDM5LjgsLTkyLjggLTAuMiwtMTIuNCAtMi4yLC0yNSAtNS44LC0zNyB6IiBmaWxsPSIjRjc5NDIxIi8+IDxwYXRoIGQ9Im0gNzEuMTgsODYuOTggMjcuNiwxMi42IGMgMTQuNiwtMzEgNDQuOCwtNTMgODAuMiwtNTYuMiB2IC0zMC42IGMgLTQ2LDIuNiAtODguNCwzMS40IC0xMDcuOCw3NC4yIHoiIGZpbGw9IiMxRTk1RDMiLz4gPHBhdGggZD0ibSAzMDQuMTgsODYuOTggYyAtMTkuNCwtNDIuOCAtNjEuOCwtNzEuNiAtMTA4LjQsLTc0LjYgdiAzMC42IGMgMzUuOCwzIDY2LDI1IDgwLjYsNTYuMiB6IiBmaWxsPSIjNkZCNTQ0Ii8+PC9zdmc+)](https://sylabs.io/docs/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/flaresimulations/synthesizer/blob/main/docs/CONTRIBUTING.md)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Python package for easy UV, optical and infra-red photometric and spectroscopic galaxy identification.

# Installation with venv
```bash
python -m venv /path_to_dir/{env_name} # Create a virtual environment
cd /path_to_dir/{env_name} # Navigate to the venv directory
source /path_to_dir/{env_name}/bin/activate # Activate the virtual environment
git clone https://github.com/duncanaustin98/galfind.git
cd galfind
pip install -e . # Install galfind in editable mode.
```

# Installation with conda
```bash
conda create -n {env_name} python==3.11 # Create a virtual environment with conda
conda activate {env_name} # Activate the conda environment
git clone https://github.com/duncanaustin98/galfind.git
cd galfind
pip install -e . # Install galfind in editable mode.
```

## Required packages

The following commonly used packages are required to install galfind:

- pip
- git (https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

It is also helpful to have SExtractor installed:

### On linux based systems: ??
```bash
sudo apt-get install sextractor
```

### On MacOS:
```bash
brew install sextractor
```
For this, you will need homebrew installed. Follow instructions at https://brew.sh/

# Installation with Apptainer

An [Apptainer](https://apptainer.org/) definition, `galfind.def`, is provided
as a reproducible, self-contained alternative to a local venv/conda install.
It bundles galfind, SExtractor, and EAZY + Bagpipes SED fitting (LePhare is
not included, since it is a separately-compiled codebase rather than a pip
package). Every build is checked automatically by the
[Apptainer workflow](https://github.com/duncanaustin98/galfind/actions/workflows/apptainer.yml).

Build the image from the repository root (needs root or `--fakeroot`).
`--notest` skips the image's built-in self-test at build time, since that
test needs `/data/GALFIND_WORK` bound to a writable directory, and nothing
can be bound to that path until the build itself has finished (see below to
run the test separately, with a bind, once the image exists):
```bash
apptainer build --fakeroot --notest galfind.sif galfind.def
```

Optionally verify the image once it's built:
```bash
mkdir -p /path/to/your/work /path/to/your/data
apptainer test \
    --bind /path/to/your/work:/data/GALFIND_WORK \
    --bind /path/to/your/data:/data/GALFIND_DATA \
    galfind.sif
```

Run a script, binding your own work/data directories onto the mount points
the shipped config expects. Apptainer also auto-binds your current directory
and `$HOME` by default, so if `your_script.py` lives under one of those it
will already be visible inside the container; otherwise, bind the directory
it lives in explicitly. `--bind /path/to/your/scripts` (no `:dest`) mounts it
at the same path inside the container, so imports/relative paths inside the
script keep working unchanged:
```bash
apptainer run \
    --bind /path/to/your/work:/data/GALFIND_WORK \
    --bind /path/to/your/data:/data/GALFIND_DATA \
    --bind /path/to/your/scripts \
    galfind.sif /path/to/your/scripts/your_script.py
```

Or drop into an interactive shell:
```bash
apptainer shell \
    --bind /path/to/your/work:/data/GALFIND_WORK \
    --bind /path/to/your/data:/data/GALFIND_DATA \
    --bind /path/to/your/scripts \
    galfind.sif
```

**_NOTE:_** `import bagpipes` currently fails at runtime inside the container
(`ModuleNotFoundError: bagpipes.configs`) due to a missing package-data bug in
the [tHarvey303/bagpipes](https://github.com/tHarvey303/bagpipes) fork itself,
not in galfind or this image. galfind and its EAZY-based SED fitting are
unaffected.

---
**_NOTE:_** Since the contribution guidelines have not yet been written, if you intend to add any new features to galfind, please raise an issue on GitHub and inform me at duncan.austin@postgrad.manchester.ac.uk or via the EPOCHS slack channel
---

# Galfind

[![workflow](https://github.com/duncanaustin98/galfind/actions/workflows/python-app.yml/badge.svg)](https://github.com/duncanaustin98/galfind/actions)
[![codecov](https://codecov.io/gh/duncanaustin98/galfind/branch/main/graph/badge.svg)](https://codecov.io/gh/duncanaustin98/galfind)
[![Documentation Status](https://github.com/duncanaustin98/galfind/actions/workflows/publish_docs.yml/badge.svg)](https://galfind.readthedocs.io/en/latest/index.html)
[![Apptainer](https://github.com/duncanaustin98/galfind/actions/workflows/apptainer.yml/badge.svg)](https://github.com/duncanaustin98/galfind/actions/workflows/apptainer.yml)
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
conda create -n {env_name} python==3.9 # Create a virtual environment with conda
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

Build the image from the repository root (needs root or `--fakeroot`):
```bash
apptainer build --fakeroot galfind.sif galfind.def
```

Run a script, binding your own work/data directories onto the mount points
the shipped config expects:
```bash
apptainer run \
    --bind /path/to/your/work:/data/GALFIND_WORK \
    --bind /path/to/your/data:/data/GALFIND_DATA \
    galfind.sif your_script.py
```

Or drop into an interactive shell:
```bash
apptainer shell --bind /path/to/your/work:/data/GALFIND_WORK --bind /path/to/your/data:/data/GALFIND_DATA galfind.sif
```

**_NOTE:_** `import bagpipes` currently fails at runtime inside the container
(`ModuleNotFoundError: bagpipes.configs`) due to a missing package-data bug in
the [tHarvey303/bagpipes](https://github.com/tHarvey303/bagpipes) fork itself,
not in galfind or this image. galfind and its EAZY-based SED fitting are
unaffected.

---
**_NOTE:_** Since the contribution guidelines have not yet been written, if you intend to add any new features to galfind, please raise an issue on GitHub and inform me at duncan.austin@postgrad.manchester.ac.uk or via the EPOCHS slack channel
---

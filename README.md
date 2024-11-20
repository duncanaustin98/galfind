# Galfind

[![workflow](https://github.com/duncanaustin98/galfind/actions/workflows/python-app.yml/badge.svg)](https://github.com/duncanaustin98/galfind/actions)
[![Documentation Status](https://github.com/duncanaustin98/galfind/actions/workflows/publish_docs.yml/badge.svg)](https://galfind.readthedocs.io/en/latest/index.html)
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

---
**_NOTE:_** Since the contribution guidelines have not yet been written, if you intend to add any new features to galfind, please raise an issue on GitHub and inform me at duncan.austin@postgrad.manchester.ac.uk or via the EPOCHS slack channel
---

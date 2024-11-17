==================
Installation Guide
==================

To install the galfind, you will first need to create a virtual environment, using either...

python -m venv galfind_env # Create a virtual environment
conda create -n galfind_env python=3.9 # Create a virtual environment with conda

You will then need to git clone the repository into the appropriate folder:
https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

You will now need to activate your conda/virtual environment:
source activate galfind_env # Activate the virtual environment
conda activate galfind_env # Activate the conda environment

Then run pip install -e . in the galfind root directory to install in editable mode.

To download SExtractor: `brew install sextractor`
You will need homebrew installed: To do this follow instructions at https://brew.sh/

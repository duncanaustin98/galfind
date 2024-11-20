==================
Installation Guide
==================


Installation with venv
=======================

.. code-block:: bash

    python -m venv /path_to_dir/{env_name} # Create a virtual environment
    cd /path_to_dir/{env_name} # Navigate to the venv directory
    source /path_to_dir/{env_name}/bin/activate # Activate the virtual environment
    git clone https://github.com/duncanaustin98/galfind.git
    cd galfind
    pip install -e . # Install galfind in editable mode.


Installation with conda
=======================

.. code-block:: bash

    conda create -n {env_name} python==3.9 # Create a virtual environment with conda
    conda activate {env_name} # Activate the conda environment
    git clone https://github.com/duncanaustin98/galfind.git
    cd galfind
    pip install -e . # Install galfind in editable mode.


Required packages
=================

The following commonly used packages are required to install galfind:

* pip
* git (https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

It is also helpful to have SExtractor installed:

On linux based systems: ??
""""""""""""""""""""""""""
.. code-block:: bash

    sudo apt-get install sextractor

On MacOS:
""""""""
.. code-block:: bash

    brew install sextractor
For this, you will need homebrew installed. Follow instructions at https://brew.sh/

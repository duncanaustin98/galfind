# Galfind

[![workflow](https://github.com/duncanaustin98/galfind/actions/workflows/python-app.yml/badge.svg)](https://github.com/duncanaustin98/galfind/actions)
[![Documentation Status](https://github.com/duncanaustin98/galfind/actions/workflows/publish_docs.yml/badge.svg)](https://galfind.readthedocs.io/en/instr_rest_post_ruff/index.html)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/flaresimulations/synthesizer/blob/main/docs/CONTRIBUTING.md)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Python package for easy UV, optical and infra-red photometric and spectroscopic galaxy identification.

# Instructions for code editing by members of Chris Conselice's Group EPOCHS at the Jodrell Bank Centre for Astrophysics at UoM
1) Clone this repository locally using ```git clone https://github.com/u92876da/galfind.git``` in the directory of your choosing. I would recommend creating a new directory called GALFIND. If using morgan you can clone this into your own "/nvme/scratch/work/{name}" working folder using ```git clone /nvme/scratch/work/austind/GALFIND```. This creates your own local repository of the local/remote galfind code.
2) Create a branch using ```git branch {branch_name}``` where the branch name should describe the new feature you would like to incorporate into the galfind code. Remember once you have finished editing the code in the branch to first stage your new code using ```git add .``` and (if you are happy with the files you have staged) commit them to your branch using ```git commit -m "{message}"``` You can switch between branches using ```git checkout {branch_name}``` if you wish to switch back to the main branch but bare in mind that you will lose all changes within your branch if you havn't already either used ```git commit``` or ```git stash```. More information on the basics of git are shown here https://rogerdudler.github.io/git-guide/ (apologies for the rude word in the google search).
3) If you have cloned locally on morgan, you can use ```git push -u origin {new_master_branch_name}``` to push to a new branch in the master galfind repository and ```git pull``` to pull files from the master galfind repository located at /nvme/scratch/work/austind/GALFIND. If you have cloned remotely, you will need to push and pull via GitHub. These are completed via pull requests which I still need to test and learn more about. These push commands will merge your (tested!) code with the master galfind code and MUST be merged with a galfind branch rather than main as to not overwrite the working pipeline.
# PLEASE NOTE: If you intend to add any new features to galfind, please raise an issue on GitHub and inform me at duncan.austin@postgrad.manchester.ac.uk or via the EPOCHS slack channel

To install your local version of galfind for testing, first make a new environment and then run ```pip install -e .``` from your clones GALFIND folder.

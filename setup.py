#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:59:40 2023

@author: austind
"""

from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import shutil
import atexit

package_name = "galfind"

def install_mplstyle():
    import matplotlib as mpl
    stylefile = "galfind_style.mplstyle"

    mpl_stylelib_dir = os.path.join(mpl.matplotlib_fname().replace("/matplotlibrc", ""), "stylelib")
    if not os.path.exists(mpl_stylelib_dir):
        os.makedirs(mpl_stylelib_dir)
    
    shutil.copy(
        os.path.join(os.getcwd(), f"{package_name}/{stylefile}"),
        os.path.join(mpl_stylelib_dir, stylefile))

def load_requirements(path):
    if os.path.isfile(path):
        with open(path) as f:
            return list(f.read().splitlines())
    else:
        raise Exception(f"No requirement path = {path}")

class PostInstallMoveFile(install):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        atexit.register(install_mplstyle)

#setup()
setup(
    name=package_name,
    version='0.0',
    description='Module for catalogue creation of galaxies from photometric imaging',
    author='Duncan Austin',
    author_email='duncan.austin@postgrad.manchester.ac.uk',
    packages=find_packages(),
    install_requires=load_requirements("requirements.txt"), #, #external packages as dependencies
    cmdclass={'install': PostInstallMoveFile}
)

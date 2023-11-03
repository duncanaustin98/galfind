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
import matplotlib as mpl

package_name = "galfind"

def install_mplstyle():
    stylefile = "galfind_style.mplstyle"

    mpl_stylelib_dir = os.path.join(mpl.matplotlib_fname().replace("/matplotlibrc", ""), "stylelib")
    if not os.path.exists(mpl_stylelib_dir):
        os.makedirs(mpl_stylelib_dir)
    
    shutil.copy(
        os.path.join(os.getcwd(), f"{package_name}/{stylefile}"),
        os.path.join(mpl_stylelib_dir, stylefile))

class PostInstallMoveFile(install):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        atexit.register(install_mplstyle)

setup(
      name=package_name,
      version='1.0',
      description='Module for catalogue creation of galaxies from photometric imaging',
      author='Duncan Austin',
      author_email='duncan.austin@postgrad.manchester.ac.uk',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib'], #, #external packages as dependencies
      cmdclass={'install': PostInstallMoveFile}
)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:59:40 2023

@author: austind
"""

from setuptools import setup, find_packages

setup(
   name='galfind',
   version='1.0',
   description='Module for catalogue creation of galaxies from photometric imaging',
   author='Duncan Austin',
   author_email='duncan.austin@postgrad.manchester.ac.uk' ,
   packages=find_packages()) #,  #same as name
   #install_requires=['bagpipes'], #external packages as dependencies
#)

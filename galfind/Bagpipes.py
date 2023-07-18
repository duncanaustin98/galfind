#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:55:57 2023

@author: austind
"""

# Bagpipes.py

from . import SED_code

# %% Bagpipes SED fitting code

class Bagpipes(SED_code):
    
    def __init__(self): #, SFH, priors, t_SFR, SPS_code):
        galaxy_property_labels = {}
        super().__init__()
    
    def make_in(self, cat):
        pass
    
    def run_fit(self, in_path):
        pass
    
    def make_fits_from_out(self, out_path):
        pass
    
    def out_fits_name(self, out_path):
        pass
    
    def extract_SEDs(self, path):
        pass
    
    def extract_z_PDF(self, path):
        pass
    
    def z_PDF_path_from_cat_path(self, cat_path, ID, low_z_run = False):
        pass
    
    def SED_path_from_cat_path(self, cat_path, ID, low_z_run = False):
        pass

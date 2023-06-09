#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 00:16:05 2023

@author: austind
"""

# Simulated_Catalogue.py

import numpy as np

from . import Simulated_Galaxy

def Simulated_Catalogue(Catalogue):
    
    def __init__(self, gals, cat_path, survey_name, sim):
        self.sim = sim
        super().__init__(gals, cat_path, survey_name)
    
    @classmethod
    def from_sim_cat(cls, cat_path, instrument, codes, survey_name, sim):
        # open the catalogue
        cat = Simulated_Catalogue.cat_from_path(cat_path)
        # produce galaxy array from each row of the catalogue
        gals = np.array([Simulated_Galaxy.from_sim_cat_row(row, instrument, codes, sim) for row in cat])
        return cls(gals, cat_path, survey_name, sim)
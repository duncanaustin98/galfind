#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:50:27 2023

@author: austind
"""

# SED_result.py

from .Photometry_rest import Photometry_rest

class SED_result:
    
    def __init__(self, phot, z, code, chi_sqs = None, z_PDF_gal = None, SEDs = None, low_z_run = False):
        self.photometry_rest = Photometry_rest.from_phot(phot, z)
        self.z = z
        self.chi_sqs = chi_sqs
        self.z_PDF_gal = z_PDF_gal
        self.SEDs = SEDs
        self.code = code
        self.low_z_run = low_z_run
        
    @classmethod
    def from_fits_cat(cls, fits_cat_row, code, phot, cat_creator, low_z_run):
        # could construct the photometry from the raw catalogue using cat_creator here
        try:
            z = float(fits_cat_row[code.galaxy_property_labels["z_phot"]])
        except:
            raise(Exception(f"SED run not performed for {code}, low_z_run = {low_z_run}"))
        chi_sqs = {name: float(fits_cat_row[chi_sq]) for name, chi_sq in code.chi_sq_labels.items()}
        ID = int(fits_cat_row[cat_creator.ID_label])
        z_PDF = [] #code.extract_z_PDF(fits_cat_row, ID, low_z_run)
        SEDs = [] #code.extract_SEDs(fits_cat_row, ID, low_z_run)
        return cls(phot, z, code, chi_sqs, z_PDF, SEDs, low_z_run)
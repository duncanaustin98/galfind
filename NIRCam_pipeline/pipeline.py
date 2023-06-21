#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:36:34 2023

@author: austind
"""

# NIRCam_pipeline.py

import astropy.units as u
import numpy as np

from galfind import Catalogue, LePhare, EAZY
from galfind.Catalogue_Creator import GALFIND_Catalogue_Creator

def NIRCam_pipeline(surveys, version, xy_offsets, aper_diams, sed_codes, min_flux_pc_errs, forced_phot_band, excl_bands, cat_type = "loc_depth", NIRCam_ZP = 28.08, n_loc_depth_samples = 5):
    for pc_err in min_flux_pc_errs:
        # make appropriate galfind catalogue creator for each aperture diameter
        cat_creator = GALFIND_Catalogue_Creator(cat_type, aper_diams[0], pc_err, NIRCam_ZP)
        for survey, xy_offset in zip(surveys, xy_offsets):
            cat = Catalogue.from_NIRCam_pipeline(survey, version, aper_diams, cat_creator, xy_offset, forced_phot_band, excl_bands, min_flux_pc_errs, n_loc_depth_samples)
            for code in sed_codes:
                cat = code.fit_cat(cat)

if __name__ == "__main__":
    version = "v8a"
    cat_type = "loc_depth"
    surveys = ["NEP-1"]
    aper_diams = [0.32] * u.arcsec
    xy_offsets = [[0, 0]]
    sed_codes = [LePhare()]
    min_flux_pc_errs = [10]
    forced_phot_band = "f444W"
    excl_bands = [] #["f090W", "f115W", "f277W", "f335M", "f356W", "f410M", "f444W"]
    n_loc_depth_samples = 5
    NIRCam_pipeline(surveys, version, xy_offsets, aper_diams, sed_codes, min_flux_pc_errs, forced_phot_band, excl_bands, cat_type = cat_type, n_loc_depth_samples = n_loc_depth_samples)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:36:34 2023

@author: austind
"""

# NIRCam_pipeline.py

import astropy.units as u
import numpy as np
import time

from galfind import Catalogue, LePhare, EAZY, config
from galfind.Catalogue_Creator import GALFIND_Catalogue_Creator

def pipeline(surveys, version,instruments, xy_offsets, aper_diams, code_names, low_z_runs, min_flux_pc_errs, forced_phot_band, excl_bands, \
             cat_type = "loc_depth", n_loc_depth_samples = 5, fast = True, eazy_templates = "fsps_larson"):
    for pc_err in min_flux_pc_errs:
        # make appropriate galfind catalogue creator for each aperture diameter
        cat_creator = GALFIND_Catalogue_Creator(cat_type, aper_diams[0], pc_err)
        for survey, xy_offset in zip(surveys, xy_offsets):
            cat = Catalogue.from_pipeline(survey = survey, version = version, instruments = instruments, aper_diams = aper_diams, cat_creator = cat_creator, code_names = code_names, low_z_runs = low_z_runs, xy_offset = xy_offset, \
                                          forced_phot_band = forced_phot_band, excl_bands = excl_bands, loc_depth_min_flux_pc_errs = min_flux_pc_errs, n_loc_depth_samples = n_loc_depth_samples, fast = fast)
            # for i, code in enumerate(sed_codes):
            #     cat = code.fit_cat(cat, templates = eazy_templates)
            #     #code.fit_cat(cat, templates = eazy_templates)
            #     # calculate the extended source corrections
            #     if code.code_name == "LePhare":
            #         cat.make_ext_src_corr_cat(code.code_name)
            #     # calculate the UV properties for this catalogue
            #     if instruments == ["NIRCam"]: # QUICK FIX!
            #         print("Instruments name is a QUICK FIX!")
            #         instruments_name = "NIRCam"
            #     cat.make_UV_fit_cat(UV_PDF_path = f"{config['RestUVProperties']['UV_PDF_PATH']}/{version}/{instruments_name}/{survey}/{code.code_name}+{pc_err}pc")  

if __name__ == "__main__":
    version = "v9"
    instruments = ['NIRCam', 'ACS_WFC', 'WFC3IR'] # Can leave this - if there is no data for an instrument it is removed automatically
    cat_type = "loc_depth"
    surveys = ["CEERSP10"]
    aper_diams = [0.32] * u.arcsec
    xy_offsets = [[200, 0]]
    code_names = ["EAZY", "EAZY"] #[LePhare()]
    low_z_runs = [False, True]
    eazy_templates = "fsps_larson"
    min_flux_pc_errs = [5, 10]
    forced_phot_band = ['f277W', "f356W","f444W"]
    fast_depths = False
    excl_bands = [] #["f606W", "f814W", "f090W", "f115W", "f277W", "f335M", "f356W", "f410M", "f444W"]
    n_loc_depth_samples = 5

    for survey in surveys:
        pipeline([survey], version,instruments, xy_offsets, aper_diams, code_names, low_z_runs, min_flux_pc_errs, forced_phot_band, excl_bands, cat_type = cat_type, n_loc_depth_samples = n_loc_depth_samples, fast = fast_depths, eazy_templates = eazy_templates)

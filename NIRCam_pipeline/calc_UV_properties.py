#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 12:20:13 2023

@author: austind
"""

# calc_UV_properties.py

import astropy.units as u
import numpy as np
import time

from galfind import Catalogue, config, LePhare, EAZY
from galfind.Catalogue_Creator import GALFIND_Catalogue_Creator

def calc_UV_properties(surveys, version,instruments, xy_offsets, aper_diams, code_names, lowz_zmax, min_flux_pc_errs, forced_phot_band, excl_bands, \
             cat_type = "loc_depth", n_loc_depth_samples = 5, fast = True, templates_arr = ["fsps", "fsps_larson", "fsps_jades"]):
    for pc_err in min_flux_pc_errs:
        # make appropriate galfind catalogue creator for each aperture diameter
        cat_creator = GALFIND_Catalogue_Creator(cat_type, aper_diams[0], pc_err)
        for survey, xy_offset in zip(surveys, xy_offsets):
            cat = Catalogue.from_pipeline(survey = survey, version = version, instruments = instruments, aper_diams = aper_diams, cat_creator = cat_creator, code_names = code_names, lowz_zmax = lowz_zmax, xy_offset = xy_offset, \
                                          forced_phot_band = forced_phot_band, excl_bands = excl_bands, loc_depth_min_flux_pc_errs = min_flux_pc_errs, n_loc_depth_samples = n_loc_depth_samples, templates_arr = templates_arr, fast = fast)
            print(cat[0].phot.flux_Jy)
            for i, code_name in enumerate(code_names):
                # calculate the extended source corrections
                cat.make_ext_src_corr_cat(code_name, templates_arr)
                # calculate the UV properties for this catalogue
                # instruments_name = "+".join(instruments)
                # print(instruments_name)
                # cat.make_UV_fit_cat(UV_PDF_path = f"{config['RestUVProperties']['UV_PDF_PATH']}/{version}/{instruments_name}/{survey}/{code_name}+{pc_err}pc")  

if __name__ == "__main__":
    version = "v9" #config["DEFAULT"]["VERSION"] #"v9_sex_test1"
    instruments = ['NIRCam', 'ACS_WFC'] #, 'WFC3IR'] # Can leave this - if there is no data for an instrument it is removed automatically
    cat_type = "loc_depth"
    surveys = ["NEP-1"] #[config["DEFAULT"]["SURVEY"]] # [f"CEERSP{int(i + 1)}" for i in range(0, 10)] #
    aper_diams = [0.32] * u.arcsec
    xy_offsets = [[0, 0]]
    code_names = ["EAZY", "EAZY", "EAZY"] #[EAZY(), EAZY(), EAZY()] #, "EAZY"] #[LePhare()]
    templates_arr = ["fsps", "fsps_larson", "fsps_jades"] #["fsps", "fsps_larson", "fsps_jades"]
    eazy_zmax_lowz = [4., 6., None]
    min_flux_pc_errs = [5, 10]
    forced_phot_band = ["f277W", "f356W", "f444W"]
    fast_depths = False
    excl_bands = [] #["f606W", "f814W", "f090W", "f115W", "f277W", "f335M", "f356W", "f410M", "f444W"]
    n_loc_depth_samples = 10

    for survey in surveys:
        calc_UV_properties([survey], version,instruments, xy_offsets, aper_diams, code_names, eazy_zmax_lowz, min_flux_pc_errs, forced_phot_band, excl_bands, cat_type = cat_type, n_loc_depth_samples = n_loc_depth_samples, fast = fast_depths, templates_arr = templates_arr)

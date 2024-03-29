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

from galfind import Catalogue, config #, LePhare, EAZY, 
from galfind.Catalogue_Creator import GALFIND_Catalogue_Creator

def pipeline(surveys, version, instruments, aper_diams, code_names, lowz_zmax, min_flux_pc_errs, forced_phot_band, excl_bands, \
             cat_type = "loc_depth", eazy_templates = ["fsps_larson"]):
    for pc_err in min_flux_pc_errs:
        # make appropriate galfind catalogue creator for each aperture diameter
        cat_creator = GALFIND_Catalogue_Creator(cat_type, aper_diams[0], pc_err)
        for survey in surveys:
            cat = Catalogue.from_pipeline(survey = survey, version = version, instruments = instruments, aper_diams = aper_diams, cat_creator = cat_creator, code_names = code_names, lowz_zmax = lowz_zmax, \
                                          forced_phot_band = forced_phot_band, excl_bands = excl_bands, loc_depth_min_flux_pc_errs = min_flux_pc_errs, templates_arr = eazy_templates)
            print(cat.cat_path)
            cat.data.calc_unmasked_area(forced_phot_band)
            
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
    version = "v11" #config["DEFAULT"]["VERSION"] #"v9_sex_test1"
    instruments = ["NIRCam"] #, 'ACS_WFC'] #, 'WFC3IR'] # Can leave this - if there is no data for an instrument it is removed automatically
    cat_type = "loc_depth"
    surveys = ["G191"] #[config["DEFAULT"]["SURVEY"]] # [f"CEERSP{int(i + 1)}" for i in range(0, 10)] #
    aper_diams = [0.32] * u.arcsec
    code_names = ["EAZY"]
    eazy_templates = ["fsps_larson"] #["fsps", "fsps_larson", "fsps_jades"]
    eazy_lowz_zmax = [[4., 6.]] #, [4., 6.], [4., 6.]]
    min_flux_pc_errs = [10]
    forced_phot_band = ["f277W", "f356W", "f444W"]
    jems_bands = ["f182M", "f210M", "f430M", "f460M", "f480M"]

    ngdeep_excl_bands = ["f435W", "f775W", "f850LP"]
    #jades_3215_excl_bands = ["f162M", "f115W", "f150W", "f200W", "f410M", "f182M", "f210M", "f250M", "f300M", "f335M", "f277W", "f356W", "f444W"] 
    excl_bands = []

    for survey in surveys:
        pipeline([survey], version, instruments, aper_diams, code_names, eazy_lowz_zmax, min_flux_pc_errs, forced_phot_band, excl_bands, cat_type = cat_type, eazy_templates = eazy_templates)

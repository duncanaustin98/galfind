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

from galfind import useful_funcs_austind as funcs
from galfind import Catalogue, config, LePhare, EAZY, NIRCam, Multiple_Catalogue
from galfind.Catalogue_Creator import GALFIND_Catalogue_Creator

def pipeline(surveys, version, instruments, aper_diams, min_flux_pc_errs, forced_phot_band, \
        excl_bands, SED_fit_params_arr, cat_type = "loc_depth", crop_by = None):
    for pc_err in min_flux_pc_errs:
        # make appropriate galfind catalogue creator for each aperture diameter
        cat_creator = GALFIND_Catalogue_Creator(cat_type, aper_diams[0], pc_err)
        cats = []
        for survey in surveys:
            cat = Catalogue.from_pipeline(survey = survey, version = version, instruments = instruments, aper_diams = aper_diams, \
                cat_creator = cat_creator, SED_fit_params_arr = SED_fit_params_arr, forced_phot_band = forced_phot_band, \
                excl_bands = excl_bands, loc_depth_min_flux_pc_errs = min_flux_pc_errs, crop_by = crop_by)
            cats.append(cat)

        cat_final = cats[0] * cats[1]   
    return cat_final

            

def make_EAZY_SED_fit_params_arr(SED_code_arr, templates_arr, lowz_zmax_arr):
    return [{"code": code, "templates": templates, "lowz_zmax": lowz_zmax} \
        for code, templates, lowz_zmaxs in zip(SED_code_arr, templates_arr, lowz_zmax_arr) for lowz_zmax in lowz_zmaxs]

if __name__ == "__main__":
    version = "v11" #config["DEFAULT"]["VERSION"]
    instruments = ["NIRCam"] #, 'ACS_WFC'] #, 'WFC3_IR']
    cat_type = "loc_depth"
    surveys = ["COSMOS-Web-0A", "COSMOS-Web-0B"] #[config["DEFAULT"]["SURVEY"]]
    aper_diams = [0.32] * u.arcsec
    SED_code_arr = [EAZY()]
    templates_arr = ["fsps_larson"] #["fsps", "fsps_larson", "fsps_jades"]
    lowz_zmax_arr = [[4., 6., None]]
    min_flux_pc_errs = [10]
    forced_phot_band = ["F444W"] # ["F444W"]
    crop_by = None

    jems_bands = ["F182M", "F210M", "F430M", "F460M", "F480M"]
    ngdeep_excl_bands = ["F435W", "F775W", "F850LP"]
    #jades_3215_excl_bands = ["f162M", "f115W", "f150W", "f200W", "f410M", "f182M", "f210M", "f250M", "f300M", "f335M", "f277W", "f356W", "f444W"] 
    excl_bands = []

    SED_fit_params_arr = make_EAZY_SED_fit_params_arr(SED_code_arr, templates_arr, lowz_zmax_arr)

    
    cat = pipeline(surveys, version, instruments, aper_diams, min_flux_pc_errs, forced_phot_band,
    excl_bands, SED_fit_params_arr, cat_type = cat_type, crop_by = crop_by)

    combined = cat.cat_arr[0] * cat.cat_arr[1]



   





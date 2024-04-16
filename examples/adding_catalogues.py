#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue April 9 16:36:34 2023

@author: tharvey
"""

# NIRCam_pipeline.py

import astropy.units as u
import numpy as np
import time

from galfind import useful_funcs_austind as funcs
from galfind import Catalogue, config #, LePhare, EAZY, 
from galfind.Catalogue_Creator import GALFIND_Catalogue_Creator
from galfind.Multiple_Catalogue import Multiple_Catalogue

def pipeline(surveys, version, instruments, aper_diams, code_names, lowz_zmax, min_flux_pc_errs, forced_phot_band, excl_bands, \
             cat_type = "loc_depth", eazy_templates = ["fsps_larson"], select_by = None):
    for pc_err in min_flux_pc_errs:
        # make appropriate galfind catalogue creator for each aperture diameter
        cat_creator = GALFIND_Catalogue_Creator(cat_type, aper_diams[0], pc_err)
        all_cats = []
        for survey in surveys:
            cat = Catalogue.from_pipeline(survey = survey, version = version, instruments = instruments, aper_diams = aper_diams, cat_creator = cat_creator, code_names = code_names, lowz_zmax = lowz_zmax, \
                                          forced_phot_band = forced_phot_band, excl_bands = excl_bands, loc_depth_min_flux_pc_errs = min_flux_pc_errs, templates_arr = eazy_templates, select_by = select_by)
            #cat.data.calc_unmasked_area("NIRCam", forced_phot_band = forced_phot_band)
            all_cats.append(cat)
        # + all cats in all_cats
        print('Single catalogue is ', len(cat))
        #combined_cat_1 = Multiple_Catalogue(all_cats, 'Combined 1')
        #print(str(combined_cat_1))
        #print(len(combined_cat_1))
        #combined_cat_2 = all_cats[0] + all_cats[1]
        #print(str(combined_cat_2))
        #print(len(combined_cat_2))

        combined_cat_3 = all_cats[0] * all_cats[1]
        print(str(combined_cat_3))
        print(len(combined_cat_3))
        # getattr()
            
       
if __name__ == "__main__":
    version = "v11" #config["DEFAULT"]["VERSION"]
    instruments = ["NIRCam"] #, 'ACS_WFC'] #, 'WFC3_IR']
    cat_type = "loc_depth"
    surveys = ["NGDEEP2", "NGDEEP2"] #[config["DEFAULT"]["SURVEY"]]
    aper_diams = [0.32] * u.arcsec
    code_names = ["EAZY"]
    eazy_templates = ["fsps_larson"] #["fsps", "fsps_larson", "fsps_jades"]
    eazy_lowz_zmax = [4., 6., None] #, [4., 6.], [4., 6.]]
    min_flux_pc_errs = [10]
    forced_phot_band = ["F277W", "F356W", "F444W"]

    jems_bands = ["F182M", "F210M", "F430M", "F460M", "F480M"]
    ngdeep_excl_bands = ["F435W", "F775W", "F850LP"]
    #jades_3215_excl_bands = ["f162M", "f115W", "f150W", "f200W", "f410M", "f182M", "f210M", "f250M", "f300M", "f335M", "f277W", "f356W", "f444W"] 
    excl_bands = []

    
    pipeline(surveys, version, instruments, aper_diams, code_names, eazy_lowz_zmax, min_flux_pc_errs, forced_phot_band, excl_bands, cat_type = cat_type, eazy_templates = eazy_templates)

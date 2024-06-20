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
from galfind import Catalogue, config, LePhare, EAZY, NIRCam
from galfind.Catalogue_Creator import GALFIND_Catalogue_Creator

def pipeline(surveys, version, instruments, aper_diams, min_flux_pc_errs, forced_phot_band, \
        excl_bands, SED_fit_params_arr, cat_type = "loc_depth", crop_by = None, timed = True):
    for pc_err in min_flux_pc_errs:
        # make appropriate galfind catalogue creator for each aperture diameter
        cat_creator = GALFIND_Catalogue_Creator(cat_type, aper_diams[0], pc_err)
        for survey in surveys:
            start = time.time()
            cat = Catalogue.from_pipeline(survey = survey, version = version, instruments = instruments, aper_diams = aper_diams, \
                cat_creator = cat_creator, SED_fit_params_arr = SED_fit_params_arr, forced_phot_band = forced_phot_band, \
                excl_bands = excl_bands, loc_depth_min_flux_pc_errs = min_flux_pc_errs, crop_by = crop_by, timed = timed)

            cat_copy = cat.select_EPOCHS(allow_lowz = False)
            cat_copy.plot_phot_diagnostics(flux_unit = u.ABmag)
            print(str(cat))

            end = time.time()
            print(f"Time to load catalogue = {(end - start):.1f}s")

            #cat.data.calc_unmasked_area("NIRCam", forced_phot_band = forced_phot_band)
            #cat.select_band_flux_radius("F277W", "gtr", 1.5)

            #cat.del_hdu(SED_fit_params_arr[-1]["code"].label_from_SED_fit_params(SED_fit_params_arr[-1]))

            # # print(cat.SED_rest_properties)
            # for name in ["EW_obs_Halpha_cont_0.1", "flux_Halpha_cont_0.1_obs_M99_C00", "lum_Halpha_cont_0.1_obs_M99_C00", "xi_ion_Halpha_cont_0.1_M99_C00_fesc0.0"]:
            #     # "EW_obs_Halpha_cont_0.0", "flux_Halpha_cont_0.0_obs_M99_C00", "lum_Halpha_cont_0.0_obs_M99_C00", \
            #     # "continuum_Halpha+[NII]-6583", "EW_rest_Halpha_cont_0.0", "flux_Halpha_cont_0.0_rest_M99_C00", "lum_Halpha_cont_0.0_rest_M99_C00", 
            #     cat.del_SED_rest_property(name)
            #     print(f"deleted {name}")

            # cat_copy = cat.select_phot_galaxy_property("z", ">", 4.5)
            # cat_copy = cat.select_unmasked_instrument(NIRCam())

            #cat_copy = cat.select_all_bands()
            #cat_copy = cat.select_phot_galaxy_property("z", ">", 4.5)

            #cat.select_rest_UV_line_emitters_sigma("CIV-1549", 2.)
            
            # cat.calc_rest_UV_properties(frame = "rest")
            # cat.calc_xi_ion() #dust_author_year = None


def make_EAZY_SED_fit_params_arr(SED_code_arr, templates_arr, lowz_zmax_arr):
    return [{"code": code, "templates": templates, "lowz_zmax": lowz_zmax} \
        for code, templates, lowz_zmaxs in zip(SED_code_arr, templates_arr, lowz_zmax_arr) for lowz_zmax in lowz_zmaxs]

if __name__ == "__main__":

    version = "Windhorst-BCGsub" #config["DEFAULT"]["VERSION"]
    instruments = ["NIRCam"] #["NIRCam", "ACS_WFC"] #,  'WFC3_IR'] # "ACS_WFC"
    cat_type = "loc_depth"
    surveys = ["MACS-0416"] #[config["DEFAULT"]["SURVEY"]]
    aper_diams = [0.32] * u.arcsec
    SED_code_arr = []#EAZY()]
    templates_arr = ["fsps_larson"] #["fsps", "fsps_larson", "fsps_jades"]
    lowz_zmax_arr = [[4., 6., None]] #[[None]] # 
    min_flux_pc_errs = [10]
    forced_phot_band = ["F277W", "F356W", "F444W"] #["F444W"]
    crop_by = None #"bands>13+EPOCHS" #"EPOCHS_lowz+z>4.5"
    timed = False

    jems_bands = ["F182M", "F210M", "F430M", "F460M", "F480M"]
    ngdeep_excl_bands = ["F435W", "F775W", "F850LP"]
    #jades_3215_excl_bands = ["f162M", "f115W", "f150W", "f200W", "f410M", "f182M", "f210M", "f250M", "f300M", "f335M", "f277W", "f356W", "f444W"] 
    excl_bands = []

    SED_fit_params_arr = make_EAZY_SED_fit_params_arr(SED_code_arr, templates_arr, lowz_zmax_arr)

    # delay_time = (8 * u.h).to(u.s).value
    # print(f"{surveys[0]} delayed by {delay_time}s")
    # time.sleep(delay_time)

    for survey in surveys:
        pipeline([survey], version, instruments, aper_diams, min_flux_pc_errs, forced_phot_band, \
        excl_bands, SED_fit_params_arr, cat_type = cat_type, crop_by = crop_by, timed = timed)

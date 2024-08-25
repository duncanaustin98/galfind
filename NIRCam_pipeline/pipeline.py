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
from galfind import Catalogue, config, LePhare, EAZY, Bagpipes, NIRCam, Number_Density_Function
from galfind.Catalogue_Creator import GALFIND_Catalogue_Creator

def pipeline(surveys, version, instruments, aper_diams, min_flux_pc_errs, forced_phot_band, \
        excl_bands, SED_fit_params_arr, cat_type = "loc_depth", crop_by = None, load_PDFs = True, \
        load_SEDs = True, timed = True, mask_stars = True, pix_scales = {"ACS_WFC": 0.03 * u.arcsec, \
        "WFC3_IR": 0.03 * u.arcsec, "NIRCam": 0.03 * u.arcsec, "MIRI": 0.09 * u.arcsec}, \
        load_SED_rest_properties = True, n_depth_reg = "auto"):
    
    for pc_err in min_flux_pc_errs:
        # make appropriate galfind catalogue creator for each aperture diameter
        cat_creator = GALFIND_Catalogue_Creator(cat_type, aper_diams[0], pc_err)
        for survey in surveys:
            start = time.time()
            cat = Catalogue.from_pipeline(survey = survey, version = version, instruments = instruments, aper_diams = aper_diams, \
                cat_creator = cat_creator, SED_fit_params_arr = SED_fit_params_arr, forced_phot_band = forced_phot_band, \
                excl_bands = excl_bands, loc_depth_min_flux_pc_errs = min_flux_pc_errs, crop_by = crop_by, load_SEDs = load_SEDs, \
                load_PDFs = load_PDFs, timed = timed, mask_stars = mask_stars, pix_scales = pix_scales, \
                load_SED_rest_properties = load_SED_rest_properties, n_depth_reg = n_depth_reg)
            breakpoint()
            pipes_origin = SED_fit_params_arr[-1] #["code"].label_from_SED_fit_params(SED_fit_params_arr[-1])
            #cat.plot("beta_C94", pipes_origin, "M_UV", pipes_origin)
            
            # M_UV_name = "M1500"
            # M_UV_bins = np.arange(-21.25, -17.25, 0.5)
            # UV_LF_z9 = Number_Density_Function.from_single_cat(cat, M_UV_name, M_UV_bins, \
            #     [8.5, 9.5], x_origin = "EAZY_fsps_larson_zfree_REST_PROPERTY")
            # UV_LF_z9.plot(x_lims = M_UV_name)
            # UV_LF_z10_5 = Number_Density_Function.from_single_cat(cat, M_UV_name, M_UV_bins, \
            #     [9.5, 11.5], x_origin = "EAZY_fsps_larson_zfree_REST_PROPERTY")
            # UV_LF_z10_5.plot(x_lims = M_UV_name)
            # UV_LF_z12_5 = Number_Density_Function.from_single_cat(cat, M_UV_name, M_UV_bins, \
            #     [11.5, 13.5], x_origin = "EAZY_fsps_larson_zfree_REST_PROPERTY")
            # UV_LF_z12_5.plot(x_lims = M_UV_name)
            # breakpoint()

            mass_name = "stellar_mass"
            mass_bins = np.arange(7.5, 11., 0.5)
            GSMF_z9 = Number_Density_Function.from_single_cat(cat, mass_name, mass_bins, \
                [8.5, 9.5], x_origin = pipes_origin)
            GSMF_z9.plot(x_lims = mass_name)
            GSMF_z10_5 = Number_Density_Function.from_single_cat(cat, mass_name, mass_bins, \
                [9.5, 11.5], x_origin = pipes_origin)
            GSMF_z10_5.plot(x_lims = mass_name)
            GSMF_z12_5 = Number_Density_Function.from_single_cat(cat, mass_name, mass_bins, \
                [11.5, 13.5], x_origin = pipes_origin)
            GSMF_z12_5.plot(x_lims = mass_name)

            #cat.calc_Vmax(cat.data, z_bin = [5.5, 6.5], timed = timed)
            #cat.calc_Vmax(cat.data, z_bin = [11.5, 13.5], timed = timed)
            #cat.calc_Vmax(cat.data, z_bin = [9.5, 11.5], timed = timed)
            #cat.calc_Vmax(cat.data, z_bin = [8.5, 9.5], timed = timed)
            #cat.calc_Vmax(cat.data, z_bin = [7.5, 8.5], timed = timed)
            #cat.calc_Vmax(cat.data, z_bin = [6.5, 7.5], timed = timed)
            
            # cat.phot_SNR_crop(0, 2., "non_detect") # 2σ non-detected in first band
            # cat.phot_bluewards_Lya_non_detect(2.) # 2σ non-detected in all bands bluewards of Lyα
            # cat.phot_redwards_Lya_detect([5., 5.], widebands_only = True) # 5σ/5σ detected in first/second band redwards of Lyα
            # cat.phot_redwards_Lya_detect(2., widebands_only = False) # 2σ detected in all bands redwards of Lyα
            # cat.select_chi_sq_lim(3., reduced = True) # χ^2_red < 3
            # cat.select_chi_sq_diff(4., delta_z_lowz = 0.5) # Δχ^2 > 4 between redshift free and low redshift SED fits, with Δz=0.5 tolerance 
            # cat.select_robust_zPDF(0.6, 0.1) # 60% of redshift PDF must lie within z ± z * 0.1
            # # ensure masked in all instruments
            # cat.select_unmasked_instrument(NIRCam()) # unmasked in all NIRCam bands
            # # hot pixel checks
            # for band_name in ["F277W", "F356W", "F444W"]:
            #     cat.select_band_flux_radius(band_name, "gtr", 1.5) # LW NIRCam wideband Re>1.5 pix
            
            # cat_copy = cat.select_EPOCHS(allow_lowz = False)
            # #cat_copy.make_cutouts(IDs = crop_by["IDs"])
            # cat_copy.plot_phot_diagnostics(flux_unit = u.ABmag)
            # print(str(cat_copy))

            # end = time.time()
            # print(f"Time to load catalogue = {(end - start):.1f}s")
            
            # #cat.data.calc_unmasked_area("NIRCam", forced_phot_band = forced_phot_band)
            # #cat.select_band_flux_radius("F277W", "gtr", 1.5)

            # # # print(cat.SED_rest_properties)
            # # for name in ["EW_obs_Halpha_cont_0.1", "flux_Halpha_cont_0.1_obs_M99_C00", "lum_Halpha_cont_0.1_obs_M99_C00", "xi_ion_Halpha_cont_0.1_M99_C00_fesc0.0"]:
            # #     # "EW_obs_Halpha_cont_0.0", "flux_Halpha_cont_0.0_obs_M99_C00", "lum_Halpha_cont_0.0_obs_M99_C00", \
            # #     # "continuum_Halpha+[NII]-6583", "EW_rest_Halpha_cont_0.0", "flux_Halpha_cont_0.0_rest_M99_C00", "lum_Halpha_cont_0.0_rest_M99_C00", 
            # #     cat.del_SED_rest_property(name)
            # #     print(f"deleted {name}")

            # # cat_copy = cat.select_phot_galaxy_property("z", ">", 4.5)
            # # cat_copy = cat.select_unmasked_instrument(NIRCam())
            # #cat_copy = cat.select_all_bands()
            # #cat_copy = cat.select_phot_galaxy_property("z", ">", 4.5)
            # #cat.select_rest_UV_line_emitters_sigma("CIV-1549", 2.)
            
            #cat.del_hdu(SED_fit_params_arr[-1]["code"].label_from_SED_fit_params(SED_fit_params_arr[-1]))
            # if load_SED_rest_properties:
            #     iters = 100
            #     cat.calc_SFR_UV_phot(frame = "obs", AUV_beta_conv_author_year = "M99", iters = iters)
                #cat.calc_SFR_UV_phot(frame = "obs", AUV_beta_conv_author_year = None, iters = iters)
                #cat.calc_rest_UV_properties(iters = iters)
                #cat.calc_cont_rest_optical(["Halpha"], iters = iters)
                #cat.calc_EW_rest_optical(["Halpha"], frame = "obs", iters = iters)
                #cat_copy.calc_xi_ion(iters = iters) #dust_author_year = None

def make_EAZY_SED_fit_params_arr(SED_code_arr, templates_arr, lowz_zmax_arr):
    return [{"code": code, "templates": templates, "lowz_zmax": lowz_zmax} \
        for code, templates, lowz_zmaxs in zip(SED_code_arr, templates_arr, lowz_zmax_arr) for lowz_zmax in lowz_zmaxs]

if __name__ == "__main__":

    version = "v11" #config["DEFAULT"]["VERSION"]
    instruments = ["ACS_WFC", "NIRCam"] #, "MIRI"] #, "ACS_WFC"] # "WFC3_IR"
    cat_type = "loc_depth"
    surveys = ["JOF"] #["JADES-Deep-GS+JEMS"]#+SMILES"] #[config["DEFAULT"]["SURVEY"]]
    aper_diams = [0.32] * u.arcsec # 0.32, 0.5, 1.0, 1.5, 2.0
    SED_code_arr = []#EAZY()]
    templates_arr = ["fsps_larson"] #["fsps", "fsps_larson", "fsps_jades"]
    lowz_zmax_arr = [[2., 4., 6., None]] #[[4., 6., None]] #[[None]] # 
    min_flux_pc_errs = [10]
    forced_phot_band = ["F277W", "F356W", "F444W"] # ["F444W"]
    crop_by = "EPOCHS" #{"ID": [893, 1685, 2171, 3400, 5532, 6492, 7389, 7540, 9036, 15476]} #"bands>13+EPOCHS" #"EPOCHS_lowz+z>4.5"
    timed = True
    mask_stars = {"ACS_WFC": False, "NIRCam": True, "WFC3_IR": False, "MIRI": False}
    MIRI_pix_scale = 0.06 * u.arcsec
    load_SED_rest_properties = False #True
    n_depth_reg = "auto"
    load_PDFs = {"EAZY": True, "Bagpipes": True}
    load_SEDs = {"EAZY": True, "Bagpipes": True}

    jems_bands = ["F182M", "F210M", "F430M", "F460M", "F480M"]
    ngdeep_excl_bands = ["F435W", "F775W", "F850LP"]
    #jades_3215_excl_bands = ["f162M", "f115W", "f150W", "f200W", "f410M", "f182M", "f210M", "f250M", "f300M", "f335M", "f277W", "f356W", "f444W"]
    excl_bands = []

    EAZY_SED_fit_params_arr = make_EAZY_SED_fit_params_arr(SED_code_arr, templates_arr, lowz_zmax_arr)
    pipes_fit_params_arr = [{"code": Bagpipes(), "dust": "Cal", "dust_prior": "log_10", \
        "metallicity_prior": "log_10", "sps_model": "BC03", "fix_z": False, \
        "z_range": (0., 25.), "sfh": "continuity_bursty"}]
    SED_fit_params_arr = EAZY_SED_fit_params_arr + pipes_fit_params_arr
    # delay_time = (8 * u.h).to(u.s).value
    # print(f"{surveys[0]} delayed by {delay_time}s")
    # time.sleep(delay_time)

    pix_scales = {**{"ACS_WFC": 0.03 * u.arcsec, "WFC3_IR": 0.03 * u.arcsec, "NIRCam": 0.03 * u.arcsec}, **{"MIRI": MIRI_pix_scale}}

    for survey in surveys:
        pipeline([survey], version, instruments, aper_diams, min_flux_pc_errs, forced_phot_band, \
            excl_bands, SED_fit_params_arr, cat_type = cat_type, crop_by = crop_by, load_PDFs = load_PDFs, \
            load_SEDs = load_SEDs, timed = timed, mask_stars = mask_stars, pix_scales = pix_scales, \
            load_SED_rest_properties = load_SED_rest_properties, n_depth_reg = n_depth_reg)

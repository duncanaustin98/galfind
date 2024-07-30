
import astropy.units as u
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from astropy.table import Table

from galfind import useful_funcs_austind as funcs
from galfind import Catalogue, config, LePhare, EAZY, NIRCam
from galfind.Catalogue_Creator import GALFIND_Catalogue_Creator

def Halpha_template_phot_color_diff(survey, version, instruments, aper_diams, min_pc_err = 10, forced_phot_band = ["F277W", "F356W", "F444W"], \
        excl_bands = [], SED_fit_params_arr = [], cat_type = "loc_depth", crop_by = None, timed = True, mask_stars = True, \
        pix_scales = {"ACS_WFC": 0.03 * u.arcsec, "WFC3_IR": 0.03 * u.arcsec, "NIRCam": 0.03 * u.arcsec, "MIRI": 0.09 * u.arcsec}, \
        load_SED_rest_properties = True, colour_arr = ["F410M", "F444W"]):
    # load catalogue
    cat_creator = GALFIND_Catalogue_Creator(cat_type, aper_diams[0], 10)
    cat = Catalogue.from_pipeline(survey = survey, version = version, instruments = instruments, aper_diams = aper_diams, \
        cat_creator = cat_creator, SED_fit_params_arr = SED_fit_params_arr, forced_phot_band = forced_phot_band, \
        excl_bands = excl_bands, loc_depth_min_flux_pc_errs = [min_pc_err], crop_by = crop_by, timed = timed, \
        mask_stars = mask_stars, pix_scales = pix_scales, load_SED_rest_properties = load_SED_rest_properties)
    # extract SEDs
    filters_colour = [band for band in NIRCam() if band.band_name in colour_arr]
    colours_arr = [gal.phot.SED_results["EAZY_fsps_larson_zfree"].SED.calc_colour(filters_colour).value for gal in tqdm(deepcopy(cat), desc = "Calculating template colours", total = len(cat))]
    colour_tab = Table({"IDs": cat.ID, "-".join(colour_arr): colours_arr}, dtype = [int, float])
    colour_tab.write(f"{survey}_{version}_{'+'.join(instruments)}_{'-'.join(colour_arr)}.fits")

def make_EAZY_SED_fit_params_arr(SED_code_arr, templates_arr, lowz_zmax_arr):
    return [{"code": code, "templates": templates, "lowz_zmax": lowz_zmax} \
        for code, templates, lowz_zmaxs in zip(SED_code_arr, templates_arr, lowz_zmax_arr) for lowz_zmax in lowz_zmaxs]

def main():
    version = "v11" #config["DEFAULT"]["VERSION"]
    instruments = ["ACS_WFC", "NIRCam"] #, "MIRI"] #, "ACS_WFC"] # "WFC3_IR"
    cat_type = "loc_depth"
    survey = "JOF" #["JADES-Deep-GS+JEMS+SMILES"] #[config["DEFAULT"]["SURVEY"]]
    aper_diams = [0.32] * u.arcsec # , 0.5, 1.0, 1.5, 2.0
    SED_code_arr = [EAZY()]
    templates_arr = ["fsps_larson"] #["fsps", "fsps_larson", "fsps_jades"]
    lowz_zmax_arr = [[None]] #[[4., 6., None]] #[[None]] # 
    min_flux_pc_errs = [10]
    forced_phot_band = ["F277W", "F356W", "F444W"] #["F444W"]
    crop_by = "EPOCHS" #{"ID": [1, 2, 3]} #"bands>13+EPOCHS" #"EPOCHS_lowz+z>4.5" # {"IDs": [30004, 26602, 2122, 28178, 17244, 23655, 1027]}
    timed = False
    mask_stars = {"ACS_WFC": False, "NIRCam": True, "WFC3_IR": False, "MIRI": False}
    MIRI_pix_scale = 0.06 * u.arcsec
    load_SED_rest_properties = True

    jems_bands = ["F182M", "F210M", "F430M", "F460M", "F480M"]
    ngdeep_excl_bands = ["F435W", "F775W", "F850LP"]
    #jades_3215_excl_bands = ["f162M", "f115W", "f150W", "f200W", "f410M", "f182M", "f210M", "f250M", "f300M", "f335M", "f277W", "f356W", "f444W"]
    excl_bands = []

    SED_fit_params_arr = make_EAZY_SED_fit_params_arr(SED_code_arr, templates_arr, lowz_zmax_arr)

    # delay_time = (8 * u.h).to(u.s).value
    # print(f"{surveys[0]} delayed by {delay_time}s")
    # time.sleep(delay_time)

    pix_scales = {**{"ACS_WFC": 0.03 * u.arcsec, "WFC3_IR": 0.03 * u.arcsec, "NIRCam": 0.03 * u.arcsec}, **{"MIRI": MIRI_pix_scale}}

    Halpha_template_phot_color_diff(survey, version, instruments, aper_diams, min_flux_pc_errs, forced_phot_band, \
        excl_bands, SED_fit_params_arr, cat_type = cat_type, crop_by = crop_by, timed = timed, \
        mask_stars = mask_stars, pix_scales = pix_scales, load_SED_rest_properties = load_SED_rest_properties)

if __name__ == "__main__":
    main()
    
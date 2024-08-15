

import astropy.units as u
import numpy as np
import time
from astropy.table import Table
import matplotlib.pyplot as plt

from galfind import useful_funcs_austind as funcs
from galfind import Catalogue, config, LePhare, EAZY, NIRCam, Number_Density_Function
from galfind.Catalogue_Creator import GALFIND_Catalogue_Creator

plt.style.use(f"{config['DEFAULT']['GALFIND_DIR']}/galfind_style.mplstyle")

def UVLF(surveys, version, instruments, aper_diams, pc_err, forced_phot_band, \
        excl_bands, SED_fit_params_arr, cat_type = "loc_depth", crop_by = None, timed = True, mask_stars = True, \
        pix_scales = {"ACS_WFC": 0.03 * u.arcsec, "WFC3_IR": 0.03 * u.arcsec, "NIRCam": 0.03 * u.arcsec, "MIRI": 0.09 * u.arcsec}, \
        load_SED_rest_properties = True, n_depth_reg = "auto"):
    # make appropriate galfind catalogue creator for each aperture diameter
    cat_creator = GALFIND_Catalogue_Creator(cat_type, aper_diams[0], pc_err)
    for survey in surveys:
        start = time.time()
        cat = Catalogue.from_pipeline(survey = survey, version = version, instruments = instruments, aper_diams = aper_diams, \
            cat_creator = cat_creator, SED_fit_params_arr = SED_fit_params_arr, forced_phot_band = forced_phot_band, \
            excl_bands = excl_bands, loc_depth_min_flux_pc_errs = [pc_err], crop_by = crop_by, timed = timed, \
            mask_stars = mask_stars, pix_scales = pix_scales, load_SED_rest_properties = load_SED_rest_properties, n_depth_reg = n_depth_reg)
        
        M_UV_name = "M1500"
        this_work_plot_kwargs = {"mfc": "gray", "marker": "D", "ms": 8., \
            "mew": 2., "mec": "black", "ecolor": "black", "elinewidth": 2.}

        UV_LF_z9 = Number_Density_Function.from_single_cat(cat, M_UV_name, np.arange(-21.25, -17.25, 0.5), \
            [8.5, 9.5], x_origin = "EAZY_fsps_larson_zfree_REST_PROPERTY")
        z9_author_years = {
            "Bouwens+21": {"z_ref": 9., "plot_kwargs": {}}, 
            "Harikane+22": {"z_ref": 9., "plot_kwargs": {}},
            "Finkelstein+23": {"z_ref": 9., "plot_kwargs": {}},
            "Leung+23": {"z_ref": 9., "plot_kwargs": {}},
            "Perez-Gonzalez+23": {"z_ref": 9., "plot_kwargs": {}},
            "Adams+24": {"z_ref": 9., "plot_kwargs": {}}
        } # also Finkelstein+22, but wide z bin used
        UV_LF_z9.plot(x_lims = M_UV_name, author_year_dict = z9_author_years, plot_kwargs = this_work_plot_kwargs)
        
        UV_LF_z10_5 = Number_Density_Function.from_single_cat(cat, M_UV_name, np.arange(-21.25, -17.25, 0.5), \
            [9.5, 11.5], x_origin = "EAZY_fsps_larson_zfree_REST_PROPERTY")
        z10_5_author_years = {
            "Oesch+18": {"z_ref": 10., "plot_kwargs": {}}, 
            "Bouwens+22": {"z_ref": 10., "plot_kwargs": {}},
            "Donnan+22": {"z_ref": 10., "plot_kwargs": {}},
            "Castellano+23": {"z_ref": 10., "plot_kwargs": {}},
            "Adams+24": {"z_ref": 10., "plot_kwargs": {}},
            "Finkelstein+22": {"z_ref": 11., "plot_kwargs": {"marker": "^"}},
            "Finkelstein+23": {"z_ref": 11., "plot_kwargs": {"marker": "^"}},
            "Leung+23": {"z_ref": 11., "plot_kwargs": {"marker": "^"}},
            "McLeod+23": {"z_ref": 11., "plot_kwargs": {"marker": "^"}},
            "Perez-Gonzalez+23": {"z_ref": 11., "plot_kwargs": {"marker": "^"}}
        }
        UV_LF_z10_5.plot(x_lims = M_UV_name, author_year_dict = z10_5_author_years, plot_kwargs = this_work_plot_kwargs)

        UV_LF_z12_5 = Number_Density_Function.from_single_cat(cat, M_UV_name, np.arange(-21.25, -17.25, 0.5), \
            [11.5, 13.5], x_origin = "EAZY_fsps_larson_zfree_REST_PROPERTY")
        z12_5_author_years = {
            "Bouwens+22": {"z_ref": 12., "plot_kwargs": {}},
            "Harikane+22": {"z_ref": 12., "plot_kwargs": {}},
            "Perez-Gonzalez+23": {"z_ref": 12., "plot_kwargs": {}},
            "Adams+24": {"z_ref": 12., "plot_kwargs": {}},
            "Robertson+24": {"z_ref": 12., "plot_kwargs": {}},
            "Donnan+22": {"z_ref": 13., "plot_kwargs": {"marker": "^"}}
        }
        UV_LF_z12_5.plot(x_lims = M_UV_name, author_year_dict = z12_5_author_years, plot_kwargs = this_work_plot_kwargs)

def compile_literature(z_ref, M_UV, phi, phi_l1, phi_u1, author_year):
    out_path = f"{config['NumberDensityFunctions']['UVLF_LIT_DIR']}/z={float(z_ref):.1f}/{author_year}.ecsv"
    funcs.make_dirs(out_path)
    tab = Table({"M_UV": M_UV, "phi": phi, "phi_l1": phi_l1, "phi_u1": phi_u1}, dtype = [float, float, float, float])
    tab.meta = {"author_year": author_year, "z_ref": z_ref}
    tab.write(out_path)

def make_EAZY_SED_fit_params_arr(SED_code_arr, templates_arr, lowz_zmax_arr):
    return [{"code": code, "templates": templates, "lowz_zmax": lowz_zmax} \
        for code, templates, lowz_zmaxs in zip(SED_code_arr, templates_arr, lowz_zmax_arr) for lowz_zmax in lowz_zmaxs]

def main():
    version = "v11"
    instruments = ["ACS_WFC", "NIRCam"]
    cat_type = "loc_depth"
    surveys = ["JOF"]
    aper_diams = [0.32] * u.arcsec
    SED_code_arr = [EAZY()]
    templates_arr = ["fsps_larson"]
    lowz_zmax_arr = [[4., 6., None]]
    min_flux_pc_errs = 10
    forced_phot_band = ["F277W", "F356W", "F444W"]
    crop_by = "EPOCHS"
    timed = False
    mask_stars = {"ACS_WFC": False, "NIRCam": True, "WFC3_IR": False, "MIRI": False}
    MIRI_pix_scale = 0.06 * u.arcsec
    load_SED_rest_properties = True
    n_depth_reg = "auto"

    jems_bands = ["F182M", "F210M", "F430M", "F460M", "F480M"]
    ngdeep_excl_bands = ["F435W", "F775W", "F850LP"]
    #jades_3215_excl_bands = ["f162M", "f115W", "f150W", "f200W", "f410M", "f182M", "f210M", "f250M", "f300M", "f335M", "f277W", "f356W", "f444W"]
    excl_bands = []

    SED_fit_params_arr = make_EAZY_SED_fit_params_arr(SED_code_arr, templates_arr, lowz_zmax_arr)

    # delay_time = (8 * u.h).to(u.s).value
    # print(f"{surveys[0]} delayed by {delay_time}s")
    # time.sleep(delay_time)

    pix_scales = {**{"ACS_WFC": 0.03 * u.arcsec, "WFC3_IR": 0.03 * u.arcsec, "NIRCam": 0.03 * u.arcsec}, **{"MIRI": MIRI_pix_scale}}

    for survey in surveys:
        UVLF([survey], version, instruments, aper_diams, min_flux_pc_errs, forced_phot_band, \
        excl_bands, SED_fit_params_arr, cat_type = cat_type, crop_by = crop_by, timed = timed, \
        mask_stars = mask_stars, pix_scales = pix_scales, load_SED_rest_properties = load_SED_rest_properties, n_depth_reg = n_depth_reg)

def convert_all_literature():
    #Bouwens et al 2022
    Bouwens22_8X = [-20.02,-18.77]
    Bouwens22_8Y = [0.000164,0.000378]
    Bouwens22_8Yerr = [0.000162,0.000306]
    compile_literature(8., Bouwens22_8X, Bouwens22_8Y, Bouwens22_8Yerr, Bouwens22_8Yerr, "Bouwens+22")

    Bouwens22_10X = [-18.65]
    Bouwens22_10Y = [0.000290]
    Bouwens22_10Yerr = [0.000238]
    compile_literature(10., Bouwens22_10X, Bouwens22_10Y, Bouwens22_10Yerr, Bouwens22_10Yerr, "Bouwens+22")

    Bouwens22_12X = [-20.31,-19.31]
    Bouwens22_12Y = [0.000116,0.000190]
    Bouwens22_12Yerr = [0.000094,0.000152]
    compile_literature(12., Bouwens22_12X, Bouwens22_12Y, Bouwens22_12Yerr, Bouwens22_12Yerr, "Bouwens+22")

    #Bouwens et al 2021

    Bouwens21_7X = [-22.19,-21.69,-21.19,-20.69,-20.19,-19.69,-19.19,-18.69,-17.94,-16.94]
    Bouwens21_7Y = [0.000001,0.000041,0.000047,0.000198,0.000283,0.000589,0.001172,0.001433,0.005760,0.008320]
    Bouwens21_7Yerr = [0.000002,0.000011,0.000015,0.000036,0.000066,0.000126,0.000336,0.000419,0.001440,0.002900]
    compile_literature(7., Bouwens21_7X, Bouwens21_7Y, Bouwens21_7Yerr, Bouwens21_7Yerr, "Bouwens+21")

    Bouwens21_8X = [-21.85,-21.35,-20.85,-20.10,-19.35,-18.60, -17.60]
    Bouwens21_8Y = [0.000003, 0.000012, 0.000041, 0.000120, 0.000657, 0.001100, 0.003020]
    Bouwens21_8Yerr = [0.000002,0.000004,0.000011,0.000040,0.000233,0.000340,0.001140]
    compile_literature(8., Bouwens21_8X, Bouwens21_8Y, Bouwens21_8Yerr, Bouwens21_8Yerr, "Bouwens+21")

    Bouwens21_9X = [-21.92, -21.12, -20.32, -19.12, -17.92]
    Bouwens21_9Y = [0.000001, 0.000007, 0.000026, 0.000187, 0.000923]
    Bouwens21_9Yerr = [0.000001,0.000003,0.000009,0.000150,0.000501]
    compile_literature(9., Bouwens21_9X, Bouwens21_9Y, Bouwens21_9Yerr, Bouwens21_9Yerr, "Bouwens+21")

    Oesch2018_10X = [-21.25,-20.25,-19.25,-18.25,-17.25]
    Oesch2018_10Y = [0.000001,0.000010,0.000034,0.000190,0.000630]
    Oesch2018_10Yerr = [0.000001,0.000005,0.000022,0.000120,0.000520]
    compile_literature(10., Oesch2018_10X, Oesch2018_10Y, Oesch2018_10Yerr, Oesch2018_10Yerr, "Oesch+18")

    #Donnan et al 2022

    Donnan22_8X = [-22.17, -21.42]
    Donnan22_8Y = [0.63* 1e-6,3.92* 1e-6]
    Donnan22_8YerrUp = [0.5* 1e-6,2.34* 1e-6]
    Donnan22_8YerrLo = [0.3* 1e-6,1.56* 1e-6] 
    compile_literature(8., Donnan22_8X, Donnan22_8Y, Donnan22_8YerrLo, Donnan22_8YerrUp, "Donnan+22")

    Donnan22_9X = [-22.3,-21.3,-18.5]
    Donnan22_9Y = [0.17* 1e-6,3.02* 1e-6,1200* 1e-6] 
    Donnan22_9YerrUp = [0.4* 1e-6,3.98* 1e-6,717* 1e-6] 
    Donnan22_9YerrLo = [0.14* 1e-6,1.5* 1e-6,476* 1e-6] 
    compile_literature(9., Donnan22_9X, Donnan22_9Y, Donnan22_9YerrLo, Donnan22_9YerrUp, "Donnan+22")

    Donnan22_10X = [-22.57, -20.10, -19.35,-18.85,-18.23]
    Donnan22_10Y = [0.18* 1e-6, 16.2* 1e-6, 136* 1e-6, 234.9* 1e-6, 630.8* 1e-6] 
    Donnan22_10YerrUp = [0.42* 1e-6, 21.4* 1e-6, 67.2* 1e-6, 107* 1e-6, 340* 1e-6] 
    Donnan22_10YerrLo = [0.15* 1e-6, 10.5* 1e-6, 47.1* 1e-6, 76.8* 1e-6, 233* 1e-6] 
    compile_literature(10., Donnan22_10X, Donnan22_10Y, Donnan22_10YerrLo, Donnan22_10YerrUp, "Donnan+22")

    Donnan22_13X = [-20.35, -19.00]
    Donnan22_13Y = [10.3* 1e-6, 27.4* 1e-6] 
    Donnan22_13YerrUp = [9.98* 1e-6,21.7* 1e-6] 
    Donnan22_13YerrLo = [5.59* 1e-6,13.1* 1e-6] 
    compile_literature(13., Donnan22_13X, Donnan22_13Y, Donnan22_13YerrLo, Donnan22_13YerrUp, "Donnan+22")

    #Harikane2022

    Harikane22_9X = [-21.03, -20.03, -19.03, -18.03]
    Harikane22_9Y = [4* 1e-5, 4.08* 1e-5, 22.4* 1e-5, 112* 1e-5] 
    Harikane22_9YerrUp = [9.42* 1e-5, 9.60* 1e-5, 18.7* 1e-5, 103* 1e-5] 
    Harikane22_9YerrLo = [3.85* 1e-5, 3.92* 1e-5, 14.6* 1e-5, 90* 1e-5]
    compile_literature(9., Harikane22_9X, Harikane22_9Y, Harikane22_9YerrLo, Harikane22_9YerrUp, "Harikane+22")

    Harikane22_12X = [-21.21,-20.21,-19.21,-18.21]
    Harikane22_12Y = [5* 1e-6, 13.1* 1e-6,24.0* 1e-6, 142* 1e-6] 
    Harikane22_12YerrUp = [11.56* 1e-6,17.5* 1e-6,23.8* 1e-6,197* 1e-6] 
    Harikane22_12YerrLo = [4.27* 1e-6,8.9* 1e-6,14.0* 1e-6,110* 1e-6]
    compile_literature(12., Harikane22_12X, Harikane22_12Y, Harikane22_12YerrLo, Harikane22_12YerrUp, "Harikane+22")

    Harikane22_16X = [-20.59]
    Harikane22_16Y = [6.62* 1e-6] 
    Harikane22_16YerrUp = [8.84* 1e-6] 
    Harikane22_16YerrLo = [4.49* 1e-6]
    compile_literature(16., Harikane22_16X, Harikane22_16Y, Harikane22_16YerrLo, Harikane22_16YerrUp, "Harikane+22")

    #Finkelstein2022c

    Fink22_11X = [-20.5, -19.8]
    Fink22_11Y = [4* 1e-5, 14.6* 1e-5] 
    Fink22_11YerrUp = [4.2* 1e-5, 8.2* 1e-5] 
    Fink22_11YerrLo = [2.5* 1e-5, 5.9* 1e-5] 
    compile_literature(11., Fink22_11X, Fink22_11Y, Fink22_11YerrLo, Fink22_11YerrUp, "Finkelstein+22")

    #Finkelstein2023

    Fink23_9X = [-22.0, -21.0, -20.5, -20.0, -19.5, -19.0]
    Fink23_9Y = [1.1* 1e-5, 2.2* 1e-5, 8.2*1e-5, 9.6* 1e-5, 28.6* 1e-5, 26.8* 1e-5] 
    Fink23_9YerrUp = [0.7* 1e-5, 1.3* 1e-5, 4.0* 1e-5, 4.6* 1e-5, 11.5* 1e-5, 12.4* 1e-5] 
    Fink23_9YerrLo = [0.6* 1e-5, 1.0* 1e-5, 3.2* 1e-5, 3.6* 1e-5, 9.1* 1e-5, 10.0* 1e-5] 
    compile_literature(9., Fink23_9X, Fink23_9Y, Fink23_9YerrLo, Fink23_9YerrUp, "Finkelstein+23")

    Fink23_11X = [-20.5, -20.0, -19.5]
    Fink23_11Y = [1.8* 1e-5, 5.4* 1e-5, 7.6* 1e-5] 
    Fink23_11YerrUp = [1.2* 1e-5, 2.7* 1e-5, 3.9* 1e-5] 
    Fink23_11YerrLo = [0.9* 1e-5, 2.1* 1e-5, 3.0* 1e-5]
    compile_literature(11., Fink23_11X, Fink23_11Y, Fink23_11YerrLo, Fink23_11YerrUp, "Finkelstein+23")

    Fink23_14X = [-20.0, -19.5]
    Fink23_14Y = [2.6* 1e-5, 5.3* 1e-5] 
    Fink23_14YerrUp = [3.3* 1e-5, 6.9* 1e-5] 
    Fink23_14YerrLo = [1.8* 1e-5, 4.4* 1e-5]
    compile_literature(14., Fink23_14X, Fink23_14Y, Fink23_14YerrLo, Fink23_14YerrUp, "Finkelstein+23")

    #Leung2023

    Leu23_9X = [-20.1, -19.1, -18.35, -17.85]
    Leu23_9Y = [14.7* 1e-5, 18.9* 1e-5, 74.0*1e-5, 170* 1e-5] 
    Leu23_9YerrUp = [11.1* 1e-5, 13.8* 1e-5, 41.4* 1e-5, 85* 1e-5] 
    Leu23_9YerrLo = [7.2* 1e-5, 8.9* 1e-5, 29* 1e-5, 65* 1e-5]
    compile_literature(9., Leu23_9X, Leu23_9Y, Leu23_9YerrLo, Leu23_9YerrUp, "Leung+23")

    Leu23_11X = [-19.35, -18.65, -17.95]
    Leu23_11Y = [18.5* 1e-5, 27.7* 1e-5, 59.1* 1e-5] 
    Leu23_11YerrUp = [11.9* 1e-5, 18.3* 1e-5, 41.9* 1e-5] 
    Leu23_11YerrLo = [8.3* 1e-5, 13* 1e-5, 29.3* 1e-5]
    compile_literature(11., Leu23_11X, Leu23_11Y, Leu23_11YerrLo, Leu23_11YerrUp, "Leung+23")

    #Bowler2020

    Bowler20_8X = [-21.65, -22.15, -22.90]
    Bowler20_8Y = [2.95* 1e-6, 0.58* 1e-6, 0.14* 1e-6]
    Bowler20_8Yerr = [0.98* 1e-6,0.33* 1e-6,0.06* 1e-6]
    compile_literature(8., Bowler20_8X, Bowler20_8Y, Bowler20_8Yerr, Bowler20_8Yerr, "Bowler+20")

    Bowler20_9X = [-21.9,-22.9]
    Bowler20_9Y = [0.84* 1e-6,0.16* 1e-6] 
    Bowler20_9Yerr = [0.49* 1e-6,0.11* 1e-6] 
    compile_literature(9., Bowler20_9X, Bowler20_9Y, Bowler20_9Yerr, Bowler20_9Yerr, "Bowler+20")

    #Stefanon2019

    Stefanon19_8X = [-22.55,-22.05,-21.55]
    Stefanon19_8Y = [0.76* 1e-6,1.38* 1e-6,4.87* 1e-6] 
    Stefanon19_8YerrUp = [0.74* 1e-6,1.09* 1e-6,2.01* 1e-6] 
    Stefanon19_8YerrLo = [0.41* 1e-6,0.66* 1e-6,1.41* 1e-6]
    compile_literature(8., Stefanon19_8X, Stefanon19_8Y, Stefanon19_8YerrLo, Stefanon19_8YerrUp, "Stefanon+19")

    Stefanon19_9X = [-22.35,-22.0,21.6,-21.2]
    Stefanon19_9Y = [0.43* 1e-6,0.43* 1e-6,1.14* 1e-6,1.64* 1e-6] 
    Stefanon19_9YerrUp = [0.99* 1e-6,0.98* 1e-6,1.5* 1e-6,2.16* 1e-6] 
    Stefanon19_9YerrLo = [0.36* 1e-6,0.36* 1e-6,0.73* 1e-6,1.06* 1e-6] 
    compile_literature(9., Stefanon19_9X, Stefanon19_9Y, Stefanon19_9YerrLo, Stefanon19_9YerrUp, "Stefanon+19")

    #Castellano+23

    Cast23_10X = [-21.5,-20.5,-19.5]
    Cast23_10Y = [2.1e-5,7.6e-5,18.0e-5] 
    Cast23_10YerrUp = [4.8e-5,6.4e-5,17.5e-5] 
    Cast23_10YerrLo = [1.7e-5,3.9e-5,9.8e-5] 
    compile_literature(10., Cast23_10X, Cast23_10Y, Cast23_10YerrLo, Cast23_10YerrUp, "Castellano+23")

    #McLure2013

    Mclure13_7X = [-21.0,-20.5,-20.0,-19.5,-19.0,-18.5,-18.0,-17.5,-17.0]
    Mclure13_7Y = [0.00003,0.00012,0.00033,0.00075,0.0011,0.0021,0.0042,0.0079,0.011]
    Mclure13_7Yerr = [0.00001,0.00002,0.00005,0.00009,0.0002,0.0006,0.0009,0.0019,0.0025]
    compile_literature(7., Mclure13_7X, Mclure13_7Y, Mclure13_7Yerr, Mclure13_7Yerr, "McLure+13")

    Mclure13_8X = [-21.25,-20.75,-20.25,-19.75,-19.25,-18.75,-18.25,-17.75,-17.25]
    Mclure13_8Y = [0.000008,0.00003,0.0001,0.0003,0.0005,0.0012,0.0018,0.0028,0.0050]
    Mclure13_8Yerr = [0.000003,0.000009,0.00003,0.00006,0.00012,0.0004,0.0006,0.0008,0.0025]
    compile_literature(8., Mclure13_8X, Mclure13_8Y, Mclure13_8Yerr, Mclure13_8Yerr, "McLure+13")

    #McLeod2023

    McL23_11X = [-21.80,-20.8,-20.05,-19.55]
    McL23_11Y = [0.129e-5,1.254e-5,3.974e-5,9.863e-5]
    McL23_11Yerr = [0.128e-5,0.428e-5,1.340e-5,4.197e-5]
    compile_literature(11., McL23_11X, McL23_11Y, McL23_11Yerr, McL23_11Yerr, "McLeod+23")

    McL23_14X = [-19.45,-18.95]
    McL23_14Y = [2.469e-5,6.199e-5]
    McL23_14Yerr = [1.659e-5,3.974e-5]
    compile_literature(14., McL23_14X, McL23_14Y, McL23_14Yerr, McL23_14Yerr, "McLeod+23")

    #Finkelstein2015

    Fink15_7X = [-22, -21.5, -21.0, -20.5, -20, -19.5, -19, -18.5, -18]
    Fink15_7Y = [46* 1e-7, 187* 1e-7, 690* 1e-7, 1301* 1e-7, 2742* 1e-7, 3848* 1e-7, 5699* 1e-7, 25650* 1e-7, 30780* 1e-7]
    Fink15_7YerrUp = [49* 1e-7, 85* 1e-7, 156* 1e-7, 239* 1e-7, 379* 1e-7, 633* 1e-7, 2229* 1e-7, 8735* 1e-7, 10837* 1e-7]
    Fink15_7YerrLo = [28* 1e-7, 67* 1e-7, 144* 1e-7, 200* 1e-7, 329* 1e-7, 586* 1e-7, 1817* 1e-7, 7161* 1e-7, 8845* 1e-7]
    compile_literature(7., Fink15_7X, Fink15_7Y, Fink15_7YerrLo, Fink15_7YerrUp, "Finkelstein+15")

    Fink15_8X = [-21.5, -21.0, -20.5, -20, -19.5, -19, -18.5]
    Fink15_8Y = [79* 1e-7, 150* 1e-7, 615* 1e-7, 1097* 1e-7, 2174* 1e-7, 6073* 1e-7, 15110* 1e-7]
    Fink15_8YerrUp = [68* 1e-7, 94* 1e-7, 197* 1e-7, 356* 1e-7, 1805* 1e-7, 3501* 1e-7, 10726* 1e-7]
    Fink15_8YerrLo = [46* 1e-7, 70* 1e-7, 165* 1e-7, 309* 1e-7, 1250* 1e-7, 2616* 1e-7, 7718* 1e-7]
    compile_literature(8., Fink15_8X, Fink15_8Y, Fink15_8YerrLo, Fink15_8YerrUp, "Finkelstein+15")

    #Perez-Gonzalez 2023

    PG23_9X = [-19.5,-18.5,-17.5,-16.5]
    PG23_9Y = [np.power(10,-4.17),np.power(10,-3.53),np.power(10,-2.77),np.power(10,-2.3)]
    PG23_9YerrUp = [np.power(10,-4.17+0.25) - np.power(10,-4.17),np.power(10,-3.53+0.14) -np.power(10,-3.53),np.power(10,-2.77+0.11) -np.power(10,-2.77),np.power(10,-2.3+0.16) -np.power(10,-2.3)]
    PG23_9YerrLo = [np.power(10,-4.17) - np.power(10,-4.17-0.61),np.power(10,-3.53) -np.power(10,-3.53-0.2),np.power(10,-2.77) -np.power(10,-2.77-0.15),np.power(10,-2.3) -np.power(10,-2.3-0.27)]
    compile_literature(9., PG23_9X, PG23_9Y, PG23_9YerrLo, PG23_9YerrUp, "Perez-Gonzalez+23")

    PG23_11X = [-18.5,-17.5]
    PG23_11Y = [np.power(10,-3.57),np.power(10,-2.82)]
    PG23_11YerrUp = [np.power(10,-3.57+0.18) - np.power(10,-3.57),np.power(10,-2.82+0.20) -np.power(10,-2.82)]
    PG23_11YerrLo = [np.power(10,-3.57) - np.power(10,-3.57-0.32),np.power(10,-2.82) -np.power(10,-2.82-0.34)]
    compile_literature(11., PG23_11X, PG23_11Y, PG23_11YerrLo, PG23_11YerrUp, "Perez-Gonzalez+23")

    PG23_12X = [-18.3,-17.3]
    PG23_12YerrUp = [np.power(10,-3.47+0.20) - np.power(10,-3.47),np.power(10,-2.87+0.23) -np.power(10,-2.87)]
    PG23_12YerrLo = [np.power(10,-3.47) - np.power(10,-3.47-0.40),np.power(10,-2.87) -np.power(10,-2.87-0.50)]
    PG23_12Y = [np.power(10,-3.47),np.power(10,-2.87)]
    compile_literature(12., PG23_12X, PG23_12Y, PG23_12YerrLo, PG23_12YerrUp, "Perez-Gonzalez+23")

    #Robertson+24

    Rob24_12X = [-18.5,-18,-17.6]
    Rob24_12Y = [1.22e-4,3.2e-4,1.54e-4]
    Rob24_12Yerr = [0.94e-4,2.46e-4,1.18e-4]
    compile_literature(12., Rob24_12X, Rob24_12Y, Rob24_12Yerr, Rob24_12Yerr, "Robertson+24")

    #Adams+24

    Ada24_9X = [-22.05,-21.55,-21.05,-20.55,-20.05, -19.55,-18.8]
    Ada24_9Y = [0.628e-5,0.628e-5,1.257e-5,6.427e-5,10.76e-5, 18.22e-5, 42.45e-5]
    Ada24_9Yerr = [0.536e-5,0.536e-5,0.851e-5, 2.534e-5, 5.825e-5, 9.442e-5, 20.32e-5]
    compile_literature(9., Ada24_9X, Ada24_9Y, Ada24_9Yerr, Ada24_9Yerr, "Adams+24")

    Ada24_10X = [-20.95,-20.45,-19.95, -19.45,-18.7]
    Ada24_10Y = [0.721e-5,1.855e-5,3.331e-5,6.674e-5,9.996e-5]
    Ada24_10Yerr = [0.48e-5,0.888e-5,1.314e-5, 3.960e-5, 6.115e-5]
    compile_literature(10., Ada24_10X, Ada24_10Y, Ada24_10Yerr, Ada24_10Yerr, "Adams+24")

    Ada24_12X = [-20.75,-20.25,-19.5]
    Ada24_12Y = [0.852e-5,2.148e-5,5.923e-5]
    Ada24_12Yerr = [0.570e-5,1.375e-5,3.883e-5]
    compile_literature(12., Ada24_12X, Ada24_12Y, Ada24_12Yerr, Ada24_12Yerr, "Adams+24")

if __name__ == "__main__":
    main()
    #convert_all_literature()
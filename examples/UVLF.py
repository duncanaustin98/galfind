import sys
import time

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from galfind import (
    EAZY,
    Catalogue,
    Number_Density_Function,
    config,
    galfind_logger,
)
from galfind import useful_funcs_austind as funcs
from galfind.Catalogue_Creator import GALFIND_Catalogue_Creator

sys.path.insert(1, config["NumberDensityFunctions"]["FLAGS_DATA_DIR"])
try:
    pass
except:
    galfind_logger.critical(
        "Could not import flags_data.distribution_functions"
    )

plt.style.use(f"{config['DEFAULT']['GALFIND_DIR']}/galfind_style.mplstyle")


def conv_flags_UVLF(name, type=np.float64):
    path = f"{config['NumberDensityFunctions']['FLAGS_DATA_DIR']}/flags_data/data/DistributionFunctions/LUV/obs/binned/{name}.ecsv"
    meta = Table.read(path).meta
    tab = {}
    tab["z"] = np.array([9.0, 9.0, 9.0, 9.0, 12.0, 12.0])
    tab["M"] = (
        np.array([-20.34, -19.34, -18.34, -17.34, -18.9, -17.9]).astype(type)
        * u.mag
    )
    tab["log10phi"] = u.Dex(
        10 ** np.array([-4.4, -3.87, -2.82, -3.08, -3.78, -3.7]).astype(type)
        * (u.mag * u.Mpc**3) ** -1
    )
    tab["log10phi_err_low"] = u.Dex(
        np.array([0.94, 0.94, 0.5, 1.64, 0.94, 1.64]).astype(type)
    )  # * (u.mag * u.Mpc ** 3) ** -1
    tab["log10phi_err_upp"] = u.Dex(
        np.array([0.57, 0.57, 0.37, 0.75, 0.57, 0.75]).astype(type)
    )  # * (u.mag * u.Mpc ** 3) ** -1
    meta["references"] = ["2023ApJ...946L..35M"]
    meta["name"] = "Morishita et al. 2023"
    meta["y"] = "log10phi"
    # breakpoint()
    tab_ = Table(tab, dtype=[float, float, float, float, float])
    tab_.meta = meta
    tab_.write(path, overwrite=True)


def UVLF(
    surveys,
    version,
    instruments,
    aper_diams,
    pc_err,
    forced_phot_band,
    excl_bands,
    SED_fit_params_arr,
    cat_type="loc_depth",
    crop_by=None,
    timed=True,
    mask_stars=True,
    pix_scales={
        "ACS_WFC": 0.03 * u.arcsec,
        "WFC3_IR": 0.03 * u.arcsec,
        "NIRCam": 0.03 * u.arcsec,
        "MIRI": 0.09 * u.arcsec,
    },
    load_SED_rest_properties=True,
    n_depth_reg="auto",
):
    # make appropriate galfind catalogue creator for each aperture diameter
    cat_creator = GALFIND_Catalogue_Creator(cat_type, aper_diams[0], pc_err)
    for survey in surveys:
        start = time.time()
        cat = Catalogue.from_pipeline(
            survey=survey,
            version=version,
            instruments=instruments,
            aper_diams=aper_diams,
            cat_creator=cat_creator,
            SED_fit_params_arr=SED_fit_params_arr,
            forced_phot_band=forced_phot_band,
            excl_bands=excl_bands,
            loc_depth_min_flux_pc_errs=[pc_err],
            crop_by=crop_by,
            timed=timed,
            mask_stars=mask_stars,
            pix_scales=pix_scales,
            load_SED_rest_properties=load_SED_rest_properties,
            n_depth_reg=n_depth_reg,
        )

        M_UV_name = "M1500"
        this_work_plot_kwargs = {
            "mfc": "gray",
            "marker": "D",
            "ms": 8.0,
            "mew": 2.0,
            "mec": "black",
            "ecolor": "black",
            "elinewidth": 2.0,
        }

        UV_LF_z9 = Number_Density_Function.from_single_cat(
            cat,
            M_UV_name,
            np.arange(-21.25, -17.25, 0.5),
            [8.5, 9.5],
            x_origin="EAZY_fsps_larson_zfree_REST_PROPERTY",
        )
        z9_author_years = {
            "Bouwens+21": {"z_ref": 9.0, "plot_kwargs": {}},
            "Harikane+22": {"z_ref": 9.0, "plot_kwargs": {}},
            "Finkelstein+23": {"z_ref": 9.0, "plot_kwargs": {}},
            "Leung+23": {"z_ref": 9.0, "plot_kwargs": {}},
            "Perez-Gonzalez+23": {"z_ref": 9.0, "plot_kwargs": {}},
            "Adams+24": {"z_ref": 9.0, "plot_kwargs": {}},
        }  # also Finkelstein+22, but wide z bin used
        UV_LF_z9.plot(
            x_lims=M_UV_name,
            author_year_dict=z9_author_years,
            plot_kwargs=this_work_plot_kwargs,
        )

        UV_LF_z10_5 = Number_Density_Function.from_single_cat(
            cat,
            M_UV_name,
            np.arange(-21.25, -17.25, 0.5),
            [9.5, 11.5],
            x_origin="EAZY_fsps_larson_zfree_REST_PROPERTY",
        )
        z10_5_author_years = {
            "Oesch+18": {"z_ref": 10.0, "plot_kwargs": {}},
            "Bouwens+22": {"z_ref": 10.0, "plot_kwargs": {}},
            "Donnan+22": {"z_ref": 10.0, "plot_kwargs": {}},
            "Castellano+23": {"z_ref": 10.0, "plot_kwargs": {}},
            "Adams+24": {"z_ref": 10.0, "plot_kwargs": {}},
            "Finkelstein+22": {"z_ref": 11.0, "plot_kwargs": {"marker": "^"}},
            "Finkelstein+23": {"z_ref": 11.0, "plot_kwargs": {"marker": "^"}},
            "Leung+23": {"z_ref": 11.0, "plot_kwargs": {"marker": "^"}},
            "McLeod+23": {"z_ref": 11.0, "plot_kwargs": {"marker": "^"}},
            "Perez-Gonzalez+23": {
                "z_ref": 11.0,
                "plot_kwargs": {"marker": "^"},
            },
        }
        UV_LF_z10_5.plot(
            x_lims=M_UV_name,
            author_year_dict=z10_5_author_years,
            plot_kwargs=this_work_plot_kwargs,
        )

        UV_LF_z12_5 = Number_Density_Function.from_single_cat(
            cat,
            M_UV_name,
            np.arange(-21.25, -17.25, 0.5),
            [11.5, 13.5],
            x_origin="EAZY_fsps_larson_zfree_REST_PROPERTY",
        )
        z12_5_author_years = {
            "Bouwens+22": {"z_ref": 12.0, "plot_kwargs": {}},
            "Harikane+22": {"z_ref": 12.0, "plot_kwargs": {}},
            "Perez-Gonzalez+23": {"z_ref": 12.0, "plot_kwargs": {}},
            "Adams+24": {"z_ref": 12.0, "plot_kwargs": {}},
            "Robertson+24": {"z_ref": 12.0, "plot_kwargs": {}},
            "Donnan+22": {"z_ref": 13.0, "plot_kwargs": {"marker": "^"}},
        }
        UV_LF_z12_5.plot(
            x_lims=M_UV_name,
            author_year_dict=z12_5_author_years,
            plot_kwargs=this_work_plot_kwargs,
        )


def compile_literature(
    z_arr,
    M_UV,
    phi,
    phi_l1,
    phi_u1,
    log_phi,
    name,
    meta,
    Ngals=None,
    dM_arr=None,
):
    out_path = f"{config['NumberDensityFunctions']['FLAGS_DATA_DIR']}/flags_data/data/DistributionFunctions/LUV/obs/binned/{name}_new.ecsv"
    # out_path = f"{config['NumberDensityFunctions']['UVLF_LIT_DIR']}/z={float(z_ref):.1f}/{author_year}.ecsv"
    funcs.make_dirs(out_path)
    input_dict = {"z": z_arr, "M": M_UV}
    if not log_phi:
        input_dict["phi"] = phi
        input_dict["phi_err_low"] = phi_l1
        input_dict["phi_err_upp"] = phi_u1
        phi_meta = "phi"
    else:
        phi_meta = "log10phi"
    dtypes = [float, float, float, float, float]
    if type(Ngals) != type(None):
        input_dict["N"] = Ngals
        dtypes += [int]
    if type(dM_arr) != type(None):
        input_dict["deltaM"] = dM_arr
        dtypes += [float]
    tab = Table(input_dict, dtype=dtypes)
    out_meta = {"x": "M", "y": phi_meta}
    out_meta["type"] = "binned"
    tab.meta = {**out_meta, **meta}
    tab.write(out_path, overwrite=True)


def make_EAZY_SED_fit_params_arr(SED_code_arr, templates_arr, lowz_zmax_arr):
    return [
        {"code": code, "templates": templates, "lowz_zmax": lowz_zmax}
        for code, templates, lowz_zmaxs in zip(
            SED_code_arr, templates_arr, lowz_zmax_arr
        )
        for lowz_zmax in lowz_zmaxs
    ]


def main():
    version = "v11"
    instruments = ["ACS_WFC", "NIRCam"]
    cat_type = "loc_depth"
    surveys = ["JOF"]
    aper_diams = [0.32] * u.arcsec
    SED_code_arr = [EAZY()]
    templates_arr = ["fsps_larson"]
    lowz_zmax_arr = [[4.0, 6.0, None]]
    min_flux_pc_errs = 10
    forced_phot_band = ["F277W", "F356W", "F444W"]
    crop_by = "EPOCHS"
    timed = False
    mask_stars = {
        "ACS_WFC": False,
        "NIRCam": True,
        "WFC3_IR": False,
        "MIRI": False,
    }
    MIRI_pix_scale = 0.06 * u.arcsec
    load_SED_rest_properties = True
    n_depth_reg = "auto"

    jems_bands = ["F182M", "F210M", "F430M", "F460M", "F480M"]
    ngdeep_excl_bands = ["F435W", "F775W", "F850LP"]
    # jades_3215_excl_bands = ["f162M", "f115W", "f150W", "f200W", "f410M", "f182M", "f210M", "f250M", "f300M", "f335M", "f277W", "f356W", "f444W"]
    excl_bands = []

    SED_fit_params_arr = make_EAZY_SED_fit_params_arr(
        SED_code_arr, templates_arr, lowz_zmax_arr
    )

    # delay_time = (8 * u.h).to(u.s).value
    # print(f"{surveys[0]} delayed by {delay_time}s")
    # time.sleep(delay_time)

    pix_scales = {
        **{
            "ACS_WFC": 0.03 * u.arcsec,
            "WFC3_IR": 0.03 * u.arcsec,
            "NIRCam": 0.03 * u.arcsec,
        },
        **{"MIRI": MIRI_pix_scale},
    }

    for survey in surveys:
        UVLF(
            [survey],
            version,
            instruments,
            aper_diams,
            min_flux_pc_errs,
            forced_phot_band,
            excl_bands,
            SED_fit_params_arr,
            cat_type=cat_type,
            crop_by=crop_by,
            timed=timed,
            mask_stars=mask_stars,
            pix_scales=pix_scales,
            load_SED_rest_properties=load_SED_rest_properties,
            n_depth_reg=n_depth_reg,
        )


def convert_all_literature():
    # # Oesch 2018

    # Oesch2018_z_arr = [10.] * 5
    # Oesch2018_10X = np.array([-21.25,-20.25,-19.25,-18.25,-17.25]) * u.mag
    # Oesch2018_10Y = np.array([0.01e-4, 0.1e-4, 0.34e-4, 1.9e-4, 6.3e-4]) * (u.mag * u.Mpc ** 3) ** -1
    # Oesch2018_10Yl1 = np.array([0.008e-4, 0.05e-4, 0.22e-4, 1.2e-4, 5.2e-4]) * (u.mag * u.Mpc ** 3) ** -1
    # Oesch2018_10Yu1 = np.array([0.022e-4, 0.1e-4, 0.45e-4, 2.5e-4, 14.9e-4]) * (u.mag * u.Mpc ** 3) ** -1

    # year = 2018
    # meta = {"redshifts": [10.], "name": f"Oesch et al. ({str(year)})", \
    #      "year": year, "observatory": ["Hubble"], "references": ["2018ApJ...855..105O"]}

    # compile_literature(Oesch2018_z_arr, Oesch2018_10X, Oesch2018_10Y, Oesch2018_10Yl1, Oesch2018_10Yu1, False, "oesch18", meta)

    # # # #Finkelstein2023

    # Fink23_z_arr = [9.] * 6 + [11.] * 3 + [14.] * 2
    # Ngals = [1, 2, 7, 7, 17, 10, 3, 8, 8, 1, 2]

    # Fink23_9X = [-22.0, -21.0, -20.5, -20.0, -19.5, -19.0]
    # Fink23_9Y = [1.1e-5, 2.2e-5, 8.2e-5, 9.6e-5, 28.6e-5, 26.8e-5]
    # Fink23_9YerrUp = [0.7e-5, 1.3e-5, 4.0e-5, 4.6e-5, 11.5e-5, 12.4e-5]
    # Fink23_9YerrLo = [0.6e-5, 1.0e-5, 3.2e-5, 3.6e-5, 9.1e-5, 10.0e-5]

    # Fink23_11X = [-20.5, -20.0, -19.5]
    # Fink23_11Y = [1.8e-5, 5.4e-5, 7.6e-5]
    # Fink23_11YerrUp = [1.2e-5, 2.7e-5, 3.9e-5]
    # Fink23_11YerrLo = [0.9e-5, 2.1e-5, 3.0e-5]

    # Fink23_14X = [-20.0, -19.5]
    # Fink23_14Y = [2.6e-5, 5.3e-5]
    # Fink23_14YerrUp = [3.3e-5, 6.9e-5]
    # Fink23_14YerrLo = [1.8e-5, 4.4e-5]

    # Fink23_M_UV_arr = np.array(Fink23_9X + Fink23_11X + Fink23_14X) * u.mag
    # Fink23_phi_arr = np.array(Fink23_9Y + Fink23_11Y + Fink23_14Y) * (u.mag * u.Mpc ** 3) ** -1
    # Fink23_phi_l1_arr = np.array(Fink23_9YerrLo + Fink23_11YerrLo + Fink23_14YerrLo) * (u.mag * u.Mpc ** 3) ** -1
    # Fink23_phi_u1_arr = np.array(Fink23_9YerrUp + Fink23_11YerrUp + Fink23_14YerrUp) * (u.mag * u.Mpc ** 3) ** -1

    # year = 2024
    # meta = {"redshifts": [9., 11., 14.], "name": f"Finkelstein et al. ({str(year)})", \
    #      "year": year, "observatory": ["Hubble", "Webb"], "references": ["2024ApJ...969L...2F"]}

    # compile_literature(Fink23_z_arr, Fink23_M_UV_arr, Fink23_phi_arr, Fink23_phi_l1_arr, Fink23_phi_u1_arr, False, "finkelstein24", meta, Ngals)

    # #Stefanon2019

    # Stefanon19_z_arr = [8.] * 3 + [9.] * 4
    # Stefanon19_8X = [-22.55,-22.05,-21.55]
    # Stefanon19_8Y = [0.76e-6, 1.38e-6, 4.87e-6]
    # Stefanon19_8YerrUp = [0.74e-6, 1.09e-6, 2.01e-6]
    # Stefanon19_8YerrLo = [0.41e-6, 0.66e-6, 1.41e-6]

    # Stefanon19_9X = [-22.35,-22.0,21.6,-21.2]
    # Stefanon19_9Y = [0.43e-6,0.43e-6,1.14e-6,1.64e-6]
    # Stefanon19_9YerrUp = [0.99e-6,0.98e-6,1.5e-6,2.16e-6]
    # Stefanon19_9YerrLo = [0.36e-6,0.36e-6,0.73e-6,1.06e-6]

    # Stefanon19_M_UV_arr = np.array(Stefanon19_8X + Stefanon19_9X) * u.mag
    # Stefanon19_phi_arr = np.array(Stefanon19_8Y + Stefanon19_9Y) * (u.mag * u.Mpc ** 3) ** -1
    # Stefanon19_phi_l1_arr = np.array(Stefanon19_8YerrLo + Stefanon19_9YerrLo) * (u.mag * u.Mpc ** 3) ** -1
    # Stefanon19_phi_u1_arr = np.array(Stefanon19_8YerrUp + Stefanon19_9YerrUp) * (u.mag * u.Mpc ** 3) ** -1

    # year = 2019
    # meta = {"redshifts": [8., 9.], "name": f"Stefanon et al. ({str(year)})", \
    #      "year": year, "observatory": ["VISTA", "Hubble", "Spitzer"], "references": ["2019ApJ...883...99S"]}

    # compile_literature(Stefanon19_z_arr, Stefanon19_M_UV_arr, Stefanon19_phi_arr, Stefanon19_phi_l1_arr, Stefanon19_phi_u1_arr, False, "stefanon19", meta)

    # #Castellano+23

    # Cast23_z_arr = [10.] * 3
    # Cast23_dM_arr = np.array([0.5] * 3) * u.mag
    # Cast23_10X = np.array([-21.5,-20.5,-19.5]) * u.mag
    # Cast23_10Y = np.array([2.1e-5,7.6e-5,18.0e-5]) * (u.mag * u.Mpc ** 3) ** -1
    # Cast23_10YerrUp = np.array([4.8e-5,6.4e-5,17.5e-5]) * (u.mag * u.Mpc ** 3) ** -1
    # Cast23_10YerrLo = np.array([1.7e-5,3.9e-5,9.8e-5]) * (u.mag * u.Mpc ** 3) ** -1

    # year = 2023
    # meta = {"redshifts": [10.], "name": f"Castellano et al. ({str(year)})", \
    #      "year": year, "observatory": ["Webb"], "references": ["2023ApJ...948L..14C"]}
    # compile_literature(Cast23_z_arr, Cast23_10X, Cast23_10Y, Cast23_10YerrLo, Cast23_10YerrUp, \
    #     False, "castellano23", meta, dM_arr = Cast23_dM_arr)

    # #McLure2013

    # Mclure13_7X = [-21.0,-20.5,-20.0,-19.5,-19.0,-18.5,-18.0,-17.5,-17.0]
    # Mclure13_7Y = [0.00003,0.00012,0.00033,0.00075,0.0011,0.0021,0.0042,0.0079,0.011]
    # Mclure13_7Yerr = [0.00001,0.00002,0.00005,0.00009,0.0002,0.0006,0.0009,0.0019,0.0025]
    # compile_literature(7., Mclure13_7X, Mclure13_7Y, Mclure13_7Yerr, Mclure13_7Yerr, "McLure+13")

    # Mclure13_8X = [-21.25,-20.75,-20.25,-19.75,-19.25,-18.75,-18.25,-17.75,-17.25]
    # Mclure13_8Y = [0.000008,0.00003,0.0001,0.0003,0.0005,0.0012,0.0018,0.0028,0.0050]
    # Mclure13_8Yerr = [0.000003,0.000009,0.00003,0.00006,0.00012,0.0004,0.0006,0.0008,0.0025]
    # compile_literature(8., Mclure13_8X, Mclure13_8Y, Mclure13_8Yerr, Mclure13_8Yerr, "McLure+13")

    # #McLeod2023

    # McL23_11X = [-21.80,-20.8,-20.05,-19.55]
    # McL23_11Y = [0.129e-5,1.254e-5,3.974e-5,9.863e-5]
    # McL23_11Yerr = [0.128e-5,0.428e-5,1.340e-5,4.197e-5]
    # compile_literature(11., McL23_11X, McL23_11Y, McL23_11Yerr, McL23_11Yerr, "McLeod+23")

    # McL23_14X = [-19.45,-18.95]
    # McL23_14Y = [2.469e-5,6.199e-5]
    # McL23_14Yerr = [1.659e-5,3.974e-5]
    # compile_literature(14., McL23_14X, McL23_14Y, McL23_14Yerr, McL23_14Yerr, "McLeod+23")

    # #Finkelstein2015

    # Fink15_7X = [-22, -21.5, -21.0, -20.5, -20, -19.5, -19, -18.5, -18]
    # Fink15_7Y = [46* 1e-7, 187* 1e-7, 690* 1e-7, 1301* 1e-7, 2742* 1e-7, 3848* 1e-7, 5699* 1e-7, 25650* 1e-7, 30780* 1e-7]
    # Fink15_7YerrUp = [49* 1e-7, 85* 1e-7, 156* 1e-7, 239* 1e-7, 379* 1e-7, 633* 1e-7, 2229* 1e-7, 8735* 1e-7, 10837* 1e-7]
    # Fink15_7YerrLo = [28* 1e-7, 67* 1e-7, 144* 1e-7, 200* 1e-7, 329* 1e-7, 586* 1e-7, 1817* 1e-7, 7161* 1e-7, 8845* 1e-7]
    # compile_literature(7., Fink15_7X, Fink15_7Y, Fink15_7YerrLo, Fink15_7YerrUp, "Finkelstein+15")

    # Fink15_8X = [-21.5, -21.0, -20.5, -20, -19.5, -19, -18.5]
    # Fink15_8Y = [79* 1e-7, 150* 1e-7, 615* 1e-7, 1097* 1e-7, 2174* 1e-7, 6073* 1e-7, 15110* 1e-7]
    # Fink15_8YerrUp = [68* 1e-7, 94* 1e-7, 197* 1e-7, 356* 1e-7, 1805* 1e-7, 3501* 1e-7, 10726* 1e-7]
    # Fink15_8YerrLo = [46* 1e-7, 70* 1e-7, 165* 1e-7, 309* 1e-7, 1250* 1e-7, 2616* 1e-7, 7718* 1e-7]
    # compile_literature(8., Fink15_8X, Fink15_8Y, Fink15_8YerrLo, Fink15_8YerrUp, "Finkelstein+15")

    # #Perez-Gonzalez 2023

    # PG23_9X = [-19.5,-18.5,-17.5,-16.5]
    # PG23_9Y = [np.power(10,-4.17),np.power(10,-3.53),np.power(10,-2.77),np.power(10,-2.3)]
    # PG23_9YerrUp = [np.power(10,-4.17+0.25) - np.power(10,-4.17),np.power(10,-3.53+0.14) -np.power(10,-3.53),np.power(10,-2.77+0.11) -np.power(10,-2.77),np.power(10,-2.3+0.16) -np.power(10,-2.3)]
    # PG23_9YerrLo = [np.power(10,-4.17) - np.power(10,-4.17-0.61),np.power(10,-3.53) -np.power(10,-3.53-0.2),np.power(10,-2.77) -np.power(10,-2.77-0.15),np.power(10,-2.3) -np.power(10,-2.3-0.27)]
    # compile_literature(9., PG23_9X, PG23_9Y, PG23_9YerrLo, PG23_9YerrUp, "Perez-Gonzalez+23")

    # PG23_11X = [-18.5,-17.5]
    # PG23_11Y = [np.power(10,-3.57),np.power(10,-2.82)]
    # PG23_11YerrUp = [np.power(10,-3.57+0.18) - np.power(10,-3.57),np.power(10,-2.82+0.20) -np.power(10,-2.82)]
    # PG23_11YerrLo = [np.power(10,-3.57) - np.power(10,-3.57-0.32),np.power(10,-2.82) -np.power(10,-2.82-0.34)]
    # compile_literature(11., PG23_11X, PG23_11Y, PG23_11YerrLo, PG23_11YerrUp, "Perez-Gonzalez+23")

    # PG23_12X = [-18.3,-17.3]
    # PG23_12YerrUp = [np.power(10,-3.47+0.20) - np.power(10,-3.47),np.power(10,-2.87+0.23) -np.power(10,-2.87)]
    # PG23_12YerrLo = [np.power(10,-3.47) - np.power(10,-3.47-0.40),np.power(10,-2.87) -np.power(10,-2.87-0.50)]
    # PG23_12Y = [np.power(10,-3.47),np.power(10,-2.87)]
    # compile_literature(12., PG23_12X, PG23_12Y, PG23_12YerrLo, PG23_12YerrUp, "Perez-Gonzalez+23")

    # #Robertson+24

    # Rob24_12X = [-18.5,-18,-17.6]
    # Rob24_12Y = [1.22e-4,3.2e-4,1.54e-4]
    # Rob24_12Yerr = [0.94e-4,2.46e-4,1.18e-4]
    # compile_literature(12., Rob24_12X, Rob24_12Y, Rob24_12Yerr, Rob24_12Yerr, "Robertson+24")

    # #Adams+24
    # Ada24_z_arr = [8.] * 6 + [9.] * 7 + [10.5] * 5 + [12.5] * 3

    # Ada24_8X = [-21.35, -20.85, -20.35, -19.85, -19.35, -18.6]
    # Ada24_8Y = [2.277e-5, 9.974e-5, 13.12e-5, 28.64e-5, 54.04e-5, 69.35e-5]
    # Ada24_8Yerr = [1.226e-5, 3.137e-5, 3.840e-5, 7.247e-5, 22.19e-5, 28.41e-5]

    # Ada24_9X = [-22.05,-21.55,-21.05,-20.55,-20.05, -19.55,-18.8]
    # Ada24_9Y = [0.628e-5,0.628e-5,1.257e-5,6.427e-5,10.76e-5, 18.22e-5, 42.45e-5]
    # Ada24_9Yerr = [0.536e-5,0.536e-5,0.851e-5, 2.534e-5, 5.825e-5, 9.442e-5, 20.32e-5]

    # Ada24_10X = [-20.95,-20.45,-19.95, -19.45,-18.7]
    # Ada24_10Y = [0.721e-5,1.855e-5,3.331e-5,6.674e-5,9.996e-5]
    # Ada24_10Yerr = [0.48e-5,0.888e-5,1.314e-5, 3.960e-5, 6.115e-5]

    # Ada24_12X = [-20.75,-20.25,-19.5]
    # Ada24_12Y = [0.852e-5,2.148e-5,5.923e-5]
    # Ada24_12Yerr = [0.570e-5,1.375e-5,3.883e-5]

    # Ada24_MUV_arr = np.array(Ada24_8X + Ada24_9X + Ada24_10X + Ada24_12X) * u.mag
    # Ada24_phi_arr = np.array(Ada24_8Y + Ada24_9Y + Ada24_10Y + Ada24_12Y) * (u.mag * u.Mpc ** 3) ** -1
    # Ada24_phi_err_arr = np.array(Ada24_8Yerr + Ada24_9Yerr + Ada24_10Yerr + Ada24_12Yerr) * (u.mag * u.Mpc ** 3) ** -1

    # year = 2024
    # meta = {"redshifts": [8., 9., 10.5, 12.5], "name": f"Adams et al. ({str(year)})", \
    #     "year": year, "observatory": ["Hubble", "Webb"], "references": ["2024ApJ...965..169A"]}

    # compile_literature(Ada24_z_arr, Ada24_MUV_arr, Ada24_phi_arr, Ada24_phi_err_arr, \
    #     Ada24_phi_err_arr, log_phi = False, name = "adams24", meta = meta)
    pass


if __name__ == "__main__":
    conv_flags_UVLF("morishita23")
    # main()
    # convert_all_literature()

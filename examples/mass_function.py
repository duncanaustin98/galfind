# mass_function.py
import sys
import time

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from galfind import (
    Bagpipes,
    Base_Number_Density_Function,
    Catalogue,
    Number_Density_Function,
    config,
    galfind_logger,
)
from galfind.Catalogue_Creator import Galfind_Catalogue_Creator

sys.path.insert(1, config["NumberDensityFunctions"]["FLAGS_DATA_DIR"])
try:
    from flags_data import distribution_functions
except:
    galfind_logger.critical(
        "Could not import flags_data.distribution_functions"
    )


def plot_mass_func_lit():
    author_years = [
        distribution_functions.read(path, verbose=True).name
        for path in distribution_functions.list_datasets("Mstar/obs")
    ]
    print(author_years)
    z_bins = [[5.5, 6.5], [8.5, 9.5], [9.5, 11.5], [11.5, 13.5]]
    for z_bin in z_bins:
        fig, ax = plt.subplots()
        for author_year in author_years:
            obs_func = Base_Number_Density_Function.from_flags_repo(
                "stellar_mass", z_bin, author_year
            )
            if type(obs_func) != type(None):
                save_annotate = (
                    True if author_year == author_years[-1] else False
                )
                obs_func.plot(
                    fig,
                    ax,
                    x_lims="stellar_mass",
                    save=True,
                    annotate=True,
                    show=False,
                    save_name=f"flags_mass_func_{z_bin[0]}<z<{z_bin[1]}.png",
                )


def mass_func(
    survey,
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
    cat_creator = Galfind_Catalogue_Creator(cat_type, aper_diams[0], pc_err)

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

    mass_name = "stellar_mass"
    this_work_plot_kwargs = {
        "mfc": "gray",
        "marker": "D",
        "ms": 8.0,
        "mew": 2.0,
        "mec": "black",
        "ecolor": "black",
        "elinewidth": 2.0,
    }
    mass_bins = np.arange(7.5, 11.0, 0.5)
    author_years = [
        distribution_functions.read(path, verbose=True).name
        for path in distribution_functions.list_datasets("Mstar/obs")
    ]
    author_years_dict = {author_year: {} for author_year in author_years}

    mass_func_z6 = Number_Density_Function.from_single_cat(
        cat, mass_name, mass_bins, [5.5, 6.5], x_origin=SED_fit_params_arr[0]
    )
    mass_func_z6.plot(
        x_lims=mass_name,
        obs_author_years=author_years_dict,
        plot_kwargs=this_work_plot_kwargs,
    )

    mass_func_z9 = Number_Density_Function.from_single_cat(
        cat, mass_name, mass_bins, [8.5, 9.5], x_origin=SED_fit_params_arr[0]
    )
    mass_func_z9.plot(
        x_lims=mass_name,
        obs_author_years=author_years_dict,
        plot_kwargs=this_work_plot_kwargs,
    )

    mass_func_z10_5 = Number_Density_Function.from_single_cat(
        cat, mass_name, mass_bins, [9.5, 11.5], x_origin=SED_fit_params_arr[0]
    )
    mass_func_z10_5.plot(
        x_lims=mass_name,
        obs_author_years=author_years_dict,
        plot_kwargs=this_work_plot_kwargs,
    )

    mass_func_z12_5 = Number_Density_Function.from_single_cat(
        cat, mass_name, mass_bins, [11.5, 13.5], x_origin=SED_fit_params_arr[0]
    )
    mass_func_z12_5.plot(
        x_lims=mass_name,
        obs_author_years=author_years_dict,
        plot_kwargs=this_work_plot_kwargs,
    )

    MUV_name = "M_UV"
    this_work_plot_kwargs = {
        "mfc": "gray",
        "marker": "D",
        "ms": 8.0,
        "mew": 2.0,
        "mec": "black",
        "ecolor": "black",
        "elinewidth": 2.0,
    }
    MUV_bins = np.arange(-21.25, -17.25, 0.5)
    author_years = [
        distribution_functions.read(path, verbose=True).name
        for path in distribution_functions.list_datasets("LUV/obs")
    ]
    author_years_dict = {author_year: {} for author_year in author_years}

    UVLF_z6 = Number_Density_Function.from_single_cat(
        cat, MUV_name, MUV_bins, [5.5, 6.5], x_origin=SED_fit_params_arr[0]
    )
    UVLF_z6.plot(
        x_lims=MUV_name,
        obs_author_years=author_years_dict,
        plot_kwargs=this_work_plot_kwargs,
    )

    UVLF_z9 = Number_Density_Function.from_single_cat(
        cat, MUV_name, MUV_bins, [8.5, 9.5], x_origin=SED_fit_params_arr[0]
    )
    UVLF_z9.plot(
        x_lims=MUV_name,
        obs_author_years=author_years_dict,
        plot_kwargs=this_work_plot_kwargs,
    )

    UVLF_z10_5 = Number_Density_Function.from_single_cat(
        cat, MUV_name, MUV_bins, [9.5, 11.5], x_origin=SED_fit_params_arr[0]
    )
    UVLF_z10_5.plot(
        x_lims=MUV_name,
        obs_author_years=author_years_dict,
        plot_kwargs=this_work_plot_kwargs,
    )

    UVLF_z12_5 = Number_Density_Function.from_single_cat(
        cat, MUV_name, MUV_bins, [11.5, 13.5], x_origin=SED_fit_params_arr[0]
    )
    UVLF_z12_5.plot(
        x_lims=MUV_name,
        obs_author_years=author_years_dict,
        plot_kwargs=this_work_plot_kwargs,
    )


def main():
    version = "v11"
    instruments = ["ACS_WFC", "NIRCam"]
    cat_type = "loc_depth"
    survey = "JOF"
    aper_diams = [0.32] * u.arcsec
    SED_code_arr = []  # EAZY()]
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

    # SED_fit_params_arr = make_EAZY_SED_fit_params_arr(SED_code_arr, templates_arr, lowz_zmax_arr)
    pipes_fit_params_arr = [
        {
            "code": Bagpipes(),
            "dust": "Cal",
            "dust_prior": "log_10",
            "metallicity_prior": "log_10",
            "sps_model": "BC03",
            "fix_z": False,
            "z_range": (0.0, 25.0),
            "sfh": "continuity_bursty",
        }
    ]
    SED_fit_params_arr = pipes_fit_params_arr
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

    mass_func(
        survey,
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


if __name__ == "__main__":
    main()

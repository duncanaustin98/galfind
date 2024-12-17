import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from galfind import Filter, Catalogue, Catalogue_Creator, Data, EAZY, LePhare, Bagpipes
from galfind import Colour_Selector, Unmasked_Instrument_Selector, EPOCHS_Selector
from galfind.Data import morgan_version_to_dir
# Load in a JOF data object
survey = "JOF"
version = "v11"
instrument_names = ["NIRCam"]
aper_diams = [0.32] * u.arcsec
forced_phot_band = ["F277W", "F356W", "F444W"]
min_flux_pc_err = 10.

def test_selection():
    SED_fit_params_arr = [
        {"templates": "fsps_larson", "lowz_zmax": 4.0},
        {"templates": "fsps_larson", "lowz_zmax": 6.0},
        {"templates": "fsps_larson", "lowz_zmax": None}
    ]

    JOF_cat = Catalogue.pipeline(
        survey,
        version,
        instrument_names = instrument_names, 
        version_to_dir_dict = morgan_version_to_dir,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        min_flux_pc_err = min_flux_pc_err,
        #crops = EPOCHS_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), allow_lowz=True)
    )
    # load sextractor half-light radii
    # JOF_cat.load_sextractor_Re()

    # load EAZY SED fitting results
    for SED_fit_params in SED_fit_params_arr:
        EAZY_fitter = EAZY(SED_fit_params)
        EAZY_fitter(JOF_cat, aper_diams[0], load_PDFs = True, load_SEDs = True, update = True)

    # perform EPOCHS selection
    # epochs_selector = EPOCHS_Selector(aper_diams[0], EAZY_fitter, allow_lowz = False, unmasked_instruments = "NIRCam")
    # EPOCHS_JOF_cat = epochs_selector(JOF_cat, return_copy = True)

    # from galfind import EPOCHS_Selector
    # epochs_selector = EPOCHS_Selector(allow_lowz = False, unmasked_instruments = "NIRCam")
    # epochs_selected_cat = epochs_selector(JOF_cat, aper_diams[0], EAZY_fitter, return_copy = True)
    # epochs_selector_lowz = EPOCHS_Selector(allow_lowz = True, unmasked_instruments = "NIRCam")
    # epochs_selected_cat_lowz = epochs_selector_lowz(JOF_cat, aper_diams[0], EAZY_fitter, return_copy = True)

    SED_fit_label = "EAZY_fsps_larson_zfree"
    from galfind import MUV_Calculator, Xi_Ion_Calculator, M99
    for beta_dust_conv in [None, M99]: #, Reddy18(C00(), 100 * u.Myr), Reddy18(C00(), 300 * u.Myr)]:
        for fesc_conv in [None]:#, "Chisholm22"]: # None, 0.1, 0.2, 0.5,
            calculator = Xi_Ion_Calculator(aper_diams[0], SED_fit_label, beta_dust_conv = beta_dust_conv, fesc_conv = fesc_conv)
            calculator(JOF_cat, n_chains = 10_000, output = False, n_jobs = 1)
    #MUV_calculator = MUV_Calculator(aper_diams[0], SED_fit_label)
    #MUV_calculator(JOF_cat, n_chains = 10_000, output = False, n_jobs = 1)

def test_pipes():
    
    # {"templates": "fsps_larson", "lowz_zmax": 4.0}, {"templates": "fsps_larson", "lowz_zmax": 6.0}, 
    SED_fit_params_arr = [{"templates": "fsps_larson", "lowz_zmax": None}]

    JOF_cat = Catalogue.pipeline(
        survey,
        version,
        instrument_names = instrument_names, 
        version_to_dir_dict = morgan_version_to_dir,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        min_flux_pc_err = min_flux_pc_err,
        crops = EPOCHS_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), allow_lowz=True)
    )

    #JOF_cat.load_sextractor_Re()

    for SED_fit_params in SED_fit_params_arr:
        EAZY_fitter = EAZY(SED_fit_params)
        EAZY_fitter(JOF_cat, aper_diams[0], load_PDFs = False, load_SEDs = False, update = True)

    # EPOCHS_JOF_cat = EPOCHS_Selector()(JOF_cat, aper_diams[0], EAZY_fitter, return_copy = True)

    pipes_SED_fit_params = {"fix_z": EAZY_fitter.label, "fesc": None}
    pipes_fitter = Bagpipes(pipes_SED_fit_params)
    pipes_fitter(JOF_cat, aper_diams[0], save_PDFs = False, load_SEDs = False, load_PDFs = False, overwrite = False)

def test_UVLF():
    from galfind import Redwards_Lya_Detect_Selector, Sextractor_Bands_Radius_Selector, Xi_Ion_Calculator, UV_Beta_Calculator, Rest_Frame_Property_Limit_Selector

    # {"templates": "fsps_larson", "lowz_zmax": 4.0}, {"templates": "fsps_larson", "lowz_zmax": 6.0}, 
    SED_fit_params_arr = [{"templates": "fsps_larson", "lowz_zmax": None}]

    test_selection_criteria = [
        Redwards_Lya_Detect_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), SNR_lims = [5.0], widebands_only = True),
        Unmasked_Instrument_Selector("NIRCam"),
    ]
    test_selection_criteria.extend([
        Sextractor_Bands_Radius_Selector( \
        band_names = ["F277W", "F356W", "F444W"], \
        gtr_or_less = "gtr", lim = 45. * u.marcsec)
    ])
    xi_ion_calc = Xi_Ion_Calculator(aper_diams[0], EAZY(SED_fit_params_arr[-1]), ext_src_corrs = None)
    low_xi_ion = deepcopy(test_selection_criteria)
    low_xi_ion.extend([Rest_Frame_Property_Limit_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), xi_ion_calc, 25.2 * u.Unit("dex(Hz/erg)"), "less")])
    high_xi_ion = deepcopy(test_selection_criteria)
    high_xi_ion.extend([Rest_Frame_Property_Limit_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), xi_ion_calc, 25.2 * u.Unit("dex(Hz/erg)"), "gtr")])
    
    for crops in [
        test_selection_criteria,
        #high_xi_ion,
        #low_xi_ion,
        #test_selection_criteria,
        #EPOCHS_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), allow_lowz=False), 
        #EPOCHS_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), allow_lowz=True)
    ]:
        JOF_cat = Catalogue.pipeline(
            survey,
            version,
            instrument_names = instrument_names, 
            version_to_dir_dict = morgan_version_to_dir,
            aper_diams = aper_diams,
            forced_phot_band = forced_phot_band,
            min_flux_pc_err = min_flux_pc_err,
            #crops = crops, #EPOCHS_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), allow_lowz=False)
        )
        #JOF_cat.load_sextractor_Re()
        #JOF_cat.load_sextractor_ext_src_corrs()

        for SED_fit_params in SED_fit_params_arr:
            EAZY_fitter = EAZY(SED_fit_params)
            EAZY_fitter(JOF_cat, aper_diams[0], load_PDFs = True, load_SEDs = True, update = True)
            
        xi_ion_calc(JOF_cat, n_chains = 10_000, output = False, n_jobs = 1)
        Rest_Frame_Property_Limit_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), xi_ion_calc, 25.2 * u.Unit("dex(Hz/erg)"), "less")(JOF_cat)
        breakpoint()

        #Redwards_Lya_Detect_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), SNR_lims = [5.0], widebands_only = True)(JOF_cat)
        #Rest_Frame_Property_Limit_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), beta_calc, -2.2 * u.dimensionless_unscaled, "gtr")(JOF_cat)
        #Rest_Frame_Property_Limit_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), beta_calc, -2.2 * u.dimensionless_unscaled, "less")(JOF_cat)   

        # epochs_selector = EPOCHS_Selector(aper_diams[0], EAZY_fitter, allow_lowz = False, unmasked_instruments = "NIRCam")
        # epochs_selected_cat = epochs_selector(JOF_cat, return_copy = True)
        # epochs_selector_lowz = EPOCHS_Selector(aper_diams[0], EAZY_fitter, allow_lowz = True, unmasked_instruments = "NIRCam")
        # epochs_selected_cat_lowz = epochs_selector_lowz(JOF_cat, return_copy = True)

        from galfind import MUV_Calculator
        MUV_calculator = MUV_Calculator(aper_diams[0], EAZY_fitter.label)
        MUV_calculator(JOF_cat, n_chains = 10_000, output = False, n_jobs = 1)

        # from galfind import Xi_Ion_Calculator, M99
        # for beta_dust_conv in [None]: #, M99]: #, Reddy18(C00(), 100 * u.Myr), Reddy18(C00(), 300 * u.Myr)]:
        #     for fesc_conv in [None]: #, "Chisholm22"]: # None, 0.1, 0.2, 0.5,
        #         calculator = Xi_Ion_Calculator(aper_diams[0], EAZY_fitter.label, beta_dust_conv = beta_dust_conv, fesc_conv = fesc_conv)
        #         calculator(JOF_cat, n_chains = 10_000, output = False, n_jobs = 1)

        from galfind import Number_Density_Function
        this_work_plot_kwargs = {
            "mfc": "gray",
            "marker": "D",
            "ms": 8.0,
            "mew": 2.0,
            "mec": "black",
            "ecolor": "black",
            "elinewidth": 2.0,
        }

        MUV_arr = np.arange(-21.25, -17.25, 0.5) * u.ABmag
        z5_obs_author_years = {
            "Bouwens+21": {}
        }
        z6_obs_author_years = {
            "Bowler+15": {},
            "Bouwens+21": {},
        }
        z7_obs_author_years = {
            "Bouwens+21": {},
        }
        z8_obs_author_years = {
            "Stefanon+19": {},
            "Bouwens+21": {},
            "Adams+24": {},
        }
        z9_obs_author_years = {
            "Stefanon+19": {},
            "Bouwens+21": {},
            "Adams+24": {},
        }
        z10_obs_author_years = {
            "Bouwens+21": {},
            "Adams+24": {},
        }
        z12_obs_author_years = {
            "Bouwens+23": {},
            "Adams+24": {},
        }
        obs_author_years = [
            z5_obs_author_years,
            z6_obs_author_years,
            z7_obs_author_years,
            z8_obs_author_years,
            z9_obs_author_years,
            z10_obs_author_years,
            z12_obs_author_years
        ]
        z_bins = [
            [4.5, 5.5],
            [5.5, 6.5],
            [6.5, 7.5],
            [7.5, 8.5],
            [8.5, 9.5],
            [9.5, 11.5],
            [11.5, 13.5]
        ]
        for z_bin, obs_author_years in zip(z_bins, obs_author_years):
            UVLF = Number_Density_Function.from_single_cat(
                JOF_cat,
                MUV_calculator,
                MUV_arr,
                z_bin,
                aper_diam = aper_diams[0],
                SED_fit_code = EAZY_fitter,
                x_origin = "phot_rest",
            )
            if UVLF is not None:
                UVLF.plot(
                    obs_author_years=obs_author_years,
                    plot_kwargs=this_work_plot_kwargs,
                )

    # xi_ion_arr = np.arange(23.5, 26.5, 0.5) * u.Unit("dex(Hz/erg)")
    # xi_ion_func_z5 = Number_Density_Function.from_single_cat(
    #     JOF_cat,
    #     calculator,
    #     xi_ion_arr,
    #     [4.5, 5.5],
    #     aper_diam = aper_diams[0],
    #     SED_fit_code = EAZY_fitter,
    #     x_origin = "phot_rest",
    # )

    # xi_ion_func_z5.plot(
    #     #x_lims=calculator.name,
    #     #obs_author_years=z5_obs_author_years,
    #     plot_kwargs=this_work_plot_kwargs,
    # )

    # xi_ion_func_z6 = Number_Density_Function.from_single_cat(
    #     JOF_cat,
    #     calculator,
    #     xi_ion_arr,
    #     [5.5, 6.5],
    #     aper_diam = aper_diams[0],
    #     SED_fit_code = EAZY_fitter,
    #     x_origin = "phot_rest",
    # )
    # xi_ion_func_z6.plot(
    #     #x_lims=calculator.name,
    #     #obs_author_years=z9_author_years,
    #     plot_kwargs=this_work_plot_kwargs,
    # )

    # UVLF_z8 = Number_Density_Function.from_single_cat(
    #     JOF_cat,
    #     calculator,
    #     np.arange(-21.25, -17.25, 0.5) * u.ABmag,
    #     [7.5, 8.5],
    #     aper_diam = aper_diams[0],
    #     SED_fit_code = EAZY_fitter,
    #     x_origin = "phot_rest",
    # )
    # UVLF_z8.plot(
    #     #x_lims=calculator.name,
    #     #obs_author_years=z9_author_years,
    #     plot_kwargs=this_work_plot_kwargs,
    # )

    # UVLF_z9 = Number_Density_Function.from_single_cat(
    #     JOF_cat,
    #     calculator,
    #     np.arange(-21.25, -17.25, 0.5) * u.ABmag,
    #     [8.5, 9.5],
    #     aper_diam = aper_diams[0],
    #     SED_fit_code = EAZY_fitter,
    #     x_origin = "phot_rest",
    # )
    # # z9_author_years = {
    # #     "Bouwens+21": {"z_ref": 9.0, "plot_kwargs": {}},
    # #     "Harikane+22": {"z_ref": 9.0, "plot_kwargs": {}},
    # #     "Finkelstein+23": {"z_ref": 9.0, "plot_kwargs": {}},
    # #     "Leung+23": {"z_ref": 9.0, "plot_kwargs": {}},
    # #     "Perez-Gonzalez+23": {"z_ref": 9.0, "plot_kwargs": {}},
    # #     "Adams+24": {"z_ref": 9.0, "plot_kwargs": {}},
    # # } # also Finkelstein+22, but wide z bin used
    # UVLF_z9.plot(
    #     x_lims=calculator.name,
    #     #obs_author_years=z9_author_years,
    #     plot_kwargs=this_work_plot_kwargs,
    # )

def split_UVLF_by_beta():
    from galfind import Redwards_Lya_Detect_Selector, Sextractor_Bands_Radius_Selector, UV_Beta_Calculator, Rest_Frame_Property_Limit_Selector

    SED_fit_params_arr = [{"templates": "fsps_larson", "lowz_zmax": None}]

    test_selection_criteria = [
        Redwards_Lya_Detect_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), SNR_lims = [5.0], widebands_only = True),
        Unmasked_Instrument_Selector("NIRCam"),
    ]
    test_selection_criteria.extend([
        Sextractor_Bands_Radius_Selector( \
        band_names = ["F277W", "F356W", "F444W"], \
        gtr_or_less = "gtr", lim = 45. * u.marcsec)
    ])
    beta_calc = UV_Beta_Calculator(aper_diams[0], EAZY(SED_fit_params_arr[-1]))
    blue_beta = deepcopy(test_selection_criteria)
    blue_beta.extend([Rest_Frame_Property_Limit_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), beta_calc, -2.2 * u.dimensionless_unscaled, "less")])
    red_beta = deepcopy(test_selection_criteria)
    red_beta.extend([Rest_Frame_Property_Limit_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), beta_calc, -2.2 * u.dimensionless_unscaled, "gtr")])
    
    JOF_cat = Catalogue.pipeline(
        survey,
        version,
        instrument_names = instrument_names, 
        version_to_dir_dict = morgan_version_to_dir,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        min_flux_pc_err = min_flux_pc_err,
        crops = test_selection_criteria,
    )
    JOF_cat_blue_beta = Catalogue.pipeline(
        survey,
        version,
        instrument_names = instrument_names, 
        version_to_dir_dict = morgan_version_to_dir,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        min_flux_pc_err = min_flux_pc_err,
        crops = blue_beta,
    )
    JOF_cat_red_beta = Catalogue.pipeline(
        survey,
        version,
        instrument_names = instrument_names, 
        version_to_dir_dict = morgan_version_to_dir,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        min_flux_pc_err = min_flux_pc_err,
        crops = red_beta,
    )

    for SED_fit_params in SED_fit_params_arr:
        EAZY_fitter = EAZY(SED_fit_params)
        EAZY_fitter(JOF_cat, aper_diams[0], load_PDFs = True, load_SEDs = True, update = True)
        EAZY_fitter(JOF_cat_blue_beta, aper_diams[0], load_PDFs = True, load_SEDs = True, update = True)
        EAZY_fitter(JOF_cat_red_beta, aper_diams[0], load_PDFs = True, load_SEDs = True, update = True)

    from galfind import MUV_Calculator
    MUV_calculator = MUV_Calculator(aper_diams[0], EAZY_fitter.label)
    MUV_calculator(JOF_cat, n_chains = 10_000, output = False, n_jobs = 1)
    MUV_calculator(JOF_cat_blue_beta, n_chains = 10_000, output = False, n_jobs = 1)
    MUV_calculator(JOF_cat_red_beta, n_chains = 10_000, output = False, n_jobs = 1)

    from galfind import Number_Density_Function
    cat_plot_kwargs = {
        "mfc": "gray",
        "marker": "D",
        "ms": 8.0,
        "mew": 2.0,
        "mec": "black",
        "ecolor": "black",
        "elinewidth": 2.0,
    }
    blue_plot_kwargs = {
        "mfc": "lightblue",
        "marker": "o",
        "ms": 8.0,
        "mew": 2.0,
        "mec": "blue",
        "ecolor": "blue",
        "elinewidth": 2.0,
    }
    red_plot_kwargs = {
        "mfc": "lightcoral",
        "marker": "o",
        "ms": 8.0,
        "mew": 2.0,
        "mec": "red",
        "ecolor": "red",
        "elinewidth": 2.0,
    }
    MUV_arr = np.arange(-21.25, -17.25, 0.5) * u.ABmag
    z5_obs_author_years = {
        "Bouwens+21": {}
    }
    z6_obs_author_years = {
        "Bowler+15": {},
        "Bouwens+21": {},
    }
    obs_author_years = [
        z5_obs_author_years,
        z6_obs_author_years,
    ]
    z_bins = [
        [4.5, 5.5],
        [5.5, 6.5],
    ]
    cats = [JOF_cat, JOF_cat_blue_beta, JOF_cat_red_beta]
    plot_kwargs_arr = [cat_plot_kwargs, blue_plot_kwargs, red_plot_kwargs]
    for z_bin, obs_author_years in zip(z_bins, obs_author_years):
        fig, ax = plt.subplots()
        for i, (cat, plot_kwargs) in enumerate(zip(cats, plot_kwargs_arr)):
            UVLF = Number_Density_Function.from_single_cat(
                cat,
                MUV_calculator,
                MUV_arr,
                z_bin,
                aper_diam = aper_diams[0],
                SED_fit_code = EAZY_fitter,
                x_origin = "phot_rest",
            )
            if UVLF is not None:
                UVLF.plot(
                    fig,
                    ax,
                    obs_author_years=obs_author_years if i == len(cats) - 1 else {},
                    plot_kwargs=plot_kwargs,
                    save = True if i == len(cats) - 1 else False,
                    save_name = f"{JOF_cat.crop_name.split('/')[-1]}+{z_bin[0]}<z<{z_bin[1]}_beta_split_-2.2" if i == len(cats) - 1 else None,
                    title = r"5$\sigma$ detected, $\beta$ split"
                )


def main():
    JOF_data = Data.pipeline(
        survey, 
        version, 
        instrument_names = instrument_names, 
        version_to_dir_dict = morgan_version_to_dir,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        min_flux_pc_err = min_flux_pc_err
    )
    cat_path = JOF_data.phot_cat_path
    filterset = JOF_data.filterset
    # [0.32] * u.arcsec hardcoded for now
    cat_creator = Catalogue_Creator(survey, version, cat_path, filterset, aper_diams)
    cat = cat_creator(cropped = False)

    # LePhare_SED_fit_params = {"GAL_TEMPLATES": "BC03_Chabrier2003_Z(m42_m62)"}
    # LePhare_fitter = LePhare(LePhare_SED_fit_params)
    # LePhare_fitter.compile(filterset)

    #SED_fit_params_arr = [{"templates": "fsps_larson", "lowz_zmax": None}]
    SED_fit_params_arr = [{"templates": "fsps_larson", "lowz_zmax": 4.0}, {"templates": "fsps_larson", "lowz_zmax": 6.0}, {"templates": "fsps_larson", "lowz_zmax": None}]
    for SED_fit_params in SED_fit_params_arr:
        EAZY_fitter = EAZY(SED_fit_params)
        EAZY_fitter(cat, aper_diams[0], load_PDFs = True, load_SEDs = True, update = True)
    EPOCHS_Selector()(cat, aper_diams[0], EAZY_fitter)

def check_multinest():
    import pymultinest as pmn

if __name__ == "__main__":
    #test_load()
    #main()
    #test_selection()
    test_UVLF()
    #split_UVLF_by_beta()
    #test_pipes()
    #check_multinest()

    # LePhare_SED_fit_params = {"GAL_TEMPLATES": "BC03_Chabrier2003_Z(m42_m62)"}
    # EAZY_SED_fit_params = {"templates": "fsps_larson", "lowz_zmax": None}
    # LePhare_fitter = LePhare(LePhare_SED_fit_params)
    # LePhare_fitter.compile()
    # print(LePhare_fitter.SED_fit_params)
    #EAZY_fitter = EAZY(EAZYSED_fit_params)
    #breakpoint()

# def test_load():
#     import numpy as np
#     import time
#     from astropy.table import Table

#     # Create a large random array
#     array = np.random.rand(10000, 1)
#     meta = {"blah": "blah"}

#     # Save as .npy
#     npy_file = "data.npy"
#     npy_meta_file = "data.meta.npy"
#     np.save(npy_file, array)
#     np.save(npy_meta_file, meta)

#     # Save as .npz
#     npz_file = "data.npz"
#     np.savez(npz_file, array=array, meta=meta)

#     save_tab = Table({"x": array})
#     save_tab.meta = meta
#     save_tab.write("data.ecsv", overwrite=True)

#     # Load .npy
#     start = time.time()
#     loaded_npy = np.load(npy_file)
#     loaded_npy_meta = np.load(npy_meta_file, allow_pickle=True).item()
#     end = time.time()
#     print(f"Loading .npy: {end - start:.6f} seconds")

#     # Load .npz
#     start = time.time()
#     loaded_npz = np.load(npz_file, allow_pickle=True)
#     loaded_array_from_npz = loaded_npz["array"] # Extract the array
#     loaded_npz_meta = loaded_npz["meta"]
#     end = time.time()
#     print(f"Loading .npz: {end - start:.6f} seconds")

#     start = time.time()
#     loaded_tab = Table.read("data.ecsv")
#     loaded_array_from_tab = loaded_tab["x"]
#     loaded_ecsv_meta = loaded_tab.meta
#     end = time.time()
#     print(f"Loading .ecsv: {end - start:.6f} seconds")
#     breakpoint()
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from galfind import Filter, Catalogue, Catalogue_Creator, Data, EAZY, LePhare, Bagpipes, config
from galfind import Colour_Selector, Unmasked_Instrument_Selector, EPOCHS_Selector
from galfind.Data import morgan_version_to_dir

plt.style.use(
    f"{config['DEFAULT']['GALFIND_DIR']}/galfind_style.mplstyle"
)

# Load in a data object
survey = "JADES-DR3-GN-Deep"
version = "v13"
instrument_names = ["NIRCam"] # "ACS_WFC"
aper_diams = [0.32] * u.arcsec
forced_phot_band = ["F277W", "F356W", "F444W"]
min_flux_pc_err = 10.

def test_euclid_filters():
    from galfind import Multiple_Filter
    filterset = Multiple_Filter.from_instruments(
        ["MegaCam", "NISP", "VIS", "IRAC"],
        excl_bands = [
            "CFHT/MegaCam.Y",
            "CFHT/MegaCam.J",
            "CFHT/MegaCam.H",
        ]
    )
    fig, ax = plt.subplots()
    filterset.plot(ax, save = True)
    breakpoint()


def test_selection():
    SED_fit_params_arr = [
        {"templates": "fsps_larson", "lowz_zmax": 4.0},
        {"templates": "fsps_larson", "lowz_zmax": 6.0},
        {"templates": "fsps_larson", "lowz_zmax": None}
    ]
    # import time
    # time.sleep(180 * 60)
    # data = Data.from_survey_version(
    #     survey,
    #     version,
    #     instrument_names = instrument_names,
    #     version_to_dir_dict = morgan_version_to_dir,
    #     aper_diams = aper_diams,
    #     forced_phot_band = forced_phot_band,
    # )
    # print(data.band_data_arr)
    # data.mask(
    #     "auto",
    #     angle = 92.0
    # )

    # fig, ax = plt.subplots()
    # data.filterset.plot(ax, save = True)
    #breakpoint()

    # # temp: define the z=6 sample
    # from galfind import Redshift_Bin_Selector, Band_SNR_Selector
    # EAZY_fitter = EAZY(SED_fit_params_arr[-1])
    # sample = [
    #     Band_SNR_Selector(
    #         aper_diams[0],
    #         "F115W",
    #         "detect",
    #         5.0
    #     ),
    #     Redshift_Bin_Selector(
    #         aper_diams[0],
    #         EAZY_fitter,
    #         [5.5, 6.5]
    #     )
    # ]

    cat = Catalogue.pipeline(
        survey,
        version,
        instrument_names = instrument_names,
        version_to_dir_dict = morgan_version_to_dir,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        min_flux_pc_err = min_flux_pc_err,
        #crops = EPOCHS_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), allow_lowz=False)
    )
    #breakpoint()

    #from galfind import Unmasked_Instrument_Selector
    #Unmasked_Instrument_Selector("ACS_WFC")(cat)

    # # load EAZY SED fitting results
    for SED_fit_params in SED_fit_params_arr:
        EAZY_fitter = EAZY(SED_fit_params)
        EAZY_fitter(cat, aper_diams[0], load_PDFs = True, load_SEDs = True, update = True)

    # #breakpoint()

    # # cat.plot(MUV_calculator, xi_ion_calculator, incl_x_errs = False, incl_y_errs = False, annotate = True, plot_type = "individual", save = True, log_y = True)

    # load sextractor half-light radii
    cat.load_sextractor_Re()

    from galfind import EPOCHS_Selector, Redwards_Lya_Detect_Selector
    epochs_selector = EPOCHS_Selector(aper_diams[0], EAZY_fitter, allow_lowz = False, unmasked_instruments = "NIRCam")
    epochs_selected_cat = epochs_selector(cat, return_copy = True)
    # epochs_selector_lowz = EPOCHS_Selector(aper_diams[0], EAZY_fitter, allow_lowz = True, unmasked_instruments = "NIRCam")
    # epochs_selected_cat_lowz = epochs_selector_lowz(cat, return_copy = True)

    from galfind import MUV_Calculator
    MUV_calculator = MUV_Calculator(aper_diams[0], EAZY_fitter.label)
    MUV_calculator(epochs_selected_cat, n_chains = 10_000, output = False, n_jobs = 1)

    epochs_selected_cat.plot_phot_diagnostics(
        aper_diams[0],
        EAZY_fitter,
        EAZY_fitter,
        imshow_kwargs = {},
        norm_kwargs = {},
        aper_kwargs = {},
        kron_kwargs = {},
        n_cutout_rows = 2,
        wav_unit = u.um,
        flux_unit = u.ABmag,
        overwrite = True
    )

    # from galfind import MUV_Calculator, Xi_Ion_Calculator, M99
    # for beta_dust_conv in [None, M99]:#, Reddy18(C00(), 100 * u.Myr), Reddy18(C00(), 300 * u.Myr)]:
    #     for fesc_conv in [None, "Chisholm22"]: # None, 0.1, 0.2, 0.5,
    #         xi_ion_calculator = Xi_Ion_Calculator(aper_diams[0], EAZY_fitter.label, beta_dust_conv = beta_dust_conv, fesc_conv = fesc_conv, logged = False)
    #         xi_ion_calculator(cat, n_chains = 10_000, output = False, n_jobs = 1)
    #         #breakpoint()

    # # Redwards_Lya_Detect_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), SNR_lims = [5.0], widebands_only = True)(JOF_cat)
    # # # SED_fit_label = "EAZY_fsps_larson_zfree"
    # #from galfind import MUV_Calculator, Xi_Ion_Calculator, M99
    # # # for beta_dust_conv in [None, M99]: #, Reddy18(C00(), 100 * u.Myr), Reddy18(C00(), 300 * u.Myr)]:
    # # #     for fesc_conv in [None]:#, "Chisholm22"]: # None, 0.1, 0.2, 0.5,
    # # #         calculator = Xi_Ion_Calculator(aper_diams[0], SED_fit_label, beta_dust_conv = beta_dust_conv, fesc_conv = fesc_conv)
    # # #         calculator(JOF_cat, n_chains = 10_000, output = False, n_jobs = 1)
    # # breakpoint()
    # # MUV_calculator = MUV_Calculator(aper_diams[0], EAZY_fitter)
    # # MUV_calculator(cat, n_chains = 10_000, output = False, n_jobs = 1)

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
    breakpoint()

    #JOF_cat.load_sextractor_Re()

    for SED_fit_params in SED_fit_params_arr:
        EAZY_fitter = EAZY(SED_fit_params)
        EAZY_fitter(JOF_cat, aper_diams[0], load_PDFs = True, load_SEDs = True, update = True)

    # EPOCHS_JOF_cat = EPOCHS_Selector()(JOF_cat, aper_diams[0], EAZY_fitter, return_copy = True)

    pipes_SED_fit_params = {"fix_z": EAZY_fitter.label, "fesc": None}
    pipes_fitter = Bagpipes(pipes_SED_fit_params)
    pipes_fitter(JOF_cat, aper_diams[0], save_PDFs = False, load_SEDs = False, load_PDFs = True, overwrite = False, update = True)

    # from galfind import Ext_Src_Property_Calculator
    # ext_src_calculator = Ext_Src_Property_Calculator("stellar_mass", "Mstar", aper_diams[0], pipes_fitter.label)
    # ext_src_calculator(JOF_cat) #, n_chains = 10_000, output = False, n_jobs = 1)
    # ext_src_calculator = Ext_Src_Property_Calculator("M_UV", "MUV", aper_diams[0], pipes_fitter.label)
    # ext_src_calculator(JOF_cat) #, n_chains = 10_000, output = False, n_jobs = 1)
    #breakpoint()

    # from galfind.Property_calculator import Redshift_Extractor, Custom_SED_Property_Extractor
    # from galfind import UV_Beta_Calculator
    # z_extractor = Redshift_Extractor(aper_diams[0], EAZY_fitter)
    # MUV_extractor = Custom_SED_Property_Extractor("M_UV", r"$M_{\mathrm{UV}}$", aper_diams[0], pipes_fitter.label)
    # stellar_mass_extractor = Custom_SED_Property_Extractor("stellar_mass", r"$\log_{10}(M_{\star}~/~\mathrm{M}_{\odot})$", aper_diams[0], pipes_fitter.label)
    # xi_ion_extractor = Custom_SED_Property_Extractor("xi_ion_caseB", r"$\xi_{\mathrm{ion}}~/~\mathrm{Hz}~\mathrm{erg}^{-1}$", aper_diams[0], pipes_fitter.label)
    # beta_extractor = Custom_SED_Property_Extractor("beta_C94", r"$\beta$", aper_diams[0], pipes_fitter.label)
    # dust_extractor = Custom_SED_Property_Extractor("dust:Av", r"$A_{\mathrm{V}}$", aper_diams[0], pipes_fitter.label)
    # beta_calculator = UV_Beta_Calculator(aper_diams[0], EAZY_fitter)
    # beta_calculator(JOF_cat, n_chains = 10_000, output = False, n_jobs = 1)
    # # Plot EAZY redshift on x axis and xi_ion on y axis
    # plt.style.use(f"{config['DEFAULT']['GALFIND_DIR']}/galfind_style.mplstyle")
    # plot_kwargs = {
    #     "mfc": "gray",
    #     "marker": "D",
    #     "ms": 8.0,
    #     "mew": 2.0,
    #     "mec": "black",
    #     "ecolor": "black",
    #     "elinewidth": 2.0,
    # }

    # for colour_by in [z_extractor]: #, dust_extractor, z_extractor, None]:

    #     fig, ax = plt.subplots()
    #     JOF_cat.plot(z_extractor, xi_ion_extractor, incl_x_errs = False, incl_y_errs = False, annotate = True, plot_type = "individual", save = True, fig = fig, ax = ax, log_y = True, c_calculator = colour_by)
    #     #JOF_cat.plot(z_calculator, xi_ion_calculator, incl_x_errs = False, incl_y_errs = False, annotate = True, plot_type = "stacked", plot_kwargs = plot_kwargs, fig = fig, ax = ax)

    #     fig, ax = plt.subplots()
    #     ax.set_xlim(-21.5, -16.5)
    #     JOF_cat.plot(MUV_extractor, xi_ion_extractor, incl_x_errs = False, incl_y_errs = False, annotate = True, plot_type = "individual", save = True, fig = fig, ax = ax, log_y = True, c_calculator = colour_by)
    #     #JOF_cat.plot(MUV_extractor, xi_ion_calculator, incl_x_errs = False, incl_y_errs = False, annotate = True, plot_type = "stacked", plot_kwargs = plot_kwargs, fig = fig, ax = ax)

    #     fig, ax = plt.subplots()
    #     JOF_cat.plot(stellar_mass_extractor, xi_ion_extractor, incl_x_errs = False, incl_y_errs = False, annotate = True, plot_type = "individual", save = True, fig = fig, ax = ax, log_y = True, c_calculator = colour_by)
    #     #JOF_cat.plot(stellar_mass_extractor, xi_ion_calculator, incl_x_errs = False, incl_y_errs = False, annotate = True, plot_type = "stacked", plot_kwargs = plot_kwargs, fig = fig, ax = ax)

    #     fig, ax = plt.subplots()
    #     ax.set_xlim(-21.5, -16.5)
    #     JOF_cat.plot(MUV_extractor, stellar_mass_extractor, incl_x_errs = False, incl_y_errs = False, annotate = True, plot_type = "individual", save = True, fig = fig, ax = ax, c_calculator = colour_by)
    #     #JOF_cat.plot(MUV_extractor, stellar_mass_extractor, incl_x_errs = False, incl_y_errs = False, annotate = True, plot_type = "stacked", plot_kwargs = plot_kwargs, fig = fig, ax = ax)

    #     fig, ax = plt.subplots()
    #     ax.set_xlim(-21.5, -16.5)
    #     JOF_cat.plot(MUV_extractor, beta_calculator, incl_x_errs = False, incl_y_errs = False, annotate = True, plot_type = "individual", save = True, fig = fig, ax = ax, c_calculator = colour_by)
    #     #JOF_cat.plot(MUV_extractor, stellar_mass_extractor, incl_x_errs = False, incl_y_errs = False, annotate = True, plot_type = "stacked", plot_kwargs = plot_kwargs, fig = fig, ax = ax)

    #     fig, ax = plt.subplots()
    #     JOF_cat.plot(stellar_mass_extractor, beta_calculator, incl_x_errs = False, incl_y_errs = False, annotate = True, plot_type = "individual", save = True, fig = fig, ax = ax, c_calculator = colour_by)
    #     #JOF_cat.plot(MUV_extractor, stellar_mass_extractor, incl_x_errs = False, incl_y_errs = False, annotate = True, plot_type = "stacked", plot_kwargs = plot_kwargs, fig = fig, ax = ax)

    #     fig, ax = plt.subplots()
    #     JOF_cat.plot(beta_calculator, xi_ion_extractor, incl_x_errs = False, incl_y_errs = False, annotate = True, plot_type = "individual", save = True, fig = fig, ax = ax, log_y = True, c_calculator = colour_by)
    #     #JOF_cat.plot(MUV_extractor, stellar_mass_extractor, incl_x_errs = False, incl_y_errs = False, annotate = True, plot_type = "stacked", plot_kwargs = plot_kwargs, fig = fig, ax = ax)


def test_UVLF():
    from galfind import Redwards_Lya_Detect_Selector, Sextractor_Bands_Radius_Selector, Xi_Ion_Calculator, UV_Beta_Calculator, Rest_Frame_Property_Limit_Selector

    # {"templates": "fsps_larson", "lowz_zmax": 4.0}, {"templates": "fsps_larson", "lowz_zmax": 6.0}, 
    SED_fit_params_arr = [{"templates": "fsps_larson", "lowz_zmax": 4.0}, {"templates": "fsps_larson", "lowz_zmax": 6.0}, {"templates": "fsps_larson", "lowz_zmax": None}]

    test_selection_criteria = [
        Redwards_Lya_Detect_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), SNR_lims = [5.0], widebands_only = True),
        Unmasked_Instrument_Selector("NIRCam"),
    ]
    test_selection_criteria.extend([
        Sextractor_Bands_Radius_Selector( \
        band_names = ["F277W", "F356W", "F444W"], \
        gtr_or_less = "gtr", lim = 45. * u.marcsec)
    ])
    # xi_ion_calc = Xi_Ion_Calculator(aper_diams[0], EAZY(SED_fit_params_arr[-1]), ext_src_corrs = None)
    # low_xi_ion = deepcopy(test_selection_criteria)
    # low_xi_ion.extend([Rest_Frame_Property_Limit_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), xi_ion_calc, 25.2 * u.Unit("dex(Hz/erg)"), "less")])
    # high_xi_ion = deepcopy(test_selection_criteria)
    # high_xi_ion.extend([Rest_Frame_Property_Limit_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), xi_ion_calc, 25.2 * u.Unit("dex(Hz/erg)"), "gtr")])
    
    for crops in [
        #test_selection_criteria,
        #high_xi_ion,
        #low_xi_ion,
        #test_selection_criteria,
        #EPOCHS_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), allow_lowz=False), 
        EPOCHS_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), allow_lowz=True)
    ]:
        JOF_data = Data.from_survey_version(
            survey,
            version,
            instrument_names = instrument_names,
            version_to_dir_dict = morgan_version_to_dir,
            aper_diams = aper_diams,
            forced_phot_band = forced_phot_band,
        )
        breakpoint()
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
        JOF_cat.load_sextractor_Re()
        JOF_cat.load_sextractor_ext_src_corrs()

        for SED_fit_params in SED_fit_params_arr:
            EAZY_fitter = EAZY(SED_fit_params)
            EAZY_fitter(JOF_cat, aper_diams[0], load_PDFs = True, load_SEDs = True, update = True)
            
        # xi_ion_calc(JOF_cat, n_chains = 10_000, output = False, n_jobs = 1)
        # Rest_Frame_Property_Limit_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), xi_ion_calc, 25.2 * u.Unit("dex(Hz/erg)"), "less")(JOF_cat)
        # Rest_Frame_Property_Limit_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), xi_ion_calc, 25.2 * u.Unit("dex(Hz/erg)"), "gtr")(JOF_cat)
        # breakpoint()

        epochs_selector = EPOCHS_Selector(aper_diams[0], EAZY_fitter, allow_lowz = False, unmasked_instruments = "NIRCam")
        epochs_selected_cat = epochs_selector(JOF_cat, return_copy = True)
        epochs_selector_lowz = EPOCHS_Selector(aper_diams[0], EAZY_fitter, allow_lowz = True, unmasked_instruments = "NIRCam")
        epochs_selected_cat_lowz = epochs_selector_lowz(JOF_cat, return_copy = True)

        five_sig_cat = Redwards_Lya_Detect_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), SNR_lims = [5.0], widebands_only = True)(JOF_cat)
        #Rest_Frame_Property_Limit_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), beta_calc, -2.2 * u.dimensionless_unscaled, "gtr")(JOF_cat)
        #Rest_Frame_Property_Limit_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), beta_calc, -2.2 * u.dimensionless_unscaled, "less")(JOF_cat)   

        from galfind import MUV_Calculator
        MUV_calculator = MUV_Calculator(aper_diams[0], EAZY_fitter.label)
        MUV_calculator(JOF_cat, n_chains = 10_000, output = False, n_jobs = 1)

        #breakpoint()

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
            #z5_obs_author_years,
            z6_obs_author_years,
            # z7_obs_author_years,
            # z8_obs_author_years,
            # z9_obs_author_years,
            # z10_obs_author_years,
            # z12_obs_author_years
        ]
        z_bins = [
            #[4.5, 5.5],
            [5.5, 6.5],
            # [6.5, 7.5],
            # [7.5, 8.5],
            # [8.5, 9.5],
            # [9.5, 11.5],
            # [11.5, 13.5]
        ]
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colours = prop_cycle.by_key()['color']
        for i, (z_bin, obs_author_years, colour) in enumerate(zip(z_bins, obs_author_years, colours)):
            UVLF = Number_Density_Function.from_single_cat(
                five_sig_cat, #JOF_cat,
                MUV_calculator,
                MUV_arr,
                z_bin,
                aper_diam = aper_diams[0],
                SED_fit_code = EAZY_fitter,
                x_origin = "phot_rest",
            )
            # from galfind import Flat_Prior, Priors, Schechter_Mag_Fitter
            # prior_arr = [
            #     Flat_Prior("log10_phi_star", [-7.5, 2.5], -2.5),
            #     Flat_Prior("M_star", [-25.0, -15.0], -20.0),
            #     Flat_Prior("alpha", [-5., 3.], -1.5)
            # ]
            # priors = Priors(prior_arr)
            # UVLF.fit(Schechter_Mag_Fitter, priors, fixed_params = {}, n_walkers = 100, n_steps = 20_000)
            # corner_kwargs = {"color": colour, "sigma_arr": [2.0], "quantiles": [0.5]}
            # if i != 0:
            #     corner_kwargs["fig"] = fig
            # fig = UVLF.fitter.plot_corner(**corner_kwargs)
            # if UVLF is not None:
            #     UVLF.plot(
            #         obs_author_years=obs_author_years,
            #         plot_kwargs=this_work_plot_kwargs,
            #     )

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

def test_plotting():
    # {"templates": "fsps_larson", "lowz_zmax": 4.0}, {"templates": "fsps_larson", "lowz_zmax": 6.0}, 

    # Load catalogue
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

    # Load EAZY SED fitting results
    for SED_fit_params in SED_fit_params_arr:
        EAZY_fitter = EAZY(SED_fit_params)
        EAZY_fitter(JOF_cat, aper_diams[0], load_PDFs = True, load_SEDs = True, update = True)

    from galfind import MUV_Calculator
    MUV_calculator = MUV_Calculator(aper_diams[0], EAZY_fitter.label)
    MUV_calculator(JOF_cat, n_chains = 10_000, output = False, n_jobs = 1)
    from galfind import Xi_Ion_Calculator #, M99
    for beta_dust_conv in [None]: #, M99]: #, Reddy18(C00(), 100 * u.Myr), Reddy18(C00(), 300 * u.Myr)]:
        for fesc_conv in [None]: #, "Chisholm22"]: # None, 0.1, 0.2, 0.5,
            xi_ion_calculator = Xi_Ion_Calculator(aper_diams[0], EAZY_fitter.label, beta_dust_conv = beta_dust_conv, fesc_conv = fesc_conv)
            xi_ion_calculator(JOF_cat, n_chains = 10_000, output = False, n_jobs = 1)

    # from galfind import SFR_Halpha_Calculator
    # SFR_Halpha_calculator = SFR_Halpha_Calculator(aper_diams[0], EAZY_fitter)
    # SFR_Halpha_calculator(JOF_cat, n_chains = 10_000, output = False, n_jobs = 1)

    from galfind import Rest_Frame_Property_Kwarg_Selector, Optical_Line_EW_Calculator
    EWrest_calculator = \
        Optical_Line_EW_Calculator(
            aper_diams[0], 
            EAZY_fitter, 
            ["Halpha"], 
            "rest", 
    )
    Ha_emission_band_selector = Rest_Frame_Property_Kwarg_Selector(aper_diams[0], EAZY_fitter, EWrest_calculator, "band", "F444W")
    JOF_cat_new = Ha_emission_band_selector(JOF_cat, return_copy = True)

    from galfind import PSF_Cutout, Galfit_Fitter, Custom_Morphology_Property_Extractor
    band_name = "F444W"
    filt = Filter.from_filt_name(band_name)
    psf_path = f"/nvme/scratch/work/westcottl/psf/PSF_Resample_03_{band_name}.fits"
    psf = PSF_Cutout.from_fits(
        fits_path=psf_path,
        filt=filt,
        unit="adu",
        pix_scale=0.03 * u.arcsec,
        size=0.96 * u.arcsec
    )
    galfit_fitter = Galfit_Fitter(psf, "sersic")
    galfit_fitter(JOF_cat_new, plot = False)

    sersic_extractor = Custom_Morphology_Property_Extractor("n", r"$n_{\mathrm{Sersic}}$", galfit_fitter)
    r_eff_extractor = Custom_Morphology_Property_Extractor("r_e", r"$r_{\mathrm{eff}}~/~\mathrm{pix}$", galfit_fitter)
    axis_ratio_extractor = Custom_Morphology_Property_Extractor("axr", r"$b/a$", galfit_fitter)

    # remove high r_e sources
    JOF_cat_new.gals = [gal for gal in JOF_cat_new if r_eff_extractor.extract_vals(gal) < 100. * u.pix]
    #breakpoint()

    # from galfind.Property_calculator import Redshift_Extractor
    # z_extractor = Redshift_Extractor(aper_diams[0], EAZY_fitter)

    pipes_SED_fit_params = {"fix_z": EAZY_fitter.label, "fesc": None}
    pipes_fitter = Bagpipes(pipes_SED_fit_params)
    pipes_fitter(JOF_cat_new, aper_diams[0], save_PDFs = False, load_SEDs = False, load_PDFs = True, overwrite = False, update = True)

    from galfind.Property_calculator import Custom_SED_Property_Extractor
    from galfind import Property_Multiplier, Property_Divider
    from galfind import Ext_Src_Property_Calculator
    from galfind import UV_Beta_Calculator
    ext_src_Mstar = Ext_Src_Property_Calculator("stellar_mass", "Mstar", aper_diams[0], pipes_fitter.label, ext_src_corrs="F444W", ext_src_uplim=None)
    ext_src_Mstar(JOF_cat_new)
    ext_src_MUV = Ext_Src_Property_Calculator("M_UV", "MUV", aper_diams[0], pipes_fitter.label)
    ext_src_MUV(JOF_cat_new)
    beta_extractor = UV_Beta_Calculator(aper_diams[0], pipes_fitter.label)
    beta_extractor(JOF_cat_new)
    stellar_mass_extractor = Custom_SED_Property_Extractor("stellar_mass_extsrc_F444W", r"$\log_{10}(M_{\star}~/~\mathrm{M}_{\odot})$", aper_diams[0], pipes_fitter.label)
    MUV_extractor = Custom_SED_Property_Extractor("M_UV_extsrc_UV<10", r"$M_{\mathrm{UV}}$", aper_diams[0], pipes_fitter.label)
    xi_ion_extractor = Custom_SED_Property_Extractor("xi_ion_caseB", r"$\xi_{\mathrm{ion}}~/~\mathrm{Hz}~\mathrm{erg}^{-1}$", aper_diams[0], pipes_fitter.label)
    SED_beta_extractor = Custom_SED_Property_Extractor("beta_C94", r"$\beta$", aper_diams[0], pipes_fitter.label)
    specific_xi_ion_calculator = Property_Divider(
        [xi_ion_extractor, stellar_mass_extractor], 
        plot_name = r"$\xi_{\mathrm{ion}}/M_{\star}~[\mathrm{Hz}~\mathrm{erg}^{-1}~\mathrm{M}_{\odot}^{-1}]$"
    )

    for x, y, c in zip(
        [beta_extractor],#, stellar_mass_extractor, MUV_extractor, stellar_mass_extractor, MUV_extractor], # stellar_mass_extractor, MUV_extractor, stellar_mass_extractor, MUV_extractor, beta_extractor, beta_extractor, 
        [stellar_mass_extractor], #, xi_ion_extractor, xi_ion_extractor, xi_ion_extractor, xi_ion_extractor], # xi_ion_extractor, xi_ion_extractor, xi_ion_extractor, xi_ion_extractor, xi_ion_extractor, xi_ion_extractor,
        [xi_ion_extractor], #, SED_beta_extractor, SED_beta_extractor, beta_extractor, beta_extractor] # MUV_extractor, stellar_mass_extractor, beta_extractor, beta_extractor, r_eff_extractor, sersic_extractor, 
    ):
        fig, ax = plt.subplots()
        JOF_cat_new.plot(
            x,
            y,
            fig = fig, 
            ax = ax,
            log_x = False,
            log_y = False, #True,
            incl_x_errs = False,
            incl_y_errs = False,
            save = False,
            plot_type = "contour",
            cmap = "teal",
            plot_kwargs = {},
        )
        JOF_cat_new.plot(
            x,
            y,
            c_calculator = c,
            fig = fig, 
            ax = ax,
            log_x = False,
            log_y = False, #True,
            log_c = True,
            incl_x_errs = False,
            incl_y_errs = False,
            save = True,
            plot_type = "individual",
            #cmap = "viridis",
            plot_kwargs = {},
        )
    
    # from galfind.Property_calculator import Redshift_Extractor
    # z_calculator = Redshift_Extractor(aper_diams[0], EAZY_fitter)
    # # Plot EAZY redshift on x axis and xi_ion on y axis
    # plt.style.use(f"{config['DEFAULT']['GALFIND_DIR']}/galfind_style.mplstyle")
    # fig, ax = plt.subplots()
    # JOF_cat.plot(z_calculator, xi_ion_calculator, incl_x_errs = False, incl_y_errs = False, annotate = False, plot_type = "individual", save = False, fig = fig, ax = ax)
    # plot_kwargs = {
    #     "mfc": "gray",
    #     "marker": "D",
    #     "ms": 8.0,
    #     "mew": 2.0,
    #     "mec": "black",
    #     "ecolor": "black",
    #     "elinewidth": 2.0,
    # }
    # JOF_cat.plot(z_calculator, xi_ion_calculator, incl_x_errs = False, incl_y_errs = True, annotate = True, plot_type = "stacked", plot_kwargs = plot_kwargs, fig = fig, ax = ax)

    # from galfind import MUV_Calculator
    # MUV_calculator = MUV_Calculator(aper_diams[0], EAZY_fitter.label)
    # MUV_calculator(JOF_cat, n_chains = 10_000, output = False, n_jobs = 1)

    # fig, ax = plt.subplots()
    # JOF_cat.plot(MUV_calculator, xi_ion_calculator, incl_x_errs = False, incl_y_errs = False, annotate = False, plot_type = "individual", save = False, fig = fig, ax = ax)
    # plot_kwargs = {
    #     "mfc": "gray",
    #     "marker": "D",
    #     "ms": 8.0,
    #     "mew": 2.0,
    #     "mec": "black",
    #     "ecolor": "black",
    #     "elinewidth": 2.0,
    # }
    # JOF_cat.plot(MUV_calculator, xi_ion_calculator, incl_x_errs = True, incl_y_errs = True, annotate = True, plot_type = "stacked", plot_kwargs = plot_kwargs, fig = fig, ax = ax)


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

def test_data_load():
    JOF_data = Data.pipeline(
        survey,
        version,
        instrument_names = instrument_names,
        version_to_dir_dict = morgan_version_to_dir,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        min_flux_pc_err = min_flux_pc_err
    )

# def update_data_names():
#     from astropy.io import fits
#     for band in ["F435W", "F606W", "F775W", "F814W", "F850LP"]:
#         for (ext, ext_name) in zip(["SCI", "WHT", "RMS"], ["drz", "wht", "rms_err"]):
#             im_name = f"{band}_{ext}.fits"
#             print(f"{band}_{ext}: {im_name}")
#             path = f"/raid/scratch/data/hst/JADES-Deep-GS/ACS_WFC/mosaic_1084_wisptemp2/30mas/ACS_WFC_{band.lower()}_JADES-Deep-GS_{ext_name}.fits"
#             #path = f"/raid/scratch/data/hst/JADES-Deep-GS/ACS_WFC/mosaic_1084_wisptemp2/30mas/ACS_WFC_{band}_JADES-Deep-GS_{ext_name}.fits"
#             hdul = fits.open(path)
#             breakpoint()
#             hdul[0].header["EXTNAME"] = ext
#             hdul.writeto(path, overwrite = True)

if __name__ == "__main__":
    #update_data_names()
    #test_load()
    #main()
    #import time
    #time.sleep((8 * u.hr).to(u.s).value)
    #test_selection()

    test_euclid_filters()

    #test_UVLF()

    #split_UVLF_by_beta()
    #test_pipes()
    #check_multinest()

    #test_plotting()

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
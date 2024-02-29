#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:10:06 2023

@author: u92876da
"""

# mock_cat_creator.py
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from pathlib import Path
import matplotlib as mpl
from astropy.table import Table, vstack, join
from scipy.interpolate import griddata, RectBivariateSpline, SmoothBivariateSpline, UnivariateSpline
from scipy.stats import binned_statistic
from matplotlib.ticker import ScalarFormatter
from copy import deepcopy
import glob
import h5py
import os
from scipy.interpolate import interp1d
from tqdm import tqdm
from galfind import Mock_SED_rest, Mock_SED_obs, config, NIRCam, ACS_WFC, Data, Catalogue, astropy_cosmo, Mock_Photometry, GALFIND_Catalogue_Creator, Photometry_rest, Emission_line, DLA
from galfind import useful_funcs_austind as funcs

plt.style.use(f"{config['DEFAULT']['GALFIND_DIR']}/galfind_style.mplstyle")

def moving_average(x, n = 3):
    ret = np.cumsum(x, dtype = float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def make_mock_galfind_cat(survey, instrument_name, n_gals, z_prior, beta_prior, m_UV_prior, out_path, version = "v9", aper_diam = 0.32 * u.arcsec, scatter_size = 1):
    # make instrument and load observational depths
    excl_bands_survey = {"JADES-Deep-GS": ["f182M", "f210M", "f430M", "f460M", "f480M"], "NGDEEP": ["f435W", "f775W", "f850LP"], "NEP-1": [], "NEP-2": [], "NEP-3": [], "NEP-4": [], \
                          "CLIO": [], "El-Gordo": [], "MACS-0416": [], "GLASS": [], "SMACS-0723": [], "CEERSP1": [], "CEERSP2": [], "CEERSP3": [], "CEERSP4": [], "CEERSP5": [], \
                              "CEERSP6": [], "CEERSP7": [], "CEERSP8": [], "CEERSP9": [], "CEERSP10": []}
    survey_bands_depths = Data.from_pipeline(survey, version, instruments = instrument_name.split("+"), excl_bands = excl_bands_survey[survey]).load_depths(aper_diam)
    survey_bands = [key for key in survey_bands_depths.keys()]
    survey_depths = [depth for depth in survey_bands_depths.values()]
    if instrument_name == "ACS_WFC+NIRCam":
        instrument = NIRCam(excl_bands = [band for band in NIRCam().bands if not band in survey_bands]) \
        + ACS_WFC(excl_bands = [band for band in ACS_WFC().bands if not band in survey_bands])
    elif instrument_name == "NIRCam":
        instrument = NIRCam(excl_bands = [band for band in NIRCam().bands if not band in survey_bands])
        
    if not Path(out_path).is_file():
        # construct m_UV prior from catalogue depths
        # mu, sigma, min_m_UV, max_m_UV = 0., 1., 24., funcs.five_to_n_sigma_mag(np.max(survey_depths), 3.)
        # print(max_m_UV)
        # mag_arr = np.random.lognormal(mu, sigma, 1_000_000)
        # m_UV_prior = np.array([mag for mag in max_m_UV - (mag_arr - np.exp(-sigma ** 2)) if mag > min_m_UV and mag < max_m_UV + np.exp(-sigma ** 2)])
        
        z_in = np.random.choice(z_prior, n_gals)
        beta_in = np.random.choice(beta_prior, n_gals)
        m_UV_in = np.random.choice(m_UV_prior, n_gals)
        M_UV_in = m_UV_in - 5 * np.log10(astropy_cosmo.luminosity_distance(z_in).to(u.pc).value / 10) + 2.5 * np.log10(1 + z_in)
        #print(z_in, beta_in, m_UV_in)
        phot_arr = np.zeros((len(z_in), scatter_size), dtype = Mock_Photometry)
        for i, (z, beta, m_UV, M_UV) in tqdm(enumerate(zip(z_in, beta_in, m_UV_in, M_UV_in)), total = n_gals, desc = f"Making mock photometry for {survey}"):
            sed_obs = Mock_SED_obs.power_law_from_beta_M_UV(z, beta, M_UV)
            sed_obs.create_mock_phot(instrument, survey_depths)
            #fig, ax = plt.subplots()
            #sed_obs.mock_photometry.plot_phot(ax, mag_units = u.Jy)
            #sed_obs.plot_SED(ax, mag_units = u.Jy)
            sed_obs.mock_photometry.scatter(size = scatter_size)
            for j, scattered_phot in enumerate(sed_obs.mock_photometry.scattered_phot):
                phot_arr[i, j] = scattered_phot
                #phot.plot_phot(ax, mag_units = u.Jy)
        phot_arr = np.array(phot_arr).flatten()
        if scatter_size != 1:
            z_in = np.array([np.full(scatter_size, z) for z in z_in]).flatten()
            beta_in = np.array([np.full(scatter_size, beta) for beta in beta_in]).flatten()
            m_UV_in = np.array([np.full(scatter_size, m_UV) for m_UV in m_UV_in]).flatten()
            M_UV_in = np.array([np.full(scatter_size, M_UV) for M_UV in M_UV_in]).flatten()
        # make catalogue out of array of scattered photometries
        output_fluxes = {f"FLUX_APER_{band}_aper_corr_Jy": [np.concatenate(([phot.flux_Jy[i].value], np.zeros(4))) for phot in phot_arr] for i, band in enumerate(instrument)}
        output_flux_errs = {f"FLUXERR_APER_{band}_loc_depth_10pc_Jy": [np.concatenate(([phot.flux_Jy_errs[i].value], np.zeros(4))) for phot in phot_arr] for i, band in enumerate(instrument)}
        out_tab = Table({**{"ID": range(len(z_in)) + np.ones(len(z_in)), "beta_int": beta_in, "m_UV_int": m_UV_in, "M_UV_int": M_UV_in, "z_int": z_in}, **output_fluxes, **output_flux_errs})
        out_tab.write(out_path, overwrite = True)
    else:
        out_tab = Table.read(out_path)
    return out_tab, instrument

def make_mock_line_bias_galfind_cat(line_name, n_gals, m_UV, line_flux, z_in, beta, out_path, version = "v9", aper_diam = 0.32 * u.arcsec):

    survey_bands = ["f090W", "f115W", "f150W", "f200W", "f277W", "f356W", "f410M", "f444W"]
    survey_depths = [30. for band in survey_bands]
    instrument = NIRCam(excl_bands = [band for band in NIRCam().bands if not band in survey_bands])
        
    if not Path(out_path).is_file():
        beta_in = np.full(n_gals, beta)
        m_UV_in = np.full(n_gals, m_UV)
        #M_UV_in = m_UV_in - 5 * np.log10(astropy_cosmo.luminosity_distance(z_in).to(u.pc).value / 10) + 2.5 * np.log10(1 + z_in)
        line_flux_in = np.full(n_gals, line_flux)
        line = Emission_line(line_name, line_flux, 150. * u.km / u.s)
        mock_sed_rest = Mock_SED_rest.power_law_from_beta_m_UV(beta, m_UV, wav_res = 0.1)
        mock_sed_rest.add_emission_lines([line])
        EW_in = np.full(n_gals, mock_sed_rest.calc_line_EW(line_name))
        #print(z_in, beta_in, m_UV_in)
        phot_arr = np.zeros(n_gals, dtype = Mock_Photometry)
        for i, z in tqdm(enumerate(z_in), total = n_gals, desc = f"Making mock {line_name} photometry"):
            mock_sed_obs = Mock_SED_obs.from_Mock_SED_rest(mock_sed_rest, z)
            mock_sed_obs.create_mock_phot(instrument, survey_depths)
            phot_arr[i] = mock_sed_obs.mock_photometry
        # make catalogue out of array of scattered photometries
        output_fluxes = {f"FLUX_APER_{band}_aper_corr_Jy": [np.concatenate(([phot.flux_Jy[i].value], np.zeros(4))) for phot in phot_arr] for i, band in enumerate(instrument)}
        output_flux_errs = {f"FLUXERR_APER_{band}_loc_depth_10pc_Jy": [np.concatenate(([phot.flux_Jy_errs[i].value], np.zeros(4))) for phot in phot_arr] for i, band in enumerate(instrument)}
        out_tab = Table({**{"ID": range(len(z_in)) + np.ones(len(z_in)), "beta_int": beta_in, "m_UV_int": m_UV_in, f"{line_name}_line_flux_rest_int": line_flux_in, f"{line_name}_EW_rest_int": EW_in, "z_int": z_in}, **output_fluxes, **output_flux_errs})
        out_tab.write(out_path, overwrite = True)
    else:
        out_tab = Table.read(out_path)
    return out_tab, instrument

def make_mock_DLA_bias_galfind_cat(N_HI, Doppler_b, n_gals, m_UV, z_in, beta, out_path, version = "v9", aper_diam = 0.32 * u.arcsec, plot = False):

    survey_bands = ["f090W", "f115W", "f150W", "f200W", "f277W", "f356W", "f410M", "f444W"]
    survey_depths = [30. for band in survey_bands]
    instrument = NIRCam(excl_bands = [band for band in NIRCam().bands if not band in survey_bands])
    DLA_obj = DLA(N_HI, Doppler_b)
    
    if not Path(out_path).is_file():
        beta_in = np.full(n_gals, beta)
        m_UV_in = np.full(n_gals, m_UV)
        N_HI_in = np.full(n_gals, N_HI)
        Doppler_b_in = np.full(n_gals, Doppler_b)
        mock_sed_rest = Mock_SED_rest.power_law_from_beta_m_UV(beta, m_UV, wav_res = 0.1)
        #print(z_in, beta_in, m_UV_in)
        phot_arr = np.zeros(n_gals, dtype = Mock_Photometry)
        for i, z in tqdm(enumerate(z_in), total = n_gals, desc = "Making mock DLA photometry"):
            mock_sed_obs = Mock_SED_obs.from_Mock_SED_rest(mock_sed_rest, z)
            if i == 0 and plot:
                fig, ax = plt.subplots()
                mock_sed_obs.plot_SED(ax, mag_units = u.Jy)
            mock_sed_obs.add_DLA(DLA_obj)
            if i == 0 and plot:
                mock_sed_obs.plot_SED(ax, mag_units = u.Jy)
                mock_sed_obs.create_mock_phot(instrument, survey_depths)
                mock_sed_obs.mock_photometry.plot_phot(ax, mag_units = u.Jy)
                ax.set_xlim(1200. * (1 + z), 3000. * (1 + z))
                ax.axvline(1250. * (1 + z), c = "blue", ls = "--")
                plt.show()
            mock_sed_obs.create_mock_phot(instrument, survey_depths)
            phot_arr[i] = mock_sed_obs.mock_photometry
        # make catalogue out of array of scattered photometries
        output_fluxes = {f"FLUX_APER_{band}_aper_corr_Jy": [np.concatenate(([phot.flux_Jy[i].value], np.zeros(4))) for phot in phot_arr] for i, band in enumerate(instrument)}
        output_flux_errs = {f"FLUXERR_APER_{band}_loc_depth_10pc_Jy": [np.concatenate(([phot.flux_Jy_errs[i].value], np.zeros(4))) for phot in phot_arr] for i, band in enumerate(instrument)}
        out_tab = Table({**{"ID": range(len(z_in)) + np.ones(len(z_in)), "beta_int": beta_in, "m_UV_int": m_UV_in, "N_HI_int": N_HI_in, "Doppler_b_int": Doppler_b_in, "z_int": z_in}, **output_fluxes, **output_flux_errs})
        out_tab.write(out_path, overwrite = True)
    else:
        out_tab = Table.read(out_path)
    return out_tab, instrument
    
def split_mock_galfind_tab(tab, instrument, out_path):
    # only include photometry from below 3000 Angstrom to avoid non-power law SED
    cat_bands = []
    z_min_dict = {}
    for i, band in enumerate(instrument):
        band_wav = instrument.band_wavelengths[band]
        band_FWHM = instrument.band_FWHMs[band]
        z_min = ((band_wav + band_FWHM / 2.) / (3000. * u.AA)).to(u.dimensionless_unscaled).value - 1.
        z_min_dict[band] = z_min
        print(band, band_wav, z_min)
        if i != 0:
            if z_min > 6.5 and z_min_dict[instrument[i - 1]] < 13.:
                cat_bands.append(list(instrument[:i]))
    cat_z_lims = [[list(z_min_dict.values())[list(z_min_dict.keys()).index(bands[-1])], list(z_min_dict.values())[list(z_min_dict.keys()).index(bands[-1]) + 1]] for bands in cat_bands]
    cat_z_lims[0][0], cat_z_lims[-1][-1] = 6.5, 13.
    print(f"cat_bands = {cat_bands}", cat_z_lims, z_min_dict)
    survey_cat_paths = []
    for i, (bands, z_lims) in enumerate(zip(cat_bands, cat_z_lims)):
        split_out_path = out_path.replace(".fits", f"_split_{str(i + 1)}.fits")
        survey_cat_paths.append(split_out_path)
        #if not Path(split_out_path).is_file():
        z_split_tab = tab[tab["z_int"] > z_lims[0]]
        z_split_tab = z_split_tab[z_split_tab["z_int"] < z_lims[1]]
        #print(len(z_split_tab))
        z_split_tab.remove_columns([f"FLUX_APER_{band}_aper_corr_Jy" for band in instrument if band not in bands])
        z_split_tab.remove_columns([f"FLUXERR_APER_{band}_loc_depth_10pc_Jy" for band in instrument if band not in bands])
        z_split_tab.rename_column("ID", "NUMBER")
        z_split_tab["ALPHA_J2000"] = 0.
        z_split_tab["DELTA_J2000"] = 0.
        z_split_tab.write(split_out_path, overwrite = True)
        print(split_out_path, np.min(z_split_tab["beta_int"]), np.max(z_split_tab["beta_int"]), np.min(z_split_tab["z_int"]), np.max(z_split_tab["z_int"]))
    return survey_cat_paths
        
def run_beta_bias_through_EAZY(survey, survey_cat_paths, instrument_in, aper_diam = 0.32 * u.arcsec, min_pc_err = 10., version = "beta_bias_v9"):
    cat_creator = GALFIND_Catalogue_Creator("loc_depth", aper_diam, min_pc_err)
    code_names = ["EAZY"]
    eazy_templates = ["fsps_larson"]
    eazy_lowz_zmax = [[6.]]
    out_cats = []
    for path in survey_cat_paths:
        instrument = deepcopy(instrument_in)
        survey_name = path.split("/")[-1].replace("_beta_bias_split", "").replace(".fits", "")
        print(f"survey_name = {survey_name}, path={path}")
        out_cats.append(Catalogue.from_fits_cat(path, version, instrument, cat_creator, code_names, survey_name, lowz_zmax_arr = eazy_lowz_zmax, templates_arr = eazy_templates, data = None, mask = False, excl_bands = []))
    return out_cats

def calc_obs_UV_properties(fits_cat_path, survey, instrument_name, aper_diam = 0.32 * u.arcsec, min_pc_err = 10., version = "beta_bias_v9"):
    # load galfind catalogue from fits_cat_path
    cat_creator = GALFIND_Catalogue_Creator("loc_depth", aper_diam, min_pc_err)
    if instrument_name == "NIRCam":
        instrument = NIRCam()
    elif instrument_name == "ACS_WFC+NIRCam":
        instrument = NIRCam() + ACS_WFC()
    galfind_cat = Catalogue.from_fits_cat(fits_cat_path, version, instrument, cat_creator, ["EAZY"], survey, [None], templates_arr = ["fsps_larson"], data = None, mask = False)
    beta_arr = np.zeros(len(galfind_cat))
    m_UV_arr = np.zeros(len(galfind_cat))
    M_UV_arr = np.zeros(len(galfind_cat))
    for i, gal in tqdm(enumerate(galfind_cat), total = len(galfind_cat), desc = f"Calculating UV properties from {galfind_cat.survey}"):
        phot = gal.phot.SED_results['EAZY']['fsps_larson'].phot_rest[Photometry_rest.rest_UV_wavs_name([1250., 3000.] * u.AA)]
        try:
            beta, m_UV = phot.basic_beta_m_UV_calc(1.)
            M_UV = m_UV - 5 * np.log10(phot.lum_distance.value / 10) + 2.5 * np.log10(1 + phot.z)
        except:
            beta, m_UV, M_UV = -99., -99., -99.
        beta_arr[i] = beta
        m_UV_arr[i] = m_UV
        M_UV_arr[i] = M_UV
    return beta_arr, m_UV_arr, M_UV_arr

def pure_power_law_beta_bias(surveys_arr = [["CEERSP9"] * 2, []], beta_in = -3.):
    #[["NGDEEP", "CEERSP2", "NEP-1", "CEERSP9"], ["MACS-0416", "GLASS", "SMACS-0723", "CLIO", "El-Gordo"]] # "JADES-Deep-GS", 
    print(surveys_arr)
    instrument_names = ["ACS_WFC+NIRCam", "NIRCam"]
    for surveys, instrument_name in zip(surveys_arr, instrument_names):
        for j, survey in enumerate(surveys):
            excl_bands_survey = {"JADES-Deep-GS": ["f182M", "f210M", "f430M", "f460M", "f480M"], "NGDEEP": ["f435W", "f775W", "f850LP"], "NEP-1": [], "NEP-2": [], "NEP-3": [], "NEP-4": [], \
                                  "CLIO": [], "El-Gordo": [], "MACS-0416": [], "GLASS": [], "SMACS-0723": [], "CEERSP1": [], "CEERSP2": [], "CEERSP3": [], "CEERSP4": [], "CEERSP5": [], \
                                      "CEERSP6": [], "CEERSP7": [], "CEERSP8": [], "CEERSP9": [], "CEERSP10": []}
            version = "v9"
            aper_diam = 0.32 * u.arcsec
            survey_bands_depths = Data.from_pipeline(survey, version, instruments = instrument_name.split("+"), excl_bands = excl_bands_survey[survey]).load_depths(aper_diam)
            survey_bands = [key for key in survey_bands_depths.keys()]
            survey_depths = [depth for depth in survey_bands_depths.values()]
            #print(survey, survey_bands_depths, funcs.five_to_n_sigma_mag(np.max(survey_depths), 3.))
            number = len(np.where(np.array(surveys[:j]) == survey)[0])
            
            out_path = f"{config['DEFAULT']['GALFIND_WORK']}/Beta_paper/beta_bias_pure_power_law/{instrument_name}/{survey}_beta={beta_in:.1f}.fits"
            os.makedirs("/".join(out_path.split("/")[:-1]), exist_ok = True)
            out_path = out_path.replace(".fits", f"_{str(number)}.fits")
            #print(out_path)
            n_gals = 10_000
            scatter_size = 10
            beta_prior = np.full(1_000_000, beta_in)
            m_UV_prior = np.linspace(26., 30., 1_000_000)
            z_prior = np.linspace(6.5, 13., 1_000_000)
            # make catalogue
            out_tab, instrument = make_mock_galfind_cat(survey, instrument_name, n_gals, z_prior, beta_prior, m_UV_prior, out_path, scatter_size = scatter_size)
            print(survey, number, beta_in, np.min(out_tab["beta_int"]), np.max(out_tab["beta_int"]))
            #survey_cat_paths = split_mock_galfind_tab(out_tab, instrument, out_path)
            #run_beta_bias_through_EAZY(survey, survey_cat_paths, instrument)
            # perform selection (DONE IN A SEPARATE SCRIPT!!!)
            
            select_cat_paths = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Catalogues/beta_bias_v9/{instrument_name}/{survey}_beta={beta_in:.1f}_{int(number)}*/{survey}*matched_selection.fits")
            print(select_cat_paths)
            select_cats = []
            for i, path in enumerate(select_cat_paths):
                #if not Path(path.replace(".fits", "_final.fits")).is_file():
                    survey_loc = path.split("/")[-2] #f"{survey}_{str(i + 1)}"
                    print(survey_loc)
                    select_tab = Table.read(path)
                    # load in enitre photometry
                    select_tab = join(left = out_tab, right = select_tab, keys_left = "ID", keys_right = "NUMBER", join_type = "inner")
                    for band in instrument:
                        try:
                            del select_tab[f"FLUX_APER_{band}_aper_corr_Jy_2"]
                            del select_tab[f"FLUXERR_APER_{band}_loc_depth_10pc_Jy_2"]
                            select_tab.rename_column(f"FLUX_APER_{band}_aper_corr_Jy_1", f"FLUX_APER_{band}_aper_corr_Jy")
                            select_tab.rename_column(f"FLUXERR_APER_{band}_loc_depth_10pc_Jy_1", f"FLUXERR_APER_{band}_loc_depth_10pc_Jy")
                        except:
                            pass
                    for name in ["ID"]:
                        del select_tab[name]
                    for name in ["z_int", "beta_int", "m_UV_int", "M_UV_int"]:
                        del select_tab[f"{name}_2"]
                        select_tab.rename_column(f"{name}_1", name)
                    select_tab = select_tab[select_tab["final_sample_highz_fsps_larson"] == True]
                    select_tab.write(path.replace(".fits", "_final.fits"), overwrite = True)
                    #old_tab = Table.read(path.replace(".fits", "_final.fits"))
                    #print(f"OLD: beta = {np.min(old_tab['beta_int'])}, {np.max(old_tab['beta_int'])}; z_in = {np.min(old_tab['z_int'])}, {np.max(old_tab['z_int'])}")
                    print(f"NEW: beta = {np.min(select_tab['beta_int'])}, {np.max(select_tab['beta_int'])}; z_in = {np.min(select_tab['z_int'])}, {np.max(select_tab['z_int'])}")
                    beta, m_UV, M_UV = calc_obs_UV_properties(path.replace(".fits", "_final.fits"), survey_loc, instrument_name, version = f"beta_bias_{version}")
                    select_tab["beta_obs"] = beta
                    select_tab["m_UV_obs"] = m_UV
                    select_tab["M_UV_obs"] = M_UV
                    select_tab.write(path.replace(".fits", "_final.fits"), overwrite = True)
                    select_cats.append(select_tab)
                #else:
                #    select_cats.append(Table.read(path.replace(".fits", "_final.fits")))
            # vstack selection catalogues
            #if not Path(out_path.replace(".fits", "_merged.fits")).is_file():
            final_tab = vstack([Table.read(path.replace(".fits", "_final.fits")) for path in select_cat_paths])
            final_tab["6.5<z<13_selected"] = [True if (row["zbest_fsps_larson"] < 13. and row["zbest_fsps_larson"] > 6.5) else False for row in final_tab]
            final_tab.write(out_path.replace(".fits", "_merged.fits"), overwrite = True)
            print(out_path)
            #else:
            #    final_tab = Table.read(out_path.replace(".fits", "_merged.fits"))
            
            # produce paper-ready plots

def line_beta_bias(line_name = "Lya"):
    EW_rest_dict = {"Lya": [5., 10., 20., 30., 40., 50., 75., 100., 150., 200., 300.], "CIV-1549": [1., 2., 3., 4., 5., 10., 15., 20., 25.], \
                    "CIII]-1909": [1., 2., 3., 4., 5., 10., 15., 20., 25.], "OIII]-1665": [1., 2., 3., 4., 5., 10., 15., 20., 25.], \
                    "HeII-1640": [1., 2., 3., 4., 5., 10., 15., 20., 25.]}
    EW_rest = EW_rest_dict[line_name] * u.AA
    n_gals = 10_000
    beta_intrinsic = -2.5
    m_UV_intrinsic = 26.
    version = "beta_bias_v9"
    aper_diam = 0.32 * u.arcsec
    survey = f"8_NIRCam_beta={beta_intrinsic:.1f}" # should have been an f-string earlier
    plot = True
    plot_tabs = []
    for EW_target in EW_rest:
        line_flux = 1e-16 * u.erg / (u.s * u.cm ** 2)
        i = 0
        z = 9.
        while True:
            #print(i)
            line = Emission_line(line_name, line_flux, 150. * u.km / u.s)
            mock_sed_rest = Mock_SED_rest.power_law_from_beta_m_UV(beta_intrinsic, m_UV_intrinsic, wav_res = 0.1)
            mock_sed_rest.add_emission_lines([line])
            mock_sed_obs = Mock_SED_obs.from_Mock_SED_rest(mock_sed_rest, z)
            EW = mock_sed_obs.calc_line_EW(line_name) / (1 + z)
            print(mock_sed_rest.calc_line_EW(line_name), EW)
            if abs(EW - EW_target) < 0.1 * u.AA:
                break
            line_flux *= EW_target / EW
            i += 1
        #print(EW, line_flux)
            
        out_path = f"{config['DEFAULT']['GALFIND_WORK']}/Beta_paper/beta_bias_pure_power_law/NIRCam/{survey}_beta_bias_{line_name}_{int(EW_target.value)}AA.fits"
        #print(out_path)
        os.makedirs("/".join(out_path.split("/")[:-1]), exist_ok = True)
        z_arr = np.linspace(6.5, 13., n_gals)
        # make catalogue
        out_tab, instrument = make_mock_line_bias_galfind_cat(line_name, n_gals, m_UV_intrinsic, line_flux, z_arr, beta_intrinsic, out_path, version = version, aper_diam = aper_diam)
        #survey_cat_paths = split_mock_galfind_tab(out_tab, instrument, out_path)
        #run_beta_bias_through_EAZY(survey, survey_cat_paths, instrument, version = version)
        # no need for selection if the sources are so bright compared to the depths
        
        select_cat_paths = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Catalogues/beta_bias_v9/NIRCam/{survey}*/{survey}*{line_name}_{str(int(EW_target.value))}AA*matched.fits")
        #print(select_cat_paths)
        select_cats = []
        for i, path in enumerate(select_cat_paths):
            if not Path(path.replace(".fits", "_final.fits")).is_file():
                survey_loc = f"{survey}_beta_bias_{line_name}_{str(int(EW_target.value))}AA_split_{str(i + 1)}"
                select_tab = Table.read(path)
                # load in enitre photometry
                select_tab = join(left = out_tab, right = select_tab, keys_left = "ID", keys_right = "NUMBER", join_type = "inner")
                for band in instrument:
                    try:
                        del select_tab[f"FLUX_APER_{band}_aper_corr_Jy_2"]
                        del select_tab[f"FLUXERR_APER_{band}_loc_depth_10pc_Jy_2"]
                        select_tab.rename_column(f"FLUX_APER_{band}_aper_corr_Jy_1", f"FLUX_APER_{band}_aper_corr_Jy")
                        select_tab.rename_column(f"FLUXERR_APER_{band}_loc_depth_10pc_Jy_1", f"FLUXERR_APER_{band}_loc_depth_10pc_Jy")
                    except:
                        pass
                for name in ["ID"]:
                    del select_tab[name]
                for name in ["z_int", "beta_int", "m_UV_int", f"{line_name}_line_flux_rest_int", f"{line_name}_EW_rest_int"]:
                    del select_tab[f"{name}_2"]
                    select_tab.rename_column(f"{name}_1", name)
                #select_tab = select_tab[select_tab["final_sample_highz_fsps_larson"] == True]
                select_tab.write(path.replace(".fits", "_final.fits"), overwrite = True)
                beta, m_UV, M_UV = calc_obs_UV_properties(path.replace(".fits", "_final.fits"), survey_loc, "NIRCam", version = version)
                select_tab["beta_obs"] = beta
                select_tab["m_UV_obs"] = m_UV
                select_tab["M_UV_obs"] = M_UV
                select_tab.write(path.replace(".fits", "_final.fits"), overwrite = True)
                select_cats.append(select_tab)
            else:
                select_cats.append(Table.read(path.replace(".fits", "_final.fits")))
        # vstack selection catalogues
        if not Path(out_path.replace(".fits", "_merged.fits")).is_file():
            final_tab = vstack([Table.read(path.replace(".fits", "_final.fits")) for path in select_cat_paths])
            final_tab["6.5<z<13_selected"] = [True if (row["zbest_fsps_larson"] < 13. and row["zbest_fsps_larson"] > 6.5) else False for row in final_tab]
            final_tab.write(out_path.replace(".fits", "_merged.fits"), overwrite = True)
        else:
            final_tab = Table.read(out_path.replace(".fits", "_merged.fits"))
        plot_tabs.append(final_tab)
        
    # make paper ready plots
    if plot:
        for z_range in [[6.5, 13.]]:
            fig, ax = plt.subplots()
            assert(len(plot_tabs) == len(EW_rest))
            # plot_interped_line_bias(ax, plot_tabs, EW_rest, line_name)
            for i, (tab, EW) in enumerate(zip(reversed(plot_tabs), reversed(EW_rest))):
                if i == len(plot_tabs) - 1:
                    show = True
                else:
                    show = False
                plot_line_bias(ax, tab, z_range, EW, line_name = line_name, show = show)
            
def plot_line_bias(ax, tab, z_range, EW, line_name = "Lya", moving_average_n = 100, show = False, save = False):
    tab = tab[tab["beta_obs"] != -99.]
    #tab = tab[((tab["z_int"] > z_range[0]) & (tab["z_int"] < z_range[1]))]
    #print(EW)
    z_int = moving_average(tab["z_int"], moving_average_n)
    delta_beta = moving_average(tab["beta_obs"] - tab["beta_int"], moving_average_n)
    ax.plot(z_int, delta_beta, label = line_name.replace('-', r' $\lambda$')) #" EW = {EW}")
    
    if show:
        ax.set_xlim(z_range[0], z_range[1])
        ax.set_xlabel("Input redshift, z")
        ax.set_ylabel(r"$\Delta\beta$")
        plt.legend(fontsize = 9.)
    if save:
        plt.savefig("/raid/scratch/work/austind/GALFIND_WORK/Bagpipes/EPOCHS_III/plots/beta_bias/UV_line_bias.png")
    if show:
        plt.show()
        
def plot_interped_line_bias(ax, tab_arr, EW_arr_in, line_name, interp_n = 1_000, moving_average_n = 100, show = True, save = False):
    z_int_arr = np.concatenate(np.array([moving_average(tab["z_int"], moving_average_n) for tab in tab_arr]))
    delta_beta_arr = np.concatenate(np.array([moving_average(tab["beta_obs"] - tab["beta_int"], moving_average_n) for tab in tab_arr]))
    EW_arr = np.concatenate(np.array([moving_average(tab[f"{line_name}_EW_rest_int"], moving_average_n) for tab in tab_arr])) # np.array(EW_arr_in.value) #
    print(z_int_arr.shape, delta_beta_arr.shape, EW_arr.shape)
    z_grid, delta_beta_grid = np.meshgrid(np.linspace(min(z_int_arr), max(z_int_arr), interp_n), np.linspace(-0.2, 0.2, interp_n))
    grid_EW = griddata((z_int_arr, delta_beta_arr), EW_arr, (z_grid, delta_beta_grid), method = "linear")
    im = ax.imshow(grid_EW, extent = (min(z_int_arr), max(z_int_arr), min(delta_beta_arr), max(delta_beta_arr)), origin = "lower", cmap = "viridis", interpolation = "bilinear")
    if save:
        pass
        #plt.savefig()
    if show:
        ax.set_ylim(-0.2, 0.2)
        ax.set_xlabel("Input redshift, z")
        ax.set_ylabel(r"$\Delta\beta$")
        plt.colorbar(im)
        plt.legend()
        plt.show()
        
def EPOCHS_III_line_bias_plot(line_name_arr = [], beta = -2.5, EW = 1. * u.AA):
    survey = f"8_NIRCam_beta={beta:.1f}"
    tab_arr = [Table.read(f"{config['DEFAULT']['GALFIND_WORK']}/Beta_paper/beta_bias_pure_power_law/NIRCam/{survey}_beta_bias_{line_name}_{int(EW.value)}AA_merged.fits") for line_name in line_name_arr]
    fig, ax = plt.subplots()
    for i, (tab, line_name) in enumerate(zip(tab_arr, line_name_arr)):
        if i == len(tab_arr) - 1:
            show = True
            save = True
        else:
            show = False
            save = False
        plot_line_bias(ax, tab, [6.5, 13.], EW, line_name = line_name, show = show, save = save)
    
def EPOCHS_III_imshow_bias_plot(bias_type = "Lya", beta = -2.5, n_bins = 5000, moving_average_n = 100, type_arr = ["dbeta", "epsilon"], cmap_name = "Reds"):
    survey = f"8_NIRCam_beta={beta:.1f}"
    if bias_type == "Lya":
        y_arr = [5., 10., 20., 30., 40., 50., 75., 100., 150., 200., 300.]
        tab_arr = [Table.read(f"{config['DEFAULT']['GALFIND_WORK']}/Beta_paper/beta_bias_pure_power_law/NIRCam/{survey}_beta_bias_Lya_{int(EW)}AA_merged.fits") for EW in y_arr]
        y_ticks = [0., 50., 100., 150., 200., 250., 300.]
        y_label = r"$\mathrm{EW}_{\mathrm{rest}}(\mathrm{Ly}\alpha)$"
    elif bias_type == "DLA":
        y_arr = [21., 21.5, 22., 22.5, 23., 23.5]
        tab_arr = [Table.read(f"{config['DEFAULT']['GALFIND_WORK']}/Beta_paper/beta_bias_pure_power_law/NIRCam/{survey}_beta_bias_DLA_{log_N_HI}cm-2_merged.fits") for log_N_HI in y_arr]
        y_ticks = y_arr #[20., 21., 22., 23.]
        y_label = r"$\log_{10}(N_{\mathrm{HI}}~/~\mathrm{cm}^{-2})$"
    n_gals = len(tab_arr[0]) #+ 2
    print(n_gals)
    #for i, tab in enumerate(tab_arr):
    #    tab_arr[i] = tab[(tab["beta_obs"] != -99.) & (tab["6.5<z<13_selected"] == True)]
    
    fig, ax = plt.subplots(ncols = 3 if bias_type == "Lya" else 2 if bias_type == "DLA" else None, nrows = 1, figsize = (15, 5) if bias_type == "Lya" else (10, 5), sharey = True)
    cmap = mpl.cm.get_cmap(cmap_name)
    cmap.set_bad(color = "black")
    for i, type_ in enumerate(type_arr):
        if type_ == "dbeta":
            if bias_type == "Lya":
                z_range_arr = [[7.1, 7.5], [9.6, 9.9]]
            elif bias_type == "DLA":
                z_range_arr = [[6.5, 13.]]
        elif type_ == "epsilon":
            z_range_arr = [[6.5, 13.]]
        for j, z_range in enumerate(z_range_arr):
            tab_arr_ = [tab_[((tab_["z_int"] >= z_range[0]) & (tab_["z_int"] <= z_range[1]))] for tab_ in tab_arr]
            z_arr = np.array(tab_arr_[0]["z_int"])
            print(z_arr)
            # z_arr = np.linspace(z_range[0], z_range[1], n_gals)
            # print(len(z_arr))
            z_bin_edges = np.histogram_bin_edges(z_arr, n_bins)
            if bias_type == "Lya":
                if type_ == "dbeta":
                    plot_arr = np.ma.array([np.ma.array(tab["beta_obs"] - tab["beta_int"], mask = (tab["beta_obs"] == -99.) | (tab["6.5<z<13_selected"] == False)) for tab in tab_arr_])
                    print(plot_arr)
                    #[list(moving_average(np.zeros(len(z_arr)), moving_average_n))] + [moving_average(interp1d(tab["z_int"], tab["beta_obs"] - tab["beta_int"], fill_value = "extrapolate")(z_arr), moving_average_n) for tab in tab_arr])
                elif type_ == "epsilon":
                    plot_arr = np.ma.array([list(moving_average(np.zeros(len(z_arr)), moving_average_n))] + [moving_average(interp1d(tab["z_int"], ((1. + tab["zbest_fsps_larson"]) / (1. + tab["z_int"])) - 1., fill_value = "extrapolate")(z_arr), moving_average_n) for tab in tab_arr_])
            elif bias_type == "DLA":
                if type_ == "dbeta":
                    plot_arr = np.ma.array([np.ma.array(tab["beta_obs"] - tab["beta_int"], mask = (tab["beta_obs"] == -99.) | (tab["6.5<z<13_selected"] == False)) for tab in tab_arr_])
                    #plot_arr = np.array([moving_average(binned_statistic(tab["z_int"], tab["beta_obs"] - tab["beta_int"], statistic = "median", bins = z_bin_edges)[0], moving_average_n) for tab in tab_arr])
                    #plot_arr = np.array([moving_average(interp1d(tab["z_int"], tab["beta_obs"] - tab["beta_int"], bounds_error = False, fill_value = )(z_arr), moving_average_n) for tab in tab_arr])
                elif type_ == "epsilon":
                    #plot_arr = np.array([moving_average(binned_statistic(tab["z_int"], ((1. + tab["zbest_fsps_larson"]) / (1. + tab["z_int"])) - 1., statistic = "median", bins = z_arr)[0], moving_average_n) for tab in tab_arr])
                    plot_arr = np.ma.array([np.ma.array(((1. + tab["zbest_fsps_larson"]) / (1. + tab["z_int"])) - 1., mask = (tab["beta_obs"] == -99.) | (tab["6.5<z<13_selected"] == False)) for tab in tab_arr_])
            #print(plot_arr, plot_arr.shape)
            ax_to_plot = ax[j if type_ == "dbeta" else -1]
            if bias_type == "Lya":
                norm = mpl.colors.SymLogNorm(1e-1, vmax = 0.) if type_ == "dbeta" else None
            else:
                norm = mpl.colors.SymLogNorm(1e0, vmin = 0.) if type_ == "dbeta" else mpl.colors.Normalize(vmax = 0.15)
            im = ax_to_plot.imshow(plot_arr, extent = [min(z_arr), max(z_arr), 0. if bias_type == "Lya" else min(y_arr), max(y_arr)], cmap = cmap, aspect = "auto", norm = norm)
            ax_to_plot.set_xlabel("Input redshift, z")
            ax_to_plot.set_yticks(y_ticks)
            if type_ == "dbeta":
                cbar_label = r"$\Delta\beta$"
                if bias_type == "Lya":
                    if j == 0:
                        ticks = [0., -0.05, -0.1, -0.25, -0.6]
                    elif j == 1:
                        ticks = [0., -0.01, -0.02, -0.03, -0.04, -0.05]
                elif bias_type == "DLA":
                    ticks = [0., 0.1, 0.2, 0.3, 0.4, 0.5]
            elif type_ == "epsilon":
                cbar_label = r"$\epsilon$"
                if bias_type == "Lya":
                    ticks = [0., -0.05, -0.1, -0.15]
                elif bias_type == "DLA":
                    ticks = [0., 0.05, 0.1, 0.15]
            cbar = fig.colorbar(im, ax = ax_to_plot, ticks = ticks, format = ScalarFormatter(), shrink = 0.9, location = "top", label = cbar_label, orientation = "horizontal")

    for k in [0, -1]:
        ax[k].set_ylabel(y_label, rotation = 270 if k == -1 else 90, labelpad = 20. if k == -1 else 4.)
    ax[-1].yaxis.set_label_position("right")
    ax[-1].tick_params(labelright = True)
    ax[-1].set_xticks([6.5, 8.5, 10, 11.5, 13.])
    plt.subplots_adjust(wspace = 0.12)
    plt.savefig(f"/raid/scratch/work/austind/GALFIND_WORK/Bagpipes/EPOCHS_III/plots/beta_bias/{bias_type}_bias.png")
    plt.show()

def plot_rest_frame_UV_line_bias(line_name_arr = ["CIV-1549", "CIII]-1909", "HeII-1640", "OIII]-1665"]):
    for line_name in line_name_arr:
        line_beta_bias(line_name)
        
def DLA_beta_bias():
    log_N_HI_arr = [23.5, 23., 22.5, 22., 21.5, 21., 20.5, 20.]
    n_gals = 10_000
    beta_intrinsic = -2.5
    m_UV_intrinsic = 26.
    Doppler_b = 150. * u.km / u.s
    version = "beta_bias_v9"
    aper_diam = 0.32 * u.arcsec
    survey = f"8_NIRCam_beta={beta_intrinsic:.1f}" # should have been an f-string earlier
    plot = True
    plot_tabs = []
    for log_N_HI in log_N_HI_arr:
        N_HI = 10 ** (log_N_HI) * (u.cm ** -2)
        out_path = f"{config['DEFAULT']['GALFIND_WORK']}/Beta_paper/beta_bias_pure_power_law/NIRCam/{survey}_beta_bias_DLA_{log_N_HI:.1f}cm-2.fits"
        #print(out_path)
        os.makedirs("/".join(out_path.split("/")[:-1]), exist_ok = True)
        z_arr = np.linspace(6.5, 13., n_gals)
        # make catalogue
        out_tab, instrument = make_mock_DLA_bias_galfind_cat(N_HI, Doppler_b, n_gals, m_UV_intrinsic, z_arr, beta_intrinsic, out_path, version = version, aper_diam = aper_diam)
        survey_cat_paths = split_mock_galfind_tab(out_tab, instrument, out_path)
        run_beta_bias_through_EAZY(survey, survey_cat_paths, instrument, version = version)
        # no need for selection if the sources are so bright compared to the depths
        
        select_cat_paths = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Catalogues/beta_bias_v9/NIRCam/{survey}*/{survey}*_DLA_{log_N_HI:.1f}cm-2*matched.fits")
        #print(select_cat_paths)
        select_cats = []
        for i, path in enumerate(select_cat_paths):
            if not Path(path.replace(".fits", "_final.fits")).is_file():
                survey_loc = f"{survey}_beta_bias_DLA_{log_N_HI:.1f}cm-2_split_{str(i + 1)}"
                select_tab = Table.read(path)
                # load in enitre photometry
                select_tab = join(left = out_tab, right = select_tab, keys_left = "ID", keys_right = "NUMBER", join_type = "inner")
                for band in instrument:
                    try:
                        del select_tab[f"FLUX_APER_{band}_aper_corr_Jy_2"]
                        del select_tab[f"FLUXERR_APER_{band}_loc_depth_10pc_Jy_2"]
                        select_tab.rename_column(f"FLUX_APER_{band}_aper_corr_Jy_1", f"FLUX_APER_{band}_aper_corr_Jy")
                        select_tab.rename_column(f"FLUXERR_APER_{band}_loc_depth_10pc_Jy_1", f"FLUXERR_APER_{band}_loc_depth_10pc_Jy")
                    except:
                        pass
                for name in ["ID"]:
                    del select_tab[name]
                for name in ["z_int", "beta_int", "m_UV_int", "N_HI_int"]:
                    del select_tab[f"{name}_2"]
                    select_tab.rename_column(f"{name}_1", name)
                #select_tab = select_tab[select_tab["final_sample_highz_fsps_larson"] == True]
                select_tab.write(path.replace(".fits", "_final.fits"), overwrite = True)
                beta, m_UV, M_UV = calc_obs_UV_properties(path.replace(".fits", "_final.fits"), survey_loc, "NIRCam", version = version)
                select_tab["beta_obs"] = beta
                select_tab["m_UV_obs"] = m_UV
                select_tab["M_UV_obs"] = M_UV
                select_tab.write(path.replace(".fits", "_final.fits"), overwrite = True)
                select_cats.append(select_tab)
            else:
                select_cats.append(Table.read(path.replace(".fits", "_final.fits")))
        # vstack selection catalogues
        if not Path(out_path.replace(".fits", "_merged.fits")).is_file():
            final_tab = vstack([Table.read(path.replace(".fits", "_final.fits")) for path in select_cat_paths])
            final_tab["6.5<z<13_selected"] = [True if (row["zbest_fsps_larson"] < 13. and row["zbest_fsps_larson"] > 6.5) else False for row in final_tab]
            final_tab.write(out_path.replace(".fits", "_merged.fits"), overwrite = True)
        else:
            final_tab = Table.read(out_path.replace(".fits", "_merged.fits"))
        plot_tabs.append(final_tab)
        
def plot_scattered_beta_bias(survey = "JADES-Deep-GS", instrument = "ACS_WFC+NIRCam", beta_arr = [-3.], z_arr = [8.5, 11.], min_bin_counts = 500, z_min = 6.5, z_max = 13., \
        x_key = "m_UV_int", input_dir = "/raid/scratch/work/austind/GALFIND_WORK/Beta_paper/beta_bias_pure_power_law", moving_average_n = 1, n_bins = 10, plot_inset = True, save = True, show = True):
    
    fig, ax = plt.subplots(nrows = len(z_arr) + 1, ncols = 1, sharex = True, sharey = True, figsize = (5, 5 * (len(z_arr) + 1)))
    if plot_inset and z_arr == []:
        ax2 = fig.add_axes([0.16, 0.145, 0.3, 0.3], transform = ax.transAxes)
    x_key_to_label_dict = {"m_UV_int": r"Input $m_{\mathrm{UV}}$", "M_UV_int": r"Input $M_{\mathrm{UV}}$"}
    
    for i, beta in enumerate(beta_arr):
        # combine scattered beta bias catalogues
        out_path = f"{input_dir}/{instrument}/{survey}_beta={beta:.1f}_EPOCHS_III_combined.fits"
        #if not Path(out_path).is_file():
        cat_names = glob.glob(f"{input_dir}/{instrument}/{survey}_beta={beta:.1f}*merged.fits")
        #print(cat_names)
        cats = [Table.read(name) for name in cat_names]
        for cat, name in zip(cats, cat_names):
            print(name, len(cat), np.min(cat["beta_int"]), np.max(cat["beta_int"]))
        tab = vstack(cats)
        tab = tab[tab["6.5<z<13_selected"] == True]
        #print(len(tab))
        tab.write(out_path, overwrite = True)
        #else:
        #    tab = Table.read(out_path)
        if "M_UV" in x_key:
            bins = np.linspace(-22.5, -17.0, n_bins)
        elif "m_UV" in x_key:
            bins = np.linspace(26., 30., n_bins)
        for j in range(len(z_arr) + 1):
            if z_arr != []:
                ax_to_plot = ax[j]
                z_bin_name = f"_{str(z_arr)}"
                if j == 0:
                    tab_ = tab[tab["z_int"] < z_arr[j]]
                elif j == len(z_arr):
                    tab_ = tab[tab["z_int"] > z_arr[j - 1]]
                else:
                    tab_ = tab[(tab["z_int"] < z_arr[j]) & (tab["z_int"] > z_arr[j - 1])]
                if plot_inset and i == 0 and j == 0:
                    ax2 = fig.add_axes([0.18, 0.165, 0.3, 0.3], transform = ax_to_plot.transAxes)
            else:
                ax_to_plot = ax
                z_bin_name = ""
                tab_ = tab
            median_delta_beta = np.median(tab["beta_obs"] - tab["beta_int"])
            print(len(tab_))
            z_label = f"{str(z_min)}<z<{str(z_max)}" if z_arr == [] else f"{str(z_min)}<z<{str(z_arr[0])}" if j == 0 else f"{str(z_arr[-1])}<z<{str(z_max)}" if j == len(z_arr) else f"{str(z_arr[j - 1])}<z<{str(z_arr[j])}"
            ax_to_plot.text(0.98, 0.98, z_label, ha = "right", va = "top", transform = ax_to_plot.transAxes)
            # bin_medians, bin_edges, bin_numbers = binned_statistic(moving_average(tab_[x_key], moving_average_n), moving_average(tab_["beta_obs"] - tab_["beta_int"], moving_average_n), statistic = "median", bins = bins)
            # bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
            # ax[i].plot(bin_centres, bin_medians, c = "black", lw = 3, label = r"All $\beta$" + f", N={len(tab_)}")
            # if beta_arr != []:
            #     for j in range(len(beta_arr) + 1):
            #         if j == 0:
            #             tab__ = tab_[tab_["beta_int"] < beta_arr[j]]
            #         elif j == len(beta_arr):
            #             tab__ = tab_[tab_["beta_int"] > beta_arr[j - 1]]
            #         else:
            #             tab__ = tab_[(tab_["beta_int"] < beta_arr[j]) & (tab_["beta_int"] > beta_arr[j - 1])]
            bin_medians, bin_edges, bin_numbers = binned_statistic(moving_average(tab_[x_key], moving_average_n), moving_average(tab_["beta_obs"] - tab_["beta_int"], moving_average_n), statistic = "median", bins = bins)
            unique, counts = np.unique(bin_numbers, return_counts = True)
            bin_centres = ((bin_edges[:-1] + bin_edges[1:]) / 2)
            if save and z_arr == []:
                # save binned bias
                save_bias({x_key: bin_centres, "delta_beta": bin_medians, "counts": counts}, {"survey": survey, "instrument": instrument, "beta_int": beta, \
                    "median_delta_beta": median_delta_beta, "tab_length": len(tab), f"{x_key.replace('_int', '')}_bin_edges": list(bin_edges)}, bias_type = x_key.replace("_int", ""))
            
            # remove bins with fewer than min_bin_counts entries
            indices_to_keep = [k for k, count in enumerate(counts) if count > min_bin_counts]
            bin_centres = bin_centres[indices_to_keep]
            bin_medians = bin_medians[indices_to_keep]
            print(z_label, f"beta={beta:.1f}", dict(zip(unique, counts)))
            label = r"$\beta=$" + f"{beta:.1f}" #str(beta_arr[0]) if j == 0 else r"$\beta>$" + str(beta_arr[-1]) if j == len(beta_arr) else str(beta_arr[j - 1]) + r"<$\beta$<" + str(beta_arr[j])
            label += r", $N=$" + str(len(tab_))
            plot = ax_to_plot.plot(bin_centres, bin_medians, label = label)
            #ax_to_plot.scatter(tab_[x_key], tab_["beta_obs"] - tab_["beta_int"], c = plot[0].get_color(), alpha = 0.1)
            ax_to_plot.set_ylabel(r"$\Delta\beta$")
            
            if plot_inset:
                # calculate x_key counts
                hist, bin_edges = np.histogram(tab[x_key], bins = bins, density = True) 
                ax2.plot(((bin_edges[:-1] + bin_edges[1:]) / 2), hist, c = plot[0].get_color())
                kwargs = {"fontsize": 12., "labelpad": 7.5}
                ax2.set_xlabel(x_key_to_label_dict[x_key].replace("Input ", ""), **kwargs)
                ax2.set_ylabel(r"d$N$/d" + x_key_to_label_dict[x_key].replace("Input ", ""), **kwargs)
                ax2.tick_params(axis = "both", labelsize = kwargs["fontsize"], which = "major", length = 5., \
                                labelbottom = False, labeltop = True, labelleft = False, labelright = True)
                ax2.tick_params(axis = "both", labelsize = kwargs["fontsize"], which = "minor", length = 2.)
                ax2.xaxis.set_label_position("top")
                ax2.yaxis.set_label_position("right")
                ax2.set_xlim(*ax.get_xlim())
                ax2.set_xticks(list(ax.get_xticks())[::2])
            else:
                ax_to_plot.legend(loc = "lower left")
            
    if plot_inset:
        ax.legend(ncol = 2, loc = "upper center", bbox_to_anchor = (0.5, -0.15))
    if z_arr == []:
        ax.set_xlabel(x_key_to_label_dict[x_key])
        ax.set_title(survey)
    else:
        ax[len(z_arr)].set_xlabel(x_key_to_label_dict[x_key])
        ax[0].set_title(survey)
    if plot_inset:
        ax2.set_xlabel(x_key_to_label_dict[x_key])
        ax_to_plot.set_ylabel(r"$\Delta\beta$")
    plt.subplots_adjust(hspace = 0.)
    if save:
        plt.savefig(f"/raid/scratch/work/austind/GALFIND_WORK/Bagpipes/EPOCHS_III/plots/beta_bias/{survey}_beta_bias_{x_key}{z_bin_name}.png")
    if show:
        plt.show()
    else:
        plt.close()
        
def beta_bias_split_check(survey, instrument, number, beta_in, input_dir = f"{config['DEFAULT']['GALFIND_WORK']}/Catalogues/beta_bias_v9"):
    cat_names = glob.glob(f"{input_dir}/{instrument}/{survey}_beta={beta_in:.1f}_{int(number)}*/*matched_selection.fits")
    for name in cat_names:
        cat = Table.read(name)
        print(name, cat.columns, np.min(cat["z_int"]), np.max(cat["z_int"]), np.min(cat["beta_int"]), np.max(cat["beta_int"]))

def plot_beta_bias(survey_arr, instrument_arr, beta_arr, x_keys = ["m_UV_int"], n_bins = 10, min_bin_counts = 500, show = False):
    assert(len(survey_arr) == len(instrument_arr))
    for i, (survey, instrument) in enumerate(zip(survey_arr, instrument_arr)):
        for j, x_key in enumerate(x_keys):
            plot_scattered_beta_bias(survey = survey, instrument = instrument, beta_arr = beta_arr, plot_inset = True, x_key = x_key, z_arr = [], n_bins = n_bins, min_bin_counts = min_bin_counts, show = show)
            #plot_scattered_beta_bias(survey = survey, instrument = instrument, beta_arr = beta_arr, plot_inset = False, x_key = x_key, z_arr = [8.5, 11.], n_bins = n_bins, min_bin_counts = min_bin_counts, show = show)
        plot_bias_vs_z(survey, instrument, beta_arr, show = show)
        plot_binned_bias_vs_z(survey, instrument, beta_arr, show = show)
        
def save_bias(data, meta, bias_type):
    # add zero to counts where appropriate
    nan_indices = [i  if i < len(data["counts"]) else len(data["counts"]) for i in list(np.array(np.argwhere(np.isnan(data["delta_beta"]))).flatten())]
    print(data)
    print(nan_indices)
    if np.isnan(np.sum(data["delta_beta"])):
        print("here")
        data["counts"] = np.insert(data["counts"], nan_indices, 0)
    print(data)
    out_tab = Table(data)
    out_tab.meta = meta
    out_path = f"/raid/scratch/work/austind/GALFIND_WORK/Bagpipes/EPOCHS_III/beta_bias_data/bias_vs_{bias_type}_int/{out_tab.meta['instrument']}/{out_tab.meta['survey']}/{out_tab.meta['survey']}_beta={out_tab.meta['beta_int']:.1f}.ecsv"
    os.makedirs("/".join(out_path.split("/")[:-1]), exist_ok = True)
    out_tab.write(out_path, overwrite = True)

def plot_bias_vs_z(survey, instrument, beta_arr, colour_by = "epsilon", cmap_name = "RdBu_r", n_bins = 20, min_bin_counts = 500, \
                   input_dir = "/raid/scratch/work/austind/GALFIND_WORK/Beta_paper/beta_bias_pure_power_law", show = True, save = True):
    for beta in beta_arr:
        fig, ax = plt.subplots(figsize = (8, 5))
        in_path = f"{input_dir}/{instrument}/{survey}_beta={beta:.1f}_EPOCHS_III_combined.fits"
        tab = Table.read(in_path)
        epsilon = ((1 + tab["zbest_fsps_larson"]) / (1 + tab["z_int"])) - 1.
        tab["epsilon"] = epsilon
        # sort to make extreme values of colour pop out
        tab["abs_epsilon"] = abs(epsilon)
        tab.sort("abs_epsilon")
        median_delta_beta = np.median(tab["beta_obs"] - tab["beta_int"])
        scatter = ax.scatter(tab["z_int"], tab["beta_obs"] - tab["beta_int"], c = tab["epsilon"], cmap = cmap_name)
        ax.text(0.98, 0.98, r"$\langle\Delta\beta\rangle=$" + f"{np.median(tab['beta_obs'] - tab['beta_int']):.2f}", ha = "right", va = "top", transform = ax.transAxes)
        bins = np.linspace(6.5, 13., n_bins)
        bin_medians, bin_edges, bin_numbers = binned_statistic(tab["z_int"], tab["beta_obs"] - tab["beta_int"], statistic = "median", bins = bins)
        bin_centres = ((bin_edges[:-1] + bin_edges[1:]) / 2)
        print(survey, f"beta={beta:.1f}", dict(zip(bin_centres, bin_medians)))
        unique, counts = np.unique(bin_numbers, return_counts = True)
        if save:
            # save binned bias
            save_bias({"z_int": bin_centres, "delta_beta": bin_medians, "counts": counts}, {"survey": survey, "instrument": instrument, "beta_int": beta, \
                "median_delta_beta": median_delta_beta, "tab_length": len(tab), "z_bin_edges": list(bin_edges)}, bias_type = "z")
        # remove bins with fewer than min_bin_counts entries
        indices_to_keep = [k for k, count in enumerate(counts) if count > min_bin_counts]
        bin_centres = bin_centres[indices_to_keep]
        bin_medians = bin_medians[indices_to_keep]
        plot = ax.plot(bin_centres, bin_medians, c = "black", lw = 3)
        ax.set_ylim(-4., 3.2)
        ax2 = fig.add_axes([0.58, 0.18, 0.15, 0.1], transform = ax.transAxes)
        ax2.plot(bin_centres, bin_medians, c = "black")
        ax2.plot(bin_centres, np.full(len(bin_centres), median_delta_beta), c = "gray", ls = "--")
        if survey == "NEP-1":
            title_survey = "NEP"
        else:
            title_survey = survey
        ax.set_title(title_survey + r", $\beta=$" + f"{beta:.1f}")
        for j, ax_loc in enumerate([ax, ax2]):
            if j == 0:
                kwargs = {}
            else:
                kwargs = {"fontsize": 10., "labelpad": 0.}
            ax_loc.set_xlabel(r"Input redshift, $z$", **kwargs)
            ax_loc.set_ylabel(r"$\Delta\beta$", **kwargs)
        ax2.tick_params(axis = "both", labelsize = 8., which = "major", length = 3.)
        ax2.tick_params(axis = "both", labelsize = 8., which = "minor", length = 0.)
        fig.colorbar(scatter, ax = ax, label = r"$\epsilon$")
        if save:
            # save plot
            plt.savefig(f"/raid/scratch/work/austind/GALFIND_WORK/Bagpipes/EPOCHS_III/plots/beta_bias/{survey}_beta={beta:.1f}_z_bias.png")
        if show:
            plt.show()
        else:
            plt.close()
            
def plot_binned_bias_vs_z(survey, instrument, beta_arr, min_bin_counts = 500, \
        input_dir = "/raid/scratch/work/austind/GALFIND_WORK/Bagpipes/EPOCHS_III/beta_bias_data/bias_vs_z_int", \
        output_dir = "/raid/scratch/work/austind/GALFIND_WORK/Bagpipes/EPOCHS_III/plots/beta_bias", show = True, save = True):
    fig, ax = plt.subplots()
    for i, beta in enumerate(beta_arr):
        # open ecsv
        input_path = f"{input_dir}/{instrument}/{survey}/{survey}_beta={beta:.1f}.ecsv"
        tab = Table.read(input_path)
        tab = tab[tab["counts"] > min_bin_counts]
        plot = ax.plot(tab["z_int"], tab["delta_beta"], label = r"$\beta=$" + f"{beta:.1f}")
        ax.axhline(tab.meta["median_delta_beta"], ls = "--", c = plot[0].get_color())
    orig_xlims = ax.get_xlim()
    ax.axvline(-99., c = "black", ls = "--", label = r"$\langle\Delta\beta\rangle$",)
    
    beta_legend = plt.legend([ax.get_lines()[i] for i in [j * 2 for j in range(len(beta_arr))]], [ax.get_lines()[i].get_label() for i in [j * 2 for j in range(len(beta_arr))]])
    med_dbeta_legend = plt.legend([ax.get_lines()[len(beta_arr) * 2]], [ax.get_lines()[len(beta_arr) * 2].get_label()], loc = "lower left", fontsize = 16., frameon = False, handletextpad = 0.2)
    for legend in [beta_legend, med_dbeta_legend]:
        ax.add_artist(legend)
    ax.set_xlabel("Input redshift, z")
    ax.set_ylabel(r"$\Delta\beta$")
    ax.set_xlim(*orig_xlims)
    
    if survey == "NEP-1":
        title_survey = "NEP"
    else:
        title_survey = survey
    ax.set_title(title_survey)
    if save:
        output_path = f"{output_dir}/{survey}_z_bias_binned.png"
        plt.savefig(output_path, overwrite = True)
    if show:
        plt.show()
    else:
        plt.close()
        
def make_bias_hdf5(surveys, instruments, beta_arr, bias_type_arr, input_dir = "/raid/scratch/work/austind/GALFIND_WORK/Bagpipes/EPOCHS_III/beta_bias_data"):
    os.makedirs(input_dir, exist_ok = True)
    hf = h5py.File(f"{input_dir}/beta_bias.h5", "w")
    assert(len(surveys) == len(instruments))
    for i, (survey, instrument) in enumerate(zip(surveys, instruments)):
        for j, bias_type in enumerate(bias_type_arr):
            delta_beta_arr = [] #np.zeros(len(beta_arr))
            counts_arr = []
            median_delta_beta_arr = []
            for k, beta in enumerate(beta_arr):
                # open appropriate bias catalogue
                input_path = f"{input_dir}/bias_vs_{bias_type}_int/{instrument}/{survey}/{survey}_beta={beta:.1f}.ecsv"
                tab = Table.read(input_path)
                if k == 0:
                    x_arr = np.array(tab[f"{bias_type}_int"])
                else:
                    assert(len(np.array(tab[f"{bias_type}_int"])) == len(x_arr))
                    for m in range(len(x_arr)):
                        assert(np.array(tab[f"{bias_type}_int"])[m] == x_arr[m])
                delta_beta_arr.append(np.array(tab["delta_beta"]))
                median_delta_beta_arr.append(tab.meta["median_delta_beta"])
                counts_arr.append(np.array(tab["counts"]))
            delta_beta_arr = np.array(delta_beta_arr)
            bias_group = hf.create_group(f"{survey}/{bias_type}_beta_bias")
            bias_group.create_dataset("counts", data = counts_arr)
            bias_group.create_dataset(bias_type, data = x_arr)
            bias_group.create_dataset("beta", data = beta_arr)
            bias_group.create_dataset("delta_beta", data = delta_beta_arr)
            if j == 0:
                survey_hf = hf.get(survey)
                median_bias_group = survey_hf.create_group("median_beta_bias")
                median_bias_group.create_dataset("beta", data = beta_arr)
                median_bias_group.create_dataset("median_delta_beta", data = median_delta_beta_arr)
    #print(list(hf.get(f"{survey}/median_beta_bias").items()))
    hf.close()
    
def calc_beta_bias(survey, z_in, m_UV_in, beta_in, min_bin_counts = 500, plot = False, \
                   beta_bias_h5_path = "/raid/scratch/work/austind/GALFIND_WORK/Bagpipes/EPOCHS_III/beta_bias_data/beta_bias.h5"):
    # open .h5 file
    hf = h5py.File(beta_bias_h5_path, "r")
    
    # interpolate z dependent beta bias -----------------------
    z_arr = np.array(hf.get(f"{survey}/z_beta_bias/z"))
    beta_arr = np.array(hf.get(f"{survey}/z_beta_bias/beta"))
    delta_beta_z = np.array(hf.get(f"{survey}/z_beta_bias/delta_beta"))
    # determine survey dependent z range to interpolate over
    indices_to_keep = [(i, j) for i, count_arr in enumerate(np.array(hf.get(f"{survey}/z_beta_bias/counts"))) for j, count in enumerate(count_arr) if count > min_bin_counts]
    min_z_index = 0
    max_z_index = len(z_arr) - 1
    for i, beta in enumerate(np.array(hf.get(f"{survey}/z_beta_bias/beta"))):
        z_indices_i = [k for j, k in indices_to_keep if j == i]
        z_indices_i_min, z_indices_i_max = np.min(z_indices_i), np.max(z_indices_i)
        if z_indices_i_min > min_z_index:
            min_z_index = z_indices_i_min
        if z_indices_i_max < max_z_index:
            max_z_index = z_indices_i_max
    z_arr = z_arr[min_z_index : max_z_index + 1]
    delta_beta_z = np.array([dbeta[min_z_index : max_z_index + 1] for dbeta in delta_beta_z])
    
    z_spline = RectBivariateSpline(beta_arr, z_arr, delta_beta_z)
    beta_bias_z = z_spline(beta_in, z_in, grid = False)
    #print(beta_bias_z)
    
    # interpolate m_UV dependent beta bias -----------------------
    m_UV_arr = np.array(hf.get(f"{survey}/m_UV_beta_bias/m_UV"))
    beta_arr = np.array(hf.get(f"{survey}/m_UV_beta_bias/beta"))
    delta_beta_m_UV = np.array(hf.get(f"{survey}/m_UV_beta_bias/delta_beta"))
    # determine survey dependent m_UV range to interpolate over
    indices_to_keep = [(i, j) for i, count_arr in enumerate(np.array(hf.get(f"{survey}/m_UV_beta_bias/counts"))) for j, count in enumerate(count_arr) if count > min_bin_counts]
    min_m_UV_index = 0
    max_m_UV_index = len(m_UV_arr) - 1
    for i, beta in enumerate(np.array(hf.get(f"{survey}/m_UV_beta_bias/beta"))):
        m_UV_indices_i = [k for j, k in indices_to_keep if j == i]
        m_UV_indices_i_min, m_UV_indices_i_max = np.min(m_UV_indices_i), np.max(m_UV_indices_i)
        if m_UV_indices_i_min > min_m_UV_index:
            min_m_UV_index = m_UV_indices_i_min
        if m_UV_indices_i_max < max_m_UV_index:
            max_m_UV_index = m_UV_indices_i_max
    m_UV_arr = m_UV_arr[min_m_UV_index : max_m_UV_index + 1]
    delta_beta_m_UV = np.array([dbeta[min_m_UV_index : max_m_UV_index + 1] for dbeta in delta_beta_m_UV])
    
    m_UV_spline = RectBivariateSpline(beta_arr, m_UV_arr, delta_beta_m_UV)
    beta_bias_m_UV = m_UV_spline(beta_in, m_UV_in, grid = False)
    print(beta_bias_m_UV)
    
    # interpolate median delta beta
    beta_arr = np.array(hf.get(f"{survey}/median_beta_bias/beta"))
    median_delta_beta_arr = np.array(hf.get(f"{survey}/median_beta_bias/median_delta_beta"))
    median_dbeta_spline = UnivariateSpline(beta_arr, median_delta_beta_arr)
    beta_bias_median_dbeta = median_dbeta_spline(beta_in)
    print(beta_bias_median_dbeta)
    hf.close()
    
    # calculate total beta bias
    tot_beta_bias = beta_bias_z + beta_bias_m_UV - beta_bias_median_dbeta
    print(tot_beta_bias)
    
    if plot:
        # plot beta_bias_z spline
        fig, ax = plt.subplots(figsize = (8, 6))
        im = ax.imshow(np.array(hf.get(f"{survey}/z_beta_bias/delta_beta")), \
                extent = (min(np.array(hf.get(f"{survey}/z_beta_bias/z"))), max(np.array(hf.get(f"{survey}/z_beta_bias/z"))), \
                min(np.array(hf.get(f"{survey}/z_beta_bias/beta"))), max(np.array(hf.get(f"{survey}/z_beta_bias/beta")))), \
                cmap = "RdBu_r", interpolation = "gaussian")
        fig.colorbar(im, label = r"$\Delta\beta$")
        ax.set_title(survey)
        ax.set_xlabel(r"Input redshift, $z$")
        ax.set_ylabel(r"Input $\beta$")
        plt.show()
        # plot beta_bias_m_UV spline
        fig, ax = plt.subplots(figsize = (8, 6))
        im = ax.imshow(np.array(hf.get(f"{survey}/m_UV_beta_bias/delta_beta")), \
                extent = (min(np.array(hf.get(f"{survey}/m_UV_beta_bias/m_UV"))), max(np.array(hf.get(f"{survey}/m_UV_beta_bias/m_UV"))), \
                min(np.array(hf.get(f"{survey}/m_UV_beta_bias/beta"))), max(np.array(hf.get(f"{survey}/m_UV_beta_bias/beta")))), \
                cmap = "RdBu_r", interpolation = "gaussian")
        fig.colorbar(im, label = r"$\Delta\beta$")
        ax.set_title(survey)
        ax.set_xlabel(r"Input $m_{\mathrm{UV}}$")
        ax.set_ylabel(r"Input $\beta$")
        plt.show()

    return tot_beta_bias, beta_bias_z, beta_bias_m_UV, beta_bias_median_dbeta

if __name__ == "__main__":
    # for i in [0, 1]:
    #     for beta_in in [-3., -2.5, -2., -1.5, -1.]:
    #         beta_bias_split_check("CLIO", "NIRCam", i, beta_in)
    
    # for beta in [-3., -2.5, -2., -1.5, -1.]:
    #     pure_power_law_beta_bias(beta_in = beta)
    
    #plot_rest_frame_UV_line_bias(["CIII]-1909"])
    #EPOCHS_III_line_bias_plot(["CIV-1549", "HeII-1640", "OIII]-1665", "CIII]-1909"])
    
    #EPOCHS_III_imshow_bias_plot("DLA", cmap_name = "Reds") #"Blues_r")
    
    #DLA_beta_bias()
    
    #plot_beta_bias(["NEP-1", "CEERSP2", "CEERSP9", "JADES-Deep-GS", "NGDEEP", "GLASS", "El-Gordo", "MACS-0416"], \
    #              ["ACS_WFC+NIRCam" for i in range(5)] + ["NIRCam" for i in range(3)], [-3., -2.5, -2., -1.5, -1.], show = True) # AND CEERSP2 + CEERSP9 + JADES-Deep-GS

    #make_bias_hdf5(["NEP-1", "CEERSP2", "CEERSP9", "JADES-Deep-GS", "NGDEEP", "GLASS", "El-Gordo", "MACS-0416"], \
    #    ["ACS_WFC+NIRCam" for i in range(5)] + ["NIRCam" for i in range(3)], [-3., -2.5, -2., -1.5, -1.], bias_type_arr = ["z", "m_UV"])
    
    calc_beta_bias("El-Gordo", [5.6, 8.2], [24.2, 28.], [-1.2, -2.])
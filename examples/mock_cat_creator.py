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
from astropy.table import Table, vstack, join
from copy import deepcopy
import glob
import os
from tqdm import tqdm
from galfind import Mock_SED_rest, Mock_SED_obs, config, NIRCam, ACS_WFC, Data, Catalogue, astropy_cosmo, Mock_Photometry, GALFIND_Catalogue_Creator, Photometry_rest
from galfind import useful_funcs_austind as funcs

def make_mock_galfind_cat(survey, instrument_name, n_gals, z_prior, beta_prior, out_path, version = "v9", aper_diam = 0.32 * u.arcsec, scatter_size = 1):
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
        mu, sigma, min_m_UV, max_m_UV = 0., 1., 24., funcs.five_to_n_sigma_mag(np.max(survey_depths), 3.)
        print(max_m_UV)
        mag_arr = np.random.lognormal(mu, sigma, 1_000_000)
        m_UV_prior = np.array([mag for mag in max_m_UV - (mag_arr - np.exp(-sigma ** 2)) if mag > min_m_UV and mag < max_m_UV + np.exp(-sigma ** 2)])
        
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
    
def split_mock_galfind_tab(tab, instrument, out_path):
    # only include photometry from below 3000 Angstrom to avoid non-power law SED
    cat_bands = []
    z_min_dict = {}
    for i, (band, band_wav, band_FWHM) in enumerate(zip(instrument, instrument.band_wavelengths.values(), instrument.band_FWHMs.values())):
        z_min = ((band_wav + band_FWHM / 2.) / (3000. * u.AA)).value - 1.
        z_min_dict[band] = z_min
        if i != 0:
            if z_min > 6.5 and z_min_dict[instrument[i - 1]] < 13.:
                cat_bands.append(list(instrument[:i]))
    cat_z_lims = [[list(z_min_dict.values())[list(z_min_dict.keys()).index(bands[-1])], list(z_min_dict.values())[list(z_min_dict.keys()).index(bands[-1]) + 1]] for bands in cat_bands]
    cat_z_lims[0][0], cat_z_lims[-1][-1] = 6.5, 13.
    #print(cat_bands, cat_z_lims, z_min_dict)
    survey_cat_paths = []
    for i, (bands, z_lims) in enumerate(zip(cat_bands, cat_z_lims)):
        split_out_path = out_path.replace(".fits", f"_split_{str(i + 1)}.fits")
        survey_cat_paths.append(split_out_path)
        if not Path(split_out_path).is_file():
            z_split_tab = tab[tab["z_int"] > z_lims[0]]
            z_split_tab = z_split_tab[z_split_tab["z_int"] < z_lims[1]]
            #print(len(z_split_tab))
            z_split_tab.remove_columns([f"FLUX_APER_{band}_aper_corr_Jy" for band in instrument if band not in bands])
            z_split_tab.remove_columns([f"FLUXERR_APER_{band}_loc_depth_10pc_Jy" for band in instrument if band not in bands])
            z_split_tab.rename_column("ID", "NUMBER")
            z_split_tab["ALPHA_J2000"] = 0.
            z_split_tab["DELTA_J2000"] = 0.
            z_split_tab.write(split_out_path, overwrite = True)
    return survey_cat_paths
        
def run_beta_bias_through_EAZY(survey, survey_cat_paths, instrument_in, aper_diam = 0.32 * u.arcsec, min_pc_err = 10., version = "beta_bias_v9"):
    cat_creator = GALFIND_Catalogue_Creator("loc_depth", aper_diam, min_pc_err)
    code_names = ["EAZY"]
    eazy_templates = ["fsps_larson"]
    eazy_lowz_zmax = [6.]
    out_cats = []
    for path in survey_cat_paths:
        instrument = deepcopy(instrument_in)
        survey_name = path.split("/")[-1].replace("_beta_bias_split", "").replace(".fits", "")
        out_cats.append(Catalogue.from_fits_cat(path, version, instrument, cat_creator, code_names, survey_name, eazy_lowz_zmax, eazy_templates, data = None, mask = False, excl_bands = []))
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
        beta, m_UV = phot.basic_beta_m_UV_calc(1.)
        M_UV = m_UV - 5 * np.log10(phot.lum_distance.value / 10) + 2.5 * np.log10(1 + phot.z)
        beta_arr[i] = beta
        m_UV_arr[i] = m_UV
        M_UV_arr[i] = M_UV
    return beta_arr, m_UV_arr, M_UV_arr

def pure_power_law_beta_bias(survey, instrument_name, version = "beta_bias_v9"):
    out_path = f"{config['DEFAULT']['GALFIND_WORK']}/Beta_paper/beta_bias_pure_power_law/{instrument_name}/{survey}_beta_bias.fits"
    os.makedirs("/".join(out_path.split("/")[:-1]), exist_ok = True)
    n_gals = 20_000
    scatter_size = 10
    beta_prior = np.linspace(-3.2, 0., 1_000_000)
    z_prior = np.linspace(6.5, 13., 1_000_000)
    # make catalogue
    out_tab, instrument = make_mock_galfind_cat(survey, instrument_name, n_gals, z_prior, beta_prior, out_path, scatter_size = scatter_size)
    #survey_cat_paths = split_mock_galfind_tab(out_tab, instrument, out_path)
    #run_beta_bias_through_EAZY(survey, survey_cat_paths, instrument, version = version)
    # perform selection (DONE IN A SEPARATE SCRIPT!!!)
    
    select_cat_paths = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Catalogues/beta_bias_v9/{instrument_name}/{survey}*/{survey}*matched_selection.fits")
    select_cats = []
    for i, path in enumerate(select_cat_paths):
        if not Path(path.replace(".fits", "_final.fits")).is_file():
            survey_loc = f"{survey}_{str(i + 1)}"
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
            beta, m_UV, M_UV = calc_obs_UV_properties(path.replace(".fits", "_final.fits"), survey_loc, instrument_name, version = version)
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
    
    # produce paper-ready plots

if __name__ == "__main__":
    surveys_arr = [["NGDEEP", "CEERSP2", "NEP-1", "CEERSP9"], ["MACS-0416", "GLASS", "SMACS-0723", "CLIO", "El-Gordo"]] # "JADES-Deep-GS", 
    instrument_names = ["ACS_WFC+NIRCam", "NIRCam"]
    for surveys, instrument_name in zip(surveys_arr, instrument_names):
        for survey in surveys:
            excl_bands_survey = {"JADES-Deep-GS": ["f182M", "f210M", "f430M", "f460M", "f480M"], "NGDEEP": ["f435W", "f775W", "f850LP"], "NEP-1": [], "NEP-2": [], "NEP-3": [], "NEP-4": [], \
                                  "CLIO": [], "El-Gordo": [], "MACS-0416": [], "GLASS": [], "SMACS-0723": [], "CEERSP1": [], "CEERSP2": [], "CEERSP3": [], "CEERSP4": [], "CEERSP5": [], \
                                      "CEERSP6": [], "CEERSP7": [], "CEERSP8": [], "CEERSP9": [], "CEERSP10": []}
            version = "v9"
            aper_diam = 0.32 * u.arcsec
            survey_bands_depths = Data.from_pipeline(survey, version, instruments = instrument_name.split("+"), excl_bands = excl_bands_survey[survey]).load_depths(aper_diam)
            survey_bands = [key for key in survey_bands_depths.keys()]
            survey_depths = [depth for depth in survey_bands_depths.values()]
            print(survey, survey_bands_depths, funcs.five_to_n_sigma_mag(np.max(survey_depths), 3.))
            #pure_power_law_beta_bias(survey, instrument_name)
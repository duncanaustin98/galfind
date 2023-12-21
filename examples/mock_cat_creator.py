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
import os
from galfind import Mock_SED_rest, Mock_SED_obs, config, NIRCam, ACS_WFC, Data, astropy_cosmo

def make_mock_galfind_cat(survey, n_gals, z_prior, beta_prior, m_UV_prior, out_path, version = "v9", aper_diam = 0.32 * u.arcsec):
    # make instrument and load observational depths
    excl_bands_survey = {"JADES-Deep-GS": ["f182M", "f210M", "f430M", "f460M", "f480M"]}
    survey_bands_depths = Data.from_pipeline(survey, version, excl_bands = excl_bands_survey[survey]).load_depths(aper_diam)
    survey_bands = [key for key in survey_bands_depths.keys()]
    survey_depths = [depth for depth in survey_bands_depths.values()]
    instrument = NIRCam(excl_bands = [band for band in NIRCam().bands if not band in survey_bands]) \
        + ACS_WFC(excl_bands = [band for band in ACS_WFC().bands if not band in survey_bands])
    
    z_in = np.random.choice(z_prior, n_gals)
    beta_in = np.random.choice(beta_prior, n_gals)
    m_UV_in = np.random.choice(m_UV_prior, n_gals)
    M_UV_in = m_UV_in - 5 * np.log10(astropy_cosmo.luminosity_distance(z_in).to(u.pc).value / 10) + 2.5 * np.log10(1 + z_in)
    print(z_in, beta_in, m_UV_in)
    for i, (z, beta, m_UV, M_UV) in enumerate(zip(z_in, beta_in, m_UV_in, M_UV_in)):
        sed_obs = Mock_SED_obs.power_law_from_beta_M_UV(z, beta, M_UV)
        sed_obs.create_mock_phot(instrument, survey_depths)

def pure_power_law_beta_bias(survey):
    out_path = f"{config['DEFAULT']['GALFIND_WORK']}Beta_paper/beta_bias_pure_power_law/{survey}"
    os.makedirs("/".join(out_path.split("/")[:-1]), exist_ok = True)
    n_gals = 10_000 #0_000
    mu, sigma, min_m_UV, max_m_UV = 0., 1., 24., 31.
    mag_arr = np.random.lognormal(mu, sigma, n_gals * 100)
    m_UV_prior = np.array([mag for mag in max_m_UV - (mag_arr - np.exp(-sigma ** 2)) if mag > min_m_UV and mag < max_m_UV])
    beta_prior = np.linspace(-3.2, 0., n_gals * 100)
    z_prior = np.linspace(6.5, 13., n_gals * 100)
    make_mock_galfind_cat(survey, n_gals, z_prior, beta_prior, m_UV_prior, out_path)

if __name__ == "__main__":
    survey = "JADES-Deep-GS"
    pure_power_law_beta_bias(survey)
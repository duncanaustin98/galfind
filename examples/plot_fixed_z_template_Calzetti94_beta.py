#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:42:29 2023

@author: u92876da
"""

# plot_fixed_z_template_Calzetti94_beta.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.units as u
from tqdm import tqdm

from galfind import SED, Mock_SED_rest, Mock_SED_obs, config, NIRCam, Photometry_rest

plt.style.use(f"{config['DEFAULT']['GALFIND_DIR']}/galfind_style.mplstyle")

z_arr = np.linspace(6.5, 12., 10) #[6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10., 10.5, 11., 11.5, 12.]
min_pc_err = 10

#JADES_depths = [29.58, 29.78, 29.68, 29.72, 30.21, 29.58, 30.17, 29.65, 29.99]
#{band: depth for band, depth in zip(instrument, JADES_depths)}

for incl_bands in [[], ["f140M", "f162M", "f182M", "f210M", "f250M", "f300M", "f335M"]]:#["f182M", "f210M"]]:
    NIRCam_excl_bands = ["f070W", "f140M", "f162M", "f182M", "f210M", "f250M", "f300M", "f335M", "f360M", "f430M", "f460M", "f480M"]
    excl_bands = [band for band in NIRCam_excl_bands if band not in incl_bands]
    instrument = NIRCam(excl_bands = excl_bands)
    depths = {band: 29.5 for band in instrument.bands} 
    
    fig, ax = plt.subplots()
    for i in range(13, 15):
        beta_C94_arr = []
        beta_phot_arr = []
        sed_rest = Mock_SED_rest.load_EAZY_in_template(25., "fsps_larson", i)
        print(sed_rest.template_name)
        for z in tqdm(z_arr, total = len(z_arr), desc = f"Calculating Δβ for {sed_rest.template_name}"):
            sed_obs = Mock_SED_obs.from_Mock_SED_rest(sed_rest, z)
            beta_C94_arr.append(sed_obs.calc_UV_slope(output_errs = False)[1])
            sed_obs.create_mock_photometry(instrument, depths, min_pc_err)
            phot_rest = Photometry_rest.from_phot(sed_obs.mock_photometry, z)
            beta_phot_arr.append(phot_rest.basic_beta_calc())
        ax.plot(np.array(z_arr), np.array(beta_phot_arr) - np.array(beta_C94_arr), label = sed_obs.template_name)
    
    plt.legend(fontsize = 12, bbox_to_anchor = (1.05, 0.5), loc = "center left")
    ax.set_xlabel("Redshift, z")
    ax.set_ylabel(r"$\Delta\beta$ (1250-3000$\mathrm{\AA}$ phot) - C+94")
    if incl_bands == []:
        ax.set_title("Standard 8 NIRCam filters only")
    else:
        ax.set_title(f"Incl bands = {str('+'.join(incl_bands))}")
    plt.show()
    
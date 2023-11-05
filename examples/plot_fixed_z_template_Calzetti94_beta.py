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
import os
from pathlib import Path
from astropy.table import Table
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

from galfind import SED, Mock_SED_rest, Mock_SED_obs, config, NIRCam, Photometry_rest

plt.style.use(f"{config['DEFAULT']['GALFIND_DIR']}/galfind_style.mplstyle")

z_arr = np.linspace(6.5, 12., 100)
min_pc_err = 10
template_set = "fsps_larson"
m_UV_norm = 25.
fixed_depth_mag = 29.
plot = False

m_UV_name = str(m_UV_norm)
depth_name = str(fixed_depth_mag)

#JADES_depths = [29.58, 29.78, 29.68, 29.72, 30.21, 29.58, 30.17, 29.65, 29.99]
#{band: depth for band, depth in zip(instrument, JADES_depths)}

rest_UV_band_arr = ["f140M", "f162M", "f182M", "f210M", "f250M", "f300M", "f335M"]
band_combinations = []
for i in range(len(rest_UV_band_arr)):
    rest_bands_i_tups = list(combinations(rest_UV_band_arr, i))
    for tup in rest_bands_i_tups:
        band_combinations.append(list(tup))
print(np.array(band_combinations).shape)

for incl_bands in tqdm(band_combinations, desc = "Saving Δβ", position = 2):
    if incl_bands == []:
        title = "8_NIRCam"
    else:
        title = f"8_NIRCam+{str('+'.join([band.replace('f', 'F') for band in incl_bands]))}"
    if plot:
        fig, ax = plt.subplots()
    NIRCam_excl_bands = ["f070W", "f140M", "f162M", "f182M", "f210M", "f250M", "f300M", "f335M", "f360M", "f430M", "f460M", "f480M"]
    excl_bands = [band for band in NIRCam_excl_bands if band not in incl_bands]
    instrument = NIRCam(excl_bands = excl_bands)
    depths = {band: fixed_depth_mag for band in instrument.bands} 
    for i in tqdm(range(13, 19), desc = f"Calculating for incl_bands = {incl_bands}", position = 1):
        sed_rest = Mock_SED_rest.load_EAZY_in_template(m_UV_norm, template_set, i)
        output_dir = f"{config['DEFAULT']['GALFIND_WORK']}/Beta_paper/Beta_phot_{template_set}/{sed_rest.template_name}"
        if not Path(f"{output_dir}/{title}.ecsv").is_file():
            print(f"{output_dir}/{title}.ecsv")
            if plot:
                beta_C94_arr = []
            beta_phot_arr = []
            for z in tqdm(z_arr, total = len(z_arr), desc = f"Calculating Δβ for {sed_rest.template_name}", position = 0):
                sed_obs = Mock_SED_obs.from_Mock_SED_rest(sed_rest, z, IGM = None)
                if plot:
                    beta_C94_arr.append(sed_obs.calc_UV_slope(output_errs = False)[1])
                sed_obs.create_mock_photometry(instrument, depths, min_pc_err)
                phot_rest = Photometry_rest.from_phot(sed_obs.mock_photometry, z)
                beta_phot_arr.append(phot_rest.basic_beta_calc())
            # save data
            out_tab = Table({"z": z_arr, "Beta_phot": beta_phot_arr})
            out_tab.meta = {"description": "The 8 standard NIRCam filters are F090W, F115W, F150W, F200W, F277W, F356W, F410M, F444W. Beta_phot calculated at rest wavelengths 1250-3000 Angstrom with no errors."}
            os.makedirs(output_dir, exist_ok = True)
            out_tab.write(f"{output_dir}/{title}.ecsv", overwrite = True, formats = {"z": np.float32, "Beta_phot": np.float32, "Beta_C94": np.float32})
        if plot:
            ax.plot(np.array(z_arr), np.array(beta_phot_arr) - np.array(beta_C94_arr), label = sed_obs.template_name)
    if plot:
        ax.set_title(title)
        ax.set_xlabel("Redshift, z")
        ax.set_ylabel(r"$\Delta\beta$ (1250-3000$\mathrm{\AA}$ phot) - C+94")
        plt.legend(fontsize = 12, bbox_to_anchor = (1.05, 0.5), loc = "center left")
        plt.show()

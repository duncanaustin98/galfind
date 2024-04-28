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
import seaborn as sns
from pathlib import Path
from astropy.table import Table
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

from galfind import SED, Mock_SED_rest, Mock_SED_obs, Mock_SED_rest_template_set, config, NIRCam, WFC3_IR, Photometry_rest

def calc_mock_beta_phot(z_arr, template_set, incl_bands, rest_UV_wav_lims_arr, m_UV_norm, fixed_depth_mag, \
        std_NIRCam_bands = ["f090W", "f115W", "f150W", "f200W", "f277W", "f356W", "f410M", "f444W"], incl_errs = False):
    output_dir = f"{config['DEFAULT']['GALFIND_WORK']}/Beta_paper/Beta_phot_{template_set}"
    os.makedirs(output_dir, exist_ok = True)
    NIRCam_excl_bands = ["f070W", "f140M", "f162M", "f182M", "f210M", "f250M", "f300M", "f335M", "f360M", "f430M", "f460M", "f480M"]
    NIRCam_excl_bands = [band for band in NIRCam_excl_bands if band not in incl_bands]
    WFC3_IR_excl_bands = ["f098M", "f105W",  "f110W", "f125W", "f127M", "f139M", "f140W", "f153M", "f160W"]
    WFC3_IR_excl_bands = [band for band in WFC3_IR_excl_bands if band not in incl_bands]
    instrument = NIRCam(excl_bands = NIRCam_excl_bands) + WFC3_IR(excl_bands = WFC3_IR_excl_bands)
    if incl_bands == []:
        title = "8_NIRCam"
    else:
        title = f"8_NIRCam+{str('+'.join([band.replace('f', 'F') for band in incl_bands]))}"
    if not Path(f"{output_dir}/{title}.ecsv").is_file():
        mock_sed_rest_set = Mock_SED_rest_template_set.load_EAZY_in_template(m_UV_norm, template_set)
        beta_phot = {}
        depths = [fixed_depth_mag for band_name in instrument.band_names]
        with tqdm(total = len(rest_UV_wav_lims_arr) * len(mock_sed_rest_set) * len(z_arr), desc = f"Calculating β for bands = {title}", leave = False) as pbar:
            for i, rest_UV_wav_lims in enumerate(rest_UV_wav_lims_arr):
                wav_name = f"{'-'.join([str(int(wav)) for wav in rest_UV_wav_lims.value])}Angstrom"
                beta_phot_loc = np.zeros(len(mock_sed_rest_set) * len(z_arr))
                # instantiate beta_phot_arr
                for j, mock_sed_rest in enumerate(mock_sed_rest_set):
                    for k, z in enumerate(z_arr):
                        mock_sed_rest.create_mock_phot(instrument, z, depths)
                        phot_rest = Photometry_rest.from_phot(mock_sed_rest.mock_photometry, z, rest_UV_wav_lims = rest_UV_wav_lims)
                        beta_phot_loc[j * len(z_arr) + k] = phot_rest.basic_beta_calc(incl_errs = incl_errs)
                        pbar.update(1)
                beta_phot[f"Beta_phot_{wav_name}"] = beta_phot_loc
        pbar.close()
        out_tab = Table({**{"template_name": np.array([mock_sed_rest.template_name for mock_sed_rest in mock_sed_rest_set for i in range(len(z_arr))]), \
                    "z": np.tile(z_arr, len(mock_sed_rest_set))}, **beta_phot})
        out_tab.meta = {"Bands": "+".join([band.replace("f", "F") for band in instrument.band_names]), "Beta_errs": False}
        out_tab.write(f"{output_dir}/{title}.ecsv", overwrite = True)

def calc_C94_beta(template_set, m_UV_norm):
    beta_C94 = []
    template_names = []
    mock_sed_rest_set = Mock_SED_rest_template_set.load_EAZY_in_template(m_UV_norm, template_set)
    for i, mock_sed_rest in enumerate(mock_sed_rest_set):
        beta_C94.append(mock_sed_rest.calc_UV_slope(output_errs = False)[1])
        template_names.append(mock_sed_rest.template_name)
    out_tab = Table({"template_name": template_names, "Beta_C94": beta_C94})
    output_dir = f"{config['DEFAULT']['GALFIND_WORK']}/Beta_paper"
    os.makedirs(output_dir, exist_ok = True)
    out_tab.write(f"{output_dir}/{template_set}_Calzetti+94.ecsv", overwrite = True)

def plot_dbeta(ax, z_arr, template_set, incl_bands, rest_UV_wav_lims, m_UV_norm, fixed_depth_mag, \
               incl_errs = False, annotate = True, show = True, save = True, cmap_name = "Oranges", \
                   plot_indices = [], output_dir = f"{config['DEFAULT']['GALFIND_WORK']}/Beta_paper/Delta_beta_fsps_larson"):
    mock_sed_rest_set = Mock_SED_rest_template_set.load_EAZY_in_templates(m_UV_norm, template_set)
    cmap = sns.color_palette(cmap_name, len(plot_indices) + 5)
    # determine appropriate file to load mock photometric beta calculations
    input_phot_dir = f"{config['DEFAULT']['GALFIND_WORK']}/Beta_paper/Beta_phot_{template_set}"
    input_spec_dir = f"{config['DEFAULT']['GALFIND_WORK']}/Beta_paper"
    if incl_bands == []:
        title = "8_NIRCam"
    else:
        title = f"8_NIRCam+{str('+'.join([band.replace('f', 'F') for band in incl_bands]))}"
    # load in mock photometric beta
    phot_tab = Table.read(f"{input_phot_dir}/{title}.ecsv")
    # load in mock spectroscopic beta
    spec_tab = Table.read(f"{input_spec_dir}/{template_set}_Calzetti+94.ecsv")
    for i, plot_index in enumerate(plot_indices):
        mock_sed_rest = mock_sed_rest_set[int(plot_index)]
        loc_phot_tab = phot_tab[phot_tab["template_name"] == mock_sed_rest.template_name]
        C94_beta = spec_tab[spec_tab["template_name"] == mock_sed_rest.template_name]["Beta_C94"]
        ax.plot(loc_phot_tab["z"], np.array(loc_phot_tab[f"Beta_phot_{'-'.join([str(int(wav)) for wav in rest_UV_wav_lims.value])}Angstrom"]) - C94_beta, \
                label = mock_sed_rest.template_name, c = cmap[int(i)])
    if incl_bands == []:
        title = "8_NIRCam"
    else:
        title = f"8_NIRCam+{str('+'.join([band.replace('f', 'F') for band in incl_bands]))}"
    if annotate:
        ax.set_title(title)
        ax.set_xlabel("Redshift, z")
        ax.set_ylabel(rf"Δβ ({'-'.join([str(int(wav)) for wav in rest_UV_wav_lims.value])}$\AA$ $-$ C+94)")
        ax.legend(bbox_to_anchor = (1.05, 0.5), loc = "center left")
    if save:
        os.makedirs(output_dir, exist_ok = True)
        plt.savefig(f"{output_dir}/{title}.png")
    if show:
        plt.show()

if __name__ == "__main__":
    rest_UV_band_arr = ["f125W", "f140W", "f140M", "f160W", "f162M", "f182M", "f210M", "f250M", "f300M", "f335M"]
    z_arr = np.linspace(6.5, 12., 100)
    min_pc_err = 10
    template_set = "fsps_larson"
    rest_UV_wav_lims_arr = [[1250, 3000], [1268, 2580]] * u.Angstrom
    m_UV_norm = 25. # arbitrary if incl_errs == False
    fixed_depth_mag = 30.
    incl_errs = False
    
    #calc_C94_beta(template_set, m_UV_norm)
    
    band_combinations = []
    for i in range(2): #range(len(rest_UV_band_arr)):
        rest_bands_i_tups = list(combinations(rest_UV_band_arr, i))
        for tup in rest_bands_i_tups:
            band_combinations.append(list(tup))
    #print(np.array(band_combinations).shape)
    for incl_bands in tqdm(band_combinations, total = len(band_combinations), desc = f"Calculating β for {template_set} λ_rest={str(rest_UV_wav_lims_arr)} incl_errs={incl_errs}", leave = False):
        calc_mock_beta_phot(z_arr, template_set, incl_bands, rest_UV_wav_lims_arr, m_UV_norm, fixed_depth_mag, incl_errs = incl_errs)
        fig, ax = plt.subplots()
        #plot_dbeta(ax, z_arr, template_set, incl_bands, rest_UV_wav_lims_arr[0], m_UV_norm, fixed_depth_mag, incl_errs, cmap_name = "Oranges_r", plot_indices = np.linspace(0, 11, 12), show = False)
        plot_dbeta(ax, z_arr, template_set, incl_bands, rest_UV_wav_lims_arr[0], m_UV_norm, fixed_depth_mag, incl_errs, cmap_name = "Blues_r", plot_indices = np.linspace(12, 17, 6))
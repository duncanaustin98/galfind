#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:27:07 2023

@author: u92876da
"""

# IGM_attenuation.py
import numpy as np
import astropy.units as u
from astropy.table import Table
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from . import config, wav_lyman_lim

def calc_Inoue14_LS_LAF_optical_depth(lyman_series, wav_obs_arr, z):
    tau_arr = np.zeros((len(lyman_series), len(wav_obs_arr)))
    for j, wav_j in enumerate(np.array(lyman_series["lambda_j"])):
        valid_series_indices = ((wav_obs_arr > wav_j) & (wav_obs_arr < wav_j * (1 + z)))
        wav_indices_1 = ((wav_obs_arr < 2.2 * wav_j) & (valid_series_indices))
        wav_indices_2 = ((wav_obs_arr < 5.7 * wav_j) & (~wav_indices_1) & (valid_series_indices))
        wav_indices_3 = ((~wav_indices_1) & (~wav_indices_2) & (valid_series_indices))
        tau_arr[j, wav_indices_1] = lyman_series["A_j1_LAF"][j] * (wav_obs_arr[wav_indices_1] / wav_j) ** 1.2
        tau_arr[j, wav_indices_2] = lyman_series["A_j2_LAF"][j] * (wav_obs_arr[wav_indices_2] / wav_j) ** 3.7
        tau_arr[j, wav_indices_3] = lyman_series["A_j3_LAF"][j] * (wav_obs_arr[wav_indices_3] / wav_j) ** 5.5
    return np.sum(tau_arr, axis = 0)

def calc_Inoue14_LS_DLA_optical_depth(lyman_series, wav_obs_arr, z):
    tau_arr = np.zeros((len(lyman_series), len(wav_obs_arr)))
    for j, wav_j in enumerate(np.array(lyman_series["lambda_j"])):
        valid_series_indices = ((wav_obs_arr > wav_j) & (wav_obs_arr < wav_j * (1 + z)))
        wav_indices_1 = ((wav_obs_arr < 3 * wav_j) & (valid_series_indices))
        wav_indices_2 = ((~wav_indices_1) & (valid_series_indices))
        tau_arr[j, wav_indices_1] = lyman_series["A_j1_DLA"][j] * (wav_obs_arr[wav_indices_1] / wav_j) ** 2
        tau_arr[j, wav_indices_2] = lyman_series["A_j2_DLA"][j] * (wav_obs_arr[wav_indices_2] / wav_j) ** 3
    return np.sum(tau_arr, axis = 0)

def calc_Inoue14_LC_LAF_optical_depth(wav_obs_arr, z):
    tau = np.zeros(len(wav_obs_arr))
    gtr_lyman_lim_indices = (wav_obs_arr > wav_lyman_lim)
    if z > 0.:
        if z < 1.2:
            wav_indices = ((wav_obs_arr < wav_lyman_lim * (1 + z)) & (gtr_lyman_lim_indices))
            tau[wav_indices] = 0.325 * (((wav_obs_arr[wav_indices] / wav_lyman_lim) ** 1.2) - ((1 + z) ** -0.9) * ((wav_obs_arr[wav_indices] / wav_lyman_lim) ** 2.1))
        elif z < 4.7:
            wav_indices_1 = ((wav_obs_arr < 2.2 * wav_lyman_lim) & (gtr_lyman_lim_indices))
            wav_indices_2 = ((wav_obs_arr < wav_lyman_lim * (1 + z)) & (~wav_indices_1) & (gtr_lyman_lim_indices))
            tau[wav_indices_1] = (2.55e-2 * ((1 + z) ** 1.6) * ((wav_obs_arr[wav_indices_1] / wav_lyman_lim) ** 2.1)) + (0.325 * ((wav_obs_arr[wav_indices_1] / wav_lyman_lim) ** 1.2)) - (0.25 * ((wav_obs_arr[wav_indices_1] / wav_lyman_lim) ** 2.1))
            tau[wav_indices_2] = 2.55e-2 * ((((1 + z) ** 1.6) * ((wav_obs_arr[wav_indices_2] / wav_lyman_lim) ** 2.1)) - ((wav_obs_arr[wav_indices_2] / wav_lyman_lim) ** 3.7))
        else:
            wav_indices_1 = ((wav_obs_arr < 2.2 * wav_lyman_lim) & (gtr_lyman_lim_indices))
            wav_indices_2 = ((wav_obs_arr < 5.7 * wav_lyman_lim) & (~wav_indices_1) & (gtr_lyman_lim_indices))
            wav_indices_3 = ((wav_obs_arr < wav_lyman_lim * (1 + z)) & (~wav_indices_1) & (~wav_indices_2) & (gtr_lyman_lim_indices))
            tau[wav_indices_1] = (5.22e-4 * ((1 + z) ** 3.4) * (wav_obs_arr[wav_indices_1] / wav_lyman_lim) ** 2.1) + (0.325 * (wav_obs_arr[wav_indices_1] / wav_lyman_lim) ** 1.2) - (3.14e-2 * (wav_obs_arr[wav_indices_1] / wav_lyman_lim) ** 2.1)
            tau[wav_indices_2] = (5.22e-4 * ((1 + z) ** 3.4) * (wav_obs_arr[wav_indices_2] / wav_lyman_lim) ** 2.1) + (0.218 * (wav_obs_arr[wav_indices_2] / wav_lyman_lim) ** 2.1) - (2.55e-2 * (wav_obs_arr[wav_indices_2] / wav_lyman_lim) ** 3.7)
            tau[wav_indices_3] = 5.22e-4 * ((((1 + z) ** 3.4) * (wav_obs_arr[wav_indices_3] / wav_lyman_lim) ** 2.1) - ((wav_obs_arr[wav_indices_3] / wav_lyman_lim) ** 5.5))
    return tau

def calc_Inoue14_LC_DLA_optical_depth(wav_obs_arr, z):
    tau = np.zeros(len(wav_obs_arr))
    gtr_lyman_lim_indices = (wav_obs_arr > wav_lyman_lim)
    if z > 0.:
        if z < 2.:
            wav_indices = ((wav_obs_arr < wav_lyman_lim * (1 + z)) & (gtr_lyman_lim_indices))
            tau[wav_indices] = (0.211 * (1 + z) ** 2) - (7.66e-2 * (1 + z) ** 2.3) * ((wav_obs_arr[wav_indices] / wav_lyman_lim) ** -1.3) - (0.135 * (wav_obs_arr[wav_indices] / wav_lyman_lim) ** 2)
        else:
            wav_indices_1 = ((wav_obs_arr < 3 * wav_lyman_lim) & gtr_lyman_lim_indices)
            wav_indices_2 = ((wav_obs_arr < wav_lyman_lim * (1 + z)) & (~wav_indices_1) & (gtr_lyman_lim_indices))
            tau[wav_indices_1] = 0.634 + (4.7e-2 * (1 + z) ** 3) - (1.78e-2 * ((1 + z) ** 3.3) * ((wav_obs_arr[wav_indices_1] / wav_lyman_lim) ** -0.3))
            tau[wav_indices_2] = 4.7e-2 * (1 + z) ** 3 - 1.78e-2 * ((1 + z) ** 3.3) * ((wav_obs_arr[wav_indices_2] / wav_lyman_lim) ** -0.3) - 2.92e-2 * ((wav_obs_arr[wav_indices_2] / wav_lyman_lim) ** 3)
    return tau

def calc_IGM_transmission(lyman_series, wav_rest_arr, z, prescription = config["MockSEDs"]["IGM_PRESCRIPTION"]):
    if isinstance(wav_rest_arr, float):
        wav_rest_arr = np.array([wav_rest_arr])
    elif isinstance(wav_rest_arr, list):
        wav_rest_arr = np.array(wav_rest_arr)
    wav_obs_arr = wav_rest_arr * (1 + z)
    if prescription == "Inoue14":
        optical_depth = np.array(calc_Inoue14_LC_DLA_optical_depth(wav_obs_arr, z) + calc_Inoue14_LC_LAF_optical_depth(wav_obs_arr, z) + \
            calc_Inoue14_LS_DLA_optical_depth(lyman_series, wav_obs_arr, z) + calc_Inoue14_LS_LAF_optical_depth(lyman_series, wav_obs_arr, z))
    transmission = np.exp(-optical_depth)
    return transmission

def make_IGM_transmission_grid(wav_rest_arr, z_arr, prescription = config["MockSEDs"]["IGM_PRESCRIPTION"]):
    # allocate 2d IGM transmission grid memory
    IGM_transmission = np.zeros((len(z_arr), len(wav_rest_arr)))
    # load lyman series .fits file
    lyman_series = Table.read(f"{config['MockSEDs']['IGM_DIR']}/LS_absorption.fits")
    # calculate 2d IGM transmission grid
    for i, z in tqdm(enumerate(z_arr), total = len(z_arr), desc = f"Making {prescription} IGM grid"): 
        IGM_transmission[i, :] = calc_IGM_transmission(lyman_series, wav_rest_arr, z, prescription = prescription)
    with h5py.File(f"{config['MockSEDs']['IGM_DIR']}/{prescription}_IGM_grid.h5", "w") as IGM_grid:
        IGM_grid.create_dataset("IGM_transmission", data = IGM_transmission)
        IGM_grid.create_dataset("Redshifts", data = z_arr)
        IGM_grid.create_dataset("Rest_wavelengths", data = wav_rest_arr)
        IGM_grid.close()
        
def load_IGM_transmission_grid(prescription = config["MockSEDs"]["IGM_PRESCRIPTION"]):
    with h5py.File(f"{config['MockSEDs']['IGM_DIR']}/{prescription}_IGM_grid.h5", "r") as IGM_grid:
        IGM_transmission = IGM_grid["IGM_transmission"][()]
        z_arr = IGM_grid["Redshifts"][()]
        wav_rest_arr = IGM_grid["Rest_wavelengths"][()]
        IGM_grid.close()
    return IGM_transmission, z_arr, wav_rest_arr
        

    
    
    
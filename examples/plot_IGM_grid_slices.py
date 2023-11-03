#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 01:10:50 2023

@author: u92876da
"""

# plot_IGM_grid.py
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import astropy.units as u

from galfind import config, IGM_attenuation, wav_lyman_lim, wav_lyman_alpha

plt.style.use(f"{config['DEFAULT']['GALFIND_DIR']}/galfind_style.mplstyle")

igm_prescription = "Inoue+14"
z_arr = [2.]
wav_rest_arr = np.linspace(wav_lyman_lim, wav_lyman_alpha, 505) * u.AA
frame = "obs"

fig, ax = plt.subplots(figsize = (8, 8))
for z in z_arr:
    igm_obj = IGM_attenuation.IGM(prescription = igm_prescription)
    plot_kwargs = {"label": f"z = {z}"}
    igm_obj.plot_slice(ax, z, wav_rest_arr, frame = frame, plot_kwargs = plot_kwargs, show = False)
plt.show()
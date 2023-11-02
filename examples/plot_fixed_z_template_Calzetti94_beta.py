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

from galfind import SED, Mock_SED_rest, Mock_SED_obs, config, NIRCam

z = 4.
min_pc_err = 10
NIRCam_excl_bands = ["f070W", "f140M", "f162M", "f182M", "f210M", "f250M", "f300M", "f335M", "f360M", "f430M", "f460M", "f480M"]

fig, ax = plt.subplots()
instrument = NIRCam(excl_bands = NIRCam_excl_bands)
for i in range(13, 19):
    sed_obj_rest = Mock_SED_rest.load_EAZY_in_template(25., "fsps_larson", i)
    sed_obj_obs = Mock_SED_obs.from_Mock_SED_rest(sed_obj_rest, z)
    beta = sed_obj_obs.calc_UV_slope(output_errs = False)[1]
    sed_obj_obs.create_mock_photometry(instrument, depths, min_pc_err)
    
    print(beta)
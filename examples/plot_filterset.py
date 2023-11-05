#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 13:54:02 2023

@author: u92876da
"""
import matplotlib.pyplot as plt
from galfind import Instrument, NIRCam, WFC3_IR, ACS_WFC

instrument = NIRCam() #+ WFC3_IR() + ACS_WFC()
plot_bands = ["f090W", "f115W", "f150W", "f200W", "f210M", "f277W", "f356W", "f360M", "f410M", "f444W"]#["f090W", "f115W", "f150W", "f200W", "f277W", "f335M", "f356W", "f410M", "f444W"]
#plot_bands = instrument.bands
fig, ax = plt.subplots()
# still need to fix title naming system
instrument.plot_filter_profiles(ax, plot_bands, show = False)
ax.set_ylim(0., 0.6)
plt.show()
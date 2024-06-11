#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 13:54:02 2023

@author: u92876da
"""
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from galfind import NIRCam, WFC3_IR, ACS_WFC
from galfind.Instrument import MICADO

def main():
    wav_units = u.um
    plot_bands = ["F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F360M", "F410M", "F444W"]
    excl_bands = [band_name for band_name in NIRCam().band_names if band_name[-1] not in ["W", "M"]]
    #instrument = MICADO(excl_bands = excl_bands)# + NIRCam()
    instrument = NIRCam(excl_bands = excl_bands) #+ WFC3_IR() + ACS_WFC()
    
    fig, ax = plt.subplots()
    instrument.plot_filter_profiles(ax, wav_units, save = True, show = False)

if __name__ == "__main__":
    main()
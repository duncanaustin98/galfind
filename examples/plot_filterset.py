#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 13:54:02 2023

@author: u92876da
"""
import matplotlib.pyplot as plt
import astropy.units as u

from galfind import Instrument, NIRCam, WFC3_IR, ACS_WFC

def main():
    wav_units = u.um
    plot_bands = ["F850LP", "F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F360M", "F410M", "F444W"]
    NIRCam_excl_bands = [band_name for band_name in NIRCam().band_names if band_name not in plot_bands]
    ACS_WFC_excl_bands = [band_name for band_name in ACS_WFC().band_names if band_name not in plot_bands]
    instrument = NIRCam(excl_bands = NIRCam_excl_bands) + ACS_WFC(ACS_WFC_excl_bands) #+ WFC3_IR() + ACS_WFC()

    fig, ax = plt.subplots()
    instrument.plot_filter_profiles(ax, wav_units, save = True, show = False)

if __name__ == "__main__":
    main()
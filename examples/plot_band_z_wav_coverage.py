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
from galfind import config
from galfind.Instrument import MICADO

plt.style.use(f"{config['DEFAULT']['GALFIND_DIR']}/galfind_style.mplstyle")

def main():
    wav_units = u.um
    excl_bands = ["Spec_IJ", "Spec_HK", "FeII", "xI1", "xI2", "xY1", "xY2", \
        "xJ1", "xJ2", "xH1", "xH2", "xK1", "xK2", "Ks"]
    instrument = NIRCam() #MICADO(excl_bands = excl_bands)
    #instrument = NIRCam(excl_bands = excl_bands) #+ WFC3_IR() + ACS_WFC()
    
    #fig, ax = plt.subplots()
    #instrument.plot_filter_profiles(ax, wav_units, save = True, show = False)

    z_arr = np.linspace(2., 13.5)
    instrument.plot_z_wav_rest_tracks(z_arr)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:14:30 2023

@author: austind
"""

# Photometry_obs.py
import numpy as np
import astropy.constants as const
import astropy.units as u
from copy import copy, deepcopy
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from . import useful_funcs_austind as funcs
from . import astropy_cosmo
from . import config

class Photometry:
    
    def __init__(self, instrument, flux_Jy, flux_Jy_errs, loc_depths):
        self.instrument = instrument
        self.flux_Jy = flux_Jy
        self.flux_Jy_errs = flux_Jy_errs
        self.loc_depths = loc_depths
    
    @classmethod
    def from_fits_cat(cls, fits_cat_row, instrument, cat_creator):
        fluxes = []
        flux_errs = []
        # Copy constructor problem here!
        instrument_copy = instrument.new_instrument(excl_bands = [band for band in instrument.new_instrument().bands if band not in instrument.bands]) # Problem loading the photometry properly here!!!
        for band in instrument_copy.bands:
            try:
                flux, err = cat_creator.load_photometry(fits_cat_row, band)
                #print(flux, err)
                fluxes.append(flux.value)
                flux_errs.append(err.value)
            except:
                # no data for the relevant band within the catalogue
                instrument.remove_band(band)
                #print(f"{band} flux not loaded")
        try:
            # local depths only currently works for one aperture diameter
            loc_depths = np.array([fits_cat_row[f"loc_depth_{band}"].T[cat_creator.aper_diam_index] for band in instrument.bands])
        except:
            #print("loc depths not loaded")
            loc_depths = None
        return cls(instrument, fluxes * u.Jy, flux_errs * u.Jy, loc_depths)
    
    def crop_phot(self, indices):
        indices = np.array(indices).astype(int)
        for index in reversed(indices):
            self.instrument.remove_band(self.instrument.bands[index])
        self.flux_Jy = np.delete(self.flux_Jy, indices)
        self.flux_Jy_errs = np.delete(self.flux_Jy_errs, indices)


            
            
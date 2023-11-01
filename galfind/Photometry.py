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
    
    def __init__(self, instrument, flux_Jy, flux_Jy_errs, depths):
        self.instrument = instrument
        # check that the fluxes and errors are in the correct units
        self.flux_Jy = flux_Jy
        self.flux_Jy_errs = flux_Jy_errs
        self.loc_depths = depths
    
    @classmethod
    def from_fits_cat(cls, fits_cat_row, instrument, cat_creator):
        fluxes, flux_errs = cat_creator.load_photometry(fits_cat_row, instrument.bands)
        try:
            # local depths only currently works for one aperture diameter
            loc_depths = np.array([fits_cat_row[f"loc_depth_{band}"].T[cat_creator.aper_diam_index] for band in instrument.bands])
        except:
            #print("local depths not loaded")
            loc_depths = None
        return cls(instrument, fluxes[0] * u.Jy, flux_errs[0] * u.Jy, loc_depths)
    
    def scatter_phot(self, n_scatter = 1):
        phot_matrix = np.random.normal(self.flux_Jy.value, self.flux_Jy_errs.value, n_scatter)
        phot_obj_arr = [Photometry(self.instrument, phot_matrix[i], self.flux_Jy_errs.value, self.loc_depths) for i in range(n_scatter)]
        if n_scatter == 1:
            return phot_obj_arr[0]
        else:
            return phot_obj_arr
    
    def crop_phot(self, indices):
        indices = np.array(indices).astype(int)
        for index in reversed(indices):
            self.instrument.remove_band(self.instrument.bands[index])
        self.flux_Jy = np.delete(self.flux_Jy, indices)
        self.flux_Jy_errs = np.delete(self.flux_Jy_errs, indices)
        
    def plot_phot(self, ax, wav_units = u.AA, mag_units = u.Jy, plot_errs = True, annotate = False, upper_limit_sigma = 3., errorbar_kwargs = {}):
        if upper_limit_sigma == None:
            uplims = np.full(len(self.flux_Jy), False)
        else:
            # calculate upper limits based on depths
            uplims = [True if flux.to(u.Jy) < depth.to(u.Jy) * upper_limit_sigma / 5. else False for (flux, depth) in zip(self.flux_Jy, self.loc_depths)]
        if plot_errs:
            yerr = [flux_err if uplim == False else 0.2 * flux for (flux, flux_err, uplim) in zip(self.flux_Jy.value, self.flux_Jy_errs.value, uplims)]
        else:
            yerr = None
            print("Unit plotting errors here!")
        plot = ax.errorbar([wav.to(wav_units).value for (band, wav) in self.instrument.band_wavelengths.items()], self.flux_Jy.value, yerr = yerr, \
                uplims = uplims, ls = "", marker = "o", ms = 8, mfc = "none", **errorbar_kwargs)
        if annotate:
            ax.legend()
        return plot

        
class Multiple_Photometry:
    
    def __init__(self, instrument, flux_Jy_arr, flux_Jy_errs_arr, loc_depths_arr):
        self.phot_arr = [Photometry(instrument, flux_Jy, flux_Jy_errs, loc_depths) for flux_Jy, flux_Jy_errs, loc_depths in zip(flux_Jy_arr, flux_Jy_errs_arr, loc_depths_arr)]        
        
    @classmethod
    def from_fits_cat(cls, fits_cat, instrument, cat_creator):
        flux_Jy_arr, flux_Jy_errs_arr = cat_creator.load_photometry(fits_cat, instrument.bands)
        # local depths not yet loaded in
        loc_depths_arr = np.full(len(flux_Jy_arr), None)
        return cls(instrument, flux_Jy_arr, flux_Jy_errs_arr, loc_depths_arr)
    
class Mock_Photometry(Photometry):
    
    def __init__(self, instrument, flux_Jy, depths, min_pc_err): # these depths should be 5σ and in units of ABmag
        assert(len(flux_Jy) == len(depths))
        # add astropy units of ABmag if depths are not already
        try:
            assert(depths.unit == u.ABmag)
        except:
            depths *= u.ABmag
        # calculate errors from ABmag depths
        flux_Jy_errs = self.flux_errs_from_depths(flux_Jy, depths, min_pc_err)
        super().__init__(instrument, flux_Jy, flux_Jy_errs, depths)
        
    @staticmethod
    def flux_errs_from_depths(flux_Jy, depths, min_pc_err):
        # calculate 1σ depths to Jy
        one_sig_depths_Jy = depths.to(u.Jy) / 5
        # apply min_pc_err criteria
        flux_Jy_errs = np.array([depth if depth > flux * min_pc_err / 100 else flux * min_pc_err / 100 for flux, depth in zip(flux_Jy.value, one_sig_depths_Jy.value)]) * u.Jy
        return flux_Jy_errs
        
            
            
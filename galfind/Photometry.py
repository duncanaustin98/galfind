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
        self.loc_depths = depths # not sure what problems renaming this will have
        self.depths = depths # stores exactly the same info as self.loc_depths, but it is a pain to propagate deletion of the above so it is left for now
        assert(len(self.instrument) == len(self.flux_Jy) == len(self.flux_Jy_errs) == len(self.depths))

    def __str__(self, print_cls_name = True, print_instrument = False, print_fluxes = True, print_depths = True):
        line_sep = "*" * 40 + "\n"
        band_sep = "-" * 10 + "\n"
        output_str = ""

        if print_cls_name:
            output_str += line_sep
            output_str += "PHOTOMETRY:\n"
            output_str += band_sep

        if print_instrument:
            output_str += str(self.instrument)
        
        #if print_fluxes:
        fluxes_str = ["%.1f ± %.1f nJy" %(flux_Jy.to(u.nJy).value, flux_Jy_err.to(u.nJy).value) \
            for flux_Jy, flux_Jy_err in zip(self.flux_Jy.filled(fill_value = np.nan), self.flux_Jy_errs.filled(fill_value = np.nan))]
        output_str += f"FLUXES: {fluxes_str}\n"
        output_str += f"MAGS: {[np.round(mag, 2) for mag in self.flux_Jy.filled(fill_value = np.nan).to(u.ABmag).value]}\n"
        #if print_depths:
        output_str += f"DEPTHS: {[np.round(depth, 2) for depth in self.depths.value]}\n"

        if print_cls_name:
            output_str += line_sep
        return output_str
    
    def __getitem__(self, i):
        if type(i) == int:
            return self.flux_Jy[i]
        elif type(i) == str:
            return self.flux_Jy[self.instrument.index_from_band(i)]
        else:
            raise(TypeError(f"i={i} in {__class__.__name__}.__getitem__ has type={type(i)} which is not in [int, str]"))
    
    def __len__(self):
        return len(self.flux_Jy)
    
    @property
    def wav(self):
        return self.instrument.band_wavelengths
        #return np.array([self.instrument.band_wavelengths[band].value for band in self.instrument.band_names]) * u.Angstrom
    
    @classmethod
    def from_fits_cat(cls, fits_cat_row, instrument, cat_creator):
        fluxes, flux_errs = cat_creator.load_photometry(fits_cat_row, instrument.band_names)
        try:
            # local depths only currently works for one aperture diameter
            loc_depths = np.array([fits_cat_row[f"loc_depth_{band_name}"].T[cat_creator.aper_diam_index] for band_name in instrument.band_names])
        except:
            #print("local depths not loaded")
            loc_depths = None
        return cls(instrument, fluxes[0], flux_errs[0], loc_depths)
    
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
            self.instrument.remove_band(self.instrument[index])
        self.flux_Jy = np.delete(self.flux_Jy, indices)
        self.flux_Jy_errs = np.delete(self.flux_Jy_errs, indices)
        
    def plot_phot(self, ax, wav_units = u.AA, mag_units = u.Jy, plot_errs = True, annotate = True, upper_limit_sigma = 3., errorbar_kwargs = {}, label = None):
        if upper_limit_sigma == None:
            uplims = np.full(len(self.flux_Jy), False)
        else:
            # calculate upper limits based on depths
            uplims = [True if flux.to(u.Jy) < depth.to(u.Jy) * upper_limit_sigma / 5. or np.isnan(flux) else False for (flux, depth) in zip(self.flux_Jy, self.loc_depths)]
        self.non_detected_indices = uplims
        if plot_errs:
            yerr = [flux_err if uplim == False else 0.2 * flux for (flux, flux_err, uplim) in zip(self.flux_Jy.value, self.flux_Jy_errs.value, uplims)]
        else:
            yerr = None
        print("Unit plotting errors here!")
        plot = ax.errorbar(self.wav.to(wav_units).value, self.flux_Jy.value, yerr = yerr, \
                uplims = uplims, ls = "", marker = "o", ms = 8, mfc = "none", label = label, **errorbar_kwargs)
        if label != None and annotate:
            ax.legend()
        return plot

class Multiple_Photometry:
    
    def __init__(self, instrument, flux_Jy_arr, flux_Jy_errs_arr, loc_depths_arr):
        self.phot_arr = [Photometry(instrument, flux_Jy, flux_Jy_errs, loc_depths) for flux_Jy, flux_Jy_errs, loc_depths in zip(flux_Jy_arr, flux_Jy_errs_arr, loc_depths_arr)]        
        
    @classmethod
    def from_fits_cat(cls, fits_cat, instrument, cat_creator):
        flux_Jy_arr, flux_Jy_errs_arr = cat_creator.load_photometry(fits_cat, instrument.band_names)
        # local depths not yet loaded in
        loc_depths_arr = np.full(flux_Jy_arr.shape, None)
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
        self.min_pc_err = min_pc_err
        super().__init__(instrument, flux_Jy, flux_Jy_errs, depths)
        
    @staticmethod
    def flux_errs_from_depths(flux_Jy, depths, min_pc_err):
        # calculate 1σ depths to Jy
        one_sig_depths_Jy = depths.to(u.Jy) / 5
        # apply min_pc_err criteria
        flux_Jy_errs = np.array([depth if depth > flux * min_pc_err / 100 else flux * min_pc_err / 100 for flux, depth in zip(flux_Jy.value, one_sig_depths_Jy.value)]) * u.Jy
        return flux_Jy_errs
    
    def scatter(self, size = 1):
        scattered_fluxes = np.zeros((len(self.flux_Jy), size))
        for i, (flux, err) in enumerate(zip(self.flux_Jy, self.flux_Jy_errs)):
            scattered_fluxes[i] = np.random.normal(flux.value, err.value, size = size)
        if size == 1:
            scattered_fluxes = scattered_fluxes.flatten()
            self.scattered_phot = [Mock_Photometry(self.instrument, scattered_fluxes * u.Jy, self.depths, self.min_pc_err)]
        else:
            self.scattered_phot = [Mock_Photometry(self.instrument, scattered_fluxes.T[i] * u.Jy, self.depths, self.min_pc_err) for i in range(size)]

        
            
            
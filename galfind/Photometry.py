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
import matplotlib.patheffects as pe

from . import useful_funcs_austind as funcs
from . import config, galfind_logger, astropy_cosmo

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

    def plot_phot(self, ax, wav_units = u.AA, mag_units = u.Jy, plot_errs = True, plot_band_widths = True, \
            annotate = True, uplim_sigma = 2., uplim_sigma_arrow = 0.5, auto_scale = True, \
            errorbar_kwargs = {"ls": "", "marker": "o", "ms": 6., "zorder": 100., "path_effects": [pe.withStroke(linewidth = 3, foreground = "white")]}, \
            filled = True, colour = "black", label = "Photometry"):

        wavs_to_plot = funcs.convert_wav_units(self.wav, wav_units)
        mags_to_plot = funcs.convert_mag_units(self.wav, self.flux_Jy, mag_units)

        if uplim_sigma == None:
            uplims = list(np.full(len(self.flux_Jy), False))
        else:
            assert uplim_sigma_arrow < uplim_sigma, \
                galfind_logger.critical(f"uplim_sigma_arrow = {uplim_sigma_arrow} < uplim_sigma = {uplim_sigma}")
            # calculate upper limits based on depths
            uplims = [True if SNR < uplim_sigma else False for SNR in self.SNR]
            # set photometry to uplim_sigma for the data to be plotted as upper limits
            uplim_indices = [i for i, is_uplim in enumerate(uplims) if is_uplim]
            uplim_vals = [funcs.convert_mag_units(self.wav, funcs.convert_mag_units(self.wav, self.depths[i], u.Jy) \
                * uplim_sigma / 5., mag_units).value for i in uplim_indices] * mag_units
            mags_to_plot.put(uplim_indices, uplim_vals)
            galfind_logger.debug("Should test whether upper plotting limits preserves the mask!")
        self.non_detected_indices = uplims

        if plot_errs:
            mag_errs_new_units = funcs.convert_mag_err_units(self.wav, self.flux_Jy, [self.flux_Jy_errs, self.flux_Jy_errs], mag_units)
            # update with upper limit errors
            uplim_l1_vals = [funcs.convert_mag_units(self.wav, funcs.convert_mag_units(self.wav, self.depths[i], u.Jy) \
                * uplim_sigma_arrow / 5., mag_units).value for i in uplim_indices] * mag_units
            #breakpoint()
            if mag_units == u.ABmag:
                # swap l1 / u1 errors
                uplim_u1_vals = (uplim_l1_vals - uplim_vals).value
                uplim_l1_vals = [np.nan for i in uplim_indices]
            else:
                uplim_l1_vals = (uplim_vals - uplim_l1_vals).value
                uplim_u1_vals = [np.nan for i in uplim_indices]
            yerr = []
            for i, uplim_errs in enumerate([uplim_l1_vals, uplim_u1_vals]):
                mag_errs = mag_errs_new_units[i].value
                mag_errs.put(uplim_indices, uplim_errs)
                yerr.append(mag_errs)
        else:
            yerr = None
        
        if plot_band_widths:
            xerr = [[funcs.convert_wav_units(filter.WavelengthCen - filter.WavelengthLower50, wav_units).value for filter in self.instrument], \
                    [funcs.convert_wav_units(filter.WavelengthUpper50 - filter.WavelengthCen, wav_units).value for filter in self.instrument]]
        else:
            xerr = None

        # update errorbar kwargs - not quite general
        if filled:
            errorbar_kwargs["mfc"] = colour
        else:
            errorbar_kwargs["mfc"] = "none"
        errorbar_kwargs["color"] = colour
        errorbar_kwargs["label"] = label

        if auto_scale:
            # auto-scale the x-axis
            lower_xlim = np.min(wavs_to_plot.value - xerr[0]) * 0.95
            upper_xlim = np.max(wavs_to_plot.value + xerr[1]) * 1.05
            ax.set_xlim(lower_xlim, upper_xlim)
            # auto-scale the y-axis based on plotting units
            if mags_to_plot.unit == u.ABmag:
                lower_ylim = np.max(mags_to_plot.value) + 0.75
                upper_ylim = np.min(mags_to_plot.value) - 1.5
            else: # auto-scale flux units
                lower_ylim = np.min(mags_to_plot.value) * 0.9
                upper_ylim = np.max(mags_to_plot.value) * 1.2
            ax.set_ylim(lower_ylim, upper_ylim)

        if mag_units == u.ABmag:
            plot_limits = {"lolims": uplims}
        else:
            plot_limits = {"uplims": uplims}
        
        plot = ax.errorbar(wavs_to_plot.value, mags_to_plot.value, xerr = xerr, yerr = yerr, **plot_limits, **errorbar_kwargs)
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

        
            
            
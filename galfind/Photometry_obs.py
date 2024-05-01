#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:03:20 2023

@author: austind
"""

# Photometry_obs.py
import numpy as np
import astropy.constants as const
import astropy.units as u
from copy import copy, deepcopy
import matplotlib.patheffects as pe

from . import useful_funcs_austind as funcs
from . import galfind_logger
from .Photometry import Photometry, Multiple_Photometry
from .SED_result import Galaxy_SED_results, Catalogue_SED_results

class Photometry_obs(Photometry):

    def __init__(self, instrument, flux_Jy, flux_Jy_errs, aper_diam, min_flux_pc_err, loc_depths, SED_results = {}):
        self.aper_diam = aper_diam
        self.min_flux_pc_err = min_flux_pc_err
        self.SED_results = SED_results # array of SED_result objects with different SED fitting runs
        self.aper_corrs = [instrument.aper_corr(self.aper_diam, band) for band in instrument.band_names]
        super().__init__(instrument, flux_Jy, flux_Jy_errs, loc_depths)

    def __str__(self):
        line_sep = "*" * 40 + "\n"
        band_sep = "-" * 10 + "\n"
        output_str = line_sep
        output_str += "PHOTOMETRY OBS:\n"
        output_str += band_sep
        output_str += f"APERTURE DIAMETER: {self.aper_diam}\n"
        output_str += f"MIN FLUX PC ERR: {self.min_flux_pc_err}%\n"
        output_str += super().__str__(print_cls_name = False)
        for result in self.SED_results.values():
            output_str += str(result)
        output_str += f"SNR: {[np.round(snr, 2) for snr in self.SNR]}\n"
        output_str += line_sep
        return output_str

    @property
    def SNR(self):
        return [(flux_Jy * 10 ** (aper_corr / -2.5)) * 5 / depth if flux_Jy > 0. else flux_Jy * 5 / depth \
            for aper_corr, flux_Jy, depth in zip(self.aper_corrs, self.flux_Jy.filled(fill_value = np.nan).to(u.Jy).value, self.depths.to(u.Jy).value)]

    @classmethod # not a gal object here, more like a catalogue row
    def from_fits_cat(cls, fits_cat_row, instrument, cat_creator, aper_diam, min_flux_pc_err, codes, lowz_zmaxs, templates):
        phot = Photometry.from_fits_cat(fits_cat_row, instrument, cat_creator)
        SED_results = Galaxy_SED_results.from_fits_cat(fits_cat_row, cat_creator, codes, lowz_zmaxs, templates, instrument = instrument)
        return cls.from_phot(phot, aper_diam, min_flux_pc_err, SED_results)
    
    @classmethod
    def from_phot(cls, phot, aper_diam, min_flux_pc_err, SED_results = {}):
        return cls(phot.instrument, phot.flux_Jy, phot.flux_Jy_errs, aper_diam, min_flux_pc_err, phot.loc_depths, SED_results)
    
    def update(self, gal_SED_results):
        if hasattr(self, "SED_results"):
            self.SED_results = {**self.SED_results, **gal_SED_results}
        else:
            self.SED_results = gal_SED_results
    
    def update_mask(self, cat, cat_creator, ID, update_phot_rest = False):
        gal_index = np.where(cat[cat_creator.ID_label] == ID)[0][0]
        mask = cat_creator.load_mask(cat, self.instrument.band_names)[gal_index]
        self.flux_Jy.mask = mask
        self.flux_Jy_errs.mask = mask
        return self
    
    #def get_SED_fit_params_arr(self, code):
    #    return [code.SED_fit_params_from_label(label) for label in self.SED_results.keys()]

    def plot_phot(self, ax, wav_units = u.AA, mag_units = u.Jy, plot_errs = True, plot_band_widths = True, \
            annotate = True, uplim_sigma = 2., uplim_sigma_arrow = 1.5, auto_scale = True, label_SNRs = False, \
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
            if mag_units == u.ABmag:
                # swap l1 / u1 errors
                uplim_u1_vals = (uplim_l1_vals - uplim_vals).to(u.dimensionless_unscaled).value
                uplim_l1_vals = [np.nan for i in uplim_indices]
            else:
                uplim_l1_vals = (uplim_vals - uplim_l1_vals).to(u.dimensionless_unscaled).value
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
                lower_ylim = np.max(mags_to_plot.value) + 1.
                upper_ylim = np.min(mags_to_plot.value) - 1.
            # Tom's auto-scaling
            # if mag.value < ax.get_ylim()[1] + 1 and mag.value > 15: 
            #     new_lim = mag.value - 1
            #     ax_photo.set_ylim(ax.get_ylim()[0], new_lim)
            # if mag.value > ax.get_ylim()[0] and mag.value < 32 and mag.value > 15:
            #     new_lim = two_sig_depth.value + 0.5
            #     ax.set_ylim(new_lim, ax.get_ylim()[1])
            ax.set_ylim(lower_ylim, upper_ylim)

        if mag_units == u.ABmag:
            plot_limits = {"lolims": uplims}
        else:
            plot_limits = {"uplims": uplims}
        plot = ax.errorbar(wavs_to_plot.value, mags_to_plot.value, xerr = xerr, yerr = yerr, **plot_limits, **errorbar_kwargs)

        # could probably go into overridden Photometry_obs method
        if label_SNRs:
            label_kwargs = {"ha": "center", "fontsize": "medium", "path_effects": [pe.withStroke(linewidth = 3, foreground = "white")], "zorder": 1_000.}
            [ax.annotate(f"{SNR:.1f}$\sigma$" if SNR < 100 else f"{SNR:.0f}$\sigma$", (funcs.convert_wav_units(filter.WavelengthCen, wav_units).value, \
                ax.get_ylim()[0] - 0.2 if i % 2 == 0 else ax.get_ylim()[0] - 0.6), **label_kwargs) for i, (SNR, filter) in enumerate(zip(self.SNR, self.instrument))]
        
        if annotate:
            # x/y labels etc here
            ax.legend()

        return plot

    #def load_local_depths(self, sex_cat_row, instrument, aper_diam_index):
    #    self.loc_depths = np.array([sex_cat_row[f"loc_depth_{band}"].T[aper_diam_index] for band in instrument.band_names])
        
    # def SNR_crop(self, band, sigma_detect_thresh):
    #     index = self.instrument.band_from_index(band)
    #     # local depth in units of Jy
    #     loc_depth_Jy = self.loc_depths[index].to(u.Jy) / 5
    #     detection_Jy = self.flux_Jy[index].to(u.Jy)
    #     sigma_detection = (detection_Jy / loc_depth_Jy).value
    #     if sigma_detection >= sigma_detect_thresh:
    #         return True
    #     else:
    #         return False

# %%    
        
class Multiple_Photometry_obs:
    
    def __init__(self, instrument, flux_Jy_arr, flux_Jy_errs_arr, aper_diam, min_flux_pc_err, loc_depths_arr, SED_results_arr = []):
        # force SED_results_arr to have the same len as the number of input fluxes
        if SED_results_arr == []:
            SED_results_arr = np.full(len(flux_Jy_arr), {})
        self.phot_obs_arr = [Photometry_obs(instrument, flux_Jy, flux_Jy_errs, aper_diam, min_flux_pc_err, loc_depths, SED_results) \
            for flux_Jy, flux_Jy_errs, loc_depths, SED_results in zip(flux_Jy_arr, flux_Jy_errs_arr, loc_depths_arr, SED_results_arr)]

    def __str__(self):
        # string representation of what is stored in this class
        return str(self.__dict__)
    
    def __len__(self):
        return len(self.phot_obs_arr)
    
    def __iter__(self):
        self.iter = 0
        return self
    
    def __next__(self):
        if self.iter > len(self) - 1:
            raise StopIteration
        else:
            phot = self[self.iter]
            self.iter += 1
            return phot
    
    def __getitem__(self, index):
        return self.phot_obs_arr[index]

    @classmethod
    def from_fits_cat(cls, fits_cat, instrument, cat_creator, SED_fit_params_arr):
        flux_Jy_arr, flux_Jy_errs_arr = cat_creator.load_photometry(fits_cat, instrument.band_names)
        depths_arr = cat_creator.load_depths(fits_cat, instrument.band_names)
        if SED_fit_params_arr != [{}]:
            SED_results_arr = Catalogue_SED_results.from_fits_cat(fits_cat, cat_creator, SED_fit_params_arr, instrument = instrument).SED_results
        else:
            SED_results_arr = np.full(len(flux_Jy_arr), {})
        return cls(instrument, flux_Jy_arr, flux_Jy_errs_arr, cat_creator.aper_diam, cat_creator.min_flux_pc_err, depths_arr, SED_results_arr)

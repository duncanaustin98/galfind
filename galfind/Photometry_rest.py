#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:04:24 2023

@author: austind
"""

# Photometry_rest.py
import numpy as np
import astropy.constants as const
import astropy.units as u
from copy import copy, deepcopy
import traceback
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from abc import ABC

from . import config, galfind_logger, astropy_cosmo, Photometry, PDF, PDF_nD
from . import useful_funcs_austind as funcs
from .Emission_lines import line_diagnostics
from .Dust_Attenuation import AUV_from_beta, Meurer99
from .decorators import ignore_warnings

class beta_fit:
    def __init__(self, z, bands):
        self.band_names = [band.band_name for band in bands]
        self.wavelength_rest = {}
        self.transmission = {}
        self.norm = {}
        for band in bands:
            self.wavelength_rest[band.band_name] = np.array(funcs.convert_wav_units(band.wav, u.AA).value / (1. + z))
            self.transmission[band.band_name] = np.array(band.trans)
            self.norm[band.band_name] = np.trapz(self.transmission[band.band_name], x = self.wavelength_rest[band.band_name])
    def beta_slope_power_law_func_conv_filt(self, _, A, beta):
        return np.array([np.trapz((10 ** A) * (self.wavelength_rest[band_name] ** beta) * self.transmission[band_name], \
            x = self.wavelength_rest[band_name]) / self.norm[band_name] for band_name in self.band_names])

SFR_conversions = \
{
    "MD14": 1.15e-28 * (u.solMass / u.yr) / (u.erg / (u.s * u.Hz))
}

fesc_from_beta_conversions = \
{
    "Chisholm22": lambda beta: np.random.normal(1.3, 0.6, len(beta)) * 10 ** (-4. - np.random.normal(1.22, 0.1, len(beta)) * beta)
}

class Photometry_rest(Photometry):
    
    def __init__(self, instrument, flux_Jy, flux_Jy_errs, depths, z, properties = {}, property_errs = {}, property_PDFs = {}):
        self.z = z
        self.properties = properties
        self.property_errs = property_errs
        self.property_PDFs = property_PDFs
        super().__init__(instrument, flux_Jy, flux_Jy_errs, depths)
    
    # these class methods need updating!
    @classmethod
    def from_fits_cat(cls, fits_cat_row, instrument, cat_creator, code):
        phot = Photometry.from_fits_cat(fits_cat_row, instrument, cat_creator)
        return cls.from_phot(phot, np.float(fits_cat_row[code.galaxy_properties["z"]]))
    
    @classmethod
    def from_phot(cls, phot, z):
        return cls(phot.instrument, phot.flux_Jy, phot.flux_Jy_errs, phot.depths, z)
    
    @classmethod
    def from_phot_obs(cls, phot):
        return cls(phot.instrument, phot.flux_Jy, phot.flux_Jy_errs, phot.depths, phot.z)
    
    def __len__(self):
        return len(self.flux_Jy)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, deepcopy(value, memo))
        return result
    
    @property
    def first_Lya_detect_band(self, Lya_wav = line_diagnostics["Lya"]["line_wav"]):
        try:
            return self._first_Lya_detect_band
        except AttributeError:
            first_band = None
            for band, lower_wav in zip(self.instrument.band_names, self.instrument.band_lower_wav_lims):
                if lower_wav > Lya_wav * (1 + self.z):
                    first_band = band
                    break
            self._first_Lya_detect_band = first_band
            return self._first_Lya_detect_band

    @property
    def first_Lya_non_detect_band(self, Lya_wav = line_diagnostics["Lya"]["line_wav"]):
        try:
            return self._first_Lya_non_detect_band
        except AttributeError:
            first_band = None
            # bands already ordered from blue -> red
            for band, upper_wav in zip(self.instrument.band_names, self.instrument.band_upper_wav_lims):
                if upper_wav < Lya_wav * (1 + self.z):
                    first_band = band
                    break
            self._first_Lya_non_detect_band = first_band
        return self._first_Lya_non_detect_band
    
    @property
    def lum_distance(self):
        try:
            return self._lum_distance
        except AttributeError:
            self._lum_distance = astropy_cosmo.luminosity_distance(self.z).to(u.pc)
            return self._lum_distance
    
    # @property
    # def rest_UV_band_index(self):
    #     return np.abs(self.wav - 1500 * u.Angstrom).argmin()
    
    # @property
    # def rest_UV_band(self):
    #     return self.instrument[self.rest_UV_band_index]
    
    # @property
    # def rest_UV_band_flux_Jy(self):
    #     return self.flux_Jy[self.rest_UV_band_index]
    
    def scatter_phot(self, n_scatter = 1):
        assert self.flux_Jy.unit != u.ABmag, galfind_logger.critical(f"{self.flux_Jy.unit=} == 'ABmag'")
        phot_matrix = np.array([np.random.normal(flux, err, n_scatter) for flux, err in zip(self.flux_Jy.value, self.flux_Jy_errs.value)])
        return [self.__class__(self.instrument, phot_matrix[:, i] * self.flux_Jy.unit, self.flux_Jy_errs, self.depths, self.z) for i in range(n_scatter)]

    @staticmethod
    def rest_UV_wavs_name(rest_UV_wav_lims):
        assert u.get_physical_type(rest_UV_wav_lims == "length"), \
            galfind_logger.critical(f"{u.get_physical_type(rest_UV_wav_lims)=} != 'length'")
        rest_UV_wav_lims = [int(funcs.convert_wav_units(rest_UV_wav_lim * rest_UV_wav_lims.unit, u.AA).value) \
            for rest_UV_wav_lim in rest_UV_wav_lims.value]
        return f"{str(rest_UV_wav_lims).replace(' ', '')}AA"

    def get_rest_UV_phot(self, rest_UV_wav_lims):
        phot_rest_copy = deepcopy(self)
        rest_UV_wav_lims = funcs.convert_wav_units(rest_UV_wav_lims, u.AA)
        crop_indices = [int(i) for i, band in enumerate(self.instrument) if \
            funcs.convert_wav_units(band.WavelengthLower50, u.AA).value < rest_UV_wav_lims.value[0] * (1. + self.z) \
            or funcs.convert_wav_units(band.WavelengthUpper50, u.AA).value > rest_UV_wav_lims.value[1] * (1. + self.z)]
        phot_rest_copy.crop_phot(crop_indices)
        phot_rest_copy.UV_wav_range = phot_rest_copy.rest_UV_wavs_name(rest_UV_wav_lims)
        return phot_rest_copy

    def is_correctly_UV_cropped(self, rest_UV_wav_lims):
        if hasattr(self, "UV_wav_range"):
            if self.UV_wav_range == self.rest_UV_wavs_name(rest_UV_wav_lims):
                assert u.get_physical_type(self.flux_Jy.unit) == "power density/spectral flux density wav"
                return True
        return False
    
    def PL_amplitude_name(self, rest_UV_wav_lims):
        return f"A_PL_{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
    
    # Rest-frame UV property calculations

    @ignore_warnings
    def calc_beta_phot(self, rest_UV_wav_lims, iters = 10, maxfev = 100_000, extract_property_name = False):
        assert iters >= 1, galfind_logger.critical(f"{iters=} < 1 in Photometry_rest.calc_beta_phot !!!")
        assert type(iters) == int, galfind_logger.critical(f"{type(iters)=} != 'int' in Photometry_rest.calc_beta_phot !!!")
        # iters = 1 -> fit without errors, iters >> 1 -> fit with errors
        property_name = f"beta_PL_{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
        if extract_property_name:
            return property_name
        # if already stored in object, do nothing
        if property_name in self.properties.keys() and property_name in self.property_errs.keys() \
                and property_name in self.property_PDFs.keys():
            galfind_logger.debug(f"{property_name=} already calculated!")
        else: # calculate beta in the relevant rest-frame UV range
            if self.is_correctly_UV_cropped(rest_UV_wav_lims):
                rest_UV_phot = self
            else:
                rest_UV_phot = self.get_rest_UV_phot(rest_UV_wav_lims)
            if iters == 1:
                f_lambda = funcs.convert_mag_units([funcs.convert_wav_units(band.WavelengthCen, u.AA).value \
                    for band in rest_UV_phot.instrument] * u.AA, rest_UV_phot.flux_Jy, u.erg / (u.s * u.AA * u.cm ** 2))
                return curve_fit(beta_fit(rest_UV_phot.z, rest_UV_phot.instrument.bands). \
                    beta_slope_power_law_func_conv_filt, None, f_lambda, maxfev = maxfev)[0]
            else:
                scattered_rest_UV_phot_arr = rest_UV_phot.scatter_phot(iters)
                popt_arr = np.array([scattered_rest_UV_phot.calc_beta_phot(rest_UV_wav_lims, iters = 1) \
                    for scattered_rest_UV_phot in tqdm(scattered_rest_UV_phot_arr, total = iters, desc = "Calculating beta_PL")])
                # save amplitude and beta PDFs
                self._update_properties_and_PDFs(self.PL_amplitude_name(rest_UV_wav_lims), (10 ** popt_arr[:, 0]) * u.erg / (u.s * u.AA * u.cm ** 2))
                self._update_properties_and_PDFs(property_name, popt_arr[:, 1] * u.dimensionless_unscaled)
                self.ampl_beta_joint_PDF = PDF_nD([self.property_PDFs[self.PL_amplitude_name(rest_UV_wav_lims)], self.property_PDFs[property_name]])
        return self, property_name
    
    def calc_fesc_from_beta_phot(self, rest_UV_wav_lims, conv_author_year, extract_property_name = False):
        assert conv_author_year in fesc_from_beta_conversions.keys()
        property_name = f"fesc_{conv_author_year}_{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
        if extract_property_name:
            return property_name
        # if already stored in object, do nothing
        if property_name in self.properties.keys() and property_name in self.property_errs.keys() \
                and property_name in self.property_PDFs.keys():
            galfind_logger.debug(f"{property_name=} already calculated!")
        else: # calculate fesc in the relevant rest-frame UV range
            beta_property_name = self.calc_beta_phot(rest_UV_wav_lims)[1]
            # update PDF
            self.property_PDFs[property_name] = self.property_PDFs[beta_property_name].manipulate_PDF(property_name, fesc_from_beta_conversions[conv_author_year])
            self._update_properties_from_PDF(property_name)
        return self, property_name
    
    def calc_AUV_from_beta_phot(self, rest_UV_wav_lims, ref_wav, conv_author_year, extract_property_name = False):
        conv_author_year_cls = globals()[conv_author_year]
        assert issubclass(conv_author_year_cls, AUV_from_beta)
        property_name = f"A{ref_wav.to(u.AA).value:.0f}_phot_{conv_author_year}_{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
        if extract_property_name:
            return property_name
        # if already stored in object, do nothing
        if property_name in self.properties.keys() and property_name in self.property_errs.keys() \
                and property_name in self.property_PDFs.keys():
            galfind_logger.debug(f"{property_name=} already calculated!")
        else: # calculate AUV in the relevant rest-frame UV range
            beta_property_name = self.calc_beta_phot(rest_UV_wav_lims)[1]
            # update PDF
            # if property_name == 'A1500_phot_Meurer99_[1250,3000]AA':
            #     breakpoint()
            self.property_PDFs[property_name] = self.property_PDFs[beta_property_name].manipulate_PDF(property_name, conv_author_year_cls())
            self._update_properties_from_PDF(property_name)
        return self, property_name
    
    def calc_mUV_phot(self, rest_UV_wav_lims, ref_wav, top_hat_width = 100. * u.AA, resolution = 1. * u.AA, extract_property_name = False):
        property_name = f"m{ref_wav.to(u.AA).value:.0f}_{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
        if extract_property_name:
            return property_name
        # if already stored in object, do nothing
        if property_name in self.properties.keys() and property_name in self.property_errs.keys() \
                and property_name in self.property_PDFs.keys():
            galfind_logger.debug(f"{property_name=} already calculated!")
        else: # calculate mUV in the relevant rest-frame UV range
            self.calc_beta_phot(rest_UV_wav_lims)
            rest_wavelengths = funcs.convert_wav_units(np.linspace(ref_wav - top_hat_width / 2, ref_wav + top_hat_width / 2, \
                int(np.round((top_hat_width / resolution).to(u.dimensionless_unscaled).value, 0))), u.AA)
            power_law_chains = self.ampl_beta_joint_PDF(funcs.power_law_beta_func, rest_wavelengths.value)
            # take the median of each chain to form a new chain
            mUV_chain = [np.median(funcs.convert_mag_units(rest_wavelengths * (1. + self.z), \
                chain * u.erg / (u.s * u.AA * u.cm ** 2), u.ABmag).value) for chain in power_law_chains] * u.ABmag
            self._update_properties_and_PDFs(property_name, mUV_chain)
        return self, property_name

    def calc_MUV_phot(self, rest_UV_wav_lims, ref_wav, extract_property_name = False):
        property_name = f"M{ref_wav.to(u.AA).value:.0f}_{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
        if extract_property_name:
            return property_name
        # if already stored in object, do nothing
        if property_name in self.properties.keys() and property_name in self.property_errs.keys() \
                and property_name in self.property_PDFs.keys():
            galfind_logger.debug(f"{property_name=} already calculated!")
        else: # calculate MUV in the relevant rest-frame UV range
            mUV_property_name = self.calc_mUV_phot(rest_UV_wav_lims, ref_wav)[1]
            self.property_PDFs[property_name] = self.property_PDFs[mUV_property_name] \
                .manipulate_PDF(property_name, lambda mUV:  mUV.unit * (mUV.value - \
                5. * np.log10(self.lum_distance.value / 10.) + 2.5 * np.log10(1. + self.z)))
            self._update_properties_from_PDF(property_name)
        return self, property_name
    
    def calc_LUV_obs_phot(self, rest_UV_wav_lims, ref_wav, extract_property_name = False):
        property_name = f"Lobs_{ref_wav.to(u.AA).value:.0f}_{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
        if extract_property_name:
            return property_name
        # if already stored in object, do nothing
        if property_name in self.properties.keys() and property_name in self.property_errs.keys() \
                and property_name in self.property_PDFs.keys():
            galfind_logger.debug(f"{property_name=} already calculated!")
        else: # calculate observed LUV in the relevant rest-frame UV range
            mUV_property_name = self.calc_mUV_phot(rest_UV_wav_lims, ref_wav)[1]
            self.property_PDFs[property_name] = self.property_PDFs[mUV_property_name] \
                .manipulate_PDF(property_name, funcs.flux_to_luminosity, \
                wavs = np.full(len(self.property_PDFs[mUV_property_name]), ref_wav), \
                z = self.z, out_units = u.erg / (u.s * u.Hz))
            self._update_properties_from_PDF(property_name)
        return self, property_name

    def calc_LUV_int_phot(self, rest_UV_wav_lims, ref_wav, AUV_beta_conv_author_year, extract_property_name = False):
        property_name = f"Lint_{ref_wav.to(u.AA).value:.0f}_{AUV_beta_conv_author_year}_{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
        if extract_property_name:
            return property_name
        # if already stored in object, do nothing
        if property_name in self.properties.keys() and property_name in self.property_errs.keys() \
                and property_name in self.property_PDFs.keys():
            galfind_logger.debug(f"{property_name=} already calculated!")
        else: # calculate intrinsic LUV in the relevant rest-frame UV range
            LUV_obs_property_name = self.calc_LUV_obs_phot(rest_UV_wav_lims, ref_wav)[1]
            AUV_property_name = self.calc_AUV_from_beta_phot(rest_UV_wav_lims, ref_wav, AUV_beta_conv_author_year)[1]
            self.property_PDFs[property_name] = self.property_PDFs[LUV_obs_property_name] \
                .manipulate_PDF(property_name, funcs.dust_correct, dust_mag = self.property_PDFs[AUV_property_name].input_arr)
            self._update_properties_from_PDF(property_name)
        return self, property_name

    def calc_SFR_UV_phot(self, rest_UV_wav_lims, ref_wav, AUV_beta_conv_author_year, kappa_UV_conv_author_year, extract_property_name = False):
        assert kappa_UV_conv_author_year in SFR_conversions.keys()
        property_name = f"SFR_{ref_wav.to(u.AA).value:.0f}_{AUV_beta_conv_author_year}_{kappa_UV_conv_author_year}_{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
        if extract_property_name:
            return property_name
        # if already stored in object, do nothing
        if property_name in self.properties.keys() and property_name in self.property_errs.keys() \
                and property_name in self.property_PDFs.keys():
            galfind_logger.debug(f"{property_name=} already calculated!")
        else: # calculate UV SFR in the relevant rest-frame UV range
            LUV_int_property_name = self.calc_LUV_int_phot(rest_UV_wav_lims, ref_wav, AUV_beta_conv_author_year)[1]
            self.property_PDFs[property_name] = self.property_PDFs[LUV_int_property_name] \
                .manipulate_PDF(property_name, lambda LUV_int: LUV_int * SFR_conversions[kappa_UV_conv_author_year])
            self._update_properties_from_PDF(property_name)
        return self, property_name

    # Function to update rest-frame UV properties and PDFs
    def _update_properties_and_PDFs(self, property_name, property_vals):
        # construct PDF from property_vals chain
        self.property_PDFs[property_name] = PDF.from_1D_arr(property_name, property_vals)
        self._update_properties_from_PDF(property_name)
    
    def _update_properties_from_PDF(self, property_name):
        self.properties[property_name] = self.property_PDFs[property_name].median
        self.property_errs[property_name] = self.property_PDFs[property_name].errs

    # def plot(self, save_dir, ID, plot_fit = True, iters = 10_000, save = True, show = False, n_interp = 100, conv_filt = False):
    #     self.make_rest_UV_phot()
    #     assert(conv_filt == False)
    #     #if not all(beta == -99. for beta in self.beta_PDF):
    #     sns.set(style="whitegrid")
    #     warnings.filterwarnings("ignore")

    #     # Create figure and axes
    #     fig, ax = plt.subplots()
        
    #     # Plotting code
    #     ax.errorbar(np.log10(self.rest_UV_phot.wav.value), self.rest_UV_phot.log_flux_lambda, yerr = self.rest_UV_phot.log_flux_lambda_errs,
    #                  ls = "none", c = "black", zorder = 10, marker = "o", markersize = 5, capsize = 3)
        
    #     if plot_fit:
    #         fit_lines = []
    #         fit_lines_interped = []
    #         wav_interp = np.linspace(np.log10(self.rest_UV_phot.wav.value)[0], np.log10(self.rest_UV_phot.wav.value)[-1], n_interp)
    #         if iters > len(self.amplitude_PDF[conv_filt]):
    #             iters = len(self.amplitude_PDF[conv_filt])
    #         for i in range(iters):
    #             #percentiles = np.percentile([16, 50, 84], axis=0)
    #             f_interp = interp1d(np.log10(self.rest_UV_phot.wav.value), np.log10(Photometry_rest.beta_slope_power_law_func(self.rest_UV_phot.wav.value, \
    #                 self.amplitude_PDF[conv_filt][i], self.beta_PDF[conv_filt][i])), kind = 'linear')
    #             y_new = f_interp(wav_interp)
    #             fit_lines.append(np.log10(Photometry_rest.beta_slope_power_law_func(self.rest_UV_phot.wav.value, self.amplitude_PDF[conv_filt][i], self.beta_PDF[conv_filt][i])))
    #             fit_lines_interped.append(y_new)
    #         fit_lines_interped = np.array(fit_lines_interped)
    #         fit_lines = np.array(fit_lines)
    #         fit_lines.reshape(iters, len(self.rest_UV_phot.wav.value))
    #         fit_lines_interped.reshape(iters, len(wav_interp))
            
    #         l1_chains = np.array([np.percentile(x, 16) for x in fit_lines_interped.T])
    #         med_chains = np.array([np.percentile(x, 50) for x in fit_lines.T])
    #         u1_chains = np.array([np.percentile(x, 84) for x in fit_lines_interped.T])
            
    #         ax.plot(np.log10(self.rest_UV_phot.wav.value), med_chains, color = "red", zorder = 2)
    #         ax.fill_between(wav_interp, l1_chains, u1_chains, color="grey", alpha=0.2, zorder=1)
        
    #     ax.set_xlabel(r"$\log_{10}(\lambda_{\mathrm{rest}} / \mathrm{\AA})$")
    #     ax.set_ylabel(r"$\log_{10}(\mathrm{f}_{\lambda_{\mathrm{rest}}} / \mathrm{erg} \, \mathrm{s}^{-1} \, \mathrm{cm}^{-2} \, \mathrm{\AA}^{-1})$")
        
    #     # Add the Galaxy ID label
    #     ax.text(0.05, 0.05, f"Galaxy ID = {str(ID)}", transform=ax.transAxes, ha="left", va="bottom", fontsize=12)
    #     # Add the Beta label
    #     ax.text(0.95, 0.95, r"$\beta$" + " = {:.2f} $^{{+{:.2f}}}_{{-{:.2f}}}$".format(np.percentile(self.beta_PDF[conv_filt], 50), \
    #         np.percentile(self.beta_PDF[conv_filt], 84) - np.percentile(self.beta_PDF[conv_filt], 50), np.percentile(self.beta_PDF[conv_filt], 50) - \
    #         np.percentile(self.beta_PDF[conv_filt], 16)), transform = ax.transAxes, ha = "right", va = "top", fontsize = 12)
            
    #     ax.set_xlim(*np.log10(self.rest_UV_wav_lims.value))
        
    #     if save:
    #         path = f"{save_dir}/plots/{ID}.png"
    #         funcs.make_dirs(path)
    #         fig.savefig(path, dpi=300, bbox_inches='tight')
        
    #     if show:
    #         plt.tight_layout()
    #         plt.show()
        
    #     plt.clf()
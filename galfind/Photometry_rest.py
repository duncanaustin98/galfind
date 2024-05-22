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
from .Dust_Attenuation import AUV_from_beta, M99, C00, Dust_Attenuation
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
    
    def __str__(self, print_PDFs = True):
        line_sep = "*" * 40 + "\n"
        band_sep = "-" * 10 + "\n"
        output_str = line_sep
        output_str += f"PHOTOMETRY_REST: z = {self.z}\n"
        output_str += band_sep
        # don't print the photometry here, only the derived properties
        if print_PDFs:
            for PDF_obj in self.property_PDFs.values():
                output_str += str(PDF_obj)
        output_str += line_sep
        return output_str

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
    def calc_beta_phot(self, rest_UV_wav_lims, iters = 10, maxfev = 100_000, beta_fit_func = None, extract_property_name = False, incl_errs = True):
        assert iters >= 1, galfind_logger.critical(f"{iters=} < 1 in Photometry_rest.calc_beta_phot !!!")
        assert type(iters) == int, galfind_logger.critical(f"{type(iters)=} != 'int' in Photometry_rest.calc_beta_phot !!!")
        # iters = 1 -> fit without errors, iters >> 1 -> fit with errors
        property_name = "beta_PL" #_{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
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
                assert type(beta_fit_func) != type(None)
                f_lambda = funcs.convert_mag_units([funcs.convert_wav_units(band.WavelengthCen, u.AA).value \
                    for band in rest_UV_phot.instrument] * u.AA, rest_UV_phot.flux_Jy, u.erg / (u.s * u.AA * u.cm ** 2))
                if not incl_errs:
                    return curve_fit(beta_fit_func, None, f_lambda, maxfev = maxfev)[0]
                else:
                    f_lambda_errs = funcs.convert_mag_err_units([funcs.convert_wav_units(band.WavelengthCen, u.AA).value \
                        for band in rest_UV_phot.instrument] * u.AA, rest_UV_phot.flux_Jy, [rest_UV_phot.flux_Jy_errs.value, \
                        rest_UV_phot.flux_Jy_errs.value] * rest_UV_phot.flux_Jy_errs.unit, u.erg / (u.s * u.AA * u.cm ** 2))
                    return curve_fit(beta_fit_func, None, f_lambda, sigma = f_lambda_errs[0], maxfev = maxfev)[0]
            else:
                assert type(beta_fit_func) == type(None)
                scattered_rest_UV_phot_arr = rest_UV_phot.scatter_phot(iters)
                beta_fit_func = beta_fit(rest_UV_phot.z, rest_UV_phot.instrument.bands).beta_slope_power_law_func_conv_filt
                popt_arr = np.array([scattered_rest_UV_phot.calc_beta_phot(rest_UV_wav_lims, iters = 1, beta_fit_func = beta_fit_func) \
                    for scattered_rest_UV_phot in tqdm(scattered_rest_UV_phot_arr, total = iters, desc = "Calculating beta_PL")])
                # save amplitude and beta PDFs
                self._update_properties_and_PDFs(self.PL_amplitude_name(rest_UV_wav_lims), (10 ** popt_arr[:, 0]) * u.erg / (u.s * u.AA * u.cm ** 2))
                self._update_properties_and_PDFs(property_name, popt_arr[:, 1] * u.dimensionless_unscaled)
                self.ampl_beta_joint_PDF = PDF_nD([self.property_PDFs[self.PL_amplitude_name(rest_UV_wav_lims)], self.property_PDFs[property_name]])
        return self, property_name
    
    def calc_fesc_from_beta_phot(self, rest_UV_wav_lims, conv_author_year, extract_property_name = False):
        assert conv_author_year in fesc_from_beta_conversions.keys()
        property_name = f"fesc_{conv_author_year}" #_{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
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
        property_name = f"A{ref_wav.to(u.AA).value:.0f}_{conv_author_year}" #_{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
        if extract_property_name:
            return property_name
        # if already stored in object, do nothing
        if property_name in self.properties.keys() and property_name in self.property_errs.keys() \
                and property_name in self.property_PDFs.keys():
            galfind_logger.debug(f"{property_name=} already calculated!")
        else: # calculate AUV in the relevant rest-frame UV range
            beta_property_name = self.calc_beta_phot(rest_UV_wav_lims)[1]
            # update PDF
            # if property_name == 'A1500_Meurer99_[1250,3000]AA':
            #     breakpoint()
            self.property_PDFs[property_name] = self.property_PDFs[beta_property_name].manipulate_PDF(property_name, conv_author_year_cls())
            self._update_properties_from_PDF(property_name)
        return self, property_name
    
    def calc_mUV_phot(self, rest_UV_wav_lims, ref_wav, top_hat_width = 100. * u.AA, resolution = 1. * u.AA, extract_property_name = False):
        property_name = f"m{ref_wav.to(u.AA).value:.0f}" #_{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
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
        property_name = f"M{ref_wav.to(u.AA).value:.0f}" #_{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
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
        property_name = f"Lobs_{ref_wav.to(u.AA).value:.0f}" #_{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
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
                z = None, out_units = u.erg / (u.s * u.Hz))
            self._update_properties_from_PDF(property_name)
        return self, property_name

    def calc_LUV_int_phot(self, rest_UV_wav_lims, ref_wav, AUV_beta_conv_author_year, extract_property_name = False):
        property_name = f"Lint_{ref_wav.to(u.AA).value:.0f}_{AUV_beta_conv_author_year}"
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
        property_name = f"SFR_{ref_wav.to(u.AA).value:.0f}_{AUV_beta_conv_author_year}_{kappa_UV_conv_author_year}" #_{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
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
    
    def calc_cont_rest_optical(self, line_names, rest_optical_wavs = [3_700., 7_000.] * u.AA, iters = 10, extract_property_name = False):
        assert all(line_name in line_diagnostics.keys() for line_name in line_names)
        included_bands = {}
        closest_band = {}
        for line_name in line_names:
            rest_frame_emission_line_wav = line_diagnostics[line_name]["line_wav"]
            assert rest_frame_emission_line_wav > rest_optical_wavs[0] and rest_frame_emission_line_wav < rest_optical_wavs[1]
            obs_frame_emission_line_wav = rest_frame_emission_line_wav * (1. + self.z)
            # determine the closest (medium) band to the emission line
            closest_band[line_name] = self.instrument.nearest_band_to_wavelength(obs_frame_emission_line_wav, medium_bands_only = False)
        property_name = f"continuum_{'+'.join(line_names)}"
        if extract_property_name:
            return property_name
        # if already stored in object, do nothing
        if property_name in self.properties.keys() and property_name in self.property_errs.keys() and property_name in self.property_PDFs.keys():
            galfind_logger.debug(f"{property_name=} already calculated!")
        else:
            # determine photometric band that traces the continuum - must avoid other strong lines
            bands_avoiding_wavs = self.instrument.bands_avoiding_wavs([line["line_wav"] \
                * (1. + self.z) for line in line_diagnostics.values()]) # \
                #if line["line_wav"] > rest_optical_wavs[0] and line["line_wav"] < rest_optical_wavs[1])
            if len(bands_avoiding_wavs) < 1 or not all(band == closest_band[line_names[0]] for band in closest_band.values()):
                self._update_properties_and_PDFs(property_name, None)
                return self, property_name
            # find closest band to the band of interest
            continuum_band = bands_avoiding_wavs[np.abs([band.WavelengthCen.to(u.AA).value for band in bands_avoiding_wavs] \
                - (line_diagnostics[line_names[0]]["line_wav"] * (1. + self.z)).value).argmin()]
            galfind_logger.debug(f"Continuum flux calculation assumes that the most prominent line in {line_names=} is {line_names[0]}")
            #print(self.z, continuum_band.band_name)
            continuum_band_index = self.instrument.index_from_band_name(continuum_band.band_name)
            # determine continuum flux of this band in Jy
            cont_flux_Jy = funcs.convert_mag_units(self.instrument[continuum_band_index].WavelengthCen, self.flux_Jy[continuum_band_index], u.nJy).value
            # errors in Jy are symmetric, therefore take the mean
            cont_flux_Jy_errs = np.mean([flux_err.value for flux_err in funcs.convert_mag_err_units(self.instrument[continuum_band_index].WavelengthCen, \
                self.flux_Jy[continuum_band_index], [self.flux_Jy_errs[continuum_band_index], self.flux_Jy_errs[continuum_band_index]], u.nJy)])
            cont_flux_Jy_chain = np.random.normal(cont_flux_Jy, cont_flux_Jy_errs, iters) * u.nJy
            PDF_kwargs = {"continuum_band": continuum_band.band_name}
            self._update_properties_and_PDFs(property_name, cont_flux_Jy_chain, PDF_kwargs)
        return self, property_name
    
    def calc_EW_rest_optical(self, line_names, medium_bands_only = True, rest_optical_wavs = [3_700., 7_000.] * u.AA, iters = 10, extract_property_name = False):
        assert all(line_name in line_diagnostics.keys() for line_name in line_names)
        included_bands = {}
        closest_band = {}
        for line_name in line_names:
            rest_frame_emission_line_wav = line_diagnostics[line_name]["line_wav"]
            assert rest_frame_emission_line_wav > rest_optical_wavs[0] and rest_frame_emission_line_wav < rest_optical_wavs[1]
            obs_frame_emission_line_wav = rest_frame_emission_line_wav * (1. + self.z)
            # determine index of the closest (medium) band to the emission line
            closest_band[line_name] = self.instrument.nearest_band_to_wavelength(obs_frame_emission_line_wav, medium_bands_only = medium_bands_only)
        property_name = f"EW_rest_{'+'.join(line_names)}"
        if extract_property_name:
            return property_name
        # if already stored in object, do nothing
        if property_name in self.properties.keys() and property_name in self.property_errs.keys() and property_name in self.property_PDFs.keys():
            galfind_logger.debug(f"{property_name=} already calculated!")
        else:
            # calculate only if these lines lie within the same band
            if any(band != closest_band[line_names[0]] for band in closest_band.values()) or any(type(band) == type(None) for band in closest_band.values()):
                self._update_properties_and_PDFs(property_name, None)
                return self, property_name
            # calculate continuum flux PDF in Jy
            cont_property_name = self.calc_cont_rest_optical(line_names, iters = iters, rest_optical_wavs = rest_optical_wavs)[1]
            if type(self.property_PDFs[cont_property_name]) == type(None):
                self._update_properties_and_PDFs(property_name, None)
                return self, property_name
            line_band_index = self.instrument.index_from_band_name(closest_band[line_names[0]].band_name)
            line_flux_Jy = funcs.convert_mag_units(self.instrument[line_band_index].WavelengthCen, self.flux_Jy[line_band_index], u.Jy).value
            # errors in Jy are symmetric, therefore take the mean
            line_flux_Jy_errs = np.mean([flux_err.value for flux_err in funcs.convert_mag_err_units(self.instrument[line_band_index].WavelengthCen, \
                self.flux_Jy[line_band_index], [self.flux_Jy_errs[line_band_index], self.flux_Jy_errs[line_band_index]], u.Jy)])
            line_flux_Jy_chain = np.random.normal(line_flux_Jy, line_flux_Jy_errs, iters) * u.Jy
            PDF_kwargs = {**self.property_PDFs[cont_property_name].kwargs, **{"emission_band": self.instrument[line_band_index].band_name}}
            bandwidth = self.instrument[line_band_index].WavelengthUpper50 - self.instrument[line_band_index].WavelengthLower50
            self.property_PDFs[property_name] = self.property_PDFs[cont_property_name].manipulate_PDF(property_name, lambda cont_flux_Jy: \
                (((line_flux_Jy_chain / cont_flux_Jy).to(u.dimensionless_unscaled) - 1.) * bandwidth / (1. + self.z)).to(u.AA), PDF_kwargs)
            self._update_properties_from_PDF(property_name)
        return self, property_name

    def calc_obs_line_flux_rest_optical(self, line_names, medium_bands_only = True, rest_optical_wavs = [3_700., 7_000.] * u.AA, iters = 10, extract_property_name = False):
        assert all(line_name in line_diagnostics.keys() for line_name in line_names)
        property_name = f"{'+'.join(line_names)}_flux_rest"
        if extract_property_name:
            return property_name
        # if already stored in object, do nothing
        if property_name in self.properties.keys() and property_name in self.property_errs.keys() and property_name in self.property_PDFs.keys():
            galfind_logger.debug(f"{property_name=} already calculated!")
        else:
            EW_property_name = self.calc_EW_rest_optical(line_names, iters = iters, rest_optical_wavs = rest_optical_wavs)[1]
            cont_property_name = self.calc_cont_rest_optical(line_names, iters = iters, rest_optical_wavs = rest_optical_wavs)[1]
            if any(type(PDF_obj) == type(None) for PDF_obj in [self.property_PDFs[EW_property_name], self.property_PDFs[cont_property_name]]):
                self._update_properties_and_PDFs(property_name, None)
                return self, property_name
            else:
                # ensure PDFs have the same length
                assert len(self.property_PDFs[EW_property_name]) == len(self.property_PDFs[cont_property_name])
                # manipulate continuum and EW PDFs
                PDF_kwargs = {**self.property_PDFs[EW_property_name].kwargs, **self.property_PDFs[cont_property_name].kwargs}
                #breakpoint()
                self.property_PDFs[property_name] = self.property_PDFs[EW_property_name].manipulate_PDF(property_name, \
                    lambda EW_rest: EW_rest.to(u.AA) * funcs.convert_mag_units( \
                    self.instrument[self.instrument.index_from_band_name(PDF_kwargs["emission_band"])].WavelengthCen / (1. + self.z), \
                    self.property_PDFs[cont_property_name].input_arr, u.erg / (u.s * u.AA * u.cm ** 2)), PDF_kwargs)
                self._update_properties_from_PDF(property_name)
        return self, property_name

    def calc_int_line_flux_rest_optical(self, line_names, dust_author_year = "M99", dust_law = "C00", \
            dust_origin = "UV", medium_bands_only = True, rest_optical_wavs = [3_700., 7_000.] * u.AA, \
            rest_UV_wav_lims = [1_250., 3_000.] * u.AA, ref_wav = 1_500. * u.AA, iters = 10, extract_property_name = False):
        assert all(line_name in line_diagnostics.keys() for line_name in line_names)
        dust_law_cls = globals()[dust_law] # un-initialized
        assert issubclass(dust_law_cls, Dust_Attenuation)
        if dust_origin == "UV":
            property_name = f"{'+'.join(line_names)}_flux_rest_{dust_author_year}_{dust_law}"
        else:
            raise NotImplementedError
        if extract_property_name:
            return property_name
        # if already stored in object, do nothing
        if property_name in self.properties.keys() and property_name in self.property_errs.keys() and property_name in self.property_PDFs.keys():
            galfind_logger.debug(f"{property_name=} already calculated!")
        else:
            obs_line_flux_property_name = self.calc_obs_line_flux_rest_optical(line_names, medium_bands_only, iters = iters, rest_optical_wavs = rest_optical_wavs)[1]
            if type(self.property_PDFs[obs_line_flux_property_name]) == type(None):
                self._update_properties_and_PDFs(property_name, None)
                return self, property_name
            if dust_origin == "UV":
                A_UV_property_name = self.calc_AUV_from_beta_phot(rest_UV_wav_lims, ref_wav, dust_author_year)[1]
                if type(self.property_PDFs[A_UV_property_name]) == type(None):
                    self._update_properties_and_PDFs(property_name, None)
                    return self, property_name
                # initialize dust_law_cls
                dust_law_cls = dust_law_cls()
                # assume first line is the strongest and most important
                A_line = self.property_PDFs[A_UV_property_name].manipulate_PDF("A_line", \
                    lambda A_UV: (A_UV.to(u.ABmag).value * dust_law_cls.k_lambda(line_diagnostics[line_names[0]]["line_wav"]) \
                    / dust_law_cls.k_lambda(ref_wav)) * u.ABmag)
                #print(A_line.input_arr)
                breakpoint()
            else:
                raise NotImplementedError
            PDF_kwargs = self.property_PDFs[obs_line_flux_property_name].kwargs
            self.property_PDFs[property_name] = self.property_PDFs[obs_line_flux_property_name]. \
                manipulate_PDF(property_name, funcs.dust_correct, dust_mag = A_line.input_arr, PDF_kwargs = PDF_kwargs)
            self._update_properties_from_PDF(property_name)
        return self, property_name
    
    def calc_int_line_lum_rest_optical(self, line_names, flux_contamination_params: dict = {"mu": 0., "sigma": 0.}, dust_author_year: str = "M99", \
            dust_law: str = "C00", dust_origin: str = "UV", medium_bands_only = True, rest_optical_wavs = [3_700., 7_000.] * u.AA, \
            rest_UV_wav_lims = [1_250., 3_000.] * u.AA, ref_wav = 1_500. * u.AA, iters = 10, extract_property_name = False):
        # assume first line in list is the line of interest!
        if dust_origin == "UV":
            property_name = f"{line_names[0]}_lum_rest_{dust_author_year}_{dust_law}"
        else:
            raise NotImplementedError
        # if already stored in object, do nothing
        if property_name in self.properties.keys() and property_name in self.property_errs.keys() and property_name in self.property_PDFs.keys():
            galfind_logger.debug(f"{property_name=} already calculated!")
        else:
            int_line_flux_property_name = self.calc_int_line_flux_rest_optical(line_names, dust_author_year, \
                dust_law, dust_origin, medium_bands_only, rest_optical_wavs, rest_UV_wav_lims, ref_wav, iters)[1]
            if type(self.property_PDFs[int_line_flux_property_name]) == type(None):
                self._update_properties_and_PDFs(property_name, None)
                return self, property_name
            lum_distance_z0 = funcs.calc_lum_distance(z = 0)
            flux_cont_keys = flux_contamination_params.keys()
            if "mu" in flux_cont_keys and "sigma" in flux_cont_keys:
                line_flux_scaling = np.random.normal(1. - flux_contamination_params["mu"], flux_contamination_params["sigma"], iters)
            elif "mu" in flux_cont_keys and "sigma" not in flux_cont_keys:
                line_flux_scaling = flux_contamination_params["mu"]
            else:
                raise NotImplementedError
            PDF_kwargs = {**self.property_PDFs[int_line_flux_property_name].kwargs, **{"cont_line_names": line_names[1:]}, \
                **{f"flux_contamination_params_{key}": value for key, value in flux_contamination_params.items()}}
            self.property_PDFs[property_name] = self.property_PDFs[int_line_flux_property_name].manipulate_PDF(property_name, \
                lambda int_line_flux: (4 * np.pi * int_line_flux * line_flux_scaling * lum_distance_z0 ** 2).to(u.erg / u.s), PDF_kwargs)
            self._update_properties_from_PDF(property_name)
        return self, property_name

    def calc_xi_ion(self, line_names = ["Halpha", "[NII]-6583"], flux_contamination_params: dict = {"mu": 0.1}, fesc_author_year: str = "fesc=0.0", \
            dust_author_year: str = "M99", dust_law: str = "C00", dust_origin: str = "UV", medium_bands_only: bool = True, \
            rest_optical_wavs: u.Quantity = [3_700., 7_000.] * u.AA, rest_UV_wav_lims: u.Quantity = [1_250., 3_000.] * u.AA, \
            ref_wav: u.Quantity = 1_500. * u.AA, iters: int = 10, extract_property_name: bool = False):
        assert "Halpha" in line_names
        if dust_origin == "UV":
            property_name = f"xi_ion_{dust_author_year}_{dust_law}_{fesc_author_year.replace('=', '')}"
        else:
            raise NotImplementedError
        if extract_property_name:
            return property_name
        # if already stored in object, do nothing
        if property_name in self.properties.keys() and property_name in self.property_errs.keys() and property_name in self.property_PDFs.keys():
            galfind_logger.debug(f"{property_name=} already calculated!")
        else:
            #breakpoint()
            if fesc_author_year in fesc_from_beta_conversions.keys():
                fesc_property_name = self.calc_fesc_from_beta_phot(rest_UV_wav_lims, fesc_author_year)[1]
                fesc_chain = self.property_PDFs[fesc_property_name]
                if type(fesc_chain) == type(None):
                    self._update_properties_and_PDFs(property_name, None)
                    return self, property_name
                else:
                    fesc_chain = fesc_chain.input_arr
            elif "fesc=" in fesc_author_year:
                fesc_chain = np.full(iters, float(fesc_author_year.split("=")[-1]))
            else:
                raise NotImplementedError
            int_line_lum_property_name = self.calc_int_line_lum_rest_optical(line_names, flux_contamination_params, dust_author_year, \
                dust_law, dust_origin, medium_bands_only, rest_optical_wavs, rest_UV_wav_lims, ref_wav, iters = iters)[1]
            L_UV_int_property_name = self.calc_LUV_int_phot(rest_UV_wav_lims = rest_UV_wav_lims, ref_wav = ref_wav, AUV_beta_conv_author_year = dust_author_year)[1]
            if any(type(self.property_PDFs[_property_name]) == type(None) for _property_name in [int_line_lum_property_name, L_UV_int_property_name]):
                self._update_properties_and_PDFs(property_name, None)
                return self, property_name
            assert self.property_PDFs[int_line_lum_property_name].kwargs["cont_line_names"] == line_names[1:]
            # calculate xi_ion from Halpha luminosity
            #breakpoint()
            PDF_kwargs = {**self.property_PDFs[int_line_lum_property_name].kwargs, **self.property_PDFs[L_UV_int_property_name].kwargs}
            self.property_PDFs[property_name] = self.property_PDFs[int_line_lum_property_name].manipulate_PDF(property_name, \
                lambda int_line_lum: (int_line_lum / (1.36e-12 * u.erg * (1. - fesc_chain) * \
                self.property_PDFs[L_UV_int_property_name].input_arr)).to(u.Hz / u.erg), PDF_kwargs)
            self._update_properties_from_PDF(property_name)
        return self, property_name
        # assumes some case for ISM recombination

    # Function to update rest-frame UV properties and PDFs
    def _update_properties_and_PDFs(self, property_name, property_vals, PDF_kwargs = {}):
        if type(property_vals) == type(None):
            self.property_PDFs[property_name] = None
        else:
            # construct PDF from property_vals chain
            self.property_PDFs[property_name] = PDF.from_1D_arr(property_name, property_vals, PDF_kwargs)
        self._update_properties_from_PDF(property_name)
    
    def _update_properties_from_PDF(self, property_name):
        if type(self.property_PDFs[property_name]) == type(None):
            self.properties[property_name] = np.nan
            self.property_errs[property_name] = [np.nan, np.nan]
        else:
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
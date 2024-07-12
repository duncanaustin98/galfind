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
from typing import Union

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
    
    # @property
    # def lum_distance(self):
    #     try:
    #         return self._lum_distance
    #     except AttributeError:
    #         self._lum_distance = astropy_cosmo.luminosity_distance(self.z).to(u.pc)
    #         return self._lum_distance
    
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
        assert iters >= 0, galfind_logger.critical(f"{iters=} < 1 in Photometry_rest.calc_beta_phot !!!")
        assert type(iters) == int, galfind_logger.critical(f"{type(iters)=} != 'int' in Photometry_rest.calc_beta_phot !!!")
        # iters = 1 -> fit without errors, iters >> 1 -> fit with errors
        property_name = f"beta_PL_{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
        PL_amplitude_name = self.PL_amplitude_name(rest_UV_wav_lims)
        if extract_property_name:
            return [PL_amplitude_name, property_name]
        #breakpoint()
        property_stored = property_name in self.properties.keys() and property_name in self.property_errs.keys() \
                and property_name in self.property_PDFs.keys() and PL_amplitude_name in self.properties.keys() \
                and PL_amplitude_name in self.property_errs.keys() and PL_amplitude_name in self.property_PDFs.keys()
        
        if iters > 0: # calculate beta in the relevant rest-frame UV range
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
                if len(rest_UV_phot) == 0:
                    A_arr = None
                    beta_arr = None
                else:
                    assert type(beta_fit_func) == type(None)
                    scattered_rest_UV_phot_arr = rest_UV_phot.scatter_phot(iters)
                    beta_fit_func = beta_fit(rest_UV_phot.z, rest_UV_phot.instrument.bands).beta_slope_power_law_func_conv_filt
                    popt_arr = np.array([scattered_rest_UV_phot.calc_beta_phot(rest_UV_wav_lims, iters = 1, beta_fit_func = beta_fit_func) \
                        for scattered_rest_UV_phot in tqdm(scattered_rest_UV_phot_arr, total = iters, desc = "Calculating beta_PL")])
                    A_arr = (10 ** popt_arr[:, 0]) * u.erg / (u.s * u.AA * u.cm ** 2)
                    beta_arr = popt_arr[:, 1] * u.dimensionless_unscaled
                update_PDFs = True
        else: # if already stored in object, do nothing
            A_arr = None
            beta_arr = None
            galfind_logger.debug(f"{property_name=} and associated {PL_amplitude_name=} already calculated!")
            if not property_stored:
                update_PDFs = True
            else:
                update_PDFs = False
        
        # save amplitude and beta PDFs
        if update_PDFs:
            PDF_kwargs = {"rest_UV_band_names": "+".join(rest_UV_phot.instrument.band_names), "n_UV_bands": len(rest_UV_phot.instrument)}
            self._update_properties_and_PDFs(PL_amplitude_name, A_arr, PDF_kwargs)
            self._update_properties_and_PDFs(property_name, beta_arr, PDF_kwargs)
        if not hasattr(self, "ampl_beta_joint_PDF") or (hasattr(self, "ampl_beta_joint_PDF") and update_PDFs):
            if type(A_arr) == None and type(beta_arr == None):
                self.ampl_beta_joint_PDF = None
            else:
                self.ampl_beta_joint_PDF = PDF_nD([self.property_PDFs[self.PL_amplitude_name(rest_UV_wav_lims)], self.property_PDFs[property_name]])
        
        return self, [PL_amplitude_name, property_name]
    
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
            if type(self.property_PDFs[beta_property_name]) == type(None):
                self.property_PDFs[property_name] = None
            else:
                self.property_PDFs[property_name] = self.property_PDFs[beta_property_name].manipulate_PDF(property_name, fesc_from_beta_conversions[conv_author_year])
            self._update_properties_from_PDF(property_name)
        return self, property_name
    
    def calc_AUV_from_beta_phot(self, rest_UV_wav_lims, ref_wav, conv_author_year, extract_property_name = False):
        conv_author_year_cls = globals()[conv_author_year]
        assert issubclass(conv_author_year_cls, AUV_from_beta)
        UV_dust_label = self._get_UV_dust_label(conv_author_year)
        property_name = f"A{ref_wav.to(u.AA).value:.0f}{UV_dust_label}" #_{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
        if extract_property_name:
            return property_name
        # if already stored in object, do nothing
        if property_name in self.properties.keys() and property_name in self.property_errs.keys() \
                and property_name in self.property_PDFs.keys():
            galfind_logger.debug(f"{property_name=} already calculated!")
        else: # calculate AUV in the relevant rest-frame UV range
            beta_property_name = self.calc_beta_phot(rest_UV_wav_lims)[1]
            # update PDF
            if type(self.property_PDFs[beta_property_name]) == type(None):
                self.property_PDFs[property_name] = None
            else:
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
            beta_name = self.calc_beta_phot(rest_UV_wav_lims)[1]
            rest_wavelengths = funcs.convert_wav_units(np.linspace(ref_wav - top_hat_width / 2, ref_wav + top_hat_width / 2, \
                int(np.round((top_hat_width / resolution).to(u.dimensionless_unscaled).value, 0))), u.AA)
            power_law_chains = self.ampl_beta_joint_PDF(funcs.power_law_beta_func, rest_wavelengths.value)
            # take the median of each chain to form a new chain
            mUV_chain = [np.median(funcs.convert_mag_units(rest_wavelengths * (1. + self.z), \
                chain * u.erg / (u.s * u.AA * u.cm ** 2), u.ABmag).value) for chain in power_law_chains] * u.ABmag
            PDF_kwargs = self.property_PDFs[beta_name].kwargs
            self._update_properties_and_PDFs(property_name, mUV_chain, PDF_kwargs)
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
                5. * np.log10(funcs.calc_lum_distance(self.z).to(u.pc).value / 10.) + 2.5 * np.log10(1. + self.z)))
            self._update_properties_from_PDF(property_name)
        return self, property_name
    
    def calc_LUV_phot(self, frame: str, rest_UV_wav_lims: u.Quantity = [1_250., 3_000.] * u.AA, \
            ref_wav: u.Quantity = 1_500. * u.AA, dust_author_year: Union[str, None] = "M99", extract_property_name: bool = False):
        assert(frame in ["rest", "obs"])
        UV_dust_label = self._get_UV_dust_label(dust_author_year)
        property_name = f"L{frame}_{ref_wav.to(u.AA).value:.0f}{UV_dust_label}" #_{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
        if extract_property_name:
            return property_name
        # if already stored in object, do nothing
        if property_name in self.properties.keys() and property_name in self.property_errs.keys() \
                and property_name in self.property_PDFs.keys():
            galfind_logger.debug(f"{property_name=} already calculated!")
        else: # calculate observed LUV in the relevant rest-frame UV range
            AUV_property_name = self.calc_AUV_from_beta_phot(rest_UV_wav_lims, ref_wav, dust_author_year)[1]
            mUV_property_name = self.calc_mUV_phot(rest_UV_wav_lims, ref_wav)[1]
            if frame == "rest":
                z = 0.
                wavs = np.full(len(self.property_PDFs[mUV_property_name]), ref_wav)
            else: # frame == "obs"
                z = self.z
                wavs = np.full(len(self.property_PDFs[mUV_property_name]), ref_wav * (1. + self.z))
            UV_lum = self.property_PDFs[mUV_property_name].manipulate_PDF(property_name, \
                funcs.flux_to_luminosity, wavs = wavs, z = z, out_units = u.erg / (u.s * u.Hz))
            if UV_dust_label == "":
                self.property_PDFs[property_name] == UV_lum
            else: # dust correct
                self.property_PDFs[property_name] = UV_lum.manipulate_PDF(property_name, \
                    funcs.dust_correct, dust_mag = self.property_PDFs[AUV_property_name].input_arr)
            self._update_properties_from_PDF(property_name)
        return self, property_name

    def calc_SFR_UV_phot(self, frame, rest_UV_wav_lims, ref_wav, dust_author_year, kappa_UV_conv_author_year, extract_property_name = False):
        assert kappa_UV_conv_author_year in SFR_conversions.keys()
        UV_dust_label = self._get_UV_dust_label(dust_author_year)
        property_name = f"SFR{frame}_{ref_wav.to(u.AA).value:.0f}{UV_dust_label}_{kappa_UV_conv_author_year}" #_{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
        if extract_property_name:
            return property_name
        # if already stored in object, do nothing
        if property_name in self.properties.keys() and property_name in self.property_errs.keys() \
                and property_name in self.property_PDFs.keys():
            galfind_logger.debug(f"{property_name=} already calculated!")
        else: # calculate UV SFR in the relevant rest-frame UV range
            LUV_property_name = self.calc_LUV_phot(frame, rest_UV_wav_lims, ref_wav, dust_author_year)[1]
            self.property_PDFs[property_name] = self.property_PDFs[LUV_property_name] \
                .manipulate_PDF(property_name, lambda LUV: LUV * SFR_conversions[kappa_UV_conv_author_year])
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
            cont_chain = np.random.normal(cont_flux_Jy, cont_flux_Jy_errs, iters) * u.nJy # funcs.convert_mag_units(self.instrument[continuum_band_index].WavelengthCen, , out_units)
            PDF_kwargs = {f"{'+'.join(line_names)}_cont_band": continuum_band.band_name}
            self._update_properties_and_PDFs(property_name, cont_chain, PDF_kwargs)
        return self, property_name
    
    def calc_EW_rest_optical(self, line_names: list, frame: str = "rest", flux_contamination_params: dict = {"mu": 0., "sigma": 0.}, \
            medium_bands_only: bool = True, rest_optical_wavs: u.Quantity = [3_700., 7_000.] * u.AA, iters: int = 10, out_units: u.Unit = u.AA, extract_property_name: bool = False):
        assert(frame in ["rest", "obs"])
        assert all(line_name in line_diagnostics.keys() for line_name in line_names)
        assert(u.get_physical_type(out_units) == "length")
        included_bands = {}
        closest_band = {}
        for line_name in line_names:
            rest_frame_emission_line_wav = line_diagnostics[line_name]["line_wav"]
            assert rest_frame_emission_line_wav > rest_optical_wavs[0] and rest_frame_emission_line_wav < rest_optical_wavs[1]
            obs_frame_emission_line_wav = rest_frame_emission_line_wav * (1. + self.z)
            # determine index of the closest (medium) band to the emission line
            closest_band[line_name] = self.instrument.nearest_band_to_wavelength(obs_frame_emission_line_wav, medium_bands_only = medium_bands_only)
        # determine flux_contamination_name
        flux_contam_name = self._get_rest_optical_flux_contam_label(line_names, flux_contamination_params)
        property_name = f"EW_{frame}_{flux_contam_name}"
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
            PDF_kwargs = {f"{'+'.join(line_names)}_emission_band": self.instrument[line_band_index].band_name}
            bandwidth = self.instrument[line_band_index].WavelengthUpper50 - self.instrument[line_band_index].WavelengthLower50
            flux_contam_scaling = self._get_rest_optical_flux_contam_scaling(flux_contamination_params, iters)
            if frame == "rest":
                calc_EW_func = lambda cont_flux_Jy: (((line_flux_Jy_chain / cont_flux_Jy).to(u.dimensionless_unscaled) - 1.) * bandwidth / (1. + self.z)).to(out_units) * flux_contam_scaling
            else: # frame == "obs"
                calc_EW_func = lambda cont_flux_Jy: (((line_flux_Jy_chain / cont_flux_Jy).to(u.dimensionless_unscaled) - 1.) * bandwidth).to(out_units) * flux_contam_scaling
            self.property_PDFs[property_name] = self.property_PDFs[cont_property_name].manipulate_PDF(property_name, calc_EW_func, PDF_kwargs)
            self._update_properties_from_PDF(property_name)
        return self, property_name

    def calc_dust_atten(self, calc_wav: u.Quantity, dust_author_year: str = "M99", dust_law: str = "C00", \
            dust_origin: str = "UV", rest_UV_wav_lims: u.Quantity = [1_250., 3_000.] * u.AA, \
            ref_wav: u.Quantity = 1_500. * u.AA, iters: int = 10, extract_property_name: bool = False):
        if any(type(name) == type(None) for name in [dust_author_year, dust_law, dust_origin]):
            return self, None
        if dust_origin != "UV":
            raise NotImplementedError
        if type(dust_law) != type(None):
            dust_law_cls = globals()[dust_law] # un-initialized
            assert issubclass(dust_law_cls, Dust_Attenuation)
        property_name = f"A{calc_wav.to(u.AA).value:.0f}{self._get_dust_corr_label(dust_author_year, dust_law, dust_origin)}"
        if extract_property_name:
            return property_name
        if property_name in self.properties.keys() and property_name in self.property_errs.keys() and property_name in self.property_PDFs.keys():
            galfind_logger.debug(f"{property_name=} already calculated!")
        else:
            A_ref_wav_name = self.calc_AUV_from_beta_phot(rest_UV_wav_lims, ref_wav, dust_author_year)[1]
            if type(self.property_PDFs[A_ref_wav_name]) == type(None):
                self._update_properties_and_PDFs(property_name, None)
                return self, property_name
            dust_law_cls = dust_law_cls() # initialize dust_law_cls
            self.property_PDFs[property_name] = self.property_PDFs[A_ref_wav_name].manipulate_PDF(property_name, \
                lambda A_ref_wav: (A_ref_wav.to(u.ABmag).value * dust_law_cls.k_lambda(calc_wav.to(u.AA)) / dust_law_cls.k_lambda(ref_wav)) * u.ABmag)
            self._update_properties_from_PDF(property_name)
        return self, property_name

    # def calc_dust_atten_line(self, line_name: str, dust_author_year: str = "M99", dust_law: str = "C00", \
    #         dust_origin: str = "UV", iters: int = 10, ref_wav: u.Quantity = 1_500. * u.AA, extract_property_name: bool = False):
    #     assert(line_name in line_diagnostics.keys())
    #     #return self.calc_dust_atten(line_diagnostics[line_name]["line_wav"].to(u.AA), dust_author_year, dust_law, )
    #     # include option to include additional dust attenuation from birth clouds here!

    def calc_line_flux_rest_optical(self, line_names: list, frame: str = "obs", flux_contamination_params: dict = {"mu": 0.}, dust_author_year: str = "M99", dust_law: str = "C00", \
            dust_origin: str = "UV", medium_bands_only: bool = True, rest_optical_wavs: u.Quantity = [3_700., 7_000.] * u.AA, rest_UV_wav_lims: u.Quantity = [1_250., 3_000.] * u.AA, \
            ref_wav: u.Quantity = 1_500. * u.AA, iters: int = 10, out_units: u.Unit = u.erg / (u.s * u.AA * u.cm ** 2), extract_property_name: bool = False):
        assert all(line_name in line_diagnostics.keys() for line_name in line_names)
        dust_label = self._get_dust_corr_label(dust_author_year = dust_author_year, dust_law = dust_law, dust_origin = dust_origin)
        flux_contam_label = self._get_rest_optical_flux_contam_label(line_names, flux_contamination_params)
        property_name = f"flux_{flux_contam_label}_{frame}{dust_label}"
        if extract_property_name:
            return property_name
        # if already stored in object, do nothing
        if property_name in self.properties.keys() and property_name in self.property_errs.keys() and property_name in self.property_PDFs.keys():
            galfind_logger.debug(f"{property_name=} already calculated!")
        else:
            cont_name = self.calc_cont_rest_optical(line_names, rest_optical_wavs, iters)[1]
            EW_name = self.calc_EW_rest_optical(line_names, frame, flux_contamination_params, medium_bands_only, rest_optical_wavs, iters)[1]
            A_line_name = self.calc_dust_atten(line_diagnostics[line_names[0]]["line_wav"], dust_author_year, dust_law, dust_origin, rest_UV_wav_lims, ref_wav, iters)[1]
            if any(type(self.property_PDFs[name]) == type(None) for name in [cont_name, EW_name]):
                self._update_properties_and_PDFs(property_name, None)
                return self, property_name
            if A_line_name in self.property_PDFs.keys():
                if type(self.property_PDFs[A_line_name]) == type(None):
                    self._update_properties_and_PDFs(property_name, None)
                    return self, property_name
            # convert EW to line flux in appropriate frame
            if dust_label == "":
                PDF_kwargs = {**self.property_PDFs[cont_name].kwargs, **self.property_PDFs[A_line_name].kwargs}
            else:
                PDF_kwargs = self.property_PDFs[cont_name].kwargs
            band_wav = self.instrument[self.instrument.index_from_band_name(self.property_PDFs[EW_name].kwargs[f"{'+'.join(line_names)}_emission_band"])].WavelengthCen
            if frame == "rest":
                band_wav /= (1. + self.z)
            line_flux_PDF = self.property_PDFs[EW_name].manipulate_PDF("line_flux", lambda EW: EW.to(u.AA) * \
                funcs.convert_mag_units(band_wav, self.property_PDFs[cont_name].input_arr, out_units), PDF_kwargs)
            if dust_label == "":
                self.property_PDFs[property_name] = line_flux_PDF
            else:
                self.property_PDFs[property_name] = line_flux_PDF.manipulate_PDF(property_name, funcs.dust_correct, dust_mag = self.property_PDFs[A_line_name].input_arr)
            self._update_properties_from_PDF(property_name)
        return self, property_name
    
    def calc_line_lum_rest_optical(self, line_names: list, frame: str = "obs", flux_contamination_params: dict = {"mu": 0.}, dust_author_year: str = "M99", \
            dust_law: str = "C00", dust_origin: str = "UV", medium_bands_only = True, rest_optical_wavs: u.Quantity = [3_700., 7_000.] * u.AA, rest_UV_wav_lims: \
            u.Quantity = [1_250., 3_000.] * u.AA, ref_wav: u.Quantity = 1_500. * u.AA, iters: int = 10, out_units: u.Unit = u.erg / u.s, extract_property_name = False):
        dust_label = self._get_dust_corr_label(dust_author_year = dust_author_year, dust_law = dust_law, dust_origin = dust_origin)
        flux_contam_label = self._get_rest_optical_flux_contam_label(line_names, flux_contamination_params)
        property_name = f"lum_{flux_contam_label}_{frame}{dust_label}"
        if extract_property_name:
            return property_name
        # if already stored in object, do nothing
        if property_name in self.properties.keys() and property_name in self.property_errs.keys() and property_name in self.property_PDFs.keys():
            galfind_logger.debug(f"{property_name=} already calculated!")
        else:
            line_flux_property_name = self.calc_line_flux_rest_optical(line_names, frame, flux_contamination_params, dust_author_year, \
                dust_law, dust_origin, medium_bands_only, rest_optical_wavs, rest_UV_wav_lims, ref_wav, iters)[1]
            if type(self.property_PDFs[line_flux_property_name]) == type(None):
                self._update_properties_and_PDFs(property_name, None)
                return self, property_name
            if frame == "rest":
                z = 0.
                lum_distance = funcs.calc_lum_distance(z = z)
            else: # frame == "obs"
                z = self.z
                lum_distance = funcs.calc_lum_distance(z = z)
            #PDF_kwargs = self.property_PDFs[line_flux_property_name].kwargs # {**, **{f"{line_names[0]}_cont_lines": str("+".join(line_names[1:]))}} #, \
                #**{f"flux_contamination_params_{key}": value for key, value in flux_contamination_params.items()}}
            self.property_PDFs[property_name] = self.property_PDFs[line_flux_property_name].manipulate_PDF(property_name, \
                lambda line_flux: (4 * np.pi * line_flux * lum_distance ** 2).to(u.erg / u.s))
            self._update_properties_from_PDF(property_name)
        return self, property_name

    def calc_xi_ion(self, frame: str = "rest", line_names: list = ["Halpha"], flux_contamination_params: dict = {}, \
            fesc_author_year: str = "fesc=0.0", dust_author_year: str = "M99", dust_law: str = "C00", dust_origin: str = "UV", \
            medium_bands_only: bool = True, rest_optical_wavs: u.Quantity = [3_700., 7_000.] * u.AA, rest_UV_wav_lims: u.Quantity = [1_250., 3_000.] * u.AA, \
            ref_wav: u.Quantity = 1_500. * u.AA, iters: int = 10, extract_property_name: bool = False):
        assert frame in ["rest", "obs"]
        assert "Halpha" in line_names
        flux_contam_label = self._get_rest_optical_flux_contam_label(line_names, flux_contamination_params)
        dust_label = self._get_dust_corr_label(dust_author_year = dust_author_year, dust_law = dust_law, dust_origin = dust_origin)
        property_name = f"xi_ion_{frame}_{flux_contam_label}{dust_label}_{fesc_author_year.replace('=', '')}"
        if extract_property_name:
            return property_name
        # if already stored in object, do nothing
        if property_name in self.properties.keys() and property_name in self.property_errs.keys() and property_name in self.property_PDFs.keys():
            galfind_logger.debug(f"{property_name=} already calculated!")
        else:
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
            line_lum_property_name = self.calc_line_lum_rest_optical(line_names, frame, flux_contamination_params, dust_author_year, \
                dust_law, dust_origin, medium_bands_only, rest_optical_wavs, rest_UV_wav_lims, ref_wav, iters = iters)[1]
            L_UV_property_name = self.calc_LUV_phot(frame = frame, rest_UV_wav_lims = rest_UV_wav_lims, ref_wav = ref_wav, dust_author_year = dust_author_year)[1]
            if any(type(self.property_PDFs[_property_name]) == type(None) for _property_name in [line_lum_property_name, L_UV_property_name]):
                self._update_properties_and_PDFs(property_name, None)
                return self, property_name
            #assert self.property_PDFs[line_lum_property_name].kwargs[f"{line_names[0]}_cont_lines"] == line_names[1:]
            # calculate xi_ion from Halpha luminosity
            PDF_kwargs = self.property_PDFs[L_UV_property_name].kwargs
            self.property_PDFs[property_name] = self.property_PDFs[line_lum_property_name].manipulate_PDF(property_name, \
                lambda int_line_lum: (int_line_lum / (1.36e-12 * u.erg * (1. - fesc_chain) * \
                self.property_PDFs[L_UV_property_name].input_arr)).to(u.Hz / u.erg), PDF_kwargs)
            self._update_properties_from_PDF(property_name)
        return self, property_name
        # assumes some case for ISM recombination
    
    # Rest optical line property naming functions

    def _get_UV_dust_label(self, dust_author_year: Union[str, None]):
        if type(dust_author_year) == type(None):
            return ""
        else:
            return f"_{dust_author_year}"

    def _get_dust_corr_label(self, dust_author_year: Union[str, None], dust_law: Union[str, None], dust_origin: str):
        if dust_origin != "UV":
            raise NotImplementedError
        if any(type(dust_name) == type(None) for dust_name in [dust_author_year, dust_law]):
            return ""
        else:
            return f"{self._get_UV_dust_label(dust_author_year)}_{dust_law}"

    def _get_rest_optical_flux_contam_label(self, line_names: list, flux_contamination_params: dict):
        assert all(line_name in line_diagnostics.keys() for line_name in line_names)
        assert(type(flux_contamination_params) == dict)
        flux_cont_keys = flux_contamination_params.keys()
        if "mu" in flux_cont_keys and "sigma" in flux_cont_keys:
            return f"{line_names[0]}_cont_G({flux_contamination_params['mu']:.1f},{flux_contamination_params['sigma']:.1f})" #_{'+'.join(line_names[1:])}"
        elif "mu" in flux_cont_keys and "sigma" not in flux_cont_keys:
            return f"{line_names[0]}_cont_{flux_contamination_params['mu']:.1f}" #_{'+'.join(line_names[1:])}"
        elif len(flux_contamination_params) == 0:
            return '+'.join(line_names)
        else:
            raise NotImplementedError

    def _get_rest_optical_flux_contam_scaling(self, flux_contamination_params: dict, iters: int):
        assert(type(flux_contamination_params) == dict)
        flux_cont_keys = flux_contamination_params.keys()
        if "mu" in flux_cont_keys and "sigma" in flux_cont_keys:
            return np.random.normal(1. - flux_contamination_params["mu"], flux_contamination_params["sigma"], iters)
        elif "mu" in flux_cont_keys and "sigma" not in flux_cont_keys:
            return 1. - flux_contamination_params["mu"]
        elif len(flux_contamination_params) == 0:
            return 1.
        else:
            raise NotImplementedError

    # Function to update rest-frame UV properties and PDFs
    def _update_properties_and_PDFs(self, property_name, property_vals, PDF_kwargs = {}):
        #breakpoint()
        if type(property_vals) == type(None):
            self.property_PDFs[property_name] = None
        else:
            # construct PDF from property_vals chain
            new_PDF = PDF.from_1D_arr(property_name, property_vals, PDF_kwargs)
            if property_name in self.property_PDFs.keys():
                old_PDF = self.property_PDFs[property_name]
                PDF_obj = old_PDF + new_PDF
            else:
                PDF_obj = new_PDF
            self.property_PDFs[property_name] = PDF_obj
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
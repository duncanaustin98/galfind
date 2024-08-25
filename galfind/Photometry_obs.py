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
from tqdm import tqdm
import time
from typing import Union
import inspect

from . import useful_funcs_austind as funcs
from . import galfind_logger, instr_to_name_dict
from .Photometry import Photometry
from .SED_result import Galaxy_SED_results, Catalogue_SED_results

class Photometry_obs(Photometry):

    def __init__(self, instrument, flux_Jy, flux_Jy_errs, aper_diam: u.Quantity, \
            min_flux_pc_err: Union[int, float], loc_depths, SED_results: dict = {}, timed: bool = False):
        if timed:
            start = time.time()
        self.aper_diam = aper_diam
        self.min_flux_pc_err = min_flux_pc_err
        self.SED_results = SED_results # array of SED_result objects with different SED fitting runs
        if timed:
            mid = time.time()
        self.aper_corrs = instrument.get_aper_corrs(self.aper_diam)
        if timed:
            mid_end = time.time()
        super().__init__(instrument, flux_Jy, flux_Jy_errs, loc_depths)
        if timed:
            end = time.time()
            print(mid - start, mid_end - mid, end - mid_end)

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
    
    def __getattr__(self, property_name: str, origin: Union[str, dict] = "phot_obs") -> Union[None, u.Quantity, u.Magnitude, u.Dex]:
        assert type(origin) in [str, dict], galfind_logger.critical(f"{origin=} with {type(origin)=} not in [str, dict]!")
        if "phot" in origin: # should have origin strings associated with Photometry.__getattr__ in list here!
            if property_name in self.__dict__.keys():
                return self.__getattribute__(property_name)
            elif "aper_corr" in property_name and property_name.split("_")[-1] in self.instrument.band_names:
                # return band aperture corrections
                return self.aper_corrs[property_name.split("_")[-1]]
            else:
                return super().__getattr__(property_name, "phot" if origin == "phot_obs" else origin.replace("phot_", ""))
        else:
            # determine property type from name
            property_type = property_name.split("_")[-1]
            if any(string == property_type.lower() for string in ["val", "errs", "l1", "u1", "pdf"]):
                property_name = property_name.replace(f"_{property_type}", "")
                property_type = property_type.lower()
            else:
                property_type = "_".join(property_name.split("_")[-2:])
                if property_type.lower() != "recently_updated":
                    # no property type, defaulting to value
                    galfind_logger.warning(f"No property_type given in suffix of {property_name=} for Photometry_rest.__getattr__. Defaulting to value")
                    property_name = property_name.replace(f"_{property_type}", "")
                    property_type = "val"
                else:
                    property_name = property_name.replace(f"_{property_type}", "")
                    property_type = "recently_updated"
            # determine relevant SED_result to use from origin keyword
            if type(origin) in [str]:
                if origin.endswith("_REST_PROPERTY"):
                    SED_results_key = origin[:-14]
                    origin = "phot_rest"
                elif origin.endswith("_SED"):
                    SED_results_key = origin[:-4]
                    origin = "SED"
                else:
                    SED_results_key = origin
                    origin = "SED_result"
            else: #type(origin) in [dict]:
                SED_results_key = origin["code"].label_from_SED_fit_params(origin)
                origin = "SED_result"
            assert SED_results_key in self.SED_results.keys(), galfind_logger.critical(f"{SED_results_key=} not in {self.SED_results.keys()=}!")
            return self.SED_results[SED_results_key].__getattr__(property_name, origin, property_type)

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
    def from_phot(cls, phot, aper_diam: u.Quantity, min_flux_pc_err: Union[int, float], SED_results: dict = {}):
        return cls(phot.instrument, phot.flux_Jy, phot.flux_Jy_errs, aper_diam, min_flux_pc_err, phot.loc_depths, SED_results)
    
    def update(self, gal_SED_results):
        if hasattr(self, "SED_results"):
            self.SED_results = {**self.SED_results, **gal_SED_results}
        else:
            self.SED_results = gal_SED_results
    
    def update_mask(self, mask, update_phot_rest: bool = False):
        assert len(self.flux_Jy) == len(mask)
        assert len(self.flux_Jy_errs) == len(mask)
        self.flux_Jy.mask = mask
        self.flux_Jy_errs.mask = mask
        return self
    
    def get_SED_fit_params_arr(self, code) -> list:
       return [code.SED_fit_params_from_label(label) for label in self.SED_results.keys()]
    
    def load_property(self, gal_property: Union[dict, u.Quantity], save_name: str) -> None:
        setattr(self, save_name, gal_property)

    def calc_ext_src_corrs(self, aper_corrs: Union[dict, None]) -> None:
        # FLUX_AUTO must already be loaded
        if not hasattr(self, "FLUX_AUTO"):
            galfind_logger.critical("Could not calculate ext_src_corrs as FLUX_AUTO not loaded in!")
            raise NotImplementedError
        # if not already calculated
        property_name = "_".join(inspect.stack()[0].function.split("_")[1:])
        if not hasattr(self, property_name):
            # calculate aperture corrections if not given
            if type(aper_corrs) == type(None):
                aper_corrs = self.instrument.get_aper_corrs(self.aper_diam)
                aper_corrs = {band_name: aper_corr for band_name, aper_corr \
                    in zip(self.instrument.band_names, aper_corrs)}
            # ensure all bands in self.instrument which have FLUX_AUTO's have aperture corrections calculated
            assert all(band_name in aper_corrs.keys() for band_name in self.instrument.band_names \
                if band_name in self.FLUX_AUTO.keys())
            # calculate extended source corrections in each band
            ext_src_corrs = {band_name: (self.FLUX_AUTO[band_name] / (self.flux_Jy[i] * \
                funcs.mag_to_flux_ratio(-aper_corrs[band_name]))).to(u.dimensionless_unscaled).unmasked \
                for i, band_name in enumerate(self.instrument.band_names) if band_name in self.FLUX_AUTO.keys()}
            # load these into self
            self.load_property(ext_src_corrs, property_name)
    
    def make_ext_src_corrs(self, gal_property: str, origin: Union[str, dict], \
            ext_src_band: Union[str, list, np.array] = "F444W") -> None:
        # determine correct SED fitting results to use
        if type(origin) in [dict]:
            origin_key = origin["code"].label_from_SED_fit_params(origin)
            rest_property = False
        elif type(origin) in [str]:
            if "_REST_PROPERTY" in origin:
                rest_property = True
            else:
                rest_property = False
            origin_key = origin.replace("_REST_PROPERTY", "")
        else:
            galfind_logger.critical(f"{type(origin)=} not in [str, dict]!")
        # skip if key not available
        if origin_key not in self.SED_results.keys():
            galfind_logger.warning(f"Could not compute ext_src_corrs for {gal_property=} as {origin_key=} not in {self.SED_results.keys()=}!")
        else:
            if rest_property:
                data_obj = self.SED_results[origin_key].phot_rest
            else:
                data_obj = self.SED_results[origin_key]
            properties = data_obj.properties
            property_errs = data_obj.property_errs
            property_PDFs = data_obj.property_PDFs
            # skip if galaxy property not in properties + property_errs + property_PDFs dicts
            if any(gal_property not in property_dict.keys() for property_dict in [properties, property_errs, property_PDFs]):
                galfind_logger.warning(f"{gal_property=},{origin_key=},{rest_property=} not in all of [{properties.keys()=},{property_errs.keys()=},{property_PDFs.keys()=}]!")
            else:
                orig_property = properties[gal_property]
                orig_property_PDF = property_PDFs[gal_property]
                assert orig_property_PDF.x.unit == orig_property.unit
                # errors may not necessarily have the same unit
                ext_src_corr = self.ext_src_corrs[ext_src_band]
                PDF_add_kwargs = {"ext_src_band": ext_src_band}
                if type(orig_property) in [u.Magnitude]:
                    correction = funcs.flux_to_mag_ratio(ext_src_corr.value) * u.mag # units are incorrect
                    updated_property = orig_property + correction
                    PDF_add_kwargs = {**PDF_add_kwargs, **{"ext_src_corr": correction}}
                    updated_property_PDF = orig_property_PDF.__add__(correction, \
                        name_ext = funcs.ext_src_label, add_kwargs = PDF_add_kwargs, save = True)
                elif type(orig_property) in [u.Dex]:
                    correction = u.Dex(np.log10(ext_src_corr.value))
                    updated_property = orig_property + correction
                    PDF_add_kwargs = {**PDF_add_kwargs, **{"ext_src_corr": correction}}
                    updated_property_PDF = orig_property_PDF.__add__(correction, \
                        name_ext = funcs.ext_src_label, add_kwargs = PDF_add_kwargs, save = True)
                elif type(orig_property) in [u.Quantity]:
                    updated_property = orig_property * ext_src_corr
                    PDF_add_kwargs = {**PDF_add_kwargs, **{"ext_src_corr": ext_src_corr}}
                    updated_property_PDF = orig_property_PDF.__mul__(ext_src_corr, \
                        name_ext = funcs.ext_src_label, add_kwargs = PDF_add_kwargs, save = True)
                else:
                    galfind_logger.warning(f"{gal_property}={orig_property} from {origin=} with {type(orig_property)=} not in [Magnitude, Dex, Quantity]")
                # update properties and property_PDFs (assume property_errs remain unaffected)
                data_obj.properties[updated_property_PDF.property_name] = updated_property
                data_obj.property_PDFs[updated_property_PDF.property_name] = updated_property_PDF
                # save non rest_property attributes outside of dict as well
                if not rest_property:
                    setattr(data_obj, updated_property_PDF.property_name, updated_property)

    def make_all_ext_src_corrs(self, ext_src_band: Union[str, list, np.array] = "F444W"):
        # extract previously calculated galaxy properties and their origins
        code_ext_src_property_dict = {key: [gal_property for gal_property in \
            self.SED_results[key].SED_fit_params["code"].ext_src_corr_properties \
            if gal_property in self.SED_results[key].properties.keys() and \
            gal_property in self.SED_results[key].property_PDFs.keys()] \
            for key in self.SED_results.keys()}
        sed_rest_ext_src_property_dict = {f"{key}_REST_PROPERTY": [gal_property \
            for gal_property in self.SED_results[key].phot_rest.properties.keys() \
            if gal_property.split("_")[0] in funcs.ext_src_properties and \
            gal_property in self.SED_results[key].phot_rest.property_PDFs.keys()] \
            for key in self.SED_results.keys()}
        ext_src_property_dict = {**code_ext_src_property_dict, **sed_rest_ext_src_property_dict}
        # make the extended source corrections
        [self.make_ext_src_corrs(gal_property, origin) for origin, gal_properties \
            in ext_src_property_dict.items() for gal_property in gal_properties] 

    def plot_phot(self, ax, wav_units: Union[str, u.Unit] = u.AA, mag_units: Union[str, u.Unit] = u.Jy, \
            plot_errs: dict = {"x": True, "y": True}, annotate: bool = True, uplim_sigma: float = 2., \
            auto_scale: bool = True, label_SNRs: bool = True, errorbar_kwargs: dict = {"ls": "", \
            "marker": "o", "ms": 4., "zorder": 100., "path_effects": [pe.withStroke(linewidth = 2., \
            foreground = "white")]}, filled: bool = True, colour: str = "black", label: str = "Photometry"):

        plot, wavs_to_plot, mags_to_plot, yerr, uplims = super().plot_phot \
            (ax, wav_units, mag_units, plot_errs, annotate, uplim_sigma, \
            auto_scale, errorbar_kwargs, filled, colour, label, return_extra = True)

        if label_SNRs:
            label_kwargs = {"ha": "center", "fontsize": "medium", "path_effects": [pe.withStroke(linewidth = 2., foreground = "white")], "zorder": 1_000.}
            label_func = lambda SNR: f"{SNR:.1f}$\sigma$" if SNR < 100 else f"{SNR:.0f}$\sigma$"
            if mag_units == u.ABmag:
                offset = 0.15
                [ax.annotate(label_func(SNR), (wav, mag - offset if is_uplim else mag + mag_u1 + offset), \
                    **label_kwargs) for i, (SNR, wav, mag, mag_l1, mag_u1, is_uplim) in \
                    enumerate(zip(self.SNR, wavs_to_plot, mags_to_plot, yerr[0], yerr[1], uplims))]
            else:
                offset = {"power density/spectral flux density wav": 0.1, "ABmag/spectral flux density": 0.1, \
                    "spectral flux density": 0.1}[str(u.get_physical_type(mag_units))]
                [ax.annotate(label_func(SNR), (wav, mag + offset if is_uplim else mag - mag_l1 - offset), \
                    **label_kwargs) for i, (SNR, wav, mag, mag_l1, mag_u1, is_uplim) in \
                    enumerate(zip(self.SNR, wavs_to_plot, mags_to_plot, yerr[0], yerr[1], uplims))]
        
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
    
    def __init__(self, instrument_arr, flux_Jy_arr, flux_Jy_errs_arr, aper_diam, min_flux_pc_err, loc_depths_arr, SED_results_arr = [], timed = True):
        # force SED_results_arr to have the same len as the number of input fluxes
        if SED_results_arr == []:
            SED_results_arr = np.full(len(flux_Jy_arr), {})
        if timed:
            self.phot_obs_arr = [Photometry_obs(instrument, flux_Jy, flux_Jy_errs, aper_diam, min_flux_pc_err, loc_depths, SED_results) \
                for instrument, flux_Jy, flux_Jy_errs, loc_depths, SED_results in tqdm(zip(instrument_arr, flux_Jy_arr, flux_Jy_errs_arr, \
                    loc_depths_arr, SED_results_arr), desc = "Initializing Multiple_Photometry_obs", total = len(instrument_arr))]
        else:
            self.phot_obs_arr = [Photometry_obs(instrument, flux_Jy, flux_Jy_errs, aper_diam, min_flux_pc_err, loc_depths, SED_results) \
                for instrument, flux_Jy, flux_Jy_errs, loc_depths, SED_results in zip(instrument_arr, flux_Jy_arr, flux_Jy_errs_arr, loc_depths_arr, SED_results_arr)]

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
    def from_fits_cat(cls, fits_cat, instrument, cat_creator, SED_fit_params_arr, timed = False):
        flux_Jy_arr, flux_Jy_errs_arr, gal_band_mask = cat_creator.load_photometry(fits_cat, instrument.band_names, timed = timed)
        depths_arr = cat_creator.load_depths(fits_cat, instrument.band_names, gal_band_mask, timed = timed)
        instrument_arr = cat_creator.load_instruments(instrument, gal_band_mask)
        if SED_fit_params_arr != [{}]:
            SED_results_arr = Catalogue_SED_results.from_fits_cat(fits_cat, cat_creator, SED_fit_params_arr, instrument = instrument).SED_results
        else:
            SED_results_arr = np.full(len(flux_Jy_arr), {})
        return cls(instrument_arr, flux_Jy_arr, flux_Jy_errs_arr, cat_creator.aper_diam, cat_creator.min_flux_pc_err, depths_arr, SED_results_arr, timed = timed)

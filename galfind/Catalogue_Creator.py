#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 13:59:59 2023

@author: austind
"""
# 
# Catalogue_Creator.py
import numpy as np
from astropy.utils.masked import Masked
import astropy.units as u
from astropy.table import Table, Row
import json
from abc import ABC, abstractmethod
import time
from tqdm import tqdm
from pathlib import Path
import h5py
from copy import copy, deepcopy

from . import useful_funcs_austind as funcs
from . import config, galfind_logger, SED_code, LePhare, EAZY, Bagpipes

class Catalogue_Creator(ABC):
    
    def __init__(self, property_conv_dict, aper_diam_index, flux_or_mag, min_flux_pc_err, ra_dec_labels, ID_label, zero_point = u.Jy.to(u.ABmag), phot_fits_ext = 0):
        self.property_conv_dict = property_conv_dict
        self.aper_diam_index = aper_diam_index # set to 'None' by default as very few people actually put arrays in catalogue columns
        self.flux_or_mag = flux_or_mag # either "flux" or "mag"
        self.min_flux_pc_err = min_flux_pc_err
        self.ra_dec_labels = ra_dec_labels
        self.ID_label = ID_label
        self.zero_point = zero_point # must be astropy units; can be either integer or dict of {band: zero_point}
        self.phot_fits_ext = phot_fits_ext # only compatible with .fits currently

    @abstractmethod
    def phot_labels(self, bands):
        pass

    @abstractmethod
    def mask_labels(self, bands):
        pass

    @abstractmethod
    def depth_labels(self, bands):
        pass

    def load_zero_points(self, bands):
        if isinstance(self.zero_point, dict):
            if all(band in self.zero_point.keys() for band in bands):
                zero_points = [self.zero_point[band] for band in bands]
            else:
                raise(Exception(f"Not all bands in self.zero_points = {self.zero_point_dict} in {self.__name__} !"))
        elif isinstance(self.zero_point, float):
            zero_points = list(np.full(len(bands), self.zero_point))
        else:
            raise(Exception(f"self.zero_point of type {type(self.zero_point)} in {__name__} must be of type 'dict' or 'int' !"))
        return np.array(zero_points)
    
    #@abstractmethod
    # def load_photometry(self, fits_cat, band):
    #     if isinstance(self.arr_index, int):
    #         zero_point = self.load_zero_point(band) # check that self.zero_point is saved in the correct format and extract ZP for this band
    #         phot_label, err_label = self.phot_conv(band, self.aper_diam_index)
    #         if self.flux_or_mag == "flux":
    #             phot = funcs.flux_image_to_Jy(fits_cat[phot_label], zero_point)
    #             phot_err = funcs.flux_image_to_Jy(fits_cat[err_label], zero_point)
    #             #phot, phot_err = self.apply_min_flux_pc_err(phot, phot_err)
    #         elif self.flux_or_mag == "mag":
    #             phot = funcs.mag_to_flux(fits_cat[phot_label], u.Jy.to(u.ABmag))
    #             phot_err = funcs.mag_to_flux
    #     else:
    #         raise(Exception(f"'arr_index' = {self.arr_index} is not valid in {__name__}! Must be either 'None' or type() = int !"))
    #     return phot, phot_err
    
    @abstractmethod
    def load_mask(self, fits_cat, bands):
        pass
    
    # def load_property(self, fits_cat, gal_property, code):
    #     return fits_cat[self.property_conv_dict[code.__class__.__name__][gal_property]]

    def load_flag(self, fits_cat, gal_flag):
        flag_label = self.flag_conv(gal_flag)
        try:
            return fits_cat[flag_label]
        except:
            return {} # None
    
    def apply_min_flux_pc_err(self, fluxes, errs):
        assert(fluxes.unit == errs.unit)
        errs = np.array([[self.min_flux_pc_err * flux / 100 if err / flux < self.min_flux_pc_err / 100 and flux > 0. else err \
            for flux, err in zip(gal_fluxes, gal_errs)] for gal_fluxes, gal_errs in zip(fluxes.value, errs.value)]) * fluxes.unit
        return fluxes, errs

# %% GALFIND conversion from photometry .fits catalogue row to Photometry_obs class

class GALFIND_Catalogue_Creator(Catalogue_Creator):
    
    def __init__(self, cat_type, aper_diam, min_flux_pc_err, zero_point = u.Jy.to(u.ABmag), flux_or_mag = "flux"):
        self.cat_type = cat_type
        self.aper_diam = aper_diam
        
        # only make these dicts once to speed up property loading
        same_key_value_properties = [] #["auto_corr_factor_UV", "auto_corr_factor_mass"]
        property_conv_dict = {sed_code: {**getattr(globals()[sed_code], sed_code)().galaxy_property_dict, **{element: element for element in same_key_value_properties}} for sed_code in json.loads(config["Other"]["CODES"])}
        
        ra_dec_labels = {"RA": "ALPHA_J2000", "DEC": "DELTA_J2000"}
        ID_label = "NUMBER"
        phot_fits_ext = 0 # check whether this works!
        aper_diam_index = int(json.loads(config.get("SExtractor", "APERTURE_DIAMS")).index(aper_diam.value))
        super().__init__(property_conv_dict, aper_diam_index, flux_or_mag, min_flux_pc_err, ra_dec_labels, ID_label, zero_point, phot_fits_ext)

    def sex_phot_labels(self, bands):
        # Updated to take a list of bands as input
        if self.flux_or_mag == "flux":
            phot_label = [f"FLUX_APER_{band}" for band in bands]
            err_label = [f"FLUXERR_APER_{band}" for band in bands]
        elif self.flux_or_mag == "mag":
            phot_label = [f"MAG_APER_{band}" for band in bands]
            err_label = [f"MAGERR_APER_{band}" for band in bands]
        else:
            raise(Exception("self.flux_or_mag = {self.flux_or_mag} is invalid! It should be either 'flux' or 'mag' !"))
        return phot_label, err_label
    
    def loc_depth_phot_labels(self, bands):
        # outputs catalogue column names for photometric fluxes + errors
        if self.flux_or_mag == "flux":
            phot_labels = [f"FLUX_APER_{band}_aper_corr_Jy" for band in bands]
            err_labels = [f"FLUXERR_APER_{band}_loc_depth_{str(int(self.min_flux_pc_err))}pc_Jy" for band in bands]
        elif self.flux_or_mag == "mag":
            raise(Exception("Not implemented mag local depth errors yet!")) # "Beware that mag errors are asymmetric!")
            phot_labels = [f"MAG_APER_{band}_aper_corr" for band in bands]
            err_labels = [[f"MAGERR_APER_{band}_l1_loc_depth", f"MAGERR_APER_{band}_u1_loc_depth"] for band in bands] # this doesn't currently work!
        else:
            raise(Exception("self.flux_or_mag = {self.flux_or_mag} is invalid! It should be either 'flux' or 'mag' !"))
        return phot_labels, err_labels
    
    def phot_labels(self, bands):
        if self.cat_type == "sex":
            return self.sex_phot_labels(bands)
        elif self.cat_type == "loc_depth":
            return self.loc_depth_phot_labels(bands)
        else:
            galfind_logger.critical(f"self.cat_type = {self.cat_type} not in ['sex', 'loc_depth']!")

    def mask_labels(self, bands):
        return [f"unmasked_{band}" for band in bands]

    def depth_labels(self, bands):
        return [f"loc_depth_{band}" for band in bands]
    
    def selection_labels(self, fits_cat):
        labels = [key.replace("SELECTED_", "") for key, value \
            in fits_cat.meta.items() if value == True and "SELECTED_" in key]
        return labels
    
    # current bottleneck
    @staticmethod
    def load_gal_instr_mask(phot, save_path, null_data_val = 0., timed = True):
        if Path(save_path).is_file():
            # load in gal_instr_mask from .h5
            hf = h5py.File(save_path, "r")
            gal_instr_mask = np.array(hf["has_data_mask"])
            galfind_logger.info(f"Loaded 'has_data_mask' from {save_path}")
        else:
            # calculate the mask that is used to crop photometry to only bands including data
            if timed:
                gal_instr_mask = [[True if val != null_data_val else False for val in gal_phot] for gal_phot in \
                    tqdm(phot, desc = "Making gal_instr_mask", total = len(phot))]
            else:
                gal_instr_mask = [[True if val != null_data_val else False for val in gal_phot] for gal_phot in phot]
            # save as .h5
            hf = h5py.File(save_path, "w")
            hf.create_dataset("has_data_mask", data = gal_instr_mask)
            galfind_logger.info(f"Saved 'has_data_mask' to {save_path}")
        hf.close()
        return gal_instr_mask
    
    @staticmethod
    def load_instruments(instrument, gal_band_mask):
        # create set of instruments to be pointed to by sources with these bands available
        unique_data_combinations = np.unique(gal_band_mask, axis = 0)
        unique_instruments = [deepcopy(instrument).remove_indices([i for i, has_data in enumerate(data_combination) if not has_data]) for data_combination in unique_data_combinations]
        instrument_arr = [unique_instruments[np.where(np.all(unique_data_combinations == data_comb, axis = 1))[0][0]] for data_comb in gal_band_mask]
        return instrument_arr

    # overriding load_photometry from parent class to include .T[aper_diam_index]'s
    def load_photometry(self, fits_cat, bands, gal_band_mask = None, timed = False):
        if timed:
            start_time = time.time()
        zero_points = self.load_zero_points(bands)
        phot_labels, err_labels = self.phot_labels(bands)
        assert len(phot_labels) == len(err_labels), galfind_logger.critical("Length of photometry and error labels inconsistent!")
        
        phot_cat = funcs.fits_cat_to_np(fits_cat, phot_labels)
        err_cat = funcs.fits_cat_to_np(fits_cat, err_labels)
        
        if self.flux_or_mag == "flux":
            phot = funcs.flux_image_to_Jy(phot_cat[:, :, self.aper_diam_index], zero_points)
            phot_err = funcs.flux_image_to_Jy(err_cat[:, :, self.aper_diam_index], zero_points)
            phot, phot_err = self.apply_min_flux_pc_err(phot, phot_err)
        elif self.flux_or_mag == "mag":
            raise(Exception("Beware that mag errors are asymmetric! FUNCTIONALITY NOT YET INCORPORATED!"))
            phot = funcs.mag_to_flux(fits_cat[phot_labels], u.Jy.to(u.ABmag))
            phot_err = funcs.mag_to_flux # this doesn't currently work!
        assert len(phot[0]) == len(bands)
        if timed:
            end_time = time.time()
            galfind_logger.info(f"Extracting photometry from fits took {(end_time - start_time):.1f}s")

        if len(bands) > 1:
            assert type(gal_band_mask) == type(None)
            # for each galaxy remove bands that have no data
            gal_band_mask_save_path = f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{fits_cat.meta['SURVEY']}/has_data_mask/{fits_cat.meta['SURVEY']}_{fits_cat.meta['VERSION']}.h5"
            funcs.make_dirs(funcs.split_dir_name(gal_band_mask_save_path, "dir"))
            gal_band_mask = self.load_gal_instr_mask(phot, gal_band_mask_save_path, timed = timed)
            if timed:
                _phot = [gal_phot[band_mask] for gal_phot, band_mask in \
                    tqdm(zip(phot, gal_band_mask), desc = "Removing photometric bands without data", total = len(phot))]
                _phot_err = [gal_phot_err[band_mask] for gal_phot_err, band_mask in \
                    tqdm(zip(phot_err, gal_band_mask), desc = "Removing photometric errors from bands without data", total = len(phot_err))]
            else:
                _phot = [gal_phot[band_mask] for gal_phot, band_mask in zip(phot, gal_band_mask)]
                _phot_err = [gal_phot_err[band_mask] for gal_phot_err, band_mask in zip(phot_err, gal_band_mask)]
        elif len(bands) == 1:
            gal_band_mask = [[True] for i in range(len(phot))]
            _phot = phot
            _phot_err = phot_err
        else:
            galfind_logger.critical("len(bands)==0 in GALFIND_Catalogue_Creator.load_photometry()")

        # mask these arrays based on whether or not each band is masked for each galaxy
        masked_arr = self.load_mask(fits_cat, bands, gal_band_mask, timed = timed)
        #assert masked_arr.shape == phot.shape, galfind_logger.critical("Length of mask arr and photometry is inconsistent!")
        if timed:
            phot = [Masked(gal_phot, mask = gal_mask) for gal_phot, gal_mask in \
                tqdm(zip(_phot, masked_arr), desc = "Loading photometry", total = len(_phot))] #Masked(_phot, mask = masked_arr)
            phot_err = [Masked(gal_phot_err, mask = gal_mask) for gal_phot_err, gal_mask in \
                tqdm(zip(_phot_err, masked_arr), desc = "Loading photometric errors", total = len(_phot_err))] #Masked(_phot_err, mask = masked_arr)
        else:
            phot = [Masked(gal_phot, mask = gal_mask) for gal_phot, gal_mask in zip(_phot, masked_arr)]
            phot_err = [Masked(gal_phot_err, mask = gal_mask) for gal_phot_err, gal_mask in zip(_phot_err, masked_arr)]
        return phot, phot_err, gal_band_mask
    
    def load_mask(self, fits_cat, bands, gal_band_mask = None, timed = False):
        band_mask_labels = self.mask_labels(bands)
        self.is_masked = True
        for label in band_mask_labels:
            if label not in fits_cat.colnames:
                galfind_logger.warning("Catalogue not yet masked in Catalogue_Creator.load_photometry()!")
                self.is_masked = False
                break
        if self.is_masked:
            masked_arr = np.invert(funcs.fits_cat_to_np(fits_cat, band_mask_labels, reshape_by_aper_diams = False))
        else: # nothing is masked
            masked_arr = np.full((len(fits_cat), len(bands)), False)
        if type(gal_band_mask) != type(None):
            if timed:
                masked_arr = np.array([[mask for has_data, mask in zip(_gal_band_mask, _gal_mask) if has_data] \
                    for _gal_band_mask, _gal_mask in tqdm(zip(gal_band_mask, masked_arr), desc = "Loading mask", total = len(gal_band_mask))])
            else:
                masked_arr = np.array([[mask for has_data, mask in zip(_gal_band_mask, _gal_mask) if has_data] \
                    for _gal_band_mask, _gal_mask in zip(gal_band_mask, masked_arr)])
        return masked_arr
    
    def load_depths(self, fits_cat, bands, gal_band_mask = None, timed = False):
        depth_labels = self.depth_labels(bands)
        self.has_depths = True
        for label in depth_labels:
            if label not in fits_cat.colnames:
                galfind_logger.warning("Catalogue not yet had depth calculations performed in Catalogue_Creator.load_photometry()!")
                self.has_depths = False
                break
        if self.has_depths:
            depths_arr = funcs.fits_cat_to_np(fits_cat, depth_labels)[:, :, self.aper_diam_index]
        else: # depths given as np.nan
            depths_arr = np.full((len(fits_cat), len(bands)), np.nan)
        if type(gal_band_mask) != type(None):
            if timed:
                depths_arr = np.array([[depth for depth, has_data in zip(_gal_depths, _gal_band_mask) if has_data] * u.ABmag for _gal_depths, _gal_band_mask \
                    in tqdm(zip(depths_arr, gal_band_mask), desc = "Loading depths", total = len(depths_arr))])
            else:
                depths_arr = np.array([[depth for depth, has_data in zip(_gal_depths, _gal_band_mask) if has_data] * u.ABmag for _gal_depths, _gal_band_mask in zip(depths_arr, gal_band_mask)])
        return depths_arr
    

# %% Common catalogue converters

class JADES_DR1_Catalogue_Creator(Catalogue_Creator):
    
    def __init__(self, aper_diam_index, phot_fits_ext, min_flux_pc_err = None, flux_or_mag = "flux"):
        super().__init__(self.phot_conv, self.property_conv, aper_diam_index, flux_or_mag, min_flux_pc_err, u.nJy.to(u.ABmag), phot_fits_ext)

    def phot_conv(self, band):
        if self.flux_or_mag == "flux":
            if self.phot_fits_ext == 4: # CIRC
                phot_label = f"{band.replace('f', 'F')}_CIRC{self.aper_diam_index}"
                err_label = f"{phot_label}_e"
        return phot_label, err_label
        
    def property_conv(self, gal_property):
        property_conv_dict = {"z": ""}
        return property_conv_dict[gal_property]
    
    # overriding load_photometry from parent class to include .T[aper_diam_index]'s
    def load_photometry(self, fits_cat, band):
        zero_point = self.load_zero_point(band) # check that self.zero_point is saved in the correct format and extract ZP for this band
        phot_label, err_label = self.phot_conv(band)
        if self.flux_or_mag == "flux":
            phot = funcs.flux_image_to_Jy(fits_cat[phot_label], zero_point)
            phot_err = funcs.flux_image_to_Jy(fits_cat[err_label], zero_point)
        elif self.flux_or_mag == "mag":
            print("Beware that mag errors are asymmetric!")
            phot = funcs.mag_to_flux(fits_cat[phot_label], u.Jy.to(u.ABmag))
            phot_err = funcs.mag_to_flux # this doesn't currently work!
        return phot, phot_err

#JADES_DR1_cat_creator = JADES_DR1_Catalogue_Creator(2, 4) # 0.15 arcsec radius apertures; CIRC .fits table extension

# JAGUAR



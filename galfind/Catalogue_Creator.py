#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 13:59:59 2023

@author: austind
"""
# 
# Catalogue_Creator.py
import numpy as np
import astropy.units as u
from astropy.table import Table, Row
import json

from . import useful_funcs_austind as funcs
from . import config
from . import SED_code, LePhare, EAZY, Bagpipes

class Catalogue_Creator:
    
    def __init__(self, phot_conv, property_conv, flag_conv, aper_diam_index, flux_or_mag, min_flux_pc_err, ra_dec_labels, ID_label, zero_point = u.Jy.to(u.ABmag), phot_fits_ext = 0):
        self.phot_conv = phot_conv
        self.property_conv = property_conv
        self.flag_conv = flag_conv
        self.aper_diam_index = aper_diam_index # set to 'None' by default as very few people actually put arrays in catalogue columns
        self.flux_or_mag = flux_or_mag # either "flux" or "mag"
        self.min_flux_pc_err = min_flux_pc_err
        self.ra_dec_labels = ra_dec_labels
        self.ID_label = ID_label
        self.zero_point = zero_point # must be astropy units; can be either integer or dict of {band: zero_point}
        self.phot_fits_ext = phot_fits_ext # only compatible with .fits currently
        
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
    
    def load_photometry(self, fits_cat, band):
        if isinstance(self.arr_index, int):
            zero_point = self.load_zero_point(band) # check that self.zero_point is saved in the correct format and extract ZP for this band
            phot_label, err_label = self.phot_conv(band, self.aper_diam_index)
            if self.flux_or_mag == "flux":
                phot = funcs.flux_image_to_Jy(fits_cat[phot_label], zero_point)
                phot_err = funcs.flux_image_to_Jy(fits_cat[err_label], zero_point)
                #phot, phot_err = self.apply_min_flux_pc_err(phot, phot_err)
            elif self.flux_or_mag == "mag":
                phot = funcs.mag_to_flux(fits_cat[phot_label], u.Jy.to(u.ABmag))
                phot_err = funcs.mag_to_flux
        else:
            raise(Exception(f"'arr_index' = {self.arr_index} is not valid in {__name__}! Must be either 'None' or type() = int !"))
        return phot, phot_err
    
    def load_property(self, fits_cat, gal_property, code):
        property_label = self.property_conv(gal_property, code)
        return fits_cat[property_label]
    
    def load_flag(self, fits_cat, gal_flag):
        flag_label = self.flag_conv(gal_flag)
        try:
            return fits_cat[flag_label]
        except:
            return None
    
    def apply_min_flux_pc_err(self, fluxes, errs):
        errs = np.array([[self.min_flux_pc_err * flux / 100 if err / flux < self.min_flux_pc_err / 100 and flux > 0. else err \
                          for flux, err in zip(gal_fluxes, gal_errs)] for gal_fluxes, gal_errs in zip(fluxes, errs)])
        return fluxes, errs

# %% GALFIND conversion from photometry .fits catalogue row to Photometry_obs class

class GALFIND_Catalogue_Creator(Catalogue_Creator):
    
    def __init__(self, cat_type, aper_diam, min_flux_pc_err, zero_point = u.Jy.to(u.ABmag), flux_or_mag = "flux"):
        self.cat_type = cat_type
        self.aper_diam = aper_diam
        if cat_type == "sex":
            phot_conv = self.sex_phot_conv
        elif cat_type == "loc_depth":
            phot_conv = self.loc_depth_phot_conv
        else:
            raise(Exception(f"'cat_type' = {cat_type} is not valid in {__name__}! Must be either 'sex' or 'loc_depth' !"))
        
        # only make these dicts once to speed up property loading
        same_key_value_properties = [] #["auto_corr_factor_UV", "auto_corr_factor_mass"]
        self.property_conv_dict = {sed_code: {**getattr(globals()[sed_code], sed_code)().galaxy_property_dict, **{element: element for element in same_key_value_properties}} for sed_code in json.loads(config["Other"]["CODES"])}
        same_key_value_flags = ["robust", "good", "robust_relaxed", "good_relaxed", "blank_module"] + [f"unmasked_{band}" for band in json.loads(config.get("Other", "ALL_BANDS"))]
        self.flag_conv_dict = {element: element for element in same_key_value_flags}
        
        property_conv = self.property_conv
        flag_conv = self.flag_conv
        ra_dec_labels = {"RA": "ALPHA_J2000", "DEC": "DELTA_J2000"}
        ID_label = "NUMBER"
        phot_fits_ext = 0 # check whether this works!
        aper_diam_index = int(json.loads(config.get("SExtractor", "APERTURE_DIAMS")).index(aper_diam.value))
        #aper_diam_index = np.where(aper_diam.value == json.loads(config.get("SExtractor", "APERTURE_DIAMS")))[0][0]
        super().__init__(phot_conv, property_conv, flag_conv, aper_diam_index, flux_or_mag, min_flux_pc_err, ra_dec_labels, ID_label, zero_point, phot_fits_ext)

    def sex_phot_conv(self, bands):
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
    
    def loc_depth_phot_conv(self, bands):
        # outputs catalogue column names for photometric fluxes + errors
        if self.flux_or_mag == "flux":
            phot_labels = [f"FLUX_APER_{band}_aper_corr_Jy" for band in bands]
            err_labels = [f"FLUXERR_APER_{band}_loc_depth_{str(int(self.min_flux_pc_err))}pc_Jy" for band in bands]
        elif self.flux_or_mag == "mag":
            raise(Exception("Not implemented mag localc depth errors yet!")) # "Beware that mag errors are asymmetric!")
            phot_labels = [f"MAG_APER_{band}_aper_corr" for band in bands]
            err_labels = [[f"MAGERR_APER_{band}_l1_loc_depth", f"MAGERR_APER_{band}_u1_loc_depth"] for band in bands] # this doesn't currently work!
        else:
            raise(Exception("self.flux_or_mag = {self.flux_or_mag} is invalid! It should be either 'flux' or 'mag' !"))
        return phot_labels, err_labels
    
    def property_conv(self, gal_property, code):
        return self.property_conv_dict[code.code_name][gal_property]
    
    def flag_conv(self, gal_flag):
        return self.flag_conv_dict[gal_flag]
    
    # overriding load_photometry from parent class to include .T[aper_diam_index]'s
    def load_photometry(self, fits_cat, bands):
        zero_points = self.load_zero_points(bands)
        phot_labels, err_labels = self.phot_conv(bands)
        assert len(phot_labels) == len(err_labels), "Length of photometry and error labels inconsistent!"
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
        return phot, phot_err

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



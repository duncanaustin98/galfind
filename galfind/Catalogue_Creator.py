#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 13:59:59 2023

@author: austind
"""

# Catalogue_Creator.py
import numpy as np
import astropy.units as u
import json

from . import useful_funcs_austind as funcs
from . import config

class Catalogue_Creator:
    
    def __init__(self, phot_conv, property_conv, aper_diam_index, flux_or_mag, zero_point = u.Jy.to(u.ABmag), phot_fits_ext = 0):
        self.phot_conv = phot_conv
        self.property_conv = property_conv
        self.aper_diam_index = aper_diam_index # set to 'None' by default as very few people actually put arrays in catalogue columns
        self.flux_or_mag = flux_or_mag # either "flux" or "mag"
        self.zero_point = zero_point # must be astropy units; can be either integer or dict of {band: zero_point}
        self.phot_fits_ext = phot_fits_ext # only compatible with .fits currently
    
    def load_zero_point(self, band):
        if isinstance(self.zero_point, dict):
            if band in self.zero_point.keys():
                # determine the ZP for this band
                zero_point = self.zero_point[band]
            else:
                raise(Exception(f"{band} not in self.zero_point = {self.zero_point_dict} in {__name__} !"))
        elif isinstance(self.zero_point, int):
            zero_point = self.zero_point
        else:
            raise(Exception(f"self.zero_point of type {type(self.zero_point)} in {__name__} must be of type 'dict' or 'int' !"))
        return zero_point
    
    def load_photometry(self, fits_cat, band):
        if isinstance(self.arr_index, int):
            zero_point = self.load_zero_point(band) # check that self.zero_point is saved in the correct format and extract ZP for this band
            phot_label, err_label = self.phot_conv(band, self.aper_diam_index)
            if self.flux_or_mag == "flux":
                phot = funcs.flux_image_to_Jy(fits_cat[phot_label], zero_point)
                phot_err = funcs.flux_image_to_Jy(fits_cat[err_label], zero_point)
            elif self.flux_or_mag == "mag":
                phot = funcs.mag_to_flux(fits_cat[phot_label], u.Jy.to(u.ABmag))
                phot_err = funcs.mag_to_flux
        else:
            raise(Exception(f"'arr_index' = {self.arr_index} is not valid in {__name__}! Must be either 'None' or type() = int !"))
        return phot, phot_err

# %% GALFIND conversion from photometry .fits catalogue row to Photometry_obs class

class GALFIND_Catalogue_Creator(Catalogue_Creator):
    
    def __init__(self, cat_type, aper_diam, zero_point, flux_or_mag = "flux"):
        if cat_type == "sex":
            phot_conv = self.sex_phot_conv
        elif cat_type == "loc_depth":
            phot_conv = self.loc_depth_phot_conv
        else:
            raise(Exception(f"'cat_type' = {cat_type} is not valid in {__name__}! Must be either 'sex' or 'loc_depth' !"))
        property_conv = self.property_conv
        phot_fits_ext = 0 # check whether this works!
        aper_diam_index = np.where(aper_diam.value == json.loads(config.get("SExtractor", "APERTURE_DIAMS")))[0][0]
        super().__init__(phot_conv, property_conv, phot_fits_ext, aper_diam_index, flux_or_mag, zero_point, phot_fits_ext)

    def sex_phot_conv(self, band):
        if self.mag_or_flux_units == "flux":
            phot_label = f"FLUX_APER_{band}"
            err_label = f"FLUXERR_APER_{band}"
        elif self.mag_or_flux_units == "mag":
            phot_label = f"MAG_APER_{band}"
            err_label = f"MAGERR_APER_{band}"
        else:
            raise(Exception("self.mag_or_flux_units = {self.mag_or_flux_units} is invalid! It should be either 'flux' or 'mag' !"))
        return phot_label, err_label
    
    def loc_depth_phot_conv(self, band):
        # outputs catalogue column names for photometric fluxes + errors
        if self.mag_or_flux_units == "flux":
            phot_label = f"FLUX_APER_{band}_aper_corr"
            err_label = f"FLUXERR_APER_{band}_loc_depth"
        elif self.mag_or_flux_units == "mag":
            print("Beware that mag errors are asymmetric!")
            phot_label = f"MAG_APER_{band}_aper_corr"
            err_label = [f"MAGERR_APER_{band}_l1_loc_depth", f"MAGERR_APER_{band}_u1_loc_depth"] # this doesn't currently work!
        else:
            raise(Exception("self.mag_or_flux_units = {self.mag_or_flux_units} is invalid! It should be either 'flux' or 'mag' !"))
        return phot_label, err_label
    
    def property_conv(self, code):
        property_conv_dict = {}
        return
    
    # overriding load_photometry from parent class to include .T[aper_diam_index]'s
    def load_photometry(self, fits_cat, band):
        if isinstance(self.arr_index, int):
            zero_point = self.load_zero_point(band) # check that self.zero_point is saved in the correct format and extract ZP for this band
            phot_label, err_label = self.phot_conv(band)
            if self.flux_or_mag == "flux":
                phot = funcs.flux_image_to_Jy(fits_cat[phot_label].T[self.aper_diam_index], zero_point)
                phot_err = funcs.flux_image_to_Jy(fits_cat[err_label].T[self.aper_diam_index], zero_point)
            elif self.flux_or_mag == "mag":
                print("Beware that mag errors are asymmetric!")
                phot = funcs.mag_to_flux(fits_cat[phot_label], u.Jy.to(u.ABmag))
                phot_err = funcs.mag_to_flux # this doesn't currently work!
        else:
            raise(Exception(f"'arr_index' = {self.arr_index} is not valid in {__name__}! Must be either 'None' or type() = int !"))
        return phot, phot_err
    

# %% Common catalogue converters

class JADES_DR1_Catalogue_Creator(Catalogue_Creator):
    
    def __init__(self, aper_diam_index, flux_or_mag = "flux"):
        super().__init__(self.phot_conv, self.property_conv, aper_diam_index, flux_or_mag, u.uJy.to(u.ABmag))

    def JADES_DR1_Catalogue_Creator(self, )

# JADES-DR1

def JADES_DR1_phot_conv(band, aper_diam_index):
    band.replace("f", "F")
    return

def JADES_DR1_property_conv(gal_property):
    return

zero_point = 0

JADES_cat_creator = Catalogue_Creator(JADES_DR1_phot_conv, JADES_DR1_property_conv, zero_point)

# JAGUAR
    
    
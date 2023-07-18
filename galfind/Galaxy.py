#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:11:23 2023

@author: austind
"""

# Galaxy.py
import numpy as np
from copy import copy, deepcopy
import astropy.units as u
from astropy.coordinates import SkyCoord

from . import useful_funcs_austind as funcs
from . import Photometry_rest, Photometry_obs

class Galaxy:
    
    def __init__(self, sky_coord, ID, phot, mask_flags = {}):
        self.sky_coord = sky_coord
        self.ID = int(ID)
        self.phot = phot
        self.mask_flags = mask_flags
        
    @classmethod
    def from_fits_cat(cls, fits_cat_row, instrument, cat_creator, codes, low_z_runs):
        # load multiple photometries from the fits catalogue
        phot = [Photometry_obs.from_fits_cat(fits_cat_row, instrument, cat_creator, cat_creator.aper_diam, cat_creator.min_flux_pc_err, codes, low_z_runs)] # \
                # for min_flux_pc_err in cat_creator.min_flux_pc_err for aper_diam in cat_creator.aper_diam]
        # load the ID and Sky Coordinate from the source catalogue
        ID = int(fits_cat_row[cat_creator.ID_label])
        sky_coord = SkyCoord(fits_cat_row[cat_creator.ra_dec_labels["RA"]] * u.deg, fits_cat_row[cat_creator.ra_dec_labels["DEC"]] * u.deg, frame = "icrs")
        # mask flags should come from cat_creator
        mask_flags = {band: cat_creator.load_flag(fits_cat_row, f"unmasked_{band}") for band in instrument.bands}
        return cls(sky_coord, ID, phot, mask_flags)
    
    def update(self, SED_result): # for now just update the single photometry
        self.phot = phot[0].update(SED_result)
        
    def update_mask_full(self, bool_values):
        pass
        
    def update_mask_band(self, band, bool_value):
        self.mask_flags[band] = bool_value
        
    def __setattr__(self, name, value, obj = "gal"):
        if obj == "gal":
            if type(name) != list and type(name) != np.array:
                super().__setattr__(name, value)
            else:
                # use setattr to set values within Galaxy dicts (e.g. properties)
                self.globals()[name[0]][name[1]] = value
        else:
            raise(Exception(f"obj = {obj} must be 'gal'!"))
    
    # STILL NEED TO LOOK FURTHER INTO THIS
    def __deepcopy__(self, memo):
        print("Overriding Galaxy.__deepcopy__()")
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, deepcopy(value, memo))
        return result
    
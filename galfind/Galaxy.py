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
from . import SED_result

class Galaxy:
    
    # should really expand this to allow for more than one redshift here (only works fro one 'code' class at the moment)
    def __init__(self, sky_coord, ID, phot, SED_results, mask_flags = {}):
        # print("'z' here for a short time not a long time (in the 'Galaxy' class)! PUT THIS INSTEAD IN THE 'CODE' class")
        self.sky_coord = sky_coord
        self.ID = int(ID)
        self.phot = phot
        self.SED_results = SED_results
        self.mask_flags = mask_flags
        
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
        
    # @classmethod
    # def from_sex_cat_row(cls, sex_cat_row, instrument, cat_creator):
    #     # load the photometry from the sextractor catalogue
    #     phot = Photometry_obs.get_phot_from_sex(sex_cat_row, instrument, cat_creator)
    #     # load the ID and Sky Coordinate from the source catalogue
    #     ID = sex_cat_row["NUMBER"]
    #     sky_coord = SkyCoord(sex_cat_row["ALPHA_J2000"] * u.deg, sex_cat_row["DELTA_J2000"] * u.deg, frame = "icrs")
    #     # perform SED fitting to measure the redshift from the photometry
    #     # for now, load in z = 0 as a placeholder
    #     return cls(sky_coord, phot, ID, {})
        
    @classmethod # currently only works for a singular code
    def from_photo_z_cat(cls, cat_path, ID, instrument, cat_creator, code_names, low_z_runs, mask_flags = {}):
        # load the photometry from the sextractor catalogue
        cat = funcs.cat_from_path(cat_path)
        photo_z_cat_row = cat[cat["NUMBER"] == ID]
        
        # include multiple photometries
        phot = Photometry_obs.get_phot_from_sex(photo_z_cat_row, instrument, cat_creator)
        # load the ID and Sky Coordinate from the source catalogue
        sky_coord = SkyCoord(photo_z_cat_row["ALPHA_J2000"] * u.deg, photo_z_cat_row["DELTA_J2000"] * u.deg, frame = "icrs")
        # also load the galaxy properties from the catalogue # remove 'None' results from this array
        raise(Exception("Put this in the Photometry_obs class!"))
        SED_results = [SED_result.from_photo_z_cat(name, phot, ID, cat_path, cat_creator, low_z_run) for name, low_z_run in zip(code_names, low_z_runs)]
        #properties = {code.code_name: {gal_property: photo_z_cat_row[property_label] for gal_property, property_label in code.galaxy_property_labels.items()} for code in codes}
        return cls(sky_coord, ID, phot, SED_results, mask_flags)
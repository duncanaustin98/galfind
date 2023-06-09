#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:58:06 2023

@author: austind
"""

# run_depths.py
#import galfind
from galfind import Data
import astropy.units as u
#from Data import *
#from useful_funcs_austind import *

def run_depths(surveys, version, xy_offsets, aper_diams, incl_offset = True):
    for survey, xy_offset in zip(surveys, xy_offsets):
        data = Data.from_NIRCam_pipeline(survey, version)
        data.make_sex_cats()
        data.combine_sex_cats()
        if incl_offset:
            data.calc_depths(xy_offset = xy_offset, aper_diams = aper_diams)
        else:
            data.calc_depths()
        data.make_loc_depth_cat(aper_diams = aper_diams)

if __name__ == "__main__":
    version = "v8"
    surveys = ["GLASS"]
    aper_diams = [0.32] * u.arcsec
    xy_offsets = [[100, 0]]
    run_depths(surveys, version, xy_offsets, aper_diams, incl_offset = True)

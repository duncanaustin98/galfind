#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:36:34 2023

@author: austind
"""

# NIRCam_pipeline.py

import astropy.units as u
import numpy as np

from galfind import Catalogue, LePhare, EAZY

def NIRCam_pipeline(surveys, version, xy_offsets, aper_diams, sed_codes, forced_phot_band, excl_bands):
    for survey, xy_offset in zip(surveys, xy_offsets):
        cat = Catalogue.from_NIRCam_pipeline(survey, version, aper_diams, xy_offset, forced_phot_band, excl_bands)
        for code in sed_codes:
            cat = code.fit_cat(cat)

if __name__ == "__main__":
    version = "lit_version"
    surveys = ["JADES-DR1"]
    aper_diams = [0.32] * u.arcsec
    xy_offsets = [[0, 0]]
    sed_codes = []#[LePhare()]
    forced_phot_band = "f444W"
    excl_bands = []#["f090W", "f115W", "f277W", "f335M", "f356W", "f410M", "f444W"]
    NIRCam_pipeline(surveys, version, xy_offsets, aper_diams, sed_codes, forced_phot_band, excl_bands)
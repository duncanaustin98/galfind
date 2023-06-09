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

def NIRCam_pipeline(surveys, version, xy_offsets, aper_diams, sed_codes):
    for survey, xy_offset in zip(surveys, xy_offsets):
        cat = Catalogue.from_NIRCam_pipeline(survey, version, aper_diams, xy_offset)
        for code in sed_codes:
            cat = code.fit_cat(cat)

if __name__ == "__main__":
    version = "v8a"
    surveys = ["NEP-3"]
    aper_diams = [0.32] * u.arcsec
    xy_offsets = [[100, 0]]
    sed_codes = [LePhare()]
    NIRCam_pipeline(surveys, version, xy_offsets, aper_diams, sed_codes)
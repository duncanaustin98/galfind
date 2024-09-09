#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:58:06 2023

@author: austind
"""

# run_depths.py
# import galfind
import astropy.units as u

from galfind import Data

# from Data import *
# from useful_funcs_austind import *


def run_depths(
    surveys,
    version,
    xy_offsets,
    aper_diams,
    forced_phot_band,
    excl_bands,
    fast_depths,
    incl_offset=True,
):
    for survey, xy_offset in zip(surveys, xy_offsets):
        data = Data.from_pipeline(survey, version, excl_bands=excl_bands)
        data.make_sex_cats(forced_phot_band=forced_phot_band)
        # data.combine_sex_cats()
        if incl_offset:
            data.calc_depths(
                xy_offset=xy_offset,
                aper_diams=aper_diams,
                fast_depths=fast_depths,
            )
        else:
            data.calc_depths()
        # data.make_loc_depth_cat(aper_diams = aper_diams)


if __name__ == "__main__":
    version = "v9"
    surveys = ["JADES-3215"]
    aper_diams = [0.32] * u.arcsec
    xy_offsets = [[0, 0]]
    forced_phot_band = "f090W"
    fast_depths = True
    jades_3215_excl_bands = [
        "f162M",
        "f115W",
        "f150W",
        "f200W",
        "f410M",
        "f182M",
        "f210M",
        "f250M",
        "f300M",
        "f335M",
        "f277W",
        "f356W",
        "f444W",
    ]
    excl_bands = jades_3215_excl_bands
    run_depths(
        surveys,
        version,
        xy_offsets,
        aper_diams,
        forced_phot_band,
        excl_bands,
        fast_depths,
        incl_offset=True,
    )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 01:21:35 2023

@author: austind
"""

# Simulated_Galaxy.py
import astropy.units as u
from astropy.coordinates import SkyCoord

from . import Galaxy, Photometry_obs

class Simulated_Galaxy(Galaxy):
    
    def __init__(self, sky_coord, phot, ID, sim_z, sim, codes):
        self.sim = sim
        super().__init__(sky_coord, phot, ID, sim_z, codes)
    
    @classmethod # currently only works for a singular code
    def from_sim_cat_row(cls, sim_cat_row, instrument, codes, sim, depths):
        phot = Photometry_obs.get_phot_from_sim(sim_cat_row, instrument)
        ID = sim_cat_row[sim.cat_keys["ID"]]
        sky_coord = SkyCoord(sim_cat_row[sim.cat_keys["RA"]] * u.deg, sim_cat_row[sim.cat_keys["DEC"]] * u.deg, frame = "icrs")
        z = sim_cat_row[sim.cat_keys["z"]]
        return cls(sky_coord, phot, ID, z, sim, codes)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:03:20 2023

@author: austind
"""

# Photometry_obs.py
import numpy as np
import astropy.constants as const
import astropy.units as u
from copy import copy, deepcopy

from . import Photometry
from . import SED_result

class Photometry_obs(Photometry):

    def __init__(self, instrument, flux_Jy, flux_Jy_errs, aper_diam, min_flux_pc_err, loc_depths, SED_results = []):
        self.aper_diam = aper_diam
        self.min_flux_pc_err = min_flux_pc_err
        self.SED_results = SED_results # array of SED_result objects with different SED fitting runs
        super().__init__(instrument, flux_Jy, flux_Jy_errs, loc_depths)

    @property
    def flux_lambda(self): # wav and flux_nu must have units here!
        return (self.flux_Jy * const.c / ((np.array([value.value for value in self.instrument.band_wavelengths.values()]) * u.Angstrom) ** 2)).to(u.erg / (u.s * (u.cm ** 2) * u.Angstrom)) # both flux_nu and wav must be in the same rest or observed frame
    
    @property
    def flux_lambda_errs(self):
        return (self.flux_Jy_errs * const.c / ((np.array([value.value for value in self.instrument.band_wavelengths.values()]) * u.Angstrom) ** 2)).to(u.erg / (u.s * (u.cm ** 2) * u.Angstrom))

    @classmethod # not a gal object here, more like a catalogue row
    def from_fits_cat(cls, fits_cat_row, instrument, cat_creator, aper_diam, min_flux_pc_err, codes, low_z_runs):
        phot = Photometry.from_fits_cat(fits_cat_row, instrument, cat_creator)
        SED_results = [SED_result.from_fits_cat(fits_cat_row, code, phot, cat_creator, low_z_run) for code, low_z_run in zip(codes, low_z_runs)]
        return cls.from_phot(phot, aper_diam, min_flux_pc_err, SED_results)
    
    @classmethod
    def from_phot(cls, phot, aper_diam, min_flux_pc_err, SED_results = []):
        return cls(phot.instrument, phot.flux_Jy, phot.flux_Jy_errs, aper_diam, min_flux_pc_err, phot.loc_depths, SED_results)
    
    def update(self, SED_result):
        self.SED_results.append(SED_result)
    
    # @classmethod
    # def get_phot_from_sim(cls, gal, instrument, sim, min_flux_err_pc = 5):
    #     fluxes = []
    #     flux_errs = []
    #     instrument_copy = instrument.copy()
    #     for band in instrument_copy.bands:
    #         try:
    #             flux = np.array(gal[sim.flux_col_name(band)])
    #             err = np.array(gal[sim.flux_err_name(band)])
    #             # encorporate minimum flux error
    #             err = np.array([err_band if err_band / flux_band >= min_flux_err_pc / 100 else \
    #                             min_flux_err_pc * flux_band / 100 for flux_band, err_band in zip(flux, err)])
    #             flux_Jy = funcs.flux_image_to_Jy(flux, sim.zero_point)
    #             err_Jy = funcs.flux_image_to_Jy(err, sim.zero_point)
    #             fluxes = np.append(fluxes, flux_Jy.value)
    #             flux_errs = np.append(flux_errs, err_Jy.value)
    #         except:
    #             instrument.remove_band(band)
    #             print(f"{band} flux not loaded")
    
    def load_local_depths(self, sex_cat_row, instrument, aper_diam_index):
        self.loc_depths = np.array([sex_cat_row[f"loc_depth_{band}"].T[aper_diam_index] for band in instrument.bands])

# %%    
        
class Multiple_Photometry_obs:
    
    def __init__(self, instrument, flux_Jy_arr, flux_Jy_errs_arr, aper_diam, min_flux_pc_err, loc_depths_arr, SED_results_arr = []):
        # force SED_results_arr to have the same len as the number of input fluxes
        self.phot_obs_arr = [Photometry_obs(instrument, flux_Jy * u.Jy, flux_Jy_errs * u.Jy, aper_diam, min_flux_pc_err, loc_depths, SED_results) \
                         for flux_Jy, flux_Jy_errs, loc_depths, SED_results in zip(flux_Jy_arr, flux_Jy_errs_arr, loc_depths_arr, SED_results_arr)]
            
    def __repr__(self):
        # string representation of what is stored in this class
        return str(self.__dict__)
    
    def __len__(self):
        return len(self.phot_obs_arr)
    
    def __iter__(self):
        self.iter = 0
        return self
    
    def __next__(self):
        if self.iter > len(self) - 1:
            raise StopIteration
        else:
            phot = self[self.iter]
            self.iter += 1
            return phot
    
    def __getitem__(self, index):
        return self.phot_obs_arr[index]
    
    @classmethod
    def from_fits_cat(cls, fits_cat, instrument, cat_creator, aper_diam, min_flux_pc_err, codes, low_z_runs):
        flux_Jy_arr, flux_Jy_errs_arr = cat_creator.load_photometry(fits_cat, instrument.bands)
        # TO DO: MAKE MULTIPLE_SED_RESULTS CLASS TO LOAD IN SED_RESULTS SIMULTANEOUSLY (LEAVE AS [] FOR NOW)
        loc_depths_arr = np.full(len(flux_Jy_arr), None)
        SED_results_arr = np.full(len(flux_Jy_arr), None)
        return cls(instrument, flux_Jy_arr, flux_Jy_errs_arr, aper_diam, min_flux_pc_err, loc_depths_arr, SED_results_arr)

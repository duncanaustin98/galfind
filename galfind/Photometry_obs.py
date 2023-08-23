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

from .Photometry import Photometry, Multiple_Photometry
from .SED_result import Galaxy_SED_results, Catalogue_SED_results

class Photometry_obs(Photometry):

    def __init__(self, instrument, flux_Jy, flux_Jy_errs, aper_diam, min_flux_pc_err, loc_depths, SED_results = {}):
        self.aper_diam = aper_diam
        self.min_flux_pc_err = min_flux_pc_err
        self.SED_results = SED_results # array of SED_result objects with different SED fitting runs
        super().__init__(instrument, flux_Jy, flux_Jy_errs, loc_depths)

    @property
    def flux_lambda(self): # wav and flux_nu must have units here!
        return (self.flux_Jy * const.c / ((np.array([self.instrument.band_wavelengths[band].value for band in self.instrument.bands]) * u.Angstrom) ** 2)).to(u.erg / (u.s * (u.cm ** 2) * u.Angstrom)) # both flux_nu and wav must be in the same rest or observed frame
    
    @property
    def flux_lambda_errs(self):
        return (self.flux_Jy_errs * const.c / ((np.array([self.instrument.band_wavelengths[band].value for band in self.instrument.bands]) * u.Angstrom) ** 2)).to(u.erg / (u.s * (u.cm ** 2) * u.Angstrom))

    @classmethod # not a gal object here, more like a catalogue row
    def from_fits_cat(cls, fits_cat_row, instrument, cat_creator, aper_diam, min_flux_pc_err, codes, lowz_zmaxs):
        phot = Photometry.from_fits_cat(fits_cat_row, instrument, cat_creator)
        #raise(Exception("Fix this!"))
        SED_results = Galaxy_SED_results.from_fits_cat(fits_cat_row, cat_creator, codes, lowz_zmaxs, instrument = instrument)
        return cls.from_phot(phot, aper_diam, min_flux_pc_err, SED_results)
    
    @classmethod
    def from_phot(cls, phot, aper_diam, min_flux_pc_err, SED_results = {}):
        return cls(phot.instrument, phot.flux_Jy, phot.flux_Jy_errs, aper_diam, min_flux_pc_err, phot.loc_depths, SED_results)
    
    def update(self, gal_SED_results):
        # gal_SED_results has type = dict(dict())
        # ensure the same code_names are keys in both self.SED_results and gal_SED_results
        for code_name, val in self.SED_results.items():
            if code_name not in gal_SED_results.keys():
                gal_SED_results[code_name] = {**{code_name: {}}, **gal_SED_results}
        for code_name, val in gal_SED_results.items():
            if code_name not in self.SED_results.keys():
                self.SED_results = {**{code_name: {}}, **self.SED_results}
        self.SED_results = {code_name: dict(**self.SED_results[code_name], **gal_SED_results[code_name]) for code_name in self.SED_results.keys()}
        #print("Post update:", self.SED_results)
    
    def load_local_depths(self, sex_cat_row, instrument, aper_diam_index):
        self.loc_depths = np.array([sex_cat_row[f"loc_depth_{band}"].T[aper_diam_index] for band in instrument.bands])
        
    def SNR_crop(self, band, sigma_detect_thresh):
        index = self.instrument.band_from_index(band)
        # local depth in units of Jy
        loc_depth_Jy = self.loc_depths[index].to(u.Jy) / 5
        detection_Jy = self.flux_Jy[index].to(u.Jy)
        sigma_detection = (detection_Jy / loc_depth_Jy).value
        if sigma_detection >= sigma_detect_thresh:
            return True
        else:
            return False

# %%    
        
class Multiple_Photometry_obs:
    
    def __init__(self, instrument, flux_Jy_arr, flux_Jy_errs_arr, aper_diam, min_flux_pc_err, loc_depths_arr, SED_results_arr = []):
        # force SED_results_arr to have the same len as the number of input fluxes
        if SED_results_arr == []:
            SED_results_arr = np.full(len(flux_Jy_arr), {})
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
    def from_fits_cat(cls, fits_cat, instrument, cat_creator, aper_diam, min_flux_pc_err, codes, lowz_zmaxs, templates_arr):
        flux_Jy_arr, flux_Jy_errs_arr = cat_creator.load_photometry(fits_cat, instrument.bands)
        # TO DO: MAKE MULTIPLE_SED_RESULTS CLASS TO LOAD IN SED_RESULTS SIMULTANEOUSLY (LEAVE AS [] FOR NOW)
        loc_depths_arr = np.full(len(flux_Jy_arr), None)
        SED_results_arr = Catalogue_SED_results.from_fits_cat(fits_cat, cat_creator, codes, lowz_zmaxs, templates_arr, instrument = instrument).SED_results
        # print("Photometry_obs SED_results = ", SED_results_arr)
        return cls(instrument, flux_Jy_arr, flux_Jy_errs_arr, aper_diam, min_flux_pc_err, loc_depths_arr, SED_results_arr)

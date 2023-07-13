#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 00:17:39 2023

@author: austind
"""

# SED_codes.py
# %% Imports

import os
import numpy as np
from abc import ABC, abstractmethod
import astropy.units as u
import itertools
from astropy.table import Table, join
import subprocess
from pathlib import Path
from astropy.io import fits

from . import useful_funcs_austind as funcs
from . import config
from. import Photometry_rest


# %% SED_code class

class SED_code(ABC):
    
    def __init__(self, code_name, ID_label, galaxy_property_labels, chi_sq_labels, low_z_run):
        self.code_name = code_name
        self.ID_label = ID_label
        self.galaxy_property_labels = galaxy_property_labels
        self.chi_sq_labels = chi_sq_labels
        #self.code_dir = f"{config['DEFAULT']['GALFIND_WORK']}/{code_name}"
        self.low_z_run = low_z_run
    
    @abstractmethod
    def from_name(self):
        pass
    
    def load_photometry(self, cat, SED_input_bands, out_units, no_data_val, upper_sigma_lim = {}):
        # load in raw photometry from the galaxies in the catalogue and convert to appropriate units
        phot = np.array([gal.phot_obs.flux_Jy.to(out_units) for gal in cat])
        phot_shape = phot.shape

        if out_units != u.ABmag:
            phot_err = np.array([gal.phot_obs.flux_Jy_errs.to(out_units) for gal in cat])
        else:
            # Not correct in general! Only for high S/N! Fails to scale mag errors asymetrically from flux errors
            phot_err = np.array([funcs.flux_pc_to_mag_err(gal.phot_obs.flux_Jy_errs / gal.phot_obs.flux_Jy) for gal in cat])
        
        # include upper limits if wanted
        if upper_sigma_lim != None and upper_sigma_lim != {}:
            # determine relevant indices
            upper_lim_indices = [[i, j] for i, gal in enumerate(cat) for j, depth in enumerate(gal.phot_obs.loc_depths) \
                                 if funcs.n_sigma_detection(depth, (phot[i][j] * out_units).to(u.ABmag).value + \
                                gal.phot_obs.instrument.aper_corr(gal.phot_obs.aper_diam, gal.phot_obs.instrument.bands[j]), u.Jy.to(u.ABmag)) < upper_sigma_lim["threshold"]]
            phot = np.array([funcs.five_to_n_sigma_mag(loc_depth, upper_sigma_lim["value"]) if [i, j] in upper_lim_indices else phot[i][j] \
                    for i, gal in enumerate(cat) for j, loc_depth in enumerate(gal.phot_obs.loc_depths)]).reshape(phot_shape)
            phot_err = np.array([-1.0 if [i, j] in upper_lim_indices else phot_err[i][j] \
                    for i, gal in enumerate(cat) for j, loc_depth in enumerate(gal.phot_obs.loc_depths)]).reshape(phot_shape)

        # insert 'no_data_val' from SED_input_bands with no data in the catalogue
        phot_in = []
        phot_err_in = []
        for band in SED_input_bands:
            if band in cat.data.instrument.bands:
                band_index = np.where(band == cat.data.instrument.bands)[0][0]
                phot_in.append(phot[:, band_index])
                phot_err_in.append(phot_err[:, band_index])
            else: # band does not exist in data but still needs to be included
                phot_in.append(np.full(phot_shape[0], no_data_val))
                phot_err_in.append(np.full(phot_shape[0], no_data_val))
        phot_in = np.array(phot_in).T
        phot_err_in = np.array(phot_err_in).T
        
        return phot_in, phot_err_in
    
    def fit_cat(self, cat, *args, **kwargs):
        print("Updated SED_code.fit_cat to trace SED_input_bands")
        in_path = self.make_in(cat, *args, **kwargs)
        out_folder = funcs.split_dir_name(in_path.replace("input", "output"), "dir")
        out_path = f"{out_folder}/{funcs.split_dir_name(in_path, 'name').replace('.in', '.out')}"
        sed_folder = f"{out_folder}/SEDs/{cat.cat_creator.min_flux_pc_err}pc"
        os.makedirs(sed_folder, exist_ok = True)
        fits_out_path = self.out_fits_name(out_path, *args, **kwargs)
        if not Path(fits_out_path).is_file() or config["DEFAULT"].getboolean("OVERWRITE"):
            self.run_fit(in_path, out_path, sed_folder, cat.data.instrument.new_instrument(), *args, **kwargs)
            self.make_fits_from_out(out_path, *args, **kwargs)
        # update galaxies within catalogue object with determined properties
        data = cat.data # work around of update_cat function
        cat = self.update_cat(cat, fits_out_path, *args, **kwargs)
        cat.data = data # work around of update_cat function
        return cat
    
    def update_cat(self, cat, fits_out_path, *args, **kwargs):
        # save concatenated catalogue
        combined_cat = join(Table.read(cat.cat_path), Table.read(fits_out_path), keys_left = "NUMBER", keys_right = "IDENT")
        combined_cat_path = f"{config['DEFAULT']['GALFIND_WORK']}/Catalogues/{cat.data.version}/{cat.data.instrument.name}/" + \
            f"{cat.data.survey}/{funcs.split_dir_name(fits_out_path.replace('.fits', '_matched.fits'), 'name')}"
        combined_cat.write(combined_cat_path, overwrite = True)
        # update 'Catalogue' object using Catalogue.__setattr__()
        codes = cat.codes + [self.from_name()]
        print("We need to make Catalogue.from_photo_z_cat() produce a data object within the catalogue object!")
        return cat.from_photo_z_cat(combined_cat_path, cat.data.instrument, cat.data.survey, cat.cat_creator, codes)
        
    @abstractmethod
    def make_in(self, cat):
        pass
    
    @abstractmethod
    def run_fit(self, in_path):
        pass
    
    @abstractmethod
    def make_fits_from_out(self, out_path):
        pass
    
    @abstractmethod
    def out_fits_name(self, out_path):
        pass
    
    @abstractmethod
    def extract_SEDs(self, cat, ID):
        pass
    
    def plot_best_fit_SED(self, ax, cat, ID, save = True, show = True):
        # extract best fitting SED
        wav, mag = self.extract_SED(cat, ID)
        # plot SED on ax
        pass
    
    @abstractmethod
    def extract_z_PDF(self, cat, ID):
        pass
    
    def plot_z_PDF(self, ax, cat, ID, save = True, show = True):
        # extract PDF
        z, PDF = self.extract_z_PDF(cat, ID)
        # plot PDF on ax
        pass
    
    @abstractmethod
    def z_PDF_path_from_cat_path(self, cat_path, ID, low_z_run = False):
        pass
    
    @abstractmethod
    def SED_path_from_cat_path(self, cat_path, ID, low_z_run = False):
        pass

class SED_result:
    
    def __init__(self, instrument, flux_Jy, flux_Jy_errs, loc_depths, z, code_name, chi_sqs = None, z_PDF_gal = None, SEDs = None, low_z_run = False):
        self.photometry_rest = Photometry_rest(instrument, flux_Jy, flux_Jy_errs, loc_depths, z, code_name)
        self.z = z
        self.chi_sqs = chi_sqs
        self.z_PDF_gal = z_PDF_gal
        self.SEDs = SEDs
        self.code_name
        self.low_z_run = low_z_run
        
    @classmethod
    def from_GALFIND(cls, code_name, phot, gal_ID, cat_path, low_z_run):
        code = SED_code.from_name(code_name)
        cat = funcs.cat_from_path(cat_path) # open catalogue
        cat = cat[cat[code.ID_label == gal_ID]]
        z = float(cat[code.galaxy_property_labels("z")])
        chi_sqs = {name: float(cat[chi_sq]) for name, chi_sq in code.chi_sq_labels.items()}
        z_PDF = code.extract_z_PDF(cat_path, gal_ID, low_z_run)
        SEDs = code.extract_SEDs(cat_path, gal_ID, low_z_run)
        return cls(code_name, phot.flux_Jy, phot.flux_errs, phot.instrument, phot.loc_depths, z, chi_sqs, z_PDF, SEDs, low_z_run)

# LePhare
LePhare_outputs = {"z": "Z_BEST", "mass": "MASS_BEST"}
LePhare_col_names = LePhare_outputs.values()

def calc_LePhare_errs(cat, col_name):
    if col_name == "Z_BEST":
        data = np.array(cat[col_name])
        data_err = np.array([np.array(cat[col_name + "68_LOW"]), np.array(cat[col_name + "68_HIGH"])])
        data, data_err = adjust_errs(data, data_err)
        return data_err
    
# EAZY
EAZY_outputs = {"z": "zbest"}
EAZY_col_names = EAZY_outputs.values()

def calc_EAZY_errs(cat, col_name):
    if col_name == "zbest":
        pass
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
import json
from abc import ABC, abstractmethod
import astropy.units as u
import itertools
from astropy.table import Table, join
import subprocess
from pathlib import Path
from astropy.io import fits
from tqdm import tqdm

from . import useful_funcs_austind as funcs
from . import config, galfind_logger
from . import SED_result, Catalogue_SED_results

# %% SED_code class

class SED_code(ABC):
    
    def __init__(self, galaxy_property_dict, galaxy_property_errs_dict, available_templates):
        self.galaxy_property_dict = galaxy_property_dict
        self.galaxy_property_errs_dict = galaxy_property_errs_dict
        self.available_templates = available_templates
    
    def load_photometry(self, cat, SED_input_bands, out_units, no_data_val, upper_sigma_lim = {}):
        # load in raw photometry from the galaxies in the catalogue and convert to appropriate units
        phot = np.array([gal.phot.flux_Jy.to(out_units) for gal in cat]) #[:, :, 0]
        phot_shape = phot.shape

        if out_units != u.ABmag:
            phot_err = np.array([gal.phot.flux_Jy_errs.to(out_units) for gal in cat])#[:, :, 0]
        else:
            # Not correct in general! Only for high S/N! Fails to scale mag errors asymetrically from flux errors
            phot_err = np.array([funcs.flux_pc_to_mag_err(gal.phot.flux_Jy_errs / gal.phot.flux_Jy) for gal in cat])#[:, :, 0]

        # include upper limits if wanted
        if upper_sigma_lim != None and upper_sigma_lim != {}:
            # determine relevant indices
            upper_lim_indices = [[i, j] for i, gal in enumerate(cat) for j, depth in enumerate(gal.phot[0].loc_depths) \
                                 if funcs.n_sigma_detection(depth, (phot[i][j] * out_units).to(u.ABmag).value + \
                                gal.phot.instrument.aper_corr(gal.phot.aper_diam, gal.phot.instrument.band_names[j]), u.Jy.to(u.ABmag)) < upper_sigma_lim["threshold"]]
            phot = np.array([funcs.five_to_n_sigma_mag(loc_depth, upper_sigma_lim["value"]) if [i, j] in upper_lim_indices else phot[i][j] \
                    for i, gal in enumerate(cat) for j, loc_depth in enumerate(gal.phot.loc_depths)]).reshape(phot_shape)
            phot_err = np.array([-1.0 if [i, j] in upper_lim_indices else phot_err[i][j] \
                    for i, gal in enumerate(cat) for j, loc_depth in enumerate(gal.phot.loc_depths)]).reshape(phot_shape)

        # insert 'no_data_val' from SED_input_bands with no data in the catalogue
        phot_in = []
        phot_err_in = []
        for band in SED_input_bands:
            if band in cat.instrument.band_names:
                band_index = np.where(band == cat.instrument.band_names)[0][0]
                phot_in.append(phot[:, band_index])
                phot_err_in.append(phot_err[:, band_index])
            else: # band does not exist in data but still needs to be included
                phot_in.append(np.full(phot_shape[0], no_data_val))
                phot_err_in.append(np.full(phot_shape[0], no_data_val))
        phot_in = np.array(phot_in).T
        phot_err_in = np.array(phot_err_in).T
        
        return phot_in, phot_err_in
    
    def fit_cat(self, cat, templates, lowz_zmax_arr): # *args, **kwargs):
        assert(templates in self.available_templates)
        in_path = self.make_in(cat) #, *args, **kwargs)
        #print(in_path)
        out_folder = funcs.split_dir_name(in_path.replace("input", "output"), "dir")
        out_path = f"{out_folder}/{funcs.split_dir_name(in_path, 'name').replace('.in', '.out')}"
        #print(out_path)
        overwrite = config[self.__class__.__name__].getboolean(f"OVERWRITE_{self.__class__.__name__}")
        tab = Table.read(cat.cat_path, memmap = True)
        for lowz_zmax in lowz_zmax_arr:
            fits_out_path = self.out_fits_name(out_path, templates, lowz_zmax)
            # run the SED fitting if not already done so or if wanted overwriting
            if f"RUN_{self.__class__.__name__}" not in tab.meta.keys() or overwrite:
                #if config[self.__class__.__name__].getboolean(f"RUN_{self.__class__.__name__}"):
                self.run_fit(in_path, fits_out_path, cat.instrument.new_instrument(), templates = templates, lowz_zmax = lowz_zmax, overwrite = overwrite) #, *args, **kwargs)
                self.make_fits_from_out(out_path, templates, lowz_zmax) #, *args, **kwargs)
            # update galaxies within catalogue object with determined properties
            self.update_fits_cat(cat, fits_out_path, templates = templates, lowz_zmax = lowz_zmax) #, *args, **kwargs)
            # update catalogue with paths to 
        # update galaxies within the catalogue with new SED fits
        cat_SED_results = Catalogue_SED_results.from_fits_cat(cat.open_cat(), cat.cat_creator, \
            [self], [templates], lowz_zmax_arr, phot_arr = [gal.phot for gal in cat], fits_cat_path = cat.cat_path).SED_results
        cat.update_SED_results(cat_SED_results)
        return cat
    
    def update_fits_cat(self, cat, fits_out_path, templates, lowz_zmax): #*args, **kwargs):
        # open original catalogue
        orig_cat = cat.open_cat()
        # combine catalogues if not already run before
        if self.galaxy_property_labels("z_phot", templates, lowz_zmax) in orig_cat.colnames:
            combined_cat = orig_cat
        else:
            combined_cat = join(orig_cat, Table.read(fits_out_path), keys_left = "NUMBER", keys_right = "IDENT")
            combined_cat.remove_column("IDENT")
            combined_cat.meta = {**combined_cat.meta, **{f"RUN_{self.__class__.__name__}": True}}
            combined_cat.write(cat.cat_path, overwrite = True)

    @abstractmethod
    @staticmethod
    def label_from_SED_fit_params(self, SED_fit_params):
        pass

    @abstractmethod
    def galaxy_property_labels(self, gal_property, SED_fit_params):
        pass

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
    def get_z_PDF_path(self, cat_path, ID, low_z_run = False):
        pass
    
    @abstractmethod
    def get_SED_path(self, cat_path, ID, low_z_run = False):
        pass

def calc_LePhare_errs(cat, col_name):
    if col_name == "Z_BEST":
        data = np.array(cat[col_name])
        data_err = np.array([np.array(cat[col_name + "68_LOW"]), np.array(cat[col_name + "68_HIGH"])])
        data, data_err = adjust_errs(data, data_err)
        return data_err
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
from tqdm import tqdm

from . import useful_funcs_austind as funcs
from . import config
from . import SED_result

# %% SED_code class

class SED_code(ABC):
    
    def __init__(self, code_name, galaxy_property_dict, available_templates):
        self.code_name = code_name
        self.galaxy_property_dict = galaxy_property_dict
        self.available_templates = available_templates
        
    @staticmethod
    def galaxy_property_labels(gal_property, templates, lowz_zmax = None):
        if templates not in ["fsps", "fsps_larson", "fsps_jades"]:
            raise(Exception(f"templates = {templates} are not yet encorporated for galfind EAZY SED fitting"))
        if lowz_zmax != None:
            lowz_suffix = f"_zmax={lowz_zmax:.1f}"
        else:
            lowz_suffix = ""
        if gal_property == "z_phot":
            return f"zbest{lowz_suffix}_{templates}"
        elif gal_property == "chi2":
            return f"chi2_best{lowz_suffix}_{templates}"
        else:
            raise(Exception(f"EAZY.galaxy_property_labels does not include option for gal_property = {gal_property}!"))
    
    def load_photometry(self, cat, SED_input_bands, out_units, no_data_val, upper_sigma_lim = {}):
        # load in raw photometry from the galaxies in the catalogue and convert to appropriate units
        phot = np.array([gal.phot.flux_Jy.to(out_units) for gal in cat])#[:, :, 0]
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
                                gal.phot.instrument.aper_corr(gal.phot.aper_diam, gal.phot.instrument.bands[j]), u.Jy.to(u.ABmag)) < upper_sigma_lim["threshold"]]
            phot = np.array([funcs.five_to_n_sigma_mag(loc_depth, upper_sigma_lim["value"]) if [i, j] in upper_lim_indices else phot[i][j] \
                    for i, gal in enumerate(cat) for j, loc_depth in enumerate(gal.phot.loc_depths)]).reshape(phot_shape)
            phot_err = np.array([-1.0 if [i, j] in upper_lim_indices else phot_err[i][j] \
                    for i, gal in enumerate(cat) for j, loc_depth in enumerate(gal.phot.loc_depths)]).reshape(phot_shape)

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
    
    def fit_cat(self, cat, z_max_lowz, *args, **kwargs):
        in_path = self.make_in(cat, *args, **kwargs)
        out_folder = funcs.split_dir_name(in_path.replace("input", "output"), "dir")
        out_path = f"{out_folder}/{funcs.split_dir_name(in_path, 'name').replace('.in', '.out')}"
        sed_folder = f"{out_folder}/SEDs/{cat.cat_creator.min_flux_pc_err}pc"
        os.makedirs(sed_folder, exist_ok = True)
        fits_out_path = self.out_fits_name(out_path, *args, **kwargs)
        #print(f"fit_cat fits_out_path = {fits_out_path}")
        if not Path(fits_out_path).is_file() or config["DEFAULT"].getboolean("OVERWRITE"):
            print(f"Running SED fitting for {self.__class__.__name__}")
            self.run_fit(in_path, out_path, sed_folder, cat.data.instrument.new_instrument(), z_max_lowz = z_max_lowz, *args, **kwargs)
            self.make_fits_from_out(out_path, *args, **kwargs)
        # update galaxies within catalogue object with determined properties
        cat = self.update_cat(cat, fits_out_path, z_max_lowz, *args, **kwargs)
        return cat
    
    def update_cat(self, cat, fits_out_path, low_z_run, *args, **kwargs):
        # save concatenated catalogue
        #print(f"update_cat fits_out_path = {fits_out_path}")
        combined_cat = join(Table.read(cat.cat_path), Table.read(fits_out_path), keys_left = "NUMBER", keys_right = "IDENT")
        combined_cat_path = f"{config['DEFAULT']['GALFIND_WORK']}/Catalogues/{cat.data.version}/{cat.data.instrument.name}/" + \
            f"{cat.data.survey}/{funcs.split_dir_name(fits_out_path.replace('_matched', '').replace('_EAZY', '').replace('.fits', '_matched.fits'), 'name')}"
        if self.__class__.__name__ == "EAZY":
            templates = kwargs.get("templates")
            combined_cat_path = combined_cat_path.replace(f"_eazy_{templates}", "_EAZY")
            #print("EAZY naming system requires updating")
        combined_cat.remove_column("IDENT")
        combined_cat.write(combined_cat_path, overwrite = True)
        combined_cat.meta["cat_path"] = combined_cat_path
        combined_cat.meta = {**combined_cat.meta, **{f"{self.code_name}_path": fits_out_path}}
        # update galaxies within the catalogue with new SED fits
        cat.cat_path = combined_cat_path
        SED_results = [] #[SED_result.from_fits_cat(combined_cat[combined_cat["NUMBER"] == gal.ID], self.__class__(), gal.phot[0], cat.cat_creator, low_z_run) \
                       #for gal in tqdm(cat, total = len(cat), desc = "Loading SED results into catalogue")]
        cat.update_SED_results(SED_results)
        return cat #.from_fits_cat(combined_cat_path, cat.data.instrument, cat.cat_creator, codes, low_z_runs, cat.data.survey)
        
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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 21:19:13 2023

@author: austind
"""

# useful_funcs_austind.py
import sys
import warnings
import sep
import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.table import Table, join
from scipy.optimize import curve_fit
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
from abc import ABC, abstractmethod
from copy import copy, deepcopy
import json
from tqdm import tqdm
import seaborn as sns
from scipy.interpolate import interp1d

from . import config
from . import astropy_cosmo

# fluxes and magnitudes
def calc_flux_from_ra_dec(ra, dec, im_data, wcs, r, unit = "deg"):
    x_pix, y_pix = skycoord_to_pixel(SkyCoord(ra, dec, unit = unit), wcs)
    flux, fluxerr, flag = sep.sum_circle(im_data, x_pix, y_pix, r)
    return flux # image units
    
def calc_1sigma_flux(depth, zero_point):
    flux_1sigma = (10 ** ((depth - zero_point) / -2.5)) / 5
    return flux_1sigma # image units

def n_sigma_detection(depth, mag, zero_point): # mag here is non aperture corrected
    flux_1sigma = calc_1sigma_flux(depth, zero_point)
    flux = 10 ** ((mag - zero_point) / -2.5)
    return flux / flux_1sigma

def flux_to_mag(flux, zero_point):
    try:
        flux = flux.value
    except:
        pass
    mag = -2.5 * np.log10(flux) + zero_point
    return mag

def mag_to_flux(mag, zero_point):
    flux = 10 ** ((mag - zero_point) / -2.5)
    return flux

def flux_to_mag_ratio(flux_ratio):
    mag_ratio = -2.5 * np.log10(flux_ratio)
    return mag_ratio

def flux_pc_to_mag_err(flux_pc_err):
    mag_err = 2.5 * flux_pc_err / (np.log(10)) # divide by 100 here to convert into percentage?
    return mag_err

def flux_image_to_Jy(flux, zero_point):
    # convert flux from image units to Jy
    return flux * (10 ** ((zero_point - 8.9) / -2.5)) * u.Jy

def five_to_n_sigma_mag(five_sigma_depth, n):
    n_sigma_mag = -2.5 * np.log10(n / 5) + five_sigma_depth
    #flux_sigma = (10 ** ((five_sigma_depth - zero_point) / -2.5)) / 5
    #n_sigma_mag = -2.5 * np.log10(flux_sigma * n) + zero_point
    return n_sigma_mag

def flux_err_to_loc_depth(flux_err, zero_point):
    return -2.5 * np.log10(flux_err * 5) + zero_point

# now in Photometry class!
# def flux_image_to_lambda(wav, flux, zero_point):
#     flux_Jy = flux_image_to_Jy(flux, zero_point)
#     flux_lambda = flux_Jy_to_lambda(wav, flux_Jy)
#     return flux_lambda # observed frame

def flux_Jy_to_lambda(flux_Jy, wav): # must akready have associated astropy units
    return (flux_Jy * const.c / (wav ** 2)).to(u.erg / (u.s * (u.cm ** 2) * u.Angstrom))

def wav_obs_to_rest(wav_obs, z):
    wav_rest = wav_obs / (1 + z)
    return wav_rest

def flux_lambda_obs_to_rest(flux_lambda_obs, z):
    flux_lambda_rest = flux_lambda_obs * ((1 + np.full(len(flux_lambda_obs), z)) ** 2)
    return flux_lambda_rest

# general functions

def adjust_errs(data, data_err):
    #print("adjusting errors:", plot_data, code)
    data_l1 = data - data_err[0]
    data_u1 = data_err[1] - data
    data_err = np.vstack([data_l1, data_u1])
    return data, data_err

def errs_to_log(data, data_err):
    log_l1 = np.log10(data) - np.log10(data - data_err[0])
    log_u1 = np.log10(data + data_err[1]) - np.log10(data)
    return np.log10(data), [log_l1, log_u1]

def PDF_hist(PDF, save_dir, obs_name, ID, show = True, save = True):
    if not all(value == -99. for value in PDF):
        plt.hist(PDF, label = ID)
        #print(f"Plotting {obs_name} hist for {ID}")
        plt.xlabel(obs_name)
        if show:
            plt.legend()
            if save:
                path = f"{split_dir_name(PDF_path(save_dir, obs_name, ID), 'dir')}/hist/{ID}.png"
                make_dirs(path)
                #print(f"Saving hist: {path}")
                plt.savefig(path)
                plt.clf()
            else:
                plt.show()
        
def split_dir_name(save_path, output):
    if output == "dir":
        return "/".join(np.array(save_path.split("/")[:-1])) + "/"
    elif output == "name":
        return save_path.split("/")[-1]
    
def save_PDF(PDF, header, path):
    if not all(value == -99. for value in PDF):
        path = f"{path}.txt"
        make_dirs(path)
        #print(f"Saving PDF: {path}")
        np.savetxt(path, PDF, header = header)

def PDF_path(save_dir, obs_name, ID):
    return f"{save_dir}/{obs_name}/{ID}"

def percentiles_from_PDF(PDF):
    try:
        PDF = np.array([val.value for val in PDF.copy()]) # remove the units
    except:
        pass
    PDF_median = np.median(PDF)
    PDF_l1 = PDF_median - np.percentile(PDF, 16)
    PDF_u1 = np.percentile(PDF, 84) - PDF_median
    return PDF_median, PDF_l1, PDF_u1

def gauss_func(x, mu, sigma):
    return (np.pi * sigma) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
def power_law_func(x, A, slope):
    return A * (x ** slope)

# Simulations
class Simulation(ABC):
    
    def __init__(self, sim_name, base_cat_path, survey_depth_name, flux_zero_point):
        self.sim_name = sim_name
        self.base_cat_path = base_cat_path
        self.survey_depth_name = survey_depth_name
        self.full_cat_path = self.make_err_cat()
        
    def make_err_cat(self):
        # open base catalogue
        base_cat = Catalogue.cat_from_path(self.base_cat_path)
        
        # load depths from specific survey
        depths = Catalogue.load_depths(self.survey_depth_name)

    
    @abstractmethod
    def flux_col_name(self, band):
        pass
    
    def flux_err_name(self, band):
        return f"{self.flux_col_name(band)}_err"

class Jaguar(Simulation):
    
    def __init__(self, survey_depth_name):
        super().__init__("Jaguar", "/nvme/scratch/work/tharvey/lightcone_models/jaguar/JADES_SF_mock_r1_v1.2.fits", survey_depth_name, 31.4)
        
    def flux_col_name(self, band):
        if band in NIRCam().bands:
            return f"NRC_{band.replace('f', 'F')}_fnu"

# SExtractor photometry



def make_dirs(path):
    os.makedirs(split_dir_name(path, "dir"), exist_ok = True)

def calc_errs_from_cat(cat, col_name, instrument):
    if col_name in LePhare_col_names:
        errs = calc_LePhare_errs(cat, col_name)
    elif col_name in EAZY_col_names:
        errs = calc_EAZY_errs(cat, col_name)
    elif col_name in instrument.bands:
        errs = [return_loc_depth_mags(cat, col_name, True)[1]]
    else:
        errs = np.array([cat[f"{col_name}_l1"], cat[f"{col_name}_u1"]])
    return errs

def source_separation(sky_coord_1, sky_coord_2, z):
    # calculate separation in arcmin
    arcmin_sep = sky_coord_1.separation(sky_coord_2).to(u.arcmin)
    #print(arcmin_sep.to(u.arcsec))
    # calculate separation in transverse comoving distance
    kpc_sep = arcmin_sep * astropy_cosmo.kpc_proper_per_arcmin(z)
    return kpc_sep

#source_separation(SkyCoord(ra = 53.26564 * u.deg, dec = -27.85555 * u.deg), SkyCoord(ra = 53.26556 * u.deg, dec = -27.85552 * u.deg), 8.0)

def tex_to_fits(tex_path, col_names, col_errs, replace = {"&": "", "\\\\": "", "\dag": "", "\ddag": "", "\S": "", \
                                            "\P": "", "$": "", "}": "", "^{+": " ", "^{": "", "_{-": " "}, empty = ["-"], comment = "%"):
    # note which columns are error columns
    is_err = col_errs.copy()
    for i in col_errs:
        if i:
            is_err[i] = False
            is_err[i:i] = np.full(2, True)
    save_data = []
    # read tex table line by line
    with open(tex_path, "r") as tab:
        line_no = 0
        while True:
            line = tab.readline()

            if not line:
                break
            
            if not line.startswith(comment): # ignore comments in the table
                line_no += 1
                # format the line into something .txt readable
                for i, (key, val) in enumerate(replace.items()):
                    line = line.replace(key, val)
                # turn each line into an array
                line_elements = line.split()
                # insert nans where there is not the appropriate data
                while True:
                    if len(line_elements) == len(is_err):
                        break
                    for i, val in enumerate(line_elements):
                        if val in empty:
                            line_elements[i] = np.nan
                            if is_err[i]:
                                line_elements[i:i] = np.full(2, np.nan)
                            break
                # append the data
                if line_no == 1:
                    save_data = line_elements
                else:
                    save_data = np.vstack([save_data, line_elements]) 
        print(save_data)
        tab.close()
    # adjust column names to include errors where appropriate
    cat_col_names = []
    for i, name in enumerate(col_names):
        cat_col_names.append(name)
        if col_errs[i]:
            cat_col_names.append(f"{name}_u1")
            cat_col_names.append(f"{name}_l1")
    cat_dtypes = np.array(np.full(len(cat_col_names), float))
    cat_dtypes[0] = str # not general
    cat_dtypes[-1] = str # not general
    fits_table = Table(save_data, names = cat_col_names, dtype = cat_dtypes)
    fits_path = tex_path.replace(".txt", "_as_fits.fits")
    fits_table.write(fits_path, overwrite = True)
    print(f"Saved {tex_path} as .fits")

#col_names = ["NAME", "RA", "DEC", "MAG_f444W", "MAG_f277W", "z_LePhare", "mass_LePhare", "Beta", "SFR", "M_UV", "References"]
#col_errs = [False, False, False, True, True, True, False, True, True, True, False]
#tex_to_fits("/nvme/scratch/work/austind/Arxiv_papers/matched_cats/HUDF-Par2/NGDEEP_paper_literature_tex.txt", col_names, col_errs)
        

        
def ext_source_corr(data, corr_factor, is_log_data = True):
    if is_log_data:
        return data + np.log10(corr_factor)
    else:
        return data * corr_factor
    
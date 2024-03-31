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
import inspect

from . import config
from . import astropy_cosmo

# fluxes and magnitudes

def convert_wav_units(wavs, out_units):
    return wavs.to(out_units)

def convert_mag_units(wavs, mags, units):
    if units == mags.unit:
        pass
    elif units == u.ABmag:
        if u.get_physical_type(mags.unit) in ["ABmag/spectral flux density", "spectral flux density"]: # f_ν -> derivative of u.Jy
            mags = mags.to(u.ABmag)
        elif u.get_physical_type(mags.unit) == "power density/spectral flux density wav": # f_λ -> derivative of u.erg / (u.s * (u.cm ** 2) * u.AA)
            mags = mags.to(u.ABmag, equivalencies = u.spectral_density(wavs))
    elif u.get_physical_type(units) in ["ABmag/spectral flux density", "spectral flux density"]: # f_ν -> derivative of u.Jy
        if mags.unit == u.ABmag:
            mags = mags.to(units)
        elif u.get_physical_type(mags.unit) == "power density/spectral flux density wav" or u.get_physical_type(mags.unit) == "ABmag/spectral flux density":
            mags = mags.to(units, equivalencies = u.spectral_density(wavs))
    
    elif u.get_physical_type(units) == "power density/spectral flux density wav": # f_λ -> derivative of u.erg / (u.s * (u.cm ** 2) * u.AA):
        mags = mags.to(units, equivalencies = u.spectral_density(wavs))
    else:
        raise(Exception("Units must be either ABmag or have physical units of 'spectral flux density' or 'power density/spectral flux density wav'!"))
    return mags

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

def flux_image_to_Jy(fluxes, zero_points):
    # convert flux from image units to Jy
    if type(fluxes) in [list, np.array]:
        return np.array([flux * (10 ** ((zero_points - 8.9) / -2.5)) for flux in fluxes]) * u.Jy
    else:
        return np.array(fluxes * (10 ** ((zero_points - 8.9) / -2.5))) * u.Jy

def five_to_n_sigma_mag(five_sigma_depth, n):
    n_sigma_mag = -2.5 * np.log10(n / 5) + five_sigma_depth
    #flux_sigma = (10 ** ((five_sigma_depth - zero_point) / -2.5)) / 5
    #n_sigma_mag = -2.5 * np.log10(flux_sigma * n) + zero_point
    return n_sigma_mag

def flux_err_to_loc_depth(flux_err, zero_point):
    return -2.5 * np.log10(flux_err * 5) + zero_point

def loc_depth_to_flux_err(loc_depth, zero_point):
    return (10 ** ((loc_depth - zero_point) /-2.5)) / 5

# now in Photometry class!
# def flux_image_to_lambda(wav, flux, zero_point):
#     flux_Jy = flux_image_to_Jy(flux, zero_point)
#     flux_lambda = flux_Jy_to_lambda(wav, flux_Jy)
#     return flux_lambda # observed frame

def flux_Jy_to_lambda(flux_Jy, wav): # must already have associated astropy units
    return (flux_Jy * const.c / (wav ** 2)).to(u.erg / (u.s * (u.cm ** 2) * u.Angstrom))

def flux_lambda_to_Jy(flux_lambda, wav):
    return (flux_lambda * (wav ** 2) / const.c).to(u.Jy)

def lum_nu_to_lum_lam(lum_nu, wav):
    return lum_nu * const.c / (wav ** 2)

def lum_lam_to_lum_nu(lum_wav, wav):
    return lum_wav * (wav ** 2) / const.c

def wav_obs_to_rest(wav_obs, z):
    wav_rest = wav_obs / (1 + z)
    return wav_rest

def wav_rest_to_obs(wav_rest, z):
    wav_obs = wav_rest * (1 + z)
    return wav_obs

def flux_lambda_obs_to_rest(flux_lambda_obs, z):
    flux_lambda_rest = flux_lambda_obs * ((1 + np.full(len(flux_lambda_obs), z)) ** 2)
    return flux_lambda_rest

def luminosity_to_flux(lum, wavs, z = None, cosmo = astropy_cosmo, out_units = u.Jy):
    # calculate luminosity distance
    if z == None:
        lum_distance = 10 * u.pc
        z = 0.
    else:
        lum_distance = cosmo.luminosity_distance(z)
    # sort out the units
    if u.get_physical_type(lum.unit) == "yank": # i.e. L_λ, Lsun / AA or equivalent
        if u.get_physical_type(out_units) == "spectral flux density": # f_ν
            return (lum_lam_to_lum_nu(lum, wavs) * (1 + z) / (4 * np.pi * lum_distance ** 2)).to(out_units)
        elif u.get_physical_type(out_units) == "power density/spectral flux density wav": # f_λ
            return (lum * (1 + z) / (4 * np.pi * lum_distance ** 2)).to(out_units)
        else:
            raise(Exception(""))
    elif u.get_physical_type(lum.unit) == "energy/torque/work": # i.e L_ν, Lsun / Hz or equivalent
        if u.get_physical_type(out_units) == "spectral flux density": # f_ν
            return (lum * (1 + z) / (4 * np.pi * lum_distance ** 2)).to(out_units)
        elif u.get_physical_type(out_units) == "power density/spectral flux density wav": # f_λ
            return (lum_nu_to_lum_lam(lum, wavs) * (1 + z) / (4 * np.pi * lum_distance ** 2)).to(out_units)
        else:
            raise(Exception(""))

# Calzetti 1994 filters
lower_Calzetti_filt = [1268., 1309., 1342., 1407., 1562., 1677., 1760., 1866., 1930., 2400.]
upper_Calzetti_filt = [1284., 1316., 1371., 1515., 1583., 1740., 1833., 1890., 1950., 2580.]

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

def PDF_hist(PDF, save_dir, obs_name, ID, show = True, save = True, rest_UV_wavs = [1250., 3000.], conv_filt = False):
    if not all(value == -99. for value in PDF):
        plt.hist(PDF, label = ID)
        #print(f"Plotting {obs_name} hist for {ID}")
        plt.xlabel(obs_name)
        if show:
            plt.legend()
            if save:
                path = f"{split_dir_name(PDF_path(save_dir, obs_name, ID, rest_UV_wavs, conv_filt = conv_filt), 'dir')}/hist/{ID}.png"
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

def PDF_path(save_dir, obs_name, ID, rest_UV_wavs, conv_filt):
    return f"{save_dir}/{obs_name}/{ID}"

def percentiles_from_PDF(PDF):
    if all(val == -99. for val in PDF):
        return -99., -99., -99.
    else:
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

def cat_from_path(path, crop_names = None):
    cat = Table.read(path, character_as_bytes = False)
    if crop_names != None:
        for name in crop_names:
            cat = cat[cat[name] == True]
    # include catalogue metadata
    cat.meta = {**cat.meta, **{"cat_path": path}}
    return cat

def fits_cat_to_np(fits_cat, column_labels, reshape_by_aper_diams = True):
    new_cat = fits_cat[column_labels].as_array()
    if type(new_cat) == np.ma.core.MaskedArray:
        new_cat = new_cat.data
    if reshape_by_aper_diams:
        n_aper_diams = len(new_cat[0][0])
        new_cat = np.lib.recfunctions.structured_to_unstructured(new_cat).reshape(len(fits_cat), len(column_labels), n_aper_diams)
    else:
        new_cat = np.lib.recfunctions.structured_to_unstructured(new_cat).reshape(len(fits_cat), len(column_labels))
    return new_cat

def lowz_label(lowz_zmax):
    if lowz_zmax != None:
        label = f"zmax={lowz_zmax:.1f}"
    else:
        label = "zfree"
    return label

def get_z_PDF_paths(fits_cat, IDs, codes, templates_arr, lowz_zmaxs, fits_cat_path = None):
    try:
        fits_cat_path = fits_cat.meta["cat_path"]
    except:
        pass
    return [code.z_PDF_paths_from_cat_path(fits_cat_path, ID, templates, lowz_label(lowz_zmax)) for code, templates, lowz_zmax in \
            zip(codes, templates_arr, lowz_zmaxs) for ID in IDs]

def get_SED_paths(fits_cat, IDs, codes, templates_arr, lowz_zmaxs, fits_cat_path = None):
    try:
        fits_cat_path = fits_cat.meta["cat_path"]
    except:
        pass
    return [code.SED_paths_from_cat_path(fits_cat_path, ID, templates, lowz_label(lowz_zmax)) for code, templates, lowz_zmax in \
            zip(codes, templates_arr, lowz_zmaxs) for ID in IDs]

# beta slope function
def beta_slope_power_law_func(wav_rest, A, beta):
    return (10 ** A) * (wav_rest ** beta)

# GALFIND specific functions
def GALFIND_SED_column_labels(codes, lowz_zmaxs, templates_arr, gal_property):
    return [code.galaxy_property_labels(gal_property, templates, lowz_zmax) for code, templates in zip(codes, templates_arr) for lowz_zmax in lowz_zmaxs]

def GALFIND_cat_path(SED_code_name, instrument_name, version, survey, forced_phot_band_name, min_flux_pc_err, cat_type = "loc_depth", masked = True, templates = "fsps_larson"):
    # should still include aper_diam here
    if masked:
        masked_name = "_masked"
    else:
        masked_name = ""
    if SED_code_name == "EAZY":
        SED_code_name = f"eazy_{templates}"
    cat_dir = f"{config['DEFAULT']['GALFIND_WORK']}{SED_code_name}/output/{instrument_name}/{version}/{survey}"
    cat_name = f"{survey}_MASTER_Sel-{forced_phot_band_name}_{version}_{cat_type}{masked_name}_{str(min_flux_pc_err)}pc_{SED_code_name}.fits"
    return f"{cat_dir}/{cat_name}"

def GALFIND_final_cat_path(survey, version, instrument_name, sel_band_name, min_pc_err, SED_code_name, masked = True):
    cat_dir = f"{config['DEFAULT']['GALFIND_WORK']}/Catalogues/{version}/{instrument_name}/{survey}"
    if masked:
        cat_name = f"{survey}_MASTER_Sel-{sel_band_name}_{version}_loc_depth_masked_{str(min_pc_err)}pc_{SED_code_name}_matched_selection.fits"
    else:
        raise(Exception("masked = False currently not implemented!"))
    return f"{cat_dir}/{cat_name}"

def inspect_info():
  info = inspect.getframeinfo(inspect.stack()[1][0])
  return info.filename, info.function, info.lineno
  
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
    
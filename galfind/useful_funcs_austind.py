#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 21:19:13 2023

@author: austind
"""

# useful_funcs_austind.py
import sys
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

from . import config

# set cosmology
astropy_cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3, Ob0 = 0.05, Tcmb0=2.725)

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
    plt.hist(PDF, label = ID)
    print(f"Plotting {obs_name} hist for {ID}")
    plt.xlabel(obs_name)
    if show:
        plt.legend()
        if save:
            path = f"{split_dir_name(PDF_path(save_dir, obs_name, ID), 'dir')}/hist/{ID}.png"
            make_dirs(path)
            print(f"Saving hist: {path}")
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
    path = f"{path}.txt"
    make_dirs(path)
    print(f"Saving PDF: {path}")
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

# each "Photometry_obs" should have an "Instrument" object inside it (e.g. NIRCam/MIRI/HST_ACS/HST_WFC3IR)
class Photometry_obs:

    def __init__(self, instrument, flux_Jy, flux_Jy_errs, aper_diam):
        self.instrument = instrument
        self.flux_Jy = flux_Jy
        self.flux_Jy_errs = flux_Jy_errs
        self.aper_diam = aper_diam
    
    @property
    def flux_lambda(self): # wav and flux_nu must have units here!
        return (self.flux_Jy * const.c / (self.instrument.wav ** 2)).to(u.erg / (u.s * (u.cm ** 2) * u.Angstrom)) # both flux_nu and wav must be in the same rest or observed frame
    
    @property
    def flux_lambda_errs(self):
        return (self.flux_Jy_errs * const.c / (self.instrument.wav ** 2)).to(u.erg / (u.s * (u.cm ** 2) * u.Angstrom))

    @classmethod # not a gal object here, more like a catalogue row
    def get_phot_from_sex(cls, sex_cat_row, instrument, cat_creator): # single unit of sextractor catalogue
        # below is contained within cat_creator
        #aper_diam_index = int(json.loads(config.get("SExtractor", "APERTURE_DIAMS")).index(cat_creator.aper_diam.value))
        fluxes = []
        flux_errs = []
        #print("instrument = ", instrument)
        # Copy constructor problem here!
        instrument_copy = instrument.new_instrument(excl_bands = [band for band in instrument.from_name(instrument.name).bands if band not in instrument.bands]) # Problem loading the photometry properly here!!!
        #print("instrument_copy = ", instrument_copy)
        for (band, zero_point) in zip(instrument_copy.bands, instrument_copy.zero_points.values()):
            try:
                flux, err = cat_creator.load_photometry(sex_cat_row, band)
                fluxes.append(flux.value)
                flux_errs.append(err.value)
            except:
                # no data for the relevant band within the catalogue
                instrument.remove_band(band)
                print(f"{band} flux not loaded")
        phot_obs = cls(instrument, fluxes * u.Jy, flux_errs * u.Jy, cat_creator.aper_diam)
        phot_obs.load_local_depths(sex_cat_row, instrument, cat_creator.aper_diam_index)
        return phot_obs
        
    @classmethod
    def get_phot_from_sim(cls, gal, instrument, sim, min_flux_err_pc = 5):
        fluxes = []
        flux_errs = []
        instrument_copy = instrument.copy()
        for band in instrument_copy.bands:
            try:
                flux = np.array(gal[sim.flux_col_name(band)])
                err = np.array(gal[sim.flux_err_name(band)])
                # encorporate minimum flux error
                err = np.array([err_band if err_band / flux_band >= min_flux_err_pc / 100 else \
                                min_flux_err_pc * flux_band / 100 for flux_band, err_band in zip(flux, err)])
                flux_Jy = flux_image_to_Jy(flux, sim.zero_point)
                err_Jy = flux_image_to_Jy(err, sim.zero_point)
                fluxes = np.append(fluxes, flux_Jy.value)
                flux_errs = np.append(flux_errs, err_Jy.value)
            except:
                instrument.remove_band(band)
                print(f"{band} flux not loaded")
    
    def load_local_depths(self, sex_cat_row, instrument, aper_diam_index):
        self.loc_depths = np.array([sex_cat_row[f"loc_depth_{band}"].T[aper_diam_index] for band in instrument.bands])
    
    def crop_phot(self, indices):
        indices = np.array(indices).astype(int)
        for index in reversed(indices):
            self.instrument.remove_band(self.bands[index])
        self.flux_Jy = np.delete(self.flux_Jy, indices)
        self.flux_Jy_errs = np.delete(self.flux_Jy_errs, indices)

class Photometry_rest:
    
    def __init__(self, phot_obs, z, code_name, rest_UV_wav_lims = [1216., 3000.] * u.Angstrom):
        self.phot_obs = phot_obs
        self.z = z
        self.code_name = code_name
        self.rest_UV_wav_lims = rest_UV_wav_lims
    
    @property
    def wav(self):
        return wav_obs_to_rest(self.phot_obs.instrument.wav, self.z)
    
    @property
    def wav_errs(self):
        return wav_obs_to_rest(self.phot_obs.instrument.wav_errs, self.z)
        
    @property
    def flux_lambda(self):
        return flux_lambda_obs_to_rest(self.phot_obs.flux_lambda, self.z)
    
    @property
    def flux_lambda_errs(self):
        return flux_lambda_obs_to_rest(self.phot_obs.flux_lambda_errs, self.z)
    
    @property
    def log_flux_lambda(self):
        return np.log10(self.flux_lambda.value)
    
    @property
    def log_flux_lambda_errs(self): # now asymmetric and unitless
        return errs_to_log(self.flux_lambda.value, [self.flux_lambda_errs.value, self.flux_lambda_errs.value])[1]
    
    @property
    def lum_distance(self, cosmo = astropy_cosmo):
        return cosmo.luminosity_distance(self.z).to(u.pc)
    
    @classmethod # this is actually a row in a catalogue, not a 'Galaxy' object
    def get_phot_from_sex(cls, gal, instrument, code):
        phot_obs = Photometry_obs.get_phot_from_sex(gal, instrument)
        return cls(phot_obs, np.float(gal[code.galaxy_properties["z"]]), code.code_name)
    
    def rest_UV_phot_only(self):
        crop_indices = []
        for i, (wav, wav_err) in enumerate(zip(self.wav, self.wav_errs)):
            if wav - wav_err < self.rest_UV_wav_lims[0] or wav + wav_err > self.rest_UV_wav_lims[1]:
                crop_indices = np.append(crop_indices, i)
        self.phot_obs.crop_phot(crop_indices)
     
    def beta_slope_power_law_func(wav_rest, A, beta):
        return (10 ** A) * (wav_rest ** beta)
    
    def set_ext_source_UV_corr(self, ext_source_UV_corr):
        #print(f"Adding extended source UV correction to {self.ID}")
        self.ext_source_UV_corr = ext_source_UV_corr
        
    def plot(self, plot_fit = False, iters = 1_000, save = False, show = True):
        self.rest_UV_phot_only()
        plt.errorbar(np.log10(self.wav.value), self.log_flux_lambda, self.log_flux_lambda_errs, \
                     ls = "none", c = "black", zorder = 10)
        if plot_fit:
            for i in range(iters):
                plt.plot(np.log10(self.wav.value), np.log10(Photometry_rest.beta_slope_power_law_func(self.wav.value, self.amplitude_PDF[i], \
                                                    self.beta_PDF[i])), c = "red", alpha = 0.1, zorder = 1)
            plt.xlabel("log10($\lambda_{rest} / \AA$)")
            plt.ylabel("log10(flux_lambda_rest / whatever these units are)")
            plt.xlim(*np.log10(self.rest_UV_wav_lims.value))
            if show:
                plt.show()
        
    def fit_UV_slope(self, save_dir, ID, z_PDF = None, iters = 100_000, plot = True): # 1D redshift PDF
        print(f"Fitting UV slope for {ID}")
        self.rest_UV_phot_only()
        fluxes = np.array([np.random.normal(mu.value, sigma.value, iters) for mu, sigma in zip(self.flux_lambda, self.flux_lambda_errs)]).T
        if z_PDF != None:
            # vary within redshift errors
            pass
        popt_arr = np.array([curve_fit(Photometry_rest.beta_slope_power_law_func, self.wav, flux)[0] for flux in fluxes])
        if len(popt_arr) >= 1:
            amplitude_PDF = popt_arr.T[0]
            beta_PDF = popt_arr.T[1]
        else:
            amplitude_PDF = [-99.]
            beta_PDF = [-99.]
        self.amplitude_PDF = amplitude_PDF # unitless
        self.beta_PDF = beta_PDF # unitless
                
        for name in ["Amplitude", "Beta"]:
            self.save_UV_fit_PDF(save_dir, name, ID)
        return amplitude_PDF, beta_PDF
    
    def calc_flux_lambda_1500_PDF(self, save_dir, ID):
        self.open_UV_fit_PDF(save_dir, "Amplitude", ID)
        self.open_UV_fit_PDF(save_dir, "Beta", ID)
        self.flux_lambda_1500_PDF = (Photometry_rest.beta_slope_power_law_func(1500., self.amplitude_PDF, self.beta_PDF) \
                                * self.ext_source_UV_corr) * u.erg / (u.s * (u.cm ** 2) * u.Angstrom)
        self.save_UV_fit_PDF(save_dir, "flux_lambda_1500", ID)
        return self.flux_lambda_1500_PDF
    
    def calc_flux_Jy_1500_PDF(self, save_dir, ID):
        self.open_UV_fit_PDF(save_dir, "flux_lambda_1500", ID)
        flux_Jy_1500_PDF = (self.flux_lambda_1500_PDF * ((1500. * u.Angstrom) ** 2) / const.c).to(u.Jy)
        self.flux_Jy_1500_PDF = flux_Jy_1500_PDF
        self.save_UV_fit_PDF(save_dir, "flux_Jy_1500", ID)
        return flux_Jy_1500_PDF
    
    def calc_M_UV_PDF(self, save_dir, ID):
        self.open_UV_fit_PDF(save_dir, "flux_Jy_1500", ID)
        m_1500_PDF = flux_to_mag(self.flux_Jy_1500_PDF, 8.9)
        self.M_UV_PDF = m_1500_PDF - 5 * np.log10(self.lum_distance.value / 10) + 2.5 * np.log10(1 + self.z)
        self.save_UV_fit_PDF(save_dir, "M_UV", ID)
        return self.M_UV_PDF
    
    def calc_A_UV_PDF(self, save_dir, ID, scatter_dex = 0.5):
        self.open_UV_fit_PDF(save_dir, "Beta", ID)
        self.A_UV_PDF = 4.43 + (1.99 * self.beta_PDF) + np.random.uniform(-scatter_dex, scatter_dex)
        self.save_UV_fit_PDF(save_dir, "A_UV", ID)
        return self.A_UV_PDF
    
    def calc_L_obs_PDF(self, save_dir, ID, alpha = 0.): # α=0 in Donnan 2022
        self.open_UV_fit_PDF(save_dir, "flux_Jy_1500", ID)
        self.L_obs_PDF = ((4 * np.pi * self.flux_Jy_1500_PDF * self.lum_distance ** 2) / ((1 + self.z) ** (1 + alpha))).to(u.erg / (u.s * u.Hz))
        self.save_UV_fit_PDF(save_dir, "L_obs", ID)
        return self.L_obs_PDF
    
    def calc_L_int_PDF(self, save_dir, ID):
        self.open_UV_fit_PDF(save_dir, "L_obs", ID)
        self.open_UV_fit_PDF(save_dir, "A_UV", ID)
        self.L_int_PDF = np.array([L_obs * 10 ** (A_UV / 2.5) if A_UV > 0 else L_obs for L_obs, A_UV in zip(self.L_obs_PDF.value, self.A_UV_PDF)]) * (u.erg / (u.s * u.Hz))
        self.save_UV_fit_PDF(save_dir, "L_int", ID)
        return self.L_int_PDF
    
    def calc_SFR_PDF(self, save_dir, ID):
        self.open_UV_fit_PDF(save_dir, "L_int", ID)
        self.SFR_PDF = 1.15e-28 * self.L_int_PDF.value * u.solMass / u.yr # Madau, Dickinson 2014, FSPS, Salpeter IMF templates
        self.save_UV_fit_PDF(save_dir, "SFR", ID)
        return self.SFR_PDF
    
    def save_UV_fit_PDF(self, save_dir, obs_name, ID):
        PDF = self.open_UV_fit_PDF(save_dir, obs_name, ID)
        try:
            unit = PDF.unit # keep track of PDF units
            PDF = np.array([val.value for val in PDF]) # make PDF unitless
        except:
            unit = "dimensionless"

        PDF_hist(PDF, save_dir, obs_name, ID, show = True, save = True)
        save_PDF(PDF, f"{obs_name}, units = {unit}, iters = {len(PDF)}", PDF_path(save_dir, obs_name, ID))
    
    def open_UV_fit_PDF(self, save_dir, obs_name, ID):
        try:
            # attempt to open PDF from object
            PDF = self.obs_name_to_PDF(obs_name)
        except: # if PDF not in object already
            try:
                # attempt to load the previously saved PDF from directory
                PDF = self.load_UV_fit_PDF(save_dir, obs_name, ID)
                print(f"Loaded {obs_name} UV fit successfully for {ID}!")
            except: # if PDF not in object and not already saved
                # calculate the PDF (can take on the order of minutes depending on PDF iters)
                if obs_name == "Amplitude" or obs_name == "Beta":
                    self.fit_UV_slope(save_dir, ID)
                elif obs_name == "flux_lambda_1500":
                    self.calc_flux_lambda_1500_PDF(save_dir, ID)
                elif obs_name == "flux_Jy_1500":
                    self.calc_flux_Jy_1500_PDF(save_dir, ID)
                elif obs_name == "M_UV":
                    self.calc_M_UV_PDF(save_dir, ID)
                elif obs_name == "A_UV":
                    self.calc_A_UV_PDF(save_dir, ID)
                elif obs_name == "L_obs":
                    self.calc_L_obs_PDF(save_dir, ID)
                elif obs_name == "L_int":
                    self.calc_L_int_PDF(save_dir, ID)
                elif obs_name == "SFR":
                    self.calc_SFR_PDF(save_dir, ID)
                else:
                    raise(Exception(f"{obs_name} not valid for calculating UV fit to photometry for {ID}!"))
                PDF = self.obs_name_to_PDF(obs_name)
        return PDF
    
    # this function and the one below could be included in the open_UV_fit_PDF
    def load_UV_fit_PDF(self, save_dir, obs_name, ID):
        # load PDF from directory
        PDF = np.array(np.loadtxt(f"{PDF_path(save_dir, obs_name, ID)}.txt"))
        if obs_name == "Amplitude":
            self.amplitude_PDF = PDF
        elif obs_name == "Beta":
            self.beta_PDF = PDF
        elif obs_name == "flux_lambda_1500":
            self.flux_lambda_1500_PDF = PDF * u.erg / (u.s * (u.cm ** 2) * u.Angstrom)
        elif obs_name == "flux_Jy_1500":
            self.flux_Jy_1500_PDF = PDF * u.Jy
        elif obs_name == "M_UV":
            self.M_UV_PDF = PDF
        elif obs_name == "A_UV":
            self.A_UV_PDF = PDF
        elif obs_name == "L_obs":
            self.L_obs_PDF = PDF * u.erg / (u.s * u.Hz)
        elif obs_name == "L_int":
            self.L_int_PDF = PDF * u.erg / (u.s * u.Hz)
        elif obs_name == "SFR":
            self.SFR_PDF = PDF * u.solMass / u.yr
        else:
            raise(Exception(f"{obs_name} not valid for loading UV fit to photometry for {ID}!"))
        return PDF
    
    # this function and the one above could be included in the open_UV_fit_PDF
    def obs_name_to_PDF(self, obs_name):
        if obs_name == "Amplitude":
            return self.amplitude_PDF
        elif obs_name == "Beta":
            return self.beta_PDF
        elif obs_name == "flux_lambda_1500":
            return self.flux_lambda_1500_PDF
        elif obs_name == "flux_Jy_1500":
            return self.flux_Jy_1500_PDF
        elif obs_name == "M_UV":
            return self.M_UV_PDF
        elif obs_name == "A_UV":
            return self.A_UV_PDF
        elif obs_name == "L_obs":
            return self.L_obs_PDF
        elif obs_name == "L_int":
            return self.L_int_PDF
        elif obs_name == "SFR":
            return self.SFR_PDF
        else:
            raise(Exception(f"No {obs_name} PDF found in object!")) 

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
        
class Galaxy:
    
    # should really expand this to allow for more than one redshift here (only works fro one 'code' class at the moment)
    def __init__(self, sky_coord, phot, ID, properties):
        # print("'z' here for a short time not a long time (in the 'Galaxy' class)! PUT THIS INSTEAD IN THE 'CODE' class")
        self.sky_coord = sky_coord
        # phot_obs is within phot_rest (it shouldn't be!)
        if properties["LePhare"]["z"] == 0:
            self.phot_rest = None
        else:
            self.phot_rest = Photometry_rest(phot, properties["LePhare"]["z"], "LePhare") # works for LePhare only currently
        self.phot_obs = phot # need to improve this still!
        self.ID = int(ID)
        #self.codes = codes
        # this should be contained within each 'code' object
        self.properties = properties
        #self.redshifts = {code.code_name: np.float(z) for code in codes}
        self.mask_flags = {}
        
    @classmethod
    def from_sex_cat_row(cls, sex_cat_row, instrument, cat_creator):
        # load the photometry from the sextractor catalogue
        phot = Photometry_obs.get_phot_from_sex(sex_cat_row, instrument, cat_creator)
        # load the ID and Sky Coordinate from the source catalogue
        ID = sex_cat_row["NUMBER"]
        sky_coord = SkyCoord(sex_cat_row["ALPHA_J2000"] * u.deg, sex_cat_row["DELTA_J2000"] * u.deg, frame = "icrs")
        # perform SED fitting to measure the redshift from the photometry
        # for now, load in z = 0 as a placeholder
        return cls(sky_coord, phot, ID, {})
        
    @classmethod # currently only works for a singular code
    def from_photo_z_cat_row(cls, photo_z_cat_row, instrument, cat_creator, codes):
        # load the photometry from the sextractor catalogue
        phot = Photometry_obs.get_phot_from_sex(photo_z_cat_row, instrument, cat_creator)
        # load the ID and Sky Coordinate from the source catalogue
        ID = photo_z_cat_row["NUMBER"]
        sky_coord = SkyCoord(photo_z_cat_row["ALPHA_J2000"] * u.deg, photo_z_cat_row["DELTA_J2000"] * u.deg, frame = "icrs")
        # also load the galaxy properties from the catalogue
        properties = {code.code_name: {gal_property: {photo_z_cat_row[code.galaxy_properties[gal_property]]}} for code in codes for gal_property in code.galaxy_properties.keys()}
        print(properties)  
        return cls(sky_coord, phot, ID, properties)
        
def ext_source_corr(data, corr_factor, is_log_data = True):
    if is_log_data:
        return data + np.log10(corr_factor)
    else:
        return data * corr_factor
    
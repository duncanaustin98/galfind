#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:20:31 2023

@author: austind
"""

from __future__ import absolute_import

import numpy as np
from astropy.io import fits
from random import randrange
from pathlib import Path
import sep # sextractor for python
import matplotlib.pyplot as plt
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.wcs.utils import pixel_to_skycoord
from astropy.coordinates import search_around_sky, SkyCoord
from matplotlib.colors import LogNorm
from astropy.table import Table, hstack
import copy
import pyregion
import subprocess
import time
import glob
import astropy.units as u
import os
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import json

from .Instrument import Instrument, ACS_WFC,WFC3IR, NIRCam, MIRI, Combined_Instrument
from . import config
from . import useful_funcs_austind as funcs
from .decorators import run_in_dir, hour_timer, email_update

# GALFIND data object
class Data:
    
    def __init__(self, instrument, im_paths, im_exts, seg_paths, mask_paths, cluster_mask_path, blank_mask_path, survey, version = "v0", is_blank = True):
        # sort dicts from blue -> red bands in ascending wavelength order
        self.im_paths = dict(sorted(im_paths.items()))
        self.im_exts = dict(sorted(im_exts.items())) # science image extension
        self.survey = survey
        self.version = version
        self.instrument = instrument
        self.is_blank = is_blank
        
        # make segmentation maps from image paths if they don't already exist
        made_new_seg_maps = False
        for i, (band, seg_path) in enumerate(seg_paths.items()):
            print(band, seg_path)
            if (seg_path == "" or seg_path == []) and not made_new_seg_maps:
                self.make_seg_maps()
                made_new_seg_maps = True
            # load new segmentation maps
            seg_paths[band] = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/SExtractor/{instrument.name}/{version}/{survey}/{survey}*{band}_{band}*{version}*seg.fits")[0]
        self.seg_paths = dict(sorted(seg_paths.items()))
        
        # make masks from image paths if they don't already exist
        print(mask_paths)
        for i, (band, mask_path) in enumerate(mask_paths.items()):
            if (mask_path == "" or mask_path == []):
                self.make_mask(band)
                # load new masks
                mask_paths[band] = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/*{band.replace('W', 'w').replace('M', 'm')}*")[0]
        self.mask_paths = dict(sorted(mask_paths.items()))
        print(f"image paths = {self.im_paths}, segmentation paths = {self.seg_paths}, mask paths = {self.mask_paths}")
        
        if is_blank:
            print(f"{survey} is a BLANK field!")
            self.blank_mask_path = ""
            self.cluster_mask_path = ""
        else:
            print(f"{survey} is a CLUSTER field!")
            self.blank_mask_path = blank_mask_path
            self.cluster_mask_path = cluster_mask_path
            if self.cluster_mask_path == "":
                print("Making cluster mask. (Not yet implemented; self.cluster_path = '' !!!)")
            # try:
            #     self.blank_mask_path = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/*blank*")[0]
            # except:
            #     self.blank_mask_path = ""
            # try:
            #     self.cluster_mask_path = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/*cluster*")[0]
            # except: # make cluster mask based on strong lensing magnification map
            #     self.cluster_mask_path = ""
            #     print("Making cluster mask. (Not yet implemented; self.cluster_path = '' !!!)")
    
    @classmethod
    def from_NIRCam_pipeline(cls, survey, version = "v8"):
        instrument = NIRCam()
        # if int(version.split("v")[1]) >= 8:
        #     pmap = "1084"
        # else:
        #     pmap = "0995"
        
        if version == "v7": #pmap == "0995":
            ceers_im_dirs = {f"CEERSP{str(i + 1)}": f"ceers/mosaic_0995/P{str(i + 1)}" for i in range(10)}
            survey_im_dirs = {"SMACS-0723": "SMACS-0723/mosaic_0995", "GLASS": "glass_0995/mosaic_v5", "MACS-0416": "MACS0416/mosaic_0995_v1", \
                       "El-Gordo": "elgordo/mosaic_0995_v1", "NEP": "NEP/mosaic", "NEP-2": "NEP-2/mosaic_0995", "NGDEEP": "NGDEEP/mosaic", \
                                    "CLIO": "CLIO/mosaic_0995_2"} | ceers_im_dirs
        elif version == "v8": #pmap == "1084":
            ceers_im_dirs = {f"CEERSP{str(i + 1)}": f"ceers/mosaic_1084/P{str(i + 1)}" for i in range(10)}
            survey_im_dirs = {"CLIO": "CLIO/mosaic_1084", "El-Gordo": "elgordo/mosaic_1084", "GLASS": "GLASS-12/mosaic_1084", "NEP": "NEP/mosaic_1084", \
                          "NEP-2": "NEP-2/mosaic_1084", "NEP-3": "NEP-3/mosaic_1084", "SMACS-0723": "SMACS0723/mosaic_1084", "MACS-0416": "MACS0416/mosaic_1084_v3"} | ceers_im_dirs
        elif version == "v8a":
            ceers_im_dirs = {f"CEERSP{str(i + 1)}": f"ceers/mosaic_1084_182/P{str(i + 1)}" for i in range(10)}
            survey_im_dirs = {"CLIO": "CLIO/mosaic_1084_182", "El-Gordo": "elgordo/mosaic_1084_182", "NEP-1": "NEP/mosaic_1084_182", "NEP-2": "NEP-2/mosaic_1084_182", \
                              "NEP-3": "NEP-3/mosaic_1084_182", "MACS-0416": "MACS0416/mosaic_1084_182", "GLASS": "GLASS-12/mosaic_1084_182"} | ceers_im_dirs
            
        survey_im_dirs = {key: f"/raid/scratch/data/jwst/{value}" for (key, value) in survey_im_dirs.items()}
        survey_dir = survey_im_dirs[survey]
        
        # don't use these if they are in the same folder
        nadams_seg_path_arr = glob.glob(f"{survey_dir}/*_seg.fits")
        nadams_bkg_path_arr = glob.glob(f"{survey_dir}/*_bkg.fits")
        
        im_path_arr = glob.glob(f"{survey_dir}/*_i2d*.fits")
        im_path_arr = np.array([path for path in im_path_arr if path not in nadams_seg_path_arr and path not in nadams_bkg_path_arr])
        
        # obtain available bands from imaging without having to hard code these
        bands = np.array([split_path.lower().replace("w", "W").replace("m", "M") for path in im_path_arr for i, split_path in \
                enumerate(path.split("-")[-1].split("/")[-1].split("_")) if split_path.lower().replace("w", "W").replace("m", "M") in instrument.bands])

        for band in instrument.bands:
            if band not in bands:
                instrument.remove_band(band)
        print(instrument.bands)
        
        im_paths = {}
        im_exts = {}
        seg_paths = {}
        mask_paths = {}
        for i, band in enumerate(bands):
            im_paths[band] = im_path_arr[i]
            im_hdul = fits.open(im_path_arr[i])
            # obtain appropriate extension from the image
            for j, im_hdu in enumerate(im_hdul):
                if im_hdu.name == "SCI":
                    im_exts[band] = int(j)
                    break
            # need to change this to work if there are no segmentation maps (with the [0] indexing)
            try:
                seg_paths[band] = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/SExtractor/{instrument.name}/{version}/{survey}/{survey}*{band}_{band}*{version}*seg.fits")[0]
            except:
                seg_paths[band] = ""
            # include just the masks corresponding to the correct bands
            try:
                mask_paths[band] = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/*{band.replace('W', 'w').replace('M', 'm')}*")[0]
            except:
                mask_paths[band] = ""
            try:
                cluster_mask_path = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/*cluster*")[0]
            except:
                cluster_mask_path = ""
            try:
                blank_mask_path = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/*blank*")[0]
            except:
                blank_mask_path = ""
            # if this mask doesn't exist, create it using automated code (NOT YET IMPLEMENTED!)
        return cls(instrument, im_paths, im_exts, seg_paths, mask_paths, cluster_mask_path, blank_mask_path, survey, version, is_blank = is_blank_survey(survey))

# %% Overloaded operators

    def __repr__(self):
        # string representation of what is stored in this class
        return str(self.__dict__)
    
    def __len__(self):
        return len(self.instrument)
    
    def __iter__(self):
        self.iter = 0
        return self
    
    def __next__(self):
        if self.iter > len(self) - 1:
            raise StopIteration
        else:
            band_data = self[self.iter]
            self.iter += 1
            return band_data
    
    def __getitem__(self, index):
        return self.load_data(self.instrument.bands[index], incl_mask = True)
    
    def __add__(self, data):
        common_bands = [band for band in data.instrument.bands if band in self.instrument.bands]
        if len(common_bands) != 0:
            raise(Exception(f"Cannot add two of the same bands from different instruments together! Culprits: {common_bands}"))
        # add all dictionaries together
        self.__dict__ = {k: self.__dict__.get(k, 0) + data.__dict__.get(k, 0) for k in set(self.__dict__()) | set(data.__dict__())}
        print(self.__repr__)
        return self
    
# %% Methods
        
    def load_data(self, band, incl_mask = True):
        im_path = self.im_paths[band]
        seg_path = self.seg_paths[band]
        im_ext = self.im_exts[band]
        #print("im_path, seg_path, band")
        #print(im_path, seg_path, band)
        im_hdul = fits.open(im_path) #directory of images and image name structure for science image
        im_data = im_hdul[im_ext].data
        im_data = im_data.byteswap().newbyteorder()
        im_header = im_hdul[im_ext].header
        seg_hdul = fits.open(seg_path) #directory of images and image name structure for segmentation map
        seg_data = seg_hdul[0].data
        seg_header = seg_hdul[0].header
        if incl_mask:
            mask_path = self.mask_paths[band]
            mask_file = pyregion.open(mask_path) # file for mask
            mask_file = mask_file.as_imagecoord(im_header)
            mask = mask_file.get_mask(hdu = im_hdul[im_ext])
            #print("mask_path", mask_path)
            return im_data, im_header, seg_data, seg_header, mask
        else:
            return im_data, im_header, seg_data, seg_header
    
    @run_in_dir(path = config['DEFAULT']['GALFIND_DIR'])
    def make_seg_maps(self):
        for band in self.instrument.bands:
            # SExtractor bash script python wrapper
            process = subprocess.Popen([f"./make_seg_map.sh", config['DEFAULT']['GALFIND_WORK'], self.im_paths[band], str(self.instrument.pixel_scales[band].value), \
                                    str(self.instrument.zero_points[band]), self.instrument.name, self.survey, band, self.version])
            process.wait()
            print(f"Made segmentation map for {self.survey} {self.version} {band}")
    
    @run_in_dir(path = config['DEFAULT']['GALFIND_DIR'])
    def make_sex_cats(self, forced_phot_band = "f444W"):
        # make individual forced photometry catalogues
        for band in self.instrument.bands:
            # if not run before
            if not Path(f"{config['DEFAULT']['GALFIND_WORK']}/SExtractor/{self.instrument.name}/{self.version}/{self.survey}/{self.survey}_{band}_{forced_phot_band}_sel_cat_{self.version}.fits").is_file():
                # SExtractor bash script python wrapper
                process = subprocess.Popen([f"./make_sex_cat.sh", config['DEFAULT']['GALFIND_WORK'], self.im_paths[band], str(self.instrument.pixel_scales[band].value), \
                                    str(self.instrument.zero_points[band]), self.instrument.name, self.survey, band, self.version, \
                                        forced_phot_band, self.im_paths[forced_phot_band]])
                process.wait()
            print(f"Finished making SExtractor catalogue for {self.survey} {self.version} {band}!")
        self.sex_cats = {band: f"{config['DEFAULT']['GALFIND_WORK']}/SExtractor/{self.instrument.name}/{self.version}/{self.survey}/{self.survey}_{band}_{forced_phot_band}_sel_cat_{self.version}.fits" for band in self.instrument.bands}
    
    def combine_sex_cats(self, forced_phot_band = "f444W"):
        self.make_sex_cats(forced_phot_band)
        # run only if this doesn't already exist
        save_name = f"{self.survey}_MASTER_Sel-{forced_phot_band}_{self.version}.fits"
        save_dir = f"{config['DEFAULT']['GALFIND_WORK']}/Catalogues/{self.version}/{self.instrument.name}/{self.survey}"
        self.sex_cat_master_path = f"{save_dir}/{save_name}"
        if not Path(self.sex_cat_master_path).is_file():
            for i, (band, path) in enumerate(self.sex_cats.items()):
                tab = Table.read(path, character_as_bytes = False)
                if i != 0:
                    # remove the duplicated IDs and RA/DECs
                    tab = remove_non_band_dependent_sex_params(tab)
                # load each one into an astropy table and update the column names by adding an "_FILT" suffix
                tab = add_band_suffix_to_cols(tab, band)
                # combine the astropy tables
                if i == 0:
                    master_tab = tab
                else:
                    master_tab = hstack([master_tab, tab])
            # save table
            os.makedirs(save_dir, exist_ok = True)
            master_tab.write(self.sex_cat_master_path, format = "fits", overwrite = True)
        
    def make_sex_plusplus_cat(self):
        pass
    
    def make_mask(self, band, stellar_dir = "GAIA DR3"):
        im_data, im_header, seg_data, seg_header = self.load_data(band, incl_mask = False)
        # works as long as your images are aligned with the stellar directory
        stellar_mask = make_stellar_mask(band, im_data, im_header, self.instrument)
        # save the stellar mask
        stellar_mask.write(f"{os.getcwd()}/Masks/{survey}/{band}_stellar_mask.reg", header = im_header)
        # maybe insert a step here to align your images
        pass
    
    def calc_unmasked_area(self):
        # calculate mask area
        # unmasked_pix = 0
        # for i in range(mask.shape[0]):
        #     for j in range(mask.shape[1]):
        #         if mask[i][j] == False: # and obs_region_mask[i][j] == True
        #             unmasked_pix = unmasked_pix + 1
        # unmasked_area = unmasked_pix * ((0.03 * u.arcsec) ** 2).to(u.arcmin ** 2)
        # print(unmasked_area)
        # unmasked_area_arr.append(unmasked_area)
        pass
        
    def make_loc_depth_cat(self, aper_diams = [0.32] * u.arcsec, n_samples = 5, forced_phot_band = "f444W", min_percentage_err = 5):
        print(f"Making local depth catalogue for {self.survey} {self.version} in {aper_diams} diameter apertures with min. error {min_percentage_err}%!")
        # if sextractor catalogue has not already been made, make it
        self.combine_sex_cats(forced_phot_band)
        # if depths havn't already been run, run them
        self.calc_depths(aper_diams = aper_diams)
        # correct the base sextractor catalogue to include local depth errors if not already done so
        self.loc_depth_cat_path = self.sex_cat_master_path.replace(".fits", "_loc_depth.fits")
        if not Path(self.loc_depth_cat_path).is_file():
            # open photometric data
            phot_data = fits.open(self.sex_cat_master_path)[1].data  
            for i, band in enumerate(tqdm(self.instrument.bands, desc = f"Making local depth catalogue", total = len(self.instrument))):
                
                im_data, im_header, seg_data, seg_header = self.load_data(band, incl_mask = False)
                wcs = WCS(im_header)
                
                # perform aperture correction
                # make new columns and overwrite them in the next couple of lines
                phot_data = make_new_fits_columns(phot_data, ["MAG_APER_" + band + "_aper_corr", "FLUX_APER_" + band + "_aper_corr"], \
                                            [phot_data["MAG_APER_" + band], phot_data["FLUX_APER_" + band]], \
                                                [phot_data.columns.formats[list(phot_data.columns.names).index("MAG_APER_" + band)], \
                                                 phot_data.columns.formats[list(phot_data.columns.names).index("FLUX_APER_" + band)]])
                for j, aper_diam in enumerate(json.loads(config.get("SExtractor", "APERTURE_DIAMS")) * u.arcsec):
                    phot_data["MAG_APER_" + band + "_aper_corr"].T[j] = (phot_data["MAG_APER_" + band].T[j] - self.instrument.aper_corr(aper_diam, band)).T
                    phot_data["FLUX_APER_" + band + "_aper_corr"].T[j] = 10 ** ((phot_data["MAG_APER_" + band + "_aper_corr"].T[j] - self.instrument.zero_points[band]) / -2.5)
                    print("Performed aperture corrections")
                
                # make new columns (fill with original errors and overwrite in a couple of lines)
                phot_data = make_new_fits_columns(phot_data, ["loc_depth_" + band, "FLUXERR_APER_" + band + "_loc_depth", "MAGERR_APER_" + band + "_l1_loc_depth", \
                                            "MAGERR_APER_" + band + "_u1_loc_depth", "FLUX_APER_" + band + "_aper_corr_Jy", "FLUXERR_APER_" + band + "_loc_depth_" + str(min_percentage_err) + "pc_Jy", \
                                            "MAGERR_APER_" + band + "_l1_loc_depth_" + str(min_percentage_err) + "pc", "MAGERR_APER_" + band + "_u1_loc_depth_" + str(min_percentage_err) + "pc", "sigma_" + band], \
                                            [phot_data["FLUXERR_APER_" + band], phot_data["FLUXERR_APER_" + band], phot_data["MAGERR_APER_" + band], phot_data["MAGERR_APER_" + band], \
                                             phot_data["FLUX_APER_" + band], phot_data["FLUXERR_APER_" + band], phot_data["MAGERR_APER_" + band], phot_data["MAGERR_APER_" + band], phot_data["MAGERR_APER_" + band]], \
                                            [phot_data.columns.formats[list(phot_data.columns.names).index("FLUXERR_APER_" + band)], \
                                             phot_data.columns.formats[list(phot_data.columns.names).index("FLUXERR_APER_" + band)], \
                                             phot_data.columns.formats[list(phot_data.columns.names).index("MAGERR_APER_" + band)], \
                                             phot_data.columns.formats[list(phot_data.columns.names).index("MAGERR_APER_" + band)], \
                                             phot_data.columns.formats[list(phot_data.columns.names).index("FLUX_APER_" + band)], \
                                             phot_data.columns.formats[list(phot_data.columns.names).index("FLUXERR_APER_" + band)], \
                                             phot_data.columns.formats[list(phot_data.columns.names).index("MAGERR_APER_" + band)], \
                                             phot_data.columns.formats[list(phot_data.columns.names).index("MAGERR_APER_" + band)], \
                                             phot_data.columns.formats[list(phot_data.columns.names).index("FLUXERR_APER_" + band)]])
                
                for j, aper_diam in enumerate(json.loads(config.get("SExtractor", "APERTURE_DIAMS")) * u.arcsec):
                    for k in range(len(phot_data["NUMBER"])):
                        # set initial values to -99. or False by default
                        phot_data["loc_depth_" + band].T[j][k] = -99.
                        phot_data["FLUXERR_APER_" + band + "_loc_depth"].T[j][k] = -99.
                        phot_data["MAGERR_APER_" + band + "_l1_loc_depth"].T[j][k] = -99.
                        phot_data["MAGERR_APER_" + band + "_u1_loc_depth"].T[j][k] = -99.
                        phot_data["FLUXERR_APER_" + band + "_loc_depth_" + str(min_percentage_err) + "pc_Jy"].T[j][k] = -99.
                        phot_data["MAGERR_APER_" + band + "_l1_loc_depth_" + str(min_percentage_err) + "pc"].T[j][k] = -99.
                        phot_data["MAGERR_APER_" + band + "_u1_loc_depth_" + str(min_percentage_err) + "pc"].T[j][k] = -99.
                        phot_data["sigma_" + band].T[j][k] = -99.
                        
                        # update column for flux in Jy
                        phot_data["FLUX_APER_" + band + "_aper_corr_Jy"].T[j][k] = funcs.flux_image_to_Jy(phot_data["FLUX_APER_" + band + "_aper_corr"].T[j][k], self.instrument.zero_points[band]).value
            
                for diam_index, aper_diam in enumerate(aper_diams):
                    r = self.calc_aper_radius_pix(aper_diam, band)
                    # open aperture positions in this band
                    aper_loc = np.loadtxt(f"{self.get_depth_dir(aper_diam)}/coord_{band}.txt")
                    xcoord = aper_loc[:, 0]
                    ycoord = aper_loc[:, 1]
                    index = np.argwhere(xcoord == 0.)
                    xcoord = np.delete(xcoord, index)
                    ycoord = np.delete(ycoord, index)
                    aper_coords = pixel_to_skycoord(xcoord, ycoord, wcs)
                    
                    # calculate local depths for all galaxies
                    loc_depths = calc_loc_depths(phot_data["ALPHA_J2000"], phot_data["DELTA_J2000"], aper_coords, xcoord, ycoord, im_data, r, self.survey, band, n_samples = n_samples, zero_point = self.instrument.zero_points[band])
                    
                    five_sigma_detected = []
                    two_sigma_non_detected = []
                    three_sigma_non_detected = []
                    nans = 0
                    # calculate local depths in each band for the relevant aperture diameters
                    for k in range(len(phot_data["NUMBER"])):
                        # calculate error based on the 5σ local depth from the n_aper nearest background aperture fluxes
                        phot_data["loc_depth_" + band].T[diam_index][k] = loc_depths[k]
                        loc_depth = loc_depths[k]
                        loc_depth_nan = False
                        if loc_depth == np.nan:
                            nans = nans + 1
                            loc_depth_nan = True
                        aper_flux_err = (10 ** ((loc_depth - self.instrument.zero_points[band]) / -2.5)) / 5 # in image units
                        if aper_flux_err == np.nan and not loc_depth_nan:
                            nans = nans + 1
                            print("loc_depth =", loc_depth)
                        phot_data["FLUXERR_APER_" + band + "_loc_depth"].T[diam_index][k] = aper_flux_err
                        
                        # add column setting flux in Jy to minimum 5pc error
                        if phot_data["FLUXERR_APER_" + band + "_loc_depth"].T[diam_index][k] / phot_data["FLUX_APER_" + band + "_aper_corr"].T[diam_index][k] < min_percentage_err / 100:
                            phot_data["FLUXERR_APER_" + band + "_loc_depth_" + str(min_percentage_err) + "pc_Jy"].T[diam_index][k] = \
                            phot_data["FLUX_APER_" + band + "_aper_corr_Jy"].T[diam_index][k] * min_percentage_err / 100
                        else:
                            phot_data["FLUXERR_APER_" + band + "_loc_depth_" + str(min_percentage_err) + "pc_Jy"].T[diam_index][k] = \
                                funcs.flux_image_to_Jy(phot_data["FLUXERR_APER_" + band + "_loc_depth"].T[diam_index][k], self.instrument.zero_points[band]).value
                        
                        # calculate local depth mag errors both with and without 5pc minimum flux errors imposed
                        for m in range(2):
                            if m == 0:
                                flux = phot_data["FLUX_APER_" + band + "_aper_corr"].T[diam_index][k]
                                aper_flux_err = phot_data["FLUXERR_APER_" + band + "_loc_depth"].T[diam_index][k]
                                add_suffix = ""
                            elif m == 1:
                                flux = phot_data["FLUX_APER_" + band + "_aper_corr_Jy"].T[diam_index][k]
                                aper_flux_err = phot_data["FLUXERR_APER_" + band + "_loc_depth_" + str(min_percentage_err) + "pc_Jy"].T[diam_index][k]
                                add_suffix = "_" + str(min_percentage_err) + "pc"
                        
                            mag_l1 = -(-2.5 * np.log10(flux) + 2.5 * np.log10(flux - aper_flux_err))
                            mag_u1 = -(-2.5 * np.log10(flux + aper_flux_err) + 2.5 * np.log10(flux))
                            #print(mag_l1)
                            #print(mag_u1)
                            if np.isfinite(mag_l1):
                                phot_data["MAGERR_APER_" + band + "_l1_loc_depth" + add_suffix].T[diam_index][k] = mag_l1
                            if np.isfinite(mag_u1):  
                                phot_data["MAGERR_APER_" + band + "_u1_loc_depth" + add_suffix].T[diam_index][k] = mag_u1
                        
                        # make boolean columns to say whether there is a local 5σ detection and 2σ non-detection in the band in the smallest aperture
                        phot_data["sigma_" + band].T[diam_index][k] = funcs.n_sigma_detection(loc_depth, phot_data[f"MAG_APER_{band}"].T[diam_index][k], self.instrument.zero_points[band])
                        if phot_data[f"MAG_APER_{band}"].T[diam_index][k] < loc_depth:
                            five_sigma_detected.append(True)
                        else:
                            five_sigma_detected.append(False)
                        if phot_data[f"MAG_APER_{band}"].T[diam_index][k] > funcs.five_to_n_sigma_mag(loc_depth, 2):
                            two_sigma_non_detected.append(True)
                        else:
                            two_sigma_non_detected.append(False)
                        if phot_data[f"MAG_APER_{band}"].T[diam_index][k] > funcs.five_to_n_sigma_mag(loc_depth, 3):
                            three_sigma_non_detected.append(True)
                        else:
                            three_sigma_non_detected.append(False)
                        #print("k =", k)
                        if k == 0:
                            print("Made boolean detection columns")
    
                    phot_data = make_new_fits_columns(phot_data, [f"5sigma_{band}_{str(aper_diam.value)}as", f"2sigma_non_detect_{band}_{str(aper_diam.value)}as", \
                                                f"3sigma_non_detect_{band}_{str(aper_diam.value)}as"], [five_sigma_detected, two_sigma_non_detected, \
                                                three_sigma_non_detected], ["L", "L", "L"], True, self.loc_depth_cat_path) # save .fits table
                    print("total nans =", nans)
        
    def get_depth_dir(self, aper_diam):
        self.depth_dir = f"{config['DEFAULT']['GALFIND_WORK']}/Depths/{self.instrument.name}/{self.version}/{self.survey}/{str(aper_diam.value)}as"
        os.makedirs(self.depth_dir, exist_ok = True)
        return self.depth_dir
    
    def calc_aper_radius_pix(self, aper_diam, band):
        return (aper_diam / (2 * self.instrument.pixel_scales[band])).value
    
    def calc_depths(self, xy_offset = [0, 0], aper_diams = [0.32] * u.arcsec, size = 500, n_busy_iters = 1_000, number = 600, \
                    mask_rad = 25, aper_disp_rad = 2, excl_bands = [], use_xy_offset_txt = True):
        
        for aper_diam in aper_diams:
            print(aper_diam)
            self.get_depth_dir(aper_diam)
        
            average_depths = []
            header = "band, average_5σ_depth"
            for band in self.instrument.bands:
                if band not in excl_bands:
                    if use_xy_offset_txt:
                        try:
                            # use the xy_offset defined in .txt in appropriate folder
                            xy_offset_path = f"{self.depth_dir}/offset_{band}.txt"
                            xy_offset = list(np.genfromtxt(xy_offset_path, dtype = int))
                            print(f"xy_offset = {xy_offset}")
                        except: # use default xy offset if this .txt does not exist
                            pass
                        
                    if not Path(f"{self.depth_dir}/coord_{band}.reg").is_file() or not Path(f"{self.depth_dir}/{self.survey}_depths.txt").is_file() or not Path(f"{self.depth_dir}/coord_{band}.txt").is_file():
                        xoff, yoff = calc_xy_offsets(xy_offset)
                        im_data, im_header, seg_data, seg_header, mask = self.load_data(band)
    
                    if not Path(f"{self.depth_dir}/coord_{band}.txt").is_file():
                        # place apertures in blank regions of sky
                        xcoord, ycoord = place_blank_regions(im_data, im_header, seg_data, mask, self.survey, xy_offset, self.instrument.pixel_scales[band], band, \
                                                         aper_diam, size, n_busy_iters, number, mask_rad, aper_disp_rad)
                        np.savetxt(f"{self.depth_dir}/coord_{band}.txt", np.column_stack((xcoord, ycoord)))
                        # save xy offset for this field and band
                        np.savetxt(f"{self.depth_dir}/offset_{band}.txt", np.column_stack((xoff, yoff)), header = "x_off, y_off", fmt = "%d %d")
                    
                    # read in aperture locations
                    if not Path(f"{self.depth_dir}/coord_{band}.reg").is_file() or not Path(f"{self.depth_dir}/{self.survey}_depths.txt").is_file():
                        aper_loc = np.loadtxt(f"{self.depth_dir}/coord_{band}.txt")
                        xcoord = aper_loc[:, 0]
                        ycoord = aper_loc[:, 1]
                        index = np.argwhere(xcoord == 0.)
                        xcoord = np.delete(xcoord, index)
                        ycoord = np.delete(ycoord, index)
                    
                    # convert these to .reg region file
                    if not Path(f"{self.depth_dir}/coord_{band}.reg").is_file():
                        aper_loc_to_reg(xcoord, ycoord, WCS(im_header), aper_diam.value, f"{self.depth_dir}/coord_{band}.reg")
                    
                    r = self.calc_aper_radius_pix(aper_diam, band)
                    if not Path(f"{self.depth_dir}/{self.survey}_depths.txt").is_file():
                        # plot the depths in the grid
                        plot_depths(im_data, self.depth_dir, band, seg_data, xcoord, ycoord, xy_offset, r, size, self.instrument.zero_points[band])
                        # calculate average depth
                        average_depths.append(calc_5sigma_depth(xcoord, ycoord, im_data, r, self.instrument.zero_points[band]))
                        
            # print table of depths for these bands
            if not Path(f"{self.depth_dir}/{self.survey}_depths.txt").is_file():
                np.savetxt(f"{self.depth_dir}/{self.survey}_depths.txt", np.column_stack((np.array(self.instrument.bands), np.array(average_depths))), header = header, fmt = "%s")
        
# match sextractor catalogue codes
sex_id_params = ["NUMBER", "X_IMAGE", "Y_IMAGE", "ALPHA_J2000", "DELTA_J2000"]

def remove_non_band_dependent_sex_params(tab, sex_id_params = sex_id_params):
    for i, param in enumerate(sex_id_params):
        tab.remove_column(param)
    return tab

def add_band_suffix_to_cols(tab, band, sex_id_params = sex_id_params):
    for i, name in enumerate(tab.columns.copy()):
        if name not in sex_id_params: 
            tab.rename_column(name, name + "_" + band)
    return tab

# depth codes (taken from background_calc.py)

def log_transform(im): # function to transform fits image to log scaling
    '''returns log(image) scaled to the interval [0,1]'''
    try:
        (min, max) = (im[im > 0].min(), im.max())
        if (max > min) and (max > 0):
            return (np.log(im.clip(min, max)) - np.log(min)) / (np.log(max) - np.log(min))
    except:
        pass
    return im

def circ_mask(h, w, center = None, radius = None):
	if center is None:
		center = (int(w/2), int(h/2))
	if radius is None:
		radius = min(center[0], center[1], w-center[0], h-center[1])

	Y,X = np.ogrid[:h, :w]
	dist = np.sqrt((X-center[1])**2 + (Y-center[0])**2)
	#print("circmask",center[0], center[1])
	mask = dist <= radius
	return mask

def calc_xy_offsets(offset):
    if type(offset) == list or type(offset) == np.array:
        xoff = offset[0]
        yoff = offset[1]
    elif type(offset) == dict:
        xoff = offset["x"]
        yoff = offset["y"]
    else:
        xoff = offset
        yoff = offset
    print(f"x_off = {xoff}, y_off = {yoff}")
    return xoff, yoff


def place_blank_regions(im_data, im_header, seg_data, mask, survey, offset, pix_scale, band, aper_diam = 0.32 * u.arcsec, size = 500, n_busy_iters = 1_000, number = 600, mask_rad = 25, aper_disp_rad = 2):
    
    im_wcs = WCS(im_header)
    r = aper_diam / (2 * pix_scale) # radius of aperture in pixels
    
    xoff, yoff = calc_xy_offsets(offset)
    
    xchunk = int(seg_data.shape[1])
    ychunk = int(seg_data.shape[0])
    xcoord = list()
    ycoord = list()
    busylist = list()
    no_space = 0
    # finds locations to place empty apertures in
    for i in tqdm(range(0, int((xchunk - (2 * xoff)) / size)), desc = f"Running {band} depths for {survey}"):
        for j in tqdm(range(0, int((ychunk - (2 * yoff)) / size)), desc = f"Current row = {i + 1}", leave = False):
            busyflag = 0
            #print(j, i)
            # narrow seg, image and mask data to appropriate size for the chunk
            seg_chunk = seg_data[(j * size) + yoff : ((j + 1) * size) + yoff, (i * size) + xoff:((i + 1) * size) + xoff]
            aper_mask_chunk = copy.deepcopy(seg_chunk)
            im_chunk = im_data[(j*size)+yoff:((j+1)*size)+yoff, (i*size)+xoff:((i+1)*size)+xoff]
            mask_chunk = mask[(j*size)+yoff:((j+1)*size)+yoff, (i*size)+xoff:((i+1)*size)+xoff] #cut the box of interest out of the images
            xlen = seg_chunk.shape[1]
            ylen = seg_chunk.shape[0]
            
            # check if there is enough space to fit apertures even if perfectly aligned
            z = np.argwhere((seg_chunk == 0) & (im_chunk != 0.) & (mask_chunk == False) & (aper_mask_chunk == 0)) #generate a list of candidate locations for empty apertures
            if len(z) > 0:
                space = True
            else:
                space = False
            if space: # there are space for "number" of apertures
                 # cycle through range of available locations for empty apertures
                 for c in range(0, number): # tqdm(), desc = "Grid square completion", total = number * 0.6, leave = False):
                     next = 0
                     iters = 0
                     
                     while next == 0:
     
                         idx = randrange(len(z)) #find random candidate location for empty aperture
                         # z[idx] is (y, x) pixel number
                         if (z[idx][0] < mask_rad or z[idx][0] > ylen - mask_rad) or (z[idx][1] < mask_rad or z[idx][1] > xlen - mask_rad): # dont place empty aperture near edges of image (not data)
                             iters += 1
                             if iters > n_busy_iters: #if struggling to place empty apertures down skip the region (happens near big resolved objects, stars and large masked/empty regions)
     							#print("busy region")
                                 busyflag = 1
                                 next += 1					
                         else:
                             iters += 1				
                             h, w = seg_chunk.shape[:2]
                             # changed to be different to maskrad, which now just looks at the region edges
                             source_mask = circ_mask(w, h, radius = mask_rad, center = z[idx]) #draw a circle on segmentation map so make sure empty aperture isn't near another object and truly empty
                             masked_source_image = copy.deepcopy(seg_chunk)
                             masked_source_image[source_mask == 0] = 0 #set all area outside of circle to 0
                             source = np.argwhere((masked_source_image != 0)) #check if any part of the masked segmentation map contains another source
                             aper_mask = circ_mask(w, h, radius = r + aper_disp_rad, center = z[idx])
                             masked_aper_image = copy.deepcopy(aper_mask_chunk)
                             masked_aper_image[aper_mask == 0] = 0 #set all area outside of circle to 0
                             aper = np.argwhere((masked_aper_image != 0))
                             img = im_chunk[z[idx][0] - mask_rad : z[idx][0] + mask_rad, z[idx][1] - mask_rad : z[idx][1] + mask_rad]
                             empty = np.argwhere(img == 0) # make sure there is data for this region
                             # could also search to see if the aperture covers any manually masked areas
                             if (len(source) == 0 and len(aper) == 0 and len(empty) == 0):# or iters > 200: #if location is good or too many iterations and given up
                                 xcoord.append(int(z[idx][1] + (i*size) + xoff))
                                 ycoord.append(int(z[idx][0] + (j*size) + yoff)) #convert coordinate of empty aperture in mini section to coordinate on full image
                                 # set segmentation map to include previous apertures
                                 aper_mask_chunk[aper_mask == 1] = 1
                                 next += 1
     						
                             elif iters > n_busy_iters: #if takes too long to find a good spot, flag section of image as busy
                                 next += 1
                                 busyflag = 1
                                 
                     if busyflag == 1: # if the region is busy, set the 200 apertures to co-ordinates (0., 0.)
                         #print(c, "apertures in (", i, ",", j, ") !")
                         xcoord.extend([0] * (number - c))
                         ycoord.extend([0] * (number - c))
                         break
             
            else: # no need for busy flag here as this has already been determined
                xcoord.extend([0] * number)
                ycoord.extend([0] * number)
        #print("len(xcoord) =", len(xcoord)) 
    return xcoord, ycoord

# depth codes (taken from convert_aper_loc_to_reg.py)

def aper_loc_to_reg(xcoord, ycoord, wcs, aper_diam, save_path):
    sky_coord = pixel_to_skycoord(xcoord, ycoord, wcs)
    #print(len(sky_coord))
    sky_coord_zero = pixel_to_skycoord(0., 0., wcs)
    sky_coord = [coord for coord in sky_coord if coord != sky_coord_zero]
    #print(len(sky_coord))

    region_str = """
    # Region file format: DS9 version 4.1
    global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
    fk5
    """
    for i in tqdm(range(len(sky_coord)), desc = f"Converting apertures to .reg in {save_path}"):
        # print(sky_coord[i].dec.hms, sky_coord[i].dec.hms[1])
        sky_coord_loc = sky_coord[i].to_string('hmsdms')
        sky_coord_loc = sky_coord_loc.replace(" ", ",")
        sky_coord_loc = sky_coord_loc.replace("+", "")
        sky_coord_loc = sky_coord_loc.replace("h", ":")
        sky_coord_loc = sky_coord_loc.replace("d", ":")
        sky_coord_loc = sky_coord_loc.replace("m", ":")
        sky_coord_loc = sky_coord_loc.replace("s", "")
        aper_artist = "circle(%s,%1.2f\")\n" % (sky_coord_loc, aper_diam / 2)
        region_str += aper_artist
    #print(region_str)
    # save the .reg file
    text_file = open(save_path, "w")
    text_file.write(region_str)
    text_file.close()
    
# depth codes (taken from plot_depths.py)

def calc_chunk_depths(im_data, xcoord_in, ycoord_in, xchunk, ychunk, xoff, yoff, r, size, zero_point, label):
    # split image into equal sized chunks
    depth_image = np.full(im_data.shape, np.nan)
    for i in tqdm(range(0,int((xchunk-(2*xoff))/size)), desc = label):
        x_invalid_low = np.argwhere((xcoord_in <= (i * size) + xoff))# or (xcoord > (i + 1) * size + xoff))
        xcoord_loc = np.delete(xcoord_in, x_invalid_low)
        ycoord_loc = np.delete(ycoord_in, x_invalid_low)
        #print(len(xcoord_loc), len(ycoord_loc))
        x_invalid_high = np.argwhere((xcoord_loc >= ((i + 1) * size) + xoff))
        xcoord_loc = np.delete(xcoord_loc, x_invalid_high)
        ycoord_loc = np.delete(ycoord_loc, x_invalid_high)
        #print(len(xcoord_loc), len(ycoord_loc))
        for j in range(0,int((ychunk-(2*yoff))/size)):       
            y_invalid_low = np.argwhere((ycoord_loc <= (j * size) + yoff))# or (ycoord > (j + 1) * size + yoff))
            xcoord_loc2 = np.delete(xcoord_loc, y_invalid_low)
            ycoord_loc2 = np.delete(ycoord_loc, y_invalid_low)
            #print(len(xcoord_loc2), len(ycoord_loc2))
            y_invalid_high = np.argwhere((ycoord_loc2 >= ((j + 1) * size) + yoff))
            xcoord_loc2 = np.delete(xcoord_loc2, y_invalid_high)
            ycoord_loc2 = np.delete(ycoord_loc2, y_invalid_high)
            #print(len(xcoord_loc2), len(ycoord_loc2))
            if len(xcoord_loc2) != 0:
                depth_5sigma = calc_5sigma_depth(list(xcoord_loc2), list(ycoord_loc2), im_data, r, zero_point)
                #print(depth_5sigma)
                depth_image[(j*size)+yoff:((j+1)*size)+yoff, (i*size)+xoff:((i+1)*size)+xoff] = depth_5sigma

    # print mean/median depths for the field
    depths_no_nans = [depth_image[j][i] for j in range(depth_image.shape[0]) for i in range(depth_image.shape[1]) if not math.isnan(depth_image[j][i])]
    print("mean depth =", np.mean(depths_no_nans), ", median depth =", np.median(depths_no_nans))
    return depths_no_nans, depth_image
    
def plot_depths(im_data, depth_dir, band, seg_data, xcoord, ycoord, offset, r, size, zero_point, cmap = cm.get_cmap("plasma")):
    
    cmap.set_bad(color = 'black')
    
    xoff = offset[0]
    yoff = offset[1]
    xchunk = int(seg_data.shape[1])
    ychunk = int(seg_data.shape[0])
    depths_all_field, depth_image = calc_chunk_depths(im_data, xcoord, ycoord, xchunk, ychunk, xoff, yoff, r, size, zero_point, label = f"{depth_dir} {band}")
    
    extent = 0., xchunk, 0., ychunk
    plt.imshow(depth_image, cmap = cmap, extent = extent, origin = "lower")
    plt.xlabel("x [pix]")
    plt.ylabel("y [pix]")
    plt.colorbar()
    plt.savefig(depth_dir + '/depth_%s.png' % band)
    plt.clf()

# depth codes (taken from print_depth_table.py)

def calc_5sigma_depth(x_pix, y_pix, im_data, r, zero_point, subpix = 5): # not sure how to get a tqdm progress bar for this
    flux, fluxerr, flag = sep.sum_circle(im_data, x_pix, y_pix, r, subpix = subpix)
    if len(flux) == 1:
        print("len(flux)=1")
    med_flux = np.nanmedian(flux)
    mad_5sigma_flux = np.nanmedian(abs(flux - med_flux)) * 1.4826 * 5
    #print(mad_5sigma_flux)
    if mad_5sigma_flux > 0.:
        depth_5sigma = -2.5 * np.log10(mad_5sigma_flux) + zero_point
    else:
        depth_5sigma = np.nan
    return depth_5sigma

# local depth sextractor catalogue (from correct_sextractor_photometry.py)

def calc_loc_depths(ra_gal, dec_gal, aper_coords_loc, xcoord, ycoord, im_data, r, survey, band, separation = 10 * u.deg, n_samples = 1, n_aper = 200, plot = False, zero_point = None):
    
    start_time = time.time()
    #print(len(ra_gal))
    loc_depths = []
    rounded_sample_size = int((len(ra_gal) / n_samples) + 1)
    for n in tqdm(range(n_samples), desc = f"{band} progress", total = n_samples):
        ra_gal_sample = ra_gal[n * rounded_sample_size : (n + 1) * rounded_sample_size]
        dec_gal_sample = dec_gal[n * rounded_sample_size : (n + 1) * rounded_sample_size]
        if n == n_samples:
            ra_gal_sample = ra_gal[n * rounded_sample_size :]
            dec_gal_sample = dec_gal[n * rounded_sample_size :]
        #print(len(ra_gal_sample))
        
        gal_coords = SkyCoord(ra = ra_gal_sample * u.degree, dec = dec_gal_sample * u.degree)
        idx1, idx2, sep2d, dist3d = search_around_sky(gal_coords, aper_coords_loc, separation)
        for i in tqdm(range(len(gal_coords)), desc = f"Calculating local depth sample {n}", leave = False):
            aper_idx = idx2[i * len(aper_coords_loc) : (i + 1) * len(aper_coords_loc)]
            sep2d_loc = sep2d[i * len(aper_idx) + aper_idx]
            aper_idx_sorted = (np.argsort(sep2d_loc))[0 : n_aper] # use only closest n_aper apertures for each galaxy
            xcoord_loc = xcoord[aper_idx_sorted]
            ycoord_loc = ycoord[aper_idx_sorted]
            loc_depths.append(calc_5sigma_depth(xcoord_loc, ycoord_loc, im_data, r, zero_point))
            
    end_time = time.time()
    print("Local depth calculation took {} seconds!".format(np.round(end_time - start_time, 2)))
    return loc_depths

def make_new_fits_columns(orig_fits_table, col_name, col_data, col_format, save = False, save_name = None):
    fits_columns = []
    column_labels = orig_fits_table.columns.names
    column_formats = orig_fits_table.columns.formats
    for i in range(len(column_labels)):
        loc_col = fits.Column(name = column_labels[i], array = orig_fits_table[column_labels[i]], format = column_formats[i])
        fits_columns.append(loc_col)
        #print(column_labels[i])
    for i in range(len(col_name)):
        new_col = fits.Column(name = col_name[i], array = col_data[i], format = col_format[i])
        fits_columns.append(new_col)
    out_fits_table = fits.BinTableHDU.from_columns(fits_columns)
    if save:
        out_fits_table.writeto(save_name, overwrite = True)
    return out_fits_table.data


# def correct_medium_band_syntax(fits_table, save_name, old_band_name):
#     if log.lower_case_m[survey]: # if medium band correction is required
#         new_band_name = old_band_name[:-1] + "M"
#         fits_columns = []
#         column_labels = fits_table.columns.names
#         column_formats = fits_table.columns.formats
#         for i in range(len(column_labels)):
#             # change the lower case m for an upper case M
#             if column_labels[i][-len(old_band_name):] == old_band_name:
#                 column_labels[i] = column_labels[i][:-len(old_band_name)] + new_band_name
#             loc_col = fits.Column(name = column_labels[i], array = fits_table[column_labels[i]], format = column_formats[i])
#             fits_columns.append(loc_col)
#         out_fits_table = fits.BinTableHDU.from_columns(fits_columns)
#         out_fits_table.writeto(save_name, overwrite = True)
        
def is_blank_survey(survey):
    cluster_surveys = ["El-Gordo", "MACS-0416", "CLIO", "SMACS-0723"]
    if survey in cluster_surveys:
        return False
    else:
        return True

if __name__ == "__main__":
    pass

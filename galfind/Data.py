#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:20:31 2023

@author: austind
"""

from __future__ import absolute_import
from photutils import Background2D, MedianBackground, SkyCircularAperture, aperture_photometry
import numpy as np
from astropy.io import fits
from random import randrange
from pathlib import Path
import sep # sextractor for python
import matplotlib.pyplot as plt
import math
import timeit
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.wcs.utils import pixel_to_skycoord
from astropy.coordinates import search_around_sky, SkyCoord
from matplotlib.colors import LogNorm
from astropy.table import Table, hstack, Column
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
from joblib import Parallel, delayed
import contextlib
import joblib
import h5py
from tqdm import tqdm
import logging

from .Instrument import Instrument, ACS_WFC, WFC3_IR, NIRCam, MIRI, Combined_Instrument
from . import config
from . import Depths
from . import useful_funcs_austind as funcs
from .decorators import run_in_dir, hour_timer, email_update
from . import galfind_logger

# GALFIND data object
class Data:
    
    def __init__(self, instrument, im_paths, im_exts, im_pixel_scales, im_shapes, im_zps, wht_paths, wht_exts, rms_err_paths, rms_err_exts, \
        seg_paths, mask_paths, cluster_mask_path, blank_mask_path, survey, version, cat_path = "", is_blank = True, alignment_band = "f444W"):
        # self, instrument, im_paths, im_exts, seg_paths, mask_paths, cluster_mask_path, blank_mask_path, survey, version = "v0", is_blank = True):
        
        # sort dicts from blue -> red bands in ascending wavelength order
        self.im_paths = dict(sorted(im_paths.items()))
        self.im_exts = dict(sorted(im_exts.items())) # science image extension
        self.survey = survey
        self.version = version
        self.instrument = instrument
        self.is_blank = is_blank
        self.im_zps = im_zps
        self.wht_paths = wht_paths
        self.wht_exts = wht_exts
        self.rms_err_paths = rms_err_paths
        self.rms_err_exts = rms_err_exts
        self.im_pixel_scales = im_pixel_scales
        self.im_shapes = im_shapes
        # ensure alignment band exists
        if alignment_band not in self.instrument.bands:
            galfind_logger.critical(f"Alignment band = {alignment_band} does not exist in instrument!")
        else:
            self.alignment_band = alignment_band
        if cat_path == "":
            pass
        elif type(cat_path) == str:
            self.cat_path = cat_path
        else:
            raise(Exception(f"cat_path = {cat_path} has type = '{type(cat_path)}', which is not 'str'!"))

        # make segmentation maps from image paths if they don't already exist
        for i, (band, seg_path) in enumerate(seg_paths.items()):
            #print(band, seg_path)
            if (seg_path == "" or seg_path == []):
                self.make_seg_map(band)
            # load segmentation map
            seg_paths[band] = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/SExtractor/{self.instrument.instrument_from_band(band).name}/{version}/{survey}/{survey}*{band}_{band}*{version}*seg.fits")[0]
        self.seg_paths = dict(sorted(seg_paths.items())) 
        # make masks from image paths if they don't already exist
        self.mask_dir = f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}"
        for i, (band, mask_path) in enumerate(mask_paths.items()):
            # the input mask path is a pixel mask
            if ".fits" in mask_path:
                pass
            # convert region mask to pixel mask
            elif ".reg" in mask_path:
                # clean region mask of any zero size regions
                mask_path = self.clean_mask_regions(mask_path)
                mask_paths[band] = self.mask_reg_to_pix(band, mask_path)
            else:
                # make an pixel mask automatically for the band
                galfind_logger.critical(f"No .fits or .reg mask for {survey} {band} and not yet implemented auto-masking.")
        self.mask_paths = dict(sorted(mask_paths.items()))

        if is_blank:
            galfind_logger.info(f"{survey} is a BLANK field!")
            self.blank_mask_path = ""
            self.cluster_mask_path = ""
        else:
            galfind_logger.info(f"{survey} is a CLUSTER field!")
            for mask_path, mask_type in zip([blank_mask_path, cluster_mask_path], ["blank", "cluster"]):
                if ".fits" in mask_path:
                    pass
                elif ".reg" in mask_path:
                    mask_path = self.clean_mask_regions(mask_path)
                    mask_path = self.mask_reg_to_pix(self.alignment_band, mask_path)
                else:
                    galfind_logger.critical(f"{mask_type.capitalize()} mask does not exist for {survey} and no auto-masking has yet been implemented!")
            self.blank_mask_path = blank_mask_path
            self.cluster_mask_path = cluster_mask_path

    @classmethod
    def from_pipeline(cls, survey, version = "v9", instruments = ['NIRCam', 'ACS_WFC', 'WFC3_IR'], excl_bands = [], pix_scales = ['30mas', '60mas']):
        instruments_obj = {'NIRCam': NIRCam(excl_bands = excl_bands), 'ACS_WFC': ACS_WFC(excl_bands = excl_bands), 'WFC3_IR': WFC3_IR(excl_bands = excl_bands)}
        # Build a combined instrument object
        comb_instrument_created = False
        
        im_paths = {} 
        im_exts = {}
        seg_paths = {}
        wht_paths = {}
        wht_exts = {}
        rms_err_paths = {}
        rms_err_exts = {}
        im_pixel_scales = {}
        im_zps = {}
        im_shapes = {}
        mask_paths = {}
        depth_dir = {}
        is_blank = is_blank_survey(survey)

        for instrument in instruments:
            instrument = instruments_obj[instrument]
            if instrument.name == "NIRCam":
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
                    ceers_im_dirs = {f"CEERSP{str(i + 1)}": f"CEERSP{str(i + 1)}/mosaic_1084_182" for i in range(10)}
                    survey_im_dirs = {"CLIO": "CLIO/mosaic_1084_182", "El-Gordo": "elgordo/mosaic_1084_182", "NEP-1": "NEP-1/mosaic_1084_182", "NEP-2": "NEP-2/mosaic_1084_182", \
                                      "NEP-3": "NEP-3/mosaic_1084_182", "NEP-4": "NEP-4/mosaic_1084_182", "MACS-0416": "MACS0416/mosaic_1084_182", "GLASS": "GLASS-12/mosaic_1084_182", "SMACS-0723": "SMACS0723/mosaic_1084_182"} | ceers_im_dirs
                elif version == "v8b":
                    survey_im_dirs = {survey: f"{survey}/mosaic_1084_wispfix"}
                elif version == "v8c":
                    survey_im_dirs = {survey: f"{survey}/mosaic_1084_wispfix2"}
                elif version == "v8d":
                    survey_im_dirs = {survey: f"{survey}/mosaic_1084_wispfix3"}
                elif version == "v8e" or version == "v8f" or version == "v9":
                    survey_im_dirs = {survey: f"{survey}/mosaic_1084_wisptemp2"}
                elif version == "lit_version":
                    survey_im_dirs = {"JADES-DR1": "JADES/DR1"}
                elif version == 'v9' or version[:2] == "v9":
                    survey_im_dirs = {survey: f"{survey}/mosaic_1084_wisptemp2"}
                elif version == 'v10' or version[:2] == "v10":
                    survey_im_dirs = {survey: f"{survey}/mosaic_1084_wispscale"}
                elif version == "v11" or version[:-2] == "v11":
                    survey_im_dirs = {survey: f"{survey}/mosaic_1084_wispnathan"}
                survey_im_dirs = {key: f"/raid/scratch/data/jwst/{value}" for (key, value) in survey_im_dirs.items()}
                survey_dir = survey_im_dirs[survey]

                if version == "lit_version":
                    im_path_arr = np.array(glob.glob(f"{survey_dir}/*_drz.fits"))
                else:
                    im_path_arr = np.array(glob.glob(f"{survey_dir}/*_i2d*.fits"))

                # obtain available bands from imaging without having to hard code these
                bands = np.array([split_path.lower().replace("w", "W").replace("m", "M") for path in im_path_arr for i, split_path in \
                        enumerate(path.split("-")[-1].split("/")[-1].split("_")) if split_path.lower().replace("w", "W").replace("m", "M") in instrument.bands])
                
                # If band not used in instrument, remove it
                for band in instrument.bands:
                    if band not in bands:
                        instrument.remove_band(band)
                    else:
                        # Maybe generalize this
                        #print("Generalize on line 177 of Data.from_pipeline()")
                        im_pixel_scales[band] = 0.03 
                        im_zps[band] = 28.08
                        galfind_logger.debug(f"im_zp[{band}] = 28.08 only for pixel scale of 0.03 arcsec! This will change if different images are used!")
                # If no images found for this instrument, don't add it to the combined instrument
                if len(bands) != 0:
                    if comb_instrument_created:
                        comb_instrument += instrument
                    else:
                        comb_instrument = instrument
                        comb_instrument_created = True

                for i, band in enumerate(bands):
                    # obtains all image paths from the correct band 
                    im_paths_band = [im_path for im_path in im_path_arr if band.lower() in im_path or band in im_path or \
                                      band.replace("f", "F").replace("W", "w") in im_path or band.upper() in im_path]
                    # checks to see if there is just one singular image for the given band
                    if len(im_paths_band) == 1:
                        im_paths[band] = im_paths_band[0]
                    else:
                        raise(Exception(f"Multiple images found for {band} in {survey} {version}"))
                    
                    im_hdul = fits.open(im_paths[band])
                    # obtain appropriate extension from the image
                    for j, im_hdu in enumerate(im_hdul):
                        #print(im_hdu.name)
                        if im_hdu.name == "SCI":
                            im_exts[band] = int(j)
                            im_shapes[band] = im_hdu.data.shape
                        if im_hdu.name == 'WHT':
                            wht_exts[band] = int(j)
                            wht_paths[band] = str(im_paths[band])
                        if im_hdu.name == 'ERR':
                            rms_err_exts[band] = int(j)
                            rms_err_paths[band] = str(im_paths[band])
                        
                    # need to change this to work if there are no segmentation maps (with the [0] indexing)

            elif instrument.name in ["ACS_WFC", 'WFC3_IR']:
                # Iterate through bands and check if images exist 
                any_path_found = False
                for band in instrument.bands:
                    path_found = False
                    for pix_scale in pix_scales:
                        glob_paths = glob.glob(f"{config['DEFAULT']['GALFIND_DATA']}/hst/{survey}/{instrument.name}/{pix_scale}/*{band.replace('W', 'w').replace('M', 'm')}*_drz.fits")
                        glob_paths += glob.glob(f"{config['DEFAULT']['GALFIND_DATA']}/hst/{survey}/{instrument.name}/{pix_scale}/*{band}*_drz.fits")
                        # Make sure no duplicates
                        glob_paths = list(set(glob_paths))
                       
                        if len(glob_paths) == 0:
                            galfind_logger.debug(f"No image path found for {survey} {version} {band} {pix_scale}!")
                        elif len(glob_paths) == 1:
                            path = Path(glob_paths[0])
                            any_path_found = True
                            path_found = True
                            break
                        else:
                            raise(Exception("Multiple image paths found for {survey} {version} {band} {pix_scale}!"))

                    # If no images found, remove band from instrument
                    if not path_found:
                        instrument.remove_band(band)
                    else:
                        # otherwise open band, work out if it has a weight map, calc zero point and image scale
                        hdul = fits.open(str(path))
                        
                        im_paths[band] = str(path)
                        # Not great to use try/except but not sure how else to do it with index_of
                        try:
                            im_exts[band] = hdul.index_of('SCI')
                        except KeyError:
                            #print(f"No 'SCI' extension for {band} image. Default to im_ext = 0!")
                            im_exts[band] = 0
                        # Get header of image extension
                        imheader = hdul[im_exts[band]].header
                        im_shapes[band] = hdul[im_exts[band]].data.shape

                        for map_paths, map_exts, map_type in zip([wht_paths, rms_err_paths], [wht_exts, rms_err_exts], ["WHT", "ERR"]):
                            try:
                                map_exts[band] = hdul.index_of(map_type)
                                map_paths[band] = str(path)
                            except KeyError:
                                glob_paths = glob.glob(f"{config['DEFAULT']['GALFIND_DATA']}/hst/{survey}/{instrument.name}/{pix_scale}/*{band.replace('W', 'w').replace('M', 'm')}*_{map_type.lower()}.fits")
                                glob_paths += glob.glob(f"{config['DEFAULT']['GALFIND_DATA']}/hst/{survey}/{instrument.name}/{pix_scale}/*{band}*_{map_type.lower()}.fits")
                                if map_type == "ERR":
                                    glob_paths += glob.glob(f"{config['DEFAULT']['GALFIND_DATA']}/hst/{survey}/{instrument.name}/{pix_scale}/*{band}*_rms.fits")
                                # Make sure no duplicates
                                glob_paths = list(set(glob_paths))
                                if len(glob_paths) == 1:
                                    map_paths[band] = str(Path(glob_paths[0]))
                                    map_exts[band] = 0
                                elif len(glob_paths) > 1:
                                    galfind_logger.critical(f"Multiple {map_type.lower()} image paths found for {survey} {version} {band} {pix_scale}!")
                                    raise(Exception(f"Multiple {map_type.lower()} image paths found for {survey} {version} {band} {pix_scale}!"))
                                else:
                                    galfind_logger.debug(f"No wht image path found for {survey} {version} {band} {pix_scale}!")
                        hdul.close()
                        # if there is neither a wht or rms_err map, raise exception
                        if band not in wht_paths.keys() and band not in rms_err_paths.keys():
                            galfind_logger.critical(f"No wht or rms_err map for {survey} {version} {band} {pix_scale}!")
                            raise(Exception(f"No wht or rms_err map for {survey} {version} {band} {pix_scale}!"))

                        #print(band, wht_paths[band])
                        im_pixel_scales[band] = float(pix_scale.split('mas')[0]) * 1e-3 
                        if instrument.name == 'ACS_WFC':
                            if "PHOTFLAM" in imheader and "PHOTPLAM" in imheader:
                                im_zps[band] = -2.5 * np.log10(imheader["PHOTFLAM"]) - 21.10 - 5 * np.log10(imheader["PHOTPLAM"]) + 18.6921
                            elif "ZEROPNT" in imheader:
                                im_zps[band] = imheader["ZEROPNT"]
                            else:
                                raise(Exception(f"ACS_WFC data for {survey} {version} {band} located at {im_paths[band]} must contain either 'ZEROPNT' or 'PHOTFLAM' and 'PHOTPLAM' in its header to calculate its ZP!"))
                            
                        elif instrument.name == 'WFC3_IR':
                        # Taken from Appendix A of https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/documentation/instrument-science-reports-isrs/_documents/2020/WFC3-ISR-2020-10.pdf
                            wfc3ir_zps = {'f098M': 25.661, 'f105W': 26.2637, 'f110W': 26.8185, 'f125W': 26.231, 'f140W': 26.4502, 'f160W': 25.9362}
                            im_zps[band] = wfc3ir_zps[band]
                        # Need to move my segmentation maps and masks to the correct place

                if any_path_found:
                    if comb_instrument_created:
                        comb_instrument += instrument
                        galfind_logger.debug(f'Added instrument = {instrument.name}')
                    else:
                        comb_instrument = instrument
                        comb_instrument_created = True
                        galfind_logger.debug("Making combined_instrument")

                        # Need to update what it suggests
            elif instrument.name == 'MIRI':
                raise NotImplementedError("MIRI not yet implemented")

        if comb_instrument_created:
            # All seg maps and masks should be in same format, so load those last when we know what bands we have
            for band in comb_instrument.bands:
                try:
                    #print(f"{config['DEFAULT']['GALFIND_WORK']}/SExtractor/{comb_instrument.instrument_from_band(band)}/{version}/{survey}/{survey}*{band}_{band}*{version}*seg.fits")
                    seg_paths[band] = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/SExtractor/{comb_instrument.instrument_from_band(band).name}/{version}/{survey}/{survey}*{band}_{band}*{version}*seg.fits")[0]
                except IndexError:
                    seg_paths[band] = ""
                # include just the masks corresponding to the correct bands
                fits_mask_paths_ = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/fits_masks/*{band.replace('W', 'w').replace('M', 'm')}*")
                if len(fits_mask_paths_) == 1:
                    mask_paths[band] = fits_mask_paths_[0]
                elif len(fits_mask_paths_) == 0:
                    mask_paths_ = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/*{band.replace('W', 'w').replace('M', 'm')}*")
                    if len(mask_paths_) == 0:
                        mask_paths[band] = ""
                    elif len(mask_paths_) == 1:
                        mask_paths[band] = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/*{band.replace('W', 'w').replace('M', 'm')}*")[0]
                    else:
                        raise(Exception(f"Too many region masks found for {survey} {band}!"))
                else:
                    raise(Exception(f"Too many fits masks found for {survey} {band}!"))
            
            if is_blank:
                cluster_mask_path = ""
                blank_mask_path = ""
            else: # load in cluster core / blank field fits/reg masks
                mask_path_dict = {}
                for mask_type in ["cluster", "blank"]:
                    fits_masks = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/fits_masks/*_{mask_type}*.fits")
                    galfind_logger.debug(f"Available {mask_type} .fits masks for {survey} = {fits_masks}")
                    if len(fits_masks) == 1:
                        mask_path = fits_masks[0]
                    elif len(fits_masks) > 1:
                        galfind_logger.critical(f"Multiple .fits {mask_type} masks exist for {survey}!")
                    else: # no .fits masks, now look for .reg masks
                        galfind_logger.info(f"No .fits {mask_type} masks exist for {survey}. Searching for .reg masks")
                        reg_masks = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/*_{mask_type}*.reg")
                        galfind_logger.debug(f"Available {mask_type} .reg masks for {survey} = {reg_masks}")
                        if len(reg_masks) == 1:
                            mask_path = reg_masks[0]
                        elif len(reg_masks) > 1:
                            galfind_logger.critical(f"Multiple .reg {mask_type} masks exist for {survey}!")
                        else: # no .reg masks
                            galfind_logger.warning(f"No .fits or .reg {mask_type} masks exist for {survey}. May cause catalogue masking issues!")
                    mask_path_dict[mask_type] = mask_path
                cluster_mask_path = mask_path_dict["cluster"]
                galfind_logger.debug(f"cluster_mask_path = {cluster_mask_path}")
                blank_mask_path = mask_path_dict["blank"]
                galfind_logger.debug(f"blank_mask_path = {blank_mask_path}")

            return cls(comb_instrument, im_paths, im_exts, im_pixel_scales, im_shapes, im_zps, wht_paths, wht_exts, rms_err_paths, rms_err_exts, \
                seg_paths, mask_paths, cluster_mask_path, blank_mask_path, survey, version, is_blank = is_blank)
        else:
            raise(Exception(f'Failed to find any data for {survey}'))  

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
        galfind_logger.debug(self.__repr__)
        return self
    
# %% Methods

    def load_data(self, band, incl_mask = True):
        if type(band) not in [str, np.str_]:
            galfind_logger.debug(f"band = {band}, type(band) = {type(band)} not in [str, np.str_] in Data.load_data")
            band = self.combine_band_names(band)
        # load science image data and header (and hdul)
        im_data, im_header = self.load_im(band)
        # load segmentation data and header
        seg_data, seg_header = self.load_seg(band)
        if incl_mask:
            # load mask
            mask = self.load_mask(band)
            return im_data, im_header, seg_data, seg_header, mask
        else:
            return im_data, im_header, seg_data, seg_header

    def load_mask(self, mask_band):
        if ".fits" in self.mask_paths[mask_band]:
            mask = fits.open(self.mask_paths[mask_band])[1].data
        else:
            galfind_logger.critical(f"Mask for {self.survey} {mask_band} at {self.mask_paths[mask_band]} is not a .fits mask!")
        return mask
    
    def load_im(self, band, return_hdul = False):
        # load image data and header
        im_hdul = fits.open(self.im_paths[band])
        im_data = im_hdul[self.im_exts[band]].data
        im_data = im_data.byteswap().newbyteorder()
        im_header = im_hdul[self.im_exts[band]].header
        if return_hdul:
            return im_data, im_header, im_hdul
        else:
            return im_data, im_header

    def load_seg(self, band):
        seg_hdul = fits.open(self.seg_paths[band])
        seg_data = seg_hdul[0].data
        seg_header = seg_hdul[0].header
        return seg_data, seg_header
    
    def load_wht(self, band):
        try:
            wht = fits.open(self.wht_paths[band])[self.wht_exts[band]].data
        except:
            wht = None
        return wht
    
    def combine_seg_data_and_mask(self, band = None, seg_data = None, mask = None):
        if type(seg_data) != type(None) and type(mask) != type(None):
            pass
        elif type(band) != type(None): # at least one of seg_data or mask is not given, but band is given
            seg_data = self.load_seg(band)[0]
            mask = self.load_mask(band)
        else:
            raise(Exception("Either band must be given or both seg_data and mask should be given in Data.combine_seg_data_and_mask()!"))
        assert(seg_data.shape == mask.shape)
        combined_mask = np.logical_or(seg_data > 0, mask == 1).astype(int)
        return combined_mask

    def plot_image_from_band(self, ax, band, norm = LogNorm(vmin = 0., vmax = 10.), show = True):
        im_data = self.load_data(band, incl_mask = False)[0]
        self.plot_image_from_data(ax, im_data, band, norm, show)
        
    @staticmethod
    def plot_image_from_data(ax, im_data, label, norm = LogNorm(vmin = 0., vmax = 10.), show = True):
        ax.imshow(im_data, norm = norm, origin = "lower")
        plt.title(f"{label} image")
        plt.xlabel("X / pix")
        plt.ylabel("Y / pix")
        if show:
            plt.show()
            
    def plot_mask_from_band(self, ax, band, show = True):
        mask = self.load_data(band, incl_mask = True)[4]
        self.plot_mask_from_data(ax, mask, band, show)
        
    @staticmethod
    def plot_mask_from_data(ax, mask, label, show = True):
        cbar_in = ax.imshow(mask, origin = "lower")
        plt.title(f"{label} mask")
        plt.xlabel("X / pix")
        plt.ylabel("Y / pix")
        plt.colorbar(cbar_in)
        if show:
            plt.show()
    
    def plot_mask_regions_from_band(self, ax, band):
        im_header = self.load_data(band, incl_mask = False)[1]
        mask_path = self.mask_paths[band]
        mask_file = pyregion.open(mask_path).as_imagecoord(im_header)
        patch_list, artist_list = mask_file.get_mpl_patches_texts()
        for p in patch_list:
            ax.add_patch(p)
        for t in artist_list:
            ax.add_artist(t)
    
    #@staticmethod
    def combine_band_names(self, bands):
        return '+'.join(bands)
    
    def get_err_map(self, band, prefer = "rms_err"):
        """ Loads either the rms_err or wht map for use in SExtractor depending on the preferred map to use

        Args:
            band (str): Filter to extract the rms_err or wht maps from
            prefer (str, optional): Preferred error map type to use. Either 'rms_err' or 'wht', anything else throws critical. Defaults to "rms_err".
        Returns:
            tuple(3): (Error map path, Error map fits extension, Error map type)
        """
        
        # determine which error map to use based on what is available or preferred
        if prefer == "rms_err":
            if band in self.rms_err_paths.keys():
                err_map_type = "MAP_RMS"
            elif band in self.wht_paths.keys():
                err_map_type = "MAP_WEIGHT"
        elif prefer == "wht":
            if band in self.wht_paths.keys():
                err_map_type = "MAP_WEIGHT"
            elif band in self.rms_err_paths.keys():
                err_map_type = "MAP_RMS"
        else:
            galfind_logger.critical(f"prefer = {prefer} not in ['rms_err', 'wht'] in Data.get_err_map()")
        
        # extract relevant fits paths and extensions
        if err_map_type == "MAP_RMS":
            err_map_path = str(self.rms_err_paths[band])
            err_map_ext = str(self.rms_err_exts[band])
        elif err_map_type == "MAP_WEIGHT":
            err_map_path = str(self.wht_paths[band])
            err_map_ext = str(self.wht_exts[band])
        else:
            galfind_logger.critical(f"No rms_err or wht maps available for {band} {self.survey} {self.version}")

        return err_map_path, err_map_ext, err_map_type

    @run_in_dir(path = config['DEFAULT']['GALFIND_DIR'])
    def make_seg_map(self, band, sex_config_path = config['SExtractor']['CONFIG_PATH'], params_path = config['SExtractor']['PARAMS_PATH']):
        if type(band) == str or type(band) == np.str_:
            pass
        elif type(band) == list or type(band) == np.array:
            band = self.combine_band_names(band)
        else:
            raise(Exception(f"Cannot make segmentation map for {band}! type(band) = {type(band)} must be either str, list, or np.array!"))

        # load relevant err map paths, preferring rms_err maps if available
        err_map_path, err_map_ext, err_map_type = self.get_err_map(band, prefer = "rms_err")
        
        # SExtractor bash script python wrapper
        process = subprocess.Popen(["./make_seg_map.sh", config['DEFAULT']['GALFIND_WORK'], self.im_paths[band], str(self.im_pixel_scales[band]), \
                                str(self.im_zps[band]), self.instrument.instrument_from_band(band).name, self.survey, band, self.version, err_map_path, \
                                err_map_ext, err_map_type, str(self.im_exts[band]), sex_config_path, params_path])
        process.wait()
        galfind_logger.info(f"Made segmentation map for {self.survey} {self.version} {band} using config = {sex_config_path} and {err_map_type}")

    def make_seg_maps(self):
        for band in self.instrument.bands:
            self.make_seg_map(band)
    
    def stack_bands(self, bands):
        for band in bands:
            if band not in self.im_paths.keys():
                bands.remove(band)
                galfind_logger.warning(f"{band} not available for {self.survey} {self.version}")
        stack_band_name = self.combine_band_names(bands)
        detection_image_dir = f"{config['DEFAULT']['GALFIND_WORK']}/Stacked_Images/{self.version}/{self.instrument.instrument_from_band(bands[0]).name}/{self.survey}"
        detection_image_name = f"{self.survey}_{stack_band_name}_{self.version}_stack.fits"
        self.im_paths[stack_band_name] = f'{detection_image_dir}/{detection_image_name}'
        self.rms_err_paths[stack_band_name] = f'{detection_image_dir}/{detection_image_name}'
        glob_mask_names = glob.glob(f"{self.mask_dir}/{stack_band_name}_basemask.reg")
        #print(glob_mask_names)
        if len(glob_mask_names) == 0:
            self.mask_paths[stack_band_name] = self.combine_masks(bands)
        elif len(glob_mask_names) == 1:
            self.mask_paths[stack_band_name] = glob_mask_names[0]
        else:
            raise(Exception(f"More than 1 mask for {stack_band_name}. Please change this in {self.mask_dir}"))
        
        if not Path(self.im_paths[stack_band_name]).is_file():
            funcs.make_dirs(self.im_paths[stack_band_name])
            for pos, band in enumerate(bands):
                if self.im_shapes[band] != self.im_shapes[bands[0]] or self.im_zps[band] != self.im_zps[bands[0]] or self.im_pixel_scales[band] != self.im_pixel_scales[bands[0]]:
                    raise Exception('All bands used in forced photometry stack must have the same shape, ZP and pixel scale!')
                
                prime_hdu = fits.open(self.im_paths[band])[0].header
                im_data, im_header = self.load_im(band)
                err = fits.open(self.rms_err_paths[band])[self.rms_err_exts[band]].data
                if pos == 0:
                    sum = im_data / err ** 2
                    sum_err = 1 / err ** 2
                else:
                    sum += im_data / err ** 2
                    sum_err += 1 / err ** 2
                
            weighted_array = sum / sum_err
            
            #https://en.wikipedia.org/wiki/Inverse-variance_weighting
            combined_err = np.sqrt(1 / sum_err)

            primary = fits.PrimaryHDU(header = prime_hdu)
            hdu = fits.ImageHDU(weighted_array, header = im_header, name = 'SCI')
            hdu_err = fits.ImageHDU(combined_err, header = im_header, name = 'ERR')
            hdul = fits.HDUList([primary, hdu, hdu_err])
            hdul.writeto(self.im_paths[stack_band_name], overwrite = True)
            galfind_logger.info(f"Finished stacking bands = {bands} for {self.survey} {self.version}")
        
        # save forced photometry band parameters
        self.im_shapes[stack_band_name] = self.im_shapes[bands[0]]
        self.im_zps[stack_band_name] = self.im_zps[bands[0]]
        self.im_pixel_scales[stack_band_name] = self.im_pixel_scales[bands[0]]
        self.im_exts[stack_band_name] = 1
        self.rms_err_exts[stack_band_name] = 2

        # could compute a wht map from the rms_err map here!

    def sex_cat_path(self, band, forced_phot_band):
        # forced phot band here is the string version
        sex_cat_dir = f"{config['DEFAULT']['GALFIND_WORK']}/SExtractor/{self.instrument.instrument_from_band(band).name}/{self.version}/{self.survey}"
        sex_cat_name = f"{self.survey}_{band}_{forced_phot_band}_sel_cat_{self.version}.fits"
        sex_cat_path = f"{sex_cat_dir}/{sex_cat_name}"
        return sex_cat_path

    def seg_path(self, band):
        # IF THIS IS CHANGED MUST ALSO CHANGE THE PATH IN __init__ AND make_seg_map.sh
        return f"{config['DEFAULT']['GALFIND_WORK']}/SExtractor/{self.instrument.instrument_from_band(band).name}/{self.version}/{self.survey}/{self.survey}_{band}_{band}_sel_cat_{self.version}_seg.fits"

    @run_in_dir(path = config['DEFAULT']['GALFIND_DIR'])
    def make_sex_cats(self, forced_phot_band = "f444W", sex_config_path = config['SExtractor']['CONFIG_PATH'], params_path = config['SExtractor']['PARAMS_PATH']):
        galfind_logger.info(f"Making SExtractor catalogues with: config file = {sex_config_path}; parameters file = {params_path}")
        # make individual forced photometry catalogues
        if type(forced_phot_band) == list:
            if len(forced_phot_band) > 1:
                # make the stacked image and save all appropriate parameters
                galfind_logger.debug(f"forced_phot_band = {forced_phot_band}")
                self.stack_bands(forced_phot_band)
                self.forced_phot_band = self.combine_band_names(forced_phot_band)
                self.seg_paths[self.forced_phot_band] = self.seg_path(self.forced_phot_band)
                if not Path(self.seg_paths[self.forced_phot_band]).is_file():
                    self.make_seg_map(forced_phot_band)
            else:
                self.forced_phot_band = forced_phot_band[0]
        else:
            self.forced_phot_band = forced_phot_band
        
        if self.forced_phot_band not in self.instrument.bands:
            sextractor_bands = np.append(self.instrument.bands, self.forced_phot_band)
        else:
            sextractor_bands = self.instrument.bands
        
        sex_cats = {}

        for band in sextractor_bands:
            sex_cat_path = self.sex_cat_path(band, self.forced_phot_band)
            galfind_logger.debug(f"band = {band}, sex_cat_path = {sex_cat_path} in Data.make_sex_cats")
            
            # if not run before
            if not Path(sex_cat_path).is_file():
                
                # check whether the image of the forced photometry band and sextraction band have the same shape
                if self.im_shapes[self.forced_phot_band] == self.im_shapes[band]:
                    sextract = True
                else:
                    sextract = False
                # check whether the band and forced phot band have error maps with consistent types
                if sextract:
                    if band in self.rms_err_paths.keys() and self.forced_phot_band in self.rms_err_paths.keys():
                        prefer = "rms_err"
                    elif band in self.rms_err_paths.keys() and self.forced_phot_band in self.rms_err_paths.keys():
                        prefer = "wht"
                    else: # do not perform sextraction
                        sextract = False
                
                # perform sextraction
                if sextract:
                    # load relevant err map paths
                    err_map_path, err_map_ext, err_map_type = self.get_err_map(band, prefer = prefer)
                    # load relevant err map paths for the forced photometry band
                    forced_phot_band_err_map_path, forced_phot_band_err_map_ext, forced_phot_band_err_map_type = self.get_err_map(self.forced_phot_band, prefer = prefer)
                    assert(err_map_type == forced_phot_band_err_map_type) # should always be true
                
                    # SExtractor bash script python wrapper
                    process = subprocess.Popen(["./make_sex_cat.sh", config['DEFAULT']['GALFIND_WORK'], self.im_paths[band], str(self.im_pixel_scales[band]), \
                        str(self.im_zps[band]), self.instrument.instrument_from_band(band).name, self.survey, band, self.version, \
                        self.forced_phot_band, self.im_paths[self.forced_phot_band], err_map_path, err_map_ext, \
                        str(self.im_exts[band]), forced_phot_band_err_map_path, str(self.im_exts[self.forced_phot_band]), err_map_type, 
                        forced_phot_band_err_map_ext, sex_config_path, params_path])
                    process.wait()

                else: # use photutils
                    self.forced_photometry(band, self.forced_phot_band)
            
            galfind_logger.info(f"Finished making SExtractor catalogue for {self.survey} {self.version} {band}!")
            sex_cats[band] = sex_cat_path
        self.sex_cats = sex_cats
    
    def combine_sex_cats(self, forced_phot_band = "f444W"):
        self.make_sex_cats(forced_phot_band)

        save_name = f"{self.survey}_MASTER_Sel-{self.combine_band_names(forced_phot_band)}_{self.version}.fits"
        save_dir = f"{config['DEFAULT']['GALFIND_WORK']}/Catalogues/{self.version}/{self.instrument.name}/{self.survey}"
        self.sex_cat_master_path = f"{save_dir}/{save_name}"
        if not Path(self.sex_cat_master_path).is_file():
            if type(forced_phot_band) == np.array or type(forced_phot_band) == list:
                forced_phot_band_name = self.combine_band_names(forced_phot_band)
            else:
                forced_phot_band_name = forced_phot_band
            print("Loading cat", self.sex_cats, forced_phot_band_name)
            for i, (band, path) in reversed(enumerate(self.sex_cats.items())):
                tab = Table.read(path, character_as_bytes = False)
                if band == forced_phot_band_name:
                    ID_detect_band = tab["NUMBER"]
                    x_image_detect_band = tab["X_IMAGE"]
                    y_image_detect_band = tab["Y_IMAGE"]
                    ra_detect_band = tab["ALPHA_J2000"]
                    dec_detect_band = tab["DELTA_J2000"]
                # remove the duplicated IDs, X_IMAGE/Y_IMAGE and RA/DECs
                tab = remove_non_band_dependent_sex_params(tab)
                # load each one into an astropy table and update the column names by adding an "_FILT" suffix
                tab = add_band_suffix_to_cols(tab, band)
                # combine the astropy tables
                if i == 0:
                    master_tab = tab
                else:
                    try:
                        master_tab = hstack([master_tab, tab])
                    except Exception as e:
                        print(e)
                        print(path)
            # add the detection band parameters to the start of the catalogue
            master_tab.add_column(ID_detect_band, name = 'NUMBER', index = 0)
            master_tab.add_column(x_image_detect_band, name = 'X_IMAGE', index = 1)
            master_tab.add_column(y_image_detect_band, name = 'Y_IMAGE', index = 2)
            master_tab.add_column(ra_detect_band, name = 'ALPHA_J2000', index = 3)
            master_tab.add_column(dec_detect_band, name = 'DELTA_J2000', index = 4)
            
            # update table header

            # create galfind catalogue README

            # save table
            os.makedirs(save_dir, exist_ok = True)
            master_tab.write(self.sex_cat_master_path, format = "fits", overwrite = True)
            galfind_logger.info(f"Saved combined SExtractor catalogue as {self.sex_cat_master_path}")

    def make_sex_plusplus_cat(self):
        pass
    
    def forced_photometry(self, band, forced_phot_band, radii = [0.16, 0.25, 0.5, 0.75, 1.] * u.arcsec, ra_col = 'ALPHA_J2000', dec_col = 'DELTA_J2000', coord_unit = u.deg, id_col = 'NUMBER', x_col = 'X_IMAGE', y_col = 'Y_IMAGE'):
        # Read in sextractor catalogue
        catalog = Table.read(self.sex_cat_path(forced_phot_band, forced_phot_band), character_as_bytes = False)
        # Open image with correct extension and get WCS
        with fits.open(self.im_paths[band]) as hdul:
            im_ext = self.im_exts[band]
            image = hdul[im_ext].data
            wcs = WCS(hdul[im_ext].header)
            
        # Check types
        assert(type(image) == np.ndarray)
        assert(type(catalog) == Table)
        
        # Get positions from sextractor catalog
        ra = catalog[ra_col]
        dec = catalog[dec_col]
         # Make SkyCoord from catlog
        positions = SkyCoord(ra, dec, unit = coord_unit)
        #print('positions', positions)
        # Define radii in sky units
        # This checks if radii is iterable and if not makes it a list
        try:
            iter(radii)
        except TypeError:
            radii = [radii]
        apertures = []

        for rad in radii:
            aperture = SkyCircularAperture(positions, r = rad)
            apertures.append(aperture)
            # Convert to pixel using image WCS

        # Do aperture photometry
        #print(image, apertures, wcs)
        phot_table = aperture_photometry(image, apertures, wcs=wcs)
        assert(len(phot_table) == len(catalog))
        # Replace detection ID with catalog ID
        sky = SkyCoord(phot_table['sky_center'])
        phot_table['id'] = catalog[id_col]
        # Rename columns
        phot_table.rename_column('id', id_col )
        phot_table.rename_column('xcenter', x_col)
        phot_table.rename_column('ycenter', y_col)

        phot_table[ra_col] = sky.ra.to('deg')
        phot_table[dec_col] = sky.dec.to('deg')
        phot_table.remove_column('sky_center')
         
        colnames = [f'aperture_sum_{i}' for i in range(len(radii))]
        aper_tab = Column(np.array(phot_table[colnames].as_array().tolist()), name=f'FLUX_APER_{band}')
        phot_table['FLUX_APER'] = aper_tab
        phot_table['FLUXERR_APER'] = phot_table['FLUX_APER'] * -99
        phot_table['MAGERR_APER'] = phot_table['FLUX_APER'] * 99 
        
        # This converts the fluxes to magnitudes using the correct zp, and puts them in the same format as the sextractor catalogue
        mag_colnames  = []
        for pos, col in enumerate(colnames):
            name = f'MAG_APER_{pos}'
            phot_table[name] = -2.5 * np.log10(phot_table[col]) + self.im_zps[band]
            phot_table[name][np.isnan(phot_table[name])] = 99.
            mag_colnames.append(name)
        aper_tab = Column(np.array(phot_table[mag_colnames].as_array().tolist()), name=f'MAG_APER_{band}')
        phot_table['MAG_APER'] = aper_tab
        # Remove old columns
        phot_table.remove_columns(colnames)
        phot_table.remove_columns(mag_colnames)
        phot_table.write(self.sex_cat_path(band, forced_phot_band), format='fits', overwrite=True)
        
    def mask_reg_to_pix(self, band, mask_path):
        # open image corresponding to band
        im_data, im_header, seg_data, seg_header = self.load_data(band, incl_mask = False)
        # open .reg mask file
        mask_regions = pyregion.open(mask_path).as_imagecoord(im_header)
        # make 2D np.array boolean pixel mask
        pix_mask = np.array(mask_regions.get_mask(header = im_header, shape = im_data.shape), dtype = bool)
        # make .fits mask
        mask_hdu = fits.ImageHDU(pix_mask.astype(np.uint8), header = WCS(im_header).to_header(), name = 'MASK')
        hdu = fits.HDUList([fits.PrimaryHDU(), mask_hdu])
        out_path = f"{'/'.join(mask_path.split('/')[:-1])}/fits_masks/{mask_path.split('/')[-1].replace('_clean', '').replace('.reg', '')}.fits"
        os.makedirs("/".join(out_path.split("/")[:-1]), exist_ok = True)
        hdu.writeto(out_path, overwrite = True)
        galfind_logger.info(f"Created fits mask from manually created reg mask, saving as {out_path}")
        return out_path
    
    def combine_masks(self, bands):
        combined_mask = np.logical_or.reduce(tuple([self.load_mask(band) for band in bands]))
        assert(combined_mask.shape == self.load_mask(bands[-1]).shape)
        # wcs taken from the reddest band
        mask_hdu = fits.ImageHDU(combined_mask.astype(np.uint8), header = WCS(self.load_im(bands[-1])[1]).to_header(), name = 'MASK')
        hdu = fits.HDUList([fits.PrimaryHDU(), mask_hdu])
        out_path = f"{self.mask_dir}/fits_cats/{self.combine_band_names(bands)}_basemask.fits"
        os.makedirs("/".join(out_path.split("/")[:-1]), exist_ok = True)
        hdu.writeto(out_path, overwrite = True)
        galfind_logger.info(f"Created combined mask for {bands}")
        return out_path

    @staticmethod
    def clean_mask_regions(mask_path):
        # open region file
        if "_clean" not in mask_path:
            with open(mask_path, 'r') as f:
                lines = f.readlines()
                clean_mask_path = mask_path.replace(".reg", "_clean.reg")
                with open(clean_mask_path, 'w') as temp:          
                    for i, line in enumerate(lines):
                        # if line.startswith('physical'):
                        #     lines[i] = line.replace('physical', 'image')
                        if i <= 2:
                            temp.write(line)
                        if not (line.endswith(',0)\n') and line.startswith('circle')):
                            if (line.startswith('ellipse') and not line.endswith(',0)\n') and not (line.split(",")[3] == "0")) or line.startswith("box"):
                               temp.write(line)
            # insert original mask ds9 region file into an unclean folder
            os.makedirs(f"{funcs.split_dir_name(mask_path,'dir')}/unclean", exist_ok = True)
            os.rename(mask_path, f"{funcs.split_dir_name(mask_path,'dir')}/unclean/{funcs.split_dir_name(mask_path,'name')}")
            return clean_mask_path
        else:
            return mask_path
    
    # can be simplified with new masks
    def calc_unmasked_area(self, forced_phot_band = ["f277W", "f356W", "f444W"], masking_instrument_name = "NIRCam"):
        masking_bands = np.array([band for band in self.instrument.bands if band in Instrument.from_name(masking_instrument_name).bands])
        # make combined mask if required
        glob_mask_names = glob.glob(f"{self.mask_dir}/*{self.combine_band_names(masking_bands)}_*")
        if len(glob_mask_names) == 0:
            self.mask_paths[masking_instrument_name] = self.combine_masks(masking_bands)
        elif len(glob_mask_names) == 1:
            self.mask_paths[masking_instrument_name] = glob_mask_names[0]
        else:
            raise(Exception(f"More than 1 mask for {masking_bands}. Please change this in {self.mask_dir}"))
        # open detection image
        if type(forced_phot_band) not in [str, np.str_]:
            forced_phot_band = self.combine_band_names(forced_phot_band)

        pixel_scale = self.im_pixel_scales[forced_phot_band]
        try:
            pixel_scale.unit
        except:
            pixel_scale = pixel_scale * u.arcsec
        print(f"pixel_scale = {pixel_scale}")
        
        full_mask = self.load_mask(masking_instrument_name, forced_phot_band)
        if self.is_blank:
            blank_mask = full_mask
        else:
            # make combined mask for masking_instrument_name blank field area
            glob_mask_names = glob.glob(f"{self.mask_dir}/*{self.combine_band_names(list(masking_bands) + ['blank'])}_*")
            if len(glob_mask_names) == 0:
                self.mask_paths[f"{masking_instrument_name}+blank"] = self.combine_masks(list(masking_bands) + ["blank"])
            elif len(glob_mask_names) == 1:
                self.mask_paths[f"{masking_instrument_name}+blank"] = glob_mask_names[0]
            else:
                raise(Exception(f"More than 1 mask for {masking_bands}. Please change this in {self.mask_dir}"))
            blank_mask = self.load_mask(f"{masking_instrument_name}+blank", forced_phot_band)
        
        unmasked_area_blank_modules = (((blank_mask.shape[0] * blank_mask.shape[1]) - np.sum(blank_mask)) * pixel_scale * pixel_scale).to(u.arcmin ** 2)
        print(f"unmasked_area_blank_modules = {unmasked_area_blank_modules}")
        output_path = f"/{config['DEFAULT']['GALFIND_WORK']}/Unmasked_areas/{masking_instrument_name}/{self.survey}_unmasked_area_{self.version}.txt"
        funcs.make_dirs(output_path)
        f = open(output_path, "w")
        f.write(f"# {self.survey} {self.version}, bands = {self.instrument.bands}; UNIT = {unmasked_area_blank_modules.unit}\n")
        f.write(f"{str(np.round(unmasked_area_blank_modules.value, 2))} # unmasked blank")
        if not self.is_blank:
            full_mask = self.load_mask(masking_instrument_name, forced_phot_band)
            unmasked_area_tot = (((full_mask.shape[0] * full_mask.shape[1]) - np.sum(full_mask)) * pixel_scale * pixel_scale).to(u.arcmin ** 2)
            unmasked_area_cluster_module = unmasked_area_tot - unmasked_area_blank_modules
            f.write("\n")
            f.write(f"{str(np.round(unmasked_area_cluster_module.value, 2))} # unmasked cluster")
        f.close()
        return unmasked_area_blank_modules

    def perform_aper_corrs(self): # not general
        overwrite = config["Depths"].getboolean("OVERWRITE_LOC_DEPTH_CAT")
        if overwrite:
            galfind_logger.info("OVERWRITE_LOC_DEPTH_CAT = YES, updating catalogue with aperture corrections.")
        cat = Table.read(self.sex_cat_master_path)
        if not "APERCORR" in cat.meta.keys() or overwrite:
            for i, band in enumerate(self.instrument.bands):
                mag_aper_corr_data = np.zeros(len(cat))
                flux_aper_corr_data = np.zeros(len(cat))
                for j, aper_diam in enumerate(json.loads(config.get("SExtractor", "APERTURE_DIAMS")) * u.arcsec):
                    # assumes these have already been calculated for each band
                    mag_aper_corr_factor = self.instrument.aper_corr(aper_diam, band)
                    flux_aper_corr_factor = 10 ** (mag_aper_corr_factor / 2.5)
                    #print(band, aper_diam, mag_aper_corr_factor, flux_aper_corr_factor)
                    if j == 0:
                        # only aperture correct if flux is positive
                        mag_aper_corr_data = [(mag_aper[0] - mag_aper_corr_factor,) if flux_aper[0] > 0. else (mag_aper[0],) \
                            for mag_aper, flux_aper in zip(cat[f"MAG_APER_{band}"], cat[f"FLUX_APER_{band}"])]
                        flux_aper_corr_data = [(flux_aper[0] * flux_aper_corr_factor,) if flux_aper[0] > 0. else (flux_aper[0],) \
                            for flux_aper in cat[f"FLUX_APER_{band}"]]
                    else:
                        mag_aper_corr_data = [mag_aper_corr + (mag_aper[j] - mag_aper_corr_factor,) if flux_aper[j] > 0. else mag_aper_corr + (mag_aper[j],) \
                            for mag_aper_corr, mag_aper, flux_aper in zip(mag_aper_corr_data, cat[f"MAG_APER_{band}"], cat[f"FLUX_APER_{band}"])]
                        flux_aper_corr_data = [flux_aper_corr + (flux_aper[j] * flux_aper_corr_factor,) if flux_aper[j] > 0. else flux_aper_corr + (flux_aper[j],) \
                            for flux_aper_corr, flux_aper in zip(flux_aper_corr_data, cat[f"FLUX_APER_{band}"])]
                cat[f"MAG_APER_{band}_aper_corr"] = mag_aper_corr_data
                cat[f"FLUX_APER_{band}_aper_corr"] = flux_aper_corr_data
                cat[f"FLUX_APER_{band}_aper_corr_Jy"] = [tuple([funcs.flux_image_to_Jy(val, self.im_zps[band]) for val in element]) for element in cat[f"FLUX_APER_{band}_aper_corr"]]

        # update catalogue metadata
        #mag_aper_corrs = {f"HIERARCH Mag_aper_corrs_{aper_diam.value}as": tuple([np.round(self.instrument.aper_corr(aper_diam, band), decimals = 4) \
        #    for band in self.instrument.bands]) for aper_diam in json.loads(config.get("SExtractor", "APERTURE_DIAMS")) * u.arcsec}
        cat.meta = {**cat.meta, **{"APERCORR": True}} #, **mag_aper_corrs}
        # overwrite original catalogue with local depth columns
        cat.write(self.sex_cat_master_path, overwrite = True)    

    def make_loc_depth_cat(self, cat_creator, depth_mode = "n_nearest"):
        overwrite = config["Depths"].getboolean("OVERWRITE_LOC_DEPTH_CAT")
        if overwrite:
            galfind_logger.info("OVERWRITE_LOC_DEPTH_CAT = YES, updating catalogue with local depths.")
        
        cat = Table.read(self.sex_cat_master_path)
        # update catalogue with local depths if not already done so
        if "DEPTHS" not in cat.meta.keys() or overwrite:
            #mean_depths = {}
            #median_depths = {}
            diagnostic_name = ""
            for i, band in enumerate(self.instrument.bands):
                for j, aper_diam in enumerate(json.loads(config.get("SExtractor", "APERTURE_DIAMS")) * u.arcsec):
                    self.get_depth_dir(aper_diam)
                    #print(band, aper_diam)
                    h5_path = f"{self.depth_dirs[band]}/{depth_mode}/{band}.h5"
                    if Path(h5_path).is_file():
                        # open depth .h5
                        hf = h5py.File(h5_path, "r")
                        depths = np.array(hf["depths"])
                        diagnostics = np.array(hf["diagnostic"])
                        diagnostic_name_ = f"d_{int(np.array(hf['n_nearest']))}" if depth_mode == "n_nearest" \
                            else f"n_aper_{float(np.array(hf['region_radius_used_pix'])):.1f}" if depth_mode == "rolling" else None
                        if diagnostic_name_ == None:
                            raise(Exception("Invalid mode!"))
                        # make sure the same depth setup has been run in each band
                        if i == 0 and j == 0:
                            diagnostic_name = diagnostic_name_
                        assert(diagnostic_name_ == diagnostic_name)
                        # update depths with average depths in each region
                        nmad_grid = np.array(hf["nmad_grid"])
                        band_mean_depth = np.round(np.nanmean(nmad_grid), decimals = 3)
                        band_median_depth = np.round(np.nanmedian(nmad_grid), decimals = 3)
                        hf.close()
                    else:
                        depths = np.full(len(cat), np.nan)
                        diagnostics = np.full(len(cat), np.nan)
                        #band_mean_depth = np.nan
                        #band_median_depth = np.nan
                    if j == 0:
                        band_depths = [(depth,) for depth in depths]
                        band_diagnostics = [(diagnostic,) for diagnostic in diagnostics]
                        band_sigmas = [(funcs.n_sigma_detection(depth, mag_aper[0], self.im_zps[band]),) for depth, mag_aper in zip(depths, cat[f"MAG_APER_{band}"])]
                        #band_mean_depths = (band_mean_depth,)
                        #band_median_depths = (band_median_depth,)
                    else:
                        band_depths = [band_depth + (aper_diam_depth,) for band_depth, aper_diam_depth in zip(band_depths, depths)]
                        band_diagnostics = [band_diagnostic + (aper_diam_diagnostic,) for band_diagnostic, aper_diam_diagnostic in zip(band_diagnostics, diagnostics)]
                        band_sigmas = [band_sigma + (funcs.n_sigma_detection(depth, mag_aper[j], self.im_zps[band]),) for band_sigma, depth, mag_aper in zip(band_sigmas, depths, cat[f"MAG_APER_{band}"])]
                        #band_mean_depths = band_mean_depths + (band_mean_depth,)
                        #band_median_depths = band_median_depths + (band_median_depth,)
                
                # update band with depths and diagnostics
                cat[f"loc_depth_{band}"] = band_depths
                cat[f"{diagnostic_name}_{band}"] = band_diagnostics
                cat[f"sigma_{band}"] = band_sigmas
                # make local depth error columns in image units
                cat[f"FLUXERR_APER_{band}_loc_depth"] = [tuple([funcs.mag_to_flux(val, self.im_zps[band]) / 5. for val in element]) for element in band_depths]
                # impose n_pc min flux error and converting to Jy where appropriate
                if "APERCORR" in cat.meta.keys():
                    cat[f"FLUXERR_APER_{band}_loc_depth_{str(int(cat_creator.min_flux_pc_err))}pc_Jy"] = \
                        [tuple([funcs.flux_image_to_Jy(flux, self.im_zps[band]) * cat_creator.min_flux_pc_err / 100. if err / flux < cat_creator.min_flux_pc_err / 100. and flux > 0. \
                        else funcs.flux_image_to_Jy(flux, self.im_zps[band]) for flux, err in zip(flux_tup, err_tup)]) for flux_tup, err_tup in zip(cat[f"FLUX_APER_{band}_aper_corr"], cat[f"FLUXERR_APER_{band}_loc_depth"])]
                else:
                    raise(Exception("Couldn't make 'FLUXERR_APER_{band}_loc_depth_{str(int(cat_creator.min_flux_pc_err))}Jy' columns!"))
                # magnitude and magnitude error columns
                #mean_depths[band] = band_mean_depths
                #median_depths[band] = band_median_depths

        # update catalogue metadata
        cat.meta = {**cat.meta, **{"DEPTHS": True, "MINPCERR": cat_creator.min_flux_pc_err}} #, "Mean_depths": mean_depths, "Median_depths": median_depths}}
        #print(cat.meta)
        # overwrite original catalogue with local depth columns
        cat.write(self.sex_cat_master_path, overwrite = True)
        
    def get_depth_dir(self, aper_diam):
        self.depth_dirs = {}
        for band in self.instrument.bands:
            self.depth_dirs[band] = f"{config['Depths']['DEPTH_DIR']}/{self.instrument.instrument_from_band(band).name}/{self.version}/{self.survey}/{format(aper_diam.value, '.2f')}as"
            os.makedirs(self.depth_dirs[band], exist_ok = True)
        return self.depth_dirs
            
    def load_depths(self, aper_diam):
        self.get_depth_dir(aper_diam)
        self.depths = {}
        for band in self.instrument.bands:
            # load depths from saved .txt file
            depths = Table.read(f"{self.depth_dirs[band]}/{self.survey}_depths.txt", names = ["band", "depth"], format = "ascii")
            self.depths[band] = float(depths[depths["band"] == band]["depth"])
        return self.depths
    
    def calc_aper_radius_pix(self, aper_diam, band):
        return (aper_diam / (2 * self.im_pixel_scales[band])).value
    
    def calc_depths(self, aper_diams = [0.32] * u.arcsec, cat_creator = None, mode = "n_nearest", scatter_size = 0.1, distance_to_mask = 30, \
        region_radius_used_pix = 300, n_nearest = 200, coord_type = "sky", split_depth_min_size = 100_000, \
        split_depths_factor = 5, step_size = 100, excl_bands = [], n_jobs = 1):
        params = []
        # Look over all aperture diameters and bands  
        for aper_diam in aper_diams:
            # Generate folder for depths
            self.get_depth_dir(aper_diam)
            for band in self.instrument.bands:
                # Only run for non excluded bands
                if band not in excl_bands:
                    params.append((band, aper_diam, self.depth_dirs[band], mode, scatter_size, distance_to_mask, region_radius_used_pix, n_nearest, \
                    coord_type, split_depth_min_size, split_depths_factor, step_size, cat_creator))
        # Parallelise the calculation of depths for each band
        with tqdm_joblib(tqdm(desc = "Calculating depths", total = len(params))) as progress_bar:
            Parallel(n_jobs = n_jobs)(delayed(self.calc_band_depth)(param) for param in params)
    
    def calc_band_depth(self, params):
        # unpack parameters
        band, aper_diam, depth_dir, mode, scatter_size, distance_to_mask, region_radius_used_pix, n_nearest, \
            coord_type, split_depth_min_size, split_depths_factor, step_size, cat_creator = params
        # determine paths and whether to overwrite
        overwrite = config["Depths"].getboolean("OVERWRITE_DEPTHS")
        if overwrite:
            galfind_logger.info("OVERWRITE_DEPTHS = YES, re-doing depths should they exist.")
        grid_depth_path = f"{depth_dir}/{mode}/{band}.h5" # {str(int(n_split))}_region_grid_depths/
        os.makedirs("/".join(grid_depth_path.split("/")[:-1]), exist_ok = True)
        
        if not Path(grid_depth_path).is_file() or overwrite:
            # load the image/segmentation/mask data for the specific band
            im_data, im_header, seg_data, seg_header, mask = self.load_data(band, incl_mask = True)
            combined_mask = self.combine_seg_data_and_mask(seg_data = seg_data, mask = mask)
            wcs = WCS(im_header)
            radius_pix = self.calc_aper_radius_pix(aper_diam, band)
            
            # Load wht data if it has the correct type
            wht_data = self.load_wht(band)
            #print(f"wht_data = {wht_data}")
            if type(wht_data) == type(None):
                n_split = 1
            else:
                n_split = "auto"

            # load catalogue of given type
            cat = Table.read(self.sex_cat_master_path)
            
            # Place apertures in empty regions in the image
            xy = Depths.make_grid(im_data, combined_mask, radius = (aper_diam / 2.).value, 
                scatter_size = scatter_size, distance_to_mask = distance_to_mask, plot = False)
            #print(f"{len(xy)} empty apertures placed in {band}")
            
            # Make ds9 region file of apertures for compatability and debugging
            region_path = f"{depth_dir}/{mode}/{self.survey}_{self.version}_{band}.reg"
            Depths.make_ds9_region_file(xy, radius_pix, region_path, coordinate_type = 'pixel', 
                convert = False, wcs = wcs, pixel_scale = self.im_pixel_scales[band])
            
            # Get fluxes in regions
            fluxes = Depths.do_photometry(im_data, xy, radius_pix)
            
            depths, diagnostic, depth_labels, final_labels = Depths.calc_depths(xy, fluxes, im_data, combined_mask, 
                    region_radius_used_pix = region_radius_used_pix, step_size = step_size, catalogue = cat, wcs = wcs, \
                    coord_type = coord_type, mode = mode, n_nearest = n_nearest, zero_point = self.im_zps[band], n_split = n_split, \
                    split_depth_min_size = split_depth_min_size, split_depths_factor = split_depths_factor, wht_data = wht_data)

            # calculate the depths for plotting purposes
            nmad_grid, num_grid, labels_grid, final_labels = Depths.calc_depths(xy, fluxes, im_data, combined_mask, 
                region_radius_used_pix = region_radius_used_pix, step_size = step_size, wcs = wcs, \
                coord_type = coord_type, mode = mode, n_nearest = n_nearest, zero_point = self.im_zps[band], \
                n_split = n_split, split_depth_min_size = split_depth_min_size, \
                split_depths_factor = split_depths_factor, wht_data = wht_data, provide_labels = final_labels)
            
            # write to .h5
            hf_save_names = self.get_depth_h5_labels()
            hf_save_data = [mode, aper_diam, scatter_size, distance_to_mask, region_radius_used_pix, \
                    n_nearest, split_depth_min_size, split_depths_factor, step_size, depths, \
                    diagnostic, depth_labels, final_labels, nmad_grid, num_grid, labels_grid]
            hf = h5py.File(grid_depth_path, "w")
            for name_i, data_i in zip(hf_save_names, hf_save_data):
                #print(name_i, data_i)
                hf.create_dataset(name_i, data = data_i)
            hf.close()

            self.plot_depth(band, cat_creator, mode, aper_diam)

    def plot_depth(self, band, cat_creator, mode, aper_diam): #, **kwargs):
        if cat_creator == None:
            galfind_logger.warning("Could not plot depths as cat_creator == None in Data.plot_depths()")
        else:
            self.get_depth_dir(aper_diam)
            save_path = f"{self.depth_dirs[band]}/{mode}/{band}_depths.png"
            # determine paths and whether to overwrite
            overwrite = config["Depths"].getboolean("OVERWRITE_DEPTH_PLOTS")
            if overwrite:
                galfind_logger.info("OVERWRITE_DEPTH_PLOTS = YES, re-doing depth plots.")
            if not Path(save_path).is_file() or overwrite:
                # load depth data
                h5_path = f"{self.depth_dirs[band]}/{mode}/{band}.h5"
                if not Path(h5_path).is_file():
                    raise(Exception(f"Must first run depths for {self.survey} {self.version} {band} {mode} {aper_diam} before plotting!"))
                hf = h5py.File(h5_path, "r")
                hf_output = {label: np.array(hf[label]) for label in self.get_depth_h5_labels()}
                hf.close()
                # load image and wcs
                im_data, im_header = self.load_im(band)
                wcs = WCS(im_header)
                # make combined mask
                combined_mask = self.combine_seg_data_and_mask(band)
                # load catalogue to calculate x/y image coordinates
                cat = Table.read(self.sex_cat_master_path)
                cat_x, cat_y = wcs.world_to_pixel(SkyCoord(cat[cat_creator.ra_dec_labels["RA"]], cat[cat_creator.ra_dec_labels["DEC"]]))
                
                depths_fig, depths_ax = Depths.show_depths(hf_output["nmad_grid"], hf_output["num_grid"], hf_output["step_size"], \
                    hf_output["region_radius_used_pix"], hf_output["labels_grid"], hf_output["depth_labels"], hf_output["depths"], hf_output["diagnostic"], cat_x, cat_y, 
                    combined_mask, hf_output["final_labels"], suptitle = f"{self.survey} {self.version} {band} Depths", save_path = save_path)

    @staticmethod
    def get_depth_h5_labels():
        return ["mode", "aper_diam", "scatter_size", "distance_to_mask", "region_radius_used_pix", \
            "n_nearest", "split_depth_min_size", "split_depths_factor", "step_size", "depths", \
            "diagnostic", "depth_labels", "final_labels", "nmad_grid", "num_grid", "labels_grid"]

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

# The below makes TQDM work with joblib
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
        
def is_blank_survey(survey):
    cluster_surveys = ["El-Gordo", "MACS-0416", "CLIO", "SMACS-0723"]
    if survey in cluster_surveys:
        return False
    else:
        return True

if __name__ == "__main__":
    pass


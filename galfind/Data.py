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
from tqdm import tqdm
from .Instrument import Instrument, ACS_WFC,WFC3IR, NIRCam, MIRI, Combined_Instrument
from . import config
from . import useful_funcs_austind as funcs
from .decorators import run_in_dir, hour_timer, email_update

# GALFIND data object
class Data:
    
    def __init__(self, instrument, im_paths, im_exts,im_pixel_scales, im_shapes, im_zps, wht_paths, wht_exts, wht_types, seg_paths, mask_paths, cluster_mask_path, blank_mask_path, survey, version = "v0", is_blank = True):
        # self, instrument, im_paths, im_exts, seg_paths, mask_paths, cluster_mask_path, blank_mask_path, survey, version = "v0", is_blank = True):
        
        # sort dicts from blue -> red bands in ascending wavelength order
        self.im_paths = dict(sorted(im_paths.items()))
        self.im_exts = dict(sorted(im_exts.items())) # science image extension
        self.survey = survey
        self.version = version
        self.instrument = instrument
        self.is_blank = is_blank
        
        self.im_zps = im_zps
        self.wht_exts = wht_exts
        self.wht_paths = wht_paths
        self.wht_types = wht_types
        self.im_pixel_scales = im_pixel_scales
        self.im_shapes = im_shapes

        print(self.im_paths)
        print(self.wht_paths)

        # make segmentation maps from image paths if they don't already exist
        made_new_seg_maps = False
        for i, (band, seg_path) in enumerate(seg_paths.items()):
            #print(band, seg_path)
            if (seg_path == "" or seg_path == []) and not made_new_seg_maps:
                self.make_seg_maps()
                made_new_seg_maps = True
            # load new segmentation maps
            seg_paths[band] = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/SExtractor/{self.instrument.instrument_from_band(band)}/{version}/{survey}/{survey}*{band}_{band}*{version}*seg.fits")[0]
        self.seg_paths = dict(sorted(seg_paths.items()))
        
        # make masks from image paths if they don't already exist
    
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
    def from_pipeline(cls, survey, version = "v8", instruments = ['NIRCam', 'ACS_WFC', 'WFC3IR'], excl_bands = [], pix_scales = ['30mas', '60mas']):
        instruments_obj = {'NIRCam': NIRCam(excl_bands = excl_bands), 'ACS_WFC': ACS_WFC(excl_bands = excl_bands), 'WFC3IR': WFC3IR(excl_bands = excl_bands)}
        # Build a combined instrument object
        comb_instrument_created = False
        
        im_paths = {} 
        im_exts = {}
        seg_paths = {}
        wht_paths = {}
        wht_exts = {}
        wht_types = {}
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
                elif version == "v8e":
                    survey_im_dirs = {survey: f"{survey}/mosaic_1084_wisptemp2"}
                elif version == "lit_version":
                    survey_im_dirs = {"JADES-DR1": "JADES/DR1"}

                survey_im_dirs = {key: f"/raid/scratch/data/jwst/{value}" for (key, value) in survey_im_dirs.items()}
                survey_dir = survey_im_dirs[survey]

                # don't use these if they are in the same folder
                nadams_seg_path_arr = glob.glob(f"{survey_dir}/*_seg.fits")
                nadams_bkg_path_arr = glob.glob(f"{survey_dir}/*_bkg.fits")

                if version == "lit_version":
                    im_path_arr = glob.glob(f"{survey_dir}/*_drz.fits")
                else:
                    im_path_arr = glob.glob(f"{survey_dir}/*_i2d*.fits")
                im_path_arr = np.array([path for path in im_path_arr if path not in nadams_seg_path_arr and path not in nadams_bkg_path_arr])

                # obtain available bands from imaging without having to hard code these
                bands = np.array([split_path.lower().replace("w", "W").replace("m", "M") for path in im_path_arr for i, split_path in \
                        enumerate(path.split("-")[-1].split("/")[-1].split("_")) if split_path.lower().replace("w", "W").replace("m", "M") in instrument.bands])
                
                # If band not used in instrument, remove it
                for band in instrument.bands:
                    if band not in bands:
                        instrument.remove_band(band)
                    else:
                        # Maybe generalize this
                        print("Generalize on line 177 of Data.from_pipeline()")
                        im_pixel_scales[band] = 0.03 
                        im_zps[band] = 28.08
                
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
                        if im_hdu.name == "SCI":
                            im_exts[band] = int(j)
                            im_shapes[band] = im_hdu.data.shape
                        if im_hdu.name == 'ERR':
                            wht_exts[band] = int(j)
                            wht_types[band] = "MAP_RMS"
                            wht_paths[band] = str(im_paths[band])
                        
                    # need to change this to work if there are no segmentation maps (with the [0] indexing)
                

            elif instrument.name in ["ACS_WFC", 'WFC3IR']:
                # Iterate through bands and check if images exist 
                any_path_found = False
                for band in instrument.bands:
                    path_found = False
                    for pix_scale in pix_scales:
                        path = Path(f"{config['DEFAULT']['GALFIND_DATA']}/hst/{survey}/{instrument.name}/{pix_scale}/{instrument.name}_{band}_{survey}_drz.fits")
                        if path.is_file():
                            any_path_found = True
                            path_found = True
                            break

                    # If no images found, remove band from instrument
                    if not path_found:
                        instrument.remove_band(band)
                    else:
                        # otherwise open band, work out if it has a weight map, calc zero point and image scale
                        hdu = fits.open(str(path))
                        
                        im_paths[band] = str(path)
                        # Not great to use try/except but not sure how else to do it with index_of
                        try:
                            im_exts[band] = hdu.index_of('SCI')
                        except KeyError:
                            im_exts[band] = 0
                        # Get header of image extension
                        imheader = hdu[im_exts[band]].header
                        im_shapes[band] = hdu[im_exts[band]].data.shape
                        hdu.close()
                        try:
                            # This nice nested loop checks if there is a wht extension, if not trys to find wht or rms file
                            wht_exts[band] = hdu.index_of('WHT')
                            wht_paths[band] = str(path)
                            wht_types[band] = "MAP_WEIGHT"
                        except KeyError:
                            try:
                                wht_exts[band] = hdu.index_of('ERR')
                                wht_paths[band] = str(path)
                                wht_types[band] = "MAP_RMS"

                            except KeyError:
                                path = Path(f"{config['DEFAULT']['GALFIND_DATA']}/hst/{survey}/{instrument.name}/{pix_scale}/{instrument.name}_{band}_{survey}_wht.fits")
                                if path.is_file():
                                    wht_paths[band] = str(path)
                                    wht_types[band] = 'MAP_WEIGHT'
                                    wht_exts[band] = 0
                                else:
                                    path = Path(f"{config['DEFAULT']['GALFIND_DATA']}/hst/{survey}/{instrument.name}/{pix_scale}/{instrument.name}_{band}_{survey}_rms.fits")
                                    if path.is_file():
                                        wht_paths[band] = str(path)
                                        wht_types[band] = 'MAP_RMS'
                                        wht_exts[band] = 0
                                    else:
                                        wht_paths[band] = ""
                                        wht_types[band] = "NONE"
                                        wht_exts[band] = ""

                        im_pixel_scales[band] = float(pix_scale.split('mas')[0]) * 1e-3 
                        if instrument.name == 'ACS_WFC':
                            im_zps[band] = -2.5 * np.log10(imheader["PHOTFLAM"]) - 21.10 - 5 * np.log10(imheader["PHOTPLAM"]) + 18.6921
                        elif instrument.name == 'WFC3IR':
                        # Taken from Appendix A of https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/documentation/instrument-science-reports-isrs/_documents/2020/WFC3-ISR-2020-10.pdf
                            wfc3ir_zps = {'f098M':25.661, 'f105W':26.2637, 'f110W':26.8185, 'f125W':26.231, 'f140W':26.4502, 'f160W':25.9362}
                            im_zps[band] = wfc3ir_zps[band]
                        # Need to move my segmentation maps and masks to the correct place
                
                if any_path_found:
                    if comb_instrument_created:
                        comb_instrument += instrument
                        print('Added instrument')
                    else:
                        comb_instrument = instrument
                        comb_instrument_created = True

                        # Need to update what it suggests
           
            elif instrument.name == 'MIRI':
                raise NotImplementedError("MIRI not yet implemented")
        if comb_instrument_created:
            # All seg maps and masks should be in same format, so load those last when we know what bands we have
            for band in comb_instrument.bands:
                try:
                    #print(f"{config['DEFAULT']['GALFIND_WORK']}/SExtractor/{comb_instrument.instrument_from_band(band)}/{version}/{survey}/{survey}*{band}_{band}*{version}*seg.fits")
            
                    seg_paths[band] = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/SExtractor/{comb_instrument.instrument_from_band(band)}/{version}/{survey}/{survey}*{band}_{band}*{version}*seg.fits")[0]
                except IndexError:
                    seg_paths[band] = ""
                # include just the masks corresponding to the correct bands
                try:
                    mask_paths[band] = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/*{band.replace('W', 'w').replace('M', 'm')}*")[0]
                except IndexError:
                    mask_paths[band] = ""
            # Moved out as not actually specific to NIRCam
            try:
                cluster_mask_path = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/*cluster*")[0]
            except IndexError:
                cluster_mask_path = ""
            try:
                blank_mask_path = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/*blank*")[0]
            except IndexError:
                blank_mask_path = ""

            #im_paths, im_exts, im_shapes, im_zps, wht_paths, wht_exts, wht_types, im_pixel_scales, seg_paths, mask_paths, cluster_mask_path, blank_mask_path 
            
            return cls(comb_instrument, im_paths, im_exts,im_pixel_scales, im_shapes, im_zps, wht_paths, wht_exts, wht_types, seg_paths, mask_paths, cluster_mask_path, blank_mask_path, survey, version = version, is_blank = is_blank)
        else:
            raise(Exception(f'Failed to find any data for {survey}'))  

    @classmethod
    def from_NIRCam_pipeline(cls, survey, version = "v8", excl_bands = []):
        instrument = NIRCam(excl_bands = excl_bands)
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
            ceers_im_dirs = {f"CEERSP{str(i + 1)}": f"CEERSP{str(i + 1)}/mosaic_1084_182" for i in range(10)}
            survey_im_dirs = {"CLIO": "CLIO/mosaic_1084_182", "El-Gordo": "elgordo/mosaic_1084_182", "NEP-1": "NEP-1/mosaic_1084_182", "NEP-2": "NEP-2/mosaic_1084_182", \
                              "NEP-3": "NEP-3/mosaic_1084_182", "NEP-4": "NEP-4/mosaic_1084_182", "MACS-0416": "MACS0416/mosaic_1084_182", "GLASS": "GLASS-12/mosaic_1084_182", "SMACS-0723": "SMACS0723/mosaic_1084_182"} | ceers_im_dirs
        elif version == "v8b":
            survey_im_dirs = {survey: f"{survey}/mosaic_1084_wispfix"}
        elif version == "v8c":
            survey_im_dirs = {survey: f"{survey}/mosaic_1084_wispfix2"}
        elif version == "lit_version":
            survey_im_dirs = {"JADES-DR1": "JADES/DR1"}
                
        survey_im_dirs = {key: f"/raid/scratch/data/jwst/{value}" for (key, value) in survey_im_dirs.items()}
        survey_dir = survey_im_dirs[survey]
        
        # don't use these if they are in the same folder
        nadams_seg_path_arr = glob.glob(f"{survey_dir}/*_seg.fits")
        nadams_bkg_path_arr = glob.glob(f"{survey_dir}/*_bkg.fits")
        
        if version == "lit_version":
            im_path_arr = glob.glob(f"{survey_dir}/*_drz.fits")
        else:
            im_path_arr = glob.glob(f"{survey_dir}/*_i2d*.fits")
        im_path_arr = np.array([path for path in im_path_arr if path not in nadams_seg_path_arr and path not in nadams_bkg_path_arr])
        
        # obtain available bands from imaging without having to hard code these
        bands = np.array([split_path.lower().replace("w", "W").replace("m", "M") for path in im_path_arr for i, split_path in \
                enumerate(path.split("-")[-1].split("/")[-1].split("_")) if split_path.lower().replace("w", "W").replace("m", "M") in instrument.bands])

        for band in instrument.bands:
            if band not in bands:
                instrument.remove_band(band)
        
        im_paths = {}
        im_exts = {}
        seg_paths = {}
        mask_paths = {}
        for band in bands:
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
            for i, im_hdu in enumerate(im_hdul):
                if im_hdu.name == "SCI":
                    im_exts[band] = int(i)
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
            print("cluster mask path = ", cluster_mask_path)
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
        #print("im_path, seg_path, band, im_ext")
        #print(im_path, seg_path, band, im_ext)
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
    
    @run_in_dir(path = config['DEFAULT']['GALFIND_DIR'])
    def make_seg_maps(self):
        for band in self.instrument.bands:
            print([config['DEFAULT']['GALFIND_WORK'], self.im_paths[band], str(self.im_pixel_scales[band]), \
                                    str(self.im_zps[band]), self.instrument.instrument_from_band(band), self.survey, band, self.version, str(self.wht_paths[band]), \
                                    str(self.wht_exts[band]),self.wht_types[band],str(self.im_exts[band]),f"{config['DEFAULT']['GALFIND_DIR']}/configs/"])
            # SExtractor bash script python wrapper
            process = subprocess.Popen([f"./make_seg_map.sh", config['DEFAULT']['GALFIND_WORK'], self.im_paths[band], str(self.im_pixel_scales[band]), \
                                    str(self.im_zps[band]), self.instrument.instrument_from_band(band), self.survey, band, self.version, str(self.wht_paths[band]), \
                                    str(self.wht_exts[band]),self.wht_types[band],str(self.im_exts[band]),f"{config['DEFAULT']['GALFIND_DIR']}/configs/"])
            process.wait()
            
            
            print(f"Made segmentation map for {self.survey} {self.version} {band}")
    
    @run_in_dir(path = config['DEFAULT']['GALFIND_DIR'])
    def make_sex_cats(self, forced_phot_band = "f444W"):
        # make individual forced photometry catalogues
        force_pho_size = self.im_shapes[forced_phot_band]
        for band in self.instrument.bands:
            # if not run before
            path = Path(f"{config['DEFAULT']['GALFIND_WORK']}/SExtractor/{self.instrument.instrument_from_band(band)}/{self.version}/{self.survey}/{self.survey}_{band}_{forced_phot_band}_sel_cat_{self.version}.fits")
            if not path.is_file():
                
                # SExtractor bash script python wrapper
                if force_pho_size == self.im_shapes[band] and self.wht_types[band] == self.wht_types[forced_phot_band]:
                   
                    process = subprocess.Popen([f"./make_sex_cat.sh", config['DEFAULT']['GALFIND_WORK'], self.im_paths[band], str(self.im_pixel_scales[band]), \
                                        str(self.im_zps[band]),self.instrument.instrument_from_band(band), self.survey, band, self.version, \
                                            forced_phot_band, str(self.im_paths[forced_phot_band]), str(self.wht_paths[band]), str(self.wht_exts[band]), \
                                            str(self.im_exts[band]), str(self.wht_paths[forced_phot_band]), str(self.im_exts[forced_phot_band]), self.wht_types[band], 
                                            str(self.wht_exts[forced_phot_band]), f"{config['DEFAULT']['GALFIND_DIR']}/configs/"])
                    process.wait()
                # Use photutils
                else:
                    forcephot_path =  Path(f"{config['DEFAULT']['GALFIND_WORK']}/SExtractor/{self.instrument.instrument_from_band(forced_phot_band)}/{self.version}/{self.survey}/{self.survey}_{forced_phot_band}_{forced_phot_band}_sel_cat_{self.version}.fits")
          
                    self.forced_photometry(band, forced_phot_band, path, forcephot_path)
                    
             
            print(f"Finished making SExtractor catalogue for {self.survey} {self.version} {band}!")
        self.sex_cats = {band: f"{config['DEFAULT']['GALFIND_WORK']}/SExtractor/{self.instrument.instrument_from_band(band)}/{self.version}/{self.survey}/{self.survey}_{band}_{forced_phot_band}_sel_cat_{self.version}.fits" for band in self.instrument.bands}
    
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
    
    def forced_photometry(self, band, forced_phot_band, path, forcephot_path, radii = [0.16, 0.25, 0.5, 0.75, 1]*u.arcsec, ra_col='ALPHA_J2000', dec_col='DELTA_J2000', coord_unit=u.deg, id_col='NUMBER', x_col='X_IMAGE', y_col='Y_IMAGE'):
        # Read in sextractor catalogue
        catalog = Table.read(forcephot_path, character_as_bytes = False)
        # Get image path
        image = self.im_paths[band]
        # Ipen image with correct extension and get WCS
        with fits.open(image) as hdul:
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
        positions = SkyCoord(ra, dec, unit=coord_unit)
    
        # Define radii in sky units
        # This checks if radii is iterable and if not makes it a list
        try:
            iter(radii)
        except TypeError:
            radii = [radii]
        apertures = []

        for rad in radii:
            aperture = SkyCircularAperture(positions, r=rad)
            apertures.append(aperture)
            # Convert to pixel using image WCS
        # Generate background
        background = Background2D(image, (64, 64), filter_size=(3, 3))
    
        image = image-background.background
        # Do aperture photometry
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
        phot_table[f'FLUX_APER'] = aper_tab
        phot_table[f'FLUXERR_APER'] = phot_table[f'FLUX_APER'] * -99
        phot_table['MAGERR_APER'] = phot_table['FLUX_APER'] * -99 
        
        zp = self.im_zps[band]
        # This converts the fluxes to magnitudes using the correct zp, and puts them in the same format as the sextractor catalogue
        mag_colnames  = []
        for pos,col in enumerate(colnames):
            name = f'MAG_APER_{pos}'
            phot_table[name] = -2.5 * np.log10(phot_table[col]) + zp
            phot_table[name][np.isnan(phot_table[name])] = -99
            mag_colnames.append(name)
        aper_tab = Column(np.array(phot_table[mag_colnames].as_array().tolist()), name=f'MAG_APER_{band}')
        phot_table[f'MAG_APER'] = aper_tab
        # Remove old columns
        phot_table.remove_columns(colnames)
        phot_table.remove_columns(mag_colnames)
        phot_table.write(path, format='fits', overwrite=True)


    # def make_mask(self, band, stellar_dir = "GAIA DR3"):
    #     im_data, im_header, seg_data, seg_header = self.load_data(band, incl_mask = False)
    #     # works as long as your images are aligned with the stellar directory
    #     stellar_mask = make_stellar_mask(band, im_data, im_header, self.instrument)
    #     # save the stellar mask
    #     stellar_mask.write(f"{os.getcwd()}/Masks/{survey}/{band}_stellar_mask.reg", header = im_header)
    #     # maybe insert a step here to align your images
    #     pass
    
    def clean_mask_regions(self, band):
        # open region file
        mask_path = self.mask_paths[band]
        with open(mask_path, 'r') as f:
            lines = f.readlines()
            with open(mask_path.replace(".reg", "_clean.reg"), 'w') as temp:          
                for i, line in enumerate(lines):
                    if line.startswith('physical'):
                        lines[i] = line.replace('physical', 'image')
                    if not ( line.endswith(',0)\n') and line.startswith('circle')):
                        if not (line.endswith(',0)\n') and line.startswith('ellipse')):
                           temp.write(line)
        # insert original mask ds9 region file into an unclean folder
        # update mask paths for the object
    
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
        
    def make_loc_depth_cat(self, aper_diams = [0.32] * u.arcsec, n_samples = 5, forced_phot_band = "f444W", min_flux_pc_err_arr = [5, 10], fast = True):
        # if sextractor catalogue has not already been made, make it
        self.combine_sex_cats(forced_phot_band)
        # if depths havn't already been run, run them
        self.calc_depths(aper_diams = aper_diams, fast = fast)
        # correct the base sextractor catalogue to include local depth errors if not already done so
        self.loc_depth_cat_path = self.sex_cat_master_path.replace(".fits", "_loc_depth.fits")
        if not Path(self.loc_depth_cat_path).is_file():
            print(f"Making local depth catalogue for {self.survey} {self.version} in {aper_diams} diameter apertures with min. error(s) {min_flux_pc_err_arr}%!")
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
                    phot_data["FLUX_APER_" + band + "_aper_corr"].T[j] = 10 ** ((phot_data["MAG_APER_" + band + "_aper_corr"].T[j] - self.im_zps[band]) / -2.5)
                    print("Performed aperture corrections")
                
                # make new columns (fill with original errors and overwrite in a couple of lines)
                phot_data = make_new_fits_columns(phot_data, ["loc_depth_" + band, "FLUXERR_APER_" + band + "_loc_depth", "MAGERR_APER_" + band + "_l1_loc_depth", \
                                            "MAGERR_APER_" + band + "_u1_loc_depth", "FLUX_APER_" + band + "_aper_corr_Jy", "sigma_" + band], \
                                            [phot_data["FLUXERR_APER_" + band], phot_data["FLUXERR_APER_" + band], phot_data["MAGERR_APER_" + band], phot_data["MAGERR_APER_" + band], \
                                             phot_data["FLUX_APER_" + band], phot_data["MAGERR_APER_" + band]], \
                                            [phot_data.columns.formats[list(phot_data.columns.names).index("FLUXERR_APER_" + band)], \
                                             phot_data.columns.formats[list(phot_data.columns.names).index("FLUXERR_APER_" + band)], \
                                             phot_data.columns.formats[list(phot_data.columns.names).index("MAGERR_APER_" + band)], \
                                             phot_data.columns.formats[list(phot_data.columns.names).index("MAGERR_APER_" + band)], \
                                             phot_data.columns.formats[list(phot_data.columns.names).index("FLUX_APER_" + band)], \
                                             phot_data.columns.formats[list(phot_data.columns.names).index("FLUXERR_APER_" + band)]])
                    
                for min_flux_pc_err in min_flux_pc_err_arr:
                    phot_data = make_new_fits_columns(phot_data, ["FLUXERR_APER_" + band + "_loc_depth_" + str(min_flux_pc_err) + "pc_Jy", \
                        "MAGERR_APER_" + band + "_l1_loc_depth_" + str(min_flux_pc_err) + "pc", "MAGERR_APER_" + band + "_u1_loc_depth_" + str(min_flux_pc_err) + "pc"], \
                        [phot_data["FLUXERR_APER_" + band], phot_data["MAGERR_APER_" + band], phot_data["MAGERR_APER_" + band]], \
                        [phot_data.columns.formats[list(phot_data.columns.names).index("FLUXERR_APER_" + band)], \
                        phot_data.columns.formats[list(phot_data.columns.names).index("MAGERR_APER_" + band)], \
                        phot_data.columns.formats[list(phot_data.columns.names).index("MAGERR_APER_" + band)]])
                
                for j, aper_diam in enumerate(json.loads(config.get("SExtractor", "APERTURE_DIAMS")) * u.arcsec):
                    for k in range(len(phot_data["NUMBER"])):
                        # set initial values to -99. or False by default
                        phot_data["loc_depth_" + band].T[j][k] = -99.
                        phot_data["FLUXERR_APER_" + band + "_loc_depth"].T[j][k] = -99.
                        phot_data["MAGERR_APER_" + band + "_l1_loc_depth"].T[j][k] = -99.
                        phot_data["MAGERR_APER_" + band + "_u1_loc_depth"].T[j][k] = -99.
                        for min_flux_pc_err in min_flux_pc_err_arr:
                            phot_data["FLUXERR_APER_" + band + "_loc_depth_" + str(min_flux_pc_err) + "pc_Jy"].T[j][k] = -99.
                            phot_data["MAGERR_APER_" + band + "_l1_loc_depth_" + str(min_flux_pc_err) + "pc"].T[j][k] = -99.
                            phot_data["MAGERR_APER_" + band + "_u1_loc_depth_" + str(min_flux_pc_err) + "pc"].T[j][k] = -99.
                        phot_data["sigma_" + band].T[j][k] = -99.
                        
                        # update column for flux in Jy
                        phot_data["FLUX_APER_" + band + "_aper_corr_Jy"].T[j][k] = funcs.flux_image_to_Jy(phot_data["FLUX_APER_" + band + "_aper_corr"].T[j][k], self.im_zps[band]).value
            
                for diam_index, aper_diam in enumerate(aper_diams):
                    r = self.calc_aper_radius_pix(aper_diam, band)
                    # open aperture positions in this band
                    self.get_depth_dir(aper_diam)
                    print(f"self.get_depth_dir(aper_diam) = {self.depth_dirs[band]}", f"aper_diam = {aper_diam}")
                    aper_loc = np.loadtxt(f"{self.depth_dirs[band]}/coord_{band}.txt")
                    xcoord = aper_loc[:, 0]
                    ycoord = aper_loc[:, 1]
                    index = np.argwhere(xcoord == 0.)
                    xcoord = np.delete(xcoord, index)
                    ycoord = np.delete(ycoord, index)
                    aper_coords = pixel_to_skycoord(xcoord, ycoord, wcs)
                    
                    # calculate local depths for all galaxies
                    loc_depths = calc_loc_depths(phot_data["ALPHA_J2000"], phot_data["DELTA_J2000"], aper_coords, xcoord, ycoord, im_data, r, self.survey, band, n_samples = n_samples, zero_point = self.im_zps[band])
                    
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
                        aper_flux_err = (10 ** ((loc_depth - self.im_zps[band]) / -2.5)) / 5 # in image units
                        if aper_flux_err == np.nan and not loc_depth_nan:
                            nans = nans + 1
                            print("loc_depth =", loc_depth)
                        phot_data["FLUXERR_APER_" + band + "_loc_depth"].T[diam_index][k] = aper_flux_err
                        
                        # add column setting flux in Jy to minimum n_pc error
                        for min_flux_pc_err in min_flux_pc_err_arr:
                            if phot_data["FLUXERR_APER_" + band + "_loc_depth"].T[diam_index][k] / phot_data["FLUX_APER_" + band + "_aper_corr"].T[diam_index][k] < min_flux_pc_err / 100:
                                phot_data["FLUXERR_APER_" + band + "_loc_depth_" + str(min_flux_pc_err) + "pc_Jy"].T[diam_index][k] = \
                                phot_data["FLUX_APER_" + band + "_aper_corr_Jy"].T[diam_index][k] * min_flux_pc_err / 100
                            else:
                                phot_data["FLUXERR_APER_" + band + "_loc_depth_" + str(min_flux_pc_err) + "pc_Jy"].T[diam_index][k] = \
                                    funcs.flux_image_to_Jy(phot_data["FLUXERR_APER_" + band + "_loc_depth"].T[diam_index][k], self.instrument.zero_points[band]).value
                        
                        # calculate local depth mag errors both with and without 5pc minimum flux errors imposed
                        for m in range(2):
                            for n, min_flux_pc_err in enumerate(min_flux_pc_err_arr):
                                if m == 0 and n == 0:
                                    flux = phot_data["FLUX_APER_" + band + "_aper_corr"].T[diam_index][k]
                                    aper_flux_err = phot_data["FLUXERR_APER_" + band + "_loc_depth"].T[diam_index][k]
                                    add_suffix = ""
                                if m == 1:
                                    flux = phot_data["FLUX_APER_" + band + "_aper_corr_Jy"].T[diam_index][k]
                                    aper_flux_err = phot_data["FLUXERR_APER_" + band + "_loc_depth_" + str(min_flux_pc_err) + "pc_Jy"].T[diam_index][k]
                                    add_suffix = "_" + str(min_flux_pc_err) + "pc"
                            
                                mag_l1 = -(-2.5 * np.log10(flux) + 2.5 * np.log10(flux - aper_flux_err))
                                mag_u1 = -(-2.5 * np.log10(flux + aper_flux_err) + 2.5 * np.log10(flux))
                                #print(mag_l1)
                                #print(mag_u1)
                                if np.isfinite(mag_l1):
                                    phot_data["MAGERR_APER_" + band + "_l1_loc_depth" + add_suffix].T[diam_index][k] = mag_l1
                                if np.isfinite(mag_u1):  
                                    phot_data["MAGERR_APER_" + band + "_u1_loc_depth" + add_suffix].T[diam_index][k] = mag_u1
                        
                        # make boolean columns to say whether there is a local 5σ detection and 2σ non-detection in the band in the smallest aperture
                        phot_data["sigma_" + band].T[diam_index][k] = funcs.n_sigma_detection(loc_depth, phot_data[f"MAG_APER_{band}"].T[diam_index][k], self.im_zps[band])
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
        self.depth_dirs = {}
        print(self.instrument.bands)
        print(self.instrument.instrument_from_band("f150W"))
        for band in self.instrument.bands:
            self.depth_dirs[band] = f"{config['DEFAULT']['GALFIND_WORK']}/Depths/{self.instrument.instrument_from_band(band)}/{self.version}/{self.survey}/{str(aper_diam.value)}as"
            os.makedirs(self.depth_dirs[band], exist_ok = True)
        
    
    def calc_aper_radius_pix(self, aper_diam, band):
        return (aper_diam / (2 * self.im_pixel_scales[band])).value
    
    def calc_depths(self, xy_offset = [0, 0], aper_diams = [0.32] * u.arcsec, size = 500, n_busy_iters = 1_000, number = 600, \
                    mask_rad = 25, aper_disp_rad = 2, excl_bands = [], use_xy_offset_txt = True, plot = False, n_jobs = 1, fast = True):   

        params = []  
        average_depths = []
        run_bands = []
        # Look over all aperture diameters and bands  
        for aper_diam in aper_diams:
            # Generate folder for depths
            self.get_depth_dir(aper_diam)
            for band in self.instrument.bands:
                # Only run for non excluded bands
                if band not in excl_bands:
                    params.append((band, xy_offset, aper_diam, size, n_busy_iters, number, mask_rad, aper_disp_rad, use_xy_offset_txt, plot, average_depths, run_bands, fast))
        # Parallelise the calculation of depths for each band
        with tqdm_joblib(tqdm(desc="Running local depths", total=len(params))) as progress_bar:
            Parallel(n_jobs=n_jobs)(delayed(self.calc_band_depth)(param) for param in params)
        # print table of depths for these bands
        header = "band, average_5sigma_depth"
        for band in run_bands:
            # Save local depths in both folders
            if not Path(f"{self.depth_dirs[band]}/{self.survey}_depths.txt").is_file():
                np.savetxt(f"{self.depth_dirs[band]}/{self.survey}_depths.txt", np.column_stack((np.array(run_bands), np.array(average_depths))), header = header, fmt = "%s")
            
    def calc_band_depth(self, params):
        band, xy_offset, aper_diam, size, n_busy_iters, number, mask_rad, aper_disp_rad, use_xy_offset_txt, plot, average_depths, run_bands, fast = params
        if plot:
            fig, ax = plt.subplots()
            self.plot_mask_regions_from_band(ax, band)
            self.plot_image_from_band(ax, band, show = True)
            fig, ax = plt.subplots()
            self.plot_mask_from_band(ax, band, show = True)
                  
        if use_xy_offset_txt:
            try:
                # use the xy_offset defined in .txt in appropriate folder
                xy_offset_path = f"{self.depth_dirs[band]}/offset_{band}.txt"
                xy_offset = list(np.genfromtxt(xy_offset_path, dtype = int))
                print(f"xy_offset = {xy_offset}")
            except: # use default xy offset if this .txt does not exist
                pass
            
        if not Path(f"{self.depth_dirs[band]}/coord_{band}.reg").is_file() or not Path(f"{self.depth_dirs[band]}/{self.survey}_depths.txt").is_file() or not Path(f"{self.depth_dirs[band]}/coord_{band}.txt").is_file():
            xoff, yoff = calc_xy_offsets(xy_offset)
            im_data, im_header, seg_data, seg_header, mask = self.load_data(band)
            print(f"Finished loading {band}")

        if not Path(f"{self.depth_dirs[band]}/coord_{band}.txt").is_file():
            # place apertures in blank regions of sky
            xcoord, ycoord = place_blank_regions(im_data, im_header, seg_data, mask, self.survey, xy_offset, self.im_pixel_scales[band], band, \
                                                aper_diam, size, n_busy_iters, number, mask_rad, aper_disp_rad, fast = fast)
            print(f"Finished placing blank regions in {band}")
            np.savetxt(f"{self.depth_dirs[band]}/coord_{band}.txt", np.column_stack((xcoord, ycoord)))
            # save xy offset for this field and band
            np.savetxt(f"{self.depth_dirs[band]}/offset_{band}.txt", np.column_stack((xoff, yoff)), header = "x_off, y_off", fmt = "%d %d")
        
        # read in aperture locations
        if not Path(f"{self.depth_dirs[band]}/coord_{band}.reg").is_file() or not Path(f"{self.depth_dirs[band]}/{self.survey}_depths.txt").is_file():
            aper_loc = np.loadtxt(f"{self.depth_dirs[band]}/coord_{band}.txt")
            xcoord = aper_loc[:, 0]
            ycoord = aper_loc[:, 1]
            index = np.argwhere(xcoord == 0.)
            xcoord = np.delete(xcoord, index)
            ycoord = np.delete(ycoord, index)
        
        # convert these to .reg region file
        if not Path(f"{self.depth_dirs[band]}/coord_{band}.reg").is_file():
            aper_loc_to_reg(xcoord, ycoord, WCS(im_header), aper_diam.value, f"{self.depth_dirs[band]}/coord_{band}.reg")
        
        r = self.calc_aper_radius_pix(aper_diam, band)
        if not Path(f"{self.depth_dirs[band]}/{self.survey}_depths.txt").is_file():
            # plot the depths in the grid
            plot_depths(im_data, self.depth_dirs[band], band, seg_data, xcoord, ycoord, xy_offset, r, size, self.im_zps[band])
            # calculate average depth
            average_depths.append(calc_5sigma_depth(xcoord, ycoord, im_data, r, self.im_zps[band])) 
            run_bands.append(band)
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


def place_blank_regions(im_data, im_header, seg_data, mask, survey, offset, pix_scale, band, aper_diam = 0.32 * u.arcsec, size = 500, n_busy_iters = 1_000, number = 600, mask_rad = 25, aper_disp_rad = 2, fast=True):
    if type(pix_scale) != u.Quantity:
       pix_scale = pix_scale * u.arcsec                          
    r = aper_diam / (2 * pix_scale) # radius of aperture in pixels
    if type(r) == u.Quantity:
        r = r.value   
    if fast:
        r = 1e-10
        
    xoff, yoff = calc_xy_offsets(offset)
    
    
    xchunk = int(seg_data.shape[1])
    ychunk = int(seg_data.shape[0])
    xcoord = list()
    ycoord = list()
    # finds locations to place empty apertures in
    for i in tqdm(range(0, int((xchunk - (2 * xoff)) / size)), desc = f"Running {band} depths for {survey}"):
        for j in tqdm(range(0, int((ychunk - (2 * yoff)) / size)), desc = f"Current row = {i + 1}", leave = False):
            busyflag = 0
            # narrow seg, image and mask data to appropriate size for the chunk
            seg_chunk = seg_data[(j * size) + yoff : ((j + 1) * size) + yoff, (i * size) + xoff:((i + 1) * size) + xoff]
            aper_mask_chunk = copy.deepcopy(seg_chunk)
            im_chunk = im_data[(j*size)+yoff:((j+1)*size)+yoff, (i*size)+xoff:((i+1)*size)+xoff]
            mask_chunk = mask[(j*size)+yoff:((j+1)*size)+yoff, (i*size)+xoff:((i+1)*size)+xoff] # cut the box of interest out of the images
            xlen = seg_chunk.shape[1]
            ylen = seg_chunk.shape[0]
            
            # check if there is enough space to fit apertures even if perfectly aligned
            # if i == 16 and j == 16:
            #     print(np.argwhere(seg_chunk == 0), np.argwhere(im_chunk != 0), np.argwhere(mask_chunk == False), np.argwhere(aper_mask_chunk == 0))
            z = np.argwhere((seg_chunk == 0) & (im_chunk != 0.) & (mask_chunk == False) & (aper_mask_chunk == 0)) #generate a list of candidate locations for empty apertures
            #print(z)
            if len(z) > 0:
                space = True
            else:
                space = False
            if space: # there is space for "number" of apertures
                 # cycle through range of available locations for empty apertures
                 for c in range(0, number): # tqdm(), desc = "Grid square completion", total = number * 0.6, leave = False):
                     next = 0
                     iters = 0
                     
                     while next == 0:
     
                         idx = randrange(len(z)) # find random candidate location for empty aperture
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
                                 
                     if busyflag == 1: # if the region is busy, set the remaining apertures to co-ordinates (0., 0.)
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


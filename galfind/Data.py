#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:20:31 2023

@author: austind
"""

from __future__ import absolute_import
import photutils
from photutils import Background2D, MedianBackground, SkyCircularAperture, aperture_photometry
import numpy as np
from astropy.io import fits
from random import randrange
from pathlib import Path
import sep # sextractor for python
import matplotlib.pyplot as plt
import cv2
import math
import timeit
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.wcs.utils import pixel_to_skycoord
from astropy.coordinates import search_around_sky, SkyCoord
from astropy.visualization.mpl_normalize import ImageNormalize
import astropy.visualization as vis
from matplotlib.colors import LogNorm
from astropy.table import Table, hstack, vstack, Column
from copy import copy, deepcopy
import pyregion
from regions import Regions
import subprocess
import time
import glob
import astropy.units as u
import os
import sys
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import json
from joblib import Parallel, delayed
#from reproject import reproject_adaptive
import contextlib
import joblib
import h5py
from tqdm import tqdm
import logging
from astroquery.gaia import Gaia

from .Instrument import Instrument, ACS_WFC, WFC3_IR, NIRCam, MIRI, Combined_Instrument
from . import config, galfind_logger, Depths
from . import useful_funcs_austind as funcs
from .decorators import run_in_dir, hour_timer, email_update

# GALFIND data object
class Data:
    
    def __init__(self, instrument, im_paths, im_exts, im_pixel_scales, im_shapes, im_zps, wht_paths, wht_exts, rms_err_paths, rms_err_exts, \
        seg_paths, mask_paths, cluster_mask_path, blank_mask_path, survey, version, cat_path = "", is_blank = True, alignment_band = "F444W", \
        RGB_method = None, mask_stars = True): # trilogy
        
        # sort dicts from blue -> red bands in ascending wavelength order
        self.im_paths = im_paths # not sure these need to be sorted
        self.im_exts = im_exts # not sure these need to be sorted
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
        if alignment_band not in self.instrument.band_names:
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
        existing_seg_paths = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/SExtractor/*/{version}/{survey}/{survey}_*_*_sel_cat_{version}_seg.fits")
        [self.make_seg_map(band) for band, path in seg_paths.items() if path not in existing_seg_paths]
        # for i, (band, seg_path) in enumerate(seg_paths.items()):
        #     #print(band, seg_path)
        #     if (seg_path == "" or seg_path == []):
        #         self.make_seg_map(band)
        #     # load segmentation map
        #     seg_paths[band] = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/SExtractor/{self.instrument.instrument_from_band(band).name}/{version}/{survey}/{survey}_{band}_{band}_sel_cat_{version}_seg.fits")[0]
        self.seg_paths = seg_paths #dict(sorted(seg_paths.items())) 
        # make masks from image paths if they don't already exist
        self.mask_dir = f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}"
        #print(mask_paths)
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
                if type(mask_stars) == bool:
                    mask = self.make_mask(band, mask_stars = mask_stars)
                elif type(mask_stars) == dict:
                    if band in mask_stars.keys():
                        mask = self.make_mask(band, mask_stars = mask_stars[band])
                    else:
                        band_instrument = instrument[np.where(band == instrument.band_names)[0][0]].instrument
                        if band_instrument in mask_stars.keys():
                            mask = self.make_mask(band, mask_stars = mask_stars[band_instrument])
                        else:
                            mask = self.make_mask(band) # default behaviour
                else:
                    galfind_logger.critical(f"{mask_stars=} with {type(mask_stars)=} not in [bool, dict]")
                mask_paths[band] = mask
                
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
                    #breakpoint()
                    mask_path = self.clean_mask_regions(mask_path)
                    mask_path = self.mask_reg_to_pix(self.alignment_band, mask_path)
                else:
                    galfind_logger.critical(f"{mask_type.capitalize()} mask does not exist for {survey} and no blank field/cluster auto-masking has yet been implemented!")
            self.blank_mask_path = blank_mask_path
            self.cluster_mask_path = cluster_mask_path
        
        # find common directories for im/seg/rms_err/wht maps
        self.common_dirs = {}
        galfind_logger.warning(f"self.common_dirs has errors when the len(rms_err_paths) = {len(self.rms_err_paths)} != len(wht_paths) = {len(self.wht_paths)}")
        if len(self.rms_err_paths) == len(self.wht_paths):
            for paths, key in zip([im_paths, seg_paths, mask_paths, rms_err_paths, wht_paths], ["SCI", "SEG", "MASK", "ERR", "WHT"]):
                try:
                    for band in self.instrument.band_names:
                        assert("/".join(paths[band].split("/")[:-1]) == "/".join(paths[self.instrument.band_names[0]].split("/")[:-1]))
                    self.common_dirs[key] = "/".join(paths[self.instrument.band_names[0]].split("/")[:-1])
                    galfind_logger.info(f"Common directory found for {key}: {self.common_dirs[key]}")
                except AssertionError:
                    galfind_logger.info(f"No common directory for {key}")

            # find other things in common between bands
            self.common = {}
            for label, item_dict in zip(["ZERO POINT", "PIXEL SCALE", "SCI SHAPE"], [self.im_zps, self.im_pixel_scales, self.im_shapes]):
                try:
                    for band in self.instrument.band_names:
                        assert(item_dict[band] == item_dict[self.instrument.band_names[0]])
                    self.common[label] = item_dict[self.instrument.band_names[0]]
                    galfind_logger.info(f"Common {label} found")
                except AssertionError:
                    galfind_logger.info(f"No common {label}")

            # make RGB using the default method if the science images have a common shape
            if "SCI SHAPE" in self.common.keys() and type(RGB_method) != type(None):
                split_bands = np.split(self.instrument.band_names, \
                    [int(np.round(len(self.instrument.band_names) / 3, 0)), -int(np.round(len(self.instrument.band_names) / 3, 0))])
                self.make_RGB(list(split_bands[0]), list(split_bands[1]), list(split_bands[2]), RGB_method)
                #split_bands = np.take(self.instrument.band_names, [0, int(len(self.instrument.band_names) / 2), -1])
                #self.make_RGB([split_bands[0]], [split_bands[1]], [split_bands[2]], RGB_method)

    @classmethod
    def from_pipeline(cls, survey, version, instrument_names = ["ACS_WFC", "WFC3_IR", "NIRCam", "MIRI"], excl_bands = [], \
            pix_scales = {"ACS_WFC": 0.03 * u.arcsec, "WFC3_IR": 0.03 * u.arcsec, "NIRCam": 0.03 * u.arcsec, "MIRI": 0.09 * u.arcsec}, \
            im_str = ["_sci", "_i2d", "_drz"], rms_err_str = ["_rms", "_err"], wht_str = ["_wht", "_weight"], mask_stars = True):
        
        # may not require all of these inits
        im_paths = {} 
        im_exts = {}
        im_shapes = {}
        seg_paths = {}
        wht_paths = {}
        wht_exts = {}
        rms_err_paths = {}
        rms_err_exts = {}
        im_pixel_scales = {}
        im_zps = {}
        mask_paths = {}
        depth_dir = {}
        is_blank = is_blank_survey(survey)
        instrument_arr = []

        # construct dict containing appropriate directories for the given version, survey etc
        NIRCam_version_to_dir = {"v8b": "mosaic_1084_wispfix", "v8c": "mosaic_1084_wispfix2", "v8d": "mosaic_1084_wispfix3", \
            "v9": "mosaic_1084_wisptemp2", "v10": "mosaic_1084_wispscale", "v11": "mosaic_1084_wispnathan"}
        #MIRI_version_to_dir = "60mas"
        instrument_version_to_dir = {**{"NIRCam": NIRCam_version_to_dir}, \
            **{instrument_name: f"{int(np.round(pix_scales[instrument_name].to(u.mas).value, 0))}mas" \
            for instrument_name in ["ACS_WFC", "WFC3_IR", "MIRI"]}}

        #breakpoint()
        for instrument_name in instrument_names:
            instrument = globals()[instrument_name](excl_bands = [band_name for band_name in excl_bands if band_name in globals()[instrument_name]().band_names])
            version_to_dir = instrument_version_to_dir[instrument_name]
            pix_scale = pix_scales[instrument_name]
            
            # determine directory where the data is stored for the version, survey and instrument
            #breakpoint()
            if type(version_to_dir) == str:
                survey_dir = f"{config['DEFAULT']['GALFIND_DATA']}/{instrument.facility.lower()}/{survey}/{instrument_name}/{version_to_dir}"
            elif type(version_to_dir) == dict:
                if version_to_dir == {}:
                    continue
                elif version.split("_")[0] in version_to_dir.keys():
                    survey_dir = f"{config['DEFAULT']['GALFIND_DATA']}/{instrument.facility.lower()}/{survey}/{instrument_name}/{version_to_dir[version.split('_')[0]]}"
                else:
                    survey_dir = f"{config['DEFAULT']['GALFIND_DATA']}/{instrument.facility.lower()}/{survey}/{instrument_name}/{version}"
                if len(version.split('_')) > 1:
                    survey_dir += f"_{'_'.join(version.split('_')[1:])}"
            else:
                galfind_logger.critical(f"{version_to_dir=} with {type(version_to_dir)=} not in [str, dict]")
            
            # extract all .fits files in this directory
            fits_path_arr = np.array(glob.glob(f"{survey_dir}/*.fits"))
            if len(fits_path_arr) == 0:
                continue

            # obtain available bands from image paths
            bands = np.array([band for path in fits_path_arr for band in instrument.band_names if band.upper() in path \
                or band.lower() in path or band.lower().replace('f', 'F') in path or band.upper().replace('F', 'f') in path])
            galfind_logger.warning("Should check more thoroughly to ensure there are not multiple band names in an image path!")
            
            #breakpoint()
            # if there are multiple images per band, separate into im/wht/rms_err
            unique_bands, band_indices, n_images = np.unique(bands, return_inverse = True, return_counts = True)
            galfind_logger.debug("These assertions may need to change in the case of e.g. stacking multiple images of same band")
            assert all(n == n_images[0] for n in n_images) # throw more appropriate warnings here
            assert n_images[0] in [1, 2, 3]
            if n_images[0] == 1:
                im_path_arr = fits_path_arr
                assert len(bands) == len(im_path_arr)
                instr_im_paths = {band: path for band, path in zip(bands, im_path_arr)}
                im_paths = {**im_paths, **instr_im_paths}
                # extract sci/rms_err/wht extensions from single band image
                for band in tqdm(bands, total = len(bands), desc = f"Extracting SCI/WHT/ERR extensions/shapes for {survey} {version} {instrument_name}"):
                    im_hdul = fits.open(im_paths[band])
                    for j, im_hdu in enumerate(im_hdul):
                        if instrument_name == "WFC3_IR":
                            breakpoint()
                        if im_hdu.name == "SCI":
                            im_exts[band] = int(j)
                            im_shapes[band] = im_hdu.data.shape
                        if im_hdu.name == 'WHT':
                            wht_exts[band] = int(j)
                            wht_paths[band] = str(im_paths[band])
                        if im_hdu.name == 'ERR':
                            rms_err_exts[band] = int(j)
                            rms_err_paths[band] = str(im_paths[band])
            else:
                im_path_arr = np.array([fits_path for band_index in np.unique(band_indices) for fits_path in \
                    fits_path_arr[band_indices == band_index] if any(str in fits_path for str in im_str) \
                    and not any(str in fits_path for str in rms_err_str) and not any(str in fits_path for str in wht_str)])
                rms_err_path_arr = np.array([fits_path for band_index in np.unique(band_indices) for fits_path in \
                    fits_path_arr[band_indices == band_index] if any(str in fits_path for str in rms_err_str) \
                    and not any(str in fits_path for str in wht_str)])
                wht_path_arr = np.array([fits_path for band_index in np.unique(band_indices) for fits_path in \
                    fits_path_arr[band_indices == band_index] if any(str in fits_path for str in wht_str)])
                # stack bands if there are more than 1 im_path for each band - NOT YET IMPLEMENTED
                assert len(np.array([path for path in fits_path_arr if path not in \
                    np.concatenate((im_path_arr, rms_err_path_arr, wht_path_arr))])) == 0
                # save paths to sci, rms_err, and wht maps
                im_paths = {**im_paths, **{band: path for band, path in zip(unique_bands, im_path_arr)}}
                rms_err_paths = {**rms_err_paths, **{band: path for band, path in zip(unique_bands, rms_err_path_arr)}}
                wht_paths = {**wht_paths, **{band: path for band, path in zip(unique_bands, wht_path_arr)}}
                # extract sci/rms_err/wht extensions from single band image

                #breakpoint()
                for band in bands:
                    im_hdul = fits.open(im_paths[band])
                    assertion_len = 1
                    for j, im_hdu in enumerate(im_hdul):
                        if im_hdu.name == "PRIMARY" and len(im_hdul) > 1:
                            assertion_len += 1
                        else:
                            im_exts[band] = int(j)
                            im_shapes[band] = im_hdul[0].data.shape
                    #breakpoint()
                    #print(band, len(im_hdul), [hdu.name for hdu in im_hdul], assertion_len)
                    assert len(im_hdul) == assertion_len
                    if band in rms_err_paths.keys():
                        rms_err_hdul = fits.open(rms_err_paths[band])
                        assertion_len = 1
                        for j, rms_err_hdu in enumerate(rms_err_hdul):
                            if rms_err_hdu.name == "PRIMARY" and len(rms_err_hdul) > 1:
                                assertion_len += 1
                            else:
                                rms_err_exts[band] = int(j)
                        #breakpoint()
                        #print(band, len(rms_err_hdul), [hdu.name for hdu in rms_err_hdul], assertion_len)
                        assert len(rms_err_hdul) == assertion_len
                    if band in wht_paths.keys():
                        wht_hdul = fits.open(wht_paths[band])
                        assertion_len = 1
                        for j, wht_hdu in enumerate(wht_hdul):
                            if wht_hdu.name == "PRIMARY" and len(wht_hdul) > 1:
                                assertion_len += 1
                            else:
                                wht_exts[band] = int(j)
                        #breakpoint()
                        #print(band, len(wht_hdul), [hdu.name for hdu in wht_hdul], assertion_len)
                        assert len(wht_hdul) == assertion_len

            #breakpoint()
            # if band not used in instrument remove it, else save pixel scale and zero point
            for band_name in deepcopy(instrument).band_names:
                if band_name not in unique_bands:
                    instrument.remove_band(band_name)
                else:
                    im_pixel_scales[band_name] = pix_scale
                    # calculate ZP given image units
                    if instrument_name == "ACS_WFC":
                        imheader = fits.open(str(im_paths[band_name]))[im_exts[band_name]].header
                        if "PHOTFLAM" in imheader and "PHOTPLAM" in imheader:
                            im_zps[band_name] = -2.5 * np.log10(imheader["PHOTFLAM"]) - 21.1 - 5 * np.log10(imheader["PHOTPLAM"]) + 18.6921
                        elif "ZEROPNT" in imheader:
                            im_zps[band_name] = imheader["ZEROPNT"]
                        else:
                            raise(Exception(f"ACS_WFC data for {survey} {version} {band_name} located at {im_paths[band_name]} must contain either 'ZEROPNT' or 'PHOTFLAM' and 'PHOTPLAM' in its header to calculate its ZP!"))
                    elif instrument_name == "WFC3_IR":
                        # Taken from Appendix A of https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/documentation/instrument-science-reports-isrs/_documents/2020/WFC3-ISR-2020-10.pdf
                        wfc3ir_zps = {'F098M': 25.661, 'F105W': 26.2637, 'F110W': 26.8185, 'F125W': 26.231, 'F140W': 26.4502, 'F160W': 25.9362}
                        im_zps[band_name] = wfc3ir_zps[band_name]
                    elif instrument_name == "NIRCam" or instrument_name == "MIRI":
                        # assume flux units of MJy/sr and calculate corresponding ZP
                        im_zps[band_name] = -2.5 * np.log10((pix_scale.to(u.rad).value ** 2) * u.MJy.to(u.Jy)) + u.Jy.to(u.ABmag)
            instrument_arr.append(instrument)
        #breakpoint()
        if len(instrument_arr) == 1:
            comb_instrument = instrument_arr[0]
        else:
            comb_instrument = np.sum(instrument_arr)
        
        # All seg maps and masks should be in same format, so load those last when we know what bands we have
        start = time.time()

        # re-write to avoid using glob.glob more than absolutely necessary
        fits_mask_paths = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/fits_masks/*.fits")
        #breakpoint()
        for i, band in enumerate(comb_instrument.band_names):
            # load path to segmentation map
            seg_paths[band] = f"{config['DEFAULT']['GALFIND_WORK']}/SExtractor/{comb_instrument.instrument_from_band(band).name}/{version}/{survey}/{survey}_{band}_{band}_sel_cat_{version}_seg.fits"

            # try:
            #     #print(f"{config['DEFAULT']['GALFIND_WORK']}/SExtractor/{comb_instrument.instrument_from_band(band)}/{version}/{survey}/{survey}*{band}_{band}*{version}*seg.fits")
            #     seg_paths[band] = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/SExtractor/{comb_instrument.instrument_from_band(band).name}/{version}/{survey}/{survey}*{band}_{band}*{version}*seg.fits")[0]
            # except IndexError:
            #     seg_paths[band] = ""

            # load fits mask for band before searching for other existing masks
            paths_to_masks = [f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/fits_masks/{path.split('/')[-1]}" for path in fits_mask_paths \
                if path.split("/")[-1].split("_")[0] == band and "basemask" in path.split("/")[-1].split("_")[1]]
            print(band)
            assert len(paths_to_masks) <= 1, galfind_logger.critical(f"{len(paths_to_masks)=} > 1")
            #breakpoint()
            if len(paths_to_masks) == 0:
                # search for manually created mask
                manual_mask_path = f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/manual/{survey}_{band}_clean.reg"
                if Path(manual_mask_path).is_file():
                    mask_paths[band] = manual_mask_path
                else: # if no manually created mask, leave blank
                    mask_paths[band] = ""
            else:
                mask_paths[band] = paths_to_masks[0]

            # # include just the masks corresponding to the correct bands
            # fits_mask_paths_ = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/fits_masks/*{band.lower()}*")
            # fits_mask_paths_ += glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/fits_masks/*{band.upper()}*")
            # fits_mask_paths_ += glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/fits_masks/*{band.lower().replace('f', 'F')}*")
            # fits_mask_paths_ += glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/fits_masks/*{band.upper().replace('F', 'f')}*")
            # fits_mask_paths_ = list(set(fits_mask_paths_))
            # # Exclude mask path if it contains the phrase 'artifact'
            # fits_mask_paths_ = [path for path in fits_mask_paths_ if 'artifact' not in path]

            # mid = time.time()
            # print(mid - start)
            # # exclude masks that include other filters
            # delete_indices = []
            # # create a new combined instrument including all possible bands
            # new_instrument = comb_instrument.new_instrument()
            # new_instrument.remove_band(band)
            # for j, path in enumerate(fits_mask_paths_):
            #     for k, band_ in enumerate(new_instrument.band_names):
            #         if band_.lower() in path or band_.upper() in path or band_.lower().replace('f', 'F') in path or band_.upper().replace('F', 'f') in path:
            #             delete_indices.append(j)
            #             break
            # fits_mask_paths_ = np.delete(fits_mask_paths_, delete_indices)

            # mid = time.time()
            # print(mid - start)

            # if len(fits_mask_paths_) == 1:
            #     mask_paths[band] = fits_mask_paths_[0]
            # elif len(fits_mask_paths_) == 0:
            #     mask_paths_ = glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/*{band.lower()}*")
            #     mask_paths_ += glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/*{band.upper()}*")
            #     mask_paths_ += glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/*{band.lower().replace('f', 'F')}*")
            #     mask_paths_ += glob.glob(f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{survey}/*{band.upper().replace('F', 'f')}*")
            #     mask_paths_ = list(set(mask_paths_))
            #     # Exclude mask path if it contains the phrase 'artifact'
            #     mask_paths_ = [path for path in mask_paths_ if 'artifact' not in path]
            #     # exclude masks that include other filters
            #     delete_indices = []
            #     for j, path in enumerate(mask_paths_):
            #         for k, band_ in enumerate(np.delete(comb_instrument.band_names, i)):
            #             if band_.lower() in path or band_.upper() in path or band_.lower().replace('f', 'F') \
            #                     in path or band_.upper().replace('F', 'f') in path:
            #                 delete_indices.append(j)
            #                 break
            #     mask_paths_ = np.delete(mask_paths_, delete_indices)
            #     if len(mask_paths_) == 0:
            #         mask_paths[band] = ""
            #     elif len(mask_paths_) == 1:
            #         mask_paths[band] = mask_paths_[0]
            #     else:
            #         raise(Exception(f"Too many region masks found for {survey} {band}! Masks are {mask_paths_}!"))
            # else:
            #     raise(Exception(f"Too many fits masks found for {survey} {band}! Masks are {fits_mask_paths_}!"))
        
            mid = time.time()
            print(mid - start)

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

        end = time.time()
        print(f"Loading mask paths took {end - start}s")
        #breakpoint()

        return cls(comb_instrument, im_paths, im_exts, im_pixel_scales, im_shapes, im_zps, wht_paths, wht_exts, rms_err_paths, rms_err_exts, \
            seg_paths, mask_paths, cluster_mask_path, blank_mask_path, survey, version, is_blank = is_blank, mask_stars = mask_stars)

# %% Overloaded operators

    def __str__(self):
        """ Function to print summary of Data class

        Returns:
            str: Summary containing survey name, version, and whether field is blank or cluster.
                Includes summary of Instrument class, including bands, instruments and facilities used.
                Image depths in relevant aperture sizes are included here if calculated.
                Masked/unmasked areas are also quoted here.
                Also includes paths/extensions to SCI/SEG/ERR/WHT/MASK in each band, pixel scales, zero points and fits shapes.
        """
        line_sep = "*" * 40 + "\n"
        band_sep = "-" * 10 + "\n"
        output_str = line_sep
        output_str += "DATA OBJECT:\n"
        output_str += band_sep
        output_str += f"SURVEY: {self.survey}\n"
        output_str += f"VERSION: {self.version}\n"
        output_str += f"FIELD TYPE: " + "BLANK\n" if self.is_blank else "CLUSTER\n"
        # print instrument string representation
        output_str += str(self.instrument) # should include aperture sizes + aperture corrections + paths to WebbPSF models
        # print basic data quantities: common ZPs, pixel scales, and SCI image shapes, as well as unmasked sky area and depths should they exist
        for (key, item) in self.common.items():
            output_str += f"{key}: {item}\n"
        try:
            unmasked_area_tab = self.calc_unmasked_area(masking_instrument_or_band_name = self.forced_phot_band, forced_phot_band = self.forced_phot_band) 
            unmasked_area = unmasked_area_tab[unmasked_area_tab["masking_instrument_band"] == 'NIRCam']['unmasked_area_total'][0] 
            output_str += f"UNMASKED AREA = {unmasked_area}\n"
        except:
            pass
        try:
            #breakpoint()
            depths = []
            for aper_diam in json.loads(config.get("SExtractor", "APERTURE_DIAMS")) * u.arcsec:
                depths.append(self.load_depths(aper_diam))
            #breakpoint()
            output_str += f"DEPTHS = {str(depths)}\n"
        except:
            pass
        # if there are common directories for data, print these
        if self.common_dirs != {}:
            output_str += line_sep
            output_str += "SHARED DIRECTORIES:\n"
            for (key, value) in self.common_dirs.items():
                output_str += f"{key}: {value}\n"
            output_str += line_sep
        # loop through available bands, printing paths, exts, ZPs, fits shapes
        output_str += "BAND DATA:\n"
        for band in self.instrument.band_names:
            output_str += band_sep
            output_str += f"{band}\n"
            if hasattr(self, "sex_cat_types"):
                if band in self.sex_cat_types.keys():
                    output_str += f"PHOTOMETRY BY: {self.sex_cat_types[band]}\n"
            band_data_paths = [self.im_paths[band], self.seg_paths[band], self.mask_paths[band]]
            band_data_exts = [self.im_exts[band], 0, 0]
            band_data_labels = ["SCI", "SEG", "MASK"]
            for paths, exts, label in zip([self.rms_err_paths, self.wht_paths], [self.rms_err_exts, self.wht_exts], ["ERR", "WHT"]):
                if band in self.rms_err_paths.keys():
                    band_data_paths.append(paths[band])
                    band_data_exts.append(exts[band])
                    band_data_labels.append(label)
            for path, ext, label in zip(band_data_paths, band_data_exts, band_data_labels):
                if label in self.common_dirs:
                    path = path.split("/")[-1]
                output_str += f"{label} path = {path}[{str(ext)}]\n"
            for label, data in zip(["ZERO POINT", "PIXEL SCALE", "SCI SHAPE"], [self.im_zps, self.im_pixel_scales, self.im_shapes]):
                if label not in self.common.keys():
                    output_str += f"{label} = {data[band]}\n"
        output_str += line_sep
        return output_str
    
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
        return self.load_data(self.instrument.band_names[index], incl_mask = True)
    
    def __add__(self, data):
        common_bands = [band for band in data.instrument.band_names if band in self.instrument.band_names]
        if len(common_bands) != 0:
            raise(Exception(f"Cannot add two of the same bands from different instruments together! Culprits: {common_bands}"))
        # add all dictionaries together
        self.__dict__ = {k: self.__dict__.get(k, 0) + data.__dict__.get(k, 0) for k in set(self.__dict__()) | set(data.__dict__())}
        galfind_logger.debug(self.__repr__)
        return self
    
# %% Methods

    def load_data(self, band, incl_mask = True):
        #print(band)
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
    
    def load_wcs(self, band, save_attr = True):
        try:
            self.wcs[band]
        except (AttributeError, KeyError) as e:
            if type(e) == AttributeError:
                self.wcs = {}
            self.wcs[band] = WCS(self.load_im(band)[1])
        return self.wcs[band]

    def load_seg(self, band):
        seg_hdul = fits.open(self.seg_paths[band])
        seg_data = seg_hdul[0].data
        seg_header = seg_hdul[0].header
        return seg_data, seg_header
    
    def load_wht(self, band):
        try:
            wht = fits.open(self.wht_paths[band])[self.wht_exts[band]].data
        except Exception as e:
            print(e)
            wht = None
        return wht

      
    def load_rms_err(self, band):
        try:
            rms_err = fits.open(self.rms_err_paths[band])[self.rms_err_exts[band]].data
        except:
            rms_err = None
        return rms_err
    
    
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

    def make_RGB(self, blue_bands = ["F090W"], green_bands = ["F200W"], red_bands = ["F444W"], method = "trilogy"):
        # ensure all blue, green and red bands are contained in the data object
        assert all(band in self.instrument.band_names for band in blue_bands + green_bands + red_bands), \
            galfind_logger.warning(f"Cannot make galaxy RGB as not all {blue_bands + green_bands + red_bands} are in {self.instrument.band_names}")
        # construct out_path
        out_path = f"{config['RGB']['RGB_DIR']}/{self.version}/{self.survey}/{method}/B={'+'.join(blue_bands)},G={'+'.join(green_bands)},R={'+'.join(red_bands)}.png"
        funcs.make_dirs(out_path)
        if not os.path.exists(out_path):
            # load RGB band paths including .fits image extensions
            RGB_paths = {}
            for colour, bands in zip(["B", "G", "R"], [blue_bands, green_bands, red_bands]):
                RGB_paths[colour] = [f"{self.im_paths[band]}[{self.im_exts[band]}]" for band in bands]
            if method == "trilogy":
                # Write trilogy.in
                in_path = out_path.replace(".png", "_trilogy.in")
                with open(in_path, "w") as f:
                    for colour, paths in RGB_paths.items():
                        f.write(f"{colour}\n")
                        for path in paths:
                            f.write(f"{path}\n")
                        f.write("\n")
                    f.write("indir  /\n")
                    f.write(f"outname  {funcs.split_dir_name(out_path, 'name').replace('.png', '')}\n")
                    f.write(f"outdir  {funcs.split_dir_name(out_path, 'dir')}\n")
                    f.write("samplesize 20000\n")
                    f.write("stampsize  2000\n")
                    f.write("showstamps  0\n")
                    f.write("satpercent  0.001\n")
                    f.write("noiselum    0.10\n")
                    f.write("colorsatfac  1\n")
                    f.write("deletetests  1\n")
                    f.write("testfirst   0\n")
                    f.write("sampledx  0\n")
                    f.write("sampledy  0\n")
                # Run trilogy
                sys.path.insert(1, "/nvme/scratch/software/trilogy") # Not sure why this path doesn't work: config["Other"]["TRILOGY_DIR"]
                from trilogy3 import Trilogy
                galfind_logger.info(f"Making full trilogy RGB image at {out_path}")
                Trilogy(in_path, images = None).run()
            elif method == "lupton":
                raise(NotImplementedError())
            
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

    def make_mask(self, band, edge_mask_distance = 50, mask_stars = True, scale_extra = 0.2,
            star_mask_override = None, exclude_gaia_galaxies = True, angle = 75., edge_value = 0., 
            element = 'ELLIPSE', gaia_row_lim = 500, plot = False):
        
        breakpoint()
        if 'NIRCam' not in self.instrument.name and mask_stars:
            galfind_logger.critical(f"Mask making only implemented for NIRCam data!")
            raise(Exception("Star mask making only implemented for NIRCam data!"))

        # if "COSMOS-Web" in self.survey:
        #     # stellar masks the same for all bands
        #     star_mask_params = { # mask_a * exp(-mag / mask_b) is the form 
        #         9000 * u.AA: {'mask_a': 700, 'mask_b': 3.7}}
        # else:
        #     star_mask_params = { # mask_a * exp(-mag / mask_b) is the form 
        #         9000 * u.AA: {'mask_a': 1300, 'mask_b': 4},
        #         11500 * u.AA: {'mask_a': 1300, 'mask_b': 4},
        #         15000 * u.AA: {'mask_a': 1300, 'mask_b': 4},
        #         20000 * u.AA: {'mask_a': 1300, 'mask_b': 4},
        #         27700 * u.AA: {'mask_a': 1000, 'mask_b': 3.7},
        #         35600 * u.AA: {'mask_a': 800, 'mask_b': 3.7},
        #         44000 * u.AA: {'mask_a': 800, 'mask_b': 3.7},
        #     }

        # update to change scaling of central circle independently of spikes
        star_mask_params_dict = { # a * exp(-mag / b) in arcsec
            11500 * u.AA: {"central": {'a': 300., 'b': 4.25}, "spikes": {"a": 400., "b": 4.5}},
        }

        if star_mask_override != None:
            assert type(star_mask_override) == dict, galfind_logger.warning(f"Mask overridden, but {type(star_mask_override)=} != dict")
            assert "central" in star_mask_override.keys() and "spikes" in star_mask_override.keys()
            assert type(star_mask_override["central"]) == dict, galfind_logger.warning(f"Mask overridden, but {type(star_mask_override['central'])=} != dict")
            assert "a" in star_mask_override["central"].keys() and "b" in star_mask_override["central"].keys()
            assert type(star_mask_override["spikes"]) == dict, galfind_logger.warning(f"Mask overridden, but {type(star_mask_override['spikes'])=} != dict")
            assert "a" in star_mask_override["spikes"].keys() and "b" in star_mask_override["spikes"].keys()
            assert all(type(scale) in [float, int] for mask_type in star_mask_override.values() for scale in mask_type.values())
            star_mask_params = star_mask_override
        else:
            band_wavelength = self.instrument.band_wavelengths[band == self.instrument.band_names]
            # Get closest wavelength parameters
            closest_wavelength = min(star_mask_params_dict.keys(), key = lambda x: abs(x - band_wavelength))
            print(band, closest_wavelength)
            star_mask_params = star_mask_params_dict[closest_wavelength]

        galfind_logger.info(f"Automasking {self.survey} {band}.")

        composite = lambda x_coord, y_coord, central_scale, spike_scale, angle: \
            f'''# Region file format: DS9 version 4.1
            global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
            image
            composite({x_coord},{y_coord},0.00) || composite=1
                circle({x_coord},{y_coord},{163*central_scale}) ||
                ellipse({x_coord},{y_coord},{29*spike_scale**(2/3)},{730*spike_scale},{str(np.round(300.15 + angle, 2))}) ||
                ellipse({x_coord},{y_coord},{29*spike_scale**(2/3)},{730*spike_scale},{str(np.round(240. + angle, 2))}) ||
                ellipse({x_coord},{y_coord},{29*spike_scale**(2/3)},{730*spike_scale},{str(np.round(360. + angle, 2))}) ||
                ellipse({x_coord},{y_coord},{29*spike_scale**(2/3)},{300*spike_scale},{str(np.round(269.48 + angle, 2))}) ||'''
        breakpoint()
        # Load data
        im_data, im_header, seg_data, seg_header = self.load_data(band, incl_mask = False)
        pixel_scale = self.im_pixel_scales[band]
        wcs = WCS(im_header)

        # Scale up the image by boundary by scale_extra factor to include diffraction spikes from stars outside image footprint
        scale_factor = scale_extra * np.array([im_data.shape[1], im_data.shape[0]])
        vertices_pix = [(-scale_factor[0], -scale_factor[1]), (-scale_factor[0], im_data.shape[0] + scale_factor[1]), \
            (im_data.shape[1] + scale_factor[0], im_data.shape[0] + scale_factor[1]), (im_data.shape[1] + scale_factor[0], -scale_factor[1])]    
        # Convert to sky coordinates
        vertices_sky = wcs.all_pix2world(vertices_pix, 0)

        # Diagnostic plot
        if plot:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection=wcs)
            stretch = vis.CompositeStretch(vis.LogStretch(), vis.ContrastBiasStretch(contrast = 30, bias = 0.08))    
            norm = ImageNormalize(stretch = stretch, vmin = 0.001, vmax = 10)

            ax.imshow(im_data, cmap='Greys', origin='lower', interpolation='None', norm=norm)

        if mask_stars:
            print(f"Making stellar mask for {band}")
            # Get list of Gaia stars in the polygon region
            Gaia.ROW_LIMIT = gaia_row_lim
            # Construct the ADQL query string
            adql_query = \
                f"""
                SELECT source_id, ra, dec, phot_g_mean_mag, radius_sersic, classlabel_dsc_joint, vari_best_class_name
                FROM gaiadr3.gaia_source 
                LEFT OUTER JOIN gaiadr3.galaxy_candidates USING (source_id) 
                WHERE 1 = CONTAINS(
                    POINT('ICRS', ra, dec), 
                    POLYGON('ICRS', 
                        POINT('ICRS', {vertices_sky[0][0]}, {vertices_sky[0][1]}), 
                        POINT('ICRS', {vertices_sky[1][0]}, {vertices_sky[1][1]}), 
                        POINT('ICRS', {vertices_sky[2][0]}, {vertices_sky[2][1]}), 
                        POINT('ICRS', {vertices_sky[3][0]}, {vertices_sky[3][1]})))"""
    
            # Execute the query asynchronously
            job = Gaia.launch_job_async(adql_query)
            gaia_stars = job.get_results()
            print(f'Found {len(gaia_stars)} stars in the region.')
            if exclude_gaia_galaxies:
                gaia_stars = gaia_stars[gaia_stars['vari_best_class_name'] != 'GALAXY']
                gaia_stars = gaia_stars[gaia_stars['classlabel_dsc_joint'] != 'galaxy']
                # Remove masked flux values
                gaia_stars = gaia_stars[~np.isnan(gaia_stars['phot_g_mean_mag'])]
        
            ra_gaia = np.asarray(gaia_stars['ra'])
            dec_gaia = np.asarray(gaia_stars['dec'])
            x_gaia, y_gaia = wcs.all_world2pix(ra_gaia, dec_gaia, 0)
            
            # Generate mask scale for each star
            central_scale_stars = (2. * star_mask_params["central"]["a"] / (730. * pixel_scale.to(u.arcsec).value)) * np.exp(-gaia_stars["phot_g_mean_mag"] / star_mask_params["central"]["b"])
            spike_scale_stars = (2. * star_mask_params["spikes"]["a"] / (730. * pixel_scale.to(u.arcsec).value)) * np.exp(-gaia_stars["phot_g_mean_mag"] / star_mask_params["spikes"]["b"])
            # Update the catalog
            gaia_stars.add_column(Column(data = x_gaia, name = 'x_pix'))
            gaia_stars.add_column(Column(data = y_gaia, name = 'y_pix'))

            diffraction_regions = []
            region_strings = []
            for pos, (row, central_scale, spike_scale) in tqdm(enumerate(zip(gaia_stars, central_scale_stars, spike_scale_stars))):    
                # Plot circle
                # if plot:
                #     ax.add_patch(Circle((row['x_pix'], row['y_pix']), 2 * row['rmask_arcsec'] / pixel_scale, color = 'r', fill = False, lw = 2))
                sky_region = composite(row['x_pix'], row['y_pix'], central_scale, spike_scale, angle)
                region_obj = Regions.parse(sky_region, format = 'ds9')
                diffraction_regions.append(region_obj)
                region_strings.append(region_obj.serialize(format = 'ds9'))

            stellar_mask = np.zeros(im_data.shape, dtype=bool)
            for regions in tqdm(diffraction_regions):
                for region in regions:
                    idx_large, idx_little = region.to_mask(mode = 'center').get_overlap_slices(im_data.shape)
                    # idx_large is x,y box containing bounds of region in image
                    if idx_large is not None:
                        stellar_mask[idx_large] = np.logical_or(region.to_mask().data[idx_little], stellar_mask[idx_large])
                    if plot:
                        artist = region.as_artist()
                        ax.add_patch(artist)
        
        # Mask image edges
        fill = np.logical_or((im_data == edge_value), np.isnan(im_data))  #true false array of where 0's are
        # also fill in nans
        edges = fill * 1 #convert to 1 for true and 0 for false
        edges = edges.astype(np.uint8) #dtype for cv2
        print('Masking edges')
        if element == 'RECT':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_mask_distance,edge_mask_distance))
        elif element == 'ELLIPSE':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_mask_distance,edge_mask_distance))
        else:
            raise ValueError(f"element = {element} must be 'RECT' or 'ELLIPSE'")

        edge_mask = cv2.dilate(edges, kernel, iterations = 1) # dilate mask using the circle
        #edge_mask = 1 - dilate # invert mask, so it is 1 where it is not masked and 0 where it is masked
        #mask = 1 - edge_mask

        # Mask up to 50 pixels from all edges - so edge is still masked if it as at edge of array
        edge_mask[:edge_mask_distance, :] = edge_mask[-edge_mask_distance:, :] = edge_mask[:, :edge_mask_distance] = edge_mask[:, -edge_mask_distance:] = 1
        
        if mask_stars:
            full_mask = np.logical_or(edge_mask.astype(np.uint8), stellar_mask.astype(np.uint8))
        else:
            full_mask = edge_mask.astype(np.uint8)
        
        if plot:
            ax.imshow(full_mask, cmap='Reds', origin='lower', interpolation='None')
        
        # Check for artefacts mask to combine with exisitng mask
        files = glob.glob(f'{self.mask_dir}/{band}*.reg')
        # Check for 'artifact' in file name
        files = [file for file in files if 'artifact' in file]
        artifact_mask = None
        if len(files) > 0:
            artifact_mask = np.zeros(im_data.shape, dtype=bool)
            print(f'Found {len(files)} artifact masks')
            for file in files:
                print(f'Adding mask {file}')
                mask = Regions.read(file)
                
                for region in mask:
                    region = region.to_pixel(wcs)
                    idx_large, idx_little = region.to_mask(mode = 'center').get_overlap_slices(im_data.shape)
                    if idx_large is not None:
                        full_mask[idx_large] = np.logical_or(region.to_mask().data[idx_little], full_mask[idx_large])
                        artifact_mask[idx_large] = np.logical_or(region.to_mask().data[idx_little], artifact_mask[idx_large])

        # Save mask - could save independent layers as well e.g. stars vs edges vs manual mask etc
        output_mask_path = f'{self.mask_dir}/fits_masks/{band}_basemask.fits'

        os.makedirs("/".join(output_mask_path.split("/")[:-1]), exist_ok = True)
        full_mask_hdu = fits.ImageHDU(full_mask.astype(np.uint8), header = wcs.to_header(), name = "MASK")
        edge_mask_hdu = fits.ImageHDU(edge_mask.astype(np.uint8), header = wcs.to_header(), name = "EDGE")
        hdulist = [fits.PrimaryHDU(), full_mask_hdu, edge_mask_hdu]
        if mask_stars:
            stellar_mask_hdu = fits.ImageHDU(stellar_mask.astype(np.uint8), header = wcs.to_header(), name = "STELLAR")
            hdulist.append(stellar_mask_hdu)
        if artifact_mask is not None:
            artifact_mask_hdu = fits.ImageHDU(artifact_mask.astype(np.uint8), header = wcs.to_header(), name = "ARTIFACT")
            hdulist.append(artifact_mask_hdu)

        hdu = fits.HDUList(hdulist)
        hdu.writeto(output_mask_path, overwrite = True)
        # Change permission to read/write for all
        os.chmod(output_mask_path, 0o777)

        if plot:
            # Save mask plot
            fig.savefig(f'{self.mask_dir}/{band}_mask.png', dpi=300)

        # Save ds9 region
        if mask_stars:
            with open(f'{self.mask_dir}/{band}_starmask.reg', 'w') as f:
                for region in region_strings:
                    f.write(region + '\n')
            os.chmod(f'{self.mask_dir}/{band}_starmask.reg', 0o777)
        return output_mask_path
    
    #@staticmethod
    def combine_band_names(self, bands):
        if type(bands) == str:
            return bands
        else:
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
        print(band, self.rms_err_paths, self.wht_paths)
        #breakpoint()
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
        # insert specified aperture diameters from config file
        as_aper_diams = json.loads(config.get("SExtractor", "APERTURE_DIAMS"))
        pix_aper_diams = str([np.round(pix_aper_diam, 2) for pix_aper_diam in as_aper_diams / self.im_pixel_scales[band].value]).replace("[", "").replace("]", "").replace(" ", "")
        # SExtractor bash script python wrapper
        process = subprocess.Popen(["./make_seg_map.sh", config['DEFAULT']['GALFIND_WORK'], self.im_paths[band], str(self.im_pixel_scales[band].value), \
                                str(self.im_zps[band]), self.instrument.instrument_from_band(band).name, self.survey, band, self.version, err_map_path, \
                                err_map_ext, err_map_type, str(self.im_exts[band]), sex_config_path, params_path, pix_aper_diams])
        process.wait()
        galfind_logger.info(f"Made segmentation map for {self.survey} {self.version} {band} using config = {sex_config_path} and {err_map_type}")

    def make_seg_maps(self):
        for band in self.instrument.band_names:
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
        combined_mask_name = f"{self.mask_dir}/combined_masks/{self.survey}_{self.combine_band_names(bands)}.fits"
        if Path(combined_mask_name).is_file():
            self.mask_paths[stack_band_name] = combined_mask_name
        else:
            self.mask_paths[stack_band_name] = self.combine_masks(bands)
    
        if all(band in self.rms_err_paths.keys() and band in self.rms_err_exts.keys() for band in bands):
            err_from = "ERR"
        elif all(band in self.wht_paths.keys() and band in self.wht_exts.keys() for band in bands):
            # determine error map from wht map
            err_from = "WHT"
        else:
            raise(Exception("Inconsistent error maps for stacking bands"))

        if not Path(self.im_paths[stack_band_name]).is_file(): # or overwrite
            # if not config["DEFAULT"].getboolean("RUN"):
            #     galfind_logger.critical("RUN = YES, so not stacking detection bands. Returning Error.")
            #     raise Exception(f"RUN = YES, and combination of {self.survey} {self.version} or {self.instrument.name} has not previously been stacked.")
            funcs.make_dirs(self.im_paths[stack_band_name])
            
            for pos, band in enumerate(bands):
                if self.im_shapes[band] != self.im_shapes[bands[0]] or self.im_zps[band] != self.im_zps[bands[0]] or self.im_pixel_scales[band] != self.im_pixel_scales[bands[0]]:
                    raise Exception('All bands used in forced photometry stack must have the same shape, ZP and pixel scale!')
                
                prime_hdu = fits.open(self.im_paths[band])[0].header
                im_data, im_header = self.load_im(band)
                if err_from == "ERR":
                    err = fits.open(self.rms_err_paths[band])[self.rms_err_exts[band]].data
                elif err_from == "WHT":
                    # determine error map from wht map
                    wht = fits.open(self.wht_paths[band])[self.wht_exts[band]].data
                    err = np.sqrt(1. / wht)
                if pos == 0:
                    sum = im_data / err ** 2
                    sum_err = 1. / err ** 2
                else:
                    sum += im_data / err ** 2
                    sum_err += 1. / err ** 2
                
            weighted_array = sum / sum_err
            if err_from == "ERR":
                combined_err_or_wht = np.sqrt(1. / sum_err)
            elif err_from == "WHT":
                combined_err_or_wht = sum_err
            else:
                galfind_logger.critical(f"{err_from=} not in ['ERR', 'WHT']")
            
            #https://en.wikipedia.org/wiki/Inverse-variance_weighting

            primary = fits.PrimaryHDU(header = prime_hdu)
            hdu = fits.ImageHDU(weighted_array, header = im_header, name = "SCI")
            hdu_err = fits.ImageHDU(combined_err_or_wht, header = im_header, name = err_from)
            hdul = fits.HDUList([primary, hdu, hdu_err])
            hdul.writeto(self.im_paths[stack_band_name], overwrite = True)
            galfind_logger.info(f"Finished stacking bands = {bands} for {self.survey} {self.version}")
        
        # save forced photometry band parameters
        self.im_shapes[stack_band_name] = self.im_shapes[bands[0]]
        self.im_zps[stack_band_name] = self.im_zps[bands[0]]
        self.im_pixel_scales[stack_band_name] = self.im_pixel_scales[bands[0]]
        self.im_exts[stack_band_name] = 1
        if err_from == "ERR":
            self.rms_err_paths[stack_band_name] = self.im_paths[stack_band_name]
            self.rms_err_exts[stack_band_name] = 2
        elif err_from == "WHT":
            self.wht_paths[stack_band_name] = self.im_paths[stack_band_name]
            self.wht_exts[stack_band_name] = 2



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
    def make_sex_cats(self, forced_phot_band = "F444W", sex_config_path = config['SExtractor']['CONFIG_PATH'], params_path = config['SExtractor']['PARAMS_PATH'], forced_phot_code = "photutils"):
        galfind_logger.info(f"Making SExtractor catalogues with: config file = {sex_config_path}; parameters file = {params_path}")
        #breakpoint()
        # make individual forced photometry catalogues
        if type(forced_phot_band) == list:
            if len(forced_phot_band) > 1:
                # make the stacked image and save all appropriate parameters
                galfind_logger.debug(f"forced_phot_band = {forced_phot_band}")
                self.stack_bands(forced_phot_band)
                self.forced_phot_band = self.combine_band_names(forced_phot_band)
                self.seg_paths[self.forced_phot_band] = self.seg_path(self.forced_phot_band)
                overwrite = config["DEFAULT"].getboolean("OVERWRITE")
                if overwrite:
                    galfind_logger.info("OVERWRITE = YES, so overwriting segmentation map if it exists.")
            
                if not Path(self.seg_paths[self.forced_phot_band]).is_file() or overwrite:
                    if not config["DEFAULT"].getboolean("RUN"):
                        galfind_logger.critical("RUN = YES, so running through sextractor. Returning Error.")
                        raise Exception(f"RUN = YES, and combination of {self.survey} {self.version} or {self.instrument.name} has not previously been run through sextractor.")
                    self.make_seg_map(forced_phot_band)
                    
            else:
                self.forced_phot_band = forced_phot_band[0]
        else:
            self.forced_phot_band = forced_phot_band
        
        if self.forced_phot_band not in self.instrument.band_names:
            sextractor_bands = np.append(self.instrument.band_names, self.forced_phot_band)
        else:
            sextractor_bands = self.instrument.band_names
        
        if not hasattr(self, "sex_cats"):
            self.sex_cats = {}
        if not hasattr(self, "sex_cat_types"):
            self.sex_cat_types = {}

        for band in sextractor_bands:
            sex_cat_path = self.sex_cat_path(band, self.forced_phot_band)
            self.sex_cats[band] = sex_cat_path
            galfind_logger.debug(f"band = {band}, sex_cat_path = {sex_cat_path} in Data.make_sex_cats")
            
            # check whether the image of the forced photometry band and sextraction band have the same shape
            if self.im_shapes[self.forced_phot_band] == self.im_shapes[band]:
                sextract = True
            else:
                sextract = False
            # check whether the band and forced phot band have error maps with consistent types
            if sextract:
                if band in self.rms_err_paths.keys() and self.forced_phot_band in self.rms_err_paths.keys():
                    prefer = "rms_err"
                elif band in self.wht_paths.keys() and self.forced_phot_band in self.wht_paths.keys():
                    prefer = "wht"
                else: # do not perform sextraction
                    sextract = False
            if sextract:
                self.sex_cat_types[band] = subprocess.check_output("sex --version", shell = True).decode("utf-8").replace("\n", "")
            else:
                self.sex_cat_types[band] = f"{forced_phot_code} v{globals()[forced_phot_code].__version__}"

            # overwrite = config["DEFAULT"].getboolean("OVERWRITE")
            # if overwrite:
            #         galfind_logger.info("OVERWRITE = YES, so overwriting sextractor output if it exists.")
            # if not run before
            if not Path(sex_cat_path).is_file(): # or overwrite
                # if not config["DEFAULT"].getboolean("RUN"):
                #     galfind_logger.critical("RUN = YES, so not running sextractor. Returning Error.")
                #     raise Exception(f"RUN = YES, and combination of {self.survey} {self.version} or {self.instrument.name} has not previously been run through sextractor.")
                
                # perform sextraction
                if sextract:
                    # load relevant err map paths
                    err_map_path, err_map_ext, err_map_type = self.get_err_map(band, prefer = prefer)
                    # load relevant err map paths for the forced photometry band
                    forced_phot_band_err_map_path, forced_phot_band_err_map_ext, forced_phot_band_err_map_type = self.get_err_map(self.forced_phot_band, prefer = prefer)
                    assert(err_map_type == forced_phot_band_err_map_type) # should always be true
                
                    # insert specified aperture diameters from config file
                    as_aper_diams = json.loads(config.get("SExtractor", "APERTURE_DIAMS"))
                    pix_aper_diams = str([np.round(pix_aper_diam, 2) for pix_aper_diam in as_aper_diams / self.im_pixel_scales[band].value]).replace("[", "").replace("]", "").replace(" ", "")
                    # SExtractor bash script python wrapper
                    process = subprocess.Popen(["./make_sex_cat.sh", config['DEFAULT']['GALFIND_WORK'], self.im_paths[band], str(self.im_pixel_scales[band].value), \
                        str(self.im_zps[band]), self.instrument.instrument_from_band(band).name, self.survey, band, self.version, \
                        self.forced_phot_band, self.im_paths[self.forced_phot_band], err_map_path, err_map_ext, \
                        str(self.im_exts[band]), forced_phot_band_err_map_path, str(self.im_exts[self.forced_phot_band]), err_map_type, 
                        forced_phot_band_err_map_ext, sex_config_path, params_path, pix_aper_diams])
                    process.wait()

                else: # use photutils
                    #breakpoint()
                    self.forced_photometry(band, self.forced_phot_band, forced_phot_code = forced_phot_code)
            
            galfind_logger.info(f"Finished making SExtractor catalogue for {self.survey} {self.version} {band}!")
    
    def combine_sex_cats(self, forced_phot_band = "F444W", readme_sep = "-" * 20):
        self.make_sex_cats(forced_phot_band)

        save_name = f"{self.survey}_MASTER_Sel-{self.combine_band_names(forced_phot_band)}_{self.version}.fits"
        save_dir = f"{config['DEFAULT']['GALFIND_WORK']}/Catalogues/{self.version}/{self.instrument.name}/{self.survey}"
        self.sex_cat_master_path = f"{save_dir}/{save_name}"
        overwrite = config["DEFAULT"].getboolean("OVERWRITE")
        if overwrite:
            galfind_logger.info("OVERWRITE = YES, so overwriting combined catalogue if it exists.")
         
        if not Path(self.sex_cat_master_path).is_file() or overwrite:
            if not config["DEFAULT"].getboolean("RUN"):
                galfind_logger.critical("RUN = YES, so not combining catalogues. Returning Error.")
                raise Exception(f"RUN = YES, and combination of {self.survey} {self.version} or {self.instrument.name} has not previously been combined into a catalogue.")

            if type(forced_phot_band) == np.array or type(forced_phot_band) == list:
                forced_phot_band_name = self.combine_band_names(forced_phot_band)
            else:
                forced_phot_band_name = forced_phot_band
            print("Loading cat", self.sex_cats, forced_phot_band_name)
            for i, (band, path) in enumerate(self.sex_cats.items()):
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
            master_tab.meta = {**master_tab.meta, **{"INSTR": self.instrument.name, \
                "SURVEY": self.survey, "VERSION": self.version, "BANDS": str(self.instrument.band_names)}}

            # create galfind catalogue README
            # sex_aper_diams = json.loads(config.get("SExtractor", "APERTURE_DIAMS")) * u.arcsec
            # text = f"""
            #     NUMBER: Galaxy ID
            #     X/Y_IMAGE: X/Y image co-ordinates in 
            #     ALPHA/DELTA_J2000: RA/Dec (J2000 co-ordinates)
            #     FLUX(ERR)_APER_'band': Aperture flux/flux errors in {str(sex_aper_diams.to(u.arcsec).value) + 'as'} diameter apertures, image units with ZPs as explained below
            #     MAG_APER_'band': Aperture magnitudes in {str(sex_aper_diams.to(u.arcsec).value) + 'as'} diameter apertures, AB mag units, defaults to 99. if flux < 0.
            #     MAGERR_APER_'band': Aperture magnitude errors in {str(sex_aper_diams.to(u.arcsec).value) + 'as'} diameter apertures, AB mag units, negative if mag == 99.
            # """
            # if 'sextractor' in [cat_type.lower() for cat_type in self.sex_cat_types.values()]:
            #     text += f"See SExtractor documentation () for descriptions of other columns. These are only available for {'+'.join([band_name for band_name, sex_cat_type in self.sex_cat_types.items() if 'sextractor' in sex_cat_type.lower()])}\n"
            # text += readme_sep + "\n"
            # self.make_sex_readme({"Photometry": text}, self.sex_cat_master_path.replace(".fits", "_README.txt"))

            # save table
            os.makedirs(save_dir, exist_ok = True)
            master_tab.write(self.sex_cat_master_path, format = "fits", overwrite = True)
            galfind_logger.info(f"Saved combined SExtractor catalogue as {self.sex_cat_master_path}")

    def produce_readme(self, ):
        pass

    def make_readme(self, col_desc_dict, save_path, overwrite = False, readme_sep = "-" * 20):
        assert type(col_desc_dict) == dict
        assert "Photometry" in col_desc_dict.keys()
        intro_text = f"""
        
        """
        # if not overwrite and README already exists, extract previous column labels to append col_desc_dict to
        f = open(save_path, "w")
        f.write(intro_text)
        f.write(readme_sep + "\n\n")
        f.write(str(self) + "\n")
        for key, value in col_desc_dict.items():
            if key == "Photometry":
                init_phot_text = f"Photometry:\n" + '\n'.join([phot_code + '= ' + '+'.join([band_name for band_name, sex_cat_type \
                    in self.sex_cat_types.items() if sex_cat_type == phot_code]) for phot_code in np.unique(self.sex_cat_types.values())]) + "\n"
                f.write(init_phot_text)
            else:
                f.write(key + "\n")
            f.write(readme_sep + "\n")
            f.write(value)
            f.write(readme_sep + "\n")
        f.close()

    def make_sex_plusplus_cat(self):
        pass
    
    def forced_photometry(self, band, forced_phot_band, radii = list(np.array(json.loads(config["SExtractor"]["APERTURE_DIAMS"])) / 2.) * u.arcsec, \
            ra_col = 'ALPHA_J2000', dec_col = 'DELTA_J2000', coord_unit = u.deg, id_col = 'NUMBER', x_col = 'X_IMAGE', y_col = 'Y_IMAGE', forced_phot_code = "photutils"):
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
        out_path = f"{'/'.join(mask_path.split('/')[:-1])}/fits_masks/{mask_path.split('/')[-1].replace('_clean', '').replace('.reg', '')}.fits"
        if not Path(out_path).is_file():
            # open image corresponding to band
            im_data, im_header, seg_data, seg_header = self.load_data(band, incl_mask = False)
            # open .reg mask file
            mask_regions = pyregion.open(mask_path).as_imagecoord(im_header)
            # make 2D np.array boolean pixel mask
            pix_mask = np.array(mask_regions.get_mask(header = im_header, shape = im_data.shape), dtype = bool)
            # make .fits mask
            mask_hdu = fits.ImageHDU(pix_mask.astype(np.uint8), header = WCS(im_header).to_header(), name = 'MASK')
            hdu = fits.HDUList([fits.PrimaryHDU(), mask_hdu])
            os.makedirs("/".join(out_path.split("/")[:-1]), exist_ok = True)
            hdu.writeto(out_path, overwrite = True)
            galfind_logger.info(f"Created fits mask from manually created reg mask, saving as {out_path}")
        return out_path
    
    def combine_masks(self, bands):
        out_path = f"{self.mask_dir}/combined_masks/{self.survey}_{self.combine_band_names(bands)}.fits"
        if not Path(out_path).is_file():
            # require pixel scales to be the same across all bands
            for i, band in enumerate(bands):
                if i == 0:
                    pix_scale = self.im_pixel_scales[band]
                else:
                    assert(self.im_pixel_scales[band] == pix_scale)
            combined_mask = np.logical_or.reduce(tuple([self.load_mask(band) for band in bands]))
            assert(combined_mask.shape == self.load_mask(bands[-1]).shape)
            # wcs taken from the reddest band
            mask_hdu = fits.ImageHDU(combined_mask.astype(np.uint8), header = WCS(self.load_im(bands[-1])[1]).to_header(), name = 'MASK')
            hdu = fits.HDUList([fits.PrimaryHDU(), mask_hdu])
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
    def calc_unmasked_area(self, masking_instrument_or_band_name = "NIRCam", forced_phot_band = ["F277W", "F356W", "F444W"]):
        
        if "PIXEL SCALE" not in self.common.keys():
            galfind_logger.warning("Masking by bands with different pixel scales is not supported!")
        
        if type(masking_instrument_or_band_name) in [list, np.array]:
            masking_instrument_or_band_name = "+".join(list(masking_instrument_or_band_name))

        # create a list of bands that need to be unmasked in order to calculate the area
        if type(masking_instrument_or_band_name) in [str, np.str_]:
            masking_instrument_or_band_name = str(masking_instrument_or_band_name)
            # mask by requiring unmasked criteria in all bands for a given Instrument
            if masking_instrument_or_band_name in [subclass.__name__ for subclass in Instrument.__subclasses__() if subclass.__name__ != "Combined_Instrument"]:
                masking_bands = np.array([band for band in self.instrument.band_names if band in Instrument.from_name(masking_instrument_or_band_name).bands])
            elif masking_instrument_or_band_name == "All":
                masking_bands = np.array(self.instrument.band_names)
            else: # string should contain individual bands, separated by a "+"
                masking_bands = masking_instrument_or_band_name.split("+")
                for band in masking_bands:
                    assert band in json.loads(config.get("Other", "ALL_BANDS")), galfind_logger.critical(f"{band} is not a valid band currently included in galfind! Cannot calculate unmasked area!")
        else:
            galfind_logger.critical(f"type(masking_instrument_or_band_name) = {type(masking_instrument_or_band_name)} must be in [str, list, np.array]!")
        
        # make combined mask if required, else load mask
        glob_mask_names = glob.glob(f"{self.mask_dir}/fits_masks/*{self.combine_band_names(masking_bands)}_*")
        if '+' not in masking_bands and len(glob_mask_names) > 1:
            for mask in glob_mask_names:
                if '+' in mask:
                    glob_mask_names.remove(mask)

        if len(glob_mask_names) == 0:
            if len(masking_bands) > 1:
                path = self.combine_masks(masking_bands)
                print(path)
                self.mask_paths[masking_instrument_or_band_name] = path
        elif len(glob_mask_names) == 1:
            self.mask_paths[masking_instrument_or_band_name] = glob_mask_names[0]
        else:
            raise(Exception(f"More than 1 mask for {masking_bands}. Please change this in {self.mask_dir}"))
        full_mask = self.load_mask(masking_instrument_or_band_name)
        
        if self.is_blank:
            blank_mask = full_mask
        else:
            # make combined mask for masking_instrument_name blank field area
            glob_mask_names = glob.glob(f"{self.mask_dir}/fits_masks/*{self.combine_band_names(list(masking_bands) + ['blank'])}_*")
            if len(glob_mask_names) == 0:
                self.mask_paths[f"{masking_instrument_or_band_name}+blank"] = self.combine_masks(list(masking_bands) + ["blank"])
            elif len(glob_mask_names) == 1:
                self.mask_paths[f"{masking_instrument_or_band_name}+blank"] = glob_mask_names[0]
            else:
                raise(Exception(f"More than 1 mask for {masking_bands}. Please change this in {self.mask_dir}"))
            blank_mask = self.load_mask(f"{masking_instrument_or_band_name}+blank")
        
        # split calculation by depth regions
        galfind_logger.warning("Area calculation for different depth regions not yet implemented!")

        # calculate areas using pixel scale of selection band
        pixel_scale = self.im_pixel_scales[self.combine_band_names(forced_phot_band)]
        unmasked_area_tot = (((full_mask.shape[0] * full_mask.shape[1]) - np.sum(full_mask)) * pixel_scale * pixel_scale).to(u.arcmin ** 2)
        unmasked_area_blank_modules = (((blank_mask.shape[0] * blank_mask.shape[1]) - np.sum(blank_mask)) * pixel_scale * pixel_scale).to(u.arcmin ** 2)
        unmasked_area_cluster_module = unmasked_area_tot - unmasked_area_blank_modules
        galfind_logger.info(f"Unmasked areas for {self.survey}, masking_instrument_or_band_name = {masking_instrument_or_band_name} - Total: {unmasked_area_tot}, Blank modules: {unmasked_area_blank_modules}, Cluster module: {unmasked_area_cluster_module}")
        
        output_path = f"{config['DEFAULT']['GALFIND_WORK']}/Unmasked_areas.ecsv"
        funcs.make_dirs(output_path)
        areas_data = {"survey": [self.survey], "masking_instrument_band": [masking_instrument_or_band_name], \
            "unmasked_area_total": [np.round(unmasked_area_tot, 3)], "unmasked_area_blank_modules": [np.round(unmasked_area_blank_modules, 3)], \
            "unmasked_area_cluster_module": [np.round(unmasked_area_cluster_module, 3)]}
        areas_tab = Table(areas_data)
        if Path(output_path).is_file():
            existing_areas_tab = Table.read(output_path)
            # if the exact same setup has already been run, overwrite
            existing_areas_tab_ = deepcopy(existing_areas_tab)
            existing_areas_tab["index"] = np.arange(0, len(existing_areas_tab), 1)

            existing_areas_tab_ = existing_areas_tab[((existing_areas_tab["survey"] == self.survey) & \
                (existing_areas_tab["masking_instrument_band"] == masking_instrument_or_band_name))]
            if len(existing_areas_tab_) > 0:
                # delete existing column using the same setup in favour of new one
                existing_areas_tab.remove_row(int(existing_areas_tab_["index"]))
            else:
                areas_tab = vstack([existing_areas_tab, areas_tab])
            for col in areas_tab.colnames:
                if 'index' in col:
                    areas_tab.remove_column(col)

        areas_tab.write(output_path, overwrite = True)
        return areas_tab

    def perform_aper_corrs(self): # not general
        overwrite = config["Depths"].getboolean("OVERWRITE_LOC_DEPTH_CAT")
        if overwrite:
            galfind_logger.info("OVERWRITE_LOC_DEPTH_CAT = YES, updating catalogue with aperture corrections.")
        cat = Table.read(self.sex_cat_master_path)
        if not "APERCORR" in cat.meta.keys() or overwrite:
            for i, band in enumerate(self.instrument.band_names):
                print(band)
                mag_aper_corr_data = np.zeros(len(cat))
                flux_aper_corr_data = np.zeros(len(cat))
                for j, aper_diam in enumerate(json.loads(config.get("SExtractor", "APERTURE_DIAMS")) * u.arcsec):
                    # assumes these have already been calculated for each band 
                    mag_aper_corr_factor = self.instrument.get_aper_corrs(aper_diam)[i]
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
                cat[f"FLUX_APER_{band}_aper_corr_Jy"] = [tuple([funcs.flux_image_to_Jy(val, self.im_zps[band]).value for val in element]) for element in cat[f"FLUX_APER_{band}_aper_corr"]]
            # update catalogue metadata
            #mag_aper_corrs = {f"HIERARCH Mag_aper_corrs_{aper_diam.value}as": tuple([np.round(self.instrument.aper_corr(aper_diam, band), decimals = 4) \
            #    for band in self.instrument.band_names]) for aper_diam in json.loads(config.get("SExtractor", "APERTURE_DIAMS")) * u.arcsec}
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
            for i, band in enumerate(self.instrument.band_names):
                galfind_logger.info(f"Finished making local depth columns for {band=}")
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
                # impose n_pc min flux error and convert to Jy where appropriate
                if "APERCORR" in cat.meta.keys():
                    cat[f"FLUXERR_APER_{band}_loc_depth_{str(int(cat_creator.min_flux_pc_err))}pc_Jy"] = \
                        [tuple([funcs.flux_image_to_Jy(flux, self.im_zps[band]).value * cat_creator.min_flux_pc_err / 100. \
                        if err / flux < cat_creator.min_flux_pc_err / 100. and flux > 0. \
                        else funcs.flux_image_to_Jy(err, self.im_zps[band]).value for flux, err in zip(flux_tup, err_tup)]) \
                        for flux_tup, err_tup in zip(cat[f"FLUX_APER_{band}_aper_corr"], cat[f"FLUXERR_APER_{band}_loc_depth"])]
                else:
                    raise(Exception(f"Couldn't make 'FLUXERR_APER_{band}_loc_depth_{str(int(cat_creator.min_flux_pc_err))}Jy' columns!"))
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
        for band in self.im_paths.keys():
            self.depth_dirs[band] = f"{config['Depths']['DEPTH_DIR']}/{self.instrument.instrument_from_band(band).name}/{self.version}/{self.survey}/{format(aper_diam.value, '.2f')}as"
            os.makedirs(self.depth_dirs[band], exist_ok = True)
        return self.depth_dirs
            
    def load_depths(self, aper_diam):
        self.get_depth_dir(aper_diam)
        self.depths = {}
        for band in self.instrument.band_names:
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
            for band in self.im_paths.keys():
                # Only run for non excluded bands
                if band not in excl_bands:
                    params.append((band, aper_diam, self.depth_dirs[band], mode, scatter_size, distance_to_mask, region_radius_used_pix, n_nearest, \
                    coord_type, split_depth_min_size, split_depths_factor, step_size, cat_creator))
        # Parallelise the calculation of depths for each band
        with tqdm_joblib(tqdm(desc = "Calculating depths", total = len(params))) as progress_bar:
            Parallel(n_jobs = n_jobs)(delayed(self.calc_band_depth)(param) for param in params)
        #self.plot_area_depth(cat_creator, mode, aper_diam, show = False)
    
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
                convert = False, wcs = wcs, pixel_scale = self.im_pixel_scales[band].value)
            
            # Get fluxes in regions
            fluxes = Depths.do_photometry(im_data, xy, radius_pix)
            depths, diagnostic, depth_labels, final_labels = Depths.calc_depths_numba(xy, fluxes, im_data, combined_mask, 
                    region_radius_used_pix = region_radius_used_pix, step_size = step_size, catalogue = cat, wcs = wcs, \
                    coord_type = coord_type, mode = mode, n_nearest = n_nearest, zero_point = self.im_zps[band], n_split = n_split, \
                    split_depth_min_size = split_depth_min_size, split_depths_factor = split_depths_factor, wht_data = wht_data)

            # calculate the depths for plotting purposes
            nmad_grid, num_grid, labels_grid, final_labels = Depths.calc_depths_numba(xy, fluxes, im_data, combined_mask, 
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

            try:
                self.plot_depth(band, cat_creator, mode, aper_diam, show = False)
            except:
                pass
    

    def plot_area_depth(self, cat_creator, mode, aper_diam, show = False, use_area_per_band=True):     
        if type(cat_creator) == type(None):
            galfind_logger.warning("Could not plot depths as cat_creator == None in Data.plot_area_depth()")
        else:
            self.get_depth_dir(aper_diam)
            area_tab = self.calc_unmasked_area(masking_instrument_or_band_name = self.forced_phot_band, forced_phot_band = self.forced_phot_band)
            overwrite = config["Depths"].getboolean("OVERWRITE_DEPTH_PLOTS")
            save_path = f"{self.depth_dirs[self.forced_phot_band]}/{mode}/depth_areas.png" # not entirely general -> need to improve self.depth_dirs
            
            if not Path(save_path).is_file() or overwrite:
                fig, ax = plt.subplots(1, 1, figsize = (5, 5))
                ax.set_title(f"{self.survey} {self.version} {aper_diam}")
                ax.set_xlabel("Area (arcmin$^{2}$)")
                ax.set_ylabel("5$\sigma$ Depth (AB mag)")
                area_row = area_tab[area_tab["masking_instrument_band"] == self.forced_phot_band]
                area_master = float(area_row["unmasked_area_total"].to(u.arcmin ** 2).value)
                bands = list(self.instrument.band_names)
                if self.forced_phot_band not in bands:
                    bands.append(self.forced_phot_band)
                colors = plt.cm.viridis(np.linspace(0, 1, len(bands)))

                for pos, band in enumerate(bands):
                    h5_path = f"{self.depth_dirs[band]}/{mode}/{band}.h5"

                    if overwrite:
                        galfind_logger.info("OVERWRITE_DEPTH_PLOTS = YES, re-doing depth plots.")

                    if not Path(h5_path).is_file():
                        raise(Exception(f"Must first run depths for {self.survey} {self.version} {band} {mode} {aper_diam} before plotting!"))
                    hf = h5py.File(h5_path, "r")
                    hf_output = {label: np.array(hf[label]) for label in self.get_depth_h5_labels()}
                    hf.close()
                    # Need unmasked area for each band
                    if use_area_per_band:
                        area_tab = self.calc_unmasked_area(masking_instrument_or_band_name = band, forced_phot_band = self.forced_phot_band)
                        area_row = area_tab[area_tab["masking_instrument_band"] == band]

                    area = area_row["unmasked_area_total"].to(u.arcmin ** 2).value

                    total_depths = hf_output["nmad_grid"].flatten()
                    total_depths = total_depths[~np.isnan(total_depths)]
                    total_depths = total_depths[total_depths != 0]
                    total_depths = total_depths[total_depths != np.inf]

                    # Round to 0.01 mag and sort
                    #total_depths = np.round(total_depths, 2)
                    total_depths = np.flip(np.sort(total_depths))

                    # Calculate the cumulative distribution scaled to area of band
                    n = len(total_depths)
                    cum_dist = np.arange(1, n + 1) / n
                    cum_dist = cum_dist * area

                    # Plot
                    ax.plot(cum_dist, total_depths, label = band if '+' not in band else 'Detection', color = colors[pos], drawstyle='steps-post')

                    # Set ylim to 2nd / 98th percentile if depth is smaller than this number
                    ylim = ax.get_ylim()
                    
                    if pos == 0:
                        min_depth = np.percentile(total_depths, 0.5)
                        max_depth = np.percentile(total_depths, 99.5)
                    else:
                        min_temp = np.percentile(total_depths, 0.5)
                        max_temp = np.percentile(total_depths, 99.5)
                        if min_temp < min_depth:
                            min_depth = min_temp
                        if max_temp > max_depth:
                            max_depth = max_temp

                ax.set_ylim(max_depth, min_depth)
                ax.legend(frameon = False, ncol = 2)
                ax.set_xlim(0, area_master)
                # Add hlines at integer depths
                depths = np.arange(20, 30, 1)
                #for depth in depths:
                #    ax.hlines(depth, 0, area_master, color = "black", linestyle = "dotted", alpha = 0.5)
                # Invert y axis
                #ax.invert_yaxis()
                ax.grid(True)

                fig.savefig(save_path, dpi = 300, bbox_inches = "tight")
                if show:
                    plt.show()


    def plot_depth(self, band, cat_creator, mode, aper_diam, show = False): #, **kwargs):
        if type(cat_creator) == type(None):
            galfind_logger.warning("Could not plot depths as cat_creator == None in Data.plot_depth()")
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
                
                Depths.show_depths(hf_output["nmad_grid"], hf_output["num_grid"], hf_output["step_size"], \
                    hf_output["region_radius_used_pix"], hf_output["labels_grid"], hf_output["depth_labels"], hf_output["depths"], hf_output["diagnostic"], cat_x, cat_y, 
                    combined_mask, hf_output["final_labels"], suptitle = f"{self.survey} {self.version} {band} Depths", save_path = save_path, show = show)

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


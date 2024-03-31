#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:11:23 2023

@author: austind
"""

# Galaxy.py
import numpy as np
from copy import copy, deepcopy
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import os
from pathlib import Path
from astropy.nddata import Cutout2D
from tqdm import tqdm
from astropy.wcs import WCS

from . import useful_funcs_austind as funcs
from . import Photometry_rest, Photometry_obs, Multiple_Photometry_obs, config, Data

class Galaxy:
    
    def __init__(self, sky_coord, ID, phot, mask_flags = {}):
        self.sky_coord = sky_coord
        self.ID = int(ID)
        self.phot = phot
        self.mask_flags = mask_flags
        
    @classmethod
    def from_fits_cat(cls, fits_cat_row, instrument, cat_creator, codes, lowz_zmax, templates_arr):
        # load multiple photometries from the fits catalogue
        phot = Photometry_obs.from_fits_cat(fits_cat_row, instrument, cat_creator, cat_creator.aper_diam, cat_creator.min_flux_pc_err, codes, lowz_zmax, templates_arr) # \
                # for min_flux_pc_err in cat_creator.min_flux_pc_err for aper_diam in cat_creator.aper_diam]
        # load the ID and Sky Coordinate from the source catalogue
        ID = int(fits_cat_row[cat_creator.ID_label])
        sky_coord = SkyCoord(fits_cat_row[cat_creator.ra_dec_labels["RA"]] * u.deg, fits_cat_row[cat_creator.ra_dec_labels["DEC"]] * u.deg, frame = "icrs")
        # mask flags should come from cat_creator
        mask_flags = {f"unmasked_{band}": cat_creator.load_flag(fits_cat_row, f"unmasked_{band}") for band in instrument.bands}
        return cls(sky_coord, ID, phot, mask_flags)
    
    def __str__(self):
        line_sep = "*" * 40 + "\n"
        band_sep = "-" * 10 + "\n"
        output_str = line_sep
        output_str += f"GALAXY {self.ID}: (RA, DEC) = ({np.round(self.sky_coord.ra, 5)}, {np.round(self.sky_coord.dec, 5)})\n"
        output_str += band_sep
        output_str += f"MASK FLAGS: {self.mask_flags}\n"
        output_str += str(self.phot)
        output_str += line_sep
        return output_str
        
    def __setattr__(self, name, value, obj = "gal"):
        if obj == "gal":
            if type(name) != list and type(name) != np.array:
                super().__setattr__(name, value)
            else:
                # use setattr to set values within Galaxy dicts (e.g. properties)
                self.globals()[name[0]][name[1]] = value
        else:
            raise(Exception(f"obj = {obj} must be 'gal'!"))
    
    # STILL NEED TO LOOK FURTHER INTO THIS
    # def __deepcopy__(self, memo):
    #     print("Overriding Galaxy.__deepcopy__()")
    #     cls = self.__class__
    #     result = cls.__new__(cls)
    #     memo[id(self)] = result
    #     for key, value in self.__dict__.items():
    #         setattr(result, key, deepcopy(value, memo))
    #     return result
    
    def update(self, gal_SED_results, index = 0): # for now just update the single photometry
        self.phot.update(gal_SED_results)

    def make_cutout(self, band, data, wcs = None, im_header = None, survey = None, version = None, cutout_size = 32):
        
        if type(data) == Data:
            survey = data.survey
            version = data.version
        if survey == None or version == None:
            raise(Exception("'survey' and 'version' must both be given to construct save paths"))
        
        out_path = f"{config['Cutouts']['CUTOUT_DIR']}/{version}/{survey}/{band}/{self.ID}.fits"
        if not Path(out_path).is_file() or config.getboolean('Cutouts', 'OVERWRITE_CUTOUTS'):
            if type(data) == Data:
                im_data, im_header, seg_data, seg_header = data.load_data(band, incl_mask = False)
                wht_data = data.load_wht(band)
                data = {"SCI": im_data, "SEG": seg_data, data.wht_types[band]: wht_data}
                wcs = WCS(im_header)
            elif type(data) == dict and type(wcs) != type(None) and type(im_header) != type(None):
                pass
            else:
                raise(Exception(""))
            hdul = [fits.PrimaryHDU(header = fits.Header({"ID": self.ID, "survey": survey, "version": version, \
                        "RA": self.sky_coord.ra.value, "DEC": self.sky_coord.dec.value, "size": cutout_size}))]
            for i, (label_i, data_i) in enumerate(data.items()):
                cutout = Cutout2D(data_i, self.sky_coord, size = (cutout_size, cutout_size), wcs = wcs)
                im_header.update(cutout.wcs.to_header())
                hdul.append(fits.ImageHDU(cutout.data, header = im_header, name = label_i))
            #print(hdul)
            os.makedirs("/".join(out_path.split("/")[:-1]), exist_ok = True)
            fits_hdul = fits.HDUList(hdul)
            fits_hdul.writeto(out_path, overwrite = True)
            print('Saved fits cutout to:', out_path)
        else:
            print(f"Already made fits cutout for {survey} {version} {self.ID}")
        
    def update_mask_full(self, bool_values):
        pass
        
    def update_mask_band(self, band, bool_value):
        self.mask_flags[band] = bool_value
        
    def phot_SNR_crop(self, band, sigma_detect_thresh, flag = True):
        self.phot.SNR_crop(band, sigma_detect_thresh)
        pass
    
class Multiple_Galaxy:
    
    def __init__(self, sky_coords, IDs, phots, mask_flags_arr):
        self.gals = [Galaxy(sky_coord, ID, phot, mask_flags) for sky_coord, ID, phot, mask_flags in zip(sky_coords, IDs, phots, mask_flags_arr)]
        
    def __repr__(self):
        # string representation of what is stored in this class
        return str(self.__dict__)
    
    def __len__(self):
        return len(self.gals)
    
    def __iter__(self):
        self.iter = 0
        return self
    
    def __next__(self):
        if self.iter > len(self) - 1:
            raise StopIteration
        else:
            gal = self[self.iter]
            self.iter += 1
            return gal
    
    def __getitem__(self, index):
        return self.gals[index]
        
    @classmethod
    def from_fits_cat(cls, fits_cat, instrument, cat_creator, codes, lowz_zmax, templates_arr):
        # load photometries from catalogue
        phots = Multiple_Photometry_obs.from_fits_cat(fits_cat, instrument, cat_creator, cat_creator.aper_diam, cat_creator.min_flux_pc_err, codes, lowz_zmax, templates_arr).phot_obs_arr
        # load the ID and Sky Coordinate from the source catalogue
        IDs = np.array(fits_cat[cat_creator.ID_label]).astype(int)
        # load sky co-ordinate one at a time (can improve efficiency here)
        sky_coords = [SkyCoord(ra * u.deg, dec * u.deg, frame = "icrs") \
            for ra, dec in zip(fits_cat[cat_creator.ra_dec_labels["RA"]], fits_cat[cat_creator.ra_dec_labels["DEC"]])]
        # mask flags should come from cat_creator
        #mask_flags_arr = [{f"unmasked_{band}": cat_creator.load_flag(fits_cat_row, f"unmasked_{band}") for band in instrument.bands} for fits_cat_row in fits_cat]
        mask_flags_arr = [{f"unmasked_{band}": None for band in instrument.bands} for fits_cat_row in fits_cat]
        return cls(sky_coords, IDs, phots, mask_flags_arr)
    
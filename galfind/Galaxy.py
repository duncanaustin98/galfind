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
from . import config, galfind_logger
from . import Photometry_rest, Photometry_obs, Multiple_Photometry_obs, Data

class Galaxy:
    
    def __init__(self, sky_coord, ID, phot, mask_flags = {}, selection_flags = {}):
        self.sky_coord = sky_coord
        self.ID = int(ID)
        self.phot = phot
        self.mask_flags = mask_flags
        self.selection_flags = selection_flags
        
    @classmethod
    def from_fits_cat(cls, fits_cat_row, instrument, cat_creator, codes, lowz_zmax, templates_arr):
        # load multiple photometries from the fits catalogue
        phot = Photometry_obs.from_fits_cat(fits_cat_row, instrument, cat_creator, cat_creator.aper_diam, cat_creator.min_flux_pc_err, codes, lowz_zmax, templates_arr) # \
                # for min_flux_pc_err in cat_creator.min_flux_pc_err for aper_diam in cat_creator.aper_diam]
        # load the ID and Sky Coordinate from the source catalogue
        ID = int(fits_cat_row[cat_creator.ID_label])
        sky_coord = SkyCoord(fits_cat_row[cat_creator.ra_dec_labels["RA"]] * u.deg, fits_cat_row[cat_creator.ra_dec_labels["DEC"]] * u.deg, frame = "icrs")
        # mask flags should come from cat_creator
        mask_flags = {f"unmasked_{band}": cat_creator.load_flag(fits_cat_row, f"unmasked_{band}") for band in instrument.band_names}
        return cls(sky_coord, ID, phot, mask_flags)
    
    def __str__(self):
        line_sep = "*" * 40 + "\n"
        band_sep = "-" * 10 + "\n"
        output_str = line_sep
        output_str += f"GALAXY {self.ID}: (RA, DEC) = ({np.round(self.sky_coord.ra, 5)}, {np.round(self.sky_coord.dec, 5)})\n"
        output_str += band_sep
        output_str += f"MASK FLAGS: {self.mask_flags}\n"
        output_str += f"SELECTION FLAGS: {self.selection_flags}\n"
        output_str += str(self.phot)
        output_str += line_sep
        return output_str
        
    # def __setattr__(self, name, value, obj = "gal"):
    #     if obj == "gal":
    #         if type(name) != list and type(name) != np.array:
    #             super().__setattr__(name, value)
    #         else:
    #             # use setattr to set values within Galaxy dicts (e.g. properties)
    #             self.globals()[name[0]][name[1]] = value
    #     else:
    #         raise(Exception(f"obj = {obj} must be 'gal'!"))
    
    # STILL NEED TO LOOK FURTHER INTO THIS
    def __deepcopy__(self, memo):
        #print("Overriding Galaxy.__deepcopy__()")
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, deepcopy(value, memo))
        return result
    
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
        
    # %% Selection methods
        
    def phot_bluewards_Lya_non_detect(self, SNR_lim, code_name = "EAZY", templates = "fsps_larson", lowz_zmax = None, update_obj = True):
        assert(type(SNR_lim) in [int, float])
        selection_name = f"bluewards_Lya_SNR<{SNR_lim:.1f}"
        # only compute this if not already done so
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            # extract bands, SNRs, mask and first Lya non-detect band
            bands = self.phot.instrument.band_names
            SNRs = self.phot.SNR
            mask = self.phot.flux_Jy.mask
            assert(len(bands) == len(SNRs) == len(mask))
            first_Lya_non_detect_band = self.phot.SED_results[code_name][templates][funcs.lowz_label(lowz_zmax)].phot_rest.first_Lya_non_detect_band
            if first_Lya_non_detect_band == None:
                if update_obj:
                    self.selection_flags[selection_name] = True
                return True, selection_name
            # find index of first Lya non-detect band
            first_Lya_non_detect_index = np.where(bands == first_Lya_non_detect_band)[0][0]
            SNR_non_detect = SNRs[:first_Lya_non_detect_index + 1]
            mask_non_detect = mask[:first_Lya_non_detect_index + 1]
            # require the first Lya non detect band and all bluewards bands to be non-detected at < SNR_lim if not masked
            if all(SNR < SNR_lim or mask for mask, SNR in zip(mask_non_detect, SNR_non_detect)):
                if update_obj:
                    self.selection_flags[selection_name] = True
                return True, selection_name
            else:
                if update_obj:
                    self.selection_flags[selection_name] = False
                return False, selection_name
        return self.selection_flags[selection_name], selection_name
    
    def phot_redwards_Lya_detect(self, SNR_lims, code_name = "EAZY", templates = "fsps_larson", \
            lowz_zmax = None, widebands_only = True, update = True):
        # work out selection name based on SNR_lims input type
        if type(SNR_lims) in [int, float]:
            # require all redwards bands to be detected at >SNR_lims
            selection_name = f"ALL_redwards_Lya_SNR>{SNR_lims:.1f}"
            SNR_lims = np.full(len(self.phot.instrument.band_names), SNR_lims)
        elif type(SNR_lims) in [list, np.array]:
            # require the n^th band after the first band redwards of Lya to be detected at >SNR_lims[n]
            assert(np.all([type(SNR) in [int, float] for SNR in SNR_lims]))
            selection_name = f"redwards_Lya_SNR>{','.join([str(np.round(SNR, 1)) for SNR in SNR_lims])}"
        else:
            galfind_logger.critical(f"SNR_lims = {SNR_lims} has type = {type(SNR_lims)} which is not in [int, float, list, np.array]")

        # only compute this if not already done so
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"Already performed {selection_name} for galaxy ID = {self.ID}, skipping!")
        else:
            galfind_logger.debug(f"Performing {selection_name} for galaxy ID = {self.ID}!")
            # extract bands, SNRs, mask and first Lya non-detect band
            bands = self.phot.instrument.band_names
            SNRs = self.phot.SNR
            mask = self.phot.flux_Jy.mask
            first_Lya_detect_band = self.phot.SED_results[code_name][templates][funcs.lowz_label(lowz_zmax)].phot_rest.first_Lya_detect_band
            if first_Lya_detect_band == None:
                #if update_cat:
                self.selection_flags[selection_name] = False
                return self, selection_name
            # find index of first Lya non-detect band
            first_Lya_detect_index = np.where(bands == first_Lya_detect_band)[0][0] # only 1 band is the first
            bands_detect = np.array(bands[first_Lya_detect_index:])
            SNR_detect = np.array(SNRs[first_Lya_detect_index:])
            mask_detect = np.array(mask[first_Lya_detect_index:])
            # option as to whether to exclude potentially shallower medium/narrow bands in this calculation
            if widebands_only:
                #breakpoint()
                wide_band_detect_indices = [True if "W" in band.upper() or "LP" in band.upper() else False for band in bands_detect]
                bands_detect = bands_detect[wide_band_detect_indices]
                SNR_detect = SNR_detect[wide_band_detect_indices]
                mask_detect = mask_detect[wide_band_detect_indices]
                selection_name += "_widebands"
            # selection criteria
            if all(SNR > SNR_lim or mask for mask, SNR, SNR_lim in zip(mask_detect, SNR_detect, SNR_lims)):
                #if update_cat:
                self.selection_flags[selection_name] = True
                #return True, selection_name
            else:
                #if update_cat:
                self.selection_flags[selection_name] = False
                #return False, selection_name
        return self, selection_name # self.selection_flags[selection_name]

    def phot_SNR_crop(self, band_name, SNR_lim):
        pass
    
class Multiple_Galaxy:
    
    def __init__(self, sky_coords, IDs, phots, mask_flags_arr, selection_flags_arr):
        self.gals = [Galaxy(sky_coord, ID, phot, mask_flags, selection_flags) for \
            sky_coord, ID, phot, mask_flags, selection_flags in zip(sky_coords, IDs, phots, mask_flags_arr, selection_flags_arr)]
        
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
        #mask_flags_arr = [{f"unmasked_{band}": cat_creator.load_flag(fits_cat_row, f"unmasked_{band}") for band in instrument.band_names} for fits_cat_row in fits_cat]
        mask_flags_arr = [{f"unmasked_{band}": None for band in instrument.band_names} for fits_cat_row in fits_cat]
        selection_flags_arr = [{selection_flag: bool(fits_cat_row[selection_flag]) for selection_flag in cat_creator.selection_labels(fits_cat)} for fits_cat_row in fits_cat]
        return cls(sky_coords, IDs, phots, mask_flags_arr, selection_flags_arr)
    
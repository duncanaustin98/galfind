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
import json
from pathlib import Path
from astropy.nddata import Cutout2D
from tqdm import tqdm
from astropy.wcs import WCS

from . import useful_funcs_austind as funcs
from . import config, galfind_logger
from . import Photometry_rest, Photometry_obs, Multiple_Photometry_obs, Data, Instrument, NIRCam, ACS_WFC, WFC3_IR

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
        mask_flags = {} #{f"unmasked_{band}": cat_creator.load_flag(fits_cat_row, f"unmasked_{band}") for band in instrument.band_names}
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
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, deepcopy(value, memo))
        return result
    
    def update(self, gal_SED_results, index = 0): # for now just update the single photometry
        self.phot.update(gal_SED_results)

    def update_mask(self, cat, catalogue_creator, update_phot_rest = False):
        self.phot.update_mask(cat, catalogue_creator, self.ID, update_phot_rest = update_phot_rest)
        return self
        
    # def update_mask_band(self, band, bool_value):
    #     self.mask_flags[band] = bool_value

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
        
    # Selection methods

    def select_min_unmasked_bands(self, min_bands, update = True):

        if type(min_bands) != int:
            min_bands = int(min_bands)

        selection_name = f"unmasked_bands>{min_bands}"
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            # extract mask
            mask = self.phot.flux_Jy.mask
            n_unmasked_bands = len([val for val in mask if val == False])
            if n_unmasked_bands >= min_bands:
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if update:
                    self.selection_flags[selection_name] = False
        return self, selection_name
    
    def select_unmasked_bands(self, band_names, update = True):
        # ensure band_names input is of the required type, convert if not, and raise error if not convertable
        if type(band_names) in [list, np.array]:
            pass # band_names already of a valid type
        elif type(band_names) == str:
            # convert to a list, assuming the bands are separated by a "+"
            band_names = band_names.split("+")
        else:
            galfind_logger.critical(f"band_names = {band_names} with type = {type(band_names)} is not in [list, np.array, str]!")
        # ensure that each band is a valid band name in galfind
        assert(all(band_name in json.loads(config.get("Other", "ALL_BANDS")) for band_name in band_names), \
            galfind_logger.critical(f"band_names = {band_names} has at least one invalid band!"))
        
        selection_name = f"unmasked_{'+'.join(band_names)}"
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            # extract band IDs belonging to the input instrument name
            band_indices = np.array([i for i, band_name in enumerate(self.phot.instrument.band_names) if band_name in band_names])
            mask = self.phot.flux_Jy.mask[band_indices]
            if all(mask_band == False for mask_band in mask):
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if update:
                    self.selection_flags[selection_name] = False
        return self, selection_name
    
    def select_unmasked_instrument(self, instrument_name, update = True):
        
        assert(type(instrument_name) == str)
        assert(instrument_name in self.phot.instrument.name.split("+"))

        selection_name = f"unmasked_{instrument_name}"
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            # make blank instrument to display all band names of the given instrument
            instrument_bands = globals()[instrument_name]().band_names
            # extract band IDs belonging to the input instrument name
            band_indices = np.array([i for i, band in enumerate(self.phot.instrument.band_names) if band in instrument_bands])
            mask = self.phot.flux_Jy.mask[band_indices]
            if all(mask_band == False for mask_band in mask):
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if update:
                    self.selection_flags[selection_name] = False
        return self, selection_name
        
    def phot_bluewards_Lya_non_detect(self, SNR_lim, code_name = "EAZY", \
            templates = "fsps_larson", lowz_zmax = None, update = True):
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
                if update:
                    self.selection_flags[selection_name] = True
                return self, selection_name
            # find index of first Lya non-detect band
            first_Lya_non_detect_index = np.where(bands == first_Lya_non_detect_band)[0][0]
            SNR_non_detect = SNRs[:first_Lya_non_detect_index + 1]
            mask_non_detect = mask[:first_Lya_non_detect_index + 1]
            # require the first Lya non detect band and all bluewards bands to be non-detected at < SNR_lim if not masked
            if all(SNR < SNR_lim or mask for mask, SNR in zip(mask_non_detect, SNR_non_detect)):
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if update:
                    self.selection_flags[selection_name] = False
        return self, selection_name
    
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
            first_Lya_detect_band = self.phot.SED_results[code_name][templates][funcs.lowz_label(lowz_zmax)].phot_rest.first_Lya_detect_band
            if first_Lya_detect_band == None:
                if update:
                    self.selection_flags[selection_name] = False
                return self, selection_name
            # find index of first Lya non-detect band
            first_Lya_detect_index = np.where(bands == first_Lya_detect_band)[0][0]
            bands_detect = np.array(bands[first_Lya_detect_index:])
            SNR_detect = np.array(self.phot.SNR[first_Lya_detect_index:])
            mask_detect = np.array(self.phot.flux_Jy.mask[first_Lya_detect_index:])
            # option as to whether to exclude potentially shallower medium/narrow bands in this calculation
            if widebands_only:
                wide_band_detect_indices = [True if "W" in band.upper() or "LP" in band.upper() else False for band in bands_detect]
                SNR_detect = SNR_detect[wide_band_detect_indices]
                mask_detect = mask_detect[wide_band_detect_indices]
                selection_name += "_widebands"
            # selection criteria
            if all(SNR > SNR_lim or mask for mask, SNR, SNR_lim in zip(mask_detect, SNR_detect, SNR_lims)):
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if update:
                    self.selection_flags[selection_name] = False
        return self, selection_name
    
    def phot_Lya_band(self, SNR_lim, detect_or_non_detect = "detect", code_name = "EAZY", \
            templates = "fsps_larson", lowz_zmax = None, widebands_only = True, update = True):
        assert(type(SNR_lim) in [int, float])
        assert(detect_or_non_detect.lower() in ["detect", "non_detect"], \
            galfind_logger.critical(f"detect_or_non_detect = {detect_or_non_detect} must be either 'detect' or 'non_detect'!"))
        selection_name = f"Lya_band_SNR{'>' if detect_or_non_detect == 'detect' else '<'}{SNR_lim:.1f}"
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            # load bands
            bands = self.phot.instrument.band_names
            # determine Lya band(s) - usually a single band, but could be two in the case of medium bands
            first_Lya_detect_band = self.phot.SED_results[code_name][templates][funcs.lowz_label(lowz_zmax)].phot_rest.first_Lya_detect_band
            first_Lya_detect_index = np.where(bands == first_Lya_detect_band)[0][0]
            first_Lya_non_detect_band = self.phot.SED_results[code_name][templates][funcs.lowz_label(lowz_zmax)].phot_rest.first_Lya_non_detect_band
            first_Lya_non_detect_index = np.where(bands == first_Lya_non_detect_band)[0][0]
            # load SNRs, cropping by the relevant bands
            bands_detect = bands[first_Lya_detect_band : first_Lya_non_detect_index + 1]
            if widebands_only:
                wide_band_detect_indices = [True if "W" in band.upper() or "LP" in band.upper() else False for band in bands_detect]
                SNRs = self.phot.SNR[first_Lya_detect_band : first_Lya_non_detect_index + 1][wide_band_detect_indices]
                selection_name += "_widebands"
            else:
                SNRs = self.phot.SNR[first_Lya_detect_band : first_Lya_non_detect_index + 1]
                mask_bands = self.phot.flux_Jy.mask[first_Lya_detect_band : first_Lya_non_detect_index + 1]
            if len(SNRs) == 0:
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if (detect_or_non_detect == "detect" and all(SNR > SNR_lim or mask for SNR, mask in zip(SNRs, mask_bands))) or \
                    (detect_or_non_detect == "non_detect" and all(SNR < SNR_lim or mask for SNR, mask in zip(SNRs, mask_bands))):
                    if update:
                        self.selection_flags[selection_name] = True
                else:
                    if update:
                        self.selection_flags[selection_name] = False
        return self, selection_name

    def phot_SNR_crop(self, band_name_or_index, SNR_lim, update = True):
        assert(type(SNR_lim) in [int, float])
        if type(band_name_or_index) == str: # band name given
            band_name = band_name_or_index
            # given str must be a valid band in the instrument, even if the galaxy does not have this data
            assert(band_name in self.phot.instrument.new_instrument().band_names)
            # get the index of the band in question
            band_index = np.where(self.phot.instrument.band_names == band_name)[0][0]
            selection_name = f"{band_name}_SNR>{SNR_lim:.1f}"
        elif type(band_name_or_index) == int: # band index of galaxy specific data
            band_index = band_name_or_index
            galfind_logger.debug("Indexing e.g. 2 and -4 when there are 6 bands results in differing behaviour even though the same band is referenced!")
            if band_index == 0:
                selection_name = f"bluest_band_SNR>{SNR_lim:.1f}"
            elif band_index == -1:
                selection_name = f"reddest_band_SNR>{SNR_lim:.1f}"
            elif band_index > 0:
                selection_name = f"{funcs.ordinal(band_index + 1)}_bluest_band_SNR>{SNR_lim:.1f}"
            elif band_index < -1:
                selection_name = f"{funcs.ordinal(abs(band_index))}_reddest_band_SNR>{SNR_lim:.1f}"
        else:
            galfind_logger.critical(f"band_name_or_index = {band_name_or_index} has type = {type(band_name_or_index)} which must be in [str, int]")
        
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            SNR = self.phot.SNR[band_index]
            mask = self.phot.flux_Jy.mask[band_index]
            if SNR > SNR_lim or mask:
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if update:
                    self.selection_flags[selection_name] = False
        return self, selection_name
    
    def select_depth_region(self, band, region_ID, update = True):
        return NotImplementedError

    # chi squared selection functions

    def select_chi_sq_lim(self, chi_sq_lim, code_name = "EAZY", templates = "fsps_larson", lowz_zmax = None, reduced = True, update = True):
        assert(type(chi_sq_lim) in [int, float])
        assert(type(reduced) == bool)
        if reduced:
            n_bands = len([mask_band for mask_band in self.phot.flux_Jy.mask if not mask_band]) # number of unmasked bands for galaxy
            chi_sq_lim *= (n_bands - 1)
        selection_name = f"chi_sq>{chi_sq_lim:.1f}"
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            # extract chi_sq
            chi_sq = self.phot.SED_results[code_name][templates][funcs.lowz_label(lowz_zmax)].chi_sq
            if chi_sq < chi_sq_lim:
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if update:
                    self.selection_flags[selection_name] = False
        return self, selection_name

    def select_chi_sq_diff(self, chi_sq_diff, code_name = "EAZY", templates = "fsps_larson", delta_z_lowz = 0.5, update = True):
        assert(type(chi_sq_diff) in [int, float])
        assert(type(delta_z_lowz) in [int, float])
        selection_name = f"chi_sq_diff>{chi_sq_diff:.1f},dz>{delta_z_lowz:.1f}"
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            lowz_zmax_arr = self.phot.get_lowz_zmax(code_name = code_name, templates = templates)
            assert(lowz_zmax_arr[-1] == None and len(lowz_zmax_arr) > 1)
            lowz_zmax_arr = lowz_zmax_arr[:-1]
            # extract redshift + chi_sq of zfree run
            zfree = self.phot.SED_results[code_name][templates][funcs.lowz_label(None)].z
            chi_sq_zfree = self.phot.SED_results[code_name][templates][funcs.lowz_label(None)].chi_sq
            # extract redshift and chi_sq of lowz runs
            z_lowz_arr = [self.phot.SED_results[code_name][templates][funcs.lowz_label(zmax)].z for zmax in lowz_zmax_arr]
            chi_sq_lowz_arr = [self.phot.SED_results[code_name][templates][funcs.lowz_label(zmax)].chi_sq for zmax in lowz_zmax_arr]
            # determine which lowz run to use for this galaxy
            z_lowz = [z for z, lowz_zmax in zip(z_lowz_arr, lowz_zmax_arr) if zfree > lowz_zmax + delta_z_lowz]
            chi_sq_lowz = [chi_sq for chi_sq, lowz_zmax in zip(chi_sq_lowz_arr, lowz_zmax_arr) if zfree > lowz_zmax + delta_z_lowz]
            if len(chi_sq_lowz) == 0:
                if update:
                    self.selection_flags[selection_name] = True
            elif (chi_sq_lowz[-1] - chi_sq_zfree > chi_sq_diff) or (chi_sq_lowz[-1] == -1.) or (z_lowz[-1] < 0.):
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if update:
                    self.selection_flags[selection_name] = False
        return self, selection_name
    
    # z-PDF selection functions

    
    
    def select_EPOCHS(self, code_name = "EAZY", templates = "fsps_larson", lowz_zmax = None, update = True):
        
        selection_name = "EPOCHS"
        if not "NIRCam" in self.phot.instrument.name:
            galfind_logger.critical(f"NIRCam data for galaxy ID = {self.ID} must be included for EPOCHS selection!")
            if update:
                self.selection_flags[selection_name] = False
            return self, selection_name
        
        # masking takes a little while
        # self.select_unmasked_instrument("NIRCam")[1], \
        selection_names = [
            self.select_chi_sq_lim(3., code_name, templates, lowz_zmax, reduced = True)[1], \
            self.select_chi_sq_diff(9., code_name, templates, delta_z_lowz = 0.5)[1], \
            self.phot_SNR_crop(0, 2.)[1], \
            self.phot_bluewards_Lya_non_detect(2., code_name, templates, lowz_zmax)[1], \
            self.phot_redwards_Lya_detect([5., 3.], code_name, templates, lowz_zmax, widebands_only = True)[1]
        ]
        # masking criteria
         # unmasked in first band
        # SNR criteria
         # 2σ non-detected in first band
         # 2σ non-detected in all bands bluewards of Lya
        # if galaxy is detected only in the LW filters
        #first_Lya_detect_band = self.phot.SED_results[code_name][templates][funcs.lowz_label(lowz_zmax)].phot_rest.first_Lya_detect_band
        #band_names = self.phot.instrument.band_names
        #first_band = self.phot.instrument[np.where()]
        # 7σ/5σ detected in 1st/2nd bands redwards of Lya
        # else
        # 5σ/3σ detected in 1st/2nd bands redwards of Lya
        
        # if the galaxy passes all criteria
        if all(self.selection_flags[name] for name in selection_names):
            if update:
                self.selection_flags[selection_name] = True
        else:
            if update:
                self.selection_flags[selection_name] = False
        return self, selection_name

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
        mask_flags_arr = [{} for fits_cat_row in fits_cat] #f"unmasked_{band}": None for band in instrument.band_names
        selection_flags_arr = [{selection_flag: bool(fits_cat_row[selection_flag]) for selection_flag in cat_creator.selection_labels(fits_cat)} for fits_cat_row in fits_cat]
        return cls(sky_coords, IDs, phots, mask_flags_arr, selection_flags_arr)
    
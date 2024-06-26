#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:27:47 2023

@author: austind
"""

# Catalogue.py
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, join
import pyregion
from copy import copy, deepcopy
from astropy.io import fits
from pathlib import Path
import traceback
import h5py
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from tqdm import tqdm
import time
import os
from typing import Union

from .Data import Data
from .Galaxy import Galaxy, Multiple_Galaxy
from . import useful_funcs_austind as funcs
from .Catalogue_Creator import GALFIND_Catalogue_Creator
from . import LePhare, Bagpipes
from .SED_codes import SED_code
from .EAZY import EAZY
from . import config
from . import Catalogue_Base
from . import Photometry_rest
from . import galfind_logger
from .Instrument import NIRCam, MIRI, ACS_WFC, WFC3_IR, Instrument
from .Emission_lines import line_diagnostics
from .Spectrum import Spectral_Catalogue

class Catalogue(Catalogue_Base):
    
    @classmethod
    def from_pipeline(cls, survey, version, aper_diams, cat_creator, SED_fit_params_arr = [{"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": 4.}, \
            {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": 6.}, {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}], \
            instruments = ['NIRCam', 'ACS_WFC', 'WFC3_IR'], forced_phot_band = "F444W", excl_bands = [], loc_depth_min_flux_pc_errs = [5, 10], crop_by = None, timed = True):
        # make 'Data' object
        data = Data.from_pipeline(survey, version, instruments, excl_bands = excl_bands)
        return cls.from_data(data, version, aper_diams, cat_creator, SED_fit_params_arr, \
            forced_phot_band, loc_depth_min_flux_pc_errs, crop_by = crop_by, timed = timed)
    
    @classmethod
    def from_data(cls, data, version, aper_diams, cat_creator, SED_fit_params_arr, forced_phot_band = "F444W", \
                loc_depth_min_flux_pc_errs = [10], mask = True, crop_by = None, timed = True):
        # make masked local depth catalogue from the 'Data' object
        data.combine_sex_cats(forced_phot_band)
        mode = str(config["Depths"]["MODE"]).lower() # mode to calculate depths (either "n_nearest" or "rolling")
        data.calc_depths(aper_diams, mode = mode, cat_creator = cat_creator)
        data.perform_aper_corrs()
        data.make_loc_depth_cat(cat_creator, depth_mode = mode)
        return cls.from_fits_cat(data.sex_cat_master_path, version, data.instrument, cat_creator, data.survey, \
            SED_fit_params_arr, data = data, mask = mask, crop_by = crop_by, timed = timed)
    
    @classmethod
    def from_fits_cat(cls, fits_cat_path, version, instrument, cat_creator, survey, \
            SED_fit_params_arr, data = None, mask = False, excl_bands = [], crop_by = None, timed = True):
        # open the catalogue
        fits_cat = funcs.cat_from_path(fits_cat_path)
        for band_name in instrument.band_names:
            try:
                cat_creator.load_photometry(Table(fits_cat[0]), [band_name])
            except:
                # no data for the relevant band within the catalogue
                instrument.remove_band(band_name)
                print(f"{band_name} flux not loaded")
        print(f"instrument band names = {instrument.band_names}")

        # crop fits catalogue by the crop_by column name should it exist
        assert(type(crop_by) in [type(None), str, list, np.array])
        if type(crop_by) in [str]:
            crop_by = crop_by.split("+")
        if type(crop_by) != type(None):
            for name in crop_by:
                if name[:3] == "ID=":
                    fits_cat = fits_cat[fits_cat[cat_creator.ID_label].astype(int) == int(name[3:])]
                    galfind_logger.info(f"Catalogue cropped to {name}")
                elif name in fits_cat.colnames: #, galfind_logger.critical(f"Cannot crop by {name}")
                    if type(fits_cat[name][0]) in [bool, np.bool_]: #, \
                        fits_cat = fits_cat[fits_cat[name]]
                        galfind_logger.info(f"Catalogue for {survey} {version} cropped by {name}")
                    else:
                        galfind_logger.warning(f"{type(fits_cat[name][0])=} not in [bool, np.bool_]")
                else:
                    galfind_logger.warning(f"Invalid crop name == {name}! Skipping")

        # produce galaxy array from each row of the catalogue
        if timed:
            start_time = time.time()
        gals = Multiple_Galaxy.from_fits_cat(fits_cat, instrument, cat_creator, [{}], timed = timed).gals #codes, lowz_zmax, templates_arr).gals
        if timed:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Finished loading in {len(gals)} galaxies. This took {elapsed_time:.6f} seconds")
        # make catalogue with no SED fitting information
        cat_obj = cls(gals, fits_cat_path, survey, cat_creator, instrument, SED_fit_params_arr, version = version, crops = crop_by)
        #print(cat_obj)
        if cat_obj != None:
            cat_obj.data = data
        if mask:
            cat_obj.mask(timed = timed)
        # run SED fitting for the appropriate SED_fit_params
        for SED_fit_params in SED_fit_params_arr:
            cat_obj = SED_fit_params["code"].fit_cat(cat_obj, SED_fit_params, timed = timed)
            cat_obj.load_SED_rest_properties(SED_fit_params, timed = timed) # load SED rest properties
        return cat_obj
    
    def save_phot_PDF_paths(self, PDF_paths, SED_fit_params):
        if "phot_PDF_paths" not in self.__dict__.keys():
            self.phot_PDF_paths = {}
        self.phot_PDF_paths[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)] = PDF_paths

    def save_phot_SED_paths(self, SED_paths, SED_fit_params):
        if "phot_SED_paths" not in self.__dict__.keys():
            self.phot_SED_paths = {}
        self.phot_SED_paths[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)] = SED_paths
    
    def update_SED_results(self, cat_SED_results, timed = True):
        assert(len(cat_SED_results) == len(self)) # if this is not the case then instead should cross match IDs between self and gal_SED_result
        galfind_logger.info("Updating SED results in galfind catalogue object")
        if timed:
            [gal.update(gal_SED_result) for gal, gal_SED_result \
                in tqdm(zip(self, cat_SED_results), desc = "Updating galaxy SED results", total = len(self))]
        else:
            [gal.update(gal_SED_result) for gal, gal_SED_result in zip(self, cat_SED_results)]
    
    # Spectroscopy
        
    def match_available_spectra(self):
        #breakpoint()
        # make catalogue consisting of spectra downloaded from the DJA
        DJA_cat = np.sum([Spectral_Catalogue.from_DJA(ra_range = self.ra_range, \
            dec_range = self.dec_range, version = version) for version in ["v1", "v2"]])
        # cross match this catalogue 
        cross_matched_cat = self * DJA_cat
        print(str(cross_matched_cat))
        return cross_matched_cat

    # %%
    
    # def calc_ext_src_corrs(self, band, ID = None):
    #     # load the catalogue (for obtaining the FLUX_AUTOs; THIS SHOULD BE MORE GENERAL IN FUTURE TO GET THESE FROM THE GALAXY OBJECT!!!)
    #     tab = self.open_full_cat()
    #     if ID != None:
    #         tab = tab[tab["NUMBER"] == ID]
    #         cat = self[self["ID"] == ID]
    #     else:
    #         cat = self
    #     # load the relevant FLUX_AUTO from SExtractor output
    #     flux_autos = funcs.flux_image_to_Jy(np.array(tab[f"FLUX_AUTO_{band}"]), self.data.im_zps[band])
    #     ext_src_corrs = self.cap_ext_src_corrs([flux_autos[i] / gal.phot.flux_Jy[np.where(band == \
    #         gal.phot.instrument.band_names)[0][0]].value for i, gal in enumerate(cat)])
    #     return ext_src_corrs

    # should also include errors here

    def load_band_properties_from_cat(self, cat_colname: str, save_name: str, multiply_factor: \
            Union[dict, u.Quantity, u.Magnitude, None] = None):
        if not hasattr(self[0], save_name):
            # load the same property from every available band
            # open catalogue with astropy
            fits_cat = self.open_cat(cropped = True)
            if type(multiply_factor) == type(None):
                multiply_factor = {band: 1. * u.dimensionless_unscaled for band in self.instrument.band_names if f"{cat_colname}_{band}" in fits_cat.colnames}
            elif type(multiply_factor) != dict:
                multiply_factor = {band: multiply_factor for band in self.instrument.band_names if f"{cat_colname}_{band}" in fits_cat.colnames}
            # load in speed can be improved here!
            cat_band_properties = {band: np.array(fits_cat[f"{cat_colname}_{band}"]) * multiply_factor[band] \
                for band in self.instrument.band_names if f"{cat_colname}_{band}" in fits_cat.colnames}
            cat_band_properties = [{band: cat_band_properties[band][i] for band in cat_band_properties.keys()} for i in range(len(fits_cat))]
            [gal.load_property(gal_properties, save_name) for gal, gal_properties in zip(self, cat_band_properties)]
            galfind_logger.info(f"Loaded {cat_colname} from {self.cat_path} saved as {save_name} for bands = {cat_band_properties[0].keys()}")

    def load_property_from_cat(self, cat_colname: str, save_name: str, multiply_factor: \
            Union[u.Quantity, u.Magnitude] = 1. * u.dimensionless_unscaled, unit: u.Unit = u.dimensionless_unscaled):
        if not hasattr(self[0], save_name):
            # open catalogue with astropy
            fits_cat = self.open_cat(cropped = True)
            cat_property = np.array(fits_cat[cat_colname])
            assert len(cat_property) == len(self)
            [gal.load_property(gal_property * multiply_factor * unit, save_name) for gal, gal_property in zip(self, cat_property)]
            galfind_logger.info(f"Loaded {cat_colname} from {self.cat_path} saved as {save_name}")

    def mask(self, timed: bool = True): #, mask_instrument = NIRCam()):
        galfind_logger.info(f"Running masking code for {self.cat_path}.")
        # determine whether to overwrite catalogue or not
        overwrite = config["Masking"].getboolean("OVERWRITE_MASK_COLS")
        if overwrite:
            galfind_logger.info("OVERWRITE_MASK_COLS = YES, updating catalogue with masking columns.")
        # open catalogue with astropy
        fits_cat = self.open_cat(cropped = True)
        # update input catalogue if it hasnt already been masked or if wanted ONLY if len(self) == len(cat)
        if len(self) != len(fits_cat):
            galfind_logger.warning(f"len(self) = {len(self)}, len(cat) = {len(fits_cat)} -> len(self) != len(cat). Skipping masking for {self.survey} {self.version}!")
        elif (not "MASKED" in fits_cat.meta.keys() or overwrite):
            galfind_logger.info(f"Masking catalogue for {self.survey} {self.version}")
            
            # calculate x,y for each galaxy in catalogue
            cat_x, cat_y = self.data.load_wcs(self.data.alignment_band).world_to_pixel(SkyCoord(fits_cat[self.cat_creator.ra_dec_labels["RA"]], fits_cat[self.cat_creator.ra_dec_labels["DEC"]]))
            
            # make columns for individual band masking
            if config["Masking"].getboolean("MASK_BANDS"):
                for band in tqdm(self.instrument.band_names, desc = "Masking galfind catalogue object", total = len(self.instrument)):
                    # open .fits mask for band
                    mask = self.data.load_mask(band)
                    # determine whether a galaxy is unmasked
                    unmasked_band = np.array([False if x < 0. or x >= mask.shape[1] or y < 0. or y >= mask.shape[0] else not bool(mask[int(y)][int(x)]) for x, y in zip(cat_x, cat_y)])
                    # update catalogue with new column
                    fits_cat[f"unmasked_{band}"] = unmasked_band # assumes order of catalogue and galaxies in self is consistent
                    # update galaxy objects in catalogue - current bottleneck
                    [gal.mask_flags.update({band: unmasked_band_gal}) for gal, unmasked_band_gal in zip(self, unmasked_band)]
                # make columns for masking by instrument
                

            # determine which cluster/blank masking columns are wanted
            mask_labels = []
            mask_paths = []
            default_blank_bool_arr = []
            if config["Masking"].getboolean("MASK_CLUSTER_MODULE"): # make blank field mask
                mask_labels.append("blank_module")
                mask_paths.append(self.data.blank_mask_path)
                default_blank_bool_arr.append(True)
            if config["Masking"].getboolean("MASK_CLUSTER_CORE"): # make cluster mask
                mask_labels.append("cluster")
                mask_paths.append(self.data.cluster_mask_path)
                default_blank_bool_arr.append(False)
            
            # mask columns in catalogue + galfind galaxies
            for mask_label, mask_path, default_blank_bool in zip(mask_labels, mask_paths, default_blank_bool_arr):
                # if using a blank field
                if self.data.is_blank:
                    galfind_logger.info(f"{self.survey} {self.version} is blank. Making '{mask_label}' boolean columns")
                    mask_data = [default_blank_bool for i in range(len(fits_cat))] # default behaviour
                else:
                    galfind_logger.info(f"{self.survey} {self.version} contains a cluster. Making '{mask_label}' boolean columns")
                    # open relevant .fits mask
                    mask = fits.open(mask_path)[1].data
                    # determine whether a galaxy is in a blank module
                    if default_blank_bool: # True if outside the mask
                        galfind_logger.warning("This masking assumes that the blank mask covers the cluster module and then invokes negatives.")
                        mask_data = np.array([False if x < 0. or x >= mask.shape[1] or y < 0. or y >= mask.shape[0] else not bool(mask[int(y)][int(x)]) for x, y in zip(cat_x, cat_y)])
                    else: # True if within the mask
                        mask_data = np.array([False if x < 0. or x >= mask.shape[1] or y < 0. or y >= mask.shape[0] else bool(mask[int(y)][int(x)]) for x, y in zip(cat_x, cat_y)])
                fits_cat[mask_label] = mask_data # update catalogue with boolean column

            # update catalogue metadata
            fits_cat.meta = {**fits_cat.meta, **{"MASKED": True, "HIERARCH MASK_BANDS": config["Masking"].getboolean("MASK_BANDS"), \
                "HIERARCH MASK_CLUSTER_MODULE": config["Masking"].getboolean("MASK_CLUSTER_MODULE"), \
                "HIERARCH MASK_CLUSTER_CORE": config["Masking"].getboolean("MASK_CLUSTER_CORE")}}
            # save catalogue
            fits_cat.write(self.cat_path, overwrite = True)
            funcs.change_file_permissions(self.cat_path)
            # update catalogue README
            galfind_logger.warning("REQUIRED UPDATE: Update README for catalogue masking columns")
            # update masking of galfind galaxy objects
            galfind_logger.info("Masking galfind galaxy objects in catalogue")
            assert(len(fits_cat) == len(self))
            mask_arr = self.cat_creator.load_mask(fits_cat, self.instrument.band_names, \
                gal_band_mask = self.cat_creator.load_photometry(fits_cat, self.instrument.band_names)[2])
            [gal.update_mask(mask, update_phot_rest = False) for gal, mask in \
                tqdm(zip(self, mask_arr), total = len(self), desc = "Masking galfind galaxy objects")]
        else:
            galfind_logger.info(f"Catalogue for {self.survey} {self.version} already masked. Skipping!")

    def make_cutouts(self, IDs, cutout_size = 32):
        if type(IDs) == int:
            IDs = [IDs]
        for band in tqdm(self.instrument.band_names, total = len(self.instrument), desc = "Making band cutouts"):
            rerun = False
            if config.getboolean("Cutouts", "OVERWRITE_CUTOUTS"):
                rerun = True
            else:
                for gal in self:
                    out_path = f"{config['Cutouts']['CUTOUT_DIR']}/{self.version}/{self.survey}/{band}/{gal.ID}.fits"
                    if Path(out_path).is_file():
                        size = fits.open(out_path)[0].header["size"]
                        if size != cutout_size:
                            rerun = True
                    else:
                        rerun = True
            if rerun:
                im_data, im_header, seg_data, seg_header = self.data.load_data(band, incl_mask = False)
                wht_data = self.data.load_wht(band)
                rms_err_data = self.data.load_rms_err(band)
                wcs = WCS(im_header)
                for gal in self:
                    if gal.ID in IDs:
                        gal.make_cutout(band, data = {"SCI": im_data, "SEG": seg_data, 'WHT': wht_data, 'RMS_ERR':rms_err_data}, \
                            wcs = wcs, im_header = im_header, survey = self.survey, version = self.version, cutout_size = cutout_size)
            else:
                for gal in self:
                    if gal.ID in IDs:
                        gal.cutout_paths[band] = f"{config['Cutouts']['CUTOUT_DIR']}/{self.version}/{self.survey}/{band}/{gal.ID}.fits"
                print(f"Cutouts for {band} already exist. Skipping.")
    def make_RGB_images(self, IDs, cutout_size = 32):
        return NotImplementedError
    
    def plot_phot_diagnostics(self, 
            SED_fit_params_arr = [{"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, {"code": EAZY(), "templates": "fsps_larson", "dz": 0.5}], \
            zPDF_plot_SED_fit_params_arr = [{"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, {"code": EAZY(), "templates": "fsps_larson", "dz": 0.5}], \
            wav_unit = u.um, flux_unit = u.ABmag):
        
        # figure size may well depend on how many bands there are
        overall_fig = plt.figure(figsize = (8, 7), constrained_layout = True)
        fig, cutout_fig = overall_fig.subfigures(2, 1, hspace = -2, height_ratios = [2, 1] if len(self.data.instrument) <= 8 else [1.8, 1])
    
        gs = fig.add_gridspec(2, 4)
        phot_ax = fig.add_subplot(gs[:, 0:3])

        PDF_ax = [fig.add_subplot(gs[0, 3:]), fig.add_subplot(gs[1, 3:])]
        
        if len(self.data.instrument) <= 8:
            gridspec_cutout = cutout_fig.add_gridspec(1, len(self.data.instrument))
        else:
            gridspec_cutout = cutout_fig.add_gridspec(2, int(np.ceil(len(self.data.instrument) / 2)))
        
        cutout_ax_list = []
        for i, band in enumerate(self.instrument):
            cutout_ax = cutout_fig.add_subplot(gridspec_cutout[i])
            cutout_ax.set_aspect('equal', adjustable='box', anchor='N')
            cutout_ax.set_xticks([])
            cutout_ax.set_yticks([])
            cutout_ax_list.append(cutout_ax)

        # plot SEDs
        out_paths = [gal.plot_phot_diagnostic([cutout_ax_list, phot_ax, PDF_ax], self.data, \
            SED_fit_params_arr, zPDF_plot_SED_fit_params_arr, wav_unit, flux_unit) \
            for gal in tqdm(self, total = len(self), desc = "Plotting photometry diagnostic plots")]

        # make a folder to store symlinked photometric diagnostic plots for selected galaxies
        if self.crops != []:
            # create symlink to selection folder for diagnostic plots
            for gal, out_path in zip(self, out_paths):
                selection_path = f"{config['Selection']['SELECTION_DIR']}/SED_plots/{self.version}/{self.instrument.name}/{'+'.join(self.crops)}/{self.survey}/{str(gal.ID)}.png"
                funcs.make_dirs(selection_path)
                try:
                    os.symlink(out_path, selection_path)
                except FileExistsError: # replace existing file
                    os.remove(selection_path)
                    os.symlink(out_path, selection_path)

    # Selection functions
                    
    def select_all_bands(self):
        return self.select_min_bands(len(self.instrument))
                    
    def select_min_bands(self, min_bands):
        return self.perform_selection(Galaxy.select_min_bands, min_bands)
        
    # Masking selection

    def select_min_unmasked_bands(self, min_bands):
        return self.perform_selection(Galaxy.select_min_unmasked_bands, min_bands)
    
    #  already made these boolean columns in the catalogue
    def select_unmasked_bands(self, bands):
        return self.perform_selection(Galaxy.select_unmasked_band, bands)
    
    def select_unmasked_instrument(self, instrument_name):
        return self.perform_selection(Galaxy.select_unmasked_instrument, instrument_name)

    # Photometric galaxy property selection functions

    def select_phot_galaxy_property(self, property_name, gtr_or_less, property_lim, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
        return self.perform_selection(Galaxy.select_phot_galaxy_property, property_name, gtr_or_less, property_lim, SED_fit_params)

    def select_phot_galaxy_property_bin(self, property_name, property_lims, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
        return self.perform_selection(Galaxy.select_phot_galaxy_property_bin, property_name, property_lims, SED_fit_params)

    # SNR selection functions

    def phot_bluewards_Lya_non_detect(self, SNR_lim, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
        return self.perform_selection(Galaxy.phot_bluewards_Lya_non_detect, SNR_lim, SED_fit_params)

    def phot_redwards_Lya_detect(self, SNR_lims, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, widebands_only = True):
        return self.perform_selection(Galaxy.phot_redwards_Lya_detect, SNR_lims, SED_fit_params, widebands_only)

    def phot_Lya_band(self, SNR_lim, detect_or_non_detect = "detect", \
            SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, widebands_only = True):
        return self.perform_selection(Galaxy.phot_Lya_band, SNR_lim, detect_or_non_detect, SED_fit_params, widebands_only)

    def phot_SNR_crop(self, band_name_or_index, SNR_lim, detect_or_non_detect = "detect"):
        return self.perform_selection(Galaxy.phot_SNR_crop, band_name_or_index, SNR_lim, detect_or_non_detect)

    # Emission line selection functions

    def select_rest_UV_line_emitters_dmag(self, emission_line_name, delta_m, rest_UV_wav_lims = [1_250., 3_000.] * u.AA, \
            medium_bands_only = True, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, update = True):
        return self.perform_selection(Galaxy.select_rest_UV_line_emitters_dmag, emission_line_name, \
            delta_m, rest_UV_wav_lims, medium_bands_only, SED_fit_params)

    def select_rest_UV_line_emitters_sigma(self, emission_line_name, sigma, rest_UV_wav_lims = [1_250., 3_000.] * u.AA, \
            medium_bands_only = True, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
        return self.perform_selection(Galaxy.select_rest_UV_line_emitters_sigma, emission_line_name, \
            sigma, rest_UV_wav_lims, medium_bands_only, SED_fit_params)

    # Colour selection functions

    def select_colour(self, colour_bands, colour_val, bluer_or_redder):
        return self.perform_selection(Galaxy.select_colour, colour_bands, colour_val, bluer_or_redder)
    
    def select_colour_colour(self, colour_bands_arr, colour_select_func):
        return self.perform_selection(Galaxy.select_colour_colour, colour_bands_arr, colour_select_func)
    
    def select_UVJ(self, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, quiescent_or_star_forming = "quiescent"):
        return self.perform_selection(Galaxy.select_UVJ, SED_fit_params, quiescent_or_star_forming)
    
    def select_Kokorev24_LRDs(self):
        # only perform this selection if all relevant bands are present
        required_bands = ["F115W", "F150W", "F200W", "F277W", "F356W", "F444W"]
        if all(band_name in self.instrument.band_names for band_name in required_bands):
            # red1 selection (z<6 LRDs)
            self.perform_selection(Galaxy.select_colour, ["F115W", "F150W"], 0.8, "bluer", make_cat_copy = False)
            self.perform_selection(Galaxy.select_colour, ["F200W", "F277W"], 0.7, "redder", make_cat_copy = False)
            self.perform_selection(Galaxy.select_colour, ["F200W", "F356W"], 1.0, "redder", make_cat_copy = False)
            # red2 selection (z>6 LRDs)
            self.perform_selection(Galaxy.select_colour, ["F150W", "F200W"], 0.8, "bluer", make_cat_copy = False)
            self.perform_selection(Galaxy.select_colour, ["F277W", "F356W"], 0.6, "redder", make_cat_copy = False)
            self.perform_selection(Galaxy.select_colour, ["F277W", "F444W"], 0.7, "redder", make_cat_copy = False)
            return self.perform_selection(Galaxy.select_Kokorev24_LRDs)
        else:
            galfind_logger.warning(f"Not all of {required_bands} in {self.instrument.band_names=}, skipping 'select_Kokorev24_LRDs' selection")
    
    # Depth region selection

    def select_depth_region(self, band, region_ID, update = True):
        return NotImplementedError
    
    # Chi squared selection functions

    def select_chi_sq_lim(self, chi_sq_lim, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, reduced = True):
        return self.perform_selection(Galaxy.select_chi_sq_lim, chi_sq_lim, SED_fit_params, reduced)

    def select_chi_sq_diff(self, chi_sq_diff, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, delta_z_lowz = 0.5):
        return self.perform_selection(Galaxy.select_chi_sq_diff, chi_sq_diff, SED_fit_params, delta_z_lowz)

    # Redshift PDF selection functions

    def select_robust_zPDF(self, integral_lim, delta_z_over_z, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
        return self.perform_selection(Galaxy.select_robust_zPDF, integral_lim, delta_z_over_z, SED_fit_params)
    
    # Morphology selection functions

    def select_band_flux_radius(self, band, gtr_or_less, lim, make_cat_copy = False):
        assert(band in self.instrument.band_names)
        # load in effective radii as calculated from SExtractor
        self.load_band_properties_from_cat("FLUX_RADIUS", "sex_Re", None)
        return self.perform_selection(Galaxy.select_band_flux_radius, band, gtr_or_less, lim, make_cat_copy = make_cat_copy)

    # Full sample selection functions - these chain the above functions

    def select_EPOCHS(self, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, allow_lowz = False, hot_pixel_bands = ["F277W", "F356W", "F444W"]):
        self.perform_selection(Galaxy.select_min_bands, 4., make_cat_copy = False) # minimum 4 photometric bands
        self.perform_selection(Galaxy.select_unmasked_instrument, NIRCam(), make_cat_copy = False) # all NIRCam bands unmasked
        [self.select_band_flux_radius(band, "gtr", 1.5, make_cat_copy = False) for band in hot_pixel_bands if band in self.instrument.band_names] # LW NIRCam wideband Re>1.5 pix
        if not allow_lowz:
            self.perform_selection(Galaxy.phot_SNR_crop, 0, 2., "non_detect", make_cat_copy = False) # 2σ non-detected in first band
        self.perform_selection(Galaxy.phot_bluewards_Lya_non_detect, 2., SED_fit_params, make_cat_copy = False) # 2σ non-detected in all bands bluewards of Lyα
        self.perform_selection(Galaxy.phot_redwards_Lya_detect, [5., 3.], SED_fit_params, True, make_cat_copy = False) # 5σ/3σ detected in first/second band redwards of Lyα
        self.perform_selection(Galaxy.select_chi_sq_lim, 3., SED_fit_params, True, make_cat_copy = False) # χ^2_red < 3
        self.perform_selection(Galaxy.select_chi_sq_diff, 9., SED_fit_params, 0.5, make_cat_copy = False) # Δχ^2 < 9 between redshift free and low redshift SED fits, with Δz=0.5 tolerance 
        self.perform_selection(Galaxy.select_robust_zPDF, 0.6, 0.1, SED_fit_params, make_cat_copy = False) # 60% of redshift PDF must lie within z ± z * 0.1
        return self.perform_selection(Galaxy.select_EPOCHS, SED_fit_params, allow_lowz, hot_pixel_bands)

    def perform_selection(self, selection_function, *args, make_cat_copy = True):
        # extract selection name from galaxy method output
        selection_name = selection_function(self[0], *args, update = False)[1]
        # open catalogue
        ##breakpoint()
        # perform selection if not previously performed
        if selection_name not in self.selection_cols:
            # perform calculation for each galaxy and update galaxies in self
            [selection_function(gal, *args, update = True)[0] for gal in \
                tqdm(self, total = len(self), desc = f"Cropping {selection_name}")]
        if make_cat_copy:
            # crop catalogue by the selection
            cat_copy = self._crop_by_selection(selection_name)
            # append .fits table if not already done so
            cat_copy._append_selection_to_fits(selection_name) # this should be written outside of make_cat_copy!
            return cat_copy

    def _crop_by_selection(self, selection_name):
        # make a deep copy of the current catalogue object
        cat_copy = deepcopy(self)
        # crop deep copied catalogue to only the selected galaxies
        cat_copy.gals = cat_copy[getattr(self, selection_name)]
        if selection_name not in cat_copy.crops:
            # make a note of this crop if it is new
            cat_copy.crops.append(selection_name)
        return cat_copy

    def _append_selection_to_fits(self, selection_name):
        ##breakpoint()
        # append .fits table if not already done so for this selection
        if not selection_name in self.selection_cols:
            assert(all(getattr(self, selection_name) == True))
            full_cat = self.open_cat()
            selection_cat = Table({"ID_temp": self.ID, selection_name: np.full(len(self), True)})
            output_cat = join(full_cat, selection_cat, keys_left = "NUMBER", keys_right = "ID_temp", join_type = "outer")
            output_cat.remove_column("ID_temp")
            # fill unselected columns with False rather than leaving as masked post-join
            output_cat[selection_name].fill_value = False
            output_cat = output_cat.filled()
            # ensure no rows are lost during this column append
            assert(len(output_cat) == len(full_cat))
            output_cat.meta = {**full_cat.meta, **{f"HIERARCH SELECTED_{selection_name}": True}}
            galfind_logger.info(f"Appending {selection_name} to {self.cat_path=}")
            output_cat.write(self.cat_path, overwrite = True)
            funcs.change_file_permissions(self.cat_path)
            self.selection_cols.append(selection_name)
        else:
            galfind_logger.info(f"Already appended {selection_name} to {self.cat_path=}")

    # %%
    # SED property functions 
            
    # Rest-frame UV property calculation functions - these are not independent of each other
    
    # beta_phot tqdm bar not working appropriately!
    def calc_beta_phot(self, rest_UV_wav_lims = [1_250., 3_000.] * u.AA, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
        self.calc_SED_rest_property(Photometry_rest.calc_beta_phot, SED_fit_params, rest_UV_wav_lims)
        
    def calc_fesc_from_beta_phot(self, rest_UV_wav_lims = [1_250., 3_000.] * u.AA, conv_author_year = "Chisholm22", \
            SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
        self.calc_beta_phot(rest_UV_wav_lims, SED_fit_params)
        self.calc_SED_rest_property(Photometry_rest.calc_fesc_from_beta_phot, SED_fit_params, rest_UV_wav_lims, conv_author_year)

    def calc_AUV_from_beta_phot(self, rest_UV_wav_lims = [1_250., 3_000.] * u.AA, ref_wav = 1_500. * u.AA, conv_author_year = "M99", \
            SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
        self.calc_beta_phot(rest_UV_wav_lims, SED_fit_params)
        self.calc_SED_rest_property(Photometry_rest.calc_AUV_from_beta_phot, SED_fit_params, rest_UV_wav_lims, ref_wav, conv_author_year)

    def calc_mUV_phot(self, rest_UV_wav_lims: u.Quantity = [1_250., 3_000.] * u.AA, ref_wav: u.Quantity = 1_500. * u.AA, \
            SED_fit_params: dict = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
        self.calc_SED_rest_property(Photometry_rest.calc_mUV_phot, SED_fit_params, rest_UV_wav_lims, ref_wav)

    def calc_MUV_phot(self, rest_UV_wav_lims: u.Quantity = [1_250., 3_000.] * u.AA, ref_wav: u.Quantity = 1_500. * u.AA, \
            SED_fit_params: dict = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
        self.calc_SED_rest_property(Photometry_rest.calc_MUV_phot, SED_fit_params, rest_UV_wav_lims, ref_wav)
    
    def calc_LUV_phot(self, frame: str = "obs", rest_UV_wav_lims = [1_250., 3_000.] * u.AA, ref_wav = 1_500. * u.AA, \
            AUV_beta_conv_author_year = "M99", SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
        self.calc_SED_rest_property(Photometry_rest.calc_LUV_phot, SED_fit_params, frame, rest_UV_wav_lims, ref_wav, AUV_beta_conv_author_year)
    
    def calc_SFR_UV_phot(self, frame: str = "obs", rest_UV_wav_lims: u.Quantity = [1_250., 3_000.] * u.AA, \
            ref_wav: u.Quantity = 1_500. * u.AA, AUV_beta_conv_author_year: Union[str, None] = "M99", kappa_UV_conv_author_year: str = "MD14", \
            SED_fit_params: dict = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
        self.calc_SED_rest_property(Photometry_rest.calc_SFR_UV_phot, SED_fit_params, frame, \
            rest_UV_wav_lims, ref_wav, AUV_beta_conv_author_year, kappa_UV_conv_author_year)
    
    def calc_rest_UV_properties(self, frame: str = "obs", rest_UV_wav_lims: u.Quantity = [1_250., 3_000.] * u.AA, \
            ref_wav: u.Quantity = 1_500. * u.AA, AUV_beta_conv_author_year: Union[str, None] = "M99", kappa_UV_conv_author_year: str = "MD14", \
            SED_fit_params: dict = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
        self.calc_beta_phot(rest_UV_wav_lims, SED_fit_params)
        self.calc_AUV_from_beta_phot(rest_UV_wav_lims, ref_wav, AUV_beta_conv_author_year, SED_fit_params)
        self.calc_mUV_phot(rest_UV_wav_lims, ref_wav, SED_fit_params)
        self.calc_MUV_phot(rest_UV_wav_lims, ref_wav, SED_fit_params)
        self.calc_LUV_phot(frame, rest_UV_wav_lims, ref_wav, AUV_beta_conv_author_year, SED_fit_params)
        self.calc_SFR_UV_phot(frame, rest_UV_wav_lims, ref_wav, AUV_beta_conv_author_year, kappa_UV_conv_author_year, SED_fit_params)

    # Emission line EWs from the rest frame UV photometry
        
    def calc_cont_rest_optical(self, line_names, rest_optical_wavs = [3_700., 7_000.] * u.AA, \
            SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
        self.calc_SED_rest_property(Photometry_rest.calc_cont_rest_optical, SED_fit_params, line_names, rest_optical_wavs)

    def calc_EW_rest_optical(self, line_names, frame: str, flux_contamination_params: dict = {"mu": 0., "sigma": 0.}, \
            medium_bands_only = True, rest_optical_wavs = [3_700., 7_000.] * u.AA, \
            SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
        self.calc_cont_rest_optical(line_names, rest_optical_wavs, SED_fit_params)
        self.calc_SED_rest_property(Photometry_rest.calc_EW_rest_optical, SED_fit_params, line_names, frame, flux_contamination_params, medium_bands_only, rest_optical_wavs)

    def calc_dust_atten(self, calc_wav: u.Quantity, dust_author_year: str = "M99", dust_law: str = "C00", \
            dust_origin: str = "UV", rest_UV_wav_lims: u.Quantity = [1_250., 3_000.] * u.AA, \
            ref_wav: u.Quantity = 1_500. * u.AA, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
        assert(all(type(name) != type(None) for name in [dust_author_year, dust_law, dust_origin]))
        self.calc_AUV_from_beta_phot(rest_UV_wav_lims, ref_wav, dust_author_year, SED_fit_params)
        self.calc_SED_rest_property(Photometry_rest.calc_dust_atten, SED_fit_params, calc_wav, dust_author_year, dust_law, dust_origin, rest_UV_wav_lims, ref_wav)

    def calc_line_flux_rest_optical(self, line_names: list, frame: str, flux_contamination_params: dict = {"mu": 0.}, dust_author_year = "M99", \
            dust_law = "C00", dust_origin = "UV", medium_bands_only = True, rest_optical_wavs = [3_700., 7_000.] * u.AA, rest_UV_wav_lims = [1_250., 3_000.] * u.AA, \
            ref_wav = 1_500. * u.AA, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
        self.calc_EW_rest_optical(line_names, frame, flux_contamination_params, medium_bands_only, rest_optical_wavs, SED_fit_params)
        if all(type(name) != type(None) for name in [dust_author_year, dust_law, dust_origin]):
            self.calc_dust_atten(line_diagnostics[line_names[0]]["line_wav"], dust_author_year, dust_law, dust_origin, rest_UV_wav_lims, ref_wav, SED_fit_params)
        self.calc_SED_rest_property(Photometry_rest.calc_line_flux_rest_optical, SED_fit_params, line_names, \
            frame, flux_contamination_params, dust_author_year, dust_law, dust_origin, medium_bands_only, rest_optical_wavs)

    def calc_line_lum_rest_optical(self, line_names: list, frame: str, flux_contamination_params: dict = {"mu": 0.}, dust_author_year: str = "M99", \
            dust_law: str = "C00", dust_origin: str = "UV", medium_bands_only = True, rest_optical_wavs = [3_700., 7_000.] * u.AA, \
            rest_UV_wav_lims = [1_250., 3_000.] * u.AA, ref_wav = 1_500. * u.AA, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
        self.calc_line_flux_rest_optical(line_names, frame, flux_contamination_params, dust_author_year, dust_law, \
            dust_origin, medium_bands_only, rest_optical_wavs, rest_UV_wav_lims, ref_wav, SED_fit_params)
        self.calc_SED_rest_property(Photometry_rest.calc_line_lum_rest_optical, SED_fit_params, line_names, frame, \
            flux_contamination_params, dust_author_year, dust_law, dust_origin, medium_bands_only, rest_optical_wavs, rest_UV_wav_lims, ref_wav)

    # should be generalized slightly more
    def calc_xi_ion(self, frame: str = "rest", line_names: list = ["Halpha", "[NII]-6583"], flux_contamination_params: dict = {"mu": 0.1}, fesc_author_year: str = "fesc=0.0", \
            dust_author_year: str = "M99", dust_law: str = "C00", dust_origin: str = "UV", medium_bands_only: bool = True, rest_optical_wavs = [3_700., 7_000.] * u.AA, \
            rest_UV_wav_lims = [1_250., 3_000.] * u.AA, ref_wav = 1_500. * u.AA, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
        self.calc_line_lum_rest_optical(line_names, frame, flux_contamination_params, dust_author_year, dust_law, dust_origin, \
            medium_bands_only, rest_optical_wavs, rest_UV_wav_lims, ref_wav, SED_fit_params)
        if "fesc" not in fesc_author_year:
            self.calc_SED_rest_property(Photometry_rest.calc_fesc_from_beta_phot, SED_fit_params, rest_UV_wav_lims, fesc_author_year)
        self.calc_SED_rest_property(Photometry_rest.calc_xi_ion, SED_fit_params, frame, line_names, flux_contamination_params, fesc_author_year, \
            dust_author_year, dust_law, dust_origin, medium_bands_only, rest_optical_wavs, rest_UV_wav_lims, ref_wav)

    # Global SED rest-frame photometry calculations

    def calc_SED_rest_property(self, SED_rest_property_function, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, *args):
        key = SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)
        property_name = SED_rest_property_function(self[0].phot.SED_results[key].phot_rest, *args, extract_property_name = True)
        # self.SED_rest_properties should contain the selections these properties have been calculated for
        if key not in self.SED_rest_properties.keys():
            self.SED_rest_properties[key] = []
        if property_name not in self.SED_rest_properties[key]:
            # perform calculation for each galaxy and update galaxies in self
            self.gals = [deepcopy(gal)._calc_SED_rest_property(SED_rest_property_function, key, *args) \
                for gal in tqdm(self, total = len(self), desc = f"Calculating {property_name}")]
            galfind_logger.info(f"Calculated {property_name}")
            #[SED_rest_property_function(gal.phot.SED_results[key].phot_rest, *args)[0] for gal in \
            #    tqdm(self, total = len(self), desc = f"Calculating {property_name}")]
            # save the property PDFs
            self._save_SED_rest_PDFs(property_name, SED_fit_params)
            # save the property name
            self.SED_rest_properties[key].append(property_name)
            self._append_SED_rest_property_to_fits(property_name, key)
        
    def _save_SED_rest_PDFs(self, property_name, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
        save_dir = f"{config['PhotProperties']['PDF_SAVE_DIR']}/{self.version}/{self.instrument.name}/{self.survey}"
        funcs.make_dirs(f"{save_dir}/dummy_path.ecsv")
        [gal._save_SED_rest_PDFs(property_name, save_dir, SED_fit_params) for gal in self]
    
    def _append_SED_rest_property_to_fits(self, property_name: str, SED_fit_params_label: str, save_kwargs: bool = True):
        try:
            SED_rest_property_tab = self.open_cat(cropped = False, hdu = SED_fit_params_label)
        except FileNotFoundError:
            SED_rest_property_tab = None
        # if the table does not exist, make the table from scratch
        if type(SED_rest_property_tab) == type(None):
            properties = self.__getattr__(property_name, phot_type = "rest", property_type = "vals")
            property_errs = self.__getattr__(property_name, phot_type = "rest", property_type = "errs")
            out_tab = Table({self.cat_creator.ID_label: np.array(self.ID).astype(int), property_name: properties, \
                f"{property_name}_l1": property_errs[:, 0], f"{property_name}_u1": property_errs[:, 1]}, dtype = [int, float, float, float])
            out_tab.meta = {f"HIERARCH SED_REST_{property_name}": True}
        # else if these properties have not already been calculated for this galaxy sample
        elif f"SED_REST_{property_name}" not in SED_rest_property_tab.meta.keys():
            # ensure this property has not been calculated for a different subset of galaxies in this field
            assert(f"SED_REST_{property_name}" not in ["_".join(label.split("_")[:-(1 + "+".join(self.crops).count("_"))]) for label in SED_rest_property_tab.meta.keys()])
            #galfind_logger.warning("Needs re-writing in the case of the same property being calculated for multiple samples of galaxies in the same field")
            properties = self.__getattr__(property_name, phot_type = "rest", property_type = "vals")
            property_errs = self.__getattr__(property_name, phot_type = "rest", property_type = "errs")
            new_SED_rest_property_tab = Table({f"{self.cat_creator.ID_label}_temp": np.array(self.ID).astype(int), property_name: properties, \
                f"{property_name}_l1": property_errs[:, 0], f"{property_name}_u1": property_errs[:, 1]}, dtype = [int, float, float, float])
            new_SED_rest_property_tab.meta = {f"HIERARCH SED_REST_{property_name}": True}
            out_tab = join(SED_rest_property_tab, new_SED_rest_property_tab, keys_left = \
                self.cat_creator.ID_label, keys_right = f"{self.cat_creator.ID_label}_temp", join_type = "outer")
            out_tab.remove_column(f"{self.cat_creator.ID_label}_temp")
            out_tab.meta = {**SED_rest_property_tab.meta, **new_SED_rest_property_tab.meta}
        else:
            galfind_logger.info(f"{property_name} already calculated!")
            return
        if save_kwargs:
            ##breakpoint()
            property_PDFs = self.__getattr__(property_name, phot_type = "rest", property_type = "PDFs")
            kwarg_names = np.unique(np.hstack([list(property_PDF.kwargs.keys()) for property_PDF in property_PDFs if type(property_PDF) != type(None)]))
            kwargs = {kwarg_name: [property_PDF.kwargs[kwarg_name] if type(property_PDF) != type(None) \
                else np.nan for property_PDF in property_PDFs] for kwarg_name in kwarg_names}
            if "Halpha_cont_lines" in kwarg_names:
                breakpoint()
            for kwarg_name, kwarg_vals in kwargs.items():
                if kwarg_name not in out_tab.colnames:
                    out_tab[kwarg_name] = kwarg_vals
        fits_tab = self.open_cat(cropped = False)
        self.write_cat([fits_tab, out_tab], ["OBJECTS", SED_fit_params_label])
    
    def _save_SED_rest_PDFs(self, property_name, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
        save_dir = f"{config['PhotProperties']['PDF_SAVE_DIR']}/{self.version}/{self.instrument.name}/{self.survey}"
        funcs.make_dirs(f"{save_dir}/dummy_path.ecsv")
        [gal._save_SED_rest_PDFs(property_name, save_dir, SED_fit_params) for gal in self]

    def load_SED_rest_properties(self, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, timed = True):
        key = SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)
        # save the names of properties that have been calculated for this sample of galaxies in the catalogue
        SED_rest_properties_tab = self.open_cat(cropped = False, hdu = key)
        if type(SED_rest_properties_tab) != type(None):
            self.SED_rest_properties[key] = list(np.unique([label.replace("SED_REST_", "") \
                for label in SED_rest_properties_tab.meta.keys() if "SED_REST" == "_".join(label.split("_")[:2])]))
            # load SED rest properties that have previously been calculated
            PDF_dir = f"{config['PhotProperties']['PDF_SAVE_DIR']}/{self.version}/{self.instrument.name}/{self.survey}/{key}"
            self.gals = [deepcopy(gal)._load_SED_rest_properties(PDF_dir, self.SED_rest_properties[key], key) for gal in self]

    def del_SED_rest_property(self, property_name, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
        key = SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)
        # SED rest property must exist for this sample
        assert (property_name in self.SED_rest_properties[key])
        # delete data from fits
        del_col_names = [property_name, f"{property_name}_l1", f"{property_name}_u1"]
        del_hdr_names = [f"SED_REST_{property_name}"]
        self.del_cols_hdrs_from_fits(del_col_names, del_hdr_names, key)
        # check whether the SED rest property kwargs are included in the catalogue, and if so delete these as well - Not Implemented Yet!

        # remove data from self, starting with catalogue, then gal for gal in self.gals
        self.SED_rest_properties[key].remove(property_name)
        self.gals = [deepcopy(gal)._del_SED_rest_properties([property_name], key) for gal in self]


    def plot_SED_properties(self, x_name, y_name, SED_fit_params):
        x_arr = []
        y_arr = []
        for i, gal in enumerate(self):
            gal_properties = getattr(gal.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)], properties)
            if x_name in gal_properties and y_name in gal_properties:
                x_arr[i] = gal_properties(x_name)
                y_arr[i] = gal_properties(y_name)
            else:
                raise(Exception(f"{x_name} and {y_name} not available for all galaxies in this catalogue!"))

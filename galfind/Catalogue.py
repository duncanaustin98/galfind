#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:27:47 2023

@author: austind
"""

# Catalogue.py
import numpy as np
from astropy.table import Table, join
import pyregion
from copy import deepcopy
from astropy.io import fits
from pathlib import Path
import traceback
import h5py
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from tqdm import tqdm
import time
import copy
import os

from .Data import Data
from .Galaxy import Galaxy, Multiple_Galaxy
from . import useful_funcs_austind as funcs
from .Catalogue_Creator import GALFIND_Catalogue_Creator
from . import SED_code, LePhare, EAZY, Bagpipes
from . import config
from . import Catalogue_Base
from . import Photometry_rest
from . import galfind_logger
from .Instrument import NIRCam, MIRI, ACS_WFC, WFC3_IR, Instrument
from .Emission_lines import line_diagnostics

class Catalogue(Catalogue_Base):
    
    # %% alternative constructors
    @classmethod
    def from_pipeline(cls, survey, version, aper_diams, cat_creator, code_names, lowz_zmax, instruments = ['NIRCam', 'ACS_WFC', 'WFC3_IR'], \
                      forced_phot_band = "F444W", excl_bands = [], loc_depth_min_flux_pc_errs = [5, 10], templates_arr = ["fsps_larson"], select_by = None):
        # make 'Data' object
        data = Data.from_pipeline(survey, version, instruments, excl_bands = excl_bands)
        return cls.from_data(data, version, aper_diams, cat_creator, code_names, lowz_zmax, forced_phot_band, \
                loc_depth_min_flux_pc_errs, templates_arr, select_by = select_by)
    
    @classmethod
    def from_data(cls, data, version, aper_diams, cat_creator, code_names, lowz_zmax, forced_phot_band = "F444W", \
                loc_depth_min_flux_pc_errs = [10], templates_arr = ["fsps_larson"], mask = True, select_by = None):
        # make masked local depth catalogue from the 'Data' object
        data.combine_sex_cats(forced_phot_band)
        mode = str(config["Depths"]["MODE"]).lower() # mode to calculate depths (either "n_nearest" or "rolling")
        data.calc_depths(aper_diams, mode = mode, cat_creator = cat_creator)
        data.perform_aper_corrs()
        data.make_loc_depth_cat(cat_creator, depth_mode = mode)
        return cls.from_fits_cat(data.sex_cat_master_path, version, data.instrument, cat_creator, \
            code_names, data.survey, lowz_zmax, templates_arr = templates_arr, data = data, mask = mask, select_by = select_by)
    
    @classmethod
    def from_fits_cat(cls, fits_cat_path, version, instrument, cat_creator, code_names, survey, \
            lowz_zmax = [4., 6., None], templates_arr = ["fsps", "fsps_larson", "fsps_jades"], \
            data = None, mask = False, excl_bands = [], select_by = None):
        # open the catalogue
        fits_cat = funcs.cat_from_path(fits_cat_path)
        for band, band_name in zip(instrument, instrument.band_names):
            try:
                cat_creator.load_photometry(Table(fits_cat[0]), [band_name])
            except:
                # no data for the relevant band within the catalogue
                instrument.remove_band(band)
                print(f"{band_name} flux not loaded")
        print(f"instrument band names = {instrument.band_names}")
        codes = [getattr(globals()[name], name)() for name in code_names]
        # produce galaxy array from each row of the catalogue
        start_time = time.time()
        gals = Multiple_Galaxy.from_fits_cat(fits_cat, instrument, cat_creator, [], [], []).gals #codes, lowz_zmax, templates_arr).gals
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished loading in {len(gals)} galaxies. This took {elapsed_time:.6f} seconds")
        # make catalogue with no SED fitting information
        cat_obj = cls(gals, fits_cat_path, survey, cat_creator, instrument, code_names, version, crops = [])
        print(cat_obj)
        breakpoint()
        if cat_obj != None:
            cat_obj.data = data
        if mask:
            cat_obj.mask()
        # run SED fitting for the appropriate code names/low-z runs
        for i, (code, templates) in enumerate(zip(codes, templates_arr)):
            cat_obj = code.fit_cat(cat_obj, templates = templates, lowz_zmax_arr = lowz_zmax)
        # crop catalogue by selection column if it exists in the galaxy objects (currently selection should be same for every galaxy)
        if type(select_by) == str and select_by in cat_obj[0].selection_flags:
            cat_obj = cat_obj.crop(True, select_by)
            galfind_logger.info(f"Catalogue for {cat_obj.survey} {cat_obj.version} cropped by {select_by}")
        else:
            galfind_logger.info(f"Catalogue for {cat_obj.survey} {cat_obj.version} could not be cropped by {select_by}!")
        return cat_obj
    
    def update_SED_results(self, cat_SED_results):
        breakpoint()
        assert(len(cat_SED_results) == len(self)) # if this is not the case then instead should cross match IDs between self and gal_SED_result
        galfind_logger.info("Updating SED results in galfind catalogue object")
        [gal.update(gal_SED_result) for gal, gal_SED_result in zip(self, cat_SED_results)]
    
    # %%
    
    def catch_redshift_minus_99(self, gal_index, code, templates, out_value = True, condition = None, condition_fail_val = None, minus_99_out = None):
        #try:
            self[gal_index].phot.SED_results[code][templates].z # i.e. if the galaxy has a finite redshift
            if condition != None:
                return out_value if condition(out_value) else condition_fail_val
            else:
                return out_value
        #except:
            #print(f"{gal_index} is at z=-99")
            return minus_99_out
    
    def cap_ext_src_corrs(self, vals, condition = lambda x: x > 1., fail_val = 1.):
        return np.array([val if condition(val) else fail_val for val in vals])
    
    def calc_ext_src_corrs(self, band, ID = None):
        # load the catalogue (for obtaining the FLUX_AUTOs; THIS SHOULD BE MORE GENERAL IN FUTURE TO GET THESE FROM THE GALAXY OBJECT!!!)
        tab = self.open_full_cat()
        if ID != None:
            tab = tab[tab["NUMBER"] == ID]
            cat = self[self["ID"] == ID]
        else:
            cat = self
        # load the relevant FLUX_AUTO from SExtractor output
        flux_autos = funcs.flux_image_to_Jy(np.array(tab[f"FLUX_AUTO_{band}"]), self.data.im_zps[band])
        ext_src_corrs = self.cap_ext_src_corrs([flux_autos[i] / gal.phot.flux_Jy[np.where(band == \
            gal.phot.instrument.band_names)[0][0]].value for i, gal in enumerate(cat)])
        return ext_src_corrs
            
    def make_ext_src_corr_cat(self, code_name = "EAZY", templates_arr = ["fsps", "fsps_larson", "fsps_jades"], join_tables = True):
        ext_src_cat_name = f"{funcs.split_dir_name(self.cat_path, 'dir')}/Extended_source_corrections_{code_name}.fits"
        overwrite = config["DEFAULT"].getboolean("OVERWRITE")
        
        if overwrite:
            galfind_logger.info(f"OVERWRITE = YES, so overwriting coord_{band}.reg if it exists.")
      
        if not Path(ext_src_cat_name).is_file() or overwrite:
            if not config["DEFAULT"].getboolean("RUN"):
                galfind_logger.critical("RUN = YES, so not making ext correction cat. Returning Error.")
                raise Exception(f"RUN = YES, and combination of {self.survey} {self.version} or {self.instrument.name} has not previously been run.")

            ext_src_corrs_band = {}
            ext_src_bands = []
            for i, band in tqdm(enumerate(self.data.instrument.band_names), total = len(self.data.instrument.band_names), desc = f"Calculating extended source corrections for {self.cat_path}"):
                try:
                    band_corrs = self.calc_ext_src_corrs(band)
                    #print(band, band_corrs)
                    ext_src_corrs_band[band] = band_corrs
                    ext_src_bands.append(band)
                except:
                    print(f"No flux auto for {band}")
            print(ext_src_bands)
            ext_src_col_names = np.array(["ID"] + [f"auto_corr_factor_{name}" for name in [band for band in self.data.instrument.band_names if band in ext_src_corrs_band.keys()]] + \
                                         [f"auto_corr_factor_UV_{code_name}_{templates}" for templates in templates_arr] + ["auto_corr_factor_mass"])
            ext_src_col_dtypes = np.array([int] + [float for name in [band for band in self.data.instrument.band_names if band in ext_src_corrs_band.keys()]] + \
                                          [float for templates in templates_arr] + [float])
                    
            # determine the relevant bands for the extended source correction (slower than it could be, but it works nevertheless)
            UV_ext_src_corrs = []
            for templates in templates_arr:
                UV_corr_bands = []
                for i, gal in enumerate(self):
                    try:  # test whether the UV band already has existing 
                        ext_src_corrs_band[gal.phot.SED_results[code_name][templates].phot_rest.rest_UV_band]
                        UV_corr_bands.append(gal.phot.SED_results[code_name][templates].phot_rest.rest_UV_band)
                    except:
                        UV_corr_bands.append(None)
                # use bluest band with an extended src correction, since it is the blue HST data that doesn't have FLUX_AUTO's
                UV_ext_src_corrs.append(np.array([ext_src_corrs_band[band][j] if band != None else ext_src_corrs_band[ext_src_bands[0]][j] for j, band in enumerate(UV_corr_bands)]))
            print(UV_ext_src_corrs, np.array(UV_ext_src_corrs).shape)
            print(f"Finished calculating UV extended source corrections using {code_name} {templates_arr} redshifts")
            mass_ext_src_corrs = np.array(ext_src_corrs_band["f444W"]) # f444W band (mass tracer)
            print("Finished calculating mass extended source corrections")
            ext_src_corr_vals = np.vstack((np.array(self.ID), np.vstack(list(ext_src_corrs_band.values())), np.vstack(UV_ext_src_corrs), mass_ext_src_corrs)) #.T
            print(ext_src_corr_vals, ext_src_col_names, ext_src_col_dtypes, ext_src_corr_vals.shape, len(ext_src_col_names), len(ext_src_col_dtypes))
            ext_src_tab = Table(ext_src_corr_vals.T, names = ext_src_col_names, dtype = ext_src_col_dtypes)
            ext_src_tab.write(ext_src_cat_name, overwrite = True)
            self.ext_src_tab = ext_src_tab
            print(f"Writing table to {ext_src_cat_name}")
        else:
            self.ext_src_tab = Table.read(ext_src_cat_name, character_as_bytes = False)
            print(f"Opening table: {ext_src_cat_name}")
            
        if join_tables:
            self.join_ext_src_cat(code_name = code_name, templates_arr = templates_arr)
        return self
        
    def join_ext_src_cat(self, match_cols = ["NUMBER", "ID"], code_name = "EAZY", templates_arr = ["fsps", "fsps_larson", "fsps_jades"]):
        # open existing cat
        init_cat = self.open_full_cat()
        joined_tab = join(init_cat, self.ext_src_tab, keys_left = match_cols[0], keys_right = match_cols[1])
        self.cat_path = self.cat_path.replace(".fits", "_ext_src.fits")
        joined_tab.write(self.cat_path, format = "fits", overwrite = True)
        print(f"Joining ext_src table to catalogue! Saving to {self.cat_path}")
        
        # set relevant properties in the galaxies contained within the catalogues (can use __setattr__ here too!)
        print("Updating Catalogue object with extended source corrections")
        for i, (gal, mass_corr) in enumerate(zip(self, self.ext_src_tab["auto_corr_factor_mass"])):
            for templates in templates_arr:
                gal.phot.SED_results[code_name][templates].ext_src_corrs = {**{"UV": self.ext_src_tab[f"auto_corr_factor_UV_{code_name}_{templates}"][i] for templates in templates_arr}, **{"mass": mass_corr}}

    def mask(self): #, mask_instrument = NIRCam()):
        galfind_logger.info(f"Running masking code for {self.cat_path}.")
        # determine whether to overwrite catalogue or not
        overwrite = config["Masking"].getboolean("OVERWRITE_MASK_COLS")
        if overwrite:
            galfind_logger.info("OVERWRITE_MASK_COLS = YES, updating catalogue with masking columns.")
        # open catalogue with astropy
        cat = Table.read(self.cat_path)
        # update input catalogue if it hasnt already been masked or if wanted ONLY if len(self) == len(cat)
        if len(self) != len(cat):
            galfind_logger.warning(f"len(self) = {len(self)}, len(cat) = {len(cat)} -> len(self) != len(cat). Skipping masking for {self.survey} {self.version}!")
        elif (not "MASKED" in cat.meta.keys() or overwrite):
            galfind_logger.info(f"Masking catalogue for {self.survey} {self.version}")
            
            # load WCS from alignment image
            wcs = WCS(self.data.load_im(self.data.alignment_band)[1])
            # calculate x,y for each galaxy in catalogue
            cat_x, cat_y = wcs.world_to_pixel(SkyCoord(cat[self.cat_creator.ra_dec_labels["RA"]], cat[self.cat_creator.ra_dec_labels["DEC"]]))
            
            # make columns for individual band masking
            if config["Masking"].getboolean("MASK_BANDS"):
                for band in tqdm(self.instrument.band_names, desc = "Masking galfind catalogue object", total = len(self.instrument)):
                    # open .fits mask for band
                    mask = self.data.load_mask(band)
                    # load image wcs
                    wcs = WCS(self.data.load_im(band)[1])
                    # determine whether a galaxy is unmasked
                    unmasked_band = np.array([False if x < 0. or x >= mask.shape[1] or y < 0. or y >= mask.shape[0] else not bool(mask[int(y)][int(x)]) for x, y in zip(cat_x, cat_y)])
                    # update catalogue with new column
                    cat[f"unmasked_{band}"] = unmasked_band # assumes order of catalogue and galaxies in self is consistent
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
                    mask_data = [default_blank_bool for i in range(len(cat))] # default behaviour
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
                cat[mask_label] = mask_data # update catalogue with boolean column

            # update catalogue metadata
            cat.meta = {**cat.meta, **{"MASKED": True, "HIERARCH MASK_BANDS": config["Masking"].getboolean("MASK_BANDS"), \
                "HIERARCH MASK_CLUSTER_MODULE": config["Masking"].getboolean("MASK_CLUSTER_MODULE"), \
                "HIERARCH MASK_CLUSTER_CORE": config["Masking"].getboolean("MASK_CLUSTER_CORE")}}
            # save catalogue
            cat.write(self.cat_path, overwrite = True)
            # update catalogue README
            galfind_logger.warning("REQUIRED UPDATE: Update README for catalogue masking columns")
            # update masking of galfind galaxy objects
            galfind_logger.info("Masking galfind galaxy objects in catalogue")
            assert(len(cat) == len(self))
            [gal.update_mask(cat, self.cat_creator, update_phot_rest = False) for gal in \
                tqdm(self, total = len(self), desc = "Masking galfind galaxy objects")]
        else:
            galfind_logger.info(f"Catalogue for {self.survey} {self.version} already masked. Skipping!")

    def make_cutouts(self, IDs, cutout_size = 32):
        for band in tqdm(self.instrument.band_names, total = len(self.instrument), desc = "Making band cutouts"):
            im_data, im_header, seg_data, seg_header = self.data.load_data(band, incl_mask = False)
            wht_data = self.data.load_wht(band)
            wcs = WCS(im_header)
            for gal in self:
                if gal.ID in IDs:
                    gal.make_cutout(band, data = {"SCI": im_data, "SEG": seg_data, self.data.wht_types[band]: wht_data}, \
                        wcs = wcs, im_header = im_header, survey = self.survey, version = self.version, cutout_size = cutout_size)

    def make_UV_fit_cat(self, code_name = "EAZY", templates = "fsps_larson", UV_PDF_path = config["RestUVProperties"]["UV_PDF_PATH"], col_names = ["Beta", "flux_lambda_1500", "flux_Jy_1500", "M_UV", "A_UV", "L_obs", "L_int", "SFR"], \
                        join_tables = True, skip_IDs = [], rest_UV_wavs_arr = [[1250., 3000.] * u.AA], conv_filt_arr = [True, False], overwrite = True):
        UV_cat_name = f"{funcs.split_dir_name(self.cat_path, 'dir')}/UV_properties_{code_name}_{templates}_{str(self.cat_creator.min_flux_pc_err)}pc.fits" # _test
        if not Path(UV_cat_name).is_file() or overwrite:
            if not config["DEFAULT"].getboolean("RUN"):
                galfind_logger.critical("RUN = YES, so not making UV corrected cat. Returning Error.")
                raise Exception(f"RUN = YES, and combination of {self.survey} {self.version} or {self.instrument.name} has not previously been run.")

            cat_data = []
            #print("Bands here: ", self[1].phot.instrument.band_names)
            for i, gal in tqdm(enumerate(self), total = len(self), desc = "Making UV fit catalogue"):
                n_bands = []
                gal_copy = gal #copy.deepcopy(gal)
                gal_data = np.array([gal_copy.ID])
                conv_filt_names = {True: "conv_filt_PL", False: "pure_PL"}
                for conv_filt in conv_filt_arr:
                    conv_filt_name = conv_filt_names[conv_filt]
                    for rest_UV_wavs in rest_UV_wavs_arr:
                        UV_PDF_path_loc = f"{UV_PDF_path}/{Photometry_rest.rest_UV_wavs_name(rest_UV_wavs)}/{conv_filt_name}"
                        os.makedirs(UV_PDF_path_loc, exist_ok = True) # not sure if this is done later or not
                        if gal.ID in skip_IDs:
                            for name in col_names:
                                gal_data = np.append(gal_data, funcs.percentiles_from_PDF([-99.]))
                            # append n_bands
                            n_bands.append(-99.)
                        else:
                            # path = f"{config['DEFAULT']['GALFIND_WORK']}/UV_PDFs/{self.data.version}/{self.data.instrument.name}/{self.survey}/{code_name}+{str(self.cat_creator.min_flux_pc_err)}pc/{templates}/Amplitude/{gal_copy.ID}.txt"
                            # #print(path)
                            # if not Path(path).is_file():
                            #print(gal.phot_obs.instrument.band_names)
                            for name in ["Amplitude", "Beta"]:
                                if name == "Beta":
                                    plot = True
                                else:
                                    plot = False
                                #raise(Exception("Failing try/except here!"))
                                try:
                                    #gal.phot.SED_results[code_name][templates].phot_rest[Photometry_rest.rest_UV_wavs_name(rest_UV_wavs)]
                                    gal.phot.SED_results[code_name][templates].phot_rest[Photometry_rest.rest_UV_wavs_name(rest_UV_wavs)] \
                                        .open_UV_fit_PDF(UV_PDF_path_loc, name, gal_copy.ID, UV_ext_src_corr = 1., conv_filt = conv_filt, plot = plot)
                                        # UV_ext_src_corr = gal_copy.phot.SED_results[code_name][templates].ext_src_corrs["UV"]
                                except Exception as e:
                                    print(f"Fitting not performed for {gal.ID}. Error code: {e}")
                                    print(traceback.format_exc())
                                    break
                            for name in col_names:
                                #print(f"{gal.ID}: {gal.phot_rest.phot_obs.instrument.band_names}")
                                try:
                                    gal_data = np.append(gal_data, funcs.percentiles_from_PDF(gal.phot.SED_results[code_name][templates].\
                                            phot_rest[Photometry_rest.rest_UV_wavs_name(rest_UV_wavs)].open_UV_fit_PDF(UV_PDF_path_loc, name, gal_copy.ID, 1., conv_filt = conv_filt, plot = plot)))
                                    # gal_copy.phot.SED_results[code_name][templates].ext_src_corrs["UV"]
                                except Exception as e:
                                    print(f"EXCEPT ID = {gal.ID}. Error code: {e}")
                                    print(traceback.format_exc())
                                    gal_data = np.append(gal_data, funcs.percentiles_from_PDF([-99.]))
                            try:
                                if not hasattr(gal.phot.SED_results[code_name][templates].phot_rest[Photometry_rest.rest_UV_wavs_name(rest_UV_wavs)], "rest_UV_phot"):
                                    gal.phot.SED_results[code_name][templates].phot_rest[Photometry_rest.rest_UV_wavs_name(rest_UV_wavs)].make_rest_UV_phot()
                                n_bands.append(int(len(gal.phot.SED_results[code_name][templates].phot_rest[Photometry_rest.rest_UV_wavs_name(rest_UV_wavs)].rest_UV_phot.flux_Jy)))
                            except Exception as e:
                                print(f"EXCEPT ID = {gal.ID}. n_bands=0 error. Error code: {e}")
                                print(traceback.format_exc())
                                n_bands.append(0.)
                        
                gal_data = np.concatenate((np.array(gal_data).flatten(), np.array(n_bands)))
                if i == 0: # if the first column
                    cat_data = gal_data
                else:
                    cat_data = np.vstack([cat_data, gal_data])
    
                UV_col_names = np.array([[f"{name}_{Photometry_rest.rest_UV_wavs_name(rest_UV_wavs.value)}_{conv_filt_names[conv_filt]}", f"{name}_l1_{Photometry_rest.rest_UV_wavs_name(rest_UV_wavs.value)}_{conv_filt_names[conv_filt]}", \
                                          f"{name}_u1_{Photometry_rest.rest_UV_wavs_name(rest_UV_wavs.value)}_{conv_filt_names[conv_filt]}"] for conv_filt in conv_filt_arr for rest_UV_wavs in rest_UV_wavs_arr for name in col_names]).flatten()
                #print(UV_col_names)
                fits_col_names = np.concatenate((np.array(["ID"]), UV_col_names, np.array([f"n_bands_{Photometry_rest.rest_UV_wavs_name(rest_UV_wavs.value)}_{conv_filt_names[conv_filt]}" for conv_filt in conv_filt_arr for rest_UV_wavs in rest_UV_wavs_arr])))
                funcs.make_dirs(self.cat_path)
                UV_tab = Table(cat_data, names = fits_col_names)
                UV_tab.write(UV_cat_name, format = "fits", overwrite = True)
                self.UV_tab = UV_tab
                # print(f"Writing UV table to {UV_cat_name}")
        else:
            self.UV_tab = Table.read(UV_cat_name, character_as_bytes = False)
            print(f"Opening table: {UV_cat_name}")
        
        if join_tables:
            self.join_UV_fit_cat()
            # set relevant properties in the galaxies contained within the catalogues
            #[setattr(gal, ["properties", name], UV_tab[name][i]) for i, gal in enumerate(self) for name in UV_col_names]
            
        return self
        
    def join_UV_fit_cat(self, match_cols = ["NUMBER", "ID"]):
        # open existing cat
        init_cat = self.open_full_cat()
        joined_tab = join(init_cat, self.UV_tab, keys_left = match_cols[0], keys_right = match_cols[1])
        self.cat_path = self.cat_path.replace('.fits', '_UV.fits')
        joined_tab.write(self.cat_path, format = "fits", overwrite = True)
        print(f"Joining UV table to catalogue! Saving to {self.cat_path}")

    # Selection functions
        
    # Masking selection

    def select_min_unmasked_bands(self, min_bands):
        return self.perform_selection(Galaxy.select_min_unmasked_bands, min_bands)
    
    def select_unmasked_bands(self, bands):
        return self.perform_selection(Galaxy.select_unmasked_band, bands)
    
    def select_unmasked_instrument(self, instrument_name):
        return self.perform_selection(Galaxy.select_unmasked_instrument, instrument_name)

    # SNR selection functions

    def phot_bluewards_Lya_non_detect(self, SNR_lim, code_name = "EAZY", templates = "fsps_larson", lowz_zmax = None):
        return self.perform_selection(Galaxy.phot_bluewards_Lya_non_detect, SNR_lim, code_name, templates, lowz_zmax)

    def phot_redwards_Lya_detect(self, SNR_lims, code_name = "EAZY", templates = "fsps_larson", lowz_zmax = None, widebands_only = True):
        return self.perform_selection(Galaxy.phot_redwards_Lya_detect, SNR_lims, code_name, templates, lowz_zmax, widebands_only)

    def phot_Lya_band(self, SNR_lim, detect_or_non_detect = "detect", code_name = "EAZY", \
            templates = "fsps_larson", lowz_zmax = None, widebands_only = True):
        return self.perform_selection(Galaxy.phot_Lya_band, SNR_lim, detect_or_non_detect, \
            code_name, templates, lowz_zmax, widebands_only)

    def phot_SNR_crop(self, band_name_or_index, SNR_lim):
        return self.perform_selection(Galaxy.phot_SNR_crop, band_name_or_index, SNR_lim)
    
    # Depth rregion selection

    def select_depth_region(self, band, region_ID, update = True):
        return NotImplementedError
    
    # Chi squared selection functions

    def select_chi_sq_lim(self, chi_sq_lim, code_name = "EAZY", templates = "fsps_larson", lowz_zmax = None, reduced = True):
        return self.perform_selection(Galaxy.select_chi_sq_lim, chi_sq_lim, code_name, templates, lowz_zmax, reduced)

    def select_chi_sq_diff(self, chi_sq_diff, code_name = "EAZY", templates = "fsps_larson", delta_z_lowz = 0.5):
        return self.perform_selection(Galaxy.select_chi_sq_diff, chi_sq_diff, code_name, templates, delta_z_lowz)

    # Redshift PDF selection functions

    def select_robust_zPDF(self, integral_lim, delta_z, code_name = "EAZY", templates = "fsps_larson", lowz_zmax = None):
        # open h5 z-PDF file
        zPDF_h5 = h5py.File(f".h5", "r")
        cat = self.perform_selection(integral_lim, delta_z, zPDF_h5, code_name, templates, lowz_zmax)
        zPDF_h5.close()
        return cat
    
    # Cutout quality selection functions

    def flag_hot_pixel(self):
        pass

    # Full sample selection functions - these chain the above functions

    def select_EPOCHS(self, code_name = "EAZY", templates = "fsps_larson", lowz_zmax = None):
        return self.perform_selection(Galaxy.select_EPOCHS, code_name, templates, lowz_zmax)

    def perform_selection(self, selection_function, *args):
        # extract selection name from galaxy method output
        selection_name = selection_function(self[0], *args, update = False)[1]
        # perform selection if not previously performed
        if selection_name not in self.crops:
            # perform calculation for each galaxy and update galaxies in self
            [selection_function(gal, *args, update = True)[0] for gal in \
                tqdm(self, total = len(self), desc = f"Cropping {selection_name}")]
        # crop catalogue by the selection
        cat_copy = self._crop_by_selection(selection_name)
        # append .fits table if not already done so
        cat_copy._append_selection_to_fits(selection_name)
        return cat_copy

    def _crop_by_selection(self, selection_name):
        # make a deep copy of the current catalogue object
        cat_copy = deepcopy(self)
        # crop deep copied catalogue to only the selected galaxies
        keep = getattr(self, selection_name)
        cat_copy.gals = cat_copy[keep]
        if selection_name not in cat_copy.crops:
            # make a note of this crop if it is new
            cat_copy.crops.append(selection_name)
        return cat_copy

    def _append_selection_to_fits(self, selection_name):
        full_cat = self.open_cat()
        # append .fits table if not already done so for this selection
        if not selection_name in full_cat.colnames:
            assert(all(getattr(self, selection_name) == True))
            selection_cat = Table({"ID_temp": getattr(self, "ID"), selection_name: np.full(len(self), True)})
            output_cat = join(full_cat, selection_cat, keys_left = "NUMBER", keys_right = "ID_temp", join_type = "outer")
            output_cat.remove_column("ID_temp")
            # fill unselected columns with False rather than leaving as masked post-join
            output_cat[selection_name].fill_value = False
            output_cat = output_cat.filled()
            # ensure no rows are lost during this column append
            assert(len(output_cat) == len(full_cat))
            output_cat.meta = {**full_cat.meta, **{f"HIERARCH SELECTED_{selection_name}": True}}
            galfind_logger.info(f"Appending {selection_name} to catalogue = {self.cat_path}")
            output_cat.write(self.cat_path, overwrite = True)
        else:
            galfind_logger.info(f"Already appended {selection_name} to catalogue = {self.cat_path}")
    
    def plot_SED_properties(self, x_name, y_name, code_name):
        x_arr = []
        y_arr = []
        for i, gal in enumerate(self):
            gal_properties = getattr(gal.phot.SED_results[code_name], properties)
            if x_name in gal_properties and y_name in gal_properties:
                x_arr[i] = gal_properties(x_name)
                y_arr[i] = gal_properties(y_name)
            else:
                raise(Exception(f"{x_name} and {y_name} not available for all galaxies in this catalogue!"))

    # def fit_sed(self, code):
    #     return code.fit_cat(self)

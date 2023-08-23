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
from astropy.io import fits
from pathlib import Path
from astropy.wcs import WCS
import astropy.units as u
from tqdm import tqdm
import time
import copy

from .Data import Data
from .Galaxy import Galaxy, Multiple_Galaxy
from . import useful_funcs_austind as funcs
from .Catalogue_Creator import GALFIND_Catalogue_Creator
from . import SED_code, LePhare, EAZY, Bagpipes
from . import config
from . import Catalogue_Base
from .Instrument import NIRCam, ACS_WFC, WFC3IR, Instrument, Combined_Instrument

class Catalogue(Catalogue_Base):
    
    # %% alternative constructors
    @classmethod
    def from_pipeline(cls, survey, version, aper_diams, cat_creator, code_names, lowz_zmax, xy_offset = [0, 0], instruments = ['NIRCam', 'ACS_WFC', 'WFC3IR'], \
                      forced_phot_band = "f444W", excl_bands = [], loc_depth_min_flux_pc_errs = [5, 10], n_loc_depth_samples = 5, templates_arr = ["fsps_larson"], fast = True):
        # make 'Data' object
        data = Data.from_pipeline(survey, version, instruments, excl_bands = excl_bands)
        return cls.from_data(data, version, aper_diams, cat_creator, code_names, lowz_zmax, xy_offset, forced_phot_band, loc_depth_min_flux_pc_errs, n_loc_depth_samples, templates_arr, fast)

    # @classmethod
    # def from_NIRCam_pipeline(cls, survey, version, aper_diams, cat_creator, xy_offset = [0, 0], forced_phot_band = "f444W", \
    #                          excl_bands = [], loc_depth_min_flux_pc_errs = [5, 10], n_loc_depth_samples = 5, fast = True):
    #     # make 'Data' object
    #     data = Data.from_NIRCam_pipeline(survey, version, excl_bands = excl_bands)
    #     return cls.from_data(data, aper_diams, cat_creator, xy_offset, forced_phot_band, loc_depth_min_flux_pc_errs, n_loc_depth_samples, fast)
    
    @classmethod
    def from_data(cls, data, version, aper_diams, cat_creator, code_names, lowz_zmax, xy_offset = [0, 0], forced_phot_band = "f444W", loc_depth_min_flux_pc_errs = [5, 10], \
                  n_loc_depth_samples = 5, templates_arr = ["fsps_larson"], fast = True, mask = True):
        # make masked local depth catalogue from the 'Data' object
        data.combine_sex_cats(forced_phot_band)
        data.calc_depths(xy_offset, aper_diams, fast = fast)
        print("from_data fast = ", fast)
        # load the catalogue that has just been created into a 'Catalogue' object
        if cat_creator.cat_type == "loc_depth":
            data.make_loc_depth_cat(aper_diams, min_flux_pc_err_arr = loc_depth_min_flux_pc_errs, forced_phot_band = forced_phot_band, n_samples = n_loc_depth_samples, fast = fast)
            cat_path = data.loc_depth_cat_path
        elif cat_creator.cat_type == "sex":
            cat_path = data.sex_cat_master_path
        return cls.from_fits_cat(cat_path, version, data.instrument, cat_creator, code_names, data.survey, lowz_zmax, templates_arr = templates_arr, data = data, mask = mask)
    
    # @classmethod
    # def from_sex_cat(cls, cat_path, instrument, survey, cat_creator):
    #     # open the catalogue
    #     cat = funcs.cat_from_path(cat_path)
    #     # produce galaxy array from each row of the catalogue
    #     gals = np.array([Galaxy.from_sex_cat_row(row, instrument, cat_creator) for row in cat])
    #     return cls(gals, cat_path, survey, cat_creator)
    
    @classmethod
    def from_fits_cat(cls, fits_cat_path, version, instrument, cat_creator, code_names, survey, lowz_zmax, templates_arr = ["fsps", "fsps_larson", "fsps_jades"], data = None, mask = True, excl_bands = []):
        # open the catalogue
        fits_cat = funcs.cat_from_path(fits_cat_path)
        if type(instrument) not in [Instrument, NIRCam, ACS_WFC, WFC3IR, Combined_Instrument]:
            instrument_name = instrument
            if type(instrument) in [list, np.ndarray]:
                instrument_name = '+'.join(instrument)
            instrument = Instrument.from_name(instrument_name, excl_bands = excl_bands)
        print("instrument bands = ", instrument.bands)
        # crop instrument bands that don't appear in the first row of the catalogue (I believe this is already done when running from data)
        # Removed comments from following
        for band in instrument.bands:
             try:
                 cat_creator.load_photometry(Table(fits_cat[0]), [band])
             except:
                 # no data for the relevant band within the catalogue
                 instrument.remove_band(band)
                 print(f"{band} flux not loaded")
        print("instrument bands = ", instrument.bands)
        codes = [getattr(globals()[name], name)() for name in code_names]
        # produce galaxy array from each row of the catalogue
        start_time = time.time()
        gals = Multiple_Galaxy.from_fits_cat(fits_cat, instrument, cat_creator, [], [], []).gals #codes, lowz_zmax, templates_arr).gals
        print(gals[0].phot.SED_results)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished loading in {len(gals)} galaxies. This took {elapsed_time:.6f} seconds")
        # make catalogue with no SED fitting information
        cat_obj = cls(gals, fits_cat_path, survey, cat_creator, instrument, code_names, version)
        if cat_obj != None:
            cat_obj.data = data
        if mask:
            cat_obj.mask(data)
        # run SED fitting for the appropriate code names/low-z runs
        for code, templates in zip(codes, templates_arr):
            cat_obj = code.fit_cat(cat_obj, lowz_zmax, templates = templates)
        return cat_obj
    
    def update_SED_results(self, cat_SED_results):
        assert(len(cat_SED_results) == len(self))
        print("Updating SED results in galfind catalogue object")
        [gal.update(gal_SED_result) for gal, gal_SED_result in zip(self, cat_SED_results)]
    
    # %% Overloaded operators
    
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
    
    def __getattr__(self, name): # only acts on attributes that don't already exist
        # get array of galaxy properties for the catalogue if they exist in all galaxies
        for gal in self:
            # property must exist in all galaxies within class
            if not hasattr(gal, name):
                raise AttributeError(f"'{name}' does not exist in all galaxies within {self.cat_name} !!!")
        return np.array([getattr(gal, name) for gal in self])
    
    def __setattr__(self, name, value, obj = "cat"):
        if obj == "cat":
            super().__setattr__(name, value)
        elif obj == "gal":
            # set attributes of individual galaxies within the catalogue
            for i, gal in enumerate(self):
                if type(value) == list or type(value) == np.array:
                    setattr(gal, name, value[i])
                else:
                    setattr(gal, name, value)
    
    # not needed!
    def __setitem__(self, index, gal):
        self.gals[index] = gal
    
    def __add__(self, cat):
        # concat catalogues
        pass
    
    def __mul__(self, cat): # self * cat
        # cross-match catalogues
        pass
    
    def __repr__(self):
        return str(self.__dict__)
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, copy.deepcopy(value, memo))
        return result
    
    # %%
    
    def open_full_cat(self):
        return Table.read(self.cat_path, character_as_bytes = False)
    
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
                                            gal.phot.instrument.bands)[0][0]].value for i, gal in enumerate(cat)])
        return ext_src_corrs
            
    def make_ext_src_corr_cat(self, code_name = "EAZY", templates_arr = ["fsps", "fsps_larson", "fsps_jades"], join_tables = True):
        ext_src_cat_name = f"{funcs.split_dir_name(self.cat_path, 'dir')}/Extended_source_corrections_{code_name}.fits"
        if not Path(ext_src_cat_name).is_file():
            ext_src_corrs_band = {}
            ext_src_bands = []
            for i, band in tqdm(enumerate(self.data.instrument.bands), total = len(self.data.instrument.bands), desc = f"Calculating extended source corrections for {self.cat_path}"):
                try:
                    band_corrs = self.calc_ext_src_corrs(band)
                    #print(band, band_corrs)
                    ext_src_corrs_band[band] = band_corrs
                    ext_src_bands.append(band)
                except:
                    print(f"No flux auto for {band}")
            print(ext_src_bands)
            ext_src_col_names = np.array(["ID"] + [f"auto_corr_factor_{name}" for name in [band for band in self.data.instrument.bands if band in ext_src_corrs_band.keys()]] + \
                                         [f"auto_corr_factor_UV_{code_name}_{templates}" for templates in templates_arr] + ["auto_corr_factor_mass"])
            ext_src_col_dtypes = np.array([int] + [float for name in [band for band in self.data.instrument.bands if band in ext_src_corrs_band.keys()]] + \
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
    
    # altered from original in mask_regions.py
    def mask(self, data, mask_instrument = NIRCam()): # mask paths is a dict of form {band: mask_path}
        print(f"Running masking code for {self.cat_path}. (Too much copying and pasting here!)")
        self.data = data # store data object in catalogue object
        masked_cat_path = self.cat_path.replace(".fits", "_masked.fits")
        
        if not Path(masked_cat_path).is_file():
            im_data, im_header, seg_data, seg_header, mask = self.data.load_data("f444W", incl_mask = True)
            wcs = WCS(im_header)
            if self.data.is_blank:
                # add 'blank_module == True' to every galaxy in the catalogue
                blank_flags = [True] * len(self.gals)
                for gal in self:
                    # changed syntax from "blank" to "blank_module"
                    gal.mask_flags["blank_module"] = True
            else: # mask cluster/blank field in reddest band (f444W for our NIRCam fields)
                blank_flags = []
                # add 'blank_module == True' to galaxies in the blank module
                if self.data.blank_mask_path != "":
                    blank_mask_file = pyregion.open(self.data.blank_mask_path).as_imagecoord(im_header) # file for mask
                    blank_mask = blank_mask_file.get_mask(hdu = fits.open(self.data.im_paths[self.data.instrument.bands[-1]])[self.data.im_exts[self.data.instrument.bands[-1]]])
                    for gal in self:
                        pix_values = wcs.world_to_pixel(gal.sky_coord)
                        x_pix = int(np.rint(pix_values[0]))
                        y_pix = int(np.rint(pix_values[1]))
                        blank_flag = blank_mask[y_pix][x_pix]
                        #print(mask_flag_gal)
                        if y_pix >= mask.shape[0] or x_pix >= mask.shape[1] or x_pix < 0 or y_pix < 0: # catch HST masking errors
                            mask_flag_gal = True
                        else:
                            mask_flag_gal = mask[y_pix][x_pix]
                        if mask_flag_gal == True:
                            gal.mask_flags["blank_module"] = False
                            blank_flags.append(False)
                        else:
                            gal.mask_flags["blank_module"] = True
                            blank_flags.append(True)
                    # update saved catalogue
                    cat = self.open_full_cat()
                    cat["blank_module"] = blank_flags #.astype(bool)
                    cat.write(masked_cat_path, overwrite = True)
                    self.cat_path = masked_cat_path
                    print("Finished masking blank field")
                else:
                    raise(Exception("Must manually create a 'blank' field mask in the 'Data' object if 'flag_blank_field == True'!"))
                # add 'cluster == True' to every galaxy within the cluster
                cluster_flags = []
                if self.data.cluster_mask_path != "":
                    cluster_mask_file = pyregion.open(self.data.cluster_mask_path).as_imagecoord(im_header) # file for mask
                    cluster_mask = cluster_mask_file.get_mask(hdu = fits.open(self.data.im_paths[self.data.instrument.bands[-1]])[self.data.im_exts[self.data.instrument.bands[-1]]])
                    for gal in self:
                        pix_values = wcs.world_to_pixel(gal.sky_coord)
                        x_pix = int(np.rint(pix_values[0]))
                        y_pix = int(np.rint(pix_values[1]))
                        cluster_flag = cluster_mask[y_pix][x_pix]
                        #print(mask_flag_gal)
                        gal.mask_flags["cluster"] = cluster_flag
                        cluster_flags.append(cluster_flag)
                    # update saved catalogue
                    cat = self.open_full_cat()
                    cat["cluster"] = cluster_flags #.astype(bool)
                    cat.write(masked_cat_path, overwrite = True)
                    self.cat_path = masked_cat_path
                    print("Finished masking cluster")
                else:
                    raise(Exception("Must manually create a 'cluster' mask in the 'Data' object!"))
                    
            # mask each band individually
            for i, (band, mask_path) in enumerate(self.data.mask_paths.items()):
                im_data, im_header, seg_data, seg_header, mask = self.data.load_data(band, incl_mask = True)
                wcs = WCS(im_header)
                
                # make a flag array to say whether each object lies within the mask or not
                mask_flag_arr = []
                for j, gal in enumerate(self.gals):
                    pix_values = wcs.world_to_pixel(gal.sky_coord)
                    x_pix = int(np.rint(pix_values[0]))
                    y_pix = int(np.rint(pix_values[1]))
                    # if j == 0:
                    #     print(mask.shape, im_data.shape, seg_data.shape)
                    if y_pix >= mask.shape[0] or x_pix >= mask.shape[1] or x_pix < 0 or y_pix < 0: # catch HST masking errors
                        mask_flag_gal = True
                    else:
                        mask_flag_gal = mask[y_pix][x_pix]
                    #print(mask_flag_gal)
                    if mask_flag_gal == True:
                        gal.mask_flags[f"unmasked_{band}"] = False
                        mask_flag_arr.append(False)
                    else:
                        gal.mask_flags[f"unmasked_{band}"] = True
                        mask_flag_arr.append(True)
                # update saved catalogue
                cat = self.open_full_cat()
                cat[f"unmasked_{band}"] = mask_flag_arr #.astype(bool)
                cat.write(masked_cat_path, overwrite = True)
                print(f"Finished masking {band}")
                self.cat_path = masked_cat_path

            # add additional boolean column to say whether an object is unmasked in all columns or not
            unmasked_blank = []
            for i, gal in enumerate(self):
                good_galaxy = True
                for band in self.data.instrument.bands:
                    if not gal.mask_flags[f"unmasked_{band}"] and band in mask_instrument.bands:
                        good_galaxy = False
                        break
                # don't include blank field galaxies in final boolean unmasked column
                if not self.data.is_blank:
                    if not gal.mask_flags["blank_module"]:
                        good_galaxy = False
                unmasked_blank.append(good_galaxy)
                gal.mask_flags[f"unmasked_blank_{mask_instrument.name}"] = good_galaxy
            cat = self.open_full_cat()
            cat[f"unmasked_blank_{mask_instrument.name}"] = unmasked_blank
            cat.write(masked_cat_path, overwrite = True)
            self.cat_path = masked_cat_path
            print("Finished masking!")
        else:
            self.cat_path = masked_cat_path
            print("Already masked!")
    
    def make_UV_fit_cat(self, code_name = "EAZY", templates = "fsps_larson", UV_PDF_path = config["RestUVProperties"]["UV_PDF_PATH"], col_names = ["Beta", "flux_lambda_1500", "flux_Jy_1500", "M_UV", "A_UV", "L_obs", "L_int", "SFR"], \
                        join_tables = True, skip_IDs = []):
        UV_cat_name = f"{funcs.split_dir_name(self.cat_path, 'dir')}/UV_properties_{code_name}_{templates}_{str(self.cat_creator.min_flux_pc_err)}pc.fits"
        if not Path(UV_cat_name).is_file():
            cat_data = []
            #print("Bands here: ", self[1].phot.instrument.bands)
            for i, gal in tqdm(enumerate(self), total = len(self), desc = "Making UV fit catalogue"):
                gal_copy = gal #copy.deepcopy(gal)
                gal_data = np.array([gal_copy.ID])
                if gal.ID in skip_IDs:
                    for name in col_names:
                        gal_data = np.append(gal_data, funcs.percentiles_from_PDF([-99.]))
                else:
                    path = f"{config['DEFAULT']['GALFIND_WORK']}/UV_PDFs/{self.data.version}/{self.data.instrument.name}/{self.survey}/{code_name}+{str(self.cat_creator.min_flux_pc_err)}pc/{templates}/Amplitude/{gal_copy.ID}.txt"
                    #print(path)
                    if not Path(path).is_file():
                        #print(gal.phot_obs.instrument.bands)
                        for name in ["Amplitude", "Beta"]:
                            if name == "Beta":
                                plot = True
                            else:
                                plot = False
                            funcs.percentiles_from_PDF(gal.phot.SED_results[code_name][templates].phot_rest.open_UV_fit_PDF(UV_PDF_path, name, gal_copy.ID, gal_copy.phot.SED_results[code_name][templates].ext_src_corrs["UV"], plot = plot))
                    for name in col_names:
                        #print(f"{gal.ID}: {gal.phot_rest.phot_obs.instrument.bands}")
                        #try:
                            gal_data = np.append(gal_data, funcs.percentiles_from_PDF(gal.phot.SED_results[code_name][templates].phot_rest.open_UV_fit_PDF(UV_PDF_path, name, gal_copy.ID, gal_copy.phot.SED_results[code_name][templates].ext_src_corrs["UV"]))) # not currently saving to object
                        #except:
                        #    print(f"EXCEPT ID = {gal.ID}")
                        #    gal_data = np.append(gal_data, funcs.percentiles_from_PDF([-99.]))
                gal_data = np.array(gal_data).flatten()
                if i == 0: # if the first column
                    cat_data = gal_data
                else:
                    cat_data = np.vstack([cat_data, gal_data])
                UV_col_names = np.array([[name, f"{name}_l1", f"{name}_u1"] for name in col_names]).flatten()
                fits_col_names = np.concatenate((np.array(["ID"]), UV_col_names))
                funcs.make_dirs(self.cat_path)
                UV_tab = Table(cat_data, names = fits_col_names)
                UV_tab.write(UV_cat_name, format = "fits", overwrite = True)
                self.UV_tab = UV_tab
                print(f"Writing UV table to {UV_cat_name}")
            
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
    
    def phot_SNR_crop(self, band, n_sigma, remove = False, flag = True):
        pass
    
    def flag_robust_high_z(self, relaxed = False):
        # could make use of an overloaded __setattr__ here!
        pass
    
    def flag_good_high_z(self, relaxed = False):
        # could make use of an overloaded __setattr__ here!
        pass
    
    def flag_hot_pixel(self):
        pass
    
    def crop(self, crop_property, crop_limits): # upper and lower limits on galaxy properties (e.g. ID, redshift, mass, SFR, SkyCoord)
        print("'Catalogue.crop_cat()' currently only works for 'ID' (and in theory any gal.property coming from SExtractor catalogue except astropy.coordinates.SkyCoord)." + \
              "Implementation for cropping by SED fitting property still not yet included !")
        if isinstance(crop_limits, int) or isinstance(crop_limits, float):
            self.gals = self[getattr(self, crop_property) == crop_limits]
        elif isinstance(crop_limits, list) or isinstance(crop_limits, np.array):
            upper_gals = self[getattr(self, crop_property) >= crop_limits[0]]
            self.gals = upper_gals
            cropped_gals = self[getattr(self, crop_property) <= crop_limits[1]]
            self.gals = cropped_gals
        else:
            raise(Exception(f"'crop_limits'={crop_limits} in 'Catalogue.crop_cat()' is inappropriate !"))
        return self
    
    # def fit_sed(self, code):
    #     return code.fit_cat(self)

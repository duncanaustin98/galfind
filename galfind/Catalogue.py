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

from .Data import Data
from .useful_funcs_austind import Galaxy
from . import useful_funcs_austind as funcs
from .Catalogue_Creator import GALFIND_Catalogue_Creator
from . import SED_code

class Catalogue:
    # later on, the gal_arr should be calculated from the Instrument and sex_cat path, with SED codes already given
    def __init__(self, gals, cat_path, survey, cat_creator, codes = []): #, UV_PDF_path):
        self.survey = survey
        self.cat_path = cat_path
        #self.UV_PDF_path = UV_PDF_path
        self.cat_creator = cat_creator
        self.codes = codes
        self.gals = gals
        
        # concat is commutative for catalogues
        self.__radd__ = self.__add__
        # cross-match is commutative for catalogues
        self.__rmul__ = self.__mul__
        
    @property
    def cat_dir(self):
        return funcs.split_dir_name(self.cat_path, "dir")
    
    @property
    def cat_name(self):
        return funcs.split_dir_name(self.cat_path, "name")
        
    # %% alternative constructors
    @classmethod
    def from_pipeline(cls, survey, version, aper_diams, cat_creator, xy_offset = [0, 0], instruments = ['NIRCam', 'ACS_WFC', 'WFC3IR'], \
                      forced_phot_band = "f444W", excl_bands = [], loc_depth_min_flux_pc_errs = [5, 10], n_loc_depth_samples = 5, fast = True):
        # make 'Data' object
        data = Data.from_pipeline(survey, version, instruments, excl_bands = excl_bands)
        return cls.from_data(data, aper_diams, cat_creator, xy_offset, forced_phot_band, loc_depth_min_flux_pc_errs, n_loc_depth_samples, fast)

    @classmethod
    def from_NIRCam_pipeline(cls, survey, version, aper_diams, cat_creator, xy_offset = [0, 0], forced_phot_band = "f444W", \
                             excl_bands = [], loc_depth_min_flux_pc_errs = [5, 10], n_loc_depth_samples = 5, fast = True):
        # make 'Data' object
        data = Data.from_NIRCam_pipeline(survey, version, excl_bands = excl_bands)
        return cls.from_data(data, aper_diams, cat_creator, xy_offset, forced_phot_band, loc_depth_min_flux_pc_errs, n_loc_depth_samples, fast)
    
    @classmethod
    def from_data(cls, data, aper_diams, cat_creator, xy_offset = [0, 0], forced_phot_band = "f444W", loc_depth_min_flux_pc_errs = [5, 10], n_loc_depth_samples = 5, fast = True):
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
            
        cat = cls.from_sex_cat(cat_path, data.instrument, data.survey, cat_creator)
        print("cat_path = ", cat.cat_path)
        cat.mask(data) # also saves the data object within the catalogue
        return cat
    
    @classmethod
    def from_sex_cat(cls, cat_path, instrument, survey, cat_creator):
        # open the catalogue
        cat = cls.cat_from_path(cat_path)
        # produce galaxy array from each row of the catalogue
        gals = np.array([Galaxy.from_sex_cat_row(row, instrument, cat_creator) for row in cat])
        return cls(gals, cat_path, survey, cat_creator)
    
    @classmethod
    def from_photo_z_cat(cls, cat_path, instrument, survey, cat_creator, codes):
        # open the catalogue
        cat = cls.cat_from_path(cat_path)
        # produce galaxy array from each row of the catalogue
        gals = np.array([Galaxy.from_photo_z_cat_row(row, instrument, cat_creator, codes) for row in cat])
        return cls(gals, cat_path, survey, cat_creator)
    
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
    
    # %%
    
    def open_full_cat(self):
        return Table.read(self.cat_path, character_as_bytes = False)

    def cat_from_path(path, crop_names = None):
        cat = Table.read(path, character_as_bytes = False)
        if crop_names != None:
            for name in crop_names:
                cat = cat[cat[name] == True]
        return cat
    
    # altered from original in mask_regions.py
    def mask(self, data, flag_blank_field = True): # mask paths is a dict of form {band: mask_path}
        print(f"Running masking code for {self.cat_path}. (Too much copying and pasting here!)")
        self.data = data # store data object in catalogue object
        masked_cat_path = self.cat_path.replace(".fits", "_masked.fits")
        
        if not Path(masked_cat_path).is_file():
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
            
            if self.data.is_blank:
                if flag_blank_field:
                    # add 'blank_module == True' to every galaxy in the catalogue
                    blank_flags = [True] * len(self.gals)
                    for gal in self:
                        gal.mask_flags["blank"] = True
            else: # mask cluster/blank field in reddest band (f444W for our NIRCam fields)
                if flag_blank_field:
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
                            if mask_flag_gal == True:
                                gal.mask_flags["blank_module"] = False
                                blank_flags.append(False)
                            else:
                                gal.mask_flags["blank_module"] = True
                                blank_flags.append(True)
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
                    cat[f"cluster"] = cluster_flags #.astype(bool)
                    cat.write(masked_cat_path, overwrite = True)
                    print(f"Finished masking cluster")
                else:
                    raise(Exception("Must manually create a 'cluster' mask in the 'Data' object!"))
            
            if flag_blank_field:
                # update saved catalogue
                cat = self.open_full_cat()
                cat[f"blank_module"] = blank_flags #.astype(bool)
                cat.write(masked_cat_path, overwrite = True)
                print(f"Finished masking blank field")
            
            # add additional boolean column to say whether an object is unmasked in all columns or not
            unmasked_blank = []
            for i, gal in enumerate(self):
                good_galaxy = True
                for band in self.data.instrument.bands:
                    if not gal.mask_flags[f"unmasked_{band}"]:
                        good_galaxy = False
                        break
                # don't include blank field galaxies in final boolean unmasked column
                if not self.data.is_blank and flag_blank_field:
                    if not gal.mask_flags["blank_module"]:
                        good_galaxy = False
                unmasked_blank.append(good_galaxy)
                gal.mask_flags["unmasked_blank"] = good_galaxy
            cat = self.open_full_cat()
            cat["unmasked_blank"] = unmasked_blank
            cat.write(masked_cat_path, overwrite = True)
            print(f"Finished masking!")
        else:
            self.cat_path = masked_cat_path
    
    def make_UV_fit_cat(self, UV_PDF_path, col_names = ["Beta", "flux_lambda_1500", "flux_Jy_1500", "M_UV", "A_UV", "L_obs", "L_int", "SFR"]):
        cat_data = []
        for i, gal in enumerate(self):
            gal_data = np.array([gal.ID])
            for name in col_names:
                gal_data = np.append(gal_data, funcs.percentiles_from_PDF(gal.phot_rest.open_UV_fit_PDF(UV_PDF_path, name, gal.ID))) # not currently saving to object
            gal_data = np.array(gal_data).flatten()
            if i == 0: # if the first column
                cat_data = gal_data
            else:
                cat_data = np.vstack([cat_data, gal_data])
        UV_col_names = np.array([[name, f"{name}_l1", f"{name}_u1"] for name in col_names]).flatten()
        fits_col_names = np.concatenate((np.array(["ID"]), UV_col_names))
        funcs.make_dirs(self.cat_path)
        UV_tab = Table(cat_data, names = fits_col_names)
        save_path = f"{self.cat_path[:-5]}_UV.fits"
        UV_tab.write(save_path, format = "fits", overwrite = True)
        self.UV_tab = UV_tab
        print(f"Writing UV table to {save_path}")
        
    def join_UV_fit_cat(self, cat, match_cols = ["NUMBER", "ID"]):
        joined_tab = join(cat, self.UV_tab, keys_left = match_cols[0], keys_right = match_cols[1])
        save_path = f"{self.cat_path[:-5]}_UV_matched.fits"
        joined_tab.write(save_path, format = "fits", overwrite = True)
        print(f"Joining UV table to catalogue! Saving to {save_path}")
        return joined_tab
    
    def flag_robust_high_z(self, relaxed = False):
        # should make use of an overloaded __setattr__ here!
        pass
    
    def flag_good_high_z(self, relaxed = False):
        # should make use of an overloaded __setattr__ here!
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

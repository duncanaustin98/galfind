# Catalogue_Base.py

import numpy as np
from astropy.table import Table
from copy import deepcopy

from . import useful_funcs_austind as useful_funcs
from .Data import Data
from .Galaxy import Galaxy
from . import useful_funcs_austind as funcs
from .Catalogue_Creator import GALFIND_Catalogue_Creator
from . import config, galfind_logger, SED_code, LePhare, EAZY, Bagpipes

class Catalogue_Base:
    # later on, the gal_arr should be calculated from the Instrument and sex_cat path, with SED codes already given
    def __init__(self, gals, cat_path, survey, cat_creator, instrument, codes = [], version = ''): #, UV_PDF_path):
        self.survey = survey
        self.cat_path = cat_path
        #self.UV_PDF_path = UV_PDF_path
        self.cat_creator = cat_creator
        self.instrument = instrument
        self.codes = codes
        self.gals = np.array(gals)
        if version == '':
            raise Exception('Version must be specified')
        self.version = version
        
        # keep a record of the crops that have been made to the catalogue
        self.crops = []
        
        # concat is commutative for catalogues
        self.__radd__ = self.__add__
        # cross-match is commutative for catalogues
        self.__rmul__ = self.__mul__

        print(str(self))

    # %% Overloaded operators

    def __str__(self, print_cls_name = True, print_data = True, print_sel_criteria = True):
        line_sep = "*" * 40 + "\n"
        band_sep = "-" * 10 + "\n"
        output_str = ""
        if print_cls_name:
            output_str += line_sep
            output_str += f"CATALOGUE: {self.survey} {self.version}\n" # could also show median RA/DEC from the array of galaxy sky coords
            output_str += band_sep
        if print_data and "data" in self.__dict__.keys():
            output_str += str(self.data)
        output_str += f"FITS CAT PATH = {self.cat_path}\n"
        # access table header to display what has been run for this catalogue
        cat = Table.read(self.cat_path, memmap = True)
        output_str += f"N_GALS_TOTAL = {len(cat)}\n"
        # display what other things have previously been calculated for this catalogue, including templates and zmax_lowz
        output_str += "CAT STATUS = SEXTRACTOR, "
        for i, (key, value) in enumerate(cat.meta.items()):
            if key in ["DEPTHS", "MASKED"] + [f"RUN_{subclass.__name__}" for subclass in SED_code.__subclasses__()]:
                output_str += key
                if i != len(cat.meta) - 1:
                    output_str += ", "
        output_str += "\n"
        # display total number of galaxies that satisfy the selection criteria previously performed
        if print_sel_criteria:
            for sel_criteria in ["EPOCHS", "BROWN_DWARF"]:
                if sel_criteria in cat.colnames:
                    output_str += f"N_GALS_{sel_criteria} = {len(cat[cat[sel_criteria]])}\n"
        output_str += band_sep
        # display crops that have been performed on this specific object
        if self.crops != []:
            output_str += f"N_GALS_OBJECT = {len(self)}\n"
            output_str += f"CROPS = {self.crops}\n"
        if print_cls_name:
            output_str += line_sep
        return output_str
    
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
    
    def __getattr__(self, name, code_name = "EAZY", templates = "fsps_larson", lowz_zmax = None): # only acts on attributes that don't already exist
        # # get array of galaxy properties for the catalogue if they exist in all galaxies
        # for gal in self:
        #     # property must exist in all galaxies within class
        #     if not hasattr(gal, name):
        #         raise AttributeError(f"'{name}' does not exist in all galaxies within {self.cat_name}!!!")
        if name in self[0].__dict__:
            return np.array([getattr(gal, name) for gal in self])
        elif name.upper() == "RA":
            return np.array([getattr(gal, "sky_coord").ra for gal in self])
        elif name.upper() == "DEC":
            return np.array([getattr(gal, "sky_coord").dec for gal in self])
        elif name in self[0].phot.SED_results[code_name][templates][funcs.lowz_label(lowz_zmax)].__dict__:
            return np.array([getattr(gal.phot.SED_results[code_name][templates][funcs.lowz_label(lowz_zmax)], name) for gal in self])
        elif name in self[0].mask_flags.__dict__:
            return np.array([getattr(gal.mask_flags, name) for gal in self])
        else:
            galfind_logger.critical(f"Galaxies do not have attribute = {name}!")
    
    def __setattr__(self, name, value, obj = "cat"):
        if obj == "cat":
            super().__setattr__(name, value)
        elif obj == "gal":
            # set attributes of individual galaxies within the catalogue
            for i, gal in enumerate(self):
                if type(value) in [list, np.array]:
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
            setattr(result, key, deepcopy(value, memo))
        return result
        
    @property
    def cat_dir(self):
        return funcs.split_dir_name(self.cat_path, "dir")
    
    @property
    def cat_name(self):
        return funcs.split_dir_name(self.cat_path, "name")
    
    def crop(self, crop_property, crop_limits): # upper and lower limits on galaxy properties (e.g. ID, redshift, mass, SFR, SkyCoord)
        cat_copy = deepcopy(self)
        if type(crop_limits) in [int, float, bool]:
            cat_copy.gals = cat_copy[getattr(cat_copy, crop_property) == crop_limits]
            cat_copy.crops.append(f"{crop_property}={crop_limits}")
        elif type(crop_limits) in [list, np.array]:
            cat_copy.gals = cat_copy[((getattr(cat_copy, crop_property) >= crop_limits[0]) & (getattr(cat_copy, crop_property) <= crop_limits[1]))]
            cat_copy.crops.append(f"{crop_limits[0]}<{crop_property}<{crop_limits[1]}")
        else:
            galfind_logger.critical(f"crop_limits={crop_limits} with type = {type(crop_limits)} not in [int, float, bool, list, np.array]")
        return cat_copy
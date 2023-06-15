#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 12:56:53 2023

@author: austind
"""

# Instrument.py
import numpy as np
from copy import copy, deepcopy
from abc import ABC, abstractmethod
import astropy.units as u
import json

from . import useful_funcs_austind as funcs
from . import config

class Instrument:
    
    def __init__(self, name, bands, band_wavelengths, band_FWHMs, zero_points, pixel_scales, excl_bands):
        self.bands = np.array(bands)
        self.band_wavelengths = band_wavelengths
        self.band_FWHMs = band_FWHMs
        #print("zero points and pixel scales should probably be in the 'Data' class rather than the 'Instrument' class!")
        self.zero_points = zero_points
        self.pixel_scales = pixel_scales
        self.name = name
        for band in excl_bands:
            self.remove_band(band)
    
# %% Overloaded operators

    def __repr__(self):
        # string representation of what is stored in this class
        return str(self.__dict__())
    
    def __len__(self):
        return len(self.bands)
    
    def __iter__(self):
        self.iter = 0
        return self
    
    def __next__(self):
        if self.iter > len(self) - 1:
            raise StopIteration
        else:
            band = self[self.iter]
            self.iter += 1
            return band
    
    def __getitem__(self, get_index): # create a new instrument with only the indexed band
        excl_bands = []
        for index, band in enumerate(self):
            if index != get_index:
                excl_bands.append(band)
        return self.new_instrument(excl_bands)
    
    def __add__(self, instrument):
        # cannot add multiple of the same bands together!!! (maybe could just ignore the problem \
        # in Instrument class and just handle it in Data, possibly stacking the two images)
        for band in instrument.bands:
            if band in self.bands:
                raise(Exception("Cannot add multiple of the same band together in 'Instrument.__add__()'!"))
        
        # add and sort bands from blue -> red
        bands = [band for band in list(config["Other"]["INSTRUMENT_NAMES"]) if band in self.bands + instrument.bands]
        band_wavelengths = self.band_wavelengths + instrument.band_wavelengths
        band_FWHMs = self.band_FWHMs + instrument.band_FWHMs
        zero_points = self.zero_points + instrument.zero_points
        pixel_scales = self.pixel_scales + instrument.pixel_scales
        # always name instruments from blue -> red
        if self.name == instrument.name:
            self.bands = bands
            self.band_wavelengths = band_wavelengths
            self.band_FWHMs = band_FWHMs
            self.zero_points = zero_points
            self.pixel_scales = pixel_scales
            out_instrument = self
        else:
            if np.where(self.name == list(config["Other"]["INSTRUMENT_NAMES"]))[0][0] < np.where(instrument.name == list(config["Other"]["INSTRUMENT_NAMES"]))[0][0]:
                # self is bluer than other
                name = str([self.name, instrument.name])
            else:
                # self is redder than other
                name = str([instrument.name, self.name])
            out_instrument = Combined_Instrument(name, bands, band_wavelengths, band_FWHMs, zero_points, pixel_scales)
            self.__del__() # delete old self
        return out_instrument
    
    def __sub__(self, instrument):
        print("Note that 'Instrument.__sub__()' only removes common bands between the two 'instrument' classes")
        for band in instrument.bands:
            if band in self.bands:
                self.remove_band(band)
            else:
                print(f"Cannot remove {band} from {self.name}!")
        return self
    
    # STILL NEED TO LOOK FURTHER INTO THIS
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    # STILL NEED TO LOOK FURTHER INTO THIS
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
    
# %% Class properties
    
    @property
    def bands_from_wavelengths(self):
        return {value: key for key, value in self.band_wavelengths.items()}
    
    @property
    def band_wavelength_errs(self):
        return {key: value / 2 for key, value in self.band_FWHMs.items()} # = FWHM/2 in Angstrom
    
# %% Class abstract methods

    @abstractmethod
    def aper_corr(self, aper_diam, band):
        pass
    
    @abstractmethod
    def new_instrument(self, excl_bands):
        pass
    
# %% Other class methods

    def remove_band(self, band):
        remove_index = np.where(self.bands == band)[0][0]
        #print(f"remove index = {remove_index}")
        self.bands = np.delete(self.bands, remove_index)
        del self.band_wavelengths[band]
        del self.band_FWHMs[band]
        del self.zero_points[band]
        
    def remove_index(self, remove_index):
        remove_band = self.bands[remove_index]
        self.remove_band(remove_band)
        
    @staticmethod
    def from_name(name):
        if name == "NIRCam":
            return NIRCam()
        elif name == "MIRI":
            return MIRI()
        elif name == "ACS":
            return ACS()
        else:
            raise(Exception(f"Instrument name: {name} does not exist in 'Instrument.from_name()'!"))

class NIRCam(Instrument):
    
    def __init__(self, zero_point = 28.08, pixel_scale = 0.03 * u.arcsec, excl_bands = []):
        bands = ["f090W", "f115W", "f150W", "f200W", "f277W", "f335M", "f356W", "f410M", "f444W"]
        band_wavelengths = {"f090W": 9_044., "f115W": 11_571., "f150W": 15_040., "f200W": 19_934., "f277W": 27_695., "f335M": 33_639, "f356W": 35_768., "f410M": 40_844., "f444W": 44_159.}
        band_wavelengths = {key: value * u.Angstrom for (key, value) in band_wavelengths.items()} # convert each individual value to Angstrom
        band_FWHMs = {"f090W": 2_101., "f115W": 2_683., "f150W": 3_371., "f200W": 4_717., "f277W": 7_110., "f335M": 3_609, "f356W": 8_408., "f410M": 4_375., "f444W": 11_055.}
        band_FWHMs = {key: value * u.Angstrom for (key, value) in band_FWHMs.items()} # convert each individual value to Angstrom
        zero_points = {band: zero_point for band in bands}
        pixel_scales = {band: pixel_scale for band in bands}
        super().__init__("NIRCam", bands, band_wavelengths, band_FWHMs, zero_points, pixel_scales, excl_bands)

    def aper_corr(self, aper_diam, band):
        aper_corr_data = np.loadtxt("/nvme/scratch/work/austind/aper_corr.txt", dtype = str, comments = "#")
        aper_diam_index = np.where(json.loads(config.get("SExtractor", "APERTURE_DIAMS")) == aper_diam.value)[0][0] + 1
        band_index = list(self.bands).index(band)
        # print((aper_diam_index, band_index))
        return float(aper_corr_data[band_index][aper_diam_index])
    
    def new_instrument(self, zero_point = 28.08, pixel_scale = 0.03 * u.arcsec, excl_bands = []):
        return NIRCam(zero_point, pixel_scale, excl_bands)
    
class MIRI(Instrument):
    
    def __init__(self, excl_bands = []):
        bands = []
        band_wavelengths = {}
        band_wavelengths = {key: value * u.Angstrom for (key, value) in band_wavelengths.items()} # convert each individual value to Angstrom
        band_FWHMs = {}
        band_FWHMs = {key: value * u.Angstrom for (key, value) in band_FWHMs.items()} # convert each individual value to Angstrom
        super().__init__("MIRI", bands, band_wavelengths, band_FWHMs, zero_points, pixel_scales, excl_bands)
    
    def aper_corr(self, aper_diam, band):
        pass
    
    def new_instrument(self, excl_bands = []):
        return MIRI(excl_bands)
    
class ACS(Instrument):
    
    def __init__(self, excl_bands = []):
        bands = []
        band_wavelengths = {}
        band_wavelengths = {key: value * u.Angstrom for (key, value) in band_wavelengths.items()} # convert each individual value to Angstrom
        band_FWHMs = {}
        band_FWHMs = {key: value * u.Angstrom for (key, value) in band_FWHMs.items()} # convert each individual value to Angstrom
        super().__init__("ACS", bands, band_wavelengths, band_FWHMs, zero_points, pixel_scales, excl_bands)
    
    def aper_corr(self, aper_diam, band):
        pass
    
    def new_instrument(self, excl_bands = []):
        return ACS(excl_bands)
    
class Combined_Instrument(Instrument):
    
    def __init__(self, name, bands, band_wavelengths, band_FWHMs, zero_points, pixel_scales):
        super().__init__(name, bands, band_wavelengths, band_FWHMs, zero_points, pixel_scales, excl_bands = [])
        
    def aper_corr(self, aper_diam, band):
        names = np.array(self.name)
        for name in names:
            instrument = Instrument.from_name(name)
            if band in instrument.bands:
                return instrument.aper_corr(aper_diam, band)
        raise(Exception(f"{band} does not exist in Instrument = {self.name}!"))
    
    def new_instrument(self):
        print("Still need to understand shallow and deep copy constructors in python!")
        return self

#aper_diams_sex = [0.32, 0.5, 1., 1.5, 2.] * u.arcsec

def return_loc_depth_mags(cat, band, incl_err = True, mag_type = ["APER", 0], min_flux_err_pc = 5):
    # insert minimum percentage error
    min_mag_err = funcs.flux_pc_to_mag_err(min_flux_err_pc)
    if "AUTO" in mag_type:
        #data = np.array(cat[f"MAG_AUTO_{band}"])
        print("NOT YET IMPLEMENTED!")
    elif "APER" in mag_type:
        data = np.array(cat[f"MAG_{mag_type[0]}_{band}_aper_corr"].T[mag_type[1]])
        #data_l1 = np.array(cat[f"MAGERR_{mag_type[0]}_{band}_l1_loc_depth"][mag_type[1]] 
        data_l1 = np.array([val if val > min_mag_err else min_mag_err for val in [cat[f"MAGERR_{mag_type[0]}_{band}_l1_loc_depth"].T[mag_type[1]]]])
        data_u1 = np.array([val if val > min_mag_err else min_mag_err for val in [cat[f"MAGERR_{mag_type[0]}_{band}_u1_loc_depth"].T[mag_type[1]]]])
        data_errs = np.array([data_l1, data_u1])
    else:
        raise(SyntaxError("Invalid mag_type in return_loc_depth_mags"))
    if incl_err:
        return data, data_errs
    else:
        return data
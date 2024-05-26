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
from pathlib import Path
from astroquery.svo_fps import SvoFps
import matplotlib.pyplot as plt
from typing import NoReturn, Union

from . import useful_funcs_austind as funcs
from . import config, galfind_logger, NIRCam_aper_corr
from .Filter import Filter

class Instrument:
    
    def __init__(self, name, bands, excl_bands, facility):
        self.bands = np.array(bands)
        self.name = name
        for band in excl_bands:
            self.remove_band(band)
        self.facility = facility

    @classmethod
    def from_SVO(cls, facility, instrument, excl_bands = []):
        filter_list = SvoFps.get_filter_list(facility = facility, instrument = instrument.split("_")[0])
        filter_list = filter_list[np.array([filter_name.split("/")[-1].split(".")[0] for filter_name in np.array(filter_list["filterID"])]) == instrument]
        bands = np.array([Filter.from_SVO(facility, instrument, filt_ID) for filt_ID in np.array(filter_list["filterID"])])
        return cls(instrument, bands, excl_bands, facility)

    @property
    def band_names(self):
        return np.array([band.band_name for band in self.bands])
    
    @property
    def band_wavelengths(self):
        # Central wavelengths
        return np.array([band.WavelengthCen.to(u.AA).value for band in self.bands]) * u.AA
    
    @property
    def band_FWHMs(self):
        return np.array([band.FWHM.to(u.AA).value for band in self.bands]) * u.AA
    
    @property
    def band_lower_wav_lims(self):
        return np.array([band.WavelengthLower50.to(u.AA).value for band in self.bands]) * u.AA
    
    @property
    def band_upper_wav_lims(self):
        return np.array([band.WavelengthUpper50.to(u.AA).value for band in self.bands]) * u.AA
    
# %% Overloaded operators

    def __str__(self):
        """ Function to print summary of Instrument class

        Returns:
            str: Summary containing facility, instrument name and filter set included in the instrument
        """
        line_sep = "*" * 40 + "\n"
        band_sep = "-" * 10 + "\n"
        output_str = line_sep
        output_str += f"FACILITY: {self.facility}\n"
        output_str += f"INSTRUMENT: {self.name}\n"
        # show individual bands used, ordered from blue to red
        output_str += f"FILTER SET: {str([f'{self.instrument_from_band(band_name).facility}/{self.instrument_from_band(band_name).name}/{band_name}' for band_name in self.band_names])}\n"
        # could also include PSF path and correction factors here
        output_str += line_sep
        return output_str
    
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
    
    def __getitem__(self, i):
        if type(i) in [int, np.int64, slice]:
            return self.bands[i]
        elif type(i) == str:
            return self.bands[self.index_from_band(i)]
        else:
            raise(TypeError(f"i={i} in {self.__class__.__name__}.__getitem__ has type={type(i)} which is not in [int, slice, str]"))
    
    def __del__(self):
        self.bands = []

    def instrument_from_band(self, band):
        # Pointless here but makes it compatible with Combined_Instrument
        if (band.split("+")[0] in self.band_names) or band in self.band_names:
            return self
        else:
            # This leads to confusing errors when passing in a bandname which doesn't exist
            return False
    
    # def __getitem__(self, get_index): # create a new instrument with only the indexed band
    #     excl_bands = []
    #     for index, band in enumerate(self):
    #         if index != get_index:
    #             excl_bands.append(band)
    #     return self.new_instrument(excl_bands)
    
    def __add__(self, instrument):
        # cannot add multiple of the same bands together!!! (maybe could just ignore the problem \
        # in Instrument class and just handle it in Data, possibly stacking the two images)
        for band in instrument.band_names:
            if band in self.band_names:
                raise(Exception("Cannot add multiple of the same band together in 'Instrument.__add__()'!"))
        
        # add and sort bands from blue -> red
        bands = [band for band in sorted(np.concatenate([self.bands, instrument.bands]), \
            key = lambda band: band.WavelengthCen.to(u.AA).value)]

        # always name instruments from blue -> red
       
        if self.name == instrument.name:
            self.bands = bands
            out_instrument = self
        else:
            all_instruments = json.loads(config.get("Other", "INSTRUMENT_NAMES"))
            all_facilities = json.loads(config.get("Other", "TELESCOPE_NAMES"))
            # split self.name and instrument.name
            names = list(set(self.name.split("+") + instrument.name.split("+")))
            facilities = list(set(self.facility.split("+") + instrument.facility.split("+")))
            name = "+".join([name for name in all_instruments if name in names])
            facility = "+".join([facility for facility in all_facilities if facility in facilities])
            out_instrument = Combined_Instrument(name, bands, excl_bands = [], facility = facility)
            #self.__del__() # delete old self
        return out_instrument
    
    def __sub__(self, instrument):
        print("Note that 'Instrument.__sub__()' only removes common bands between the two 'instrument' classes")
        for band in instrument.band_names:
            if band in self.band_names:
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
        for key, value in self.__dict__.items():
            setattr(result, key, deepcopy(value, memo))
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

    def remove_band(self, band_name) -> NoReturn:
        assert type(band_name) in [str, np.str_], galfind_logger.critical(f"{band_name=} with {type(band_name)=} not in ['str', 'np.str_']")
        assert band_name in self.band_names, galfind_logger.critical(f"{band_name=} not in {self.band_names=}")
        self.remove_index(self.index_from_band_name(band_name))
        
    def remove_index(self, remove_index: int) -> NoReturn:
        if not remove_index == None:
            self.bands = np.delete(self.bands, remove_index)
        
    def index_from_band_name(self, band_name) -> int:
        if band_name in self.band_names:
            return np.where(self.band_names == band_name)[0][0]
        else:
            return None
    
    def band_name_from_index(self, index) -> str:
        return self.band_names[index]
    
    def bands_from_wavelength(self, wavelength) -> list[Filter]:
        return [band for band in self if wavelength > band.WavelengthLower50 and wavelength < band.WavelengthUpper50]

    def nearest_band_to_wavelength(self, wavelength, medium_bands_only = False, check_wavelength_in_band = True) -> Union[Filter, None]:
        if medium_bands_only:
            search_bands = [band for band in self if "M" == band.band_name[-1]]
        else:
            search_bands = self.bands
        nearest_band = search_bands[np.abs([funcs.convert_wav_units(band.WavelengthCen, u.AA).value for band in search_bands] \
            - funcs.convert_wav_units(wavelength, u.AA).value).argmin()]
        
        if check_wavelength_in_band and not nearest_band in self.bands_from_wavelength(wavelength):
            return None
        else:
            return nearest_band
    
    def bands_avoiding_wavs(self, wavs):
        # extract the unique band names
        unique_band_names = np.unique(np.array([band.band_name for wav in wavs for band in self.bands_from_wavelength(wav)]))
        # return an array of Filter objects corresponding to the band names
        return np.array([self[self.index_from_band_name(band_name)] for band_name in self.band_names if not band_name in unique_band_names])
    
    @staticmethod
    def from_name(name, excl_bands = []):
        if name == "NIRCam":
            return NIRCam(excl_bands = excl_bands)
        elif name == "MIRI":
            return MIRI(excl_bands = excl_bands)
        elif name == "ACS_WFC":
            return ACS_WFC(excl_bands = excl_bands)
        elif name == "WFC3_IR":
            return WFC3_IR(excl_bands = excl_bands)
        elif "+" in name:
            new_instruments = Combined_Instrument.instruments_from_name(name, excl_bands = excl_bands)
            for i, instrument in enumerate(new_instruments):
                if i == 0:
                    new_instrument = instrument
                else:
                    new_instrument += instrument
            return new_instrument
        else:
            raise(Exception(f"Instrument name: {name} does not exist in 'Instrument.from_name()'!"))
    
    # def load_instrument_filter_profiles(self, from_SVO = True):
    #     for band in self:
    #         band.load_band_filter_profile(band, from_SVO = from_SVO)

    def plot_filter_profiles(self, ax, wav_units = u.um, from_SVO = True, \
            cmap_name = "Spectral_r", annotate = True, show = True, save = False) -> NoReturn:
        cmap = plt.get_cmap(cmap_name, len(self))
        for i, band in enumerate(self):
            band.plot_filter_profile(ax, from_SVO = from_SVO, color = cmap[i])
        if annotate:
            ax.set_title(f"{self.name} filters")
            ax.set_xlabel(r"$\lambda_{\mathrm{obs}}$ / " + funcs.unit_labels_dict[wav_units])
            ax.set_ylabel("Transmission")
            ax.set_ylim(0., np.max([trans for trans in band.trans for band in self]) + 0.1)
        if save:
            plt.savefig(f"{self.name}_filter_profiles.png")
        if show:
            plt.show()

class NIRCam(Instrument):
    
    def __init__(self, excl_bands = []):
        instr = Instrument.from_SVO("JWST", self.__class__.__name__, excl_bands = [])
        super().__init__(instr.name, instr.bands, excl_bands, instr.facility)
        # bands = ["f070W", "f090W", "f115W", "f140M", "f150W", "f162M", "f182M", "f200W", "f210M", "f250M", "f277W", "f300M", "f335M", "f356W", "f360M", "f410M", "f430M", "f444W", "f460M", "f480M"]
        # band_wavelengths = {"f070W": 7_056., "f090W": 9_044., "f115W": 11_571., "f140M": 14_060., "f150W": 15_040., "f162M": 16_281., "f182M": 18_466., "f200W": 19_934., "f210M": 20_964., \
        #                     "f250M": 25_038., "f277W": 27_695., "f300M": 29_908., "f335M": 33_639., "f356W": 35_768., "f360M": 36_261., "f410M": 40_844., "f430M": 42_818., "f444W": 44_159., \
        #                         "f460M": 46_305., "f480M": 48_192.}
        # band_wavelengths = {key: value * u.Angstrom for (key, value) in band_wavelengths.items()} # convert each individual value to Angstrom
        # band_FWHMs = {"f070W": 1_600., "f090W": 2_101., "f115W": 2_683., "f140M": 1_478., "f150W": 3_371., "f162M": 1_713., "f182M": 2_459., "f200W": 4_717., "f210M": 2_089., \
        #               "f250M": 1_825., "f277W": 7_110., "f300M": 3_264., "f335M": 3_609., "f356W": 8_408., "f360M": 3_873., "f410M": 4_375., "f430M": 2_312., "f444W": 11_055., \
        #                   "f460M": 2_322., "f480M": 3_145.}
        # band_FWHMs = {key: value * u.Angstrom for (key, value) in band_FWHMs.items()} # convert each individual value to Angstrom
        # super().__init__("NIRCam", bands, band_wavelengths, band_FWHMs, excl_bands, "JWST")

    def aper_corr(self, aper_diam, band):
        aper_corr_path = f'{config["Other"]["GALFIND_DIR"]}/Aperture_corrections/NIRCam_aper_corr.txt'
        if not Path(aper_corr_path).is_file():
            # perform aperture corrections
            NIRCam_aper_corr.main(self.band_names)
        # load aperture corrections from appropriate path (bands in NIRCam class must be the same as those saved in aper_corr)
        aper_corr_data = np.loadtxt(aper_corr_path, dtype = str, comments = "#")
        aper_diam_index = np.where(json.loads(config.get("SExtractor", "APERTURE_DIAMS")) == aper_diam.value)[0][0] + 1
        band_index = list(self.band_names).index(band)
        # print((aper_diam_index, band_index))
        return float(aper_corr_data[band_index][aper_diam_index])
    
    def new_instrument(self, excl_bands = []):
        return NIRCam(excl_bands)
    
class MIRI(Instrument):
    
    def __init__(self, excl_bands = []):
        instr = Instrument.from_SVO("JWST", self.__class__.__name__, excl_bands = [])
        super().__init__(instr.name, instr.bands, excl_bands, instr.facility)
        # bands = ['f560W', 'f770W', 'f1000W', 'f1130W', 'f1280W', 'f1500W', 'f1800W', 'f2100W', 'f2550W']
        # band_wavelengths = {'f560W':55870.25, 'f770W':75224.94, 'f1000W': 98793.45, 'f1130W':112960.71, 'f1280W':127059.68, 'f1500W':149257.07, 'f1800W':178734.17, 'f2100W':205601.06, 'f2550W':251515.99}
        # band_wavelengths = {key: value * u.Angstrom for (key, value) in band_wavelengths.items()} # convert each individual value to Angstrom
        # band_FWHMs = {'f560W': 11114.05, 'f770W': 20734.55, 'f1000W': 18679.18, 'f1130W': 7091.01, 'f1280W': 25306.74, 'f1500W': 31119.13, 'f1800W': 29839.89,'f2100W':46711.97, 'f2550W':36393.71}
        # band_FWHMs = {key: value * u.Angstrom for (key, value) in band_FWHMs.items()} # convert each individual value to Angstrom    
        # super().__init__("MIRI", bands, band_wavelengths, band_FWHMs, excl_bands, "JWST")
    
    def aper_corr(self, aper_diam, band):
        pass
    
    def new_instrument(self, excl_bands = []):
        return MIRI(excl_bands)
    
class ACS_WFC(Instrument):
    
    def __init__(self, excl_bands = []):
        instr = Instrument.from_SVO("HST", self.__class__.__name__, excl_bands = [])
        super().__init__(instr.name, instr.bands, excl_bands, instr.facility)
        # bands = ["f435W", "fr459M", "f475W", "f550M", "f555W", "f606W", "f625W", "fr647M", "f775W", "f814W", "f850LP", "fr914M"]
        # # Wavelengths corrrespond to lambda effective of the filters from SVO Filter Profile Service
        # band_wavelengths = {"f435W": 4_340., "fr459M": 4_590., "f475W": 4_766., "f550M": 5_584., "f555W": 5_373., "f606W": 5_960., \
        #                     "f625W": 6_325., "fr647M": 6_472., "f775W": 7_706., "f814W": 8_073., "f850LP": 9_047., "fr914M": 9_072.}
        # band_wavelengths = {key: value * u.Angstrom for (key, value) in band_wavelengths.items()} # convert each individual value to Angstrom
        # # FWHMs corrrespond to FWHM of the filters from SVO Filter Profile Service
        # band_FWHMs = {"f435W": 937., "fr459M": 350., "f475W": 1_437., "f550M": 546., "f555W": 1_240., "f606W": 2_322., \
        #                     "f625W": 1_416., "fr647M": 501., "f775W": 1_511., "f814W": 1_858., "f850LP": 1_208., "fr914M": 774.}
        # band_FWHMs = {key: value * u.Angstrom for (key, value) in band_FWHMs.items()} # convert each individual value to Angstrom    
        # super().__init__("ACS_WFC", bands, band_wavelengths, band_FWHMs, excl_bands, "HST")
    
    def aper_corr(self, aper_diam, band):
        aper_corr_path = f'{config["Other"]["GALFIND_DIR"]}/Aperture_corrections/hst_acs_wfc_aper_corr.dat'
        aper_corr_data = np.loadtxt(aper_corr_path, comments = "#", dtype=[('band', 'U10'), ('0.32', 'f4'), ('0.5', 'f4'), ('1.0', 'f4'), ('1.5', 'f4'), ('2.0', 'f4')])
        return aper_corr_data[aper_corr_data['band'] == band.upper()][str(aper_diam.to('arcsec').value)][0]
    
    def new_instrument(self, excl_bands = []):
        return ACS_WFC(excl_bands)
   
class WFC3_IR(Instrument):

    def __init__(self, excl_bands = []):
        instr = Instrument.from_SVO("HST", self.__class__.__name__, excl_bands = [])
        super().__init__(instr.name, instr.bands, excl_bands, instr.facility)
        # bands = ["f098M", "f105W",  "f110W", "f125W", "f127M", "f139M", "f140W", "f153M", "f160W"]
        # # Wavelengths corrrespond to lambda effective of the filters from SVO Filter Profile Service
        # band_wavelengths = {"f098M": 9_875., "f105W": 10_584.,  "f110W": 11_624., "f125W": 12_516., "f127M": 12_743., "f139M": 13_843., "f140W": 13_970., "f153M": 15_334., "f160W": 15_392.}
        # band_wavelengths = {key: value * u.Angstrom for (key, value) in band_wavelengths.items()} # convert each individual value to Angstrom
        # # FWHMs corrrespond to FWHM of the filters from SVO Filter Profile Service
        # band_FWHMs = {"f098M": 1_692., "f105W": 2_917.,  "f110W": 4_994., "f125W": 3_005., "f127M": 692., "f139M": 652., "f140W": 3_941., "f153M": 693., "f160W": 2_875.}
        # band_FWHMs = {key: value * u.Angstrom for (key, value) in band_FWHMs.items()} # convert each individual value to Angstrom
        # super().__init__("WFC3_IR", bands, band_wavelengths, band_FWHMs, excl_bands, "HST")
    
    def aper_corr(self, aper_diam, band):
        aper_corr_path = f'{config["Other"]["GALFIND_DIR"]}/Aperture_corrections/wfc3ir_aper_corr.dat'
        aper_corr_data = np.loadtxt(aper_corr_path, comments = "#", dtype=[('band', 'U10'), ('0.32', 'f4'), ('0.5', 'f4'), ('1.0', 'f4'), ('1.5', 'f4'), ('2.0', 'f4')])
        return aper_corr_data[aper_corr_data['band'] == band.upper()][str(aper_diam.to('arcsec').value)][0]
    
    def new_instrument(self, excl_bands = []):
        return WFC3_IR(excl_bands)
    
class Combined_Instrument(Instrument):
    
    def __init__(self, name, bands, excl_bands, facility):
        super().__init__(name, bands, excl_bands, facility)
        
    @classmethod
    def instruments_from_name(cls, name, excl_bands = []):
        combined_instrument_names = name.split("+")
        return [cls.from_name(combined_instrument_name, excl_bands) for combined_instrument_name in combined_instrument_names]

    @classmethod
    def combined_instrument_from_name(cls, name, excl_bands = []):
        return cls.from_name(name, excl_bands)
      
    def aper_corr(self, aper_diam, band):
        names = self.name.split("+")
        for name in names:
            instrument = self.from_name(name)
            if band in instrument.band_names:
                return instrument.aper_corr(aper_diam, band)
        raise(Exception(f"{band} does not exist in Instrument = {self.name}!"))
    
    def instrument_from_band(self, band, return_name = True):
        names = self.name.split("+")
        for name in names:
            instrument = Instrument.from_name(name)
            if instrument.instrument_from_band(band) != False:
                return instrument
        
    def new_instrument(self, excl_bands = []):
        instruments = self.instruments_from_name(self.name, excl_bands)
        for i, instrument in enumerate(instruments):
            if i == 0:
                new_instrument = instrument
            else:
                new_instrument += instrument
        # print("Still need to understand shallow and deep copy constructors in python!")
        return new_instrument

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

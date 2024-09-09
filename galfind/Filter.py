# Filter.py

import numpy as np
import astropy.units as u
from astroquery.svo_fps import SvoFps
import itertools

from . import galfind_logger
from . import useful_funcs_austind as funcs

class Filter:

    def __init__(self, facility, instrument, band_name, wav, trans, properties = {}):
        assert(type(facility) == type(instrument) == type(band_name) == str)
        assert(len(wav) == len(trans))
        self.facility = facility
        self.instrument = instrument
        self.band_name = band_name
        self.wav = wav
        self.trans = trans
        for key, value in properties.items():
            self.__setattr__(key, value)
        self.properties = properties # currently just used in __str__ only

    def __str__(self):
        line_sep = "*" * 40 + "\n"
        band_sep = "-" * 10 + "\n"
        output_str = line_sep
        output_str += f"{self.facility}/{self.instrument}/{self.band_name}\n"
        output_str += band_sep
        for key, value in self.properties.items():
            output_str += f"{key}: {value}\n"
        output_str += line_sep
        return output_str

    def __len__(self):
        return 1

    def __eq__(self, other):
        # ensure types are the same
        if type(self) != type(other):
            return False
        # ensure both have the same attribute keys
        elif not all(other_key in self.__dict__.keys() for other_key in other.__dict__.keys()) or \
                not all(self_key in other.__dict__.keys() for self_key in self.__dict__.keys()):
            return False
        # ensure both have matching wav and trans arrays
        elif len(self.wav) != len(other.wav) or any(self_wav != other_wav \
                for self_wav, other_wav in zip(self.wav, other.wav)) \
                or len(self.trans) != len(other.trans) or any(self_trans != other_trans \
                for self_trans, other_trans in zip(self.trans, other.trans)):
            return False
        # ensure attribute values are the same
        elif any(getattr(self, self_key) != getattr(other, self_key) \
                for self_key in self.__dict__.keys() if self_key not in ["wav", "trans", "properties"]):
            return False
        # ensure SVO property keys stored are the same
        elif not all(key in self.properties.keys() for key in other.properties.keys()) or \
                not all(key in other.properties.keys() for key in self.properties.keys()):
            return False
        # ensure SVO property values stored are the same
        else:
            return all(self.properties[key] == other.properties[key] for key in self.properties.keys())

    def __setattr__(self, key, value):
        super().__setattr__(key, value)

    @classmethod
    def from_SVO(cls, facility, instrument, filter_name):
        full_name = f"{facility}/{instrument}.{filter_name}"
        try:
            filter_profile = SvoFps.get_transmission_data(full_name)
        except:
            galfind_logger.critical(f"{full_name} is not a valid SvoFps filter!")
        wav = np.array(filter_profile["Wavelength"])
        trans = np.array(filter_profile["Transmission"])
        properties = SvoFps.data_from_svo(query = {"Facility": facility, "Instrument": instrument.split("_")[0]}, \
            error_msg = f"'{facility}/{instrument.split('_')[0]}' is not a valid facility/instrument combination!")
        properties = properties[properties["filterID"] == full_name]
        wav *= u.Unit(str(np.array(properties["WavelengthUnit"])[0]))

        output_prop = {}
        for key, value in properties.items():
            if ("Wavelength" in key or key in ["WidthEff", "FWHM"]) and key not in ["WavelengthUnit", "WavelengthUCD"]:
                output_prop[key] = float(np.array(value)[0]) * u.Unit(str(np.array(properties["WavelengthUnit"])[0]))
            elif key in ["Description", "Comments"]:
                output_prop[key] = str(np.array(value)[0])
            elif key == "DetectorType":
                detector_type_dict = {1: "photon counter"}
                output_prop[key] = str(detector_type_dict[int(np.array(value)[0])])
        output_prop["WavelengthUpper50"] = output_prop["WavelengthCen"] + output_prop["FWHM"] / 2.
        output_prop["WavelengthLower50"] = output_prop["WavelengthCen"] - output_prop["FWHM"] / 2.
        return cls(facility, instrument, filter_name, wav, trans, output_prop)
    
    #def crop_wav_range(self, lower_throughput, upper_throughput):
    #    self.wavs = self.wavs[self.trans > 1e-1]
    
    def plot_filter_profile(self, ax, wav_units = u.um, from_SVO = True, color = "black"):
        wavs = funcs.convert_wav_units(self.wav, wav_units).value
        ax.fill_between(wavs, 0., self.trans, color = color, alpha = 0.6)
        ax.plot(wavs, self.trans, color = "black", lw = 2) #cmap[np.where(self.bands == band)])
        ax.text(funcs.convert_wav_units(self.WavelengthCen, wav_units).value, \
            np.max(self.trans) + 0.03, self.band_name, ha = "center", fontsize = 8)

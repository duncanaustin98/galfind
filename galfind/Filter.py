# Filter.py

import numpy as np
import astropy.units as u
from astroquery.svo_fps import SvoFps

from . import galfind_logger

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

    def __str__(self):
        pass

    def __setattr__(self, key, value):
        super().__setattr__(key, value)

    @classmethod
    def from_SVO(cls, facility, instrument, filter_name):
        band = filter_name.split(".")[-1]
        try:
            filter_profile = SvoFps.get_transmission_data(filter_name)
        except:
            galfind_logger.critical(f"{filter_name} is not a valid filter name!")
        wav = np.array(filter_profile["Wavelength"])
        trans = np.array(filter_profile["Transmission"])

        properties = SvoFps.data_from_svo(query = {"Facility": facility, "Instrument": instrument.split("_")[0]}, error_msg = f"'{facility}/{instrument.split('_')[0]}' is not a valid facility/instrument combination!")
        properties = properties[properties["filterID"] == filter_name]
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
        return cls(facility, instrument, band, wav, trans, output_prop)
    
    def plot_filter_profile(self, ax, band, from_SVO = True, color = "black"):
        ax.fill_between(self.wav, 0., self.trans, color = color, alpha = 0.6)
        ax.plot(self.wav, self.trans, color = "black", lw = 2) #cmap[np.where(self.bands == band)])
        mid_wav = np.median(self.wav[self.trans > 1e-3])
        ax.text(self.WavelengthCen, np.max(self.trans) + 0.03, band, ha = "center", fontsize = 8)
        ax.grid(False)
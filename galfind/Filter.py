# Filter.py

import numpy as np
import astropy.units as u
from astroquery.svo_fps import SvoFps

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
    
    @classmethod
    def from_spextra(cls, passband):
        facility = passband.filter_name.split("/")[0].upper()
        instrument = passband.filter_name.split("/")[1].upper()
        band = passband.basename
        wav = np.array(passband.model.points[0]) * u.AA
        trans = np.array(passband.model.lookup_table)
        # for output_property
        _wav = [__wav for __wav, _trans in zip(wav.value, trans) if _trans > 0.5]
        output_prop = {"WavelengthCen": np.mean(_wav) * u.AA, \
            "WavelengthLower50": np.min(_wav) * u.AA, "WavelengthUpper50": np.max(_wav) * u.AA}
        return cls(facility, instrument, band, wav, trans, output_prop)

    #def crop_wav_range(self, lower_throughput, upper_throughput):
    #    self.wavs = self.wavs[self.trans > 1e-1]
    
    def plot_filter_profile(self, ax, wav_units = u.um, from_SVO = True, colour = "black"):
        wavs = funcs.convert_wav_units(self.wav, wav_units).value
        ax.fill_between(wavs, 0., self.trans, color = colour, alpha = 0.6)
        ax.plot(wavs, self.trans, color = "black", lw = 2) #cmap[np.where(self.bands == band)])
        ax.text(funcs.convert_wav_units(self.WavelengthCen, wav_units).value, \
            np.max(self.trans) + 0.03, self.band_name, ha = "center", fontsize = 8)
        
    def plot_z_wav_rest_track(self, ax, z_arr, wav_units = u.AA, outline = True, colour = "black", alpha = 0.5):
        l1_track = self.WavelengthLower50.to(wav_units).value / (1. + z_arr)
        u1_track = self.WavelengthUpper50.to(wav_units).value / (1. + z_arr)
        ax.fill_between(z_arr, l1_track, u1_track, color = colour, alpha = alpha)
        if outline:
            ax.plot(z_arr, l1_track, color = "black")
            ax.plot(z_arr, u1_track, color = "black")

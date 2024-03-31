# Filter.py

import numpy as np
import astropy.units as u
from astroquery.svo_fps import SvoFps

#from . import galfind_logger

class Filter:

    def __init__(self, facility, instrument, band, wav, trans, properties = {}):
        assert(type(facility) == type(instrument) == type(band) == str)
        assert(len(wav) == len(trans))
        self.facility = facility
        self.instrument = instrument
        self.band = band
        self.wav = wav
        self.trans = trans
        for key, value in properties.items():
            self.__setattr__(key, value)

    def __str__(self):
        pass

    def __setattr__(self, key, value):
        super().__setattr__(key, value)

    @classmethod
    def from_SVO(cls, facility, instrument, band):
        band = band.upper()
        filter_name = f"{facility}/{instrument}.{band}"
        try:
            filter_profile = SvoFps.get_transmission_data(filter_name)
        except:
            pass
            #galfind_logger.critical(f"No filter profile for {filter_name}")
        wav = np.array(filter_profile["Wavelength"])
        trans = np.array(filter_profile["Transmission"])

        properties = SvoFps.data_from_svo(query = {"Facility": facility, "Instrument": instrument})
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
    
    

if __name__ == "__main__":
    Filter.from_SVO("JWST", "NIRCam", "F444W")
# Instrument.py
from __future__ import annotations

from typing import NoReturn, Dict, Any, Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from . import Band_Data, Band_Data_Base, Data
    from . import PSF_Base, PSF_Cutout
    from . import Filter, Multiple_Filter

import astropy.units as u
from astropy.table import Table
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from astroquery.svo_fps import SvoFps
from copy import deepcopy
import json
from abc import ABC, abstractmethod

try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import config, galfind_logger
from . import useful_funcs_austind as funcs


class Facility(ABC):
    # Facility class to store the name of the facility
    # and other facility-specific attributes/methods

    def __init__(self) -> None:
        if not hasattr(self, "SVO_name"):
            self.SVO_name = self.__class__.__name__

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __eq__(self, other: Type[Self]) -> bool:
        if isinstance(other, Facility):
            return self.__class__.__name__ == other.__class__.__name__
        else:
            return False

    def __copy__(self) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, deepcopy(value, memo))
        return result


class HST(Facility, funcs.Singleton):
    pass


class JWST(Facility, funcs.Singleton):
    pass

class Paranal(Facility, funcs.Singleton):
    pass

class Spitzer(Facility, funcs.Singleton):
    pass

class Euclid(Facility, funcs.Singleton):
    pass

class CFHT(Facility, funcs.Singleton):
    pass

class Subaru(Facility, funcs.Singleton):
    pass


class Instrument(ABC):
    def __init__(
        self: Type[Self],
        facility: Union[str, Facility],
        filt_names: List[str],
    ) -> None:
        if isinstance(facility, str):
            self.facility = globals()[facility]()
            assert isinstance(self.facility, tuple(Facility.__subclasses__()))
        else:
            self.facility = facility
        self.filt_names = filt_names
        self._load_aper_corrs()

        if not hasattr(self, "SVO_name"):
            self.SVO_name = self.__class__.__name__


    def __str__(self) -> str:
        # print filter_names?
        output_str = funcs.line_sep
        output_str += (
            f"{self.facility.__class__.__name__}/{self.__class__.__name__}\n"
        )
        if len(self.facility.__dict__) > 0 or len(self.__dict__) > 0:
            output_str += funcs.line_sep
        if len(self.__dict__) > 0:
            for key, value in self.facility.__dict__.items():
                output_str += f"{key}: {value}\n"
            output_str += funcs.band_sep
        if len(self.facility.__dict__) > 0:
            output_str += f"FACILITY:\n"
            output_str += funcs.band_sep
            for key, value in self.facility.__dict__.items():
                output_str += f"{key}: {value}\n"
        output_str += funcs.line_sep
        return output_str

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __eq__(self, other: Type[Self]) -> bool:
        if isinstance(other, Instrument):
            return (
                self.facility == other.facility
                and self.__class__.__name__ == other.__class__.__name__
            )
        else:
            return False

    def __copy__(self) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, deepcopy(value, memo))
        return result

    def make_PSF(self, band_data: Band_Data, method: str) -> Type[PSF_Base]:
        if method == "model":
            # no real data needed for model PSF
            return self.make_model_PSF(band_data.filt)
        elif method == "empirical":
            return self.make_empirical_PSF(band_data)
        else:
            raise NotImplementedError

    def make_PSFs(self, data: Data, method: str) -> List[Type[PSF_Base]]:
        return [self.make_PSF(data, band, method) for band in self]

    @abstractmethod
    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        pass

    def calc_pix_scale(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        pass

    @abstractmethod
    def make_empirical_PSF(self, band_data: Band_Data) -> Type[PSF_Cutout]:
        pass

    def make_empirical_PSFs(self, data: Data) -> List[Type[PSF_Cutout]]:
        return [self.make_empirical_PSF(data, band) for band in self]

    @abstractmethod
    def make_model_PSF(self, filt: Union[str, Filter]) -> Type[PSF_Cutout]:
        pass

    def make_model_PSFs(self, filterset: Multiple_Filter) -> List[Type[PSF_Cutout]]:
        return [self.make_model_PSF(filt) for filt in filterset]
    
    def _load_aper_corrs(self) -> Dict[u.Quantity, Any]:
        # open Aperture corrections file
        aper_corr_path = config['DEFAULT']['APER_CORR_DIR'] + \
            f"/{self.__class__.__name__}_aper_corr.txt"

        if Path(aper_corr_path).is_file():
            aper_corr_tab = Table.read(aper_corr_path, format="ascii")
            aper_diams = [0.2, 0.32, 0.5, 1.0, 1.5, 2.0] * u.arcsec
            # save aperture corrections in self
            if not hasattr(self, "aper_corrs"):
                self.aper_corrs = {}
            for filt_name in list(aper_corr_tab["col1"]):
                aper_corr_row = aper_corr_tab[
                        aper_corr_tab["col1"] == filt_name
                    ]
                self.aper_corrs[filt_name] = {aper_diam: float(aper_corr_row[f"col{str(i + 2)}"]) \
                    for i, aper_diam in enumerate(aper_diams)}
            galfind_logger.debug(
                f"Aperture corrections for {self.__class__.__name__} loaded from {aper_corr_path}"
            )
        else:
            galfind_logger.warning(
                f"Aperture corrections for {self.__class__.__name__} not found in {aper_corr_path}"
            )


class NIRCam(Instrument, funcs.Singleton):
    def __init__(self) -> None:
        NIRCam_band_names = [
            "F070W",
            "F090W",
            "F115W",
            "F140M",
            "F150W",
            "F162M",
            "F164N",
            "F150W2",
            "F182M",
            "F187N",
            "F200W",
            "F210M",
            "F212N",
            "F250M",
            "F277W",
            "F300M",
            "F323N",
            "F322W2",
            "F335M",
            "F356W",
            "F360M",
            "F405N",
            "F410M",
            "F430M",
            "F444W",
            "F460M",
            "F466N",
            "F470N",
            "F480M",
        ]
        super().__init__("JWST", NIRCam_band_names)

    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        # assume flux units of MJy/sr and calculate corresponding ZP
        ZP = -2.5 * np.log10(
            (band_data.pix_scale.to(u.rad).value ** 2) * u.MJy.to(u.Jy)
        ) + u.Jy.to(u.ABmag)
        return ZP

    def make_model_PSF(self, band: Union[str, Filter]) -> Type[PSF_Cutout]:
        pass

    def make_empirical_PSF(self, band_data: Band_Data) -> Type[PSF_Cutout]:
        pass


class MIRI(Instrument, funcs.Singleton):
    def __init__(self) -> None:
        MIRI_band_names = [
            "F560W",
            "F770W",
            "F1000W",
            "F1065C",
            "F1140C",
            "F1130W",
            "F1280W",
            "F1500W",
            "F1550C",
            "F1800W",
            "F2100W",
            "F2300C",
            "F2550W",
        ]
        super().__init__("JWST", MIRI_band_names)

    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        # assume flux units of MJy/sr and calculate corresponding ZP
        ZP = -2.5 * np.log10(
            (band_data.pix_scale.to(u.rad).value ** 2) * u.MJy.to(u.Jy)
        ) + u.Jy.to(u.ABmag)
        return ZP

    def make_model_PSF(self, band: Union[str, Filter]) -> Type[PSF_Base]:
        pass

    def make_empirical_PSF(self, data: Data, band: Union[str, Filter]) -> Type[PSF_Base]:
        pass


class ACS_WFC(Instrument, funcs.Singleton):
    def __init__(self) -> None:
        ACS_WFC_band_names = [
            "FR388N",
            "FR423N",
            "F435W",
            "FR459M",
            "FR462N",
            "F475W",
            "F502N",
            "FR505N",
            "F555W",
            "FR551N",
            "F550M",
            "F606W",
            "FR601N",
            "F625W",
            "FR647M",
            "FR656N",
            "F658N",
            "F660N",
            "FR716N",
            #"POL_UV",
            "G800L",
            #"POL_V",
            "F775W",
            "FR782N",
            "F814W",
            "FR853N",
            "F892N",
            "F850LP",
            "FR914M",
            "FR931N",
            "FR1016N",
        ]
        self.SVO_name = "ACS"
        super().__init__("HST", ACS_WFC_band_names)

    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        im_header = band_data.load_im()[1]
        if "PHOTFLAM" in im_header and "PHOTPLAM" in im_header:
            ZP = (
                -2.5 * np.log10(im_header["PHOTFLAM"])
                - 21.1
                - 5.0 * np.log10(im_header["PHOTPLAM"])
                + 18.6921
            )
        elif "ZEROPNT" in im_header:
            ZP = im_header["ZEROPNT"]
        # elif "BUNIT" in im_header:
        #     unit = im_header["BUNIT"].replace(" ", "")
        #     assert unit == "MJy/sr"
        #     ZP = -2.5 * np.log10(
        #         (band_data.pix_scale.to(u.rad).value ** 2) * u.MJy.to(u.Jy)
        #     ) + u.Jy.to(u.ABmag)
        else:
            err_message = f"ACS_WFC data for {band_data.filt_name}" + \
                " must contain either 'ZEROPNT' or 'PHOTFLAM' and 'PHOTPLAM' " + \
                "in its header to calculate its ZP!" # or 'BUNIT'=MJy/sr 
            galfind_logger.critical(err_message)
            raise (Exception(err_message))
        return ZP

    def make_model_PSF(self, band: Union[str, Filter]) -> Type[PSF_Base]:
        pass

    def make_empirical_PSF(self, band_data: Band_Data) -> Type[PSF_Base]:
        pass


class WFC3_IR(Instrument, funcs.Singleton):
    def __init__(self) -> None:
        WFC3_IR_band_names = [
            "F098M",
            "G102",
            "F105W",
            "F110W",
            "F125W",
            "F126N",
            "F127M",
            "F128N",
            "F130N",
            "F132N",
            "F139M",
            "F140W",
            "G141",
            "F153M",
            "F160W",
            "F164N",
            "F167N",
        ]
        self.SVO_name = "WFC3"
        super().__init__("HST", WFC3_IR_band_names)

    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        # Taken from Appendix A of
        # https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/documentation/instrument-science-reports-isrs/_documents/2020/WFC3-ISR-2020-10.pdf
        wfc3ir_zps = {
            "F098M": 25.661,
            "F105W": 26.2637,
            "F110W": 26.8185,
            "F125W": 26.231,
            "F140W": 26.4502,
            "F160W": 25.9362,
        }
        return wfc3ir_zps[band_data.filt.filt_name]

    def make_model_PSF(self, band: Union[str, Filter]) -> Type[PSF_Base]:
        pass

    def make_empirical_PSF(self, band_data: Band_Data) -> Type[PSF_Base]:
        pass


class VISTA(Instrument, funcs.Singleton):
    def __init__(self) -> None:
        VISTA_band_names = [
            "Z_filter",
            "Z",
            "NB980_filter",
            "NB980",
            "NB990_filter",
            "NB990",
            "Y_filter",
            "Y",
            "NB118_filter",
            "NB118",
            "J",
            "J_filter",
            "H",
            "H_filter",
            "Ks_filter",
            "Ks",
        ]
        self.SVO_name = "VIRCam"
        super().__init__("Paranal", VISTA_band_names)

    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        ZP = band_data.load_im()[1]["PHOTZP"]
        return ZP

    def make_model_PSF(self, band: Union[str, Filter]) -> Type[PSF_Base]:
        pass

    def make_empirical_PSF(self, data: Data, band: Union[str, Filter]) -> Type[PSF_Base]:
        pass


class MegaCam(Instrument, funcs.Singleton):
    def __init__(self) -> None:
        Megacam_band_names = [
            "u",
            "u_1",
            "g",
            "g_1",
            "r",
            "r_1",
            "i",
            "i_1",
            "i_2",
            "z",
            "z_1",
            #"gri",
        ]
        self.SVO_name = "MegaCam"
        super().__init__("CFHT", Megacam_band_names)

    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        ZP = band_data.load_im()[1]["PHOTZP"]
        return ZP

    def make_model_PSF(self, band: Union[str, Filter]) -> Type[PSF_Base]:
        pass

    def make_empirical_PSF(self, data: Data, band: Union[str, Filter]) -> Type[PSF_Base]:
        pass

class HSC(Instrument, funcs.Singleton):
    def __init__(self) -> None:
        HSC_band_names = [
            "g",
            "r",
            "i",
            "z",
            "Y",
            "NB387_filter",
            "NB468_filter",
            "g_filter",
            "NB515_filter",
            "r2_filter",
            "r_filter",
            "NB656_filter",
            "NB718_filter",
            "i_filter",
            "i2_filter",
            "NB816_filter",
            "z_filter",
            "NB921_filter",
            "NB926_filter",
            "IB945_filter",
            "NB973_filter",
            "Y_filter",
        ]
        self.SVO_name = "HSC"
        super().__init__("Subaru", HSC_band_names)

    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        ZP = band_data.load_im()[1]["PHOTZP"]
        return ZP
    
    def make_model_PSF(self, band: Union[str, Filter]) -> Type[PSF_Base]:
        pass

    def make_empirical_PSF(self, data: Data, band: Union[str, Filter]) -> Type[PSF_Base]:
        pass


class VIS(Instrument, funcs.Singleton):
    def __init__(self) -> None:
        VIS_band_names = [
            "vis"
        ]
        self.SVO_name = "VIS"
        super().__init__("Euclid", VIS_band_names)

    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        ZP = band_data.load_im()[1]["PHOTZP"]
        return ZP

    def make_model_PSF(self, band: Union[str, Filter]) -> Type[PSF_Base]:
        pass

    def make_empirical_PSF(self, data: Data, band: Union[str, Filter]) -> Type[PSF_Base]:
        pass


class NISP(Instrument, funcs.Singleton):
    def __init__(self) -> None:
        NISP_band_names = [
            "Y",
            "J",
            "H",
        ]
        self.SVO_name = "NISP"
        super().__init__("Euclid", NISP_band_names)

    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        ZP = band_data.load_im()[1]["PHOTZP"]
        return ZP

    def make_model_PSF(self, band: Union[str, Filter]) -> Type[PSF_Base]:
        pass

    def make_empirical_PSF(self, data: Data, band: Union[str, Filter]) -> Type[PSF_Base]:
        pass


class IRAC(Instrument, funcs.Singleton):
    def __init__(self) -> None:
        IRAC_band_names = [
            "I1",
            "I2",
            "I3",
            "I4",
        ]
        self.SVO_name = "IRAC"
        super().__init__("Spitzer", IRAC_band_names)

    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        ZP = band_data.load_im()[1]["PHOTZP"]
        return ZP

    def make_model_PSF(self, band: Union[str, Filter]) -> Type[PSF_Base]:
        pass

    def make_empirical_PSF(self, data: Data, band: Union[str, Filter]) -> Type[PSF_Base]:
        pass


# Instrument attributes

# TODO: Generalize this so the user does not 
# have to update upon the addition of a new instrument
expected_instr_bands = {
    "ACS_WFC": ACS_WFC().filt_names,
    "WFC3_IR": WFC3_IR().filt_names,
    "NIRCam": NIRCam().filt_names,
    "MIRI": MIRI().filt_names,
    "VISTA": VISTA().filt_names,
    "MegaCam": MegaCam().filt_names,
    "HSC": HSC().filt_names,
    "VIS": VIS().filt_names,
    "NISP": NISP().filt_names,
    "IRAC": IRAC().filt_names,
}

expected_instr_facilities = {
    "ACS_WFC": "HST",
    "WFC3_IR": "HST",
    "NIRCam": "JWST",
    "MIRI": "JWST",
    "VISTA": "Paranal",
    "MegaCam": "CFHT",
    "HSC": "Subaru",
    "VIS": "Euclid",
    "NISP": "Euclid",
    "IRAC": "Spitzer",
}


# Instrument.py
from __future__ import annotations

from typing import NoReturn, Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from . import Band_Data, Band_Data_Base, Data
    from . import PSF
    from . import Filter

import astropy.units as u
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

from . import NIRCam_aper_corr, config, galfind_logger
from . import useful_funcs_austind as funcs


class Facility(ABC):
    # Facility class to store the name of the facility
    # and other facility-specific attributes/methods

    def __init__(self) -> None:
        pass

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

    @abstractmethod
    def make_PSF_model(self, band: Union[str, Filter]) -> PSF:
        pass


class HST(Facility, funcs.Singleton):
    def make_PSF_model(self, band: Union[str, Filter]) -> PSF:
        pass


class JWST(Facility, funcs.Singleton):
    def make_PSF_model(self, band: Union[str, Filter]) -> PSF:
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

    def make_PSF(self, band_data: Band_Data, method: str) -> PSF:
        if method == "model":
            # no real data needed for model PSF
            return self.facility.make_PSF_model(band_data.filt)
        elif method == "empirical":
            return self.make_empirical_PSF(band_data)
        else:
            raise NotImplementedError

    def make_PSFs(self, data: Data, method: str) -> List[PSF]:
        return [self.make_PSF(data, band, method) for band in self]

    @abstractmethod
    def calc_ZP(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        pass

    def calc_pix_scale(self, band_data: Type[Band_Data_Base]) -> u.Quantity:
        pass

    @abstractmethod
    def make_empirical_PSF(self, band_data: Band_Data) -> PSF:
        pass

    def make_empirical_PSFs(self, data: Data) -> List[PSF]:
        return [self.make_empirical_PSF(data, band) for band in self]


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

    def make_empirical_PSF(self, band_data: Band_Data) -> PSF:
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

    def make_empirical_PSF(self, data: Data, band: Union[str, Filter]) -> PSF:
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
            "POL_UV",
            "G800L",
            "POL_V",
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
        elif "BUNIT" in im_header:
            unit = im_header["BUNIT"].replace(" ", "")
            assert unit == "MJy/sr"
            ZP = -2.5 * np.log10(
                (band_data.pix_scale.to(u.rad).value ** 2) * u.MJy.to(u.Jy)
            ) + u.Jy.to(u.ABmag)
        else:
            raise (
                Exception(
                    f"ACS_WFC data for {band_data.filt.filt_name}"
                    + " must contain either 'ZEROPNT' or 'PHOTFLAM' and 'PHOTPLAM' "
                    + "or 'BUNIT'=MJy/sr in its header to calculate its ZP!"
                )
            )
        return ZP

    def make_empirical_PSF(self, band_data: Band_Data) -> PSF:
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

    def make_empirical_PSF(self, band_data: Band_Data) -> PSF:
        pass


# Instrument attributes

expected_instr_bands = {
    "ACS_WFC": ACS_WFC().filt_names,
    "WFC3_IR": WFC3_IR().filt_names,
    "NIRCam": NIRCam().filt_names,
    "MIRI": MIRI().filt_names,
}

expected_instr_facilities = {
    "ACS_WFC": "HST",
    "WFC3_IR": "HST",
    "NIRCam": "JWST",
    "MIRI": "JWST",
}

# class Instrument_:

#     @classmethod
#     def from_band_names(
#         cls: Type[Self], band_names: Union[list, np.array, str]
#     ) -> Self:
#         if isinstance(band_names, str):
#             band_names = "+".join(band_names)
#         # invert expected_instr_bands dict
#         expected_band_instr = {
#             band: instr
#             for instr, bands in expected_instr_bands.items()
#             for band in bands
#         }
#         band_instruments = [
#             expected_band_instr[band_name] for band_name in band_names
#         ]
#         band_facilities = [
#             expected_instr_facilities[band_instrument]
#             for band_instrument in band_instruments
#         ]
#         bands = [
#             Filter.from_SVO(band_facility, band_instr, band_name)
#             for band_facility, band_instr, band_name in zip(
#                 band_facilities, band_instruments, band_names
#             )
#         ]
#         return cls.from_bands(bands)

# @property
# def band_FWHMs(self):
#     return (
#         np.array([band.FWHM.to(u.AA).value for band in self.bands]) * u.AA
#     )

# @property # same for upper
# def band_lower_wav_lims(self):
#     return (
#         np.array(
#             [band.WavelengthLower50.to(u.AA).value for band in self.bands]
#         )
#         * u.AA
#     )

# %% Overloaded operators

# def __getitem__(self, i):
#     if type(i) in [int, np.int64, slice]:
#         return self.bands[i]
#     elif type(i) == str:
#         return self.bands[self.index_from_band(i)]
#     else:
#         raise (
#             TypeError(
#                 f"i={i} in {self.__class__.__name__}.__getitem__ has type={type(i)} which is not in [int, slice, str]"
#             )
#         )

# def instrument_from_band(self, band):
#     # Pointless here but makes it compatible with Combined_Instrument
#     if (band.split("+")[0] in self.band_names) or band in self.band_names:
#         return self
#     else:
#         # This leads to confusing errors when passing in a bandname which doesn't exist
#         return False

# def __eq__(self, other):
#     # ensure same type
#     if type(self) != type(other):
#         return False
#     # ensure same facility and instrument name
#     elif self.facility != other.facility or self.name != other.name:
#         return False
#     # ensure same bands are present in both instruments
#     else:
#         if len(self) == len(other) and all(
#             self_band == other_band
#             for self_band, other_band in zip(self, other)
#         ):
#             return True
#         else:
#             return False

# # %% Class properties

# @property
# def bands_from_wavelengths(self):
#     return {value: key for key, value in self.band_wavelengths.items()}

# @property
# def band_wavelength_errs(self):
#     return {
#         key: value / 2 for key, value in self.band_FWHMs.items()
#     }  # = FWHM/2 in Angstrom

# # %% Other class methods

# def remove_band(self, band_name: str) -> NoReturn:
#     assert type(band_name) in [str, np.str_], galfind_logger.critical(
#         f"{band_name=} with {type(band_name)=} not in ['str', 'np.str_']"
#     )
#     assert band_name in self.band_names, galfind_logger.critical(
#         f"{band_name=} not in {self.band_names=}"
#     )
#     self.remove_index(self.index_from_band_name(band_name))
#     return self

# def remove_bands(self, band_names: str) -> type[Self]:
#     assert all(band in self.band_names for band in band_names)
#     remove_indices = self.indices_from_band_names(band_names)
#     if remove_indices != []:
#         self.remove_indices(remove_indices)
#     return self

# def remove_index(self, remove_index: int) -> NoReturn:
#     if not type(remove_index) == type(None):
#         self.bands = np.delete(self.bands, remove_index)
#     return self

# def remove_indices(self, remove_indices: list) -> NoReturn:
#     if not type(remove_indices) == type(None):
#         self.bands = np.delete(self.bands, remove_indices)
#     return self

# def index_from_band_name(self, band_name: str) -> Union[int, None]:
#     if band_name in self.band_names:
#         return np.where(self.band_names == band_name)[0][0]
#     else:
#         return None

# def indices_from_band_names(self, band_names: list) -> list:
#     return [
#         self.index_from_band_name(band_name) for band_name in band_names
#     ]

# def band_name_from_index(self, index) -> str:
#     return self.band_names[index]

# def bands_from_wavelength(self, wavelength) -> list[Filter]:
#     return [
#         band
#         for band in self
#         if wavelength > band.WavelengthLower50
#         and wavelength < band.WavelengthUpper50
#     ]

# def nearest_band_to_wavelength(
#     self,
#     wavelength,
#     medium_bands_only=False,
#     check_wavelength_in_band=True,
# ) -> Union[Filter, None]:
#     if medium_bands_only:
#         search_bands = [band for band in self if "M" == band.band_name[-1]]
#     else:
#         search_bands = self.bands
#     nearest_band = search_bands[
#         np.abs(
#             [
#                 funcs.convert_wav_units(band.WavelengthCen, u.AA).value
#                 for band in search_bands
#             ]
#             - funcs.convert_wav_units(wavelength, u.AA).value
#         ).argmin()
#     ]
#     if (
#         check_wavelength_in_band
#         and nearest_band not in self.bands_from_wavelength(wavelength)
#     ):
#         return None
#     else:
#         return nearest_band

# def bands_avoiding_wavs(self, wavs):
#     # extract the unique band names
#     unique_band_names = np.unique(
#         np.array(
#             [
#                 band.band_name
#                 for wav in wavs
#                 for band in self.bands_from_wavelength(wav)
#             ]
#         )
#     )
#     # return an array of Filter objects corresponding to the band names
#     return np.array(
#         [
#             self[self.index_from_band_name(band_name)]
#             for band_name in self.band_names
#             if band_name not in unique_band_names
#         ]
#     )

# def get_aper_corrs(self, aper_diam: u.Quantity, cache: bool = True):
#     # load aperture correction from object should it exist
#     if hasattr(self, "aper_corrs"):
#         assert type(self.aper_corrs) in [dict]
#         if aper_diam in self.aper_corrs.keys():
#             return self.aper_corrs[aper_diam]
#     else:
#         self.aper_corrs = {}
#     if self.name in globals():
#         assert globals()[self.name] in Instrument.__subclasses__()

#     # TEMPORARY for F444W PSF-homogenized.
#     if config.getboolean("DataReduction", "PSF_HOMOGENIZED"):
#         galfind_logger.warning(
#             "Temporary aperture correction for F444W PSF-homogenized"
#         )
#         aper_corr = {
#             0.2: 0.8259490203345642,
#             0.32: 0.4691650961749638,
#             0.5: 0.31013820485999116,
#             1.0: 0.16205011984149864,
#             1.5: 0.10985494543655024,
#             2.0: 0.09046584834037462,
#         }
#         aper_corrs = [aper_corr[aper_diam.to(u.arcsec).value]] * len(self)
#         if cache:
#             self.aper_corrs[aper_diam] = aper_corrs
#         return aper_corrs

#     aper_corr_path = f'{config["Other"]["GALFIND_DIR"]}/Aperture_corrections/{self.name}_aper_corr{".txt" if self.name in ["NIRCam", "MIRI"] else ".dat"}'
#     # if no aperture corrections in object, load from aperture corrections txt
#     if Path(aper_corr_path).is_file():
#         aper_corr_data = np.loadtxt(
#             aper_corr_path,
#             comments="#",
#             dtype=[
#                 ("band", "U10"),
#                 ("0.32", "f4"),
#                 ("0.5", "f4"),
#                 ("1.0", "f4"),
#                 ("1.5", "f4"),
#                 ("2.0", "f4"),
#             ],
#         )
#         if (
#             all(
#                 [
#                     True if band_name in aper_corr_data["band"] else False
#                     for band_name in self.band_names
#                 ]
#             )
#             and str(aper_diam.to(u.arcsec).value)
#             in aper_corr_data.dtype.names[1:]
#         ):
#             band_indices = [
#                 list(aper_corr_data["band"]).index(band_name)
#                 for band_name in self.band_names
#             ]
#             aper_corrs = list(
#                 aper_corr_data[str(aper_diam.to(u.arcsec).value)][
#                     band_indices
#                 ]
#             )
#             if cache:
#                 self.aper_corrs[aper_diam] = aper_corrs
#             return aper_corrs
#         else:
#             raise (Exception())
#     else:
#         # THIS CODE IS NOT AT ALL GENERAL!
#         # if no aperture corrections txt, create it
#         if "+" not in self.name:
#             NIRCam_aper_corr.main(
#                 self.new_instrument().band_names, instrument_name=self.name
#             )

#     @staticmethod
#     def from_name(name, excl_bands=[]):
#         if name == "NIRCam":
#             return NIRCam(excl_bands=excl_bands)
#         elif name == "MIRI":
#             return MIRI(excl_bands=excl_bands)
#         elif name == "ACS_WFC":
#             return ACS_WFC(excl_bands=excl_bands)
#         elif name == "WFC3_IR":
#             return WFC3_IR(excl_bands=excl_bands)
#         elif "+" in name:
#             new_instruments = Combined_Instrument.instruments_from_name(
#                 name, excl_bands=excl_bands
#             )
#             for i, instrument in enumerate(new_instruments):
#                 if i == 0:
#                     new_instrument = instrument
#                 else:
#                     new_instrument += instrument
#             return new_instrument
#         else:
#             raise (
#                 Exception(
#                     f"Instrument name: {name} does not exist in 'Instrument.from_name()'!"
#                 )
#             )

# class Combined_Instrument(Instrument):
#     def __init__(self, name, bands, excl_bands, facility):
#         super().__init__(name, bands, excl_bands, facility)

#     @classmethod
#     def instruments_from_name(cls, name, excl_bands=[]):
#         combined_instrument_names = name.split("+")
#         return [
#             cls.from_name(combined_instrument_name, excl_bands)
#             for combined_instrument_name in combined_instrument_names
#         ]

#     @classmethod
#     def combined_instrument_from_name(cls, name, excl_bands=[]):
#         return cls.from_name(name, excl_bands)

#     def get_bands_from_instrument(self, instrument_name):
#         assert instrument_name in [
#             subcls.__name__
#             for subcls in Instrument.__subclasses__()
#             if subcls.__name__ != "Combined_Instrument"
#         ]
#         return [band for band in self if band.instrument == instrument_name]

#     def get_aper_corrs(self, aper_diam, cache=True):
#         # load from object if already calculated
#         if hasattr(self, "aper_corrs"):
#             assert type(self.aper_corrs) in [dict]
#             if aper_diam in self.aper_corrs.keys():
#                 return self.aper_corrs[aper_diam]
#         else:
#             self.aper_corrs = {}
#         # calculate aperture corrections for each instrument
#         instrument_arr = [
#             globals()[name](
#                 excl_bands=[
#                     band_name
#                     for band_name in globals()[name]().band_names
#                     if band_name
#                     not in [
#                         band.band_name
#                         for band in self.get_bands_from_instrument(name)
#                     ]
#                 ]
#             )
#             for name in self.name.split("+")
#         ]
#         instrument_band_names = np.hstack(
#             [instrument.band_names for instrument in instrument_arr]
#         )
#         aper_corrs = np.hstack(
#             [
#                 instrument.get_aper_corrs(aper_diam, cache=False)
#                 for instrument in instrument_arr
#             ]
#         )
#         # re-order aperture corrections
#         band_aper_corr_dict = {
#             band_name: aper_corr
#             for band_name, aper_corr in zip(instrument_band_names, aper_corrs)
#         }
#         _aper_corrs = [
#             band_aper_corr_dict[band_name] for band_name in self.band_names
#         ]
#         if cache:  # save in self
#             self.aper_corrs[aper_diam] = _aper_corrs
#         # breakpoint()
#         return _aper_corrs

#     def instrument_from_band(self, band, return_name=True):
#         names = self.name.split("+")
#         for name in names:
#             instrument = Instrument.from_name(name)
#             if instrument.instrument_from_band(band) != False:
#                 return instrument

#     def new_instrument(self, excl_bands=[]):
#         instruments = self.instruments_from_name(self.name, excl_bands)
#         for i, instrument in enumerate(instruments):
#             if i == 0:
#                 new_instrument = instrument
#             else:
#                 new_instrument += instrument
#         # print("Still need to understand shallow and deep copy constructors in python!")
#         return new_instrument


# aper_diams_sex = [0.32, 0.5, 1., 1.5, 2.] * u.arcsec

# def return_loc_depth_mags(
#     cat, band, incl_err=True, mag_type=["APER", 0], min_flux_err_pc=5
# ):
#     # insert minimum percentage error
#     min_mag_err = funcs.flux_pc_to_mag_err(min_flux_err_pc)
#     if "AUTO" in mag_type:
#         # data = np.array(cat[f"MAG_AUTO_{band}"])
#         raise NotImplementedError
#     elif "APER" in mag_type:
#         data = np.array(
#             cat[f"MAG_{mag_type[0]}_{band}_aper_corr"].T[mag_type[1]]
#         )
#         # data_l1 = np.array(cat[f"MAGERR_{mag_type[0]}_{band}_l1_loc_depth"][mag_type[1]]
#         data_l1 = np.array(
#             [
#                 val if val > min_mag_err else min_mag_err
#                 for val in [
#                     cat[f"MAGERR_{mag_type[0]}_{band}_l1_loc_depth"].T[
#                         mag_type[1]
#                     ]
#                 ]
#             ]
#         )
#         data_u1 = np.array(
#             [
#                 val if val > min_mag_err else min_mag_err
#                 for val in [
#                     cat[f"MAGERR_{mag_type[0]}_{band}_u1_loc_depth"].T[
#                         mag_type[1]
#                     ]
#                 ]
#             ]
#         )
#         data_errs = np.array([data_l1, data_u1])
#     else:
#         raise (SyntaxError("Invalid mag_type in return_loc_depth_mags"))
#     if incl_err:
#         return data, data_errs
#     else:
#         return data

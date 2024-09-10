#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 12:56:53 2023

@author: austind
"""

# Instrument.py
from __future__ import absolute_import

import json
import warnings
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import NoReturn, Union, List

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astroquery.svo_fps import SvoFps

try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import NIRCam_aper_corr, config, galfind_logger
from . import useful_funcs_austind as funcs
from .Filter import Filter

# Instrument attributes

ACS_WFC_bands = [
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
    "FR601N",
    "F606W",
    "F625W",
    "FR647M",
    "FR656N",
    "F658N",
    "F660N",
    "FR716N",
    "POL_UV",
    "POL_V",
    "G800L",
    "F775W",
    "FR782N",
    "F814W",
    "FR853N",
    "F892N",
    "FR914M",
    "F850LP",
    "FR931N",
    "FR1016N",
]
WFC3_IR_bands = [
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
NIRCam_bands = [
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
MIRI_bands = [
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
expected_instr_bands = {
    "ACS_WFC": ACS_WFC_bands,
    "WFC3_IR": WFC3_IR_bands,
    "NIRCam": NIRCam_bands,
    "MIRI": MIRI_bands,
}

expected_instr_facilities = {
    "ACS_WFC": "HST",
    "WFC3_IR": "HST",
    "NIRCam": "JWST",
    "MIRI": "JWST",
}


class Instrument:
    def __init__(
        self, name, bands, excl_bands, facility, band_order="ascending"
    ):
        self.bands = np.array(self.sort_bands(bands, order=band_order))
        self.name = name
        for band in excl_bands:
            self.remove_band(band)
        self.facility = facility
        # TODO: calculate name and facility on the fly from band information (and overridden getattr)
        # TODO: make this Instrument class a base for NIRCam/ACS_WFC/etc singletons containing PSF info./methods;
        #       there should then be Multiple_Filter object which is the equivalent of this
        # TODO: excl_bands instead included in 2 class methods (Multiple_Filter.from_instrument(str) AND Multiple_Filter.from_facility(str))

    @staticmethod
    def sort_bands(bands, order="ascending"):
        if order == "ascending":
            # sort bands from blue -> red
            bands = [
                band
                for band in sorted(
                    bands,
                    key=lambda band: band.WavelengthCen.to(u.AA).value,
                )
            ]
        else:
            raise NotImplementedError
        return bands

    @classmethod
    def from_SVO(cls, facility, instrument, excl_bands=[]):
        filter_list = SvoFps.get_filter_list(
            facility=facility, instrument=instrument.split("_")[0]
        )
        filter_list = filter_list[
            np.array(
                [
                    filter_name.split("/")[-1].split(".")[0]
                    for filter_name in np.array(filter_list["filterID"])
                ]
            )
            == instrument
        ]
        bands = np.array(
            [
                Filter.from_SVO(
                    facility,
                    instrument,
                    filt_ID.replace(f"{facility}/{instrument}.", ""),
                )
                for filt_ID in np.array(filter_list["filterID"])
            ]
        )
        return cls(instrument, bands, excl_bands, facility)

    @classmethod
    def from_band_names(
        cls: Type[Self], band_names: Union[list, np.array, str]
    ) -> Self:
        if isinstance(band_names, str):
            band_names = "+".join(band_names)
        # invert expected_instr_bands dict
        expected_band_instr = {
            band: instr
            for instr, bands in expected_instr_bands.items()
            for band in bands
        }
        band_instruments = [
            expected_band_instr[band_name] for band_name in band_names
        ]
        band_facilities = [
            expected_instr_facilities[band_instrument]
            for band_instrument in band_instruments
        ]
        bands = [
            Filter.from_SVO(band_facility, band_instr, band_name)
            for band_facility, band_instr, band_name in zip(
                band_facilities, band_instruments, band_names
            )
        ]
        return cls.from_bands(bands)

    @classmethod
    def from_bands(cls: Type[Self], bands: List[Filter]) -> Self:
        # sort bands from blue -> red
        bands = cls.sort_bands(bands)
        # extract instrument and facility name for each band and hence the class
        name = "+".join(np.unique([band.instrument for band in bands]))
        facility = "+".join(np.unique([band.facility for band in bands]))
        return cls(name, bands, [], facility)

    @property
    def band_names(self):
        return np.array([band.band_name for band in self.bands])

    @property
    def band_wavelengths(self):
        # Central wavelengths
        return (
            np.array(
                [band.WavelengthCen.to(u.AA).value for band in self.bands]
            )
            * u.AA
        )

    @property
    def band_FWHMs(self):
        return (
            np.array([band.FWHM.to(u.AA).value for band in self.bands]) * u.AA
        )

    @property
    def band_lower_wav_lims(self):
        return (
            np.array(
                [band.WavelengthLower50.to(u.AA).value for band in self.bands]
            )
            * u.AA
        )

    @property
    def band_upper_wav_lims(self):
        return (
            np.array(
                [band.WavelengthUpper50.to(u.AA).value for band in self.bands]
            )
            * u.AA
        )

    # %% Overloaded operators

    def __str__(self):
        """Function to print summary of Instrument class

        Returns:
            str: Summary containing facility, instrument name and filter set included in the instrument
        """
        line_sep = "*" * 40 + "\n"
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
        if isinstance(i, (int, np.int64, slice)):
            return self.bands[i]
        elif isinstance(i, str):
            return self.bands[self.index_from_band(i)]
        else:
            raise (
                TypeError(
                    f"i={i} in {self.__class__.__name__}.__getitem__ has type={type(i)} which is not in [int, slice, str]"
                )
            )

    def __del__(self):
        self.bands = []

    def instrument_from_band(self, band):
        # Pointless here but makes it compatible with Combined_Instrument
        if (band.split("+")[0] in self.band_names) or band in self.band_names:
            return self
        else:
            # This leads to confusing errors when passing in a bandname which doesn't exist
            return None

    # def __getitem__(self, get_index): # create a new instrument with only the indexed band
    #     excl_bands = []
    #     for index, band in enumerate(self):
    #         if index != get_index:
    #             excl_bands.append(band)
    #     return self.new_instrument(excl_bands)

    def __add__(self, other):
        # cannot add multiple of the same bands together!!! (maybe could just ignore the problem \
        # in Instrument class and just handle it in Data, possibly stacking the two images)

        if isinstance(other, tuple(Instrument.__subclasses__())):
            new_bands = np.array(
                [
                    band
                    for band in other
                    if band.band_name not in self.band_names
                ]
            )
        elif isinstance(other, Filter):
            if other.band_name not in self.band_names:
                new_bands = np.array([other])
            else:
                new_bands = np.array([])
        else:
            galfind_logger.critical(f"{type(other)=} not in \
                {[instr.__name__ for instr in Instrument.__subclasses__()] + ['Filter']}")
            raise (Exception())

        if len(new_bands) == 0:
            # produce warning message
            warning_message = (
                "No new bands to add, returning self from Instrument.__add__"
            )
            galfind_logger.warning(warning_message)
            warnings.warn(UserWarning(warning_message))
            # nothing else to do
            return deepcopy(self)
        elif len(new_bands) < len(other):
            # produce warning message
            warning_message = f"Not all bands in {other.band_names=} in {self.band_names=}, adding only {new_bands=}"
            galfind_logger.warning(warning_message)
            warnings.warn(UserWarning(warning_message))

        # add and sort bands from blue -> red
        bands = [
            band
            for band in sorted(
                np.concatenate([self.bands, new_bands]),
                key=lambda band: band.WavelengthCen.to(u.AA).value,
            )
        ]

        if type(other) in Instrument.__subclasses__():
            other_instr_name = other.name
        else:  # type == Filter
            other_instr_name = other.instrument

        if all(name in self.name for name in other_instr_name.split("+")):
            self.bands = bands
            out_instrument = deepcopy(self)
        else:  # re-compute blue -> red instrument_name and facility
            all_instruments = json.loads(
                config.get("Other", "INSTRUMENT_NAMES")
            )
            all_facilities = json.loads(config.get("Other", "TELESCOPE_NAMES"))
            # split self.name and instrument.name
            names = list(
                set(self.name.split("+") + other_instr_name.split("+"))
            )
            facilities = list(
                set(self.facility.split("+") + other.facility.split("+"))
            )
            name = "+".join(
                [name for name in all_instruments if name in names]
            )
            facility = "+".join(
                [
                    facility
                    for facility in all_facilities
                    if facility in facilities
                ]
            )
            out_instrument = Combined_Instrument(
                name, bands, excl_bands=[], facility=facility
            )
        return out_instrument

    def __sub__(self, instrument):
        print(
            "Note that 'Instrument.__sub__()' only removes common bands between the two 'instrument' classes"
        )
        for band in instrument.band_names:
            if band in self.band_names:
                self.remove_band(band)
            else:
                print(f"Cannot remove {band} from {self.name}!")
        return self

    def __eq__(self, other):
        # ensure same type
        if not isinstance(self, other):
            return False
        # ensure same facility and instrument name
        elif self.facility != other.facility or self.name != other.name:
            return False
        # ensure same bands are present in both instruments
        else:
            if len(self) == len(other) and all(
                self_band == other_band
                for self_band, other_band in zip(self, other)
            ):
                return True
            else:
                return False

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
        return {
            key: value / 2 for key, value in self.band_FWHMs.items()
        }  # = FWHM/2 in Angstrom

    # %% Class abstract methods

    @abstractmethod
    def new_instrument(self, excl_bands):
        pass

    # %% Other class methods

    def remove_band(self, band_name: str) -> NoReturn:
        assert type(band_name) in [str, np.str_], galfind_logger.critical(
            f"{band_name=} with {type(band_name)=} not in ['str', 'np.str_']"
        )
        assert band_name in self.band_names, galfind_logger.critical(
            f"{band_name=} not in {self.band_names=}"
        )
        self.remove_index(self.index_from_band_name(band_name))
        return self

    def remove_bands(self, band_names: str) -> "Instrument":
        assert all(band in self.band_names for band in band_names)
        remove_indices = self.indices_from_band_names(band_names)
        if remove_indices != []:
            self.remove_indices(remove_indices)
        return self

    def remove_index(self, remove_index: int) -> NoReturn:
        if remove_index is not None:
            self.bands = np.delete(self.bands, remove_index)
        return self

    def remove_indices(self, remove_indices: list) -> NoReturn:
        if remove_indices is not None:
            self.bands = np.delete(self.bands, remove_indices)
        return self

    def index_from_band_name(self, band_name: str) -> Union[int, None]:
        if band_name in self.band_names:
            return np.where(self.band_names == band_name)[0][0]
        else:
            return None

    def indices_from_band_names(self, band_names: list) -> list:
        return [
            self.index_from_band_name(band_name) for band_name in band_names
        ]

    def band_name_from_index(self, index) -> str:
        return self.band_names[index]

    def bands_from_wavelength(self, wavelength) -> list[Filter]:
        return [
            band
            for band in self
            if wavelength > band.WavelengthLower50
            and wavelength < band.WavelengthUpper50
        ]

    def nearest_band_to_wavelength(
        self,
        wavelength,
        medium_bands_only=False,
        check_wavelength_in_band=True,
    ) -> Union[Filter, None]:
        if medium_bands_only:
            search_bands = [band for band in self if "M" == band.band_name[-1]]
        else:
            search_bands = self.bands
        nearest_band = search_bands[
            np.abs(
                [
                    funcs.convert_wav_units(band.WavelengthCen, u.AA).value
                    for band in search_bands
                ]
                - funcs.convert_wav_units(wavelength, u.AA).value
            ).argmin()
        ]
        if (
            check_wavelength_in_band
            and nearest_band not in self.bands_from_wavelength(wavelength)
        ):
            return None
        else:
            return nearest_band

    def bands_avoiding_wavs(self, wavs):
        # extract the unique band names
        unique_band_names = np.unique(
            np.array(
                [
                    band.band_name
                    for wav in wavs
                    for band in self.bands_from_wavelength(wav)
                ]
            )
        )
        # return an array of Filter objects corresponding to the band names
        return np.array(
            [
                self[self.index_from_band_name(band_name)]
                for band_name in self.band_names
                if band_name not in unique_band_names
            ]
        )

    def get_aper_corrs(self, aper_diam, cache=True):
        # print(self.name)
        # breakpoint()
        # load aperture correction from object should it exist
        if hasattr(self, "aper_corrs"):
            assert type(self.aper_corrs) in [dict]
            if aper_diam in self.aper_corrs.keys():
                return self.aper_corrs[aper_diam]
        else:
            self.aper_corrs = {}
        if self.name in globals():
            assert globals()[self.name] in Instrument.__subclasses__()

        # HACK: for F444W PSF-homogenized.
        if config.getboolean("DataReduction", "PSF_HOMOGENIZED"):
            galfind_logger.warning(
                "Temporary aperture correction for F444W PSF-homogenized"
            )
            aper_corr = {
                0.2: 0.8259490203345642,
                0.32: 0.4691650961749638,
                0.5: 0.31013820485999116,
                1.0: 0.16205011984149864,
                1.5: 0.10985494543655024,
                2.0: 0.09046584834037462,
            }
            aper_corrs = [aper_corr[aper_diam.to(u.arcsec).value]] * len(self)
            if cache:
                self.aper_corrs[aper_diam] = aper_corrs
            return aper_corrs

        aper_corr_path = f'{config["Other"]["GALFIND_DIR"]}/Aperture_corrections/{self.name}_aper_corr{".txt" if self.name in ["NIRCam", "MIRI"] else ".dat"}'
        # if no aperture corrections in object, load from aperture corrections txt
        if Path(aper_corr_path).is_file():
            aper_corr_data = np.loadtxt(
                aper_corr_path,
                comments="#",
                dtype=[
                    ("band", "U10"),
                    ("0.32", "f4"),
                    ("0.5", "f4"),
                    ("1.0", "f4"),
                    ("1.5", "f4"),
                    ("2.0", "f4"),
                ],
            )
            if (
                all(
                    [
                        True if band_name in aper_corr_data["band"] else False
                        for band_name in self.band_names
                    ]
                )
                and str(aper_diam.to(u.arcsec).value)
                in aper_corr_data.dtype.names[1:]
            ):
                band_indices = [
                    list(aper_corr_data["band"]).index(band_name)
                    for band_name in self.band_names
                ]
                aper_corrs = list(
                    aper_corr_data[str(aper_diam.to(u.arcsec).value)][
                        band_indices
                    ]
                )
                if cache:
                    self.aper_corrs[aper_diam] = aper_corrs
                return aper_corrs
            else:
                raise (Exception())
        else:
            # THIS CODE IS NOT AT ALL GENERAL!
            # if no aperture corrections txt, create it
            if "+" not in self.name:
                NIRCam_aper_corr.main(
                    self.new_instrument().band_names, instrument_name=self.name
                )

    # eww TODO
    @staticmethod
    def from_name(name, excl_bands=[]):
        if name == "NIRCam":
            return NIRCam(excl_bands=excl_bands)
        elif name == "MIRI":
            return MIRI(excl_bands=excl_bands)
        elif name == "ACS_WFC":
            return ACS_WFC(excl_bands=excl_bands)
        elif name == "WFC3_IR":
            return WFC3_IR(excl_bands=excl_bands)
        elif "+" in name:
            new_instruments = Combined_Instrument.instruments_from_name(
                name, excl_bands=excl_bands
            )
            for i, instrument in enumerate(new_instruments):
                if i == 0:
                    new_instrument = instrument
                else:
                    new_instrument += instrument
            return new_instrument
        else:
            raise (
                Exception(
                    f"Instrument name: {name} does not exist in 'Instrument.from_name()'!"
                )
            )

    def plot_filter_profiles(
        self,
        ax,
        wav_units=u.um,
        from_SVO=True,
        cmap_name="Spectral_r",
        annotate=True,
        show=True,
        save=False,
    ) -> NoReturn:
        cmap = plt.get_cmap(cmap_name, len(self))
        for i, band in enumerate(self):
            band.plot_filter_profile(ax, from_SVO=from_SVO, color=cmap[i])
        if annotate:
            ax.set_title(f"{self.name} filters")
            ax.set_xlabel(
                r"$\lambda_{\mathrm{obs}}$ / "
                + funcs.unit_labels_dict[wav_units]
            )
            ax.set_ylabel("Transmission")
            ax.set_ylim(
                0.0,
                np.max([trans for trans in band.trans for band in self]) + 0.1,
            )
        if save:
            plt.savefig(f"{self.name}_filter_profiles.png")
            funcs.change_file_permissions(f"{self.name}_filter_profiles.png")
        if show:
            plt.show()


class NIRCam(Instrument):
    def __init__(self, excl_bands=[]):
        instr = Instrument.from_SVO(
            "JWST", self.__class__.__name__, excl_bands=[]
        )
        super().__init__(instr.name, instr.bands, excl_bands, instr.facility)

    def new_instrument(self, excl_bands=[]):
        return NIRCam(excl_bands)

    @classmethod
    def from_bands(
        cls: Type[Self], bands: List[Filter]
    ) -> NotImplementedError:
        raise NotImplementedError


class MIRI(Instrument):
    def __init__(self, excl_bands=[]):
        instr = Instrument.from_SVO(
            "JWST", self.__class__.__name__, excl_bands=[]
        )
        super().__init__(instr.name, instr.bands, excl_bands, instr.facility)

    def new_instrument(self, excl_bands=[]):
        return MIRI(excl_bands)

    @classmethod
    def from_bands(
        cls: Type[Self], bands: List[Filter]
    ) -> NotImplementedError:
        raise NotImplementedError


class ACS_WFC(Instrument):
    def __init__(self, excl_bands=[]):
        instr = Instrument.from_SVO(
            "HST", self.__class__.__name__, excl_bands=[]
        )
        super().__init__(instr.name, instr.bands, excl_bands, instr.facility)

    def new_instrument(self, excl_bands=[]):
        return ACS_WFC(excl_bands)

    @classmethod
    def from_bands(
        cls: Type[Self], bands: List[Filter]
    ) -> NotImplementedError:
        raise NotImplementedError


class WFC3_IR(Instrument):
    def __init__(self, excl_bands=[]):
        instr = Instrument.from_SVO(
            "HST", self.__class__.__name__, excl_bands=[]
        )
        super().__init__(instr.name, instr.bands, excl_bands, instr.facility)

    def new_instrument(self, excl_bands=[]):
        return WFC3_IR(excl_bands)

    @classmethod
    def from_bands(
        cls: Type[Self], bands: List[Filter]
    ) -> NotImplementedError:
        raise NotImplementedError


class Combined_Instrument(Instrument):
    def __init__(self, name, bands, excl_bands, facility):
        super().__init__(name, bands, excl_bands, facility)

    @classmethod
    def instruments_from_name(cls, name, excl_bands=[]):
        combined_instrument_names = name.split("+")
        return [
            cls.from_name(combined_instrument_name, excl_bands)
            for combined_instrument_name in combined_instrument_names
        ]

    @classmethod
    def combined_instrument_from_name(cls, name, excl_bands=[]):
        return cls.from_name(name, excl_bands)

    def get_bands_from_instrument(self, instrument_name):
        assert instrument_name in [
            subcls.__name__
            for subcls in Instrument.__subclasses__()
            if subcls.__name__ != "Combined_Instrument"
        ]
        return [band for band in self if band.instrument == instrument_name]

    def get_aper_corrs(self, aper_diam, cache=True):
        # load from object if already calculated
        if hasattr(self, "aper_corrs"):
            assert type(self.aper_corrs) in [dict]
            if aper_diam in self.aper_corrs.keys():
                return self.aper_corrs[aper_diam]
        else:
            self.aper_corrs = {}
        # calculate aperture corrections for each instrument
        instrument_arr = [
            globals()[name](
                excl_bands=[
                    band_name
                    for band_name in globals()[name]().band_names
                    if band_name
                    not in [
                        band.band_name
                        for band in self.get_bands_from_instrument(name)
                    ]
                ]
            )
            for name in self.name.split("+")
        ]
        instrument_band_names = np.hstack(
            [instrument.band_names for instrument in instrument_arr]
        )
        aper_corrs = np.hstack(
            [
                instrument.get_aper_corrs(aper_diam, cache=False)
                for instrument in instrument_arr
            ]
        )
        # re-order aperture corrections
        band_aper_corr_dict = {
            band_name: aper_corr
            for band_name, aper_corr in zip(instrument_band_names, aper_corrs)
        }
        _aper_corrs = [
            band_aper_corr_dict[band_name] for band_name in self.band_names
        ]
        if cache:  # save in self
            self.aper_corrs[aper_diam] = _aper_corrs
        # breakpoint()
        return _aper_corrs

    def instrument_from_band(self, band, return_name=True):
        names = self.name.split("+")
        for name in names:
            instrument = Instrument.from_name(name)
            if instrument.instrument_from_band(band) is not None:
                return instrument

    def new_instrument(self, excl_bands=[]):
        instruments = self.instruments_from_name(self.name, excl_bands)
        for i, instrument in enumerate(instruments):
            if i == 0:
                new_instrument = instrument
            else:
                new_instrument += instrument
        # print("Still need to understand shallow and deep copy constructors in python!")
        return new_instrument


# aper_diams_sex = [0.32, 0.5, 1., 1.5, 2.] * u.arcsec


def return_loc_depth_mags(
    cat, band, incl_err=True, mag_type=["APER", 0], min_flux_err_pc=5
):
    # insert minimum percentage error
    min_mag_err = funcs.flux_pc_to_mag_err(min_flux_err_pc)
    if "AUTO" in mag_type:
        # data = np.array(cat[f"MAG_AUTO_{band}"])
        raise NotImplementedError
    elif "APER" in mag_type:
        data = np.array(
            cat[f"MAG_{mag_type[0]}_{band}_aper_corr"].T[mag_type[1]]
        )
        # data_l1 = np.array(cat[f"MAGERR_{mag_type[0]}_{band}_l1_loc_depth"][mag_type[1]]
        data_l1 = np.array(
            [
                val if val > min_mag_err else min_mag_err
                for val in [
                    cat[f"MAGERR_{mag_type[0]}_{band}_l1_loc_depth"].T[
                        mag_type[1]
                    ]
                ]
            ]
        )
        data_u1 = np.array(
            [
                val if val > min_mag_err else min_mag_err
                for val in [
                    cat[f"MAGERR_{mag_type[0]}_{band}_u1_loc_depth"].T[
                        mag_type[1]
                    ]
                ]
            ]
        )
        data_errs = np.array([data_l1, data_u1])
    else:
        raise (SyntaxError("Invalid mag_type in return_loc_depth_mags"))
    if incl_err:
        return data, data_errs
    else:
        return data

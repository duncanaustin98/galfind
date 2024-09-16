# Filter.py
from __future__ import annotations

import astropy.units as u
import numpy as np
from astroquery.svo_fps import SvoFps
from copy import deepcopy
import json
import matplotlib as plt
from typing import List, Union, NoReturn, TYPE_CHECKING

if TYPE_CHECKING:
    from . import Facility, Instrument, Data
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import galfind_logger, config
from . import useful_funcs_austind as funcs
from . import Facility, Instrument
from . import ACS_WFC, WFC3_IR, NIRCam, MIRI, JWST, HST  # noqa: F401


class Filter:
    def __init__(
        self,
        facility: Union[None, str, Facility],
        instrument: Union[None, str, Instrument],
        band_name: str,
        wav: u.Quantity,
        trans: List[float],
        properties: dict = {},
    ):
        assert len(wav) == len(trans)
        if isinstance(facility, str):
            facility = globals()[facility]()
            assert isinstance(facility, tuple(Facility.__subclasses__()))
        self.facility = facility
        if isinstance(instrument, str):
            instrument = globals()[instrument]()
        self.instrument = instrument
        self.band_name = band_name
        self.wav = wav
        self.trans = trans
        for key, value in properties.items():
            self.__setattr__(key, value)
        self.properties = properties  # currently just used in __str__ only

    @classmethod
    def from_SVO(cls, facility, instrument, filter_name):
        full_name = f"{facility}/{instrument}.{filter_name}"
        try:
            filter_profile = SvoFps.get_transmission_data(full_name)
        except:
            galfind_logger.critical(
                f"{full_name} is not a valid SvoFps filter!"
            )
        wav = np.array(filter_profile["Wavelength"])
        trans = np.array(filter_profile["Transmission"])
        properties = SvoFps.data_from_svo(
            query={
                "Facility": facility,
                "Instrument": instrument.split("_")[0],
            },
            error_msg=f"'{facility}/{instrument.split('_')[0]}' is not a valid facility/instrument combination!",
        )
        properties = properties[properties["filterID"] == full_name]
        wav *= u.Unit(str(np.array(properties["WavelengthUnit"])[0]))

        output_prop = {}
        for key, value in properties.items():
            if (
                "Wavelength" in key or key in ["WidthEff", "FWHM"]
            ) and key not in ["WavelengthUnit", "WavelengthUCD"]:
                output_prop[key] = float(np.array(value)[0]) * u.Unit(
                    str(np.array(properties["WavelengthUnit"])[0])
                )
            elif key in ["Description", "Comments"]:
                output_prop[key] = str(np.array(value)[0])
            elif key == "DetectorType":
                detector_type_dict = {1: "photon counter"}
                output_prop[key] = str(
                    detector_type_dict[int(np.array(value)[0])]
                )
        output_prop["WavelengthUpper50"] = (
            output_prop["WavelengthCen"] + output_prop["FWHM"] / 2.0
        )
        output_prop["WavelengthLower50"] = (
            output_prop["WavelengthCen"] - output_prop["FWHM"] / 2.0
        )
        return cls(facility, instrument, filter_name, wav, trans, output_prop)

    def __str__(self):
        output_str = funcs.line_sep
        output_str += f"{self.facility}/{self.instrument}/{self.band_name}\n"
        output_str += funcs.band_sep
        for key, value in self.properties.items():
            output_str += f"{key}: {value}\n"
        output_str += funcs.line_sep
        return output_str

    def __len__(self):
        return 1

    def __eq__(self, other):
        # ensure types are the same
        if type(self) != type(other):
            return False
        # ensure both have the same attribute keys
        elif not all(
            other_key in self.__dict__.keys()
            for other_key in other.__dict__.keys()
        ) or not all(
            self_key in other.__dict__.keys()
            for self_key in self.__dict__.keys()
        ):
            return False
        # ensure both have matching wav and trans arrays
        elif (
            len(self.wav) != len(other.wav)
            or any(
                self_wav != other_wav
                for self_wav, other_wav in zip(self.wav, other.wav)
            )
            or len(self.trans) != len(other.trans)
            or any(
                self_trans != other_trans
                for self_trans, other_trans in zip(self.trans, other.trans)
            )
        ):
            return False
        # ensure attribute values are the same
        elif any(
            getattr(self, self_key) != getattr(other, self_key)
            for self_key in self.__dict__.keys()
            if self_key not in ["wav", "trans", "properties"]
        ):
            return False
        # ensure SVO property keys stored are the same
        elif not all(
            key in self.properties.keys() for key in other.properties.keys()
        ) or not all(
            key in other.properties.keys() for key in self.properties.keys()
        ):
            return False
        # ensure SVO property values stored are the same
        else:
            return all(
                self.properties[key] == other.properties[key]
                for key in self.properties.keys()
            )

    def __setattr__(self, key, value):
        super().__setattr__(key, value)

    # def crop_wav_range(self, lower_throughput, upper_throughput):
    #    self.wavs = self.wavs[self.trans > 1e-1]

    def make_PSF(self, data: Data, method: str):
        self.instrument.make_PSF(self, method)

    def plot_filter_profile(
        self, ax, wav_units=u.um, from_SVO=True, color="black"
    ):
        wavs = funcs.convert_wav_units(self.wav, wav_units).value
        ax.fill_between(wavs, 0.0, self.trans, color=color, alpha=0.6)
        ax.plot(
            wavs, self.trans, color="black", lw=2
        )  # cmap[np.where(self.bands == band)])
        ax.text(
            funcs.convert_wav_units(self.WavelengthCen, wav_units).value,
            np.max(self.trans) + 0.03,
            self.band_name,
            ha="center",
            fontsize=8,
        )


class U(Filter):
    def __init__(
        self,
    ):
        # construct the top hat filter profile
        wav = [] * u.AA
        trans = [1.0 for _ in range(len(wav))]
        # super().__init__(


class Multiple_Filter:
    def __init__(
        self: Type[Self], filters: List[Filter], sort_order: str = "ascending"
    ) -> None:
        self.filters = filters
        self.sort_order = sort_order
        self.sort_bands()

    @classmethod
    def from_instrument(
        cls: Type[Self],
        instrument: Union[str, Instrument],
        excl_bands: Union[List[str], List[Filter]] = [],
        origin: str = "SVO",
        sort_order: str = "ascending",
    ) -> Self:
        if not isinstance(instrument, Instrument):
            # construct instrument object from string
            instrument = globals()[instrument]()
        # determine facility from instrument
        facility = instrument.facility

        if origin == "SVO":
            filter_list = SvoFps.get_filter_list(
                facility=facility, instrument=instrument.name.split("_")[0]
            )
            filter_list = filter_list[
                np.array(
                    [
                        filter_name.split("/")[-1].split(".")[0]
                        for filter_name in np.array(filter_list["filterID"])
                    ]
                )
                == instrument.name
            ]
            filters = np.array(
                [
                    Filter.from_SVO(
                        facility.name,
                        instrument.name,
                        filt_ID.replace(
                            f"{facility.name}/{instrument.name}.", ""
                        ),
                    )
                    for filt_ID in np.array(filter_list["filterID"])
                    if filt_ID.replace(
                        f"{facility.name}/{instrument.name}.", ""
                    )
                    not in excl_bands
                ]
            )
        else:
            raise NotImplementedError
        return cls(filters, sort_order=sort_order)

    @classmethod
    def from_facility(
        cls: Type[Self],
        facility: Union[str, Facility],
        excl_bands: Union[List[str], List[Filter]],
        origin: str = "SVO",
    ) -> Self:
        if isinstance(facility, Facility):
            facility = facility.name
        # TODO: determine the name of all instruments associated with the facility
        pass

    @classmethod
    def from_instruments(
        cls: Type[Self],
        instruments: List[Union[str, Instrument]],
        excl_bands: Union[List[str], List[Filter]],
        origin: str = "SVO",
    ) -> Self:
        # TODO: call from_instrument for each instrument in instruments and add the results together
        pass

    def __len__(self) -> int:
        return len(self.filters)

    def __iter__(self) -> Self:
        self.iter = 0
        return self

    def __next__(self) -> Filter:
        if self.iter > len(self) - 1:
            raise StopIteration
        else:
            band = self[self.iter]
            self.iter += 1
            return band

    def __getitem__(self, i: Union[int, slice]) -> Union[Filter, List[Filter]]:
        if isinstance(i, (int, slice)):
            return self.filters[i]
        else:
            raise (
                TypeError(
                    f"i={i} in {self.__class__.__name__}.__getitem__ has type={type(i)} which is not in [int, slice]"
                )
            )

    def __add__(self, other: Union[Filter, Type[Self]]) -> Type[Self]:
        if isinstance(other, Filter):
            return Multiple_Filter(self.filters + [other])
        elif isinstance(other, Multiple_Filter):
            return Multiple_Filter(self.filters + other.filters)
        else:
            raise TypeError(
                f"Cannot add {type(other)} to {self.__class__.__name__}"
            )

    def __sub__(self, other: Union[Filter, Type[Self]]) -> Type[Self]:
        if isinstance(other, Filter):
            return Multiple_Filter([band for band in self if band != other])
        elif isinstance(other, Multiple_Filter):
            return Multiple_Filter(
                [band for band in self if band not in other]
            )
        else:
            raise TypeError(
                f"Cannot subtract {type(other)} from {self.__class__.__name__}"
            )

    def __eq__(self, other: Type[Self]) -> bool:
        if isinstance(other, Multiple_Filter):
            if len(self) == len(other):
                return all(
                    self_filt == other_filt
                    for self_filt, other_filt in zip(
                        self.filters, other.filters
                    )
                )
        return False

    def __str__(self) -> str:
        """Function to print summary of Instrument class

        Returns:
            str: Summary containing facility, instrument name and filter set included in the instrument
        """
        output_str = funcs.line_sep
        output_str += f"FACILITY: {self.facility}\n"
        output_str += f"INSTRUMENT: {self.name}\n"
        # show individual bands used, ordered from blue to red
        output_str += f"FILTER SET: {str([f'{band.facility}/{band.instrument}/{band.band_name}' for band in self])}\n"
        # could also include PSF path and correction factors here
        output_str += funcs.line_sep
        return output_str

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.band_names})"

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

    def __del__(self) -> None:
        self.filters = []

    @property
    def instrument_name(self) -> str:
        """
        Returns the name of the instrument(s) associated with the bands in the current object.

        This method retrieves all possible instrument names from the configuration,
        identifies the unique instrument names from the bands in the current object,
        and returns a concatenated string of these names in ascending order.

        Returns:
            str: A concatenated string of unique instrument names in ascending wavelength order.

        Raises:
            NotImplementedError: If the sort order is not "ascending".
        """
        all_instrument_names = json.loads(
            config.get("Other", "INSTRUMENT_NAMES")
        )
        unique_instrument_names = np.unique(
            [band.instrument.name for band in self if band is not None]
        )
        if self.sort_order == "ascending":
            return "+".join(
                [
                    name
                    for name in all_instrument_names
                    if name in unique_instrument_names
                ]
            )
        else:
            raise NotImplementedError

    @property
    def facility_name(self) -> str:
        """
        Returns the name of the facility/facilities associated with the bands in the current object.

        This method retrieves all possible facility names from the configuration,
        identifies the unique facility names from the bands in the current object,
        and returns a concatenated string of these names in ascending order.

        Returns:
            str: A concatenated string of unique facility names in ascending wavelength order.

        Raises:
            NotImplementedError: If the sort order is not "ascending".
        """
        all_facility_names = json.loads(config.get("Other", "TELESCOPE_NAMES"))
        unique_facility_names = np.unique(
            [band.facility.name for band in self if band is not None]
        )
        if self.sort_order == "ascending":
            return "+".join(
                [
                    name
                    for name in all_facility_names
                    if name in unique_facility_names
                ]
            )
        else:
            raise NotImplementedError

    @property
    def band_names(self) -> List[str]:
        return [band.band_name for band in self]

    @property
    def band_wavelengths(self):
        # Central wavelengths
        return (
            np.array(
                [band.WavelengthCen.to(u.AA).value for band in self.bands]
            )
            * u.AA
        )

    def sort_bands(self) -> None:
        if self.sort_order == "ascending":
            # sort filters from blue -> red
            self.filters = [
                band
                for band in sorted(
                    self.filters,
                    key=lambda band: band.WavelengthCen.to(u.AA).value,
                )
            ]
        else:
            raise NotImplementedError

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

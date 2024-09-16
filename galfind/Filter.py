# Filter.py
from __future__ import annotations

import astropy.units as u
import numpy as np
from astroquery.svo_fps import SvoFps
from copy import deepcopy
import json
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import List, Union, NoReturn, TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from . import Facility, Instrument, Data
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import galfind_logger, config, instr_to_name_dict
from . import useful_funcs_austind as funcs
from . import Facility, Instrument
from . import ACS_WFC, WFC3_IR, NIRCam, MIRI, JWST, HST  # noqa: F401


class Filter:
    def __init__(
        self,
        instrument: Union[None, str, Instrument],
        band_name: str,
        wav: u.Quantity,
        trans: List[float],
        properties: dict = {},
    ):
        assert len(wav) == len(trans)
        if isinstance(instrument, str):
            instrument = instr_to_name_dict[instrument]
        self.instrument = instrument
        self.band_name = band_name
        self.wav = wav
        self.trans = trans
        for key, value in properties.items():
            self.__setattr__(key, value)
        self.properties = properties  # currently just used in __str__ only

        # create WavelengthCen if required
        if "WavelengthCen" not in properties.keys():
            self.WavelengthCen = np.mean(self.wav.value) * self.wav.unit

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
        return cls(instrument, filter_name, wav, trans, output_prop)

    def __str__(self):
        output_str = funcs.line_sep
        if self.instrument is not None:
            output_str += f"{self.instrument}/{self.band_name}\n"
        else:
            output_str += f"{self.band_name}\n"
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

    def plot(
        self, ax, wav_units: u.Quantity = u.um, colour: str = "black", save_dir: str = "", label: bool = True, save: bool = False, show: bool = False
    ):
        # convert wavelength units
        wavs = funcs.convert_wav_units(self.wav, wav_units).value
        # add extra points to ensure the plot is closed
        wavs = list(np.concatenate(([wavs[0]], wavs, [wavs[-1]])))
        trans = list(np.concatenate(([0.], self.trans, [0.])))
        # plot the filter profile
        ax.fill_between(wavs, 0., trans, color=colour, alpha=0.6)
        ax.plot(
            wavs, trans, color="black", lw=2, label = self.band_name
        )  # cmap[np.where(self.bands == band)])
        if label:
            ax.text(
            funcs.convert_wav_units(self.WavelengthCen, wav_units).value,
            np.max(trans) + 0.03,
            self.band_name,
            ha="center",
            fontsize=8,
        )
        if save or show:
            leg_labels = ax.get_legend_handles_labels()[1]
            if len(leg_labels) == 1:
                title = f"{leg_labels[0]} filter"
            else:
                title = f"{'+'.join(leg_labels)} filters"
            # annotate plot
            ax.set_title(title)
            ax.set_xlabel(
                r"$\lambda_{\mathrm{obs}}$ / "
                + funcs.unit_labels_dict[wav_units]
            )
            ax.set_ylabel("Transmission")
            ax.set_ylim(
                0.0,
                np.max(trans) + 0.1,
            )
        if save:
            save_path = f"{save_dir}/{title.replace(' ', '_')}.png"
            funcs.make_dirs(save_path)
            plt.savefig(save_path)
            funcs.change_file_permissions(save_path)
        if show:
            plt.show()

class Tophat_Filter(Filter):
    def __init__(self, band_name: str, lower_wav: u.Quantity, upper_wav: u.Quantity, throughput: float = 1., resolution: u.Quantity = 1. * u.AA, properties: dict = {}):
        # construct the top hat filter profile
        n_elements = int(((upper_wav - lower_wav) / resolution).to(u.dimensionless_unscaled).value)
        wav = list(np.linspace(lower_wav, upper_wav, n_elements)) * u.AA
        trans = list(np.full(len(wav), throughput))
        properties = {**properties, **{"WavelengthCen": (lower_wav + upper_wav) / 2., "FWHM": upper_wav - lower_wav}}
        super().__init__(None, band_name, wav, trans, properties=properties)
                               
class U(Tophat_Filter):
    def __init__(
        self,
        throughput: float = 1.,
        resolution: u.Quantity = 1. * u.AA
    ):
        super().__init__(self.__class__.__name__, 3_320. * u.AA, 3_980. * u.AA, throughput = throughput, resolution = resolution)

class V(Tophat_Filter):
    def __init__(self, throughput: float = 1., resolution: u.Quantity = 1. * u.AA):
        super().__init__(self.__class__.__name__, 5_070. * u.AA, 5_950. * u.AA, throughput = throughput, resolution = resolution)

class J(Tophat_Filter):
    def __init__(self, throughput: float = 1., resolution: u.Quantity = 1. * u.AA):
        super().__init__(self.__class__.__name__, 11_135. * u.AA, 13_265. * u.AA, throughput = throughput, resolution = resolution)

class Multiple_Filter:
    def __init__(
        self: Type[Self], filters: List[Filter], sort_order: str = "ascending"
    ) -> None:
        self.filters = filters
        self.sort_order = sort_order
        self.sort_bands()

    @classmethod
    def from_facility(
        cls: Type[Self],
        facility: Union[str, Facility],
        excl_bands: Union[List[str], List[Filter]] = [],
        origin: str = "SVO",
        sort_order: str = "ascending",
        keep_suffix: str = "All",
    ) -> Self:
        if isinstance(facility, Facility):
            facility = facility.name
        instruments_from_facility = [name for name, instr in instr_to_name_dict.items() if instr.facility.name == facility]
        return cls.from_instruments(instruments_from_facility, excl_bands, origin, sort_order, keep_suffix)

    @classmethod
    def from_instruments(
        cls: Type[Self],
        instruments: List[Union[str, Instrument]],
        excl_bands: Union[List[str], List[Filter]] = [],
        origin: str = "SVO",
        sort_order: str = "ascending",
        keep_suffix: str = "All",
    ) -> Self:
        assert len(instruments) > 0
        for i, instrument in enumerate(instruments):
            # convert instrument object to string
            if isinstance(instrument, tuple(instr.__class__ for instr in instr_to_name_dict.values())):
                instrument = instrument.name
            # ensure the instrument is a valid instrument
            assert instrument in json.loads(config.get("Other", "INSTRUMENT_NAMES"))
            new_multi_filt = cls.from_instrument(instrument, excl_bands, origin, sort_order, keep_suffix)
            if i == 0:
                multi_filt = new_multi_filt
            else:
                multi_filt += new_multi_filt
        return multi_filt

    @classmethod
    def from_instrument(
        cls: Type[Self],
        instrument: Union[str, Instrument],
        excl_bands: Union[List[str], List[Filter]] = [],
        origin: str = "SVO",
        sort_order: str = "ascending",
        keep_suffix: str = "All",
    ) -> Self:
        if not isinstance(instrument, Instrument):
            # construct instrument object from string
            instrument = globals()[instrument]()

        if origin == "SVO":
            filter_list = SvoFps.get_filter_list(
                facility=instrument.facility.name, instrument=instrument.name.split("_")[0]
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
                        instrument.facility.name,
                        instrument.name,
                        filt_ID.replace(
                            f"{instrument}.", ""
                        ),
                    )
                    for filt_ID in np.array(filter_list["filterID"])
                    if filt_ID.replace(
                        f"{instrument}.", ""
                    )
                    not in excl_bands and (filt_ID.endswith(keep_suffix) or keep_suffix == "All")
                ]
            )
        else:
            raise NotImplementedError
        return cls(filters, sort_order=sort_order)

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

    # def __add__(self, other: Union[Filter, Type[Self]]) -> Type[Self]:
    #     if isinstance(other, Filter):
    #         return Multiple_Filter(self.filters + [other])
    #     elif isinstance(other, Multiple_Filter):
    #         return Multiple_Filter(self.filters + other.filters)
    #     else:
    #         raise TypeError(
    #             f"Cannot add {type(other)} to {self.__class__.__name__}"
    #         )

    # def __sub__(self, other: Union[Filter, Type[Self]]) -> Type[Self]:
    #     if isinstance(other, Filter):
    #         return Multiple_Filter([band for band in self if band != other])
    #     elif isinstance(other, Multiple_Filter):
    #         return Multiple_Filter(
    #             [band for band in self if band not in other]
    #         )
    #     else:
    #         raise TypeError(
    #             f"Cannot subtract {type(other)} from {self.__class__.__name__}"
    #         )
        
    def __add__(self, other: Union[str, Filter, Multiple_Filter, List[Union[str, Filter, Multiple_Filter]]]) -> Self:

        if type(other) in instr_to_name_dict.keys():
            new_bands = np.array(
                [
                    band
                    for band in other
                    if band.band_name not in self.band_names
                ]
            )
        elif type(other) == Filter:
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
            all_facilities = json.loads(config.get("Other", "FACILITY_NAMES"))
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
        return out_instrument

    def __sub__(self, other: Union[str, Filter, Multiple_Filter, List[Union[str, Filter, Multiple_Filter]]]) -> Self:
        # If other is a subclass of Instrument, remove bands in other from self
        if isinstance(other, tuple(Instrument.__subclasses__())):
            other_band_names = other.band_names
        else:
            # Make a list if required
            if not isinstance(other, list):
                other = [other]
            else:
                # Ensure all elements are either an instance of Filter or str
                assert all(
                    isinstance(band, (Filter, str)) for band in other
                ), galfind_logger.critical(
                    f"Not all elements in {other} have the type (Filter, str)!"
                )
            # Work out which bands need to be removed from the instrument
            other_band_names = [
                band if isinstance(band, str) else band.band_name
                for band in other
            ]
        # Ensure all bands in other_band_names are included in the instrument already
        assert all(band in self.band_names for band in other_band_names)
        # Remove the bands from the instrument
        self.bands = [
            band for band in self if band.band_name not in other_band_names
        ]
        # Work out the modified instrument name
        all_instruments = json.loads(config.get("Other", "INSTRUMENT_NAMES"))
        instrument_names = np.unique([band.instrument for band in self])
        self.name = "+".join(
            [name for name in all_instruments if name in instrument_names]
        )
        # Work out the modified instrument facility
        all_facilities = json.loads(config.get("Other", "FACILITY_NAMES"))
        facility_names = np.unique([band.facility for band in self])
        self.facility = "+".join(
            [
                facility
                for facility in all_facilities
                if facility in facility_names
            ]
        )
        # Return the modified instrument
        return self

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
        all_facility_names = json.loads(config.get("Other", "FACILITY_NAMES"))
        unique_facility_names = np.unique(
            [band.instrument.facility.name for band in self if band is not None]
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

    def plot(
        self,
        ax,
        wav_units: u.Unit = u.um,
        cmap_name: str = "Spectral_r",
        save_dir: str = "",
        label: bool = True,
        show: bool = False,
        save: bool = False,
    ) -> NoReturn:
        # determine appropriate colours from the colour map
        cmap = plt.get_cmap(cmap_name, len(self))
        norm = Normalize(vmin=0, vmax=len(self) - 1)
        colours = [cmap(norm(i)) for i in range(len(self))]
        # plot each filter profile
        for i, (band, colour) in enumerate(zip(self, colours)):
            band.plot(ax, colour=colour, label = label, show = False, save = False)
        # annotate plot if needed
        if save or show:
            ax.set_title(f"{self.instrument_name} filters")
            ax.set_xlabel(
                r"$\lambda_{\mathrm{obs}}$ / "
                + funcs.unit_labels_dict[wav_units]
            )
            ax.set_ylabel("Transmission")
            ax.set_ylim(
                0.0,
                np.max([trans for band in self for trans in band.trans]) + 0.1,
            )
            print(np.max([trans for band in self for trans in band.trans]))
        if save:
            save_path = f"{save_dir}/{self.instrument_name}_filter_profiles.png"
            funcs.make_dirs(save_path)
            plt.savefig(save_path)
            funcs.change_file_permissions(save_path)
        if show:
            plt.show()

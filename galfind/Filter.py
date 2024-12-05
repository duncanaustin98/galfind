# Filter.py
from __future__ import annotations

import astropy.units as u
import numpy as np
from astroquery.svo_fps import SvoFps
from copy import deepcopy
import json
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import List, Tuple, Union, Optional, NoReturn, TYPE_CHECKING
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
    def from_SVO(cls, facility: str, instrument: str, filter_name: str, \
            SVO_facility_name: Optional[str] = None, SVO_instr_name: Optional[str] = None):
        full_name = f"{facility}/{instrument}.{filter_name}"
        try:
            filter_profile = SvoFps.get_transmission_data(full_name)
        except Exception as e:
            err_message = f"{full_name} is not a valid SvoFps filter! Exception={e}"
            galfind_logger.critical(err_message)
            raise(Exception(err_message))
        wav = np.array(filter_profile["Wavelength"])
        trans = np.array(filter_profile["Transmission"])
        if SVO_facility_name is None:
            SVO_facility_name = instr_to_name_dict[instrument].facility.SVO_name
        if SVO_instr_name is None:
            SVO_instr_name = instr_to_name_dict[instrument].SVO_name
        try:
            properties = SvoFps.data_from_svo(
                query={
                    "Facility": SVO_facility_name,
                    "Instrument": SVO_instr_name,
                },
                error_msg=f"'{SVO_facility_name}/{SVO_instr_name}' is not a valid facility/instrument combination!",
            )
        except Exception as e:
            err_message = f"Could not retrieve properties for {SVO_facility_name}/{SVO_instr_name}! Exception={e}"
            galfind_logger.critical(err_message)
            raise(Exception(err_message))
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

    @classmethod
    def from_filt_name(cls, filt_name: str):
        return cls.from_SVO(*cls._get_facility_instrument_filt(filt_name))

    def __str__(self):
        output_str = funcs.line_sep
        output_str += "FILTER: "
        if self.instrument is not None:
            output_str += f"{self.instrument}/{self.band_name}\n"
        else:
            output_str += f"{self.band_name}\n"
        output_str += funcs.line_sep
        for key, value in self.properties.items():
            output_str += f"{key}: {value}\n"
        output_str += funcs.line_sep
        return output_str

    def __repr__(self):
        return f"{self.__class__.__name__}({self.band_name})"

    def __len__(self):
        return 1

    def __add__(
        self: Self,
        other: Union[
            str,
            Filter,
            Multiple_Filter,
            List[Union[str, Filter, Multiple_Filter]],
        ],
    ) -> Union[Self, Multiple_Filter]:
        # make relevant new filters from other that aren't in [self]
        new_filters = Multiple_Filter._make_new_filt([self], other)
        if new_filters is None:
            return self
        else:
            # add new filters to existing filters
            return Multiple_Filter([self] + new_filters)

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

    @property
    def instrument_name(self) -> str:
        return self.instrument.__class__.__name__
    
    @property
    def facility_name(self) -> str:
        return self.instrument.facility.__class__.__name__

    @staticmethod
    def _get_facility_instrument_filt(
        filt_name: str
    ) -> Tuple[str, str, str, str, str]:
        # determine facility and instrument names of filter string
        split_str = filt_name.split("/")
        if len(split_str) == 3:
            # formatted as e.g. JWST/NIRCam/F444W
            facility, instrument, filt = split_str
        elif len(split_str) == 2:
            # formatted as e.g. JWST/NIRCam.F444W
            facility, filt_substr = split_str
            filt_substr_split = filt_substr.split(".")
            assert len(filt_substr_split) == 2
            instrument, filt = filt_substr_split
        elif len(split_str) == 1:
            # formatted as e.g. F444W
            # try to determine facility and instrument from band name alone
            filt = split_str[0].upper()
            instruments_with_filt = [
                instr_name
                for instr_name, instrument in instr_to_name_dict.items()
                if filt in instrument.filt_names
            ]
            assert len(instruments_with_filt) == 1, \
                galfind_logger.critical(
                    f"Could not determine instrument from band name {filt}"
                )
            instrument = instruments_with_filt[0]
            facility = instr_to_name_dict[instrument].facility.__class__.__name__
        filt = filt.upper()
        # determine instrument and facility SVO names
        SVO_facility_name = instr_to_name_dict[instrument].facility.SVO_name
        SVO_instr_name = instr_to_name_dict[instrument].SVO_name
        return facility, instrument, filt, SVO_facility_name, SVO_instr_name

    @staticmethod
    def _make_new_filt(
        current_filt_names: List[str], filt_or_name: Union[str, Filter]
    ) -> Union[Filter, None]:
        already_included = False
        if isinstance(filt_or_name, str):
            # extract facility, instrument and filter name from string
            facility, instrument, filt, SVO_facility_name, SVO_instr_name = \
                Filter._get_facility_instrument_filt(filt_or_name)
            if filt_or_name in current_filt_names:
                already_included = True
            else:
                new_filt = Filter.from_SVO(facility, instrument, filt, \
                    SVO_facility_name = SVO_facility_name, SVO_instr_name = SVO_instr_name)
        elif isinstance(filt_or_name, Filter):
            if filt_or_name.band_name in current_filt_names:
                already_included = True
            else:
                new_filt = filt_or_name
        # print warning if filter already included
        if already_included:
            already_included_warning = (
                f"{repr(filt_or_name)} duplicated, not adding"
            )
            galfind_logger.warning(already_included_warning)
            # warnings.warn(UserWarning(already_included_warning))
            return None
        else:
            return new_filt

    # def crop_wav_range(self, lower_throughput, upper_throughput):
    #    self.wavs = self.wavs[self.trans > 1e-1]

    def make_PSF(self, data: Data, method: str):
        self.instrument.make_PSF(self, method)

    def plot(
        self,
        ax,
        wav_units: u.Quantity = u.um,
        colour: str = "black",
        save_dir: str = "",
        label: bool = True,
        save: bool = False,
        show: bool = False,
    ):
        # convert wavelength units
        wavs = funcs.convert_wav_units(self.wav, wav_units).value
        # add extra points to ensure the plot is closed
        wavs = list(np.concatenate(([wavs[0]], wavs, [wavs[-1]])))
        trans = list(np.concatenate(([0.0], self.trans, [0.0])))
        # plot the filter profile
        ax.fill_between(wavs, 0.0, trans, color=colour, alpha=0.6)
        ax.plot(
            wavs, trans, color="black", lw=2, label=self.band_name
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
    def __init__(
        self,
        band_name: str,
        lower_wav: u.Quantity,
        upper_wav: u.Quantity,
        throughput: float = 1.0,
        resolution: u.Quantity = 1.0 * u.AA,
        properties: dict = {},
    ):
        # construct the top hat filter profile
        n_elements = int(
            ((upper_wav - lower_wav) / resolution)
            .to(u.dimensionless_unscaled)
            .value
        )
        wav = list(np.linspace(lower_wav, upper_wav, n_elements)) * u.AA
        trans = list(np.full(len(wav), throughput))
        properties = {
            **properties,
            **{
                "WavelengthCen": (lower_wav + upper_wav) / 2.0,
                "FWHM": upper_wav - lower_wav,
            },
        }
        super().__init__(None, band_name, wav, trans, properties=properties)


class U(Tophat_Filter):
    def __init__(
        self, throughput: float = 1.0, resolution: u.Quantity = 1.0 * u.AA
    ):
        super().__init__(
            self.__class__.__name__,
            3_320.0 * u.AA,
            3_980.0 * u.AA,
            throughput=throughput,
            resolution=resolution,
        )


class V(Tophat_Filter):
    def __init__(
        self, throughput: float = 1.0, resolution: u.Quantity = 1.0 * u.AA
    ):
        super().__init__(
            self.__class__.__name__,
            5_070.0 * u.AA,
            5_950.0 * u.AA,
            throughput=throughput,
            resolution=resolution,
        )


class J(Tophat_Filter):
    def __init__(
        self, throughput: float = 1.0, resolution: u.Quantity = 1.0 * u.AA
    ):
        super().__init__(
            self.__class__.__name__,
            11_135.0 * u.AA,
            13_265.0 * u.AA,
            throughput=throughput,
            resolution=resolution,
        )


class Multiple_Filter:
    def __init__(
        self: Type[Self], filters: List[Filter], sort_order: str = "ascending"
    ) -> None:
        self.filters = filters
        self.sort_order = sort_order
        self.sort_bands()

    @classmethod
    def from_facilities(
        cls: Type[Self],
        facility_arr = List[Union[str, Facility]],
        excl_bands: Union[
            str,
            Filter,
            Multiple_Filter,
            List[Union[str, Filter, Multiple_Filter]],
        ] = [],
        origin: str = "SVO",
        sort_order: str = "ascending",
        keep_suffix: str = "All",
    ) -> Self:
        for i, facility in enumerate(facility_arr):
            filterset_ = cls.from_facility(
                facility, excl_bands, origin, sort_order, keep_suffix
            )
            if i == 0:
                filterset = filterset_
            else:
                filterset += filterset_
        return filterset
        

    @classmethod
    def from_facility(
        cls: Type[Self],
        facility: Union[str, Facility],
        excl_bands: Union[
            str,
            Filter,
            Multiple_Filter,
            List[Union[str, Filter, Multiple_Filter]],
        ] = [],
        origin: str = "SVO",
        sort_order: str = "ascending",
        keep_suffix: str = "All",
    ) -> Self:
        if isinstance(facility, Facility):
            facility = facility.__class__.__name__
        instruments_from_facility = [
            name
            for name, instr in instr_to_name_dict.items()
            if instr.facility.__class__.__name__ == facility
        ]
        return cls.from_instruments(
            instruments_from_facility,
            excl_bands,
            origin,
            sort_order,
            keep_suffix,
        )

    @classmethod
    def from_instruments(
        cls: Type[Self],
        instruments: List[Union[str, Instrument]],
        excl_bands: Union[
            str,
            Filter,
            Multiple_Filter,
            List[Union[str, Filter, Multiple_Filter]],
        ] = [],
        origin: str = "SVO",
        sort_order: str = "ascending",
        keep_suffix: str = "All",
    ) -> Self:
        assert len(instruments) > 0
        for i, instrument in enumerate(instruments):
            # convert instrument object to string
            if isinstance(
                instrument,
                tuple(
                    instr.__class__ for instr in instr_to_name_dict.values()
                ),
            ):
                instrument = instrument.__class__.__name__
            # ensure the instrument is a valid instrument
            assert instrument in json.loads(
                config.get("Other", "INSTRUMENT_NAMES")
            )
            new_multi_filt = cls.from_instrument(
                instrument, excl_bands, origin, sort_order, keep_suffix
            )
            if i == 0:
                multi_filt = new_multi_filt
            else:
                multi_filt += new_multi_filt
        return multi_filt

    @classmethod
    def from_instrument(
        cls: Type[Self],
        instrument: Union[str, Instrument],
        excl_bands: Union[
            str,
            Filter,
            Multiple_Filter,
            List[Union[str, Filter, Multiple_Filter]],
        ] = [],
        origin: str = "SVO",
        sort_order: str = "ascending",
        keep_suffix: Union[str, List[str]] = "All",
    ) -> Self:
        # construct instrument object from string
        if isinstance(instrument, str):
            instrument = instr_to_name_dict[instrument]
        # make excl_bands a list of string filter names if not already
        excl_bands = cls._get_name_from_filt(excl_bands)
        excl_bands = [
            Filter._get_facility_instrument_filt(filt)[2]
            for filt in excl_bands
        ]
        # make keep_suffix a list of strings if not already
        if isinstance(keep_suffix, str):
            keep_suffix = [keep_suffix]
        # get filter list from "origin" source
        if origin == "SVO":
            try:
                filter_list = SvoFps.get_filter_list(
                    facility=instrument.facility.SVO_name,
                    instrument=instrument.SVO_name,
                )
            except:
                err_message = "Could not retrieve filter list from SVO for " + \
                    f"{instrument.facility.SVO_name}/{instrument.SVO_name}!"
                galfind_logger.critical(err_message)
                raise(Exception(err_message))
            # only include filters from the requested instrument
            filter_list = filter_list[
                np.array(
                    [
                        filter_name.split("/")[-1].split(".")[0]
                        for filter_name in np.array(filter_list["filterID"])
                    ]
                )
                == instrument.__class__.__name__
            ]
            # only include filters without an underscore in the name
            filter_list = filter_list[~np.array(["_" in filt_name.split("/")[-1].split(".")[-1] for filt_name in filter_list["filterID"]])]
            filters = np.array(
                [
                    Filter.from_SVO(
                        instrument.facility.__class__.__name__,
                        instrument.__class__.__name__,
                        filt_ID.replace(f"{instrument.facility.__class__.__name__}/{instrument.__class__.__name__}.", ""),
                        SVO_facility_name = instrument.facility.SVO_name,
                        SVO_instr_name = instrument.SVO_name,
                    )
                    for filt_ID in np.array(filter_list["filterID"])
                    if filt_ID.replace(f"{instrument.facility.__class__.__name__}/{instrument.__class__.__name__}.", "") not in excl_bands
                    and (
                        any(filt_ID.endswith(suffix) for suffix in keep_suffix)
                        or "All" in keep_suffix
                        and len(keep_suffix) == 1
                    )
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

    def __getitem__(self, i: Union[int, slice, str, list, np.ndarray]) -> Union[Filter, List[Filter]]:
        if isinstance(i, (int, slice)):
            return self.filters[i]
        elif isinstance(i, str):
            return list(np.array(self.filters)[[index for index, filt in enumerate(self) if filt.band_name == i]])[0]
        elif isinstance(i, (list, np.ndarray)):
            # if all(isinstance(j, int) for j in i):
            #     # convert to boolean array

            if isinstance(i, list):
                return Multiple_Filter(list(np.array(self.filters)[np.array(i)]))
            else:
                return Multiple_Filter(list(np.array(self.filters)[i]))
        else:
            raise (
                TypeError(
                    f"i={i} in {self.__class__.__name__}.__getitem__" + \
                    f" has type={type(i)} which is not in " + \
                    "[int, slice, str, list, np.ndarray]"
                )
            )

    def __add__(
        self,
        other: Union[
            str,
            Filter,
            Multiple_Filter,
            List[Union[str, Filter, Multiple_Filter]],
        ],
    ) -> Self:
        # make relevant new filters from other that aren't in [self]
        new_filters = Multiple_Filter._make_new_filt(self.filters, other)
        if new_filters is not None:
            # add new filters to existing filters
            self.filters += new_filters
        self.sort_bands()
        return self

    def __sub__(
        self,
        other: Union[
            str,
            Filter,
            Multiple_Filter,
            List[Union[str, Filter, Multiple_Filter]],
        ],
    ) -> Self:
        # make a list if not already
        if not isinstance(other, list):
            other = [other]
        # if list elements include Multiple_Filter objects,
        # flatten the Multiple_Filter objects to make a list of types (Filter, str)
        other = self._flatten_multi_filters(other)
        # populate array of filters to remove
        remove_filt_names = []
        for i, filt in enumerate(other):
            remove = True
            if isinstance(filt, str):
                # extract facility, instrument and filter name from string
                facility, instrument, filt, SVO_facility_name, SVO_instr_name = (
                    Filter._get_facility_instrument_filt(filt)
                )
                if filt not in self.band_names or filt in remove_filt_names:
                    remove = False
                else:
                    remove_filt_names.extend([filt])
            elif isinstance(filt, Filter):
                # extract name from Filter object
                filt = filt.band_name
                if filt not in self.band_names or filt in remove_filt_names:
                    remove = False
                else:
                    remove_filt_names.extend([filt])
            # print warning if filter already included
            if not remove:
                already_included_warning = (
                    f"{repr(filt)} not in {self.band_names}, not removing"
                )
                galfind_logger.warning(already_included_warning)
                # warnings.warn(UserWarning(already_included_warning))
        # print warning if no new filters to add
        if len(remove_filt_names) == 0:
            warning_message = (
                "No filters to remove, returning self from Instrument.__sub__"
            )
            galfind_logger.warning(warning_message)
            # warnings.warn(UserWarning(warning_message))
        else:
            # add new filters to existing filters
            self.filters = [
                filt
                for filt in self
                if filt.band_name not in remove_filt_names
            ]
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
        output_str += "MULTIPLE_FILTER\n"
        output_str += funcs.band_sep
        for i, instrument in enumerate(self.instrument_name.split("+")):
            if i != 0:
                output_str += funcs.band_sep
            if instrument != "UserDefined":
                output_str += f"FACILITY: {instr_to_name_dict[instrument].facility.__class__.__name__}\n"
            else:
                output_str += f"FACILITY: UserDefined\n"
            output_str += f"INSTRUMENT: {instrument}\n"
            instr_filt = []
            for filt in self:
                if filt.instrument is None:
                    if instrument == "UserDefined":
                        instr_filt.extend([filt])
                    else:
                        continue
                elif filt.instrument.__class__.__name__ == instrument:
                    instr_filt.extend([filt])
            output_str += f"FILTERS: {str([f'{band.band_name}' for band in instr_filt])}\n"
        # could also include PSF path and correction factors here
        output_str += funcs.line_sep
        return output_str

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.instrument_name})"

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
            [
                band.instrument.__class__.__name__
                if band.instrument is not None
                else "UserDefined"
                for band in self
            ]
        )
        if self.sort_order == "ascending":
            name = "+".join(
                [
                    name
                    for name in all_instrument_names
                    if name in unique_instrument_names
                ]
            )
            if name == "" and "UserDefined" in unique_instrument_names:
                name = "UserDefined"
            else:
                name += (
                    "+UserDefined"
                    if "UserDefined" in unique_instrument_names
                    else ""
                )
            return name
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
            [
                filt.facility_name
                if filt.instrument is not None
                else "UserDefined"
                for filt in self
            ]
        )
        if self.sort_order == "ascending":
            name = "+".join(
                [
                    name
                    for name in all_facility_names
                    if name in unique_facility_names
                ]
            )
            if name == "" and "UserDefined" in unique_facility_names:
                name = "UserDefined"
            else:
                name += (
                    "+UserDefined"
                    if "UserDefined" in unique_facility_names
                    else ""
                )
            return name
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

    @staticmethod
    def _get_name_from_filt(
        filters: Union[
            str,
            Filter,
            Multiple_Filter,
            List[Union[str, Filter, Multiple_Filter]],
        ],
    ) -> List[str]:
        # make a list if not already
        if not isinstance(filters, list):
            filters = [filters]
        # if list elements include Multiple_Filter objects, flatten the Multiple_Filter objects to make a list of str
        _filters = [
            filt if isinstance(filt, str) else filt.band_name
            for filt in filters
            if not isinstance(filt, Multiple_Filter)
        ]
        [
            _filters.extend(filt.band_names)
            for filt in filters
            if isinstance(filt, Multiple_Filter)
        ]
        return _filters

    @staticmethod
    def _flatten_multi_filters(
        filters: List[Union[str, Filter, Multiple_Filter]],
    ) -> List[Filter]:
        # if list elements include Multiple_Filter objects, flatten the Multiple_Filter objects to make a list of (Filter, str)
        _other = [
            filt for filt in filters if not isinstance(filt, Multiple_Filter)
        ]
        [
            _other.extend(filt.filters)
            for filt in filters
            if isinstance(filt, Multiple_Filter)
        ]
        return _other

    @staticmethod
    def _make_new_filt(current_band_names, other) -> Union[Filter, None]:
        # turn other into a list if not already
        if not isinstance(other, list):
            other = [other]
        # if list elements include Multiple_Filter objects,
        # flatten the Multiple_Filter objects to make a list of types (Filter, str)
        other = Multiple_Filter._flatten_multi_filters(other)
        # populate array of new filters
        new_filters = []
        for i, val in enumerate(other):
            new_filt_names = [filt.band_name for filt in new_filters]
            new_filt = Filter._make_new_filt(
                current_band_names + new_filt_names, val
            )
            if new_filt is not None:
                new_filters.extend([new_filt])
        # print warning if no new filters to add
        if len(new_filters) == 0:
            warning_message = (
                "No new bands to add, returning self from __add__"
            )
            galfind_logger.warning(warning_message)
            # warnings.warn(UserWarning(warning_message))
            return None
        else:
            return new_filters

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
            band.plot(ax, colour=colour, label=label, show=False, save=False)
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
        if save:
            if save_dir == "":
                save_path = f"{self.instrument_name}_filter_profiles.png"
            else:
                save_path = (
                    f"{save_dir}/{self.instrument_name}_filter_profiles.png"
                )
            funcs.make_dirs(save_path)
            plt.savefig(save_path)
            funcs.change_file_permissions(save_path)
        if show:
            plt.show()


class UVJ(Multiple_Filter):
    def __init__(self, sort_order: str = "ascending"):
        super().__init__([U(), V(), J()], sort_order=sort_order)

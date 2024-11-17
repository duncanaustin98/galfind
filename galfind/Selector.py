
from __future__ import annotations

from abc import ABC, abstractmethod
import astropy.units as u
import numpy as np
from copy import deepcopy
import json
from tqdm import tqdm
from typing import TYPE_CHECKING, List, Union, NoReturn, Callable, Optional
if TYPE_CHECKING:
    from . import SED_code
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import useful_funcs_austind as funcs
from . import galfind_logger, config
from . import Galaxy, Catalogue, Instrument
from .Instrument import expected_instr_bands

class Selector(ABC):

    def __init__(self: Self, **kwargs):
        assert (key in kwargs.keys() for key in self._include_kwargs), \
            galfind_logger.critical(
                f"Selection {self.__class__.__name__} given {kwargs=}" + \
                f" missing required keys = {self._include_kwargs}."
            )
        self.kwargs = kwargs
        assert self._assertions()

    @property
    @abstractmethod
    def _selection_name(self) -> str:
        pass

    @property
    @abstractmethod
    def _include_kwargs(self) -> List[str]:
        pass

    @abstractmethod
    def _assertions(self: Self) -> bool:
        pass

    #@abstractmethod
    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        aper_diam: u.Quantity,
        SED_fit_label: Optional[Union[SED_code, str]] = None,
    ) -> bool:
        # always pass by default
        return False

    @abstractmethod
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        aper_diam: u.Quantity,
        SED_fit_label: Optional[Union[SED_code, str]] = None,
    ) -> bool:
        pass

    def _get_selection_name(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: str,
    ) -> str:
        return f"{self._selection_name}{SED_fit_label}" + \
            f"_{aper_diam.to(u.arcsec).value:.2f}as"
    
    @staticmethod
    def _get_SED_fit_label(
        object: Union[Galaxy, Catalogue],
        aper_diam: u.Quantity,
        SED_fit_label: Optional[Union[SED_code, str]]
    ) -> str:
        if SED_fit_label is None:
            return ""
        else:
            if isinstance(SED_fit_label, SED_code):
                SED_fit_label = f"_{SED_fit_label.label}"
            elif isinstance(SED_fit_label, str):
                SED_fit_label = f"_{SED_fit_label}"
            else:
                raise ValueError(
                    f"{SED_fit_label=}, with {type(SED_fit_label)=}" + \
                    " not in ['SED_code', 'str']!"
                )
            if isinstance(object, Galaxy):
                assert SED_fit_label in object.aper_phot[aper_diam].SED_results.keys(), \
                    galfind_logger.critical(
                        f"SED fitting results for {SED_fit_label=} " + \
                        f"not loaded for {repr(object)}."
                    )
            elif isinstance(object, Catalogue):
                assert any(SED_fit_label in gal.aper_phot[aper_diam].\
                        SED_results.keys() for gal in object), \
                    galfind_logger.critical(
                        f"SED fitting results for {SED_fit_label=} " + \
                        f"not loaded for any galaxy in {repr(object)}."
                    )
            else:
                raise ValueError(
                    f"{object=} with {type(object)=} not in ['Galaxy', 'Catalogue']!"
                )
            return SED_fit_label
        
    @staticmethod
    def _check_phot_exists(
        gal: Galaxy,
        aper_diam: u.Quantity,
    ) -> bool:
        return len(gal.aper_phot[aper_diam]) != 0
    
    @staticmethod
    def _check_SED_fit_exists(
        gal: Galaxy,
        aper_diam: u.Quantity,
        SED_fit_label: str,
    ) -> NoReturn:
        return SED_fit_label in gal.aper_phot[aper_diam].SED_results.keys()

    def __call__(
        self: Self,
        object: Union[Galaxy, Catalogue],
        aper_diam: u.Quantity,
        SED_fit_label: Optional[Union[SED_code, str]] = None,
        return_copy: bool = True,
    ) -> Optional[Union[Galaxy, Catalogue]]:
        SED_fit_label = self._get_SED_fit_label(object, aper_diam, SED_fit_label)
        if isinstance(object, Galaxy):
            obj = self._call_gal(object, aper_diam, SED_fit_label, return_copy)
        elif isinstance(object, Catalogue):
            obj = self._call_cat(object, aper_diam, SED_fit_label, return_copy)
        else:
            raise ValueError(
                f"{object=} must be either a Galaxy or Catalogue object."
            )
        if obj is not None:
            return obj
    
    def _call_gal(
        self: Self, 
        gal: Galaxy,
        aper_diam: u.Quantity,
        SED_fit_label: str,
        return_copy: bool = True,
    ) -> Union[NoReturn, Galaxy]:
        selection_name = self._get_selection_name(aper_diam)
        if not selection_name in gal.selection_flags.keys():
            if self._failure_criteria(gal, aper_diam, SED_fit_label) \
                    or not self._check_phot_exists(gal, aper_diam) \
                    or not self._check_SED_fit_exists(gal, aper_diam, SED_fit_label):
                gal.selection_flags[selection_name] = False
            else:
                if self._selection_criteria(gal, aper_diam, SED_fit_label):
                    gal.selection_flags[selection_name] = True
                else:
                    gal.selection_flags[selection_name] = False
        if return_copy:
            return deepcopy(gal)

    def call_cat(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity,
        SED_fit_label: str,
        return_copy: bool = True,
    ) -> Union[NoReturn, Catalogue]:
        # ensure results have been loaded for 
        # at least 1 galaxy in the catalogue
        assert any(gal._check_SED_fit_exists(aper_diam, SED_fit_label) for gal in cat), \
            galfind_logger.critical(
                f"SED fitting results for {SED_fit_label=} " + \
                f"not loaded for any galaxy in {repr(cat)}."
            )
        selection_name = self._get_selection_name(aper_diam)
        [self._call_gal(gal, aper_diam, SED_fit_label, return_copy = False) for gal \
            in tqdm(cat, total = len(cat), desc = f"Selecting {selection_name}")]
        cat._append_property_to_tab(selection_name, "SELECTION")
        if return_copy:
            # append cat creator crops
            cat.cat_creator.update_crops(selection_name)
            # NOTE: return self.cat_creator(cropped = True) slower but should be identical
            cat_copy = deepcopy(cat)
            cat_copy.gals = cat_copy[getattr(cat_copy, selection_name)]
            return cat_copy


class Colour_Selector(Selector):

    @property
    def _selection_name(self) -> str:
        return f"{'-'.join(self.kwargs['colour_bands'])}" + \
            f"{'<' if self.kwargs['bluer_or_redder'] == 'bluer' else '>'}" + \
            f"{self.kwargs['colour_val']:.2f}"

    @property
    def _include_kwargs(self) -> List[str]:
        return ["colour_bands", "bluer_or_redder", "colour_val"]

    def _assertions(self: Self) -> bool:
        try:
            assert self.kwargs["bluer_or_redder"] in ["bluer", "redder"]
            colour_bands = self.kwargs["colour_bands"]
            assert isinstance(colour_bands, (str, list))
            if isinstance(colour_bands, str):
                colour_bands = colour_bands.split("-")
            colour_bands = list(colour_bands)
            assert len(colour_bands) == 2
            passed = True
        except:
            passed = False
        return passed
        
    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        aper_diam: u.Quantity,
        **kwargs
    ) -> bool:
        try:
            assertions = []
            assertions.extend([
                all(
                    colour in gal.aper_phot[aper_diam].filterset.band_names
                    for colour in self.kwargs["colour_bands"]
                )
            ])
            # ensure bands are ordered blue -> red
            assertions.extend([
                np.where(np.array(gal.aper_phot[aper_diam].filterset.band_names) == self.kwargs["colour_bands"][0])[0][0] \
                    < np.where(np.array(gal.aper_phot[aper_diam].filterset.band_names) == self.kwargs["colour_bands"][1])[0][0]
            ])
            failed = not all(assertions)
        except:
            failed = True
        return failed
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        aper_diam: u.Quantity,
        **kwargs
    ) -> bool:
        band_indices = [
            int(np.where(np.array(gal.aper_phot[aper_diam].filterset.band_names) == band_name)[0][0])
            for band_name in self.kwargs["colour_bands"]
        ]
        colour = (
            funcs.convert_mag_units(
                gal.aper_phot[aper_diam].filterset[band_indices[0]].WavelengthCen,
                gal.aper_phot[aper_diam].flux[band_indices[0]],
                u.ABmag,
            )
            - funcs.convert_mag_units(
                gal.aper_phot[aper_diam].filterset[band_indices[1]].WavelengthCen,
                gal.aper_phot[aper_diam].flux[band_indices[1]],
                u.ABmag,
            )
        ).value
        return (colour < self.kwargs["colour_val"] and \
            self.kwargs["bluer_or_redder"] == "bluer") or \
            (colour > self.kwargs["colour_val"] and \
            self.kwargs["bluer_or_redder"] == "redder")
    

class Min_Band_Selector(Selector):

    @property
    def _selection_name(self) -> str:
        return f"bands>{self.kwargs['min_bands'] - 1}"

    @property
    def _include_kwargs(self) -> List[str]:
        return ["min_bands"]

    def _assertions(self: Self) -> bool:
        return isinstance(self.kwargs["min_bands"], int)
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        aper_diam: u.Quantity,
        **kwargs
    ) -> bool:
        return len(gal.aper_phot[aper_diam]) >= self.kwargs["min_bands"]


class Unmasked_Band_Selector(Selector):

    @property
    def _selection_name(self) -> str:
        return f"unmasked_{self.kwargs['band_name']}"

    @property
    def _include_kwargs(self) -> List[str]:
        return ["band_name"]

    def _assertions(self: Self) -> bool:
        # ensure that each band is a valid band name in galfind
        return self.kwargs["band_name"] in json.loads(config.get("Other", "ALL_BANDS"))
        
    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        aper_diam: u.Quantity,
        **kwargs
    ) -> bool:
        try:
            failed = self.kwargs["band_name"] not in \
                gal.aper_phot[aper_diam].filterset.band_names
        except:
            failed = True
        return failed
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        aper_diam: u.Quantity,
        **kwargs
    ) -> bool:
        band_index = int([i for i, band_name in enumerate( \
            gal.aper_phot[aper_diam].filterset.band_names) \
            if band_name == self.kwargs["band_name"]][0])
        return not gal.aper_phot[aper_diam].flux.mask[band_index]
    
    def call_cat(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity,
        SED_fit_label: str,
        return_copy: bool = True,
    ) -> Union[NoReturn, Catalogue]:
        assert self.kwargs["band_name"] in cat.filterset.band_names, \
            galfind_logger.critical(
                f"{self.kwargs['band_name']} not in {cat.filterset.band_names}."
            )
        return super().call_cat(cat, aper_diam, SED_fit_label, return_copy)


class Min_Unmasked_Band_Selector(Selector):

    @property
    def _selection_name(self) -> str:
        return f"unmasked_bands>{self.kwargs['min_bands'] - 1}"

    @property
    def _include_kwargs(self) -> List[str]:
        return ["min_bands"]

    def _assertions(self: Self) -> bool:
        return isinstance(self.kwargs["min_bands"], int)
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        aper_diam: u.Quantity,
        **kwargs
    ) -> bool:
        mask = gal.aper_phot[aper_diam].flux.mask
        n_unmasked_bands = len([val for val in mask if not val])
        return n_unmasked_bands >= self.kwargs["min_bands"]


class Bluewards_Lya_Non_Detect_Selector(Selector):

    @property
    def _selection_name(self) -> str:
        return f"bluewards_Lya_SNR<{self.kwargs['SNR_lim']:.1f}"

    @property
    def _include_kwargs(self) -> List[str]:
        return ["SNR_lim"]

    def _assertions(self: Self) -> bool:
        return isinstance(self.kwargs["SNR_lim"], (int, float))
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        aper_diam: u.Quantity,
        SED_fit_label: str,
    ) -> bool:
        # extract first Lya non-detect band
        first_Lya_non_detect_band = gal.aper_phot[aper_diam]. \
            SED_results[SED_fit_label].phot_rest.first_Lya_non_detect_band
        # if no bands bluewards of Lyman alpha, do not select the galaxy
        if first_Lya_non_detect_band is None:
            return False # True
        # find index of first Lya non-detect band
        first_Lya_non_detect_index = np.where(
            gal.aper_phot[aper_diam].filterset.band_names \
            == first_Lya_non_detect_band)[0][0]
        SNR_non_detect = gal.aper_phot[aper_diam].SNR[: first_Lya_non_detect_index + 1]
        mask_non_detect = gal.aper_phot[aper_diam].flux.mask[: first_Lya_non_detect_index + 1]
        # require the first Lya non detect band and all bluewards bands 
        # to be non-detected at < SNR_lim if not masked
        return all(SNR < self.kwargs["SNR_lim"] or mask for mask, SNR in 
            zip(mask_non_detect, SNR_non_detect))


class Redwards_Lya_Detect_Selector(Selector):

    @property
    def _selection_name(self) -> str:
        if isinstance(self.kwargs["SNR_lims"], (int, float)):
            # require all redwards bands to be detected at >SNR_lims
            selection_name = f"ALL_redwards_Lya_SNR>{self.kwargs['SNR_lims']:.1f}"
        else: # isinstance(SNR_lims, (list, np.array)):
            # require the n^th band redwards of Lya 
            # to be detected at >SNR_lims[n]
            SNR_str = ",".join([str(np.round(SNR, 1)) \
                for SNR in self.kwargs["SNR_lims"]])
            selection_name = f"redwards_Lya_SNR>{SNR_str}"
        if self.kwargs["widebands_only"]:
            selection_name += "_widebands"
        return selection_name

    @property
    def _include_kwargs(self) -> List[str]:
        return ["SNR_lims", "widebands_only"]

    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            if isinstance(self.kwargs["SNR_lims"], (int, float)):
                pass
            elif isinstance(self.kwargs["SNR_lims"], (list, np.ndarray)):
                assertions.extend([all(isinstance(SNR, (int, float)) \
                    for SNR in self.kwargs["SNR_lims"])])
            else:
                assertions.extend([False])
            assertions.extend([isinstance(self.kwargs["widebands_only"], bool)])
            failed = not all(assertions)
        except:
            failed = True
        return failed
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        aper_diam: u.Quantity,
        SED_fit_label: str,
    ) -> bool:
        if isinstance(self.kwargs["SNR_lims"], (int, float)):
            SNR_lims = np.full(len(self.aper_phot[aper_diam].\
                filterset.band_names), self.kwargs["SNR_lims"])
        else:
            SNR_lims = self.kwargs["SNR_lims"]

        # extract first Lya non-detect band
        first_Lya_detect_band = gal.aper_phot[aper_diam]. \
            SED_results[SED_fit_label].phot_rest.first_Lya_detect_band
        # if no bands redwards of Lyman alpha, do not select the galaxy
        if first_Lya_detect_band is None:
            return False
        # find index of first Lya non-detect band
        first_Lya_detect_index = np.where(gal.aper_phot[aper_diam]. \
            filterset.band_names == first_Lya_detect_band)[0][0]
        detect_bands = gal.aper_phot[aper_diam]. \
            filterset.band_names[first_Lya_detect_index:]
        SNR_detect = gal.aper_phot[aper_diam].SNR[first_Lya_detect_index:]
        mask_detect = gal.aper_phot[aper_diam].flux.mask[first_Lya_detect_index:]
        # option as to whether to exclude potentially 
        # shallower medium/narrow bands in this calculation
        if self.kwargs["widebands_only"]:
            wide_band_detect_indices = [
                True if "W" in band.upper() or "LP" in band.upper()
                else False for band in detect_bands
            ]
            SNR_detect = list(np.array(SNR_detect)[wide_band_detect_indices])
            mask_detect = list(np.array(mask_detect)[wide_band_detect_indices])
        # selection criteria
        return all(SNR > SNR_lim or mask for mask, SNR, SNR_lim \
            in zip(mask_detect, SNR_detect, SNR_lims))


class Lya_Band_Selector(Selector):

    @property
    def _selection_name(self) -> str:
        detect_or_non_detect_str = ">" if self.kwargs \
            ["detect_or_non_detect"] == "detect" else "<"
        selection_name = f"Lya_band_SNR{detect_or_non_detect_str}" + \
            f"{self.kwargs['SNR_lim']:.1f}"
        if self.kwargs["widebands_only"]:
            selection_name += "_widebands"
        return selection_name

    @property
    def _include_kwargs(self) -> List[str]:
        return ["SNR_lim", "detect_or_non_detect", "widebands_only"]

    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["SNR_lim"], (int, float))])
            assertions.extend([self.kwargs["detect_or_non_detect"].lower() in ["detect", "non_detect"]])
            assertions.extend([isinstance(self.kwargs["widebands_only"], bool)])
            failed = not all(assertions)
        except:
            failed = True
        return failed
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        aper_diam: u.Quantity,
        SED_fit_label: str,
    ) -> bool:
        bands = gal.aper_phot[aper_diam].filterset.band_names
        # determine Lya band(s) - usually a single band, 
        # but could be two in the case of medium bands
        first_Lya_detect_band = gal.aper_phot[aper_diam]. \
            SED_results[SED_fit_label].phot_rest.first_Lya_detect_band
        first_Lya_detect_index = np.where(bands == \
            first_Lya_detect_band)[0][0]
        first_Lya_non_detect_band = gal.aper_phot[aper_diam]. \
            SED_results[SED_fit_label].phot_rest.first_Lya_non_detect_band
        first_Lya_non_detect_index = np.where(bands == \
            first_Lya_non_detect_band)[0][0]
        # load SNRs, cropping by the relevant bands
        detect_bands = bands[
            first_Lya_detect_index : first_Lya_non_detect_index + 1
        ]
        if len(detect_bands) == 0:
            return False
        SNRs = gal.aper_phot[aper_diam].SNR[
            first_Lya_detect_band : first_Lya_non_detect_index + 1
        ]
        mask_bands = gal.aper_phot[aper_diam].flux.mask[
            first_Lya_detect_band : first_Lya_non_detect_index + 1
        ]
        if self.kwargs["widebands_only"]:
            wide_band_detect_indices = [
                True
                if "W" in band.upper() or "LP" in band.upper()
                else False
                for band in detect_bands
            ]
            SNRs = SNRs[wide_band_detect_indices]
        if len(SNRs) == 0:
            return False
        if self.kwargs["detect_or_non_detect"].lower() == "detect":
            return all(
                SNR > self.kwargs["SNR_lim"] or mask
                for SNR, mask in zip(SNRs, mask_bands)
            )
        else:
            return all(
                SNR < self.kwargs["SNR_lim"] or mask
                for SNR, mask in zip(SNRs, mask_bands)
            )


class Band_SNR_Selector(Selector):

    @property
    def _selection_name(self) -> str:
        if isinstance(self.kwargs["band"], str):
            selection_name = self.kwargs['band'] + \
                f"_SNR{sign}{self.kwargs['SNR_lim']:.1f}"
        else: # isinstance(self.kwargs["band"], int):
            galfind_logger.warning(
                "Indexing e.g. 2 and -4 when there are 6 bands " + \
                f"results in differing {self.__class__.__name__} selection " + \
                "names even though the same band is referenced!"
            )
            if self.kwargs["detect_or_non_detect"].lower() == "detect":
                sign = ">"
            else: # self.kwargs["detect_or_non_detect"].lower() == "non_detect"
                sign = "<"
            if self.kwargs["band"] == 0:
                selection_name = "bluest_band_SNR" + \
                    f"{sign}{self.kwargs['SNR_lim']:.1f}"
            elif self.kwargs["band"] == -1:
                selection_name = f"reddest_band_SNR" + \
                    f"{sign}{self.kwargs['SNR_lim']:.1f}"
            elif self.kwargs["band"] > 0:
                selection_name = funcs.ordinal(self.kwargs["band"] + 1) + \
                    f"_bluest_band_SNR{sign}{self.kwargs['SNR_lim']:.1f}"
            elif self.kwargs["band"] < -1:
                selection_name = funcs.ordinal(abs(self.kwargs["band"])) + \
                    f"_reddest_band_SNR{sign}{self.kwargs['SNR_lim']:.1f}"
        return selection_name

    @property
    def _include_kwargs(self) -> List[str]:
        return ["band", "SNR_lim", "detect_or_non_detect"]

    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["band"], (int, str))])
            if isinstance(self.kwargs["band"], str):
                assertions.extend([self.kwargs["band"] in json.loads(config.get("Other", "ALL_BANDS"))])
            assertions.extend([isinstance(self.kwargs["SNR_lim"], (int, float))])
            assertions.extend([self.kwargs["detect_or_non_detect"].lower() in ["detect", "non_detect"]])
            failed = not all(assertions)
        except:
            failed = True
        return failed
    
    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        aper_diam: u.Quantity,
        **kwargs
    ) -> bool:
        if isinstance(self.kwargs["band"], str):
            try:
                failed = self.kwargs["band"] not in \
                    gal.aper_phot[aper_diam].filterset.band_names
            except:
                failed = True
            return failed
        else:
            return False
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        aper_diam: u.Quantity,
        **kwargs
    ) -> bool:
        if isinstance(self.kwargs["band"], str):
            band_index = int(np.where(np.array( \
                gal.aper_phot[aper_diam].filterset.band_names) \
                == self.kwargs["band"])[0][0])
        else:
            band_index = self.kwargs["band"]
        SNR = gal.aper_phot[aper_diam].SNR[band_index]
        masked = gal.aper_phot[aper_diam].flux.mask[band_index]
        # fails if masked
        return (
            not masked
            and ((self.kwargs["detect_or_non_detect"].lower() \
            == "detect" and SNR > self.kwargs["SNR_lim"])
            or (self.kwargs["detect_or_non_detect"].lower() \
            == "non_detect" and SNR < self.kwargs["SNR_lim"]))
        )

    def call_cat(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity,
        SED_fit_label: str,
        return_copy: bool = True,
    ) -> Union[NoReturn, Catalogue]:
        if isinstance(self.kwargs["band"], str):
            assert (self.kwargs["band"] in cat.filterset.band_names), \
                galfind_logger.critical(
                    f"{self.kwargs['band']} not in {cat.filterset.band_names}."
                )
        return super().call_cat(cat, aper_diam, SED_fit_label, return_copy)


class UVJ_Selector(Selector):

    @property
    def _selection_name(self) -> str:
        return f"UVJ_{self.kwargs['quiescent_or_star_forming'].lower()}"

    @property
    def _include_kwargs(self) -> List[str]:
        return ["quiescent_or_star_forming"]

    def _assertions(self: Self) -> bool:
        try:
            failed = self.kwargs["quiescent_or_star_forming"].lower() \
                not in ["quiescent", "star_forming"]
        except:
            failed = True
        return failed
    
    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        aper_diam: u.Quantity,
        SED_fit_label: str,
    ) -> bool:
        try:
            assertions = [f"{band}_flux" not in \
                gal.aper_phot[aper_diam].SED_results \
                [SED_fit_label].properties.keys() \
                for band in ["U", "V", "J"]]
            failed = not all(assertions)
        except:
            failed = True
        return failed
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        aper_diam: u.Quantity,
        SED_fit_label: str,
    ) -> bool:
        # extract UVJ colours
        U_minus_V = -2.5 * np.log10(
            (
                gal.aper_phot[aper_diam].SED_results[SED_fit_label].properties["U_flux"]
                / gal.aper_phot[aper_diam].SED_results[SED_fit_label].properties["V_flux"]
            )
            .to(u.dimensionless_unscaled).value
        )
        V_minus_J = -2.5 * np.log10(
            (
                gal.aper_phot[aper_diam].SED_results[SED_fit_label].properties["V_flux"]
                / gal.aper_phot[aper_diam].SED_results[SED_fit_label].properties["J_flux"]
            )
            .to(u.dimensionless_unscaled).value
        )
        # selection from Antwi-Danso2022
        is_quiescent = (
            U_minus_V > 1.23
            and V_minus_J < 1.67
            and U_minus_V > V_minus_J * 0.98 + 0.38
        )
        return (self.kwargs["quiescent_or_star_forming"].lower() \
            == "quiescent" and is_quiescent) or \
            (self.kwargs["quiescent_or_star_forming"].lower() \
            == "star_forming" and not is_quiescent)


class Chi_Sq_Lim_Selector(Selector):
 
    @property
    def _selection_name(self) -> str:
        selection_name = f"chi_sq<{self.kwargs['chi_sq_lim']:.1f}"
        if self.kwargs["reduced"]:
            selection_name = "red_" + selection_name
        return selection_name

    @property
    def _include_kwargs(self) -> List[str]:
        return ["chi_sq_lim", "reduced"]
    
    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["chi_sq_lim"], (int, float))])
            assertions.extend([isinstance(self.kwargs["reduced"], bool)])
            failed = not all(assertions)
        except:
            failed = True
        return failed

    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        aper_diam: u.Quantity,
        SED_fit_label: str,
    ) -> bool:
        if self.kwargs["reduced"]:
            n_bands = len(
                [
                    mask_band
                    for mask_band in gal.aper_phot[aper_diam].flux.mask
                    if not mask_band
                ]
            )
            chi_sq_lim *= n_bands - 1
        else:
            chi_sq_lim = self.kwargs["chi_sq_lim"]
        chi_sq = gal.aper_phot[aper_diam].SED_results[SED_fit_label].chi_sq
        return chi_sq < chi_sq_lim


class Chi_Sq_Diff_Selector(Selector):

    @property
    def _selection_name(self) -> str:
        chi_sq_name = f"chi_sq_diff>{self.kwargs['chi_sq_diff']:.1f}"
        dz_name = f"dz>{self.kwargs['dz']:.1f}"
        return f"{chi_sq_name},{dz_name}"
    
    @property
    def _include_kwargs(self) -> List[str]:
        return ["chi_sq_diff", "dz"]
    
    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["chi_sq_diff"], (int, float))])
            assertions.extend([isinstance(self.kwargs["dz"], (int, float))])
            assertions.extend([self.kwargs["dz"] > 0.0])
            assertions.extend([self.kwargs["chi_sq_diff"] > 0.0])
            failed = not all(assertions)
        except:
            failed = True
        return failed

    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        aper_diam: u.Quantity,
        SED_fit_label: str,
    ) -> bool:
        try:
            assertions = []
            assertions.extend([SED_fit_label in gal.aper_phot[aper_diam].SED_results.keys()])
            gal_SED_fit_labels = self._get_lowz_SED_fit_labels(gal, aper_diam, SED_fit_label)
            assertions.extend([len(gal_SED_fit_labels) > 0])
            failed = not all(assertions)
        except:
            failed = True
        return failed
    
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        aper_diam: u.Quantity,
        SED_fit_label: str,
    ) -> bool:
        # extract redshift + chi_sq of zfree run
        zfree = gal.aper_phot[aper_diam].SED_results[SED_fit_label].z
        chi_sq_zfree = gal.aper_phot[aper_diam].SED_results[SED_fit_label].chi_sq
        # extract redshift and chi_sq of lowz runs
        lowz_SED_fit_labels = filter( \
            lambda label: float(label.split("zmax=")[-1][:3]) \
            < zfree - self.kwargs["dz"], \
            self._get_lowz_SED_fit_labels(gal, aper_diam, SED_fit_label)
        )
        # if no lowz runs, do not select galaxy
        if len(lowz_SED_fit_labels) == 0:
            return False
        else:
            lowz_SED_fit_label = lowz_SED_fit_labels[0]
        z_lowz = gal.aper_phot[aper_diam].SED_results[lowz_SED_fit_label].z
        chi_sq_lowz = gal.aper_phot[aper_diam].SED_results[lowz_SED_fit_label].chi_sq
        return (
            (chi_sq_lowz - chi_sq_zfree > self.kwargs["chi_sq_diff"])
            or (chi_sq_lowz == -1.0)
            or (z_lowz < 0.0)
        )
    
    @staticmethod
    def _extract_lowz_SED_fit_labels(
        gal: Galaxy,
        aper_diam: u.Quantity,
        SED_fit_label: str,
    ) -> List[str]:
        # TODO: Works for EAZY, but not for a general 
        # SED fitting code with different zmax syntax
        return [label for label in \
            gal.aper_phot[aper_diam].SED_results.keys() \
            if "zmax=" in label and label.remove( \
            f"_zmax={label.split('zmax=')[-1][:3]}") \
            == SED_fit_label]
    
    def call_cat(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity,
        SED_fit_label: str,
        return_copy: bool = True,
    ) -> Union[NoReturn, Catalogue]:
        # ensure a lowz run has been run for at least 1 galaxy in the catalogue
        cat_SED_fit_labels = [self._get_lowz_SED_fit_labels(gal, aper_diam, SED_fit_label) for gal in cat]
        assert any(len(gal_labels) > 0 for gal_labels in cat_SED_fit_labels), \
            galfind_logger.critical(
                f"lowz run not run for {SED_fit_label=} for every galaxy."
            )
        return super().call_cat(cat, aper_diam, SED_fit_label, return_copy)


class Robust_zPDF_Selector(Selector):

    @property
    def _selection_name(self) -> str:
        return f"zPDF>{int(self.kwargs['integral_lim'] * 100)}%," + \
            f"|dz|/z<{self.kwargs['dz_over_z']}"
    
    @property
    def _include_kwargs(self) -> List[str]:
        return ["integral_lim", "dz_over_z"]
    
    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["integral_lim"], float)])
            assertions.extend([self.kwargs["integral_lim"] > 0.0])
            assertions.extend([(self.kwargs["integral_lim"] * 100).is_integer()])
            assertions.extend([isinstance(self.kwargs["dz_over_z"], (int, float))])
            assertions.extend([self.kwargs["dz_over_z"] > 0.0])
            failed = not all(assertions)
        except:
            failed = True
        return failed

    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        aper_diam: u.Quantity,
        SED_fit_label: str,
    ) -> bool:
        try:
            failed = gal.aper_phot[aper_diam].SED_results[SED_fit_label].z < 0.0
        except:
            failed = True
        return failed

    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        aper_diam: u.Quantity,
        SED_fit_label: str,
    ) -> bool:
        # extract best fitting redshift - peak of the redshift PDF
        zbest = gal.aper_phot[aper_diam].SED_results[SED_fit_label].z
        integral = gal.aper_phot[aper_diam].SED_results[SED_fit_label]. \
            property_PDFs["z"].integrate_between_lims(
                float(self.kwargs["dz_over_z"]), float(zbest)
            )
        return integral > self.kwargs["integral_lim"]


class Sextractor_Band_Radius_Selector(Selector):

    @property
    def _selection_name(self) -> str:
        if self.kwargs["lim"].unit == u.pix:
            lim_str = f"{self.kwargs['lim'].value:.1f}pix"
        else:
            lim_str = f"{self.kwargs['lim'].to(u.arcsec).value:.1f}as"
        gtr_or_less_str = ">" if self.kwargs["gtr_or_less"].lower() == "gtr" else "<"
        return f"sex_Re_{self.kwargs['band']}{gtr_or_less_str}{lim_str}"

    @property
    def _include_kwargs(self) -> List[str]:
        return ["band", "gtr_or_less", "lim"]

    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["band"], str)])
            assertions.extend([self.kwargs["band"] in json.loads(config.get("Other", "ALL_BANDS"))])
            assertions.extend([self.kwargs["gtr_or_less"].lower() in ["gtr", "less"]])
            assertions.extend([isinstance(self.kwargs["lim"], u.Quantity)])
            assertions.extend([self.kwargs["lim"].unit in [u.pix, u.arcsec]])
            assertions.extend([self.kwargs["lim"].value > 0.0])
            failed = not all(assertions)
        except:
            failed = True
        return failed

    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        **kwargs
    ) -> bool:
        if self.kwargs["gtr_or_less"].lower() == "gtr":
            return gal.sex_Re[self.kwargs["band"]] > self.kwargs["lim"]
        else: # self.kwargs["gtr_or_less"].lower() == "less"
            return gal.sex_Re[self.kwargs["band"]] < self.kwargs["lim"]

    def call_cat(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity,
        SED_fit_label: str,
        return_copy: bool = True,
    ) -> Union[NoReturn, Catalogue]:
        if isinstance(self.kwargs["band"], str):
            assert (self.kwargs["band"] in cat.filterset.band_names), \
                galfind_logger.critical(
                    f"{self.kwargs['band']} not in" + \
                    f" {cat.filterset.band_names}!"
                )
        # load in effective radii as calculated from SExtractor
        cat.load_band_properties_from_cat("FLUX_RADIUS", "sex_Re", None)
        return super().call_cat(cat, aper_diam, SED_fit_label, return_copy)


class Multiple_Selector(Selector):

    def __init__(
        self: Self, 
        selectors: List[Type[Selector]],
        selection_name: Optional[str] = None
    ):
        self.selectors = selectors
        if selection_name is not None:
            self.selection_name = selection_name
        super().__init__()
    
    @property
    def _selection_name(self: Self) -> str:
        if hasattr(self, "selection_name"):
            return self.selection_name
        else:
            return f"{'+'.join([selector._selection_name for selector in self.selectors])}"

    @property
    def _include_kwargs(self: Self) -> List[str]:
        return []
    
    def _assertions(self: Self) -> bool:
        return all(
            selector._assertions()
            for selector in self.selectors
        )

    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        aper_diam: u.Quantity,
        SED_fit_label: str,
    ) -> bool:
        return any(
            selector._failure_criteria(gal, aper_diam, SED_fit_label)
            for selector in self.selectors
        )

    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        aper_diam: u.Quantity,
        SED_fit_label: str,
    ) -> bool:
        return all(
            selector._selection_criteria(gal, aper_diam, SED_fit_label)
            for selector in self.selectors
        )

    
class Kokorev24_LRD_red1(Multiple_Selector):

    def __init__(self):
        super().__init__([
            Colour_Selector(
                colour_bands = ["F115W", "F150W"],
                bluer_or_redder = "bluer",
                colour_val = 0.8
            ),
            Colour_Selector(
                colour_bands = ["F200W", "F277W"],
                bluer_or_redder = "redder",
                colour_val = 0.7
            ),
            Colour_Selector(
                colour_bands = ["F200W", "F356W"],
                bluer_or_redder = "redder",
                colour_val = 1.0
            ),
        ], 
        selection_name = "Kokorev+24_LRD_red1")

class Kokorev24_LRD_red2(Multiple_Selector):

    def __init__(self):
        super().__init__([
            Colour_Selector(
                colour_bands = ["F150W", "F200W"],
                bluer_or_redder = "bluer",
                colour_val = 0.8
            ),
            Colour_Selector(
                colour_bands = ["F277W", "F356W"],
                bluer_or_redder = "redder",
                colour_val = 0.6
            ),
            Colour_Selector(
                colour_bands = ["F277W", "F444W"],
                bluer_or_redder = "redder",
                colour_val = 0.7
            ),
        ], 
        selection_name = "Kokorev+24_LRD_red2")

class Kokorev24_LRD(Multiple_Selector):

    def __init__(self: Self):
        super().__init__([
            Kokorev24_LRD_red1(), 
            Kokorev24_LRD_red2()
        ], 
        selection_name = "Kokorev+24_LRD")


class Unmasked_Bands_Selector(Multiple_Selector):

    def __init__(
        self: Self, 
        band_names: Union[str, List[str]]
    ):
        if isinstance(band_names, str):
            band_names = band_names.split("+")
        selectors = [Unmasked_Band_Selector(band_name = name) for name in band_names]
        super().__init__(selectors, f"unmasked_{'+'.join(band_names)}")

class Unmasked_Instrument_Selector(Multiple_Selector):

    def __init__(
        self: Self, 
        instrument: Union[str, Type[Instrument]]
    ):
        if isinstance(instrument, str):
            assert instrument in expected_instr_bands.keys(), \
                galfind_logger.critical(
                    f"{instrument=} not a valid instrument name."
                )
            instrument = [instr() for instr in Instrument.__subclasses__() \
                if instr.__name__ == instrument][0]
        selectors = [Unmasked_Band_Selector(band_name = name) for name in instrument.filt_names]
        super().__init__(selectors, f"unmasked_{instrument.__class__.__name__}")

    def call_cat(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity,
        SED_fit_label: str,
        return_copy: bool = True,
    ) -> Union[NoReturn, Catalogue]:
        self.selectors = [selector for selector in self.selectors \
            if selector.kwargs["band_name"] in cat.filterset.band_names]
        return super().call_cat(cat, aper_diam, SED_fit_label, return_copy)


class Colour_Colour_Selector(Selector):
   
    @property
    def _selection_name(self: str) -> str:
       raise NotImplementedError
   
    @property
    def _include_kwargs(self: List[str]) -> List[str]:
        return ["colour_bands_arr", "select_func"]


class Sextractor_Bands_Radius_Selector(Multiple_Selector):

    def __init__(
        self: Self,
        band_names: List[str],
        gtr_or_less: str,
        lim: u.Quantity
    ):
        selectors = [Sextractor_Band_Radius_Selector(
            band = band_name, gtr_or_less = gtr_or_less, lim = lim)
            for band_name in band_names]
        if lim.unit == u.pix:
            lim_str = f"{self.kwargs['lim'].value:.1f}pix"
        else:
            lim_str = f"{self.kwargs['lim'].to(u.arcsec).value:.1f}as"
        gtr_or_less_str = ">" if self.kwargs["gtr_or_less"].lower() == "gtr" else "<"
        selection_name = f"sex_Re_{'+'.join(band_names)}{gtr_or_less_str}{lim_str}"
        super().__init__(selectors, selection_name)


class Sextractor_Instrument_Radius_Selector(Multiple_Selector):

    def __init__(
        self: Self,
        instrument: str,
        gtr_or_less: str,
        lim: u.Quantity
    ):
        if isinstance(instrument, str):
            assert instrument in expected_instr_bands.keys(), \
                galfind_logger.critical(
                    f"{instrument=} not a valid instrument name."
                )
            instrument = [instr() for instr in Instrument.__subclasses__() \
                if instr.__name__ == instrument][0]
        selectors = [Sextractor_Band_Radius_Selector(
            band = name, gtr_or_less = gtr_or_less, lim = lim) \
            for name in instrument.filt_names]
        if lim.unit == u.pix:
            lim_str = f"{self.kwargs['lim'].value:.1f}pix"
        else:
            lim_str = f"{self.kwargs['lim'].to(u.arcsec).value:.1f}as"
        gtr_or_less_str = ">" if self.kwargs["gtr_or_less"].lower() == "gtr" else "<"
        selection_name = f"sex_Re_{instrument.__class__.__name__}{gtr_or_less_str}{lim_str}"
        super().__init__(selectors, selection_name)
    
    def call_cat(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity,
        SED_fit_label: str,
        return_copy: bool = True,
    ) -> Union[NoReturn, Catalogue]:
        self.selectors = [selector for selector in self.selectors \
            if selector.kwargs["band_name"] in cat.filterset.band_names]
        return super().call_cat(cat, aper_diam, SED_fit_label, return_copy)


class EPOCHS_Selector(Multiple_Selector):

    def __init__(
        self: Self,
        allow_lowz: bool = False,
        unmasked_instruments: Union[str, List[str]] = "NIRCam"
    ):
        selectors = [
            Bluewards_Lya_Non_Detect_Selector(SNR_lim = 2.0),
            Redwards_Lya_Detect_Selector(SNR_lims = [5.0, 5.0], widebands_only = True),
            Redwards_Lya_Detect_Selector(SNR_lims = 2.0, widebands_only = False),
            Chi_Sq_Lim_Selector(chi_sq_lim = 3.0, reduced = True),
            Chi_Sq_Diff_Selector(chi_sq_diff = 4.0, dz = 0.5),
            Robust_zPDF_Selector(integral_lim = 0.6, dz_over_z = 0.1)
        ]
        # add unmasked instrument selections
        if isinstance(unmasked_instruments, str):
            unmasked_instruments = unmasked_instruments.split("+")
        selectors.extend([Unmasked_Instrument_Selector(instrument) \
            for instrument in unmasked_instruments.split("+")])
        # add 2σ non-detection in first band if wanted
        if not allow_lowz:
            selectors.extend([Band_SNR_Selector( \
                band = 0, SNR_lim = 2.0, detect_or_non_detect = "non_detect")])
        # add hot pixel checks in LW widebands
        selectors.extend([
            Sextractor_Bands_Radius_Selector( \
            band_names = ["F277W", "F356W", "F444W"], \
            gtr_or_less = "gtr", lim = 1.5 * u.pix)
        ])
        lowz_name = "_lowz" if allow_lowz else ""
        unmasked_instr_name = "_" + "+".join(unmasked_instruments)
        selection_name = f"EPOCHS{lowz_name}{unmasked_instr_name}"
        super().__init__(selectors, selection_name = selection_name)


# Catalogue Level Selection functions

# def select_all_bands(self):
#     return self.select_min_bands(len(self.instrument))

# Photometric galaxy property selection functions

    # def select_phot_galaxy_property(
    #     self,
    #     property_name,
    #     gtr_or_less,
    #     property_lim,
    #     SED_fit_params,
    #     update=True,
    # ):
    #     key = SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)
    #     assert property_name in self.phot.SED_results[key].properties.keys()
    #     galfind_logger.warning(
    #         "Ideally need to include appropriate units for photometric galaxy property selection"
    #     )
    #     assert type(property_lim) in [int, float]
    #     assert gtr_or_less in ["gtr", "less", ">", "<"]
    #     if gtr_or_less in ["gtr", ">"]:
    #         selection_name = f"{property_name}>{property_lim}"
    #     else:
    #         selection_name = f"{property_name}<{property_lim}"
    #     if selection_name in self.selection_flags.keys():
    #         galfind_logger.debug(
    #             f"{selection_name} already performed for galaxy ID = {self.ID}!"
    #         )
    #     else:
    #         if (
    #             len(self.phot) == 0
    #         ):  # no data at all (not sure why sextractor does this)
    #             if update:
    #                 self.selection_flags[selection_name] = False
    #             return self, selection_name
    #         property_val = self.phot.SED_results[key].properties[property_name]
    #         if (
    #             gtr_or_less in ["gtr", ">"] and property_val > property_lim
    #         ) or (
    #             gtr_or_less in ["less", "<"] and property_val < property_lim
    #         ):
    #             if update:
    #                 self.selection_flags[selection_name] = True
    #         else:
    #             if update:
    #                 self.selection_flags[selection_name] = False
    #     return self, selection_name

    # def select_phot_galaxy_property_bin(
    #     self,
    #     property_name,
    #     property_lims,
    #     SED_fit_params,
    #     update=True,
    # ):
    #     key = SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)
    #     assert property_name in self.phot.SED_results[key].properties.keys()
    #     galfind_logger.warning(
    #         "Ideally need to include appropriate units for photometric galaxy property selection"
    #     )
    #     assert type(property_lims) in [np.ndarray, list]
    #     assert len(property_lims) == 2
    #     assert property_lims[1] > property_lims[0]
    #     selection_name = (
    #         f"{property_lims[0]}<{property_name}<{property_lims[1]}"
    #     )
    #     if selection_name in self.selection_flags.keys():
    #         galfind_logger.debug(
    #             f"{selection_name} already performed for galaxy ID = {self.ID}!"
    #         )
    #     else:
    #         if (
    #             len(self.phot) == 0
    #         ):  # no data at all (not sure why sextractor does this)
    #             if update:
    #                 self.selection_flags[selection_name] = False
    #             return self, selection_name
    #         property_val = self.phot.SED_results[key].properties[property_name]
    #         if (
    #             property_val > property_lims[0]
    #             and property_val < property_lims[1]
    #         ):
    #             if update:
    #                 self.selection_flags[selection_name] = True
    #         else:
    #             if update:
    #                 self.selection_flags[selection_name] = False
    #     return self, selection_name

# Emission line selection functions

    # def select_rest_UV_line_emitters_dmag(
    #     self,
    #     emission_line_name,
    #     delta_m,
    #     rest_UV_wav_lims=[1_250.0, 3_000.0] * u.AA,
    #     medium_bands_only=True,
    #     SED_fit_params=EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
    #     update=True,
    # ):
    #     assert (
    #         line_diagnostics[emission_line_name]["line_wav"]
    #         > rest_UV_wav_lims[0] * rest_UV_wav_lims.unit
    #         and line_diagnostics[emission_line_name]["line_wav"]
    #         < rest_UV_wav_lims[1] * rest_UV_wav_lims.unit
    #     )
    #     assert type(delta_m) in [int, np.int64, float, np.float64]
    #     assert u.has_physical_type(rest_UV_wav_lims) == "length"
    #     assert type(medium_bands_only) in [bool, np.bool_]
    #     selection_name = f"{emission_line_name},dm{'_med' if medium_bands_only else ''}>{delta_m:.1f},UV_{str(list(np.array(rest_UV_wav_lims.value).astype(int))).replace(' ', '')}AA"
    #     if selection_name in self.selection_flags.keys():
    #         galfind_logger.debug(
    #             f"{selection_name} already performed for galaxy ID = {self.ID}!"
    #         )
    #     else:
    #         if (
    #             len(self.phot) == 0
    #         ):  # no data at all (not sure why sextractor does this)
    #             if update:
    #                 self.selection_flags[selection_name] = False
    #             return self, selection_name
    #         phot_rest = deepcopy(
    #             self.phot.SED_results[
    #                 SED_fit_params["code"].label_from_SED_fit_params(
    #                     SED_fit_params
    #                 )
    #             ].phot_rest
    #         )
    #         # find bands that the emission line lies within
    #         obs_frame_emission_line_wav = line_diagnostics[emission_line_name][
    #             "line_wav"
    #         ] * (1.0 + phot_rest.z)
    #         included_bands = self.phot.instrument.bands_from_wavelength(
    #             obs_frame_emission_line_wav
    #         )
    #         # determine index of the closest band to the emission line
    #         closest_band_index = (
    #             self.phot.instrument.nearest_band_index_to_wavelength(
    #                 obs_frame_emission_line_wav, medium_bands_only
    #             )
    #         )
    #         central_wav = self.phot.instrument[
    #             closest_band_index
    #         ].WavelengthCen
    #         # if there are no included bands or the closest band is masked
    #         if (
    #             len(included_bands) == 0
    #             or self.phot.flux.mask[closest_band_index]
    #         ):
    #             if update:
    #                 self.selection_flags[selection_name] = False
    #             return self, selection_name
    #         # calculate beta excluding the bands that the emission line contaminates
    #         phot_rest.crop_phot(
    #             [
    #                 self.phot.instrument.index_from_band_name(band.band_name)
    #                 for band in included_bands
    #             ]
    #         )
    #         A, beta = phot_rest.calc_beta_phot(rest_UV_wav_lims, iters=1)
    #         # make mock SED to calculate bandpass averaged flux from
    #         mock_SED_obs = Mock_SED_obs.from_Mock_SED_rest(
    #             Mock_SED_rest.power_law_from_beta_m_UV(
    #                 beta,
    #                 funcs.power_law_beta_func(1_500.0, 10**A, beta),
    #                 mag_units=u.Jy,
    #                 wav_lims=[
    #                     self.phot.instrument[
    #                         closest_band_index
    #                     ].WavelengthLower50,
    #                     self.phot.instrument[
    #                         closest_band_index
    #                     ].WavelengthUpper50,
    #                 ],
    #             ),
    #             self.z,
    #             IGM=None,
    #         )
    #         mag_cont = funcs.convert_mag_units(
    #             central_wav,
    #             mock_SED_obs.calc_bandpass_averaged_flux(
    #                 self.phot.instrument[closest_band_index].wav,
    #                 self.phot.instrument[closest_band_index].trans,
    #             )
    #             * u.erg
    #             / (u.s * (u.cm**2) * u.AA),
    #             u.ABmag,
    #         )
    #         # determine observed magnitude
    #         mag_obs = funcs.convert_mag_units(
    #             central_wav, self.phot[closest_band_index], u.ABmag
    #         )
    #         if (mag_cont - mag_obs).value > delta_m:
    #             if update:
    #                 self.selection_flags[selection_name] = True
    #         else:
    #             if update:
    #                 self.selection_flags[selection_name] = False
    #     return self, selection_name

    # def select_rest_UV_line_emitters_sigma(
    #     self,
    #     emission_line_name,
    #     sigma,
    #     rest_UV_wav_lims=[1_250.0, 3_000.0] * u.AA,
    #     medium_bands_only=True,
    #     SED_fit_params=EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
    #     update=True,
    # ) -> tuple[Self, str]:
    #     assert (
    #         line_diagnostics[emission_line_name]["line_wav"]
    #         > rest_UV_wav_lims[0]
    #         and line_diagnostics[emission_line_name]["line_wav"]
    #         < rest_UV_wav_lims[1]
    #     )
    #     assert type(sigma) in [int, np.int64, float, np.float64]
    #     assert u.get_physical_type(rest_UV_wav_lims) == "length"
    #     assert type(medium_bands_only) in [bool, np.bool_]
    #     selection_name = f"{emission_line_name},sigma{'_med' if medium_bands_only else ''}>{sigma:.1f},UV_{str(list(np.array(rest_UV_wav_lims.value).astype(int))).replace(' ', '')}AA"
    #     if selection_name in self.selection_flags.keys():
    #         galfind_logger.debug(
    #             f"{selection_name} already performed for galaxy ID = {self.ID}!"
    #         )
    #     else:
    #         if (
    #             len(self.phot) == 0
    #         ):  # no data at all (not sure why sextractor does this)
    #             if update:
    #                 self.selection_flags[selection_name] = False
    #             return self, selection_name
    #         SED_results_key = SED_fit_params["code"].label_from_SED_fit_params(
    #             SED_fit_params
    #         )
    #         phot_rest = deepcopy(
    #             self.phot.SED_results[
    #                 SED_fit_params["code"].label_from_SED_fit_params(
    #                     SED_fit_params
    #                 )
    #             ].phot_rest
    #         )
    #         # find bands that the emission line lies within
    #         obs_frame_emission_line_wav = line_diagnostics[emission_line_name][
    #             "line_wav"
    #         ] * (1.0 + phot_rest.z)
    #         included_bands = self.phot.instrument.bands_from_wavelength(
    #             obs_frame_emission_line_wav
    #         )
    #         # determine index of the closest band to the emission line
    #         closest_band = self.phot.instrument.nearest_band_to_wavelength(
    #             obs_frame_emission_line_wav, medium_bands_only
    #         )
    #         if type(closest_band) == type(None):
    #             if update:
    #                 self.selection_flags[selection_name] = False
    #             return self, selection_name
    #         closest_band_index = self.phot.instrument.index_from_band_name(
    #             closest_band.band_name
    #         )
    #         central_wav = self.phot.instrument[
    #             closest_band_index
    #         ].WavelengthCen
    #         # if there are no included bands or the closest band is masked
    #         if (
    #             len(included_bands) == 0
    #             or self.phot.flux.mask[closest_band_index]
    #         ):
    #             if update:
    #                 self.selection_flags[selection_name] = False
    #             return self, selection_name
    #         # calculate beta excluding the bands that the emission line contaminates
    #         phot_rest.crop_phot(
    #             [
    #                 self.phot.instrument.index_from_band_name(band.band_name)
    #                 for band in included_bands
    #             ]
    #         )
    #         A, beta = phot_rest.calc_beta_phot(
    #             rest_UV_wav_lims, iters=1, incl_errs=True
    #         )
    #         m_UV = funcs.convert_mag_units(
    #             1_500.0
    #             * (1.0 + self.phot.SED_results[SED_results_key].z)
    #             * u.AA,
    #             funcs.power_law_beta_func(1_500.0, 10**A, beta)
    #             * u.erg
    #             / (u.s * (u.cm**2) * u.AA),
    #             u.ABmag,
    #         )
    #         # make mock SED to calculate bandpass averaged flux from
    #         mock_SED_rest = Mock_SED_rest.power_law_from_beta_m_UV(
    #             beta, m_UV
    #         )  # , wav_range = \
    #         # [funcs.convert_wav_units(self.phot.instrument[closest_band_index].WavelengthLower50, u.AA).value / (1. + self.phot.SED_results[SED_results_key].z), \
    #         # funcs.convert_wav_units(self.phot.instrument[closest_band_index].WavelengthUpper50, u.AA).value / (1. + self.phot.SED_results[SED_results_key].z)] * u.AA)
    #         mock_SED_obs = Mock_SED_obs.from_Mock_SED_rest(
    #             mock_SED_rest,
    #             self.phot.SED_results[SED_results_key].z,
    #             IGM=None,
    #         )
    #         flux_cont = funcs.convert_mag_units(
    #             central_wav,
    #             mock_SED_obs.calc_bandpass_averaged_flux(
    #                 self.phot.instrument[closest_band_index].wav,
    #                 self.phot.instrument[closest_band_index].trans,
    #             )
    #             * u.erg
    #             / (u.s * (u.cm**2) * u.AA),
    #             u.Jy,
    #         )
    #         # determine observed magnitude
    #         flux_obs_err = funcs.convert_mag_err_units(
    #             central_wav,
    #             self.phot.flux[closest_band_index],
    #             [
    #                 self.phot.flux_errs[closest_band_index].value,
    #                 self.phot.flux_errs[closest_band_index].value,
    #             ]
    #             * self.phot.flux_errs.unit,
    #             u.Jy,
    #         )
    #         flux_obs = funcs.convert_mag_units(
    #             central_wav, self.phot.flux[closest_band_index], u.Jy
    #         )
    #         snr_band = abs((flux_obs - flux_cont).value) / np.mean(
    #             flux_obs_err.value
    #         )
    #         mag_cont = funcs.convert_mag_units(
    #             central_wav,
    #             mock_SED_obs.calc_bandpass_averaged_flux(
    #                 self.phot.instrument[closest_band_index].wav,
    #                 self.phot.instrument[closest_band_index].trans,
    #             )
    #             * u.erg
    #             / (u.s * (u.cm**2) * u.AA),
    #             u.ABmag,
    #         )
    #         print(
    #             self.ID,
    #             snr_band,
    #             beta,
    #             mag_cont,
    #             self.phot.SED_results[SED_results_key].z,
    #             closest_band.band_name,
    #         )
    #         if snr_band > sigma:
    #             if update:
    #                 self.selection_flags[selection_name] = True
    #         else:
    #             if update:
    #                 self.selection_flags[selection_name] = False
    #     return self, selection_name

# Depth region selection

# def select_depth_region(self, band, region_ID, update=True):
#     return NotImplementedError
"""Base class for galaxy catalogue containers.

Provides abstract base class wrapping Galaxy lists with
catalogue-level operations
including cropping, cross-matching, and concatenation.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    NoReturn,
    Optional,
    Tuple,
    Union,
)

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, join, vstack
from numpy.typing import NDArray
from tqdm import tqdm

if TYPE_CHECKING:
    from . import (
        Catalogue_Creator,
        Data,
        Galaxy,
        Mask_Selector,
        Property_Calculator_Base,
        Selector,
    )
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from .. import config, galfind_logger
from ..sed_fitting.SED_codes import SED_code
from ..utils import useful_funcs_austind as funcs
from ..utils.exceptions import (
    EmptyCatalogueError,
    GalfindError,
    GalfindTypeError,
    IncompatibleKwargsError,
    InvalidOptionError,
    InvalidUnitError,
    LengthMismatchError,
    MissingDataError,
    MissingKeyError,
    RangeError,
)


class Catalogue_Base:
    """Base class for galaxy catalogue containers.

    Wraps a list of `Galaxy` objects together with a `Catalogue_Creator`
    used to read/write the underlying FITS catalogue, providing
    catalogue-level operations such as cropping, cross-matching,
    concatenation, plotting and FITS I/O. Subclassed by e.g. `Catalogue`
    and `Spectral_Catalogue`. Attributes not defined directly on the
    instance (e.g. ``survey``, ``version``, ``filterset``, ``cat_path``)
    are looked up on `cat_creator`, or computed per-galaxy from the
    `Galaxy` objects in `gals`, via `__getattr__`.

    Parameters
    ----------
    gals : `list` of `Galaxy`
        Galaxies making up this catalogue.
    cat_creator : `Catalogue_Creator`
        Object responsible for reading/writing the underlying FITS
        catalogue and tracking applied crops.

    Attributes
    ----------
    gals : `list` of `Galaxy`
        Galaxies making up this catalogue.
    cat_creator : `Catalogue_Creator`
        Object responsible for reading/writing the underlying FITS
        catalogue and tracking applied crops.
    """

    # later on, the gal_arr should be calculated from the Instrument
    # and sex_cat path, with SED codes already given
    def __init__(
        self,
        gals: List[Galaxy],
        cat_creator: Catalogue_Creator,
        # SED_rest_properties: Dict[
        #     str, Union[u.Quantity, u.Magnitude, u.Dex]
        # ] = {},
    ):
        self.gals = gals
        self.cat_creator = cat_creator
        # self.SED_rest_properties = SED_rest_properties
        # concat is commutative for catalogues
        self.__radd__ = self.__add__
        # cross-match is commutative for catalogues - not if they
        # are of different classes
        # self.__rmul__ = self.__mul__

    # %% Overloaded operators

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__.upper()}({self.survey},"
            + f"{self.version},{self.filterset.instrument_name})"
        )

    def __str__(self: Self) -> str:
        # display_selections = ["EPOCHS", "BROWN_DWARF"]
        output_str = funcs.line_sep
        output_str += f"{repr(self)}:\n"
        output_str += funcs.band_sep
        output_str += f"CAT PATH = {self.cat_path}\n"
        # access table header to display what has been run for this catalogue
        tab = self.cat_creator._open_tab("ID")
        output_str += f"TOTAL GALAXIES = {len(tab)}\n"
        output_str += f"RA RANGE = {self.ra_range}\n"
        output_str += f"DEC RANGE = {self.dec_range}\n"
        output_str += funcs.band_sep
        output_str += str(self.filterset)
        # display what other things have previously been calculated
        # for this catalogue
        if hasattr(self, "SED_results"):
            output_str += "SED FITTING RESULTS:\n"
            for aper_diam, SED_results in self.SED_results.items():
                output_str += funcs.band_sep
                SED_results_str = ",".join(
                    [repr(SED_result) for SED_result in SED_results]
                )
                output_str += f"{aper_diam}: {SED_results_str}\n"
            output_str += funcs.line_sep
        # breakpoint()
        # output_str += "CAT STATUS = SEXTRACTOR, "
        # for i, (key, value) in enumerate(cat.meta.items()):
        #     if key in ["DEPTHS", "MASKED"] + [
        #         f"RUN_{subclass.__name__}"
        #         for subclass in SED_code.__subclasses__()
        #     ]:
        #         output_str += f"{key.split('_')[-1]}, "
        # for sel_criteria in display_selections:
        #     if sel_criteria in cat.colnames:
        #         output_str += f"{sel_criteria} SELECTION, "
        # output_str += "\n"
        # # display total number of galaxies that satisfy the
        # # selection criteria previously performed
        # if print_sel_criteria:
        #     for sel_criteria in display_selections:
        #         if sel_criteria in cat.colnames:
        #             output_str += (
        #                 f"N_GALS_{sel_criteria} = "
        #                 f"{len(cat[cat[sel_criteria]])}\n"
        #             )
        # output_str += funcs.band_sep
        # # display crops that have been performed on this specific object
        # if self.crops != []:
        #     output_str += f"N_GALS_OBJECT = {len(self)}\n"
        #     output_str += f"CROPS = {' + '.join(self.crops)}\n"

        # if hasattr(self, "SED_rest_properties"):
        #     if len(self.SED_rest_properties) >= 1:
        #         output_str += funcs.band_sep
        #         for key, properties in self.SED_rest_properties.items():
        #             output_str += "Rest frame SED properties:\n"
        #             output_str += f"{key}: {str(properties)}\n"
        #             output_str += funcs.band_sep
        # if print_cls_name:
        #     output_str += funcs.line_sep
        return output_str

    def __len__(self):
        return len(self.gals)

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter > len(self) - 1:
            raise StopIteration
        else:
            gal = self[self.iter]
            self.iter += 1
            return gal

    def __getitem__(self, index: Any) -> Optional[Union[Galaxy, List[Galaxy]]]:
        if len(self) == 0:
            raise EmptyCatalogueError(
                f"Cannot index {repr(self)}: it has 0 galaxies. At "
                "least one Galaxy is required in self.gals."
            )
        from ..selection.Selector import Selector

        if isinstance(index, int):
            return self.gals[index]
        elif isinstance(index, (list, np.ndarray)):
            if len(index) == 1 and all(
                isinstance(index_, int) for index_ in index
            ):
                return self.gals[index[0]]
            elif all(
                isinstance(index_, tuple(Selector.__subclasses__()))
                for index_ in index
            ):
                keep_arr = []
                for i, index_ in enumerate(index):
                    # run selection if not already done
                    if not all(
                        index_.name in gal.selection_flags for gal in self
                    ):
                        [index_(gal, return_copy=False) for gal in self]
                    else:
                        pass
                    keep_arr.append(
                        [gal.selection_flags[index_.name] for gal in self]
                    )
                return list(
                    np.array(self.gals)[np.logical_and.reduce(keep_arr)]
                )
        if isinstance(index, (slice, np.ndarray)):
            return list(np.array(self.gals)[index])
        elif isinstance(index, list):
            return list(np.array(self.gals)[np.array(index)])
        # elif isinstance(index, dict):
        #     # make this more general
        #     keep_arr = []
        #     for key, values in index.items():
        #         if key == "ID":
        #             if not isinstance(values, (list, np.ndarray)):
        #                 values = [int(values)]
        #             values = np.array(values).astype(int)
        #             keep_arr.extend(
        #                 [np.array(
        #                     [
        #                         True if getattr(gal, key) in values
        #                         else False
        #                         for i, gal in enumerate(self)
        #                     ]
        #                 )]
        #             )
        #     if len(keep_arr) > 0:
        #         self_copy.gals = list(np.array(self.gals)[np.array( \
        #             np.logical_and.reduce(keep_arr)).astype(bool)])
        #         if len(self_copy.gals) == 1:
        #             return self_copy.gals[0]
        #     else:
        #         return None
        elif isinstance(index, tuple(Selector.__subclasses__())):
            # run selection if not already done
            if not all(index.name in gal.selection_flags for gal in self):
                [index(gal, return_copy=False) for gal in self]
            keep_arr = [gal.selection_flags[index.name] for gal in self]
            return list(np.array(self.gals)[np.array(keep_arr)])

    def crop(
        self: Self,
        selector: Type[Selector],
    ) -> Self:
        """Crop this catalogue in place to galaxies satisfying a selector.

        If `selector` has not already been applied (tracked via
        `self.crops`), runs it on every galaxy (if not already run) and
        keeps only those satisfying it, recording `selector` as an
        applied crop on `self.cat_creator`.

        Parameters
        ----------
        selector : `Selector`
            Selection criterion to crop the catalogue with.

        Returns
        -------
        `Self`
            This catalogue, cropped in place.
        """
        if not any(
            crops._selection_name == selector._selection_name
            for crops in self.crops
        ):
            if len(self) == 0:
                galfind_logger.warning(f"No galaxies in {repr(self)} to crop!")
            else:
                self.gals = np.array(self[selector])
            self.cat_creator.crops.append(selector)
        return self

    # only acts on attributes that don't already exist in Catalogue
    def __getattr__(
        self: Self,
        property_name: str,  # , origin: Union[str, dict] = "gal"
    ) -> np.ndarray:
        if property_name in self.cat_creator.__dict__.keys():
            return getattr(self.cat_creator, property_name)
        elif (
            all(hasattr(gal, property_name) for gal in self) and len(self) > 0
        ):
            attr_arr = [getattr(gal, property_name) for gal in self]
            # attr_arr = [gal.__getattr__(property_name, origin)
            # for gal in self]
            # ensure all units are the same
            if all(
                isinstance(attr, (u.Quantity, u.Magnitude, u.Dex))
                for attr in attr_arr
            ):
                if not all(attr.unit == attr_arr[0].unit for attr in attr_arr):
                    raise InvalidUnitError(
                        f"Galaxies in {repr(self)} have mismatched units "
                        f"for property {property_name!r}: "
                        f"{[str(attr.unit) for attr in attr_arr]!r}; all "
                        "galaxies must share the same unit."
                    )
                attr_arr = [attr.value for attr in attr_arr] * u.Unit(
                    attr_arr[0].unit
                )
            return attr_arr
        else:
            raise AttributeError

    def __setattr__(self, name, value, obj="cat"):
        if obj == "cat":
            super().__setattr__(name, value)
        elif obj == "gal":
            # set attributes of individual galaxies within the catalogue
            for i, gal in enumerate(self):
                if isinstance(value, tuple([list, np.array])):
                    setattr(gal, name, value[i])
                else:
                    setattr(gal, name, value)

    def __setitem__(self, index, gal):
        self.gals[index] = gal

    def __deepcopy__(self, memo):
        galfind_logger.debug(f"deepcopy({self.__class__.__name__})")
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            try:
                setattr(result, key, deepcopy(value, memo))
            except Exception as e:
                raise GalfindError(
                    f"deepcopy({self.__class__.__name__}) failed for "
                    f"attribute {key!r}={value!r}: {e}"
                ) from e
        return result

    def __add__(self, cat, out_survey=None):
        if not cat.__class__.__name__ == "Spectral_Catalogue":
            from .Multiple_Catalogue import Combined_Catalogue

            # concat catalogues
            if out_survey is None:
                out_survey = "+".join([self.survey, cat.survey])
            return Combined_Catalogue([self, cat], survey=out_survey)

    def __sub__(self):
        pass

    def __mul__(
        self,
        other,
        out_survey=None,
        max_sep=1.0 * u.arcsec,
        match_type="compare_within_radius",
    ):
        """
        'Multiply' two catalogues by performing a cross-match and
        filtering the best matching galaxies.
        Ensures no duplicate galaxies are present in the output catalogue.

        Parameters:
        - other: Catalogue_Base
            The other catalogue to multiply with.
        - out_survey: str, optional
            The name of the output survey. Default is None.
        - max_sep: Quantity, optional
            The maximum separation allowed for matching galaxies.
            Default is 1.0 arcsec.
        - match_type: str, optional
            The type of matching to perform. Options are 'nearest'
            and 'compare_within_radius'.
            Default is 'compare_within_radius'.

        Returns:
        - Combined_Catalogue
            The resulting Combined_Catalogue after performing the
            multiplication, with duplicates removed.

        Raises:
        - LengthMismatchError
            If the cross-match produces a different number of matched
            galaxies in each catalogue.

        """
        from .Multiple_Catalogue import Combined_Catalogue

        # cross-match catalogues
        # update .fits tables with cross-matched version
        # open tables
        self_copy = deepcopy(self)

        if isinstance(other, Combined_Catalogue):
            other_copy_gals = other.cat_arr
        else:
            other_copy_gals = [other]

        new_copies = []
        for o in other_copy_gals:
            other_copy = deepcopy(o)
            # Convert from list of SkyCoord to SkyCoord(arra)
            sky_coords_cat = SkyCoord(
                self_copy.RA, self_copy.DEC, unit=(u.deg, u.deg), frame="icrs"
            )
            other_sky_coords = SkyCoord(
                other_copy.RA,
                other_copy.DEC,
                unit=(u.deg, u.deg),
                frame="icrs",
            )
            """
            band = self_copy.data.forced_phot_band
            wcs_self = WCS(
                fits.getheader(
                    self_copy.data.im_paths[band],
                    ext=self_copy.data.im_exts[band],
                )
            )
            wcs_other = WCS(
                fits.getheader(
                    other_copy.data.im_paths[band],
                    ext=other_copy.data.im_exts[band],
                )
            )
            if not (
                any(sky_coords_cat.contained_by(wcs_self))
                and any(other_sky_coords.contained_by(wcs_other))
            ):
                cat_matches = []
                other_cat_matches = []
                print('Skipping')
                continue

            # Check if likely to have any matches
            idx, _, _, _ = sky_coords_cat.search_around_sky(
                other_sky_coords, 5 * u.arcsec
            )


            if len(idx) == 0:
                print('Skipping')
                cat_matches = []
                other_cat_matches = []

                continue
            """
            if match_type == "nearest":
                # This just takes the nearest galaxy as the best match
                idx, d2d, d3d = sky_coords_cat.match_to_catalog_sky(
                    other_sky_coords
                )
                # Also check mask - don't keep masked galaxies where
                # there is an unmasked match
                sep_constraint = d2d < max_sep
                # Get indexes of matches
                cat_matches = np.arange(len(sky_coords_cat))[sep_constraint]
                other_cat_matches = idx[sep_constraint]

            elif match_type == "compare_within_radius":
                # This finds all matches within a certain radius and
                # compares the photometry of the galaxies
                cat_matches = []
                other_cat_matches = []

                for pos, coord in tqdm(
                    enumerate(self_copy.sky_coord),
                    desc="Cross-matching galaxies",
                    disable=galfind_logger.getEffectiveLevel() > logging.INFO,
                ):
                    # Need to save index of match in other_sky_coords

                    d2d = coord.separation(other_sky_coords)

                    indexes = np.argwhere(d2d < max_sep)
                    indexes = np.ndarray.flatten(indexes)

                    if len(indexes) == 0:
                        continue
                    elif len(indexes) == 1:
                        cat_matches.append(pos)
                        other_cat_matches.append(indexes[0])
                    else:
                        # Compare fluxes and choose the closest match
                        # Get bands that are in both galaxies

                        coord_gal = self_copy.gals[pos]

                        other_gals = np.ndarray.flatten(
                            other_copy.gals[indexes]
                        )
                        # Save indexes of other_gals in other_sky_coords
                        other_gals_indexes = np.arange(len(other_sky_coords))[
                            indexes
                        ]

                        bands_gal1 = np.array(
                            coord_gal.phot.instrument.filt_names
                        )[coord_gal.phot.flux.mask]

                        chi_squareds = []
                        for other_gal in other_gals:
                            bands_gal2 = np.array(
                                other_gal.phot.instrument.filt_names
                            )[other_gal.phot.flux.mask]
                            matched_bands = list(
                                set(bands_gal1).union(set(bands_gal2))
                            )

                            if len(matched_bands) == 0:
                                continue
                            # Compare fluxes in matched bands
                            indexes_bands = np.argwhere(
                                [
                                    band in matched_bands
                                    for band in (
                                        coord_gal.phot.instrument.filt_names
                                    )
                                ]
                            )
                            indexes_other_bands = np.argwhere(
                                [
                                    band in matched_bands
                                    for band in (
                                        other_gal.phot.instrument.filt_names
                                    )
                                ]
                            )
                            coord_gal_fluxes = coord_gal.phot.flux[
                                indexes_bands
                            ]
                            coord_gal_flux_errs = coord_gal.phot.flux_errs[
                                indexes_bands
                            ]
                            other_gal_fluxes = other_gal.phot.flux[
                                indexes_other_bands
                            ]

                            # Chi-squared comparison
                            try:
                                chi_squared = np.sum(
                                    (coord_gal_fluxes - other_gal_fluxes) ** 2
                                    / (coord_gal_flux_errs**2)
                                )
                            except ValueError:
                                chi_squareds.append(1000000)
                                continue

                            chi_squareds.append(chi_squared)
                        if len(chi_squareds) == 0:
                            continue
                        best_match_index = int(
                            np.squeeze(np.argmin(chi_squareds))
                        )
                        # pop empty dimensions
                        cat_matches.append(pos)
                        other_cat_matches.append(
                            other_gals_indexes[best_match_index]
                        )

            if len(cat_matches) != len(other_cat_matches):
                raise LengthMismatchError(
                    f"cat_matches has length {len(cat_matches)} but "
                    f"other_cat_matches has length "
                    f"{len(other_cat_matches)}; cross-match indices "
                    "must be 1-to-1."
                )  # check that the matches are 1-to-1
            print("Getting galaxies")
            print(cat_matches)
            cat_matches = np.array(cat_matches, dtype=int)
            gal_matched_cat = self_copy[cat_matches]
            # Use indexes instead
            other_cat_matches = np.array(other_cat_matches, dtype=int)
            breakpoint()
            gal_matched_other = other_copy[other_cat_matches]
            print("Obtained matched galaxies")
            if len(gal_matched_cat) != len(gal_matched_other):
                raise LengthMismatchError(
                    f"gal_matched_cat has length {len(gal_matched_cat)} "
                    f"but gal_matched_other has length "
                    f"{len(gal_matched_other)}; cross-match indices "
                    "must be 1-to-1."
                )  # check that the matches are 1-to-1

            if len(gal_matched_cat) > 0:
                if (
                    self_copy.__class__.__name__ == "Catalogue"
                    and other_copy.__class__.__name__ == "Spectral_Catalogue"
                ):
                    # update catalogue and galaxies
                    self_copy.gals = [
                        deepcopy(gal).add_spectra(spectra)
                        for gal, spectra in tqdm(
                            zip(gal_matched_cat, gal_matched_other),
                            total=len(gal_matched_cat),
                            desc="Appending spectra to catalogue!",
                            disable=galfind_logger.getEffectiveLevel()
                            > logging.INFO,
                        )
                    ]
                    return self_copy
                else:
                    print(f"Total matches = {len(gal_matched_cat)}")
                    for gal1, gal2 in tqdm(
                        zip(gal_matched_cat, gal_matched_other),
                        desc="Filtering best galaxy for matches",
                    ):
                        # Compare the two galaxies and choose the better one
                        bands_gal1 = np.array(gal1.phot.instrument.filt_names)[
                            gal1.phot.flux.mask
                        ]
                        bands_gal2 = np.array(gal2.phot.instrument.filt_names)[
                            gal2.phot.flux.mask
                        ]

                        filt_names_union = list(
                            set(bands_gal1).union(set(bands_gal2))
                        )

                        if len(bands_gal2) > len(bands_gal1):
                            self_copy.remove_gal(id=gal1.ID)
                        elif len(bands_gal2) < len(bands_gal1):
                            other_copy.remove_gal(id=gal2.ID)
                        else:
                            # If same bands, choose galaxy with deeper depth
                            # Get matching bands between the two
                            # galaxies - only use if not masked
                            # logical comparison of depth in each
                            # band, keeping the galaxy with the
                            # deeper depth in more bands
                            # gal1.phot.depths is just an array.
                            # Need to slice by position
                            indexes_gal1 = np.argwhere(
                                [
                                    band in filt_names_union
                                    for band in gal1.phot.instrument.filt_names
                                ]
                            )
                            depths_gal1 = gal1.phot.depths[indexes_gal1]
                            indexes_gal2 = np.argwhere(
                                [
                                    band in filt_names_union
                                    for band in gal2.phot.instrument.filt_names
                                ]
                            )
                            depths_gal2 = gal2.phot.depths[indexes_gal2]
                            # Compare depths
                            if np.sum(depths_gal1 > depths_gal2) > np.sum(
                                depths_gal1 < depths_gal2
                            ):
                                self_copy.remove_gal(id=gal1.ID)
                            elif np.sum(depths_gal1 > depths_gal2) < np.sum(
                                depths_gal1 < depths_gal2
                            ):
                                other_copy.remove_gal(id=gal2.ID)
                            else:
                                # Choose first galaxy
                                self_copy.remove_gal(id=gal1.ID)

            new_copies.append(other_copy)

        out_survey = "+".join([self.survey, other.survey])
        return Combined_Catalogue([self_copy, *new_copies], survey=out_survey)

    @property
    def crop_name(self) -> List[str]:
        """`list` of `str`: Names of the crops applied to this
        catalogue, delegated to `cat_creator.crop_name`."""
        return self.cat_creator.crop_name

    # # Need to save the cross-match distances

    # def combine_and_remove_duplicates(
    #     self,
    #     other,
    #     out_survey=None,
    #     max_sep=1.0 * u.arcsec,
    #     match_type="nearest",
    # ):
    #     "Alias for self * other"
    #     return self.__mul__(
    #         other,
    #         out_survey=out_survey,
    #         max_sep=max_sep,
    #         match_type=match_type,
    #     )

    @property
    def cat_dir(self):
        """`str`: Directory component of `self.cat_path`."""
        return funcs.split_dir_name(self.cat_path, "dir")

    @property
    def cat_name(self):
        """`str`: File name component of `self.cat_path`."""
        return funcs.split_dir_name(self.cat_path, "name")

    @property
    def ra_range(self):
        """`astropy.units.Quantity`: [min, max] right ascension of
        the galaxies in this catalogue, cached after first access."""
        try:
            return self._ra_range
        except Exception:
            self._ra_range = [
                np.min(self.RA.value),
                np.max(self.RA.value),
            ] * self.RA.unit
            return self._ra_range

    @property
    def dec_range(self):
        """`astropy.units.Quantity`: [min, max] declination of the
        galaxies in this catalogue, cached after first access."""
        try:
            return self._dec_range
        except Exception:
            self._dec_range = [
                np.min(self.DEC.value),
                np.max(self.DEC.value),
            ] * self.DEC.unit
            return self._dec_range

    @property
    def select_colnames(self) -> List[str]:
        """`list` of `str`: Column names of the boolean selection
        flags stored in the ``"SELECTION"`` HDU, or `[]` if that HDU
        does not exist."""
        tab = self.cat_creator._open_tab("SELECTION")
        if tab is None:
            return []
        else:
            return self.cat_creator.get_selection_labels(tab)

    # TODO: Time this function with decorator
    def cross_match(
        self: Self,
        other: Type[Catalogue_Base],
        max_sep: u.Quantity,
    ) -> Dict[Galaxy, List[Tuple[float, Galaxy]]]:
        """Cross-match the galaxies in this catalogue against another
        catalogue.

        Finds all pairs of galaxies (one from this catalogue, one from
        `other`) separated on sky by less than `max_sep`.

        Parameters
        ----------
        other : `Catalogue_Base`
            Catalogue to cross-match this catalogue's galaxies against.
        max_sep : `astropy.units.Quantity`
            Maximum on-sky separation for a pair of galaxies to be
            considered a match.

        Returns
        -------
        `dict`
            Dictionary keyed by `Galaxy`, with values given by a list of
            ``(separation, Galaxy)`` tuples for each matched galaxy
            within `max_sep`.

        Raises
        ------
        GalfindTypeError
            If `max_sep` is `None`.
        """
        if max_sep is None:
            raise GalfindTypeError(
                "max_sep=None passed to cross_match; must be an "
                "astropy.units.Quantity giving the maximum on-sky "
                "separation."
            )
        galfind_logger.info(
            f"Cross-matching {repr(self)} with {repr(other)} within {max_sep}!"
        )
        max_sep = funcs.validate_quantity(max_sep, "angle")
        # make sky coordinates of self and other
        self_coords = SkyCoord(self.RA, self.DEC, frame="icrs")
        other_coords = SkyCoord(other.RA, other.DEC, frame="icrs")
        # match and populate dict
        idx_self, idx_other, sep2d, _ = self_coords.search_around_sky(
            other_coords, max_sep
        )
        matches = {}
        self_gals_copy = deepcopy(self.gals)
        other_gals_copy = deepcopy(other.gals)
        sep2d = sep2d.to(u.arcsec)
        for i_src, i_match, sep in zip(idx_self, idx_other, sep2d):
            try:
                key = self_gals_copy[i_match]
                item = (sep, other_gals_copy[i_src])
                if key not in matches.keys():
                    matches[key] = [item]
                else:
                    matches[key].extend([item])
            except Exception:
                breakpoint()

        return matches

    # TODO: should be __sub__ instead
    def remove_gal(self, index=None, id=None):
        """Remove a galaxy from `self.gals` by index or ID.

        Parameters
        ----------
        index : `int`, optional
            Index of the galaxy to remove from `self.gals`. Default is
            `None`.
        id : `int`, optional
            ID of the galaxy to remove from `self.gals`. Only used if
            `index` is `None`. Default is `None`.

        Raises
        ------
        IncompatibleKwargsError
            If both `index` and `id` are `None`.
        """
        if index is not None:
            self.gals = np.delete(self.gals, index)
        elif id is not None:
            self.gals = np.delete(self.gals, np.where(self.ID == id))
        else:
            raise IncompatibleKwargsError(
                "remove_gal called with index=None and id=None; at "
                "least one of 'index' or 'id' must be provided."
            )

    # def crop(
    #     self: Self,
    #     crop_limits: Union[int, float, bool, list, NDArray],
    #     crop_property: str,
    #     aper_diam: u.Quantity,
    #     SED_fit_label: Union[str, SED_code],
    #     origin: str,
    # ) -> Self:

    #     assert origin in ["phot_rest", "SED_result"]
    #     if isinstance(SED_fit_label, tuple(SED_code.__subclasses__())):
    #         SED_fit_label = SED_fit_label.label

    #     # extract appropriate properties
    #     if origin == "phot_rest":
    #         property_arr = [
    #             gal.aper_phot[aper_diam].SED_results[SED_fit_label]
    #             .phot_rest.properties[crop_property] for gal in self
    #         ]
    #     elif origin == "SED_result":
    #         property_arr = [
    #             gal.aper_phot[aper_diam].SED_results[SED_fit_label]
    #             .properties[crop_property] for gal in self
    #         ]
    #     assert all(
    #         prop.unit == property_arr[0].unit for prop in property_arr
    #     )
    #     property_arr = (
    #         np.array([prop.value for prop in property_arr])
    #         * property_arr[0].unit
    #     )

    #     cat_copy = deepcopy(self)
    #     if isinstance(crop_limits, (int, float, bool)):
    #         cat_copy.gals = cat_copy[
    #             property_arr == crop_limits
    #         ]
    #         # if crop_limits:
    #         #     cat_copy.cat_creator.crops.append(crop_property)
    #         # else:
    #         #     cat_copy.cat_creator.crops.append(
    #         #         f"{crop_property}={crop_limits}"
    #         #     )
    #     elif isinstance(crop_limits, (list, np.array)):
    #         cat_copy.gals = cat_copy[
    #             (
    #                 (property_arr >= crop_limits[0])
    #                 & (property_arr <= crop_limits[1])
    #             )
    #         ]
    #         # cat_copy.cat_creator.crops.append(
    #         #     f"{crop_limits[0]}<{crop_property}<{crop_limits[1]}"
    #         # )
    #     else:
    #         galfind_logger.warning(
    #             f"{crop_limits=} with {type(crop_limits)=}" + \
    #             f" not in [int, float, bool, list, np.array]"
    #         )
    #     return cat_copy

    def open_cat(
        self: Self,
        cropped: bool = False,
        hdu: Optional[str, Type[SED_code]] = None,
    ) -> Optional[Table]:
        """Read the FITS catalogue table for this catalogue.

        Loads either the primary catalogue table or a named/`SED_code`
        extension of `self.cat_path`, optionally cropping the resulting
        table to only the galaxy IDs currently held in `self.gals` (or,
        if this catalogue has no galaxies loaded, to the IDs satisfying
        `self.crops` in the ``"SELECTION"`` HDU).

        Parameters
        ----------
        cropped : `bool`, optional
            Whether to crop the returned table to the galaxy IDs in this
            catalogue (or, if empty, to the applied crops). Default is
            `False`.
        hdu : `str` or `SED_code`, optional
            Name of the FITS extension to open, or an `SED_code` whose
            `hdu_name` is used. If `None`, the primary catalogue table is
            opened. Default is `None`.

        Returns
        -------
        `astropy.table.Table` or `None`
            The requested catalogue table, or `None` if the requested
            `hdu` does not exist in `self.cat_path`.

        Raises
        ------
        GalfindTypeError
            If `hdu` is not `None`, a `str`, or an `SED_code` instance.
        MissingDataError
            If `cropped` is `True`, this catalogue has no galaxies
            loaded, and not all of `self.crops` have been performed on
            the ``"SELECTION"`` HDU.
        InvalidOptionError
            If `hdu` is a `str` that does not match any recognised HDU
            naming convention.
        """
        if hdu is None:
            fits_cat = Table.read(
                self.cat_path,
                character_as_bytes=False,
                memmap=True,
            )
        else:
            if isinstance(hdu, SED_code):
                hdu_name = hdu.hdu_name.upper()
            elif isinstance(hdu, str):
                hdu_name = hdu
            else:
                raise GalfindTypeError(
                    f"hdu={hdu!r} has type {type(hdu).__name__}; must "
                    "be None, a str, or an SED_code instance."
                )
            if self.check_hdu_exists(hdu_name):
                fits_cat = Table.read(
                    self.cat_path,
                    character_as_bytes=False,
                    memmap=True,
                    hdu=hdu_name,
                )
            else:
                galfind_logger.warning(
                    f"{hdu=} does not exist in {self.cat_path=}!"
                )
                return None
        if cropped:
            if len(self) == 0:
                galfind_logger.debug(
                    f"No galaxies in {repr(self)} to crop the catalogue with! "
                    + "Cropping from .fits table!"
                )
                # load selection table
                select_cat = self.open_cat(cropped=False, hdu="SELECTION")
                if any(
                    crop.name not in select_cat.colnames for crop in self.crops
                ):
                    raise MissingDataError(
                        f"Not all crops in {self.crops} have been "
                        f"performed on {repr(self)}. Run the missing "
                        "crop(s) before cropping from the .fits table."
                    )
                crop_mask = np.array(
                    np.logical_and.reduce(
                        [select_cat[crop.name] for crop in self.crops]
                    )
                )
                IDs_temp = np.array(select_cat[self.cat_creator.ID_label])[
                    crop_mask
                ]
            else:
                IDs_temp = self.ID
            ID_tab = Table({"IDs_temp": IDs_temp}, dtype=[int])
            if hdu is None:
                keys_left = self.cat_creator.ID_label
            elif hdu_name.upper() in [
                "OBJECTS",
                "SELECTION",
            ]:  # , "GALFIND_CAT"]:
                keys_left = self.cat_creator.ID_label
            elif "PROPERTIES" in hdu_name.upper():
                keys_left = "ID"
            elif isinstance(hdu, SED_code):
                keys_left = hdu.ID_label
            elif hdu_name.split("_")[0].upper() in [
                subcls.__name__.upper()
                for subcls in funcs.all_subclasses(SED_code)
            ]:
                keys_left = [
                    subcls
                    for subcls in funcs.all_subclasses(SED_code)
                    if subcls.__name__.upper()
                    == hdu_name.split("_")[0].upper()
                ][0].ID_label
            elif any(
                hdu_name_.upper() == subcls.__name__.upper()
                for subcls in funcs.all_subclasses(SED_code)
                for hdu_name_ in hdu_name.split("_")
            ):
                keys_left = [
                    subcls
                    for subcls in funcs.all_subclasses(SED_code)
                    if any(
                        hdu_name_.upper() == subcls.__name__.upper()
                        for hdu_name_ in hdu_name.split("_")
                    )
                ][0].ID_label
            elif hdu_name in [
                subcls.__name__.upper()
                for subcls in funcs.all_subclasses(SED_code)
            ]:
                keys_left = [
                    subcls
                    for subcls in funcs.all_subclasses(SED_code)
                    if subcls.__name__.upper() == hdu_name.upper()
                ][0].ID_label
            else:
                raise InvalidOptionError(
                    f"hdu_name={hdu_name!r} (from hdu={hdu!r}) does not "
                    "match any recognised HDU naming convention "
                    "('OBJECTS', 'SELECTION', a '*PROPERTIES*' HDU, an "
                    "SED_code instance, or an SED_code subclass name)."
                )
            # convert ID column to appropriate units if not already
            if fits_cat[keys_left].dtype != int:
                fits_cat[keys_left] = fits_cat[keys_left].astype(int)
            combined_tab = join(
                fits_cat, ID_tab, keys_left=keys_left, keys_right="IDs_temp"
            )
            combined_tab.remove_column("IDs_temp")
            combined_tab.meta = fits_cat.meta
            return combined_tab
        else:
            return fits_cat

    def get_hdu_names(self):
        """List the names of all non-primary FITS extensions.

        Returns
        -------
        `list` of `str`
            Names of every HDU in `self.cat_path` except ``"PRIMARY"``.
        """
        hdul = fits.open(self.cat_path)
        return [hdu_.name for hdu_ in hdul if hdu_.name != "PRIMARY"]

    def check_hdu_exists(self: Self, hdu_name: str) -> bool:
        """Check whether a named HDU extension exists in the FITS catalogue.

        Parameters
        ----------
        hdu_name : `str`
            Name of the FITS extension to check for.

        Returns
        -------
        `bool`
            `True` if `hdu_name` exists in `self.cat_path`, else `False`.
        """
        # check whether the hdu extension exists
        hdul = fits.open(self.cat_path)
        return any(hdu_.name == hdu_name for hdu_ in hdul)

    def write_cat(
        self: Self,
        tab_arr: List[Table],
        tab_names: List[str],
        cat_path: str,
    ) -> None:
        """Write a set of tables to a multi-extension FITS catalogue.

        Parameters
        ----------
        tab_arr : `list` of `astropy.table.Table`
            Tables to write, one per FITS extension.
        tab_names : `list` of `str`
            Extension names corresponding to `tab_arr`.
        cat_path : `str`
            Path to write the FITS catalogue to. Overwrites any existing
            file.
        """
        self._write_cat(tab_arr, tab_names, cat_path)
        # Update cache with written tables
        if hasattr(self, "cat_creator"):
            for tab, name in zip(tab_arr, tab_names):
                self.cat_creator._tab_cache[name] = tab

    @staticmethod
    def _write_cat(
        tab_arr: List[Table],
        tab_names: List[str],
        cat_path: str,
    ) -> None:
        try:
            hdu_list = fits.HDUList()
            [
                hdu_list.append(
                    fits.BinTableHDU(
                        data=tab.as_array(),
                        header=fits.Header(tab.meta),
                        name=name,
                    )
                )
                for (tab, name) in zip(tab_arr, tab_names)
            ]
        except:
            galfind_logger.critical("Failed to write .fits table!")
            raise
        funcs.make_dirs(cat_path)
        hdu_list.writeto(cat_path, overwrite=True)
        funcs.change_file_permissions(cat_path)
        galfind_logger.info(f"Writing table to {cat_path}")

    @staticmethod
    def update_fits_cat(
        new_tab: Table,
        cat_path: str,
        hdu_name: str,
    ) -> None:
        """Update a single HDU of a FITS catalogue with new/updated columns.

        Merges the columns of `new_tab` into the existing `hdu_name`
        extension of `cat_path` (overwriting columns of the same name,
        appending new ones, and updating table metadata), while
        retaining all other extensions unchanged.

        Parameters
        ----------
        new_tab : `astropy.table.Table`
            Table containing the new/updated columns and metadata.
        cat_path : `str`
            Path to the FITS catalogue to update.
        hdu_name : `str`
            Name of the extension to update.
        """
        # retain any pre-existing column and meta names
        orig_tab = Table.read(cat_path, hdu=hdu_name)
        # update original table with new master table columns
        for col in new_tab.colnames:
            out_colname = funcs.truncate_colname(col)
            if out_colname in orig_tab.colnames:
                orig_tab[out_colname] = new_tab[col]
            else:
                orig_tab.add_column(new_tab[col], name=out_colname)
        orig_meta = orig_tab.meta
        orig_meta.update(new_tab.meta)
        # retain all fits extensions
        hdul = fits.open(cat_path, memmap=False)
        non_objects_hdul = [
            hdul[i]
            for i in range(len(hdul))
            if hdul[i].name not in ["PRIMARY", hdu_name]
        ]
        # write the fits table atomically to prevent truncation if interrupted
        out_hdul = fits.HDUList(
            [fits.PrimaryHDU(), fits.BinTableHDU(orig_tab, name=hdu_name)]
            + non_objects_hdul
        )
        tmp_path = f"{cat_path}.tmp.fits"
        try:
            out_hdul.writeto(tmp_path, overwrite=True)
            os.replace(tmp_path, cat_path)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            error_msg = (
                f"Failed to write FITS catalogue to {cat_path}. "
                f"Error: {e}. "
                f"If a temporary file exists at {tmp_path}, it can "
                f"be safely deleted. "
                f"You may need to re-run the catalogue loading process."
            )
            galfind_logger.error(error_msg)
            raise
        funcs.change_file_permissions(cat_path)
        galfind_logger.info(
            f"Updated {hdu_name} in {cat_path} with new table!"
        )

    def write_hdu(self: Self, tab: Table, hdu: str) -> None:
        """Write (or overwrite) a single HDU extension in the FITS catalogue.

        Parameters
        ----------
        tab : `astropy.table.Table`
            Table to write to the `hdu` extension.
        hdu : `str`
            Name of the extension to write `tab` to. If it already
            exists it is overwritten in place; otherwise it is appended
            as a new extension.
        """
        # if hdu exists, overwrite it
        if self.check_hdu_exists(hdu):
            tab_arr = [
                self.open_cat(cropped=False, hdu=hdu_.name)
                if hdu_.name != hdu.upper()
                else tab
                for hdu_ in fits.open(self.cat_path)
                if hdu_.name != "PRIMARY"
            ]
            tab_names = [
                hdu_.name
                for hdu_ in fits.open(self.cat_path)
                if hdu_.name != "PRIMARY"
            ]
        else:  # make new hdu
            tab_arr = [
                self.open_cat(cropped=False, hdu=hdu_.name)
                for hdu_ in fits.open(self.cat_path)
                if hdu_.name != "PRIMARY"
            ] + [tab]
            tab_names = [
                hdu_.name
                for hdu_ in fits.open(self.cat_path)
                if hdu_.name != "PRIMARY"
            ] + [hdu]
        self.write_cat(tab_arr, tab_names, self.cat_path)
        # Cache is updated by write_cat()

    def del_hdu(self, hdu: str):
        """Delete a named HDU extension from the FITS catalogue, if present.

        Parameters
        ----------
        hdu : `str`
            Name of the extension to delete.
        """
        if self.check_hdu_exists(hdu.upper()):  # delete hdu if it exists
            tab_arr = [
                self.open_cat(cropped=False, hdu=hdu_.name)
                for hdu_ in fits.open(self.cat_path)
                if hdu_.name != hdu.upper() and hdu_.name != "PRIMARY"
            ]
            tab_names = [
                hdu_.name
                for hdu_ in fits.open(self.cat_path)
                if hdu_.name != hdu.upper() and hdu_.name != "PRIMARY"
            ]
            self.write_cat(tab_arr, tab_names, self.cat_path)
            # Remove deleted HDU from cache
            if (
                hasattr(self, "cat_creator")
                and hdu.upper() in self.cat_creator._tab_cache
            ):
                del self.cat_creator._tab_cache[hdu.upper()]
            galfind_logger.info(
                f"Deleted {hdu.upper()=} from {self.cat_path=}!"
            )
        else:
            galfind_logger.info(
                f"{hdu.upper()=} does not exist in "
                f"{self.cat_path=}, could not delete!"
            )

    def del_cols_hdrs_from_fits(
        self: Self,
        col_names: List[str] = [],
        hdr_names: List[str] = [],
        hdu: Optional[str] = None,
    ) -> NoReturn:
        """Delete columns and/or header keywords from a FITS extension.

        Rewrites every extension of `self.cat_path`, removing the
        `col_names` columns and `hdr_names` metadata keywords from the
        `hdu` extension only; all other extensions are left unchanged.

        Parameters
        ----------
        col_names : `list` of `str`, optional
            Column names to remove from the `hdu` extension. Default is
            ``[]``.
        hdr_names : `list` of `str`, optional
            Header/metadata keys to remove from the `hdu` extension.
            Default is ``[]``.
        hdu : `str`, optional
            Name of the extension to remove `col_names`/`hdr_names` from.
            Default is `None`.

        Raises
        ------
        MissingKeyError
            If any name in `col_names` is not a column of the `hdu`
            table, or any name in `hdr_names` is not a metadata key of
            the `hdu` table.
        """
        # open up all fits extensions
        tab_arr = []
        tab_names = []
        for i, hdu_ in enumerate(fits.open(self.cat_path)):
            if hdu_.name != "PRIMARY":
                # open fits extension
                tab = self.open_cat(cropped=False, hdu=hdu_.name)
                # append required fits extension
                if hdu_.name == hdu.upper():
                    # ensure every column/header name is in catalogue
                    if not all(name in tab.colnames for name in col_names):
                        raise MissingKeyError(
                            f"col_names={col_names} contains names not "
                            f"present in {hdu.upper()} HDU columns "
                            f"{tab.colnames}."
                        )
                    if not all(name in tab.meta.keys() for name in hdr_names):
                        raise MissingKeyError(
                            f"hdr_names={hdr_names} contains keys not "
                            f"present in {hdu.upper()} HDU metadata "
                            f"{list(tab.meta.keys())}."
                        )
                    [tab.remove_column(name) for name in col_names]
                    tab.meta = {
                        key: value
                        for key, value in dict(tab.meta).items()
                        if key not in hdr_names
                    }
                tab_arr.append(tab)
                tab_names.append(hdu_.name)
        self.write_cat(tab_arr, tab_names, self.cat_path)

    def hist(
        self: Self,
        x_calculator: Type[Property_Calculator_Base],
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        log: bool = False,
        # n_bins: int = 50,
        save: bool = True,
        show: bool = False,
        overwrite: bool = True,
        from_pdf: bool = True,
        **plot_kwargs: Dict[str, Any],
    ) -> NoReturn:
        """Plot a histogram of a galaxy property across this catalogue.

        Extracts values of `x_calculator` for every galaxy (either by
        sampling from each galaxy's PDF or from single point estimates),
        and plots a histogram of the resulting distribution. Skips
        re-plotting if a plot already exists at the computed save path
        and `overwrite` is `False`.

        Parameters
        ----------
        x_calculator : `Property_Calculator_Base`
            Property to histogram.
        fig : `matplotlib.figure.Figure`, optional
            Figure to plot on. Created along with `ax` if either is
            `None`. Default is `None`.
        ax : `matplotlib.axes.Axes`, optional
            Axis to plot on. Created along with `fig` if either is
            `None`. Default is `None`.
        log : `bool`, optional
            Whether to plot the base-10 logarithm of the extracted
            values. Default is `False`.
        save : `bool`, optional
            Whether to save the resulting plot to disk. Default is
            `True`.
        show : `bool`, optional
            Whether to call `plt.show()`. Default is `False`.
        overwrite : `bool`, optional
            Whether to remake and overwrite the plot if it already
            exists at the computed save path. Default is `True`.
        from_pdf : `bool`, optional
            Whether to sample values from each galaxy's PDF (`True`) or
            use single point estimates (`False`). Default is `True`.
        **plot_kwargs : `dict`
            Additional keyword arguments passed to `ax.hist`.

        Raises
        ------
        GalfindTypeError
            If `x_calculator` is not a `Property_Calculator_Base`
            subclass instance.
        """
        save_path = (
            f"{config['DEFAULT']['GALFIND_WORK']}/Plots/{self.version}/"
            + f"{self.filterset.instrument_name}/{self.survey}/hist/"
            + f"{self.crop_name}/{x_calculator.name}.png"
        )
        if not Path(save_path).is_file() or overwrite:
            galfind_logger.info(
                f"Making histogram for {x_calculator.name}"
                f"{' from PDFs' if from_pdf else ''}!"
            )
            if any(fig_ax is None for fig_ax in [fig, ax]):
                fig, ax = plt.subplots()
            from . import Property_Calculator_Base

            if isinstance(
                x_calculator, tuple(Property_Calculator_Base.__subclasses__())
            ):
                if from_pdf:
                    # sample x values from PDFs
                    # unit = x_calculator.extract_vals(self).unit
                    x = np.array(
                        [
                            x_.input_arr
                            for x_ in x_calculator.extract_PDFs(self)
                        ]
                    ).flatten()  # * unit
                    # remove nans
                    x = np.array(
                        [x_ for x_ in x if not np.isnan(x_)]
                    )  # .value
                    x_label = f"{x_calculator.name} PDF samples"
                    n_bins = 50
                else:
                    x = x_calculator.extract_vals(self).value
                    x = np.array(
                        [x_ for x_ in x if not np.isnan(x_)]
                    )  # .value * x.unit
                    x_label = x_calculator.name
                    n_bins = int(np.sqrt(len(x)))
            else:
                raise GalfindTypeError(
                    f"x_calculator={x_calculator!r} has type "
                    f"{type(x_calculator).__name__}; must be a "
                    "Property_Calculator_Base subclass instance."
                )
            if log:
                x = np.log10(x)
                # x_name = f"log({x_name})"
                x_label = f"log({x_label})"
            ax.hist(x, bins=n_bins, label=x_label, **plot_kwargs)
            # ax.set_ylabel("Number of Galaxies")
            # ax.set_title(f"{x_label} histogram")
            if save:
                ax.legend()
                ax.set_xlabel(x_calculator.plot_name)
                funcs.make_dirs(save_path)
                plt.savefig(save_path)
                galfind_logger.info(f"Saved histogram to {save_path}!")
                funcs.change_file_permissions(save_path)
            if show:
                plt.show()
        else:
            galfind_logger.info(
                f"{x_calculator.name} histogram already exists and "
                f"{overwrite=}, skipping!"
            )

    def plot(
        self: Self,
        x_calculator: Type[Property_Calculator_Base],
        y_calculator: Type[Property_Calculator_Base],
        c_calculator: Optional[Type[Property_Calculator_Base]] = None,
        incl_x_errs: bool = True,
        incl_y_errs: bool = True,
        log_x: bool = False,
        log_y: bool = False,
        log_c: bool = False,
        mean_err: bool = False,
        annotate: bool = True,
        save: bool = True,
        show: bool = False,
        legend_kwargs: dict = {},
        plot_legend: bool = False,
        plot_kwargs: dict = {},
        cmap: str = "viridis",
        save_path: Optional[str] = None,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        plot_type: str = "individual",
        n_samples: int = 100_000,
        n_bins: int = 25,
        contour_levels: List[float] = [68.0, 95.0, 100.0],
        x_hist_ax: Optional[plt.Axes] = None,
        y_hist_ax: Optional[plt.Axes] = None,
        hist_kwargs: Dict[str, Any] = {},
        plot_label: Optional[str] = "default",
        plot_cbar: bool = True,
        inf_val: float = 1.0e6,
        xlims: Optional[List[float]] = None,
        ylims: Optional[List[float]] = None,
    ) -> Optional[plt.Axes]:
        """Scatter,
        contour, or stacked plot of one galaxy property against another.

        Extracts `x_calculator` and `y_calculator` values (and
        optionally `c_calculator` for colouring) for every galaxy in
        this catalogue and plots them against each other, with three
        possible plotting modes controlled by `plot_type`:

        - ``"individual"``: scatter plot of per-galaxy point estimates,
          optionally with error bars and/or colouring by `c_calculator`.
        - ``"contour"``: contours of a 2D histogram built from samples
          drawn from each galaxy's x/y PDFs.
        - ``"stacked"``: a single point (with errors) from the sum of
          every galaxy's x/y PDFs.

        Parameters
        ----------
        x_calculator : `Property_Calculator_Base`
            Property to plot on the x-axis.
        y_calculator : `Property_Calculator_Base`
            Property to plot on the y-axis.
        c_calculator : `Property_Calculator_Base`, optional
            Property to colour points by (only used for
            ``plot_type="individual"``). Default is `None`.
        incl_x_errs : `bool`, optional
            Whether to include x errorbars. Default is `True`.
        incl_y_errs : `bool`, optional
            Whether to include y errorbars. Default is `True`.
        log_x : `bool`, optional
            Whether to plot log10(x). Also triggered automatically if
            `x_calculator.full_name` is in `funcs.logged_properties`.
            Default is `False`.
        log_y : `bool`, optional
            Whether to plot log10(y). Also triggered automatically if
            `y_calculator.full_name` is in `funcs.logged_properties`.
            Default is `False`.
        log_c : `bool`, optional
            Whether to colour by log10(c). Also triggered automatically
            if `c_calculator.full_name` is in `funcs.logged_properties`.
            Default is `False`.
        mean_err : `bool`, optional
            Whether to plot the mean error rather than per-point errors.
            Currently unimplemented and has no effect. Default is
            `False`.
        annotate : `bool`, optional
            Whether to set the axis title and labels. Default is `True`.
        save : `bool`, optional
            Whether to save the plot to disk. Default is `True`.
        show : `bool`, optional
            Whether to call `plt.show()`. Default is `False`.
        legend_kwargs : `dict`, optional
            Extra keyword arguments passed to `ax.legend`. Default is
            ``{}``.
        plot_legend : `bool`, optional
            Whether to draw a legend. Default is `False`.
        plot_kwargs : `dict`, optional
            Extra keyword arguments passed to the scatter/errorbar
            plotting calls. Default is ``{}``.
        cmap : `str`, optional
            Name of the colormap used for `c_calculator` colouring and
            for the ``"contour"`` plot type. Default is ``"viridis"``.
        save_path : `str`, optional
            Path to save the plot to. If `None` and `save` is `True`, a
            path is generated from the catalogue and property names.
            Default is `None`.
        fig : `matplotlib.figure.Figure`, optional
            Figure to plot on. Created along with `ax` if either is
            `None`. Default is `None`.
        ax : `matplotlib.axes.Axes`, optional
            Axis to plot on. Created along with `fig` if either is
            `None`. Default is `None`.
        plot_type : `str`, optional
            One of ``"individual"``, ``"contour"`` or ``"stacked"``.
            Default is ``"individual"``.
        n_samples : `int`, optional
            Number of samples drawn per galaxy PDF for
            ``plot_type="contour"``. Default is `100_000`.
        n_bins : `int`, optional
            Number of bins per axis of the 2D histogram for
            ``plot_type="contour"``. Default is `25`.
        contour_levels : `list` of `float`, optional
            Percentile levels of the 2D histogram to draw contours at.
            Default is `[68., 95., 100.]`.
        x_hist_ax : `matplotlib.axes.Axes`, optional
            If given, an additional axis to plot a marginal histogram of
            x values on. Default is `None`.
        y_hist_ax : `matplotlib.axes.Axes`, optional
            If given, an additional axis to plot a marginal histogram of
            y values on. Default is `None`.
        hist_kwargs : `dict`, optional
            Extra keyword arguments passed to `x_hist_ax.hist`/
            `y_hist_ax.hist`. Default is ``{}``.
        plot_label : `str`, optional
            Title to annotate the plot with. If ``"default"``, built
            from the catalogue's version, instrument and survey names.
            Default is ``"default"``.
        plot_cbar : `bool`, optional
            Whether to draw a colourbar when `c_calculator` is given.
            Default is `True`.
        inf_val : `float`, optional
            Value substituted for infinities when converting y errors to
            log space. Default is `1.0e6`.
        xlims : `list` of `float`, optional
            x-axis limits. Default is `None`.
        ylims : `list` of `float`, optional
            y-axis limits. Default is `None`.

        Returns
        -------
        `matplotlib.axes.Axes` or `None`
            The plotted scatter/errorbar artist for ``"individual"`` and
            ``"stacked"`` plot types, or `None` for ``"contour"``.

        Raises
        ------
        GalfindTypeError
            If `x_calculator`, `y_calculator` or `c_calculator` is not a
            `Property_Calculator_Base` subclass instance.
        IncompatibleKwargsError
            If `c_calculator` is given together with
            ``plot_type="contour"`` or ``plot_type="stacked"``.
        InvalidOptionError
            If `plot_type` is not one of ``"individual"``, ``"contour"``
            or ``"stacked"``.
        """
        from . import Property_Calculator_Base

        x_name = x_calculator.full_name
        y_name = y_calculator.full_name
        x_label = x_calculator.plot_name
        y_label = y_calculator.plot_name
        colour_name = (
            c_calculator.full_name if c_calculator is not None else ""
        )
        colour_label = (
            c_calculator.plot_name if c_calculator is not None else None
        )

        if plot_type.lower() == "individual":
            if isinstance(
                x_calculator, tuple(Property_Calculator_Base.__subclasses__())
            ):
                x = x_calculator.extract_vals(self)
                if incl_x_errs:
                    x_err = x_calculator.extract_errs(self)
                else:
                    x_err = None
            else:
                raise GalfindTypeError(
                    f"x_calculator={x_calculator!r} has type "
                    f"{type(x_calculator).__name__}; must be a "
                    "Property_Calculator_Base subclass instance."
                )
            if log_x or x_name in funcs.logged_properties:
                if incl_x_errs:
                    unit = x.unit
                    x_err = np.array(x_err)
                    if x_err.shape[1] == 2:
                        x_err = x_err.T
                    # try:
                    x, x_err, _ = funcs.errs_to_log(x.flatten().value, x_err)
                    # except:
                    #     breakpoint()
                    x *= unit
                    x_err *= unit
                else:
                    x = np.log10(x.value) * u.dimensionless_unscaled
                x_name = f"log({x_name})"
                x_label = f"log({x_label})"

            if isinstance(
                y_calculator, tuple(Property_Calculator_Base.__subclasses__())
            ):
                y = y_calculator.extract_vals(self)
                if incl_y_errs:
                    y_err = y_calculator.extract_errs(self)
                else:
                    y_err = None
                y_ = deepcopy(y.flatten().value)
                y_err_ = deepcopy(y_err)
                y_uplims_1 = np.full(y_.shape, False)
            else:
                raise GalfindTypeError(
                    f"y_calculator={y_calculator!r} has type "
                    f"{type(y_calculator).__name__}; must be a "
                    "Property_Calculator_Base subclass instance."
                )
            if log_y or y_name in funcs.logged_properties:
                if incl_y_errs:
                    unit = y.unit
                    y_err = np.array(y_err)
                    try:
                        y_, y_err_, y_uplims_1 = funcs.errs_to_log(
                            y.flatten().value,
                            y_err.T,
                            uplim_sigma=None,
                            inf_val=inf_val,
                        )
                    except Exception:
                        y_, y_err_, y_uplims_1 = funcs.errs_to_log(
                            y.flatten().value,
                            y_err,
                            uplim_sigma=None,
                            inf_val=inf_val,
                        )
                    y_ *= unit
                    y_err_ *= unit
                else:
                    y_ = np.log10(y.value) * u.dimensionless_unscaled
                y_name = f"log({y_name})"
                y_label = f"log({y_label})"

            if c_calculator is not None:
                if isinstance(
                    c_calculator,
                    tuple(Property_Calculator_Base.__subclasses__()),
                ):
                    c = c_calculator.extract_vals(self).value.flatten()
                else:
                    raise GalfindTypeError(
                        f"c_calculator={c_calculator!r} has type "
                        f"{type(c_calculator).__name__}; must be a "
                        "Property_Calculator_Base subclass instance."
                    )
                if log_c or colour_name in funcs.logged_properties:
                    c = np.log10(c)
                    colour_name = f"log({colour_name})"
                    colour_label = f"log({colour_label})"

        elif plot_type.lower() == "contour":
            if c_calculator is not None:
                raise IncompatibleKwargsError(
                    "Cannot use plot_type='contour' together with a "
                    f"c_calculator={c_calculator!r}: contour plots "
                    "cannot colour galaxies by another property."
                )
            # extract the PDFs
            x_PDFs = x_calculator.extract_PDFs(self)
            y_PDFs = y_calculator.extract_PDFs(self)
            x = []
            y = []
            for x_PDF_, y_PDF_ in zip(x_PDFs, y_PDFs):
                if x_PDF_ is not None and y_PDF_ is not None:
                    # extract n_samples from each PDF
                    x.extend(x_PDF_.draw_sample(n_samples).value)
                    y.extend(y_PDF_.draw_sample(n_samples).value)

            if log_x or x_name in funcs.logged_properties:
                x = np.log10(x)
                x_name = f"log({x_name})"
                x_label = f"log({x_label})"

            if log_y or y_name in funcs.logged_properties:
                y = np.log10(y)
                y_name = f"log({y_name})"
                y_label = f"log({y_label})"

        elif plot_type.lower() == "stacked":
            if c_calculator is not None:
                raise IncompatibleKwargsError(
                    "Cannot use plot_type='stacked' together with a "
                    f"c_calculator={c_calculator!r}: stacked plots "
                    "cannot colour galaxies by another property."
                )
            # extract the PDFs
            x_PDFs = x_calculator.extract_PDFs(self)
            y_PDFs = y_calculator.extract_PDFs(self)
            # add all the PDFs together
            x_PDF = None
            y_PDF = None
            for x_PDF_, y_PDF_ in zip(x_PDFs, y_PDFs):
                if x_PDF_ is not None and y_PDF_ is not None:
                    if x_PDF is None and y_PDF is None:
                        x_PDF = x_PDF_
                        y_PDF = y_PDF_
                    else:
                        x_PDF += x_PDF_
                        y_PDF += y_PDF_

            x = x_PDF.median.value
            y = y_PDF.median.value
            if incl_x_errs:
                x_err = [[x_PDF.errs[0].value], [x_PDF.errs[1].value]]
            else:
                x_err = None
            if incl_y_errs:
                y_err = [[y_PDF.errs[0].value], [y_PDF.errs[1].value]]
            else:
                y_err = None

        else:
            raise InvalidOptionError(
                f"plot_type={plot_type!r} not recognised; must be one "
                "of 'individual', 'contour' or 'stacked'."
            )

        # setup matplotlib figure/axis if not already given
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        if (
            "label" not in plot_kwargs.keys()
            and plot_type.lower() != "contour"
        ):
            plot_kwargs["label"] = self.crop_name

        if c_calculator is not None:
            plot_kwargs["c"] = c
            if "cmap" not in plot_kwargs.keys():
                plot_kwargs["cmap"] = cmap

        if plot_type.lower() == "contour":
            nan_mask = np.logical_and(~np.isnan(x), ~np.isnan(y))
            x = np.array(x)[nan_mask]
            y = np.array(y)[nan_mask]
            x_interval = (np.max(x) - np.min(x)) / (n_bins + 1)
            y_interval = (np.max(y) - np.min(y)) / (n_bins + 1)
            x_edges = np.linspace(
                np.min(x) - x_interval, np.max(x) + x_interval, n_bins
            )
            y_edges = np.linspace(
                np.min(y) - y_interval, np.max(y) + y_interval, n_bins
            )
            N, x_edges, y_edges = np.histogram2d(x, y, bins=(x_edges, y_edges))
            x_mid_bins = (x_edges[:-1] + x_edges[1:]) / 2
            y_mid_bins = (y_edges[:-1] + y_edges[1:]) / 2
            x_mesh, y_mesh = np.meshgrid(x_mid_bins, y_mid_bins)
            levels = np.percentile(N, contour_levels)
            # n_bins = int(len(x) / (25 * n_samples))
            for i in range(len(levels) - 1):
                ax.contourf(
                    x_mesh,
                    y_mesh,
                    N.T,
                    levels=[levels[i], levels[i + 1]],
                    alpha=(i + 1) / len(levels),
                    colors=[cmap],
                    **plot_kwargs,
                )
                # , norm = norm, cmap = cmap, norm = norm,
                # cmap = cm.get_cmap(cmap).resampled(
                #     len(plot_kwargs["levels"]) - 1
                # )
            for level in levels:
                ax.contour(
                    x_mesh,
                    y_mesh,
                    N.T,
                    levels=(level,),
                    colors=cmap,
                    linewidths=1.0,
                    **plot_kwargs,
                )
            ax.set_xlim([x_edges[0], x_edges[-1]])
            ax.set_ylim([y_edges[0], y_edges[-1]])
        else:
            if incl_x_errs or incl_y_errs and not mean_err:
                # if "ls" not in plot_kwargs.keys():
                #     plot_kwargs["ls"] = ""
                # old_to_new_names = {
                #     "s": "ms", "edgecolor": "mec", "facecolor": "mfc"
                # }
                # for old_name, new_name in old_to_new_names.items():
                #     if old_name in plot_kwargs.keys():
                #         plot_kwargs[new_name] = plot_kwargs.pop(old_name)
                if "mec" in plot_kwargs.keys():
                    colour = plot_kwargs["mec"]
                elif "edgecolor" in plot_kwargs.keys():
                    colour = plot_kwargs["edgecolor"]
                elif "c" in plot_kwargs.keys():
                    colour = plot_kwargs["c"]
                if incl_x_errs and isinstance(
                    x_err, (u.Quantity, u.Magnitude, u.Dex)
                ):
                    x_err = x_err.T.value
                if incl_y_errs and isinstance(
                    y_err, (u.Quantity, u.Magnitude, u.Dex)
                ):
                    y_err_ = y_err_.value  # .T
                if isinstance(x, (u.Quantity, u.Magnitude, u.Dex)):
                    x = x.value
                if isinstance(y_, (u.Quantity, u.Magnitude, u.Dex)):
                    y_ = y_.value
                x = np.array(x)
                y_ = np.array(y_)
                x_err = np.array(x_err)
                y_err_ = np.array(y_err_)
                if isinstance(colour, np.ndarray):
                    errorbar_colour = "gray"
                else:
                    errorbar_colour = colour
                try:
                    plot = ax.errorbar(
                        x,
                        y_,
                        xerr=x_err,
                        yerr=y_err_,
                        uplims=y_uplims_1,
                        marker="none",
                        color=errorbar_colour,
                        ls="",
                        capsize=0.0,
                        alpha=0.5,
                    )  # , **errorbar_kwargs) # **plot_kwargs)
                except Exception:
                    try:
                        plot = ax.errorbar(
                            x,
                            y_,
                            xerr=x_err,
                            yerr=y_err_.T,
                            uplims=y_uplims_1,
                            marker="none",
                            color=errorbar_colour,
                            ls="",
                            capsize=0.0,
                            alpha=0.5,
                        )  # , **errorbar_kwargs) # **plot_kwargs)
                    except Exception:
                        try:
                            plot = ax.errorbar(
                                x,
                                y_,
                                xerr=x_err.T,
                                yerr=y_err_,
                                uplims=y_uplims_1,
                                marker="none",
                                color=errorbar_colour,
                                ls="",
                                capsize=0.0,
                                alpha=0.5,
                            )  # , **errorbar_kwargs) # **plot_kwargs)
                        except Exception:
                            plot = ax.errorbar(
                                x,
                                y_,
                                xerr=x_err.T,
                                yerr=y_err_.T,
                                uplims=y_uplims_1,
                                marker="none",
                                color=errorbar_colour,
                                ls="",
                                capsize=0.0,
                                alpha=0.5,
                            )
            if "zorder" not in plot_kwargs.keys():
                plot_kwargs["zorder"] = plot.lines[0].zorder + 1
            plot = ax.scatter(x, y_, **plot_kwargs)

        if mean_err and (incl_x_errs or incl_y_errs):
            # plot the mean error
            pass

        # if histograms are wanted to be displayed
        if x_hist_ax is not None or y_hist_ax is not None:
            # if "bins" not in hist_kwargs.keys():
            #     hist_kwargs["bins"] = 30
            if isinstance(x, u.Magnitude):
                x = x.value
            if isinstance(y, u.Magnitude):
                y = y.value
            # divider = make_axes_locatable(ax)
        if x_hist_ax is not None:
            if not isinstance(x_hist_ax, plt.Axes):
                raise GalfindTypeError(
                    f"x_hist_ax={x_hist_ax!r} has type "
                    f"{type(x_hist_ax).__name__}; must be a "
                    "matplotlib.axes.Axes."
                )
            # ax_hist_top = divider.append_axes(
            #     "top", size="20%", pad=0.1, sharex=ax
            # )
            x_hist_ax.hist(x, **hist_kwargs)
        if y_hist_ax is not None:
            if not isinstance(y_hist_ax, plt.Axes):
                raise GalfindTypeError(
                    f"y_hist_ax={y_hist_ax!r} has type "
                    f"{type(y_hist_ax).__name__}; must be a "
                    "matplotlib.axes.Axes."
                )
            # Right histogram
            # ax_hist_right = divider.append_axes(
            #     "right", size="20%", pad=0.1, sharey=ax
            # )
            y_hist_ax.hist(y, orientation="horizontal", **hist_kwargs)

        # sort plot aesthetics - automatically annotate if coloured
        if annotate or c_calculator is not None:
            if plot_label == "default":
                plot_label = (
                    f"{self.version}, "
                    f"{self.filterset.instrument_name}, {self.survey}"
                )
            if plot_label is not None:
                ax.set_title(plot_label)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            if c_calculator is not None and plot_cbar:
                fig.colorbar(plot, label=colour_label)
            if plot_legend:
                default_legend_kwargs = {"loc": "best"}
                for key, value in legend_kwargs.items():
                    default_legend_kwargs[key] = value
                ax.legend(**default_legend_kwargs)

        # TODO: Determine these dynamically
        if xlims is not None:
            ax.set_xlim(xlims)
        if ylims is not None:
            ax.set_ylim(ylims)

        if save:
            # determine appropriate save path
            if save_path is None:
                save_colour_name = (
                    f"_c={colour_name}" if c_calculator is not None else ""
                )
                save_path = (
                    f"{config['Other']['PLOT_DIR']}/{self.version}/"
                    + f"{self.filterset.instrument_name}/{self.survey}/"
                    f"{y_name}_vs_{x_name}/{self.crop_name}{save_colour_name}.png"
                )
            funcs.make_dirs(save_path)
            plt.savefig(save_path)
            funcs.change_file_permissions(save_path)
            galfind_logger.info(f"Saved plot to {save_path}!")

        if show:
            plt.show()

        if plot_type.lower() != "contour":
            return plot

    def get_Vmax_ecsv_path(
        self: Self,
        data: Data,
        Vmax_method: str = "uniform_depth",
    ) -> str:
        """Construct the save path of this catalogue's Vmax .ecsv file.

        Parameters
        ----------
        data : `Data`
            Data object the Vmax values are computed with respect to.
        Vmax_method : `str`, optional
            Name of the Vmax calculation method. Default is
            ``"uniform_depth"``.

        Returns
        -------
        `str`
            Path to the Vmax .ecsv file for this catalogue, `data` and
            `Vmax_method`. The containing directory is created if it
            does not already exist.
        """
        full_data_name = funcs.get_full_survey_name(
            data.survey, data.version, data.filterset
        )
        save_path = (
            f"{config['NumberDensityFunctions']['VMAX_DIR']}/"
            + f"{self.version}/{self.filterset.instrument_name}/"
            + f"{self.survey}/{self.crop_name}/{Vmax_method}/"
            f"Vmax_field={full_data_name}.ecsv"
        )
        funcs.make_dirs(save_path)
        return save_path

    def _calc_Vmax(
        self: Self,
        data: Data,
        z_bin: List[float],
        aper_diam: u.Quantity,
        SED_fit_code: SED_code,
        z_step: float = 0.01,
        unmasked_area: Union[
            str, List[str], u.Quantity, Type[Mask_Selector]
        ] = "selection",
        Vmax_method: str = "uniform_depth",
        n_jobs: int = 1,
    ) -> Dict[str, NDArray[float]]:
        if len(z_bin) != 2:
            raise LengthMismatchError(
                f"z_bin={z_bin!r} has length {len(z_bin)}; must have "
                "length 2 (a [z_min, z_max] pair)."
            )
        if not z_bin[0] < z_bin[1]:
            raise RangeError(
                f"z_bin={z_bin!r}: z_bin[0] must be less than z_bin[1]."
            )
        if not isinstance(SED_fit_code, tuple(SED_code.__subclasses__())):
            raise GalfindTypeError(
                f"SED_fit_code={SED_fit_code!r} has type "
                f"{type(SED_fit_code).__name__}; must be an SED_code "
                "subclass instance."
            )
        if not all(
            SED_fit_code.label in gal.aper_phot[aper_diam].SED_results.keys()
            for gal in self
        ):
            raise MissingDataError(
                f"SED fitting results for {SED_fit_code.label!r} not "
                f"loaded at aper_diam={aper_diam!r} for every galaxy "
                f"in {repr(self)}. Run the SED fit first."
            )
        if not isinstance(n_jobs, int):
            raise GalfindTypeError(
                f"n_jobs={n_jobs!r} has type {type(n_jobs).__name__}; "
                "must be an int."
            )
        if not n_jobs >= 1:
            raise RangeError(f"n_jobs={n_jobs} must be >= 1.")

        save_path = self.get_Vmax_ecsv_path(data, Vmax_method=Vmax_method)
        # if this file already exists
        if Path(save_path).is_file():
            # open file
            old_tab = Table.read(save_path)
            update_gals = np.array(
                [
                    gal
                    for gal in self
                    if gal.ID
                    not in old_tab[old_tab["survey"] == gal.survey]["ID"]
                ]
            )
        else:
            update_gals = self.gals
        # HACK: remove update gals from self
        # update_gals_ids = np.array([gal.ID for gal in update_gals])
        # self.gals = np.array(
        #     [gal for gal in self if gal.ID not in update_gals_ids]
        # )
        # update_gals = []
        if len(update_gals) > 0:
            full_survey_name = funcs.get_full_survey_name(
                self.survey, self.version, self.filterset
            )
            full_data_name = funcs.get_full_survey_name(
                data.survey, data.version, data.filterset
            )

            # load in area-depth curves before parallelization
            # if Vmax_method.lower() == "area_depth_scaled_split_region":
            z_arr = np.arange(z_bin[0], z_bin[1] + z_step, z_step)
            if hasattr(data, "region_selector"):
                region_selector = data.region_selector
                invert_region = [
                    True,
                    False,
                ]  # region != self.data.region_selector.name
            else:
                region_selector = None
                invert_region = [False]
                setattr(data, "regions", ["all"])
            for invert_region_ in invert_region:
                for z in z_arr:
                    # compute area depth for all relevant redshift bin areas
                    data.calc_area_depth(
                        aper_diam=aper_diam,
                        mask_selector=unmasked_area,
                        # mask_type = mask_type,
                        region_selector=region_selector,
                        invert_region=invert_region_,
                        z=z,
                        plot=True,
                    )

            # calculate Vmax's and append them to Vmax ecsv
            if n_jobs == 1:
                Vmax_outputs = [
                    gal.calc_Vmax(
                        data,
                        z_bin,
                        aper_diam,
                        SED_fit_code,
                        self.cat_creator.crops,
                        z_step,
                        unmasked_area=unmasked_area,
                        Vmax_method=Vmax_method,
                    )
                    for gal in tqdm(
                        update_gals,
                        total=len(update_gals),
                        desc=f"Calculating {full_survey_name} Vmax's "
                        f"in {full_data_name}",
                        disable=galfind_logger.getEffectiveLevel()
                        > logging.INFO,
                    )
                ]
            else:
                params_arr = [
                    (
                        gal,
                        data,
                        z_bin,
                        aper_diam,
                        SED_fit_code,
                        self.cat_creator.crops,
                        z_step,
                        unmasked_area,
                        Vmax_method,
                    )
                    for gal in update_gals
                ]
                from joblib import Parallel, delayed, parallel_config

                with funcs.tqdm_joblib(
                    tqdm(
                        desc=f"Calculating {full_survey_name} Vmax's "
                        f"in {full_data_name}",
                        total=len(update_gals),
                        disable=galfind_logger.getEffectiveLevel()
                        > logging.INFO,
                    )
                ):
                    with parallel_config(backend="loky", n_jobs=n_jobs):
                        Vmax_outputs = Parallel()(
                            delayed(self._calc_Vmax_multi_process)(params)
                            for params in params_arr
                        )

            # prepare ecsv data
            region_keys = [key for key in Vmax_outputs[0][0].keys()]
            if region_keys != data.regions:
                raise LengthMismatchError(
                    f"region_keys={region_keys!r} from Vmax calculation "
                    f"does not match data.regions={data.regions!r}."
                )
            if not all(
                len(Vmax_output[0]) == len(data.regions)
                for Vmax_output in Vmax_outputs
            ):
                raise LengthMismatchError(
                    f"Vmax output region counts "
                    f"{[len(Vmax_output[0]) for Vmax_output in Vmax_outputs]} "
                    f"do not all match len(data.regions)={len(data.regions)}."
                )
            zbin_keys = [
                key for key in Vmax_outputs[0][1][region_keys[0]].keys()
            ]
            # assert all([
            #     Vmax_output[1][region_keys[0]].keys() == zbin_keys
            #     for Vmax_output in Vmax_outputs
            # ]), \
            #     galfind_logger.critical(
            #         "z_bin keys from Vmax calculation do not "
            #         "match across galaxies!"
            #     )
            meta = Vmax_outputs[0][2]
            # TODO: check meta data consistency across galaxies
            # tricky due to dict element assertion requirements
            # assert all(
            #     Vmax_output[2] == meta for Vmax_output in Vmax_outputs
            # ), \
            #     galfind_logger.critical(
            #         "meta data from Vmax calculation do not "
            #         "match across galaxies!"
            #     )
            Vmax = np.zeros(
                (len(update_gals), len(region_keys), len(zbin_keys))
            )
            kwargs = {}
            for i, Vmax_output in enumerate(Vmax_outputs):
                for j, region in enumerate(region_keys):
                    for k, zbin in enumerate(zbin_keys):
                        try:
                            Vmax[i, j, k] = Vmax_output[0][region][zbin]
                            if isinstance(
                                Vmax_output[1][region][zbin],
                                (list, np.ndarray),
                            ):
                                for h, kwarg in enumerate(
                                    Vmax_output[1][region][zbin]
                                ):
                                    if isinstance(kwarg, dict):
                                        for key, value in kwarg.items():
                                            key = f"{key}_{str(h + 1)}"
                                            if key not in kwargs.keys():
                                                kwargs[key] = [value]
                                            else:
                                                kwargs[key].extend([value])
                                    else:
                                        raise GalfindTypeError(
                                            f"Vmax output kwarg={kwarg!r} "
                                            f"has type "
                                            f"{type(kwarg).__name__}; "
                                            "must be a dict."
                                        )
                            elif isinstance(
                                Vmax_output[1][region][zbin], dict
                            ):
                                for key, value in Vmax_output[1][region][
                                    zbin
                                ].items():
                                    if key not in kwargs.keys():
                                        kwargs[key] = [value]
                                    else:
                                        kwargs[key].extend([value])
                            else:
                                val = Vmax_output[1][region][zbin]
                                raise GalfindTypeError(
                                    f"Vmax output kwarg={val!r} has type "
                                    f"{type(val).__name__}; must be a "
                                    "list, ndarray or dict."
                                )
                        except Exception as e:
                            galfind_logger.critical(
                                f"Error when assigning Vmax for "
                                f"galaxy ID {update_gals[i].ID}! "
                                f"Error={e}"
                            )
                            raise e
            IDs = np.repeat(
                np.array([gal.ID for gal in update_gals]),
                len(zbin_keys) * len(region_keys),
            )
            regions = np.tile(
                np.repeat(data.regions, len(zbin_keys)), len(update_gals)
            )
            zbin_labels = np.tile(
                zbin_keys, len(region_keys) * len(update_gals)
            )
            surveys = np.repeat(
                np.array([gal.survey for gal in update_gals]),
                len(zbin_keys) * len(region_keys),
            )
            aper_diams = np.full(len(IDs), aper_diam.value)
            SED_fit_codes = np.full(len(IDs), SED_fit_code.label)
            ecsv_data = {
                "ID": IDs,
                "survey": surveys,
                "region": regions,
                "aper_diam": aper_diams,
                "SED_fit_code": SED_fit_codes,
                "zbin_label": zbin_labels,
                "Vmax_total": Vmax.flatten(),
                **kwargs,
            }
            try:
                new_tab = Table(
                    ecsv_data,
                    dtype=[int, str, str, float, str, str, float]
                    + [str] * len(kwargs),
                )
            except Exception as e:
                galfind_logger.critical(
                    f"Error when creating Vmax ecsv table! Error={e}"
                )
                raise e
            new_tab.meta = {"Vmax_invalid_val": -1.0, **meta}
            out_tab = self._save_ecsv(save_path, new_tab)
            self._load_Vmax_from_ecsv(
                out_tab, data.survey, aper_diam, SED_fit_code, data.full_name
            )
        else:
            if len(self) != 0:
                old_tab = Table.read(save_path)
                self._load_Vmax_from_ecsv(
                    old_tab,
                    data.survey,
                    aper_diam,
                    SED_fit_code,
                    data.full_name,
                )

    @staticmethod
    def _calc_Vmax_multi_process(
        params: Dict[str, Any],
    ) -> Tuple[Dict[str, float], Dict[str, Any], Dict[str, Any]]:
        (
            gal,
            data,
            z_bin,
            aper_diam,
            SED_fit_code,
            crops,
            z_step,
            unmasked_area,
            Vmax_method,
        ) = params
        return gal.calc_Vmax(
            data,
            z_bin,
            aper_diam,
            SED_fit_code,
            crops,
            z_step,
            unmasked_area=unmasked_area,
            Vmax_method=Vmax_method,
        )

    def _save_ecsv(self: Self, save_path: str, tab: Table) -> Table:
        if Path(save_path).is_file():  # update and save table
            old_tab = Table.read(save_path)
            out_tab = vstack([old_tab, tab])
            out_tab.meta = {**old_tab.meta, **tab.meta}
        else:  # save table
            out_tab = tab
        out_tab.write(save_path, overwrite=True)
        return out_tab

    def _load_Vmax_from_ecsv(
        self: Self,
        tab: Table,
        data_survey: str,
        aper_diam: u.Quantity,
        SED_fit_code: SED_code,
        full_survey_name: str,
    ) -> None:
        if any(gal.survey == full_survey_name.split("_")[0] for gal in self):
            load_gals_arr = [
                gal
                for gal in self
                if gal.survey == full_survey_name.split("_")[0]
            ]
        else:
            load_gals_arr = self.gals
        # save appropriate Vmax properties
        # Vmax_arr = np.zeros(len(load_gals_arr))
        for i, gal in enumerate(load_gals_arr):
            Vmax_rows = tab[
                np.logical_and(
                    (tab["ID"] == gal.ID), (tab["survey"] == gal.survey)
                )
            ]
            gal.set_Vmax(Vmax_rows, data_survey=data_survey)

# Catalogue_Base.py

from __future__ import annotations

from copy import deepcopy
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, join
from tqdm import tqdm
from typing import TYPE_CHECKING, List, Dict, Union
if TYPE_CHECKING:
    from . import Galaxy, Catalogue_Creator

from . import (
    Multiple_Catalogue,
    SED_code,
    galfind_logger,
    sed_code_to_name_dict,
)
from . import useful_funcs_austind as funcs


class Catalogue_Base:
    # later on, the gal_arr should be calculated from the Instrument and sex_cat path, with SED codes already given
    def __init__(
        self,
        gals: List[Galaxy],
        cat_creator: Catalogue_Creator,
        SED_rest_properties: Dict[str, Union[u.Quantity, u.Magnitude, u.Dex]] = {},
    ):
        self.gals = gals
        self.cat_creator = cat_creator

        # keep a record of the crops that have been made to the catalogue
        # TODO: Ensure this is updated appropriately after Data class rewrite
        # self.selection_cols = [
        #     key.replace("SELECTED_", "")
        #     for key in self.open_cat().meta.keys()
        #     if "SELECTED_" in key
        # ]
        # if crops is None:
        #     crops = []
        # self.crops = crops
        self.SED_rest_properties = SED_rest_properties

        # concat is commutative for catalogues
        self.__radd__ = self.__add__
        # cross-match is commutative for catalogues - not if they are of different classes
        # self.__rmul__ = self.__mul__

    # %% Overloaded operators

    def __repr__(self) -> str:
        return f"{self.__class__.__name__.upper()}({self.survey}," + \
            f"{self.version},{self.filterset.instrument_name})"

    def __str__(
        self,
    ) -> str:
        #display_selections = ["EPOCHS", "BROWN_DWARF"]
        output_str = funcs.line_sep
        output_str += f"{repr(self)}:\n"
        output_str += funcs.band_sep
        output_str += f"CAT PATH = {self.cat_path}\n"
        # access table header to display what has been run for this catalogue
        cat = self.cat_creator.open_cat(self.cat_path, "ID")
        output_str += f"TOTAL GALAXIES = {len(cat)}\n"
        output_str += f"RA RANGE = {self.ra_range}\n"
        output_str += f"DEC RANGE = {self.dec_range}\n"
        output_str += funcs.band_sep
        output_str += str(self.filterset)
        # display what other things have previously been calculated for this catalogue, including templates and zmax_lowz
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
        # display total number of galaxies that satisfy the selection criteria previously performed
        #if print_sel_criteria:
        # for sel_criteria in display_selections:
        #     if sel_criteria in cat.colnames:
        #         output_str += f"N_GALS_{sel_criteria} = {len(cat[cat[sel_criteria]])}\n"
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
        #if print_cls_name:
        #output_str += funcs.line_sep
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

    def __getitem__(self, index):
        # if type(self.gals) != np.ndarray:
        #     self.gals = np.array(self.gals)
        return self.gals[index]

    # only acts on attributes that don't already exist in Catalogue
    def __getattr__(
        self, property_name: str, origin: Union[str, dict] = "gal"
    ) -> np.array:
        # get attributes from stored galaxy objects
        if property_name in self.__dict__.keys():
            return self.__getattribute__(property_name)
        elif property_name in self.cat_creator.__dict__.keys():
            return self.cat_creator.__getattribute__(property_name)
        else:
            attr_arr = [gal.__getattr__(property_name, origin) for gal in self]
            # sort the units
            if all(
                type(attr) in [u.Quantity, u.Magnitude, u.Dex]
                for attr in attr_arr
            ):
                assert all(
                    attr.unit == attr_arr[0].unit for attr in attr_arr
                )  # ensure all units are the same
                attr_arr = np.array(
                    [attr.value for attr in attr_arr]
                ) * u.Unit(attr_arr[0].unit)
            else:
                attr_arr = np.array(attr_arr)
            return attr_arr

    def __setattr__(self, name, value, obj="cat"):
        if obj == "cat":
            super().__setattr__(name, value)
        elif obj == "gal":
            # set attributes of individual galaxies within the catalogue
            for i, gal in enumerate(self):
                if type(value) in [list, np.array]:
                    setattr(gal, name, value[i])
                else:
                    setattr(gal, name, value)

    # not needed!
    def __setitem__(self, index, gal):
        self.gals[index] = gal

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            try:
                setattr(result, key, deepcopy(value, memo))
            except:
                galfind_logger.critical(
                    f"deepcopy({self.__class__.__name__}) {key}: {value} FAIL!"
                )
                breakpoint()
        return result

    def __add__(self, cat, out_survey=None):
        if not cat.__class__.__name__ == "Spectral_Catalogue":
            # concat catalogues
            if out_survey == None:
                out_survey = "+".join([self.survey, cat.survey])
            return Multiple_Catalogue([self, cat], survey=out_survey)
    
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
        'Multiply' two catalogues by performing a cross-match and filtering the best matching galaxies.
        Ensures no duplicate galaxies are present in the output catalogue.

        Parameters:
        - other: Catalogue_Base
            The other catalogue to multiply with.
        - out_survey: str, optional
            The name of the output survey. Default is None.
        - max_sep: Quantity, optional
            The maximum separation allowed for matching galaxies. Default is 1.0 arcsec.
        - match_type: str, optional
            The type of matching to perform. Options are 'nearest' and 'compare_within_radius'.
            Default is 'compare_within_radius'.

        Returns:
        - Multiple_Catalogue
            The resulting MultipleCatalogue after performing the multiplication, with duplicates removed.

        """
        # cross-match catalogues
        # update .fits tables with cross-matched version
        # open tables
        self_copy = deepcopy(self)

        if isinstance(other, Multiple_Catalogue):
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
            wcs_self = WCS(fits.getheader(self_copy.data.im_paths[band], ext=self_copy.data.im_exts[band]))
            wcs_other = WCS(fits.getheader(other_copy.data.im_paths[band], ext=other_copy.data.im_exts[band]))
            if not (any(sky_coords_cat.contained_by(wcs_self)) and any(other_sky_coords.contained_by(wcs_other))):
                cat_matches = []
                other_cat_matches = []
                print('Skipping')
                continue

            # Check if likely to have any matches
            idx, _, _, _ = sky_coords_cat.search_around_sky(other_sky_coords, 5*u.arcsec)

        
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
                # Also check mask - don't keep masked galaxies where there is an unmasked match
                sep_constraint = d2d < max_sep
                # Get indexes of matches
                cat_matches = np.arange(len(sky_coords_cat))[sep_constraint]
                other_cat_matches = idx[sep_constraint]

            elif match_type == "compare_within_radius":
                # This finds all matches within a certain radius and compares the photometry of the galaxies
                cat_matches = []
                other_cat_matches = []

                for pos, coord in tqdm(
                    enumerate(self_copy.sky_coord),
                    desc="Cross-matching galaxies",
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
                            coord_gal.phot.instrument.band_names
                        )[coord_gal.phot.flux.mask]

                        chi_squareds = []
                        for other_gal in other_gals:
                            bands_gal2 = np.array(
                                other_gal.phot.instrument.band_names
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
                                    for band in coord_gal.phot.instrument.band_names
                                ]
                            )
                            indexes_other_bands = np.argwhere(
                                [
                                    band in matched_bands
                                    for band in other_gal.phot.instrument.band_names
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

            assert (
                len(cat_matches) == len(other_cat_matches)
            ), (
                f"{len(cat_matches)} != {len(other_cat_matches)}"
            )  # check that the matches are 1-to-1
            print("Getting galaxies")
            print(cat_matches)
            cat_matches = np.array(cat_matches, dtype=int)
            gal_matched_cat = self_copy[cat_matches]
            # Use indexes instead
            other_cat_matches = np.array(other_cat_matches, dtype=int)
            gal_matched_other = other_copy[other_cat_matches]
            print("Obtained matched galaxies")
            assert len(gal_matched_cat) == len(
                gal_matched_other
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
                        bands_gal1 = np.array(gal1.phot.instrument.band_names)[
                            gal1.phot.flux.mask
                        ]
                        bands_gal2 = np.array(gal2.phot.instrument.band_names)[
                            gal2.phot.flux.mask
                        ]

                        band_names_union = list(
                            set(bands_gal1).union(set(bands_gal2))
                        )

                        if len(bands_gal2) > len(bands_gal1):
                            self_copy.remove_gal(id=gal1.ID)
                        elif len(bands_gal2) < len(bands_gal1):
                            other_copy.remove_gal(id=gal2.ID)
                        else:
                            # If same bands, choose galaxy with deeper depth
                            # Get matching bands between the two galaxies - only use if not masked
                            # logical comparison of depth in each band, keeping the galaxy with the deeper depth in more bands
                            # gal1.phot.depths is just an array. Need to slice by position
                            indexes_gal1 = np.argwhere(
                                [
                                    band in band_names_union
                                    for band in gal1.phot.instrument.band_names
                                ]
                            )
                            depths_gal1 = gal1.phot.depths[indexes_gal1]
                            indexes_gal2 = np.argwhere(
                                [
                                    band in band_names_union
                                    for band in gal2.phot.instrument.band_names
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
        return Multiple_Catalogue([self_copy, *new_copies], survey=out_survey)

    # Need to save the cross-match distances

    def combine_and_remove_duplicates(
        self,
        other,
        out_survey=None,
        max_sep=1.0 * u.arcsec,
        match_type="nearest",
    ):
        "Alias for self * other"
        return self.__mul__(
            other,
            out_survey=out_survey,
            max_sep=max_sep,
            match_type=match_type,
        )

    @property
    def cat_dir(self):
        return funcs.split_dir_name(self.cat_path, "dir")

    @property
    def cat_name(self):
        return funcs.split_dir_name(self.cat_path, "name")

    @property
    def ra_range(self):
        try:
            return self._ra_range
        except:
            self._ra_range = [np.min(self.RA.value), np.max(self.RA.value)] * self.RA.unit
            return self._ra_range

    @property
    def dec_range(self):
        try:
            return self._dec_range
        except:
            self._dec_range = [np.min(self.DEC.value), np.max(self.DEC.value)] * self.DEC.unit
            return self._dec_range

    # TODO: should be __sub__ instead
    def remove_gal(self, index=None, id=None):
        if index is not None:
            self.gals = np.delete(self.gals, index)
        elif id is not None:
            self.gals = np.delete(self.gals, np.where(self.ID == id))
        else:
            galfind_logger.critical("No index or ID provided to remove_gal!")

    def crop(
        self,
        crop_limits: Union[int, float, bool, list, np.array],
        crop_property: str,
        SED_fit_params: Union[dict, str] = "EAZY_fsps_larson_zfree",
        phot_type: str = "obs",
    ):  # -> self.__class__
        cat_copy = deepcopy(self)
        if type(crop_limits) in [int, float, bool]:
            cat_copy.gals = cat_copy[
                cat_copy.__getattr__(crop_property, origin=SED_fit_params)
                == crop_limits
            ]
            if crop_limits == True:
                cat_copy.crops.append(crop_property)
            else:
                cat_copy.crops.append(f"{crop_property}={crop_limits}")
        elif type(crop_limits) in [list, np.array]:
            cat_copy.gals = cat_copy[
                (
                    (
                        cat_copy.__getattr__(
                            crop_property, origin=SED_fit_params
                        )
                        >= crop_limits[0]
                    )
                    & (
                        cat_copy.__getattr__(
                            crop_property, origin=SED_fit_params
                        )
                        <= crop_limits[1]
                    )
                )
            ]
            cat_copy.crops.append(
                f"{crop_limits[0]}<{crop_property}<{crop_limits[1]}"
            )
        else:
            galfind_logger.critical(
                f"crop_limits={crop_limits} with type = {type(crop_limits)} not in [int, float, bool, list, np.array]"
            )
        return cat_copy

    def open_cat(self, cropped=False, hdu=None):
        if type(hdu) == type(None):
            fits_cat = Table.read(
                self.cat_path, character_as_bytes=False, memmap=True
            )
        elif self.check_hdu_exists(hdu):
            fits_cat = Table.read(
                self.cat_path, character_as_bytes=False, memmap=True, hdu=hdu
            )
        else:
            galfind_logger.warning(
                f"{hdu.upper()=} does not exist in {self.cat_path=}!"
            )
            return None
        if cropped:
            ID_tab = Table({"IDs_temp": self.ID}, dtype=[int])
            if type(hdu) == type(None):
                keys_left = self.cat_creator.ID_label
            elif hdu.upper() in ["OBJECTS"]:  # , "GALFIND_CAT"]:
                keys_left = self.cat_creator.ID_label
            else:
                keys_left = sed_code_to_name_dict[hdu.split("_")[0]].ID_label
            combined_tab = join(
                fits_cat, ID_tab, keys_left=keys_left, keys_right="IDs_temp"
            )
            combined_tab.remove_column("IDs_temp")
            combined_tab.meta = fits_cat.meta
            return combined_tab
        else:
            return fits_cat

    def check_hdu_exists(self, hdu):
        # check whether the hdu extension exists
        hdul = fits.open(self.cat_path)
        return any(hdu_.name == hdu.upper() for hdu_ in hdul)

    def write_cat(self, tab_arr, tab_names):
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
        hdu_list.writeto(self.cat_path, overwrite=True)
        funcs.change_file_permissions(self.cat_path)
        galfind_logger.info(f"Writing table to {self.cat_path}")

    def write_hdu(self, tab: Table, hdu: str):
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
        self.write_cat(tab_arr, tab_names)

    def del_hdu(self, hdu):
        if self.check_hdu_exists(hdu):  # delete hdu if it exists
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
            self.write_cat(tab_arr, tab_names)
            galfind_logger.info(
                f"Deleted {hdu.upper()=} from {self.cat_path=}!"
            )
        else:
            galfind_logger.info(
                f"{hdu.upper()=} does not exist in {self.cat_path=}, could not delete!"
            )

    def del_cols_hdrs_from_fits(self, col_names=[], hdr_names=[], hdu=None):
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
                    assert all(name in tab.colnames for name in col_names)
                    assert all(name in tab.meta.keys() for name in hdr_names)
                    [tab.remove_column(name) for name in col_names]
                    tab.meta = {
                        key: value
                        for key, value in dict(tab.meta).items()
                        if key not in hdr_names
                    }
                tab_arr.append(tab)
                tab_names.append(hdu_.name)
        self.write_cat(tab_arr, tab_names)

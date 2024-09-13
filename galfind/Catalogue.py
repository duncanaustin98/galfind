#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:27:47 2023

@author: austind
"""

# Catalogue.py
import glob
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Union

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, join, vstack
from astropy.wcs import WCS
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Catalogue_Creator
    from . import Instrument

from . import (
    Catalogue_Base,
    Photometry_rest,
    config,
    galfind_logger,
)
from . import useful_funcs_austind as funcs
from .Data import Data
from .EAZY import EAZY
from .Emission_lines import line_diagnostics
from .Galaxy import Galaxy, Multiple_Galaxy
from .SED_codes import SED_code
from .Spectrum import Spectral_Catalogue


class Catalogue(Catalogue_Base):
    @classmethod
    def from_pipeline(
        cls,
        survey: str,
        version: str,
        cat_creator: type[Catalogue_Creator],
        SED_fit_params_arr: Union[list, np.array] = [
            {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": 4.0},
            {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": 6.0},
            {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None},
        ],
        instrument_name: Union[list, np.array] = [
            "NIRCam",
            "ACS_WFC",
            "WFC3_IR",
        ],
        forced_phot_band: Union[list, np.array, str] = [
            "F277W",
            "F356W",
            "F444W",
        ],
        excl_bands: Union[list, np.array] = [],
        pix_scales: dict = {
            "ACS_WFC": 0.03 * u.arcsec,
            "WFC3_IR": 0.03 * u.arcsec,
            "NIRCam": 0.03 * u.arcsec,
            "MIRI": 0.09 * u.arcsec,
        },
        crop_by: Union[None, str, dict, list, np.array] = None,
        load_PDFs: Union[bool, dict] = True,
        load_SEDs: Union[bool, dict] = True,
        timed: bool = True,
        mask_stars: bool = True,
        load_SED_rest_properties: bool = True,
        sex_prefer: str = "wht",
        n_depth_reg: Union[str, int] = "auto",
        load_ext_src_corrs: bool = True,
    ):
        data = Data.from_pipeline(
            survey,
            version,
            instrument_name,
            excl_bands=excl_bands,
            mask_stars=mask_stars,
            pix_scales=pix_scales,
            sex_prefer=sex_prefer,
        )

        return cls.from_data(
            data,
            version,
            cat_creator,
            SED_fit_params_arr,
            forced_phot_band,
            crop_by=crop_by,
            load_PDFs=load_PDFs,
            load_SEDs=load_SEDs,
            timed=timed,
            load_SED_rest_properties=load_SED_rest_properties,
            n_depth_reg=n_depth_reg,
            load_ext_src_corrs=load_ext_src_corrs,
            sex_prefer=sex_prefer,
        )

    @classmethod
    def from_data(
        cls,
        data: Data,
        version: str,
        cat_creator: type[Catalogue_Creator],
        SED_fit_params_arr: Union[list, np.array],
        forced_phot_band: Union[str, list, np.array] = [
            "F277W",
            "F356W",
            "F444W",
        ],
        mask: bool = True,
        crop_by: Union[None, str, dict, list, np.array] = None,
        load_PDFs: Union[bool, dict] = True,
        load_SEDs: Union[bool, dict] = True,
        timed: bool = True,
        load_SED_rest_properties: bool = True,
        sex_prefer: str = "rms_err",
        n_depth_reg: Union[str, int] = "auto",
        load_ext_src_corrs: bool = True,
    ):
        # make masked local depth catalogue from the 'Data' object
        data.combine_sex_cats(forced_phot_band, prefer=sex_prefer, timed=timed)
        mode = str(
            config["Depths"]["MODE"]
        ).lower()  # mode to calculate depths (either "n_nearest" or "rolling")
        for aper_diam in json.loads(config.get("SExtractor", "APER_DIAMS")):
            data.calc_depths(
                aper_diam * u.arcsec,
                mode=mode,
                cat_creator=cat_creator,
                n_split=n_depth_reg,
                timed=timed,
            )
            data.perform_aper_corrs()
            data.make_loc_depth_cat(cat_creator, depth_mode=mode)

        return cls.from_fits_cat(
            data.sex_cat_master_path,
            version,
            data.instrument,
            cat_creator,
            data.survey,
            SED_fit_params_arr,
            data=data,
            mask=mask,
            crop_by=crop_by,
            load_PDFs=load_PDFs,
            load_SEDs=load_SEDs,
            timed=timed,
            load_SED_rest_properties=load_SED_rest_properties,
            load_ext_src_corrs=load_ext_src_corrs,
        )

    @classmethod
    def from_fits_cat(
        cls,
        fits_cat_path: str,
        version: str,
        instrument: type[Instrument],
        cat_creator: type[Catalogue_Creator],
        survey: str,
        SED_fit_params_arr: Union[list, np.array],
        data: Union[None, Data] = None,
        mask: bool = False,
        excl_bands: Union[list, np.array] = [],
        crop_by: Union[None, str, dict, list, np.array] = None,
        load_PDFs: Union[bool, dict] = True,
        load_SEDs: Union[bool, dict] = True,
        timed: bool = True,
        load_SED_rest_properties: bool = True,
        load_ext_src_corrs: bool = True,
    ):
        # open the catalogue
        fits_cat = funcs.cat_from_path(fits_cat_path)
        for band in deepcopy(instrument):
            try:
                cat_creator.load_photometry(
                    Table(fits_cat[0]), [band.band_name]
                )
            except:
                # no data for the relevant band within the catalogue
                instrument -= band
                print(f"{band.band_name} flux not loaded")
        print(f"instrument band names = {instrument.band_names}")

        # crop fits catalogue by the crop_by column name should it exist
        assert type(crop_by) in [type(None), str, list, np.array, dict]
        if type(crop_by) in [str]:
            crop_by = crop_by.split("+")
        if type(crop_by) == type(None):
            pass
        elif type(crop_by) == dict:
            for key, values in crop_by.items():
                # currently only crops by ID
                if "ID" in key.upper():
                    fits_cat = fits_cat[
                        np.logical_or.reduce(
                            [
                                fits_cat[cat_creator.ID_label].astype(int)
                                == int(value)
                                for value in values
                            ]
                        )
                    ]
        elif type(crop_by) in [list, np.array]:
            for name in crop_by:
                if name[:3] == "ID=":
                    fits_cat = fits_cat[
                        fits_cat[cat_creator.ID_label].astype(int)
                        == int(name[3:])
                    ]
                    galfind_logger.info(f"Catalogue cropped to {name}")
                elif (
                    name in fits_cat.colnames
                ):  # , galfind_logger.critical(f"Cannot crop by {name}")
                    if type(fits_cat[name][0]) in [bool, np.bool_]:  # , \
                        fits_cat = fits_cat[fits_cat[name]]
                        galfind_logger.info(
                            f"Catalogue for {survey} {version} cropped by {name}"
                        )
                    else:
                        galfind_logger.warning(
                            f"{type(fits_cat[name][0])=} not in [bool, np.bool_]"
                        )
                else:
                    galfind_logger.warning(
                        f"Invalid crop name == {name}! Skipping"
                    )
        # produce galaxy array from each row of the catalogue
        if timed:
            start_time = time.time()
        gals = Multiple_Galaxy.from_fits_cat(
            fits_cat, instrument, cat_creator, [{}], timed=timed
        ).gals  # codes, lowz_zmax, templates_arr).gals
        if timed:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(
                f"Finished loading in {len(gals)} galaxies. This took {elapsed_time:.6f} seconds"
            )
        # make catalogue with no SED fitting information
        cat_obj = cls(
            gals,
            fits_cat_path,
            survey,
            cat_creator,
            instrument,
            SED_fit_params_arr,
            version=version,
            crops=crop_by,
        )

        if cat_obj is not None:
            cat_obj.data = data
        if mask:
            cat_obj.mask(timed=timed)

        # run SED fitting for the appropriate SED_fit_params
        for SED_fit_params in SED_fit_params_arr:
            if type(load_PDFs) in [dict]:
                assert (
                    SED_fit_params["code"].__class__.__name__
                    in load_PDFs.keys()
                )
                _load_PDFs = load_PDFs[
                    SED_fit_params["code"].__class__.__name__
                ]
                assert type(_load_PDFs) in [bool]
            else:
                _load_PDFs = load_PDFs
            if type(load_SEDs) in [dict]:
                assert (
                    SED_fit_params["code"].__class__.__name__
                    in load_SEDs.keys()
                )
                _load_SEDs = load_SEDs[
                    SED_fit_params["code"].__class__.__name__
                ]
                assert type(_load_SEDs) in [bool]
            else:
                _load_SEDs = load_SEDs
            cat_obj = SED_fit_params["code"].fit_cat(
                cat_obj,
                SED_fit_params,
                load_PDFs=_load_PDFs,
                load_SEDs=_load_SEDs,
                timed=timed,
            )
            if load_SED_rest_properties:
                cat_obj.load_SED_rest_properties(SED_fit_params, timed=timed)
        # make extended source corrections for properties that require it
        if load_ext_src_corrs:
            cat_obj.make_all_ext_src_corrs()
        return cat_obj

    def save_phot_PDF_paths(self, PDF_paths, SED_fit_params):
        if "phot_PDF_paths" not in self.__dict__.keys():
            self.phot_PDF_paths = {}
        self.phot_PDF_paths[
            SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)
        ] = PDF_paths

    def save_phot_SED_paths(self, SED_paths, SED_fit_params):
        if "phot_SED_paths" not in self.__dict__.keys():
            self.phot_SED_paths = {}
        self.phot_SED_paths[
            SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)
        ] = SED_paths

    def update_SED_results(self, cat_SED_results, timed=True):
        assert (
            len(cat_SED_results) == len(self)
        )  # if this is not the case then instead should cross match IDs between self and gal_SED_result
        galfind_logger.info("Updating SED results in galfind catalogue object")
        # deepcopying here?
        if timed:
            [
                gal.update(gal_SED_result)
                for gal, gal_SED_result in tqdm(
                    zip(self, cat_SED_results),
                    desc="Updating galaxy SED results",
                    total=len(self),
                )
            ]
        else:
            [
                gal.update(gal_SED_result)
                for gal, gal_SED_result in zip(self, cat_SED_results)
            ]

    # Spectroscopy

    def match_available_spectra(self):
        # make catalogue consisting of spectra downloaded from the DJA
        DJA_cat = np.sum(
            [
                Spectral_Catalogue.from_DJA(
                    ra_range=self.ra_range,
                    dec_range=self.dec_range,
                    version=version,
                )
                for version in ["v1", "v2"]
            ]
        )
        # cross match this catalogue
        cross_matched_cat = self * DJA_cat
        print(str(cross_matched_cat))
        return cross_matched_cat

    # %%

    def calc_ext_src_corrs(self) -> None:
        self.load_sex_flux_mag_autos()
        # calculate aperture corrections if not already
        if not hasattr(self.instrument, "aper_corrs"):
            aper_corrs = self.instrument.get_aper_corrs(
                self.cat_creator.aper_diam
            )
        else:
            aper_corrs = self.instrument.aper_corrs[self.cat_creator.aper_diam]
        assert len(aper_corrs) == len(self.instrument)
        aper_corrs = {
            band_name: aper_corr
            for band_name, aper_corr in zip(
                self.instrument.band_names, aper_corrs
            )
        }
        # calculate and save dict of ext_src_corrs for each galaxy in self
        galfind_logger.debug(
            "Photometry_obs.calc_ext_src_corrs takes 2min 20s for JOF with 16,000 galaxies. Fairly slow!"
        )
        [
            gal.phot.calc_ext_src_corrs(aper_corrs=aper_corrs)
            for gal in tqdm(
                self,
                desc=f"Calculating extended source corrections for {self.survey} {self.version}",
                total=len(self),
            )
        ]
        # save the extended source corrections to catalogue
        self._append_property_to_tab(
            "_".join(inspect.stack()[0].function.split("_")[1:]), "phot_obs"
        )

    def make_ext_src_corrs(
        self, gal_property: str, origin: Union[str, dict]
    ) -> str:
        # calculate pre-requisites
        self.calc_ext_src_corrs()
        # make extended source correction for given property
        [gal.phot.make_ext_src_corrs(gal_property, origin) for gal in self]
        # save properties to fits table
        property_name = f"{gal_property}{funcs.ext_src_label}"
        self._append_property_to_tab(property_name, origin)
        return property_name

    def make_all_ext_src_corrs(self) -> None:
        self.calc_ext_src_corrs()
        properties_dict = [gal.phot.make_all_ext_src_corrs() for gal in self]
        unique_origins = np.unique(
            [property_dict.keys() for property_dict in properties_dict]
        )[0]
        unique_properties_origins_dict = {
            key: np.unique(
                [property_dict[key] for property_dict in properties_dict]
            )
            for key in unique_origins
        }
        unique_properties_origins_dict = {
            key: value
            for key, value in unique_properties_origins_dict.items()
            if len(value) > 0
        }
        # breakpoint()
        [
            self._append_property_to_tab(property_name, origin)
            for origin, property_names in unique_properties_origins_dict.items()
            for property_name in property_names
        ]

    def _append_property_to_tab(
        self,
        property_name: str,
        origin: Union[str, dict],
        overwrite: bool = False,
    ) -> None:
        galfind_logger.info(
            "Catalogue._append_property_to_tab doesn't work if trying to append newly updated galaxy properties YET!"
        )
        # extract catalogue to append to
        if type(origin) in [dict]:
            # convert SED_fit_params origin to str
            assert "code" in origin.keys()
            origin = origin["code"].label_from_SED_fit_params(origin)
        if origin in ["gal", "phot_obs"]:
            hdu = "OBJECTS"
            ID_label = self.cat_creator.ID_label
        else:
            hdu = origin.replace("_REST_PROPERTY", "")
            ID_label = sed_code_to_name_dict[origin.split("_")[0]].ID_label

        append_tab = self.open_cat(
            cropped=False, hdu=hdu
        )  # this should really be cached in Catalogue_Base
        if type(append_tab) == type(None):
            galfind_logger.critical(
                f"Skipping appending of {property_name=}, {origin=} as append_tab does not exist!"
            )
            return None
        # append to .fits table only if not already
        if property_name in append_tab.colnames:
            galfind_logger.info(
                f"{property_name=} already appended to {origin=} .fits table!"
            )
            return None
        else:
            galfind_logger.info(
                f"Appending {property_name=} to {origin=} .fits table!"
            )
            # make new table with calculated properties
            gal_IDs = self.__getattr__("ID")
            gal_properties = self.__getattr__(property_name, origin=origin)
            if all(
                type(gal_property) == dict for gal_property in gal_properties
            ):
                all_keys = np.unique(
                    [
                        f"{property_name}_{key}"
                        for gal_property in gal_properties
                        for key in gal_property.keys()
                    ]
                )
                # skip if all columns already appended and overwrite == False
                if (
                    all(key in append_tab.colnames for key in all_keys)
                    and not overwrite
                ):
                    return None
                property_dict = {
                    key: np.array(
                        [
                            gal_property[key.replace(f"{property_name}_", "")]
                            if key.replace(f"{property_name}_", "")
                            in gal_property.keys()
                            else None
                            for gal_property in gal_properties
                        ]
                    )
                    for key in all_keys
                }
                appended_keys = [
                    key
                    for key in property_dict.keys()
                    if key in append_tab.colnames
                ]
                # ensure the type of each element in the dict is the same
                expected_type = type(
                    gal_properties[0][list(gal_properties[0].keys())[0]]
                )
                assert all(
                    type(value) == expected_type
                    for gal_property in gal_properties
                    for value in gal_property.values()
                )
                if overwrite:
                    # remove old columns that already exist
                    [append_tab.remove_column(key) for key in appended_keys]
                else:
                    # remove new columns that already exist
                    [property_dict.pop(key) for key in appended_keys]
                property_types = [
                    bool if expected_type in [bool, np.bool_] else float
                ] * len(appended_keys)
            elif any(
                type(gal_property) == dict for gal_property in gal_properties
            ):
                galfind_logger.critical(
                    f"{property_name}={gal_properties} from {origin=} should not have mixed 'dict' + other element types!"
                )
                breakpoint()
            else:
                assert all(
                    type(gal_property)
                    in [bool, np.bool_, u.Quantity, u.Magnitude, u.Dex]
                    for gal_property in gal_properties
                )
                assert all(
                    type(gal_property) == type(gal_properties[0])
                    for gal_property in gal_properties
                )
                property_dict = {property_name: gal_properties}
                property_types = [
                    bool
                    if type(gal_properties[0]) in [bool, np.bool_]
                    else float
                ]
            new_tab = Table(
                {**{"ID_temp": gal_IDs}, **property_dict},
                dtype=[int] + property_types,
            )
            # join new and old tables
            out_tab = join(
                append_tab,
                new_tab,
                keys_left=ID_label,
                keys_right="ID_temp",
                join_type="outer",
            )
            out_tab.remove_column("ID_temp")
            breakpoint()
            # save multi-extension table
            self.write_hdu(out_tab, hdu=hdu)

    # def calc_new_property(self, func: Callable[..., float], arg_names: Union[list, np.array]):
    #     pass

    def load_band_properties_from_cat(
        self,
        cat_colname: str,
        save_name: str,
        multiply_factor: Union[dict, u.Quantity, u.Magnitude, None] = None,
        dest: str = "gal",
    ) -> None:
        assert dest in ["gal", "phot_obs"]
        if dest == "gal":
            has_attr = hasattr(self[0], save_name)
        else:  # dest == "phot_obs"
            has_attr = hasattr(self[0].phot, save_name)
        if not has_attr:
            # load the same property from every available band
            # open catalogue with astropy
            fits_cat = self.open_cat(cropped=True)
            if type(multiply_factor) == type(None):
                multiply_factor = {
                    band: 1.0 * u.dimensionless_unscaled
                    for band in self.instrument.band_names
                    if f"{cat_colname}_{band}" in fits_cat.colnames
                }
            elif type(multiply_factor) != dict:
                multiply_factor = {
                    band: multiply_factor
                    for band in self.instrument.band_names
                    if f"{cat_colname}_{band}" in fits_cat.colnames
                }
            # load in speed can be improved here!
            cat_band_properties = {
                band: np.array(fits_cat[f"{cat_colname}_{band}"])
                * multiply_factor[band]
                for band in self.instrument.band_names
                if f"{cat_colname}_{band}" in fits_cat.colnames
            }
            if len(cat_band_properties) == 0:
                galfind_logger.info(
                    f"Could not load {cat_colname=} from {self.cat_path}, as no '{cat_colname}_band' exists for band in {self.instrument.band_names=}!"
                )
            else:
                cat_band_properties = [
                    {
                        band: cat_band_properties[band][i]
                        for band in cat_band_properties.keys()
                    }
                    for i in range(len(fits_cat))
                ]
                if dest == "gal":
                    [
                        gal.load_property(gal_properties, save_name)
                        for gal, gal_properties in zip(
                            self, cat_band_properties
                        )
                    ]
                else:  # dest == "phot_obs"
                    [
                        gal.phot.load_property(gal_properties, save_name)
                        for gal, gal_properties in zip(
                            self, cat_band_properties
                        )
                    ]
                galfind_logger.info(
                    f"Loaded {cat_colname} from {self.cat_path} saved as {save_name} for bands = {cat_band_properties[0].keys()}"
                )

    def load_property_from_cat(
        self,
        cat_colname: str,
        save_name: str,
        multiply_factor: Union[u.Quantity, u.Magnitude] = 1.0
        * u.dimensionless_unscaled,
        dest: str = "gal",
    ):
        assert dest in ["gal", "phot_obs"]
        if dest == "gal":
            has_attr = hasattr(self[0], save_name)
        else:  # dest == "phot_obs"
            has_attr = hasattr(self[0].phot, save_name)
        if not has_attr:
            # open catalogue with astropy
            fits_cat = self.open_cat(cropped=True)
            if cat_colname in fits_cat.colnames:
                cat_property = np.array(fits_cat[cat_colname])
                assert len(cat_property) == len(self)
                if dest == "gal":
                    [
                        gal.load_property(
                            gal_property * multiply_factor, save_name
                        )
                        for gal, gal_property in zip(self, cat_property)
                    ]
                else:  # dest == "phot_obs"
                    [
                        gal.phot.load_property(
                            gal_property * multiply_factor, save_name
                        )
                        for gal, gal_property in zip(self, cat_property)
                    ]
                galfind_logger.info(
                    f"Loaded {cat_colname=} from {self.cat_path} saved as {save_name}!"
                )
            else:
                galfind_logger.info(
                    f"{cat_colname=} does not exist in {self.cat_path}, skipping!"
                )

    def load_sex_flux_mag_autos(self):
        # sex_band_names = [band_name for band_name, cat_type in self.data.sex_cat_types.items() if "SExtractor" in cat_type]
        flux_im_to_Jy_conv = {
            band_name: funcs.flux_image_to_Jy(1.0, self.data.im_zps[band_name])
            for band_name in self.instrument.band_names
        }
        self.load_band_properties_from_cat(
            "FLUX_AUTO",
            "FLUX_AUTO",
            multiply_factor=flux_im_to_Jy_conv,
            dest="phot_obs",
        )
        self.load_band_properties_from_cat(
            "MAG_AUTO", "MAG_AUTO", multiply_factor=u.ABmag, dest="phot_obs"
        )

    def mask(self, timed: bool = True):  # , mask_instrument = NIRCam()):
        galfind_logger.info(f"Running masking code for {self.cat_path}.")
        # determine whether to overwrite catalogue or not
        overwrite = config["Masking"].getboolean("OVERWRITE_MASK_COLS")
        if overwrite:
            galfind_logger.info(
                "OVERWRITE_MASK_COLS = YES, updating catalogue with masking columns."
            )
        # open catalogue with astropy
        fits_cat = self.open_cat(cropped=True)

        # update input catalogue if it hasnt already been masked or if wanted ONLY if len(self) == len(cat)
        if len(self) != len(fits_cat):
            galfind_logger.warning(
                f"len(self) = {len(self)}, len(cat) = {len(fits_cat)} -> len(self) != len(cat). Skipping masking for {self.survey} {self.version}!"
            )

        elif "MASKED" not in fits_cat.meta.keys() or overwrite:
            galfind_logger.info(
                f"Masking catalogue for {self.survey} {self.version}"
            )

            # calculate x,y for each galaxy in catalogue
            # cat_x, cat_y = self.data.load_wcs(self.data.alignment_band).world_to_pixel(cat_sky_coords)
            cat_sky_coords = SkyCoord(
                fits_cat[self.cat_creator.ra_dec_labels["RA"]],
                fits_cat[self.cat_creator.ra_dec_labels["DEC"]],
            )

            # make columns for individual band masking
            if config["Masking"].getboolean("MASK_BANDS"):
                unmasked_band_dict = {}
                masks = [
                    self.data.load_mask(band)
                    for band in self.instrument.band_names
                ]
                # if masks are all the same shape
                if all(mask.shape == masks[0].shape for mask in masks):
                    cat_x, cat_y = self.data.load_wcs(
                        self.data.alignment_band
                    ).world_to_pixel(cat_sky_coords)
                    unmasked_band_dict = {
                        band: np.array(
                            [
                                False
                                if x < 0.0
                                or x >= mask.shape[1]
                                or y < 0.0
                                or y >= mask.shape[0]
                                else not bool(mask[int(y)][int(x)])
                                for x, y in zip(cat_x, cat_y)
                            ]
                        )
                        for band, mask in tqdm(
                            zip(self.instrument.band_names, masks),
                            desc="Masking galfind catalogue object",
                            total=len(self.instrument),
                        )
                    }
                else:
                    unmasked_band_dict = {}
                    for band, mask in tqdm(
                        zip(self.instrument.band_names, masks),
                        desc="Masking galfind catalogue object",
                        total=len(self.instrument),
                    ):
                        # convert catalogue RA/Dec to mask X/Y co-ordinates using image wcs
                        cat_x, cat_y = self.data.load_wcs(band).world_to_pixel(
                            cat_sky_coords
                        )
                        # determine whether a galaxy is unmasked
                        unmasked_band_dict[band] = np.array(
                            [
                                False
                                if x < 0.0
                                or x >= mask.shape[1]
                                or y < 0.0
                                or y >= mask.shape[0]
                                else not bool(mask[int(y)][int(x)])
                                for x, y in zip(cat_x, cat_y)
                            ]
                        )
                # update catalogue with new columns
                for band, unmasked_band in unmasked_band_dict.items():
                    fits_cat[f"unmasked_{band}"] = (
                        unmasked_band  # assumes order of catalogue and galaxies in self is consistent
                    )
                    # update galaxy objects in catalogue - current bottleneck
                    [
                        gal.mask_flags.update({band: unmasked_band_gal})
                        for gal, unmasked_band_gal in zip(self, unmasked_band)
                    ]

            # determine which cluster/blank masking columns are wanted
            mask_labels = []
            mask_paths = []
            default_blank_bool_arr = []
            if config["Masking"].getboolean(
                "MASK_CLUSTER_MODULE"
            ):  # make blank field mask
                mask_labels.append("blank_module")
                mask_paths.append(self.data.blank_mask_path)
                default_blank_bool_arr.append(True)
            if config["Masking"].getboolean(
                "MASK_CLUSTER_CORE"
            ):  # make cluster mask
                mask_labels.append("cluster")
                mask_paths.append(self.data.cluster_mask_path)
                default_blank_bool_arr.append(False)

            # mask columns in catalogue + galfind galaxies
            cat_x, cat_y = self.data.load_wcs(
                self.data.alignment_band
            ).world_to_pixel(cat_sky_coords)
            for mask_label, mask_path, default_blank_bool in zip(
                mask_labels, mask_paths, default_blank_bool_arr
            ):
                # if using a blank field
                if self.data.is_blank:
                    galfind_logger.info(
                        f"{self.survey} {self.version} is blank. Making '{mask_label}' boolean columns"
                    )
                    mask_data = [
                        default_blank_bool for i in range(len(fits_cat))
                    ]  # default behaviour
                else:
                    galfind_logger.info(
                        f"{self.survey} {self.version} contains a cluster. Making '{mask_label}' boolean columns"
                    )
                    # open relevant .fits mask
                    mask = fits.open(mask_path)[1].data
                    # determine whether a galaxy is in a blank module
                    if default_blank_bool:  # True if outside the mask
                        galfind_logger.warning(
                            "This masking assumes that the blank mask covers the cluster module and then invokes negatives."
                        )
                        mask_data = np.array(
                            [
                                False
                                if x < 0.0
                                or x >= mask.shape[1]
                                or y < 0.0
                                or y >= mask.shape[0]
                                else not bool(mask[int(y)][int(x)])
                                for x, y in zip(cat_x, cat_y)
                            ]
                        )
                    else:  # True if within the mask
                        mask_data = np.array(
                            [
                                False
                                if x < 0.0
                                or x >= mask.shape[1]
                                or y < 0.0
                                or y >= mask.shape[0]
                                else bool(mask[int(y)][int(x)])
                                for x, y in zip(cat_x, cat_y)
                            ]
                        )
                fits_cat[mask_label] = (
                    mask_data  # update catalogue with boolean column
                )

            # update catalogue metadata
            fits_cat.meta = {
                **fits_cat.meta,
                **{
                    "MASKED": True,
                    "HIERARCH MASK_BANDS": config["Masking"].getboolean(
                        "MASK_BANDS"
                    ),
                    "HIERARCH MASK_CLUSTER_MODULE": config[
                        "Masking"
                    ].getboolean("MASK_CLUSTER_MODULE"),
                    "HIERARCH MASK_CLUSTER_CORE": config["Masking"].getboolean(
                        "MASK_CLUSTER_CORE"
                    ),
                },
            }
            # save catalogue
            fits_cat.write(self.cat_path, overwrite=True)
            funcs.change_file_permissions(self.cat_path)
            # update catalogue README
            galfind_logger.warning(
                "REQUIRED UPDATE: Update README for catalogue masking columns"
            )
            # update masking of galfind galaxy objects
            galfind_logger.info("Masking galfind galaxy objects in catalogue")
            assert len(fits_cat) == len(self)
            mask_arr = self.cat_creator.load_mask(
                fits_cat,
                self.instrument.band_names,
                gal_band_mask=self.cat_creator.load_photometry(
                    fits_cat, self.instrument.band_names
                )[2],
            )
            [
                gal.update_mask(mask, update_phot_rest=False)
                for gal, mask in tqdm(
                    zip(self, mask_arr),
                    total=len(self),
                    desc="Masking galfind galaxy objects",
                )
            ]
        else:
            galfind_logger.info(
                f"Catalogue for {self.survey} {self.version} already masked. Skipping!"
            )

    def make_cutouts(
        self,
        IDs: Union[list, np.array],
        cutout_size: Union[u.Quantity, dict] = 0.96 * u.arcsec,
    ) -> None:
        if type(IDs) == int:
            IDs = [IDs]
        for band in tqdm(
            self.instrument.band_names,
            total=len(self.instrument),
            desc="Making band cutouts",
        ):
            #             rerun = False
            #             if config.getboolean("Cutouts", "OVERWRITE_CUTOUTS"):
            #                 rerun = True
            #             else:
            #                 for gal in self:
            #                     out_path = f"{config['Cutouts']['CUTOUT_DIR']}/{self.version}/{self.survey}/{band}/{gal.ID}.fits"
            #                     if Path(out_path).is_file():
            #                         size = fits.open(out_path)[0].header["size"]
            #                         if size != cutout_size:
            #                             rerun = True
            #                     else:
            #                         rerun = True
            #             if rerun:
            start = time.time()
            im_data, im_header, seg_data, seg_header = self.data.load_data(
                band, incl_mask=False
            )
            end1 = time.time()
            print("Time to load im/seg data:", end1 - start)
            wht_data = self.data.load_wht(band)
            end2 = time.time()
            print("Time to load wht data:", end2 - end1)
            rms_err_data = self.data.load_rms_err(band)
            end3 = time.time()
            print("Time to load rms_err data:", end3 - end2)
            wcs = WCS(im_header)
            pos = 0
            end = time.time()
            print("Time to load data:", end - start)

            for gal in self:
                if gal.ID in IDs:
                    if type(cutout_size) in [dict]:
                        cutout_size_gal = cutout_size[gal.ID]
                    else:
                        cutout_size_gal = cutout_size
                    gal.make_cutout(
                        band,
                        data={
                            "SCI": im_data,
                            "SEG": seg_data,
                            "WHT": wht_data,
                            "RMS_ERR": rms_err_data,
                        },
                        wcs=wcs,
                        im_header=im_header,
                        survey=self.survey,
                        version=self.version,
                        pix_scale=self.data.im_pixel_scales[band],
                        cutout_size=cutout_size_gal,
                    )
                    pos += 1

            end2 = time.time()
            print("Time to make cutouts:", end2 - end)

    #             else:
    #                 for gal in self:
    #                     if gal.ID in IDs:
    #                         gal.cutout_paths[band] = f"{config['Cutouts']['CUTOUT_DIR']}/{self.version}/{self.survey}/{band}/{gal.ID}.fits"
    #                 print(f"Cutouts for {band} already exist. Skipping.")

    def make_RGB_images(self, IDs, cutout_size=0.96 * u.arcsec):
        return NotImplementedError

    def plot_phot_diagnostics(
        self,
        SED_fit_params_arr=[
            {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None},
            {"code": EAZY(), "templates": "fsps_larson", "dz": 0.5},
        ],
        zPDF_plot_SED_fit_params_arr=[
            {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None},
            {"code": EAZY(), "templates": "fsps_larson", "dz": 0.5},
        ],
        wav_unit=u.um,
        flux_unit=u.ABmag,
    ):
        # figure size may well depend on how many bands there are
        overall_fig = plt.figure(figsize=(8, 7), constrained_layout=True)
        fig, cutout_fig = overall_fig.subfigures(
            2,
            1,
            hspace=-2,
            height_ratios=[2, 1]
            if len(self.data.instrument) <= 8
            else [1.8, 1],
        )

        gs = fig.add_gridspec(2, 4)
        phot_ax = fig.add_subplot(gs[:, 0:3])

        PDF_ax = [fig.add_subplot(gs[0, 3:]), fig.add_subplot(gs[1, 3:])]

        # plot SEDs
        out_paths = [
            gal.plot_phot_diagnostic(
                [cutout_fig, phot_ax, PDF_ax],
                self.data,
                SED_fit_params_arr,
                zPDF_plot_SED_fit_params_arr,
                wav_unit,
                flux_unit,
                aper_diam=self.cat_creator.aper_diam,
            )
            for gal in tqdm(
                self,
                total=len(self),
                desc="Plotting photometry diagnostic plots",
            )
        ]

        # make a folder to store symlinked photometric diagnostic plots for selected galaxies
        if self.crops != []:
            # create symlink to selection folder for diagnostic plots
            for gal, out_path in zip(self, out_paths):
                selection_path = f"{config['Selection']['SELECTION_DIR']}/SED_plots/{self.version}/{self.instrument.name}/{'+'.join(self.crops)}/{self.survey}/{str(gal.ID)}.png"
                funcs.make_dirs(selection_path)
                try:
                    os.symlink(out_path, selection_path)
                except FileExistsError:  # replace existing file
                    os.remove(selection_path)
                    os.symlink(out_path, selection_path)

    def plot(
        self,
        x_name: str,
        x_origin: Union[str, dict],
        y_name: str,
        y_origin: Union[str, dict],
        colour_by: Union[None, str] = None,
        c_origin: Union[str, dict, None] = None,
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
        plot_kwargs: dict = {},
        cmap: str = "viridis",
        save_type: str = ".png",
        fig=None,
        ax=None,
    ):
        if type(x_origin) in [dict]:
            assert "code" in x_origin.keys()
            assert x_origin["code"].__class__.__name__ in [
                code.__name__ for code in SED_code.__subclasses__()
            ]
        x = self.__getattr__(
            x_name, SED_fit_params=x_origin, property_type="vals"
        )
        if incl_x_errs:
            x_err = self.__getattr__(
                x_name, SED_fit_params=x_origin, property_type="errs"
            )
            x_err = np.array([x_err[:, 0], x_err[:, 1]])
        else:
            x_err = None
        if type(x_origin) in [dict]:
            x_label = x_origin["code"].gal_property_fmt_dict[x_name]
        else:
            NotImplementedError
        if log_x or x_name in funcs.logged_properties:
            if incl_x_errs:
                x, x_err = funcs.errs_to_log(x, x_err)
            else:
                x = np.log10(x)
            x_name = f"log({x_name})"
            x_label = f"log({x_label})"

        if type(y_origin) in [dict]:
            assert "code" in y_origin.keys()
            assert y_origin["code"].__class__.__name__ in [
                code.__name__ for code in SED_code.__subclasses__()
            ]
        y = self.__getattr__(
            y_name, SED_fit_params=y_origin, property_type="vals"
        )
        if incl_y_errs:
            y_err = self.__getattr__(
                y_name, SED_fit_params=y_origin, property_type="errs"
            )
            y_err = np.array([y_err[:, 0], y_err[:, 1]])
        else:
            y_err = None
        if type(y_origin) in [dict]:
            y_label = y_origin["code"].gal_property_fmt_dict[y_name]
        else:
            NotImplementedError
        if log_y or y_name in funcs.logged_properties:
            if incl_y_errs:
                y, y_err = funcs.errs_to_log(y, y_err)
            else:
                y = np.log10(y)
            y_name = f"log({y_name})"
            y_label = f"log({y_label})"

        if type(colour_by) == type(None):
            # plot all as a single colour
            pass
        else:
            if type(c_origin) in [dict]:
                assert "code" in c_origin.keys()
                assert c_origin["code"].__class__.__name__ in [
                    code.__name__ for code in SED_code.__subclasses__()
                ]
            c = getattr(
                self, colour_by, SED_fit_params=c_origin, property_type="vals"
            )
            if type(c_origin) in [dict]:
                cbar_label = c_origin["code"].gal_property_fmt_dict[colour_by]
            else:
                NotImplementedError
            if log_c or c in funcs.logged_properties:
                c = np.log10(c)
                colour_by = f"log({colour_by})"
                cbar_label = f"log({cbar_label})"

        # setup matplotlib figure/axis if not already given
        plt.style.use(
            f"{config['DEFAULT']['GALFIND_DIR']}/galfind_style.mplstyle"
        )
        if type(fig) == type(None) or type(ax) == type(None):
            fig, ax = plt.subplots()

        if "label" not in plot_kwargs.keys():
            plot_kwargs["label"] = "+".join(self.crops)

        if mean_err:
            # produce scatter plot
            if type(colour_by) == type(None):
                plot = ax.scatter(x, y, **plot_kwargs)
            else:
                if "cmap" not in plot_kwargs.keys():
                    plot_kwargs["cmap"] = cmap
                plot = ax.scatter(x, y, c=c, **plot_kwargs)
            if incl_x_errs and incl_y_errs:
                # plot the mean error
                pass
        else:
            # produce errorbar plot
            if "ls" not in plot_kwargs.keys():
                plot_kwargs["ls"] = ""
            if type(colour_by) == type(None):
                plot = ax.errorbar(x, y, xerr=x_err, yerr=y_err, **plot_kwargs)
            else:
                if "cmap" not in plot_kwargs.keys():
                    plot_kwargs["cmap"] = cmap
                plot = ax.errorbar(
                    x, y, xerr=x_err, yerr=y_err, c=c, **plot_kwargs
                )

        # sort plot aesthetics
        if annotate:
            plot_label = (
                f"{self.version}, {self.instrument.name}, {self.survey}"
            )
            ax.set_title(plot_label)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            if type(colour_by) != type(None):
                # make colourbar
                pass
            ax.legend(**legend_kwargs)

        if save:
            # determine origin_str
            origin_str = ""
            if type(x_origin) in [str]:
                origin_str += f"x={x_origin},"
            else:
                origin_str += f"x={x_origin['code'].label_from_SED_fit_params(x_origin)},"
            if type(y_origin) in [str]:
                origin_str += f"y={y_origin},"
            else:
                origin_str += (
                    f"y={y_origin['code'].label_from_SED_fit_params(y_origin)}"
                )
            if any(type(var) == type(None) for var in [colour_by, c_origin]):
                pass
            elif type(c_origin) in [str]:
                origin_str += f",c={c_origin}"
            else:  # dict
                origin_str += f",c={c_origin['code'].label_from_SED_fit_params(c_origin)}"

            # determine appropriate save path
            save_dir = f"{config['Other']['PLOT_DIR']}/{self.version}/{self.instrument.name}/{self.survey}/{origin_str}"
            if type(colour_by) == type(None):
                colour_label = f"_c={colour_by}"
            else:
                colour_label = ""
            save_name = f"{y_name}_vs_{x_name}{colour_label}"
            save_path = f"{save_dir}/{save_name}{save_type}"
            funcs.make_dirs(save_path)
            plt.savefig(save_path)

        if show:
            plt.show()

    # Selection functions

    def select_all_bands(self):
        return self.select_min_bands(len(self.instrument))

    def select_min_bands(self, min_bands):
        return self.perform_selection(Galaxy.select_min_bands, min_bands)

    # Masking selection

    def select_min_unmasked_bands(self, min_bands):
        return self.perform_selection(
            Galaxy.select_min_unmasked_bands, min_bands
        )

    #  already made these boolean columns in the catalogue
    def select_unmasked_bands(self, bands):
        return self.perform_selection(Galaxy.select_unmasked_band, bands)

    def select_unmasked_instrument(self, instrument_name):
        return self.perform_selection(
            Galaxy.select_unmasked_instrument, instrument_name
        )

    # Photometric galaxy property selection functions

    def select_phot_galaxy_property(
        self,
        property_name,
        gtr_or_less,
        property_lim,
        SED_fit_params={
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
    ):
        return self.perform_selection(
            Galaxy.select_phot_galaxy_property,
            property_name,
            gtr_or_less,
            property_lim,
            SED_fit_params,
        )

    def select_phot_galaxy_property_bin(
        self,
        property_name,
        property_lims,
        SED_fit_params={
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
    ):
        return self.perform_selection(
            Galaxy.select_phot_galaxy_property_bin,
            property_name,
            property_lims,
            SED_fit_params,
        )

    # SNR selection functions

    def phot_bluewards_Lya_non_detect(
        self,
        SNR_lim,
        SED_fit_params={
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
    ):
        return self.perform_selection(
            Galaxy.phot_bluewards_Lya_non_detect, SNR_lim, SED_fit_params
        )

    def phot_redwards_Lya_detect(
        self,
        SNR_lims,
        SED_fit_params={
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
        widebands_only=True,
    ):
        return self.perform_selection(
            Galaxy.phot_redwards_Lya_detect,
            SNR_lims,
            SED_fit_params,
            widebands_only,
        )

    def phot_Lya_band(
        self,
        SNR_lim,
        detect_or_non_detect="detect",
        SED_fit_params={
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
        widebands_only=True,
    ):
        return self.perform_selection(
            Galaxy.phot_Lya_band,
            SNR_lim,
            detect_or_non_detect,
            SED_fit_params,
            widebands_only,
        )

    def phot_SNR_crop(
        self, band_name_or_index, SNR_lim, detect_or_non_detect="detect"
    ):
        return self.perform_selection(
            Galaxy.phot_SNR_crop,
            band_name_or_index,
            SNR_lim,
            detect_or_non_detect,
        )

    # Emission line selection functions

    def select_rest_UV_line_emitters_dmag(
        self,
        emission_line_name,
        delta_m,
        rest_UV_wav_lims=[1_250.0, 3_000.0] * u.AA,
        medium_bands_only=True,
        SED_fit_params={
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
        update=True,
    ):
        return self.perform_selection(
            Galaxy.select_rest_UV_line_emitters_dmag,
            emission_line_name,
            delta_m,
            rest_UV_wav_lims,
            medium_bands_only,
            SED_fit_params,
        )

    def select_rest_UV_line_emitters_sigma(
        self,
        emission_line_name,
        sigma,
        rest_UV_wav_lims=[1_250.0, 3_000.0] * u.AA,
        medium_bands_only=True,
        SED_fit_params={
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
    ):
        return self.perform_selection(
            Galaxy.select_rest_UV_line_emitters_sigma,
            emission_line_name,
            sigma,
            rest_UV_wav_lims,
            medium_bands_only,
            SED_fit_params,
        )

    # Colour selection functions

    def select_colour(self, colour_bands, colour_val, bluer_or_redder):
        return self.perform_selection(
            Galaxy.select_colour, colour_bands, colour_val, bluer_or_redder
        )

    def select_colour_colour(self, colour_bands_arr, colour_select_func):
        return self.perform_selection(
            Galaxy.select_colour_colour, colour_bands_arr, colour_select_func
        )

    def select_UVJ(
        self,
        SED_fit_params={
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
        quiescent_or_star_forming="quiescent",
    ):
        return self.perform_selection(
            Galaxy.select_UVJ, SED_fit_params, quiescent_or_star_forming
        )

    def select_Kokorev24_LRDs(self):
        # only perform this selection if all relevant bands are present
        required_bands = ["F115W", "F150W", "F200W", "F277W", "F356W", "F444W"]
        if all(
            band_name in self.instrument.band_names
            for band_name in required_bands
        ):
            # red1 selection (z<6 LRDs)
            self.perform_selection(
                Galaxy.select_colour,
                ["F115W", "F150W"],
                0.8,
                "bluer",
                make_cat_copy=False,
            )
            self.perform_selection(
                Galaxy.select_colour,
                ["F200W", "F277W"],
                0.7,
                "redder",
                make_cat_copy=False,
            )
            self.perform_selection(
                Galaxy.select_colour,
                ["F200W", "F356W"],
                1.0,
                "redder",
                make_cat_copy=False,
            )
            # red2 selection (z>6 LRDs)
            self.perform_selection(
                Galaxy.select_colour,
                ["F150W", "F200W"],
                0.8,
                "bluer",
                make_cat_copy=False,
            )
            self.perform_selection(
                Galaxy.select_colour,
                ["F277W", "F356W"],
                0.6,
                "redder",
                make_cat_copy=False,
            )
            self.perform_selection(
                Galaxy.select_colour,
                ["F277W", "F444W"],
                0.7,
                "redder",
                make_cat_copy=False,
            )
            return self.perform_selection(Galaxy.select_Kokorev24_LRDs)
        else:
            galfind_logger.warning(
                f"Not all of {required_bands} in {self.instrument.band_names=}, skipping 'select_Kokorev24_LRDs' selection"
            )

    # Depth region selection

    def select_depth_region(self, band, region_ID, update=True):
        return NotImplementedError

    # Chi squared selection functions

    def select_chi_sq_lim(
        self,
        chi_sq_lim,
        SED_fit_params={
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
        reduced=True,
    ):
        return self.perform_selection(
            Galaxy.select_chi_sq_lim, chi_sq_lim, SED_fit_params, reduced
        )

    def select_chi_sq_diff(
        self,
        chi_sq_diff,
        SED_fit_params={
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
        delta_z_lowz=0.5,
    ):
        return self.perform_selection(
            Galaxy.select_chi_sq_diff,
            chi_sq_diff,
            SED_fit_params,
            delta_z_lowz,
        )

    # Redshift PDF selection functions

    def select_robust_zPDF(
        self,
        integral_lim,
        delta_z_over_z,
        SED_fit_params={
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
    ):
        return self.perform_selection(
            Galaxy.select_robust_zPDF,
            integral_lim,
            delta_z_over_z,
            SED_fit_params,
        )

    # Morphology selection functions

    def select_band_flux_radius(self, band, gtr_or_less, lim):
        assert band in self.instrument.band_names
        # load in effective radii as calculated from SExtractor
        self.load_band_properties_from_cat("FLUX_RADIUS", "sex_Re", None)
        return self.perform_selection(
            Galaxy.select_band_flux_radius, band, gtr_or_less, lim
        )

    # Full sample selection functions - these chain the above functions

    def select_EPOCHS(
        self,
        SED_fit_params={
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
        allow_lowz=False,
        hot_pixel_bands=["F277W", "F356W", "F444W"],
        mask_instruments=["NIRCam"],
    ):
        instruments_to_mask = [
            globals()[instr_name]()
            for instr_name in self.instrument.name.split("+")
            if instr_name in mask_instruments
        ]
        self.perform_selection(
            Galaxy.select_min_bands, 4.0, make_cat_copy=False
        )  # minimum 4 photometric bands
        [
            self.perform_selection(
                Galaxy.select_unmasked_instrument,
                instrument,
                make_cat_copy=False,
            )
            for instrument in instruments_to_mask
        ]  # all bands unmasked
        [
            self.perform_selection(
                Galaxy.select_band_flux_radius,
                band,
                "gtr",
                1.5,
                make_cat_copy=False,
            )
            for band in hot_pixel_bands
            if band in self.instrument.band_names
        ]  # LW NIRCam wideband Re>1.5 pix
        if not allow_lowz:
            self.perform_selection(
                Galaxy.phot_SNR_crop, 0, 2.0, "non_detect", make_cat_copy=False
            )  # 2σ non-detected in first band
        self.perform_selection(
            Galaxy.phot_bluewards_Lya_non_detect,
            2.0,
            SED_fit_params,
            make_cat_copy=False,
        )  # 2σ non-detected in all bands bluewards of Lyα
        self.perform_selection(
            Galaxy.phot_redwards_Lya_detect,
            [5.0, 5.0],
            SED_fit_params,
            True,
            make_cat_copy=False,
        )  # 5σ/3σ detected in first/second band redwards of Lyα
        self.perform_selection(
            Galaxy.phot_redwards_Lya_detect,
            2.0,
            SED_fit_params,
            False,
            make_cat_copy=False,
        )  # 2σ detected in all bands redwards of Lyα
        self.perform_selection(
            Galaxy.select_chi_sq_lim,
            3.0,
            SED_fit_params,
            True,
            make_cat_copy=False,
        )  # χ^2_red < 3
        self.perform_selection(
            Galaxy.select_chi_sq_diff,
            4.0,
            SED_fit_params,
            0.5,
            make_cat_copy=False,
        )  # Δχ^2 < 4 between redshift free and low redshift SED fits, with Δz=0.5 tolerance
        self.perform_selection(
            Galaxy.select_robust_zPDF,
            0.6,
            0.1,
            SED_fit_params,
            make_cat_copy=False,
        )  # 60% of redshift PDF must lie within z ± z * 0.1
        return self.perform_selection(
            Galaxy.select_EPOCHS,
            SED_fit_params,
            allow_lowz,
            hot_pixel_bands,
            instruments_to_mask,
        )

    def perform_selection(self, selection_function, *args, make_cat_copy=True):
        # extract selection name from galaxy method output
        selection_name = selection_function(self[0], *args, update=False)[1]
        # open catalogue
        # perform selection if not previously performed
        if selection_name not in self.selection_cols:
            # perform calculation for each galaxy and update galaxies in self
            [
                selection_function(gal, *args, update=True)[0]
                for gal in tqdm(
                    self, total=len(self), desc=f"Cropping {selection_name}"
                )
            ]
            # work out origin of property from arguments
            origin = "gal"
            # append .fits table in extension 1
            self._append_property_to_tab(selection_name, origin)
            # self._append_selection_to_fits(selection_name)
        if make_cat_copy:
            # crop catalogue by the selection
            cat_copy = self._crop_by_selection(selection_name)
            return cat_copy

    def _crop_by_selection(self, selection_name):
        # make a deep copy of the current catalogue object
        cat_copy = deepcopy(self)
        # crop deep copied catalogue to only the selected galaxies
        cat_copy.gals = cat_copy[getattr(self, selection_name)]
        if selection_name not in cat_copy.crops:
            # make a note of this crop if it is new
            cat_copy.crops.append(selection_name)
        return cat_copy

    # def _append_selection_to_fits(self, selection_name):
    #     # append .fits table if not already done so for this selection
    #     if selection_name not in self.selection_cols:
    #         assert(all(getattr(self, selection_name)))
    #         full_cat = self.open_cat()
    #         selection_cat = Table({"ID_temp": self.ID, selection_name: np.full(len(self), True)})
    #         output_cat = join(full_cat, selection_cat, keys_left = "NUMBER", keys_right = "ID_temp", join_type = "outer")
    #         output_cat.remove_column("ID_temp")
    #         # fill unselected columns with False rather than leaving as masked post-join
    #         output_cat[selection_name].fill_value = False
    #         output_cat = output_cat.filled()
    #         # ensure no rows are lost during this column append
    #         assert(len(output_cat) == len(full_cat))
    #         output_cat.meta = {**full_cat.meta, **{f"HIERARCH SELECTED_{selection_name}": True}}
    #         galfind_logger.info(f"Appending {selection_name} to {self.cat_path=}")
    #         output_cat.write(self.cat_path, overwrite = True)
    #         funcs.change_file_permissions(self.cat_path)
    #         self.selection_cols.append(selection_name)
    #     else:
    #         galfind_logger.info(f"Already appended {selection_name} to {self.cat_path=}")

    # %%
    # SED property functions

    # Rest-frame UV property calculation functions - these are not independent of each other

    # beta_phot tqdm bar not working appropriately!
    def calc_beta_phot(
        self,
        rest_UV_wav_lims=[1_250.0, 3_000.0] * u.AA,
        SED_fit_params={
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
        iters=10_000,
    ):
        self.calc_SED_rest_property(
            SED_rest_property_function=Photometry_rest.calc_beta_phot,
            iters=iters,
            SED_fit_params=SED_fit_params,
            rest_UV_wav_lims=rest_UV_wav_lims,
        )

    def calc_fesc_from_beta_phot(
        self,
        rest_UV_wav_lims=[1_250.0, 3_000.0] * u.AA,
        conv_author_year="Chisholm22",
        SED_fit_params={
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
        iters=10_000,
    ):
        self.calc_beta_phot(rest_UV_wav_lims, SED_fit_params, iters)
        self.calc_SED_rest_property(
            SED_rest_property_function=Photometry_rest.calc_fesc_from_beta_phot,
            iters=iters,
            SED_fit_params=SED_fit_params,
            rest_UV_wav_lims=rest_UV_wav_lims,
            conv_author_year=conv_author_year,
        )

    def calc_AUV_from_beta_phot(
        self,
        rest_UV_wav_lims=[1_250.0, 3_000.0] * u.AA,
        ref_wav=1_500.0 * u.AA,
        conv_author_year="M99",
        SED_fit_params={
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
        iters=10_000,
    ):
        self.calc_beta_phot(rest_UV_wav_lims, SED_fit_params, iters)
        self.calc_SED_rest_property(
            SED_rest_property_function=Photometry_rest.calc_AUV_from_beta_phot,
            iters=iters,
            SED_fit_params=SED_fit_params,
            rest_UV_wav_lims=rest_UV_wav_lims,
            ref_wav=ref_wav,
            conv_author_year=conv_author_year,
        )

    def calc_mUV_phot(
        self,
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        SED_fit_params: dict = {
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
        iters=10_000,
    ):
        self.calc_beta_phot(rest_UV_wav_lims, SED_fit_params, iters)
        self.calc_SED_rest_property(
            SED_rest_property_function=Photometry_rest.calc_mUV_phot,
            iters=iters,
            SED_fit_params=SED_fit_params,
            rest_UV_wav_lims=rest_UV_wav_lims,
            ref_wav=ref_wav,
        )

    def calc_MUV_phot(
        self,
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        SED_fit_params: dict = {
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
        iters=10_000,
    ):
        self.calc_mUV_phot(rest_UV_wav_lims, ref_wav, SED_fit_params, iters)
        self.calc_SED_rest_property(
            SED_rest_property_function=Photometry_rest.calc_MUV_phot,
            iters=iters,
            SED_fit_params=SED_fit_params,
            rest_UV_wav_lims=rest_UV_wav_lims,
            ref_wav=ref_wav,
        )

    def calc_LUV_phot(
        self,
        frame: str = "obs",
        rest_UV_wav_lims=[1_250.0, 3_000.0] * u.AA,
        ref_wav=1_500.0 * u.AA,
        AUV_beta_conv_author_year="M99",
        SED_fit_params={
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
        iters=10_000,
    ):
        if type(AUV_beta_conv_author_year) != type(None):
            self.calc_AUV_from_beta_phot(
                rest_UV_wav_lims,
                ref_wav,
                AUV_beta_conv_author_year,
                SED_fit_params,
                iters,
            )
        self.calc_mUV_phot(rest_UV_wav_lims, ref_wav, SED_fit_params, iters)
        self.calc_SED_rest_property(
            SED_rest_property_function=Photometry_rest.calc_LUV_phot,
            iters=iters,
            SED_fit_params=SED_fit_params,
            frame=frame,
            rest_UV_wav_lims=rest_UV_wav_lims,
            ref_wav=ref_wav,
            AUV_beta_conv_author_year=AUV_beta_conv_author_year,
        )

    def calc_SFR_UV_phot(
        self,
        frame: str = "obs",
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        AUV_beta_conv_author_year: Union[str, None] = "M99",
        kappa_UV_conv_author_year: str = "MD14",
        SED_fit_params: dict = {
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
        iters=10_000,
    ):
        self.calc_LUV_phot(
            frame,
            rest_UV_wav_lims,
            ref_wav,
            AUV_beta_conv_author_year,
            SED_fit_params,
            iters,
        )
        self.calc_SED_rest_property(
            SED_rest_property_function=Photometry_rest.calc_SFR_UV_phot,
            iters=iters,
            SED_fit_params=SED_fit_params,
            frame=frame,
            rest_UV_wav_lims=rest_UV_wav_lims,
            ref_wav=ref_wav,
            AUV_beta_conv_author_year=AUV_beta_conv_author_year,
            kappa_UV_conv_author_year=kappa_UV_conv_author_year,
        )

    def calc_rest_UV_properties(
        self,
        frame: str = "obs",
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        fesc_conv_author_year: Union[str, None] = "Chisholm22",
        AUV_beta_conv_author_year: Union[str, None] = "M99",
        kappa_UV_conv_author_year: str = "MD14",
        SED_fit_params: dict = {
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
        iters=10_000,
    ):
        if type(fesc_conv_author_year) != type(None):
            self.calc_fesc_from_beta_phot(
                rest_UV_wav_lims, fesc_conv_author_year, SED_fit_params, iters
            )
        self.calc_SFR_UV_phot(
            frame,
            rest_UV_wav_lims,
            ref_wav,
            AUV_beta_conv_author_year,
            kappa_UV_conv_author_year,
            SED_fit_params,
            iters,
        )

    # Emission line EWs from the rest frame UV photometry

    def calc_cont_rest_optical(
        self,
        strong_line_names: Union[str, list],
        rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
        SED_fit_params: dict = {
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
        iters: int = 10_000,
    ):
        self.calc_SED_rest_property(
            SED_rest_property_function=Photometry_rest.calc_cont_rest_optical,
            iters=iters,
            SED_fit_params=SED_fit_params,
            strong_line_names=strong_line_names,
            rest_optical_wavs=rest_optical_wavs,
        )

    def calc_EW_rest_optical(
        self,
        strong_line_names: Union[str, list],
        frame: str,
        rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
        SED_fit_params: dict = {
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
        iters: int = 10_000,
    ):
        self.calc_cont_rest_optical(
            strong_line_names, rest_optical_wavs, SED_fit_params, iters
        )
        self.calc_SED_rest_property(
            SED_rest_property_function=Photometry_rest.calc_EW_rest_optical,
            iters=iters,
            SED_fit_params=SED_fit_params,
            strong_line_names=strong_line_names,
            frame=frame,
            rest_optical_wavs=rest_optical_wavs,
        )

    def calc_dust_atten(
        self,
        calc_wav: u.Quantity,
        dust_author_year: Union[None, str] = "M99",
        dust_law: str = "C00",
        dust_origin: str = "UV",
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        SED_fit_params: dict = {
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
        iters: int = 10_000,
    ):
        assert all(
            type(name) != type(None) for name in [dust_law, dust_origin]
        )
        if type(dust_author_year) != type(None):
            self.calc_AUV_from_beta_phot(
                rest_UV_wav_lims,
                ref_wav,
                dust_author_year,
                SED_fit_params,
                iters,
            )
        self.calc_SED_rest_property(
            SED_rest_property_function=Photometry_rest.calc_dust_atten,
            iters=iters,
            SED_fit_params=SED_fit_params,
            calc_wav=calc_wav,
            dust_author_year=dust_author_year,
            dust_law=dust_law,
            dust_origin=dust_origin,
            rest_UV_wav_lims=rest_UV_wav_lims,
            ref_wav=ref_wav,
        )

    def calc_line_flux_rest_optical(
        self,
        strong_line_names: Union[str, list],
        frame: str,
        dust_author_year="M99",
        dust_law="C00",
        dust_origin="UV",
        rest_optical_wavs=[4_200.0, 10_000.0] * u.AA,
        rest_UV_wav_lims=[1_250.0, 3_000.0] * u.AA,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        SED_fit_params: dict = {
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
        iters: int = 10_000,
    ):
        self.calc_EW_rest_optical(
            strong_line_names, frame, rest_optical_wavs, SED_fit_params, iters
        )
        if all(
            type(name) != type(None)
            for name in [dust_author_year, dust_law, dust_origin]
        ):
            self.calc_dust_atten(
                line_diagnostics[strong_line_names[0]]["line_wav"],
                dust_author_year,
                dust_law,
                dust_origin,
                rest_UV_wav_lims,
                ref_wav,
                SED_fit_params,
                iters,
            )
        self.calc_SED_rest_property(
            SED_rest_property_function=Photometry_rest.calc_line_flux_rest_optical,
            iters=iters,
            SED_fit_params=SED_fit_params,
            frame=frame,
            strong_line_names=strong_line_names,
            dust_author_year=dust_author_year,
            dust_law=dust_law,
            dust_origin=dust_origin,
            rest_optical_wavs=rest_optical_wavs,
            rest_UV_wav_lims=rest_UV_wav_lims,
            ref_wav=ref_wav,
        )

    def calc_line_lum_rest_optical(
        self,
        strong_line_names: Union[str, list],
        frame: str,
        dust_author_year: Union[str, None] = "M99",
        dust_law: str = "C00",
        dust_origin: str = "UV",
        rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        SED_fit_params: dict = {
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
        iters: int = 10_000,
    ):
        self.calc_line_flux_rest_optical(
            strong_line_names,
            frame,
            dust_author_year,
            dust_law,
            dust_origin,
            rest_optical_wavs,
            rest_UV_wav_lims,
            ref_wav,
            SED_fit_params,
            iters,
        )
        self.calc_SED_rest_property(
            SED_rest_property_function=Photometry_rest.calc_line_lum_rest_optical,
            iters=iters,
            SED_fit_params=SED_fit_params,
            strong_line_names=strong_line_names,
            frame=frame,
            dust_author_year=dust_author_year,
            dust_law=dust_law,
            dust_origin=dust_origin,
            rest_optical_wavs=rest_optical_wavs,
            rest_UV_wav_lims=rest_UV_wav_lims,
            ref_wav=ref_wav,
        )

    # should be generalized slightly more
    def calc_xi_ion(
        self,
        frame: str = "rest",
        strong_line_names: Union[str, list] = ["Halpha"],
        fesc_author_year: str = "fesc=0.0",
        dust_author_year: Union[str, None] = "M99",
        dust_law: str = "C00",
        dust_origin: str = "UV",
        rest_optical_wavs=[4_200.0, 10_000.0] * u.AA,
        rest_UV_wav_lims=[1_250.0, 3_000.0] * u.AA,
        ref_wav=1_500.0 * u.AA,
        SED_fit_params={
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
        iters=10_000,
    ):
        self.calc_line_lum_rest_optical(
            strong_line_names,
            frame,
            dust_author_year,
            dust_law,
            dust_origin,
            rest_optical_wavs,
            rest_UV_wav_lims,
            ref_wav,
            SED_fit_params,
            iters,
        )
        if "fesc" not in fesc_author_year:
            self.calc_SED_rest_property(
                SED_rest_property_function=Photometry_rest.calc_fesc_from_beta_phot,
                iters=iters,
                SED_fit_params=SED_fit_params,
                rest_UV_wav_lims=rest_UV_wav_lims,
                fesc_author_year=fesc_author_year,
            )
        self.calc_SED_rest_property(
            SED_rest_property_function=Photometry_rest.calc_xi_ion,
            iters=iters,
            SED_fit_params=SED_fit_params,
            frame=frame,
            strong_line_names=strong_line_names,
            fesc_author_year=fesc_author_year,
            dust_author_year=dust_author_year,
            dust_law=dust_law,
            dust_origin=dust_origin,
            rest_optical_wavs=rest_optical_wavs,
            rest_UV_wav_lims=rest_UV_wav_lims,
            ref_wav=ref_wav,
        )

    # Global SED rest-frame photometry calculations

    def calc_SED_rest_property(
        self,
        SED_rest_property_function,
        iters,
        SED_fit_params={
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
        **kwargs,
    ):
        key = SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)
        property_name = SED_rest_property_function(
            self[0].phot.SED_results[key].phot_rest,
            **kwargs,
            extract_property_name=True,
        )
        # self.SED_rest_properties should contain the selections these properties have been calculated for
        if key not in self.SED_rest_properties.keys():
            self.SED_rest_properties[key] = []

        PDF_dir = f"{config['PhotProperties']['PDF_SAVE_DIR']}/{self.version}/{self.instrument.name}/{self.survey}"
        # perform calculation for each galaxy and update galaxies in self
        if type(property_name) in [str]:
            property_name = [property_name]
        for name in property_name:
            self.gals = [
                deepcopy(gal)._calc_SED_rest_property(
                    SED_rest_property_function=SED_rest_property_function,
                    SED_fit_params_label=key,
                    save_dir=PDF_dir,
                    iters=iters,
                    **kwargs,
                )
                for gal in tqdm(
                    self, total=len(self), desc=f"Calculating {name}"
                )
            ]
            galfind_logger.info(f"Calculated {name}")
            self._append_SED_rest_property_to_fits(name, key)

    # def _save_SED_rest_PDFs(self, property_name, save_dir, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
    #     [gal._save_SED_rest_PDFs(property_name, save_dir, SED_fit_params) for gal in self]

    def _append_SED_rest_property_to_fits(
        self,
        property_name: str,
        SED_fit_params_label: str,
        save_kwargs: bool = True,
        type_fill_vals: dict = {int: -99, float: None, str: ""},
    ):
        try:
            SED_rest_property_tab = self.open_cat(
                cropped=False, hdu=SED_fit_params_label
            )
        except FileNotFoundError:
            SED_rest_property_tab = None
        # obtain full list of catalogue IDs
        fits_tab = self.open_cat(cropped=False)
        IDs = np.array(fits_tab[self.cat_creator.ID_label]).astype(int)
        if type(SED_rest_property_tab) == type(None):
            SED_rest_property_tab = Table(
                {self.cat_creator.ID_label: IDs}, dtype=[int]
            )
        # if the table does not include the required column names, instantiate blank columns
        if property_name not in SED_rest_property_tab.colnames:
            blank_floats = np.full(
                len(SED_rest_property_tab), type_fill_vals[float]
            )
            new_colname_tab = Table(
                {
                    f"{self.cat_creator.ID_label}_temp": IDs,
                    property_name: blank_floats,
                    f"{property_name}_l1": blank_floats,
                    f"{property_name}_u1": blank_floats,
                },
                dtype=[int] + [float] * 3,
            )
            SED_rest_property_tab = join(
                SED_rest_property_tab,
                new_colname_tab,
                keys_left=self.cat_creator.ID_label,
                keys_right=f"{self.cat_creator.ID_label}_temp",
                join_type="inner",
            )
            SED_rest_property_tab.remove_column(
                f"{self.cat_creator.ID_label}_temp"
            )
            new_cols = True
        else:
            new_cols = False
        # extract names of properties that have been recently updated
        if new_cols:  # all columns that havn't previously existed are updates
            is_property_updated = np.full(len(self), True)
        else:
            is_property_updated = self.__getattr__(
                property_name,
                phot_type="rest",
                property_type="recently_updated",
            )
        if type(is_property_updated) == type(None):
            # breakpoint()
            pass
        else:
            if any(
                type(updated) == type(None) for updated in is_property_updated
            ):
                # breakpoint()
                pass
        # update properties and kwargs for those galaxies that have been updated, or if the columns have just been made

        if is_property_updated is not None:
            if any(updated for updated in is_property_updated):
                # extract the kwargs for this property
                calculated_property_PDFs = self.__getattr__(
                    property_name, phot_type="rest", property_type="PDFs"
                )[is_property_updated]
                kwarg_names = np.unique(
                    np.hstack(
                        [
                            list(property_PDF.kwargs.keys())
                            for property_PDF in calculated_property_PDFs
                            if type(property_PDF) != type(None)
                        ]
                    )
                )
                kwarg_types_arr = [
                    [
                        type(property_PDF.kwargs[kwarg_name])
                        for property_PDF in calculated_property_PDFs
                        if type(property_PDF) != type(None)
                    ]
                    for kwarg_name in kwarg_names
                ]
                for kwarg_types in kwarg_types_arr:
                    assert all(
                        types == kwarg_types[0] for types in kwarg_types
                    )
                kwarg_types = [
                    kwarg_types[0] for kwarg_types in kwarg_types_arr
                ]
                # make new columns for any kwarg names that have not previously been created
                for kwarg_name, kwarg_type in zip(kwarg_names, kwarg_types):
                    assert kwarg_types[0] in type_fill_vals.keys()
                    if kwarg_name not in SED_rest_property_tab.colnames:
                        blank_col = np.full(
                            len(SED_rest_property_tab),
                            type_fill_vals[kwarg_type],
                        )
                        new_colname_tab = Table(
                            {
                                f"{self.cat_creator.ID_label}_temp": IDs,
                                kwarg_name: blank_col,
                            },
                            dtype=[int] + [kwarg_type],
                        )
                        SED_rest_property_tab = join(
                            SED_rest_property_tab,
                            new_colname_tab,
                            keys_left=self.cat_creator.ID_label,
                            keys_right=f"{self.cat_creator.ID_label}_temp",
                            join_type="outer",
                        )
                        SED_rest_property_tab.remove_column(
                            f"{self.cat_creator.ID_label}_temp"
                        )
                # create new columns of properties
                calculated_IDs = np.array(self.__getattr__("ID")).astype(int)[
                    is_property_updated
                ]
                non_calculated_IDs = np.array(
                    [ID for ID in IDs if ID not in calculated_IDs]
                ).astype(int)
                new_IDs = np.concatenate((calculated_IDs, non_calculated_IDs))
                calculated_properties = self.__getattr__(
                    property_name, phot_type="rest", property_type="vals"
                )[is_property_updated]
                # slice old catalogue to just those IDs which have not been updated
                old_SED_rest_property_tab = SED_rest_property_tab[
                    np.array(
                        [
                            True if ID in non_calculated_IDs else False
                            for ID in SED_rest_property_tab[
                                self.cat_creator.ID_label
                            ]
                        ]
                    )
                ]
                new_properties = np.concatenate(
                    (
                        calculated_properties,
                        np.array(
                            old_SED_rest_property_tab[property_name]
                        ).astype(float),
                    )
                )
                calculated_property_errs = self.__getattr__(
                    property_name, phot_type="rest", property_type="errs"
                )
                new_property_l1 = np.concatenate(
                    (
                        np.array(calculated_property_errs[:, 0])[
                            is_property_updated
                        ],
                        np.array(
                            old_SED_rest_property_tab[f"{property_name}_l1"]
                        ).astype(float),
                    )
                )
                new_property_u1 = np.concatenate(
                    (
                        np.array(calculated_property_errs[:, 1])[
                            is_property_updated
                        ],
                        np.array(
                            old_SED_rest_property_tab[f"{property_name}_u1"]
                        ).astype(float),
                    )
                )
                # create new columns of kwargs
                new_kwargs = {
                    kwarg_name: np.concatenate(
                        (
                            np.array(
                                [
                                    property_PDF.kwargs[kwarg_name]
                                    if type(property_PDF) != type(None)
                                    else type_fill_vals[kwarg_type]
                                    for property_PDF in calculated_property_PDFs
                                ]
                            ),
                            np.full(
                                len(non_calculated_IDs),
                                type_fill_vals[kwarg_type],
                            ),
                        )
                    )
                    for kwarg_name, kwarg_type in zip(kwarg_names, kwarg_types)
                }
                # make new table of the same length as the global .fits catalogue to be joined
                new_tab = Table(
                    {
                        **{
                            f"{self.cat_creator.ID_label}_temp": new_IDs,
                            property_name: new_properties,
                            f"{property_name}_l1": new_property_l1,
                            f"{property_name}_u1": new_property_u1,
                        },
                        **new_kwargs,
                    },
                    dtype=[int] + [float] * 3 + kwarg_types,
                )
                # update .fits table
                # remove old columns before appending the newer ones
                for name in [
                    property_name,
                    f"{property_name}_l1",
                    f"{property_name}_u1",
                ] + list(new_kwargs.keys()):
                    SED_rest_property_tab.remove_column(name)
                SED_rest_property_tab = join(
                    SED_rest_property_tab,
                    new_tab,
                    keys_left=self.cat_creator.ID_label,
                    keys_right=f"{self.cat_creator.ID_label}_temp",
                    join_type="outer",
                )
                SED_rest_property_tab.remove_column(
                    f"{self.cat_creator.ID_label}_temp"
                )
                SED_rest_property_tab.sort(self.cat_creator.ID_label)
                self.write_cat(
                    [fits_tab, SED_rest_property_tab],
                    ["OBJECTS", SED_fit_params_label],
                )

    def load_SED_rest_properties(
        self,
        SED_fit_params={
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
        timed=True,
    ):
        key = SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)
        PDF_dir = f"{config['PhotProperties']['PDF_SAVE_DIR']}/{self.version}/{self.instrument.name}/{self.survey}/{key}"
        property_paths = glob.glob(f"{PDF_dir}/*")
        if len(property_paths) != 0:
            property_names = [
                property_path.split("/")[-1]
                for property_path in property_paths
            ]
            self.gals = [
                deepcopy(gal)._load_SED_rest_properties(
                    PDF_dir, property_names, key
                )
                for gal in tqdm(
                    deepcopy(self),
                    desc=f"Loading SED rest properties for {key}",
                    total=len(self),
                )
            ]
            for name in property_names:
                self._append_SED_rest_property_to_fits(name, key)

    def del_SED_rest_property(
        self,
        property_name,
        SED_fit_params={
            "code": EAZY(),
            "templates": "fsps_larson",
            "lowz_zmax": None,
        },
    ):
        key = SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)
        # SED rest property must exist for this sample
        assert property_name in self.SED_rest_properties[key]
        # delete data from fits
        del_col_names = [
            property_name,
            f"{property_name}_l1",
            f"{property_name}_u1",
        ]
        del_hdr_names = [f"SED_REST_{property_name}"]
        self.del_cols_hdrs_from_fits(del_col_names, del_hdr_names, key)
        # check whether the SED rest property kwargs are included in the catalogue, and if so delete these as well - Not Implemented Yet!

        # remove data from self, starting with catalogue, then gal for gal in self.gals
        self.SED_rest_properties[key].remove(property_name)
        self.gals = [
            deepcopy(gal)._del_SED_rest_properties([property_name], key)
            for gal in self
        ]

    # Number Density Function (e.g. UVLF and mass functions) methods

    def calc_Vmax(
        self,
        data_arr: Union[list, np.array],
        z_bin: Union[list, np.array],
        SED_fit_params: Union[dict, str] = "EAZY_fsps_larson_zfree",
        z_step: float = 0.01,
        timed: bool = False,
    ) -> None:
        assert len(z_bin) == 2
        assert z_bin[0] < z_bin[1]
        if type(SED_fit_params) == dict:
            SED_fit_params_key = SED_fit_params[
                "code"
            ].label_from_SED_fit_params(SED_fit_params)
        elif type(SED_fit_params) == str:
            SED_fit_params_key = SED_fit_params
        else:
            galfind_logger.critical(
                f"{SED_fit_params=} with {type(SED_fit_params)=} is not in [dict, str]!"
            )
        z_bin_name = f"{SED_fit_params_key}_{z_bin[0]:.1f}<z<{z_bin[1]:.1f}"
        for data in data_arr:
            save_path = f"{config['NumberDensityFunctions']['VMAX_DIR']}/{self.version}/{self.instrument.name}/{self.survey}/{z_bin_name}/Vmax_field={data.full_name}.ecsv"
            funcs.make_dirs(save_path)
            # if this file already exists
            if Path(save_path).is_file():
                # open file
                old_tab = Table.read(save_path)
                update_IDs = np.array(
                    [gal.ID for gal in self if gal.ID not in old_tab["ID"]]
                )
            else:
                update_IDs = self.ID
            if len(update_IDs) > 0:
                self.gals = [
                    deepcopy(gal).calc_Vmax(
                        self.data.full_name,
                        [data],
                        z_bin,
                        SED_fit_params_key,
                        z_step,
                        timed=timed,
                    )
                    for gal in tqdm(
                        self,
                        total=len(self),
                        desc=f"Calculating Vmax's for {self.data.full_name} in {z_bin_name} {data.full_name}",
                    )
                ]
                # table with uncalculated Vmax's
                Vmax_arr = np.array(
                    [
                        gal.V_max[z_bin_name][data.full_name]
                        .to(u.Mpc**3)
                        .value
                        if type(gal.V_max[z_bin_name][data.full_name])
                        in [u.Quantity]
                        else gal.V_max[z_bin_name][data.full_name]
                        for gal in self
                        if gal.ID in update_IDs
                    ]
                )
                # Vmax_simple_arr = np.array([gal.V_max_simple[z_bin_name][data.full_name].to(u.Mpc ** 3).value \
                #    if type(gal.V_max_simple[z_bin_name][data.full_name]) in [u.Quantity] else \
                #    gal.V_max_simple[z_bin_name][data.full_name] for gal in self if gal.ID in update_IDs])
                obs_zmin = np.array(
                    [
                        gal.obs_zrange[z_bin_name][data.full_name][0]
                        for gal in self
                        if gal.ID in update_IDs
                    ]
                )
                obs_zmax = np.array(
                    [
                        gal.obs_zrange[z_bin_name][data.full_name][1]
                        for gal in self
                        if gal.ID in update_IDs
                    ]
                )
                # new_tab = Table({"ID": update_IDs, "Vmax": Vmax_arr, "Vmax_simple": Vmax_simple_arr, \
                #     "obs_zmin": obs_zmin, "obs_zmax": obs_zmax}, dtype = [int, float, float, float, float])
                new_tab = Table(
                    {
                        "ID": update_IDs,
                        "Vmax": Vmax_arr,
                        "obs_zmin": obs_zmin,
                        "obs_zmax": obs_zmax,
                    },
                    dtype=[int, float, float, float],
                )
                new_tab.meta = {
                    "Vmax_invalid_val": -1.0,
                    "Vmax_unit": u.Mpc**3,
                }
                if Path(save_path).is_file():  # update and save table
                    out_tab = vstack([old_tab, new_tab])
                    out_tab.meta = {**old_tab.meta, **new_tab.meta}
                else:  # save table
                    out_tab = new_tab
                out_tab.sort("ID")
                out_tab.write(save_path, overwrite=True)
            else:  # Vmax table already opened
                Vmax_tab = old_tab[
                    np.array([row["ID"] in self.ID for row in old_tab])
                ]
                Vmax_tab.sort("ID")
                # save appropriate Vmax properties
                self.gals = [
                    deepcopy(gal).save_Vmax(
                        Vmax, z_bin_name, data.full_name, is_simple_Vmax=False
                    )
                    for gal, Vmax in zip(self, np.array(Vmax_tab["Vmax"]))
                ]
                # self.gals = [deepcopy(gal).save_Vmax(Vmax, z_bin_name, data.full_name, is_simple_Vmax = True) \
                #    for gal, Vmax in zip(self, np.array(Vmax_tab["Vmax_simple"]))]

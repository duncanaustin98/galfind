#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:27:47 2023

@author: austind
"""

from __future__ import annotations

import glob
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, join, vstack, MaskedColumn
from astropy.wcs import WCS
import itertools
from astropy.utils.masked import Masked
from tqdm import tqdm
from typing import Union, Tuple, Any, List, Dict, Callable, Optional, NoReturn, TYPE_CHECKING

if TYPE_CHECKING:
    from . import SED_code
    from . import Band_Data_Base
    from . import Multiple_Filter, Data

try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import (
    EAZY,  # noqa F501
    Catalogue_Base,
    Photometry_rest,
    Photometry_obs,
    config,
    galfind_logger,
)
from . import useful_funcs_austind as funcs
from .Cutout import Multiple_Band_Cutout, Multiple_RGB, Stacked_RGB
from .Data import Data
from .EAZY import EAZY
from .Emission_lines import line_diagnostics
from .Galaxy import Galaxy
from .Spectrum import Spectral_Catalogue


def load_IDs_Table(
    cat: Table, 
    ID_label: str, 
    **kwargs
) -> List[int]:
    return list(cat[ID_label])

def load_skycoords_Table(
    cat: Table, 
    skycoords_labels: Dict[str, str], 
    skycoords_units: Dict[str, u.Unit], 
    **kwargs
) -> SkyCoord:
    ra = cat[skycoords_labels["RA"]]
    dec = cat[skycoords_labels["DEC"]]
    ra_unit = skycoords_units["RA"]
    dec_unit = skycoords_units["DEC"]
    return SkyCoord(ra=ra, dec=dec, unit=(ra_unit, dec_unit))

# def load_phot_Table(
#     cat: Table, 
#     phot_labels: List[str], 
#     err_labels: List[str], 
#     **kwargs
# ) -> Tuple[np.ndarray, np.ndarray]:
#     assert "ZP" in kwargs.keys(), \
#         galfind_logger.critical("ZP not in kwargs!")
#     phot_cat = load_cols_Table(cat, phot_labels, **kwargs)
#     err_cat = load_cols_Table(cat, err_labels, **kwargs)
#     phot = funcs.flux_image_to_Jy(phot_cat, kwargs["ZP"])
#     phot_err = funcs.flux_image_to_Jy(err_cat, kwargs["ZP"])
#     return phot, phot_err

def phot_property_from_galfind_tab(
    cat: Table,
    labels: Dict[u.Quantity, List[str]],
    **kwargs
) -> np.ndarray:
    aper_diams = [label.value for label in labels.keys()] * list(labels.keys())[0].unit
    if "cat_aper_diams" not in kwargs.keys():
        galfind_logger.warning(
            f"cat_aper_diams not in {kwargs.keys()=}! " + \
            f"Setting to {aper_diams=}"
        )
        kwargs["cat_aper_diams"] = aper_diams
    else:
        assert isinstance(kwargs["cat_aper_diams"], u.Quantity), \
            galfind_logger.critical(f"{type(kwargs['cat_aper_diams'])=} != u.Quantity!")
        assert isinstance(kwargs["cat_aper_diams"].value, (list, np.ndarray)), \
            galfind_logger.critical(f"{type(kwargs['cat_aper_diams'])=} != list!")
    # load aperture diameter indices
    aper_diam_indices = [i for i, aper_diam in enumerate(
        kwargs["cat_aper_diams"]) if aper_diam in aper_diams]
    assert len(aper_diam_indices) > 0, \
        galfind_logger.critical("len(aper_diam_indices) <= 0")
    # ensure labels are formatted properly
    assert all(label == labels[aper_diams[0]] for label in labels.values()), \
        galfind_logger.critical("All phot_labels not equal!")
    properties = {}
    _properties = funcs.fits_cat_to_np(cat, labels[aper_diams[0]])
    for aper_diam_index in aper_diam_indices:
        aper_diam = kwargs["cat_aper_diams"][aper_diam_index]
        properties[aper_diam] = _properties[:, :, aper_diam_index]
    return properties

def load_galfind_phot(
    cat: Table,
    phot_labels: Dict[u.Quantity, List[str]],
    err_labels: Dict[u.Quantity, List[str]],
    **kwargs
) -> Tuple[Dict[u.Quantity, np.ndarray], Dict[u.Quantity, np.ndarray]]:
    assert phot_labels.keys() == err_labels.keys(), \
        galfind_logger.critical(f"{phot_labels.keys()=} != {err_labels.keys()=}!")
    assert "ZP" in kwargs.keys(), \
        galfind_logger.critical("ZP not in kwargs!")
    phot = {aper_diam: funcs.flux_image_to_Jy(_phot, kwargs["ZP"]) for aper_diam, _phot \
        in phot_property_from_galfind_tab(cat, phot_labels, **kwargs).items()}
    phot_err = {aper_diam: funcs.flux_image_to_Jy(_phot_err, kwargs["ZP"]) for aper_diam, _phot_err \
        in phot_property_from_galfind_tab(cat, err_labels, **kwargs).items()}
    return phot, phot_err

def load_galfind_mask(
    cat: Table,
    mask_labels: List[str], 
    **kwargs
) -> np.ndarray:
    mask = np.invert(
        funcs.fits_cat_to_np(
            cat, mask_labels, reshape_by_aper_diams=False
        )
    )
    return mask

def load_galfind_depths(
    cat: Table,
    depth_labels: Dict[u.Quantity, List[str]],
    **kwargs
) -> Dict[u.Quantity, np.ndarray]:
    return {aper_diam: depth * u.ABmag for aper_diam, depth in \
        phot_property_from_galfind_tab(cat, depth_labels, **kwargs).items()}

# def load_cols_Table(
#     cat: Table, 
#     labels: List[str], 
#     **kwargs
# ) -> np.ndarray:
#     return funcs.fits_cat_to_np(cat, labels)

def check_hdu_exists(cat_path: str, hdu: str) -> bool:
    # check whether the hdu extension exists
    hdul = fits.open(cat_path)
    return any(hdu_.name == hdu.upper() for hdu_ in hdul)

def open_galfind_cat(cat_path: str, cat_type: str) -> Optional[Table]:
    if cat_type in ["ID", "sky_coord", "phot", "mask", "depths"]:
        tab = Table.read(
            cat_path, character_as_bytes=False, memmap=True
        )
    elif check_hdu_exists(cat_path, cat_type):
        tab = Table.read(
            cat_path, character_as_bytes=False, memmap=True, hdu=cat_type
        )
    else:
        err_message = f"cat_type = {cat_type=} not in " + \
            "['ID', 'sky_coord', 'phot', 'mask', 'depths'] and " + \
            f"not a valid HDU extension in {cat_path}!"
        galfind_logger.warning(err_message)
        return None
        #raise Exception(err_message)
    return tab

def open_galfind_hdr(cat_path: str, cat_type: str) -> Dict[str, str]:
    return open_galfind_cat(cat_path, cat_type).meta

def galfind_phot_labels(
    filterset: Multiple_Filter, 
    aper_diams: u.Quantity, 
    **kwargs
) -> Tuple[Dict[str, str], Dict[str, str]]:
    assert "min_flux_pc_err" in kwargs.keys(), \
        galfind_logger.critical("min_flux_pc_err not in kwargs!")
    phot_labels = {aper_diam * aper_diams.unit: [f"FLUX_APER_{filt_name}_aper_corr_Jy" for filt_name in filterset.band_names] for aper_diam in aper_diams.value}
    err_labels = {aper_diam * aper_diams.unit: [f"FLUXERR_APER_{filt_name}_loc_depth_{str(int(kwargs['min_flux_pc_err']))}pc_Jy" for filt_name in filterset.band_names] for aper_diam in aper_diams.value}
    return phot_labels, err_labels

def galfind_mask_labels(
    filterset: Multiple_Filter, 
    **kwargs
) -> Dict[str, str]:
    return [f"unmasked_{filt_name}" for filt_name in filterset.band_names]

def galfind_depth_labels(
    filterset: Multiple_Filter, 
    aper_diams: u.Quantity, 
    **kwargs
) -> Dict[str, str]:
    return {aper_diam * aper_diams.unit: [f"loc_depth_{filt_name}" \
        for filt_name in filterset.band_names] \
        for aper_diam in aper_diams.value}

def load_bool_Table(
    tab: Table,
    select_names: List[str], 
    **kwargs
) -> Dict[str, List[bool]]:
    return {name: list(tab[name]) for name in select_names}

def galfind_selection_labels(
    tab: Table,
    **kwargs
) -> List[str]:
    # load selection names dict from header
    return [name for name in tab.colnames if name not in "NUMBER"]


class Catalogue_Creator:

    def __init__(self,
        survey: str,
        version: str, 
        cat_path: str,
        filterset: Multiple_Filter,
        aper_diams: u.Quantity,
        crops: Optional[Dict[str, Any], str, int, List[str], List[int]] = None,
        open_cat: Callable[[str, str], Any] = open_galfind_cat,
        open_hdr: Callable[[Any], Dict[str, str]] = open_galfind_hdr,
        load_ID_func: Optional[Callable] = load_IDs_Table,
        ID_label: str = "NUMBER",
        load_ID_kwargs: Dict[str, Any] = {},
        load_skycoords_func: Optional[Callable] = load_skycoords_Table,
        skycoords_labels: Dict[str, str] = {"RA": "ALPHA_J2000", "DEC": "DELTA_J2000"},
        skycoords_units: Dict[str, u.Unit] = {"RA": u.deg, "DEC": u.deg},
        load_skycoords_kwargs: Dict[str, Any] = {},
        load_phot_func: Callable = load_galfind_phot,
        get_phot_labels: Callable[[Multiple_Filter], Dict[str, str]] = galfind_phot_labels,
        load_phot_kwargs: Dict[str, Any] = {"ZP": u.Jy.to(u.ABmag), "min_flux_pc_err": 10.},
        load_mask_func: Optional[Callable] = load_galfind_mask,
        get_mask_labels: Callable[[Multiple_Filter], Dict[str, str]] = galfind_mask_labels,
        load_mask_kwargs: Dict[str, Any] = {},
        load_depths_func: Optional[Callable] = load_galfind_depths,
        get_depth_labels: Callable[[Multiple_Filter], Dict[str, str]] = galfind_depth_labels,
        load_depths_kwargs: Dict[str, Any] = {},
        load_selection_func: Optional[Callable[[], Dict[u.Quantity, Dict[str, List[Any]]]]] = load_bool_Table,
        get_selection_labels: Callable[[Table, List[str]], List[str]] = galfind_selection_labels,
        load_selection_kwargs: Dict[str, Any] = {},
        load_SED_result_func: Optional[Callable] = None,
    ):
        self.survey = survey
        self.version = version
        self.cat_path = cat_path
        self.filterset = filterset
        assert isinstance(aper_diams, u.Quantity), \
            galfind_logger.critical(f"{type(aper_diams)=} != u.Quantity!")
        assert isinstance(aper_diams.value, (list, np.ndarray)), \
            galfind_logger.critical(f"{type(aper_diams.value)=} != list!")
        self.aper_diams = aper_diams
        self.open_cat = open_cat
        self.open_hdr = open_hdr
        self.ID_label = ID_label
        self.skycoords_labels = skycoords_labels
        self.load_ID_func = load_ID_func
        self.load_ID_kwargs = load_ID_kwargs
        self.load_skycoords_func = load_skycoords_func
        self.skycoords_units = skycoords_units
        self.load_skycoords_kwargs = load_skycoords_kwargs
        self.load_phot_func = load_phot_func
        self.get_phot_labels = get_phot_labels
        self.load_phot_kwargs = load_phot_kwargs
        self.load_mask_func = load_mask_func
        self.get_mask_labels = get_mask_labels
        self.load_mask_kwargs = load_mask_kwargs
        self.load_depth_func = load_depths_func
        self.get_depth_labels = get_depth_labels
        self.load_depth_kwargs = load_depths_kwargs
        self.load_selection_func = load_selection_func
        self.get_selection_labels = get_selection_labels
        self.load_selection_kwargs = load_selection_kwargs
        self.load_SED_result_func = load_SED_result_func

        self.set_crops(crops)
        self.make_gal_instr_mask()
        # ensure survey/version is correct
        # in primary header which stores the IDs
        hdr = self.open_hdr(self.cat_path, "ID")
        if "SURVEY" in hdr.keys():
            assert hdr["SURVEY"] == self.survey, \
                galfind_logger.critical(
                    f"{hdr['SURVEY']=} != {self.survey=}"
                )
        if "VERSION" in hdr.keys():
            assert hdr["VERSION"] == self.version, \
                galfind_logger.critical(
                    f"{hdr['VERSION']=} != {self.version=}"
                )
    
    @classmethod
    def from_data(
        cls, 
        data: Data, 
        crops: Optional[Dict[str, Any], str, int, List[str], List[int]] = None,
    ) -> Catalogue_Creator:
        cat_creator = cls(
            data.survey,
            data.version,
            data.phot_cat_path,
            data.filterset,
            data.aper_diams,
            crops = crops,
        )
        cat_creator.data = data
        return cat_creator

    def __call__(self, cropped: bool = False) -> Catalogue:
        galfind_logger.info(
            f"Making {self.survey} {self.version} {self.cat_name} catalogue!"
        )
        galfind_logger.debug(
            f"Loading {self.survey} {self.version} {self.cat_name} photometry!"
        )
        # make array of Photometry_obs for each aperture diameter
        IDs = self.load_IDs(cropped)
        sky_coords = self.load_skycoords(cropped)
        phot, phot_err = self.load_phot(cropped)
        depths = self.load_depths(cropped)
        selection_flags = self.load_selection_flags(cropped)
        filterset_arr = self.load_gal_filtersets(cropped)
        SED_results = {}
        phot_obs_arr = [{aper_diam: Photometry_obs(filterset_arr[i], \
            phot[aper_diam][i], phot_err[aper_diam][i], depths[aper_diam][i], \
            aper_diam, SED_results=SED_results) for aper_diam in self.aper_diams} \
            for i in range(len(filterset_arr))]
        assert len(IDs) == len(sky_coords) == len(phot_obs_arr), \
            galfind_logger.critical(
                f"{len(IDs)=} != {len(sky_coords)=} != {len(phot_obs_arr)=}!"
            )
        # make an array of galaxy objects to be stored in the catalogue
        galfind_logger.debug(
            f"Loading {self.survey} {self.version} {self.cat_name} galaxies!"
        )
        gals = [Galaxy(ID, sky_coord, phot_obs, flags) \
            for ID, sky_coord, phot_obs, flags \
            in zip(IDs, sky_coords, phot_obs_arr, selection_flags)]
        cat = Catalogue(gals, self)
        # point to data if provided
        if hasattr(self, "data"):
            cat.data = self.data
        galfind_logger.info(f"Made {self.cat_path} catalogue!")
        return cat

    @property
    def cat_name(self) -> str:
        #f"{meta['SURVEY']}_{meta['VERSION']}_{meta['INSTR']}"
        return ".".join(self.cat_path.split("/")[-1].split(".")[:-1])

    def load_tab(self, cat_type: str, cropped: bool = True) -> Table:
        tab = self.open_cat(self.cat_path, cat_type)
        if tab is None:
            return None
        else:
            if cropped:
                return tab[self.crop_mask]
            else:
                return tab
    
    @staticmethod
    def _convert_crops_to_dict(crops: Optional[Union[Dict[str, Any], str, int, List[str], List[int]]]) -> Dict[str, Any]:
        # TODO: update this function to be more general
        if crops is not None:
            # make this a list of crops
            if isinstance(crops, str):
                crops_ = {crop: True for crop in crops.split("+")}
            elif isinstance(crops, int):
                crops_ = {"ID": crops}
            elif isinstance(crops, list):
                if isinstance(crops[0], int):
                    # TODO: ensure all elements are integers
                    crops_ = {"ID": crops}
                elif isinstance(crops[0], str):
                    # TODO: ensure all elements are strings
                    crops_ = {crop: True for crop in crops}
            elif isinstance(crops, dict):
                crops_ = {key: [value] if isinstance(value, int) \
                    else value for key, value in crops.items()}
            else:
                err_message = f"{type(crops)=} is invalid!"
                galfind_logger.critical(err_message)
                raise Exception(err_message)
        else:
            crops_ = None
        return crops_

    def set_crops(self, crops: Optional[Union[Dict[str, Any], str, int, List[str], List[int]]]) -> NoReturn:
        crops = self._convert_crops_to_dict(crops)
        self.crops = crops
        self._get_crop_mask()
    
    def update_crops(self, crops: Optional[Union[Dict[str, Any], str, int, List[str], List[int]]]) -> NoReturn:
        new_crops = self._convert_crops_to_dict(crops)
        if new_crops is None:
            crops = self.crops
        elif self.crops is None:
            crops = new_crops
        else:
            crops = {**self.crops, **new_crops}
        self.set_crops(crops)

    def _get_crop_mask(self) -> NoReturn:
        # crop table
        # TODO: implement more general catalogue loading
        tab = self.open_cat(self.cat_path, "ID")
        if self.crops is not None:
            # crop table using crop dict
            keep_arr = []
            for key, values in self.crops.items():
                # currently only crops by ID
                if "ID" in key.upper():
                    keep_arr.extend([np.array([True if ID in values else False \
                        for ID in self.load_IDs(cropped = False)])])
                    # galfind_logger.info(
                    #     f"Catalogue cropped by 'ID' to {values}"
                    # )
                elif key in tab.colnames:
                    if isinstance(tab[key][0], (bool, np.bool_)):
                        keep_arr.extend([np.array(tab[key]).astype(bool)])
                        # galfind_logger.info(
                        #     f"Catalogue cropped by {key}"
                        # )
                    else:
                        pass
                        # galfind_logger.warning(
                        #     f"{type(tab[key][0])=} not in [bool, np.bool_]"
                        # )
                else:
                    pass
                    # galfind_logger.warning(
                    #     f"Invalid crop name == {key}! Skipping"
                    # )
            # crop table
            self.crop_mask = np.array(np.logical_and.reduce(keep_arr)).astype(bool)
        else:
            self.crop_mask = np.full(len(tab), True)


    def load_IDs(self, cropped: bool = True) -> List[int]:
        tab = self.load_tab("ID", cropped)
        return self.load_ID_func(tab, self.ID_label, **self.load_ID_kwargs)

    def load_skycoords(self, cropped: bool = True) -> SkyCoord:
        tab = self.load_tab("sky_coord", cropped)
        return self.load_skycoords_func(tab, self.skycoords_labels, self.skycoords_units, **self.load_skycoords_kwargs)

    def load_phot(self, cropped: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        tab = self.load_tab("phot", cropped)
        # load the photometric fluxes and errors in units of Jy
        phot_labels, err_labels = self.get_phot_labels(self.filterset, self.aper_diams, **self.load_phot_kwargs)
        phot, phot_err = self.load_phot_func(tab, phot_labels, err_labels, **self.load_phot_kwargs)
        if cropped:
            gal_instr_mask = self.gal_instr_mask[self.crop_mask]
        else:
            gal_instr_mask = self.gal_instr_mask
        phot = {aper_diam: self._apply_gal_instr_mask(_phot, gal_instr_mask) for aper_diam, _phot in phot.items()}
        phot_err = {aper_diam: self._apply_gal_instr_mask(_phot_err, gal_instr_mask) for aper_diam, _phot_err in phot_err.items()}

        mask = self.load_mask(cropped)
        if mask is not None:
            phot = {aper_diam: [Masked(gal_phot, mask=gal_mask)
                for gal_phot, gal_mask in zip(_phot, mask)]
                for aper_diam, _phot in phot.items()}
            phot_err = {aper_diam: [Masked(gal_phot_err, mask=gal_mask)
                for gal_phot_err, gal_mask in zip(_phot_err, mask)]
                for aper_diam, _phot_err in phot_err.items()}
        return phot, phot_err
    
    def load_mask(self, cropped: bool = True) -> np.ndarray:
        tab = self.load_tab("mask", cropped)
        if self.load_mask_func is not None:
            mask_labels = self.get_mask_labels(self.filterset)
            try:
                mask = self.load_mask_func(tab, mask_labels, **self.load_mask_kwargs)
            except Exception as e:
                err_message = f"Error loading mask: {e}"
                galfind_logger.critical(err_message)
                raise(Exception(err_message))
            if cropped:
                gal_instr_mask = self.gal_instr_mask[self.crop_mask]
            else:
                gal_instr_mask = self.gal_instr_mask
            mask = self._apply_gal_instr_mask(mask, gal_instr_mask)
            return mask
        else:
            galfind_logger.warning(
                "Either load mask or mask label function not provided!"
            )
            return None

    def load_depths(self, cropped: bool = True) -> np.ndarray:
        tab = self.load_tab("depths", cropped)
        if self.load_depth_func is not None and self.get_depth_labels is not None:
            depth_labels = self.get_depth_labels(self.filterset, self.aper_diams)
            try:
                depths = self.load_depth_func(tab, depth_labels, **self.load_depth_kwargs)
            except Exception as e:
                err_message = f"Error loading depths: {e}"
                galfind_logger.critical(err_message)
                raise(Exception(err_message))
            if cropped:
                gal_instr_mask = self.gal_instr_mask[self.crop_mask]
            else:
                gal_instr_mask = self.gal_instr_mask
            depths = {aper_diam: self._apply_gal_instr_mask(_depths, gal_instr_mask) \
                for aper_diam, _depths in depths.items()}
            return depths
        else:
            galfind_logger.warning(
                "Either load depth or depth label function not provided!"
            )
            return None
        
    def load_selection_flags(self, cropped: bool = True) -> List[Dict[str, bool]]:
        tab = self.load_tab("selection", cropped)
        if self.load_selection_func is not None and self.get_selection_labels is not None and tab is not None:
            select_labels = self.get_selection_labels(tab, **self.load_selection_kwargs)
            select_dict = self.load_selection_func(tab, select_labels, **self.load_selection_kwargs)
            # ensure all selection dict values are the same length as the catalogue
            assert all(len(selection) == len(tab) for selection in select_dict.values()), \
                galfind_logger.critical("Not all selection values are the same length as the catalogue!")
            gal_selection = [{key: bool(values[i]) for key, values in select_dict.items()} for i in range(len(tab))]
            return gal_selection
        elif tab is None:
            galfind_logger.warning(
                "selection tab is None!"
            )
            tab = self.load_tab("ID", cropped)
        else:
            galfind_logger.warning(
                "Selection function not provided!"
            )
        return list(itertools.repeat(None, len(tab)))

    # current bottleneck
    def make_gal_instr_mask(
        self,
        null_data_vals: List[Union[float, np.nan]] = [0.0, np.nan],
        overwrite: bool = False,
        timed: bool = True
    ) -> NoReturn:
        meta = self.open_hdr(self.cat_path, "ID")
        if 'SURVEY' in meta.keys():
            save_dir = f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{meta['SURVEY']}/has_data_mask"
        elif self.survey is not None:
            save_dir = f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{self.survey}/has_data_mask"
        else:
            err_message = f"Neither 'SURVEY' in {self.cat_path} nor 'survey' provided!"
            galfind_logger.critical(err_message)
            raise Exception(err_message)
        save_path = f"{save_dir}/{self.cat_name}.h5"
        funcs.make_dirs(save_path)
        if Path(save_path).is_file() or overwrite:
            # load in gal_instr_mask from .h5
            hf = h5py.File(save_path, "r")
            gal_instr_mask = np.array(hf["has_data_mask"])
            galfind_logger.info(f"Loaded 'has_data_mask' from {save_path}")
        else:
            galfind_logger.info(f"Making 'has_data_mask' for {self.cat_name}!")
            # calculate the mask that is used to crop photometry to only bands including data
            tab = self.load_tab("phot", cropped = False)
            phot_labels, err_labels = self.get_phot_labels(self.filterset, self.aper_diams, **self.load_phot_kwargs)
            phot = self.load_phot_func(tab, phot_labels, err_labels, **self.load_phot_kwargs)[0]
            phot = list(phot.values())[0]
            if timed:
                gal_instr_mask = np.array(
                    [
                        [
                            True if val not in null_data_vals else False
                            for val in gal_phot
                        ]
                        for gal_phot in tqdm(
                            phot, desc="Making has_data_mask", total=len(phot)
                        )
                    ]
                )
            else:
                gal_instr_mask = np.array(
                    [
                        [
                            True if val not in null_data_vals else False
                            for val in gal_phot
                        ]
                        for gal_phot in phot
                    ]
                )
            # save as .h5
            hf = h5py.File(save_path, "w")
            hf.create_dataset("has_data_mask", data=gal_instr_mask)
            galfind_logger.info(f"Saved 'has_data_mask' to {save_path}")
        hf.close()
        if not hasattr(self, "gal_instr_mask"):
            self.gal_instr_mask = gal_instr_mask
    
    @staticmethod
    def _apply_gal_instr_mask(
        arr: np.ndarray,
        gal_instr_mask: np.ndarray,
    ) -> np.ndarray:
        assert len(gal_instr_mask) == len(arr), \
            galfind_logger.critical(f"{len(gal_instr_mask)} != {len(arr)}!")
        return [ele[mask] for ele, mask in zip(arr, gal_instr_mask)]

    def load_gal_filtersets(self, cropped: bool = True):
        # create set of filtersets to be pointed to by sources with these bands available
        galfind_logger.debug(f"Making {self.cat_name} unique filtersets!")
        if cropped:
            gal_instr_mask = self.gal_instr_mask[self.crop_mask]
        else:
            gal_instr_mask = self.gal_instr_mask
        unique_filt_comb = np.unique(gal_instr_mask, axis=0)
        unique_filtersets = [deepcopy(self.filterset)[data_comb] for data_comb in unique_filt_comb]
        filterset_arr = [
            unique_filtersets[
                np.where(
                    np.all(unique_filt_comb == data_comb, axis=1)
                )[0][0]
            ]
            for data_comb in gal_instr_mask
        ]
        galfind_logger.debug(f"Made {self.cat_name} unique filtersets!")
        return filterset_arr

class Catalogue(Catalogue_Base):

    @classmethod
    def pipeline(
        cls, 
        survey: str,
        version: str,
        instrument_names: List[str] = json.loads(
            config.get("Other", "INSTRUMENT_NAMES")
        ),
        pix_scales: Union[u.Quantity, Dict[str, u.Quantity]] = {
            "ACS_WFC": 0.03 * u.arcsec,
            "WFC3_IR": 0.03 * u.arcsec,
            "NIRCam": 0.03 * u.arcsec,
            "MIRI": 0.09 * u.arcsec,
        },
        im_str: List[str] = ["_sci", "_i2d", "_drz"],
        rms_err_str: List[str] = ["_rms_err", "_rms", "_err"],
        wht_str: List[str] = ["_wht", "_weight"],
        version_to_dir_dict: Optional[Dict[str, str]] = None,
        im_ext_name: Union[str, List[str]] = "SCI",
        rms_err_ext_name: Union[str, List[str]] = "ERR",
        wht_ext_name: Union[str, List[str]] = "WHT",
        aper_diams: Optional[u.Quantity] = None,
        forced_phot_band: Optional[
            Union[str, List[str], Type[Band_Data_Base]]
        ] = None,
        min_flux_pc_err: Union[int, float] = 10.,
        crops: Optional[Union[str, Dict[str, Union[str, List[str]]]]] = None,
    ) -> Catalogue:
        data = Data.pipeline(
            survey,
            version,
            instrument_names=instrument_names,
            pix_scales=pix_scales,
            im_str=im_str,
            rms_err_str=rms_err_str,
            wht_str=wht_str,
            version_to_dir_dict=version_to_dir_dict,
            im_ext_name=im_ext_name,
            rms_err_ext_name=rms_err_ext_name,
            wht_ext_name=wht_ext_name,
            aper_diams=aper_diams,
            forced_phot_band=forced_phot_band,
            min_flux_pc_err=min_flux_pc_err,
        )
        return cls.from_data(data, crops)

    @classmethod
    def from_data(
        cls, 
        data: Data,
        crops: Optional[Union[str, Dict[str, Union[str, List[str]]]]] = None,
    ) -> Catalogue:
        cat_creator = Catalogue_Creator.from_data(data, crops=crops)
        return cat_creator(cropped = True)

    def __repr__(self):
        return super().__repr__()
    
    def __str__(self) -> str:
        return super().__str__()

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
        galfind_logger.info(
            "Updating SED results in galfind catalogue object"
        )
        [
            gal.update_SED_results(gal_SED_result)
            for gal, gal_SED_result in tqdm(
                zip(self, cat_SED_results),
                desc="Updating galaxy SED results",
                total=len(self),
            )
        ]

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
        hdu: str = "OBJECTS",
        overwrite: bool = False,
    ) -> NoReturn:
        if hdu in ["OBJECTS", "SELECTION"]:
            ID_label = self.cat_creator.ID_label
        elif any([True for code in SED_code.__subclasses__() if hdu.find(code.__name__) != -1]):
            raise NotImplementedError
        else:
            raise NotImplementedError

        # TODO: Need to attach self.open_cat to catalogue_creator.open_cat
        append_tab = self.open_cat(cropped=False, hdu=hdu)
        # append to .fits table only if not already
        if append_tab is not None:
            if property_name in append_tab.colnames:
                if overwrite:
                    # remove old column that already exists
                    append_tab.remove_column(property_name)
                else:
                    err_message = f"{property_name=} already appended to " + \
                        f"{hdu=} .fits table, not overwriting!"
                    galfind_logger.warning(err_message)
                    return
        # return None
        galfind_logger.info(
            f"Appending {property_name=} to {hdu=} .fits table!"
        )
        # make new table with calculated properties
        gal_IDs = getattr(self, "ID")
        # TODO: get an array of properties to save (NOT an array in all cases!)
        gal_properties = getattr(self, property_name)
        assert len(gal_properties) == len(gal_IDs)
        # mask any NaN values
        # i = 0
        # property_type = None
        # while property_type is None:
        #     property_type = type(gal_properties[i])
        #     i += 1
        #gal_properties = MaskedColumn(gal_properties, mask=np.isnan(gal_properties), dtype = property_type)
        new_tab = Table(
            {"ID_temp": gal_IDs, property_name: gal_properties},
            dtype = [int, type(gal_properties[0])]
        )
        if append_tab is None:
            out_tab = new_tab
            out_tab.rename_column("ID_temp", ID_label)
        else:
            # join new and old tables
            out_tab = join(
                append_tab,
                new_tab,
                keys_left=ID_label,
                keys_right="ID_temp",
                join_type="outer",
            )
            out_tab.remove_column("ID_temp")
        # save multi-extension table
        self.write_hdu(out_tab, hdu=hdu)

    # def calc_new_property(self, func: Callable[..., float], arg_names: Union[list, np.array]):
    #     pass

    def load_sextractor_Re(self):
        if hasattr(self, "data"):
            # load Re from SExtractor
            pix_to_as_dict = {band_data.filt_name: band_data.pix_scale for band_data in self.data}
            self.load_band_properties_from_cat("FLUX_RADIUS", "sex_Re", multiply_factor = pix_to_as_dict)
        else:
            err_message = "Loading SExtractor Re from catalogue " + \
                f"only works when hasattr({repr(self)}, data)!"
            galfind_logger.critical(err_message)
            raise Exception(err_message)

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
            if multiply_factor is None:
                multiply_factor = {
                    filt.band_name: 1.0 * u.dimensionless_unscaled
                    for filt in self.filterset
                    if f"{cat_colname}_{filt.band_name}" in fits_cat.colnames
                }
            elif not isinstance(multiply_factor, dict):
                multiply_factor = {
                    filt.band_name: multiply_factor
                    for filt in self.filterset
                    if f"{cat_colname}_{filt.band_name}" in fits_cat.colnames
                }
            # load in speed can be improved here!
            cat_band_properties = {
                filt.band_name: np.array(fits_cat[f"{cat_colname}_{filt.band_name}"])
                * multiply_factor[filt.band_name]
                for filt in self.filterset
                if f"{cat_colname}_{filt.band_name}" in fits_cat.colnames
            }
            if len(cat_band_properties) == 0:
                galfind_logger.info(
                    f"Could not load {cat_colname=} from {self.cat_path}," + \
                    f" as no '{cat_colname}_band' exists for band in {self.instrument.band_names=}!"
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
                    raise NotImplementedError
                    [
                        gal.phot.load_property(gal_properties, save_name)
                        for gal, gal_properties in zip(
                            self, cat_band_properties
                        )
                    ]
                galfind_logger.info(
                    f"Loaded {cat_colname} from {self.cat_path} " + \
                    f"saved as {save_name} for {cat_band_properties[0].keys()=}"
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

    def make_cutouts(
        self,
        cutout_size: Union[u.Quantity, dict] = 0.96 * u.arcsec,
    ) -> None:
        # loop over galaxies, making a cutout of each one
        for band in tqdm(
            self.instrument.band_names,
            total=len(self.instrument),
            desc="Making band cutouts",
        ):
            # TODO: Requires update to use new Cutout class
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
                if isinstance(cutout_size, dict):
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
                    pix_scale=self.data.pix_scales[band],
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

    def stack_gals(
        self, cutout_size: u.Quantity = 0.96 * u.arcsec
    ) -> Multiple_Band_Cutout:
        # stack all galaxies in catalogue for a given band
        if not hasattr(self, "stacked_cutouts"):
            self.stacked_cutouts = {}
        cutout_size_str = f"{cutout_size.to(u.arcsec).value:.2f}as"
        if cutout_size_str not in self.stacked_cutouts.keys():
            self.stacked_cutouts[cutout_size_str] = (
                Multiple_Band_Cutout.from_cat(self, cutout_size=cutout_size)
            )
        return self.stacked_cutouts[cutout_size_str]

    def plot_stacked_gals(
        self,
        cutout_size: u.Quantity = 0.96 * u.arcsec,
        save_name: Optional[str] = None,
    ) -> NoReturn:
        stacked_cutouts = self.stack_gals(cutout_size=cutout_size)
        stacked_cutouts.plot(save_name=save_name)

    def make_RGBs(
        self,
        cutout_size: u.Quantity = 0.96 * u.arcsec,
        rgb_bands: Dict[str, List[str]] = {
            "B": ["F090W"],
            "G": ["F200W"],
            "R": ["F444W"],
        },
    ) -> Multiple_RGB:
        if not hasattr(self, "RGBs"):
            self.RGBs = {}
        cutout_size_str = f"{cutout_size.to(u.arcsec).value:.2f}as"
        if cutout_size_str not in self.RGBs.keys():
            self.RGBs[cutout_size_str] = {}
        rgb_key = ",".join(
            f"{colour}={'+'.join(self.get_colour_band_names[colour])}"
            for colour in ["B", "G", "R"]
        )
        if (
            rgb_key
            not in self.RGBs[f"{cutout_size.to(u.arcsec).value:.2f}as"].keys()
        ):
            self.RGBs[cutout_size_str][rgb_key] = Multiple_RGB.from_cat(
                self, cutout_size=cutout_size
            )
        return self.RGBs[cutout_size_str][rgb_key]

    def plot_RGBs(
        self,
        cutout_size: u.Quantity = 0.96 * u.arcsec,
        rgb_bands: Dict[str, List[str]] = {
            "B": ["F090W"],
            "G": ["F200W"],
            "R": ["F444W"],
        },
        method: str = "trilogy",
        save_name: Optional[str] = None,
    ) -> NoReturn:
        RGBs = self.make_RGBs(cutout_size=cutout_size, rgb_bands=rgb_bands)
        RGBs.plot(save_name=save_name)

    def make_stacked_RGB(
        self,
        cutout_size: u.Quantity = 0.96 * u.arcsec,
        rgb_bands: Dict[str, List[str]] = {
            "B": ["F090W"],
            "G": ["F200W"],
            "R": ["F444W"],
        },
    ) -> Stacked_RGB:
        if not hasattr(self, "stacked_RGB"):
            self.stacked_RGB = {}
        cutout_size_str = f"{cutout_size.to(u.arcsec).value:.2f}as"
        if cutout_size_str not in self.stacked_RGB.keys():
            self.stacked_RGB[cutout_size_str] = Stacked_RGB.from_cat(
                self, cutout_size=cutout_size, rgb_bands=rgb_bands
            )
        return self.stacked_RGB[cutout_size_str]

    def plot_phot_diagnostics(
        self,
        SED_fit_params_arr,#=[
        #     EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
        #     EAZY({"templates": "fsps_larson", "dz": 0.5, "lowz_zma"}),
        # ],
        zPDF_plot_SED_fit_params_arr,#=[
        #     EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
        #     EAZY({"templates": "fsps_larson", "dz": 0.5}),
        # ],
        wav_unit=u.um,
        flux_unit=u.ABmag,
    ):
        # loop over galaxies and plot photometry diagnostic plots for each one
        # figure size may well depend on how many bands there are
        overall_fig = plt.figure(figsize=(8, 7), constrained_layout=True)
        fig, cutout_fig = overall_fig.subfigures(
            2,
            1,
            hspace=-2,
            height_ratios=[2.0, 1.0]
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
        self: Type[Self],
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
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
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

    # %% SED property functions
    # Rest-frame UV property calculation functions - these are not independent of each other

    # beta_phot tqdm bar not working appropriately!
    def calc_beta_phot(
        self,
        rest_UV_wav_lims=[1_250.0, 3_000.0] * u.AA,
        SED_fit_params=EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
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
        SED_fit_params=EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
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
        SED_fit_params=EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
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
        SED_fit_params = EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
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
        SED_fit_params = EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
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
        SED_fit_params=EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
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
        SED_fit_params = EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
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
        SED_fit_params = EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
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
        SED_fit_params = EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
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
        SED_fit_params = EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
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
        SED_fit_params = EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
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
        SED_fit_params = EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
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
        SED_fit_params = EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
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
        SED_fit_params=EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
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
        SED_fit_params,
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

    # def _save_SED_rest_PDFs(self, property_name, save_dir, SED_fit_params = EAZY({"templates": "fsps_larson", "lowz_zmax": None})):
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
        SED_fit_params,
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
        SED_fit_params,
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

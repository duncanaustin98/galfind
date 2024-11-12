#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 13:59:59 2023

@author: austind
"""

from __future__ import annotations

# Catalogue_Creator.py
import json
import time
from abc import ABC, abstractmethod
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.io import fits
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, Tuple, Optional, List, Callable, Union, Dict
if TYPE_CHECKING:
    from . import Multiple_Filter, SED_code, Data
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

import astropy.units as u
import h5py
import numpy as np
from astropy.utils.masked import Masked
from tqdm import tqdm

from . import (
    EAZY,  # noqa F501
    Bagpipes,  # noqa F501
    LePhare,  # noqa F501
    config,
    galfind_logger,
    Photometry_obs,
    Galaxy,
    Catalogue,
    Catalogue_Base, # noqa F501
)
from . import useful_funcs_austind as funcs


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
    return {aper_diam: depth for aper_diam, depth in \
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

def open_galfind_cat(cat_path: str, cat_type: str) -> Table:
    if cat_type in ["ID", "sky_coord", "phot", "mask", "depths", "selection"]:
        tab = Table.read(
            cat_path, character_as_bytes=False, memmap=True
        )
    elif check_hdu_exists(cat_path, cat_type):
        tab = Table.read(
            cat_path, character_as_bytes=False, memmap=True, hdu=cat_type
        )
    else:
        err_message = f"cat_type = {cat_type} not in " + \
            "['ID', 'sky_coord', 'phot', 'mask', 'depths', 'selection'] and " + \
            f"not a valid HDU extension in {cat_path}!"
        galfind_logger.critical(err_message)
        raise Exception(err_message)
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
    return {aper_diam * aper_diams.unit: [f"loc_depth_{filt_name}" for filt_name in filterset.band_names] for aper_diam in aper_diams.value}

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
        load_selection_func: Optional[Callable] = None,
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
        self.crops = crops
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
        self.load_selection_kwargs = load_selection_kwargs
        self.load_SED_result_func = load_SED_result_func

        self._get_crop_mask()
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
        filterset_arr = self.load_gal_filtersets(cropped)
        SED_results = None
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
        gals = [Galaxy(ID, sky_coord, phot_obs) \
            for ID, sky_coord, phot_obs in zip(IDs, sky_coords, phot_obs_arr)]
        cat = Catalogue(gals, self)
        galfind_logger.info(f"Made {self.cat_path} catalogue!")
        return cat

    @property
    def cat_name(self) -> str:
        #f"{meta['SURVEY']}_{meta['VERSION']}_{meta['INSTR']}"
        return ".".join(self.cat_path.split("/")[-1].split(".")[:-1])

    def load_tab(self, cat_type: str, cropped: bool = True) -> Table:
        tab = self.open_cat(self.cat_path, cat_type)
        if cropped:
            return tab[self.crop_mask]
        else:
            return tab
    
    def _get_crop_mask(self) -> NoReturn:
        # crop table
        # TODO: implement more general catalogue loading
        tab = self.open_cat(self.cat_path, "ID")
        if self.crops is not None:
            if isinstance(self.crops, str):
                crops = {crop: True for crop in self.crops.split("+")}
            elif isinstance(self.crops, int):
                crops = {"ID": self.crops}
            elif isinstance(self.crops, list):
                if isinstance(self.crops[0], int):
                    # TODO: ensure all elements are integers
                    crops = {"ID": self.crops}
                elif isinstance(self.crops[0], str):
                    # TODO: ensure all elements are strings
                    crops = {crop: True for crop in self.crops}
            elif isinstance(self.crops, dict):
                crops = {key: [value] if isinstance(value, int) \
                    else value for key, value in self.crops.items()}
            else:
                err_message = f"{type(self.crops)=} is invalid!"
                galfind_logger.critical(err_message)
                raise Exception(err_message)
            # crop table using crop dict
            keep_arr = []
            for key, values in crops.items():
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
        
    # def load_selection(self, cat: Any, selection_names: List[str]) -> List[str]:
    #     if isinstance(cat, str):
    #         tab = self.load_tab("selection")
    #     if self.load_selection_func is not None:
    #         return self.load_selection_func(tab, selection_names)
    #     else:
    #         galfind_logger.warning("No selection function provided!")
    #         return None

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


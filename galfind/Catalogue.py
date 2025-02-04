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
from matplotlib import cm
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
    from . import Filter, Band_Data_Base, Selector, Multiple_Filter, Data, Property_Calculator_Base, Band_Cutout_Base, Band_Cutout
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
from . import SED_code
from .Cutout import Multiple_Band_Cutout, Multiple_RGB, Stacked_RGB
from .Data import Data
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

    def __init__(
        self: Self,
        survey: str,
        version: str, 
        cat_path: str,
        filterset: Multiple_Filter,
        aper_diams: u.Quantity,
        crops: Optional[Union[Type[Selector], List[Type[Selector]]]] = None,
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
        crops: Optional[Union[Type[Selector], List[Type[Selector]]]] = None,
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
        gals = [Galaxy(ID, sky_coord, phot_obs, flags, cat_filterset) \
            for ID, sky_coord, phot_obs, flags, cat_filterset \
            in zip(IDs, sky_coords, phot_obs_arr, selection_flags, filterset_arr)]
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

    @property
    def crop_name(self) -> List[str]:
        return funcs.get_crop_name(self.crops)

    def load_tab(self, cat_type: str, cropped: bool = True) -> Table:
        tab = self.open_cat(self.cat_path, cat_type)
        if tab is None:
            return None
        else:
            if cropped:
                return tab[self.crop_mask]
            else:
                return tab
    
    def set_crops(self, crops: Optional[Union[Type[Selector], List[Type[Selector]]]]) -> NoReturn:
        if crops is None:
            self.crops = []
        elif not isinstance(crops, (list, np.ndarray)):
            self.crops = [crops]
        else:
            self.crops = crops
        self._get_crop_mask()

    def _get_crop_mask(self) -> NoReturn:
        tab = self.open_cat(self.cat_path, "SELECTION")
        if tab is None:
            tab = self.open_cat(self.cat_path, "ID")
        elif len(self.crops) > 0:
            # crop table using crop dict
            keep_arr = []
            for selector in self.crops:
                if selector.name in tab.colnames:
                    keep_arr.extend([np.array(tab[selector.name]).astype(bool)])
                    galfind_logger.info(
                        f"Catalogue cropped by {selector.name}"
                    )
                else:
                    err_message = f"{selector.name} not yet performed!"
                    galfind_logger.warning(err_message)
                    raise Exception(err_message)
            # crop table
            if len(keep_arr) > 0:
                self.crop_mask = np.array(np.logical_and.reduce(keep_arr)).astype(bool)
                return
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
        tab = self.load_tab("SELECTION", cropped)
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
        if all(name in meta.keys() for name in ["SURVEY", "INSTR"]):
            save_dir = f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{meta['SURVEY']}/has_data_mask/{meta['INSTR']}"
        elif self.survey is not None and self.filterset is not None:
            save_dir = f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{self.survey}/has_data_mask/{self.filterset.instrument_name}"
        else:
            err_message = f"Not both of ['SURVEY', 'INSTR'] in {self.cat_path} nor 'survey' and 'filterset' provided!"
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
        crops: Optional[Union[Type[Selector], List[Type[Selector]]]] = None,
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
        crops: Optional[Union[Type[Selector], List[Type[Selector]]]] = None,
    ) -> Catalogue:
        cat_creator = Catalogue_Creator.from_data(data, crops=crops)
        return cat_creator(cropped = True)

    @property
    def crop_name(self) -> List[str]:
        return self.cat_creator.crop_name

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

        # # calculate aperture corrections if not already
        # if not hasattr(self.instrument, "aper_corrs"):
        #     aper_corrs = self.instrument.get_aper_corrs(
        #         self.cat_creator.aper_diam
        #     )
        # else:
        #     aper_corrs = self.instrument.aper_corrs[self.cat_creator.aper_diam]
        # assert len(aper_corrs) == len(self.instrument)
        # aper_corrs = {
        #     band_name: aper_corr
        #     for band_name, aper_corr in zip(
        #         self.instrument.band_names, aper_corrs
        #     )
        # }
        # # calculate and save dict of ext_src_corrs for each galaxy in self
        # galfind_logger.debug(
        #     "Photometry_obs.calc_ext_src_corrs takes 2min 20s for JOF with 16,000 galaxies. Fairly slow!"
        # )
        # [
        #     gal.phot.calc_ext_src_corrs(aper_corrs=aper_corrs)
        #     for gal in tqdm(
        #         self,
        #         desc=f"Calculating extended source corrections for {self.survey} {self.version}",
        #         total=len(self),
        #     )
        # ]
        # # save the extended source corrections to catalogue
        # self._append_property_to_tab(
        #     "_".join(inspect.stack()[0].function.split("_")[1:]), "phot_obs"
        # )

    # def make_ext_src_corrs(
    #     self, gal_property: str, origin: Union[str, dict]
    # ) -> str:
    #     # calculate pre-requisites
    #     self.calc_ext_src_corrs()
    #     # make extended source correction for given property
    #     [aper_phot_.make_ext_src_corrs(gal_property, origin) for gal in self for aper_phot_ in gal.aper_phot.values()]
    #     # save properties to fits table
    #     #property_name = f"{gal_property}{funcs.ext_src_label}"
    #     #self._append_property_to_tab(property_name, origin)
    #     #return property_name

    # def make_all_ext_src_corrs(self) -> None:
    #     self.calc_ext_src_corrs()
    #     properties_dict = [gal.phot.make_all_ext_src_corrs() for gal in self]
    #     unique_origins = np.unique(
    #         [property_dict.keys() for property_dict in properties_dict]
    #     )[0]
    #     unique_properties_origins_dict = {
    #         key: np.unique(
    #             [property_dict[key] for property_dict in properties_dict]
    #         )
    #         for key in unique_origins
    #     }
    #     unique_properties_origins_dict = {
    #         key: value
    #         for key, value in unique_properties_origins_dict.items()
    #         if len(value) > 0
    #     }
    #     # breakpoint()
    #     [
    #         self._append_property_to_tab(property_name, origin)
    #         for origin, property_names in unique_properties_origins_dict.items()
    #         for property_name in property_names
    #     ]

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
        
    def load_sextractor_auto_mags(self):
        if hasattr(self, "data"):
            # load Re from SExtractor
            self.load_band_properties_from_cat("MAG_AUTO", "sex_MAG_AUTO", multiply_factor = u.ABmag)
            for gal in self:
                for aper_diam in gal.aper_phot.keys():
                    gal.aper_phot[aper_diam].sex_MAG_AUTO = gal.sex_MAG_AUTO
        else:
            err_message = "Loading SExtractor auto mags from catalogue " + \
                f"only works when hasattr({repr(self)}, data)!"
            galfind_logger.critical(err_message)
            raise Exception(err_message)

    def load_sextractor_auto_fluxes(self):
        if hasattr(self, "data"):
            # load Re from SExtractor
            multiply_factor = {band_data.filt_name: \
                funcs.flux_image_to_Jy(1.0, band_data.ZP) \
                for band_data in self.data}
            self.load_band_properties_from_cat(
                "FLUX_AUTO", 
                "sex_FLUX_AUTO", 
                multiply_factor = multiply_factor
            )
            for gal in self:
                for aper_diam in gal.aper_phot.keys():
                    gal.aper_phot[aper_diam].sex_FLUX_AUTO = gal.sex_FLUX_AUTO
        else:
            err_message = "Loading SExtractor flux autos from catalogue " + \
                f"only works when hasattr({repr(self)}, data)!"
            galfind_logger.critical(err_message)
            raise Exception(err_message)
        
    def load_sextractor_kron_radii(self):
        if hasattr(self, "data"):
            # load Kron radius from SExtractor
            kron_radii = self.load_band_properties_from_cat(
                "KRON_RADIUS", 
                "sex_KRON_RADIUS", 
            )
            A_image_arr = self.load_band_properties_from_cat(
                "A_IMAGE",
                "sex_A_IMAGE",
                update = False
            )
            [setattr(gal, "sex_A_IMAGE", A_image[self.data[0].filt_name]) for gal, A_image in zip(self, A_image_arr)]
            A_image_as_arr = [{band_data.filt_name: kron_radius[band_data.filt_name] * A_image[band_data.filt_name] * band_data.pix_scale \
                for band_data in self.data} for kron_radius, A_image in zip(kron_radii, A_image_arr)]
            [gal.load_property(A_image_as, "sex_A_IMAGE_AS") for gal, A_image_as in zip(self, A_image_as_arr)]
            B_image_arr = self.load_band_properties_from_cat(
                "B_IMAGE",
                "sex_B_IMAGE",
                update = False
            )
            [setattr(gal, "sex_B_IMAGE", B_image[self.data[0].filt_name]) for gal, B_image in zip(self, B_image_arr)]
            B_image_as_arr = [{band_data.filt_name: kron_radius[band_data.filt_name] * B_image[band_data.filt_name] * band_data.pix_scale \
                for band_data in self.data} for kron_radius, B_image in zip(kron_radii, B_image_arr)]
            [gal.load_property(B_image_as, "sex_B_IMAGE_AS") for gal, B_image_as in zip(self, B_image_as_arr)]
            theta_image_arr = self.load_band_properties_from_cat(
                "THETA_IMAGE",
                "sex_THETA_IMAGE",
                multiply_factor = u.deg,
                update = False,
            )
            [setattr(gal, "sex_THETA_IMAGE", theta_image[self.data[0].filt_name]) for gal, theta_image in zip(self, theta_image_arr)]
            
            for gal in self:
                for aper_diam in gal.aper_phot.keys():
                    for name in ["KRON_RADIUS", "A_IMAGE", "B_IMAGE", "THETA_IMAGE", "A_IMAGE_AS", "B_IMAGE_AS"]:
                        name = f"sex_{name}"
                        if hasattr(gal, name):
                            setattr(gal.aper_phot[aper_diam], name, getattr(gal, name))
                        else:
                            galfind_logger.warning(f"{name} not in {gal.ID=}")
            
        
    def load_sextractor_ext_src_corrs(self) -> None:
        self.load_sextractor_auto_fluxes()
        galfind_logger.info(
            f"Loading SExtractor extended source corrections for {self.cat_name}!"
        )
        [filt.instrument._load_aper_corrs() for filt in self.data.filterset]
        aper_corrs = {filt.band_name: filt.instrument. \
            aper_corrs[filt.band_name] for filt in self.data.filterset}
        [gal.load_sextractor_ext_src_corrs(aper_corrs) for gal in self]

    def load_band_properties_from_cat(
        self,
        cat_colname: str,
        save_name: str,
        multiply_factor: Union[dict, u.Quantity, u.Magnitude, None] = None,
        dest: str = "gal",
        update: bool = True,
    ) -> Optional[List[Dict[str, Union[u.Quantity, u.Magnitude, u.Dex]]]]:
        assert dest in ["gal", "phot_obs"]
        if dest == "gal":
            has_attr = hasattr(self[0], save_name)
        else:  # dest == "phot_obs"
            has_attr = hasattr(self[0].phot, save_name)
        if not has_attr:
            galfind_logger.info(
                f"Loading {cat_colname=} from {self.cat_path} saved as {save_name}!"
            )
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
                err_message = f"Could not load {cat_colname=} from {self.cat_path} " + \
                    f"as no '{cat_colname}_band' exists for band in {self.instrument.band_names=}!"
                galfind_logger.info(err_message)
                raise Exception(err_message)
            
            cat_band_properties = [
                {
                    band: cat_band_properties[band][i]
                    for band in cat_band_properties.keys()
                }
                for i in range(len(fits_cat))
            ]
            if update:
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
            return cat_band_properties

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
            "MAG_AUTO", 
            "MAG_AUTO", 
            multiply_factor=u.ABmag, 
            dest="phot_obs",
        )

    def make_cutouts(
        self: Self,
        cutout_size: Union[u.Quantity, dict] = 0.96 * u.arcsec,
    ) -> List[Dict[str, Type[Band_Cutout_Base]]]:
        return [gal.make_cutouts(self.data, cutout_size) for gal in self]
    
    def make_band_cutouts(
        self: Self,
        filt: Union[str, Filter],
        cutout_size: u.Quantity = 0.96 * u.arcsec,
    ) -> List[Band_Cutout]:
        band_data = self.data[filt.band_name]
        return [gal.make_band_cutout(band_data, cutout_size) for gal in self]

    def plot_cutouts(
        self: Self,
        cutout_size: u.Quantity = 0.96 * u.arcsec,
        #save_name: Optional[str] = None,
        plot_kwargs: Dict[str, Any] = {},
        crop_name: Optional[str] = None,
        collate_dir: Optional[str] = None,
        overwrite: bool = False,
        overwrite_sample: bool = True,
    ) -> NoReturn:
        out_paths = np.full(len(self), None)
        for i, gal in enumerate(self):
            cutouts = gal.make_cutouts(self.data, cutout_size, overwrite = overwrite)
            cutouts.plot(close_fig = True, **plot_kwargs)
            out_paths[i] = cutouts._get_save_path()
        out_paths.astype(str)
        # collate plots for the specified galaxies
        if collate_dir is None:
            if crop_name is None:
                crop_name = self.crop_name
            collate_dir = "/".join(out_paths[0].replace("/png", "").split("/")[:-2] + [crop_name]) \
                + out_paths[0].replace("/png", "").split("/")[-2]
        self._collate_plots(out_paths, collate_dir, overwrite = overwrite_sample)

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
        aper_diam: u.Quantity,
        SED_arr: List[SED_code],
        zPDF_arr: List[SED_code],
        plot_lowz: bool = True,
        wav_unit: u.Unit = u.um,
        flux_unit: u.Unit = u.ABmag,
        crop_name: Optional[str] = None,
        collate_dir: Optional[str] = None,
        imshow_kwargs: Dict[str, Any] = {},
        norm_kwargs: Dict[str, Any] = {},
        aper_kwargs: Dict[str, Any] = {},
        kron_kwargs: Dict[str, Any] = {},
        n_cutout_rows: int = 2,
        overwrite: bool = False,
    ):
        # load sextractor parameters
        self.load_sextractor_auto_mags()
        self.load_sextractor_auto_fluxes()
        self.load_sextractor_kron_radii()
        self.load_sextractor_Re()

        # loop over galaxies and plot photometry diagnostic plots for each one
        # figure size may well depend on how many bands there are
        overall_fig = plt.figure(figsize=(8, 7), constrained_layout=True)
        fig, cutout_fig = overall_fig.subfigures(
            2,
            1,
            hspace=-2.,
            height_ratios=[2.0, 1.0]
            if len(self.data) <= 8
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
                aper_diam = aper_diam,
                SED_arr = SED_arr,
                zPDF_arr = zPDF_arr,
                plot_lowz = plot_lowz,
                n_cutout_rows = n_cutout_rows,
                wav_unit = wav_unit,
                flux_unit = flux_unit,
                imshow_kwargs = imshow_kwargs,
                norm_kwargs = norm_kwargs,
                aper_kwargs = aper_kwargs,
                kron_kwargs = kron_kwargs,
                overwrite = overwrite,
                save = True,
                show = False,
            )
            for gal in tqdm(
                self,
                total=len(self),
                desc="Plotting photometry diagnostic plots",
            )
        ]
        if collate_dir is None:
            if crop_name is None:
                crop_name = self.crop_name
            collate_dir = f"{config['Other']['PLOT_DIR']}/{self.version}/" + \
                f"{self.filterset.instrument_name}/{self.survey}/SED_plots/" + \
                f"{aper_diam.to(u.arcsec).value:.2f}as/{crop_name}/"
        self._collate_plots(out_paths, collate_dir)

    def _collate_plots(
        self: Self,
        out_paths: List[str],
        collate_dir: str,
        overwrite: bool = True
    ):
        # make a folder to store symlinked photometric diagnostic plots for selected galaxies
        if self.crops != []:
            if overwrite:
                # remove existing symlinks in folder
                symlink_paths = glob.glob(f"{collate_dir}/*.png")
                for symlink_path in symlink_paths:
                    os.remove(symlink_path)
            # create symlink to selection folder for diagnostic plots
            for gal, out_path in zip(self, out_paths):
                selection_path = f"{collate_dir}/{str(gal.ID)}.png"
                funcs.make_dirs(selection_path)
                try:
                    os.symlink(out_path, selection_path)
                except FileExistsError:  # replace existing file
                    if Path(out_path).is_file():
                        os.remove(selection_path)
                        os.symlink(out_path, selection_path)

    def hist(
        self: Self,
        x_calculator: Type[Property_Calculator_Base],
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        log: bool = False,
        n_bins: int = 50, 
        save: bool = True,
        show: bool = False,
        overwrite: bool = True,
    ) -> NoReturn:
        save_path = f"{config['DEFAULT']['GALFIND_WORK']}/Plots/{self.version}/" + \
            f"{self.filterset.instrument_name}/{self.survey}/hist/" + \
            f"{self.crop_name}/{x_calculator.name}.png"
        if not Path(save_path).is_file() or overwrite:
            galfind_logger.info(
                f"Making histogram for {x_calculator.name}!"
            )
            from . import Property_Calculator_Base
            if isinstance(x_calculator, tuple(Property_Calculator_Base.__subclasses__())):
                # calculate x values
                x = [gal.aper_phot[x_calculator.aper_diam].SED_results \
                    [x_calculator.SED_fit_label].phot_rest.properties \
                    [x_calculator.name] for gal in self]
                # remove nans
                x = np.array([x_.value for x_ in x if not np.isnan(x_)])
                #x_name = x_calculator.name
                x_label = x_calculator.name
            else:
                raise NotImplementedError
            if log:
                x = np.log10(x)
                #x_name = f"log({x_name})"
                x_label = f"log({x_label})"
            if any(fig_ax is None for fig_ax in [fig, ax]):
                fig, ax = plt.subplots()
            ax.hist(x, bins = n_bins)
            ax.set_xlabel(x_label)
            #ax.set_ylabel("Number of Galaxies")
            #ax.set_title(f"{x_label} histogram")
            if save:
                funcs.make_dirs(save_path)
                plt.savefig(save_path)
                funcs.change_file_permissions(save_path)
            if show:
                plt.show()
        else:
            galfind_logger.info(
                f"{x_calculator.name} histogram already exists and {overwrite=}, skipping!"
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
        contour_levels: List[float] = [68., 95., 100.],
        x_hist_ax: Optional[plt.Axes] = None,
        y_hist_ax: Optional[plt.Axes] = None,
        hist_kwargs: Dict[str, Any] = {},
    ):
        from . import Property_Calculator_Base
        x_name = x_calculator.full_name
        y_name = y_calculator.full_name
        x_label = x_calculator.plot_name
        y_label = y_calculator.plot_name
        colour_name = c_calculator.full_name if c_calculator is not None else ""
        colour_label = c_calculator.plot_name if c_calculator is not None else None

        if plot_type.lower() == "individual":
            if isinstance(x_calculator, tuple(Property_Calculator_Base.__subclasses__())):
                x = x_calculator.extract_vals(self)
                if incl_x_errs:
                    raise NotImplementedError
                else:
                    x_err = None
            else:
                raise NotImplementedError
            if log_x or x_name in funcs.logged_properties:
                if incl_x_errs:
                    x, x_err = funcs.errs_to_log(x, x_err)
                else:
                    x = np.log10(x.value)
                x_name = f"log({x_name})"
                x_label = f"log({x_label})"

            if isinstance(y_calculator, tuple(Property_Calculator_Base.__subclasses__())):
                y = y_calculator.extract_vals(self)
                if incl_y_errs:
                    raise NotImplementedError
                else:
                    y_err = None
            else:
                raise NotImplementedError
            if log_y or y_name in funcs.logged_properties:
                if incl_y_errs:
                    y, y_err = funcs.errs_to_log(y, y_err)
                else:
                    y = np.log10(y.value)
                y_name = f"log({y_name})"
                y_label = f"log({y_label})"

            if c_calculator is not None:
                if isinstance(c_calculator, tuple(Property_Calculator_Base.__subclasses__())):
                    c = c_calculator.extract_vals(self).value
                else:
                    raise NotImplementedError
                if log_c or colour_name in funcs.logged_properties:
                    c = np.log10(c)
                    colour_name = f"log({colour_name})"
                    colour_label = f"log({colour_label})"

        elif plot_type.lower() == "contour":

            assert c_calculator is None, galfind_logger.critical(
                "Cannot contour galaxies and colour by another property!"
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

            assert c_calculator is None, galfind_logger.critical(
                "Cannot stack galaxies and colour by another property!"
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
            err_message = f"{plot_type=} not recognised!"
            galfind_logger.critical(err_message)
            raise Exception(err_message)

        # setup matplotlib figure/axis if not already given
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        if "label" not in plot_kwargs.keys() and plot_type.lower() != "contour":
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
            x_edges = np.linspace(np.min(x) - x_interval, np.max(x) + x_interval, n_bins)
            y_edges = np.linspace(np.min(y) - y_interval, np.max(y) + y_interval, n_bins)
            N, x_edges, y_edges = np.histogram2d(x, y, bins=(x_edges, y_edges))
            x_mid_bins = (x_edges[:-1] + x_edges[1:]) / 2
            y_mid_bins = (y_edges[:-1] + y_edges[1:]) / 2
            x_mesh, y_mesh = np.meshgrid(x_mid_bins, y_mid_bins)
            levels = np.percentile(N, contour_levels)
            # n_bins = int(len(x) / (25 * n_samples))
            for i in range(len(levels) - 1):
                ax.contourf(x_mesh, y_mesh, N.T, levels = [levels[i], levels[i + 1]], alpha = (i + 1) / len(levels), colors = [cmap], **plot_kwargs) # , norm = norm, cmap = cmap, norm = norm, cmap = cm.get_cmap(cmap).resampled(len(plot_kwargs["levels"]) - 1)
            for level in levels:
                ax.contour(x_mesh, y_mesh, N.T, levels = (level,), colors = cmap, linewidths = 1.0, **plot_kwargs)
            ax.set_xlim([x_edges[0], x_edges[-1]])
            ax.set_ylim([y_edges[0], y_edges[-1]])
        else:
            if incl_x_errs or incl_y_errs and not mean_err:
                if "ls" not in plot_kwargs.keys():
                    plot_kwargs["ls"] = ""
                plot = ax.errorbar(x, y, xerr=x_err, yerr=y_err, **plot_kwargs)
            else:
                plot = ax.scatter(x, y, **plot_kwargs)

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
            #divider = make_axes_locatable(ax)
        if x_hist_ax is not None:
            assert isinstance(x_hist_ax, plt.Axes)
            #ax_hist_top = divider.append_axes("top", size="20%", pad=0.1, sharex=ax)
            x_hist_ax.hist(x, **hist_kwargs)
        if y_hist_ax is not None:
            assert isinstance(y_hist_ax, plt.Axes)
            # Right histogram
            #ax_hist_right = divider.append_axes("right", size="20%", pad=0.1, sharey=ax)
            y_hist_ax.hist(y, orientation='horizontal', **hist_kwargs)

        # sort plot aesthetics - automatically annotate if coloured
        if annotate or c_calculator is not None:
            plot_label = (
                f"{self.version}, {self.filterset.instrument_name}, {self.survey}"
            )
            ax.set_title(plot_label)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            if c_calculator is not None:
                fig.colorbar(plot, label = colour_label)
            if plot_legend:
                ax.legend(**legend_kwargs)

        if save:
            # determine appropriate save path
            save_colour_name = f"_c={colour_name}" if c_calculator is not None else ""
            if save_path is None:
                save_path = f"{config['Other']['PLOT_DIR']}/{self.version}/" + \
                    f"{self.filterset.instrument_name}/{self.survey}/" + \
                    f"{y_name}_vs_{x_name}/{self.crop_name}{save_colour_name}.png"
            funcs.make_dirs(save_path)
            plt.savefig(save_path)
            funcs.change_file_permissions(save_path)
            galfind_logger.info(f"Saved plot to {save_path}!")

        if show:
            plt.show()

    # Number Density Function (e.g. UVLF and mass functions) methods

    def calc_Vmax(
        self: Self,
        data_arr: Union[list, np.array],
        z_bin: Union[list, np.array],
        aper_diam: u.Quantity,
        SED_fit_code: SED_code,
        z_step: float = 0.01,
        timed: bool = False,
    ) -> None:
        assert len(z_bin) == 2
        assert z_bin[0] < z_bin[1]
        assert isinstance(SED_fit_code, tuple(SED_code.__subclasses__()))
        assert all(SED_fit_code.label in gal.aper_phot[aper_diam].SED_results.keys() for gal in self)

        for data in data_arr:
            save_path = f"{config['NumberDensityFunctions']['VMAX_DIR']}/" + \
                f"{self.version}/{self.filterset.instrument_name}/{self.survey}/" + \
                f"{self.crop_name}/Vmax_field={data.full_name}.ecsv"
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
                [
                    gal.calc_Vmax(
                        self.data.full_name,
                        [data],
                        z_bin,
                        aper_diam,
                        SED_fit_code,
                        self.cat_creator.crops,
                        z_step,
                        timed=timed,
                    )
                    for gal in tqdm(
                        self,
                        total=len(self),
                        desc=f"Calculating Vmax's for {self.data.full_name}" + \
                            f" in {self.crop_name} {data.full_name}",
                    )
                ]
                # table with uncalculated Vmax's
                Vmax_arr = np.array(
                    [
                        gal.aper_phot[aper_diam].SED_results[SED_fit_code.label]. \
                        V_max[self.crop_name.split("/")[-1]][data.full_name].to(u.Mpc**3).value
                        if isinstance(gal.aper_phot[aper_diam].SED_results[SED_fit_code.label]. \
                        V_max[self.crop_name.split("/")[-1]][data.full_name], u.Quantity)
                        else gal.aper_phot[aper_diam].SED_results[SED_fit_code.label]. \
                        V_max[self.crop_name.split("/")[-1]][data.full_name]
                        for gal in self if gal.ID in update_IDs
                    ]
                )
                obs_zmin = np.array(
                    [
                        gal.aper_phot[aper_diam].SED_results[SED_fit_code.label]. \
                        obs_zrange[self.crop_name.split("/")[-1]][data.full_name][0]
                        for gal in self
                        if gal.ID in update_IDs
                    ]
                )
                obs_zmax = np.array(
                    [
                        gal.aper_phot[aper_diam].SED_results[SED_fit_code.label]. \
                        obs_zrange[self.crop_name.split("/")[-1]][data.full_name][1]
                        for gal in self
                        if gal.ID in update_IDs
                    ]
                )
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
                for gal, Vmax in zip(self, np.array(Vmax_tab["Vmax"])):
                    gal._make_Vmax_storage(aper_diam, SED_fit_code, self.crop_name.split("/")[-1])
                    gal.aper_phot[aper_diam].SED_results[SED_fit_code.label]. \
                        V_max[self.crop_name.split("/")[-1]][data.full_name] = Vmax #* u.Mpc ** 3
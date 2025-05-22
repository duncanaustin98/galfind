#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:20:31 2023

@author: austind
"""

from __future__ import annotations

# from reproject import reproject_adaptive
from abc import ABC, abstractmethod
import contextlib
import glob
import json
import os
import subprocess
import sys
from matplotlib.colors import LinearSegmentedColormap
import time
import itertools
from matplotlib import cm
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    List,
    Type,
    Dict,
    Union,
    Tuple,
    Optional,
    NoReturn,
    TYPE_CHECKING,
)

try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

if TYPE_CHECKING:
    from . import Multiple_Filter, Region_Selector, Mask_Selector
    from .PSF import PSF

import astropy.units as u
import astropy.visualization as vis
import cv2
import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import convolve, convolve_fft
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Column, Table, hstack, vstack
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.wcs import WCS
from astroquery.gaia import Gaia
from joblib import Parallel, delayed, parallel_config
from matplotlib.colors import LogNorm, Normalize
from regions import Regions
from tqdm import tqdm

from . import Depths, SExtractor, config, galfind_logger
from . import useful_funcs_austind as funcs
from . import Masking
from .decorators import run_in_dir
from . import Filter, Multiple_Filter
from .Instrument import ACS_WFC, WFC3_IR, NIRCam, MIRI, Instrument  # noqa F501

morgan_version_to_dir = {
    "v8b": "mosaic_1084_wispfix",
    "v8c": "mosaic_1084_wispfix2",
    "v8d": "mosaic_1084_wispfix3",
    "v9": "mosaic_1084_wisptemp2",
    "v10": "mosaic_1084_wispscale",
    "v11": "mosaic_1084_wispnathan",
    "v12": "mosaic_1210_wispnathan",
    "v12test": "mosaic_1210_wispnathan_test", # not sure if this is needed?
    "v13": "mosaic_1293_wispnathan",
    "v14": "mosaic_1364_wispnathan",
}

class Band_Data_Base(ABC):
    def __init__(
        self,
        survey: str,
        version: str,
        im_path: str,
        im_ext: int,
        rms_err_path: Optional[str] = None,
        rms_err_ext: Optional[int] = None,
        wht_path: Optional[str] = None,
        wht_ext: Optional[int] = None,
        pix_scale: u.Quantity = 0.03 * u.arcsec,
        im_ext_name: Union[str, List[str]] = "SCI",
        rms_err_ext_name: Union[str, List[str]] = "ERR",
        wht_ext_name: Union[str, List[str]] = "WHT",
        use_galfind_err: bool = True,
        aper_diams: Optional[u.Quantity] = None,
    ):
        self.survey = survey
        self.version = version
        self.im_path = im_path
        self.im_ext = im_ext
        self.im_ext_name = im_ext_name
        self.rms_err_path = rms_err_path
        self.rms_err_ext = rms_err_ext
        self.rms_err_ext_name = rms_err_ext_name
        self.wht_path = wht_path
        self.wht_ext = wht_ext
        self.wht_ext_name = wht_ext_name
        self.pix_scale = pix_scale
        if aper_diams is not None:
            self.aper_diams = aper_diams
        self._psf_match = None
        # make rms error / wht maps using galfind if required
        if use_galfind_err:
            if (
                (self.rms_err_path is None or self.rms_err_ext is None)
                and self.wht_path is not None
                and self.wht_ext is not None
            ):
                # make rms_err from wht if rms_err is not available
                self._make_rms_err_from_wht()
            elif (
                (self.wht_path is None or self.wht_ext is None)
                and self.rms_err_path is not None
                and self.rms_err_ext is not None
            ):
                # make wht from rms_err if wht is not available
                self._make_wht_from_rms_err()
        else:
            self._use_galfind_err = False
        # ensure all paths/exts link to valid data
        self._check_data(
            incl_rms_err=(
                self.rms_err_path is not None and self.rms_err_ext is not None
            ),
            incl_wht=(self.wht_path is not None and self.wht_ext is not None),
        )

    @property
    @abstractmethod
    def instr_name(self) -> str:
        pass

    @property
    @abstractmethod
    def filt_name(self) -> str:
        pass

    @property
    @abstractmethod
    def ZP(self) -> float:
        pass

    @property
    def data_shape(self) -> Tuple[int, int]:
        return self.load_im()[0].shape

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.instr_name}/{self.filt_name})"
    
    def __str__(self) -> str:
        output_str = funcs.line_sep
        output_str += f"{repr(self)} {self.__class__.__name__.upper().replace('_', ' ')}:\n"
        output_str += funcs.band_sep
        output_str += f"SURVEY: {self.survey}\n"
        output_str += f"VERSION: {self.version}\n"
        output_str += f"PIX SCALE: {self.pix_scale}\n"
        output_str += f"ZP: {self.ZP}\n"
        output_str += f"SHAPE: {self.data_shape}\n"
        if hasattr(self, "aper_diams"):
            output_str += f"APERTURE DIAMETERS: {self.aper_diams}\n"
        for attr in ["im", "rms_err", "wht"]:
            if getattr(self, f"{attr}_path") is not None:
                output_str += f"{attr.upper().replace('_', ' ')} PATH: " \
                    + f"{getattr(self, f'{attr}_path')}[{getattr(self, f'{attr}_ext')}]\n"
        for attr in ["mask", "seg", "forced_phot"]:
            if hasattr(self, f"{attr}_args"):
                output_str += f"{attr.upper().replace('_', ' ')}" \
                    + f" PATH: {getattr(self, f'{attr}_path')}\n"
                output_str += f"{attr.upper().replace('_', ' ')}" \
                    + f" ARGS: {getattr(self, f'{attr}_args')}\n"
        if hasattr(self, "depth_args"):
            output_str += funcs.line_sep
            output_str += f"DEPTHS:\n"
            for aper_diam in self.aper_diams:
                output_str += funcs.band_sep
                output_str += f"{aper_diam}\n"
                output_str += f"MEDIAN DEPTH: {self.med_depth[aper_diam]}\n"
                output_str += f"MEAN DEPTH: {self.mean_depth[aper_diam]}\n"
                output_str += f"H5 PATH: {self.depth_path[aper_diam]}\n"
                output_str += f"ARGS: {self.depth_args[aper_diam]}\n"
                output_str += funcs.band_sep
        output_str += funcs.line_sep
        return output_str
        

    def __eq__(self, other: Type[Band_Data_Base]) -> bool:
        if not isinstance(other, tuple(Band_Data_Base.__subclasses__())):
            return False
        else:
            # check if all attributes are the same
            return (
                self.survey == other.survey
                and self.version == other.version
                and self.im_path == other.im_path
                and self.im_ext == other.im_ext
                and self.rms_err_path == other.rms_err_path
                and self.rms_err_ext == other.rms_err_ext
                and self.wht_path == other.wht_path
                and self.wht_ext == other.wht_ext
                and self.pix_scale == other.pix_scale
                and self.im_ext_name == other.im_ext_name
                and self.rms_err_ext_name == other.rms_err_ext_name
                and self.wht_ext_name == other.wht_ext_name
            )

    def __copy__(self) -> Type[Band_Data_Base]:
        # copy the object
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(result, k, v)
        return result

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

    def _check_data(self, incl_rms_err: bool = True, incl_wht: bool = True):
        # make im_ext_name lists if not already
        if isinstance(self.im_ext_name, str):
            self.im_ext_name = [self.im_ext_name]
        if isinstance(self.rms_err_ext_name, str):
            self.rms_err_ext_name = [self.rms_err_ext_name]
        if isinstance(self.wht_ext_name, str):
            self.wht_ext_name = [self.wht_ext_name]
        # load image header
        im_hdr = self.load_im()[1]
        assert im_hdr["EXTNAME"] in self.im_ext_name, galfind_logger.critical(
            f"Image extension name {im_hdr['EXTNAME']} "
            + f"not in {str(self.im_ext_name)} for {self.filt.band_name}"
        )
        if incl_rms_err:
            # load rms error header
            rms_err_hdr = self.load_rms_err(output_hdr=True)[1]
            assert (
                rms_err_hdr["EXTNAME"] in self.rms_err_ext_name
            ), galfind_logger.critical(
                f"RMS error extension name {rms_err_hdr['EXTNAME']} "
                + f"not in {str(self.rms_err_ext_name)} for {self.filt.band_name}"
            )
        if incl_wht:
            # load weight header
            wht_hdr = self.load_wht(output_hdr=True)[1]
            assert (
                wht_hdr["EXTNAME"] in self.wht_ext_name
            ), galfind_logger.critical(
                f"Weight extension name {wht_hdr['EXTNAME']} "
                + f"not in {str(self.wht_ext_name)} for {self.filt.band_name}"
            )

    # %% Loading methods

    def load_aper_diams(self, aper_diams: u.Quantity) -> NoReturn:
        if hasattr(self, "aper_diams"):
            galfind_logger.warning(
                f"{self.aper_diams=} already loaded for {self.filt_name},"
                + f" skipping {aper_diams=} load-in"
            )
        else:
            self.aper_diams = aper_diams
            galfind_logger.info(f"Loaded {aper_diams=} for {self.filt_name}")

    def load_data(self, incl_mask: bool = True):
        assert hasattr(self, "seg_args")
        # load science image data and header (and hdul)
        im_data, im_header = self.load_im()
        # load segmentation data and header
        seg_data, seg_header = self.load_seg()
        if incl_mask:
            mask = self.load_mask()
            return im_data, im_header, seg_data, seg_header, mask
        else:
            return im_data, im_header, seg_data, seg_header

    def load_im(
        self, return_hdul: bool = False
    ) -> Union[
        Tuple[np.ndarray, fits.Header],
        Tuple[np.ndarray, fits.Header, fits.HDUList],
    ]:
        # load image data and header
        if not Path(self.im_path).is_file():
            err_message = (
                f"Image for {self.survey} {self.filt.band_name}"
                + f" at {self.im_path} is not a .fits image!"
            )
            galfind_logger.critical(err_message)
            raise (Exception(err_message))
        im_hdul = fits.open(self.im_path, ignore_missing_simple = True)
        im_data = im_hdul[self.im_ext].data
        # im_data = im_data.byteswap().newbyteorder() slow
        im_header = im_hdul[self.im_ext].header
        if return_hdul:
            return im_data, im_header, im_hdul
        else:
            return im_data, im_header

    def load_wcs(self) -> WCS:
        try:
            self.wcs
        except (AttributeError, KeyError) as e:
            if type(e) == AttributeError:
                self.wcs = {}
            self.wcs = WCS(self.load_im()[1])
        return self.wcs

    def load_wht(
        self, output_hdr: bool = False
    ) -> Union[Tuple[np.ndarray, fits.Header], np.ndarray]:
        if Path(self.wht_path).is_file():
            hdu = fits.open(self.wht_path, ignore_missing_simple = True)[self.wht_ext]
            wht = hdu.data
            hdr = hdu.header
        else:
            err_message = (
                f"Weight image for {self.survey} {self.filt.band_name}"
                + f" at {self.wht_path} is not a .fits image!"
            )
            galfind_logger.critical(err_message)
            wht = None
            hdr = None
        if output_hdr:
            return wht, hdr
        else:
            return wht

    def load_rms_err(   
        self, output_hdr: bool = False
    ) -> Union[Tuple[np.ndarray, fits.Header], np.ndarray]:
        if Path(self.rms_err_path).is_file():
            hdu = fits.open(self.rms_err_path, ignore_missing_simple = True)[self.rms_err_ext]
            rms_err = hdu.data
            hdr = hdu.header
        else:
            err_message = (
                f"RMS error for {self.survey} {self.filt.band_name}"
                + f" at {self.rms_err_path} is not a .fits image!"
            )
            galfind_logger.critical(err_message)
            rms_err = None
            hdr = None
        if output_hdr:
            return rms_err, hdr
        else:
            return rms_err

    def load_seg(
        self, incl_hdr: bool = True
    ) -> Tuple[np.ndarray, fits.Header]:
        # TODO: load from the correct hdu rather than the first one
        if not Path(self.seg_path).is_file():
            err_message = (
                f"Segmentation map for {self.survey} "
                f"{self.filt.band_name} at {self.seg_path} is not a .fits image!"
            )
            galfind_logger.critical(err_message)
            raise (Exception(err_message))
        seg_hdul = fits.open(self.seg_path, ignore_missing_simple = True)
        seg_data = seg_hdul[0].data
        seg_header = seg_hdul[0].header
        if incl_hdr:
            return seg_data, seg_header
        else:
            return seg_data

    def load_mask(
        self: Self,
        ext: Optional[str] = None,
        invert: bool = False,
    ) -> Optional[Union[np.ndarray, Dict[str, np.ndarray]]]:
        if hasattr(self, "mask_args"):
            # load mask
            if ".fits" in self.mask_path:
                hdul = fits.open(self.mask_path, mode = "readonly", ignore_missing_simple = True)
                hdu_names_indices = {
                    hdu.name.upper(): i
                    for i, hdu in enumerate(hdul)
                    if hdu.name != "PRIMARY"
                }
                if ext is not None:
                    ext = ext.upper()
                    assert (
                        ext in hdu_names_indices.keys()
                    ), galfind_logger.critical(
                        f"{ext=} not in mask extensions: {list(hdu_names_indices.keys())}"
                    )
                    mask = hdul[hdu_names_indices[ext]].data
                    if invert:
                        mask = (~mask.astype(bool)).astype(int)
                    hdr = hdul[hdu_names_indices[ext]].header
                else:
                    mask = {
                        hdu_name: hdul[index].data
                        for hdu_name, index in hdu_names_indices.items()
                    }
                    hdr = {
                        hdu_name: hdul[index].header
                        for hdu_name, index in hdu_names_indices.items()
                    }
            else:
                galfind_logger.critical(
                    f"Mask for {self.survey} {self.filt_name}"
                    + f" at {self.mask_path} is not a .fits mask!"
                )
        else:
            galfind_logger.critical(
                f"Mask for {self.survey} {self.filt_name} not set!"
            )
            mask = None
            hdr = None
        return mask, hdr

    # %% Complex methods

    def psf_homogenize(self, psf: PSF):
        """Homogenize the SCI/RMS_ERR/WHT images to the given PSF"""
        raise NotImplementedError
        self._psf_match = psf
        # Functionality in PSF_homogenization.py

    def segment(
        self,
        err_type: str = "rms_err",
        method: str = "sextractor",
        config_name: str = "default.sex",
        params_name: str = "default.param",
        overwrite: bool = False,
    ) -> NoReturn:
        """
        Segment the image using the specified method and error type
        if it has not already been done.

        Parameters:
        -----------
        err_type : str, optional
            The type of error to use for segmentation. Default is "rms_err".
        method : str, optional
            The segmentation method to use. Default is "sextractor".
        overwrite : bool, optional
            Whether to overwrite existing segmentation data. Default is False.

        Returns:
        --------
        NoReturn
            This method does not return any value.

        Notes:
        ------
        - If the method is "sextractor", it uses the Segmentation.segment_sextractor function.
        - The segmentation arguments used here stored in the `seg_args` attribute.
        - The `overwrite` parameter determines if existing segmentation data should be replaced.
            Segments the data using the specified method and error type.
        """
        # do not re-segment if already done
        if not (hasattr(self, "seg_args") and hasattr(self, "seg_path")):
            # segment the data
            if "sextractor" in method.lower():
                self.seg_path = SExtractor.segment_sextractor(
                    self,
                    err_type,
                    config_name=config_name,
                    params_name=params_name,
                    overwrite=overwrite,
                )
            else:
                raise (
                    Exception(f"segmentation {method.lower()=} does not contain 'sextractor'")
                )
            self.seg_args = {
                "err_type": err_type,
                "method": method,
                "config_name": config_name,
                "params_name": params_name,
            }

    def _perform_forced_phot(
        self,
        forced_phot_band: Type[Band_Data_Base],
        err_type: str = "rms_err",
        method: str = "sextractor",
        config_name: str = "default.sex",
        params_name: str = "default.param",
        overwrite: bool = False,
    ) -> NoReturn:
        # do not re-perform forced photometry if already done
        if not (
            hasattr(self, "forced_phot_args")
            and hasattr(self, "forced_phot_path")
        ):
            if "sextractor" in method.lower():
                self.forced_phot_path, self.forced_phot_args = \
                    SExtractor.perform_forced_phot(
                        self,
                        forced_phot_band,
                        err_type,
                        config_name=config_name,
                        params_name=params_name,
                        overwrite=overwrite,
                    )
            else:
                raise (Exception(f"{method.lower()=} does not contain 'sextractor'"))

    def _get_master_tab(
        self, output_ids_locs: bool = False
    ) -> Table:
        tab = Table.read(self.forced_phot_path, character_as_bytes=False, format="fits")
        if "sextractor" in self.forced_phot_args["method"].lower():
            id_loc_params = [
                "NUMBER",
                "X_IMAGE",
                "Y_IMAGE",
                "ALPHA_J2000",
                "DELTA_J2000",
            ]
        else:
            raise (
                Exception(
                    f"{self.forced_phot_args['method'].lower()} does not contain 'sextractor'"
                )
            )
        if output_ids_locs:
            if "sextractor" in self.forced_phot_args["method"].lower():
                append_ids_loc = {
                    "ID": tab["NUMBER"],
                    "X_IMAGE": tab["X_IMAGE"].value,
                    "Y_IMAGE": tab["Y_IMAGE"].value,
                    "RA": tab["ALPHA_J2000"].value,
                    "DEC": tab["DELTA_J2000"].value,
                }
        else:
            append_ids_loc = {}

        # remove non band-dependent forced photometry parameters
        for param in id_loc_params:
            if not output_ids_locs:
                tab.remove_column(param)
        # add band suffix to columns
        for name in tab.columns.copy():
            if name not in id_loc_params:
                tab.rename_column(name, name + "_" + self.filt_name)
        return tab


    def mask(
        self,
        method: Union[str, List[str], Dict[str, str]] = "auto",
        fits_mask_path: Optional[Union[str, List[str], Dict[str, str]]] = None,
        star_mask_params: Optional[
            Union[
                Dict[str, Dict[str, float]],
                Dict[u.Quantity, Dict[str, Dict[str, float]]],
            ]
        ] = {
            "central": {"a": 300.0, "b": 4.25},
            "spikes": {"a": 400.0, "b": 4.5},
        },
        edge_mask_distance: Union[
            int, float, List[Union[int, float]], Dict[str, Union[int, float]]
        ] = 50,
        scale_extra: Union[float, List[float], Dict[str, float]] = 0.2,
        exclude_gaia_galaxies: Union[bool, List[bool], Dict[str, bool]] = True,
        angle: Union[float, List[float], Dict[str, float]] = 0.0,
        edge_value: Union[float, List[float], Dict[str, float]] = 0.0,
        element: Union[str, List[str], Dict[str, str]] = "ELLIPSE",
        gaia_row_lim: Union[int, List[int], Dict[str, int]] = 500,
        overwrite: Union[bool, List[bool], Dict[str, bool]] = False,
    ) -> Union[None, NoReturn]:
        if not (hasattr(self, "mask_args") and hasattr(self, "mask_path")):
            # load in already made fits mask
            if fits_mask_path is not None:
                mask_args = Masking.get_mask_method(fits_mask_path)
                if mask_args is not None:
                    self.mask_path = fits_mask_path
                    self.mask_args = {"method": mask_args}
                    return
            # create fits mask
            if method.lower() == "manual":
                self.mask_path = Masking.manually_mask(
                    self, overwrite=overwrite
                )
                self.mask_args = {"method": method}
            elif method.lower() == "auto":
                self.mask_path, self.mask_args = Masking.auto_mask(
                    self,
                    star_mask_params,
                    edge_mask_distance,
                    scale_extra,
                    exclude_gaia_galaxies,
                    angle,
                    edge_value,
                    element,
                    gaia_row_lim,
                    overwrite,
                )
            else:
                raise (
                    Exception(
                        f"Invalid masking method {method} (not in ['auto', 'manual'])"
                    )
                )

    def run_depths(
        self: Self,
        mode: str = "n_nearest",
        scatter_size: float = 0.1,
        distance_to_mask: Union[int, float] = 30,
        region_radius_used_pix: Union[int, float] = 300,
        n_nearest: int = 200,
        coord_type: str = "sky",
        split_depth_min_size: int = 100_000,
        split_depths_factor: int = 5,
        step_size: int = 100,
        n_split: Union[str, int] = "auto",
        n_retry_box: int = 1,
        grid_offset_times: int = 1,
        plot: bool = True,
        overwrite: bool = False,
        master_cat_path: Optional[str] = None,
    ) -> NoReturn:
        if not hasattr(self, "depth_args"):
            # load parameters (i.e. for each aper_diams in self)
            params_arr = self._sort_run_depth_params(
                mode,
                scatter_size,
                distance_to_mask,
                region_radius_used_pix,
                n_nearest,
                coord_type,
                split_depth_min_size,
                split_depths_factor,
                step_size,
                n_split,
                n_retry_box,
                grid_offset_times,
                overwrite,
                master_cat_path,
            )
            # run depths
            for params in params_arr:
                Depths.calc_band_depth(params)
            # load depths into object
            self._load_depths_from_params(params_arr)
            # plot depths
            if plot:
                self.plot_depth_diagnostics(save = True, overwrite = False, master_cat_path = master_cat_path)
        else:
            galfind_logger.warning(
                f"Depths loaded for {self.filt_name}, skipping!"
            )
    
    def get_hf_output(self, aper_diam: u.Quantity) -> Dict[str, Any]:
        return Depths.get_hf_output(self, aper_diam)

    def _sort_run_depth_params(
        self,
        mode: str = "n_nearest",
        scatter_size: float = 0.1,
        distance_to_mask: Union[int, float] = 30,
        region_radius_used_pix: Union[int, float] = 300,
        n_nearest: int = 200,
        coord_type: str = "sky",
        split_depth_min_size: int = 100_000,
        split_depths_factor: int = 5,
        step_size: int = 100,
        n_split: Union[str, int] = "auto",
        n_retry_box: int = 1,
        grid_offset_times: int = 1,
        overwrite: bool = False,
        master_cat_path: Optional[str] = None,
    ) -> List[Tuple[Any, ...]]:
        params = []
        for aper_diam in self.aper_diams:
            params.extend(
                [
                    (
                        self,
                        aper_diam,
                        mode,
                        scatter_size,
                        distance_to_mask,
                        region_radius_used_pix,
                        n_nearest,
                        coord_type,
                        split_depth_min_size,
                        split_depths_factor,
                        step_size,
                        n_split,
                        n_retry_box,
                        grid_offset_times,
                        overwrite,
                        master_cat_path,
                    )
                ]
            )
        return params

    def _load_depths_from_params(
        self: Self, 
        params: List[Tuple[Any, ...]],
    ) -> NoReturn:
        if hasattr(self, "depth_args"):
            if all(param[1] in self.depth_args.keys() for param in params):
                galfind_logger.warning(
                    f"Depth data already loaded for {self.filt_name}, skipping load-in"
                )
        else:
            self.depth_path = {
                param[1]: Depths.get_grid_depth_path(self, param[1], param[2])
                for param in params
            }
            depths = [
                Depths.get_depths_from_h5(self, param[1], param[2])
                for param in params
            ]
            for depth, param in zip(depths, params):
                for key in depth[0].keys():
                    self._update_depths(param[1], depth[0][key], depth[1][key], key)
            self.depth_args = {
                param[1]: Depths.get_depth_args(param) for param in params
            }

    def _update_depths(
        self: Self,
        aper_diam: u.Quantity,
        med_depth: float,
        mean_depth: float,
        label: str,
    ):
        if not hasattr(self, "med_depth"):
            self.med_depth = {}
            if aper_diam not in self.med_depth.keys():
                self.med_depth[aper_diam] = {}
        if not hasattr(self, "mean_depth"):
            self.mean_depth = {}
            if aper_diam not in self.mean_depth.keys():
                self.mean_depth[aper_diam] = {}
        self.med_depth[aper_diam][label] = med_depth
        self.mean_depth[aper_diam][label] = mean_depth
    

    def _load_depths(
        self: Self,
        aper_diam: u.Quantity,
        mode: str,
        region: str = "all", 
    ) -> NoReturn:
        params = (aper_diam, mode, region)
        return self._load_depths_from_params([params])

    def plot_depths(
        self,
        aper_diam: u.Quantity,
        plot_type: str,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        save: bool = False,
        show: bool = True,
        cmap_name: str = "plasma",
        label_suffix: Optional[str] = None,
        title: Optional[str] = None,
    ) -> NoReturn:
        assert aper_diam in self.aper_diams, \
            galfind_logger.critical(
                f"{aper_diam=} not in {self.aper_diams} for {self.filt_name}"
            )
        assert plot_type in [
            "rolling_average", 
            "rolling_average_diag", 
            "labels", 
            "hist", 
            "cat_depths", 
            "cat_diag"
            ], galfind_logger.critical(
                f"{plot_type=} not valid!"
            )
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        hf_output = Depths.get_hf_output(self, aper_diam)
        if plot_type.lower() == "rolling_average":
            Depths._plot_rolling_average(fig, ax, hf_output, cm.get_cmap(cmap_name))
        elif plot_type.lower() == "rolling_average_diag":
            Depths._plot_rolling_average_diagnostic(fig, ax, hf_output, cm.get_cmap(cmap_name))
        elif plot_type.lower() in ["labels", "hist"]:
            num_labels = len(np.unique(hf_output["labels_grid"]))
            labels_cmap = LinearSegmentedColormap.from_list
            (
                "custom",
                [cm.get_cmap("Set2")(i / num_labels) for i in range(num_labels)],
                num_labels,
            )
            if plot_type.lower() == "labels":
                Depths._plot_labels(fig, ax, hf_output, labels_cmap)
            else:
                labels_arr, possible_labels, colours = \
                    Depths._get_labels(hf_output, cmap = labels_cmap, cmap_name = cmap_name)
                Depths._plot_depth_hist(fig, ax, hf_output, labels_arr, 
                    possible_labels, colours, annotate = True if show or save else False,
                    label_suffix = label_suffix, title = title)
        elif plot_type.lower() in ["cat_depths", "cat_diag"]:
            cmap = cm.get_cmap(cmap_name)
            cmap.set_bad(color="black")
            cat_x, cat_y = Depths.get_cat_xy(hf_output)
            combined_mask = Depths._combine_seg_data_and_mask(self)
            if plot_type.lower() == "cat_depths":
                Depths._plot_cat_depths(fig, ax, hf_output, cmap, cat_x, cat_y, combined_mask)
            else:
                Depths._plot_cat_diagnostic(fig, ax, hf_output, cmap, cat_x, cat_y, combined_mask)
    
        if save:
            label = Depths.get_depth_dir(self, aper_diam, self.depth_args[aper_diam]['mode']) \
                + f"/{plot_type.lower()}/{self.filt_name}.png"
            funcs.make_dirs(label)
            plt.savefig(label)
            galfind_logger.info(f"Saved plot to {label}")
        if show:
            plt.show()


    def plot_depth_diagnostic(
        self,
        aper_diam: u.Quantity,
        save: bool = False,
        show: bool = False,
        overwrite: bool = True,
        master_cat_path: Optional[str] = None,
    ) -> NoReturn:
        save_path = Depths.get_depth_plot_path(self, aper_diam)
        if not Path(save_path).is_file() or overwrite:
            Depths.plot_depth_diagnostic(
                self,
                aper_diam,
                save = save,
                show = show,
                master_cat_path = master_cat_path,
            )
    
    def plot_depth_diagnostics(
        self,
        save: bool = False,
        overwrite: bool = True,
        master_cat_path: Optional[str] = None,
    ) -> NoReturn:
        for aper_diam in self.aper_diams:
            self.plot_depth_diagnostic(
                aper_diam,
                save = save,
                overwrite = overwrite,
                master_cat_path = master_cat_path
            )

    def plot_area_depth(
        self,
    ) -> NoReturn:
        Depths.plot_band_data_area_depth(self)

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        ext: str = "SCI",
        norm: Type[Normalize] = LogNorm(vmin=0.0, vmax=10.0),
        cmap: str = "plasma",
        save: bool = False,
        show: bool = True,
    ) -> NoReturn:
        """
        Plots the specified image data on the given matplotlib Axes.

        Parameters:
        -----------
        ax : plt.Axes
            The matplotlib Axes object where the image will be plotted.
        ext : str, optional
            The type of image data to plot. Must be one of ['SCI', 'RMS_ERR', 'WHT', 'SEG', 'MASK'].
            Default is 'SCI'.
        norm : Type[Normalize], optional
            The normalization for the image data. Default is LogNorm(vmin=0.0, vmax=10.0).
        save : bool, optional
            If True, the plot will be saved to a file. Default is False.
        show : bool, optional
            If True, the plot will be displayed. Default is True.

        Raises:
        -------
        Exception
            If the provided extension `ext` is not one of the allowed values.

        Returns:
        --------
        NoReturn
        """
        normalize = True
        if ext.lower() in ["sci", "im"]:
            data = self.load_im()[0]
        elif ext.lower() == "rms_err":
            data = self.load_rms_err()[0]
        elif ext.lower() == "wht":
            data = self.load_wht()[0]
        elif ext.lower() == "seg":
            data = self.load_seg()[0]
        elif ext.lower() == "mask":
            # TODO: plot different masks in different colours
            data = self.load_mask()[0]["MASK"]
            normalize = False
        else:
            raise (
                Exception(
                    f"Invalid extension {ext}. "
                    + "Must be one of ['SCI', 'RMS_ERR', 'WHT', 'SEG', 'MASK']"
                )
            )
        # make a fresh axis if one is not provided
        if ax is None:
            fig, ax = plt.subplots()
        # plot the image data
        if normalize:
            ax.imshow(data, norm=norm, cmap=cmap, origin="lower")
        else:
            ax.imshow(data, cmap=cmap, origin="lower")
        # annotate if required
        if show or save:
            plt.title(ext.upper())
            ax.set_xlabel("X / pix")
            ax.set_ylabel("Y / pix")
        if save:
            label = None
            # plt.savefig(label)
        if show:
            plt.show()

    def _combine_seg_data_and_mask(self) -> np.ndarray:
        seg_data = self.load_seg()[0]
        mask = self.load_mask()[0]["MASK"]
        assert seg_data.shape == mask.shape, \
            galfind_logger.critical(
                f"{self.seg_path=} with {self.seg_data.shape=} != {self.mask_path=} with {mask.shape=}"
            )
        combined_mask = np.logical_or(seg_data > 0, mask == 1).astype(int)
        return combined_mask

    @staticmethod
    def _pix_scale_to_str(pix_scale: u.Quantity):
        return f"{round(pix_scale.to(u.marcsec).value)}mas"

    def _make_rms_err_from_wht(self, overwrite: bool = False) -> NoReturn:
        save_path = self.im_path.replace(
            self.im_path.split("/")[-1],
            f"rms_err/{self.filt.band_name}_rms_err.fits",
        )
        if not Path(save_path).is_file() or overwrite:
            # make rms_err map from wht map
            wht, hdr = self.load_wht(output_hdr=True)
            err = 1.0 / (wht**0.5)
            primary_hdr = deepcopy(hdr)
            primary_hdr["EXTNAME"] = "PRIMARY"
            primary = fits.PrimaryHDU(header=primary_hdr)
            hdu = fits.ImageHDU(err, header=hdr, name="ERR")
            hdul = fits.HDUList([primary, hdu])
            # save and overwrite object attributes
            funcs.make_dirs(save_path)
            hdul.writeto(save_path, overwrite=True)
            funcs.change_file_permissions(save_path)
            galfind_logger.info(
                f"Finished making {self.survey} {self.version} {self.filt} rms_err map"
            )
        galfind_logger.info(
            f"Loading galfind created rms_err for {self.filt_name}"
        )
        self.rms_err_path = save_path
        self.rms_err_ext = 1
        self.rms_err_ext_name = ["ERR"]
        self._use_galfind_err = True

    def _make_wht_from_rms_err(self, overwrite: bool = False) -> NoReturn:
        save_path = self.im_path.replace(
            self.im_path.split("/")[-1], f"wht/{self.filt.band_name}_wht.fits"
        )
        if not Path(save_path).is_file() or overwrite:
            err, hdr = self.load_rms_err(output_hdr=True)
            wht = 1.0 / (err**2)
            primary_hdr = deepcopy(hdr)
            primary_hdr["EXTNAME"] = "PRIMARY"
            primary = fits.PrimaryHDU(header=primary_hdr)
            hdu = fits.ImageHDU(wht, header=hdr, name="WHT")
            hdul = fits.HDUList([primary, hdu])
            # save and overwrite object attributes
            funcs.make_dirs(save_path)
            hdul.writeto(save_path, overwrite=True)
            funcs.change_file_permissions(save_path)
            galfind_logger.info(
                f"Finished making {self.survey} {self.version} {self.filt} wht map"
            )
        galfind_logger.info(
            f"Loading galfind created wht for {self.filt_name}"
        )
        self.wht_path = save_path
        self.wht_ext = 1
        self.wht_ext_name = ["WHT"]
        self._use_galfind_err = True

    # can be simplified with new masks
    def calc_unmasked_area(
        self: Self,
        mask_type: str = "All",
    ) -> NoReturn:
        # calculate areas for given mask
        if mask_type == "All":
            masks = self.load_mask()[0]
            for mask_name, mask in masks.items():
                self._calc_area_given_mask(mask_name, mask)
        else:
            mask_types = mask_type.split("+")
            if len(mask_types) == 1:
                self._calc_area_given_mask(mask_type)
            elif len(mask_types) > 1:
                masks = tuple([self.load_mask()[0][mask_type] for mask_type in mask_types])
                self._calc_area_given_mask("+".join(np.sort(mask_types)), masks)
    
    def _calc_area_given_mask(
        self,
        mask_name: str,
        mask: Optional[np.ndarray, Tuple[np.ndarray]] = None,
    ) -> NoReturn:
        if not hasattr(self, "unmasked_area"):
            self.unmasked_area = {}
        if mask_name not in self.unmasked_area.keys():
            # load mask
            if mask is None:
                mask = self.load_mask()[0][mask_name.upper()]
            if isinstance(mask, tuple):
                mask = np.logical_or.reduce(mask)
            # ensure mask is the same shape as your imaging
            assert mask.shape == self.data_shape, galfind_logger.critical(
                f"{mask_name=} shape {mask.shape} != {self.data_shape=}"
            )
            # calculate unmasked area
            unmasked_area = funcs.calc_unmasked_area(mask, self.pixel_scale)
            self.unmasked_area[mask_name.upper()] = unmasked_area

class Band_Data(Band_Data_Base):
    def __init__(
        self,
        filt: Type[Filter],
        survey: str,
        version: str,
        im_path: str,
        im_ext: int,
        rms_err_path: Optional[str] = None,
        rms_err_ext: Optional[int] = None,
        wht_path: Optional[str] = None,
        wht_ext: Optional[int] = None,
        pix_scale: u.Quantity = 0.03 * u.arcsec,
        im_ext_name: Union[str, List[str]] = "SCI",
        rms_err_ext_name: Union[str, List[str]] = "ERR",
        wht_ext_name: Union[str, List[str]] = "WHT",
        use_galfind_err: bool = True,
        aper_diams: Optional[u.Quantity] = None,
    ):
        self.filt = filt
        super().__init__(
            survey,
            version,
            im_path,
            im_ext,
            rms_err_path,
            rms_err_ext,
            wht_path,
            wht_ext,
            pix_scale,
            im_ext_name,
            rms_err_ext_name,
            wht_ext_name,
            use_galfind_err,
            aper_diams,
        )

    @classmethod
    def from_band_data_arr(cls, band_data_arr: List[Type[Band_Data_Base]]):
        raise (NotImplementedError)
        # make sure all filters are the same
        # stack bands by multiplication

    @property
    def instr_name(self):
        return self.filt.instrument.__class__.__name__

    @property
    def filt_name(self):
        return self.filt.band_name

    @property
    def ZP(self) -> Dict[str, float]:
        return float(self.filt.instrument.calc_ZP(self))

    def __add__(
        self, other: Union[Band_Data, List[Band_Data], Data, List[Data]]
    ) -> Data:
        # if other is not a list, make it one
        if not isinstance(other, list):
            other = [other]
        # if other is an array of data objects, make a list of band_data objects
        if isinstance(other[0], Data):
            assert all(isinstance(_other, Data) for _other in other)
            other_band_data = []
            for _other in other:
                other_band_data.extend(_other.band_data_arr)
            other = other_band_data
        new_band_data_arr = [self] + other
        # ensure all bands come from the same survey and version
        if all(
            [
                band_data.survey == self.survey
                and band_data.version == self.version
                for band_data in new_band_data_arr
            ]
        ):
            # if all bands being added are different
            if len(
                np.unique(
                    [band_data.filt_name for band_data in new_band_data_arr]
                )
            ) == len(new_band_data_arr):
                return Data(new_band_data_arr)
            else:
                raise (
                    Exception(
                        "Cannot add Data/Band_Data objects with the same filters."
                        + " You may want to use Band_Data.__mul__() to stack!"
                    )
                )
        else:
            raise (
                Exception(
                    "Cannot add Data/Band_Data objects from different surveys or versions."
                )
            )

    # stacking/mosaicing
    def __mul__(
        self, other: Union[Type[Band_Data_Base], List[Type[Band_Data_Base]]]
    ) -> Type[Band_Data_Base]:
        # if other is not a list, make it one
        if not isinstance(other, list):
            other = [other]
        assert all(
            isinstance(_other, tuple(Band_Data_Base.__subclasses__()))
            for _other in other
        )
        # flatten array of other band_data objects
        band_data_arr = []
        for _other in other:
            if isinstance(_other, Band_Data):
                band_data_arr.extend([_other])
            elif isinstance(_other, Stacked_Band_Data):
                assert hasattr(other, "band_data_arr")
                band_data_arr.extend(_other.band_data_arr)
        # stack/mosaic bands
        if all(band_data.filt == self.filt for band_data in band_data_arr):
            return Band_Data.from_band_data_arr(
                [deepcopy(self), *band_data_arr]
            )
        else:
            return Stacked_Band_Data.from_band_data_arr(
                [deepcopy(self), *band_data_arr]
            )


class Stacked_Band_Data(Band_Data_Base):
    def __init__(
        self,
        filterset: Union[List[Filter], Multiple_Filter],
        survey: str,
        version: str,
        im_path: str,
        im_ext: int,
        rms_err_path: Optional[str] = None,
        rms_err_ext: Optional[int] = None,
        wht_path: Optional[str] = None,
        wht_ext: Optional[int] = None,
        pix_scale: u.Quantity = 0.03 * u.arcsec,
        im_ext_name: Union[str, List[str]] = "SCI",
        rms_err_ext_name: Union[str, List[str]] = "ERR",
        wht_ext_name: Union[str, List[str]] = "WHT",
        use_galfind_err: bool = True,
        aper_diams: Optional[u.Quantity] = None,
    ):
        # ensure every band_data is from the same survey and version,
        # have the same pixel scale and are from different filters
        self.filterset = filterset
        super().__init__(
            survey,
            version,
            im_path,
            im_ext,
            rms_err_path,
            rms_err_ext,
            wht_path,
            wht_ext,
            pix_scale,
            im_ext_name,
            rms_err_ext_name,
            wht_ext_name,
            use_galfind_err,
            aper_diams,
        )

    @classmethod
    def from_band_data_arr(
        cls, band_data_arr: List[Band_Data], err_type: str = "rms_err"
    ) -> Stacked_Band_Data:
        # make sure all filters are different
        assert all(
            band_data.filt_name != band_data_arr[0].filt_name
            for i, band_data in enumerate(band_data_arr)
            if i != 0
        )

        # TODO: if all band_data in band_data_arr have been PSF homogenized, update the stacking path names

        # stack bands
        input_data = Stacked_Band_Data._stack_band_data(
            band_data_arr, err_type=err_type
        )
        # make filterset from filters
        filterset = Multiple_Filter(
            [band_data.filt for band_data in band_data_arr]
        )
        # instantiate the stacked band data object
        stacked_band_data = cls(filterset, **input_data)

        # if all band_data in band_data_arr have aper_diams included
        if all(hasattr(band_data, "aper_diams") for band_data in band_data_arr):
            if all(all(diam == diam_0 for diam, diam_0 in zip(band_data.aper_diams, band_data_arr[0].aper_diams)) for band_data in band_data_arr):
                stacked_band_data.load_aper_diams(band_data_arr[0].aper_diams)

        # if all band_data in band_data_arr have been segmented, segment the stacked band data
        if all(hasattr(band_data, "seg_args") for band_data in band_data_arr):
            stacked_band_data.segment()

        # if all band_data in band_data_arr have been masked, mask the stacked band data
        if all(hasattr(band_data, "mask_args") for band_data in band_data_arr):
            # if all mask arguments are the same, use the same mask method
            # as for the individual bands
            if all(
                band_data.mask_args == band_data_arr[0].mask_args
                for band_data in band_data_arr
            ):
                stacked_band_data.mask(**band_data_arr[0].mask_args)
            else:
                # perform default masking
                stacked_band_data.mask()

        # TODO: if all band_data in band_data_arr have run depths,
        # run depths for the stacked band data

        # TODO: if all band_data in band_data_arr have performed
        # forced photometry, perform forced photometry for the stacked band data

        # save original band_data inputs in the class, sorted blue -> red
        stacked_band_data.band_data_arr = funcs.sort_band_data_arr(
            band_data_arr
        )
        return stacked_band_data

    @property
    def instr_name(self) -> str:
        return self.filterset.instrument_name

    @property
    def filt_name(self) -> str:
        return self._get_stacked_band_data_name(self.filterset)

    @property
    def ZP(self) -> Dict[str, float]:
        assert all(
            filt.instrument.calc_ZP(self)
            == self.filterset[0].instrument.calc_ZP(self)
            for filt in self.filterset
        )
        return float(self.filterset[0].instrument.calc_ZP(self))

    # stacking/mosaicing
    def __mul__(
        self, other: Union[Type[Band_Data_Base], List[Type[Band_Data_Base]]]
    ) -> Type[Band_Data_Base]:
        # if other is not a list, make it one
        if not isinstance(other, list):
            other = [other]
        assert all(
            isinstance(_other, tuple(Band_Data_Base.__subclasses__()))
            for _other in other
        )
        # flatten array of other band_data objects
        band_data_arr = []
        for _other in other:
            if isinstance(_other, Band_Data):
                band_data_arr.extend([_other])
            elif isinstance(_other, Stacked_Band_Data):
                assert hasattr(other, "band_data_arr")
                band_data_arr.extend(_other.band_data_arr)

        return Stacked_Band_Data.from_band_data_arr(
            [*deepcopy(self).band_data_arr, *band_data_arr]
        )

    @staticmethod
    def _get_stacked_band_data_name(
        filterset: Union[List[Filter], Multiple_Filter],
    ) -> str:
        return "+".join([filt.band_name for filt in filterset])

    @staticmethod
    def _get_stacked_band_data_path(
        band_data_arr: List[Band_Data], err_type: str = "rms_err"
    ) -> str:
        assert all(
            getattr(band_data, name) == getattr(band_data_arr[0], name)
            for name in ["survey", "version", "pix_scale"]
            for band_data in band_data_arr
        )
        # make stacked band data path, creating directory if it does not exist
        stacked_band_data_dir = (
            f"{config['DEFAULT']['GALFIND_WORK']}/Stacked_Images/"
            + f"{band_data_arr[0].version}/{band_data_arr[0].instr_name}/{band_data_arr[0].survey}/{err_type.lower()}"
        )
        stacked_band_data_name = (
            f"{band_data_arr[0].survey}_"
            + Stacked_Band_Data._get_stacked_band_data_name(
                [band_data.filt for band_data in band_data_arr]
            )
            + f"_{band_data_arr[0].version}_stack.fits"
        )
        stacked_band_data_path = (
            f"{stacked_band_data_dir}/{stacked_band_data_name}"
        )
        funcs.make_dirs(stacked_band_data_path)
        return stacked_band_data_path

    @staticmethod
    def _stack_band_data(
        band_data_arr: List[Band_Data],
        err_type: str = "rms_err",
        overwrite: bool = False,
    ) -> Tuple[str, Dict[str, Union[str, int]]]:
        assert err_type.lower() in ["rms_err", "wht"], galfind_logger.critical(
            f"{err_type=} not in ['rms_err', 'wht']"
        )

        # make rms_err/wht maps if they do not exist and are required
        # used_galfind_err = False
        if err_type.lower() == "rms_err":
            if any(band_data.rms_err_path is None for band_data in band_data_arr):
                run = True
            elif not all(
                Path(band_data.rms_err_path).is_file()
                for band_data in band_data_arr
            ):
                run = True
            else:
                run = False
            if run:
                for band_data in band_data_arr:
                    band_data._make_rms_err_from_wht()
                # used_galfind_err = True
        else:  # err_type.lower() == "wht"
            if any(band_data.wht_path is None for band_data in band_data_arr):
                run = True
            elif not all(
                Path(band_data.wht_path).is_file()
                for band_data in band_data_arr
            ):
                run = True
            else:
                run = False
            if run:
                for band_data in band_data_arr:
                    band_data._make_wht_from_rms_err()
                # used_galfind_err = True
        # load output path and perform stacking if required
        stacked_band_data_path = Stacked_Band_Data._get_stacked_band_data_path(
            band_data_arr, err_type
        )
        if not Path(stacked_band_data_path).is_file() or overwrite:
            # ensure all shapes are the same for the band data images
            assert all(
                band_data.data_shape == band_data_arr[0].data_shape
                for band_data in band_data_arr
            ), galfind_logger.critical(
                "All band data images in stacking bands must have the same shape!"
            )
            # ensure all band data images have the same ZP
            assert all(
                band_data.ZP == band_data_arr[0].ZP
                for band_data in band_data_arr
            ), galfind_logger.critical(
                "All image ZPs must be the same!"
            )
            # ensure all band data images have the same pixel scale
            assert all(
                band_data.pix_scale == band_data_arr[0].pix_scale
                for band_data in band_data_arr
            ), galfind_logger.critical(
                "All image pixel scales must be the same!"
            )
            # stack band data SCI/ERR/WHT images (inverse variance weighted)
            galfind_logger.info(
                f"Stacking {[band_data.filt_name for band_data in band_data_arr]}"
                + f" for {band_data_arr[0].survey} {band_data_arr[0].version}"
            )
            for i, band_data in enumerate(band_data_arr):
                if i == 0:
                    im_data, im_header, im_hdul = band_data.load_im(
                        return_hdul=True
                    )
                    prime_hdu = im_hdul[0].header
                else:
                    im_data, im_header = band_data.load_im()
                if err_type.lower() == "rms_err":
                    rms_err_data = band_data.load_rms_err()
                    wht_data = 1.0 / (rms_err_data**2)
                else:  # err_type.lower() == "wht"
                    wht_data = band_data.load_wht()
                    rms_err_data = np.sqrt(1.0 / wht)
                if i == 0:
                    sum = im_data * wht_data
                    sum_wht = wht_data
                else:
                    sum += im_data * wht_data
                    sum_wht += wht_data

            sci = sum / sum_wht
            err = np.sqrt(1.0 / sum_wht)
            wht = sum_wht

            primary = fits.PrimaryHDU(header=prime_hdu)
            hdu = fits.ImageHDU(sci, header=im_header, name="SCI")
            hdu_err = fits.ImageHDU(err, header=im_header, name="ERR")
            hdu_wht = fits.ImageHDU(wht, header=im_header, name="WHT")
            hdul = fits.HDUList([primary, hdu, hdu_err, hdu_wht])
            hdul.writeto(stacked_band_data_path, overwrite=True)
            funcs.change_file_permissions(stacked_band_data_path)

        output_dict = {
            "survey": band_data_arr[0].survey,
            "version": band_data_arr[0].version,
            "pix_scale": band_data_arr[0].pix_scale,
            "im_path": stacked_band_data_path,
            "im_ext": 1,
            "im_ext_name": "SCI",
            "rms_err_path": stacked_band_data_path,
            "rms_err_ext": 2,
            "rms_err_ext_name": "ERR",
            "wht_path": stacked_band_data_path,
            "wht_ext": 3,
            "wht_ext_name": "WHT",
        }
        return output_dict

    def mask(
        self,
        method: Union[str, List[str], Dict[str, str]] = "auto",
        fits_mask_path: Optional[Union[str, List[str], Dict[str, str]]] = None,
        star_mask_params: Optional[
            Union[
                Dict[str, Dict[str, float]],
                Dict[u.Quantity, Dict[str, Dict[str, float]]],
            ]
        ] = {
            "central": {"a": 300.0, "b": 4.25},
            "spikes": {"a": 400.0, "b": 4.5},
        },
        edge_mask_distance: Union[
            int, float, List[Union[int, float]], Dict[str, Union[int, float]]
        ] = 50,
        scale_extra: Union[float, List[float], Dict[str, float]] = 0.2,
        exclude_gaia_galaxies: Union[bool, List[bool], Dict[str, bool]] = True,
        angle: Union[float, List[float], Dict[str, float]] = 0.0,
        edge_value: Union[float, List[float], Dict[str, float]] = 0.0,
        element: Union[str, List[str], Dict[str, str]] = "ELLIPSE",
        gaia_row_lim: Union[int, List[int], Dict[str, int]] = 500,
        overwrite: Union[bool, List[bool], Dict[str, bool]] = False,
    ) -> Union[None, NoReturn]:
        # if the individual bands have not been loaded
        if not hasattr(self, "band_data_arr"):
            # mask the stacked band data
            super().mask(
                method=method,
                fits_mask_path=fits_mask_path,
                star_mask_params=star_mask_params,
                edge_mask_distance=edge_mask_distance,
                scale_extra=scale_extra,
                exclude_gaia_galaxies=exclude_gaia_galaxies,
                angle=angle,
                edge_value=edge_value,
                element=element,
                gaia_row_lim=gaia_row_lim,
                overwrite=overwrite,
            )
        else:
            # make these masks if they do not exist
            for band_data in self.band_data_arr:
                band_data.mask(
                    method=method,
                    fits_mask_path=fits_mask_path,
                    star_mask_params=star_mask_params,
                    edge_mask_distance=edge_mask_distance,
                    scale_extra=scale_extra,
                    exclude_gaia_galaxies=exclude_gaia_galaxies,
                    angle=angle,
                    edge_value=edge_value,
                    element=element,
                    gaia_row_lim=gaia_row_lim,
                    overwrite=overwrite,
                )
            # combine masks from individual bands
            self.mask_path, self.mask_args = Masking.combine_masks(self)


class Data:
    def __init__(
        self,
        band_data_arr: List[Type[Band_Data]],
        forced_phot_band: Optional[
            Union[str, List[str], Type[Band_Data_Base]]
        ] = None,
    ):
        # save and sort band_arr by central wavelength
        self.band_data_arr = funcs.sort_band_data_arr(band_data_arr)
        # load forced photometry band
        if forced_phot_band is not None:
            self.load_forced_phot_band(forced_phot_band)

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
        min_flux_pc_err: Union[int, float] = 10.
    ) -> Type[Data]:
        data = cls.from_survey_version( \
            survey,
            version,
            instrument_names,
            pix_scales,
            im_str,
            rms_err_str,
            wht_str,
            version_to_dir_dict,
            im_ext_name,
            rms_err_ext_name,
            wht_ext_name,
            aper_diams,
            forced_phot_band,
        )
        data.mask()
        data.segment()
        data.perform_forced_phot()
        data.append_aper_corr_cols()
        data.append_mask_cols()
        data.run_depths()
        data.append_loc_depth_cols(min_flux_pc_err = min_flux_pc_err)
        return data

    @classmethod
    def from_survey_version(
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
    ):
        # make im/rms_err/wht extension names lists if not already
        if isinstance(im_ext_name, str):
            im_ext_name = [im_ext_name]
        if isinstance(rms_err_ext_name, str):
            rms_err_ext_name = [rms_err_ext_name]
        if isinstance(wht_ext_name, str):
            wht_ext_name = [wht_ext_name]
        # search on an instrument-by-instrument basis
        instr_to_name_dict = {
            name: globals()[name]()
            for name in instrument_names
            if name in json.loads(config.get("Other", "INSTRUMENT_NAMES"))
        }
        band_data_arr = []
        for instr_name, instrument in instr_to_name_dict.items():
            if isinstance(pix_scales, dict):
                pix_scale = pix_scales[instr_name]
            else:
                pix_scale = pix_scales
            search_dir = cls._get_data_dir(
                survey,
                version,
                instrument,
                pix_scales[instr_name],
                version_to_dir_dict,
            )
            galfind_logger.debug(
                f"Searching for {survey} {version} {instr_name} data in {search_dir}"
            )
            # determine which filters have data
            fits_paths = list(glob.glob(f"{search_dir}/*.fits"))
            filt_names_paths = {
                filt: [
                    path
                    for path in fits_paths
                    if any(
                        path.find(substr) != -1
                        for substr in [
                            filt.upper(),
                            filt.lower(),
                            filt.lower().replace("f", "F"),
                            filt.upper().replace("F", "f"),
                        ]
                    )
                    and not any(
                        path.find(substr) != -1
                        for other_filt in instrument.filt_names
                        if other_filt != filt
                        for substr in [
                            other_filt.upper(),
                            other_filt.lower(),
                            other_filt.lower().replace("f", "F"),
                            other_filt.upper().replace("F", "f"),
                        ]
                    )
                ]
                for filt in instrument.filt_names
            }
            if all(len(values) == 0 for values in filt_names_paths.values()):
                err_message = f"No data found for {survey} {version} {instr_name} in {search_dir}"
                galfind_logger.critical(err_message)
                raise Exception(err_message)
            else:
                bands_found = [
                    key
                    for key, val in filt_names_paths.items()
                    if len(val) != 0
                ]
                galfind_logger.debug(
                    f"Found {'+'.join(bands_found)} filters for {survey} {version} {instr_name}"
                )
            # sort into paths and extensions for each image type
            (
                im_paths,
                im_exts,
                rms_err_paths,
                rms_err_exts,
                wht_paths,
                wht_exts,
            ) = cls._sort_paths(
                filt_names_paths,
                im_str,
                rms_err_str,
                wht_str,
                im_ext_name,
                rms_err_ext_name,
                wht_ext_name,
            )
            for filt_name in im_paths.keys():
                if len(im_paths[filt_name]) > 1:
                    # stack sci/rms_err/wht images together and move the old ones to a new directory
                    err_message = (
                        f"Multiple images found for {filt_name}."
                        + "Stacking multiple images in the same band not yet implemented"
                    )
                    # NOTE: This can only be done when the images are
                    # in the same fits file but different extensions.
                    raise NotImplementedError(err_message)
                else:
                    #breakpoint()
                    band_data = Band_Data(
                        Filter.from_filt_name(filt_name),
                        survey,
                        version,
                        im_paths[filt_name][0],
                        im_exts[filt_name][0],
                        rms_err_paths[filt_name][0],
                        rms_err_exts[filt_name][0],
                        wht_paths[filt_name][0],
                        wht_exts[filt_name][0],
                        pix_scale,
                        im_ext_name,
                        rms_err_ext_name,
                        wht_ext_name,
                        aper_diams=aper_diams,
                    )
                band_data_arr.extend([band_data])
        return cls(band_data_arr, forced_phot_band=forced_phot_band)

    @staticmethod
    def _get_data_dir(
        survey: str,
        version: str,
        instrument: Type[Instrument],
        pix_scale: u.Quantity = 0.03 * u.arcsec,
        version_to_dir_dict: Optional[Dict[str, str]] = None,
    ) -> Self:
        if version_to_dir_dict is not None:
            version = version_to_dir_dict[version.split("_")[0]]
        # else:
        #     version_substr = version
        # if len(version.split("_")) > 1:
        #     version_substr += f"_{'_'.join(version.split('_')[1:])}"
        return (
            f"{config['DEFAULT']['GALFIND_DATA']}/"
            + f"{instrument.facility.__class__.__name__.lower()}/{survey}/"
            + f"{instrument.__class__.__name__}/{version}/"
            + f"{Band_Data_Base._pix_scale_to_str(pix_scale)}"
        )

    @staticmethod
    def _sort_paths(
        filt_names_paths: Dict[str, List[str]],
        im_str: List[str] = ["_sci", "_i2d", "_drz"],
        rms_err_str: List[str] = ["_rms_err", "_rms", "_err"],
        wht_str: List[str] = ["_wht", "_weight"],
        im_ext_name: Union[str, List[str]] = "SCI",
        rms_err_ext_name: Union[str, List[str]] = "ERR",
        wht_ext_name: Union[str, List[str]] = "WHT",
    ) -> Tuple[
        Dict[str, List[str]],
        Dict[str, List[int]],
        Dict[str, List[str]],
        Dict[str, List[int]],
        Dict[str, List[str]],
        Dict[str, List[int]],
    ]:
        # determine which bands/image types correspond to which paths
        im_paths = {}
        im_exts = {}
        rms_err_paths = {}
        rms_err_exts = {}
        wht_paths = {}
        wht_exts = {}
        for filt_name, paths in filt_names_paths.items():
            if len(paths) == 0:
                galfind_logger.debug(f"No data found for {filt_name}")
                continue
            if filt_name not in im_paths.keys():
                im_paths[filt_name] = []
                im_exts[filt_name] = []
            if filt_name not in rms_err_paths.keys():
                rms_err_paths[filt_name] = []
                rms_err_exts[filt_name] = []
            if filt_name not in wht_paths.keys():
                wht_paths[filt_name] = []
                wht_exts[filt_name] = []
            # make arrays to determine where the data is stored for each band
            is_sci = {path: any([str in path for str in im_str]) for path in paths}
            is_rms_err = {
                path: any([str in path for str in rms_err_str]) for path in paths
            }
            is_wht = {path: any([str in path for str in wht_str]) for path in paths}
            # check to see if all paths are science images
            if all(is_sci_ext for is_sci_ext in is_sci.values()):
                all_sci = True
            else:
                all_sci = False
            for path in paths:
                # if all paths are science images
                if all_sci:
                    # all extensions must be within the same image
                    single_path = True
                    im_paths[filt_name].extend([path])
                    rms_err_paths[filt_name].extend([path])
                    wht_paths[filt_name].extend([path])
                else:
                    # ensure the path only belongs to one (or none) of the image types
                    n_unique_types = ([is_sci[path]] + [is_rms_err[path]] + [is_wht[path]]).count(True)
                    assert n_unique_types < 2, galfind_logger.critical(
                        f"Multiple image types found for {filt_name}, {path}"
                    )
                    single_path = False
                    if (
                        is_sci[path]
                        and not is_rms_err[path]
                        and not is_wht[path]
                    ):
                        im_paths[filt_name].extend([path])
                    elif (
                        not is_sci[path]
                        and is_rms_err[path]
                        and not is_wht[path]
                    ):
                        rms_err_paths[filt_name].extend([path])
                    elif (
                        not is_sci[path]
                        and not is_rms_err[path]
                        and is_wht[path]
                    ):
                        wht_paths[filt_name].extend([path])
                    else:
                        galfind_logger.critical(
                            f"{filt_name}, {path} not recognised as im, rms_err, or wht!"
                            + "Consider updating 'im_str', ''rms_err_str', and 'wht_str'!"
                        )
                # breakpoint()
                # extract sci/rms_err/wht extensions
                try:
                    hdul = fits.open(path, ignore_missing_simple = True)
                except:
                    breakpoint()
                if not single_path:
                    for j, hdu in enumerate(hdul):
                        if is_sci[path] and hdu.name in list(im_ext_name):
                            im_exts[filt_name].extend([int(j)])
                            break
                        elif is_rms_err[path] and hdu.name in list(rms_err_ext_name):
                            rms_err_exts[filt_name].extend([int(j)])
                            break
                        elif is_wht[path] and hdu.name in list(wht_ext_name):
                            wht_exts[filt_name].extend([int(j)])
                            break
                else:
                    for j, hdu in enumerate(hdul):
                        if hdu.name in im_ext_name:
                            im_exts[filt_name].extend([int(j)])
                        if hdu.name in rms_err_ext_name:
                            rms_err_exts[filt_name].extend([int(j)])
                        if hdu.name in wht_ext_name:
                            wht_exts[filt_name].extend([int(j)])
            # ensure a None is inserted if either rms_err/wht path/ext is missing
            # compared to im path length
            n_rms_err_path_missing = len(im_paths[filt_name]) - len(
                rms_err_paths[filt_name]
            )
            if n_rms_err_path_missing > 0:
                rms_err_paths[filt_name].extend(
                    list(itertools.repeat(None, n_rms_err_path_missing))
                )
            n_wht_path_missing = len(im_paths[filt_name]) - len(
                wht_paths[filt_name]
            )
            if n_wht_path_missing > 0:
                wht_paths[filt_name].extend(
                    list(itertools.repeat(None, n_wht_path_missing))
                )
            n_im_ext_missing = len(im_paths[filt_name]) - len(
                im_exts[filt_name]
            )
            if n_im_ext_missing != 0:
                err_message = f"SCI image extension not found for {filt_name}"
                galfind_logger.critical(err_message)
                raise (Exception(err_message))
            n_rms_err_ext_missing = len(im_paths[filt_name]) - len(
                rms_err_exts[filt_name]
            )
            if n_rms_err_ext_missing > 0:
                rms_err_exts[filt_name].extend(
                    list(itertools.repeat(None, n_rms_err_ext_missing))
                )
            n_wht_ext_missing = len(im_paths[filt_name]) - len(
                wht_exts[filt_name]
            )
            if n_wht_ext_missing > 0:
                wht_exts[filt_name].extend(
                    list(itertools.repeat(None, n_wht_ext_missing))
                )
            assert (
                len(im_paths[filt_name])
                == len(im_exts[filt_name])
                == len(rms_err_paths[filt_name])
                == len(rms_err_exts[filt_name])
                == len(wht_paths[filt_name])
                == len(wht_exts[filt_name])
            )

        return (
            im_paths,
            im_exts,
            rms_err_paths,
            rms_err_exts,
            wht_paths,
            wht_exts,
        )

    @property
    def survey(self):
        assert all(band_data.survey == self[0].survey for band_data in self)
        return self[0].survey

    @property
    def version(self):
        assert all(band_data.version == self[0].version for band_data in self)
        return self[0].version

    @property
    def filterset(self):
        return Multiple_Filter(band_data.filt for band_data in self if isinstance(band_data, Band_Data))

    # @property
    # def ZPs(self) -> Dict[str, float]:
    #     return {band_data.filt_name: band_data.ZP for band_data in self}

    # @property
    # def pix_scales(self) -> Dict[str, u.Quantity]:
    #     return {band_data.filt_name: band_data.pix_scale for band_data in self}

    @property
    def full_name(self):
        return funcs.get_full_survey_name(self.survey, self.version, self.filterset)
    
    @property
    def aper_diams(self) -> u.Quantity:
        all_aper_diams, aper_diam_counts = np.unique(np.concatenate([values for values in self.aper_diamss.values()]), return_counts = True)
        return [aper_diam.to(u.arcsec) for aper_diam, counts in zip(all_aper_diams, aper_diam_counts) if counts == len(self.aper_diamss)] * u.arcsec

    # def load_cluster_blank_mask_paths(self):
    #     # load in cluster core / blank field fits/reg masks
    #     mask_path_dict = {}
    #     for mask_type in ["cluster", "blank"]:
    #         # look for .fits masks first
    #         fits_masks = glob.glob(
    #             f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{self.survey}/fits_masks/*_{mask_type}*.fits"
    #         )
    #         if len(fits_masks) == 1:
    #             mask_path = fits_masks[0]
    #         elif len(fits_masks) > 1:
    #             galfind_logger.critical(
    #                 f"Multiple .fits {mask_type} masks exist for {self.survey}!"
    #             )
    #         else:
    #             # no .fits masks, now look for .reg masks
    #             reg_masks = glob.glob(
    #                 f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{self.survey}/*_{mask_type}*.reg"
    #             )
    #             if len(reg_masks) == 1:
    #                 mask_path = reg_masks[0]
    #             elif len(reg_masks) > 1:
    #                 galfind_logger.critical(
    #                     f"Multiple .reg {mask_type} masks exist for {self.survey}!"
    #                 )
    #             else:
    #                 # no .reg masks
    #                 mask_path = None
    #                 galfind_logger.info(
    #                     f"No {mask_type} mask found for {self.survey}"
    #                 )
    #         mask_path_dict[mask_type] = mask_path
    #     self.cluster_mask_path = mask_path_dict["cluster"]
    #     galfind_logger.debug(f"cluster_mask_path = {self.cluster_mask_path}")
    #     self.blank_mask_path = mask_path_dict["blank"]
    #     galfind_logger.debug(f"blank_mask_path = {self.blank_mask_path}")

    # %% Overloaded operators

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.full_name.replace('_', ', ')})"
        )

    def __str__(self):
        """Function to print summary of Data class

        Returns:
            str: Summary containing survey name, version, and whether field is blank or cluster.
                Includes summary of Instrument class, including bands, instruments and facilities used.
                Image depths in relevant aperture sizes are included here if calculated.
                Masked/unmasked areas are also quoted here.
                Also includes paths/extensions to SCI/SEG/ERR/WHT/MASK in each band, pixel scales, zero points and fits shapes.
        """
        output_str = funcs.line_sep
        output_str += "DATA OBJECT:\n"
        output_str += funcs.band_sep
        output_str += f"SURVEY: {self.survey}\n"
        output_str += f"VERSION: {self.version}\n"
        # TODO: Print survey areas
        # if a catalogue has been created, print the path
        if hasattr(self, "phot_cat_path"):
            output_str += f"PHOTOMETRIC CATALOGUE: {self.phot_cat_path}\n"
            output_str += f"APERTURE DIAMETERS: {self[0].aper_diams}\n"
            output_str += f"SELECTION BAND: {repr(self.forced_phot_band)}\n"
        output_str += f"FILTERSET: {repr(self.filterset)}\n"
        # print common attributes between bands
        self._get_common_attrs()
        for instr_name, common in self.common_attrs.items():
            if len(common) > 0:
                output_str += f"{instr_name} COMMON ATTRIBUTES:\n"
                output_str += funcs.band_sep
                for key, value in common.items():
                    output_str += f"{key.upper().replace('_', ' ')}: {value}\n"
                output_str += funcs.band_sep
        output_str += funcs.line_sep
        # loop through bands printing key attributes not in common, and depths if available
        for band_data in self:
            # output_str += str(band_data)
            output_str += f"{repr(band_data)}\n"
            output_str += funcs.band_sep
            # print the im, rms_err and wht paths/extensions
            for attr in ["im", "rms_err", "wht"]:
                if not (f"{attr}_dir" in self.common_attrs[band_data.instr_name].keys() \
                        and f"{attr}_ext" in self.common_attrs[band_data.instr_name].keys()):
                    output_str += f"{attr.upper().replace('_', ' ')} PATH: " + \
                        f"{getattr(band_data, f'{attr}_path')}[{getattr(band_data, f'{attr}_ext')}]\n"
                else:
                    output_str += f"{attr.upper().replace('_', ' ')} NAME: " + \
                        f"{getattr(band_data, f'{attr}_path').split('/')[-1]}[{getattr(band_data, f'{attr}_ext')}]\n"
            # print other attributes that are not in common
            for attr in [
                    "seg_path", 
                    "mask_path", 
                    "forced_phot_path", 
                    "mask_args", 
                    "seg_args", 
                    "forced_phot_args", 
                    "depth_args",
                    "ZP", 
                    "pix_scale", 
                    "data_shape",
                ]:
                if not attr in self.common_attrs[band_data.instr_name].keys() and hasattr(band_data, attr):
                    band_data_attr = getattr(band_data, attr)
                    if attr == "ZP":
                        band_data_attr = np.round(band_data_attr, decimals = 4)
                        name = attr.upper().replace('_', ' ')
                    if "_path" in attr:
                        band_data_attr = band_data_attr.split("/")[-1]
                        name = attr.upper().replace('_path', '_name').replace('_', ' ')
                    output_str += f"{name}: {band_data_attr}\n"
            
            if hasattr(band_data, "depth_args"):
                for aper_diam in band_data.aper_diams:
                    output_str += funcs.band_sep
                    output_str += f"{aper_diam}\n"
                    depth_keys = list(band_data.med_depth[aper_diam].keys())
                    depth_keys.remove("all")
                    for depth_key in depth_keys:
                        if len(depth_keys) > 1:
                            output_str += f"REGION {depth_key}:\n"
                        output_str += funcs.band_sep
                        output_str += f"MEDIAN DEPTH: {np.round(band_data.med_depth[aper_diam][depth_key], decimals = 3)}\n"
                        output_str += f"MEAN DEPTH: {np.round(band_data.mean_depth[aper_diam][depth_key], decimals = 3)}\n"
                    output_str += f"H5 PATH: {band_data.depth_path[aper_diam]}\n"
                    if "depth_args" not in self.common_attrs[band_data.instr_name].keys():
                        output_str += f"ARGS: {band_data.depth_args[aper_diam]}\n"
            # TODO: print total area if available
            output_str += funcs.line_sep
        output_str += funcs.line_sep
        return output_str

    def __len__(self):
        return len(self.band_data_arr)

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self) -> Band_Data:
        if self.iter > len(self) - 1:
            raise StopIteration
        else:
            band_data = self[self.iter]
            self.iter += 1
            return band_data

    def __getitem__(
        self, other: Union[int, slice, str, List[int], List[bool]]
    ) -> Band_Data:
        # convert other to integer indices if string
        # or a list of filter names are given
        if isinstance(other, str):
            other_split = other.split("+")
            other = self._indices_from_filt_names(other_split)
        elif isinstance(other, list):
            if isinstance(other[0], str):
                other = self._indices_from_filt_names(other)
        if isinstance(other, list):
            item = list(np.array(self.band_data_arr)[other])
        else:
            item = self.band_data_arr[other]
        if isinstance(item, Band_Data):
            return item
        else:
            if len(item) == 1:
                return item[0]
            else:
                return item

    def __getattr__(self, attr: str) -> Any:
        # attr inserted here must be pluralised with 's' suffix
        if all(attr[:-1] in band_data.__dict__.keys() for band_data in self):
            if hasattr(self, "forced_phot_band"):
                if attr[:-1] in self.forced_phot_band.__dict__.keys():
                    self_band_data_arr = self.band_data_arr + [self.forced_phot_band]
                else:
                    self_band_data_arr = self.band_data_arr
            else:
                self_band_data_arr = self.band_data_arr
            return {
                band_data.filt_name: getattr(band_data, attr[:-1])
                for band_data in self_band_data_arr
            }
        else:
            if attr not in [
                "__array_struct__",
                "__array_interface__",
                "__array__",
            ]:
                galfind_logger.debug(f"Data has no {attr=}!")
            raise AttributeError

    def __add__(
        self, other: Union[Type[Band_Data_Base], List[Type[Band_Data_Base]], Data, List[Data]]
    ) -> Data:
        # if other is not a list, make it one
        if not isinstance(other, list):
            other = [other]
        # if other is an array of data objects, make a list of band_data objects
        if isinstance(other[0], Data):
            assert all(isinstance(_other, Data) for _other in other)
            other_band_data = []
            for _other in other:
                other_band_data.extend(_other.band_data_arr)
            other = other_band_data
        assert all(isinstance(_other, tuple(Band_Data_Base.__subclasses__())) for _other in other)
        new_band_data_arr = self.band_data_arr + other
        # ensure all bands come from the same survey and version
        if all(
            [
                band_data.survey == self.survey
                and band_data.version == self.version
                for band_data in new_band_data_arr
            ]
        ):
            # if all bands being added are different
            if len(
                np.unique(
                    [
                        band_data.filt_name
                        for band_data in new_band_data_arr
                    ]
                )
            ) == len(new_band_data_arr):
                return Data(new_band_data_arr)
            else:
                raise (
                    Exception(
                        "Cannot add Data objects with the same filters."
                        + " You may want to use Data.__mul__() to stack!"
                    )
                )
        else:
            raise (
                Exception(
                    "Cannot add Data objects from different surveys or versions."
                )
            )

    def __eq__(self, other: Data) -> bool:
        if not isinstance(other, Data):
            return False
        elif len(self) != len(other):
            return False
        else:
            return all(
                [
                    self_band == other_band
                    for self_band, other_band in zip(self, other)
                ]
            )

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

    def _indices_from_filt_names(
        self, filt_names: Union[str, List[str]]
    ) -> int:
        if isinstance(filt_names, str):
            filt_names = filt_names.split("+")
        # make sure all names are filters in the filterset
        assert all(
            name in [band.filt_name for band in self] for name in filt_names
        ), galfind_logger.warning(
            f"Not all {filt_names} in {self.filterset.band_names}"
        )
        return [i for i in range(len(self)) if self[i].filt_name in filt_names]

    def _sort_band_dependent_params(
        self,
        filt_name: str,
        params: Union[Any, List[Any], Dict[str, Any]],
    ):
        if isinstance(params, list):
            # ensure params is the same length as the bands
            assert len(params) == len(self)
            return params[self._indices_from_filt_names(filt_name)]
        elif isinstance(params, dict):
            # if filter name is the name of a Stacked_Band_Data object
            if filt_name not in params.keys():
                galfind_logger.debug(
                    f"{filt_name} not in {params.keys()}!"
                )
                split_filt_names = filt_name.split("+")
                # ensure all filters are included in the object
                assert all(name in params.keys() for name in split_filt_names), \
                    galfind_logger.critical(
                        f"Not all {filt_name} in {params.keys()}"
                    )
                # ensure all parameters are the same
                assert all(params[name] == params[split_filt_names[0]] for i, name in enumerate(split_filt_names)), \
                    galfind_logger.critical(
                        f"Not all {params} are the same for {filt_name}"
                    )
                return params[split_filt_names[0]]
            else:
                assert filt_name in params.keys(), \
                    galfind_logger.critical(
                        f"{filt_name} not in {params.keys()}"
                    )
                return params[filt_name]
        else:
            return params

    def _get_common_attrs(self) -> NoReturn:
        common_attrs = {}
        # split by instrument
        for instr_name in self.filterset.instrument_name.split("+"):
            instr_band_data_arr = [band_data for band_data in self \
                if band_data.filt.instrument.__class__.__name__ in instr_name]
            common_attrs[instr_name] = {}
            # determine instrument dependent common path directories
            for attr in ["im_path", "rms_err_path", "wht_path", "mask_path", "seg_path", "forced_phot_path"]: 
                # NOTE: Could also do for "seg_path", "mask_path", "forced_phot_path"
                if all(hasattr(band_data, attr) for band_data in instr_band_data_arr):
                    if all("/".join(getattr(band_data, attr).split("/")[:-1]) == \
                            "/".join(getattr(instr_band_data_arr[0], attr).split("/")[:-1]) \
                            for band_data in instr_band_data_arr):
                        common_attrs[instr_name][f"{'_'.join(attr.split('_')[:-1])}_dir"] = \
                            "/".join(getattr(instr_band_data_arr[0], attr).split("/")[:-1])
            # NOTE: Could also determine instrument dependent common depth directories here - aperture diameter dependent
                
            # determine instrument dependent common aatributes
            for attr in ["im_ext", "rms_err_ext", "wht_ext", "ZP", "pix_scale", "data_shape"]:
                if all(getattr(band_data, attr) == getattr(instr_band_data_arr[0], attr) for band_data in instr_band_data_arr):
                    band_data_attr = getattr(instr_band_data_arr[0], attr)
                    if attr == "ZP":
                        band_data_attr = np.round(band_data_attr, decimals = 4)
                    common_attrs[instr_name][attr] = band_data_attr
            # determine instrument dependent mask, seg, forced phot, and depth arguments
            for attr in ["mask_args", "seg_args", "forced_phot_args", "depth_args"]:
                if all(hasattr(band_data, attr) for band_data in instr_band_data_arr):
                    if all(getattr(band_data, attr) == getattr(instr_band_data_arr[0], attr) for band_data in instr_band_data_arr):
                        common_attrs[instr_name][attr] = getattr(instr_band_data_arr[0], attr)
        # save common attributes in self
        self.common_attrs = common_attrs
    
    # %% Methods

    def load_data(
        self,
        band: Union[int, str, Filter, List[Filter], Multiple_Filter],
        incl_mask: bool = True,
    ):
        return self[band].load_data(incl_mask)

    def load_im(
        self,
        band: Union[int, str, Filter, List[Filter], Multiple_Filter],
        return_hdul: bool = False,
    ):
        return self[band].load_im(return_hdul)

    def load_wcs(
        self, band: Union[int, str, Filter, List[Filter], Multiple_Filter]
    ):
        return self[band].load_wcs()

    def load_wht(
        self,
        band: Union[int, str, Filter, List[Filter], Multiple_Filter],
        output_hdr: bool = False,
    ):
        return self[band].load_wht(output_hdr)

    def load_rms_err(
        self,
        band: Union[int, str, Filter, List[Filter], Multiple_Filter],
        output_hdr: bool = False,
    ):
        return self[band].load_rms_err(output_hdr)

    def load_seg(
        self, band: Union[int, str, Filter, List[Filter], Multiple_Filter]
    ):
        return self[band].load_seg()

    def load_mask(
        self,
        band: Union[int, str, Filter, List[Filter], Multiple_Filter],
        ext: Optional[str] = None,
        invert: bool = False,
    ):
        return self[band].load_mask(ext, invert)

    def load_aper_diams(self, aper_diams: u.Quantity) -> NoReturn:
        if hasattr(self, "forced_phot_band"):
            self.forced_phot_band.load_aper_diams(aper_diams)
        [band_data.load_aper_diams(aper_diams) for band_data in self]
    
    def _load_depths(
        self: Self,
        aper_diam: u.Quantity,
        mode: str,
        region: str = "all",
    ) -> NoReturn:
        [band_data._load_depths(aper_diam, mode, region) for band_data in self]
        

    def psf_homogenize(self):
        raise(NotImplementedError())

    def segment(
        self,
        err_type: str = "rms_err",
        method: str = "sextractor",
        config_name: str = "default.sex",
        params_name: str = "default.param",
        overwrite: bool = False,
    ) -> NoReturn:
        """
        Segments the data using the specified error type and method.

        Args:
            err_type (str): The type of error map to use for segmentation. Default is "rms_err".
            method (str): The method to use for segmentation. Default is "sextractor".

        Returns:
            NoReturn: This method does not return any value.
        """

        if hasattr(self, "forced_phot_band"):
            if (
                self.forced_phot_band.filt_name
                not in self.filterset.band_names
            ):
                self_band_data_arr = self.band_data_arr + [self.forced_phot_band]
            else:
                self_band_data_arr = self.band_data_arr
        else:
            self_band_data_arr = self.band_data_arr

        [
            band_data.segment(
                err_type, method, config_name, params_name, overwrite
            )
            for band_data in self_band_data_arr
        ]

    def perform_forced_phot(
        self,
        forced_phot_band: Optional[Union[str, List[str], Type[Band_Data_Base]]] = None,
        err_type: Union[str, List[str], Dict[str, str]] = "rms_err",
        method: Union[str, List[str], Dict[str, str]] = "sextractor",
        config_name: str = "default.sex",
        params_name: str = "default.param",
        overwrite: bool = False,
    ) -> NoReturn:
        if hasattr(self, "phot_cat_path"):
            galfind_logger.critical("MASTER Photometric catalogue already exists!")
            return
        # create a forced_phot_band object from given string
        self.load_forced_phot_band(forced_phot_band)

        if hasattr(self, "forced_phot_band"):
            if (
                self.forced_phot_band.filt_name
                not in self.filterset.band_names
            ):
                self_ = deepcopy(self) + deepcopy(self.forced_phot_band)
                self_band_data_arr = self.band_data_arr + [
                    self.forced_phot_band
                ]
            else:
                self_ = deepcopy(self)
                self_band_data_arr = self.band_data_arr

        # run for every band in the Data object
        [
            band_data._perform_forced_phot(
                self.forced_phot_band,
                self_._sort_band_dependent_params(band_data.filt_name, err_type),
                self_._sort_band_dependent_params(band_data.filt_name, method),
                config_name,
                params_name,
                overwrite,
            )
            for band_data in self_band_data_arr
        ]

        # combined forced photometry catalogues into a single photometric catalogue
        self._combine_forced_phot_cats(overwrite=overwrite)

    def load_forced_phot_band(
        self,
        forced_phot_band: Union[str, List[str], Type[Band_Data_Base]],
    ) -> Optional[Type[Band_Data_Base]]:
        if forced_phot_band is not None:
            # create a forced_phot_band object from given string
            if isinstance(forced_phot_band, str):
                filt_names = forced_phot_band.split("+")
            elif isinstance(forced_phot_band, list):
                filt_names = forced_phot_band
            if isinstance(forced_phot_band, tuple([str, list])):
                assert all(name in self.filterset.band_names for name in filt_names), \
                    galfind_logger.critical(
                        f"Not all {filt_names.split('+')} in {self.filterset.band_names}"
                    )
                if len(filt_names) == 1:
                    forced_phot_band = self[filt_names[0]]
                else:
                    forced_phot_band = Stacked_Band_Data.from_band_data_arr(
                        self[filt_names]
                    )
            # save forced phot band in self
            if hasattr(self, "forced_phot_band"):
                assert (
                    forced_phot_band == self.forced_phot_band
                ), galfind_logger.critical(
                    f"{self.forced_phot_band=} already loaded"
                )
            else:
                self.forced_phot_band = forced_phot_band
            return self.forced_phot_band
        else:
            return None

    def _get_phot_cat_path(self) -> str:
        # ensure aperture diamters are the same for all bands
        assert all(all(diam == diam_0 for diam, diam_0 in zip(band_data.aper_diams, self[0].aper_diams)) for band_data in self)
        assert all(diam == diam_0 for diam, diam_0 in zip(self.forced_phot_band.aper_diams, self[0].aper_diams))
        # ensure all bands have the same forced photometry band
        assert all(band_data.forced_phot_args["forced_phot_band"] == self.forced_phot_band for band_data in self)
        assert self.forced_phot_band.forced_phot_args["forced_phot_band"] == self.forced_phot_band # points to itself?
        # ensure all bands are made using the same err map
        assert all(band_data.forced_phot_args["err_type"] == self[0].forced_phot_args["err_type"] for band_data in self)
        assert self.forced_phot_band.forced_phot_args["method"] == self[0].forced_phot_args["method"]

        # determine photometric catalogue path
        phot_cat_path = funcs.get_phot_cat_path(
            self.survey,
            self.version,
            self.filterset.instrument_name,
            self[0].aper_diams,
            self.forced_phot_band.filt_name
        )
        funcs.make_dirs(phot_cat_path)
        return phot_cat_path

    def _combine_forced_phot_cats(self, overwrite: bool = False) -> NoReturn:
        # readme_sep: str = "-" * 20,
        phot_cat_path = self._get_phot_cat_path()
        funcs.make_dirs(phot_cat_path)
        if not hasattr(self, "phot_cat_path"):
            self.phot_cat_path = phot_cat_path
        else:
            raise (Exception("MASTER Photometric catalogue already exists!"))
        
        if not Path(phot_cat_path).is_file() or overwrite:

            master_tab_arr = [self.forced_phot_band._get_master_tab(
                output_ids_locs=True
            )]
            for band_data in self:
                if band_data.filt_name != self.forced_phot_band.filt_name: 
                    master_tab_arr.extend(
                        [band_data._get_master_tab(output_ids_locs=False)])
            master_tab = hstack(master_tab_arr)
            # update table header
            self_band_data_arr = self.band_data_arr + [self.forced_phot_band]
            master_tab.meta = {
                **master_tab.meta,
                **{
                    "INSTR": self.filterset.instrument_name,
                    "SURVEY": self.survey,
                    "VERSION": self.version,
                    "BANDS": str(self.filterset.band_names),
                    "APERDIAM": funcs.aper_diams_to_str(self.forced_phot_band.aper_diams),
                    "ERR_TYPE": "+".join(np.unique(band_data.forced_phot_args["err_type"] for band_data in self_band_data_arr)[0]),
                    "METHODS": "+".join(np.unique(band_data.forced_phot_args["method"] for band_data in self_band_data_arr)[0]),
                },
            }
            # save master table
            master_tab.write(self.phot_cat_path, format="fits", overwrite=True)
            galfind_logger.info(
                f"Saved combined SExtractor catalogue as {self.phot_cat_path}"
            )
            self._create_phot_cat_readme()

    def _create_phot_cat_readme(self):
        pass
        # create galfind catalogue README
        # sex_aper_diams = json.loads(config.get("SExtractor", "APERTURE_DIAMS")) * u.arcsec
        # text = f"""
        #     NUMBER: Galaxy ID
        #     X/Y_IMAGE: X/Y image co-ordinates in
        #     ALPHA/DELTA_J2000: RA/Dec (J2000 co-ordinates)
        #     FLUX(ERR)_APER_'band': Aperture flux/flux errors in {str(sex_aper_diams.to(u.arcsec).value) + 'as'} diameter apertures, image units with ZPs as explained below
        #     MAG_APER_'band': Aperture magnitudes in {str(sex_aper_diams.to(u.arcsec).value) + 'as'} diameter apertures, AB mag units, defaults to 99. if flux < 0.
        #     MAGERR_APER_'band': Aperture magnitude errors in {str(sex_aper_diams.to(u.arcsec).value) + 'as'} diameter apertures, AB mag units, negative if mag == 99.
        # """
        # if 'sextractor' in [cat_type.lower() for cat_type in self.sex_cat_types.values()]:
        #     text += f"See SExtractor documentation () for descriptions of other columns. These are only available for {'+'.join([band_name for band_name, sex_cat_type in self.sex_cat_types.items() if 'sextractor' in sex_cat_type.lower()])}\n"
        # text += readme_sep + "\n"
        # self.make_sex_readme({"Photometry": text}, self.sex_cat_master_path.replace(".fits", "_README.txt"))

    def mask(
        self,
        method: Union[str, List[str], Dict[str, str]] = "auto",
        fits_mask_path: Optional[Union[str, List[str], Dict[str, str]]] = None,
        star_mask_params: Optional[
            Union[
                Dict[str, Dict[str, float]],
                Dict[u.Quantity, Dict[str, Dict[str, float]]],
            ]
        ] = {
            "central": {"a": 300.0, "b": 4.25},
            "spikes": {"a": 400.0, "b": 4.5},
        },
        edge_mask_distance: Union[
            int, float, List[Union[int, float]], Dict[str, Union[int, float]]
        ] = 50,
        scale_extra: Union[float, List[float], Dict[str, float]] = 0.2,
        exclude_gaia_galaxies: Union[bool, List[bool], Dict[str, bool]] = True,
        angle: Union[float, List[float], Dict[str, float]] = 0.0,
        edge_value: Union[float, List[float], Dict[str, float]] = 0.0,
        element: Union[str, List[str], Dict[str, str]] = "ELLIPSE",
        gaia_row_lim: Union[int, List[int], Dict[str, int]] = 500,
        overwrite: Union[bool, List[bool], Dict[str, bool]] = False,
    ) -> Union[None, NoReturn]:
        assert method in ["auto", "manual"], galfind_logger.warning(
            f"Method {method} not recognised. Must be 'auto' or 'manual'"
        )

        if hasattr(self, "forced_phot_band"):
            if (
                self.forced_phot_band.filt_name
                not in self.filterset.band_names
            ):
                self_ = deepcopy(self) + deepcopy(self.forced_phot_band)
                self_band_data_arr = self.band_data_arr + [
                    self.forced_phot_band
                ]
                no_forced_phot_band = False
            else:
                no_forced_phot_band = True
        else:
            no_forced_phot_band = True
        if no_forced_phot_band:
            self_ = deepcopy(self)
            self_band_data_arr = self.band_data_arr

        # mask each band, sorting the potentially band dependent input parameters
        [
            band_data.mask(
                method,
                self_._sort_band_dependent_params(
                    band_data.filt_name, fits_mask_path
                ),
                Masking.sort_band_dependent_star_mask_params(
                    band_data.filt if isinstance(band_data, Band_Data) else band_data.filterset[0], star_mask_params
                ),
                self_._sort_band_dependent_params(
                    band_data.filt_name, edge_mask_distance
                ),
                self_._sort_band_dependent_params(
                    band_data.filt_name, scale_extra
                ),
                self_._sort_band_dependent_params(
                    band_data.filt_name, exclude_gaia_galaxies
                ),
                self_._sort_band_dependent_params(band_data.filt_name, angle),
                self_._sort_band_dependent_params(
                    band_data.filt_name, edge_value
                ),
                self_._sort_band_dependent_params(
                    band_data.filt_name, element
                ),
                self_._sort_band_dependent_params(
                    band_data.filt_name, gaia_row_lim
                ),
                self_._sort_band_dependent_params(
                    band_data.filt_name, overwrite
                ),
            )
            for band_data in self_band_data_arr
        ]

    def run_depths(
        self,
        mode: Union[str, List[str], Dict[str, str]] = "n_nearest",
        scatter_size: Union[float, List[float], Dict[str, float]] = 0.1,
        distance_to_mask: Union[
            int, float, List[Union[int, float]], Dict[str, Union[int, float]]
        ] = 30,
        region_radius_used_pix: Union[
            int, float, List[Union[int, float]], Dict[str, Union[int, float]]
        ] = 300,
        n_nearest: Union[int, List[int], Dict[str, int]] = 200,
        coord_type: Union[str, List[str], Dict[str, str]] = "sky",
        split_depth_min_size: Union[int, List[int], Dict[str, int]] = 100_000,
        split_depths_factor: Union[int, List[int], Dict[str, int]] = 5,
        step_size: Union[int, List[int], Dict[str, int]] = 100,
        n_jobs: int = 1,
        n_split: Union[
            str, int, List[Union[str, int]], Dict[str, Union[str, int]]
        ] = "auto",
        n_retry_box: Union[int, List[int], Dict[str, int]] = 1,
        grid_offset_times: Union[int, List[int], Dict[str, int]] = 1,
        plot: Union[bool, List[bool], Dict[str, bool]] = True,
        overwrite: Union[bool, List[bool], Dict[str, bool]] = False,
        timed: bool = False,
    ) -> NoReturn:
        if timed:
            start = time.time()
        if hasattr(self, "phot_cat_path"):
            master_cat_path = self.phot_cat_path
        else:
            master_cat_path = None
        if hasattr(self, "forced_phot_band"):
            if (
                self.forced_phot_band.filt_name
                not in self.filterset.band_names
            ):
                self_ = deepcopy(self) + deepcopy(self.forced_phot_band)
                self_band_data_arr = self.band_data_arr + [
                    self.forced_phot_band
                ]
                no_forced_phot_band = False
            else:
                no_forced_phot_band = True
        else:
            no_forced_phot_band = True
        if no_forced_phot_band:
            self_ = deepcopy(self)
            self_band_data_arr = self.band_data_arr

        params = []
        # Look over all aperture diameters and bands
        for band_data in self_band_data_arr:
            if not hasattr(band_data, "depth_args"):
                params.extend(
                    band_data._sort_run_depth_params(
                        self_._sort_band_dependent_params(
                            band_data.filt_name, mode
                        ),
                        self_._sort_band_dependent_params(
                            band_data.filt_name, scatter_size
                        ),
                        self_._sort_band_dependent_params(
                            band_data.filt_name, distance_to_mask
                        ),
                        self_._sort_band_dependent_params(
                            band_data.filt_name, region_radius_used_pix
                        ),
                        self_._sort_band_dependent_params(
                            band_data.filt_name, n_nearest
                        ),
                        self_._sort_band_dependent_params(
                            band_data.filt_name, coord_type
                        ),
                        self_._sort_band_dependent_params(
                            band_data.filt_name, split_depth_min_size
                        ),
                        self_._sort_band_dependent_params(
                            band_data.filt_name, split_depths_factor
                        ),
                        self_._sort_band_dependent_params(
                            band_data.filt_name, step_size
                        ),
                        self_._sort_band_dependent_params(
                            band_data.filt_name, n_split
                        ),
                        self_._sort_band_dependent_params(
                            band_data.filt_name, n_retry_box
                        ),
                        self_._sort_band_dependent_params(
                            band_data.filt_name, grid_offset_times
                        ),
                        self_._sort_band_dependent_params(
                            band_data.filt_name, overwrite
                        ),
                        master_cat_path,
                    )
                )
            else:
                galfind_logger.warning(
                    f"Depths for {band_data.filt_name} already run, skipping!"
                )
        if len(params) > 0:
            # Parallelise the calculation of depths for each band
            with funcs.tqdm_joblib(
                tqdm(desc="Calculating depths", total=len(params))
            ) as progress_bar:
                Parallel(n_jobs=n_jobs)(
                    delayed(Depths.calc_band_depth)(param) for param in params
                )

            # save properties to individual band_data objects
            for band_data in self_band_data_arr:
                [
                    band_data._load_depths_from_params(params)
                    for _params in params
                    if _params[0] == band_data
                ]
                if plot:
                    band_data.plot_depth_diagnostics(
                        save = True, 
                        overwrite = False, 
                        master_cat_path = master_cat_path
                    )

            # make depth table
            Depths.make_depth_tab(self)

            finishing_message = (
                "Calculated/loaded depths for "
                + f"{self.survey} {self.version} {self.filterset.instrument_name}"
            )
            if timed:
                end = time.time()
                finishing_message += f" ({end - start:.1f}s)"
            galfind_logger.info(finishing_message)
        else:
            galfind_logger.warning(
                f"Depths run for {self.survey} {self.version}"
                + f" {self.filterset.instrument_name}, skipping!"
            )


    def plot_depth_diagnostic(
        self,
        band: Union[int, str, Filter, List[Filter], Multiple_Filter],
        aper_diam: u.Quantity,
        save: bool = False, 
        show: bool = False,
        overwrite: bool = True,
    ) -> NoReturn:
        try:
            master_cat_path = self._get_phot_cat_path()
        except:
            master_cat_path = None
        self[band].plot_depth_diagnostic(
            aper_diam,
            save = save,
            show = show,
            overwrite = overwrite,
            master_cat_path = master_cat_path
        )

    def plot_depth_diagnostics(
        self,
        save: bool = False,
        overwrite: bool = True,
    ) -> NoReturn:
        try:
            master_cat_path = self._get_phot_cat_path()
        except:
            master_cat_path = None
        for band_data in self:
            band_data.plot_depth_diagnostics(
                save = save,
                overwrite = overwrite,
                master_cat_path = master_cat_path
            )

    def plot_area_depth(
        self,
    ) -> NoReturn:
        Depths.plot_data_area_depth(self)
        # Depths.plot_area_depth(self, cat_creator, mode, aper_diam, show=False)

    def plot(
        self,
        band: Union[int, str, Filter, List[Filter], Multiple_Filter],
        ax: Optional[plt.Axes] = None,
        ext: str = "SCI",
        norm: Type[Normalize] = LogNorm(vmin=0.0, vmax=10.0),
        save: bool = False,
        show: bool = True,
    ) -> NoReturn:
        self[band].plot(ax, ext, norm, save, show)

    def plot_RGB(
        self,
        ax: Optional[plt.Axes] = None,
        blue_bands: List[Union[str, Filter]] = ["F090W"],
        green_bands: List[Union[str, Filter]] = ["F200W"],
        red_bands: List[Union[str, Filter]] = ["F444W"],
        method: str = "trilogy",
    ):
        # ensure all blue, green and red bands are contained in the data object
        assert all(
            band in self.instrument.band_names
            for band in blue_bands + green_bands + red_bands
        ), galfind_logger.warning(
            f"Cannot make galaxy RGB as not all {blue_bands + green_bands + red_bands} are in {self.instrument.band_names}"
        )
        # construct out_path
        out_path = f"{config['RGB']['RGB_DIR']}/{self.version}/{self.survey}/{method}/B={'+'.join(blue_bands)},G={'+'.join(green_bands)},R={'+'.join(red_bands)}.png"
        funcs.make_dirs(out_path)
        if not os.path.exists(out_path):
            # load RGB band paths including .fits image extensions
            RGB_paths = {}
            for colour, bands in zip(
                ["B", "G", "R"], [blue_bands, green_bands, red_bands]
            ):
                RGB_paths[colour] = [
                    f"{self.im_paths[band]}[{self.im_exts[band]}]"
                    for band in bands
                ]
            if method == "trilogy":
                # Write trilogy.in
                in_path = out_path.replace(".png", "_trilogy.in")
                with open(in_path, "w") as f:
                    for colour, paths in RGB_paths.items():
                        f.write(f"{colour}\n")
                        for path in paths:
                            f.write(f"{path}\n")
                        f.write("\n")
                    f.write("indir  /\n")
                    f.write(
                        f"outname  {funcs.split_dir_name(out_path, 'name').replace('.png', '')}\n"
                    )
                    f.write(
                        f"outdir  {funcs.split_dir_name(out_path, 'dir')}\n"
                    )
                    f.write("samplesize 20000\n")
                    f.write("stampsize  2000\n")
                    f.write("showstamps  0\n")
                    f.write("satpercent  0.001\n")
                    f.write("noiselum    0.10\n")
                    f.write("colorsatfac  1\n")
                    f.write("deletetests  1\n")
                    f.write("testfirst   0\n")
                    f.write("sampledx  0\n")
                    f.write("sampledy  0\n")

                funcs.change_file_permissions(in_path)
                # Run trilogy
                sys.path.insert(
                    1, "/nvme/scratch/software/trilogy"
                )  # Not sure why this path doesn't work: config["Other"]["TRILOGY_DIR"]
                from trilogy3 import Trilogy

                galfind_logger.info(
                    f"Making full trilogy RGB image at {out_path}"
                )
                Trilogy(in_path, images=None).run()
            elif method == "lupton":
                raise (NotImplementedError())

    def append_loc_depth_cols(
        self,
        min_flux_pc_err: Union[int, float],
        overwrite: bool = False
    ) -> NoReturn:
        Depths.append_loc_depth_cols(self, min_flux_pc_err, overwrite)

    def append_aper_corr_cols(
        self,
        overwrite: bool = False,
        psf_wanted: str = "model"
    ) -> NoReturn:
        assert psf_wanted in ["model", "empirical"], \
            galfind_logger.critical(
                f"PSF '{psf_wanted}' not in ['model', 'empirical']"
            )
        # not general
        cat = Table.read(self.phot_cat_path)
        if f"MAG_APER_{self[0].filt_name}_aper_corr" not in cat.colnames or overwrite:
            # ensure aperture diameters are the same for all bands
            assert all(all(diam == diam_0 for diam, diam_0 
                in zip(band_data.aper_diams, self[0].aper_diams)) 
                for band_data in self.band_data_arr + 
                [self.forced_phot_band]), galfind_logger.critical(
                "Aperture diameters are not the same for all bands!"
                )
            if overwrite:
                # TODO: Delete already existing columns
                raise(Exception())
            aper_diams = self[0].aper_diams.to(u.arcsec).value
            for i, band_data in tqdm(enumerate(self), \
                    total=len(self), desc="Appending aperture correction columns"):
                mag_aper_corr_data = np.zeros(len(cat))
                flux_aper_corr_data = np.zeros(len(cat))
                if len(aper_diams) == 1:
                    mag_aper_corr_factor = band_data.filt.instrument.\
                        aper_corrs[band_data.filt_name][aper_diams[0] * u.arcsec]
                    flux_aper_corr_factor = 10 ** (mag_aper_corr_factor / 2.5)
                    # only aperture correct if flux is positive
                    mag_aper_corr_data = [
                        mag_aper - mag_aper_corr_factor
                        if flux_aper > 0.0 else mag_aper
                        for mag_aper, flux_aper in zip(
                            cat[f"MAG_APER_{band_data.filt_name}"],
                            cat[f"FLUX_APER_{band_data.filt_name}"],
                        )
                    ]
                    flux_aper_corr_data = [
                        flux_aper * flux_aper_corr_factor
                        if flux_aper > 0.0 else flux_aper
                        for flux_aper in cat[f"FLUX_APER_{band_data.filt_name}"]
                    ]
                else:
                    for j, aper_diam in enumerate(aper_diams):
                        # assumes these have already been calculated for each band
                        mag_aper_corr_factor = band_data.filt.instrument.\
                            aper_corrs[band_data.filt_name][aper_diam * u.arcsec]
                        flux_aper_corr_factor = 10 ** (mag_aper_corr_factor / 2.5)
                        
                        if j == 0:
                            # only aperture correct if flux is positive
                            mag_aper_corr_data = [
                                (mag_aper[0] - mag_aper_corr_factor,)
                                if flux_aper[0] > 0.0
                                else (mag_aper[0],)
                                for mag_aper, flux_aper in zip(
                                    cat[f"MAG_APER_{band_data.filt_name}"],
                                    cat[f"FLUX_APER_{band_data.filt_name}"],
                                )
                            ]
                            flux_aper_corr_data = [
                                (flux_aper[0] * flux_aper_corr_factor,)
                                if flux_aper[0] > 0.0
                                else (flux_aper[0],)
                                for flux_aper in cat[f"FLUX_APER_{band_data.filt_name}"]
                            ]
                        else:
                            mag_aper_corr_data = [
                                mag_aper_corr
                                + (mag_aper[j] - mag_aper_corr_factor,)
                                if flux_aper[j] > 0.0
                                else mag_aper_corr + (mag_aper[j],)
                                for mag_aper_corr, mag_aper, flux_aper in zip(
                                    mag_aper_corr_data,
                                    cat[f"MAG_APER_{band_data.filt_name}"],
                                    cat[f"FLUX_APER_{band_data.filt_name}"],
                                )
                            ]
                            flux_aper_corr_data = [
                                flux_aper_corr
                                + (flux_aper[j] * flux_aper_corr_factor,)
                                if flux_aper[j] > 0.0
                                else flux_aper_corr + (flux_aper[j],)
                                for flux_aper_corr, flux_aper in zip(
                                    flux_aper_corr_data, cat[f"FLUX_APER_{band_data.filt_name}"]
                                )
                            ]
                cat[f"MAG_APER_{band_data.filt_name}_aper_corr"] = mag_aper_corr_data
                cat[f"FLUX_APER_{band_data.filt_name}_aper_corr"] = flux_aper_corr_data
                if len(aper_diams) == 1:
                    cat[f"FLUX_APER_{band_data.filt_name}_aper_corr_Jy"] = [
                        funcs.flux_image_to_Jy(element, band_data.ZP).value
                        for element in cat[f"FLUX_APER_{band_data.filt_name}_aper_corr"]
                    ]
                else:
                    cat[f"FLUX_APER_{band_data.filt_name}_aper_corr_Jy"] = [
                        tuple(
                            [
                                funcs.flux_image_to_Jy(
                                    val, band_data.ZP
                                ).value
                                for val in element
                            ]
                        )
                        for element in cat[f"FLUX_APER_{band_data.filt_name}_aper_corr"]
                    ]
            # TODO: update catalogue metadata with PSF representation
            
            # overwrite original catalogue with local depth columns
            cat.write(self.phot_cat_path, overwrite=True)
            funcs.change_file_permissions(self.phot_cat_path)
            galfind_logger.info(
                f"Appended aperture correction columns to {self.phot_cat_path}"
            )
        else:
            galfind_logger.warning(
                f"Aperture correction columns already in {self.phot_cat_path}"
            )
    
    def append_mask_cols(
        self, 
        overwrite: bool = False
    ) -> NoReturn:
        # ensure forced photometry has been run on every band in catalogue
        assert all(hasattr(band_data, "forced_phot_args") for band_data in self), \
            galfind_logger.critical(
                "Forced photometry not performed on all bands!"
            )
        assert all(band_data.forced_phot_args["ra_label"] == \
                self[0].forced_phot_args["ra_label"] for band_data in self) \
                and all(band_data.forced_phot_args["dec_label"] == \
                self[0].forced_phot_args["dec_label"] for band_data in self), \
            galfind_logger.critical(
                "RA/DEC labels not the same for all bands!"
            )
        cat = Table.read(self.phot_cat_path)
        if f"unmasked_{self[0].filt_name}" not in cat.colnames or overwrite:
            if overwrite:
                # TODO: Delete already existing columns
                raise(NotImplementedError())
                galfind_logger.info(
                    f"Deleting {self.phot_cat_path} mask columns!"
                )
            # make sky_coords
            ra = cat[self[0].forced_phot_args["ra_label"]]
            dec = cat[self[0].forced_phot_args["dec_label"]]
            sky_coords = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
            # append mask columns to catalogue
            for band_data in self:
                galfind_logger.info(
                    f"Appending {band_data.filt_name} mask" + \
                    f" columns to {self.phot_cat_path}"
                )
                wcs = band_data.load_wcs()
                cat_x, cat_y = wcs.world_to_pixel(sky_coords)
                mask = band_data.load_mask()[0]["MASK"]
                cat[f"unmasked_{band_data.filt_name}"] = \
                    np.array(
                        [
                            False if x < 0.0
                            or x >= mask.shape[1]
                            or y < 0.0 or y >= mask.shape[0]
                            else not bool(mask[int(y)][int(x)])
                            for x, y in zip(cat_x, cat_y)
                        ]
                    )
            cat.write(self.phot_cat_path, overwrite=True)
            funcs.change_file_permissions(self.phot_cat_path)
            galfind_logger.info(
                f"Appended mask columns to {self.phot_cat_path}"
            )
            # TODO: update README
            galfind_logger.debug(
                f"Updating README for mask not implemented!"
            )

    # @staticmethod
    # def mosaic_images(
    #     image_paths,
    #     extract_ext_names={"data": "SCI", "err": "RMS_ERR"},
    #     pix_scale_hdr_name="PIXSCALE",
    # ):
    #     # ensure images are .fits images
    #     assert all(".fits" in path for path in image_paths)
    #     # open all images
    #     hdul_arr = [fits.open(path) for path in image_paths]
    #     # ensure images have the same number of extensions
    #     assert all(len(hdul_arr[0]) == hdul for hdul in hdul_arr)
    #     # ensure images have all of the relevant extensions - NOT IMPLEMENTED YET
    #     # extract the header files for each extension for each fits image
    #     headers_arr = np.array(
    #         [[hdu.header for hdu in hdul] for hdul in hdul_arr]
    #     )
    #     # ensure they have been taken using the same filter - NOT IMPLEMENTED YET (assume this comes from the header files)
    #     ext_names_arr = [
    #         [header["EXTNAME"] for header in hdul_headers]
    #         for hdul_headers in headers_arr
    #     ]
    #     # extract the raw data for each extension for each fits image
    #     data_arr = np.array(
    #         [
    #             [
    #                 hdu.data
    #                 for hdu, ext_name in zip(hdul, ext_names)
    #                 if ext_name == extract_ext_names["data"]
    #             ][0]
    #             for hdul, ext_names in zip(hdul_arr, ext_names_arr)
    #         ]
    #     )
    #     err_arr = np.array(
    #         [
    #             [
    #                 hdu.data
    #                 for hdu, ext_name in zip(hdul, ext_names)
    #                 if ext_name == extract_ext_names["err"]
    #             ][0]
    #             for hdul, ext_names in zip(hdul_arr, ext_names_arr)
    #         ]
    #     )
    #     # if the files have the same wcs, the same x/y dimensions, and the same pixel scale
    #     same_wcs = all(
    #         WCS(header) == WCS(headers_arr[0][0])
    #         for hdul_headers in headers_arr
    #         for header in hdul_headers
    #     )
    #     same_dimensions = (
    #         all(data.shape == data_arr[0].shape for data in data_arr)
    #     ) & (all(err.shape == err_arr[0].shape for err in err_arr))
    #     same_pix_scale = all(
    #         float(header[pix_scale_hdr_name])
    #         == float(headers_arr[0][0][pix_scale_hdr_name])
    #         for hdul_headers in headers_arr
    #         for header in hdul_headers
    #     )
    #     if same_wcs and same_dimensions and same_pix_scale:
    #         # ensure images are PSF homogenized to the same filter
    #         for i, (data, err) in enumerate(zip(data_arr, err_arr)):
    #             if i == 0:
    #                 sum_data = data
    #                 sum_err = err
    #             else:
    #                 # convert np.nans to zeros in both science and error maps so as to allow data only covered by one image
    #                 data[data == np.nan] = 0.0
    #                 err[err == np.nan] = 0.0
    #                 sum_data += data / err**2
    #                 sum_err += 1 / err**2
    #         weighted_array = sum_data / sum_err  # output sci map
    #         combined_err = np.sqrt(1 / sum_err)  # output err map

    #         # determine new combined image path
    #         combined_image_path = image_paths[0].replace(
    #             ".fits", "_stack.fits"
    #         )
    #         # save combined image at this path
    #         primary = fits.PrimaryHDU(header=prime_hdu)
    #         hdu = fits.ImageHDU(weighted_array, header=im_header, name="SCI")
    #         hdu_err = fits.ImageHDU(combined_err, header=im_header, name="ERR")
    #         hdul = fits.HDUList([primary, hdu, hdu_err])
    #         hdul.writeto(combined_image_path, overwrite=True)
    #         galfind_logger.info(
    #             f"Finished mosaicing images at {image_paths=}, saved to {combined_image_path}"
    #         )
    #         hdul.writeto(combined_image_path, overwrite=True)
    #         # move the individual images into a "stacked" folder
    #         for path in image_paths:
    #             os.makedirs(
    #                 f"{funcs.split_dir_name(path, 'dir')}/stacked",
    #                 exist_ok=True,
    #             )
    #             os.rename(
    #                 path,
    #                 f"{funcs.split_dir_name(path, 'dir')}/stacked/{funcs.split_dir_name(path, 'name')}",
    #             )
    #         return combined_image_path

    #     else:  # else convert all of the images to the required wcs, x/y dimensions, and pixel scale
    #         raise (
    #             NotImplementedError(
    #                 galfind_logger.critical(
    #                     f"Cannot convert images as all of {same_wcs=}, {same_dimensions=}, {same_pix_scale=} != True"
    #                 )
    #             )
    #         )

    # def make_readme(
    #     self, col_desc_dict, save_path, overwrite=False, readme_sep="-" * 20
    # ):
    #     assert type(col_desc_dict) == dict
    #     assert "Photometry" in col_desc_dict.keys()
    #     intro_text = """

    #     """
    #     # if not overwrite and README already exists, extract previous column labels to append col_desc_dict to
    #     f = open(save_path, "w")
    #     f.write(intro_text)
    #     f.write(readme_sep + "\n\n")
    #     f.write(str(self) + "\n")
    #     for key, value in col_desc_dict.items():
    #         if key == "Photometry":
    #             init_phot_text = (
    #                 "Photometry:\n"
    #                 + "\n".join(
    #                     [
    #                         phot_code
    #                         + "= "
    #                         + "+".join(
    #                             [
    #                                 band_name
    #                                 for band_name, sex_cat_type in self.sex_cat_types.items()
    #                                 if sex_cat_type == phot_code
    #                             ]
    #                         )
    #                         for phot_code in np.unique(
    #                             self.sex_cat_types.values()
    #                         )
    #                     ]
    #                 )
    #                 + "\n"
    #             )
    #             f.write(init_phot_text)
    #         else:
    #             f.write(key + "\n")
    #         f.write(readme_sep + "\n")
    #         f.write(value)
    #         f.write(readme_sep + "\n")
    #     f.close()


    def calc_unmasked_area(
        self: Self,
        mask_selector: Union[str, List[str], Type[Mask_Selector]],
        mask_type: Union[str, List[str]] = "MASK",
        region_selector: Optional[Type[Region_Selector], List[Type[Region_Selector]]] = None,
        invert_region: bool = True,
        out_units: u.Quantity = u.arcmin ** 2,
    ) -> u.Quantity:

        from . import Mask_Selector

        if not hasattr(self, "unmasked_area"):
            self.unmasked_area = {}

        if isinstance(mask_selector, str):
            mask_selector = mask_selector.split("+")
        if isinstance(mask_type, str):
            mask_type = mask_type.split("+")
        
        if isinstance(mask_selector, tuple(Mask_Selector.__subclasses__())):
            mask_selector_name = mask_selector.name
        else:
            mask_selector_name = f"{'+'.join(np.sort(mask_selector))}_{reg_name}"

        if region_selector is None:
            reg_name = "All"
        else:
            if not isinstance(region_selector, list):
                region_selector = [region_selector]
            reg_name = "+".join([
                region_selector_.name if not invert_region 
                else region_selector_.fail_name 
                for region_selector_ in region_selector
            ])
        
        mask_save_name = "+".join(np.sort(mask_type))

        if mask_selector_name not in self.unmasked_area.keys():
            self.unmasked_area[mask_selector_name] = {}
        area_tab_path = f"{config['DEFAULT']['GALFIND_WORK']}/Unmasked_areas/{self.survey}_{self.version}.ecsv"
        if Path(area_tab_path).is_file():
            area_tab = Table.read(area_tab_path)
            funcs.make_dirs(area_tab_path)
            area_tab_ = area_tab[(
                (area_tab["mask_instr_band"] == mask_selector_name) \
                & (area_tab["mask_type"] == mask_save_name) \
                & (area_tab["region"] == reg_name)
            )]
            if len(area_tab_) == 0:
                calculate = True
            else:
                calculate = False
        else:
            calculate = True
            
        if calculate:
            if isinstance(mask_selector, tuple(Mask_Selector.__subclasses__())):
                masks = [mask_selector.load_mask(self, invert = not invert_region)]
                pix_scales = [band_data.pix_scale for band_data in self]
                assert all(pix_scale == pix_scales[0] for pix_scale in pix_scales), \
                    galfind_logger.critical(
                        "All pixel scales must be the same!"
                    )
                pix_scale = pix_scales[0]
            else:
                masks = []
                for name in mask_selector:
                    if name in self.filterset.instrument_name.split("+"):
                        pix_scales = [band_data.pix_scale for band_data in self \
                            if band_data.instr_name == name]
                        assert all(pix_scale == pix_scales[0] for pix_scale in pix_scales), \
                            galfind_logger.critical(
                                "All pixel scales for bands in the same instrument must be the same!"
                            )
                        pix_scale = pix_scales[0]
                        masks.extend([band_data.load_mask()[0][mask_type_] for band_data in self \
                            for mask_type_ in mask_type if band_data.instr_name == name])
                    elif name in self.filterset.band_names:
                        pix_scale = self[name].pix_scale
                        masks.extend([self[name].load_mask()[0][mask_type_] for mask_type_ in mask_type])
                    else:
                        possible_names = self.filterset.instrument_name.split("+") + self.filterset.band_names
                        err_message = f"{name} not in {possible_names}"
                        galfind_logger.critical(
                            err_message
                        )
                        raise(Exception(err_message))
            
            if region_selector is not None:
                for region_selector_ in region_selector:
                    masks.extend([region_selector_.load_mask(self, invert = not invert_region)])
            
            if len(masks) == 0:
                galfind_logger.critical(
                    f"Could not find any masks for {mask_selector_name}"
                )
            elif len(masks) == 1:
                mask = masks[0]
            else:
                mask = np.logical_or.reduce(tuple(masks))
            mask = np.invert(mask)

            if isinstance(mask_selector, list) and len(mask_selector) == 1:
                self[mask_selector_name[0]]._calc_area_given_mask(mask_save_name, mask)
                unmasked_area = self[mask_selector_name[0]].unmasked_area[mask_save_name]
            else:
                unmasked_area = funcs.calc_unmasked_area(mask, pix_scale)

            self.unmasked_area[mask_selector_name][mask_save_name] = unmasked_area

            area_data = {
                "mask_instr_band": [mask_selector_name],
                "mask_type": [mask_save_name],
                "region": [reg_name],
                "unmasked_area": [np.round(self.unmasked_area \
                    [mask_selector_name][mask_save_name].to(out_units), 3)],
            }

            new_area_tab = Table(area_data)
            if Path(area_tab_path).is_file():
                area_tab = vstack([area_tab, new_area_tab])
            else:
                area_tab = new_area_tab
            area_tab.write(area_tab_path, overwrite=True)
            funcs.change_file_permissions(area_tab_path)

        # return unmasked area
        unmasked_area = (
            area_tab[
                area_tab["mask_instr_band"]
                == mask_selector_name
            ]["unmasked_area"][0]
            * area_tab["unmasked_area"].unit
        )
        return unmasked_area

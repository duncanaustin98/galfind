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
import time
import itertools
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
    from . import Multiple_Filter
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
from joblib import Parallel, delayed
from matplotlib.colors import LogNorm, Normalize
from photutils.aperture import (
    SkyCircularAperture,
    aperture_photometry,
)
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
        use_galfind_err: bool = False,
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
    def data_shape(self) -> Tuple[int, int]:
        return self.load_im()[0].shape


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
        assert self._seg_method is not None
        # load science image data and header (and hdul)
        im_data, im_header = self.load_im()
        # load segmentation data and header
        seg_data, seg_header = self.load_seg()
        if incl_mask:
            assert self._mask_method is not None
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
        im_hdul = fits.open(self.im_path)
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
            hdu = fits.open(self.wht_path)[self.wht_ext]
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
            hdu = fits.open(self.rms_err_path)[self.rms_err_ext]
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

    def load_seg(self, incl_hdr: bool = True) -> Tuple[np.ndarray, fits.Header]:
        if not Path(self.seg_path).is_file():
            err_message = (
                f"Segmentation map for {self.survey} "
                f"{self.filt.band_name} at {self.seg_path} is not a .fits image!"
            )
            galfind_logger.critical(err_message)
            raise (Exception(err_message))
        seg_hdul = fits.open(self.seg_path)
        seg_data = seg_hdul[0].data
        seg_header = seg_hdul[0].header
        if incl_hdr:
            return seg_data, seg_header
        else:
            return seg_data

    def load_mask(
        self, ext: Optional[str] = None
    ) -> Optional[Union[np.ndarray, Dict[str, np.ndarray]]]:
        if hasattr(self, "mask_args"):
            # load mask
            if ".fits" in self.mask_path:
                hdul = fits.open(self.mask_path, mode="readonly")
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
                else:
                    mask = {
                        hdu_name: hdul[index].data
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
        return mask

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
            if method == "sextractor":
                self.seg_path = SExtractor.segment_sextractor(
                    self, err_type, config_name=config_name, params_name=params_name, overwrite=overwrite
                )
            else:
                raise (
                    Exception(f"segmentation {method=} not in ['sextractor']")
                )
            self.seg_args = {"err_type": err_type, "method": method, "config_name": config_name, "params_name": params_name}

    def perform_forced_phot(
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
            if method == "sextractor":
                self.forced_phot_path = SExtractor.perform_forced_phot(
                    self, forced_phot_band, err_type, config_name=config_name, params_name=params_name, overwrite=overwrite
                )
            else:
                raise (Exception(f"{method=} not in ['sextractor']"))
            self.forced_phot_args = {
                "forced_phot_band": forced_phot_band,
                "err_type": err_type,
                "method": method,
                "config_name": config_name,
                "params_name": params_name
            }

    def _get_master_tab(
        self, output_ids_loc: bool = False
    ) -> Tuple[Table, List[int]]:
        tab = Table.read(self.forced_phot_path, character_as_bytes=False)
        if self.forced_phot_args["method"] == "sextractor":
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
                    f"{self.forced_phot_args['method']} not in ['sextractor']"
                )
            )
        if output_ids_loc:
            if self.forced_phot_args["method"] == "sextractor":
                append_ids_loc = {
                    "ID": tab["NUMBER"],
                    "X_IMAGE": tab["X_IMAGE"],
                    "Y_IMAGE": tab["Y_IMAGE"],
                    "RA": tab["ALPHA_J2000"],
                    "DEC": tab["DELTA_J2000"],
                }
        else:
            append_ids_loc = {}

        # remove non band-dependent forced photometry parameters
        for i, param in enumerate(id_loc_params):
            if not output_ids_loc:
                tab.remove_column(param)
        # add band suffix to columns
        for i, name in enumerate(tab.columns.copy()):
            if name not in id_loc_params:
                tab.rename_column(name, name + "_" + self.filt_name)
        # combine the astropy tables
        if i == 0:
            master_tab = tab
            len_required = len(master_tab)
        else:
            master_tab = hstack([master_tab, tab])
            assert (
                len(tab) == len_required
            ), f"Lengths of sextractor catalogues do not match! Check same detection image used {len(tab)} != {len_required} for {band}"

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
        angle: Union[float, List[float], Dict[str, float]] = -70.0,
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
                    self.mask_args = mask_args
                    return
            # create fits mask
            if method.lower() == "manual":
                self.mask_path = Masking.manually_mask(
                    self, overwrite=overwrite
                )
                self.mask_args = {"method": method}
            elif method.lower() == "auto":
                self.mask_path, self.mask_args = Masking.auto_mask(
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
        overwrite: bool = False,
    ) -> NoReturn:
        if not hasattr(self, "depth_args"):
            # load parameters (i.e. for each aper_diams in self)
            params = self._sort_run_depth_params(
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
                overwrite,
            )
            # run depths
            Depths.calc_band_depth(params)
            # load depths into object
            self._load_depths(params)
        else:
            galfind_logger.warning(
                f"Depths loaded for {self.filt_name}, skipping!"
            )

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
        overwrite: bool = False,
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
                        overwrite,
                    )
                ]
            )
        return params

    def _load_depths(self, params: List[Tuple[Any, ...]]) -> NoReturn:
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
            self.med_depth = {
                param[1]: depth[0] for depth, param in zip(depths, params)
            }
            self.mean_depth = {
                param[1]: depth[1] for depth, param in zip(depths, params)
            }
            self.depth_args = {
                param[1]: Depths.get_depth_args(param) for param in params
            }

    def plot_depths(self) -> NoReturn:
        Depths.plot_band_data_depth(self)

    def plot_area_depth(
        self,
    ) -> NoReturn:
        Depths.plot_band_data_area_depth(self)

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        ext: str = "SCI",
        norm: Type[Normalize] = LogNorm(vmin=0.0, vmax=10.0),
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
        if ext.lower() in ["sci", "im"]:
            data = self.load_im()[0]
        elif ext.lower() == "rms_err":
            data = self.load_rms_err()[0]
        elif ext.lower() == "wht":
            data = self.load_wht()[0]
        elif ext.lower() == "seg":
            data = self.load_seg()[0]
        elif ext.lower() == "mask":
            data = self.load_mask()
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
        ax.imshow(data, norm=norm, origin="lower")
        # annotate if required
        if show or save:
            label = ""
            plt.title(f"{label} image")
            ax.set_xlabel("X / pix")
            ax.set_ylabel("Y / pix")
        if save:
            label = None
            # plt.savefig(label)
        if show:
            plt.show()

    # def _combine_seg_data_and_mask(self, band=None, seg_data=None, mask=None):
    #     if type(seg_data) != type(None) and type(mask) != type(None):
    #         pass
    #     elif type(band) != type(
    #         None
    #     ):  # at least one of seg_data or mask is not given, but band is given
    #         seg_data = self.load_seg(band)[0]
    #         mask = self.load_mask(band)
    #     else:
    #         raise (
    #             Exception(
    #                 "Either band must be given or both seg_data and mask should be given in Data.combine_seg_data_and_mask()!"
    #             )
    #         )
    #     assert seg_data.shape == mask.shape
    #     combined_mask = np.logical_or(seg_data > 0, mask == 1).astype(int)
    #     return combined_mask

    @staticmethod
    def _pix_scale_to_str(pix_scale: u.Quantity):
        return f"{round(pix_scale.to(u.marcsec).value)}mas"

    def _make_rms_err_from_wht(self):
        # make rms_err map from wht map
        wht, hdr = self.load_wht(output_hdr=True)
        err = 1.0 / (wht**0.5)
        primary_hdr = deepcopy(hdr)
        primary_hdr["EXTNAME"] = "PRIMARY"
        primary = fits.PrimaryHDU(header=primary_hdr)
        hdu = fits.ImageHDU(err, header=hdr, name="ERR")
        hdul = fits.HDUList([primary, hdu])
        # save and overwrite object attributes
        save_path = self.im_path.replace(
            self.im_path.split("/")[-1],
            f"rms_err/{self.filt.band_name}_rms_err.fits",
        )
        funcs.make_dirs(save_path)
        hdul.writeto(save_path, overwrite=True)
        funcs.change_file_permissions(save_path)
        galfind_logger.info(
            f"Finished making {self.survey} {self.version} {self.filt} rms_err map"
        )
        self.rms_err_path = save_path
        self.rms_err_ext = 1
        self.rms_err_ext_name = ["ERR"]
        self._use_galfind_err = True

    def _make_wht_from_rms_err(self):
        err, hdr = self.load_rms_err(output_hdr=True)
        wht = 1.0 / (err**2)
        primary_hdr = deepcopy(hdr)
        primary_hdr["EXTNAME"] = "PRIMARY"
        primary = fits.PrimaryHDU(header=primary_hdr)
        hdu = fits.ImageHDU(wht, header=hdr, name="WHT")
        hdul = fits.HDUList([primary, hdu])
        # save and overwrite object attributes
        save_path = self.im_path.replace(
            self.im_path.split("/")[-1], f"wht/{self.filt.band_name}_wht.fits"
        )
        funcs.make_dirs(save_path)
        hdul.writeto(save_path, overwrite=True)
        funcs.change_file_permissions(save_path)
        galfind_logger.info(
            f"Finished making {self.survey} {self.version} {self.filt} wht map"
        )
        self.wht_path = save_path
        self.wht_ext = 1
        self.rms_err_ext_name = ["WHT"]
        self._use_galfind_err = True


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
        use_galfind_err: bool = False,
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
        raise(NotImplementedError)
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
        return self.filt.instrument.calc_ZP(self)

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
            return Band_Data.from_band_data_arr([deepcopy(self), *band_data_arr])
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
        use_galfind_err: bool = False,
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
        return self.filterset[0].instrument.calc_ZP(self)

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
        used_galfind_err = False
        if err_type.lower() == "rms_err":
            if not all(
                Path(band_data.rms_err_path).is_file()
                for band_data in band_data_arr
            ):
                for band_data in band_data_arr:
                    band_data._make_rms_err_from_wht()
                used_galfind_err = True
        else:  # err_type.lower() == "wht"
            if not all(
                Path(band_data.wht_path).is_file()
                for band_data in band_data_arr
            ):
                for band_data in band_data_arr:
                    band_data._make_wht_from_rms_err()
                used_galfind_err = True
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
                "All band data images must have the same shape!"
            )
            # ensure all band data images have the same ZP
            assert all(
                band_data.ZP == band_data_arr[0].ZP
                for band_data in band_data_arr
            ), galfind_logger.critical(
                "All image ZPs must have the same shape!"
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
        angle: Union[float, List[float], Dict[str, float]] = -70.0,
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
                if not Path(band_data.mask_path).is_file():
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
            Masking.combine_masks(self)


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
        forced_phot_band: Optional[Union[str, List[str], Type[Band_Data_Base]]] = None,
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
            if len(filt_names_paths) == 0:
                galfind_logger.warning(
                    f"No data found for {survey} {version} {instr_name} in {search_dir}"
                )
                continue
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
        #if version_to_dir_dict is not None:
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
            is_sci = {path: [str in path for str in im_str] for path in paths}
            is_rms_err = {
                path: [str in path for str in rms_err_str] for path in paths
            }
            is_wht = {path: [str in path for str in wht_str] for path in paths}
            for path in paths:
                # if all paths are science images
                if all(path_is_sci for path_is_sci in is_sci.values()):
                    # all extensions must be within the same image
                    single_path = True
                    im_paths[filt_name].extend([path])
                    rms_err_paths[filt_name].extend([path])
                    wht_paths[filt_name].extend([path])
                else:
                    # ensure the path only belongs to one (or none) of the image types
                    assert all(
                        not all(i for i in pair)
                        for pair in itertools.combinations(
                            (is_sci, is_rms_err, is_wht), 2
                        )
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

                # extract sci/rms_err/wht extensions
                hdul = fits.open(path)
                if not single_path:
                    assertion_len = 1
                    for j, hdu in enumerate(hdul):
                        # skip primary if there are multiple extensions
                        if hdu.name == "PRIMARY" and len(hdul) > 1:
                            assertion_len += 1
                        else:
                            if is_sci:
                                im_exts[filt_name].extend([int(j)])
                            elif is_rms_err:
                                rms_err_exts[filt_name].extend([int(j)])
                            elif is_wht:
                                wht_exts[filt_name].extend([int(j)])
                    assert len(hdul) == assertion_len
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

    @classmethod
    def pipeline(
        cls,
        survey: str,
        version: str,
    ):
        pass

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
        return Multiple_Filter(band_data.filt for band_data in self)

    @property
    def ZPs(self) -> Dict[str, float]:
        return {band_data.filt_name: band_data.ZP for band_data in self}

    @property
    def pixel_scales(self) -> Dict[str, u.Quantity]:
        return {band_data.filt_name: band_data.pix_scale for band_data in self}

    @property
    def full_name(self):
        return f"{self.survey}_{self.version}_{self.instrument.name}"

    def load_cluster_blank_mask_paths(self):
        # load in cluster core / blank field fits/reg masks
        mask_path_dict = {}
        for mask_type in ["cluster", "blank"]:
            # look for .fits masks first
            fits_masks = glob.glob(
                f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{self.survey}/fits_masks/*_{mask_type}*.fits"
            )
            if len(fits_masks) == 1:
                mask_path = fits_masks[0]
            elif len(fits_masks) > 1:
                galfind_logger.critical(
                    f"Multiple .fits {mask_type} masks exist for {self.survey}!"
                )
            else:
                # no .fits masks, now look for .reg masks
                reg_masks = glob.glob(
                    f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{self.survey}/*_{mask_type}*.reg"
                )
                if len(reg_masks) == 1:
                    mask_path = reg_masks[0]
                elif len(reg_masks) > 1:
                    galfind_logger.critical(
                        f"Multiple .reg {mask_type} masks exist for {self.survey}!"
                    )
                else:
                    # no .reg masks
                    mask_path = None
                    galfind_logger.info(
                        f"No {mask_type} mask found for {self.survey}"
                    )
            mask_path_dict[mask_type] = mask_path
        self.cluster_mask_path = mask_path_dict["cluster"]
        galfind_logger.debug(f"cluster_mask_path = {self.cluster_mask_path}")
        self.blank_mask_path = mask_path_dict["blank"]
        galfind_logger.debug(f"blank_mask_path = {self.blank_mask_path}")

    #     # find common directories for im/seg/rms_err/wht maps
    #     self.common_dirs = {}
    #     galfind_logger.warning(
    #         f"self.common_dirs has errors when the len(rms_err_paths) = {len(self.rms_err_paths)} != len(wht_paths) = {len(self.wht_paths)}"
    #     )
    #     if len(self.rms_err_paths) == len(self.wht_paths):
    #         for paths, key in zip(
    #             [im_paths, seg_paths, mask_paths, rms_err_paths, wht_paths],
    #             ["SCI", "SEG", "MASK", "ERR", "WHT"],
    #         ):
    #             try:
    #                 for band in self.instrument.band_names:
    #                     assert "/".join(
    #                         paths[band].split("/")[:-1]
    #                     ) == "/".join(
    #                         paths[self.instrument.band_names[0]].split("/")[
    #                             :-1
    #                         ]
    #                     )
    #                 self.common_dirs[key] = "/".join(
    #                     paths[self.instrument.band_names[0]].split("/")[:-1]
    #                 )
    #                 galfind_logger.info(
    #                     f"Common directory found for {key}: {self.common_dirs[key]}"
    #                 )
    #             except AssertionError:
    #                 galfind_logger.info(f"No common directory for {key}")

    #     key_failed = {}
    #     for paths, key in zip(
    #         [im_paths, seg_paths, mask_paths, rms_err_paths, wht_paths],
    #         ["SCI", "SEG", "MASK", "ERR", "WHT"],
    #     ):
    #         try:
    #             for band in self.instrument.band_names:
    #                 key_failed[key] = []
    #                 try:
    #                     assert "/".join(
    #                         paths[band].split("/")[:-1]
    #                     ) == "/".join(
    #                         paths[self.instrument.band_names[-1]].split("/")[
    #                             :-1
    #                         ]
    #                     )
    #                 except (AssertionError, KeyError):
    #                     key_failed[key].append(band)
    #             self.common_dirs[key] = "/".join(
    #                 paths[self.instrument.band_names[-1]].split("/")[:-1]
    #             )
    #             galfind_logger.info(
    #                 f"Common directory found for {key}: {self.common_dirs[key]}"
    #             )
    #         except AssertionError:
    #             galfind_logger.info(f"No common directory for {key}")

    #     # find other things in common between bands
    #     self.common = {}
    #     for label, item_dict in zip(
    #         ["ZERO POINT", "PIXEL SCALE", "SCI SHAPE"],
    #         [self.im_zps, self.im_pixel_scales, self.im_shapes],
    #     ):
    #         try:
    #             for band in self.instrument.band_names:
    #                 assert (
    #                     item_dict[band]
    #                     == item_dict[self.instrument.band_names[0]]
    #                 )
    #             self.common[label] = item_dict[self.instrument.band_names[0]]
    #             galfind_logger.info(f"Common {label} found")
    #         except AssertionError:
    #             galfind_logger.info(f"No common {label}")

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
        # output_str += (
        #     "FIELD TYPE: " + "BLANK\n" if self.is_blank else "CLUSTER\n"
        # )
        # print filterset string representation
        output_str += str(self.filterset)
        # print basic data quantities: common ZPs, pixel scales, and SCI image shapes, as well as unmasked sky area and depths should they exist
        # for key, item in self.common.items():
        #     output_str += f"{key}: {item}\n"
        # try:
        #     unmasked_area_tab = self.calc_unmasked_area(
        #         masking_instrument_or_band_name=self.forced_phot_band,
        #         forced_phot_band=self.forced_phot_band,
        #     )
        #     unmasked_area = unmasked_area_tab[
        #         unmasked_area_tab["masking_instrument_band"] == "NIRCam"
        #     ]["unmasked_area_total"][0]
        #     output_str += f"UNMASKED AREA = {unmasked_area}\n"
        # except:
        #     pass
        # try:
        #     depths = []
        #     for aper_diam in (
        #         json.loads(config.get("SExtractor", "APERTURE_DIAMS"))
        #         * u.arcsec
        #     ):
        #         depths.append(self.load_depths(aper_diam, depth_mode))
        #     output_str += f"DEPTHS = {str(depths)}\n"
        # except:
        #     pass
        # # if there are common directories for data, print these
        # if self.common_dirs != {}:
        #     output_str += line_sep
        #     output_str += "SHARED DIRECTORIES:\n"
        #     for key, value in self.common_dirs.items():
        #         output_str += f"{key}: {value}\n"
        #     output_str += line_sep
        # # loop through available bands, printing paths, exts, ZPs, fits shapes
        # output_str += "BAND DATA:\n"
        # for band in self.instrument.band_names:
        #     output_str += band_sep
        #     output_str += f"{band}\n"
        #     if hasattr(self, "sex_cat_types"):
        #         if band in self.sex_cat_types.keys():
        #             output_str += (
        #                 f"PHOTOMETRY BY: {self.sex_cat_types[band]}\n"
        #             )
        #     band_data_paths = [
        #         self.im_paths[band],
        #         self.seg_paths[band],
        #         self.mask_paths[band],
        #     ]
        #     band_data_exts = [self.im_exts[band], 0, 0]
        #     band_data_labels = ["SCI", "SEG", "MASK"]
        #     for paths, exts, label in zip(
        #         [self.rms_err_paths, self.wht_paths],
        #         [self.rms_err_exts, self.wht_exts],
        #         ["ERR", "WHT"],
        #     ):
        #         if band in self.rms_err_paths.keys():
        #             band_data_paths.append(paths[band])
        #             band_data_exts.append(exts[band])
        #             band_data_labels.append(label)
        #     for path, ext, label in zip(
        #         band_data_paths, band_data_exts, band_data_labels
        #     ):
        #         if label in self.common_dirs:
        #             path = path.split("/")[-1]
        #         output_str += f"{label} path = {path}[{str(ext)}]\n"
        #     for label, data in zip(
        #         ["ZERO POINT", "PIXEL SCALE", "SCI SHAPE"],
        #         [self.im_zps, self.im_pixel_scales, self.im_shapes],
        #     ):
        #         if label not in self.common.keys():
        #             output_str += f"{label} = {data[band]}\n"
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
            return {
                band_data.filt_name: getattr(band_data, attr[:-1])
                for band_data in self
            }
        else:
            if attr not in [
                "__array_struct__",
                "__array_interface__",
                "__array__",
            ]:
                galfind_logger.debug(
                    f"Data has no {attr=}!"
                )
            raise AttributeError
    
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
        assert all(isinstance(_other, Band_Data) for _other in other)
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
                        band_data.filt.band_name
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
            assert filt_name in params.keys()
            return params[filt_name]
        else:
            return params

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
    ):
        return self[band].load_mask(ext)

    def load_aper_diams(self, aper_diams: u.Quantity) -> NoReturn:
        if hasattr(self, "forced_phot_band"):
            self.forced_phot_band.load_aper_diams(aper_diams)
        [band_data.load_aper_diams(aper_diams) for band_data in self]

    def psf_homogenize(self):
        pass

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
        [band_data.segment(err_type, method, config_name, params_name, overwrite) for band_data in self]

    def perform_forced_phot(
        self,
        forced_phot_band: Union[str, List[str], Type[Band_Data_Base]],
        err_type: str = "rms_err",
        method: Union[str, List[str], Dict[str, str]] = "sextractor",
        config_name: str = "default.sex",
        params_name: str = "default.param",
        overwrite: bool = False,
    ) -> NoReturn:
        if hasattr(self, "phot_cat_path"):
            raise (Exception("MASTER Photometric catalogue already exists!"))
        # create a forced_phot_band object from given string
        self.load_forced_phot_band(forced_phot_band)
        if isinstance(method, str):
            method = {band_data.filt.band_name: method for band_data in self}
        elif isinstance(method, list):
            assert len(method) == len(self)
            method = {
                band_data.filt.band_name: method[i]
                for i, band_data in enumerate(self)
            }
        else:
            assert isinstance(method, dict)
            assert all(
                band_data.filt.band_name in method.keys() for band_data in self
            )
        # run for every band in the Data object
        [
            band_data.perform_forced_phot(
                forced_phot_band,
                err_type,
                method[band_data.filt.band_name],
                config_name,
                params_name,
                overwrite,
            )
            for band_data in self
        ]
        # ensure method for all bands in the forced phot band are the same
        assert all(
            method[filt_name]
            == method[self.forced_phot_band.filt_name.split("+")[0]]
            for i, filt_name in enumerate(
                self.forced_phot_band.filt_name.split("+")
            )
        )
        # segment and run for the forced_phot_band too
        self.forced_phot_band.segment(err_type, method, config_name, params_name, overwrite)
        self.forced_phot_band.perform_forced_phot(
            self.forced_phot_band,
            err_type,
            method[self.forced_phot_band.filt_name.split("+")[0]],
            config_name,
            params_name,
            overwrite=overwrite,
        )
        # combined forced photometry catalogues into a single photometric catalogue
        self._combine_forced_phot_cats(overwrite=overwrite)

    def load_forced_phot_band(
        self,
        forced_phot_band: Union[str, List[str], Type[Band_Data_Base]],
    ) -> Type[Band_Data_Base]:
        # create a forced_phot_band object from given string
        if isinstance(forced_phot_band, str):
            filt_names = forced_phot_band.split("+")
        elif isinstance(forced_phot_band, list):
            filt_names = forced_phot_band
        if isinstance(forced_phot_band, tuple([str, list])):
            if len(filt_names) == 0:
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

    def _get_phot_cat_path(
        self, forced_phot_band: Type[Band_Data_Base]
    ) -> str:
        # determine photometric catalogue path
        save_dir = (
            f"{config['DEFAULT']['GALFIND_WORK']}/Catalogues/"
            + f"{self.version}/{self.instrument.name}/{self.survey}"
        )
        save_name = (
            f"{self.survey}_MASTER_Sel-"
            + f"{self.forced_phot_band.filt_name}_{self.version}.fits"
        )
        phot_cat_path = f"{save_dir}/{save_name}"
        funcs.make_dirs(phot_cat_path)
        return phot_cat_path

    def _combine_forced_phot_cats(self, overwrite: bool = False) -> NoReturn:
        # readme_sep: str = "-" * 20,
        phot_cat_path = self._get_phot_cat_path(self.forced_phot_band)
        funcs.make_dirs(phot_cat_path)
        if not hasattr(self, "phot_cat_path"):
            self.phot_cat_path = phot_cat_path
        else:
            raise (Exception("MASTER Photometric catalogue already exists!"))
        if not Path(phot_cat_path).is_file() or overwrite:
            master_tab = self.forced_phot_band._get_master_tab(
                output_ids_locs=True
            )
            for i, band_data in enumerate(self):
                master_tab.extend(
                    band_data._get_master_tab(output_ids_locs=False)[0]
                )
            # update table header
            master_tab.meta = {
                **master_tab.meta,
                **{
                    "INSTR": self.filterset.instrument_name,
                    "SURVEY": self.survey,
                    "VERSION": self.version,
                    "BANDS": str(self.filterset.band_names),
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
        angle: Union[float, List[float], Dict[str, float]] = -70.0,
        edge_value: Union[float, List[float], Dict[str, float]] = 0.0,
        element: Union[str, List[str], Dict[str, str]] = "ELLIPSE",
        gaia_row_lim: Union[int, List[int], Dict[str, int]] = 500,
        overwrite: Union[bool, List[bool], Dict[str, bool]] = False,
    ) -> Union[None, NoReturn]:
        assert method in ["auto", "manual"], galfind_logger.warning(
            f"Method {method} not recognised. Must be 'auto' or 'manual'"
        )

        if hasattr(self, "forced_phot_band"):
            print(self, self.forced_phot_band)
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
                self_._sort_band_dependent_params(
                    band_data.filt_name,
                    Masking.sort_band_dependent_star_mask_params(
                        band_data.filt, star_mask_params
                    ),
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
        overwrite: Union[bool, List[bool], Dict[str, bool]] = False,
        timed: bool = False,
    ) -> NoReturn:
        if timed:
            start = time.time()
        print(getattr(self, "forced_phot_band"))
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
                            band_data.filt_name, overwrite
                        ),
                    )
                )
            else:
                galfind_logger.warning(
                    f"Depths for {band_data.filt_name} already run, skipping!"
                )
        if len(params) > 0:
            # Parallelise the calculation of depths for each band
            with tqdm_joblib(
                tqdm(desc="Calculating depths", total=len(params))
            ) as progress_bar:
                Parallel(n_jobs=n_jobs)(
                    delayed(Depths.calc_band_depth)(param) for param in params
                )

            # save properties to individual band_data objects
            for band_data in self.band_data_arr + [self.forced_phot_band]:
                [
                    band_data._load_depths(params)
                    for _params in params
                    if _params[0] == band_data
                ]

            # make depth table
            modes = np.unique([param[2] for param in params])
            [Depths.make_depth_tab(self, mode) for mode in modes]

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

    def plot_depths(
        self, band: Union[int, str, Filter, List[Filter], Multiple_Filter]
    ) -> NoReturn:
        self[band].plot_depths()

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

    def make_RGB(
        self,
        blue_bands=["F090W"],
        green_bands=["F200W"],
        red_bands=["F444W"],
        method="trilogy",
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

    def forced_photometry(
        self,
        band,
        forced_phot_band,
        radii=list(
            np.array(json.loads(config["SExtractor"]["APERTURE_DIAMS"])) / 2.0
        )
        * u.arcsec,
        ra_col="ALPHA_J2000",
        dec_col="DELTA_J2000",
        coord_unit=u.deg,
        id_col="NUMBER",
        x_col="X_IMAGE",
        y_col="Y_IMAGE",
        forced_phot_code="photutils",
    ):
        # Read in sextractor catalogue
        catalog = Table.read(
            self.sex_cat_path(forced_phot_band, forced_phot_band),
            character_as_bytes=False,
        )
        # Open image with correct extension and get WCS
        with fits.open(self.im_paths[band]) as hdul:
            im_ext = self.im_exts[band]
            image = hdul[im_ext].data
            wcs = WCS(hdul[im_ext].header)

        # Check types
        assert type(image) == np.ndarray
        assert type(catalog) == Table

        # Get positions from sextractor catalog
        ra = catalog[ra_col]
        dec = catalog[dec_col]
        # Make SkyCoord from catlog
        positions = SkyCoord(ra, dec, unit=coord_unit)
        # print('positions', positions)
        # Define radii in sky units
        # This checks if radii is iterable and if not makes it a list
        try:
            iter(radii)
        except TypeError:
            radii = [radii]
        apertures = []

        for rad in radii:
            aperture = SkyCircularAperture(positions, r=rad)
            apertures.append(aperture)
            # Convert to pixel using image WCS

        # Do aperture photometry
        # print(image, apertures, wcs)
        phot_table = aperture_photometry(image, apertures, wcs=wcs)
        assert len(phot_table) == len(catalog)
        # Replace detection ID with catalog ID
        sky = SkyCoord(phot_table["sky_center"])
        phot_table["id"] = catalog[id_col]
        # Rename columns
        phot_table.rename_column("id", id_col)
        phot_table.rename_column("xcenter", x_col)
        phot_table.rename_column("ycenter", y_col)

        phot_table[ra_col] = sky.ra.to("deg")
        phot_table[dec_col] = sky.dec.to("deg")
        phot_table.remove_column("sky_center")

        colnames = [f"aperture_sum_{i}" for i in range(len(radii))]
        aper_tab = Column(
            np.array(phot_table[colnames].as_array().tolist()),
            name=f"FLUX_APER_{band}",
        )
        phot_table["FLUX_APER"] = aper_tab
        phot_table["FLUXERR_APER"] = phot_table["FLUX_APER"] * -99
        phot_table["MAGERR_APER"] = phot_table["FLUX_APER"] * 99

        # This converts the fluxes to magnitudes using the correct zp, and puts them in the same format as the sextractor catalogue
        mag_colnames = []
        for pos, col in enumerate(colnames):
            name = f"MAG_APER_{pos}"
            phot_table[name] = (
                -2.5 * np.log10(phot_table[col]) + self.im_zps[band]
            )
            phot_table[name][np.isnan(phot_table[name])] = 99.0
            mag_colnames.append(name)
        aper_tab = Column(
            np.array(phot_table[mag_colnames].as_array().tolist()),
            name=f"MAG_APER_{band}",
        )
        phot_table["MAG_APER"] = aper_tab
        # Remove old columns
        phot_table.remove_columns(colnames)
        phot_table.remove_columns(mag_colnames)
        phot_table.write(
            self.sex_cat_path(band, forced_phot_band),
            format="fits",
            overwrite=True,
        )
        funcs.change_file_permissions(
            self.sex_cat_path(band, forced_phot_band)
        )

    # can be simplified with new masks
    def calc_unmasked_area(
        self,
        masking_instrument_or_band_name="NIRCam",
        forced_phot_band=["F277W", "F356W", "F444W"],
    ):
        if "PIXEL SCALE" not in self.common.keys():
            galfind_logger.warning(
                "Masking by bands with different pixel scales is not supported!"
            )

        if type(masking_instrument_or_band_name) in [list, np.array]:
            masking_instrument_or_band_name = "+".join(
                list(masking_instrument_or_band_name)
            )

        # create a list of bands that need to be unmasked in order to calculate the area
        if type(masking_instrument_or_band_name) in [str, np.str_]:
            masking_instrument_or_band_name = str(
                masking_instrument_or_band_name
            )
            # mask by requiring unmasked criteria in all bands for a given Instrument
            if masking_instrument_or_band_name in [
                subclass.__name__
                for subclass in Instrument.__subclasses__()
                if subclass.__name__ != "Combined_Instrument"
            ]:
                masking_bands = np.array(
                    [
                        band
                        for band in self.instrument.band_names
                        if band
                        in Instrument.from_name(
                            masking_instrument_or_band_name
                        ).bands
                    ]
                )
            elif masking_instrument_or_band_name == "All":
                masking_bands = np.array(self.instrument.band_names)
            else:  # string should contain individual bands, separated by a "+"
                masking_bands = masking_instrument_or_band_name.split("+")
                for band in masking_bands:
                    assert band in json.loads(
                        config.get("Other", "ALL_BANDS")
                    ), galfind_logger.critical(
                        f"{band} is not a valid band currently included in galfind! Cannot calculate unmasked area!"
                    )
        else:
            galfind_logger.critical(
                f"type(masking_instrument_or_band_name) = {type(masking_instrument_or_band_name)} must be in [str, list, np.array]!"
            )

        # make combined mask if required, else load mask
        glob_mask_names = glob.glob(
            f"{self.mask_dir}/fits_masks/*{self.combine_band_names(masking_bands)}_*"
        )
        if "+" not in masking_bands and len(glob_mask_names) > 1:
            for mask in glob_mask_names:
                if "+" in mask:
                    glob_mask_names.remove(mask)

        if len(glob_mask_names) == 0:
            if len(masking_bands) > 1:
                path = self.combine_masks(masking_bands)
                print(path)
                self.mask_paths[masking_instrument_or_band_name] = path
        elif len(glob_mask_names) == 1:
            self.mask_paths[masking_instrument_or_band_name] = glob_mask_names[
                0
            ]
        else:
            raise (
                Exception(
                    f"More than 1 mask for {masking_bands}. Please change this in {self.mask_dir}"
                )
            )
        full_mask = self.load_mask(masking_instrument_or_band_name)

        if self.is_blank:
            blank_mask = full_mask
        else:
            # make combined mask for masking_instrument_name blank field area
            glob_mask_names = glob.glob(
                f"{self.mask_dir}/fits_masks/*{self.combine_band_names(list(masking_bands) + ['blank'])}_*"
            )
            if len(glob_mask_names) == 0:
                self.mask_paths[f"{masking_instrument_or_band_name}+blank"] = (
                    self.combine_masks(list(masking_bands) + ["blank"])
                )
            elif len(glob_mask_names) == 1:
                self.mask_paths[f"{masking_instrument_or_band_name}+blank"] = (
                    glob_mask_names[0]
                )
            else:
                raise (
                    Exception(
                        f"More than 1 mask for {masking_bands}. Please change this in {self.mask_dir}"
                    )
                )
            blank_mask = self.load_mask(
                f"{masking_instrument_or_band_name}+blank"
            )

        # split calculation by depth regions
        galfind_logger.warning(
            "Area calculation for different depth regions not yet implemented!"
        )

        # calculate areas using pixel scale of selection band
        pixel_scale = self.im_pixel_scales[
            self.combine_band_names(forced_phot_band)
        ]
        unmasked_area_tot = (
            ((full_mask.shape[0] * full_mask.shape[1]) - np.sum(full_mask))
            * pixel_scale
            * pixel_scale
        ).to(u.arcmin**2)
        print(unmasked_area_tot)
        unmasked_area_blank_modules = (
            ((blank_mask.shape[0] * blank_mask.shape[1]) - np.sum(blank_mask))
            * pixel_scale
            * pixel_scale
        ).to(u.arcmin**2)
        unmasked_area_cluster_module = (
            unmasked_area_tot - unmasked_area_blank_modules
        )
        galfind_logger.info(
            f"Unmasked areas for {self.survey}, masking_instrument_or_band_name = {masking_instrument_or_band_name} - Total: {unmasked_area_tot}, Blank modules: {unmasked_area_blank_modules}, Cluster module: {unmasked_area_cluster_module}"
        )

        # save in self
        if not hasattr(self, "area"):
            self.area = {}
        self.area["all"] = unmasked_area_tot

        output_path = (
            f"{config['DEFAULT']['GALFIND_WORK']}/Unmasked_areas.ecsv"
        )
        funcs.make_dirs(output_path)
        areas_data = {
            "survey": [self.survey],
            "masking_instrument_band": [masking_instrument_or_band_name],
            "unmasked_area_total": [np.round(unmasked_area_tot, 3)],
            "unmasked_area_blank_modules": [
                np.round(unmasked_area_blank_modules, 3)
            ],
            "unmasked_area_cluster_module": [
                np.round(unmasked_area_cluster_module, 3)
            ],
        }
        areas_tab = Table(areas_data)
        if Path(output_path).is_file():
            existing_areas_tab = Table.read(output_path)
            # if the exact same setup has already been run, overwrite
            existing_areas_tab_ = deepcopy(existing_areas_tab)
            existing_areas_tab["index"] = np.arange(
                0, len(existing_areas_tab), 1
            )

            existing_areas_tab_ = existing_areas_tab[
                (
                    (existing_areas_tab["survey"] == self.survey)
                    & (
                        existing_areas_tab["masking_instrument_band"]
                        == masking_instrument_or_band_name
                    )
                )
            ]
            if len(existing_areas_tab_) > 0:
                # delete existing column using the same setup in favour of new one
                existing_areas_tab.remove_row(
                    int(existing_areas_tab_["index"])
                )
            else:
                areas_tab = vstack([existing_areas_tab, areas_tab])
            for col in areas_tab.colnames:
                if "index" in col:
                    areas_tab.remove_column(col)

        areas_tab.write(output_path, overwrite=True)
        funcs.change_file_permissions(output_path)
        return areas_tab

    def perform_aper_corrs(self):  # not general
        overwrite = config["Depths"].getboolean("OVERWRITE_LOC_DEPTH_CAT")
        if overwrite:
            galfind_logger.info(
                "OVERWRITE_LOC_DEPTH_CAT = YES, updating catalogue with aperture corrections."
            )
        cat = Table.read(self.sex_cat_master_path)
        if "APERCORR" not in cat.meta.keys() or overwrite:
            for i, band in enumerate(self.instrument.band_names):
                print(band)
                mag_aper_corr_data = np.zeros(len(cat))
                flux_aper_corr_data = np.zeros(len(cat))
                for j, aper_diam in enumerate(
                    json.loads(config.get("SExtractor", "APERTURE_DIAMS"))
                    * u.arcsec
                ):
                    # assumes these have already been calculated for each band
                    mag_aper_corr_factor = self.instrument.get_aper_corrs(
                        aper_diam
                    )[i]
                    flux_aper_corr_factor = 10 ** (mag_aper_corr_factor / 2.5)
                    # print(band, aper_diam, mag_aper_corr_factor, flux_aper_corr_factor)
                    if j == 0:
                        # only aperture correct if flux is positive
                        mag_aper_corr_data = [
                            (mag_aper[0] - mag_aper_corr_factor,)
                            if flux_aper[0] > 0.0
                            else (mag_aper[0],)
                            for mag_aper, flux_aper in zip(
                                cat[f"MAG_APER_{band}"],
                                cat[f"FLUX_APER_{band}"],
                            )
                        ]
                        flux_aper_corr_data = [
                            (flux_aper[0] * flux_aper_corr_factor,)
                            if flux_aper[0] > 0.0
                            else (flux_aper[0],)
                            for flux_aper in cat[f"FLUX_APER_{band}"]
                        ]
                    else:
                        mag_aper_corr_data = [
                            mag_aper_corr
                            + (mag_aper[j] - mag_aper_corr_factor,)
                            if flux_aper[j] > 0.0
                            else mag_aper_corr + (mag_aper[j],)
                            for mag_aper_corr, mag_aper, flux_aper in zip(
                                mag_aper_corr_data,
                                cat[f"MAG_APER_{band}"],
                                cat[f"FLUX_APER_{band}"],
                            )
                        ]
                        flux_aper_corr_data = [
                            flux_aper_corr
                            + (flux_aper[j] * flux_aper_corr_factor,)
                            if flux_aper[j] > 0.0
                            else flux_aper_corr + (flux_aper[j],)
                            for flux_aper_corr, flux_aper in zip(
                                flux_aper_corr_data, cat[f"FLUX_APER_{band}"]
                            )
                        ]
                cat[f"MAG_APER_{band}_aper_corr"] = mag_aper_corr_data
                cat[f"FLUX_APER_{band}_aper_corr"] = flux_aper_corr_data
                cat[f"FLUX_APER_{band}_aper_corr_Jy"] = [
                    tuple(
                        [
                            funcs.flux_image_to_Jy(
                                val, self.im_zps[band]
                            ).value
                            for val in element
                        ]
                    )
                    for element in cat[f"FLUX_APER_{band}_aper_corr"]
                ]
            # update catalogue metadata
            # mag_aper_corrs = {f"HIERARCH Mag_aper_corrs_{aper_diam.value}as": tuple([np.round(self.instrument.aper_corr(aper_diam, band), decimals = 4) \
            #    for band in self.instrument.band_names]) for aper_diam in json.loads(config.get("SExtractor", "APERTURE_DIAMS")) * u.arcsec}
            cat.meta = {
                **cat.meta,
                **{"APERCORR": True},
            }  # , **mag_aper_corrs}
            # overwrite original catalogue with local depth columns
            cat.write(self.sex_cat_master_path, overwrite=True)
            funcs.change_file_permissions(self.sex_cat_master_path)

    def make_loc_depth_cat(self, cat_creator, depth_mode="n_nearest"):
        overwrite = config["Depths"].getboolean("OVERWRITE_LOC_DEPTH_CAT")
        if overwrite:
            galfind_logger.info(
                "OVERWRITE_LOC_DEPTH_CAT = YES, updating catalogue with local depths."
            )

        cat = Table.read(self.sex_cat_master_path)
        # update catalogue with local depths if not already done so
        if "DEPTHS" not in cat.meta.keys() or overwrite:
            # mean_depths = {}
            # median_depths = {}
            diagnostic_name = ""
            for i, band in enumerate(self.instrument.band_names):
                # breakpoint()
                galfind_logger.info(f"Making local depth columns for {band=}")
                for j, aper_diam in enumerate(
                    json.loads(config.get("SExtractor", "APERTURE_DIAMS"))
                    * u.arcsec
                ):
                    # print(band, aper_diam)
                    self.load_depth_dirs(aper_diam, depth_mode)
                    h5_path = f"{self.depth_dirs[aper_diam][depth_mode][band]}/{band}.h5"
                    if Path(h5_path).is_file():
                        # open depth .h5
                        hf = h5py.File(h5_path, "r")
                        depths = np.array(hf["depths"])
                        diagnostics = np.array(hf["diagnostic"])
                        diagnostic_name_ = (
                            f"d_{int(np.array(hf['n_nearest']))}"
                            if depth_mode == "n_nearest"
                            else f"n_aper_{float(np.array(hf['region_radius_used_pix'])):.1f}"
                            if depth_mode == "rolling"
                            else None
                        )
                        if diagnostic_name_ == None:
                            raise (Exception("Invalid mode!"))
                        # make sure the same depth setup has been run in each band
                        if i == 0 and j == 0:
                            diagnostic_name = diagnostic_name_
                        assert diagnostic_name_ == diagnostic_name
                        # update depths with average depths in each region
                        nmad_grid = np.array(hf["nmad_grid"])
                        band_mean_depth = np.round(
                            np.nanmean(nmad_grid), decimals=3
                        )
                        band_median_depth = np.round(
                            np.nanmedian(nmad_grid), decimals=3
                        )
                        hf.close()
                    else:
                        depths = np.full(len(cat), np.nan)
                        diagnostics = np.full(len(cat), np.nan)
                        # band_mean_depth = np.nan
                        # band_median_depth = np.nan
                    if j == 0:
                        band_depths = [(depth,) for depth in depths]
                        band_diagnostics = [
                            (diagnostic,) for diagnostic in diagnostics
                        ]
                        band_sigmas = [
                            (
                                funcs.n_sigma_detection(
                                    depth, mag_aper[0], self.im_zps[band]
                                ),
                            )
                            for depth, mag_aper in zip(
                                depths, cat[f"MAG_APER_{band}"]
                            )
                        ]
                        # band_mean_depths = (band_mean_depth,)
                        # band_median_depths = (band_median_depth,)
                    else:
                        band_depths = [
                            band_depth + (aper_diam_depth,)
                            for band_depth, aper_diam_depth in zip(
                                band_depths, depths
                            )
                        ]
                        band_diagnostics = [
                            band_diagnostic + (aper_diam_diagnostic,)
                            for band_diagnostic, aper_diam_diagnostic in zip(
                                band_diagnostics, diagnostics
                            )
                        ]
                        band_sigmas = [
                            band_sigma
                            + (
                                funcs.n_sigma_detection(
                                    depth, mag_aper[j], self.im_zps[band]
                                ),
                            )
                            for band_sigma, depth, mag_aper in zip(
                                band_sigmas, depths, cat[f"MAG_APER_{band}"]
                            )
                        ]
                        # band_mean_depths = band_mean_depths + (band_mean_depth,)
                        # band_median_depths = band_median_depths + (band_median_depth,)

                # update band with depths and diagnostics
                cat[f"loc_depth_{band}"] = band_depths
                cat[f"{diagnostic_name}_{band}"] = band_diagnostics
                cat[f"sigma_{band}"] = band_sigmas
                # make local depth error columns in image units
                cat[f"FLUXERR_APER_{band}_loc_depth"] = [
                    tuple(
                        [
                            funcs.mag_to_flux(val, self.im_zps[band]) / 5.0
                            for val in element
                        ]
                    )
                    for element in band_depths
                ]
                # impose n_pc min flux error and convert to Jy where appropriate
                if "APERCORR" in cat.meta.keys():
                    cat[
                        f"FLUXERR_APER_{band}_loc_depth_{str(int(cat_creator.min_flux_pc_err))}pc_Jy"
                    ] = [
                        tuple(
                            [
                                np.nan
                                if flux == 0.0
                                else funcs.flux_image_to_Jy(
                                    flux, self.im_zps[band]
                                ).value
                                * cat_creator.min_flux_pc_err
                                / 100.0
                                if err / flux
                                < cat_creator.min_flux_pc_err / 100.0
                                and flux > 0.0
                                else funcs.flux_image_to_Jy(
                                    err, self.im_zps[band]
                                ).value
                                for flux, err in zip(flux_tup, err_tup)
                            ]
                        )
                        for flux_tup, err_tup in zip(
                            cat[f"FLUX_APER_{band}_aper_corr"],
                            cat[f"FLUXERR_APER_{band}_loc_depth"],
                        )
                    ]
                else:
                    raise (
                        Exception(
                            f"Couldn't make 'FLUXERR_APER_{band}_loc_depth_{str(int(cat_creator.min_flux_pc_err))}Jy' columns!"
                        )
                    )
                # magnitude and magnitude error columns
                # mean_depths[band] = band_mean_depths
                # median_depths[band] = band_median_depths

            # update catalogue metadata
            cat.meta = {
                **cat.meta,
                **{
                    "DEPTHS": True,
                    "MINPCERR": cat_creator.min_flux_pc_err,
                    "ZEROPNT": str(self.im_zps),
                },
            }  # , "Mean_depths": mean_depths, "Median_depths": median_depths}}
            # print(cat.meta)
            # overwrite original catalogue with local depth columns
            cat.write(self.sex_cat_master_path, overwrite=True)
            funcs.change_file_permissions(self.sex_cat_master_path)


# The below makes TQDM work with joblib
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

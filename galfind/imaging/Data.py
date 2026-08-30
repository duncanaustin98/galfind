#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Imaging data management for single photometric bands.

Provides Band_Data_Base abstract class and concrete Band_Data class for
wrapping science images, RMS/weight maps, segmentation maps, masks, PSFs, and
depths. Handles loading, masking, segmentation, forced photometry, and depth
calculation.
"""

from __future__ import annotations

import glob
import itertools
import json
import logging
import os
import shutil
import sys
import time

# from reproject import reproject_adaptive
from abc import ABC, abstractmethod
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
    Type,
    Union,
)

import astropy
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from numpy.typing import NDArray

try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

if TYPE_CHECKING:
    from . import (
        Mask_Selector,
        Multiple_Filter,
        PSF_Base,
        PSF_Cutout,
        Region_Selector,
    )

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import convolve, convolve_fft
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import QTable, Table, hstack, vstack
from astropy.wcs import WCS

# from astroquery.gaia import Gaia
from joblib import Parallel, delayed
from matplotlib.colors import LogNorm, Normalize
from tqdm import tqdm

from .. import config, galfind_logger
from ..photometry import Photutils, SExtractor
from ..utils import Depths, Masking
from ..utils import useful_funcs_austind as funcs
from ..utils.exceptions import (
    AbstractMethodError,
    ExternalToolError,
    GalfindError,
    GalfindTypeError,
    IncompatibleKwargsError,
    InvalidOptionError,
    InvalidUnitError,
    LengthMismatchError,
    MissingDataError,
    MissingFileError,
    MissingKeyError,
    RangeError,
)
from .Filter import Filter, Multiple_Filter
from .Instrument import (  # noqa F501
    ACS_SBC,
    ACS_WFC,
    MIRI,
    WFC3_IR,
    Instrument,
    NIRCam,
)

morgan_version_to_dir = {
    "v8b": "mosaic_1084_wispfix",
    "v8c": "mosaic_1084_wispfix2",
    "v8d": "mosaic_1084_wispfix3",
    "v9": "mosaic_1084_wisptemp2",
    "v10": "mosaic_1084_wispscale",
    "v11": "mosaic_1084_wispnathan",
    "v12": "mosaic_1210_wispnathan",
    "v12test": "mosaic_1210_wispnathan_test",  # not sure if this is needed?
    "v13": "mosaic_1293_wispnathan",
    "v14": "mosaic_1364_wispnathan",
}


class Band_Data_Base(ABC):
    """Abstract base class representing imaging data for a single band.

    Wraps the science image, RMS error/weight maps, and derived products
    (segmentation map, mask, PSF, depths) for one photometric band of a
    given survey/version/pixel scale, and provides the common loading,
    masking, segmentation, forced photometry, and depth-calculation
    machinery shared by `Band_Data` (a single filter) and
    `Stacked_Band_Data` (a stack of multiple filters). Concrete subclasses
    must implement the `instr_name`, `filt_name`, and `ZP` properties.

    Parameters
    ----------
    survey : `str`
        Name of the survey/field this data belongs to.
    version : `str`
        Data reduction version string.
    im_path : `str`
        Path to the FITS file containing the science image.
    im_ext : `int`
        FITS extension index of the science image within `im_path`.
    rms_err_path : `str`, optional
        Path to the FITS file containing the RMS error map. Default is `None`.
    rms_err_ext : `int`, optional
        FITS extension index of the RMS error map. Default is `None`.
    wht_path : `str`, optional
        Path to the FITS file containing the weight map. Default is `None`.
    wht_ext : `int`, optional
        FITS extension index of the weight map. Default is `None`.
    pix_scale : `astropy.units.Quantity`, optional
        Pixel scale of the imaging. Default is `0.03 * u.arcsec`.
    im_ext_name : `str` or `list` of `str`, optional
        Expected FITS `EXTNAME`(s) for the science image extension.
        Default is `"SCI"`.
    rms_err_ext_name : `str` or `list` of `str`, optional
        Expected FITS `EXTNAME`(s) for the RMS error extension.
        Default is `"ERR"`.
    wht_ext_name : `str` or `list` of `str`, optional
        Expected FITS `EXTNAME`(s) for the weight extension.
        Default is `"WHT"`.
    use_galfind_err : `bool`, optional
        If `True`, automatically derive a missing RMS error map from the
        weight map (or vice versa). Default is `True`.
    aper_diams : `astropy.units.Quantity`, optional
        Aperture diameters to associate with this band. Default is `None`.
    psf : `PSF_Base`, optional
        PSF object associated with this band's imaging. Default is `None`.

    Attributes
    ----------
    survey : `str`
        Survey/field name.
    version : `str`
        Data reduction version string.
    im_path, rms_err_path, wht_path : `str`
        Paths to the science, RMS error, and weight FITS files.
    im_ext, rms_err_ext, wht_ext : `int`
        FITS extension indices of the science, RMS error, and weight data.
    pix_scale : `astropy.units.Quantity`
        Pixel scale of the imaging.
    aper_diams : `astropy.units.Quantity`
        Aperture diameters associated with this band, if loaded.
    psf : `PSF_Base` or `None`
        PSF object associated with this band's imaging.
    is_native : `bool`
        Whether this object represents the native (pre-PSF-homogenized)
        version of the data.
    """

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
        psf: Optional[Type[PSF_Base]] = None,
    ):
        """Initialize the Band_Data_Base instance.

        See the class docstring for detailed parameter descriptions.
        """
        self.survey = survey
        self.version = version
        # store paths as absolute, since methods decorated with
        # `run_in_dir` change the working directory before running, which
        # would otherwise break resolution of relative paths
        self.im_path = os.path.abspath(im_path)
        self.im_ext = im_ext
        self.im_ext_name = im_ext_name
        self.rms_err_path = (
            None if rms_err_path is None else os.path.abspath(rms_err_path)
        )
        self.rms_err_ext = rms_err_ext
        self.rms_err_ext_name = rms_err_ext_name
        self.wht_path = None if wht_path is None else os.path.abspath(wht_path)
        self.wht_ext = wht_ext
        self.wht_ext_name = wht_ext_name
        self.pix_scale = pix_scale
        if aper_diams is not None:
            self.aper_diams = aper_diams
        self.psf = psf
        self.is_native = False

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
        """`str`: Name of the instrument this band's data was taken with.

        Must be implemented by subclasses.
        """
        pass

    @property
    @abstractmethod
    def filt_name(self) -> str:
        """`str`: Name of the filter (or combination of filters, for a
        stack) this data corresponds to.

        Must be implemented by subclasses.
        """
        pass

    @property
    @abstractmethod
    def ZP(self) -> float:
        """`float`: Photometric zero point of the image.

        Must be implemented by subclasses.
        """
        pass

    @property
    def data_shape(self) -> Tuple[int, int]:
        """`tuple` of `int`: Pixel dimensions ``(ny, nx)`` of the science
        image, loaded from the FITS file on access.
        """
        return self.load_im()[0].shape

    def __repr__(self) -> str:
        """Return the official string representation of the Band_Data object.

        Returns
        -------
        `str`
            Representation showing class name, instrument, and filter.
        """
        return f"{self.__class__.__name__}({self.instr_name}/{self.filt_name})"

    def __str__(self) -> str:
        """Return a human-readable string representation of the
        Band_Data object.

        Returns
        -------
        `str`
            Formatted string with survey details, paths, and depth information.
        """
        output_str = funcs.line_sep
        output_str += (
            f"{repr(self)} "
            f"{self.__class__.__name__.upper().replace('_', ' ')}:\n"
        )
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
                output_str += (
                    f"{attr.upper().replace('_', ' ')} PATH: "
                    + f"{getattr(self, f'{attr}_path')}["
                    + f"{getattr(self, f'{attr}_ext')}]\n"
                )
        for attr in ["mask", "seg", "forced_phot"]:
            if hasattr(self, f"{attr}_args"):
                output_str += (
                    f"{attr.upper().replace('_', ' ')}"
                    + f" PATH: {getattr(self, f'{attr}_path')}\n"
                )
                output_str += (
                    f"{attr.upper().replace('_', ' ')}"
                    + f" ARGS: {getattr(self, f'{attr}_args')}\n"
                )
        if hasattr(self, "depth_args"):
            output_str += funcs.line_sep
            output_str += "DEPTHS:\n"
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
        """Compare two Band_Data objects for equality.

        Checks if all configuration attributes (paths, extensions, pixel scale)
        are identical between two Band_Data instances.

        Parameters
        ----------
        other : `Band_Data_Base`
            Another Band_Data object to compare with.

        Returns
        -------
        `bool`
            `True` if all attributes are equal, `False` otherwise.
        """
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
        """Create a shallow copy of the Band_Data object.

        Returns
        -------
        `Band_Data_Base`
            A shallow copy of this Band_Data instance.
        """
        # copy the object
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(result, k, v)
        return result

    def __deepcopy__(self, memo):
        """Create a deep copy of the Band_Data object.

        Parameters
        ----------
        memo : `dict`
            Memo dictionary tracking already-copied objects.

        Returns
        -------
        `Band_Data_Base`
            A deep copy of this Band_Data instance.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            try:
                setattr(result, key, deepcopy(value, memo))
            except Exception:
                galfind_logger.critical(
                    f"deepcopy({self.__class__.__name__}) {key}: {value} FAIL!"
                )
                breakpoint()
        return result

    def _check_data(
        self: Type[Self], incl_rms_err: bool = True, incl_wht: bool = True
    ):
        """Validate and normalize the data file configuration.

        Ensures all FITS extension names are in list format and checks that
        required science image and error/weight data are available.

        Parameters
        ----------
        incl_rms_err : `bool`, optional
            Whether to require RMS error data. Default is `True`.
        incl_wht : `bool`, optional
            Whether to require weight data. Default is `True`.

        Raises
        ------
        ExternalToolError
            If the science image, RMS error, or weight FITS file's
            `EXTNAME` header keyword is not one of the extension names
            configured for this band.
        """
        # make im_ext_name lists if not already
        if isinstance(self.im_ext_name, str):
            self.im_ext_name = [self.im_ext_name]
        if isinstance(self.rms_err_ext_name, str):
            self.rms_err_ext_name = [self.rms_err_ext_name]
        if isinstance(self.wht_ext_name, str):
            self.wht_ext_name = [self.wht_ext_name]
        # load image header
        im_hdr = self.load_im()[1]
        if im_hdr["EXTNAME"] not in self.im_ext_name:
            raise ExternalToolError(
                f"Image extension name EXTNAME={im_hdr['EXTNAME']!r} "
                f"not in expected im_ext_name={self.im_ext_name!r} for "
                f"filt={self.filt.filt_name!r}."
            )
        if incl_rms_err:
            # load rms error header
            rms_err_hdr = self.load_rms_err(output_hdr=True)[1]
            if rms_err_hdr["EXTNAME"] not in self.rms_err_ext_name:
                raise ExternalToolError(
                    f"RMS error file at rms_err_path="
                    f"{self.rms_err_path!r} has "
                    f"EXTNAME={rms_err_hdr['EXTNAME']!r}, not in expected "
                    f"rms_err_ext_name={self.rms_err_ext_name!r}."
                )
        if incl_wht:
            # load weight header
            wht_hdr = self.load_wht(output_hdr=True)[1]
            if wht_hdr["EXTNAME"] not in self.wht_ext_name:
                raise ExternalToolError(
                    f"Weight extension name EXTNAME={wht_hdr['EXTNAME']!r} "
                    f"not in expected wht_ext_name={self.wht_ext_name!r} "
                    f"for filt={self.filt.filt_name!r}."
                )

    def _check_aper_diams(self: Self) -> NoReturn:
        """Validate that aperture diameters are properly configured.

        Checks that aperture diameters have been loaded and are astropy
        Quantities with angular units.

        Raises
        ------
        MissingDataError
            If aperture diameters have not been loaded for this band.
        GalfindTypeError
            If `self.aper_diams` is not an `astropy.units.Quantity`.
        InvalidUnitError
            If `self.aper_diams` does not have angular units.
        """
        if hasattr(self, "aper_diams"):
            if not isinstance(self.aper_diams, u.Quantity):
                raise GalfindTypeError(
                    f"aper_diams for filt_name={self.filt_name!r} has "
                    f"type={type(self.aper_diams).__name__}; must be an "
                    f"astropy.units.Quantity."
                )
            elif not u.get_physical_type(self.aper_diams.unit) == "angle":
                raise InvalidUnitError(
                    f"aper_diams for filt_name={self.filt_name!r} has "
                    f"unit={self.aper_diams.unit!r} with physical_type="
                    f"{u.get_physical_type(self.aper_diams.unit)!r}; must "
                    f"have angular units."
                )
            else:
                pass
        else:
            raise MissingDataError(
                f"aper_diams not loaded for filt_name={self.filt_name!r}. "
                f"Call set_aper_diams() first."
            )

    # %% Loading methods

    def set_aper_diams(self: Self, aper_diams: u.Quantity) -> NoReturn:
        """Set the aperture diameters to use for this band, if not already set.

        If aperture diameters are already loaded for this band, the call is a
        no-op (a debug message is logged instead of overwriting them).

        Parameters
        ----------
        aper_diams : `astropy.units.Quantity`
            Aperture diameters (angular units) to associate with this band.

        Raises
        ------
        Exception
            If `aper_diams` is not an `astropy.units.Quantity` with angular
            units (raised via `_check_aper_diams`).
        """
        if (
            hasattr(self, "aper_diams")
            and getattr(self, "aper_diams", None) is not None
        ):
            galfind_logger.debug(
                f"{self.aper_diams=} already loaded for {self.filt_name},"
                + f" skipping {aper_diams=} load-in"
            )
        else:
            self.aper_diams = aper_diams
            self._check_aper_diams()
            galfind_logger.info(f"Loaded {aper_diams=} for {self.filt_name}")

    def load_data(self, incl_mask: bool = True):
        """Load the science image, segmentation map, and (optionally) mask.

        Parameters
        ----------
        incl_mask : `bool`, optional
            If `True`, also load and return the mask. Default is `True`.

        Returns
        -------
        `tuple`
            ``(im_data, im_header, seg_data, seg_header)`` if `incl_mask` is
            `False`, or ``(im_data, im_header, seg_data, seg_header, mask)``
            if `incl_mask` is `True`.

        Raises
        ------
        MissingDataError
            If segmentation has not yet been performed for this band (i.e.
            `seg_args` is not set).
        """
        if not hasattr(self, "seg_args"):
            raise MissingDataError(
                f"seg_args not set for filt_name={self.filt_name!r}; "
                f"segmentation has not been performed yet. Run "
                f"segmentation first."
            )
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
        self: Self,
        return_hdul: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Union[
        Tuple[np.ndarray, fits.Header],
        Tuple[np.ndarray, fits.Header, fits.HDUList],
    ]:
        """Load the science image data and header from `im_path`/`im_ext`.

        Parameters
        ----------
        return_hdul : `bool`, optional
            If `True`, also return the opened `astropy.io.fits.HDUList`.
            Default is `False`.
        **kwargs : `dict`
            Additional keyword arguments passed to `astropy.io.fits.open`.

        Returns
        -------
        `tuple`
            ``(im_data, im_header)``, or ``(im_data, im_header, im_hdul)``
            if `return_hdul` is `True`.

        Raises
        ------
        MissingFileError
            If `im_path` does not point to an existing FITS file.
        """
        # load image data and header
        if not Path(self.im_path).is_file():
            raise MissingFileError(
                f"Image for survey={self.survey!r} "
                f"filt_name={self.filt_name!r} at im_path="
                f"{self.im_path!r} is not an existing .fits image."
            )
        im_hdul = fits.open(self.im_path, ignore_missing_simple=True, **kwargs)
        im_data = im_hdul[self.im_ext].data
        # im_data = im_data.byteswap().newbyteorder() slow
        im_header = im_hdul[self.im_ext].header
        if return_hdul:
            return im_data, im_header, im_hdul
        else:
            return im_data, im_header

    def load_wcs(self: Type[Self]) -> WCS:
        """Load (and cache) the WCS of the science image.

        Returns
        -------
        `astropy.wcs.WCS`
            The world coordinate system of the science image header.
        """
        try:
            self.wcs
        except (AttributeError, KeyError) as e:
            if isinstance(e, AttributeError):
                self.wcs = {}
            self.wcs = WCS(self.load_im()[1])
        return self.wcs

    def load_wht(
        self: Type[Self],
        output_hdr: bool = False,
        return_hdul: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Union[Tuple[np.ndarray, fits.Header], np.ndarray]:
        """Load the weight map data (and optionally header/HDUList).

        Parameters
        ----------
        output_hdr : `bool`, optional
            If `True`, also return the FITS header. Default is `False`.
        return_hdul : `bool`, optional
            If `True`, also return the opened `astropy.io.fits.HDUList`.
            Default is `False`.
        **kwargs : `dict`
            Additional keyword arguments passed to `astropy.io.fits.open`.

        Returns
        -------
        `numpy.ndarray` or `tuple`
            The weight map data, optionally accompanied by the header
            and/or HDUList depending on `output_hdr`/`return_hdul`.
            Returns `None` (with a logged critical error) in place of the
            data if `wht_path` does not point to an existing FITS file.
        """
        if Path(self.wht_path).is_file():
            hdul = fits.open(
                self.wht_path, ignore_missing_simple=True, **kwargs
            )
            hdu = hdul[self.wht_ext]
            wht = hdu.data
            hdr = hdu.header
        else:
            err_message = (
                f"Weight image for {self.survey} {self.filt_name}"
                + f" at {self.wht_path} is not a .fits image!"
            )
            galfind_logger.critical(err_message)
            wht = None
            hdr = None
        if output_hdr:
            if return_hdul:
                return wht, hdr, hdul
            else:
                return wht, hdr
        else:
            if return_hdul:
                return wht, hdul
            else:
                return wht

    def load_rms_err(
        self: Type[Self],
        output_hdr: bool = False,
        return_hdul: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Union[Tuple[np.ndarray, fits.Header], np.ndarray]:
        """Load the RMS error map data (and optionally header/HDUList).

        Parameters
        ----------
        output_hdr : `bool`, optional
            If `True`, also return the FITS header. Default is `False`.
        return_hdul : `bool`, optional
            If `True`, also return the opened `astropy.io.fits.HDUList`.
            Default is `False`.
        **kwargs : `dict`
            Additional keyword arguments passed to `astropy.io.fits.open`.

        Returns
        -------
        `numpy.ndarray` or `tuple`
            The RMS error map data, optionally accompanied by the header
            and/or HDUList depending on `output_hdr`/`return_hdul`.
            Returns `None` (with a logged critical error) in place of the
            data if `rms_err_path` does not point to an existing FITS file.
        """
        if Path(self.rms_err_path).is_file():
            hdul = fits.open(
                self.rms_err_path, ignore_missing_simple=True, **kwargs
            )
            hdu = hdul[self.rms_err_ext]
            rms_err = hdu.data
            hdr = hdu.header
        else:
            err_message = (
                f"RMS error for {self.survey} {self.filt_name}"
                + f" at {self.rms_err_path} is not a .fits image!"
            )
            galfind_logger.critical(err_message)
            rms_err = None
            hdr = None
        if output_hdr:
            if return_hdul:
                return rms_err, hdr, hdul
            else:
                return rms_err, hdr
        else:
            if return_hdul:
                return rms_err, hdul
            else:
                return rms_err

    def load_seg(
        self: Self,
        incl_hdr: bool = True,
        **kwargs: Dict[str, Any],
    ) -> Tuple[np.ndarray, fits.Header]:
        """Load the segmentation map data (and optionally header).

        Parameters
        ----------
        incl_hdr : `bool`, optional
            If `True`, also return the FITS header. Default is `True`.
        **kwargs : `dict`
            Additional keyword arguments passed to `astropy.io.fits.open`.

        Returns
        -------
        `numpy.ndarray` or `tuple`
            The segmentation map data, or ``(seg_data, seg_header)`` if
            `incl_hdr` is `True`.

        Raises
        ------
        MissingFileError
            If `seg_path` does not point to an existing FITS file.
        """
        # TODO: load from the correct hdu rather than the first one
        if not Path(self.seg_path).is_file():
            raise MissingFileError(
                f"Segmentation map for survey={self.survey!r} "
                f"filt_name={self.filt_name!r} at seg_path="
                f"{self.seg_path!r} is not an existing .fits image."
            )
        seg_hdul = fits.open(
            self.seg_path, ignore_missing_simple=True, **kwargs
        )
        if np.sum(["SEG" in hdu.name.upper() for hdu in seg_hdul]) == 1:
            seg_hdu = [hdu for hdu in seg_hdul if "SEG" in hdu.name.upper()][0]
        else:
            seg_hdu = seg_hdul[0]
        seg_data = seg_hdu.data
        seg_header = seg_hdu.header
        if incl_hdr:
            return seg_data, seg_header
        else:
            return seg_data

    def load_mask(
        self: Self,
        ext: Optional[str] = None,
        invert: bool = False,
    ) -> Optional[Union[np.ndarray, Dict[str, np.ndarray]]]:
        """Load the mask data (and header) for this band, if masking has
        been performed.

        Parameters
        ----------
        ext : `str`, optional
            Name of a specific mask extension to load (e.g. ``"MASK"``). If
            `None`, all mask extensions are returned as dictionaries keyed
            by extension name. Default is `None`.
        invert : `bool`, optional
            If `True` (and `ext` is given), invert the boolean mask before
            returning it. Default is `False`.

        Returns
        -------
        `tuple`
            ``(mask, hdr)`` where `mask` (and `hdr`) is either a single
            `numpy.ndarray` (`astropy.io.fits.Header`) if `ext` is given, or
            a `dict` of `numpy.ndarray` (`astropy.io.fits.Header`) keyed by
            extension name otherwise. Both elements are `None` if no mask
            has been set for this band.

        Raises
        ------
        MissingKeyError
            If `ext` is given but is not one of the extension names present
            in the mask FITS file.
        """
        if hasattr(self, "mask_args"):
            # load mask
            if ".fits" in self.mask_path:
                hdul = fits.open(
                    self.mask_path, mode="readonly", ignore_missing_simple=True
                )
                hdu_names_indices = {
                    hdu.name.upper(): i
                    for i, hdu in enumerate(hdul)
                    if hdu.name != "PRIMARY"
                }
                if ext is not None:
                    ext = ext.upper()
                    if ext not in hdu_names_indices.keys():
                        raise MissingKeyError(
                            f"ext={ext!r} not in mask extensions: "
                            f"{list(hdu_names_indices.keys())}."
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

    def get_area_tab_path(self: Self) -> str:
        """Get the path to the unmasked-area lookup table for this
        survey/version.

        Creates the parent directory if it does not already exist.

        Returns
        -------
        `str`
            Path to the ``.ecsv`` unmasked-area table.
        """
        area_tab_path = (
            f"{config['DEFAULT']['GALFIND_WORK']}/"
            + f"Unmasked_areas/{self.survey}_{self.version}.ecsv"
        )
        funcs.make_dirs(area_tab_path)
        return area_tab_path

    # %% Complex methods

    def psf_homogenize(
        self: Self,
        psf: PSF_Cutout,
        use_fft_conv: bool = True,
        overwrite: bool = False,
    ) -> None:
        """PSF-homogenize this band's science, RMS error, and weight images
        to a target PSF.

        Constructs a convolution kernel from this band's PSF to the target
        `psf`, convolves the science/RMS-error/weight images with it (or
        simply symlinks them if the kernel is trivial), writes the results
        to a new version directory, and updates `self` in place (`im_path`,
        `rms_err_path`, `wht_path`, `version`, and `psf` are all updated to
        point at the new, homogenized data).

        Parameters
        ----------
        psf : `PSF_Cutout`
            Target PSF to homogenize this band's imaging to.
        use_fft_conv : `bool`, optional
            If `True`, use FFT-based convolution
            (`astropy.convolution.convolve_fft`);
            otherwise use direct convolution. Default is `True`.
        overwrite : `bool`, optional
            If `True`, redo the convolution even if the output files
            already exist. Default is `False`.

        Raises
        ------
        MissingDataError
            If no PSF has been loaded for this band (`self.psf` not set).
        """
        if self.psf is None:
            raise MissingDataError(
                f"psf not loaded for filt_name={self.filt_name!r}; cannot "
                f"make PSF homogenization kernel."
            )

        if use_fft_conv:
            convolve_func = convolve_fft
            convolve_kwargs = {"allow_huge": True}
        else:
            convolve_func = convolve
            convolve_kwargs = {}

        filenames = {
            "SCI": self.im_path,
            "RMS_ERR": self.rms_err_path,
            "WHT": self.wht_path,
        }
        open_funcs = {
            "SCI": self.load_im,
            "RMS_ERR": self.load_rms_err,
            "WHT": self.load_wht,
        }
        # Some extensions (e.g. WHT for a band with no dedicated weight
        # file) fall back to reusing another extension's original file,
        # possibly from a different subdirectory with the same basename
        # (e.g. "F090W_test.fits" alongside "wht/F090W_test.fits"). The
        # output paths below are built from basenames only (the
        # subdirectory is stripped), so any extensions sharing a
        # basename would otherwise collide and silently overwrite each
        # other's freshly-convolved output -- suffix any extension
        # whose basename isn't unique among the others to avoid that.
        basenames = {
            ext: name.split("/")[-1]
            for ext, name in filenames.items()
            if name is not None
        }
        needs_suffix = {
            ext: list(basenames.values()).count(basename) > 1
            for ext, basename in basenames.items()
        }
        orig_wht = self.load_wht(output_hdr=False)

        # determine new version and data directory based on kernel choice
        new_version = f"{self.version}_psfmatch_{psf.name}"
        new_data_dir = Data._get_data_dir(
            self.survey,
            new_version,
            self.filt.instrument,
            self.pix_scale,
        )

        # make psf homogenization kernel
        kernel = self.psf.make_kernel(psf)
        if kernel is None:
            for ext, filename in filenames.items():
                if filename is not None:
                    galfind_logger.debug(f"Symlinking {repr(self)} {ext}!")
                    out_filepath = f"{new_data_dir}/{filename.split('/')[-1]}"
                    if needs_suffix[ext]:
                        # add ext to suffix
                        out_filepath = out_filepath.replace(
                            ".fits", f"_{ext.lower()}.fits"
                        )
                    funcs.symlink(filename, out_filepath)
            setattr(self, "version", new_version)
            return
        # convolve the image using the kernel
        kernel_hdr, kernel_data = kernel.cutout.load()

        for ext in ["SCI", "RMS_ERR", "WHT"]:
            filename = filenames[ext]
            if filename is not None:
                out_filepath = f"{new_data_dir}/{filename.split('/')[-1]}"
                if needs_suffix[ext]:
                    # add ext to suffix
                    out_filepath = out_filepath.replace(
                        ".fits", f"_{ext.lower()}.fits"
                    )
                if not Path(out_filepath).is_file() or overwrite:
                    galfind_logger.info(
                        f"Convolving {repr(self)} {ext} with "
                        f"{repr(kernel)} kernel!"
                    )
                    starttime = time.time()
                    data, hdr = open_funcs[ext](output_hdr=True)
                    out_hdul = fits.HDUList([])
                    if ext == "WHT":
                        rms_err = np.where(data == 0, 0, 1 / np.sqrt(data))
                        out_rms_err_data = convolve_func(
                            rms_err, kernel_data, **convolve_kwargs
                        ).astype(np.float32)
                        # convert back to weight map, retaining 0's
                        convolved_data = np.where(
                            out_rms_err_data == 0,
                            0,
                            1.0 / (out_rms_err_data**2),
                        )
                    else:
                        convolved_data = convolve_func(
                            data, kernel_data, **convolve_kwargs
                        ).astype(np.float32)
                    convolved_data[orig_wht == 0] = 0.0

                    out_hdu = fits.PrimaryHDU(
                        convolved_data, header=fits.Header(hdr)
                    )
                    out_hdu.name = ext
                    out_hdu.header["PSFHOMOG"] = kernel.name
                    out_hdul.append(out_hdu)
                    funcs.make_dirs(out_filepath)
                    out_hdul.writeto(out_filepath, overwrite=True)
                    endtime = time.time()
                    galfind_logger.info(
                        f"Written file to {out_filepath}! "
                        f"Convolution took {endtime - starttime:.1f} seconds!"
                    )
                    funcs.change_file_permissions(out_filepath)
                else:
                    galfind_logger.debug(
                        f"{new_version} {repr(self)} {ext} "
                        f"image exists! Skipping!"
                    )
            # update self paths to point to new psf homogenized imaging
            if ext == "SCI":
                self.im_path = out_filepath
                self.im_ext = 0
                self.im_ext_name = [ext]
            elif ext == "RMS_ERR":
                self.rms_err_path = out_filepath
                self.rms_err_ext = 0
                self.rms_err_ext_name = [ext]
            else:  # ext == "WHT":
                self.wht_path = out_filepath
                self.wht_ext = 0
                self.wht_ext_name = [ext]

        # update the version and psf attributes
        setattr(self, "version", new_version)
        setattr(self, "psf", psf)

    @staticmethod
    def _parallel_psf_homogenize(params: Dict[str, Any]) -> None:
        """Perform PSF homogenization for a single band (
            parallel worker function).

        Parameters
        ----------
        params : `dict`
            Dictionary containing ``band_data`` (Band_Data instance),
            ``psf`` (PSF object), ``use_fft_conv`` (bool), and
            ``overwrite`` (bool).
        """
        # unpack parameters
        band_data, psf, use_fft_conv, overwrite = params
        # run psf homogenization
        band_data.psf_homogenize(
            psf,
            use_fft_conv=use_fft_conv,
            overwrite=overwrite,
        )

    def segment(
        self: Self,
        err_type: str = "rms_err",
        method: str = "sextractor",
        config_name: str = "default.sex",
        params_name: str = "default.param",
        overwrite: bool = False,
    ) -> None:
        """Segment the image using the specified method and error type, if
        not already done.

        Parameters
        ----------
        err_type : `str`, optional
            The type of error map to use for segmentation. Default is
            ``"rms_err"``.
        method : `str`, optional
            The segmentation method to use; must contain ``"sextractor"``.
            Default is ``"sextractor"``.
        config_name : `str`, optional
            Name of the SExtractor configuration file to use. Default is
            ``"default.sex"``.
        params_name : `str`, optional
            Name of the SExtractor output parameters file to use. Default
            is ``"default.param"``.
        overwrite : `bool`, optional
            Whether to overwrite existing segmentation data. Default is
            `False`.

        Notes
        -----
        The segmentation arguments used are stored in the `seg_args`
        attribute, and the resulting segmentation map path in `seg_path`.

        Raises
        ------
        InvalidOptionError
            If `method` does not contain ``"sextractor"``.
        """
        # do not re-segment if already done
        if (
            not (hasattr(self, "seg_args") and hasattr(self, "seg_path"))
            or overwrite
        ):
            # segment the data
            if "sextractor" in method.lower():
                self.seg_path = SExtractor.segment(
                    self,
                    err_type,
                    config_name=config_name,
                    params_name=params_name,
                    overwrite=overwrite,
                )
            else:
                raise InvalidOptionError(
                    f"segmentation method={method.lower()!r} does not "
                    f"contain 'sextractor'; no other segmentation method "
                    f"is currently supported."
                )
            self.seg_args = {
                "err_type": err_type,
                "method": method,
                "config_name": config_name,
                "params_name": params_name,
            }
        else:
            galfind_logger.warning(
                f"Segmentation already performed for "
                f"{self.filt_name}, skipping!"
            )

    def perform_forced_phot(
        self: Self,
        forced_phot_band: Type[Band_Data_Base],
        err_type: str = "rms_err",
        method: str = "sextractor",
        config_name: str = "default.sex",
        params_name: str = "default.param",
        overwrite: bool = False,
    ) -> NoReturn:
        """Perform forced photometry on this band using a given detection
        band, if not already done.

        Parameters
        ----------
        forced_phot_band : `Band_Data_Base`
            Band (or stack) whose segmentation map/detections are used to
            force photometry on this band's image.
        err_type : `str`, optional
            The type of error map to use. Default is ``"rms_err"``.
        method : `str`, optional
            The forced photometry method to use; must contain
            ``"sextractor"``. Default is ``"sextractor"``.
        config_name : `str`, optional
            Name of the SExtractor configuration file to use. Default is
            ``"default.sex"``.
        params_name : `str`, optional
            Name of the SExtractor output parameters file to use. Default
            is ``"default.param"``.
        overwrite : `bool`, optional
            Whether to overwrite existing forced photometry results.
            Default is `False`.

        Raises
        ------
        InvalidOptionError
            If `method` does not contain ``"sextractor"``.
        """
        # do not re-perform forced photometry if already done
        if not (
            hasattr(self, "forced_phot_args")
            and hasattr(self, "forced_phot_path")
        ):
            if "sextractor" in method.lower():
                self.forced_phot_path, self.forced_phot_args = (
                    SExtractor.perform_forced_phot(
                        self,
                        forced_phot_band,
                        err_type,
                        config_name=config_name,
                        params_name=params_name,
                        overwrite=overwrite,
                    )
                )
            else:
                raise InvalidOptionError(
                    f"forced photometry method={method.lower()!r} does "
                    f"not contain 'sextractor'; no other forced "
                    f"photometry method is currently supported."
                )

    def _get_master_tab(self, output_ids_locs: bool = False) -> Table:
        tab = Table.read(
            self.forced_phot_path, character_as_bytes=False, format="fits"
        )
        if "sextractor" in self.forced_phot_args["method"].lower():
            id_loc_params = [
                "NUMBER",
                "X_IMAGE",
                "Y_IMAGE",
                "ALPHA_J2000",
                "DELTA_J2000",
            ]
        else:
            raise InvalidOptionError(
                f"forced_phot_args['method']="
                f"{self.forced_phot_args['method'].lower()!r} does not "
                f"contain 'sextractor'; no other forced photometry method "
                f"is currently supported."
            )
        if output_ids_locs:
            if "sextractor" in self.forced_phot_args["method"].lower():
                {
                    "ID": tab["NUMBER"],
                    "X_IMAGE": tab["X_IMAGE"].value,
                    "Y_IMAGE": tab["Y_IMAGE"].value,
                    "RA": tab["ALPHA_J2000"].value,
                    "DEC": tab["DELTA_J2000"].value,
                }
        else:
            pass

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
        self: Self,
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
        angle: Optional[Union[float, List[float], Dict[str, float]]] = None,
        edge_value: Union[float, List[float], Dict[str, float]] = 0.0,
        edge_threshold: Optional[
            Union[float, List[Optional[float]], Dict[str, Optional[float]]]
        ] = None,
        element: Union[str, List[str], Dict[str, str]] = "ELLIPSE",
        gaia_row_lim: Union[int, List[int], Dict[str, int]] = 500,
        overwrite: Union[bool, List[bool], Dict[str, bool]] = False,
    ) -> None:
        """Create or load a mask for this band's imaging, if not already done.

        Either loads a pre-existing FITS mask (`fits_mask_path`), or
        generates one using either manual region-based masking or automatic
        star/edge masking, storing the result path and arguments in
        `mask_path` and `mask_args`.

        Parameters
        ----------
        method : `str`, optional
            Masking method to use, one of ``"auto"`` or ``"manual"``.
            Default is ``"auto"``.
        fits_mask_path : `str`, optional
            Path to a pre-existing FITS mask to use directly, skipping mask
            generation. Default is `None`.
        star_mask_params : `dict`, optional
            Parameters controlling the size/shape of masked regions around
            bright stars (central cores and diffraction spikes). Default is
            ``{"central": {"a": 300.0, "b": 4.25},
            "spikes": {"a": 400.0, "b": 4.5}}``.
        edge_mask_distance : `int` or `float`, optional
            Distance in pixels from the image edge to mask. Default is `50`.
        scale_extra : `float`, optional
            Additional fractional scaling applied to masked star regions.
            Default is `0.2`.
        exclude_gaia_galaxies : `bool`, optional
            Whether to exclude Gaia sources classified as galaxies from
            star masking. Default is `True`.
        angle : `float`, optional
            Position angle override for masked regions. Default is `None`.
        edge_value : `float`, optional
            Pixel value used to identify the image edge/blank border.
            Default is `0.0`.
        edge_threshold : `float`, optional
            Threshold used when detecting the image edge. Default is `None`.
        element : `str`, optional
            Shape of the masking element (e.g. ``"ELLIPSE"``). Default is
            ``"ELLIPSE"``.
        gaia_row_lim : `int`, optional
            Maximum number of Gaia catalogue rows to query. Default is `500`.
        overwrite : `bool`, optional
            Whether to regenerate the mask even if one already exists.
            Default is `False`.

        Raises
        ------
        InvalidOptionError
            If `method` is not one of ``"auto"`` or ``"manual"``.
        """
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
                    self,
                    overwrite=overwrite,
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
                    edge_threshold,
                    element,
                    gaia_row_lim,
                    overwrite,
                )
            else:
                raise InvalidOptionError(
                    f"Invalid masking method={method!r}; must be one of "
                    f"['auto', 'manual']."
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
        """Calculate local depths for this band,
        for each loaded aperture diameter.

        Runs the depth calculation (via `Depths.calc_band_depth`), loads the
        resulting median/mean depths into `self`, and optionally produces
        diagnostic plots, if depths have not already been calculated.

        Parameters
        ----------
        mode : `str`, optional
            Depth calculation mode. Default is ``"n_nearest"``.
        scatter_size : `float`, optional
            Size of the scatter used when placing empty apertures. Default
            is `0.1`.
        distance_to_mask : `int` or `float`, optional
            Minimum distance (pixels) from masked regions for a placed
            aperture to be valid. Default is `30`.
        region_radius_used_pix : `int` or `float`, optional
            Radius (pixels) of the local region used to estimate depth.
            Default is `300`.
        n_nearest : `int`, optional
            Number of nearest empty apertures used for the ``"n_nearest"``
            mode. Default is `200`.
        coord_type : `str`, optional
            Coordinate system used when placing apertures, e.g. ``"sky"``.
            Default is ``"sky"``.
        split_depth_min_size : `int`, optional
            Minimum image size above which the depth calculation is split
            into sub-regions. Default is `100_000`.
        split_depths_factor : `int`, optional
            Factor by which to split the image when calculating depths in
            sub-regions. Default is `5`.
        step_size : `int`, optional
            Grid step size (pixels) used to place empty apertures. Default
            is `100`.
        n_split : `str` or `int`, optional
            Number of sub-regions to split the calculation into, or
            ``"auto"`` to determine automatically. Default is ``"auto"``.
        n_retry_box : `int`, optional
            Number of retries allowed when placing apertures in a sub-box.
            Default is `1`.
        grid_offset_times : `int`, optional
            Number of grid offsets used when placing empty apertures.
            Default is `1`.
        plot : `bool`, optional
            Whether to produce diagnostic depth plots. Default is `True`.
        overwrite : `bool`, optional
            Whether to recompute depths even if already loaded. Default is
            `False`.
        master_cat_path : `str`, optional
            Path to the master photometric catalogue, used for catalogue
            overlay diagnostics. Default is `None`.

        Raises
        ------
        RangeError
            If more than one set of depth parameters is generated
            internally (should not normally occur for a single band).
        """
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
            if len(params_arr) != 1:
                raise RangeError(
                    f"Depths run for filt_name={self.filt_name!r} with "
                    f"len(params_arr)={len(params_arr)}; only one set of "
                    f"parameters should be used."
                )
            params = params_arr[0]
            # run depths
            Depths.calc_band_depth(params)
            # load depths into object
            self._load_depths_from_params(params)
            # plot depths
            if plot:
                self.plot_depth_diagnostics(
                    save=True, overwrite=False, master_cat_path=master_cat_path
                )
        else:
            galfind_logger.warning(
                f"Depths loaded for {self.filt_name}, skipping!"
            )

    def get_hf_output(self, aper_diam: u.Quantity) -> Dict[str, Any]:
        """Get the stored HDF5 depth-calculation output for a given
        aperture diameter.

        Parameters
        ----------
        aper_diam : `astropy.units.Quantity`
            Aperture diameter to retrieve the depth output for.

        Returns
        -------
        `dict`
            Dictionary of depth-calculation outputs loaded from the HDF5
            depth file for this band and aperture diameter.
        """
        return Depths.get_hf_output(self, aper_diam)

    def _sort_run_depth_params(
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
        params: Tuple[Any, ...],
    ) -> None:
        if hasattr(self, "depth_args"):
            if params[1] in self.depth_args.keys():
                galfind_logger.warning(
                    f"Depth data already loaded for {self.filt_name}, "
                    f"skipping load-in"
                )
        else:
            self.depth_path = {
                params[1]: Depths.get_grid_depth_path(
                    self, params[1], params[2]
                )
            }
            depths = Depths.get_depths_from_h5(self, params[1], params[2])
            if not all(key in depths[0].keys() for key in depths[1].keys()):
                raise MissingKeyError(
                    f"Depths keys {list(depths[1].keys())} not all in "
                    f"{list(depths[0].keys())} for filt_name="
                    f"{self.filt_name!r}, aper_diam={params[1]!r}, "
                    f"mode={params[2]!r}."
                )
            for key in depths[0].keys():
                self._update_depths(
                    params[1], depths[0][key], depths[1][key], key
                )
            self.depth_args = {params[1]: Depths.get_depth_args(params)}

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
        return self._load_depths_from_params(params)

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
        """Plot depth diagnostics of a given type for this band.

        Parameters
        ----------
        aper_diam : `astropy.units.Quantity`
            Aperture diameter to plot depths for. Must be one of
            `self.aper_diams`.
        plot_type : `str`
            Type of plot to make. One of ``"rolling_average"``,
            ``"rolling_average_diag"``, ``"labels"``, ``"hist"``,
            ``"cat_depths"``, or ``"cat_diag"``.
        fig : `matplotlib.figure.Figure`, optional
            Figure to plot onto. A new figure/axes pair is created if not
            given together with `ax`. Default is `None`.
        ax : `matplotlib.axes.Axes`, optional
            Axes to plot onto. Default is `None`.
        save : `bool`, optional
            If `True`, save the plot to disk. Default is `False`.
        show : `bool`, optional
            If `True`, display the plot. Default is `True`.
        cmap_name : `str`, optional
            Name of the matplotlib colormap to use. Default is ``"plasma"``.
        label_suffix : `str`, optional
            Suffix appended to plot labels. Default is `None`.
        title : `str`, optional
            Plot title, used for histogram-type plots. Default is `None`.

        Raises
        ------
        InvalidOptionError
            If `aper_diam` is not in `self.aper_diams`, or `plot_type` is
            not a recognised value.
        """
        if aper_diam not in self.aper_diams:
            raise InvalidOptionError(
                f"aper_diam={aper_diam!r} not in "
                f"aper_diams={self.aper_diams!r} for filt_name="
                f"{self.filt_name!r}."
            )
        valid_plot_types = [
            "rolling_average",
            "rolling_average_diag",
            "labels",
            "hist",
            "cat_depths",
            "cat_diag",
        ]
        if plot_type not in valid_plot_types:
            raise InvalidOptionError(
                f"plot_type={plot_type!r} not valid; must be one of "
                f"{valid_plot_types}."
            )
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        hf_output = Depths.get_hf_output(self, aper_diam)
        if plot_type.lower() == "rolling_average":
            Depths._plot_rolling_average(
                fig, ax, hf_output, colormaps.get_cmap(cmap_name)
            )
        elif plot_type.lower() == "rolling_average_diag":
            Depths._plot_rolling_average_diagnostic(
                fig, ax, hf_output, colormaps.get_cmap(cmap_name)
            )
        elif plot_type.lower() in ["labels", "hist"]:
            num_labels = len(np.unique(hf_output["labels_grid"]))
            labels_cmap = LinearSegmentedColormap.from_list
            (
                "custom",
                [
                    colormaps.get_cmap("Set2")(i / num_labels)
                    for i in range(num_labels)
                ],
                num_labels,
            )
            if plot_type.lower() == "labels":
                Depths._plot_labels(fig, ax, hf_output, labels_cmap)
            else:
                labels_arr, possible_labels, colours = Depths._get_labels(
                    hf_output, cmap=labels_cmap, cmap_name=cmap_name
                )
                Depths._plot_depth_hist(
                    fig,
                    ax,
                    hf_output,
                    labels_arr,
                    possible_labels,
                    colours,
                    annotate=True if show or save else False,
                    label_suffix=label_suffix,
                    title=title,
                )
        elif plot_type.lower() in ["cat_depths", "cat_diag"]:
            cmap = colormaps.get_cmap(cmap_name)
            cmap.set_bad(color="black")
            cat_x, cat_y = Depths.get_cat_xy(hf_output)
            combined_mask = Depths._combine_seg_data_and_mask(self)
            if plot_type.lower() == "cat_depths":
                Depths._plot_cat_depths(
                    fig, ax, hf_output, cmap, cat_x, cat_y, combined_mask
                )
            else:
                Depths._plot_cat_diagnostic(
                    fig, ax, hf_output, cmap, cat_x, cat_y, combined_mask
                )

        if save:
            label = (
                Depths.get_depth_dir(
                    self, aper_diam, self.depth_args[aper_diam]["mode"]
                )
                + f"/{plot_type.lower()}/{self.filt_name}.png"
            )
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
        """Produce (and optionally save/show) the depth diagnostic plot for a
        single aperture diameter.

        Parameters
        ----------
        aper_diam : `astropy.units.Quantity`
            Aperture diameter to plot the depth diagnostic for.
        save : `bool`, optional
            If `True`, save the plot to disk. Default is `False`.
        show : `bool`, optional
            If `True`, display the plot. Default is `False`.
        overwrite : `bool`, optional
            If `True`, regenerate the plot even if it already exists on
            disk. Default is `True`.
        master_cat_path : `str`, optional
            Path to the master photometric catalogue used for overlaying
            catalogue sources on the diagnostic. Default is `None`.
        """
        save_path = Depths.get_depth_plot_path(self, aper_diam)
        if not Path(save_path).is_file() or overwrite:
            Depths.plot_depth_diagnostic(
                self,
                aper_diam,
                save=save,
                show=show,
                master_cat_path=master_cat_path,
            )

    def plot_depth_diagnostics(
        self,
        save: bool = False,
        overwrite: bool = True,
        master_cat_path: Optional[str] = None,
    ) -> NoReturn:
        """Produce depth diagnostic plots for every aperture diameter loaded
        for this band.

        Parameters
        ----------
        save : `bool`, optional
            If `True`, save each plot to disk. Default is `False`.
        overwrite : `bool`, optional
            If `True`, regenerate plots even if they already exist on disk.
            Default is `True`.
        master_cat_path : `str`, optional
            Path to the master photometric catalogue used for overlaying
            catalogue sources on the diagnostics. Default is `None`.
        """
        for aper_diam in self.aper_diams:
            self.plot_depth_diagnostic(
                aper_diam,
                save=save,
                overwrite=overwrite,
                master_cat_path=master_cat_path,
            )

    def calc_area_depth(
        self: Type[Self],
        aper_diam: u.Quantity,
        mask_selector: Union[str, List[str], Type[Mask_Selector]] = None,
        mask_type: Union[str, List[str]] = "MASK",
        region_selector: Optional[
            Union[Type[Region_Selector], List[Type[Region_Selector]]]
        ] = None,
        invert_region: bool = False,
        zbin: Optional[float] = None,
    ) -> Tuple[NDArray[float], NDArray[float], u.Quantity]:
        """Calculate cumulative area as a function of depth for this band.

        Parameters
        ----------
        aper_diam : `astropy.units.Quantity`
            Aperture diameter to calculate the area-depth relation for.
        mask_selector : `str`, `list` of `str`, or `Mask_Selector`, optional
            Mask selector(s) defining which mask(s) to apply. Default is
            `None`.
        mask_type : `str` or `list` of `str`, optional
            Mask extension type(s) to use. Default is ``"MASK"``.
        region_selector : `Region_Selector` or `list` of `Region_Selector`,
        optional
            Region selector(s) restricting the calculation to a sub-region
            of the image. Default is `None`.
        invert_region : `bool`, optional
            If `True`, invert the region selection. Default is `False`.
        zbin : `float`, optional
            Redshift bin to restrict the calculation to, if relevant.
            Default is `None`.

        Returns
        -------
        `tuple`
            ``(total_depths, cum_dist, area)`` giving the depth grid, the
            cumulative distribution, and the total unmasked area. Also
            stored on `self.area_depth`.
        """
        total_depths, cum_dist, area = Depths.calc_band_data_area_depth(
            self,
            aper_diam,
            mask_selector,
            mask_type,
            region_selector,
            invert_region,
            zbin,
        )
        self.area_depth = {
            "total_depths": total_depths,
            "cum_dist": cum_dist,
            "area": area,
        }
        return total_depths, cum_dist, area

    def plot_area_depth(
        self: Type[Self],
        aper_diam: u.Quantity,
        mask_selector: Union[str, List[str], Type[Mask_Selector]] = None,
        mask_type: Union[str, List[str]] = "MASK",
        region_selector: Optional[
            Type[Region_Selector], List[Type[Region_Selector]]
        ] = None,
        invert_region: bool = False,
        zbin: Optional[float] = None,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        save: bool = False,
        show: bool = False,
        close: bool = False,
        **plot_kwargs: Dict[str, Any],
    ) -> None:
        """Plot cumulative area as a function of depth for this band.

        Parameters
        ----------
        aper_diam : `astropy.units.Quantity`
            Aperture diameter to plot the area-depth relation for.
        mask_selector : `str`, `list` of `str`, or `Mask_Selector`, optional
            Mask selector(s) defining which mask(s) to apply. Default is
            `None`.
        mask_type : `str` or `list` of `str`, optional
            Mask extension type(s) to use. Default is ``"MASK"``.
        region_selector : `Region_Selector` or `list` of `Region_Selector`,
        optional
            Region selector(s) restricting the calculation to a sub-region
            of the image. Default is `None`.
        invert_region : `bool`, optional
            If `True`, invert the region selection. Default is `False`.
        zbin : `float`, optional
            Redshift bin to restrict the calculation to. Default is `None`.
        fig : `matplotlib.figure.Figure`, optional
            Figure to plot onto. Default is `None`.
        ax : `matplotlib.axes.Axes`, optional
            Axes to plot onto. Default is `None`.
        save : `bool`, optional
            If `True`, save the plot to disk. Default is `False`.
        show : `bool`, optional
            If `True`, display the plot. Default is `False`.
        close : `bool`, optional
            If `True`, close the figure after plotting. Default is `False`.
        **plot_kwargs : `dict`
            Additional keyword arguments passed to the underlying plotting
            call.

        Returns
        -------
        None
            Delegates to `Depths.plot_band_data_area_depth`.
        """
        return Depths.plot_band_data_area_depth(
            self,
            aper_diam,
            mask_selector,
            mask_type,
            region_selector,
            invert_region,
            zbin,
            fig,
            ax,
            save,
            show,
            close,
            **plot_kwargs,
        )

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        ext: str = "SCI",
        norm: Type[Normalize] = LogNorm(vmin=0.0, vmax=10.0),
        cmap: str = "plasma",
        save: bool = False,
        show: bool = True,
    ) -> None:
        """Plot the specified image data for this band on a matplotlib Axes.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`, optional
            The matplotlib Axes to plot onto. A new figure/axes pair is
            created if not given. Default is `None`.
        ext : `str`, optional
            The type of image data to plot. Must be one of ``"SCI"``,
            ``"RMS_ERR"``, ``"WHT"``, ``"SEG"``, or ``"MASK"``. Default is
            ``"SCI"``.
        norm : `matplotlib.colors.Normalize`, optional
            The normalization to use for the image data (ignored for
            ``"MASK"``). Default is `LogNorm(vmin=0.0, vmax=10.0)`.
        cmap : `str`, optional
            Name of the matplotlib colormap to use. Default is ``"plasma"``.
        save : `bool`, optional
            If `True`, the plot will be saved to a file. Default is `False`.
        show : `bool`, optional
            If `True`, the plot will be displayed. Default is `True`.

        Raises
        ------
        InvalidOptionError
            If the provided extension `ext` is not one of the allowed
            values.
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
            raise InvalidOptionError(
                f"Invalid ext={ext!r}; must be one of "
                f"['SCI', 'RMS_ERR', 'WHT', 'SEG', 'MASK']."
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
            pass
            # plt.savefig(label)
        if show:
            plt.show()

    def _combine_seg_data_and_mask(self) -> np.ndarray:
        seg_data = self.load_seg()[0]
        mask = self.load_mask()[0]["MASK"]
        if seg_data.shape != mask.shape:
            raise LengthMismatchError(
                f"seg_path={self.seg_path!r} has seg_data.shape="
                f"{seg_data.shape!r} != mask.shape={mask.shape!r} from "
                f"mask_path={self.mask_path!r}."
            )
        combined_mask = np.logical_or(seg_data > 0, mask == 1).astype(int)
        return combined_mask

    @staticmethod
    def _pix_scale_to_str(pix_scale: u.Quantity):
        return f"{round(pix_scale.to(u.marcsec).value)}mas"

    def _make_rms_err_from_wht(
        self, overwrite: bool = False, rms_err_ext_name: str = "ERR"
    ) -> NoReturn:
        save_path = self.im_path.replace(
            self.im_path.split("/")[-1],
            f"rms_err/{self.filt.filt_name}_rms_err.fits",
        )
        if not Path(save_path).is_file() or overwrite:
            # make rms_err map from wht map
            wht, hdr = self.load_wht(output_hdr=True)
            err = 1.0 / (wht**0.5)
            primary_hdr = deepcopy(hdr)
            primary_hdr["EXTNAME"] = "PRIMARY"
            primary = fits.PrimaryHDU(header=primary_hdr)
            hdu = fits.ImageHDU(err, header=hdr, name=rms_err_ext_name)
            hdul = fits.HDUList([primary, hdu])
            # save and overwrite object attributes
            funcs.make_dirs(save_path)
            hdul.writeto(save_path, overwrite=True)
            funcs.change_file_permissions(save_path)
            galfind_logger.info(
                f"Finished making {self.survey} {self.version} "
                f"{self.filt} rms_err map"
            )
        galfind_logger.debug(
            f"Loading galfind created rms_err for {self.filt_name}"
        )
        self.rms_err_path = save_path
        self.rms_err_ext = 1
        self.rms_err_ext_name = [rms_err_ext_name]
        self._use_galfind_err = True

    def _make_wht_from_rms_err(
        self, overwrite: bool = False, wht_ext_name: str = "WHT"
    ) -> NoReturn:
        save_path = self.im_path.replace(
            self.im_path.split("/")[-1], f"wht/{self.filt.filt_name}_wht.fits"
        )
        if not Path(save_path).is_file() or overwrite:
            err, hdr = self.load_rms_err(output_hdr=True)
            wht = 1.0 / (err**2)
            primary_hdr = deepcopy(hdr)
            primary_hdr["EXTNAME"] = "PRIMARY"
            primary = fits.PrimaryHDU(header=primary_hdr)
            hdu = fits.ImageHDU(wht, header=hdr, name=wht_ext_name)
            hdul = fits.HDUList([primary, hdu])
            # save and overwrite object attributes
            funcs.make_dirs(save_path)
            hdul.writeto(save_path, overwrite=True)
            funcs.change_file_permissions(save_path)
            galfind_logger.info(
                f"Finished making {self.survey} {self.version} "
                f"{self.filt} wht map"
            )
        galfind_logger.info(
            f"Loading galfind created wht for {self.filt_name}"
        )
        self.wht_path = save_path
        self.wht_ext = 1
        self.wht_ext_name = [wht_ext_name]
        self._use_galfind_err = True

    # can be simplified with new masks
    def calc_unmasked_area(
        self: Self,
        mask_type: str = "All",
    ) -> NoReturn:
        """Calculate the unmasked area for one or more mask extension types.

        Results are stored in `self.unmasked_area`, keyed by mask name.

        Parameters
        ----------
        mask_type : `str`, optional
            Mask extension type(s) to compute the unmasked area for.
            ``"All"`` computes the area for every extension in the mask
            file; multiple extension names may be combined with ``"+"`` to
            compute the area unmasked by all of them jointly. Default is
            ``"All"``.
        """
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
                masks = tuple(
                    [
                        self.load_mask(mask_type, invert=True)[0]
                        for mask_type in mask_types
                    ]
                )
                self._calc_area_given_mask(
                    "+".join(np.sort(mask_types)), masks
                )

    def _calc_area_given_mask(
        self,
        mask_name: str,
        mask: Optional[NDArray, Tuple[NDArray]] = None,
    ) -> NoReturn:
        if not hasattr(self, "unmasked_area"):
            self.unmasked_area = {}
        if mask_name not in self.unmasked_area.keys():
            # load mask
            if mask is None:
                mask = self.load_mask(mask_name.upper(), invert=True)[0]
            if isinstance(mask, tuple):
                mask = np.logical_and.reduce(mask)
            # ensure mask is the same shape as your imaging
            if mask.shape != self.data_shape:
                raise LengthMismatchError(
                    f"mask_name={mask_name!r} shape={mask.shape!r} != "
                    f"data_shape={self.data_shape!r}."
                )
            # calculate unmasked area
            unmasked_area = funcs.calc_unmasked_area(mask, self.pixel_scale)
            self.unmasked_area[mask_name.upper()] = unmasked_area


class Band_Data(Band_Data_Base):
    """Imaging data for a single filter of a survey/version.

    Concrete `Band_Data_Base` subclass representing one filter's science,
    RMS error, and weight imaging, along with derived products (mask,
    segmentation map, PSF, depths).

    Parameters
    ----------
    filt : `Filter`
        Filter this imaging data was taken through.
    survey : `str`
        Name of the survey/field this data belongs to.
    version : `str`
        Data reduction version string.
    im_path : `str`
        Path to the FITS file containing the science image.
    im_ext : `int`
        FITS extension index of the science image within `im_path`.
    rms_err_path : `str`, optional
        Path to the FITS file containing the RMS error map. Default is `None`.
    rms_err_ext : `int`, optional
        FITS extension index of the RMS error map. Default is `None`.
    wht_path : `str`, optional
        Path to the FITS file containing the weight map. Default is `None`.
    wht_ext : `int`, optional
        FITS extension index of the weight map. Default is `None`.
    pix_scale : `astropy.units.Quantity`, optional
        Pixel scale of the imaging. Default is `0.03 * u.arcsec`.
    im_ext_name : `str` or `list` of `str`, optional
        Expected FITS `EXTNAME`(s) for the science image extension.
        Default is `"SCI"`.
    rms_err_ext_name : `str` or `list` of `str`, optional
        Expected FITS `EXTNAME`(s) for the RMS error extension.
        Default is `"ERR"`.
    wht_ext_name : `str` or `list` of `str`, optional
        Expected FITS `EXTNAME`(s) for the weight extension.
        Default is `"WHT"`.
    use_galfind_err : `bool`, optional
        If `True`, automatically derive a missing RMS error map from the
        weight map (or vice versa). Default is `True`.
    aper_diams : `astropy.units.Quantity`, optional
        Aperture diameters to associate with this band. Default is `None`.
    psf : `PSF_Base`, optional
        PSF object associated with this band's imaging. Default is `None`.

    Attributes
    ----------
    filt : `Filter`
        Filter this imaging data was taken through.
    """

    def __init__(
        self: Self,
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
        psf: Optional[Type[PSF_Base]] = None,
    ):
        """Initialize the Band_Data instance.

        See the class docstring for detailed parameter descriptions.
        """
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
            psf,
        )

    @classmethod
    def from_band_data_arr(cls, band_data_arr: List[Type[Band_Data_Base]]):
        """Construct a `Band_Data` by stacking multiple same-filter
        band data objects.

        Parameters
        ----------
        band_data_arr : `list` of `Band_Data_Base`
            Band data objects (expected to share the same filter) to stack.

        Raises
        ------
        AbstractMethodError
            This method is not yet implemented.
        """
        raise AbstractMethodError(
            "Band_Data.from_band_data_arr() is not yet implemented."
        )
        # make sure all filters are the same
        # stack bands by multiplication

    @property
    def instr_name(self):
        """`str`: Class name of the instrument this filter belongs to."""
        return self.filt.instrument.__class__.__name__

    @property
    def filt_name(self):
        """`str`: Name of this band's filter."""
        return self.filt.filt_name

    @property
    def ZP(self) -> Dict[str, float]:
        """`float`: Photometric zero point of the image, computed from the
        instrument."""
        return float(self.filt.instrument.calc_ZP(self))

    def __add__(
        self, other: Union[Band_Data, List[Band_Data], Data, List[Data]]
    ) -> Data:
        # if other is not a list, make it one
        if not isinstance(other, list):
            other = [other]
        # if other is an array of data objects, make a list
        # of band_data objects
        if isinstance(other[0], Data):
            if not all(isinstance(_other, Data) for _other in other):
                raise GalfindTypeError(
                    f"Band_Data.__add__: all elements of other={other!r} "
                    f"must be Data instances when other[0] is a Data "
                    f"instance."
                )
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
                raise IncompatibleKwargsError(
                    "Cannot add Data/Band_Data objects with the same "
                    "filters. You may want to use Band_Data.__mul__() to "
                    "stack!"
                )
        else:
            raise IncompatibleKwargsError(
                "Cannot add Data/Band_Data objects from different surveys "
                "or versions."
            )

    # stacking/mosaicing
    def __mul__(
        self: Self,
        other: Union[Type[Band_Data_Base], List[Type[Band_Data_Base]]],
    ) -> Type[Band_Data_Base]:
        # if other is not a list, make it one
        if not isinstance(other, list):
            other = [other]
        if not all(
            isinstance(_other, tuple(Band_Data_Base.__subclasses__()))
            for _other in other
        ):
            raise GalfindTypeError(
                f"Band_Data.__mul__: all elements of other={other!r} must "
                f"be instances of a Band_Data_Base subclass."
            )
        # flatten array of other band_data objects
        band_data_arr = []
        for _other in other:
            if isinstance(_other, Band_Data):
                band_data_arr.extend([_other])
            elif isinstance(_other, Stacked_Band_Data):
                if not hasattr(other, "band_data_arr"):
                    raise MissingDataError(
                        "Band_Data.__mul__: other must have a "
                        "band_data_arr attribute when it contains a "
                        "Stacked_Band_Data."
                    )
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

    def sky_align(
        self: Self,
        align_band_data: Band_Data,
        wcs_name: str = "TWEAK",
        **kwargs: Dict[str, Any],
    ) -> NoReturn:
        """Align this band's WCS on-sky to match a reference band's detections.

        Matches source catalogues (from segmentation) of this band and
        `align_band_data`, fits a new WCS solution using `tweakwcs`, and
        updates the science/RMS-error/weight FITS headers in place (backing
        up the originals first).

        Parameters
        ----------
        align_band_data : `Band_Data`
            Reference band to align this band's astrometry to.
        wcs_name : `str`, optional
            Name to give the new WCS solution written to the FITS headers.
            Default is ``"TWEAK"``.
        **kwargs : `dict`
            Alignment parameters (``searchrad``, ``separation``,
            ``tolerance``, ``max_sep``); any not given are taken from the
            instrument's default `align_params`.

        Raises
        ------
        MissingKeyError
            If a required alignment parameter is missing from both
            `kwargs` and the instrument's `align_params`.
        """

        from drizzlepac import updatehdr
        from stwcs.wcsutil import HSTWCS
        from tweakwcs import FITSWCSCorrector, XYXYMatch, fit_wcs

        if self.filt_name == align_band_data.filt_name:
            galfind_logger.warning(
                f"Cannot align {repr(self)} to itself, skipping alignment!"
            )
            return

        req_params = ["searchrad", "separation", "tolerance", "max_sep"]
        for name in req_params:
            if name not in kwargs.keys():
                # use default from instrument
                if name not in self.filt.instrument.align_params.keys():
                    raise MissingKeyError(
                        f"{self!r}.sky_align: name={name!r} not in "
                        f"kwargs and not in "
                        f"align_params={list(self.filt.instrument.align_params.keys())!r}."
                    )
                kwargs[name] = self.filt.instrument.align_params[name]

        galfind_logger.info(
            f"Sky aligning {repr(self)} to "
            f"{repr(align_band_data)} with parameters: {kwargs}"
        )

        # copy original image to a subfolder
        copied_filenames = []
        for name, path in zip(
            ["sci", "rms_err", "wht"],
            [self.im_path, self.rms_err_path, self.wht_path],
        ):
            pre_align_filename = (
                f"{'/'.join(path.split('/')[:-1])}/pre_sky_alignment/"
                f"{path.split('/')[-1]}"
            )
            if pre_align_filename not in copied_filenames:
                copied_filenames.append(pre_align_filename)
                funcs.make_dirs(pre_align_filename)
                if not Path(pre_align_filename).is_file():
                    shutil.copy(path, pre_align_filename)
                    galfind_logger.info(
                        f"Successfully copied {name} image from {path} to "
                        f"{pre_align_filename}."
                    )
        # TODO: should assert that segmentation has already been performed,
        # rather than use segmentation from Photutils by default
        seg_cat_path = Photutils.segment(self)
        input_cat = QTable.read(seg_cat_path)

        ref_cat_path = Photutils.segment(align_band_data)
        ref_cat = QTable.read(ref_cat_path)

        sci_data, sci_hdr, sci_im = self.load_im(
            return_hdul=True, mode="update"
        )
        # TODO: Include HSTWCS wcs loading option in self.load_wcs
        sci_wcs = HSTWCS(sci_im, self.im_ext)

        # match catalogs
        match = XYXYMatch(
            searchrad=kwargs["searchrad"],
            separation=kwargs["separation"],
            tolerance=kwargs["tolerance"],
            use2dhist=False,
            xoffset=0.0,
            yoffset=0.0,
        )
        input_wcs_corrector = FITSWCSCorrector(sci_wcs)
        ridx, iidx = match(ref_cat, input_cat, tp_wcs=input_wcs_corrector)
        galfind_logger.info(
            f"Number of alignment matches = ({len(ridx)=}, {len(iidx)=}"
        )

        seps = np.zeros(len(ridx))
        for i, (ri, ii) in enumerate(zip(ridx, iidx)):
            sep = ref_cat[ri]["sky_centroid"].separation(
                input_cat[ii]["sky_centroid"]
            )
            seps[i] = (sep.to(u.arcsec) / align_band_data.pix_scale).value
        mask_sep = seps < kwargs["max_sep"]
        galfind_logger.info(
            f"Applying rejection of separations > {kwargs['max_sep']} pixels"
        )
        galfind_logger.info(
            f"Reducing selection to {len(input_cat[iidx][mask_sep])}"
        )
        galfind_logger.info(f"Mean OFFSET: {seps[mask_sep].mean()}")

        # if plot:
        #     plt.scatter(ref_cat['RA'][ridx],ref_cat['DEC'][ridx], c=seps)
        #     plt.colorbar()
        #     plt.show()

        aligned_imwcs = fit_wcs(
            ref_cat[ridx][mask_sep],
            input_cat[iidx][mask_sep],
            input_wcs_corrector,
            nclip=0,
            fitgeom="general",
        ).wcs

        rms_err_im = self.load_rms_err(
            output_hdr=False, return_hdul=True, mode="update"
        )[1]
        wht_im = self.load_wht(
            output_hdr=False, return_hdul=True, mode="update"
        )[1]
        aligned_paths = []
        for name, im, ext, path in zip(
            ["sci", "rms_err", "wht"],
            [sci_im, rms_err_im, wht_im],
            [self.im_ext, self.rms_err_ext, self.wht_ext],
            [self.im_path, self.rms_err_path, self.wht_path],
        ):
            if path not in aligned_paths:
                galfind_logger.info(
                    f"Updating {name} WCS to {wcs_name} for "
                    f"{repr(self)}, aligned to {repr(align_band_data)}"
                )
                aligned_paths.append(path)
                updatehdr.update_wcs(
                    im,
                    ext,
                    aligned_imwcs,
                    wcsname=wcs_name,
                    reusename=True,
                    verbose=True,
                )
                im.flush()
            im.close()
        galfind_logger.info(
            f"Finished aligning {repr(self)} to {repr(align_band_data)}"
        )

    def xy_align(
        self: Type[Self],
        align_band_data: Band_Data,
        n_cores: int = 1,
    ) -> NoReturn:
        """Reproject this band's imaging onto the pixel grid of a
        reference band.

        Uses `reproject.reproject_interp` to resample the science, RMS
        error, and weight images onto `align_band_data`'s WCS/pixel grid
        (backing up the originals first), and updates `self` in place if
        the pixel scale changes.

        Parameters
        ----------
        align_band_data : `Band_Data`
            Reference band whose pixel grid this band's imaging should be
            reprojected onto.
        n_cores : `int`, optional
            Number of cores to use for the reprojection. Default is `1`.

        Raises
        ------
        MissingKeyError
            If neither band's header contains any recognised zero-point
            keyword (``PHOTFLAM``, ``PHOTPLAM``, ``PHOTZP``, ``ZEROPNT``,
            ``ZP``, ``PIX_SCALE``, or ``PIXSCALE``).
        """

        from reproject import reproject_interp

        if align_band_data == self:
            galfind_logger.warning(
                f"Cannot align {repr(self)} to itself, skipping alignment!"
            )
            return
        if self.data_shape == align_band_data.data_shape:
            galfind_logger.warning(
                f"Data shapes are the same for {repr(self)} and "
                f"{repr(align_band_data)}, skipping alignment!"
            )
            return
        # TODO: Allow for XY pixel matching to convert the pixel scale of
        # self to the alignment band pixel scale!
        # if self.pix_scale != align_band_data.pix_scale:
        # breakpoint()
        # galfind_logger.warning(
        #     f"{repr(self)} {self.pix_scale=} != "
        #     f"{repr(align_band_data)} {align_band_data.pix_scale=}, "
        #     f"skipping alignment!"
        # )
        # return
        # scaling_factor = (align_band_data.pix_scale / self.pix_scale).value
        sci_hdr = self.load_im(return_hdul=False)[-1]
        ref_hdr = align_band_data.load_im(return_hdul=False)[-1]
        # add required ZP information to reference header
        possible_ZP_keys = [
            "PHOTFLAM",
            "PHOTPLAM",
            "PHOTZP",
            "ZEROPNT",
            "ZP",
            "PIX_SCALE",
            "PIXSCALE",
        ]
        ZP_info = {
            ZP_key: sci_hdr[ZP_key]
            for ZP_key in possible_ZP_keys
            if ZP_key in sci_hdr.keys()
        }
        if len(ZP_info) == 0:
            raise MissingKeyError(
                f"{self!r} header with keys={list(sci_hdr.keys())!r} does "
                f"not contain any of possible_ZP_keys={possible_ZP_keys!r}."
            )

        copied_filenames = []
        for name, path, ext, load_func in zip(
            ["sci", "rms_err", "wht"],
            [self.im_path, self.rms_err_path, self.wht_path],
            [self.im_ext, self.rms_err_ext, self.wht_ext],
            [self.load_im, self.load_rms_err, self.load_wht],
        ):
            pre_align_filename = (
                f"{'/'.join(path.split('/')[:-1])}/pre_xy_alignment/"
                f"{path.split('/')[-1]}"
            )
            if pre_align_filename not in copied_filenames:
                copied_filenames.append(pre_align_filename)
                funcs.make_dirs(pre_align_filename)
                if not Path(pre_align_filename).is_file():
                    shutil.copy(path, pre_align_filename)
                    galfind_logger.info(
                        f"Successfully copied {name} image from {path} to "
                        f"{pre_align_filename}."
                    )
                else:
                    galfind_logger.warning(
                        f"File {pre_align_filename} already exists, "
                        f"skipping copy."
                    )
            galfind_logger.info(
                f"XY pixel aligning {repr(self)} {name} to "
                f"{repr(align_band_data)}"
            )
            if self.pix_scale != align_band_data.pix_scale:
                if self.pix_scale > align_band_data.pix_scale:
                    galfind_logger.warning(
                        f"Reprojecting {repr(self)} {name} from "
                        f"{self.pix_scale=} to smaller "
                        f"{align_band_data.pix_scale=}!"
                    )
                    hdul = load_func(return_hdul=True)[-1]
                else:  # self.pix_scale < align_band_data.pix_scale:
                    galfind_logger.warning(
                        f"Reprojecting {repr(self)} {name} from "
                        f"{self.pix_scale=} to larger "
                        f"{align_band_data.pix_scale=},"
                        " this may result in loss of information"
                    )
                hdul = load_func(return_hdul=True)[-1]
            else:
                hdul = load_func(return_hdul=True, mode="update")[-1]

            hdul[ext].header = sci_hdr
            array = reproject_interp(hdul[ext], ref_hdr, parallel=n_cores)[0]
            hdul[ext].header = deepcopy(ref_hdr)
            hdul[ext].header["SIMPLE"] = "T"
            hdul[ext].header["EXTNAME"] = name.split("_")[-1].upper()
            for key, val in ZP_info.items():
                hdul[ext].header[key] = val
            hdul[ext].data = array
            hdul[ext].verify("fix+warn")
            if self.pix_scale != align_band_data.pix_scale:
                target_pix_scale_str = Band_Data_Base._pix_scale_to_str(
                    align_band_data.pix_scale
                )
                out_path = path.replace(
                    f"{Band_Data_Base._pix_scale_to_str(self.pix_scale)}/",
                    f"{target_pix_scale_str}/",
                )
                if out_path == path:
                    raise GalfindError(
                        f"Failed to change pixel scale in path={path!r} "
                        f"to align_band_data.pix_scale="
                        f"{align_band_data.pix_scale!r}."
                    )
                funcs.make_dirs(out_path)
                hdul.writeto(out_path, overwrite=True)
                funcs.change_file_permissions(out_path)
                # update pixel scale and path of self
                if name == "sci":
                    self.im_path = out_path
                elif name == "rms_err":
                    self.rms_err_path = out_path
                elif name == "wht":
                    self.wht_path = out_path
            else:
                hdul.flush()
                hdul.close()
        self.pix_scale = align_band_data.pix_scale

    def load_psf(
        self: Self,
        method: str = "default",
    ) -> None:
        """Load and set the PSF for this band using the instrument's
        PSF-making routine.

        Parameters
        ----------
        method : `str`, optional
            Method used to construct/retrieve the PSF. Default is
            ``"default"``.
        """
        self.psf = self.filt.instrument.make_psf(
            self,
            method=method,
        )


class Stacked_Band_Data(Band_Data_Base):
    """Imaging data formed by stacking multiple single-filter bands together.

    Concrete `Band_Data_Base` subclass representing an inverse-variance
    weighted stack of several `Band_Data` filters (e.g. a detection image),
    sharing the common loading/masking/segmentation/depth machinery of the
    base class. Typically constructed via `from_band_data_arr` rather than
    calling the constructor directly.

    Parameters
    ----------
    filterset : `list` of `Filter` or `Multiple_Filter`
        Filters that were combined to make this stack.
    survey : `str`
        Name of the survey/field this data belongs to.
    version : `str`
        Data reduction version string.
    im_path : `str`
        Path to the FITS file containing the stacked science image.
    im_ext : `int`
        FITS extension index of the science image within `im_path`.
    rms_err_path : `str`, optional
        Path to the FITS file containing the RMS error map. Default is `None`.
    rms_err_ext : `int`, optional
        FITS extension index of the RMS error map. Default is `None`.
    wht_path : `str`, optional
        Path to the FITS file containing the weight map. Default is `None`.
    wht_ext : `int`, optional
        FITS extension index of the weight map. Default is `None`.
    pix_scale : `astropy.units.Quantity`, optional
        Pixel scale of the imaging. Default is `0.03 * u.arcsec`.
    im_ext_name : `str` or `list` of `str`, optional
        Expected FITS `EXTNAME`(s) for the science image extension.
        Default is `"SCI"`.
    rms_err_ext_name : `str` or `list` of `str`, optional
        Expected FITS `EXTNAME`(s) for the RMS error extension.
        Default is `"ERR"`.
    wht_ext_name : `str` or `list` of `str`, optional
        Expected FITS `EXTNAME`(s) for the weight extension.
        Default is `"WHT"`.
    use_galfind_err : `bool`, optional
        If `True`, automatically derive a missing RMS error map from the
        weight map (or vice versa). Default is `True`.
    aper_diams : `astropy.units.Quantity`, optional
        Aperture diameters to associate with this stack. Default is `None`.
    psf : `PSF_Base`, optional
        PSF object associated with this stack's imaging. Default is `None`.

    Attributes
    ----------
    filterset : `list` of `Filter` or `Multiple_Filter`
        Filters combined to make this stack.
    band_data_arr : `list` of `Band_Data`
        The individual band data objects that were stacked, sorted blue to
        red (set by `from_band_data_arr`).
    """

    def __init__(
        self: Self,
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
        psf: Optional[Type[PSF_Base]] = None,
    ):
        """Initialize the Stacked_Band_Data instance.

        See the class docstring for detailed parameter descriptions.
        """
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
            psf,
        )

    @classmethod
    def from_band_data_arr(
        cls,
        band_data_arr: List[Band_Data],
        err_type: str = "rms_err",
    ) -> Stacked_Band_Data:
        """Construct a `Stacked_Band_Data` by inverse-variance
        stacking several bands.

        Stacks the science images (weighted by RMS error or weight map, as
        selected by `err_type`), instantiates the resulting
        `Stacked_Band_Data` object, and propagates aperture diameters,
        segmentation, and masking from the input bands where they are
        consistently defined across all of them.

        Parameters
        ----------
        band_data_arr : `list` of `Band_Data`
            Band data objects (must all have distinct filters) to stack.
        err_type : `str`, optional
            Error map type to use when weighting the stack, one of
            ``"rms_err"`` or ``"wht"``. Default is ``"rms_err"``.

        Returns
        -------
        `Stacked_Band_Data`
            The newly constructed stacked band data object, with
            `band_data_arr` set to the (blue-to-red sorted) input bands.

        Raises
        ------
        IncompatibleKwargsError
            If any two bands in `band_data_arr` share the same filter.
        """
        # make sure all filters are different
        if not all(
            band_data.filt_name != band_data_arr[0].filt_name
            for i, band_data in enumerate(band_data_arr)
            if i != 0
        ):
            raise IncompatibleKwargsError(
                f"Stacked_Band_Data.from_band_data_arr: band_data_arr="
                f"{[bd.filt_name for bd in band_data_arr]!r} contains "
                f"bands with duplicate filters; all bands must have "
                f"distinct filters."
            )

        # TODO: if all band_data in band_data_arr have been PSF homogenized,
        # update the stacking path names

        # stack bands
        input_data = Stacked_Band_Data._stack_band_data(
            band_data_arr,
            err_type=err_type,
        )
        # make filterset from filters
        filterset = Multiple_Filter(
            [band_data.filt for band_data in band_data_arr]
        )
        # instantiate the stacked band data object
        try:
            if not all(
                getattr(band_data, "psf") == getattr(band_data_arr[0], "psf")
                for band_data in band_data_arr
            ):
                raise IncompatibleKwargsError(
                    "Stacked_Band_Data.from_band_data_arr: all band_data "
                    "in band_data_arr must have the same PSF "
                    "homogenization status."
                )
        except Exception as e:
            galfind_logger.error(
                f"Error checking PSF homogenization status of "
                f"band_data in band_data_arr: {e}"
            )
            breakpoint()
        input_data["psf"] = getattr(band_data_arr[0], "psf")
        stacked_band_data = cls(filterset, **input_data)

        # if all band_data in band_data_arr have aper_diams included
        if all(
            hasattr(band_data, "aper_diams") for band_data in band_data_arr
        ):
            if all(
                all(
                    diam == diam_0
                    for diam, diam_0 in zip(
                        band_data.aper_diams, band_data_arr[0].aper_diams
                    )
                )
                for band_data in band_data_arr
            ):
                stacked_band_data.set_aper_diams(band_data_arr[0].aper_diams)

        # if all band_data in band_data_arr have been segmented, segment the
        # stacked band data
        if all(hasattr(band_data, "seg_args") for band_data in band_data_arr):
            stacked_band_data.segment()

        # if all band_data in band_data_arr have been masked, mask the
        # stacked band data
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

        # TODO: if all band_data in band_data_arr have performed forced
        # photometry, perform forced photometry for the stacked band data

        # save original band_data inputs in the class, sorted blue -> red
        stacked_band_data.band_data_arr = funcs.sort_band_data_arr(
            band_data_arr
        )
        return stacked_band_data

    @property
    def instr_name(self) -> str:
        """`str`: Combined instrument name of the filters in this stack."""
        return self.filterset.instrument_name

    @property
    def filt_name(self) -> str:
        """`str`: Name of this stack, formed by joining its filter names
        with ``"+"``."""
        return self._get_stacked_band_data_name(self.filterset)

    @property
    def ZP(self) -> Dict[str, float]:
        """`float`: Photometric zero point of the stacked image.

        If all constituent filters share the same zero point, that value is
        returned; otherwise the ``ZEROPNT`` header keyword of the stacked
        image is used.
        """
        if all(
            filt.instrument.calc_ZP(self)
            == self.filterset[0].instrument.calc_ZP(self)
            for filt in self.filterset
        ):
            galfind_logger.debug(
                f"All filters in {repr(self)} have the same ZP="
                + f"{self.filterset[0].instrument.calc_ZP(self):.2f}"
            )
            return float(self.filterset[0].instrument.calc_ZP(self))
        else:
            # extract ZP information from the header
            # open image
            im_hdr = self.load_im(return_hdul=False)[1]
            if "ZEROPNT" not in im_hdr.keys():
                raise MissingKeyError(
                    f"{self!r} header does not contain 'ZEROPNT' key; "
                    f"available keys={list(im_hdr.keys())!r}."
                )
            return float(im_hdr["ZEROPNT"])

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

    def __len__(self) -> int:
        return len(self.filterset)

    def __getitem__(
        self: Self,
        idx: Any,
    ) -> Band_Data:
        return self.band_data_arr[idx]

    # stacking/mosaicing
    def __mul__(
        self, other: Union[Type[Band_Data_Base], List[Type[Band_Data_Base]]]
    ) -> Type[Band_Data_Base]:
        # if other is not a list, make it one
        if not isinstance(other, list):
            other = [other]
        if not all(
            isinstance(_other, tuple(Band_Data_Base.__subclasses__()))
            for _other in other
        ):
            raise GalfindTypeError(
                f"Stacked_Band_Data.__mul__: all elements of "
                f"other={other!r} must be instances of a Band_Data_Base "
                f"subclass."
            )
        # flatten array of other band_data objects
        band_data_arr = []
        for _other in other:
            if isinstance(_other, Band_Data):
                band_data_arr.extend([_other])
            elif isinstance(_other, Stacked_Band_Data):
                if not hasattr(other, "band_data_arr"):
                    raise MissingDataError(
                        "Stacked_Band_Data.__mul__: other must have a "
                        "band_data_arr attribute when it contains a "
                        "Stacked_Band_Data."
                    )
                band_data_arr.extend(_other.band_data_arr)

        return Stacked_Band_Data.from_band_data_arr(
            [*deepcopy(self).band_data_arr, *band_data_arr]
        )

    @staticmethod
    def _get_stacked_band_data_name(
        filterset: Union[List[Filter], Multiple_Filter],
    ) -> str:
        return "+".join([filt.filt_name for filt in filterset])

    @staticmethod
    def _get_stacked_band_data_path(
        band_data_arr: List[Band_Data],
        err_type: str = "rms_err",
    ) -> str:
        if not all(
            getattr(band_data, name) == getattr(band_data_arr[0], name)
            for name in ["survey", "version", "pix_scale"]
            for band_data in band_data_arr
        ):
            raise IncompatibleKwargsError(
                "Stacked_Band_Data._get_stacked_band_data_path: all "
                "band_data in band_data_arr must share the same survey, "
                "version, and pix_scale."
            )
        survey = band_data_arr[0].survey
        version = band_data_arr[0].version
        band_data_arr = funcs.sort_band_data_arr(band_data_arr)
        instr_name = "+".join(
            np.unique([band_data.instr_name for band_data in band_data_arr])
        )
        # make stacked band data path, creating directory if it does not exist
        stacked_band_data_dir = (
            f"{config['DEFAULT']['GALFIND_WORK']}/Stacked_Images/"
            + f"{version}/{instr_name}/{survey}/{err_type.lower()}"
        )
        stacked_band_data_name = (
            f"{survey}_"
            + Stacked_Band_Data._get_stacked_band_data_name(
                [band_data.filt for band_data in band_data_arr]
            )
            + f"_{version}_stack.fits"
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
        if err_type.lower() not in ["rms_err", "wht"]:
            raise InvalidOptionError(
                f"err_type={err_type!r} not in ['rms_err', 'wht']."
            )

        # make rms_err/wht maps if they do not exist and are required
        # used_galfind_err = False
        if err_type.lower() == "rms_err":
            if any(
                band_data.rms_err_path is None for band_data in band_data_arr
            ):
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
            band_data_arr,
            err_type=err_type,
        )
        if not Path(stacked_band_data_path).is_file() or overwrite:
            # ensure all shapes are the same for the band data images
            if not all(
                band_data.data_shape == band_data_arr[0].data_shape
                for band_data in band_data_arr
            ):
                raise LengthMismatchError(
                    "All band data images in stacking bands must have "
                    "the same shape; got shapes="
                    f"{[bd.data_shape for bd in band_data_arr]!r}."
                )
            # ensure all band data images have the same pixel scale
            if not all(
                band_data.pix_scale == band_data_arr[0].pix_scale
                for band_data in band_data_arr
            ):
                raise IncompatibleKwargsError(
                    "All image pixel scales must be the same when "
                    "stacking; got pix_scales="
                    f"{[bd.pix_scale for bd in band_data_arr]!r}."
                )
            # stack band data SCI/ERR/WHT images (inverse variance weighted)
            galfind_logger.info(
                f"Stacking "
                f"{[band_data.filt_name for band_data in band_data_arr]} "
                f"for {band_data_arr[0].survey} {band_data_arr[0].version}"
            )
            # ensure all band data images have the same ZP
            if all(
                band_data.ZP == band_data_arr[0].ZP
                for band_data in band_data_arr
            ):
                galfind_logger.debug(
                    f"All ZPs are the same when stacking data: "
                    f"{band_data_arr[0].ZP}"
                )
                same_ZP = True
            else:
                galfind_logger.debug(
                    "Not all ZPs are the same when stacking data: "
                    + ", ".join(
                        [
                            f"({band_data.filt_name}: {band_data.ZP})"
                            for band_data in band_data_arr
                        ]
                    )
                )
                same_ZP = False

            for i, band_data in enumerate(band_data_arr):
                im_data, im_header, im_hdul = band_data.load_im(
                    return_hdul=True
                )
                if i == 0:
                    prime_hdu = im_hdul[0].header
                if err_type.lower() == "rms_err":
                    rms_err_data = band_data.load_rms_err()
                    wht_data = 1.0 / (rms_err_data**2)
                else:  # err_type.lower() == "wht"
                    wht_data = band_data.load_wht()
                    # rms_err_data = np.sqrt(1.0 / wht_data)
                if not same_ZP:
                    # convert image to Jy and update ZP information in header
                    scale = funcs.flux_image_to_Jy(1.0, band_data.ZP).value
                    im_data *= scale
                    for key in band_data.filt.instrument.ZP_keys:
                        if key in im_header.keys():
                            im_header[f"HIERARCH OLD_{key}"] = im_header[key]
                            im_header.remove(key)
                        if i == 0 and key in prime_hdu.keys():
                            prime_hdu[f"HIERARCH OLD_{key}"] = prime_hdu[key]
                            prime_hdu.remove(key)
                    im_header["ZEROPNT"] = (u.Jy).to(u.ABmag)
                    if i == 0:
                        prime_hdu["ZEROPNT"] = (u.Jy).to(u.ABmag)
                    wht_data /= scale**2

                # handle non-finite values in the image and weight data by
                # setting them to 0
                valid = np.isfinite(im_data) & np.isfinite(wht_data)
                im_data = np.where(valid, im_data, 0.0)
                wht_data = np.where(valid, wht_data, 0.0)

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
        angle: Optional[Union[float, List[float], Dict[str, float]]] = None,
        edge_value: Union[float, List[float], Dict[str, float]] = 0.0,
        edge_threshold: Optional[
            Union[float, List[Optional[float]], Dict[str, Optional[float]]]
        ] = None,
        element: Union[str, List[str], Dict[str, str]] = "ELLIPSE",
        gaia_row_lim: Union[int, List[int], Dict[str, int]] = 500,
        overwrite: Union[bool, List[bool], Dict[str, bool]] = False,
    ) -> Union[None, NoReturn]:
        """Create or load masks for this stack, and for its constituent
        bands if known.

        If the individual bands that made up this stack are not tracked
        (`band_data_arr` not set), masks the stacked image directly via the
        base class implementation. Otherwise, masks each constituent band
        individually and then combines the per-band masks into a single
        mask for the stack.

        Parameters
        ----------
        method : `str`, optional
            Masking method to use, one of ``"auto"`` or ``"manual"``.
            Default is ``"auto"``.
        fits_mask_path : `str`, optional
            Path to a pre-existing FITS mask to use directly. Default is
            `None`.
        star_mask_params : `dict`, optional
            Parameters controlling the size/shape of masked regions around
            bright stars. Default is
            ``{"central": {"a": 300.0, "b": 4.25},
            "spikes": {"a": 400.0, "b": 4.5}}``.
        edge_mask_distance : `int` or `float`, optional
            Distance in pixels from the image edge to mask. Default is `50`.
        scale_extra : `float`, optional
            Additional fractional scaling applied to masked star regions.
            Default is `0.2`.
        exclude_gaia_galaxies : `bool`, optional
            Whether to exclude Gaia sources classified as galaxies from
            star masking. Default is `True`.
        angle : `float`, optional
            Position angle override for masked regions. Default is `None`.
        edge_value : `float`, optional
            Pixel value used to identify the image edge/blank border.
            Default is `0.0`.
        edge_threshold : `float`, optional
            Threshold used when detecting the image edge. Default is `None`.
        element : `str`, optional
            Shape of the masking element (e.g. ``"ELLIPSE"``). Default is
            ``"ELLIPSE"``.
        gaia_row_lim : `int`, optional
            Maximum number of Gaia catalogue rows to query. Default is `500`.
        overwrite : `bool`, optional
            Whether to regenerate masks even if they already exist. Default
            is `False`.
        """
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
                edge_threshold=edge_threshold,
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
                    edge_threshold=edge_threshold,
                    element=element,
                    gaia_row_lim=gaia_row_lim,
                    overwrite=overwrite,
                )
            # combine masks from individual bands
            self.mask_path, self.mask_args = Masking.combine_masks(
                self,
                edge_value=edge_value,
                edge_mask_distance=edge_mask_distance,
                edge_threshold=edge_threshold,
                element=element,
            )


class Multiple_Band_Data_Base:
    """Lightweight container grouping several `Band_Data_Base` objects
    under a common name.

    Provides sequence-like access (iteration, indexing, length) and
    convenience properties that report a common value across all
    constituent bands where one exists, or a ``"+"``-joined combination
    otherwise.

    Parameters
    ----------
    band_data_arr : `list` of `Band_Data_Base`
        Band data objects (or subclass instances) to group together.
    name : `str`
        Name to identify this group of bands (returned by the `survey`
        property).

    Attributes
    ----------
    band_data_arr : `list` of `Band_Data_Base`
        The grouped band data objects.
    name : `str`
        Name identifying this group.
    """

    def __init__(
        self: Self,
        band_data_arr: List[Type[Band_Data_Base]],
        name: str,
    ):
        if not all(
            isinstance(band_data, funcs.all_subclasses(Band_Data_Base))
            for band_data in band_data_arr
        ):
            raise GalfindTypeError(
                "All band_data in band_data_arr must be subclasses of "
                "Band_Data_Base."
            )
        self.band_data_arr = band_data_arr
        self.name = name

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self) -> Type[Band_Data_Base]:
        if self.iter > len(self.band_data_arr) - 1:
            raise StopIteration
        else:
            band_data = self.band_data_arr[self.iter]
            self.iter += 1
            return band_data

    def __len__(self) -> int:
        return len(self.band_data_arr)

    def __getitem__(
        self: Self,
        idx: Any,
    ) -> Type[Band_Data_Base]:
        return self.band_data_arr[idx]

    def _get_property(self: Self, name: str) -> Union[str, List[str]]:
        if all(
            getattr(band_data, name) == getattr(self.band_data_arr[0], name)
            for band_data in self.band_data_arr
        ):
            return getattr(self.band_data_arr[0], name)
        else:
            return "+".join(
                np.unique(
                    [
                        getattr(band_data, name)
                        for band_data in self.band_data_arr
                    ]
                )
            )

    @property
    def survey(self) -> str:
        """`str`: Name identifying this group of bands."""
        return self.name
        # return self._get_property("survey")

    @property
    def version(self) -> str:
        """`str`: Common data reduction version across all bands, or a
        ``"+"``-joined combination."""
        return self._get_property("version")

    @property
    def pix_scale(self) -> u.Quantity:
        """`astropy.units.Quantity`: Common pixel scale across all bands,
        or a ``"+"``-joined combination."""
        return self._get_property("pix_scale")

    @property
    def filt_name(self) -> str:
        """`str`: Common filter name across all bands, or a ``"+"``-joined
        combination."""
        return self._get_property("filt_name")

    @property
    def instr_name(self) -> str:
        """`str`: Common instrument name across all bands, or a
        ``"+"``-joined combination."""
        return self._get_property("instr_name")

    @property
    def filt(self) -> Union[Filter, Multiple_Filter]:
        """`Filter` or `list` of `Filter`: Common filter across all bands,
        or a list of each band's filter if they differ."""
        if all(
            getattr(band_data, "filt")
            == getattr(self.band_data_arr[0], "filt")
            for band_data in self.band_data_arr
        ):
            return getattr(self.band_data_arr[0], "filt")
        else:
            return [
                getattr(band_data, "filt") for band_data in self.band_data_arr
            ]


class Data:
    """Top-level container for a survey/version's multi-band imaging data.

    Wraps a collection of `Band_Data` (and optionally `Stacked_Band_Data`)
    objects for a single survey/version, providing collective access to
    per-band loading, masking, segmentation, forced photometry, depth
    calculation, and photometric catalogue construction. Individual bands
    can be accessed via indexing (e.g. ``data["F444W"]`` or ``data[0]``) or
    iteration. Most single-band operations delegate to the corresponding
    `Band_Data_Base` method for each band. See `Data.pipeline` for the
    typical end-to-end construction/processing entry point.

    Parameters
    ----------
    band_data_arr : `list` of `Band_Data`
        Band data objects making up this dataset, one per filter.
    forced_phot_band : `str`, `list` of `str`, or `Band_Data_Base`, optional
        Band (or filter name(s) identifying a band or stack) to use as the
        detection/forced-photometry band. Default is `None`.

    Attributes
    ----------
    band_data_arr : `list` of `Band_Data`
        Band data objects, sorted by central wavelength.
    forced_phot_band : `Band_Data_Base`
        Detection/forced-photometry band, if loaded.
    is_native : `bool`
        Whether this object represents the native (pre-PSF-homogenized)
        version of the data.
    """

    def __init__(
        self,
        band_data_arr: List[Type[Band_Data]],
        forced_phot_band: Optional[
            Union[str, List[str], Type[Band_Data_Base]]
        ] = None,
        # xy_align_filt_name: str = "F444W",
    ):
        """Initialize the Data container with band data objects.

        See the class docstring for detailed parameter descriptions.
        """
        # save and sort band_arr by central wavelength
        self.band_data_arr = funcs.sort_band_data_arr(band_data_arr)
        # self._xy_align(xy_align_filt_name)
        # load forced photometry band
        if forced_phot_band is not None:
            self.load_forced_phot_band(forced_phot_band)
        self.is_native = False

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
        min_flux_pc_err: Union[int, float] = 10.0,
        stacked_band_data: Optional[
            Union[
                str,
                List[str],
                Type[Stacked_Band_Data],
                List[Union[str, List[str], Type[Stacked_Band_Data]]],
            ]
        ] = None,
        mask_method: str = "auto",
        psf_method: str = "default",
        psf_homog_filt: Optional[str] = "F444W",
        psf_homog_overwrite: bool = False,
        update: bool = False,
    ) -> Type[Data]:
        """Run the full galfind data-reduction pipeline for a survey/version.

        Discovers imaging on disk (via `from_survey_version_psfs`), loads
        PSFs, optionally PSF-homogenizes and stacks bands, masks, segments,
        performs forced photometry, and calculates depths and aperture/mask
        catalogue columns -- returning a fully processed `Data` object.

        Parameters
        ----------
        survey : `str`
            Name of the survey/field to process.
        version : `str`
            Data reduction version string.
        instrument_names : `list` of `str`, optional
            Names of instruments to search for imaging from. Default is
            read from the galfind config (``Other.INSTRUMENT_NAMES``).
        pix_scales : `astropy.units.Quantity` or `dict`, optional
            Pixel scale to use, either a single value or one per
            instrument. Default is
            ``{"ACS_WFC": 0.03, "WFC3_IR": 0.03, "NIRCam": 0.03,
            "MIRI": 0.09} * u.arcsec``.
        im_str : `list` of `str`, optional
            Filename substrings identifying science images. Default is
            ``["_sci", "_i2d", "_drz"]``.
        rms_err_str : `list` of `str`, optional
            Filename substrings identifying RMS error maps. Default is
            ``["_rms_err", "_rms", "_err"]``.
        wht_str : `list` of `str`, optional
            Filename substrings identifying weight maps. Default is
            ``["_wht", "_weight"]``.
        version_to_dir_dict : `dict`, optional
            Mapping from version string to data directory name, if
            different from `version`. Default is `None`.
        im_ext_name : `str` or `list` of `str`, optional
            Expected FITS `EXTNAME`(s) for science images. Default is
            ``"SCI"``.
        rms_err_ext_name : `str` or `list` of `str`, optional
            Expected FITS `EXTNAME`(s) for RMS error maps. Default is
            ``"ERR"``.
        wht_ext_name : `str` or `list` of `str`, optional
            Expected FITS `EXTNAME`(s) for weight maps. Default is
            ``"WHT"``.
        aper_diams : `astropy.units.Quantity`, optional
            Aperture diameters to use for photometry. Default is `None`.
        forced_phot_band : `str`, `list` of `str`, or `Band_Data_Base`,
        optional
            Band(s) to use for detection/forced photometry. Default is
            `None`.
        min_flux_pc_err : `int` or `float`, optional
            Minimum flux percentage error floor applied when computing
            local-depth-based flux errors. Default is `10.0`.
        stacked_band_data : `str`, `list` of `str`, `Stacked_Band_Data`,
            or `list` thereof, optional
            Band combination(s) to additionally stack. Default is `None`.
        mask_method : `str`, optional
            Masking method to use, one of ``"auto"`` or ``"manual"``.
            Default is ``"auto"``.
        psf_method : `str`, optional
            Method used to construct/retrieve PSFs. Default is
            ``"default"``.
        psf_homog_filt : `str`, optional
            Filter to PSF-homogenize all bands to. If `None`, PSF
            homogenization is skipped. Default is ``"F444W"``.
        psf_homog_overwrite : `bool`, optional
            Forwarded to `Data.psf_homogenize` as `overwrite`; if `True`,
            redo the PSF-homogenization convolution even if the output
            (`{version}_psfmatch_{psf_name}`) files already exist on
            disk, rather than silently reusing them. Default is `False`.
        update : `bool`, optional
            Whether to update existing catalogue columns rather than
            recomputing them from scratch. Default is `False`.

        Returns
        -------
        `Data`
            The fully processed `Data` object.
        """
        data = cls.from_survey_version_psfs(
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
            psfs=None,
        )
        data.load_psfs(method=psf_method)
        if psf_homog_filt is not None:
            data.psf_homogenize(psf_homog_filt, overwrite=psf_homog_overwrite)
        if stacked_band_data is not None:
            if not isinstance(stacked_band_data, (list, np.ndarray)):
                stacked_band_data = [stacked_band_data]
            for stacked_band_data_ in stacked_band_data:
                data.load_stacked_band_data(stacked_band_data_)
        data.mask(method=mask_method)
        data.segment()
        data.perform_forced_phot(update=update)
        data.append_aper_corr_cols()
        data.append_mask_cols()
        data.run_depths()
        data.append_loc_depth_cols(
            min_flux_pc_err=min_flux_pc_err,
            update=update,
        )
        return data

    @classmethod
    def from_survey_version_psfs(
        cls,
        survey: str,
        version: str,
        instrument_names: List[str] = json.loads(
            config.get("Other", "INSTRUMENT_NAMES")
        ),
        pix_scales: Union[u.Quantity, Dict[str, u.Quantity]] = {
            "ACS_SBC": 0.025 * u.arcsec,
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
        psfs: Optional[Type[PSF_Base], Dict[str, Type[PSF_Base]]] = None,
    ):
        """Discover on-disk imaging for a survey/version and build a
        `Data` object.

        Searches the galfind data directory for each requested instrument,
        matches FITS files to filters and image types (science/RMS
        error/weight) by filename substring and header extension name, and
        constructs a `Band_Data` object per discovered filter.

        Parameters
        ----------
        survey : `str`
            Name of the survey/field to search for.
        version : `str`
            Data reduction version string.
        instrument_names : `list` of `str`, optional
            Names of instruments to search for imaging from. Default is
            read from the galfind config (``Other.INSTRUMENT_NAMES``).
        pix_scales : `astropy.units.Quantity` or `dict`, optional
            Pixel scale to use, either a single value or one per
            instrument. Default is
            ``{"ACS_WFC": 0.03, "WFC3_IR": 0.03, "NIRCam": 0.03,
            "MIRI": 0.09} * u.arcsec``.
        im_str : `list` of `str`, optional
            Filename substrings identifying science images. Default is
            ``["_sci", "_i2d", "_drz"]``.
        rms_err_str : `list` of `str`, optional
            Filename substrings identifying RMS error maps. Default is
            ``["_rms_err", "_rms", "_err"]``.
        wht_str : `list` of `str`, optional
            Filename substrings identifying weight maps. Default is
            ``["_wht", "_weight"]``.
        version_to_dir_dict : `dict`, optional
            Mapping from version string to data directory name, if
            different from `version`. Default is `None`.
        im_ext_name : `str` or `list` of `str`, optional
            Expected FITS `EXTNAME`(s) for science images. Default is
            ``"SCI"``.
        rms_err_ext_name : `str` or `list` of `str`, optional
            Expected FITS `EXTNAME`(s) for RMS error maps. Default is
            ``"ERR"``.
        wht_ext_name : `str` or `list` of `str`, optional
            Expected FITS `EXTNAME`(s) for weight maps. Default is
            ``"WHT"``.
        aper_diams : `astropy.units.Quantity`, optional
            Aperture diameters to associate with each discovered band.
            Default is `None`.
        forced_phot_band : `str`, `list` of `str`, or `Band_Data_Base`,
        optional
            Band(s) to use for detection/forced photometry. Default is
            `None`.
        psfs : `PSF_Base` or `dict` of `str` to `PSF_Base`, optional
            PSF(s) to associate with the discovered bands, either a single
            PSF applied to all filters or a dict keyed by filter name.
            Default is `None`.

        Returns
        -------
        `Data`
            The constructed `Data` object for this survey/version.

        Raises
        ------
        MissingFileError
            If no imaging data is found for a requested instrument.
        ExternalToolError
            If multiple images are found for the same filter (band
            stacking/mosaicing across files is not yet implemented), or if
            the number of images/RMS-errors/weights found is inconsistent.
        """
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
                pix_scale,
                version_to_dir_dict,
            )
            galfind_logger.debug(
                f"Searching for {survey} {version} {instr_name} data "
                f"in {search_dir}"
            )
            # determine which filters have data
            fits_paths = list(glob.glob(f"{search_dir}/*.fits"))
            filt_names_paths = {
                filt: [
                    path
                    for path in fits_paths
                    if any(
                        path.split("/")[-1].find(substr) != -1
                        for substr in [
                            filt.upper(),
                            filt.lower(),
                            filt.lower().replace("f", "F"),
                            filt.upper().replace("F", "f"),
                        ]
                    )
                    and not any(
                        path.split("/")[-1].find(substr) != -1
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
                raise MissingFileError(
                    f"No data found for survey={survey!r} "
                    f"version={version!r} instr_name={instr_name!r} in "
                    f"search_dir={search_dir!r}."
                )
            else:
                bands_found = [
                    key
                    for key, val in filt_names_paths.items()
                    if len(val) != 0
                ]
                galfind_logger.debug(
                    f"Found {'+'.join(bands_found)} filters for "
                    f"{survey} {version} {instr_name}"
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
            if psfs is not None:
                from . import PSF_Base

                if isinstance(psfs, funcs.all_subclasses(PSF_Base)):
                    psfs_dict = {
                        filt_name: psfs for filt_name in im_paths.keys()
                    }
                else:
                    psfs_dict = psfs
                if not all(
                    filt_name in psfs_dict.keys()
                    for filt_name in im_paths.keys()
                ):
                    raise MissingKeyError(
                        f"psfs dictionary keys={list(psfs_dict.keys())!r} "
                        f"must include all filt_names="
                        f"{list(im_paths.keys())!r} with data."
                    )

            for filt_name in im_paths.keys():
                if len(im_paths[filt_name]) > 1:
                    # stack sci/rms_err/wht images together and move the
                    # old ones to a new directory
                    # NOTE: This can only be done when the images are
                    # in the same fits file but different extensions.
                    raise AbstractMethodError(
                        f"Multiple images found for filt_name={filt_name!r}. "
                        f"Stacking multiple images in the same band is not "
                        f"yet implemented."
                    )
                else:
                    if psfs is None:
                        psf = None
                    else:
                        psf = psfs_dict[filt_name]
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
                        psf=psf,
                    )
                band_data_arr.extend([band_data])
        return cls(
            band_data_arr,
            forced_phot_band=forced_phot_band,
        )

    @property
    def psf_matched(self: Self) -> Optional[str]:
        """`str` or `None`: Name of the PSF all bands are homogenized to,
        or `None` if not all bands share a common PSF."""
        if not all(band_data.psf is not None for band_data in self):
            return None
        else:
            psf_match_names = [band_data.psf.name for band_data in self]
            if all(name == psf_match_names[0] for name in psf_match_names):
                return psf_match_names[0]
            else:
                return None

    @staticmethod
    def _get_data_dir(
        survey: str,
        version: str,
        instrument: Union[str, Type[Instrument]],
        pix_scale: u.Quantity = 0.03 * u.arcsec,
        version_to_dir_dict: Optional[Dict[str, str]] = None,
        data_dir: str = config["DEFAULT"]["GALFIND_DATA"],
    ) -> Self:
        if isinstance(instrument, str):
            instrument_arr = [
                instr
                for instr in Instrument.__subclasses__()
                if instr.__name__ == instrument
            ]
            if len(instrument_arr) != 1:
                valid_names = [
                    instr.__name__ for instr in Instrument.__subclasses__()
                ]
                raise InvalidOptionError(
                    f"instrument={instrument!r} not found; must be one of "
                    f"{valid_names}."
                )
            instrument = instrument_arr[0]()
        if version_to_dir_dict is not None:
            version = version_to_dir_dict[version.split("_")[0]]
        # else:
        #     version_substr = version
        # if len(version.split("_")) > 1:
        #     version_substr += f"_{'_'.join(version.split('_')[1:])}"
        out_dir = os.path.abspath(
            f"{data_dir}/{instrument.facility.__class__.__name__.lower()}"
            + f"/{survey}/{instrument.__class__.__name__}/{version}/"
            + f"{Band_Data_Base._pix_scale_to_str(pix_scale)}"
        )
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

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
            is_sci = {
                path: any([str in path for str in im_str]) for path in paths
            }
            is_rms_err = {
                path: any([str in path for str in rms_err_str])
                for path in paths
            }
            is_wht = {
                path: any([str in path for str in wht_str]) for path in paths
            }
            # ensure name only appears in one of the image types
            unique_ext_names, unique_ext_counts = np.unique(
                list(im_ext_name)
                + list(rms_err_ext_name)
                + list(wht_ext_name),
                return_counts=True,
            )
            duplicate_names = [
                name
                for name, count in zip(unique_ext_names, unique_ext_counts)
                if count > 1
            ]
            if not all(count == 1 for count in unique_ext_counts):
                raise IncompatibleKwargsError(
                    f"Extension names duplicate_names={duplicate_names!r} "
                    f"appear in multiple of im_ext_name/rms_err_ext_name/"
                    f"wht_ext_name; each extension name must be unique "
                    f"across image types."
                )
            # check to see if all paths are science images
            for path in paths:
                # if all paths are science images
                if all(is_sci_ext for is_sci_ext in is_sci.values()):
                    # all extensions must be within the same image
                    single_path = True
                    im_paths[filt_name].extend([path])
                    rms_err_paths[filt_name].extend([path])
                    wht_paths[filt_name].extend([path])
                else:
                    # ensure the path only belongs to one (or none) of the
                    # image types
                    n_unique_types = (
                        [is_sci[path]] + [is_rms_err[path]] + [is_wht[path]]
                    ).count(True)
                    if n_unique_types >= 2:
                        raise ExternalToolError(
                            f"Multiple image types found for filt_name="
                            f"{filt_name!r}, path={path!r}; filename "
                            f"cannot be unambiguously classified as sci/"
                            f"rms_err/wht."
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
                        galfind_logger.warning(
                            f"{filt_name}, {path} not recognised as im, "
                            f"rms_err, or wht! "
                            "Consider updating 'im_str', 'rms_err_str', and "
                            "'wht_str'!"
                        )
                # extract sci/rms_err/wht extensions
                try:
                    hdul = fits.open(
                        path, ignore_missing_simple=True, mode="update"
                    )
                except Exception as e:
                    galfind_logger.critical(
                        f"Failed to open {path}! Error: {e}!"
                    )
                    breakpoint()
                if not single_path:
                    is_data_hdu = [
                        True
                        if (
                            isinstance(hdu.data, np.ndarray)
                            and hdu.data.ndim == 2
                        )
                        else False
                        for hdu in hdul
                    ]
                    n_data_hdul = len(
                        [is_data for is_data in is_data_hdu if is_data]
                    )
                    if n_data_hdul == 0:
                        raise ExternalToolError(
                            f"No 2D data HDU found in path={path!r}."
                        )
                    for j, (hdu, is_data) in enumerate(zip(hdul, is_data_hdu)):
                        if is_data:
                            if is_sci[path]:
                                if n_data_hdul == 1 or hdu.name in list(
                                    im_ext_name
                                ):
                                    im_exts[filt_name].extend([int(j)])
                                    if n_data_hdul == 1:
                                        if isinstance(
                                            hdu,
                                            astropy.io.fits.hdu.image.PrimaryHDU,
                                        ):
                                            galfind_logger.warning(
                                                f"Creating new non-PRIMARY "
                                                f"{list(im_ext_name)[0]=} hdu "
                                                f"for {filt_name}, {path}!"
                                            )
                                            non_primary_hdu = fits.ImageHDU(
                                                data=hdu.data,
                                                header=hdu.header,
                                                name=list(im_ext_name)[0],
                                            )
                                            hdul.append(non_primary_hdu)
                                            im_exts[filt_name].pop(-1)
                                            im_exts[filt_name].extend(
                                                [int(len(hdul) - 1)]
                                            )
                                            # remove primary HDU data
                                            hdu.data = None
                                        elif hdu.name not in list(im_ext_name):
                                            galfind_logger.warning(
                                                f"Updating {hdu.name=} to "
                                                f"{list(im_ext_name)[0]=} for "
                                                f"{filt_name}, {path}!"
                                            )
                                            hdu.name = list(im_ext_name)[0]
                                            hdu.header["EXTNAME"] = list(
                                                im_ext_name
                                            )[0]
                                    break
                            elif is_rms_err[path]:
                                if n_data_hdul == 1 or hdu.name in list(
                                    rms_err_ext_name
                                ):
                                    rms_err_exts[filt_name].extend([int(j)])
                                    if n_data_hdul == 1:
                                        if isinstance(
                                            hdu,
                                            astropy.io.fits.hdu.image.PrimaryHDU,
                                        ):
                                            n = list(rms_err_ext_name)[0]
                                            galfind_logger.warning(
                                                f"Creating new non-PRIMARY "
                                                f"{n=} hdu for "
                                                f"{filt_name}, {path}!"
                                            )
                                            non_primary_hdu = fits.ImageHDU(
                                                data=hdu.data,
                                                header=hdu.header,
                                                name=list(rms_err_ext_name)[0],
                                            )
                                            hdul.append(non_primary_hdu)
                                            # remove last element of
                                            # rms_err_exts as it was the
                                            # primary HDU
                                            rms_err_exts[filt_name].pop(-1)
                                            rms_err_exts[filt_name].extend(
                                                [int(len(hdul) - 1)]
                                            )
                                            # remove primary HDU data
                                            hdu.data = None
                                        elif hdu.name not in list(
                                            rms_err_ext_name
                                        ):
                                            n = list(rms_err_ext_name)[0]
                                            galfind_logger.warning(
                                                f"Updating {hdu.name=} to "
                                                f"{n=} for {filt_name}, "
                                                f"{path}!"
                                            )
                                            hdu.name = list(rms_err_ext_name)[
                                                0
                                            ]
                                            hdu.header["EXTNAME"] = list(
                                                rms_err_ext_name
                                            )[0]
                                    break
                            elif is_wht[path]:
                                if n_data_hdul == 1 or hdu.name in list(
                                    wht_ext_name
                                ):
                                    wht_exts[filt_name].extend([int(j)])
                                    if n_data_hdul == 1:
                                        if isinstance(
                                            hdu,
                                            astropy.io.fits.hdu.image.PrimaryHDU,
                                        ):
                                            w = list(wht_ext_name)[0]
                                            galfind_logger.warning(
                                                f"Creating new non-PRIMARY "
                                                f"{w=} hdu for "
                                                f"{filt_name}, {path}!"
                                            )
                                            non_primary_hdu = fits.ImageHDU(
                                                data=hdu.data,
                                                header=hdu.header,
                                                name=list(wht_ext_name)[0],
                                            )
                                            hdul.append(non_primary_hdu)
                                            wht_exts[filt_name].pop(-1)
                                            wht_exts[filt_name].extend(
                                                [int(len(hdul) - 1)]
                                            )
                                            # remove primary HDU data
                                            hdu.data = None
                                        elif hdu.name not in list(
                                            wht_ext_name
                                        ):
                                            w = list(wht_ext_name)[0]
                                            galfind_logger.warning(
                                                f"Updating {hdu.name=} to "
                                                f"{w=} for {filt_name}, "
                                                f"{path}!"
                                            )
                                            hdu.name = list(wht_ext_name)[0]
                                            hdu.header["EXTNAME"] = list(
                                                wht_ext_name
                                            )[0]
                                    break
                            galfind_logger.warning(
                                f"Data HDU not recognised as im, rms_err, "
                                f"or wht for {filt_name}, {path}, "
                                f"{hdu.name=}!"
                            )
                    hdul.flush()
                    hdul.close()
                else:
                    for j, hdu in enumerate(hdul):
                        if hdu.name in im_ext_name:
                            im_exts[filt_name].extend([int(j)])
                        if hdu.name in rms_err_ext_name:
                            rms_err_exts[filt_name].extend([int(j)])
                        if hdu.name in wht_ext_name:
                            wht_exts[filt_name].extend([int(j)])
            # ensure a None is inserted if either rms_err/wht
            # path/ext is missing compared to im path length
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
                raise ExternalToolError(
                    f"SCI image extension not found for filt_name="
                    f"{filt_name!r}."
                )
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
            if not (
                len(im_paths[filt_name])
                == len(im_exts[filt_name])
                == len(rms_err_paths[filt_name])
                == len(rms_err_exts[filt_name])
                == len(wht_paths[filt_name])
                == len(wht_exts[filt_name])
            ):
                raise LengthMismatchError(
                    f"For filt_name={filt_name!r}, mismatched lengths: "
                    f"im_paths={len(im_paths[filt_name])}, "
                    f"im_exts={len(im_exts[filt_name])}, "
                    f"rms_err_paths={len(rms_err_paths[filt_name])}, "
                    f"rms_err_exts={len(rms_err_exts[filt_name])}, "
                    f"wht_paths={len(wht_paths[filt_name])}, "
                    f"wht_exts={len(wht_exts[filt_name])}."
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
    def survey(self: Self) -> str:
        """`str`: Survey name shared by all bands in `self`.

        Raises
        ------
        IncompatibleKwargsError
            If bands in `self` do not all share the same survey.
        """
        if not all(band_data.survey == self[0].survey for band_data in self):
            raise IncompatibleKwargsError(
                "Multiple surveys found across bands; all bands in a "
                "Data object must share the same survey."
            )
        return self[0].survey

    @property
    def version(self: Self) -> str:
        """`str`: Reduction version shared by all bands in `self`.

        Raises
        ------
        IncompatibleKwargsError
            If bands in `self` do not all share the same version.
        """
        if not all(band_data.version == self[0].version for band_data in self):
            raise IncompatibleKwargsError(
                "Multiple versions found across bands; all bands in a "
                "Data object must share the same version."
            )
        return self[0].version

    @property
    def filterset(self):
        """`Multiple_Filter`: Filters of every `Band_Data` object in `self`."""
        return Multiple_Filter(
            band_data.filt
            for band_data in self
            if isinstance(band_data, Band_Data)
        )

    # @property
    # def ZPs(self) -> Dict[str, float]:
    #     return {band_data.filt_name: band_data.ZP for band_data in self}

    # @property
    # def pix_scales(self) -> Dict[str, u.Quantity]:
    #     return {
    #         band_data.filt_name: band_data.pix_scale
    #         for band_data in self
    #     }

    @property
    def full_name(self: Self) -> str:
        """`str`: Full survey name, combining the survey, version and
        filterset."""
        return funcs.get_full_survey_name(
            self.survey, self.version, self.filterset
        )

    @property
    def aper_diams(self: Self) -> u.Quantity:
        """`astropy.units.Quantity`: Aperture diameters common to every band
        in `self`."""
        all_aper_diams, aper_diam_counts = np.unique(
            np.concatenate([values for values in self.aper_diamss.values()]),
            return_counts=True,
        )
        return [
            aper_diam.to(u.arcsec)
            for aper_diam, counts in zip(all_aper_diams, aper_diam_counts)
            if counts == len(self.aper_diamss)
        ] * u.arcsec

    # def load_cluster_blank_mask_paths(self):
    #     # load in cluster core / blank field fits/reg masks
    #     mask_path_dict = {}
    #     for mask_type in ["cluster", "blank"]:
    #         # look for .fits masks first
    #         fits_masks = glob.glob(
    #             f"{config['DEFAULT']['GALFIND_WORK']}/Masks/"
    #             f"{self.survey}/fits_masks/*_{mask_type}*.fits"
    #         )
    #         if len(fits_masks) == 1:
    #             mask_path = fits_masks[0]
    #         elif len(fits_masks) > 1:
    #             galfind_logger.critical(
    #                 f"Multiple .fits {mask_type} masks exist for "
    #                 f"{self.survey}!"
    #             )
    #         else:
    #             # no .fits masks, now look for .reg masks
    #             reg_masks = glob.glob(
    #                 f"{config['DEFAULT']['GALFIND_WORK']}/Masks/"
    #                 f"{self.survey}/*_{mask_type}*.reg"
    #             )
    #             if len(reg_masks) == 1:
    #                 mask_path = reg_masks[0]
    #             elif len(reg_masks) > 1:
    #                 galfind_logger.critical(
    #                     f"Multiple .reg {mask_type} masks exist for "
    #                     f"{self.survey}!"
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

    def __repr__(self: Self) -> str:
        return f"{self.__class__.__name__}({self.full_name})".replace(
            "_", ", "
        )

    def __str__(self):
        """Function to print summary of Data class

        Returns:
            str: Summary containing survey name, version, and whether field
                is blank or cluster. Includes summary of Instrument class,
                including bands, instruments and facilities used. Image
                depths in relevant aperture sizes are included here if
                calculated. Masked/unmasked areas are also quoted here.
                Also includes paths/extensions to SCI/SEG/ERR/WHT/MASK in
                each band, pixel scales, zero points and fits shapes.
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
        # loop through bands printing key attributes not in common, and
        # depths if available
        for band_data in self:
            # output_str += str(band_data)
            output_str += f"{repr(band_data)}\n"
            output_str += funcs.band_sep
            # print the im, rms_err and wht paths/extensions
            for attr in ["im", "rms_err", "wht"]:
                if not (
                    f"{attr}_dir"
                    in self.common_attrs[band_data.instr_name].keys()
                    and f"{attr}_ext"
                    in self.common_attrs[band_data.instr_name].keys()
                ):
                    output_str += (
                        f"{attr.upper().replace('_', ' ')} PATH: "
                        + f"{getattr(band_data, f'{attr}_path')}["
                        + f"{getattr(band_data, f'{attr}_ext')}]\n"
                    )
                else:
                    attr_path_name = getattr(band_data, f"{attr}_path").split(
                        "/"
                    )[-1]
                    attr_ext = getattr(band_data, f"{attr}_ext")
                    output_str += (
                        f"{attr.upper().replace('_', ' ')} NAME: "
                        f"{attr_path_name}[{attr_ext}]\n"
                    )
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
                if attr not in self.common_attrs[
                    band_data.instr_name
                ].keys() and hasattr(band_data, attr):
                    band_data_attr = getattr(band_data, attr)
                    if attr == "ZP":
                        band_data_attr = np.round(band_data_attr, decimals=4)
                        name = attr.upper().replace("_", " ")
                    if "_path" in attr:
                        band_data_attr = band_data_attr.split("/")[-1]
                        name = (
                            attr.upper()
                            .replace("_path", "_name")
                            .replace("_", " ")
                        )
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
                        med_depth = np.round(
                            band_data.med_depth[aper_diam][depth_key],
                            decimals=3,
                        )
                        output_str += f"MEDIAN DEPTH: {med_depth}\n"
                        mean_depth = np.round(
                            band_data.mean_depth[aper_diam][depth_key],
                            decimals=3,
                        )
                        output_str += f"MEAN DEPTH: {mean_depth}\n"
                    output_str += (
                        f"H5 PATH: {band_data.depth_path[aper_diam]}\n"
                    )
                    if (
                        "depth_args"
                        not in self.common_attrs[band_data.instr_name].keys()
                    ):
                        output_str += (
                            f"ARGS: {band_data.depth_args[aper_diam]}\n"
                        )
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
        self: Type[Self],
        other: Union[int, slice, str, List[int], List[bool]],
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
        elif isinstance(other, np.ndarray) and other.dtype == bool:
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
        # Avoid recursion for pickling-related attributes
        if attr in {"__getstate__", "__setstate__"}:
            raise AttributeError(attr)

        # attr inserted here must be pluralised with 's' suffix
        if all(attr[:-1] in band_data.__dict__.keys() for band_data in self):
            if hasattr(self, "forced_phot_band"):
                if attr[:-1] in self.forced_phot_band.__dict__.keys():
                    self_band_data_arr = self.band_data_arr + [
                        self.forced_phot_band
                    ]
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
        self: Self,
        other: Union[
            Type[Band_Data_Base], List[Type[Band_Data_Base]], Data, List[Data]
        ],
    ) -> Data:
        # if other is not a list, make it one
        if not isinstance(other, list):
            other = [other]
        # if other is an array of data objects, make a list
        # of band_data objects
        if isinstance(other[0], Data):
            if not all(isinstance(_other, Data) for _other in other):
                raise GalfindTypeError(
                    f"Data.__add__: all elements of other={other!r} must "
                    f"be Data instances when other[0] is a Data instance."
                )
            other_band_data = []
            for _other in other:
                other_band_data.extend(_other.band_data_arr)
            other = other_band_data
        if not all(
            isinstance(_other, tuple(Band_Data_Base.__subclasses__()))
            for _other in other
        ):
            raise GalfindTypeError(
                f"Data.__add__: all elements of other={other!r} must be "
                f"instances of a Band_Data_Base subclass."
            )
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
                    [band_data.filt_name for band_data in new_band_data_arr]
                )
            ) == len(new_band_data_arr):
                return Data(new_band_data_arr)
            else:
                raise IncompatibleKwargsError(
                    "Cannot add Data objects with the same filters. You "
                    "may want to use Data.__mul__() to stack!"
                )
        else:
            raise IncompatibleKwargsError(
                "Cannot add Data objects from different surveys or versions."
            )

    def __eq__(self: Self, other: Data) -> bool:
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
            except Exception:
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
        if not all(
            name in [band.filt_name for band in self] for name in filt_names
        ):
            raise InvalidOptionError(
                f"Not all filt_names={filt_names!r} in "
                f"filterset={self.filterset.filt_names!r}."
            )
        return [i for i in range(len(self)) if self[i].filt_name in filt_names]

    def _sort_band_dependent_params(
        self,
        filt_name: str,
        params: Union[Any, List[Any], Dict[str, Any]],
    ):
        if isinstance(params, list):
            # ensure params is the same length as the bands
            if len(params) != len(self):
                raise LengthMismatchError(
                    f"_sort_band_dependent_params: len(params)="
                    f"{len(params)} != len(self)={len(self)}."
                )
            return params[self._indices_from_filt_names(filt_name)]
        elif isinstance(params, dict):
            # if filter name is the name of a Stacked_Band_Data object
            if filt_name not in params.keys():
                galfind_logger.debug(f"{filt_name} not in {params.keys()}!")
                split_filt_names = filt_name.split("+")
                # ensure all filters are included in the object
                if not all(name in params.keys() for name in split_filt_names):
                    raise MissingKeyError(
                        f"Not all split_filt_names={split_filt_names!r} "
                        f"in params.keys()={list(params.keys())!r} for "
                        f"filt_name={filt_name!r}."
                    )
                # ensure all parameters are the same
                if not all(
                    params[name] == params[split_filt_names[0]]
                    for i, name in enumerate(split_filt_names)
                ):
                    raise IncompatibleKwargsError(
                        f"Not all params={params!r} are the same for "
                        f"filt_name={filt_name!r}."
                    )
                return params[split_filt_names[0]]
            else:
                if filt_name not in params.keys():
                    raise MissingKeyError(
                        f"filt_name={filt_name!r} not in "
                        f"params.keys()={list(params.keys())!r}."
                    )
                return params[filt_name]
        else:
            return params

    def _get_common_attrs(self) -> NoReturn:
        common_attrs = {}
        # split by instrument
        for instr_name in self.filterset.instrument_name.split("+"):
            instr_band_data_arr = [
                band_data
                for band_data in self
                if band_data.filt.instrument.__class__.__name__ in instr_name
            ]
            common_attrs[instr_name] = {}
            # determine instrument dependent common path directories
            for attr in [
                "im_path",
                "rms_err_path",
                "wht_path",
                "mask_path",
                "seg_path",
                "forced_phot_path",
            ]:
                # NOTE: Could also do for "seg_path", "mask_path",
                "forced_phot_path"
                if all(
                    hasattr(band_data, attr)
                    for band_data in instr_band_data_arr
                ):
                    if all(
                        "/".join(getattr(band_data, attr).split("/")[:-1])
                        == "/".join(
                            getattr(instr_band_data_arr[0], attr).split("/")[
                                :-1
                            ]
                        )
                        for band_data in instr_band_data_arr
                    ):
                        common_attrs[instr_name][
                            f"{'_'.join(attr.split('_')[:-1])}_dir"
                        ] = "/".join(
                            getattr(instr_band_data_arr[0], attr).split("/")[
                                :-1
                            ]
                        )
            # NOTE: Could also determine instrument dependent common depth
            # directories here - aperture diameter dependent

            # determine instrument dependent common aatributes
            for attr in [
                "im_ext",
                "rms_err_ext",
                "wht_ext",
                "ZP",
                "pix_scale",
                "data_shape",
            ]:
                if all(
                    getattr(band_data, attr)
                    == getattr(instr_band_data_arr[0], attr)
                    for band_data in instr_band_data_arr
                ):
                    band_data_attr = getattr(instr_band_data_arr[0], attr)
                    if attr == "ZP":
                        band_data_attr = np.round(band_data_attr, decimals=4)
                    common_attrs[instr_name][attr] = band_data_attr
            # determine instrument dependent mask, seg, forced phot, and
            # depth arguments
            for attr in [
                "mask_args",
                "seg_args",
                "forced_phot_args",
                "depth_args",
            ]:
                if all(
                    hasattr(band_data, attr)
                    for band_data in instr_band_data_arr
                ):
                    if all(
                        getattr(band_data, attr)
                        == getattr(instr_band_data_arr[0], attr)
                        for band_data in instr_band_data_arr
                    ):
                        common_attrs[instr_name][attr] = getattr(
                            instr_band_data_arr[0], attr
                        )
        # save common attributes in self
        self.common_attrs = common_attrs

    # %% Methods

    def load_data(
        self,
        band: Union[int, str, Filter, List[Filter], Multiple_Filter],
        incl_mask: bool = True,
    ):
        """Load the science image, segmentation map, and (optionally) mask
        for a given band.

        Delegates to `Band_Data_Base.load_data` for the selected band.

        Parameters
        ----------
        band : `int`, `str`, `Filter`, `list` of `Filter`, or `Multiple_Filter`
            Band to load data for; used to index `self`.
        incl_mask : `bool`, optional
            If `True`, also load and return the mask. Default is `True`.

        Returns
        -------
        `tuple`
            ``(im_data, im_header, seg_data, seg_header)`` if `incl_mask` is
            `False`, or ``(im_data, im_header, seg_data, seg_header, mask)``
            if `incl_mask` is `True`.
        """
        return self[band].load_data(incl_mask)

    def load_im(
        self,
        band: Union[int, str, Filter, List[Filter], Multiple_Filter],
        return_hdul: bool = False,
    ):
        """Load the science image data and header for a given band.

        Delegates to `Band_Data_Base.load_im` for the selected band.

        Parameters
        ----------
        band : `int`, `str`, `Filter`, `list` of `Filter`, or `Multiple_Filter`
            Band to load the image for; used to index `self`.
        return_hdul : `bool`, optional
            If `True`, also return the opened `astropy.io.fits.HDUList`.
            Default is `False`.

        Returns
        -------
        `tuple`
            ``(im_data, im_header)``, or ``(im_data, im_header, im_hdul)``
            if `return_hdul` is `True`.
        """
        return self[band].load_im(return_hdul)

    def load_wcs(
        self: Self,
        band: Union[int, str, Filter, List[Filter], Multiple_Filter],
    ):
        """Load (and cache) the WCS of a given band's science image.

        Delegates to `Band_Data_Base.load_wcs` for the selected band.

        Parameters
        ----------
        band : `int`, `str`, `Filter`, `list` of `Filter`, or `Multiple_Filter`
            Band to load the WCS for; used to index `self`.

        Returns
        -------
        `astropy.wcs.WCS`
            The world coordinate system of the selected band's science
            image header.
        """
        return self[band].load_wcs()

    def load_wht(
        self: Self,
        band: Union[int, str, Filter, List[Filter], Multiple_Filter],
        output_hdr: bool = False,
    ):
        """Load the weight map data (and optionally header) for a given band.

        Delegates to `Band_Data_Base.load_wht` for the selected band.

        Parameters
        ----------
        band : `int`, `str`, `Filter`, `list` of `Filter`, or `Multiple_Filter`
            Band to load the weight map for; used to index `self`.
        output_hdr : `bool`, optional
            If `True`, also return the FITS header. Default is `False`.

        Returns
        -------
        `numpy.ndarray` or `tuple`
            The weight map data, optionally accompanied by the header
            depending on `output_hdr`.
        """
        return self[band].load_wht(output_hdr)

    def load_rms_err(
        self,
        band: Union[int, str, Filter, List[Filter], Multiple_Filter],
        output_hdr: bool = False,
    ):
        """Load the RMS error map data (
            and optionally header) for a given band.

        Delegates to `Band_Data_Base.load_rms_err` for the selected band.

        Parameters
        ----------
        band : `int`, `str`, `Filter`, `list` of `Filter`, or `Multiple_Filter`
            Band to load the RMS error map for; used to index `self`.
        output_hdr : `bool`, optional
            If `True`, also return the FITS header. Default is `False`.

        Returns
        -------
        `numpy.ndarray` or `tuple`
            The RMS error map data, optionally accompanied by the header
            depending on `output_hdr`.
        """
        return self[band].load_rms_err(output_hdr)

    def load_seg(
        self: Self,
        band: Union[int, str, Filter, List[Filter], Multiple_Filter],
    ):
        """Load the segmentation map data and header for a given band.

        Delegates to `Band_Data_Base.load_seg` for the selected band.

        Parameters
        ----------
        band : `int`, `str`, `Filter`, `list` of `Filter`, or `Multiple_Filter`
            Band to load the segmentation map for; used to index `self`.

        Returns
        -------
        `tuple`
            ``(seg_data, seg_header)`` for the selected band.
        """
        return self[band].load_seg()

    def load_mask(
        self,
        band: Union[int, str, Filter, List[Filter], Multiple_Filter],
        ext: Optional[str] = None,
        invert: bool = False,
    ):
        """Load the mask data (and header) for a given band, if masking
        has been performed.

        Delegates to `Band_Data_Base.load_mask` for the selected band.

        Parameters
        ----------
        band : `int`, `str`, `Filter`, `list` of `Filter`, or `Multiple_Filter`
            Band to load the mask for; used to index `self`.
        ext : `str`, optional
            Name of a specific mask extension to load (e.g. ``"MASK"``). If
            `None`, all mask extensions are returned. Default is `None`.
        invert : `bool`, optional
            If `True` (and `ext` is given), invert the boolean mask before
            returning it. Default is `False`.

        Returns
        -------
        `tuple`
            ``(mask, hdr)`` for the selected band, as returned by
            `Band_Data_Base.load_mask`.
        """
        return self[band].load_mask(ext, invert)

    def load_aper_diams(self, aper_diams: u.Quantity) -> NoReturn:
        """Set the aperture diameters to use for every band (and the forced
        photometry/stacked bands).

        Parameters
        ----------
        aper_diams : `astropy.units.Quantity`
            Aperture diameters (angular units) to associate with each
            band in `self`.
        """
        if hasattr(self, "forced_phot_band"):
            self.forced_phot_band.load_aper_diams(aper_diams)
        if hasattr(self, "stacked_band_data_arr"):
            [
                stacked_band_data.load_aper_diams(aper_diams)
                for stacked_band_data in self.stacked_band_data_arr
            ]
        [band_data.load_aper_diams(aper_diams) for band_data in self]

    def load_psfs(
        self: Self,
        method: str = "default",
    ) -> None:
        """Load a PSF for every band in `self`.

        Parameters
        ----------
        method : `str`, optional
            Method used to obtain each band's PSF, passed to
            `Band_Data_Base.load_psf`. Default is `"default"`.
        """
        for band_data in self:
            band_data.load_psf(method)

    def _load_depths(
        self: Self,
        aper_diam: u.Quantity,
        mode: str,
        region: str = "all",
    ) -> NoReturn:
        [band_data._load_depths(aper_diam, mode, region) for band_data in self]

    def sky_align(
        self: Self,
        align_band_data: Union[str, Type[Band_Data]],
        wcs_name: str = "TWEAK",
        **kwargs: Dict[str, Any],
    ) -> NoReturn:
        """Astrometrically align every band in `self` to a reference band.

        Calls `Band_Data.sky_align` on every band other than
        `align_band_data`, aligning their WCS to it, and stores the
        reference band as `self.align_band_data`.

        Parameters
        ----------
        align_band_data : `str` or `Band_Data`
            Reference band (or its filter name) to align all other bands
            to. Must be one of the bands in `self`.
        wcs_name : `str`, optional
            Name of the WCS solution to align to/produce. Default is
            `"TWEAK"`.
        **kwargs : `dict`
            Additional keyword arguments passed to `Band_Data.sky_align`.

        Raises
        ------
        MissingDataError
            If forced photometry has already been performed on `self`.
        InvalidOptionError
            If `align_band_data` (given as a `str`) is not a filter name
            present in `self.filterset`, or if `align_band_data` (given
            as a `Band_Data` object) has a filter name not present in
            `self.filterset`.
        GalfindTypeError
            If `align_band_data` is not a `str`/`Band_Data`.
        """
        if hasattr(self, "forced_phot_band"):
            raise MissingDataError(
                f"Should not have already loaded "
                f"forced_phot_band={self.forced_phot_band!r} if trying to "
                f"sky align!"
            )
        if isinstance(align_band_data, str):
            if align_band_data not in self.filterset.filt_names:
                raise InvalidOptionError(
                    f"align_band_data={align_band_data!r} not in "
                    f"filterset={self.filterset.filt_names!r}, cannot sky "
                    f"align."
                )
            align_band_data = self[align_band_data]
        elif isinstance(
            align_band_data, tuple(Band_Data_Base.__subclasses__())
        ):
            if align_band_data.filt_name not in self.filterset.filt_names:
                raise InvalidOptionError(
                    f"align_band_data.filt_name="
                    f"{align_band_data.filt_name!r} not in "
                    f"filterset={self.filterset.filt_names!r}, cannot sky "
                    f"align."
                )
        else:
            raise GalfindTypeError(
                f"align_band_data={align_band_data!r} must be a string or "
                f"Band_Data object, got type "
                f"{type(align_band_data).__name__}."
            )

        for band_data in self:
            if band_data.filt_name != align_band_data.filt_name:
                band_data.sky_align(
                    align_band_data, wcs_name=wcs_name, **kwargs
                )
        self.align_band_data = align_band_data

    # TODO:
    # check astrometry!
    def check_astrometry(
        self: Self,
        align_band: Union[str, Type[Filter]],
    ):
        """Check the astrometric alignment of `self` against a reference band.

        Not yet implemented.

        Parameters
        ----------
        align_band : `str` or `Filter`
            Reference band (or its filter name) to check astrometry
            against.
        """
        # load segmentation map source catalogue
        pass

    def xy_align(
        self: Self,
        align_band_data: Union[str, Type[Band_Data]],
        n_cores: int = 1,
    ) -> NoReturn:
        """Pixel-align every band in `self` to a reference band.

        Calls `Band_Data_Base.xy_align` on every band other than
        `align_band_data`, aligning their pixel grid to it.

        Parameters
        ----------
        align_band_data : `str` or `Band_Data`
            Reference band (or its filter name) to align all other bands
            to. Must be one of the bands in `self`.
        n_cores : `int`, optional
            Number of cores to use for the alignment, passed to
            `Band_Data_Base.xy_align`. Default is `1`.

        Raises
        ------
        InvalidOptionError
            If `align_band_data` (given as a `str`) is not a filter name
            present in `self.filterset`, or if `align_band_data` (given
            as a `Band_Data` object) is not present in `self`.
        GalfindTypeError
            If `align_band_data` is not a `str` or `Band_Data` object.
        """
        # determine shape of every band_data in self
        if isinstance(align_band_data, str):
            if align_band_data not in self.filterset.filt_names:
                raise InvalidOptionError(
                    f"align_band_data={align_band_data!r} not in "
                    f"filterset={self.filterset.filt_names!r}, cannot xy "
                    f"align."
                )
            align_band_data = self[align_band_data]
        elif isinstance(
            align_band_data, tuple(Band_Data_Base.__subclasses__())
        ):
            if align_band_data not in self:
                raise InvalidOptionError(
                    f"align_band_data={align_band_data!r} not in "
                    f"self={self!r}, cannot xy align."
                )
        else:
            raise GalfindTypeError(
                f"align_band_data={align_band_data!r} must be a string or "
                f"of type {tuple(Band_Data_Base.__subclasses__())!r}, got "
                f"type {type(align_band_data).__name__}."
            )
        for band_data in self:
            if band_data.filt_name != align_band_data.filt_name:
                band_data.xy_align(align_band_data, n_cores=n_cores)
            else:
                galfind_logger.debug(
                    f"Cannot align {repr(band_data)} to itself"
                )

    def psf_homogenize(
        self: Self,
        psf: Union[str, PSF_Cutout],
        use_fft_conv: bool = True,
        save_native: bool = True,
        overwrite: bool = False,
        n_jobs: int = 1,
    ):
        """PSF-homogenize every band in `self` to a target PSF.

        Optionally caches the native (pre-homogenization) data as
        `self.native`, then calls `Band_Data.psf_homogenize` on every band
        (serially if `n_jobs == 1`, otherwise in parallel via
        `joblib.Parallel`), and finally reloads the forced photometry band
        and any stacked bands so they point at the new, homogenized data.

        Parameters
        ----------
        psf : `str` or `PSF_Cutout`
            Target PSF to homogenize every band's imaging to, either
            given directly or as the name of a filter already loaded
            in `self` (its loaded PSF is then used).
        use_fft_conv : `bool`, optional
            If `True`, use FFT-based convolution; otherwise use direct
            convolution. Default is `True`.
        save_native : `bool`, optional
            If `True`, deep-copy `self` (and mark it as native) to
            `self.native` before homogenizing. Default is `True`.
        overwrite : `bool`, optional
            If `True`, redo the convolution even if the output files
            already exist. Default is `False`.
        n_jobs : `int`, optional
            Number of parallel jobs to use for homogenizing the bands.
            Default is `1`.

        Raises
        ------
        MissingDataError
            If `self` is already PSF-matched (`self.psf_matched` is not
            `None`).
        GalfindTypeError
            If `n_jobs` is not an `int`.
        RangeError
            If `n_jobs` is not positive.
        InvalidOptionError
            If `psf` (given as a `str`) is not a filter name present in
            `self.filterset`.
        """
        if getattr(self, "psf_matched") is not None:
            raise MissingDataError(
                f"Data already has psf_matched={self.psf_matched!r}, "
                f"cannot PSF homogenize."
            )
        if not isinstance(n_jobs, int):
            raise GalfindTypeError(
                f"n_jobs={n_jobs!r} has type {type(n_jobs).__name__}; "
                f"must be a positive int."
            )
        if n_jobs <= 0:
            raise RangeError(f"n_jobs={n_jobs!r} must be a positive int.")
        if isinstance(psf, str):
            if psf not in self.filterset.filt_names:
                raise InvalidOptionError(
                    f"psf={psf!r} not in "
                    f"filterset={self.filterset.filt_names!r}, cannot PSF "
                    f"homogenize."
                )
            psf = self[psf].psf

        if save_native:
            self.native = deepcopy(self)
            setattr(self.native, "is_native", True)
            for band_data in self.native:
                setattr(band_data, "is_native", True)

        if n_jobs == 1:
            for band_data in self:
                band_data.psf_homogenize(
                    psf,
                    use_fft_conv=use_fft_conv,
                    overwrite=overwrite,
                )
        else:
            params = [
                (
                    band_data,
                    psf,
                    use_fft_conv,
                    overwrite,
                )
                for band_data in self
            ]
            with funcs.tqdm_joblib(
                tqdm(
                    desc=f"PSF homogenizing to {repr(psf)} with {n_jobs=}",
                    total=len(params),
                    disable=galfind_logger.getEffectiveLevel() > logging.INFO,
                )
            ):
                Parallel(n_jobs=n_jobs)(
                    delayed(Band_Data._parallel_psf_homogenize)(param)
                    for param in params
                )
        # re-do stacks from new versions

        if hasattr(self, "forced_phot_band"):
            orig_filt_names = self.forced_phot_band.filt_name
            delattr(self, "forced_phot_band")
            self.load_forced_phot_band(orig_filt_names)

        if hasattr(self, "stacked_band_data_arr"):
            orig_filt_names = [
                stacked_band_data.filt_name
                for stacked_band_data in self.stacked_band_data_arr
            ]
            delattr(self, "stacked_band_data_arr")
            self.load_stacked_band_data_arr(orig_filt_names)

    def segment(
        self: Self,
        err_type: str = "rms_err",
        method: str = "sextractor",
        config_name: str = "default.sex",
        params_name: str = "default.param",
        overwrite: bool = False,
    ) -> NoReturn:
        """
        Segments the data using the specified error type and method.

        Args:
            err_type (str): The type of error map to use for
                segmentation. Default is "rms_err".
            method (str): The method to use for segmentation. Default is
            "sextractor".

        Returns:
            NoReturn: This method does not return any value.
        """

        if hasattr(self, "forced_phot_band"):
            if (
                self.forced_phot_band.filt_name
                not in self.filterset.filt_names
            ):
                self_band_data_arr = self.band_data_arr + [
                    self.forced_phot_band
                ]
            else:
                self_band_data_arr = self.band_data_arr
        else:
            self_band_data_arr = self.band_data_arr
        if hasattr(self, "stacked_band_data_arr"):
            self_band_data_arr += self.stacked_band_data_arr

        [
            band_data.segment(
                err_type,
                method,
                config_name,
                params_name,
                overwrite,
            )
            for band_data in self_band_data_arr
        ]

        # segment native data if it exists
        if hasattr(self, "native"):
            self.native.segment(
                err_type,
                method,
                config_name,
                params_name,
                overwrite,
            )

    def perform_forced_phot(
        self,
        forced_phot_band: Optional[
            Union[str, List[str], Type[Band_Data_Base]]
        ] = None,
        err_type: Union[str, List[str], Dict[str, str]] = "rms_err",
        method: Union[str, List[str], Dict[str, str]] = "sextractor",
        config_name: str = "default.sex",
        params_name: str = "default.param",
        update: bool = True,
        overwrite: bool = False,
    ) -> None:
        """Run forced photometry (via SExtractor or similar) for every
        band and combine the results.

        Loads (or creates) the forced photometry detection band, runs
        forced photometry on every band in `self` (plus the forced
        photometry band itself, if not already one of `self`'s bands, and
        any stacked bands) using that detection band, and combines the
        resulting per-band catalogues into the master photometric
        catalogue via `self._combine_forced_phot_cats`. Does nothing if
        `self` already has a `phot_cat_path` (i.e. forced photometry has
        already been run).

        Parameters
        ----------
        forced_phot_band : `str`, `list` of `str`, or `Band_Data_Base`,
        optional
            Band(s) to use as the forced photometry detection band,
            passed to `self.load_forced_phot_band`. Default is `None`.
        err_type : `str`, `list` of `str`, or `dict`, optional
            Error map type(s) to use for forced photometry, per band.
            Default is `"rms_err"`.
        method : `str`, `list` of `str`, or `dict`, optional
            Forced photometry method(s) to use, per band. Default is
            `"sextractor"`.
        config_name : `str`, optional
            Name of the SExtractor configuration file. Default is
            `"default.sex"`.
        params_name : `str`, optional
            Name of the SExtractor output parameters file. Default is
            `"default.param"`.
        update : `bool`, optional
            Whether to update an existing master catalogue rather than
            overwriting it, passed to `self._combine_forced_phot_cats`.
            Default is `True`.
        overwrite : `bool`, optional
            Whether to overwrite existing per-band forced photometry
            output. Default is `False`.

        Returns
        -------
        `None`
            Nothing is returned; the master photometric catalogue is
            written to disk as a side effect.
        """
        if hasattr(self, "phot_cat_path"):
            galfind_logger.critical(
                "MASTER Photometric catalogue already exists!"
            )
            return

        # create a forced_phot_band object from given string
        self.load_forced_phot_band(forced_phot_band)

        if hasattr(self, "forced_phot_band"):
            if (
                self.forced_phot_band.filt_name
                not in self.filterset.filt_names
            ):
                self_ = deepcopy(self) + deepcopy(self.forced_phot_band)
                self_band_data_arr = self.band_data_arr + [
                    self.forced_phot_band
                ]
            else:
                self_ = deepcopy(self)
                self_band_data_arr = self.band_data_arr

        if hasattr(self, "stacked_band_data_arr"):
            self_band_data_arr += self.stacked_band_data_arr

        # run for every band in the Data object
        [
            band_data.perform_forced_phot(
                self.forced_phot_band,
                self_._sort_band_dependent_params(
                    band_data.filt_name, err_type
                ),
                self_._sort_band_dependent_params(band_data.filt_name, method),
                config_name,
                params_name,
                overwrite,
            )
            for band_data in self_band_data_arr
        ]

        self._combine_forced_phot_cats(
            update=update,
            overwrite=overwrite,
        )

    def _make_band_data_base(
        self: Self,
        band_data_base: Union[str, List[str], Type[Band_Data_Base]],
    ) -> Optional[Type[Band_Data_Base]]:
        if isinstance(band_data_base, tuple(Band_Data_Base.__subclasses__())):
            if not all(
                name in self.filterset.filt_names
                for name in band_data_base.filt_name.split("+")
            ):
                raise InvalidOptionError(
                    f"band_data_base.filt_name="
                    f"{band_data_base.filt_name!r} not in "
                    f"filterset={self.filterset.filt_names!r}, cannot "
                    f"load forced photometry band."
                )
        else:
            # create a forced_phot_band object from given string
            if isinstance(band_data_base, str):
                filt_names = band_data_base.split("+")
            elif isinstance(band_data_base, list):
                filt_names = band_data_base
            else:
                raise GalfindTypeError(
                    f"band_data_base={band_data_base!r} must be a string, "
                    f"list of strings, or Band_Data_Base subclass; got "
                    f"type {type(band_data_base).__name__}."
                )
            if not all(
                name in self.filterset.filt_names for name in filt_names
            ):
                raise InvalidOptionError(
                    f"Not all filt_names={filt_names!r} in "
                    f"filterset={self.filterset.filt_names!r}."
                )
            if len(filt_names) == 1:
                band_data_base = self[filt_names[0]]
            else:
                band_data_base = Stacked_Band_Data.from_band_data_arr(
                    self[filt_names]
                )
        return band_data_base

    def load_stacked_band_data_arr(
        self: Self,
        stacked_band_data_arr: Union[
            str, List[str], Multiple_Filter, List[Multiple_Filter]
        ],
    ) -> None:
        """Load or create a stacked band data from a filter combination.

        Creates a `Stacked_Band_Data` object from the specified filter(s)
        and stores it in this Data object.

        Parameters
        ----------
        stacked_band_data_arr : `str`, `list` of `str`,
            `Multiple_Filter`, or `list` of `Multiple_Filter`
            Filter name(s) or `Multiple_Filter` object(s) identifying the
            filters to stack together.

        Raises
        ------
        GalfindTypeError
            If the resulting stacked band data is not a `Stacked_Band_Data`
            instance.
        AbstractMethodError
            If a stacked band data has already been loaded.
        """
        if isinstance(stacked_band_data_arr, list) and isinstance(
            stacked_band_data_arr[0], Multiple_Filter
        ):
            stacked_band_data_arr = [
                self._make_band_data_base(stacked_band_data.filt_names)
                for stacked_band_data in stacked_band_data_arr
            ]
        else:
            stacked_band_data_arr = [
                self._make_band_data_base(stacked_band_data_arr)
            ]
        if not all(
            isinstance(stacked_band_data, Stacked_Band_Data)
            for stacked_band_data in stacked_band_data_arr
        ):
            raise GalfindTypeError(
                f"stacked_band_data_arr={stacked_band_data_arr!r} must be "
                f"a list of Stacked_Band_Data objects."
            )
        # save stacked_band_data in self
        if hasattr(self, "stacked_band_data_arr"):
            raise AbstractMethodError(
                "Currently cannot load multiple stacked_band_data objects!"
            )
            # if stacked_band_data in self.stacked_band_data_arr:
            #     pass
        else:
            self.stacked_band_data_arr = stacked_band_data_arr

    def load_forced_phot_band(
        self: Self,
        forced_phot_band: Optional[
            Union[str, List[str], Type[Band_Data_Base]]
        ],
    ) -> Optional[Type[Band_Data_Base]]:
        """Load or retrieve the forced photometry reference band.

        Parameters
        ----------
        forced_phot_band : `str`, `list` of `str`, `Band_Data_Base`, or `None`
            Filter name, list of filter names, or band data object identifying
            the detection/forced-photometry band. If `None`, returns `None`.

        Returns
        -------
        `Band_Data_Base` or `None`
            The forced photometry band, or `None` if not specified.

        Raises
        ------
        MissingDataError
            If a different forced photometry band has already been loaded.
        """
        if forced_phot_band is not None:
            forced_phot_band = self._make_band_data_base(forced_phot_band)
            # save forced phot band in self
            if hasattr(self, "forced_phot_band"):
                if forced_phot_band != self.forced_phot_band:
                    raise MissingDataError(
                        f"forced_phot_band="
                        f"{self.forced_phot_band!r} already loaded and "
                        f"differs from the requested "
                        f"forced_phot_band={forced_phot_band!r}."
                    )
            else:
                self.forced_phot_band = forced_phot_band
            return self.forced_phot_band
        else:
            return None

    def _get_phot_cat_path(self) -> str:
        # ensure aperture diamters are the same for all bands
        if not all(
            all(
                diam == diam_0
                for diam, diam_0 in zip(
                    band_data.aper_diams, self[0].aper_diams
                )
            )
            for band_data in self
        ):
            raise IncompatibleKwargsError(
                "_get_phot_cat_path: all bands in self must share the "
                "same aper_diams."
            )
        if not all(
            diam == diam_0
            for diam, diam_0 in zip(
                self.forced_phot_band.aper_diams, self[0].aper_diams
            )
        ):
            raise IncompatibleKwargsError(
                "_get_phot_cat_path: forced_phot_band.aper_diams must "
                "match self[0].aper_diams."
            )
        # ensure all bands have the same forced photometry band
        if hasattr(self.forced_phot_band, "forced_phot_args"):
            if not all(
                band_data.forced_phot_args["forced_phot_band"]
                == self.forced_phot_band
                for band_data in self
            ):
                raise IncompatibleKwargsError(
                    "_get_phot_cat_path: all bands in self must have been "
                    "forced-photometered with the same forced_phot_band."
                )
            if (
                self.forced_phot_band.forced_phot_args["forced_phot_band"]
                != self.forced_phot_band
            ):  # points to itself?
                raise IncompatibleKwargsError(
                    "_get_phot_cat_path: forced_phot_band's own "
                    "forced_phot_args['forced_phot_band'] must point to "
                    "itself."
                )
            # ensure all bands are made using the same err map
            if not all(
                band_data.forced_phot_args["err_type"]
                == self[0].forced_phot_args["err_type"]
                for band_data in self
            ):
                raise IncompatibleKwargsError(
                    "_get_phot_cat_path: all bands in self must share the "
                    "same forced_phot_args['err_type']."
                )
            if (
                self.forced_phot_band.forced_phot_args["method"]
                != self[0].forced_phot_args["method"]
            ):
                raise IncompatibleKwargsError(
                    "_get_phot_cat_path: forced_phot_band.forced_phot_args"
                    "['method'] must match self[0].forced_phot_args"
                    "['method']."
                )

        # determine photometric catalogue path
        phot_cat_path = funcs.get_phot_cat_path(
            self.survey,
            self.version,
            self.filterset.instrument_name,
            self[0].aper_diams,
            self.forced_phot_band.filt_name,
        )
        funcs.make_dirs(phot_cat_path)
        return phot_cat_path

    def _combine_forced_phot_cats(
        self: Self,
        update: bool = True,
        overwrite: bool = False,
    ) -> None:
        # readme_sep: str = "-" * 20,
        phot_cat_path = self._get_phot_cat_path()
        funcs.make_dirs(phot_cat_path)
        if not hasattr(self, "phot_cat_path"):
            self.phot_cat_path = phot_cat_path
        else:
            raise MissingDataError(
                "MASTER Photometric catalogue already exists at "
                f"phot_cat_path={self.phot_cat_path!r}."
            )

        if (
            not Path(phot_cat_path).is_file()
            or overwrite
            or (Path(phot_cat_path).is_file() and update)
        ):
            master_tab_arr = [
                self.forced_phot_band._get_master_tab(output_ids_locs=True)
            ]
            non_forced_phot_band_data_arr = deepcopy(self.band_data_arr)
            if hasattr(self, "stacked_band_data_arr"):
                non_forced_phot_band_data_arr += self.stacked_band_data_arr

            for band_data in non_forced_phot_band_data_arr:
                if band_data.filt_name != self.forced_phot_band.filt_name:
                    master_tab_arr.extend(
                        [band_data._get_master_tab(output_ids_locs=False)]
                    )
            master_tab = hstack(master_tab_arr)

            self_band_data_arr = [
                self.forced_phot_band
            ] + non_forced_phot_band_data_arr
            # update table header
            master_tab.meta = {
                **master_tab.meta,
                **{
                    "INSTR": self.filterset.instrument_name,
                    "SURVEY": self.survey,
                    "VERSION": self.version,
                    "BANDS": str(self.filterset.filt_names),
                    "APERDIAM": funcs.aper_diams_to_str(
                        self.forced_phot_band.aper_diams
                    ),
                    "ERR_TYPE": "+".join(
                        np.unique(
                            [
                                band_data.forced_phot_args["err_type"]
                                for band_data in self_band_data_arr
                            ]
                        )
                    ),
                    "METHODS": "+".join(
                        np.unique(
                            [
                                band_data.forced_phot_args["method"]
                                for band_data in self_band_data_arr
                            ]
                        )
                    ),
                },
            }
            if not Path(phot_cat_path).is_file() or overwrite:
                # save master table
                master_tab.write(
                    self.phot_cat_path, format="fits", overwrite=True
                )
            else:
                from ..catalogues import Catalogue

                Catalogue.update_fits_cat(
                    master_tab,
                    self.phot_cat_path,
                    "OBJECTS",
                )
            galfind_logger.info(
                f"Saved combined SExtractor catalogue as {self.phot_cat_path}"
            )
            self._create_phot_cat_readme()

    def _create_phot_cat_readme(self):
        pass
        # create galfind catalogue README
        # sex_aper_diams = json.loads(
        #     config.get("SExtractor", "APERTURE_DIAMS")
        # ) * u.arcsec
        # text = f"""
        #     NUMBER: Galaxy ID
        #     X/Y_IMAGE: X/Y image co-ordinates in
        #     ALPHA/DELTA_J2000: RA/Dec (J2000 co-ordinates)
        #     FLUX(ERR)_APER_'band': Aperture flux/flux errors in
        #         {str(sex_aper_diams.to(u.arcsec).value) + 'as'}
        #         diameter apertures, image units with ZPs as
        #         explained below
        #     MAG_APER_'band': Aperture magnitudes in
        #         {str(sex_aper_diams.to(u.arcsec).value) + 'as'}
        #         diameter apertures, AB mag units, defaults to 99.
        #         if flux < 0.
        #     MAGERR_APER_'band': Aperture magnitude errors in
        #         {str(sex_aper_diams.to(u.arcsec).value) + 'as'}
        #         diameter apertures, AB mag units, negative if
        #         mag == 99.
        # """
        # if 'sextractor' in [
        #     cat_type.lower() for cat_type in
        #     self.sex_cat_types.values()
        # ]:
        #     text += (
        #         f"See SExtractor documentation () for descriptions "
        #         f"of other columns. These are only available for "
        #         f"{'+'.join([filt_name for filt_name, sex_cat_type "
        #         f"in self.sex_cat_types.items() if 'sextractor' "
        #         f"in sex_cat_type.lower()])}\n"
        #     )
        # text += readme_sep + "\n"
        # self.make_sex_readme(
        #     {"Photometry": text},
        #     self.sex_cat_master_path.replace(".fits", "_README.txt")
        # )

    def mask(
        self: Self,
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
        angle: Optional[Union[float, List[float], Dict[str, float]]] = None,
        edge_value: Union[float, List[float], Dict[str, float]] = 0.0,
        edge_threshold: Optional[
            Union[float, List[Optional[float]], Dict[str, Optional[float]]]
        ] = None,
        element: Union[str, List[str], Dict[str, str]] = "ELLIPSE",
        gaia_row_lim: Union[int, List[int], Dict[str, int]] = 500,
        overwrite: Union[bool, List[bool], Dict[str, bool]] = False,
    ) -> Union[None, NoReturn]:
        """Create or load bad-pixel masks for all bands in the Data object.

        Delegates to each `Band_Data`'s `mask()` method with the specified
        parameters, creating or loading edge masks, star masks, and other
        exclusion masks as configured.

        Parameters
        ----------
        method : `str`, `list`, or `dict`, optional
            Masking method: ``"auto"`` (automatic masking) or ``"manual"``
            (user-provided masks). Default is ``"auto"``.
        fits_mask_path : `str`, `list`, or `dict`, optional
            Path(s) to pre-made FITS mask files. Default is `None`.
        star_mask_params : `dict`, optional
            Parameters for automatic star masking (central/spike regions).
        edge_mask_distance : `int`, `float`, `list`, or `dict`, optional
            Distance from image edge to mask (pixels). Default is 50.
        scale_extra : `float`, `list`, or `dict`, optional
            Extra scaling factor for mask regions. Default is 0.2.
        exclude_gaia_galaxies : `bool`, `list`, or `dict`, optional
            Whether to mask GAIA extended objects. Default is `True`.
        angle : `float`, `list`, `dict`, or `None`, optional
            Rotation angle for masks (degrees). Default is `None`.
        edge_value : `float`, `list`, or `dict`, optional
            Pixel value for edge mask. Default is 0.0.
        edge_threshold : `float`, `list`, `dict`, or `None`, optional
            Threshold for edge detection. Default is `None`.
        element : `str`, `list`, or `dict`, optional
            Morphological element shape: ``"ELLIPSE"``, ``"BOX"``, etc.
            Default is ``"ELLIPSE"``.
        gaia_row_lim : `int`, `list`, or `dict`, optional
            Maximum GAIA row index to include in mask. Default is 500.
        overwrite : `bool`, `list`, or `dict`, optional
            Whether to regenerate masks even if they exist. Default is `False`.

        Raises
        ------
        InvalidOptionError
            If `method` is not one of ``"auto"`` or ``"manual"``.
        """
        if method not in ["auto", "manual"]:
            raise InvalidOptionError(
                f"method={method!r} not recognised; must be 'auto' or "
                f"'manual'."
            )

        if hasattr(self, "forced_phot_band"):
            if (
                self.forced_phot_band.filt_name
                not in self.filterset.filt_names
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

        if hasattr(self, "stacked_band_data_arr"):
            for stacked_band_data in self.stacked_band_data_arr:
                self_ += deepcopy(stacked_band_data)
            self_band_data_arr += self.stacked_band_data_arr

        # mask each band, sorting the potentially band
        # dependent input parameters
        [
            band_data.mask(
                method,
                self_._sort_band_dependent_params(
                    band_data.filt_name, fits_mask_path
                ),
                Masking.sort_band_dependent_star_mask_params(
                    band_data.filt
                    if isinstance(band_data, Band_Data)
                    else band_data.filterset[0],
                    star_mask_params,
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
                self._sort_band_dependent_params(
                    band_data.filt_name, edge_threshold
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
        self: Self,
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
        """Calculate photometric depths for all bands.

        Computes depth (sensitivity) information for each band using a variety
        of algorithms (e.g., nearest-neighbor, grid-based) and optionally
        generates diagnostic plots.

        Parameters
        ----------
        mode : `str`, `list`, or `dict`, optional
            Depth calculation method. Default is ``"n_nearest"``.
        scatter_size : `float`, `list`, or `dict`, optional
            Size of scatter in plot generation. Default is 0.1.
        distance_to_mask : `int`, `float`, `list`, or `dict`, optional
            Minimum distance from masked pixels (pixels). Default is 30.
        region_radius_used_pix : `int`, `float`, `list`, or `dict`, optional
            Radius of depth calculation region (pixels). Default is 300.
        n_nearest : `int`, `list`, or `dict`, optional
            Number of nearest sources for nearest-neighbor mode. Default is
            200.
        coord_type : `str`, `list`, or `dict`, optional
            Coordinate type: ``"sky"`` or ``"pixel"``. Default is ``"sky"``.
        split_depth_min_size : `int`, `list`, or `dict`, optional
            Minimum size for splitting depth calculations. Default is 100,000.
        split_depths_factor : `int`, `list`, or `dict`, optional
            Scaling factor for split depth. Default is 5.
        step_size : `int`, `list`, or `dict`, optional
            Step size for grid calculations (pixels). Default is 100.
        n_jobs : `int`, optional
            Number of parallel jobs. Default is 1 (serial).
        n_split : `str`, `int`, `list`, or `dict`, optional
            Number of splits. Default is ``"auto"``.
        n_retry_box : `int`, `list`, or `dict`, optional
            Number of retry attempts. Default is 1.
        grid_offset_times : `int`, `list`, or `dict`, optional
            Number of grid offset attempts. Default is 1.
        plot : `bool`, `list`, or `dict`, optional
            Whether to generate diagnostic plots. Default is `True`.
        overwrite : `bool`, `list`, or `dict`, optional
            Whether to recalculate depths even if cached. Default is `False`.
        timed : `bool`, optional
            Whether to time the calculation. Default is `False`.
        """
        if timed:
            start = time.time()
        if hasattr(self, "phot_cat_path"):
            master_cat_path = self.phot_cat_path
        else:
            master_cat_path = None
        if hasattr(self, "forced_phot_band"):
            if (
                self.forced_phot_band.filt_name
                not in self.filterset.filt_names
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

        if hasattr(self, "stacked_band_data_arr"):
            for stacked_band_data in self.stacked_band_data_arr:
                self_ += deepcopy(stacked_band_data)
            self_band_data_arr += self.stacked_band_data_arr

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
            if n_jobs == 1:
                [Depths.calc_band_depth(param) for param in params]
            else:
                with funcs.tqdm_joblib(
                    tqdm(
                        desc="Calculating depths",
                        total=len(params),
                        disable=galfind_logger.getEffectiveLevel()
                        > logging.INFO,
                    )
                ):
                    # TODO: Fix pointer parallelization issues
                    Parallel(n_jobs=n_jobs)(
                        delayed(Depths.calc_band_depth)(param)
                        for param in params
                    )
                    # self_band_data_arr = outputs
            # save properties to individual band_data objects
            for band_data in self_band_data_arr:
                [
                    band_data._load_depths_from_params(band_params)
                    for band_params in params
                    if band_params[0] == band_data
                ]
                if plot:
                    band_data.plot_depth_diagnostics(
                        save=True,
                        overwrite=False,
                        master_cat_path=master_cat_path,
                    )
            # make depth table
            Depths.make_depth_tab(self)

            survey_info = (
                f"{self.survey} {self.version} "
                f"{self.filterset.instrument_name}"
            )
            finishing_message = f"Calculated/loaded depths for {survey_info}"
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
        """Plot depth diagnostic for a single band and aperture diameter.

        Parameters
        ----------
        band : `int`, `str`, `Filter`, `list` of `Filter`, or `Multiple_Filter`
            Band to plot (by index, filter name, or filter object).
        aper_diam : `astropy.units.Quantity`
            Aperture diameter to plot depth for.
        save : `bool`, optional
            Whether to save the plot. Default is `False`.
        show : `bool`, optional
            Whether to display the plot. Default is `False`.
        overwrite : `bool`, optional
            Whether to overwrite existing plots. Default is `True`.
        """
        try:
            master_cat_path = self._get_phot_cat_path()
        except Exception:
            master_cat_path = None
        self[band].plot_depth_diagnostic(
            aper_diam,
            save=save,
            show=show,
            overwrite=overwrite,
            master_cat_path=master_cat_path,
        )

    def plot_depth_diagnostics(
        self,
        save: bool = False,
        overwrite: bool = True,
    ) -> NoReturn:
        """Plot depth diagnostics for all bands and aperture diameters.

        Parameters
        ----------
        save : `bool`, optional
            Whether to save plots. Default is `False`.
        overwrite : `bool`, optional
            Whether to overwrite existing plots. Default is `True`.
        """
        try:
            master_cat_path = self._get_phot_cat_path()
        except Exception:
            master_cat_path = None
        for band_data in self:
            band_data.plot_depth_diagnostics(
                save=save, overwrite=overwrite, master_cat_path=master_cat_path
            )

    def calc_area_depth(
        self: Type[Self],
        aper_diam: u.Quantity,
        mask_selector: Union[str, List[str], Type[Mask_Selector]] = None,
        mask_type: Union[str, List[str]] = "MASK",
        region_selector: Optional[
            Type[Region_Selector], List[Type[Region_Selector]]
        ] = None,
        invert_region: bool = False,
        z: Optional[float] = None,
        plot: bool = True,
    ) -> Tuple[
        Dict[str, NDArray[float]], Dict[str, NDArray[float]], Dict[str, float]
    ]:
        """Calculate depth as a function of unmasked area.

        Computes how depth varies with the unmasked survey area for a given
        aperture diameter, optionally restricting to a region or redshift bin.

        Parameters
        ----------
        aper_diam : `astropy.units.Quantity`
            Aperture diameter.
        mask_selector : `str`, `list`, or `Mask_Selector`, optional
            Selector defining masked regions. Default is `None`.
        mask_type : `str` or `list`, optional
            Type of mask to apply. Default is ``"MASK"``.
        region_selector : `Region_Selector`, `list`, or `None`, optional
            Region to restrict depth calculation to. Default is `None`.
        invert_region : `bool`, optional
            Whether to invert the region selection. Default is `False`.
        z : `float` or `None`, optional
            Redshift bin identifier. Default is `None`.
        plot : `bool`, optional
            Whether to generate diagnostic plots. Default is `True`.

        Returns
        -------
        `tuple` of (3 items)
            - Depth values by band
            - Depth errors by band
            - Unmasked area by band (in square arcsec)

        Raises
        ------
        RangeError
            If `z` is given but matches zero or more than one redshift bin.
        """

        # extract zbin label
        if z is not None:
            zbins = mask_selector.extract_zbins(self)
            # select zbin which matches the redshift of the data
            zbin = [zbin_ for zbin_ in zbins if zbin_[0] <= z < zbin_[1]]
            if len(zbin) != 1:
                raise RangeError(
                    f"Found {len(zbin)} zbins matching z={z!r}; must "
                    f"match exactly 1."
                )
            zbin = zbin[0]
            zbin_label = f"{zbin[0]:.2f}<z<{zbin[1]:.2f}"
        else:
            zbin = None
            zbin_label = "All-z"
        # extract region label
        if hasattr(self, "region_selector"):
            if invert_region:
                region = self.region_selector.fail_name
            else:
                region = self.region_selector.name
        else:
            region = None
        # try:
        if hasattr(self, "area_depths") and (
            (
                region in self.area_depths.keys()
                and zbin_label in self.area_depths[region].keys()
            )
            or zbin_label in self.area_depths.keys()
        ):
            area_depths_msg = (
                f"Area depths already calculated in {repr(self)} for "
                f"{region=}, {zbin_label=}, skipping!"
            )
            galfind_logger.debug(area_depths_msg)
        else:
            if not hasattr(self, "area_depths"):
                self.area_depths = {"all": {}}
            if region is not None and region not in self.area_depths.keys():
                self.area_depths[region] = {}
            total_depths, cum_dist, area = Depths.calc_data_area_depth(
                self,
                aper_diam,
                mask_selector,
                mask_type,
                region_selector,
                invert_region,
                zbin,
            )
            if region is not None:
                if zbin_label not in self.area_depths[region].keys():
                    self.area_depths[region][zbin_label] = {
                        "total_depths": total_depths,
                        "cum_dist": cum_dist,
                        "area": area,
                    }
            else:
                if zbin_label not in self.area_depths["all"].keys():
                    self.area_depths["all"][zbin_label] = {
                        "total_depths": total_depths,
                        "cum_dist": cum_dist,
                        "area": area,
                    }
        # except Exception as e:
        #     galfind_logger.critical(
        #         f"Could not calculate area depths for {repr(self)}: {e}"
        #     )
        #     breakpoint()
        #     raise e

        if plot:
            self.plot_area_depth(
                aper_diam,
                mask_selector,
                mask_type,
                region_selector,
                invert_region,
                save=True,
                close=True,
                overwrite=False,
                z=z,
            )

    def plot_area_depth(
        self: Self,
        aper_diam: u.Quantity,
        mask_selector: Union[str, List[str], Type[Mask_Selector]] = None,
        mask_type: Union[str, List[str]] = "MASK",
        region_selector: Optional[
            Union[Type[Region_Selector], List[Type[Region_Selector]]]
        ] = None,
        invert_region: bool = False,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        cmap_name: str = "RdYlBu_r",
        overwrite: bool = False,
        save: bool = True,
        show: bool = False,
        close: bool = True,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Plot depth as a function of unmasked survey area.

        Parameters
        ----------
        aper_diam : `astropy.units.Quantity`
            Aperture diameter.
        mask_selector : `str`, `list`, or `Mask_Selector`, optional
            Selector defining masked regions. Default is `None`.
        mask_type : `str` or `list`, optional
            Type of mask. Default is ``"MASK"``.
        region_selector : `Region_Selector`, `list`, or `None`, optional
            Region to restrict to. Default is `None`.
        invert_region : `bool`, optional
            Whether to invert region. Default is `False`.
        fig : `matplotlib.figure.Figure` or `None`, optional
            Existing figure to plot in. Default is `None` (creates new).
        ax : `matplotlib.axes.Axes` or `None`, optional
            Existing axes to plot in. Default is `None` (creates new).
        cmap_name : `str`, optional
            Colormap name. Default is ``"RdYlBu_r"``.
        overwrite : `bool`, optional
            Whether to overwrite existing plots. Default is `False`.
        save : `bool`, optional
            Whether to save the plot. Default is `True`.
        show : `bool`, optional
            Whether to display the plot. Default is `False`.
        close : `bool`, optional
            Whether to close the figure after. Default is `True`.
        **kwargs
            Additional arguments passed to plotting function.
        """
        return Depths.plot_data_area_depth(
            self,
            aper_diam,
            mask_selector,
            mask_type,
            region_selector,
            invert_region,
            fig,
            ax,
            cmap_name,
            overwrite,
            save,
            show,
            close,
            **kwargs,
        )

    def plot(
        self,
        band: Union[int, str, Filter, List[Filter], Multiple_Filter],
        ax: Optional[plt.Axes] = None,
        ext: str = "SCI",
        norm: Type[Normalize] = LogNorm(vmin=0.0, vmax=10.0),
        save: bool = False,
        show: bool = True,
    ) -> NoReturn:
        """Plot image data for a specific band.

        Parameters
        ----------
        band : `int`, `str`, `Filter`, `list` of `Filter`, or `Multiple_Filter`
            Band(s) to plot (identifier, name, or object).
        ax : `matplotlib.axes.Axes`, optional
            Axes to plot on. A new one is created if `None`. Default is `None`.
        ext : `str`, optional
            FITS extension name to plot (e.g., "SCI", "ERR"). Default is "SCI".
        norm : `matplotlib.colors.Normalize`, optional
            Normalization for the image. Default is ``LogNorm(vmin=0.0,
            vmax=10.0)``.
        save : `bool`, optional
            Whether to save the figure. Default is `False`.
        show : `bool`, optional
            Whether to display the figure. Default is `True`.
        """
        self[band].plot(ax, ext, norm, save, show)

    def plot_psf_eec(
        self: Self,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        cmap: str = "cmr.guppy_r",
        save: bool = True,
        show: bool = False,
        close: bool = True,
        **kwargs: Dict[str, Any],
    ):
        """Plot encircled energy curves (EEC) for PSFs in all bands.

        Parameters
        ----------
        fig : `matplotlib.figure.Figure` or `None`, optional
            Existing figure to plot in. Default is `None` (creates new).
        ax : `matplotlib.axes.Axes` or `None`, optional
            Existing axes to plot in. Default is `None` (creates new).
        cmap : `str`, optional
            Colormap name for band-dependent colors. Default is
            ``"cmr.guppy_r"``.
        save : `bool`, optional
            Whether to save the plot. Default is `True`.
        show : `bool`, optional
            Whether to display the plot. Default is `False`.
        close : `bool`, optional
            Whether to close figure after. Default is `True`.
        **kwargs
            Additional keyword arguments passed to plotting function.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        colours = colormaps.get_cmap(cmap)(np.linspace(0.0, 1.0, len(self)))
        labels = [band_data.filt_name for band_data in self]
        for i, band_data in enumerate(self):
            plot_kwargs = deepcopy(kwargs)
            if (
                "color" not in plot_kwargs.keys()
                and "c" not in plot_kwargs.keys()
            ):
                plot_kwargs["color"] = colours[i]
            plot_kwargs["label"] = labels[i]
            band_data.psf.plot_eec(
                ax,
                annotate=True if i == len(self) - 1 else False,
                **plot_kwargs,
            )
        if save:
            title_str = (
                f"{self.survey} {self.version} "
                f"{self.filterset.instrument_name}"
            )
            ax.set_title(title_str)
            # TODO: determine whether psf is model or empirical
            # and include in title
            out_path = (
                f"{config['PSF']['PSF_PLOT_DIR']}/{self.version}/"
                f"{self.survey}/{self.filterset.instrument_name}_EEC.png"
            )
            funcs.make_dirs(out_path)
            plt.savefig(out_path, dpi=600)
            funcs.change_file_permissions(out_path)
            galfind_logger.info(f"Saved {repr(self)} EEC plot at {out_path}")
        if show:
            plt.show()
        if close:
            plt.close(fig)

    def plot_RGB(
        self,
        ax: Optional[plt.Axes] = None,
        blue_bands: List[Union[str, Filter]] = ["F090W"],
        green_bands: List[Union[str, Filter]] = ["F200W"],
        red_bands: List[Union[str, Filter]] = ["F444W"],
        method: str = "trilogy",
    ):
        """Create and display a false-color RGB image from three bands.

        Combines specified bands to create a composite RGB image using the
        specified method (e.g., "trilogy").

        Parameters
        ----------
        ax : `matplotlib.axes.Axes` or `None`, optional
            Axes to plot RGB image in. Default is `None`.
        blue_bands : `list`, optional
            Filter names for blue channel. Default is ``["F090W"]``.
        green_bands : `list`, optional
            Filter names for green channel. Default is ``["F200W"]``.
        red_bands : `list`, optional
            Filter names for red channel. Default is ``["F444W"]``.
        method : `str`, optional
            RGB creation method: ``"trilogy"`` or similar. Default is
            ``"trilogy"``.

        Raises
        ------
        InvalidOptionError
            If any of the specified bands are not available in this Data
            object.
        """
        # ensure all blue, green and red bands are contained in the data object
        if not all(
            band in self.instrument.filt_names
            for band in blue_bands + green_bands + red_bands
        ):
            raise InvalidOptionError(
                f"Cannot make galaxy RGB as not all bands="
                f"{blue_bands + green_bands + red_bands!r} are in "
                f"instrument.filt_names={self.instrument.filt_names!r}."
            )
        # construct out_path
        band_str = (
            f"B={'+'.join(blue_bands)},"
            f"G={'+'.join(green_bands)},"
            f"R={'+'.join(red_bands)}"
        )
        out_path = (
            f"{config['RGB']['RGB_DIR']}/{self.version}/{self.survey}/"
            f"{method}/{band_str}.png"
        )
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
                    outname = funcs.split_dir_name(out_path, "name").replace(
                        ".png", ""
                    )
                    f.write(f"outname  {outname}\n")
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
                )  # TRILOGY_DIR config not working here
                from trilogy3 import Trilogy

                galfind_logger.info(
                    f"Making full trilogy RGB image at {out_path}"
                )
                Trilogy(in_path, images=None).run()
            elif method == "lupton":
                raise AbstractMethodError(
                    "plot_RGB(method='lupton') is not yet implemented."
                )

    def append_loc_depth_cols(
        self: Self,
        min_flux_pc_err: Union[int, float],
        update: bool = True,
        overwrite: bool = False,
    ) -> None:
        """Append local depth columns to the photometric catalogue.

        Parameters
        ----------
        min_flux_pc_err : `int` or `float`
            Minimum flux fraction error threshold.
        update : `bool`, optional
            Whether to update the catalogue in-place. Default is `True`.
        overwrite : `bool`, optional
            Whether to overwrite existing columns. Default is `False`.
        """
        return Depths.append_loc_depth_cols(
            self,
            min_flux_pc_err=min_flux_pc_err,
            update=update,
            overwrite=overwrite,
        )

    def append_aper_corr_cols(
        self: Self,
        overwrite: bool = False,
    ) -> NoReturn:
        """Append aperture correction columns to the photometric catalogue.

        Adds corrected magnitudes for each aperture diameter using the stored
        aperture correction values.

        Parameters
        ----------
        overwrite : `bool`, optional
            Whether to overwrite existing columns. Default is `False`.

        Raises
        ------
        IncompatibleKwargsError
            If aperture diameters are not the same for all bands.
        AbstractMethodError
            If `overwrite` is `True` (deleting existing columns is not yet
            implemented).
        """
        cat = Table.read(self.phot_cat_path)
        if (
            f"MAG_APER_{self[0].filt_name}_aper_corr" not in cat.colnames
            or overwrite
        ):
            # ensure aperture diameters are the same for all bands
            all_bands = deepcopy(self.band_data_arr)
            if getattr(self, "forced_phot_band") is not None:
                if getattr(self, "forced_phot_band").filt_name not in [
                    band_data.filt_name for band_data in all_bands
                ]:
                    all_bands += [self.forced_phot_band]
            if hasattr(self, "stacked_band_data_arr"):
                all_bands += self.stacked_band_data_arr
            if not all(
                all(
                    diam == diam_0
                    for diam, diam_0 in zip(
                        band_data.aper_diams, self[0].aper_diams
                    )
                )
                for band_data in all_bands
            ):
                raise IncompatibleKwargsError(
                    "Aperture diameters are not the same for all bands."
                )
            if overwrite:
                # TODO: Delete already existing columns
                raise AbstractMethodError(
                    "append_aper_corr_cols(overwrite=True): deleting "
                    "already existing columns is not yet implemented."
                )
            aper_diams = self[0].aper_diams.to(u.arcsec).value
            for i, band_data in tqdm(
                enumerate(self),
                total=len(self),
                desc="Appending aperture correction columns",
                disable=galfind_logger.getEffectiveLevel() > logging.INFO,
            ):
                mag_aper_corr_data = np.zeros(len(cat))
                flux_aper_corr_data = np.zeros(len(cat))
                if len(aper_diams) == 1:
                    mag_aper_corr_factor = band_data.psf.get_aper_corrs(
                        aper_diams[0] * u.arcsec,
                        out_type="mag",
                    )
                    flux_aper_corr_factor = band_data.psf.get_aper_corrs(
                        aper_diams[0] * u.arcsec,
                        out_type="flux",
                    )
                    # only aperture correct if flux is positive
                    mag_aper_corr_data = [
                        mag_aper - mag_aper_corr_factor
                        if flux_aper > 0.0
                        else mag_aper
                        for mag_aper, flux_aper in zip(
                            cat[f"MAG_APER_{band_data.filt_name}"],
                            cat[f"FLUX_APER_{band_data.filt_name}"],
                        )
                    ]
                    flux_aper_corr_data = [
                        flux_aper * flux_aper_corr_factor
                        if flux_aper > 0.0
                        else flux_aper
                        for flux_aper in cat[
                            f"FLUX_APER_{band_data.filt_name}"
                        ]
                    ]
                else:
                    for j, aper_diam in enumerate(aper_diams):
                        mag_aper_corr_factor = band_data.psf.get_aper_corrs(
                            aper_diam * u.arcsec,
                            out_type="mag",
                        )
                        flux_aper_corr_factor = band_data.psf.get_aper_corrs(
                            aper_diam * u.arcsec,
                            out_type="flux",
                        )

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
                                for flux_aper in cat[
                                    f"FLUX_APER_{band_data.filt_name}"
                                ]
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
                                    flux_aper_corr_data,
                                    cat[f"FLUX_APER_{band_data.filt_name}"],
                                )
                            ]
                cat[f"MAG_APER_{band_data.filt_name}_aper_corr"] = (
                    mag_aper_corr_data
                )
                cat[f"FLUX_APER_{band_data.filt_name}_aper_corr"] = (
                    flux_aper_corr_data
                )
                if len(aper_diams) == 1:
                    cat[f"FLUX_APER_{band_data.filt_name}_aper_corr_Jy"] = [
                        funcs.flux_image_to_Jy(element, band_data.ZP).value
                        for element in cat[
                            f"FLUX_APER_{band_data.filt_name}_aper_corr"
                        ]
                    ]
                else:
                    cat[f"FLUX_APER_{band_data.filt_name}_aper_corr_Jy"] = [
                        tuple(
                            [
                                funcs.flux_image_to_Jy(val, band_data.ZP).value
                                for val in element
                            ]
                        )
                        for element in cat[
                            f"FLUX_APER_{band_data.filt_name}_aper_corr"
                        ]
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
        self: Self,
        overwrite: bool = False,
    ) -> None:
        """Append mask flag columns to the photometric catalogue.

        Adds flags indicating which sources are masked in each band and
        aperture.

        Parameters
        ----------
        overwrite : `bool`, optional
            Whether to overwrite existing columns. Default is `False`.

        Raises
        ------
        MissingDataError
            If forced photometry has not been performed on all bands.
        IncompatibleKwargsError
            If RA/DEC labels differ across bands.
        """
        # ensure forced photometry has been run on every band in catalogue
        if not all(
            hasattr(band_data, "forced_phot_args") for band_data in self
        ):
            raise MissingDataError(
                "Forced photometry not performed on all bands!"
            )
        if not (
            all(
                band_data.forced_phot_args["ra_label"]
                == self[0].forced_phot_args["ra_label"]
                for band_data in self
            )
            and all(
                band_data.forced_phot_args["dec_label"]
                == self[0].forced_phot_args["dec_label"]
                for band_data in self
            )
        ):
            raise IncompatibleKwargsError(
                "RA/DEC labels not the same for all bands!"
            )
        tab = Table.read(self.phot_cat_path)

        all_bands = deepcopy(self.band_data_arr)
        if getattr(self, "forced_phot_band") is not None:
            if getattr(self, "forced_phot_band").filt_name not in [
                band_data.filt_name for band_data in all_bands
            ]:
                all_bands += [self.forced_phot_band]
        if hasattr(self, "stacked_band_data_arr"):
            all_bands += self.stacked_band_data_arr
        if (
            not all(
                f"unmasked_{band_data.filt_name}" in tab.colnames
                for band_data in all_bands
            )
            or overwrite
        ):
            # make sky_coords
            ra = tab[self[0].forced_phot_args["ra_label"]]
            dec = tab[self[0].forced_phot_args["dec_label"]]
            sky_coords = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
            # append mask columns to catalogue
            for band_data in all_bands:
                galfind_logger.info(
                    f"Appending {band_data.filt_name} mask"
                    + f" columns to {self.phot_cat_path}"
                )
                wcs = band_data.load_wcs()
                cat_x, cat_y = wcs.world_to_pixel(sky_coords)
                mask = band_data.load_mask()[0]["MASK"]
                if f"unmasked_{band_data.filt_name}" not in tab.colnames:
                    tab[f"unmasked_{band_data.filt_name}"] = np.array(
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
            if not overwrite:
                from ..catalogues import Catalogue

                Catalogue.update_fits_cat(
                    tab,
                    self.phot_cat_path,
                    "OBJECTS",
                )
            else:
                tab.write(self.phot_cat_path, overwrite=True)
                funcs.change_file_permissions(self.phot_cat_path)
            galfind_logger.info(
                f"Appended mask columns to {self.phot_cat_path}"
            )
            # TODO: update README
            galfind_logger.debug("Updating README for mask not implemented!")
        else:
            galfind_logger.debug(
                f"Mask columns already in {self.phot_cat_path}, skipping!"
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
    # # ensure images have all of the relevant extensions - NOT IMPLEMENTED YET
    #     # extract the header files for each extension for each fits image
    #     headers_arr = np.array(
    #         [[hdu.header for hdu in hdul] for hdul in hdul_arr]
    #     )
    #     # ensure they have been taken using the same filter
    #     # NOT IMPLEMENTED YET (assume this comes from header files)
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
    #     # if files have same wcs, x/y dimensions, and pixel scale
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
    #                 # convert np.nans to zeros in science and error
    #                 # maps to allow data only covered by one image
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
    #         hdu_err = fits.ImageHDU(
    #             combined_err, header=im_header, name="ERR")
    #         hdul = fits.HDUList([primary, hdu, hdu_err])
    #         hdul.writeto(combined_image_path, overwrite=True)
    #         mosaic_msg = (
    #             f"Finished mosaicing images at {image_paths=}, "
    #             f"saved to {combined_image_path}"
    #         )
    #         galfind_logger.info(mosaic_msg)
    #         hdul.writeto(combined_image_path, overwrite=True)
    #         # move the individual images into a "stacked" folder
    #         for path in image_paths:
    #             os.makedirs(
    #                 f"{funcs.split_dir_name(path, 'dir')}/stacked",
    #                 exist_ok=True,
    #             )
    #             os.rename(
    #                 path,
    #                 f"{funcs.split_dir_name(path, 'dir')}/stacked/"
    #                 f"{funcs.split_dir_name(path, 'name')}",
    #             )
    #         return combined_image_path

    #     else:  # convert all images to required wcs, x/y dims, pix scale
    #         raise (
    #             NotImplementedError(
    #                 galfind_logger.critical(
    #                     f"Cannot convert images as all of "
    #                     f"{same_wcs=}, {same_dimensions=}, "
    #                     f"{same_pix_scale=} != True"
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
    #     # if not overwrite and README exists, extract previous
    #     # column labels to append col_desc_dict to
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
    #                                 filt_name
    #                                 for filt_name, sex_cat_type in (
    #                                     self.sex_cat_types.items()
    #                                 )
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

    def get_area_tab_path(self: Self) -> str:
        """Get the unmasked-area table path (must be the same for all bands).

        Returns
        -------
        `str`
            Path to the unmasked-area table.

        Raises
        ------
        IncompatibleKwargsError
            If bands have different area table paths.
        """
        area_tab_paths = [band_data.get_area_tab_path() for band_data in self]
        if not all(
            area_tab_path == area_tab_paths[0]
            for area_tab_path in area_tab_paths
        ):
            raise IncompatibleKwargsError(
                "Area table paths for all bands are not the same!"
            )
        return area_tab_paths[0]

    def calc_unmasked_area(
        self: Self,
        mask_selector: Union[str, List[str], Type[Mask_Selector]],
        mask_type: Union[str, List[str]] = "MASK",
        region_selector: Optional[
            Type[Region_Selector], List[Type[Region_Selector]]
        ] = None,
        invert_region: bool = True,
        out_units: u.Quantity = u.arcmin**2,
        **kwargs: Dict[str, Any],
    ) -> u.Quantity:
        """Calculate the total unmasked survey area.

        Computes the area of the survey that is unmasked, accounting for
        specified mask types and regions.

        Parameters
        ----------
        mask_selector : `str`, `list`, or `Mask_Selector`
            Selector defining mask types to exclude.
        mask_type : `str` or `list`, optional
            Mask extension type(s). Default is ``"MASK"``.
        region_selector : `Region_Selector`, `list`, or `None`, optional
            Region to restrict calculation to. Default is `None`.
        invert_region : `bool`, optional
            Whether to invert region selection. Default is `True`.
        out_units : `astropy.units.Quantity`, optional
            Output units for area. Default is arcmin² .
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        `astropy.units.Quantity`
            Unmasked area in the specified units.

        Raises
        ------
        RangeError
            If ``kwargs["z"]`` is given but matches zero or more than one
            redshift bin.
        """

        from ..selection import Mask_Selector

        if not hasattr(self, "unmasked_area"):
            self.unmasked_area = {}

        if isinstance(mask_selector, str):
            mask_selector = mask_selector.split("+")
        if isinstance(mask_type, str):
            mask_type = mask_type.split("+")

        if region_selector is None:
            reg_name = "All"
        else:
            if not isinstance(region_selector, list):
                region_selector = [region_selector]
            reg_name = "+".join(
                [
                    region_selector_.name
                    if not invert_region
                    else region_selector_.fail_name
                    for region_selector_ in region_selector
                ]
            )

        if isinstance(mask_selector, funcs.all_subclasses(Mask_Selector)):
            if "z" not in kwargs.keys():
                z_warn = (
                    "'z' not included in Data.calc_unmasked_area, "
                    "assuming no mask z dependence"
                )
                galfind_logger.warning(z_warn)
                zbin = None
            else:
                zbins = mask_selector.extract_zbins(self)
                # select zbin which matches the redshift of the data
                zbin = [
                    zbin_
                    for zbin_ in zbins
                    if zbin_[0] <= kwargs["z"] < zbin_[1]
                ]
                if len(zbin) != 1:
                    raise RangeError(
                        f"Found {len(zbin)} zbins matching "
                        f"z={kwargs['z']!r}; must match exactly 1."
                    )
                zbin = zbin[0]
        else:
            zbin = None

        mask_selector_name = self._get_mask_selector_name(
            mask_selector, reg_name, zbin
        )
        mask_save_name = "+".join(np.sort(mask_type))

        if mask_selector_name not in self.unmasked_area.keys():
            self.unmasked_area[mask_selector_name] = {}

        area_tab_path = self.get_area_tab_path()
        if Path(area_tab_path).is_file():
            area_tab = Table.read(area_tab_path)
            funcs.make_dirs(area_tab_path)
            area_tab_ = area_tab[
                (
                    (area_tab["mask_instr_band"] == mask_selector_name)
                    & (area_tab["mask_type"] == mask_save_name)
                    & (area_tab["region"] == reg_name)
                )
            ]
            # if zbin is not None:
            #     breakpoint()
            #     area_tab_ = area_tab_[
            #         (area_tab_["zbin_min"] == zbin[0])
            #         & (area_tab_["zbin_max"] == zbin[1])
            #     ]
            if len(area_tab_) == 0:
                calculate = True
            else:
                calculate = False
        else:
            calculate = True

        if calculate:
            Masking.make_area_mask_from_data(
                self,
                mask_selector,
                mask_type,
                region_selector,
                invert_region,
                zbin=zbin,
                **kwargs,
            )

            area_data = {
                "mask_instr_band": [mask_selector_name],
                "mask_type": [mask_save_name],
                "region": [reg_name],
                "unmasked_area": [
                    np.round(
                        self.unmasked_area[mask_selector_name][
                            mask_save_name
                        ].to(out_units),
                        3,
                    )
                ],
            }

            new_area_tab = Table(area_data)
            if Path(area_tab_path).is_file():
                area_tab = vstack([area_tab, new_area_tab])
            else:
                area_tab = new_area_tab
            area_tab.write(area_tab_path, overwrite=True)
            funcs.change_file_permissions(area_tab_path)

        # return unmasked area
        unmasked_area_tab = area_tab[
            (area_tab["mask_instr_band"] == mask_selector_name)
            & (area_tab["mask_type"] == mask_save_name)
            & (area_tab["region"] == reg_name)
        ]["unmasked_area"]
        if len(unmasked_area_tab) != 1:
            raise RangeError(
                f"Found {len(unmasked_area_tab)} unmasked areas for "
                f"mask_selector_name={mask_selector_name!r}, "
                f"mask_save_name={mask_save_name!r}, reg_name="
                f"{reg_name!r}; must match exactly 1."
            )
        unmasked_area = unmasked_area_tab[0] * area_tab["unmasked_area"].unit
        return unmasked_area

    @staticmethod
    def _get_mask_selector_name(
        mask_selector: Type[Mask_Selector],
        reg_name: str,
        zbin: Optional[Tuple[float, float]] = None,
    ) -> str:
        from ..selection import Mask_Selector

        if isinstance(mask_selector, tuple(Mask_Selector.__subclasses__())):
            mask_selector_name = mask_selector.name
        else:
            mask_selector_name = (
                f"{'+'.join(np.sort(mask_selector))}_{reg_name}"
            )

        if zbin is not None:
            if len(zbin) != 2:
                raise LengthMismatchError(
                    f"zbin must be a tuple of (zbin_min, zbin_max) with "
                    f"length 2, got zbin={zbin!r} with length {len(zbin)}."
                )
            if not zbin[0] < zbin[1]:
                raise RangeError(
                    f"zbin_min must be less than zbin_max, got zbin={zbin!r}."
                )
            mask_selector_name += f"_{zbin[0]:.2f}<z<{zbin[1]:.2f}"
        return mask_selector_name

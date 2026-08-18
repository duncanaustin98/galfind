#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Galaxy catalogue with FITS I/O and catalogue-level operations.

Provides Catalogue class as a collection of Galaxy objects with FITS
read/write,
cropping, cross-matching, concatenation, and visualization capabilities.
"""

from __future__ import annotations

import glob
import itertools
import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    NoReturn,
    Optional,
    Tuple,
    Union,
)

import astropy.units as u
import h5py
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, join, vstack
from astropy.utils.masked import Masked
from numpy.typing import NDArray
from tqdm import tqdm

from galfind.imaging.PSF import PSF_Base

if TYPE_CHECKING:
    from . import (
        Band_Cutout,
        Band_Cutout_Base,
        Band_Data_Base,
        Data,
        Filter,
        Mask_Selector,
        Multiple_Filter,
        Region_Selector,
        Selector,
    )
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from .. import config, galfind_logger
from ..galaxy.Galaxy import Galaxy
from ..imaging.Data import Data
from ..imaging.Filter import Multiple_Filter
from ..imaging.Instrument import MIRI, NIRCam
from ..photometry.Photometry_obs import Photometry_obs
from ..sed_fitting.EAZY import EAZY  # noqa F501
from ..sed_fitting.SED_codes import SED_code
from ..spectra.Spectrum import Spectral_Catalogue
from ..utils import Depths
from ..utils import useful_funcs_austind as funcs
from ..visualization.Cutout import (
    Multiple_Band_Cutout,
    Multiple_RGB,
    Stacked_RGB,
)
from .Catalogue_Base import Catalogue_Base


def load_IDs_Table(cat: Table, ID_label: str, **kwargs) -> List[int]:
    """Extract the galaxy ID column from a catalogue table.

    Parameters
    ----------
    cat : `Table`
        Catalogue table to load IDs from.
    ID_label : `str`
        Name of the column containing the galaxy IDs.
    **kwargs
        Unused, accepted for a consistent ``load_ID_func`` call signature.

    Returns
    -------
    `list` of `int`
        Galaxy IDs from the ``ID_label`` column.
    """
    return list(cat[ID_label])


def load_skycoords_Table(
    cat: Table,
    skycoords_labels: Dict[str, str],
    skycoords_units: Dict[str, u.Unit],
    **kwargs,
) -> SkyCoord:
    """Build a `SkyCoord` array from RA/Dec columns of a catalogue table.

    Parameters
    ----------
    cat : `Table`
        Catalogue table to load sky coordinates from.
    skycoords_labels : `dict`
        Mapping with keys ``"RA"`` and ``"DEC"`` giving the names of the
        right ascension and declination columns in `cat`.
    skycoords_units : `dict`
        Mapping with keys ``"RA"`` and ``"DEC"`` giving the
        `astropy.units.Unit`
        of the corresponding columns.
    **kwargs
        Unused, accepted for a consistent ``load_skycoords_func`` call
        signature.

    Returns
    -------
    `SkyCoord`
        Sky coordinates of every galaxy in `cat`.
    """
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
    cat: Table, labels: Dict[u.Quantity, List[str]], **kwargs
) -> np.ndarray:
    """Load a per-band,
    per-aperture-diameter property from a GALFIND-format table.

    GALFIND catalogues store one column per band with all aperture diameters
    stacked together in a single array-valued cell. This unpacks those
    columns into a separate array for each requested aperture diameter.

    Parameters
    ----------
    cat : `Table`
        Catalogue table to load the property from.
    labels : `dict`
        Mapping from aperture diameter (`u.Quantity`) to the `list` of `str`
        column names (one per band) storing the property for that aperture
        diameter. All values must be identical lists of column names.
    **kwargs
        Must include ``cat_aper_diams`` (`u.Quantity` array) giving the full
        set of aperture diameters stored in `cat`, used to work out which
        index of the stacked column corresponds to each requested aperture
        diameter. If not given, it is assumed to equal ``labels.keys()``,
        with a warning logged. May also include ``reshape_by_aper_diams``
        (`bool`) which is forwarded to `useful_funcs_austind.fits_cat_to_np`.

    Returns
    -------
    `dict`
        Mapping from aperture diameter (`u.Quantity`) to `np.ndarray` of the
        requested property for that aperture diameter, or an empty `dict` if
        `cat` is empty.
    """
    if len(cat) == 0:
        galfind_logger.warning(
            "Catalogue is empty! Returning empty properties."
        )
        return {}
    else:
        aper_diams = [label.value for label in labels.keys()] * list(
            labels.keys()
        )[0].unit
        if "cat_aper_diams" not in kwargs.keys():
            galfind_logger.warning(
                f"cat_aper_diams not in {kwargs.keys()=}! "
                + f"Setting to {aper_diams=}"
            )
            kwargs["cat_aper_diams"] = aper_diams
        else:
            assert isinstance(
                kwargs["cat_aper_diams"], u.Quantity
            ), galfind_logger.critical(
                f"{type(kwargs['cat_aper_diams'])=} != u.Quantity!"
            )
            assert isinstance(
                kwargs["cat_aper_diams"].value, (list, np.ndarray)
            ), galfind_logger.critical(
                f"{type(kwargs['cat_aper_diams'])=} != list!"
            )
        # load aperture diameter indices
        aper_diam_indices = [
            i
            for i, aper_diam in enumerate(kwargs["cat_aper_diams"])
            if aper_diam in aper_diams
        ]
        assert len(aper_diam_indices) > 0, galfind_logger.critical(
            "len(aper_diam_indices) <= 0"
        )
        # ensure labels are formatted properly
        assert all(
            label == labels[aper_diams[0]] for label in labels.values()
        ), galfind_logger.critical("All phot_labels not equal!")
        properties = {}
        if "reshape_by_aper_diams" in kwargs.keys() and isinstance(
            kwargs["reshape_by_aper_diams"], bool
        ):
            _properties = funcs.fits_cat_to_np(
                cat,
                labels[aper_diams[0]],
                reshape_by_aper_diams=kwargs["reshape_by_aper_diams"],
            )
        else:
            _properties = funcs.fits_cat_to_np(
                cat,
                labels[aper_diams[0]],
            )
        for aper_diam_index in aper_diam_indices:
            aper_diam = kwargs["cat_aper_diams"][aper_diam_index]
            properties[aper_diam] = _properties[:, :, aper_diam_index]
        return properties


def phot_property_from_fits(
    cat: Table, labels: Dict[u.Quantity, List[str]], **kwargs
) -> np.ndarray:
    """Load a per-band property from a plain (non-GALFIND) FITS table.

    Unlike `phot_property_from_galfind_tab`, each aperture diameter is
    assumed to already have its own set of per-band columns in `cat`, so no
    unpacking of stacked aperture-diameter values is required.

    Parameters
    ----------
    cat : `Table`
        Catalogue table to load the property from.
    labels : `dict`
        Mapping from aperture diameter (`u.Quantity`) to the `list` of `str`
        column names (one per band) storing the property for that aperture
        diameter. All values must be identical lists of column names.
    **kwargs
        May include ``cat_aper_diams`` (`u.Quantity` array); if not given it
        is assumed to equal ``labels.keys()``, with a warning logged.

    Returns
    -------
    `dict`
        Mapping from aperture diameter (`u.Quantity`) to `np.ndarray` of the
        requested property for that aperture diameter.
    """
    aper_diams = [label.value for label in labels.keys()] * list(
        labels.keys()
    )[0].unit
    assert len(aper_diams) > 0, galfind_logger.critical(
        f"{len(aper_diams)=} <= 0"
    )
    if "cat_aper_diams" not in kwargs.keys():
        galfind_logger.warning(
            f"cat_aper_diams not in {kwargs.keys()=}! "
            + f"Setting to {aper_diams=}"
        )
        kwargs["cat_aper_diams"] = aper_diams
    else:
        assert isinstance(
            kwargs["cat_aper_diams"], u.Quantity
        ), galfind_logger.critical(
            f"{type(kwargs['cat_aper_diams'])=} != u.Quantity!"
        )
        assert isinstance(
            kwargs["cat_aper_diams"].value, (list, np.ndarray)
        ), galfind_logger.critical(
            f"{type(kwargs['cat_aper_diams'])=} != list!"
        )
    # ensure labels are formatted properly
    assert all(
        label == labels[aper_diams[0]] for label in labels.values()
    ), galfind_logger.critical("All phot_labels not equal!")
    # extract properties from fits table
    properties = {}
    for aper_diam in aper_diams:
        properties[aper_diam] = np.lib.recfunctions.structured_to_unstructured(
            cat[labels[aper_diams[0]]].as_array()
        )
    return properties


def load_galfind_phot(
    cat: Table,
    phot_labels: Dict[u.Quantity, List[str]],
    err_labels: Dict[u.Quantity, List[str]],
    **kwargs,
) -> Tuple[Dict[u.Quantity, np.ndarray], Dict[u.Quantity, np.ndarray]]:
    """Load photometric fluxes and errors from a GALFIND-format table, in Jy.

    Parameters
    ----------
    cat : `Table`
        Catalogue table to load photometry from.
    phot_labels : `dict`
        Mapping from aperture diameter (`u.Quantity`) to the `list` of `str`
        flux column names (one per band), as returned by e.g.
        `galfind_phot_labels`.
    err_labels : `dict`
        Mapping from aperture diameter (`u.Quantity`) to the `list` of `str`
        flux error column names (one per band). Must have the same keys as
        `phot_labels`.
    **kwargs
        Must include ``ZP`` (image zeropoint) used to convert fluxes/errors
        into Jy via `useful_funcs_austind.flux_image_to_Jy`. Forwarded to
        `phot_property_from_galfind_tab`.

    Returns
    -------
    `tuple` of `dict`
        ``(phot, phot_err)``, each a mapping from aperture diameter
        (`u.Quantity`) to `np.ndarray` fluxes/errors in Jy.
    """
    assert phot_labels.keys() == err_labels.keys(), galfind_logger.critical(
        f"{phot_labels.keys()=} != {err_labels.keys()=}!"
    )
    assert "ZP" in kwargs.keys(), galfind_logger.critical("ZP not in kwargs!")
    phot = {
        aper_diam: funcs.flux_image_to_Jy(_phot, kwargs["ZP"])
        for aper_diam, _phot in phot_property_from_galfind_tab(
            cat, phot_labels, **kwargs
        ).items()
    }
    phot_err = {
        aper_diam: funcs.flux_image_to_Jy(_phot_err, kwargs["ZP"])
        for aper_diam, _phot_err in phot_property_from_galfind_tab(
            cat, err_labels, **kwargs
        ).items()
    }
    return phot, phot_err


def load_phot(
    cat: Table,
    phot_labels: Dict[u.Quantity, List[str]],
    err_labels: Dict[u.Quantity, List[str]],
    **kwargs,
) -> Tuple[Dict[u.Quantity, np.ndarray], Dict[u.Quantity, np.ndarray]]:
    """Load photometric fluxes and errors from a plain FITS table, in Jy.

    Parameters
    ----------
    cat : `Table`
        Catalogue table to load photometry from.
    phot_labels : `dict`
        Mapping from aperture diameter (`u.Quantity`) to the `list` of `str`
        flux column names (one per band).
    err_labels : `dict`
        Mapping from aperture diameter (`u.Quantity`) to the `list` of `str`
        flux error column names (one per band). Must have the same keys as
        `phot_labels`.
    **kwargs
        Must include ``ZP`` (image zeropoint) used to convert fluxes/errors
        into Jy. If ``incl_errs`` is present and `False`, errors are not
        loaded and an array of `None` is returned for `phot_err` instead.
        Forwarded to `phot_property_from_fits`.

    Returns
    -------
    `tuple` of `dict`
        ``(phot, phot_err)``, each a mapping from aperture diameter
        (`u.Quantity`) to `np.ndarray` fluxes/errors in Jy.
    """
    assert phot_labels.keys() == err_labels.keys(), galfind_logger.critical(
        f"{phot_labels.keys()=} != {err_labels.keys()=}!"
    )
    assert "ZP" in kwargs.keys(), galfind_logger.critical("ZP not in kwargs!")
    phot = {
        aper_diam: funcs.flux_image_to_Jy(_phot, kwargs["ZP"])
        for aper_diam, _phot in phot_property_from_fits(
            cat, phot_labels, **kwargs
        ).items()
    }
    if "incl_errs" in kwargs.keys():
        if not kwargs["incl_errs"]:
            phot_err = {
                aper_diam: np.array(list(itertools.repeat(None, len(cat))))
                for aper_diam in phot_labels.keys()
            }
            return phot, phot_err
    phot_err = {
        aper_diam: funcs.flux_image_to_Jy(_phot_err, kwargs["ZP"])
        for aper_diam, _phot_err in phot_property_from_fits(
            cat, err_labels, **kwargs
        ).items()
    }
    return phot, phot_err


def load_galfind_mask(
    cat: Table, mask_labels: List[str], **kwargs
) -> np.ndarray:
    """Load a per-band masked-out flag array from a GALFIND-format table.

    Parameters
    ----------
    cat : `Table`
        Catalogue table to load the mask from.
    mask_labels : `list` of `str`
        Names of the ``unmasked_<band>`` boolean columns (`True` where a
        galaxy is unmasked/has good data in that band).
    **kwargs
        Unused, accepted for a consistent ``load_mask_func`` call signature.

    Returns
    -------
    `np.ndarray`
        Boolean array of shape ``(n_gals, n_bands)``, `True` where a galaxy
        is masked out in a given band (i.e. the inverse of `mask_labels`).
    """
    mask = np.invert(
        funcs.fits_cat_to_np(cat, mask_labels, reshape_by_aper_diams=False)
    )
    return mask


def load_galfind_depths(
    cat: Table, depth_labels: Dict[u.Quantity, List[str]], **kwargs
) -> Dict[u.Quantity, np.ndarray]:
    """Load per-band local depths from a GALFIND-format table, in AB mag.

    Parameters
    ----------
    cat : `Table`
        Catalogue table to load depths from.
    depth_labels : `dict`
        Mapping from aperture diameter (`u.Quantity`) to the `list` of `str`
        depth column names (one per band), as returned by e.g.
        `galfind_depth_labels`.
    **kwargs
        Forwarded to `phot_property_from_galfind_tab`.

    Returns
    -------
    `dict`
        Mapping from aperture diameter (`u.Quantity`) to `np.ndarray` of
        depths in `u.ABmag` for each band.
    """
    return {
        aper_diam: depth * u.ABmag
        for aper_diam, depth in phot_property_from_galfind_tab(
            cat, depth_labels, **kwargs
        ).items()
    }


# def load_cols_Table(
#     cat: Table,
#     labels: List[str],
#     **kwargs
# ) -> np.ndarray:
#     return funcs.fits_cat_to_np(cat, labels)


def check_hdu_exists(cat_path: str, hdu: str) -> bool:
    """Check whether a named HDU extension exists in a FITS file.

    Parameters
    ----------
    cat_path : `str`
        Path to the FITS file to check.
    hdu : `str`
        Name of the HDU extension to look for (case-insensitive).

    Returns
    -------
    `bool`
        `True` if an HDU named `hdu` (upper-cased) exists in `cat_path`.
    """
    # check whether the hdu extension exists
    hdul = fits.open(cat_path)
    return any(hdu_.name == hdu.upper() for hdu_ in hdul)


def open_galfind_cat(
    cat_path: str,
    cat_type: str,
) -> Optional[Table]:
    """Open a single extension of a GALFIND-format multi-extension
    FITS catalogue.

    Parameters
    ----------
    cat_path : `str`
        Path to the multi-extension FITS catalogue.
    cat_type : `str`
        Extension to open. One of ``"ID"``, ``"sky_coord"``, ``"phot"``,
        ``"mask"``, ``"depths"`` (all read from the primary/first extension),
        or any other HDU extension name present in `cat_path`.

    Returns
    -------
    `Table` or `None`
        The requested table extension, or `None` if `cat_type` is not one of
        the recognised keys and not a valid HDU extension name in `cat_path`
        (a warning is logged in this case).
    """
    first_ext_keys = ["ID", "sky_coord", "phot", "mask", "depths"]
    if cat_type in first_ext_keys:
        tab = Table.read(
            cat_path,
            character_as_bytes=False,
            memmap=True,
        )
    elif check_hdu_exists(cat_path, cat_type):
        tab = Table.read(
            cat_path,
            character_as_bytes=False,
            memmap=True,
            hdu=cat_type,
        )
    else:
        err_message = (
            f"{cat_type=} not in {first_ext_keys} and "
            + f"not a valid HDU extension in {cat_path}!"
        )
        galfind_logger.warning(err_message)
        return None

    return tab


def open_galfind_hdr(cat_path: str, cat_type: str) -> Dict[str, str]:
    """Load the FITS header (metadata) of a GALFIND catalogue extension.

    Parameters
    ----------
    cat_path : `str`
        Path to the multi-extension FITS catalogue.
    cat_type : `str`
        Extension to load the header of. See `open_galfind_cat` for the
        recognised values.

    Returns
    -------
    `dict`
        Header/metadata of the requested table extension.
    """
    return open_galfind_cat(cat_path, cat_type).meta


def galfind_phot_labels(
    filterset: Multiple_Filter,
    aper_diams: u.Quantity,
    **kwargs,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Construct GALFIND-format flux/error column names for each band
    and aperture diameter.

    Parameters
    ----------
    filterset : `Multiple_Filter`
        Set of filters/bands to construct column names for.
    aper_diams : `u.Quantity`
        Array of aperture diameters to construct column names for.
    **kwargs
        Must include ``min_flux_pc_err`` (`float`), the minimum percentage
        flux error used in the local-depth error column naming convention.

    Returns
    -------
    `tuple` of `dict`
        ``(phot_labels, err_labels)``, each mapping aperture diameter
        (`u.Quantity`) to the `list` of `str` flux/error column names (one
        per band in `filterset`).
    """
    assert "min_flux_pc_err" in kwargs.keys(), galfind_logger.critical(
        "min_flux_pc_err not in kwargs!"
    )
    phot_labels = {
        aper_diam * aper_diams.unit: [
            f"FLUX_APER_{filt.filt_name}_aper_corr_Jy" for filt in filterset
        ]
        for aper_diam in aper_diams.value
    }
    err_labels = {
        aper_diam * aper_diams.unit: [
            f"FLUXERR_APER_{filt.filt_name}_loc_depth_{str(int(kwargs['min_flux_pc_err']))}pc_Jy"
            for filt in filterset
        ]
        for aper_diam in aper_diams.value
    }
    return phot_labels, err_labels


def jaguar_phot_labels(
    filterset: Multiple_Filter, aper_diams: u.Quantity, **kwargs
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Construct JAGUAR-format flux column names for each band and
    aperture diameter.

    Instrument-specific column-name prefixes (``NRC_``, ``MIRI_``, ``HST_``)
    are used depending on each filter's instrument. JAGUAR simulated
    catalogues have no per-band error columns, so `err_labels` is empty.

    Parameters
    ----------
    filterset : `Multiple_Filter`
        Set of filters/bands to construct column names for.
    aper_diams : `u.Quantity`
        Array of aperture diameters to construct column names for.
    **kwargs
        Must include ``min_flux_pc_err`` (`float`), though it is not used in
        the constructed column names for this convention.

    Returns
    -------
    `tuple` of `dict`
        ``(phot_labels, err_labels)``, mapping aperture diameter
        (`u.Quantity`) to the `list` of `str` flux column names (one per
        band in `filterset`) and to an empty `list` respectively.
    """
    assert "min_flux_pc_err" in kwargs.keys(), galfind_logger.critical(
        "min_flux_pc_err not in kwargs!"
    )
    phot_labels = {
        aper_diam * aper_diams.unit: [
            f"NRC_{filt.filt_name}_fnu"
            if isinstance(filt.instrument, NIRCam)
            else f"MIRI_{filt.filt_name}_fnu"
            if isinstance(filt.instrument, MIRI)
            else f"HST_{filt.filt_name}_fnu"
            for filt in filterset
        ]
        for aper_diam in aper_diams.value
    }
    err_labels = {
        aper_diam * aper_diams.unit: [] for aper_diam in aper_diams.value
    }
    return phot_labels, err_labels


def scattered_phot_labels(
    filterset: Multiple_Filter, aper_diams: u.Quantity, **kwargs
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Construct flux/error column names for scattered-photometry catalogues.

    Parameters
    ----------
    filterset : `Multiple_Filter`
        Set of filters/bands to construct column names for.
    aper_diams : `u.Quantity`
        Array of aperture diameters to construct column names for.
    **kwargs
        Must include ``min_flux_pc_err`` (`float`), though it is not used in
        the constructed column names for this convention.

    Returns
    -------
    `tuple` of `dict`
        ``(phot_labels, err_labels)``, each mapping aperture diameter
        (`u.Quantity`) to the `list` of `str` ``<instrument>.<band>_scattered``
        / ``<instrument>.<band>_err`` column names (one per band in
        `filterset`).
    """
    assert "min_flux_pc_err" in kwargs.keys(), galfind_logger.critical(
        "min_flux_pc_err not in kwargs!"
    )
    phot_labels = {
        aper_diam * aper_diams.unit: [
            f"{filt.instrument_name}.{filt.filt_name}_scattered"
            for filt in filterset
        ]
        for aper_diam in aper_diams.value
    }
    err_labels = {
        aper_diam * aper_diams.unit: [
            f"{filt.instrument_name}.{filt.filt_name}_err"
            for filt in filterset
        ]
        for aper_diam in aper_diams.value
    }
    return phot_labels, err_labels


# def scattered_phot_labels(
#     filterset: Multiple_Filter,
#     aper_diams: u.Quantity,
#     **kwargs
# ) -> Tuple[Dict[str, str], Dict[str, str]]:
#     assert "min_flux_pc_err" in kwargs.keys(), \
#         galfind_logger.critical("min_flux_pc_err not in kwargs!")
#     phot_labels = {
#         aper_diam * aper_diams.unit: [
#             f"{filt.filt_name}_scattered" for filt in filterset
#         ]
#         for aper_diam in aper_diams.value
#     }
#     err_labels = {
#         aper_diam * aper_diams.unit: [
#             f"{filt.filt_name}_err" for filt in filterset
#         ]
#         for aper_diam in aper_diams.value
#     }
#     return phot_labels, err_labels


def galfind_mask_labels(filterset: Multiple_Filter, **kwargs) -> List[str]:
    """Construct GALFIND-format ``unmasked_<band>`` column names.

    Parameters
    ----------
    filterset : `Multiple_Filter`
        Set of filters/bands to construct column names for.
    **kwargs
        Unused, accepted for a consistent ``get_mask_labels`` call signature.

    Returns
    -------
    `list` of `str`
        ``unmasked_<band>`` column names, one per band in `filterset`.
    """
    return [f"unmasked_{filt_name}" for filt_name in filterset.filt_names]


def galfind_depth_labels(
    filterset: Multiple_Filter, aper_diams: u.Quantity, **kwargs
) -> Dict[str, str]:
    """Construct GALFIND-format ``loc_depth_<band>`` column names per
    aperture diameter.

    Parameters
    ----------
    filterset : `Multiple_Filter`
        Set of filters/bands to construct column names for.
    aper_diams : `u.Quantity`
        Array of aperture diameters to construct column names for.
    **kwargs
        Unused, accepted for a consistent ``get_depth_labels`` call signature.

    Returns
    -------
    `dict`
        Mapping from aperture diameter (`u.Quantity`) to the `list` of `str`
        ``loc_depth_<band>`` column names (one per band in `filterset`).
    """
    return {
        aper_diam * aper_diams.unit: [
            f"loc_depth_{filt_name}" for filt_name in filterset.filt_names
        ]
        for aper_diam in aper_diams.value
    }


def scattered_depth_labels(
    filterset: Multiple_Filter, aper_diams: u.Quantity, **kwargs
) -> Dict[str, str]:
    """Construct ``loc_depth_<instrument>.<band>`` column names per
    aperture diameter.

    Parameters
    ----------
    filterset : `Multiple_Filter`
        Set of filters/bands to construct column names for.
    aper_diams : `u.Quantity`
        Array of aperture diameters to construct column names for.
    **kwargs
        Unused, accepted for a consistent ``get_depth_labels`` call signature.

    Returns
    -------
    `dict`
        Mapping from aperture diameter (`u.Quantity`) to the `list` of `str`
        ``loc_depth_<instrument>.<band>`` column names (one per band in
        `filterset`).
    """
    return {
        aper_diam * aper_diams.unit: [
            f"loc_depth_{filt.instrument_name}.{filt.filt_name}"
            for filt in filterset
        ]
        for aper_diam in aper_diams.value
    }


def load_bool_Table(
    tab: Table, select_names: List[str], **kwargs
) -> Dict[str, List[bool]]:
    """Load a set of boolean selection-flag columns from a table.

    Parameters
    ----------
    tab : `Table`
        Table (typically the ``SELECTION`` extension) to load flags from.
    select_names : `list` of `str`
        Names of the boolean columns to load.
    **kwargs
        Unused, accepted for a consistent ``load_selection_func`` call
        signature.

    Returns
    -------
    `dict`
        Mapping from column name to the `list` of `bool` values in that column.
    """
    return {name: list(tab[name]) for name in select_names}


def galfind_selection_labels(tab: Table, **kwargs) -> List[str]:
    """Get the names of all boolean (selection-flag) columns in a table.

    Parameters
    ----------
    tab : `Table`
        Table (typically the ``SELECTION`` extension) to inspect.
    **kwargs
        Unused, accepted for a consistent ``get_selection_labels`` call
        signature.

    Returns
    -------
    `list` of `str`
        Names of the columns in `tab` with a `bool`/`np.bool_` dtype.
    """
    # load selection names dict from header
    return [
        name for name in tab.colnames if tab[name].dtype in (bool, np.bool_)
    ]


def galfind_snr_labels(
    filterset: Multiple_Filter,
    aper_diams: u.Quantity,
    **kwargs,
):
    """Construct per-band,
    per-aperture-diameter stacked-detection SNR column names.

    Parameters
    ----------
    filterset : `Multiple_Filter`
        Set of filters/bands to construct column names for.
    aper_diams : `u.Quantity`
        Array of aperture diameters to construct column names for.
    **kwargs
        Unused, accepted for a consistent call signature.

    Returns
    -------
    `dict`
        Mapping from aperture diameter (`u.Quantity`) to the `list` of `str`
        (truncated) ``sigma_<band>`` column names, one per band in
        `filterset`.
    """
    return {
        aper_diam * aper_diams.unit: [
            funcs.truncate_colname(f"sigma_{filt.filt_name}")
            for filt in filterset
        ]
        for aper_diam in aper_diams.value
    }


class Catalogue_Creator:
    """Factory that builds a `Catalogue` (or a single `Galaxy`) from a
    catalogue FITS file.

    Encapsulates everything needed to open a multi-extension catalogue FITS
    file and turn its rows into `Galaxy`/`Photometry_obs` objects: which
    extensions/columns to read for IDs, sky coordinates, photometry, masks,
    depths and selection flags, how to crop the catalogue, and how to work
    out which bands actually have data for each galaxy. Calling an instance
    (`__call__`) performs the loading and returns a `Catalogue`.

    Parameters
    ----------
    survey : `str`
        Name of the survey/field this catalogue belongs to.
    version : `str`
        Data reduction version of the catalogue.
    cat_path : `str`
        Path to the catalogue FITS file.
    filterset : `Multiple_Filter`
        Full set of filters/bands available in the catalogue.
    aper_diams : `u.Quantity`
        Array of aperture diameters stored in the catalogue.
    crops : `Selector` or `list` of `Selector`, optional
        Selector(s) to crop the catalogue by when loading. Default is `None`
        (no cropping).
    open_cat : `Callable`, optional
        Function used to open a named catalogue extension, called as
        ``open_cat(cat_path, cat_type)``. Default is `open_galfind_cat`.
    open_hdr : `Callable`, optional
        Function used to load the header of a named catalogue extension.
        Default is `open_galfind_hdr`.
    load_ID_func : `Callable`, optional
        Function used to load galaxy IDs from the ``"ID"`` extension.
        Default is `load_IDs_Table`.
    ID_label : `str`, optional
        Name of the ID column. Default is ``"NUMBER"``.
    load_ID_kwargs : `dict`, optional
        Extra keyword arguments passed to `load_ID_func`. Default is ``{}``.
    load_skycoords_func : `Callable`, optional
        Function used to load sky coordinates from the ``"sky_coord"``
        extension. Default is `load_skycoords_Table`.
    skycoords_labels : `dict`, optional
        Mapping of ``"RA"``/``"DEC"`` to their column names. Default is
        ``{"RA": "ALPHA_J2000", "DEC": "DELTA_J2000"}``.
    skycoords_units : `dict`, optional
        Mapping of ``"RA"``/``"DEC"`` to their `astropy.units.Unit`. Default
        is ``{"RA": u.deg, "DEC": u.deg}``.
    load_skycoords_kwargs : `dict`, optional
        Extra keyword arguments passed to `load_skycoords_func`. Default is
        ``{}``.
    load_phot_func : `Callable`, optional
        Function used to load photometric fluxes/errors from the ``"phot"``
        extension. Default is `load_galfind_phot`.
    get_phot_labels : `Callable`, optional
        Function returning the flux/error column names for `filterset` and
        `aper_diams`. Default is `galfind_phot_labels`.
    load_phot_kwargs : `dict`, optional
        Extra keyword arguments passed to `load_phot_func`/`get_phot_labels`.
        Default is ``{"ZP": u.Jy.to(u.ABmag), "min_flux_pc_err": 10.}``.
    load_mask_func : `Callable`, optional
        Function used to load the per-band mask from the ``"mask"``
        extension. Default is `load_galfind_mask`.
    get_mask_labels : `Callable`, optional
        Function returning the mask column names for `filterset`. Default is
        `galfind_mask_labels`.
    load_mask_kwargs : `dict`, optional
        Extra keyword arguments passed to `load_mask_func`. Default is ``{}``.
    load_depth_func : `Callable`, optional
        Function used to load per-band depths from the ``"depths"``
        extension. Default is `load_galfind_depths`.
    get_depth_labels : `Callable`, optional
        Function returning the depth column names for `filterset` and
        `aper_diams`. Default is `galfind_depth_labels`.
    load_depth_kwargs : `dict`, optional
        Extra keyword arguments passed to `load_depth_func`. Default is ``{}``.
    load_selection_func : `Callable`, optional
        Function used to load selection flags from the ``"SELECTION"``
        extension. Default is `load_bool_Table`.
    get_selection_labels : `Callable`, optional
        Function returning the names of the selection-flag columns. Default
        is `galfind_selection_labels`.
    load_selection_kwargs : `dict`, optional
        Extra keyword arguments passed to `load_selection_func`. Default is
        ``{}``.
    load_SED_result_func : `Callable`, optional
        Function used to load SED fitting results. Default is `None`.
    apply_gal_instr_mask : `bool`, optional
        Whether to compute/apply a per-galaxy, per-band "has data" mask
        (see `make_gal_instr_mask`) when loading photometry, masks and
        depths. Default is `True`.
    cache_fits_handle : `bool`, optional
        Whether to keep a single FITS file handle open and reuse it for all
        extension reads, instead of opening the file separately for each
        extension. Significantly speeds up loading many galaxies from the
        same catalogue by avoiding repeated file opens. Falls back to standard
        reading if any errors occur. Default is `True`.
    simulated : `bool`, optional
        Whether the catalogue is simulated rather than observed. Default is
        `False`.

    Attributes
    ----------
    survey : `str`
        Name of the survey/field this catalogue belongs to.
    version : `str`
        Data reduction version of the catalogue.
    cat_path : `str`
        Path to the catalogue FITS file.
    filterset : `Multiple_Filter`
        Full set of filters/bands available in the catalogue.
    aper_diams : `u.Quantity`
        Array of aperture diameters stored in the catalogue.
    crops : `list` of `Selector`
        Selector(s) the catalogue is cropped by, set by `load_crops`.
    crop_mask : `np.ndarray` or `None`
        Boolean row mask corresponding to `crops`, set by `load_crops`.
    apply_gal_instr_mask : `bool`
        Whether a per-galaxy, per-band "has data" mask is applied when
        loading photometry, masks and depths.
    simulated : `bool`
        Whether the catalogue is simulated rather than observed.
    gal_instr_mask : `np.ndarray`
        Per-galaxy, per-band boolean "has data" mask, set the first time
        `make_gal_instr_mask` is called.
    data : `Data`
        Dataset the catalogue was built from, set only when constructed via
        `from_data`.
    """

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
        skycoords_labels: Dict[str, str] = {
            "RA": "ALPHA_J2000",
            "DEC": "DELTA_J2000",
        },
        skycoords_units: Dict[str, u.Unit] = {"RA": u.deg, "DEC": u.deg},
        load_skycoords_kwargs: Dict[str, Any] = {},
        load_phot_func: Callable = load_galfind_phot,
        get_phot_labels: Callable[
            [Multiple_Filter], Dict[str, str]
        ] = galfind_phot_labels,
        load_phot_kwargs: Dict[str, Any] = {
            "ZP": u.Jy.to(u.ABmag),
            "min_flux_pc_err": 10.0,
        },
        load_mask_func: Optional[Callable] = load_galfind_mask,
        get_mask_labels: Callable[
            [Multiple_Filter], Dict[str, str]
        ] = galfind_mask_labels,
        load_mask_kwargs: Dict[str, Any] = {},
        load_depth_func: Optional[Callable] = load_galfind_depths,
        get_depth_labels: Callable[
            [Multiple_Filter], Dict[str, str]
        ] = galfind_depth_labels,
        load_depth_kwargs: Dict[str, Any] = {},
        load_selection_func: Optional[
            Callable[[], Dict[u.Quantity, Dict[str, List[Any]]]]
        ] = load_bool_Table,
        get_selection_labels: Callable[
            [Table, List[str]], List[str]
        ] = galfind_selection_labels,
        load_selection_kwargs: Dict[str, Any] = {},
        load_SED_result_func: Optional[Callable] = None,
        apply_gal_instr_mask: bool = True,
        cache_fits_handle: bool = True,
        simulated: bool = False,
    ):
        """Initialize the Catalogue_Creator with configuration for
        loading catalogues.

        See the class docstring for detailed parameter descriptions.
        """
        self.survey = survey
        self.version = version
        self.cat_path = cat_path
        self.filterset = filterset
        assert isinstance(aper_diams, u.Quantity), galfind_logger.critical(
            f"{type(aper_diams)=} != u.Quantity!"
        )
        assert isinstance(
            aper_diams.value, (list, np.ndarray)
        ), galfind_logger.critical(f"{type(aper_diams.value)=} != list!")
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
        self.load_depth_func = load_depth_func
        self.get_depth_labels = get_depth_labels
        self.load_depth_kwargs = load_depth_kwargs
        self.load_selection_func = load_selection_func
        self.get_selection_labels = get_selection_labels
        self.load_selection_kwargs = load_selection_kwargs
        self.load_SED_result_func = load_SED_result_func
        self.apply_gal_instr_mask = apply_gal_instr_mask
        self.cache_fits_handle = cache_fits_handle
        self.simulated = simulated
        self._tab_cache = {}
        self._fits_handle = None

        self.load_crops(crops)

        # ensure survey/version is correct
        # in primary header which stores the IDs
        hdr = self.open_hdr(self.cat_path, "ID")
        if "SURVEY" in hdr.keys():
            assert hdr["SURVEY"] == self.survey, galfind_logger.critical(
                f"{hdr['SURVEY']=} != {self.survey=}"
            )
        if "VERSION" in hdr.keys():
            assert hdr["VERSION"] == self.version, galfind_logger.critical(
                f"{hdr['VERSION']=} != {self.version=}"
            )

    def __del__(self: Self) -> None:
        """Close the cached FITS file handle when the object is destroyed."""
        if self._fits_handle is not None:
            try:
                self._fits_handle.close()
            except Exception:
                pass  # Silently ignore errors on cleanup

    @classmethod
    def from_data(
        cls,
        data: Data,
        crops: Optional[Union[Type[Selector], List[Type[Selector]]]] = None,
        **kwargs: Dict[str, Any],
    ) -> Catalogue_Creator:
        """Construct a `Catalogue_Creator` from a `Data` object.

        Derives `survey`, `version`, `cat_path`, `filterset` and `aper_diams`
        from `data` rather than requiring them to be passed explicitly, and
        attaches `data` to the resulting instance.

        Parameters
        ----------
        data : `Data`
            Dataset to build the catalogue creator from.
        crops : `Selector` or `list` of `Selector`, optional
            Selector(s) to crop the catalogue by when loading. Default is
            `None` (no cropping).
        **kwargs
            Additional keyword arguments forwarded to the `Catalogue_Creator`
            constructor.

        Returns
        -------
        `Catalogue_Creator`
            New instance with its `data` attribute set to `data`.
        """
        cat_creator = cls(
            data.survey,
            data.version,
            data._get_phot_cat_path(),
            data.filterset,
            data.aper_diams,
            crops=crops,
            **kwargs,
        )
        cat_creator.data = data
        return cat_creator

    def __call__(
        self: Self,
        psfs: Optional[
            Union[
                List[Optional[Type[PSF_Base]]],
                NDArray[Optional[Type[PSF_Base]]],
                Dict[str, Optional[Type[PSF_Base]]],
            ]
        ] = None,
        cropped: bool = True,
        load_gals: bool = True,
        # extra_crops: Union[
        #     Type[Selector],
        #     List[Type[Selector]]
        # ] = [],
    ) -> Catalogue:
        """Load and construct a Catalogue from the configured FITS file.

        Reads the catalogue table(s), applies any configured crops, and
        constructs
        a Catalogue object with Galaxy instances. Applies PSF information and
        selection flags if available.

        Parameters
        ----------
        psfs : `dict`, `list`, `ndarray` of `PSF_Base`, or `None`, optional
            PSF objects to associate with each band. Default is `None`.
        cropped : `bool`, optional
            Whether to apply configured crops to the loaded catalogue.
            Default is `True`.
        load_gals : `bool`, optional
            Whether to construct Galaxy objects. If `False`, returns an empty
            catalogue. Default is `True`.

        Returns
        -------
        `Catalogue`
            The loaded catalogue, possibly empty if load_gals is False.
        """
        galfind_logger.info(f"Making {repr(self)} catalogue!")
        if load_gals:
            # make array of Photometry_obs for each aperture diameter
            galfind_logger.debug(f"Loading {repr(self)} photometry!")
            # if len(extra_crops) > 0:
            #     self.load_crops(self.crops, extra_crops = extra_crops)
            IDs = self.load_IDs(cropped)
            sky_coords = self.load_skycoords(cropped)
            phot, phot_err = self.load_phot(cropped)
            depths = self.load_depths(cropped)
            selection_flags_arr, selection_kwargs_arr = (
                self.load_selection_flags_kwargs(cropped)
            )
            filterset_arr = self.load_gal_filtersets(
                length=len(IDs), cropped=cropped
            )
            SED_results = {}
            phot_obs_arr = [
                {
                    aper_diam: Photometry_obs(
                        filterset_arr[i],
                        phot[aper_diam][i],
                        phot_err[aper_diam][i],
                        depths[aper_diam][i],
                        aper_diam,
                        SED_results=SED_results,
                        psfs=psfs,
                        simulated=self.simulated,
                    )
                    for aper_diam in self.aper_diams
                }
                for i in range(len(filterset_arr))
            ]
            assert (
                len(IDs) == len(sky_coords) == len(phot_obs_arr)
            ), galfind_logger.critical(
                f"{len(IDs)=} != {len(sky_coords)=} "
                f"!= {len(phot_obs_arr)=}!"
            )
            # make an array of galaxy objects to be stored in the catalogue
            galfind_logger.debug(
                f"Loading {self.survey} {self.version} "
                f"{self.cat_name} galaxies!"
            )
            # , origin_survey = self.survey
            gals = [
                Galaxy(
                    ID,
                    sky_coord,
                    phot_obs,
                    selection_flags=selection_flags,
                    selection_kwargs=selection_kwargs,
                    cat_filterset=self.filterset,
                    survey=self.survey,
                    version=self.version,
                    simulated=self.simulated,
                )
                for (
                    ID,
                    sky_coord,
                    phot_obs,
                    selection_flags,
                    selection_kwargs,
                ) in zip(
                    IDs,
                    sky_coords,
                    phot_obs_arr,
                    selection_flags_arr,
                    selection_kwargs_arr,
                )
            ]
            # if len(extra_crops) > 0:
            #     self.load_crops(self.crops)
        else:
            galfind_logger.info(
                f"Not loading galaxies for {self.survey} {self.version} "
                f"{self.cat_name} catalogue!"
            )
            gals = []
        cat = Catalogue(gals, self)
        # if len(extra_crops) > 0:
        #     setattr(cat, "extra_crops", extra_crops)
        if load_gals and cropped and len(self._crops_to_perform) != 0:
            # perform the crops
            crops_performed = set()
            for crop in self._crops_to_perform:
                galfind_logger.info(f"Performing {repr(crop)}!")
                crop(cat, return_copy=False)
                crops_performed.add(id(crop))
            # Only recurse if there are new crops to perform
            self.load_crops(self.crops)
            if self._crops_to_perform and not all(
                id(c) in crops_performed for c in self._crops_to_perform
            ):
                cat = self(
                    psfs=psfs,
                    cropped=True,
                    load_gals=load_gals,
                )
            else:
                # point to data if provided
                if hasattr(self, "data"):
                    cat.data = self.data
                galfind_logger.info(f"Made {self.cat_path} catalogue!")
        else:
            # point to data if provided
            if hasattr(self, "data"):
                cat.data = self.data
            galfind_logger.info(f"Made {self.cat_path} catalogue!")
        return cat

    @property
    def cat_name(self) -> str:
        """`str`: Name describing the crop(s) currently applied via `crops`."""
        return funcs.get_crop_name(self.crops)

    def _open_tab(self: Self, cat_type: str) -> Any:
        """Open and cache a named catalogue table extension.

        When `cache_fits_handle` is True, keeps a single FITS file handle open
        and reads all extensions from it, avoiding repeated file opens.

        Parameters
        ----------
        cat_type : `str`
            Name of the catalogue extension to open (e.g., "ID", "phot").

        Returns
        -------
        `Table` or `None`
            The requested table extension, cached for future access.
        """
        if cat_type not in self._tab_cache:
            if self.cache_fits_handle:
                try:
                    from astropy.io import fits

                    # Open FITS file once and keep it open
                    if self._fits_handle is None:
                        self._fits_handle = fits.open(
                            self.cat_path, memmap=True
                        )

                    # Read the requested extension from the open handle
                    first_ext_keys = [
                        "ID",
                        "sky_coord",
                        "phot",
                        "mask",
                        "depths",
                    ]
                    if cat_type in first_ext_keys:
                        # These are typically in the primary HDU
                        hdu = self._fits_handle[0]
                    else:
                        # Look for the HDU by name
                        hdu = self._fits_handle[cat_type]

                    # Read with Table.read() to preserve metadata
                    self._tab_cache[cat_type] = Table.read(hdu, format="fits")
                except Exception as e:
                    # Fall back to default behavior if caching fails
                    galfind_logger.warning(
                        f"FITS handle caching failed for {cat_type}: {e}. "
                        f"Falling back to standard catalogue reading."
                    )
                    self._tab_cache[cat_type] = self.open_cat(
                        self.cat_path, cat_type
                    )
            else:
                # Use the original open_cat function (default behavior)
                self._tab_cache[cat_type] = self.open_cat(
                    self.cat_path, cat_type
                )

        return self._tab_cache[cat_type]

    def load_tab(
        self: Self,
        cat_type: str,
        cropped: bool = True,
    ) -> Table:
        """Load (and cache) a named extension of the catalogue,
        optionally cropped.

        Parameters
        ----------
        cat_type : `str`
            Extension to load. See `open_galfind_cat` for the recognised
            values.
        cropped : `bool`, optional
            Whether to apply `crop_mask` (from `load_crops`) to the loaded
            table. Default is `True`.

        Returns
        -------
        `Table` or `None`
            The requested (optionally cropped) table extension, or `None` if
            the extension does not exist.
        """
        tab = self._open_tab(cat_type)
        if tab is None:
            return None
        else:
            if cropped and self.crop_mask is not None:
                return tab[self.crop_mask]
            else:
                return tab

    def load_crops(
        self: Self,
        crops: Optional[Union[Type[Selector], List[Type[Selector]]]] = None,
        # extra_crops: Union[
        #     Type[Selector],
        #     List[Type[Selector]]
        # ] = [],
    ) -> None:
        """Set the crop selector(s) to apply and (
            re)compute the resulting row mask.

        Normalises `crops` into `self.crops`, recomputes `self.crop_mask`
        (via `_get_crop_mask`), and, if `apply_gal_instr_mask` is `True`,
        (re)computes `self.gal_instr_mask` (via `make_gal_instr_mask`).

        Parameters
        ----------
        crops : `Selector` or `list` of `Selector`, optional
            Selector(s) to crop the catalogue by. Default is `None`, which
            clears `self.crops` (no cropping).
        """
        if crops is None:
            self.crops = []
        elif not isinstance(crops, (list, np.ndarray)):
            self.crops = [crops]
        else:
            self.crops = crops
        self._get_crop_mask()  # extra_crops = extra_crops)
        if self.apply_gal_instr_mask:
            self.make_gal_instr_mask()

    @staticmethod
    def _get_selection_kwarg_colnames(selector: Type[Selector]) -> List[str]:
        """Extract selection keyword argument column names from a Selector.

        Recursively retrieves the names of all columns storing selector
        parameters,
        handling both single selectors and compound Multiple_Selector
        instances.

        Parameters
        ----------
        selector : `Selector`
            Selector to extract column names from.

        Returns
        -------
        `list` of `str`
            Column names of format ``<selector_name>__<kwarg_name>``.
        """
        from ..selection.Selector import Multiple_Selector

        kwarg_colnames = []
        if isinstance(selector, funcs.all_subclasses(Multiple_Selector)):
            for sub_selector in selector:
                kwarg_colnames.extend(
                    Catalogue_Creator._get_selection_kwarg_colnames(
                        sub_selector
                    )
                )
        else:
            kwarg_colnames.extend(
                [
                    f"{selector.name}__{kwarg_name}"
                    for kwarg_name, kwarg_dtype in zip(
                        *selector.select_kwarg_names_dtypes
                    )
                ]
            )
        return kwarg_colnames

    def _get_crop_mask(
        self: Self,
        # extra_crops: Union[
        #     Type[Selector],
        #     List[Type[Selector]]
        # ] = [],
    ) -> Optional[NDArray[bool]]:
        """Compute the boolean row mask for configured crops.

        Reads selection-flag columns from the SELECTION extension and computes
        a boolean mask indicating which rows satisfy all configured crop
        selectors.
        Stores crops that are not yet in the selection table for deferred
        application.

        Returns
        -------
        `np.ndarray` of `bool` or `None`
            Boolean mask of shape (n_rows,) where `True` indicates rows
            that pass
            all crops. `None` if deferred crops remain.
        """
        # from . import Selector
        # if isinstance(extra_crops, funcs.all_subclasses(Selector)):
        #     extra_crops = [extra_crops]
        tab = self._open_tab("SELECTION")
        if tab is None:
            tab = self._open_tab("ID")
            self._crops_to_perform = self.crops
        elif len(self.crops) > 0:  #  + extra_crops
            # crop table using crop dict
            self._crops_to_perform = []
            keep_arr = []
            for selector in self.crops:  # + extra_crops:
                # iterate through selectors, appending all kwarg colnames
                kwarg_colnames = self._get_selection_kwarg_colnames(selector)
                kwarg_colnames = [
                    selector.shorten_kwarg_colname(name, max_len=68)
                    for name in kwarg_colnames
                ]
                if selector.name in tab.colnames and all(
                    [
                        kwarg_colname in tab.colnames
                        for kwarg_colname in kwarg_colnames
                    ]
                ):
                    keep_arr.extend(
                        [np.array(tab[selector.name]).astype(bool)]
                    )
                    galfind_logger.info(
                        f"Catalogue cropped by {selector.name}"
                    )
                else:
                    # if selector in id_crop:
                    #     err_message = (
                    #         f"extra_crop = {selector.name} not "
                    #         f"in {tab.colnames=}!"
                    #     )
                    #     galfind_logger.critical(err_message)
                    #     raise Exception(err_message)
                    # else:
                    galfind_logger.info(f"{selector.name} not yet performed!")
                    self._crops_to_perform.append(selector)
            # crop table if required
            if len(keep_arr) > 0 and len(self._crops_to_perform) == 0:
                self.crop_mask = np.array(
                    np.logical_and.reduce(keep_arr)
                ).astype(bool)
                return self.crop_mask
        else:
            self._crops_to_perform = []
        if len(self._crops_to_perform) == 0:
            self.crop_mask = np.full(len(tab), True)
        else:
            self.crop_mask = None
        return self.crop_mask

    def load_IDs(self: Self, cropped: bool = True) -> List[int]:
        """Load the galaxy IDs from the catalogue's ``"ID"`` extension.

        Parameters
        ----------
        cropped : `bool`, optional
            Whether to apply `crop_mask` before loading IDs. Default is `True`.

        Returns
        -------
        `list` of `int`
            Galaxy IDs, as returned by `load_ID_func`.
        """
        tab = self.load_tab("ID", cropped)
        return self.load_ID_func(tab, self.ID_label, **self.load_ID_kwargs)

    def load_skycoords(self: Self, cropped: bool = True) -> SkyCoord:
        """Load the sky coordinates from the catalogue's
        ``"sky_coord"`` extension.

        Parameters
        ----------
        cropped : `bool`, optional
            Whether to apply `crop_mask` before loading sky coordinates.
            Default is `True`.

        Returns
        -------
        `SkyCoord`
            Sky coordinates of every (cropped) galaxy, as returned by
            `load_skycoords_func`.
        """
        tab = self.load_tab("sky_coord", cropped)
        return self.load_skycoords_func(
            tab,
            self.skycoords_labels,
            self.skycoords_units,
            **self.load_skycoords_kwargs,
        )

    def load_phot(
        self: Self, cropped: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load photometric fluxes and errors from the catalogue's
        ``"phot"`` extension.

        Loads fluxes/errors via `load_phot_func`, optionally restricts each
        galaxy's bands to those with data (via `gal_instr_mask`, if
        `apply_gal_instr_mask` is `True`), and wraps the per-band values in
        an `astropy.utils.masked.Masked` array using the catalogue mask (see
        `load_mask`), if available.

        Parameters
        ----------
        cropped : `bool`, optional
            Whether to apply `crop_mask` before loading photometry. Default
            is `True`.

        Returns
        -------
        `tuple` of `dict`
            ``(phot, phot_err)``, each a mapping from aperture diameter
            (`u.Quantity`) to a per-galaxy list of flux/error arrays in Jy.
        """
        tab = self.load_tab("phot", cropped)
        # load the photometric fluxes and errors in units of Jy
        phot_labels, err_labels = self.get_phot_labels(
            self.filterset, self.aper_diams, **self.load_phot_kwargs
        )
        phot, phot_err = self.load_phot_func(
            tab, phot_labels, err_labels, **self.load_phot_kwargs
        )
        if self.apply_gal_instr_mask:
            if cropped and self.crop_mask is not None:
                gal_instr_mask = self.gal_instr_mask[self.crop_mask]
            else:
                gal_instr_mask = self.gal_instr_mask
            phot = {
                aper_diam: self._apply_gal_instr_mask(_phot, gal_instr_mask)
                for aper_diam, _phot in phot.items()
            }
            phot_err = {
                aper_diam: self._apply_gal_instr_mask(
                    _phot_err, gal_instr_mask
                )
                for aper_diam, _phot_err in phot_err.items()
            }
        mask = self.load_mask(cropped)
        if mask is not None:
            phot = {
                aper_diam: [
                    Masked(gal_phot, mask=gal_mask)
                    for gal_phot, gal_mask in zip(_phot, mask)
                ]
                for aper_diam, _phot in phot.items()
            }
            phot_err = {
                aper_diam: [
                    Masked(gal_phot_err, mask=gal_mask)
                    for gal_phot_err, gal_mask in zip(_phot_err, mask)
                ]
                for aper_diam, _phot_err in phot_err.items()
            }
        return phot, phot_err

    def load_mask(self: Self, cropped: bool = True) -> Optional[NDArray[bool]]:
        """Load the per-galaxy, per-band mask from the catalogue's
        ``"mask"`` extension.

        Parameters
        ----------
        cropped : `bool`, optional
            Whether to apply `crop_mask` before loading the mask. Default is
            `True`.

        Returns
        -------
        `np.ndarray` or `None`
            Boolean mask array (`True` where a galaxy is masked out in a
            given band), optionally restricted to bands with data via
            `gal_instr_mask`, or `None` if there is no mask data or no
            `load_mask_func`/`get_mask_labels` provided.

        Raises
        ------
        Exception
            If `load_mask_func` raises while loading the mask.
        """
        tab = self.load_tab("mask", cropped)
        if len(tab) == 0:
            galfind_logger.warning(
                f"No rows in the 'mask' table extension of {self.cat_path} "
                f"for crop '{self.cat_name}'; proceeding without a mask."
            )
            return None
        if self.load_mask_func is not None:
            mask_labels = self.get_mask_labels(self.filterset)
            try:
                mask = self.load_mask_func(
                    tab, mask_labels, **self.load_mask_kwargs
                )
            except Exception as e:
                err_message = f"Error loading mask: {e}"
                galfind_logger.critical(err_message)
                raise (Exception(err_message))
            if self.apply_gal_instr_mask:
                if cropped and self.crop_mask is not None:
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
            # {aper_diam: list(itertools.repeat(None, len(tab)))
            # for aper_diam in self.aper_diams}

    def load_depths(self, cropped: bool = True) -> np.ndarray:
        """Load per-galaxy, per-band local depths from the catalogue's
        ``"depths"`` extension.

        Parameters
        ----------
        cropped : `bool`, optional
            Whether to apply `crop_mask` before loading depths. Default is
            `True`.

        Returns
        -------
        `dict`
            Mapping from aperture diameter (`u.Quantity`) to a per-galaxy
            array of depths, optionally restricted to bands with data via
            `gal_instr_mask`. If no `load_depth_func`/`get_depth_labels` is
            provided, a `dict` of `None`-filled lists is returned instead
            (with a warning logged).

        Raises
        ------
        Exception
            If `load_depth_func` raises while loading the depths.
        """
        tab = self.load_tab("depths", cropped)
        if (
            self.load_depth_func is not None
            and self.get_depth_labels is not None
        ):
            depth_labels = self.get_depth_labels(
                self.filterset, self.aper_diams
            )
            try:
                depths = self.load_depth_func(
                    tab, depth_labels, **self.load_depth_kwargs
                )
            except Exception as e:
                err_message = f"Error loading depths: {e}"
                galfind_logger.critical(err_message)
                raise (Exception(err_message))
            if self.apply_gal_instr_mask:
                if cropped and self.crop_mask is not None:
                    gal_instr_mask = self.gal_instr_mask[self.crop_mask]
                else:
                    gal_instr_mask = self.gal_instr_mask
                depths = {
                    aper_diam: self._apply_gal_instr_mask(
                        _depths, gal_instr_mask
                    )
                    for aper_diam, _depths in depths.items()
                }
            return depths
        else:
            galfind_logger.warning(
                "Either load depth or depth label function not provided!"
            )
            return {
                aper_diam: list(itertools.repeat(None, len(tab)))
                for aper_diam in self.aper_diams
            }

    def load_selection_flags_kwargs(
        self: Self,
        cropped: bool = True,
    ) -> Tuple[List[Dict[str, bool]], List[Dict[str, Dict[str, Any]]]]:
        """Load per-galaxy selection flags and the keyword arguments used
        to make them.

        Reads the boolean selection-flag columns from the catalogue's
        ``"SELECTION"`` extension along with any accompanying
        ``"<selector>__<kwarg>"`` columns storing the parameters used to
        generate each flag.

        Parameters
        ----------
        cropped : `bool`, optional
            Whether to apply `crop_mask` before loading selection flags.
            Default is `True`.

        Returns
        -------
        `tuple` of `list`
            ``(selection_flags, selection_kwargs)``. `selection_flags` is a
            per-galaxy `list` of `dict` mapping selector name to `bool`.
            `selection_kwargs` is a per-galaxy `list` of `dict` mapping
            selector name to a `dict` of the keyword arguments used for that
            selection. If the ``"SELECTION"`` extension does not exist, both
            are `list` of `None` (one per galaxy).

        Raises
        ------
        Exception
            If the ``"SELECTION"`` extension exists but no
            `load_selection_func`/`get_selection_labels` is provided.
        """
        tab = self.load_tab("SELECTION", cropped)
        if (
            self.load_selection_func is not None
            and self.get_selection_labels is not None
            and tab is not None
        ):
            select_labels = self.get_selection_labels(
                tab, **self.load_selection_kwargs
            )
            select_dict = self.load_selection_func(
                tab, select_labels, **self.load_selection_kwargs
            )
            # ensure all selection dict values are the same length as
            # the catalogue
            assert all(
                len(selection) == len(tab)
                for selection in select_dict.values()
            ), galfind_logger.critical(
                "Not all selection values are the same length "
                "as the catalogue!"
            )
            selection_flags = [
                {key: bool(values[i]) for key, values in select_dict.items()}
                for i in range(len(tab))
            ]
            # TODO: Generalize for non galfind-like tables
            selection_kwarg_colnames = {
                select_label: [
                    colname.split("__")[1]
                    for colname in tab.colnames
                    if colname.startswith(select_label)
                    and tab[colname].dtype not in (bool, np.bool_)
                ]
                for select_label in select_labels
            }
            # TODO: Cast fits values to None instead of the numpy
            # default (e.g. 'None', np.nan, etc.)
            selection_kwargs = [
                {
                    select_label: {
                        kwarg_name: tab[f"{select_label}__{kwarg_name}"][i]
                        for kwarg_name in selection_kwarg_colnames[
                            select_label
                        ]
                        if tab[f"{select_label}__{kwarg_name}"].dtype
                        not in (bool, np.bool_)
                    }
                    for select_label in select_labels
                }
                for i in range(len(tab))
            ]
        elif tab is None:
            galfind_logger.warning("'SELECTION' tab is None!")
            tab = self.load_tab("ID", cropped)
            selection_flags = list(itertools.repeat(None, len(tab)))
            selection_kwargs = list(itertools.repeat(None, len(tab)))
        else:
            err_message = "Selection function not provided!"
            galfind_logger.critical(err_message)
            raise Exception(err_message)

        return selection_flags, selection_kwargs

    # current bottleneck
    def make_gal_instr_mask(
        self,
        null_data_vals: List[Union[float, np.nan]] = [0.0],
        overwrite: bool = False,
        timed: bool = True,
    ) -> NoReturn:
        """Compute (or load a cached) per-galaxy, per-band "has data" mask.

        For every galaxy and band in `filterset`, flags whether that band's
        (uncropped) flux is finite and not one of `null_data_vals`, i.e.
        whether real photometric data exists for that galaxy/band
        combination. The result is cached to an HDF5 file under
        ``GALFIND_WORK/Masks/<survey>/has_data_mask/...`` and stored as
        `self.gal_instr_mask` (only set if not already present).

        Parameters
        ----------
        null_data_vals : `list`, optional
            Values (in addition to non-finite values) treated as "no data".
            Default is ``[0.0]``.
        overwrite : `bool`, optional
            If `True`, recompute the mask even if a cached ``.h5`` file
            already exists (rather than loading it). Default is `False`.
        timed : `bool`, optional
            Whether to show a progress bar while computing the mask.
            Default is `True`.

        Returns
        -------
        `None`

        Raises
        ------
        Exception
            If neither the catalogue header nor `self.survey`/`self.filterset`/
            `self.aper_diams` provide enough information to determine the
            cache directory.
        """
        meta = self.open_hdr(self.cat_path, "ID")
        req_metakeys = ["SURVEY", "INSTR", "APERDIAM"]
        if all(name in meta.keys() for name in req_metakeys):
            save_dir = (
                f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{meta['SURVEY']}/"
                + f"has_data_mask/{meta['INSTR']}/{meta['APERDIAM']}"
            )
        elif (
            self.survey is not None
            and self.filterset is not None
            and self.aper_diams is not None
        ):
            if len(self.aper_diams) != 1:
                galfind_logger.warning(
                    f"len({self.aper_diams=}) != 1! Using "
                    f"{self.aper_diams[0]} for gal_instr_mask!"
                )
            save_dir = (
                f"{config['DEFAULT']['GALFIND_WORK']}/Masks/{self.survey}/"
                + f"has_data_mask/{self.filterset.instrument_name}"
            )
        else:
            err_message = (
                f"Not all of {req_metakeys} in {self.cat_path} "
                + "nor 'survey', 'filterset', and 'aper_diams' "
                "provided in self!"
            )
            galfind_logger.critical(err_message)
            raise Exception(err_message)
        cat_basename = Path(self.cat_path).stem
        save_path = f"{save_dir}/{cat_basename}.h5"
        funcs.make_dirs(save_path)
        if Path(save_path).is_file() or overwrite:
            # load in gal_instr_mask from .h5
            hf = h5py.File(save_path, "r")
            gal_instr_mask = np.array(hf["has_data_mask"])
            galfind_logger.info(f"Loaded 'has_data_mask' from {save_path}")
        else:
            galfind_logger.info(f"Making 'has_data_mask' for {self.cat_name}!")
            # calculate the mask that is used to crop photometry to
            # only bands including data
            tab = self.load_tab("phot", cropped=False)
            phot_labels, err_labels = self.get_phot_labels(
                self.filterset, self.aper_diams, **self.load_phot_kwargs
            )
            phot = self.load_phot_func(
                tab, phot_labels, err_labels, **self.load_phot_kwargs
            )[0]
            phot = list(phot.values())[0]
            if timed:
                gal_instr_mask = np.array(
                    [
                        [
                            True
                            if val not in null_data_vals and np.isfinite(val)
                            else False
                            for val in gal_phot
                        ]
                        for gal_phot in tqdm(
                            phot,
                            desc="Making has_data_mask",
                            total=len(phot),
                            disable=galfind_logger.getEffectiveLevel()
                            > logging.INFO,
                        )
                    ]
                )
            else:
                gal_instr_mask = np.array(
                    [
                        [
                            True
                            if val not in null_data_vals and np.isfinite(val)
                            else False
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
        """Mask out bands with no data for each galaxy.

        Applies a per-galaxy, per-band boolean mask to filter out bands where
        data is unavailable, returning only data for bands with valid
        measurements.

        Parameters
        ----------
        arr : `np.ndarray`
            Per-galaxy array of per-band values to mask.
        gal_instr_mask : `np.ndarray`
            Boolean mask of shape (n_galaxies, n_bands), `True` where data
            exists.

        Returns
        -------
        `list` of `np.ndarray`
            Masked arrays, one per galaxy, containing only bands with data.
        """
        assert len(gal_instr_mask) == len(arr), galfind_logger.critical(
            f"{len(gal_instr_mask)} != {len(arr)}!"
        )
        return [ele[mask] for ele, mask in zip(arr, gal_instr_mask)]

    def load_gal_filtersets(
        self: Self,
        length: int,
        cropped: bool = True,
    ) -> List[Multiple_Filter]:
        """Build the per-galaxy `Multiple_Filter` set of bands with
        available data.

        If `apply_gal_instr_mask` is `True`, each galaxy is assigned the
        subset of `filterset` for which `gal_instr_mask` indicates data is
        present (unique combinations are computed once and re-used).
        Otherwise every galaxy is assigned the full `filterset`.

        Parameters
        ----------
        length : `int`
            Number of galaxies to build filtersets for.
        cropped : `bool`, optional
            Whether to apply `crop_mask` to `gal_instr_mask` before use.
            Default is `True`.

        Returns
        -------
        `list` of `Multiple_Filter`
            Filterset for each of the `length` galaxies.
        """
        if self.apply_gal_instr_mask:
            # create set of filtersets to be pointed to by sources
            # with these bands available
            galfind_logger.debug(f"Making {self.cat_name} unique filtersets!")
            if cropped and self.crop_mask is not None:
                gal_instr_mask = self.gal_instr_mask[self.crop_mask]
            else:
                gal_instr_mask = self.gal_instr_mask
            unique_filt_comb = np.unique(gal_instr_mask, axis=0)
            unique_filtersets = [
                deepcopy(self.filterset)[data_comb]
                for data_comb in unique_filt_comb
            ]
            filterset_arr = [
                unique_filtersets[
                    np.where(np.all(unique_filt_comb == data_comb, axis=1))[0][
                        0
                    ]
                ]
                for data_comb in gal_instr_mask
            ]
            galfind_logger.debug(f"Made {self.cat_name} unique filtersets!")
        else:
            filterset_arr = list(itertools.repeat(self.filterset, length))
        return filterset_arr


class Catalogue(Catalogue_Base):
    """A galaxy catalogue, storing a list of `Galaxy` objects for a
    single survey/version.

    The central catalogue-level class in GALFIND: a thin, list-like wrapper
    around a collection of `Galaxy` objects (see `Catalogue_Base` for
    indexing, iteration and cross-matching behaviour) plus catalogue-wide
    convenience methods for loading SExtractor/photometric properties,
    making cutouts/RGBs, running photometry diagnostics, computing
    number-density-function quantities (e.g. Vmax) and scattering
    photometry. Instances are normally constructed via `pipeline`,
    `from_data`, or by calling a `Catalogue_Creator` instance, rather than
    directly.

    Attributes not set directly on the instance (e.g. `survey`, `version`,
    `filterset`, `aper_diams`, `data`) are transparently looked up on
    `cat_creator`, or (failing that) collected from each `Galaxy` in `gals`,
    via `Catalogue_Base.__getattr__`.

    Parameters
    ----------
    gals : `list` of `Galaxy`
        Galaxies making up this catalogue.
    cat_creator : `Catalogue_Creator`
        Creator object used to build this catalogue, providing access to
        the underlying catalogue FITS file and loading configuration.

    Attributes
    ----------
    gals : `list` of `Galaxy`
        Galaxies making up this catalogue.
    cat_creator : `Catalogue_Creator`
        Creator object this catalogue was built with.
    data : `Data`
        Dataset the catalogue's photometry was extracted from, set when
        available from `cat_creator.data`.
    """

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
        crops: Optional[Union[Type[Selector], List[Type[Selector]]]] = None,
        load_gals: bool = True,
        mask_method: str = "auto",
        psf_method: Optional[str] = "empirical",
        psf_homog_filt: Optional[str] = "F444W",
    ) -> Catalogue:
        """Run the full galfind data-reduction pipeline and load the
        resulting catalogue.

        Convenience wrapper that runs `Data.pipeline` to reduce the imaging
        for a survey/version into a photometric catalogue, then loads that
        catalogue via `from_data`.

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
        crops : `Selector` or `list` of `Selector`, optional
            Selector(s) to crop the resulting catalogue by. Default is
            `None`.
        load_gals : `bool`, optional
            Whether to load `Galaxy` objects into the returned catalogue.
            Default is `True`.
        mask_method : `str`, optional
            Masking method to use, one of ``"auto"`` or ``"manual"``.
            Default is ``"auto"``.
        psf_method : `str`, optional
            Method used to construct/retrieve PSFs. Default is
            ``"empirical"``.
        psf_homog_filt : `str`, optional
            Filter to PSF-homogenize all bands to. If `None`, PSF
            homogenization is skipped. Default is ``"F444W"``.

        Returns
        -------
        `Catalogue`
            The catalogue loaded from the pipeline's reduced data.
        """
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
            mask_method=mask_method,
            psf_method=psf_method,
            psf_homog_filt=psf_homog_filt,
        )
        return cls.from_data(data, crops, load_gals=load_gals)

    @classmethod
    def from_data(
        cls,
        data: Data,
        crops: Optional[Union[Type[Selector], List[Type[Selector]]]] = None,
        load_gals: bool = True,
        **cat_creator_kwargs: Dict[str, Any],
    ) -> Catalogue:
        """Construct a `Catalogue` directly from a `Data` object.

        Builds a `Catalogue_Creator` via `Catalogue_Creator.from_data` and
        immediately calls it to load and return the `Catalogue`.

        Parameters
        ----------
        data : `Data`
            Dataset to build the catalogue from.
        crops : `Selector` or `list` of `Selector`, optional
            Selector(s) to crop the catalogue by when loading. Default is
            `None` (no cropping).
        load_gals : `bool`, optional
            Whether to load `Galaxy` objects into the returned catalogue.
            Default is `True`.
        **cat_creator_kwargs
            Additional keyword arguments forwarded to
            `Catalogue_Creator.from_data`.

        Returns
        -------
        `Catalogue`
            The catalogue loaded from `data`.
        """
        cat_creator = Catalogue_Creator.from_data(
            data,
            crops=crops,
            **cat_creator_kwargs,
        )
        return cat_creator(
            psfs=data.psfs,
            cropped=True,
            load_gals=load_gals,
        )

    def __repr__(self):
        """Return the official string representation of the Catalogue.

        Returns
        -------
        `str`
            Representation delegating to parent class.
        """
        return super().__repr__()

    def __str__(self) -> str:
        """Return a human-readable string representation of the Catalogue.

        Returns
        -------
        `str`
            String representation delegating to parent class.
        """
        return super().__str__()

    def load_stacked_band_data_snrs(
        self: Self,
        get_snr_colnames: Callable[
            [Multiple_Filter, u.Quantity], Dict[u.Quantity, str]
        ] = galfind_snr_labels,
        # filterset_arr: Union[Multiple_Filter, List[Multiple_Filter]],
        overwrite: bool = False,
    ) -> None:
        """Load stacked-band-detection-image SNRs into each galaxy's
        aperture photometry.

        Reads per-band, per-aperture-diameter SNR columns (named via
        `get_snr_colnames`) from the ``"phot"`` extension for each of
        `self.data.stacked_band_data_arr`, and stores them as a
        ``stacked_band_snrs`` attribute on each galaxy's `Photometry_obs`
        (one per aperture diameter). Does nothing (with a warning logged) if
        the catalogue is empty, has no `data` attribute, or `data` has no
        `stacked_band_data_arr`.

        Parameters
        ----------
        get_snr_colnames : `Callable`, optional
            Function returning the per-aperture-diameter SNR column names
            for a set of bands. Default is `galfind_snr_labels`.
        overwrite : `bool`, optional
            Whether to overwrite an existing ``stacked_band_snrs`` attribute
            on galaxies that already have one. Default is `False`.

        Returns
        -------
        `None`
        """
        if len(self) == 0:
            galfind_logger.warning(
                "Catalogue is empty! Cannot load stacked band data!"
            )
            return
        if not hasattr(self, "data"):
            galfind_logger.warning(
                "No data attribute to load stacked band data from!"
            )
            return
        if not hasattr(self.data, "stacked_band_data_arr"):
            galfind_logger.warning("No stacked band data to load!")
            return

        # if isinstance(filterset_arr, Multiple_Filter):
        #     filterset_arr = [filterset_arr]
        phot_tab = self.cat_creator.load_tab("phot", cropped=True)
        mask_tab = self.cat_creator.load_tab("mask", cropped=True)
        assert (
            len(phot_tab) == len(mask_tab) == len(self)
        ), galfind_logger.critical(
            f"{len(phot_tab)=} != {len(mask_tab)=} != {len(self)=}!"
        )
        snr_labels = get_snr_colnames(
            self.data.stacked_band_data_arr, self.aper_diams
        )
        snrs = {
            aper_diam: {
                band_data.filt_name: np.array(phot_tab[snr_labels_[i]]).astype(
                    float
                )
                for i, band_data in enumerate(self.data.stacked_band_data_arr)
            }
            for aper_diam, snr_labels_ in snr_labels.items()
        }
        # add snrs to galaxy aperture photometry objects
        galfind_logger.info(
            f"Loading {self.data.stacked_band_data_arr=} SNRs "
            f"into {repr(self)}!"
        )
        for i, gal in tqdm(
            enumerate(self),
            total=len(self),
            desc=f"Loading {self.data.stacked_band_data_arr} SNRs "
            f"into {repr(self)}",
            disable=galfind_logger.getEffectiveLevel() > logging.INFO,
        ):
            if not hasattr(gal, "stacked_band_snrs") or overwrite:
                for aper_diam in self.aper_diams:
                    if aper_diam in gal.aper_phot.keys():
                        setattr(
                            gal.aper_phot[aper_diam],
                            "stacked_band_snrs",
                            {
                                band_data.filt_name: snrs[aper_diam][
                                    band_data.filt_name
                                ][i]
                                for band_data in (
                                    self.data.stacked_band_data_arr
                                )
                            },
                        )
                    else:
                        galfind_logger.warning(
                            f"{repr(gal)} does not have photometry "
                            f"for {aper_diam=}! " + f"Cannot load "
                            f"{self.data.stacked_band_data_arr=} SNRs "
                            f"for this aperture diameter!"
                        )
            else:
                galfind_logger.debug(
                    f"{repr(gal)} already has 'stacked_band_snrs' "
                    f"attribute! Not overwriting!"
                )

    def save_phot_PDF_paths(self, PDF_paths, SED_fit_params):
        """Store the paths to saved SED-fitting redshift/property PDFs
        for this catalogue.

        Parameters
        ----------
        PDF_paths : `Any`
            PDF path(s) to store, keyed under the SED fitting run's label.
        SED_fit_params : `dict`
            SED fitting parameters, must contain a ``"code"`` key (an
            `SED_code` instance/subclass) used to derive the storage label
            via
            ``SED_fit_params["code"].label_from_SED_fit_params(
                SED_fit_params)``.
        """
        if "phot_PDF_paths" not in self.__dict__.keys():
            self.phot_PDF_paths = {}
        self.phot_PDF_paths[
            SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)
        ] = PDF_paths

    def save_phot_SED_paths(self, SED_paths, SED_fit_params):
        """Store the paths to saved best-fit SED files for this catalogue.

        Parameters
        ----------
        SED_paths : `Any`
            SED path(s) to store, keyed under the SED fitting run's label.
        SED_fit_params : `dict`
            SED fitting parameters, must contain a ``"code"`` key (an
            `SED_code` instance/subclass) used to derive the storage label
            via
            ``SED_fit_params["code"].label_from_SED_fit_params(
                SED_fit_params)``.
        """
        if "phot_SED_paths" not in self.__dict__.keys():
            self.phot_SED_paths = {}
        self.phot_SED_paths[
            SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)
        ] = SED_paths

    def update_SED_results(self, cat_SED_results, timed=True):
        """Update each galaxy's stored SED fitting results.

        Parameters
        ----------
        cat_SED_results : `list`
            Per-galaxy SED fitting result objects, in the same order as
            `self`. Must be the same length as `self`.
        timed : `bool`, optional
            Unused (accepted for a stable call signature); a progress bar is
            always shown unless logging is suppressed. Default is `True`.
        """
        assert len(cat_SED_results) == len(
            self
        )  # if this is not the case then instead should cross match
        # IDs between self and gal_SED_result
        galfind_logger.info("Updating SED results in galfind catalogue object")
        [
            gal.update_SED_results(gal_SED_result)
            for gal, gal_SED_result in tqdm(
                zip(self, cat_SED_results),
                desc="Updating galaxy SED results",
                total=len(self),
                disable=galfind_logger.getEffectiveLevel() > logging.INFO,
            )
        ]

    def update_SED_result_lowz_zmax_info(
        self, aper_diam, SED_result_key, zmax_info_arr
    ):
        """Update each galaxy's low-redshift-solution zmax information
        for a given SED fit.

        Parameters
        ----------
        aper_diam : `u.Quantity`
            Aperture diameter identifying which `Photometry_obs` the SED
            result belongs to.
        SED_result_key : `str`
            Key identifying which stored SED fitting result to update.
        zmax_info_arr : `list`
            Per-galaxy low-z zmax info, in the same order as `self`. Must be
            the same length as `self`.
        """
        assert len(zmax_info_arr) == len(self), galfind_logger.critical(
            "Length of zmax_info_dict must be the same as the catalogue!"
        )
        galfind_logger.info(
            "Updating low-z zmax info in galfind catalogue object"
        )
        [
            gal.update_SED_result_lowz_zmax_info(
                aper_diam, SED_result_key, zmax_info_arr[i]
            )
            for i, gal in tqdm(
                enumerate(self),
                desc="Updating galaxy low-z zmax info",
                total=len(self),
                disable=galfind_logger.getEffectiveLevel() > logging.INFO,
            )
        ]

    def match_available_spectra(
        self: Self,
        versions: List[str] = ["v1", "v2", "v3", "v4_2"],
        **match_kwargs: Dict[str, Any],
    ) -> Self:
        """Cross-match with available spectra from JWST DJA
        spectroscopic releases.

        Queries and cross-matches this photometric catalogue with spectroscopic
        observations from specified JWST DJA data release versions.

        Parameters
        ----------
        versions : `list` of `str`, optional
            DJA release versions to query. Default is ["v1", "v2", "v3",
            "v4_2"].
        **match_kwargs
            Additional keyword arguments passed to cross-matching functions.

        Returns
        -------
        `Catalogue`
            New catalogue containing only galaxies with matching spectroscopy.
        """
        # make catalogue consisting of spectra downloaded from the DJA
        for i, version in enumerate(versions):
            DJA_cat_ = Spectral_Catalogue.from_DJA(
                ra_range=self.ra_range,
                dec_range=self.dec_range,
                version=version,
                **match_kwargs,
            )
            if i == 0:
                DJA_cat = DJA_cat_
            else:
                DJA_cat += DJA_cat_
        # cross match this catalogue
        cross_matched_cat = self * DJA_cat
        # print(str(cross_matched_cat))
        return cross_matched_cat

        # # calculate aperture corrections if not already
        # # calculate and save dict of ext_src_corrs for each galaxy in self
        # galfind_logger.debug(
        #     "Photometry_obs.calc_ext_src_corrs takes 2min 20s for "
        #     "JOF with 16,000 galaxies. Fairly slow!"
        # )
        # [
        #     gal.phot.calc_ext_src_corrs(aper_corrs=aper_corrs)
        #     for gal in tqdm(
        #         self,
        #         desc=f"Calculating extended source corrections "
        #         f"for {self.survey} {self.version}",
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
    #     [aper_phot_.make_ext_src_corrs(gal_property, origin)
    #     for gal in self for aper_phot_ in gal.aper_phot.values()]
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
    #         for origin, property_names in
    #         unique_properties_origins_dict.items()
    #         for property_name in property_names
    #     ]

    def _append_property_to_tab(
        self,
        property_name: str,
        hdu: str = "OBJECTS",
        dtype: Type = object,
        overwrite: bool = False,
    ) -> None:
        """Append a computed galaxy property to a FITS catalogue table.

        Computes a property from all galaxies in the catalogue and joins it
        with
        the existing FITS table at the specified HDU extension.

        Parameters
        ----------
        property_name : `str`
            Name of the galaxy property to append (accessed via getattr).
        hdu : `str`, optional
            HDU extension to append to. One of "OBJECTS" or "SELECTION".
            Default is "OBJECTS".
        dtype : `type`, optional
            Data type for the appended column. Default is `object`.
        overwrite : `bool`, optional
            Whether to overwrite an existing column with the same name.
            Default is `False`.
        """
        from ..selection.Selector import Selector

        save_property_name = Selector.shorten_kwarg_colname(property_name)
        if hdu in ["OBJECTS", "SELECTION"]:
            ID_label = self.cat_creator.ID_label
        elif any(
            [
                True
                for code in funcs.all_subclasses(SED_code)
                if hdu.find(code.__name__) != -1
            ]
        ):
            raise NotImplementedError
        else:
            raise NotImplementedError
        # TODO: Need to attach self.open_cat to catalogue_creator.open_cat
        append_tab = self.open_cat(cropped=False, hdu=hdu)
        # append to .fits table only if not already
        if append_tab is not None:
            if save_property_name in append_tab.colnames:
                if overwrite:
                    # remove old column that already exists
                    append_tab.remove_column(save_property_name)
                else:
                    err_message = (
                        f"{save_property_name=} already appended to "
                        + f"{hdu=} .fits table, not overwriting!"
                    )
                    galfind_logger.debug(err_message)
                    return
        # return None
        galfind_logger.info(
            f"Appending {save_property_name=} to {hdu=} .fits table!"
        )
        # make new table with calculated properties
        gal_IDs = getattr(self, "ID")
        # TODO: get an array of properties to save (NOT an array in all cases!)
        gal_properties = np.array(getattr(self, property_name)).astype(dtype)
        assert len(gal_properties) == len(gal_IDs)
        # mask any NaN values
        # i = 0
        # property_type = None
        # while property_type is None:
        #     property_type = type(gal_properties[i])
        #     i += 1
        # gal_properties = MaskedColumn(gal_properties,
        # mask=np.isnan(gal_properties), dtype = property_type)
        new_tab = Table(
            {"ID_temp": gal_IDs, save_property_name: gal_properties},
            dtype=[int, type(gal_properties[0])],
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

    # def calc_new_property(self, func: Callable[..., float],
    # arg_names: Union[list, np.array]):
    #     pass

    def load_sextractor_Re(self) -> None:
        """Load SExtractor effective radius (Re) for all galaxies and bands.

        Extracts the FLUX_RADIUS parameter from SExtractor catalogues and
        converts to physical units (arcsec) accounting for pixel scale.

        Raises
        ------
        Exception
            If data object is not available for unit conversion.
        """
        if hasattr(self, "data"):
            # load Re from SExtractor
            pix_to_as_dict = {
                band_data.filt_name: band_data.pix_scale
                for band_data in self.data
            }
            self.load_band_properties_from_cat(
                "FLUX_RADIUS", "sex_Re", multiply_factor=pix_to_as_dict
            )
        else:
            err_message = (
                "Loading SExtractor Re from catalogue "
                + f"only works when hasattr({repr(self)}, data)!"
            )
            galfind_logger.critical(err_message)
            raise Exception(err_message)

    def load_sextractor_auto_mags(self) -> None:
        """Load SExtractor AUTO magnitudes for all galaxies and bands.

        Extracts MAG_AUTO (isophotal magnitudes) from SExtractor catalogues.
        """
        if hasattr(self, "data"):
            # load Re from SExtractor
            self.load_band_properties_from_cat(
                "MAG_AUTO", "sex_MAG_AUTO", multiply_factor=u.ABmag
            )
            for gal in self:
                for aper_diam in gal.aper_phot.keys():
                    gal.aper_phot[aper_diam].sex_MAG_AUTO = gal.sex_MAG_AUTO
        else:
            err_message = (
                "Loading SExtractor auto mags from catalogue "
                + f"only works when hasattr({repr(self)}, data)!"
            )
            galfind_logger.critical(err_message)
            raise Exception(err_message)

    def load_sextractor_auto_fluxes(
        self: Self, multiply_factor: Optional[Dict[str, float]] = None
    ) -> None:
        """Load SExtractor AUTO fluxes for all galaxies and bands.

        Extracts FLUX_AUTO from SExtractor catalogues, converting to Jansky
        using zero-points if data object is available.

        Parameters
        ----------
        multiply_factor : `dict` of `str`: `float` or `None`, optional
            Per-band multiplication factor for flux conversion. If `None`,
            computed from data object zero-points. Default is `None`.

        Raises
        ------
        Exception
            If neither data object nor multiply_factor is provided.
        """
        if hasattr(self, "data") or multiply_factor is not None:
            # load Re from SExtractor
            if multiply_factor is None:
                assert hasattr(self, "data"), galfind_logger.critical(
                    "Either provide multiply_factor or have self.data!"
                )
                multiply_factor = {
                    band_data.filt_name: funcs.flux_image_to_Jy(
                        1.0, band_data.ZP
                    )
                    for band_data in self.data
                }
            else:
                assert isinstance(
                    multiply_factor, dict
                ), galfind_logger.critical(
                    f"{type(multiply_factor)=} != dict!"
                )
                assert all(
                    [
                        key in self.filterset.filt_names
                        for key in multiply_factor.keys()
                    ]
                ), galfind_logger.critical(
                    f"{len(multiply_factor)=} != " + f"{len(self.filterset)=}!"
                )
            self.load_band_properties_from_cat(
                "FLUX_AUTO", "sex_FLUX_AUTO", multiply_factor=multiply_factor
            )
            for gal in self:
                for aper_diam in gal.aper_phot.keys():
                    gal.aper_phot[aper_diam].sex_FLUX_AUTO = gal.sex_FLUX_AUTO
        else:
            err_message = (
                "Loading SExtractor flux autos from catalogue "
                + f"only works when hasattr({repr(self)}, data) "
                f"or multiply_factor is not None!"
            )
            galfind_logger.critical(err_message)
            raise Exception(err_message)

    def load_sextractor_kron_radii(self) -> None:
        """Load SExtractor Kron radii and morphological parameters.

        Extracts Kron radius, A_IMAGE, B_IMAGE, and THETA_IMAGE parameters
        from SExtractor catalogues for morphological analysis.
        """
        if hasattr(self, "data"):
            # load Kron radius from SExtractor
            kron_radii = self.load_band_properties_from_cat(
                "KRON_RADIUS",
                "sex_KRON_RADIUS",
            )
            try:
                A_image_arr = self.load_band_properties_from_cat(
                    "A_IMAGE", "sex_A_IMAGE", update=False
                )
                [
                    setattr(
                        gal, "sex_A_IMAGE", A_image[self.data[0].filt_name]
                    )
                    for gal, A_image in zip(self, A_image_arr)
                ]
                A_image_as_arr = [
                    {
                        band_data.filt_name: kron_radius[band_data.filt_name]
                        * A_image[band_data.filt_name]
                        * band_data.pix_scale
                        for band_data in self.data
                    }
                    for kron_radius, A_image in zip(kron_radii, A_image_arr)
                ]
                [
                    gal.load_property(A_image_as, "sex_A_IMAGE_AS")
                    for gal, A_image_as in zip(self, A_image_as_arr)
                ]
                B_image_arr = self.load_band_properties_from_cat(
                    "B_IMAGE",
                    "sex_B_IMAGE",
                    update=False,
                )
                [
                    setattr(
                        gal, "sex_B_IMAGE", B_image[self.data[0].filt_name]
                    )
                    for gal, B_image in zip(self, B_image_arr)
                ]
                B_image_as_arr = [
                    {
                        band_data.filt_name: kron_radius[band_data.filt_name]
                        * B_image[band_data.filt_name]
                        * band_data.pix_scale
                        for band_data in self.data
                    }
                    for kron_radius, B_image in zip(kron_radii, B_image_arr)
                ]
                [
                    gal.load_property(B_image_as, "sex_B_IMAGE_AS")
                    for gal, B_image_as in zip(self, B_image_as_arr)
                ]
                theta_image_arr = self.load_band_properties_from_cat(
                    "THETA_IMAGE",
                    "sex_THETA_IMAGE",
                    multiply_factor=u.deg,
                    update=False,
                )
                [
                    setattr(
                        gal,
                        "sex_THETA_IMAGE",
                        theta_image[self.data[0].filt_name],
                    )
                    for gal, theta_image in zip(self, theta_image_arr)
                ]
            except Exception:
                pass

            for gal in self:
                for aper_diam in gal.aper_phot.keys():
                    for name in [
                        "KRON_RADIUS",
                        "A_IMAGE",
                        "B_IMAGE",
                        "THETA_IMAGE",
                        "A_IMAGE_AS",
                        "B_IMAGE_AS",
                    ]:
                        name = f"sex_{name}"
                        if hasattr(gal, name):
                            setattr(
                                gal.aper_phot[aper_diam],
                                name,
                                getattr(gal, name),
                            )
                        else:
                            galfind_logger.warning(f"{name} not in {gal.ID=}")

    def load_sextractor_ext_src_corrs(
        self: Self,
        multiply_factor: Optional[Dict[str, float]] = None,
        aper_corrs: Optional[Dict[str, float]] = None,
    ) -> None:
        """Load extended source corrections from SExtractor AUTO fluxes.

        Computes corrections to aperture photometry for extended sources
        by comparing AUTO flux estimates with aperture photometry.

        Parameters
        ----------
        multiply_factor : `dict` or `None`, optional
            Per-band flux conversion factors. Default is `None`.
        aper_corrs : `dict` or `None`, optional
            Per-aperture correction values. Default is `None`.
        """
        if multiply_factor is None:
            filt_names = None
        else:
            filt_names = list(multiply_factor.keys())
        self.load_sextractor_auto_fluxes(multiply_factor=multiply_factor)
        [
            gal.load_sextractor_ext_src_corrs(
                filt_names, aper_corrs=aper_corrs
            )
            for gal in tqdm(
                self,
                desc=f"Loading SExtractor extended source "
                f"corrections for {repr(self)}",
                total=len(self),
                disable=galfind_logger.getEffectiveLevel() > logging.INFO,
            )
        ]

        if len(self) == len(self.cat_creator._open_tab("ID")):
            galfind_logger.info(
                f"Saving SExtractor extended source corrections "
                f"to {self.cat_name}!"
            )
            # save to catalogue
            for filt in self.filterset:
                if filt_names is None or filt.filt_name in filt_names:
                    for aper_diam in self.aper_diams:
                        self._append_property_to_tab(
                            f"ext_src_corr_{aper_diam.to(u.arcsec).value:.2f}as_{filt.filt_name}",
                            "OBJECTS",
                            dtype=float,
                        )

    def load_sextractor_params(self) -> None:
        """Load all SExtractor parameters for all galaxies.

        Convenience method that loads effective radius, AUTO magnitudes/fluxes,
        Kron radii, and extended source corrections in sequence.
        """
        self.load_sextractor_auto_mags()
        self.load_sextractor_auto_fluxes()
        self.load_sextractor_kron_radii()
        self.load_sextractor_Re()
        self.load_sextractor_ext_src_corrs()

    def load_band_properties_from_cat(
        self: Self,
        cat_colname: str,
        save_name: str,
        multiply_factor: Union[dict, u.Quantity, u.Magnitude, None] = None,
        dest: str = "gal",
        update: bool = True,
    ) -> Optional[List[Dict[str, Union[u.Quantity, u.Magnitude, u.Dex]]]]:
        """Load a per-band property from the catalogue for all galaxies.

        Reads a set of per-band columns from the catalogue FITS file,
        optionally applies unit conversions, and stores on each galaxy object.

        Parameters
        ----------
        cat_colname : `str`
            Base name of the catalogue column (e.g., "FLUX_RADIUS").
            Per-band columns are named as ``{colname}_{filt_name}``.
        save_name : `str`
            Name to store the property as on each galaxy object.
        multiply_factor : `dict`, `Quantity`, `Magnitude`, or `None`, optional
            Per-band multiplication factor(s) for unit conversion. Can be a
            single value (applied to all bands) or a dictionary mapping filter
            names to factors. Default is `None` (no conversion).
        dest : `str`, optional
            Destination to store the property on: either `"gal"` (galaxy
            object)
            or `"phot_obs"` (photometry object). Default is `"gal"`.
        update : `bool`, optional
            Whether to update the galaxy objects with the loaded property.
            Default is `True`.

        Returns
        -------
        `list` of `dict`
            List of dictionaries mapping band names to per-band property values
            for each galaxy.
        """
        if len(self) == 0:
            galfind_logger.warning(
                f"No galaxies in {self.cat_name}, skipping "
                f"loading {cat_colname=}"
            )
            return None
        assert dest in ["gal", "phot_obs"]
        if dest == "gal":
            has_attr = hasattr(self[0], save_name)
        else:  # dest == "phot_obs"
            has_attr = hasattr(self[0].phot, save_name)
        if not has_attr:
            galfind_logger.info(
                f"Loading {cat_colname=} from {self.cat_path} "
                f"saved as {save_name}!"
            )
            # load the same property from every available band
            # open catalogue with astropy
            fits_cat = self.open_cat(cropped=True)
            if multiply_factor is None:
                multiply_factor = {
                    filt.filt_name: 1.0 * u.dimensionless_unscaled
                    for filt in self.filterset
                    if f"{cat_colname}_{filt.filt_name}" in fits_cat.colnames
                }
            elif not isinstance(multiply_factor, dict):
                multiply_factor = {
                    filt.filt_name: multiply_factor
                    for filt in self.filterset
                    if f"{cat_colname}_{filt.filt_name}" in fits_cat.colnames
                }
            # load in speed can be improved here!
            cat_band_properties = {
                filt.filt_name: np.array(
                    fits_cat[f"{cat_colname}_{filt.filt_name}"]
                )
                * multiply_factor[filt.filt_name]
                for filt in self.filterset
                if f"{cat_colname}_{filt.filt_name}" in fits_cat.colnames
                and filt.filt_name in multiply_factor.keys()
            }
            if len(cat_band_properties) == 0:
                err_message = (
                    f"Could not load {cat_colname=} from {self.cat_path} "
                    + f"as no '{cat_colname}_band' exists for band "
                    f"in {self.instrument.filt_names=}!"
                )
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
                    f"Loaded {cat_colname} from {self.cat_path} "
                    f"saved as {save_name} for "
                    f"{cat_band_properties[0].keys()=}"
                )
            return cat_band_properties

    def load_property_from_cat(
        self,
        cat_colname: str,
        save_name: str,
        multiply_factor: Union[u.Quantity, u.Magnitude] = 1.0
        * u.dimensionless_unscaled,
    ) -> None:
        """Load a single property column from the catalogue for all galaxies.

        Reads a catalogue column and stores it on each galaxy object,
        optionally applying a unit conversion factor.

        Parameters
        ----------
        cat_colname : `str`
            Name of the catalogue column to load.
        save_name : `str`
            Name to store the property as on each galaxy object.
        multiply_factor : `Quantity` or `Magnitude`, optional
            Multiplication factor for unit conversion. Default is
            dimensionless.
        dest : `str`, optional
            Destination: ``"gal"`` to store on galaxy, or ``"phot_obs"`` to
            store on photometry object. Default is ``"gal"``.
        """
        dest = "gal"
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
                    f"Loaded {cat_colname=} from {self.cat_path} "
                    f"saved as {save_name}!"
                )
            else:
                galfind_logger.info(
                    f"{cat_colname=} does not exist in "
                    f"{self.cat_path}, skipping!"
                )

    def load_sex_flux_mag_autos(self) -> None:
        """Load SExtractor AUTO fluxes and magnitudes into photometry objects.

        Extracts FLUX_AUTO and MAG_AUTO from SExtractor catalogues and
        stores them on each galaxy's photometry object, converting fluxes
        to Jansky.
        """
        # sex_filt_names = [filt_name for filt_name, cat_type in
        # self.data.sex_cat_types.items() if "SExtractor" in cat_type]
        flux_im_to_Jy_conv = {
            filt_name: funcs.flux_image_to_Jy(1.0, self.data.im_zps[filt_name])
            for filt_name in self.instrument.filt_names
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

    def load_fixz_SED_results(
        self: Self,
        aper_diam: u.Quantity,
        z_arr: Union[List[float], NDArray[float]],
        z_label: str = "z",
    ) -> None:
        """Load pre-computed fixed-redshift SED fitting results.

        Parameters
        ----------
        aper_diam : `astropy.units.Quantity`
            Aperture diameter that results correspond to.
        z_arr : `list` or `ndarray` of `float`
            Redshift value for each galaxy. Must have same length as catalogue.
        z_label : `str`, optional
            Label for the redshift parameter. Default is ``"z"``.

        Raises
        ------
        AssertionError
            If redshift array length doesn't match catalogue length.
        """
        assert len(z_arr) == len(self), galfind_logger.critical(
            f"{len(z_arr)} != {len(self)}! {z_arr=} and {self=}"
        )

        [
            gal.load_fixz_SED_result(aper_diam, z_value, z_label)
            for gal, z_value in zip(self, z_arr)
        ]

    def make_cutouts(
        self: Self,
        cutout_size: Union[u.Quantity, dict] = 0.96 * u.arcsec,
        native: bool = False,
    ) -> List[Dict[str, Type[Band_Cutout_Base]]]:
        """Generate multi-band cutout images for all galaxies in the catalogue.

        Parameters
        ----------
        cutout_size : `astropy.units.Quantity` or `dict`, optional
            Size of cutouts in angular units. Can be a single value (applied
            to all bands) or a dictionary mapping band names to sizes.
            Default is 0.96 arcsec.
        native : `bool`, optional
            Whether to use native (non-PSF-homogenized) data. Default is
            `False`.

        Returns
        -------
        `list` of `dict`
            List of cutout dictionaries, one per galaxy, each mapping band
            names to `Band_Cutout_Base` objects.
        """
        if native:
            assert hasattr(self.data, "native"), galfind_logger.critical(
                "Native data not available"
            )
            data = self.data.native
        else:
            data = self.data
        return [gal.make_cutouts(data, cutout_size) for gal in self]

    def make_band_cutouts(
        self: Self,
        filt: Union[str, Filter],
        cutout_size: u.Quantity = 0.96 * u.arcsec,
        native: bool = False,
        overwrite: bool = False,
    ) -> List[Band_Cutout]:
        """Generate single-band cutout images for all galaxies.

        Parameters
        ----------
        filt : `str` or `Filter`
            Filter/band to extract cutouts for.
        cutout_size : `astropy.units.Quantity`, optional
            Size of cutouts. Default is 0.96 arcsec.
        native : `bool`, optional
            Whether to use native data. Default is `False`.
        overwrite : `bool`, optional
            Whether to regenerate cutouts. Default is `False`.

        Returns
        -------
        `list` of `Band_Cutout`
            Cutout objects, one per galaxy, for the specified band.
        """
        if native:
            assert hasattr(self.data, "native"), galfind_logger.critical(
                "Native data not available"
            )
            data = self.data.native
        else:
            data = self.data
        band_data = data[filt.filt_name]
        return [
            gal.make_band_cutout(
                band_data,
                cutout_size,
                overwrite=overwrite,
            )
            for gal in tqdm(
                self,
                desc=f"Making {filt.filt_name} band cutouts",
                total=len(self),
                disable=galfind_logger.getEffectiveLevel() > logging.INFO,
            )
        ]

    def plot_cutouts(
        self: Self,
        cutout_size: u.Quantity = 0.96 * u.arcsec,
        # save_name: Optional[str] = None,
        plot_kwargs: Dict[str, Any] = {},
        crop_name: Optional[str] = None,
        collate_dir: Optional[str] = None,
        overwrite: bool = False,
        overwrite_sample: bool = True,
    ) -> None:
        """Plot multi-band cutout images for all galaxies.

        Parameters
        ----------
        cutout_size : `astropy.units.Quantity`, optional
            Size of cutouts. Default is 0.96 arcsec.
        plot_kwargs : `dict`, optional
            Keyword arguments for plotting. Default is empty.
        crop_name : `str` or `None`, optional
            Crop/selection name for organizing plots. Default is `None`.
        collate_dir : `str` or `None`, optional
            Directory to collate plots into. If `None`, uses default path
            based on crop name. Default is `None`.
        overwrite : `bool`, optional
            Whether to overwrite existing cutouts. Default is `False`.
        overwrite_sample : `bool`, optional
            Whether to overwrite sample collation. Default is `True`.
        """
        out_paths = np.full(len(self), None)
        for i, gal in enumerate(self):
            cutouts = gal.make_cutouts(
                self.data, cutout_size, overwrite=overwrite
            )
            cutouts.plot(close_fig=True, **plot_kwargs)
            out_paths[i] = cutouts._get_save_path()
        out_paths.astype(str)
        # collate plots for the specified galaxies
        if collate_dir is None:
            if crop_name is None:
                crop_name = self.crop_name
            collate_dir = (
                "/".join(
                    out_paths[0].replace("/png", "").split("/")[:-2]
                    + [crop_name]
                )
                + out_paths[0].replace("/png", "").split("/")[-2]
            )
        self._collate_plots(out_paths, collate_dir, overwrite=overwrite_sample)

    def stack_gals(
        self, cutout_size: u.Quantity = 0.96 * u.arcsec
    ) -> Multiple_Band_Cutout:
        """Create a stacked multi-band cutout image from all galaxies
        in the catalogue.

        Generates aligned cutouts of the specified size for all galaxies,
        stacks them band-by-band, and caches the result for reuse.

        Parameters
        ----------
        cutout_size : `astropy.units.Quantity`, optional
            Size of individual cutouts. Default is 0.96 arcsec.

        Returns
        -------
        `Multiple_Band_Cutout`
            Stacked cutout image with one layer per band.
        """
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
        """Plot a visualization of the stacked galaxy cutouts.

        Creates and displays a stacked multi-band cutout image from all
        galaxies, with optional saving to file.

        Parameters
        ----------
        cutout_size : `astropy.units.Quantity`, optional
            Size of individual cutouts. Default is 0.96 arcsec.
        save_name : `str` or `None`, optional
            File path to save the plot to. If `None`, plot is not saved.
            Default is `None`.
        """
        stacked_cutouts = self.stack_gals(cutout_size=cutout_size)
        stacked_cutouts.plot(save_name=save_name)

    def make_RGBs(
        self: Self,
        cutout_size: u.Quantity = 0.96 * u.arcsec,
        rgb_bands: Dict[str, List[str]] = {
            "B": ["F090W"],
            "G": ["F200W"],
            "R": ["F444W"],
        },
    ) -> Multiple_RGB:
        """Generate RGB composite images for all galaxies in the catalogue.

        Creates cutout images from specified bands, combines them into RGB
        composites, and caches the results for reuse.

        Parameters
        ----------
        cutout_size : `astropy.units.Quantity`, optional
            Size of individual galaxy cutouts. Default is 0.96 arcsec.
        rgb_bands : `dict` of `str` to `list` of `str`, optional
            Mapping of RGB channels ("B", "G", "R") to lists of band names
            to use for each channel. Default uses F090W, F200W, F444W.

        Returns
        -------
        `Multiple_RGB`
            RGB composite images for all galaxies.
        """
        if not hasattr(self, "RGBs"):
            self.RGBs = {}
        cutout_size_str = f"{cutout_size.to(u.arcsec).value:.2f}as"
        if cutout_size_str not in self.RGBs.keys():
            self.RGBs[cutout_size_str] = {}
        rgb_key = ",".join(
            f"{colour}={'+'.join(self.get_colour_filt_names[colour])}"
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
        """Plot RGB composite images for all galaxies.

        Generates and displays RGB composite images from specified filter
        bands,
        optionally saving to file.

        Parameters
        ----------
        cutout_size : `astropy.units.Quantity`, optional
            Size of individual galaxy cutouts. Default is 0.96 arcsec.
        rgb_bands : `dict`, optional
            Mapping of RGB channels to band names. Default uses standard bands.
        method : `str`, optional
            RGB combination method (e.g., "trilogy"). Default is "trilogy".
        save_name : `str` or `None`, optional
            File path to save the plot to. Default is `None`.
        """
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
        """Create a stacked RGB composite image from all galaxies in
        the catalogue.

        Generates RGB composites for each galaxy, stacks them, and caches
        the result for reuse.

        Parameters
        ----------
        cutout_size : `astropy.units.Quantity`, optional
            Size of individual galaxy cutouts. Default is 0.96 arcsec.
        rgb_bands : `dict`, optional
            Mapping of RGB channels to band names. Default uses standard bands.

        Returns
        -------
        `Stacked_RGB`
            Stacked RGB composite image.
        """
        if not hasattr(self, "stacked_RGB"):
            self.stacked_RGB = {}
        cutout_size_str = f"{cutout_size.to(u.arcsec).value:.2f}as"
        if cutout_size_str not in self.stacked_RGB.keys():
            self.stacked_RGB[cutout_size_str] = Stacked_RGB.from_cat(
                self, cutout_size=cutout_size, rgb_bands=rgb_bands
            )
        return self.stacked_RGB[cutout_size_str]

    def plot_phot_diagnostics(
        self: Self,
        aper_diam: u.Quantity,
        SED_arr: List[SED_code],
        zPDF_arr: List[SED_code],
        plot_lowz: bool = True,
        wav_unit: u.Unit = u.um,
        flux_unit: u.Unit = u.ABmag,
        log_fluxes: bool = False,
        cutout_size: u.Quantity = 0.96 * u.arcsec,
        crop_name: Optional[str] = None,
        collate_dir: Optional[str] = None,
        imshow_kwargs: Dict[str, Any] = {},
        norm_kwargs: Dict[str, Any] = {},
        aper_kwargs: Dict[str, Any] = {},
        kron_kwargs: Dict[str, Any] = {},
        cutout_label_kwargs: Dict[str, Any] = {},
        sed_plot_kwargs: Dict[str, Any] = {},
        cutout_gridspec_kwargs: Dict[str, Any] = {},
        legend_kwargs: Dict[str, Any] = {"loc": "best"},
        errorbar_kwargs: Dict[str, Any] = {},
        SNR_label_kwargs: Dict[str, Any] = {},
        phot_axes_labels_kwargs: Dict[str, Any] = {},
        phot_axes_ticks_kwargs: Dict[str, Any] = {},
        PDF_axes_ticks_kwargs: Dict[str, Any] = {},
        PDF_axes_labels_kwargs: Dict[str, Any] = {},
        PDF_axes_title_kwargs: Dict[str, Any] = {},
        title_kwargs: Dict[str, Any] = {},
        n_cutout_rows: int = 2,
        overwrite: bool = False,
        split_by_instr: bool = True,
        split_by_instr_cmap: str = "plasma",
        SED_cmap: Optional[str] = None,
        # fig_axs: Optional[Tuple[plt.Figure, NDArray[plt.Axes]]] = None,
    ):
        """Generate and save photometry diagnostic plots for all galaxies.

        Creates multi-panel diagnostic figures showing cutouts, photometric
        SEDs,
        and redshift PDFs for each galaxy, saving them to a results directory.

        Parameters
        ----------
        aper_diam : `astropy.units.Quantity`
            Aperture diameter to plot diagnostics for.
        SED_arr : `list` of `SED_code`
            List of SED fitting code instances for each SED model type.
        zPDF_arr : `list` of `SED_code`
            List of redshift PDF code instances for each PDF type.
        plot_lowz : `bool`, optional
            Whether to plot low-redshift solutions. Default is `True`.
        wav_unit : `astropy.units.Unit`, optional
            Wavelength unit for plots. Default is micrometers.
        flux_unit : `astropy.units.Unit`, optional
            Flux unit for plots. Default is AB magnitudes.
        log_fluxes : `bool`, optional
            Whether to use logarithmic flux scale. Default is `False`.
        cutout_size : `astropy.units.Quantity`, optional
            Size of galaxy cutout images. Default is 0.96 arcsec.
        crop_name : `str` or `None`, optional
            Name of crop/selection for organizing outputs. Default is `None`.
        collate_dir : `str` or `None`, optional
            Directory to collate symlinks to diagnostic plots. Default is
            `None`.
        imshow_kwargs : `dict`, optional
            Keyword arguments for imshow. Default is empty.
        norm_kwargs : `dict`, optional
            Normalization keyword arguments. Default is empty.
        aper_kwargs : `dict`, optional
            Aperture marking keyword arguments. Default is empty.
        kron_kwargs : `dict`, optional
            Kron radius marking keyword arguments. Default is empty.
        cutout_label_kwargs : `dict`, optional
            Label keyword arguments for cutouts. Default is empty.
        sed_plot_kwargs : `dict`, optional
            SED plot keyword arguments. Default is empty.
        cutout_gridspec_kwargs : `dict`, optional
            Gridspec keyword arguments. Default is empty.
        legend_kwargs : `dict`, optional
            Legend keyword arguments. Default is {"loc": "best"}.
        errorbar_kwargs : `dict`, optional
            Error bar keyword arguments. Default is empty.
        SNR_label_kwargs : `dict`, optional
            SNR label keyword arguments. Default is empty.
        phot_axes_labels_kwargs : `dict`, optional
            Photometry axes labels keyword arguments. Default is empty.
        phot_axes_ticks_kwargs : `dict`, optional
            Photometry axes ticks keyword arguments. Default is empty.
        PDF_axes_ticks_kwargs : `dict`, optional
            PDF axes ticks keyword arguments. Default is empty.
        PDF_axes_labels_kwargs : `dict`, optional
            PDF axes labels keyword arguments. Default is empty.
        PDF_axes_title_kwargs : `dict`, optional
            PDF axes title keyword arguments. Default is empty.
        title_kwargs : `dict`, optional
            Figure title keyword arguments. Default is empty.
        n_cutout_rows : `int`, optional
            Number of rows for cutout grid. Default is 2.
        overwrite : `bool`, optional
            Whether to overwrite existing plots. Default is `False`.
        split_by_instr : `bool`, optional
            Whether to color-code by instrument. Default is `True`.
        split_by_instr_cmap : `str`, optional
            Colormap for instrument colors. Default is "plasma".
        SED_cmap : `str` or `None`, optional
            Colormap for SED plotting. Default is `None`.
        """
        self.load_sextractor_params()

        # if fig_axs is None:
        #     fig, fig_axs = figs.make_phot_diagnostic_fig(len(self.data))

        # loop over galaxies and make photometry diagnostic plots for each one
        out_paths = [
            gal.plot_phot_diagnostic(
                self.data,
                # fig = fig,
                # ax = fig_axs,
                aper_diam=aper_diam,
                SED_arr=SED_arr,
                zPDF_arr=zPDF_arr,
                plot_lowz=plot_lowz,
                n_cutout_rows=n_cutout_rows,
                wav_unit=wav_unit,
                flux_unit=flux_unit,
                log_fluxes=log_fluxes,
                cutout_size=cutout_size,
                imshow_kwargs=imshow_kwargs,
                norm_kwargs=norm_kwargs,
                aper_kwargs=aper_kwargs,
                kron_kwargs=kron_kwargs,
                cutout_label_kwargs=cutout_label_kwargs,
                sed_plot_kwargs=sed_plot_kwargs,
                legend_kwargs=legend_kwargs,
                errorbar_kwargs=errorbar_kwargs,
                SNR_label_kwargs=SNR_label_kwargs,
                phot_axes_labels_kwargs=phot_axes_labels_kwargs,
                phot_axes_ticks_kwargs=phot_axes_ticks_kwargs,
                PDF_axes_ticks_kwargs=PDF_axes_ticks_kwargs,
                PDF_axes_labels_kwargs=PDF_axes_labels_kwargs,
                PDF_axes_title_kwargs=PDF_axes_title_kwargs,
                title_kwargs=title_kwargs,
                overwrite=overwrite,
                save=True,
                show=False,
                incl_nodata_cutouts=True,
                split_by_instr=split_by_instr,
                split_by_instr_cmap=split_by_instr_cmap,
                SED_cmap=SED_cmap,
                cutout_gridspec_kwargs=cutout_gridspec_kwargs,
            )
            for gal in tqdm(
                self,
                total=len(self),
                desc="Plotting photometry diagnostic plots",
                disable=galfind_logger.getEffectiveLevel() > logging.INFO,
            )
        ]
        if collate_dir is None:
            if crop_name is None:
                crop_name = self.crop_name
            collate_dir = (
                f"{config['Other']['PLOT_DIR']}/{self.version}/"
                + f"{self.filterset.instrument_name}/{self.survey}/SED_plots/"
                + f"{aper_diam.to(u.arcsec).value:.2f}as/{crop_name}/"
            )
        self._collate_plots(out_paths, collate_dir)

    def _collate_plots(
        self: Self,
        out_paths: List[str],
        collate_dir: str,
        overwrite: bool = True,
    ):
        """Create symlinks to plots in a collation directory for quick access.

        Organizes individual galaxy plots by creating symbolic links in a
        collation directory, useful for browsing selected subsets of results.

        Parameters
        ----------
        out_paths : `list` of `str`
            Paths to individual galaxy plot files, one per galaxy.
        collate_dir : `str`
            Directory to create symlinks in.
        overwrite : `bool`, optional
            Whether to remove and recreate existing symlinks. Default is
            `True`.
        """
        # make a folder to store symlinked photometric diagnostic
        # plots for selected galaxies
        if self.crops != []:
            if overwrite:
                # remove existing symlinks in folder
                symlink_paths = glob.glob(f"{collate_dir}/*.png")
                for symlink_path in symlink_paths:
                    os.remove(symlink_path)
            # create symlink to selection folder for diagnostic plots
            for gal, out_path in zip(self, out_paths):
                if out_path is not None:
                    selection_path = f"{collate_dir}/{str(gal.ID)}.png"
                    funcs.make_dirs(selection_path)
                    try:
                        os.symlink(out_path, selection_path)
                    except FileExistsError:  # replace existing file
                        if Path(out_path).is_file():
                            os.remove(selection_path)
                            os.symlink(out_path, selection_path)

    # Number Density Function (e.g. UVLF and mass functions) methods

    def calc_Vmax(
        self: Self,
        z_bin: List[float],
        aper_diam: u.Quantity,
        SED_fit_code: SED_code,
        z_step: float = 0.01,
        unmasked_area: Union[
            str, List[str], u.Quantity, Type[Mask_Selector]
        ] = "selection",
        Vmax_method: str = "uniform_depth",
        n_jobs: int = 1,
    ) -> NDArray[float]:
        """Calculate maximum co-moving volume (
            Vmax) for galaxies in a redshift bin.

        Computes Vmax corrections for luminosity/mass function calculations,
        accounting for observational limits and survey sensitivity as a
        function of redshift.

        Parameters
        ----------
        z_bin : `list` of `float`
            Redshift bin edges [z_min, z_max] to calculate Vmax over.
        aper_diam : `astropy.units.Quantity`
            Aperture diameter for photometry.
        SED_fit_code : `SED_code`
            SED fitting code used for redshift determination.
        z_step : `float`, optional
            Redshift step size for integration. Default is 0.01.
        unmasked_area : `str`, `list` of `str`, `Quantity`, or
            `Mask_Selector`, optional
            Definition of survey area with data. Can be "selection", a region
            selector, or an area value. Default is "selection".
        Vmax_method : `str`, optional
            Method for Vmax calculation ("uniform_depth" or other). Default is
            "uniform_depth".
        n_jobs : `int`, optional
            Number of parallel jobs for calculation. Default is 1.

        Returns
        -------
        `np.ndarray` of `float`
            Vmax values for each galaxy in the catalogue.
        """
        assert hasattr(self, "data"), galfind_logger.critical(
            f"{repr(self)} does not have data loaded!"
        )
        return self._calc_Vmax(
            self.data,
            z_bin=z_bin,
            aper_diam=aper_diam,
            SED_fit_code=SED_fit_code,
            z_step=z_step,
            unmasked_area=unmasked_area,
            Vmax_method=Vmax_method,
            n_jobs=n_jobs,
        )

    # class Simulated_Catalogue(Catalogue_Base):
    #     # TODO: Store Simulated_Galaxy rather than Galaxy objects

    def scatter(
        self: Self,
        aper_diam: u.Quantity,
        mode: str = "n_nearest",
        depth_region: str = "all",
        min_flux_pc_err: float = 10.0,
        update_errs: bool = True,
    ):
        """Add random scatter to photometry based on local depth measurements.

        Applies Gaussian scatter to galaxy fluxes at a given aperture diameter,
        scaled by photometric errors derived from the survey's local depths.
        Useful for simulating measurement uncertainties.

        Parameters
        ----------
        aper_diam : `astropy.units.Quantity`
            Aperture diameter to scatter fluxes for.
        mode : `str`, optional
            Method for depth calculation ("n_nearest" or other). Default is
            "n_nearest".
        depth_region : `str`, optional
            Region to calculate depth statistics from ("all" or region name).
            Default is "all".
        min_flux_pc_err : `float`, optional
            Minimum flux percentage error floor. Default is 10.0.
        update_errs : `bool`, optional
            Whether to recompute flux errors after scattering. Default is
            `True`.
        """
        assert all(aper_diam in gal.aper_phot.keys() for gal in self)
        # load galaxy depths from the average depths of the field
        if hasattr(self, "data"):
            self._update_depths_from_data(aper_diam, mode, depth_region)
        # calculate photometric errors from these newly inserted depths
        if update_errs:
            self._update_errs_from_depths(
                aper_diam, apply_min_flux_pc_err=False
            )
        # scatter each set of fluxes once by the calculated errors
        [
            gal.aper_phot[aper_diam].scatter_fluxes(update=True)
            for gal in tqdm(
                self,
                desc="Scattering catalogue fluxes",
                total=len(self),
                disable=galfind_logger.getEffectiveLevel() > logging.INFO,
            )
        ]
        if update_errs:
            self._update_errs_from_depths(aper_diam)

    def _update_depths_from_data(
        self: Self,
        aper_diam: u.Quantity,
        mode: str = "n_nearest",
        depth_region: str = "all",
    ) -> NoReturn:
        """Update all galaxy depths to values from the survey data object.

        Queries depth measurements from the loaded data object and assigns
        them to each galaxy's photometry.

        Parameters
        ----------
        aper_diam : `astropy.units.Quantity`
            Aperture diameter to update depths for.
        mode : `str`, optional
            Method for depth calculation. Default is "n_nearest".
        depth_region : `str`, optional
            Region to calculate depth statistics from. Default is "all".
        """
        assert hasattr(self, "data"), galfind_logger.critical(
            f"{self.cat_name} does not have data loaded!"
        )
        self.load_depths(aper_diam, mode)
        assert all(
            [hasattr(band_data, "med_depth") for band_data in self.data]
        )
        assert all(
            [
                aper_diam in band_data.med_depth.keys()
                for band_data in self.data
            ]
        )
        assert all(
            [
                depth_region in band_data.med_depth[aper_diam].keys()
                for band_data in self.data
            ]
        )
        depths = (
            np.array(
                [
                    band_data.med_depth[aper_diam][depth_region]
                    for band_data in self.data
                ]
            )
            * u.ABmag
        )
        [
            setattr(gal.aper_phot[aper_diam], "depths", depths)
            for gal in tqdm(
                self,
                desc="Updating catalogue depths",
                total=len(self),
                disable=galfind_logger.getEffectiveLevel() > logging.INFO,
            )
        ]

    def _update_errs_from_depths(
        self: Self,
        aper_diam: u.Quantity,
        default_min_flux_pc_err: float = 10.0,
        apply_min_flux_pc_err: bool = True,
    ) -> NoReturn:
        """Recompute all galaxy flux errors from their depths.

        Updates each galaxy's photometric errors based on the local depths,
        applying an optional minimum percentage error floor.

        Parameters
        ----------
        aper_diam : `astropy.units.Quantity`
            Aperture diameter to update errors for.
        default_min_flux_pc_err : `float`, optional
            Minimum percentage error if not found in configuration. Default
            is 10.0.
        apply_min_flux_pc_err : `bool`, optional
            Whether to apply a minimum percentage error floor. Default is
            `True`.
        """
        if apply_min_flux_pc_err:
            if "min_flux_pc_err" in self.cat_creator.load_phot_kwargs.keys():
                min_flux_pc_err = self.cat_creator.load_phot_kwargs[
                    "min_flux_pc_err"
                ]
            else:
                galfind_logger.warning(
                    f"No 'min_flux_pc_err' in "
                    f"{self.cat_creator.load_phot_kwargs.keys()=}."
                    + f" Using {default_min_flux_pc_err=}!"
                )
                min_flux_pc_err = default_min_flux_pc_err
        else:
            min_flux_pc_err = 0.0

        [
            gal.aper_phot[aper_diam]._update_errs_from_depths(min_flux_pc_err)
            for gal in tqdm(
                self,
                desc="Updating catalogue errors from average depths",
                total=len(self),
                disable=galfind_logger.getEffectiveLevel() > logging.INFO,
            )
        ]

    def load_depths(
        self: Self,
        aper_diam: u.Quantity,
        mode: str,
        region_selector: Optional[Region_Selector] = None,
        invert_region: bool = False,
        update_ecsv: bool = False,
    ) -> None:
        """Load and compute local depths for a given aperture diameter.

        Queries depth measurements from the survey data, optionally filtering
        by a region selector. Can save computed depths to an ECSV file.

        Parameters
        ----------
        aper_diam : `astropy.units.Quantity`
            Aperture diameter to load depths for.
        mode : `str`
            Method for depth calculation (e.g., "n_nearest").
        region_selector : `Region_Selector` or `None`, optional
            Region to compute depths over. If `None`, uses all data.
            Default is `None`.
        invert_region : `bool`, optional
            Whether to invert the region selection. Default is `False`.
        update_ecsv : `bool`, optional
            Whether to save computed depths to ECSV file. Default is `False`.
        """
        if region_selector is None:
            self.data._load_depths(
                aper_diam,
                mode,
            )
        else:
            region_name = (
                region_selector.name
                if not invert_region
                else region_selector.fail_name
            )
            galfind_logger.info(
                f"Loading {aper_diam} {mode} {region_name} depths!"
            )
            gal_depths = {
                filt_name: [
                    gal.aper_phot[aper_diam]
                    .depths[
                        np.where(
                            np.array(
                                gal.aper_phot[aper_diam].filterset.filt_names
                            )
                            == filt_name
                        )[0][0]
                    ]
                    .value
                    for gal in self
                    if filt_name
                    in gal.aper_phot[aper_diam].filterset.filt_names
                    and (
                        (
                            gal.selection_flags[region_selector.name]
                            and not invert_region
                        )
                        or (
                            not gal.selection_flags[region_selector.name]
                            and invert_region
                        )
                    )
                ]
                for filt_name in self.data.filterset.filt_names
            }
            med_depths = {
                filt_name: np.nanmedian(gal_band_depths)
                for filt_name, gal_band_depths in gal_depths.items()
            }
            depths_16 = {
                filt_name: np.nanpercentile(gal_band_depths, 16.0)
                for filt_name, gal_band_depths in gal_depths.items()
            }
            depths_84 = {
                filt_name: np.nanpercentile(gal_band_depths, 84.0)
                for filt_name, gal_band_depths in gal_depths.items()
            }
            mean_depths = {
                filt_name: np.nanmean(gal_band_depths)
                for filt_name, gal_band_depths in gal_depths.items()
            }

            # pass to data objects for storage
            for band_data in self.data:
                band_data._update_depths(
                    aper_diam,
                    med_depths[band_data.filt_name],
                    mean_depths[band_data.filt_name],
                    region_selector.name
                    if not invert_region
                    else region_selector.fail_name,
                )

        if update_ecsv:
            if region_selector is None:
                gal_depths = {
                    filt_name: [
                        gal.aper_phot[aper_diam]
                        .depths[
                            np.where(
                                np.array(
                                    gal.aper_phot[
                                        aper_diam
                                    ].filterset.filt_names
                                )
                                == filt_name
                            )[0][0]
                        ]
                        .value
                        for gal in self
                        if filt_name
                        in gal.aper_phot[aper_diam].filterset.filt_names
                    ]
                    for filt_name in self.data.filterset.filt_names
                }
                med_depths = {
                    filt_name: np.nanmedian(gal_band_depths)
                    for filt_name, gal_band_depths in gal_depths.items()
                }
                depths_16 = {
                    filt_name: np.nanpercentile(gal_band_depths, 16.0)
                    for filt_name, gal_band_depths in gal_depths.items()
                }
                depths_84 = {
                    filt_name: np.nanpercentile(gal_band_depths, 84.0)
                    for filt_name, gal_band_depths in gal_depths.items()
                }
                mean_depths = {
                    filt_name: np.nanmean(gal_band_depths)
                    for filt_name, gal_band_depths in gal_depths.items()
                }

            tab = self.open_cat(cropped=False, hdu="OBJECTS")
            if len(self) == len(tab):  # and region_selector is not None:
                # save depths to fits
                depth_dir = Depths.get_depth_tab_dir(self.data)
                gal_reg_depth_path = (
                    f"{depth_dir}/reg_depths/{self.survey}_"
                    f"{aper_diam.to(u.arcsec).value:.2f}as.ecsv"
                )
                append_data = {
                    "reg_name": [],
                    "filt_name": [],
                    "depth_50": [],
                    "depth_16": [],
                    "depth_84": [],
                    "mean_depth": [],
                }
                append_data_types = {
                    "reg_name": str,
                    "filt_name": str,
                    "depth_50": float,
                    "depth_16": float,
                    "depth_84": float,
                    "mean_depth": float,
                }
                if not Path(gal_reg_depth_path).is_file():
                    orig_tab = Table(
                        append_data,
                        dtype=[append_data_types[key] for key in append_data],
                    )
                else:
                    orig_tab = Table.read(
                        gal_reg_depth_path, format="ascii.ecsv"
                    )
                if region_selector is not None:
                    reg_name = (
                        region_selector.name
                        if not invert_region
                        else region_selector.fail_name
                    )
                else:
                    reg_name = "all"
                for band_data in self.data:
                    filt_name = band_data.filt_name
                    if not any(
                        row["reg_name"] == reg_name
                        and row["filt_name"] == filt_name
                        for row in orig_tab
                    ):
                        append_data["reg_name"].append(reg_name)
                        append_data["filt_name"].append(filt_name)
                        append_data["depth_50"].append(med_depths[filt_name])
                        append_data["depth_16"].append(depths_16[filt_name])
                        append_data["depth_84"].append(depths_84[filt_name])
                        append_data["mean_depth"].append(
                            mean_depths[filt_name]
                        )
                # combine append data with original table
                if len(append_data) > 0:
                    append_tab = Table(
                        append_data,
                        dtype=[append_data_types[key] for key in append_data],
                    )
                    out_tab = vstack([orig_tab, append_tab])
                    funcs.make_dirs(gal_reg_depth_path)
                    out_tab.write(
                        gal_reg_depth_path, format="ascii.ecsv", overwrite=True
                    )
                    funcs.change_file_permissions(gal_reg_depth_path)
                    galfind_logger.info(
                        f"Saved {reg_name} depths to {gal_reg_depth_path}!"
                    )

    def make_readme(
        self: Self,
        update: bool = False,
        hdu_divider: str = "------------------",
    ) -> None:
        """Generate or update a README file documenting the catalogue.

        Creates a human-readable README file describing the catalogue contents,
        column definitions, and data provenance for each HDU extension.

        Parameters
        ----------
        update : `bool`, optional
            Whether to update an existing README. Default is `False`.
        hdu_divider : `str`, optional
            String to use as divider between HDU sections. Default is 18
            dashes.
        """
        out_path = f"{self.fits_path.replace('.fits', '')}_README.txt"
        if not Path(out_path).is_file() or update:
            out_str = ""
            if Path(out_path).is_file():
                with open(out_path, "r") as f:
                    out_str = f.read()
                # split by hdu_divider
                out_str = out_str.split(hdu_divider)

            readme_info = self._readme_info_from_fits_tab(self.fits_path)
        for origin, columns in readme_info.items():
            pass

    @staticmethod
    def _readme_info_from_fits_tab(
        fits_path: str,
    ) -> Dict[str, Dict[str, str]]:
        """Extract column metadata from a FITS catalogue for README generation.

        Parses a multi-extension FITS catalogue file, extracting header and
        column information from each HDU to generate documentation.

        Parameters
        ----------
        fits_path : `str`
            Path to the FITS catalogue file.

        Returns
        -------
        `dict` of `str` to `dict` of `str` to `str`
            Nested dictionary mapping HDU names to dictionaries of column
            names and their descriptions.
        """
        hdul = fits.open(fits_path)
        hdu_names = [hdu.name for hdu in hdul]
        readme_info = {}
        for i, hdu_name in enumerate(hdu_names):
            hdul[i]
            if hdu_name == "OBJECTS":
                # check for SExtractor columns
                pass
        return readme_info

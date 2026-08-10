"""General utility functions and helpers for GALFIND operations.

Provides unit conversion utilities, coordinates transformation, statistical
calculations, and miscellaneous helper functions used throughout galfind.
"""

from __future__ import annotations

import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sep
import re
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs.utils import skycoord_to_pixel
from scipy.stats import chi2
import inspect
import os
import contextlib
import joblib
from numba import njit
from numpy.typing import NDArray
from typing import Union, List, Tuple, TYPE_CHECKING, Optional, Any, Dict
if TYPE_CHECKING:
    from .Data import Band_Data_Base, Band_Data, Stacked_Band_Data
    from . import Selector, Filter, Multiple_Filter, Mask_Selector, Photometry_rest, Catalogue
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import astropy_cosmo, galfind_logger, config

# fluxes and magnitudes

def convert_wav_units(wavs, units):
    """Convert a wavelength quantity to the requested units.

    Parameters
    ----------
    wavs : `astropy.units.Quantity`
        Wavelength value(s) to convert.
    units : `astropy.units.Unit`
        Target wavelength units.

    Returns
    -------
    `astropy.units.Quantity`
        ``wavs`` converted to ``units`` (returned unchanged if already in ``units``).
    """
    if units == wavs.unit:
        return wavs
    else:
        return wavs.to(units)


def convert_mag_units(wavs, mags, units):
    """Convert a magnitude/flux quantity between AB magnitude, f_nu, and f_lambda representations.

    Uses the astropy spectral density equivalency (with the associated
    wavelengths) whenever converting between f_lambda and AB magnitude/f_nu
    representations.

    Parameters
    ----------
    wavs : `astropy.units.Quantity`
        Wavelength(s) associated with ``mags``, used for the spectral density equivalency.
    mags : `astropy.units.Quantity` or `astropy.units.Magnitude`
        Magnitude or flux density value(s) to convert.
    units : `astropy.units.Unit`
        Target units: either `astropy.units.ABmag`, or a unit with physical type
        "spectral flux density"/"ABmag/spectral flux density" (f_nu), or
        "power density/spectral flux density wav" (f_lambda).

    Returns
    -------
    `astropy.units.Quantity` or `astropy.units.Magnitude`
        ``mags`` converted to ``units``.

    Raises
    ------
    Exception
        If ``units`` is not `astropy.units.ABmag` and does not have a recognised
        spectral flux density physical type.
    """
    if units == mags.unit:
        pass
    elif units == u.ABmag:
        if u.get_physical_type(mags.unit) in [
            "ABmag/spectral flux density",
            "spectral flux density",
        ]:  # f_ν -> derivative of u.Jy
            mags = mags.to(u.ABmag)
        elif (
            u.get_physical_type(mags.unit)
            == "power density/spectral flux density wav"
        ):  # f_λ -> derivative of u.erg / (u.s * (u.cm ** 2) * u.AA)
            mags = mags.to(u.ABmag, equivalencies=u.spectral_density(wavs))
    elif u.get_physical_type(units) in [
        "ABmag/spectral flux density",
        "spectral flux density",
    ]:  # f_ν -> derivative of u.Jy
        if mags.unit == u.ABmag:
            mags = mags.to(units)
        elif u.get_physical_type(mags.unit) in [
            "power density/spectral flux density wav",
            "ABmag/spectral flux density",
        ]:
            mags = mags.to(units, equivalencies=u.spectral_density(wavs))
        else:
            mags = mags.to(units)
    elif (
        u.get_physical_type(units) == "power density/spectral flux density wav"
    ):  # f_λ -> derivative of u.erg / (u.s * (u.cm ** 2) * u.AA):
        if mags.unit == u.ABmag:
            mags = mags.to(units, equivalencies=u.spectral_density(wavs))
        elif (
            u.get_physical_type(mags.unit)
            == "power density/spectral flux density wav"
        ):
            mags = mags.to(units)
        else:  # different phyiscal type that isn't ABmag
            mags = mags.to(units, equivalencies=u.spectral_density(wavs))
    else:
        raise (
            Exception(
                "Units must be either ABmag or have physical units of 'spectral flux density' or 'power density/spectral flux density wav'!"
            )
        )
    return mags


def convert_mag_err_units(wavs, mags, mag_errs, units):
    """Convert asymmetric magnitude/flux errors to the requested units.

    Parameters
    ----------
    wavs : `astropy.units.Quantity`
        Wavelength(s) associated with ``mags``, used for spectral density equivalencies.
    mags : `astropy.units.Quantity` or `astropy.units.Magnitude`
        Central magnitude/flux value(s).
    mag_errs : `list` of `astropy.units.Quantity`
        Two-element ``[lower, upper]`` 1-sigma errors on ``mags``, in the same units as ``mags``.
    units : `astropy.units.Unit`
        Target units to convert the errors into.

    Returns
    -------
    `list` of `astropy.units.Quantity`
        Two-element ``[lower, upper]`` errors on ``mags``, converted to ``units``.

    Raises
    ------
    AssertionError
        If ``mags`` and ``mag_errs`` do not share the same units, or ``mag_errs``
        is not a two-element sequence of array-likes each with length > 1.
    Exception
        If ``units`` is not `astropy.units.ABmag` and does not have a recognised
        spectral flux density physical type.
    """
    assert (
        mags.unit == mag_errs[0].unit == mag_errs[1].unit
    ), galfind_logger.critical(
        f"Could not convert mag error units as mags.unit = {mags.unit} != mag_errs.unit = ({mag_errs[0].unit}, {mag_errs[1].unit})"
    )
    assert len(mag_errs) == 2 and len(mag_errs[0]) > 1 and len(mag_errs[1]) > 1, \
        galfind_logger.critical(
            f"Could not convert mag error units as mag_errs = {mag_errs} with {len(mag_errs)=} != 2"
            f" and {len(mag_errs[0])=}, {len(mag_errs[1])=}"
        )

    if units == mags.unit:
        return mag_errs
    else:
        mags_new_units = convert_mag_units(wavs, mags, units)
        mags_u1_new_units = convert_mag_units(wavs, mags + mag_errs[1], units)
        mags_l1_new_units = convert_mag_units(wavs, mags - mag_errs[0], units)

        # work out whether the order needs swapping
        if units == u.ABmag:
            swap_order = True
        elif u.get_physical_type(units) in [
            "ABmag/spectral flux density",
            "spectral flux density",
        ]:  # f_ν -> derivative of u.Jy
            if mags.unit == u.ABmag:
                swap_order = True
            else:
                swap_order = False
        elif (
            u.get_physical_type(units)
            == "power density/spectral flux density wav"
        ):  # f_λ -> derivative of u.erg / (u.s * (u.cm ** 2) * u.AA):
            if mags.unit == u.ABmag:
                swap_order = True
            else:  # f_ν -> derivative of u.Jy
                swap_order = False
        else:
            raise (
                Exception(
                    "Units must be either ABmag or have physical units of 'spectral flux density' or 'power density/spectral flux density wav'!"
                )
            )
        if swap_order:  # swap order of l1 / u1
            return [
                mags_new_units - mags_u1_new_units,
                mags_l1_new_units - mags_new_units,
            ]
        else:
            return [
                mags_new_units - mags_l1_new_units,
                mags_u1_new_units - mags_new_units,
            ]


def log_scale_fluxes(fluxes):  # removes unit
    """Take the base-10 logarithm of flux values, discarding their unit.

    Parameters
    ----------
    fluxes : `astropy.units.Quantity`
        Flux value(s).

    Returns
    -------
    `numpy.ndarray`
        ``log10(fluxes.value)``, unitless.
    """
    log_flux_unit = fluxes.unit
    log_fluxes = np.log10(fluxes.value)
    return log_fluxes


def log_scale_flux_errors(fluxes, flux_errs):  # removes unit
    """Propagate asymmetric flux errors into log10 space.

    Parameters
    ----------
    fluxes : `astropy.units.Quantity`
        Central flux value(s).
    flux_errs : `list` of `astropy.units.Quantity`
        Two-element ``[lower, upper]`` 1-sigma flux errors, in the same units as ``fluxes``.

    Returns
    -------
    `list` of `numpy.ndarray`
        Two-element ``[lower, upper]`` errors on ``log10(fluxes)``.

    Raises
    ------
    AssertionError
        If ``flux_errs`` does not have length 2, or its units do not match ``fluxes``.
    """
    assert len(flux_errs) == 2, galfind_logger.warning(
        f"{flux_errs=} with {len(flux_errs)=} != 2"
    )
    assert (
        fluxes.unit == flux_errs[0].unit == flux_errs[1].unit
    ), galfind_logger.warning(
        f"{fluxes.unit =} != flux_errs.unit = ({flux_errs[0].unit, flux_errs[1].unit})"
    )
    log_flux_l1 = log_scale_fluxes(fluxes) - log_scale_fluxes(
        fluxes - flux_errs[0]
    )
    log_flux_u1 = log_scale_fluxes(fluxes + flux_errs[1]) - log_scale_fluxes(
        fluxes
    )
    return [log_flux_l1, log_flux_u1]


def calc_flux_from_ra_dec(ra, dec, im_data, wcs, r, unit="deg"):
    """Compute the aperture flux at a sky position from image data using SEP.

    Parameters
    ----------
    ra : `float` or array-like
        Right ascension of the aperture centre(s).
    dec : `float` or array-like
        Declination of the aperture centre(s).
    im_data : `numpy.ndarray`
        2D image array to sum flux from.
    wcs : `astropy.wcs.WCS`
        WCS solution used to convert the sky position(s) to pixel coordinates.
    r : `float`
        Aperture radius in pixels.
    unit : `str`, optional
        Unit of ``ra``/``dec``, as accepted by `astropy.coordinates.SkyCoord`. Default is "deg".

    Returns
    -------
    `numpy.ndarray`
        Summed flux within the aperture(s), in image units.
    """
    x_pix, y_pix = skycoord_to_pixel(SkyCoord(ra, dec, unit=unit), wcs)
    flux, fluxerr, flag = sep.sum_circle(im_data, x_pix, y_pix, r)
    return flux  # image units


def calc_1sigma_flux(
    depth: Union[float, u.Magnitude],
    zero_point: float,
) -> float:
    """Convert a 5-sigma depth to the corresponding 1-sigma flux.

    Parameters
    ----------
    depth : `float` or `astropy.units.Magnitude`
        5-sigma limiting magnitude depth.
    zero_point : `float`
        Magnitude zero point.

    Returns
    -------
    `float`
        1-sigma flux, in image units consistent with ``zero_point``.
    """
    if isinstance(depth, u.Magnitude):
        depth = depth.value
    flux_1sigma = (10 ** ((depth - zero_point) / -2.5)) / 5
    return flux_1sigma  # image units


def n_sigma_detection(
    depth,
    mag,
    zero_point,
):  # mag here is non aperture corrected
    """Compute the detection significance (in units of sigma) of a magnitude relative to a depth.

    Parameters
    ----------
    depth : `float` or `astropy.units.Magnitude`
        5-sigma limiting magnitude depth.
    mag : `float`
        Non-aperture-corrected magnitude of the source.
    zero_point : `float`
        Magnitude zero point.

    Returns
    -------
    `float`
        Ratio of the source flux to the 1-sigma flux (i.e. detection significance in sigma).
    """
    flux_1sigma = calc_1sigma_flux(depth, zero_point)
    flux = 10 ** ((mag - zero_point) / -2.5)
    return flux / flux_1sigma


def flux_to_mag(flux, zero_point):
    """Convert flux to magnitude given a zero point.

    Parameters
    ----------
    flux : `float` or `astropy.units.Quantity`
        Flux value(s); if a `Quantity` is given, its unit is stripped.
    zero_point : `float`
        Magnitude zero point.

    Returns
    -------
    `float`
        Magnitude corresponding to ``flux``.
    """
    try:
        flux = flux.value
    except:
        pass
    mag = -2.5 * np.log10(flux) + zero_point
    return mag


def mag_to_flux(mag, zero_point):
    """Convert magnitude to flux given a zero point.

    Parameters
    ----------
    mag : `float`
        Magnitude value(s).
    zero_point : `float`
        Magnitude zero point.

    Returns
    -------
    `float`
        Flux corresponding to ``mag``, in units consistent with ``zero_point``.
    """
    flux = 10 ** ((mag - zero_point) / -2.5)
    return flux


def flux_to_mag_ratio(flux_ratio):
    """Convert a flux ratio to a magnitude difference.

    Parameters
    ----------
    flux_ratio : `float`
        Ratio of two fluxes.

    Returns
    -------
    `float`
        Corresponding magnitude difference.
    """
    mag_ratio = -2.5 * np.log10(flux_ratio)
    return mag_ratio


def mag_to_flux_ratio(mag_ratio):
    """Convert a magnitude difference to a flux ratio.

    Parameters
    ----------
    mag_ratio : `float`
        Magnitude difference.

    Returns
    -------
    `float`
        Corresponding flux ratio.
    """
    flux_ratio = 10 ** (mag_ratio / -2.5)
    return flux_ratio


def flux_pc_to_mag_err(flux_pc_err):
    """Convert a fractional flux error to a magnitude error.

    Parameters
    ----------
    flux_pc_err : `float`
        Fractional error on flux (i.e. ``flux_err / flux``).

    Returns
    -------
    `float`
        Corresponding magnitude error.
    """
    mag_err = (
        2.5 * flux_pc_err / (np.log(10))
    )  # divide by 100 here to convert into percentage?
    return mag_err


def flux_image_to_Jy(fluxes, zero_points):
    """Convert flux(es) in image (zero-point) units to Jy.

    Parameters
    ----------
    fluxes : `float`, `list`, or `numpy.ndarray`
        Flux value(s) in image units.
    zero_points : `float` or array-like
        Magnitude zero point(s) corresponding to ``fluxes``.

    Returns
    -------
    `astropy.units.Quantity`
        Flux(es) converted to Jy.
    """
    # convert flux from image units to Jy
    if isinstance(fluxes, (list, np.ndarray,)):
        return (
            np.array(
                [
                    flux * (10 ** ((zero_points - (u.Jy).to(u.ABmag)) / -2.5))
                    for flux in fluxes
                ]
            )
            * u.Jy
        )
    else:
        return np.array(fluxes * (10 ** ((zero_points - (u.Jy).to(u.ABmag)) / -2.5))) * u.Jy


def five_to_n_sigma_mag(
    five_sigma_depth: Union[int, float, u.Magnitude],
    n: Union[int, float],
):
    """Convert a 5-sigma limiting magnitude to the n-sigma limiting magnitude.

    Parameters
    ----------
    five_sigma_depth : `int`, `float`, or `astropy.units.Magnitude`
        5-sigma limiting magnitude.
    n : `int` or `float`
        Target significance level in sigma; must be > 0.

    Returns
    -------
    `float`
        n-sigma limiting magnitude.

    Raises
    ------
    AssertionError
        If ``n`` is not greater than 0.
    """
    assert n > 0, galfind_logger.critical(f"{n=} must be > 0")
    if isinstance(five_sigma_depth, u.Magnitude):
        five_sigma_depth = five_sigma_depth.value
    n_sigma_mag = -2.5 * np.log10(n / 5) + five_sigma_depth
    # flux_sigma = (10 ** ((five_sigma_depth - zero_point) / -2.5)) / 5
    # n_sigma_mag = -2.5 * np.log10(flux_sigma * n) + zero_point
    return n_sigma_mag


def flux_err_to_loc_depth(flux_err, zero_point):
    """Convert a 1-sigma flux error to a local 5-sigma depth.

    Parameters
    ----------
    flux_err : `float`
        1-sigma flux error.
    zero_point : `float`
        Magnitude zero point.

    Returns
    -------
    `float`
        5-sigma limiting magnitude (local depth).
    """
    return -2.5 * np.log10(flux_err * 5) + zero_point


def loc_depth_to_flux_err(loc_depth, zero_point):
    """Convert a local 5-sigma depth to a 1-sigma flux error.

    Parameters
    ----------
    loc_depth : `float`
        5-sigma limiting magnitude (local depth).
    zero_point : `float`
        Magnitude zero point.

    Returns
    -------
    `float`
        1-sigma flux error.
    """
    return (10 ** ((loc_depth - zero_point) / -2.5)) / 5


# now in Photometry class!
# def flux_image_to_lambda(wav, flux, zero_point):
#     flux = flux_image_to_Jy(flux, zero_point)
#     flux_lambda = flux_to_lambda(wav, flux)
#     return flux_lambda # observed frame


def flux_Jy_to_lambda(
    flux_Jy, wav
):  # must already have associated astropy units
    """Convert a flux density from f_nu to f_lambda.

    Parameters
    ----------
    flux_Jy : `astropy.units.Quantity`
        Flux density in f_nu units (e.g. Jy).
    wav : `astropy.units.Quantity`
        Wavelength associated with ``flux_Jy``.

    Returns
    -------
    `astropy.units.Quantity`
        Flux density in f_lambda units (erg / s / cm^2 / Angstrom).
    """
    return (flux_Jy * const.c / (wav**2)).to(
        u.erg / (u.s * (u.cm**2) * u.Angstrom)
    )


def flux_lambda_to_Jy(flux_lambda, wav):
    """Convert a flux density from f_lambda to f_nu.

    Parameters
    ----------
    flux_lambda : `astropy.units.Quantity`
        Flux density in f_lambda units.
    wav : `astropy.units.Quantity`
        Wavelength associated with ``flux_lambda``.

    Returns
    -------
    `astropy.units.Quantity`
        Flux density in Jy.
    """
    return (flux_lambda * (wav**2) / const.c).to(u.Jy)


def lum_nu_to_lum_lam(lum_nu, wav):
    """Convert a specific luminosity from L_nu to L_lambda.

    Parameters
    ----------
    lum_nu : `astropy.units.Quantity`
        Specific luminosity per unit frequency.
    wav : `astropy.units.Quantity`
        Wavelength associated with ``lum_nu``.

    Returns
    -------
    `astropy.units.Quantity`
        Specific luminosity per unit wavelength (units not automatically simplified).
    """
    return lum_nu * const.c / (wav**2)


def lum_lam_to_lum_nu(lum_wav, wav):
    """Convert a specific luminosity from L_lambda to L_nu.

    Parameters
    ----------
    lum_wav : `astropy.units.Quantity`
        Specific luminosity per unit wavelength.
    wav : `astropy.units.Quantity`
        Wavelength associated with ``lum_wav``.

    Returns
    -------
    `astropy.units.Quantity`
        Specific luminosity per unit frequency (units not automatically simplified).
    """
    return lum_wav * (wav**2) / const.c


def wav_obs_to_rest(wav_obs, z):
    """Convert an observed-frame wavelength to the rest frame.

    Parameters
    ----------
    wav_obs : `astropy.units.Quantity` or `float`
        Observed-frame wavelength.
    z : `float`
        Redshift.

    Returns
    -------
    same type as ``wav_obs``
        Rest-frame wavelength, ``wav_obs / (1 + z)``.
    """
    wav_rest = wav_obs / (1 + z)
    return wav_rest


def wav_rest_to_obs(wav_rest, z):
    """Convert a rest-frame wavelength to the observed frame.

    Parameters
    ----------
    wav_rest : `astropy.units.Quantity` or `float`
        Rest-frame wavelength.
    z : `float`
        Redshift.

    Returns
    -------
    same type as ``wav_rest``
        Observed-frame wavelength, ``wav_rest * (1 + z)``.
    """
    wav_obs = wav_rest * (1 + z)
    return wav_obs


def flux_lambda_obs_to_rest(flux_lambda_obs, z):
    """Convert an observed-frame f_lambda flux density to the rest frame.

    Applies the ``(1 + z) ** 2`` k-correction factor appropriate for f_lambda.

    Parameters
    ----------
    flux_lambda_obs : array-like
        Observed-frame f_lambda flux density values.
    z : `float`
        Redshift.

    Returns
    -------
    array-like
        Rest-frame f_lambda flux density.
    """
    flux_lambda_rest = flux_lambda_obs * (
        (1 + np.full(len(flux_lambda_obs), z)) ** 2
    )
    return flux_lambda_rest


def luminosity_to_flux(lum, wavs, z, cosmo=astropy_cosmo, out_units=u.Jy):
    """Convert an intrinsic (rest-frame) luminosity to an observed-frame flux.

    Parameters
    ----------
    lum : `astropy.units.Quantity`
        Rest-frame (intrinsic) specific luminosity, either L_lambda (physical
        type "yank") or L_nu (physical type "energy/torque/work").
    wavs : `astropy.units.Quantity`
        Rest-frame wavelength(s) associated with ``lum``.
    z : `float`
        Redshift.
    cosmo : `astropy.cosmology.FLRW`, optional
        Cosmology used to compute the luminosity distance. Default is `astropy_cosmo`.
    out_units : `astropy.units.Unit`, optional
        Desired output flux units. Default is `astropy.units.Jy`.

    Returns
    -------
    `astropy.units.Quantity`
        Observed-frame flux density in ``out_units``.

    Raises
    ------
    Exception
        If ``lum``'s physical type is L_lambda or L_nu but ``out_units`` is not
        a recognised flux physical type.
    """

    # calculate luminosity distance
    lum_distance = cosmo.luminosity_distance(z)
    # sort out the units
    if (
        u.get_physical_type(lum.unit) == "yank"
    ):  # i.e. L_λ, Lsun / AA or equivalent
        if u.get_physical_type(out_units) in [
            "ABmag/spectral flux density",
            "spectral flux density",
        ]:  # f_ν
            lum = lum_lam_to_lum_nu(lum, wavs)
        elif (
            u.get_physical_type(out_units)
            == "power density/spectral flux density wav"
        ):  # f_λ
            pass
        else:
            raise (Exception(""))
    elif (
        u.get_physical_type(lum.unit) == "energy/torque/work"
    ):  # i.e L_ν, Lsun / Hz or equivalent
        if u.get_physical_type(out_units) in [
            "ABmag/spectral flux density",
            "spectral flux density",
        ]:  # f_ν
            pass
        elif (
            u.get_physical_type(out_units)
            == "power density/spectral flux density wav"
        ):  # f_λ
            lum = lum_nu_to_lum_lam(lum, wavs)
        else:
            raise (Exception(""))
    return (lum * (1. + z) / (4 * np.pi * lum_distance ** 2)).to(out_units)


def flux_to_luminosity(
    flux,
    wavs,
    z,
    cosmo = astropy_cosmo,
    out_units = u.erg / (u.s * u.Hz),
):
    """Convert an observed-frame flux to an intrinsic (rest-frame) luminosity.

    Parameters
    ----------
    flux : `astropy.units.Quantity`
        Observed-frame flux density (`astropy.units.ABmag` or a spectral flux
        density unit).
    wavs : `astropy.units.Quantity`
        Wavelength(s) associated with ``flux``.
    z : `float`
        Redshift.
    cosmo : `astropy.cosmology.FLRW`, optional
        Cosmology used to compute the luminosity distance. Default is `astropy_cosmo`.
    out_units : `astropy.units.Unit`, optional
        Desired output luminosity units. Default is ``erg / (s * Hz)``.

    Returns
    -------
    `astropy.units.Quantity`
        Rest-frame intrinsic luminosity in ``out_units``.
    """

    # sort out the units
    if flux.unit == u.ABmag:
        # convert to f_ν
        flux = flux.to(u.Jy)
    if u.get_physical_type(flux.unit) in [
        "ABmag/spectral flux density",
        "spectral flux density",
    ]:  # f_ν
        if (
            u.get_physical_type(out_units) == "yank"
        ):  # i.e. L_λ, Lsun / AA or equivalent
            # convert f_ν -> f_λ
            flux = convert_mag_units(
                wavs, flux, u.erg / (u.s * u.AA * u.cm**2)
            )
        elif (
            u.get_physical_type(out_units) == "energy/torque/work"
        ):  # i.e L_ν, Lsun / Hz or equivalent
            pass
        else:
            galfind_logger.critical(
                f"{out_units=} not in ['yank', 'energy/torque/work']"
            )
    elif (
        u.get_physical_type(flux.unit)
        == "power density/spectral flux density wav"
    ):  # f_λ
        if (
            u.get_physical_type(out_units) == "yank"
        ):  # i.e. L_λ, Lsun / AA or equivalent
            pass
        elif (
            u.get_physical_type(out_units) == "energy/torque/work"
        ):  # i.e L_ν, Lsun / Hz or equivalent
            # convert f_λ -> f_ν
            flux = convert_mag_units(wavs, flux, u.Jy)
        else:
            galfind_logger.critical(
                f"{out_units=} not in ['yank', 'energy/torque/work']"
            )
    else:
        galfind_logger.critical(
            f"{flux.unit=} not in ['spectral flux density', 'power density/spectral flux density wav']"
        )
    # calculate luminosity distance
    lum_distance = cosmo.luminosity_distance(z)
    return (4 * np.pi * flux * lum_distance ** 2 / (1. + z)).to(out_units)


def dust_correct(lum, dust_mag):
    """Apply a per-element dust attenuation correction to luminosity values.

    Parameters
    ----------
    lum : `astropy.units.Quantity`
        Luminosity value(s) (array-like).
    dust_mag : `astropy.units.Quantity`
        Dust attenuation in magnitudes for each element of ``lum``; elements
        ``<= 0`` are left uncorrected.

    Returns
    -------
    `astropy.units.Quantity`
        Dust-corrected luminosity values.
    """
    return [
        lum_i * (10 ** (dust_mag_i / 2.5)) if dust_mag_i > 0.0 else lum_i
        for lum_i, dust_mag_i in zip(lum.value, dust_mag.value)
    ] * lum.unit

SFR_conversions = {
    "MD14": 1.15e-28 * (u.solMass / u.yr) / (u.erg / (u.s * u.Hz))
}

fesc_from_beta_conversions = {
    "Chisholm22": lambda beta: np.random.normal(1.3, 0.6, len(beta))
        * 10 ** (-4.0 - np.random.normal(1.22, 0.1, len(beta)) * beta)
}

# unit labelling

unit_labels_dict = {
    u.AA: r"$\mathrm{\AA}$",
    u.um: r"$\mu\mathrm{m}$",
    u.erg
    / (
        u.s * u.AA * u.cm**2
    ): r"$\mathrm{erg s}^{-1}\mathrm{AA}^{-1}\mathrm{cm}^{-2}$",
    u.Jy: r"$\mathrm{Jy}$",
    u.nJy: r"$\mathrm{nJy}$",
    u.uJy: r"$\mathrm{\mu Jy}$",
    u.ABmag: r"$m_{\rm AB}$",
    u.Hz / u.erg: r"$\mathrm{Hz}\mathrm{erg}^{-1}$",
}

property_name_to_label = {
    "z": r"Redshift, $z$",
    "M_UV": r"$M_{\mathrm{UV}}$",
    "xi_ion_caseB_rest": r"$\xi_{\mathrm{ion,0}}~/~\mathrm{Hz}~\mathrm{erg}^{-1}$",
}


def label_log(label):
    """Wrap a LaTeX label string in a base-10 logarithm expression.

    Parameters
    ----------
    label : `str`
        LaTeX label to wrap, e.g. an axis label.

    Returns
    -------
    `str`
        ``label`` wrapped as ``$\\log_{10}(...)$``.
    """
    return r"$\log_{10}($" + label + r"$)$"


def label_wavelengths(unit, is_log_scaled, frame):
    """Build a LaTeX axis label for a wavelength quantity.

    Parameters
    ----------
    unit : `astropy.units.Unit`
        Wavelength unit; must be a key of ``unit_labels_dict``.
    is_log_scaled : `bool`
        Whether to wrap the label in a log10 expression.
    frame : `str`
        Reference-frame subscript, one of ``""``, ``"rest"``, or ``"obs"``.

    Returns
    -------
    `str`
        LaTeX-formatted wavelength axis label.

    Raises
    ------
    AssertionError
        If ``frame`` is not one of ``""``, ``"rest"``, ``"obs"``.
    """
    assert frame in ["", "rest", "obs"]
    wavelength_label = r"$\lambda_{%s}~/~$" % frame
    wavelength_label += unit_labels_dict[unit]
    if is_log_scaled:
        return label_log(wavelength_label)
    else:
        return wavelength_label


def label_fluxes(unit, is_log_scaled):
    """Build a LaTeX axis label for a flux density quantity.

    Parameters
    ----------
    unit : `astropy.units.Unit`
        Flux unit; must be a key of ``unit_labels_dict``.
    is_log_scaled : `bool`
        Whether to wrap the label in a log10 expression. Must be `False` when ``unit`` is `astropy.units.ABmag`.

    Returns
    -------
    `str`
        LaTeX-formatted flux axis label (f_nu or f_lambda notation, or the AB magnitude label).

    Raises
    ------
    AssertionError
        If ``unit`` is not a recognised key of ``unit_labels_dict``, or if
        ``unit`` is `astropy.units.ABmag` and ``is_log_scaled`` is `True`.
    """
    assert unit in unit_labels_dict.keys()
    if unit == u.ABmag:
        assert not is_log_scaled
        return unit_labels_dict[unit]
    elif u.get_physical_type(unit) in [
        "ABmag/spectral flux density",
        "spectral flux density",
    ]:
        flux_label = r"$f_{\nu}$"
    elif (
        u.get_physical_type(unit) == "power density/spectral flux density wav"
    ):
        flux_label = r"$f_{\lambda}$"
    else:
        galfind_logger.critical(f"{unit=} not valid!")
    flux_label += r"$~/~$" + unit_labels_dict[unit]
    if is_log_scaled:
        return label_log(flux_label)
    else:
        return flux_label


# properties that are by default logged
logged_properties = ["stellar_mass", "formed_mass", "ssfr", "ssfr_10myr"]

# extended source corrections
ext_src_label = "_ext_src_corr"
ext_src_properties = ["Lrest", "Lobs", "m1500", "M1500", "SFRrest", "SFRobs"]

# Calzetti 1994 filters
lower_Calzetti_filt = [
    1268.0,
    1309.0,
    1342.0,
    1407.0,
    1562.0,
    1677.0,
    1760.0,
    1866.0,
    1930.0,
    2400.0,
]
upper_Calzetti_filt = [
    1284.0,
    1316.0,
    1371.0,
    1515.0,
    1583.0,
    1740.0,
    1833.0,
    1890.0,
    1950.0,
    2580.0,
]

# mass IMF conversion
mass_IMF_factor = {}

# General number density function tools

default_lims = {
    "M1500": [-24.0, -16.0],
    "M_UV": [-24.0, -16.0],
    "M1500_[1250,3000]AA": [-24.0, -16.0],
    "M1500_[1250,3000]AA_extsrc": [-24.0, -16.0],
    "M1500_[1250,3000]AA_extsrc_UV<10": [-24.0, -16.0],
    "xi_ion_Halpha_fesc=0": [10 ** 23.5, 10 ** 26.5],
    "log_xi_ion_Halpha_fesc=0": [23.5, 26.5],
    "M_UV_ext_src_corr": [-24.0, -16.0],
    "stellar_mass": [7.5, 11.0],
    "stellar_mass_ext_src_corr": [7.5, 11.0],
}


def get_z_bin_name(z_bin: Union[list, np.array]) -> str:
    """Format a redshift bin as a ``"zmin<z<zmax"`` string.

    Parameters
    ----------
    z_bin : `list` or `numpy.array`
        Two-element ``[zmin, zmax]`` redshift bin edges.

    Returns
    -------
    `str`
        Formatted bin name, e.g. ``"6.0<z<7.0"``.
    """
    return f"{z_bin[0]:.1f}<z<{z_bin[1]:.1f}"


def get_SED_fit_label_aper_diam_z_bin_name(
    SED_fit_params_key: str,
    aper_diam: u.Quantity,
    z_bin: Union[list, np.array]
) -> str:
    """Build a combined label identifying an SED-fitting run, aperture diameter, and redshift bin.

    Parameters
    ----------
    SED_fit_params_key : `str`
        Label identifying the SED-fitting configuration.
    aper_diam : `astropy.units.Quantity`
        Aperture diameter.
    z_bin : `list` or `numpy.array`
        Two-element ``[zmin, zmax]`` redshift bin edges.

    Returns
    -------
    `str`
        Combined label of the form ``"{SED_fit_params_key}_{aper_diam}as_{zmin}<z<{zmax}"``.
    """
    return f"{SED_fit_params_key}_{aper_diam.to(u.arcsec).value:.2f}as_{get_z_bin_name(z_bin)}"


def get_crop_name(crops: List[Selector]) -> str:
    """Build a descriptive name summarising a list of selection/crop objects.

    Concatenates the aperture diameter (if common to all crops), the
    SED-fitter label (if common to all crops, with an EAZY-specific "zfree"
    suffix), and each selector's own name.

    Parameters
    ----------
    crops : `list` of `Selector`
        Selection objects applied to a catalogue/galaxy.

    Returns
    -------
    `str`
        Combined crop name, or an empty string if ``crops`` is `None`.
    """
    from . import EAZY
    if crops is not None:
        aper_diam = np.unique([selector.aper_diam.to(u.arcsec).value for selector \
            in crops if hasattr(selector, "aper_diam") and selector.aper_diam is not None])
        if len(aper_diam) == 1:
            aper_diam = aper_diam[0]
        else:
            aper_diam = None
        SED_fit_label = np.unique([selector.SED_fitter.label for selector \
            in crops if getattr(selector, "SED_fitter", None) is not None])
        if len(SED_fit_label) == 1:
            SED_fit_label = SED_fit_label[0]
        else:
            SED_fit_label = None
        if aper_diam is not None and SED_fit_label is not None:
            # SED_fit_aper_diam_name = f"{SED_fit_label}_{aper_diam:.2f}as"
            out_crop_name = f"{aper_diam:.2f}as/" + \
                "+".join([
                    selector.name.replace(f"_{aper_diam:.2f}as", "") \
                    if not isinstance(getattr(selector, "SED_fitter", None), EAZY) \
                    else f"{selector.name.replace(f'_{aper_diam:.2f}as', '')}_zfree"
                    for selector in crops
                ])
        elif aper_diam is not None:
            aper_diam_name = f"{aper_diam:.2f}as"
            out_crop_name = f"{aper_diam_name}/" + \
                "+".join([selector.name.replace(f"_{aper_diam_name}", "") \
                for selector in crops])
        else:
            out_crop_name = "+".join([selector.name for selector in crops])
        return out_crop_name
    else:
        return ""

def get_full_survey_name(
    survey: str,
    version: str,
    filterset: Multiple_Filter,
) -> str:
    """Build the full identifying name of a survey/version/instrument combination.

    Parameters
    ----------
    survey : `str`
        Survey name.
    version : `str`
        Data reduction version.
    filterset : `Multiple_Filter`
        Filter set defining the instrument combination.

    Returns
    -------
    `str`
        Combined name ``"{survey}_{version}_{filterset.instrument_name}"``.
    """
    return f"{survey}_{version}_{filterset.instrument_name}"

def calc_Vmax(area, zmin, zmax):
    """Compute the comoving volume subtended by a sky area between two redshifts.

    Parameters
    ----------
    area : `astropy.units.Quantity`
        Sky (solid) angle.
    zmin : `float`
        Lower redshift bound.
    zmax : `float`
        Upper redshift bound.

    Returns
    -------
    `astropy.units.Quantity`
        Comoving volume in Mpc^3.
    """
    return (
        (4 / 3 * np.pi)
        * (area / (4.0 * np.pi * u.sr))
        * (
            astropy_cosmo.comoving_distance(zmax) ** 3.0
            - astropy_cosmo.comoving_distance(zmin) ** 3.0
        )
    ).to(u.Mpc**3)


def poisson_interval(k, alpha=0.05):
    """Compute the Poisson confidence interval for a count using the chi-squared method.

    Parameters
    ----------
    k : `int`
        Observed count.
    alpha : `float`, optional
        Significance level (e.g. 0.05 for a 95% interval). Default is 0.05.

    Returns
    -------
    `tuple` of `float`
        ``(low, high)`` confidence interval bounds on the Poisson rate.

    Notes
    -----
    Uses `scipy.stats.chi2`. Adapted from
    https://stackoverflow.com/questions/14813530/poisson-confidence-interval-with-numpy
    """
    low, high = (
        chi2.ppf(alpha / 2.0, 2 * k) / 2,
        chi2.ppf(1.0 - alpha / 2.0, (2 * k) + 2) / 2,
    )
    if k == 0:
        low = 0.0
    return low, high


def calc_cv_proper(
    z_bin: Union[list, np.array],
    data_arr: Union[list, np.array],
    masked_selector: Type[Mask_Selector],
    rectangular_geometry_y_to_x: Union[int, float, list, np.array, dict] = 1.0,
    data_region: Union[str, int] = "all",
    **kwargs: Dict[str, Any],
) -> float:
    """Estimate the cosmic variance of number counts across one or more survey fields.

    Combines the (approximately) rectangular-field cosmic variance recipe in
    quadrature across multiple fields/data regions.

    Parameters
    ----------
    z_bin : `list` or `numpy.array`
        Two-element ``[zmin, zmax]`` redshift bin edges.
    data_arr : `list` or `numpy.array`
        Data objects (one per field), each requiring a ``calc_unmasked_area``
        method and a ``full_name`` attribute.
    masked_selector : `type`
        A `Mask_Selector` subclass/instance used to compute the unmasked area of each field.
    rectangular_geometry_y_to_x : `int`, `float`, `list`, `numpy.array`, or `dict`, optional
        Ratio of the rectangular field's y to x dimension: a single value
        applied to all fields, one value per field, or a `dict` keyed by
        ``data.full_name``. Default is 1.0.
    data_region : `str` or `int`, optional
        Region identifier passed through to ``calc_unmasked_area``. Default is "all".
    **kwargs : `dict`
        Additional keyword arguments passed to ``data.calc_unmasked_area``.

    Returns
    -------
    `float`
        Combined fractional cosmic variance across all fields (0.0 if the
        total area is zero).
    """
    if isinstance(data_region, int):
        data_region = str(data_region)
    if isinstance(rectangular_geometry_y_to_x, int):
        rectangular_geometry_y_to_x = float(rectangular_geometry_y_to_x)
    if isinstance(rectangular_geometry_y_to_x, float):
        rectangular_geometry_y_to_x = [
            rectangular_geometry_y_to_x for i in range(len(data_arr))
        ]
    elif isinstance(rectangular_geometry_y_to_x, (list, np.ndarray)):
        assert len(rectangular_geometry_y_to_x) == len(data_arr)
    elif isinstance(rectangular_geometry_y_to_x, dict):
        assert all(
            data.full_name in rectangular_geometry_y_to_x.keys()
            for data in data_arr
        )
        rectangular_geometry_y_to_x = [
            float(rectangular_geometry_y_to_x[data.full_name])
            for data in data_arr
        ]
    cos_var_tot = 0.0
    total_area = 0.0
    for data, y_to_x in zip(data_arr, rectangular_geometry_y_to_x):
        # calculate area of field
        area = data.calc_unmasked_area(masked_selector, **kwargs) #data.forced_phot_band.filt_name)
        # field is square if y_to_x == 1
        dimensions_x = np.sqrt(area.value / y_to_x) * u.arcmin
        dimensions_y = np.sqrt(area.value * y_to_x) * u.arcmin

        volume = (
            (
                astropy_cosmo.comoving_volume(z_bin[1])
                - astropy_cosmo.comoving_volume(z_bin[0])
            )
            * area
            / (4.0 * np.pi * u.sr)
        ).to(u.Mpc**3)

        codist_low = astropy_cosmo.comoving_distance(z_bin[0]).to(u.Mpc)
        codist_high = astropy_cosmo.comoving_distance(z_bin[1]).to(u.Mpc)
        C = codist_high - codist_low
        A = (
            np.cos(dimensions_y.to(u.rad).value)
            * dimensions_x.to(u.deg).value
            / 360.0
            * (codist_low + 0.5 * C)
        )
        B = dimensions_x.to(u.deg).value / 180.0 * (codist_low + 0.5 * C)
        scale = np.sqrt(
            (volume / (A * B * C)).to(u.dimensionless_unscaled)
        ).decompose()
        A *= scale
        B *= scale
        N = 1
        cos_var = (
            (1.0 - 0.03 * np.sqrt(np.max([A / B, B / A]) - 1.0))
            * (
                219.7
                - 52.4 * np.log10(A.value * B.value * 291.0)
                + 3.21 * (np.log10(A.value * B.value * 291.0)) ** 2.0
            )
            / np.sqrt(N * C.value / 291.0)
        ) / 100.0
        total_area += area
        cos_var_tot += (area**2) * (cos_var**2)
    if total_area != 0.0:
        cosmic_variance = np.sqrt(
            (cos_var_tot / (total_area**2.0))
            .to(u.dimensionless_unscaled)
            .value
        )
    else:
        cosmic_variance = 0.0
    return cosmic_variance


# general functions

def adjust_errs(data, data_err):
    """Convert lower/upper bound values into lower/upper 1-sigma errors relative to the data.

    Parameters
    ----------
    data : array-like
        Central data value(s).
    data_err : `list` or `tuple` of array-like
        Two-element ``[lower_bound, upper_bound]`` values.

    Returns
    -------
    `tuple`
        ``(data, data_err)`` where ``data_err`` is now a stacked 2xN array of
        ``[lower_error, upper_error]``.
    """
    # print("adjusting errors:", plot_data, code)
    data_l1 = data - data_err[0]
    data_u1 = data_err[1] - data
    data_err = np.vstack([data_l1, data_u1])
    return data, data_err


def errs_to_log(data, data_err, uplim_sigma = None, uplim_arrowsize = 0.2, inf_val = 1e6):
    """Propagate data and asymmetric errors into log10 space, handling upper limits.

    Parameters
    ----------
    data : `numpy.ndarray`
        Data value(s).
    data_err : `list` of `numpy.ndarray`
        Two-element ``[lower, upper]`` 1-sigma errors on ``data``.
    uplim_sigma : `float`, optional
        If given, points where the log upper error is undefined are treated as
        upper limits: ``data`` is replaced by ``log10(data + uplim_sigma * upper_err)``
        at those points. Default is `None`.
    uplim_arrowsize : `float`, optional
        Lower log-error value assigned to upper-limit points when ``uplim_sigma`` is set. Default is 0.2.
    inf_val : `float`, optional
        Value substituted for the lower log error where it would otherwise be undefined
        (e.g. ``data - err <= 0``). Default is 1e6.

    Returns
    -------
    `tuple`
        If ``uplim_sigma`` is not `None`: ``(log_data, [log_l1, log_u1], uplim_indices)``
        where ``uplim_indices`` is a boolean mask of upper-limit points.
        Otherwise: ``(log_data, [log_l1, log_u1], all_false_mask)``.
    """
    log_data = np.log10(data)
    log_l1 = log_data - np.log10(data - data_err[0])
    log_u1 = np.log10(data + data_err[1]) - log_data
    if uplim_sigma is not None:
        u1_nans = np.isnan(log_u1)
        log_data[u1_nans] = np.log10(data + uplim_sigma * data_err[1])[u1_nans]
        log_l1[u1_nans] = uplim_arrowsize
        log_u1[u1_nans] = 0.0
        uplim_indices = u1_nans
    l1_nans = np.isnan(log_l1)
    log_l1[l1_nans] = inf_val
    if uplim_sigma is not None:
        return log_data, [log_l1, log_u1], uplim_indices
    else:
        return log_data, [log_l1, log_u1], np.full(len(log_data), False)


def PDF_hist(
    PDF,
    save_dir,
    obs_name,
    ID,
    show=True,
    save=True,
    rest_UV_wavs=[1250.0, 3000.0],
    conv_filt=False,
):
    """Plot (and optionally save) a histogram of a PDF for a single object.

    Does nothing if every value in ``PDF`` equals -99.0 (flagged as invalid).

    Parameters
    ----------
    PDF : array-like
        Sampled PDF values.
    save_dir : `str`
        Base output directory used to construct the save path.
    obs_name : `str`
        Name of the quantity being histogrammed (used as the x-axis label).
    ID : `int` or `str`
        Object ID (used as the legend label and file name).
    show : `bool`, optional
        Whether to draw the legend (and, if not saving, display the figure). Default is `True`.
    save : `bool`, optional
        Whether to save the figure to disk (only applies if ``show`` is `True`);
        otherwise the figure is displayed with `matplotlib.pyplot.show`. Default is `True`.
    rest_UV_wavs : `list`, optional
        Rest-frame UV wavelength range passed to `PDF_path`. Default is ``[1250.0, 3000.0]``.
    conv_filt : `bool`, optional
        Passed through to `PDF_path`. Default is `False`.
    """
    if not all(value == -99.0 for value in PDF):
        plt.hist(PDF, label=ID)
        # print(f"Plotting {obs_name} hist for {ID}")
        plt.xlabel(obs_name)
        if show:
            plt.legend()
            if save:
                path = f"{split_dir_name(PDF_path(save_dir, obs_name, ID, rest_UV_wavs, conv_filt = conv_filt), 'dir')}/hist/{ID}.png"
                make_dirs(path)
                # print(f"Saving hist: {path}")
                plt.savefig(path)
                change_file_permissions(path)
                plt.clf()
            else:
                plt.show()


def split_dir_name(save_path, output):
    """Split a file path into its directory or its base file name.

    Parameters
    ----------
    save_path : `str`
        Full path to a file.
    output : `str`
        Either ``"dir"`` to return the directory portion (with trailing slash),
        or ``"name"`` to return the file name.

    Returns
    -------
    `str`
        The requested portion of ``save_path``.
    """
    if output == "dir":
        return "/".join(np.array(save_path.split("/")[:-1])) + "/"
    elif output == "name":
        return save_path.split("/")[-1]


def gauss_func(x, mu, sigma):
    """Evaluate a Gaussian-shaped function of ``x``.

    Note the normalisation prefactor used here is ``pi * sigma`` rather than
    the standard ``1 / (sigma * sqrt(2 * pi))``.

    Parameters
    ----------
    x : array-like
        Points at which to evaluate the function.
    mu : `float`
        Mean.
    sigma : `float`
        Standard deviation.

    Returns
    -------
    array-like
        Function values at ``x``.
    """
    return (np.pi * sigma) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def power_law_func(x, A, slope):
    """Evaluate a power law ``A * x ** slope``.

    Parameters
    ----------
    x : array-like
        Points at which to evaluate the function.
    A : `float`
        Normalisation.
    slope : `float`
        Power-law index.

    Returns
    -------
    array-like
        Function values at ``x``.
    """
    return A * (x**slope)


def simple_power_law_func(x, c, m):
    """Evaluate a linear function ``m * x + c``.

    Parameters
    ----------
    x : array-like
        Points at which to evaluate the function.
    c : `float`
        Intercept.
    m : `float`
        Slope.

    Returns
    -------
    array-like
        Function values at ``x``.
    """
    return (m * x) + c


def cat_from_path(path, crop_names=None):
    """Load a catalogue table from disk, optionally cropping rows and recording the source path.

    Parameters
    ----------
    path : `str`
        Path to the catalogue file (read via `astropy.table.Table.read`).
    crop_names : `list` of `str`, optional
        Column names; rows are kept only where every listed column is `True`. Default is `None` (no crop).

    Returns
    -------
    `astropy.table.Table`
        Loaded (and optionally cropped) catalogue, with ``cat_path`` stored in ``.meta``.
    """
    cat = Table.read(path, character_as_bytes=False)
    if crop_names != None:
        for name in crop_names:
            cat = cat[cat[name] == True]
    # include catalogue metadata
    cat.meta = {**cat.meta, **{"cat_path": path}}
    return cat


def get_phot_cat_path(
    survey: str,
    version: str,
    instrument_name: str,
    aper_diams: u.Quantity,
    forced_phot_filt_name: Optional[str],
):
    """Construct the standard path to a photometric catalogue FITS file, creating its directory.

    Parameters
    ----------
    survey : `str`
        Survey name.
    version : `str`
        Data reduction version.
    instrument_name : `str`
        Combined instrument name string.
    aper_diams : `astropy.units.Quantity`
        Aperture diameters used in the catalogue.
    forced_phot_filt_name : `str` or `None`
        Name of the forced-photometry detection band/image, or `None` if not applicable.

    Returns
    -------
    `str`
        Full path to the catalogue FITS file.
    """
    save_dir = (
        f"{config['DEFAULT']['GALFIND_WORK']}/Catalogues/{version}/" + \
        f"{instrument_name}/{survey}/{aper_diams_to_str(aper_diams)}"
    )
    if forced_phot_filt_name is None:
        forced_phot_filt_name = ""
    else:
        forced_phot_filt_name = f"_MASTER_Sel-{forced_phot_filt_name}"
    save_name = f"{survey}{forced_phot_filt_name}_{version}.fits"
    save_path = f"{save_dir}/{save_name}"
    make_dirs(save_path)
    return save_path


def fits_cat_to_np(
    fits_cat: Table,
    column_labels: List[str],
    reshape_by_aper_diams: bool = True
):
    """Extract a subset of columns from a FITS table into a plain numpy array.

    Parameters
    ----------
    fits_cat : `astropy.table.Table`
        Source catalogue.
    column_labels : `list` of `str`
        Columns to extract.
    reshape_by_aper_diams : `bool`, optional
        If `True`, reshape the output to ``(n_rows, n_columns, n_aper_diams)``,
        inferring ``n_aper_diams`` from the first row/column entry. Default is `True`.

    Returns
    -------
    `numpy.ndarray`
        Extracted data, reshaped as described.

    Raises
    ------
    AssertionError
        If ``fits_cat`` is empty.
    """
    new_cat = fits_cat[column_labels].as_array()
    assert len(new_cat) > 0, \
        galfind_logger.critical(
            "Cannot convert empty fits_cat!"
        )
    if isinstance(new_cat, np.ma.core.MaskedArray):
        new_cat = new_cat.data
    if reshape_by_aper_diams:
        if isinstance(new_cat[0][0], (float, int)):
            n_aper_diams = 1
        else:
            n_aper_diams = len(new_cat[0][0])
        new_cat = np.lib.recfunctions.structured_to_unstructured(
            new_cat
        ).reshape(len(fits_cat), len(column_labels), n_aper_diams)
    else:
        new_cat = np.lib.recfunctions.structured_to_unstructured(
            new_cat
        ).reshape(len(fits_cat), len(column_labels))
    return new_cat


def lowz_label(lowz_zmax):
    """Build the label used for a low-redshift-restricted SED-fitting run.

    Parameters
    ----------
    lowz_zmax : `float` or `None`
        Maximum redshift imposed on the fit, or `None` for an unrestricted ("zfree") fit.

    Returns
    -------
    `str`
        ``"zmax={lowz_zmax:.1f}"``, or ``"zfree"`` if ``lowz_zmax`` is `None`.
    """
    if lowz_zmax != None:
        label = f"zmax={lowz_zmax:.1f}"
    else:
        label = "zfree"
    return label


def zmax_from_lowz_label(label):
    """Recover the zmax value encoded in a low-redshift fit label.

    Parameters
    ----------
    label : `str`
        Label produced by `lowz_label`.

    Returns
    -------
    `float` or `None`
        The zmax value, or `None` if ``label`` is ``"zfree"``.
    """
    if label == "zfree":
        zmax = None
    else:
        zmax = float(label.replace("zmax=", ""))
    return zmax


def get_z_PDF_paths(
    fits_cat, IDs, codes, templates_arr, lowz_zmaxs, fits_cat_path=None
):
    """Build the redshift-PDF file paths for a set of objects and SED-fitting codes.

    Parameters
    ----------
    fits_cat : `astropy.table.Table`
        Catalogue; its ``.meta["cat_path"]`` is used as the base path if available.
    IDs : `list`
        Object IDs to build paths for.
    codes : `list` of `SED_code`
        SED-fitting code instance(s) providing ``z_PDF_paths_from_cat_path``.
    templates_arr : `list`
        Template set identifier corresponding to each entry in ``codes``.
    lowz_zmaxs : `list`
        Low-redshift zmax value (or `None`) corresponding to each entry in ``codes``.
    fits_cat_path : `str`, optional
        Explicit catalogue path to use instead of ``fits_cat.meta["cat_path"]``. Default is `None`.

    Returns
    -------
    `list`
        Flattened list of PDF file paths for every ``(code, templates, lowz_zmax)`` x ``ID`` combination.
    """
    try:
        fits_cat_path = fits_cat.meta["cat_path"]
    except:
        pass
    return [
        code.z_PDF_paths_from_cat_path(
            fits_cat_path, ID, templates, lowz_label(lowz_zmax)
        )
        for code, templates, lowz_zmax in zip(codes, templates_arr, lowz_zmaxs)
        for ID in IDs
    ]


def get_SED_paths(
    fits_cat, IDs, codes, templates_arr, lowz_zmaxs, fits_cat_path=None
):
    """Build the best-fit SED file paths for a set of objects and SED-fitting codes.

    Parameters
    ----------
    fits_cat : `astropy.table.Table`
        Catalogue; its ``.meta["cat_path"]`` is used as the base path if available.
    IDs : `list`
        Object IDs to build paths for.
    codes : `list` of `SED_code`
        SED-fitting code instance(s) providing ``SED_paths_from_cat_path``.
    templates_arr : `list`
        Template set identifier corresponding to each entry in ``codes``.
    lowz_zmaxs : `list`
        Low-redshift zmax value (or `None`) corresponding to each entry in ``codes``.
    fits_cat_path : `str`, optional
        Explicit catalogue path to use instead of ``fits_cat.meta["cat_path"]``. Default is `None`.

    Returns
    -------
    `list`
        Flattened list of SED file paths for every ``(code, templates, lowz_zmax)`` x ``ID`` combination.
    """
    try:
        fits_cat_path = fits_cat.meta["cat_path"]
    except:
        pass
    return [
        code.SED_paths_from_cat_path(
            fits_cat_path, ID, templates, lowz_label(lowz_zmax)
        )
        for code, templates, lowz_zmax in zip(codes, templates_arr, lowz_zmaxs)
        for ID in IDs
    ]


def ordinal(n: int):
    """Convert an integer to its ordinal string representation.

    Parameters
    ----------
    n : `int`
        Integer to convert.

    Returns
    -------
    `str`
        ``n`` with the appropriate ordinal suffix appended (e.g. ``"1st"``, ``"22nd"``).
    """
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]
    return str(n) + suffix


def date_finder(text: str):
    """Find ISO- or slash/dash-formatted dates within a string.

    Parameters
    ----------
    text : `str`
        Text to search.

    Returns
    -------
    `list` of `str`
        All date-like substrings matched in ``text``.
    """
    pattern = r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}[/-]\d{2}[/-]\d{4}\b"
    dates = re.findall(pattern, text)
    return dates


def validate_quantity(
    quant: Optional[Any],
    physical_type: str,
):
    """Validate that a value is an `astropy.units.Quantity` of the expected physical type.

    Parameters
    ----------
    quant : `Any`, optional
        Value to validate; may be `None`.
    physical_type : `str`
        Required astropy physical type string (as returned by `astropy.units.get_physical_type`).

    Returns
    -------
    `astropy.units.Quantity` or `None`
        ``quant`` unchanged if it is a `Quantity` of the correct physical type;
        `None` if ``quant`` is `None` or not a `Quantity`.

    Raises
    ------
    AssertionError
        If ``quant`` is a `Quantity` but does not have the required ``physical_type``.
    """
    if quant is not None:
        if not isinstance(quant, u.Quantity):
            galfind_logger.warning(
                f"{quant} must be a Quantity! Changing to None"
            )
            quant = None
        else:
            assert u.get_physical_type(quant) == physical_type, \
                galfind_logger.critical(
                    f"{quant} must have units of type {physical_type}!"
                )
    return quant


# beta slope function
def beta_slope_power_law_func(wav_rest, A, beta):
    """Evaluate a UV continuum power-law spectrum ``f(wav) = 10 ** A * wav ** beta``.

    Parameters
    ----------
    wav_rest : array-like
        Rest-frame wavelength(s).
    A : `float`
        Log-normalisation.
    beta : `float`
        UV continuum slope.

    Returns
    -------
    array-like
        Function values at ``wav_rest``.
    """
    return (10**A) * (wav_rest**beta)

def crop_to_Calzetti94_filters(wavs, mags):
    """Crop wavelength/magnitude arrays to the Calzetti (1994) UV continuum windows.

    Parameters
    ----------
    wavs : `astropy.units.Quantity`
        Wavelengths, convertible to Angstrom.
    mags : array-like
        Magnitude or flux values corresponding to ``wavs``.

    Returns
    -------
    `tuple` of (`astropy.units.Quantity`, array-like)
        ``(wavs, mags)`` cropped to only the points falling within the Calzetti94 filter windows.
    """
    wavs = wavs.to(u.AA)
    Calzetti94_filter_indices = np.logical_or.reduce(
        [
            (wavs.value > low_lim) & (wavs.value < up_lim)
            for low_lim, up_lim in zip(
                lower_Calzetti_filt,
                upper_Calzetti_filt,
            )
        ]
    )
    wavs = wavs[Calzetti94_filter_indices]
    mags = mags[Calzetti94_filter_indices]
    return wavs, mags


def inspect_info():
    """Get the file name, function name, and line number of the calling frame.

    Returns
    -------
    `tuple` of (`str`, `str`, `int`)
        ``(filename, function_name, line_number)`` of the caller.
    """
    info = inspect.getframeinfo(inspect.stack()[1][0])
    return info.filename, info.function, info.lineno


def make_dirs(path, permissions=0o777):
    """Create the parent directory of a file path if it doesn't exist, and set its permissions.

    Parameters
    ----------
    path : `str`
        File path whose parent directory should be created.
    permissions : `int`, optional
        Octal permissions to apply to the created directory. Default is ``0o777``.
    """
    os.makedirs(split_dir_name(path, "dir"), exist_ok=True)
    try:
        os.chmod(split_dir_name(path, "dir"), permissions)
    except PermissionError:
        galfind_logger.warning(
            f"Could not change permissions of {path} to {oct(permissions)}."
        )


def change_file_permissions(path, permissions=0o777, log=False):
    """Change file permissions for one or more files, silently ignoring missing files or permission errors.

    Parameters
    ----------
    path : `str` or `list` of `str`
        File path(s) to change the permissions of.
    permissions : `int`, optional
        Octal permissions to apply. Default is ``0o777``.
    log : `bool`, optional
        Whether to log successful permission changes at INFO level. Default is `False`.
    """
    if type(path) != list:
        path = [path]
    for p in path:
        try:
            os.chmod(p, permissions)
            if log:
                galfind_logger.info(
                    f"Changed permissions of {p} to {oct(permissions)}"
                )
        except (PermissionError, FileNotFoundError):
            pass


def source_separation(sky_coord_1, sky_coord_2, z):
    """Compute the transverse proper separation between two sky positions at a given redshift.

    Parameters
    ----------
    sky_coord_1 : `astropy.coordinates.SkyCoord`
        First sky position.
    sky_coord_2 : `astropy.coordinates.SkyCoord`
        Second sky position.
    z : `float`
        Redshift used to convert angular separation to a physical separation.

    Returns
    -------
    `astropy.units.Quantity`
        Separation in kpc.
    """
    # calculate separation in arcmin
    arcmin_sep = sky_coord_1.separation(sky_coord_2).to(u.arcmin)
    # print(arcmin_sep.to(u.arcsec))
    # calculate separation in transverse comoving distance
    kpc_sep = arcmin_sep * astropy_cosmo.kpc_proper_per_arcmin(z)
    return kpc_sep


def tex_to_fits(
    tex_path,
    col_names,
    col_errs,
    replace={
        "&": "",
        "\\\\": "",
        r"\dag": "",
        r"\ddag": "",
        r"\S": "",
        r"\P": "",
        "$": "",
        "}": "",
        "^{+": " ",
        "^{": "",
        "_{-": " ",
    },
    empty=["-"],
    comment="%",
):
    """Parse a LaTeX table into a FITS catalogue.

    Reads a LaTeX tabular file line by line, strips LaTeX formatting/markup,
    splits each row into columns, inserts NaNs for missing/placeholder
    entries, and writes the result to disk as a FITS table.

    Parameters
    ----------
    tex_path : `str`
        Path to the input LaTeX table file.
    col_names : `list` of `str`
        Base column names, one per data column (excluding auto-generated error columns).
    col_errs : `list` of `bool`
        Flags (one per entry in ``col_names``) indicating whether that column
        has associated upper/lower error columns immediately following it in the LaTeX table.
    replace : `dict`, optional
        Mapping of LaTeX substrings to their plain-text replacement, applied to every line.
    empty : `list` of `str`, optional
        Placeholder tokens treated as missing data. Default is ``["-"]``.
    comment : `str`, optional
        Line prefix marking comment lines to skip. Default is ``"%"``.

    Returns
    -------
    None
        Writes a FITS file to disk (path derived from ``tex_path`` by replacing
        ``".txt"`` with ``"_as_fits.fits"``).
    """
    # note which columns are error columns
    is_err = col_errs.copy()
    for i in col_errs:
        if i:
            is_err[i] = False
            is_err[i:i] = np.full(2, True)
    save_data = []
    # read tex table line by line
    with open(tex_path, "r") as tab:
        line_no = 0
        while True:
            line = tab.readline()

            if not line:
                break

            if not line.startswith(comment):  # ignore comments in the table
                line_no += 1
                # format the line into something .txt readable
                for i, (key, val) in enumerate(replace.items()):
                    line = line.replace(key, val)
                # turn each line into an array
                line_elements = line.split()
                # insert nans where there is not the appropriate data
                while True:
                    if len(line_elements) == len(is_err):
                        break
                    for i, val in enumerate(line_elements):
                        if val in empty:
                            line_elements[i] = np.nan
                            if is_err[i]:
                                line_elements[i:i] = np.full(2, np.nan)
                            break
                # append the data
                if line_no == 1:
                    save_data = line_elements
                else:
                    save_data = np.vstack([save_data, line_elements])
        print(save_data)
        tab.close()

    change_file_permissions(tex_path)
    # adjust column names to include errors where appropriate
    cat_col_names = []
    for i, name in enumerate(col_names):
        cat_col_names.append(name)
        if col_errs[i]:
            cat_col_names.append(f"{name}_u1")
            cat_col_names.append(f"{name}_l1")
    cat_dtypes = np.array(np.full(len(cat_col_names), float))
    cat_dtypes[0] = str  # not general
    cat_dtypes[-1] = str  # not general
    fits_table = Table(save_data, names=cat_col_names, dtype=cat_dtypes)
    fits_path = tex_path.replace(".txt", "_as_fits.fits")
    fits_table.write(fits_path, overwrite=True)
    change_file_permissions(fits_path)
    print(f"Saved {tex_path} as .fits")


def ext_source_corr(data, corr_factor, is_log_data=True):
    """Apply an extended-source correction factor to a (possibly logged) quantity.

    Parameters
    ----------
    data : array-like
        Data value(s) to correct.
    corr_factor : array-like
        Multiplicative correction factor.
    is_log_data : `bool`, optional
        If `True`, ``data`` is assumed to be in log10 space and
        ``log10(corr_factor)`` is added to it; otherwise ``data`` is
        multiplied by ``corr_factor`` directly. Default is `True`.

    Returns
    -------
    array-like
        Corrected data.
    """
    if is_log_data:
        return data + np.log10(corr_factor)
    else:
        return data * corr_factor


def power_law_beta_func(wav, A, beta):
    """Evaluate a power law ``A * wav ** beta``.

    Parameters
    ----------
    wav : array-like
        Wavelength(s) at which to evaluate the function.
    A : `float`
        Normalisation.
    beta : `float`
        Power-law index.

    Returns
    -------
    array-like
        Function values at ``wav``.
    """
    return A * (wav**beta)


class Singleton(object):
    """Base class implementing the singleton design pattern.

    Subclassing `Singleton` and instantiating the subclass multiple times
    always returns the same shared instance.

    Attributes
    ----------
    _instance : `object` or `None`
        The single shared instance, created on first instantiation.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Instantiate or retrieve the singleton instance.

        Returns
        -------
        `object`
            The singleton instance, created on first call and reused thereafter.
        """
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance


# for __str__ methods
line_sep = "*" * 40 + "\n"
band_sep = "-" * 10 + "\n"

def aper_diams_to_str(aper_diams: u.Quantity):
    """Format aperture diameters as a compact string label.

    Parameters
    ----------
    aper_diams : `astropy.units.Quantity`
        Array of aperture diameters.

    Returns
    -------
    `str`
        Comma-separated diameters (2 d.p., in arcsec) wrapped in parentheses
        with an "as" suffix, e.g. ``"(0.32,0.50)as"``.
    """
    return f"({','.join([f'{aper_diam:.2f}' for aper_diam in aper_diams.value])})as"

def calc_unmasked_area(
    mask: Union[np.ndarray, Tuple[np.ndarray]],
    pixel_scale: u.Quantity
) -> u.Quantity:
    """Compute the sky area covered by unmasked (`True`) pixels.

    Parameters
    ----------
    mask : `numpy.ndarray` or `tuple` of `numpy.ndarray`
        Boolean mask array, or a tuple of boolean masks combined with a logical AND.
    pixel_scale : `astropy.units.Quantity`
        Angular size of one pixel.

    Returns
    -------
    `astropy.units.Quantity`
        Unmasked area in arcmin^2.
    """
    if isinstance(mask, tuple):
        mask = np.logical_and.reduce(mask)
    return ((np.sum(mask)) * (pixel_scale ** 2)).to(u.arcmin**2)

def sort_band_data_arr(band_data_arr: List[Type[Band_Data_Base]]):
    """Sort band-data objects blue to red by central wavelength.

    Any `Stacked_Band_Data` objects are appended, unsorted, after the sorted
    `Band_Data` objects.

    Parameters
    ----------
    band_data_arr : `list` of `Band_Data_Base`
        Mixture of `Band_Data` and `Stacked_Band_Data` objects.

    Returns
    -------
    `list` of `Band_Data_Base`
        `Band_Data` objects sorted by ascending central wavelength, followed
        by any `Stacked_Band_Data` objects.
    """
    stacked_band_data_arr = [band_data for band_data in band_data_arr if band_data.__class__.__name__ == "Stacked_Band_Data"]
    sorted_band_data_arr = [
        band_data
        for band_data in sorted(
            [band_data for band_data in band_data_arr if band_data.__class__.__name__ == "Band_Data"],
            key=lambda band_data: band_data.filt.WavelengthCen.to(u.AA).value,
        )
    ]
    sorted_band_data_arr.extend(stacked_band_data_arr)
    return sorted_band_data_arr

def rolling_average(y_array, window_size):
    """Compute a rolling (moving) average of an array using a boxcar kernel.

    Parameters
    ----------
    y_array : array-like
        Input data.
    window_size : `int`
        Number of samples to average over.

    Returns
    -------
    `numpy.ndarray`
        Rolling average, with length ``len(y_array) - window_size + 1`` (valid-mode convolution).
    """
    kernel = np.ones(window_size) / window_size
    return np.convolve(y_array, kernel, mode='valid')

# The below makes TQDM work with joblib
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager that reports joblib parallel progress into a tqdm progress bar.

    Parameters
    ----------
    tqdm_object : `tqdm.tqdm`
        A tqdm progress bar instance to update as joblib batches complete.

    Yields
    ------
    `tqdm.tqdm`
        The same ``tqdm_object``, updated as batches complete.
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        """Callback that updates a tqdm progress bar when a joblib batch completes.

        Extends `joblib.parallel.BatchCompletionCallBack` to notify the parent
        tqdm progress bar each time a batch of parallel jobs finishes.
        """
        def __call__(self, *args, **kwargs):
            """Invoke the callback, updating the tqdm progress bar.

            Updates the associated tqdm progress bar by the batch size and then
            calls the parent callback implementation.
            """
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# useful for rest frame SED property calculations

def get_first_bluewards_band(
    z: float,
    filterset: Multiple_Filter,
    ref_wav: u.Quantity,
    ignore_bands: Optional[Union[str, List[str]]] = None,
) -> Optional[str]:
    """Find the reddest filter that lies entirely bluewards of a redshifted reference wavelength.

    Parameters
    ----------
    z : `float`
        Redshift; if negative, no band is returned.
    filterset : `Multiple_Filter`
        Filter set ordered blue to red.
    ref_wav : `astropy.units.Quantity`
        Rest-frame reference wavelength.
    ignore_bands : `str` or `list` of `str`, optional
        Band name(s) to exclude from consideration. Default is `None`.

    Returns
    -------
    `str` or `None`
        Name of the first (reddest) band found bluewards of ``ref_wav * (1 + z)``,
        or `None` if none is found or ``z < 0``.
    """
    # convert ignore_bands to List[str] if not already
    if ignore_bands is None:
        ignore_bands = []
    elif isinstance(ignore_bands, str):
        ignore_bands = [ignore_bands]
    first_band = None
    if z < 0.0:
        return first_band
    # bands already ordered from blue -> red
    for filt in reversed(filterset):
        if filt.filt_name not in ignore_bands:
            upper_wav = filt.WavelengthUpper50
            if upper_wav < ref_wav * (1.0 + z):
                first_band = filt.filt_name
                break
    return first_band

def get_first_redwards_band(
    z: float,
    filterset: Multiple_Filter,
    ref_wav: u.Quantity,
    ignore_bands: Optional[Union[str, List[str]]] = None,
) -> Optional[str]:
    """Find the bluest filter that lies entirely redwards of a redshifted reference wavelength.

    Parameters
    ----------
    z : `float`
        Redshift.
    filterset : `Multiple_Filter`
        Filter set ordered blue to red.
    ref_wav : `astropy.units.Quantity`
        Rest-frame reference wavelength.
    ignore_bands : `str` or `list` of `str`, optional
        Band name(s) to exclude from consideration. Default is `None`.

    Returns
    -------
    `str` or `None`
        Name of the first (bluest) band found redwards of ``ref_wav * (1 + z)``,
        or `None` if none is found.
    """
    # convert ignore_bands to List[str] if not already
    if ignore_bands is None:
        ignore_bands = []
    elif isinstance(ignore_bands, str):
        ignore_bands = [ignore_bands]
    first_band = None
    for filt in filterset:
        if filt.filt_name not in ignore_bands:
            lower_wav = filt.WavelengthLower50
            if lower_wav > ref_wav * (1.0 + z):
                first_band = filt.filt_name
                break
    return first_band

def group_positions(
    sky_coords: SkyCoord,
    match_radius: u.Quantity = 2.0 * u.arcsec
) -> Dict[int, List[int]]:
    """Group sky positions by proximity within a matching radius.

    Parameters
    ----------
    sky_coords : `astropy.coordinates.SkyCoord`
        Sky positions to group.
    match_radius : `astropy.units.Quantity`, optional
        Matching radius. Default is 2 arcsec.

    Returns
    -------
    `dict`
        Mapping of a group name (derived from the median RA/Dec of the group,
        e.g. ``"j1234p5678"``) to the list of indices into ``sky_coords``
        belonging to that group.
    """

    # adjacency matrix for matches
    coords_len = len(sky_coords)
    groups = {}
    visited = np.zeros(coords_len, dtype=bool)
    for i in range(coords_len):
        if visited[i]:
            continue
        # Find all neighbors within radius of point i that havn't already been visited
        sep = sky_coords[i].separation(sky_coords)
        mask = (sep < match_radius) & ~visited
        indices = np.where(mask)[0]

        # name group by median RA/DEC of each group
        median_ra = np.median(sky_coords[indices].ra).to_string(unit=u.hourangle, sep=('h', 'm', 's'))
        ra_label = f"{round(float(median_ra.split('h')[0])):02d}" + \
            f"{round(float(median_ra.split('h')[-1].split('m')[0])):02d}" + \
            f"{round(float(median_ra.split('h')[-1].split('m')[-1].split('s')[0])):02d}"
        median_dec = np.median(sky_coords[indices].dec)
        dec_sign = "p" if median_dec >= 0.0 * u.deg else "m"
        median_dec = median_dec.to_string(unit=u.deg, sep=('d', 'm'))
        dec_label = f"{round(abs(float(median_dec.split('d')[0]))):02d}" + \
            f"{round(float(median_dec.split('d')[-1].split('m')[0])):02d}"
        group_name = f"j{ra_label}{dec_sign}{dec_label}"

        groups[group_name] = indices.tolist()
        visited[indices] = True
    return groups

def parse_s_region(s_region):
    """Parse an S_REGION FITS header polygon string into vertex coordinates.

    Parameters
    ----------
    s_region : `str`
        S_REGION string of the form ``"POLYGON <frame> ra1 dec1 ra2 dec2 ..."``.

    Returns
    -------
    `numpy.ndarray` or `None`
        Nx2 array of ``(ra, dec)`` vertex coordinates, or `None` if
        ``s_region`` does not match the expected polygon format or has an
        invalid number of values.
    """
    # Expect "POLYGON ICRS ra1 dec1 ra2 dec2 ..."
    m = re.search(r'POLYGON\s+\w+\s+(.+)', s_region, flags=re.IGNORECASE)
    if not m:
        return None
    vals = list(map(float, m.group(1).split()))
    if len(vals) < 6 or len(vals) % 2 != 0:
        return None
    coords = np.array(list(zip(vals[0::2], vals[1::2])))
    return coords

def footprints_from_files(files):
    """Extract sky-region footprint polygons from the S_REGION header keyword of a list of FITS files.

    Parameters
    ----------
    files : `list` of `str`
        Paths to FITS files with an ``"SCI"`` extension containing an S_REGION header keyword.

    Returns
    -------
    `dict`
        Mapping of file path to Nx2 `numpy.ndarray` of ``(ra, dec)`` polygon
        vertices, for files where a valid S_REGION was found and parsed.
    """
    from astropy.io import fits
    from matplotlib.patches import Polygon
    footprints = {}
    for f in files:
        try:
            with fits.open(f) as hdul:
                sreg = hdul["SCI"].header.get("S_REGION")
            if sreg:
                coords = parse_s_region(sreg)
                if coords is not None:
                    footprints[f] = coords
        except Exception as e:
            print(f"Skipping {f}: {e}")
    return footprints

@njit
def linear_fit(x: NDArray[np.float64], y: NDArray[np.float64]) -> Tuple[float, float]:
    """Perform a linear least-squares fit ``y = slope * x + intercept``.

    Compiled with numba (`@njit`) for performance.

    Parameters
    ----------
    x : `numpy.ndarray` of `float64`
        Independent variable (1D array).
    y : `numpy.ndarray` of `float64`
        Dependent variable (1D array).

    Returns
    -------
    `tuple` of `float`
        ``(slope, intercept)`` of the best-fit line.
    """
    n = len(x)

    # Compute sums for least squares
    sum_x = 0.0
    sum_y = 0.0
    sum_x2 = 0.0
    sum_xy = 0.0
    for i in range(n):
        sum_x += x[i]
        sum_y += y[i]
        sum_x2 += x[i] * x[i]
        sum_xy += x[i] * y[i]

    # Calculate slope and intercept
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept

@njit
def interpolate_linear_fit(x: NDArray[np.float64], y: NDArray[np.float64], x_out: float) -> float:
    """Evaluate a linear least-squares fit of ``(x, y)`` at a new point.

    Compiled with numba (`@njit`) for performance.

    Parameters
    ----------
    x : `numpy.ndarray` of `float64`
        Independent variable used to fit the line.
    y : `numpy.ndarray` of `float64`
        Dependent variable used to fit the line.
    x_out : `float`
        Point at which to evaluate the fitted line.

    Returns
    -------
    `float`
        Fitted y value at ``x_out``.
    """
    slope, intercept = linear_fit(x, y)
    return slope * x_out + intercept

@njit
def residual_sum_of_squares(params, x, y):
    """Compute the residual sum of squares for a linear model ``y = m * x + c``.

    Compiled with numba (`@njit`) for performance.

    Parameters
    ----------
    params : array-like
        Two-element ``(m, c)`` linear model parameters.
    x : array-like
        Independent variable.
    y : array-like
        Observed dependent variable.

    Returns
    -------
    `float`
        Sum of squared residuals between ``y`` and the model prediction.
    """
    m, c = params
    residuals = y - (m * x + c)
    return np.sum(residuals ** 2)

@njit
def gradient_descent_beta_fit(x, y, initial_params, learning_rate=0.01, max_iter=1000, tol=1e-6):
    """Fit a linear model to ``(x, y)`` via gradient descent on the residual sum of squares.

    Compiled with numba (`@njit`) for performance.

    Parameters
    ----------
    x : array-like
        Independent variable.
    y : array-like
        Dependent variable.
    initial_params : array-like
        Initial ``(m, c)`` parameter guess.
    learning_rate : `float`, optional
        Gradient descent step size. Default is 0.01.
    max_iter : `int`, optional
        Maximum number of iterations. Default is 1000.
    tol : `float`, optional
        Convergence tolerance on the gradient norm. Default is 1e-6.

    Returns
    -------
    `tuple`
        ``(params, i)`` where ``params`` is the array ``[m, c]`` of fitted
        parameters and ``i`` is the number of iterations taken.
    """
    params = np.array(initial_params, dtype=np.float64)
    for i in range(max_iter):
        m, c = params
        residuals = y - (m * x + c)

        # Compute gradients
        grad_m = -2 * np.sum(x * residuals)
        grad_c = -2 * np.sum(residuals)

        # Update parameters
        params[0] -= learning_rate * grad_m
        params[1] -= learning_rate * grad_c

        # Check for convergence
        if np.sqrt(grad_m ** 2 + grad_c ** 2) < tol:
            break

    return params, i  # Return optimized parameters and iterations taken

def symlink(target_path, symlink_path):
    """Create a symlink to a target file, creating parent directories as needed.

    Parameters
    ----------
    target_path : `str`
        Path to the file the symlink should point to.
    symlink_path : `str`
        Path at which to create the symlink.
    """
    make_dirs(symlink_path)
    if Path(target_path).is_file():
        try:
            os.symlink(target_path, symlink_path)
            galfind_logger.info(f"Created symlink: {symlink_path} -> {target_path}")
        except FileExistsError:
            galfind_logger.debug(f"Symlink already exists: {symlink_path}")
    else:
        breakpoint()
        galfind_logger.warning(f"Target file does not exist for symlink: {target_path}")

def get_depth_dir(galfind_work_dir, survey, version, instrument_names):
    """Build the depths output directory path(s) for each instrument.

    Parameters
    ----------
    galfind_work_dir : `str`
        Root GALFIND working directory.
    survey : `str`
        Survey name.
    version : `str`
        Data reduction version.
    instrument_names : `list` of `str`
        Instrument names to build a directory for.

    Returns
    -------
    `numpy.ndarray` of `str`
        One depths directory path per instrument.
    """
    out_dirs = []
    for instrument_name in instrument_names:
        out_dirs.append(f"{galfind_work_dir}/Depths/{instrument_name}/{version}/{survey}")
    return np.array(out_dirs)

def get_eazy_dir(galfind_work_dir, survey, version, instrument_names):
    """Build the EAZY input/output directory paths for a combined instrument set.

    Parameters
    ----------
    galfind_work_dir : `str`
        Root GALFIND working directory.
    survey : `str`
        Survey name.
    version : `str`
        Data reduction version.
    instrument_names : `list` of `str`
        Instrument names, combined with ``"+"`` into a single identifier.

    Returns
    -------
    `numpy.ndarray` of `str`
        Two paths: ``[input_dir, output_dir]``.
    """
    instrument_name = "+".join(instrument_names)
    out_dirs = []
    for subdir in ["input", "output"]:
        out_dirs.append(f"{galfind_work_dir}/EAZY/{subdir}/{instrument_name}/{version}/{survey}")
    return np.array(out_dirs)

def get_mask_dir(galfind_work_dir, survey):
    """Build the mask directory path for a survey.

    Parameters
    ----------
    galfind_work_dir : `str`
        Root GALFIND working directory.
    survey : `str`
        Survey name.

    Returns
    -------
    `numpy.ndarray` of `str`
        Single-element array containing the masks directory path.
    """
    return np.array([f"{galfind_work_dir}/Masks/{survey}"])

def get_sex_dir(galfind_work_dir, survey, version, instrument_names):
    """Build the SExtractor output directory path(s) for each instrument.

    Parameters
    ----------
    galfind_work_dir : `str`
        Root GALFIND working directory.
    survey : `str`
        Survey name.
    version : `str`
        Data reduction version.
    instrument_names : `list` of `str`
        Instrument names to build a directory for.

    Returns
    -------
    `numpy.ndarray` of `str`
        One SExtractor directory path per instrument.
    """
    out_dirs = []
    for instrument_name in instrument_names:
        out_dirs.append(f"{galfind_work_dir}/SExtractor/{instrument_name}/{version}/{survey}")
    return np.array(out_dirs)

def get_stacked_images_dir(galfind_work_dir, survey, version, instrument_names):
    """Build the stacked-images directory path(s) for each instrument.

    Parameters
    ----------
    galfind_work_dir : `str`
        Root GALFIND working directory.
    survey : `str`
        Survey name.
    version : `str`
        Data reduction version.
    instrument_names : `list` of `str`
        Instrument names to build a directory for.

    Returns
    -------
    `numpy.ndarray` of `str`
        One stacked-images directory path per instrument.
    """
    out_dirs = []
    for instrument_name in instrument_names:
        out_dirs.append(f"{galfind_work_dir}/Stacked_Images/{version}/{instrument_name}/{survey}")
    return np.array(out_dirs)

def find_target_dir(galfind_work_dir, survey, version, instrument_names, keyword):
    """Dispatch to the appropriate directory-builder function based on a keyword.

    Parameters
    ----------
    galfind_work_dir : `str`
        Root GALFIND working directory.
    survey : `str`
        Survey name.
    version : `str`
        Data reduction version.
    instrument_names : `list` of `str`
        Instrument names.
    keyword : `str`
        One of ``"Depths"``, ``"EAZY"``, ``"Masks"``, ``"SExtractor"``, ``"Stacked_Images"``.

    Returns
    -------
    `numpy.ndarray` of `str`
        Directory path(s) for the requested product type.

    Raises
    ------
    ValueError
        If ``keyword`` is not one of the recognised values.
    """
    if keyword == "Depths":
        return get_depth_dir(galfind_work_dir, survey, version, instrument_names)
    elif keyword == "EAZY":
        return get_eazy_dir(galfind_work_dir, survey, version, instrument_names)
    elif keyword == "Masks":
        return get_mask_dir(galfind_work_dir, survey)
    elif keyword == "SExtractor":
        return get_sex_dir(galfind_work_dir, survey, version, instrument_names)
    elif keyword == "Stacked_Images":
        return get_stacked_images_dir(galfind_work_dir, survey, version, instrument_names)
    else:
        raise ValueError(f"Keyword {keyword} not recognised")

def make_symlinks(target_galfind_work, symlink_galfind_work, survey, version, instrument_names, keywords):
    """Create symlinks mirroring a set of GALFIND work-directory products into another location.

    Parameters
    ----------
    target_galfind_work : `str`
        Source GALFIND working directory containing the real files.
    symlink_galfind_work : `str`
        Destination GALFIND working directory in which to create symlinks.
    survey : `str`
        Survey name.
    version : `str`
        Data reduction version.
    instrument_names : `list` of `str`
        Instrument names.
    keywords : `list` of `str`
        Product-type keywords (as accepted by `find_target_dir`) to symlink.
    """
    for keyword in keywords:
        target_dirs = find_target_dir(target_galfind_work, survey, version, instrument_names, keyword)
        for target_dir in target_dirs:
            target_paths = [str(path) for path in Path(target_dir).rglob("*") if path.is_file()]
            symlink_paths = [path.replace(target_galfind_work, symlink_galfind_work) for path in target_paths]
            for target_path, symlink_path in zip(target_paths, symlink_paths):
                symlink(target_path, symlink_path)

def get_ext_src_corr(
    phot_rest: Photometry_rest,
    ext_src_key: Optional[str] = "UV",
    ext_src_uplim: Optional[Union[int, float]] = 10.0,
    ref_wav: u.Quantity = 1_500.0 * u.AA,
) -> float:
    """Look up (and clip) the extended-source photometric correction for a galaxy.

    Parameters
    ----------
    phot_rest : `Photometry_rest`
        Rest-frame photometry object providing ``filterset``, ``z``, and ``ext_src_corrs``.
    ext_src_key : `str`, optional
        Either ``"UV"`` to select the correction from the band nearest
        ``ref_wav`` in the rest frame, a specific band name, or `None` to
        disable the correction entirely. Default is ``"UV"``.
    ext_src_uplim : `int` or `float`, optional
        Upper limit clipped onto the correction factor; `None` disables clipping. Default is 10.0.
    ref_wav : `astropy.units.Quantity`, optional
        Rest-frame reference wavelength used when ``ext_src_key`` is ``"UV"``. Default is 1500 Angstrom.

    Returns
    -------
    `float`
        Extended source correction factor (``>= 1.0``), or ``1.0`` if
        ``ext_src_key`` is `None`, or `numpy.nan` if ``phot_rest`` has no filters.

    Raises
    ------
    AttributeError
        If ``phot_rest`` has no ``ext_src_corrs`` attribute.
    """
    if ext_src_key is None:
        return 1.0
    else:
        if len(phot_rest.filterset) == 0:
            galfind_logger.debug(
                f"{repr(phot_rest)} has {len(phot_rest.filterset)=}! " +
                "Unable to compute extended source correction!"
            )
            return np.nan
    if not hasattr(phot_rest, "ext_src_corrs"):
        err_message = f"{repr(phot_rest)} has no attribute ext_src_corrs! " + \
            "Unable to compute extended source correction!"
        galfind_logger.critical(err_message)
        raise AttributeError(err_message)
    if ext_src_key == "UV":
        # calculate band nearest to the rest frame UV reference wavelength
        band_wavs = [filt.WavelengthCen.to(u.AA).value \
            for filt in phot_rest.filterset] * u.AA / (1. + phot_rest.z.value)
        ref_band = phot_rest.filterset.filt_names[np.argmin(np.abs(band_wavs - ref_wav))]
        ext_src_corr = phot_rest.ext_src_corrs[ref_band]
    else: # band given
        ext_src_corr = phot_rest.ext_src_corrs[ext_src_key]
    # apply limit to extended source correction
    if ext_src_uplim is not None:
        if ext_src_corr > ext_src_uplim:
            ext_src_corr = ext_src_uplim
    if ext_src_corr < 1.0:
        ext_src_corr = 1.0
    return ext_src_corr

def get_ext_src_corr_label(
    ext_src_key: Optional[str] = "UV",
    ext_src_uplim: Optional[Union[int, float]] = 10.0,
) -> str:
    """Build the catalogue column-name suffix identifying an extended-source correction configuration.

    Parameters
    ----------
    ext_src_key : `str`, optional
        Band or ``"UV"`` key identifying the correction; `None` disables the suffix. Default is ``"UV"``.
    ext_src_uplim : `int` or `float`, optional
        Upper limit applied to the correction, appended to the label if given. Default is 10.0.

    Returns
    -------
    `str`
        Label suffix, e.g. ``"_extsrc_UV<10"``, or ``""`` if ``ext_src_key`` is `None`.
    """
    if ext_src_key is None:
        return ""
    else:
        ext_src_name = f"_extsrc_{ext_src_key}"
        if ext_src_uplim is None:
            ext_src_lim_label = ""
        else:
            ext_src_lim_label = f"<{ext_src_uplim:.0f}"
        return ext_src_name + ext_src_lim_label

def truncate_colname(col):
    """Truncate a FITS column name to the 68-character FITS format limit.

    Attempts to shorten the name by stripping filter-name letter codes
    (``'F'``, ``'W'``, ``'M'``, ``'LP'``) from any embedded JWST/HST-style
    filter names before falling back to a hard truncation.

    Parameters
    ----------
    col : `str`
        Candidate column name.

    Returns
    -------
    `str`
        ``col`` unchanged if its length is <= 68 characters; otherwise a
        shortened name, hard-truncated to 68 characters if still too long.
    """
    out_colname = col
    if len(col) > 68:
        # truncate column name to 68 characters for fits format
        # remove 'F', 'W', 'M', 'LP' from filter names
        # find all filter names in column name
        filter_names = re.findall(r'F[0-9]+[WMLP]+', col)
        for filter_name in filter_names:
            out_colname = out_colname.replace(
                filter_name,
                filter_name.replace('F', '').replace('W', '').replace('M', '').replace('LP', '')
            )
        galfind_logger.warning(
            f"{col=} with {len(col)=}>68 is too long for fits format! " + \
            f"Using truncated {out_colname} instead!"
        )
        if len(out_colname) > 68:
            out_colname = out_colname[:68]
            galfind_logger.warning(
                f"Truncated {out_colname=} with {len(out_colname)=}>68!" +
                f"Further truncating to {out_colname}!"
            )
    return out_colname

def all_subclasses(cls):
    """Recursively collect every (in)direct subclass of a class.

    Parameters
    ----------
    cls : `type`
        Class to walk the subclass tree of.

    Returns
    -------
    `tuple` of `type`
        All (in)direct subclasses of ``cls``.
    """
    out = set()
    stack = [cls]
    while stack:
        parent = stack.pop()
        for sub in parent.__subclasses__():
            if sub not in out:
                out.add(sub)
                stack.append(sub)
    out = tuple(out)
    return out

def cat_from_gal(
    gal: Galaxy,
    data: Optional[Data],
    gal_creator_kwargs: Optional[Dict[str, Any]],
) -> Catalogue:
    """Build a length-1 `Catalogue` containing a single `Galaxy` object.

    Parameters
    ----------
    gal : `Galaxy`
        Galaxy to wrap in a catalogue.
    data : `Data`
        `Data` object the galaxy belongs to; must be an instance of `Data`.
    gal_creator_kwargs : `dict`, optional
        Keyword arguments used to construct a `Galaxy_Creator` if ``gal`` does
        not already have one attached.

    Returns
    -------
    `Catalogue`
        A `Catalogue` object containing only ``gal``.

    Raises
    ------
    AssertionError
        If ``data`` is not an instance of `Data`.
    """
    from . import Data, Galaxy_Creator, Catalogue
    assert isinstance(data, Data), \
        galfind_logger.critical(
            f"funcs.cat_from_gal requires {type(data)}==Data!"
        )
    # TODO: galaxy should already have an associated Galaxy_Creator
    # make catalogue of length 1 from galaxy object
    if hasattr(gal, "gal_creator"):
        gal_creator = gal.gal_creator
    else:
        galfind_logger.debug(
            f"{gal_creator_kwargs=} in funcs.cat_from_gal()"
        )
        gal_creator = Galaxy_Creator.from_data(
            data,
            gal.ID,
            **gal_creator_kwargs,
        )
    # BUG: Galaxy_Creator.__call__() produces Galaxy rather than Catalogue object
    cat = Catalogue([gal], gal_creator)
    return cat

# def _dicts_equal(d1, d2, name1="dict1", name2="dict2") -> bool:
#     """Iteratively compare two dicts, tolerating values that don't support
#     a simple boolean == (e.g. numpy arrays)."""
#     keys1, keys2 = set(d1.keys()), set(d2.keys())

#     only_in_1 = keys1 - keys2
#     only_in_2 = keys2 - keys1
#     if only_in_1 or only_in_2:
#         galfind_logger.critical(
#             f"Key mismatch between {name1} and {name2}: "
#             f"only in {name1}={only_in_1}, only in {name2}={only_in_2}"
#         )
#         return False

#     for key in keys1:
#         v1, v2 = d1[key], d2[key]
#         try:
#             equal = bool(v1 == v2)
#         except ValueError:
#             # e.g. numpy arrays -> elementwise comparison
#             equal = bool(np.array_equal(v1, v2))
#         if not equal:
#             galfind_logger.critical(f"{name1}[{key}]={v1!r} != {name2}[{key}]={v2!r}")
#             return False

#     return True

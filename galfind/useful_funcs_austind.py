
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
    from . import Selector, Filter, Multiple_Filter, Mask_Selector, Photometry_rest
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import astropy_cosmo, galfind_logger, config

# fluxes and magnitudes

def convert_wav_units(wavs, units):
    if units == wavs.unit:
        return wavs
    else:
        return wavs.to(units)


def convert_mag_units(wavs, mags, units):
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
    log_flux_unit = fluxes.unit
    log_fluxes = np.log10(fluxes.value)
    return log_fluxes


def log_scale_flux_errors(fluxes, flux_errs):  # removes unit
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
    x_pix, y_pix = skycoord_to_pixel(SkyCoord(ra, dec, unit=unit), wcs)
    flux, fluxerr, flag = sep.sum_circle(im_data, x_pix, y_pix, r)
    return flux  # image units


def calc_1sigma_flux(
    depth: Union[float, u.Magnitude],
    zero_point: float,
) -> float:
    if isinstance(depth, u.Magnitude):
        depth = depth.value
    flux_1sigma = (10 ** ((depth - zero_point) / -2.5)) / 5
    return flux_1sigma  # image units


def n_sigma_detection(
    depth,
    mag,
    zero_point,
):  # mag here is non aperture corrected
    flux_1sigma = calc_1sigma_flux(depth, zero_point)
    flux = 10 ** ((mag - zero_point) / -2.5)
    return flux / flux_1sigma


def flux_to_mag(flux, zero_point):
    try:
        flux = flux.value
    except:
        pass
    mag = -2.5 * np.log10(flux) + zero_point
    return mag


def mag_to_flux(mag, zero_point):
    flux = 10 ** ((mag - zero_point) / -2.5)
    return flux


def flux_to_mag_ratio(flux_ratio):
    mag_ratio = -2.5 * np.log10(flux_ratio)
    return mag_ratio


def mag_to_flux_ratio(mag_ratio):
    flux_ratio = 10 ** (mag_ratio / -2.5)
    return flux_ratio


def flux_pc_to_mag_err(flux_pc_err):
    mag_err = (
        2.5 * flux_pc_err / (np.log(10))
    )  # divide by 100 here to convert into percentage?
    return mag_err


def flux_image_to_Jy(fluxes, zero_points):
    # convert flux from image units to Jy
    if isinstance(fluxes, (list, np.ndarray,)):
        return (
            np.array(
                [
                    flux * (10 ** ((zero_points - 8.9) / -2.5))
                    for flux in fluxes
                ]
            )
            * u.Jy
        )
    else:
        return np.array(fluxes * (10 ** ((zero_points - 8.9) / -2.5))) * u.Jy


def five_to_n_sigma_mag(
    five_sigma_depth: Union[int, float, u.Magnitude],
    n: Union[int, float],
):
    assert n > 0, galfind_logger.critical(f"{n=} must be > 0")
    if isinstance(five_sigma_depth, u.Magnitude):
        five_sigma_depth = five_sigma_depth.value
    n_sigma_mag = -2.5 * np.log10(n / 5) + five_sigma_depth
    # flux_sigma = (10 ** ((five_sigma_depth - zero_point) / -2.5)) / 5
    # n_sigma_mag = -2.5 * np.log10(flux_sigma * n) + zero_point
    return n_sigma_mag


def flux_err_to_loc_depth(flux_err, zero_point):
    return -2.5 * np.log10(flux_err * 5) + zero_point


def loc_depth_to_flux_err(loc_depth, zero_point):
    return (10 ** ((loc_depth - zero_point) / -2.5)) / 5


# now in Photometry class!
# def flux_image_to_lambda(wav, flux, zero_point):
#     flux = flux_image_to_Jy(flux, zero_point)
#     flux_lambda = flux_to_lambda(wav, flux)
#     return flux_lambda # observed frame


def flux_Jy_to_lambda(
    flux_Jy, wav
):  # must already have associated astropy units
    return (flux_Jy * const.c / (wav**2)).to(
        u.erg / (u.s * (u.cm**2) * u.Angstrom)
    )


def flux_lambda_to_Jy(flux_lambda, wav):
    return (flux_lambda * (wav**2) / const.c).to(u.Jy)


def lum_nu_to_lum_lam(lum_nu, wav):
    return lum_nu * const.c / (wav**2)


def lum_lam_to_lum_nu(lum_wav, wav):
    return lum_wav * (wav**2) / const.c


def wav_obs_to_rest(wav_obs, z):
    wav_rest = wav_obs / (1 + z)
    return wav_rest


def wav_rest_to_obs(wav_rest, z):
    wav_obs = wav_rest * (1 + z)
    return wav_obs


def flux_lambda_obs_to_rest(flux_lambda_obs, z):
    flux_lambda_rest = flux_lambda_obs * (
        (1 + np.full(len(flux_lambda_obs), z)) ** 2
    )
    return flux_lambda_rest


def luminosity_to_flux(lum, wavs, z, cosmo=astropy_cosmo, out_units=u.Jy):

    """
        Input luminosity should be intrinsic (i.e. rest frame luminosity), leading to observed frame output flux units.
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
    
    """
        Input should be in observed frame units, leading to output intrinsic luminosity units.
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
    u.ABmag: r"$\mathrm{AB mag}$",
    u.Hz / u.erg: r"$\mathrm{Hz}\mathrm{erg}^{-1}$",
}

property_name_to_label = {
    "z": r"Redshift, $z$",
    "M_UV": r"$M_{\mathrm{UV}}$",
    "xi_ion_caseB_rest": r"$\xi_{\mathrm{ion,0}}~/~\mathrm{Hz}~\mathrm{erg}^{-1}$",
}


def label_log(label):
    return r"$\log_{10}($" + label + r"$)$"


def label_wavelengths(unit, is_log_scaled, frame):
    assert frame in ["", "rest", "obs"]
    wavelength_label = r"$\lambda_{%s}~/~$" % frame
    wavelength_label += unit_labels_dict[unit]
    if is_log_scaled:
        return label_log(wavelength_label)
    else:
        return wavelength_label


def label_fluxes(unit, is_log_scaled):
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
    return f"{z_bin[0]:.1f}<z<{z_bin[1]:.1f}"


def get_SED_fit_label_aper_diam_z_bin_name(
    SED_fit_params_key: str,
    aper_diam: u.Quantity,
    z_bin: Union[list, np.array]
) -> str:
    return f"{SED_fit_params_key}_{aper_diam.to(u.arcsec).value:.2f}as_{get_z_bin_name(z_bin)}"


def get_crop_name(crops: List[Selector]) -> str:
    if crops is not None:
        aper_diam = np.unique([selector.aper_diam.to(u.arcsec).value for selector \
            in crops if hasattr(selector, "aper_diam") and selector.aper_diam is not None])
        if len(aper_diam) == 1:
            aper_diam = aper_diam[0]
        else:
            aper_diam = None
        SED_fit_label = np.unique([selector.SED_fit_label for selector \
            in crops if hasattr(selector, "SED_fit_label") and selector.SED_fit_label is not None])
        if len(SED_fit_label) == 1:
            SED_fit_label = SED_fit_label[0]
        else:
            SED_fit_label = None
        if aper_diam is not None and SED_fit_label is not None:
            SED_fit_aper_diam_name = f"{SED_fit_label}_{aper_diam:.2f}as"
            out_crop_name = f"{SED_fit_aper_diam_name}/" + \
                "+".join([selector.name.replace( \
                f"_{SED_fit_aper_diam_name}", "") for selector in crops])
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
    return f"{survey}_{version}_{filterset.instrument_name}"

def calc_Vmax(area, zmin, zmax):
    return (
        (4 / 3 * np.pi)
        * (area / (4.0 * np.pi * u.sr))
        * (
            astropy_cosmo.comoving_distance(zmax) ** 3.0
            - astropy_cosmo.comoving_distance(zmin) ** 3.0
        )
    ).to(u.Mpc**3)


def poisson_interval(k, alpha=0.05):
    """
    uses chisquared info to get the poisson interval. Uses scipy.stats
    From https://stackoverflow.com/questions/14813530/poisson-confidence-interval-with-numpy
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
    # print("adjusting errors:", plot_data, code)
    data_l1 = data - data_err[0]
    data_u1 = data_err[1] - data
    data_err = np.vstack([data_l1, data_u1])
    return data, data_err


def errs_to_log(data, data_err, uplim_sigma = None, uplim_arrowsize = 0.2, inf_val = 1e6):
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
    if output == "dir":
        return "/".join(np.array(save_path.split("/")[:-1])) + "/"
    elif output == "name":
        return save_path.split("/")[-1]
    

def gauss_func(x, mu, sigma):
    return (np.pi * sigma) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def power_law_func(x, A, slope):
    return A * (x**slope)


def simple_power_law_func(x, c, m):
    return (m * x) + c


def cat_from_path(path, crop_names=None):
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
    forced_phot_band_name: Optional[str],
):
    save_dir = (
        f"{config['DEFAULT']['GALFIND_WORK']}/Catalogues/{version}/" + \
        f"{instrument_name}/{survey}/{aper_diams_to_str(aper_diams)}"
    )
    if forced_phot_band_name is None:
        forced_phot_band_name = ""
    else:
        forced_phot_band_name = f"_MASTER_Sel-{forced_phot_band_name}"
    save_name = f"{survey}{forced_phot_band_name}_{version}.fits"
    save_path = f"{save_dir}/{save_name}"
    make_dirs(save_path)
    return save_path


def fits_cat_to_np(
    fits_cat: Table,
    column_labels: List[str],
    reshape_by_aper_diams: bool = True
):
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
    if lowz_zmax != None:
        label = f"zmax={lowz_zmax:.1f}"
    else:
        label = "zfree"
    return label


def zmax_from_lowz_label(label):
    if label == "zfree":
        zmax = None
    else:
        zmax = float(label.replace("zmax=", ""))
    return zmax


def get_z_PDF_paths(
    fits_cat, IDs, codes, templates_arr, lowz_zmaxs, fits_cat_path=None
):
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
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]
    return str(n) + suffix


def date_finder(text: str):
    pattern = r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}[/-]\d{2}[/-]\d{4}\b"
    dates = re.findall(pattern, text)
    return dates


def validate_quantity(
    quant: Optional[Any],
    physical_type: str,
):
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
    return (10**A) * (wav_rest**beta)


def inspect_info():
    info = inspect.getframeinfo(inspect.stack()[1][0])
    return info.filename, info.function, info.lineno


def make_dirs(path, permissions=0o777):
    os.makedirs(split_dir_name(path, "dir"), exist_ok=True)
    try:
        os.chmod(split_dir_name(path, "dir"), permissions)
    except PermissionError:
        galfind_logger.warning(
            f"Could not change permissions of {path} to {oct(permissions)}."
        )


def change_file_permissions(path, permissions=0o777, log=False):
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
    if is_log_data:
        return data + np.log10(corr_factor)
    else:
        return data * corr_factor


def power_law_beta_func(wav, A, beta):
    return A * (wav**beta)


class Singleton(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance


# for __str__ methods
line_sep = "*" * 40 + "\n"
band_sep = "-" * 10 + "\n"

def aper_diams_to_str(aper_diams: u.Quantity):
    return f"({','.join([f'{aper_diam:.2f}' for aper_diam in aper_diams.value])})as"

def calc_unmasked_area(
    mask: Union[np.ndarray, Tuple[np.ndarray]],
    pixel_scale: u.Quantity
) -> u.Quantity:
    if isinstance(mask, tuple):
        mask = np.logical_and.reduce(mask)
    return ((np.sum(mask)) * (pixel_scale ** 2)).to(u.arcmin**2)

def sort_band_data_arr(band_data_arr: List[Type[Band_Data_Base]]):
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
    kernel = np.ones(window_size) / window_size
    return np.convolve(y_array, kernel, mode='valid')

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

# useful for rest frame SED property calculations

def get_first_bluewards_band(
    z: float,
    filterset: Multiple_Filter,
    ref_wav: u.Quantity,
    ignore_bands: Optional[Union[str, List[str]]] = None,
) -> Optional[str]:
    """
    Get the first band bluewards of a reference wavelength, ignoring required bands
    """
    # convert ignore_bands to List[str] if not already
    if ignore_bands is None:
        ignore_bands = []
    elif isinstance(ignore_bands, str):
        ignore_bands = [ignore_bands]
    first_band = None
    # bands already ordered from blue -> red
    for filt in filterset:
        upper_wav = filt.WavelengthUpper50
        if upper_wav < ref_wav * (1.0 + z):
            first_band = filt.band_name
            break
    return first_band

def get_first_redwards_band(
    z: float,
    filterset: Multiple_Filter,
    ref_wav: u.Quantity,
    ignore_bands: Optional[Union[str, List[str]]] = None,
) -> Optional[str]:
    """
    Get the first band redwards of a reference wavelength, ignoring required bands
    """
    # convert ignore_bands to List[str] if not already
    if ignore_bands is None:
        ignore_bands = []
    elif isinstance(ignore_bands, str):
        ignore_bands = [ignore_bands]
    first_band = None
    for filt in filterset:
        if filt.band_name not in ignore_bands:
            lower_wav = filt.WavelengthLower50
            if lower_wav > ref_wav * (1.0 + z):
                first_band = filt.band_name
                break
    return first_band

def group_positions(
    sky_coords: SkyCoord,
    match_radius: u.Quantity = 2.0 * u.arcsec
) -> Dict[int, List[int]]:
    """
    Group sky positions by proximity within a matching radius.

    Parameters
    ----------
    ra_list : array-like
        List of Right Ascensions in degrees.
    dec_list : array-like
        List of Declinations in degrees.
    match_radius : astropy.units.Quantity
        Matching radius (default 2 arcsec).

    Returns
    -------
    groups : dict
        Dictionary mapping group_id -> list of indices belonging to that group.
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
    """
    Parse S_REGION polygon string into matplotlib Polygon coordinates.
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
    """
    Extract matplotlib Polygons from files with S_REGION.
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
    """
    Performs linear least-squares fitting: y = slope * x + intercept.
    
    Parameters:
        x (ndarray): The independent variable (1D array).
        y (ndarray): The dependent variable (1D array).
        
    Returns:
        slope (float): The slope of the best-fit line.
        intercept (float): The intercept of the best-fit line.
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
    slope, intercept = linear_fit(x, y)
    return slope * x_out + intercept

@njit
def residual_sum_of_squares(params, x, y):
    """
    Calculate the residual sum of squares for a linear model y = mx + b.
    """
    m, c = params
    residuals = y - (m * x + c)
    return np.sum(residuals ** 2)

@njit
def gradient_descent_beta_fit(x, y, initial_params, learning_rate=0.01, max_iter=1000, tol=1e-6):
    """
    Perform gradient descent to minimize the residual sum of squares.
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
    make_dirs(symlink_path)
    if Path(target_path).is_file():
        try:
            os.symlink(target_path, symlink_path)
            galfind_logger.info(f"Created symlink: {symlink_path} -> {target_path}")
        except FileExistsError:
            galfind_logger.info(f"Symlink already exists: {symlink_path}")
    else:
        breakpoint()
        galfind_logger.warning(f"Target file does not exist for symlink: {target_path}")

def get_depth_dir(galfind_work_dir, survey, version, instrument_names):
    out_dirs = []
    for instrument_name in instrument_names:
        out_dirs.append(f"{galfind_work_dir}/Depths/{instrument_name}/{version}/{survey}")
    return np.array(out_dirs)

def get_eazy_dir(galfind_work_dir, survey, version, instrument_names):
    instrument_name = "+".join(instrument_names)
    out_dirs = []
    for subdir in ["input", "output"]:
        out_dirs.append(f"{galfind_work_dir}/EAZY/{subdir}/{instrument_name}/{version}/{survey}")
    return np.array(out_dirs)

def get_mask_dir(galfind_work_dir, survey):
    return np.array([f"{galfind_work_dir}/Masks/{survey}"])

def get_sex_dir(galfind_work_dir, survey, version, instrument_names):
    out_dirs = []
    for instrument_name in instrument_names:
        out_dirs.append(f"{galfind_work_dir}/SExtractor/{instrument_name}/{version}/{survey}")
    return np.array(out_dirs)

def get_stacked_images_dir(galfind_work_dir, survey, version, instrument_names):
    out_dirs = []
    for instrument_name in instrument_names:
        out_dirs.append(f"{galfind_work_dir}/Stacked_Images/{version}/{instrument_name}/{survey}")
    return np.array(out_dirs)

def find_target_dir(galfind_work_dir, survey, version, instrument_names, keyword):
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
        ref_band = phot_rest.filterset.band_names[np.argmin(np.abs(band_wavs - ref_wav))]
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
    if ext_src_key is None:
        return ""
    else:
        ext_src_name = f"_extsrc_{ext_src_key}"
        if ext_src_uplim is None:
            ext_src_lim_label = ""
        else:
            ext_src_lim_label = f"<{ext_src_uplim:.0f}"
        return ext_src_name + ext_src_lim_label
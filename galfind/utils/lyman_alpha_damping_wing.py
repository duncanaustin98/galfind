#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lyman-alpha damping wing calculations.

Miralda-Escude (1998) integral evaluation, background HI density calculations
for damping wing optical depth computation in Lyman-alpha systems.
"""

# lyman alpha damping wing and DLAs
import astropy.constants as const
import astropy.cosmology.units as cu
import astropy.units as u
import numpy as np
from scipy.special import wofz

from .. import astropy_cosmo
from .Emission_lines import wav_lyman_alpha

lambda_alpha_classical = (
    8
    * (np.pi * const.e.esu) ** 2
    / (3 * const.m_e * const.c * (wav_lyman_alpha * u.AA) ** 2)
).to(1 / u.s)
lyman_alpha_oscillator_strength = 0.4162
# print(lambda_alpha_classical)
lambda_alpha = lambda_alpha_classical * lyman_alpha_oscillator_strength
# print(lambda_alpha)
R_alpha = (lambda_alpha * wav_lyman_alpha * u.AA / (4 * np.pi * const.c)).to(
    u.dimensionless_unscaled
)
lyman_alpha_photon_absorption_const = (
    lyman_alpha_oscillator_strength
    * 4
    * np.sqrt(np.pi**3)
    * (const.e.esu**2)
    / (const.m_e * const.c * lambda_alpha)
).to(u.cm**2)
# print(lyman_alpha_photon_absorption_const)


def integral_result(x):
    """Evaluate the analytic antiderivative used in the Miralda-Escude (1998) damping-wing integral.

    Computes the closed-form result of the integral appearing in the
    calculation of the Lyman-alpha damping wing optical depth through an
    ionized bubble surrounded by neutral IGM (Miralda-Escude 1998).

    Parameters
    ----------
    x : `float` or array-like
        Dimensionless argument of the antiderivative, typically of the
        form ``(1 + z') / (1 + z_obs)``.

    Returns
    -------
    `float` or array-like
        Value of the antiderivative at `x`.
    """
    term_1 = (x ** (9 / 2)) / (1 - x)
    term_2 = 9 * (x ** (7 / 2)) / 7
    term_3 = 9 * (x ** (5 / 2)) / 5
    term_4 = 3 * (x ** (3 / 2))
    term_5 = 9 * (x ** (1 / 2))
    term_6 = -(9 / 2) * np.log10((1 + np.sqrt(x)) / (1 - np.sqrt(x)))
    return term_1 + term_2 + term_3 + term_4 + term_5 + term_6


def bg_HI_density(z, x_HI, helium_mass_frac, cosmo=astropy_cosmo):
    """Compute the mean background neutral hydrogen number density at redshift `z`.

    Scales the cosmic critical baryon density at `z` by the neutral
    hydrogen fraction and the hydrogen mass fraction (``1 - helium_mass_frac``).

    Parameters
    ----------
    z : `float`
        Redshift at which to evaluate the density.
    x_HI : `float`
        Volume-averaged neutral hydrogen fraction of the IGM.
    helium_mass_frac : `float`
        Cosmic helium mass fraction.
    cosmo : `astropy.cosmology.Cosmology`, optional
        Cosmology used to compute ``Ob0`` and the critical density.
        Default is `astropy_cosmo`.

    Returns
    -------
    `astropy.units.Quantity`
        Background neutral hydrogen number density, in `u.cm**-3`.
    """
    return (
        x_HI
        * (1 - helium_mass_frac)
        * cosmo.Ob0
        * ((1 + z) ** 3)
        * cosmo.critical_density0
        / (const.m_e + const.m_p)
    ).to(u.cm**-3)


def tau_GP(z, x_HI, helium_mass_frac, cosmo=astropy_cosmo):
    """Compute the Gunn-Peterson optical depth at redshift `z`.

    Parameters
    ----------
    z : `float`
        Redshift at which to evaluate the optical depth.
    x_HI : `float`
        Volume-averaged neutral hydrogen fraction of the IGM.
    helium_mass_frac : `float`
        Cosmic helium mass fraction.
    cosmo : `astropy.cosmology.Cosmology`, optional
        Cosmology used to compute the background HI density and Hubble
        parameter. Default is `astropy_cosmo`.

    Returns
    -------
    `astropy.units.Quantity`
        Dimensionless Gunn-Peterson optical depth at `z`.
    """
    n_HI = bg_HI_density(z, x_HI, helium_mass_frac, cosmo=cosmo)
    return (
        3
        * ((wav_lyman_alpha * u.AA) ** 3)
        * lambda_alpha
        * n_HI
        / (8 * np.pi * cosmo.H(z))
    ).to(u.dimensionless_unscaled)


def tau_DW(
    wav_rest,
    z_gal,
    R_b,
    x_HI,
    helium_mass_frac,
    z_re_end=6.0,
    cosmo=astropy_cosmo,
):
    """Compute the Lyman-alpha damping wing optical depth from the neutral IGM outside an ionized bubble.

    Implements the Miralda-Escude (1998) damping wing model for a galaxy
    at redshift `z_gal` sitting at the centre of an ionized bubble of
    comoving radius `R_b`, surrounded by a partially neutral IGM (neutral
    fraction `x_HI`) extending back to the end of reionization at
    `z_re_end`.

    Parameters
    ----------
    wav_rest : `astropy.units.Quantity`
        Rest-frame wavelength(s) at which to evaluate the optical depth.
    z_gal : `float`
        Redshift of the galaxy.
    R_b : `astropy.units.Quantity`
        Comoving radius of the ionized bubble surrounding the galaxy.
    x_HI : `float`
        Volume-averaged neutral hydrogen fraction of the IGM outside the
        bubble.
    helium_mass_frac : `float`
        Cosmic helium mass fraction.
    z_re_end : `float`, optional
        Redshift at the end of the reionization epoch, i.e. the far edge
        of the integration over neutral IGM. Default is `6.0`.
    cosmo : `astropy.cosmology.Cosmology`, optional
        Cosmology used for distance and Gunn-Peterson optical depth
        calculations. Default is `astropy_cosmo`.

    Returns
    -------
    `astropy.units.Quantity`
        Dimensionless Lyman-alpha damping wing optical depth at each
        wavelength in `wav_rest`.
    """
    tau_0 = tau_GP(z_gal, x_HI, helium_mass_frac, cosmo=cosmo)
    z_bubble_near = (cosmo.comoving_distance(z_gal) - R_b.to(u.Mpc)).to(
        cu.redshift, cu.redshift_distance(cosmo, kind="comoving")
    )
    z_obs = (wav_rest * (1 + z_gal) / (wav_lyman_alpha * u.AA)) - 1
    integral = integral_result(
        (1 + z_bubble_near) / (1 + z_obs)
    ) - integral_result((1 + z_re_end) / (1 + z_obs))
    return (
        tau_0
        * R_alpha
        * (((1 + z_obs) / (1 + z_gal)) ** (3 / 2))
        * integral
        / np.pi
    )


# proximate DLA system
def Doppler_parameter(gas_temp):
    """Compute the thermal Doppler broadening parameter of hydrogen gas.

    Parameters
    ----------
    gas_temp : `astropy.units.Quantity`
        Gas temperature.

    Returns
    -------
    `astropy.units.Quantity`
        Doppler parameter, ``b = sqrt(2 * k_B * gas_temp / m_p)``.
    """
    return np.sqrt(2 * const.k_B * gas_temp / const.m_p)


def delta_lambda_lyman_alpha_from_gas_temp(gas_temp):
    """Compute the Lyman-alpha Doppler wavelength width from a gas temperature.

    Parameters
    ----------
    gas_temp : `astropy.units.Quantity`
        Gas temperature.

    Returns
    -------
    `astropy.units.Quantity`
        Lyman-alpha Doppler wavelength width, via `Doppler_parameter` and
        `delta_lambda_lyman_alpha_from_b`.
    """
    return delta_lambda_lyman_alpha_from_b(Doppler_parameter(gas_temp))


def delta_lambda_lyman_alpha_from_b(b):
    """Compute the Lyman-alpha Doppler wavelength width from a Doppler parameter.

    Parameters
    ----------
    b : `astropy.units.Quantity`
        Doppler broadening parameter (velocity units).

    Returns
    -------
    `astropy.units.Quantity`
        Lyman-alpha Doppler wavelength width, ``(b / c) * wav_lyman_alpha``.
    """
    print(b.to(u.km / u.s))
    return (b / const.c) * wav_lyman_alpha * u.AA


def DLA_damping_param(delta_lambda):
    """Compute the dimensionless Voigt damping parameter for a DLA at the given Doppler width.

    Parameters
    ----------
    delta_lambda : `astropy.units.Quantity`
        Lyman-alpha Doppler wavelength width, e.g. from
        `delta_lambda_lyman_alpha_from_b`.

    Returns
    -------
    `astropy.units.Quantity`
        Dimensionless damping parameter, ``a = wav_lyman_alpha**2 *
        lambda_alpha / (4 * pi * c * delta_lambda)``.
    """
    return (
        ((wav_lyman_alpha * u.AA) ** 2)
        * lambda_alpha
        / (4 * np.pi * const.c * delta_lambda)
    ).to(u.dimensionless_unscaled)


def full_voigt_profile(x, alpha, gamma):
    """Evaluate the exact Voigt line profile using the Faddeeva function.

    Parameters
    ----------
    x : `float` or array-like
        Offset from line centre.
    alpha : `float`
        Gaussian (Doppler) half-width at half-maximum.
    gamma : `float`
        Lorentzian (damping) half-width at half-maximum.

    Returns
    -------
    `float` or array-like
        Value of the (unit-normalised) Voigt profile at `x`, computed via
        `scipy.special.wofz`.
    """
    sigma = alpha / np.sqrt(2 * np.log(2))
    return (
        np.real(wofz((x + 1j * gamma) / sigma / np.sqrt(2)))
        / sigma
        / np.sqrt(2 * np.pi)
    )


def Tepper_Garcia06_voigt_profile(a, x):
    """Evaluate the Tepper-Garcia (2006) approximation to the Voigt-Hjerting function, H(a, x).

    A fast pseudo-analytic approximation to the Voigt-Hjerting function,
    valid for small damping parameter `a`, used to model DLA/proximate-DLA
    Lyman-alpha absorption profiles.

    Parameters
    ----------
    a : `float`
        Dimensionless Voigt damping parameter.
    x : `float` or array-like
        Dimensionless offset from line centre in units of the Doppler
        width.

    Returns
    -------
    `float` or array-like
        Approximate value of the Voigt-Hjerting function, H(a, x).
    """
    x_sq = x**2
    H_0 = np.exp(-x_sq)
    Q = 1.5 / x_sq
    return H_0 - (a / (np.sqrt(np.pi) * x_sq)) * (
        ((H_0**2) * (4 * x_sq**2 + 7 * x_sq + 4 + Q)) - 1 - Q
    )


def Tepper_Garcia06_lyman_alpha_voigt_profile(wav_rest, delta_lambda):
    """Evaluate the Tepper-Garcia (2006) Voigt-Hjerting profile of the Lyman-alpha line at given wavelengths.

    Computes the Voigt damping parameter from `delta_lambda` and the
    dimensionless offset from the Lyman-alpha line centre, then evaluates
    `Tepper_Garcia06_voigt_profile`.

    Parameters
    ----------
    wav_rest : `astropy.units.Quantity`
        Rest-frame wavelength(s) at which to evaluate the profile.
    delta_lambda : `astropy.units.Quantity`
        Lyman-alpha Doppler wavelength width.

    Returns
    -------
    `float` or array-like
        Voigt-Hjerting function value, H(a, x), at each wavelength in
        `wav_rest`.
    """
    a = DLA_damping_param(delta_lambda)
    # print(a)
    x = ((wav_rest - wav_lyman_alpha * u.AA) / delta_lambda).to(
        u.dimensionless_unscaled
    )
    return Tepper_Garcia06_voigt_profile(a, x)


def tau_proximate_DLA(
    wav_rest, N_HI, delta_lambda, voigt_method="Tepper-Garcia+06"
):
    """Compute the Lyman-alpha optical depth of a proximate damped Lyman-alpha (DLA) system.

    Parameters
    ----------
    wav_rest : `astropy.units.Quantity`
        Rest-frame wavelength(s) at which to evaluate the optical depth.
    N_HI : `astropy.units.Quantity`
        Neutral hydrogen column density of the DLA system.
    delta_lambda : `astropy.units.Quantity`
        Lyman-alpha Doppler wavelength width of the absorbing gas.
    voigt_method : `str`, optional
        Method used to compute the Voigt-Hjerting profile. Only
        ``"Tepper-Garcia+06"`` is currently implemented. Default is
        ``"Tepper-Garcia+06"``.

    Returns
    -------
    `astropy.units.Quantity`
        Lyman-alpha optical depth of the DLA at each wavelength in
        `wav_rest`.
    """
    if voigt_method == "Tepper-Garcia+06":
        H_a_x = Tepper_Garcia06_lyman_alpha_voigt_profile(
            wav_rest, delta_lambda
        )
    tau = (
        N_HI
        * DLA_damping_param(delta_lambda)
        * lyman_alpha_photon_absorption_const
        * H_a_x
    )
    # print(N_HI, a, lyman_alpha_photon_absorption_const)
    return tau


def get_transmission(tau_arr):
    """Convert a set of optical depths into a combined transmission.

    Parameters
    ----------
    tau_arr : array-like
        Optical depths to be summed (e.g. contributions from the IGM
        damping wing and a proximate DLA).

    Returns
    -------
    `float`
        Combined transmission, ``exp(-sum(tau_arr))``.
    """
    return np.exp(-np.sum(tau_arr))

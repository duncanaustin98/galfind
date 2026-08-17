#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Damped Lyman-alpha absorption system models.

Stores HI column density, Doppler broadening, and velocity offset; computes
Voigt profiles and photon absorption cross-sections for DLA modeling.
"""

# DLA.py
import astropy.units as u

from .Emission_lines import line_diagnostics
from ..utils.lyman_alpha_damping_wing import *


class DLA:
    """Damped Lyman-Alpha (DLA) absorption system for spectral analysis.

    Models a damped Lyman-alpha system with specified HI column density,
    Doppler broadening, and velocity offset. Computes properties like
    Voigt profile parameters and photon absorption cross-sections.

    Parameters
    ----------
    N_HI : `astropy.units.Quantity`
        HI column density (number per unit area, typically cm^-2).
    Doppler_b : `astropy.units.Quantity`
        Doppler broadening parameter (velocity width), typically in km/s.
    vel_offset : `astropy.units.Quantity`, optional
        Velocity offset from the expected DLA velocity. Default is 0 km/s.
    z_offset : `float`, optional
        Redshift offset from the parent absorption. Default is 0.0.
    voigt_method : `str`, optional
        Method for computing Voigt profile ("Tepper-Garcia+06" or similar).
        Default is "Tepper-Garcia+06".

    Attributes
    ----------
    N_HI : `astropy.units.Quantity`
        HI column density.
    Doppler_b : `astropy.units.Quantity`
        Doppler broadening parameter.
    """
    def __init__(
        self,
        N_HI,
        Doppler_b,
        vel_offset = 0.0 * u.km / u.s,
        z_offset = 0.0,
        voigt_method="Tepper-Garcia+06",
    ):
        """Initialize a DLA absorption system.

        Parameters
        ----------
        N_HI : `astropy.units.Quantity`
            HI column density (cm^-2).
        Doppler_b : `astropy.units.Quantity`
            Doppler broadening parameter (velocity width, km/s).
        vel_offset : `astropy.units.Quantity`, optional
            Velocity offset from the expected DLA velocity. Default is 0 km/s.
        z_offset : `float`, optional
            Redshift offset from the parent absorption. Default is 0.0.
        voigt_method : `str`, optional
            Voigt profile computation method. Default is ``"Tepper-Garcia+06"``.
        """
        self.N_HI = N_HI
        self.Doppler_b = Doppler_b
        self.vel_offset = vel_offset
        self.z_offset = z_offset
        self.voigt_method = voigt_method

    @property
    def delta_lambda(self):
        """Wavelength broadening from the Doppler parameter.

        Returns
        -------
        `astropy.units.Quantity`
            Wavelength width of the Doppler broadening at Lyman-alpha.
        """
        return (self.Doppler_b / const.c) * line_diagnostics["Lya"]["line_wav"]

    @property
    def photon_absorption_const(self):
        """Photon absorption cross-section constant for Lyman-alpha.

        Returns
        -------
        `astropy.units.Quantity`
            Absorption cross-section coefficient (cm^2).
        """
        return (
            line_diagnostics["Lya"]["oscillator_strength"]
            * 4
            * np.sqrt(np.pi**3)
            * (const.e.esu**2)
            / (const.m_e * const.c * line_diagnostics["Lya"]["rel_lambda"])
        ).to(u.cm**2)

    @property
    def a(self):
        """Damping parameter of the DLA (dimensionless).

        Returns
        -------
        `astropy.units.Quantity`
            DLA damping parameter (dimensionless).
        """
        return (
            (line_diagnostics["Lya"]["line_wav"] ** 2)
            * line_diagnostics["Lya"]["rel_lambda"]
            / (4 * np.pi * const.c * self.delta_lambda)
        ).to(u.dimensionless_unscaled)

    @property
    def z_vel_offset(self):
        """Velocity offset converted to relativistic Doppler shift.

        Returns
        -------
        `astropy.units.Quantity`
            Velocity offset in km/s.
        """
        # copying astropy unit conversion from base code
        zponesq = (1 + self.z_offset) ** 2
        out = (const.c * (zponesq - 1) / (zponesq + 1)).to(u.km / u.s)
        return out
        # print((self.z_offset * u.dimensionless_unscaled).to(u.AA, equivalencies = u.doppler_redshift()))

    def tau(self, wav_rest):
        """Compute the optical depth as a function of rest-frame wavelength.

        Parameters
        ----------
        wav_rest : `astropy.units.Quantity`
            Rest-frame wavelength(s).

        Returns
        -------
        `astropy.units.Quantity`
            Optical depth (dimensionless).
        """
        if self.voigt_method == "Tepper-Garcia+06":
            x = (
                (
                    (self.z_vel_offset + self.vel_offset).to(
                        u.AA, equivalencies=u.doppler_relativistic(wav_rest)
                    )
                    - line_diagnostics["Lya"]["line_wav"]
                )
                / self.delta_lambda
            ).to(u.dimensionless_unscaled)
            H_a_x = Tepper_Garcia06_voigt_profile(self.a, x)
        tau = self.N_HI * self.a * self.photon_absorption_const * H_a_x
        return tau

    def transmission(self, wav_rest):
        """Compute the transmission (e^-tau) as a function of rest-frame wavelength.

        Parameters
        ----------
        wav_rest : `astropy.units.Quantity`
            Rest-frame wavelength(s).

        Returns
        -------
        `numpy.ndarray`
            Transmission (dimensionless, between 0 and 1).
        """
        return np.exp(-self.tau(wav_rest))

    def plot_transmission_profile(self, ax, wav_rest):
        """Plot the transmission profile on the given axes.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes on which to plot.
        wav_rest : `astropy.units.Quantity`
            Rest-frame wavelengths for plotting.
        """
        ax.plot(wav_rest, self.transmission(wav_rest))

    def plot_voigt_profile(self, ax, wav_rest):
        """Plot the Voigt profile on the given axes.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes on which to plot.
        wav_rest : `astropy.units.Quantity`
            Rest-frame wavelengths for plotting.
        """
        ax.plot(
            wav_rest,
            Tepper_Garcia06_lyman_alpha_voigt_profile(
                wav_rest, self.delta_lambda
            ),
        )

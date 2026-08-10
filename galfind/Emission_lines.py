#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Emission line diagnostics and wavelength definitions.

Stores Lyman-alpha and optical emission line parameters including wavelengths,
feature widths, and continuum band definitions for emission line analysis.
"""

# Emission_lines.py
import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

wav_lyman_alpha = 1215.67  # u.AA
line_diagnostics = {
    "Lya": {
        "line_wav": 1_215.67 * u.AA,
        "feature_wavs": [1_211.0, 1_221.0] * u.AA,
        "cont_wavs": [[1_226.0, 1_246.0]] * u.AA,
        "rel_lambda": 6.262e8 / u.s,
        "oscillator_strength": 0.4162,
    },
    "CIV-1549": {
        "line_wav": 1_549.0 * u.AA,
        "feature_wavs": [1_540.0, 1_556.0] * u.AA,
        "cont_wavs": [[1_528.0, 1_538.0], [1_556.0, 1_566.0]] * u.AA,
        "rel_lambda": None,
    },
    "HeII-1640": {
        "line_wav": 1_640.0 * u.AA,
        "feature_wavs": [1_630.0, 1_652.5] * u.AA,
        "cont_wavs": [[1_620.0, 1_630.0], [1_675.0, 1_685.0]] * u.AA,
        "rel_lambda": None,
    },
    "OIII]-1665": {
        "line_wav": 1_665.0 * u.AA,
        "feature_wavs": [1_652.5, 1_675.0] * u.AA,
        "cont_wavs": [[1_620.0, 1_630.0], [1_675.0, 1_685.0]] * u.AA,
        "rel_lambda": None,
    },
    "CIII]-1909": {
        "line_wav": 1_909.0 * u.AA,
        "feature_wavs": [1_901.0, 1_914.0] * u.AA,
        "cont_wavs": [[1_864.0, 1_874.0], [1_914.0, 1_924.0]] * u.AA,
        "rel_lambda": None,
    },
    "MgII]-2796,2803": {
        "line_wav": 2_800.0 * u.AA,
        "feature_wavs": [2_786.0, 2_813.0] * u.AA,
        "cont_wavs": [[2_776.0, 2_786.0], [2_813.0, 2_823.0]] * u.AA,
        "rel_lambda": None,
    },
    "[OII]-3727": {
        "line_wav": 3_727.0 * u.AA,
        "feature_wavs": [3_717.0, 3_737.0] * u.AA,
        "cont_wavs": [[3_707.0, 3_717.0], [3_737.0, 3_747.0]] * u.AA,
        "rel_lambda": None,
    },
    "[NeIII]-3869": {
        "line_wav": 3_868.8 * u.AA,
        "feature_wavs": [3_858.8, 3_878.8] * u.AA,
        "cont_wavs": [[3_848.8, 3_858.8], [3_878.8, 3_888.8]] * u.AA,
        "rel_lambda": None,
    },
    "Hgamma": {
        "line_wav": 4_340.5 * u.AA,
        "feature_wavs": [4_330.5, 4_350.5] * u.AA,
        "cont_wavs": [[4_320.5, 4_330.5], [4_350.5, 4_360.5]] * u.AA,
        "rel_lambda": None,
    },
    "Hbeta": {
        "line_wav": 4_861.3 * u.AA,
        "feature_wavs": [4_857.3, 4_865.3] * u.AA,
        "cont_wavs": [[4_852.3, 4_857.3], [4_865.3, 4_870.3]] * u.AA,
        "rel_lambda": None,
    },
    "[OIII]-4959": {
        "line_wav": 4_958.9 * u.AA,
        "feature_wavs": [4_948.9, 4_968.9] * u.AA,
        "cont_wavs": [[4_938.9, 4_948.9], [4_968.9, 4_978.9]] * u.AA,
        "rel_lambda": None,
    },
    "[OIII]-5007": {
        "line_wav": 5_006.8 * u.AA,
        "feature_wavs": [5_003.8, 5_008.8] * u.AA,
        "cont_wavs": [[4_999.8, 5_003.8], [5_022.8, 5_027.8]] * u.AA,
        "rel_lambda": None,
    },
    "[NII]-6548": {
        "line_wav": 6_548.0 * u.AA,
        "feature_wavs": [6_543.0, 6_553.0] * u.AA,
        "cont_wavs": [[6_533.0, 6_543.0], [6_553.0, 6_558.0]] * u.AA,
        "rel_lambda": None,
    },
    "Halpha": {
        "line_wav": 6_562.8 * u.AA,
        "feature_wavs": [6_558.0, 6_570.0] * u.AA,
        "cont_wavs": [[6_555.0, 6_558.0], [6_570.0, 6_575.0]] * u.AA,
        "rel_lambda": None,
    },
    "[NII]-6583": {
        "line_wav": 6_583.4 * u.AA,
        "feature_wavs": [6_575.0, 6_593.4] * u.AA,
        "cont_wavs": [[6_563.4, 6_573.4], [6_593.4, 6_603.4]] * u.AA,
        "rel_lambda": None,
    },
    "[SII]-6716": {
        "line_wav": 6_716.4 * u.AA,
        "feature_wavs": [6_706.4, 6_726.4] * u.AA,
        "cont_wavs": [[6_696.4, 6_706.4], [6_726.4, 6_736.4]] * u.AA,
        "rel_lambda": None,
    },
    "[SII]-6730": {
        "line_wav": 6_730.8 * u.AA,
        "feature_wavs": [6_720.8, 6_740.8] * u.AA,
        "cont_wavs": [[6_710.8, 6_720.8], [6_740.8, 6_750.8]] * u.AA,
        "rel_lambda": None,
    },
}

strong_optical_lines = ["Hbeta", "[OIII]-4959", "[OIII]-5007", "Halpha"]


class Emission_line:
    """Model for a single emission line with Voigt profile.

    Represents an emission line with specified flux and Doppler broadening,
    computing wavelength-dependent properties like the Voigt profile shape,
    damping parameter, and line profile.

    Parameters
    ----------
    line_name : `str`
        Name of the emission line (must be in ``line_diagnostics`` dict).
        Examples: "Lya", "Hbeta", "[OIII]-4959", etc.
    line_flux : `astropy.units.Quantity`
        Integrated flux of the emission line.
    Doppler_b : `astropy.units.Quantity`, optional
        Doppler broadening (velocity width). Default is 0 km/s.
    voigt_type : `str`, optional
        Method for computing Voigt profile. Currently supports "Tepper-Garcia+06".
        Default is "Tepper-Garcia+06".

    Attributes
    ----------
    line_diagnostics : `dict`
        Dictionary of line properties (wavelength, feature windows, continuum windows,
        oscillator strength, etc.) from the module-level `line_diagnostics`.
    """
    def __init__(
        self,
        line_name: str,
        line_flux: u.Quantity,
        Doppler_b: u.Quantity = 0.0 * u.km / u.s,
        voigt_type: str = "Tepper-Garcia+06",
    ):
        """Initialize an Emission_line instance.

        Parameters
        ----------
        line_name : `str`
            Name of the emission line (e.g. "Halpha", "Hbeta", etc.).
        line_flux : `astropy.units.Quantity`
            Total flux of the emission line.
        Doppler_b : `astropy.units.Quantity`, optional
            Doppler broadening (velocity width). Default is 0 km/s.
        voigt_type : `str`, optional
            Method for computing Voigt profile. Default is ``"Tepper-Garcia+06"``.
        """
        self.line_name = line_name
        self.line_flux = line_flux
        self.line_diagnostics = line_diagnostics[line_name]
        self.Doppler_b = Doppler_b
        self.voigt_type = voigt_type

    def __repr__(self):
        """Return a string representation of the emission line.

        Returns
        -------
        `str`
            String containing the instance dictionary.
        """
        # string representation of what is stored in this class
        return str(self.__dict__)

    @property
    def delta_lambda(self):
        """Wavelength broadening from the Doppler parameter.

        Returns
        -------
        `astropy.units.Quantity`
            Wavelength width of the Doppler broadening at the line wavelength.
        """
        return (self.Doppler_b / const.c) * self.line_diagnostics["line_wav"]

    @property
    def R(self):
        """Radiative damping parameter (dimensionless).

        Returns
        -------
        `float` or `astropy.units.Quantity`
            Radiative damping parameter, or 0.0 if not defined.
        """
        if self.line_diagnostics["rel_lambda"] == None:
            return 0.0
        else:
            return (
                self.line_diagnostics["rel_lambda"]
                * self.line_diagnostics["line_wav"]
                / (4 * np.pi * const.c)
            ).to(u.dimensionless_unscaled)

    @property
    def a(self):
        """Damping parameter (dimensionless).

        Returns
        -------
        `float` or `astropy.units.Quantity`
            Damping parameter, or 0.0 if not defined.
        """
        if self.line_diagnostics["rel_lambda"] == None:
            return 0.0
        else:
            return (
                (self.line_diagnostics["line_wav"] ** 2)
                * self.line_diagnostics["rel_lambda"]
                / (4 * np.pi * const.c * self.delta_lambda)
            ).to(u.dimensionless_unscaled)

    @property
    def line_profile(self):
        """Normalized emission line profile.

        Returns
        -------
        `dict`
            Dictionary with keys ``"wavs"`` and ``"flux"`` containing the
            computed and normalized line profile.
        """
        if self.voigt_type == "Tepper-Garcia+06":
            profile = self.Tepper_Garcia06_profile()
        else:
            raise (
                Exception(
                    f"voigt_type={self.voigt_type} not available. Must be one of ['Tepper-Garcia+06']!"
                )
            )
        # normalize profile
        line_flux = np.trapz(profile["flux"], profile["wavs"])
        profile["flux"] *= self.line_flux / line_flux
        return profile

    @property
    def line_width(self, lim=1e-4):
        """Wavelength range containing the line above a threshold.

        Parameters
        ----------
        lim : `float`, optional
            Flux threshold as a fraction of peak flux. Default is 1e-4.

        Returns
        -------
        `astropy.units.Quantity` or `None`
            Wavelength range of the line, or `None` if line_width is incomplete.
        """
        mask = self.line_profile["flux"] < lim * np.max(
            self.line_profile["flux"]
        )
        cont_wavs = self.wavs[mask]

    def Tepper_Garcia06_profile(self, bins=1_000):
        """Compute line profile using Tepper-Garcia+06 Voigt approximation.

        Parameters
        ----------
        bins : `int`, optional
            Number of wavelength bins to sample the profile over. Default is 1000.

        Returns
        -------
        `dict`
            Dictionary with keys ``"wavs"`` and ``"flux"`` containing the
            wavelength array and unnormalized line profile.
        """
        line_profile = {}
        line_profile["wavs"] = np.linspace(
            self.line_diagnostics["feature_wavs"][0],
            self.line_diagnostics["feature_wavs"][1],
            bins,
        )
        x = (
            (line_profile["wavs"] - self.line_diagnostics["line_wav"])
            / self.delta_lambda
        ).to(u.dimensionless_unscaled)
        x_sq = x**2
        H_0 = np.exp(-x_sq)
        Q = 1.5 / x_sq
        line_profile["flux"] = H_0 - (self.a / (np.sqrt(np.pi) * x_sq)) * (
            ((H_0**2) * (4 * x_sq**2 + 7 * x_sq + 4 + Q)) - 1 - Q
        )
        return line_profile

    def plot_profile(self, ax, kwargs={}, show=False, save=False):
        """Plot the normalized emission line profile on the given axes.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes on which to plot the profile.
        kwargs : `dict`, optional
            Keyword arguments to pass to `ax.plot`. Default is empty dict.
        show : `bool`, optional
            Whether to display the plot. Default is `False`.
        save : `bool`, optional
            Unused; present for interface compatibility. Default is `False`.
        """
        ax.plot(self.line_profile["wavs"], self.line_profile["flux"], **kwargs)
        if show:
            plt.legend()
            plt.show()

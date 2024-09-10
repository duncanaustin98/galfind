#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:36:38 2024

@author: austind
"""

# DLA.py
import astropy.units as u

from .lyman_alpha_damping_wing import *
from .Emission_lines import line_diagnostics


class DLA:
    def __init__(
        self,
        N_HI,
        Doppler_b,
        vel_offset=0.0 * u.km / u.s,
        z_offset=0.0,
        voigt_method="Tepper-Garcia+06",
    ):
        self.N_HI = N_HI
        self.Doppler_b = Doppler_b
        self.vel_offset = vel_offset
        self.z_offset = z_offset
        self.voigt_method = voigt_method

    @property
    def delta_lambda(self):
        return (self.Doppler_b / const.c) * line_diagnostics["Lya"]["line_wav"]

    @property
    def photon_absorption_const(self):
        return (
            line_diagnostics["Lya"]["oscillator_strength"]
            * 4
            * np.sqrt(np.pi**3)
            * (const.e.esu**2)
            / (const.m_e * const.c * line_diagnostics["Lya"]["rel_lambda"])
        ).to(u.cm**2)

    @property
    def a(self):  # DLA damping parameter
        return (
            (line_diagnostics["Lya"]["line_wav"] ** 2)
            * line_diagnostics["Lya"]["rel_lambda"]
            / (4 * np.pi * const.c * self.delta_lambda)
        ).to(u.dimensionless_unscaled)

    @property
    def z_vel_offset(self):
        # copying astropy unit conversion from base code
        zponesq = (1 + self.z_offset) ** 2
        out = (const.c * (zponesq - 1) / (zponesq + 1)).to(u.km / u.s)
        return out
        # print((self.z_offset * u.dimensionless_unscaled).to(u.AA, equivalencies = u.doppler_redshift()))

    # not sure this is completely accurate for non zero velocity offsets
    def tau(self, wav_rest):
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
        return np.exp(-self.tau(wav_rest))

    def plot_transmission_profile(self, ax, wav_rest):
        ax.plot(wav_rest, self.transmission(wav_rest))

    def plot_voigt_profile(self, ax, wav_rest):
        ax.plot(
            wav_rest,
            Tepper_Garcia06_lyman_alpha_voigt_profile(wav_rest, self.delta_lambda),
        )

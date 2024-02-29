#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 19:14:15 2023

@author: u92876da
"""

# Emission_lines.py
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
from scipy.interpolate import interp1d
from scipy.integrate import quad

wav_lyman_alpha = 1215.67 # u.AA
line_diagnostics = {
    "Lya": {"line_wav": 1_215.67 * u.AA, "feature_wavs": [1_211., 1_221.] * u.AA, "cont_wavs": [[1_226., 1_246.]] * u.AA, "rel_lambda": 6.262e8 / u.s, "oscillator_strength": 0.4162}, \
    "CIV-1549": {"line_wav": 1_549. * u.AA, "feature_wavs": [1_540., 1_556.] * u.AA, "cont_wavs": [[1_528., 1_538.], [1_556., 1_566.]] * u.AA, "rel_lambda": None}, \
    "HeII-1640": {"line_wav": 1_640. * u.AA, "feature_wavs": [1_630., 1_652.5] * u.AA, "cont_wavs": [[1_620., 1_630.], [1_675., 1_685.]] * u.AA, "rel_lambda": None}, \
    "OIII]-1665": {"line_wav": 1_665. * u.AA, "feature_wavs": [1_652.5, 1_675.] * u.AA, "cont_wavs": [[1_620., 1_630.], [1_675., 1_685.]] * u.AA, "rel_lambda": None}, \
    "CIII]-1909": {"line_wav": 1_909. * u.AA, "feature_wavs": [1_901., 1_914.] * u.AA, "cont_wavs": [[1_864., 1_874.], [1_914., 1_924.]] * u.AA, "rel_lambda": None}
}


class Emission_line:
    
    def __init__(self, line_name, line_flux, Doppler_b = 0. * u.km / u.s, voigt_type = "Tepper-Garcia+06"):
        self.line_name = line_name
        self.line_flux = line_flux
        self.line_diagnostics = line_diagnostics[line_name]
        self.Doppler_b = Doppler_b
        self.voigt_type = voigt_type
        
    def __repr__(self):
        # string representation of what is stored in this class
        return str(self.__dict__)
    
    @property
    def delta_lambda(self):
        return (self.Doppler_b / const.c) * self.line_diagnostics["line_wav"]
    
    @property
    def R(self):
        if self.line_diagnostics["rel_lambda"] == None:
            return 0.
        else:
            return (self.line_diagnostics["rel_lambda"] * self.line_diagnostics["line_wav"] / (4 * np.pi * const.c)).to(u.dimensionless_unscaled)
    
    @property
    def a(self):
        if self.line_diagnostics["rel_lambda"] == None:
            return 0.
        else:
            return ((self.line_diagnostics["line_wav"] ** 2) * self.line_diagnostics["rel_lambda"] / (4 * np.pi * const.c * self.delta_lambda)).to(u.dimensionless_unscaled)
    
    @property
    def line_profile(self):
        if self.voigt_type == "Tepper-Garcia+06":
            profile = self.Tepper_Garcia06_profile()
        else:
            raise(Exception(f"voigt_type={self.voigt_type} not available. Must be one of ['Tepper-Garcia+06']!"))
        # normalize profile
        line_flux = np.trapz(profile["flux"], profile["wavs"])
        profile["flux"] *= self.line_flux / line_flux
        return profile
    
    @property
    def line_width(self, lim = 1e-4):
        mask = (self.line_profile["flux"] < lim * np.max(self.line_profile["flux"]))
        cont_wavs = self.wavs[mask]
    
    def Tepper_Garcia06_profile(self, bins = 1_000):
        line_profile = {}
        line_profile["wavs"] = np.linspace(self.line_diagnostics["feature_wavs"][0], self.line_diagnostics["feature_wavs"][1], bins)
        x = ((line_profile["wavs"] - self.line_diagnostics["line_wav"]) / self.delta_lambda).to(u.dimensionless_unscaled)
        x_sq = x ** 2
        H_0 = np.exp(-x_sq)
        Q = 1.5 / x_sq
        line_profile["flux"] = H_0 - (self.a / (np.sqrt(np.pi) * x_sq)) * (((H_0 ** 2) * (4 * x_sq ** 2 + 7 * x_sq + 4 + Q)) - 1 - Q)
        return line_profile
    
    def plot_profile(self, ax, kwargs = {}, show = False, save = False):
        ax.plot(self.line_profile["wavs"], self.line_profile["flux"], **kwargs)
        if show:
            plt.legend()
            plt.show()
        
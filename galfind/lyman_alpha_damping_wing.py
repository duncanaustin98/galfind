#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:03:59 2023

@author: u92876da
"""

# lyman alpha damping wing and DLAs
import numpy as np
import astropy.constants as const
import astropy.units as u
import astropy.cosmology.units as cu
from scipy.special import wofz

from . import astropy_cosmo
from .Emission_lines import wav_lyman_alpha

lambda_alpha_classical = (8 * (np.pi * const.e.esu) ** 2 / (3 * const.m_e * const.c * (wav_lyman_alpha * u.AA) ** 2)).to(1 / u.s)
lyman_alpha_oscillator_strength = 0.4162
#print(lambda_alpha_classical)
lambda_alpha = lambda_alpha_classical * lyman_alpha_oscillator_strength
#print(lambda_alpha)
R_alpha = (lambda_alpha * wav_lyman_alpha * u.AA / (4 * np.pi * const.c)).to(u.dimensionless_unscaled)
lyman_alpha_photon_absorption_const = (lyman_alpha_oscillator_strength * 4 * np.sqrt(np.pi ** 3) * (const.e.esu ** 2) / (const.m_e * const.c * lambda_alpha)).to(u.cm ** 2)
#print(lyman_alpha_photon_absorption_const)

def integral_result(x):
    term_1 = (x ** (9 / 2)) / (1 - x)
    term_2 = 9 * (x ** (7 / 2)) / 7
    term_3 = 9 * (x ** (5 / 2)) / 5
    term_4 = 3 * (x ** (3 / 2))
    term_5 = 9 * (x ** (1 / 2))
    term_6 = - (9 / 2) * np.log10((1 + np.sqrt(x)) / (1 - np.sqrt(x)))
    return term_1 + term_2 + term_3 + term_4 + term_5 + term_6

def bg_HI_density(z, x_HI, helium_mass_frac, cosmo = astropy_cosmo):
    return (x_HI * (1 - helium_mass_frac) * cosmo.Ob0 * ((1 + z) ** 3) * cosmo.critical_density0 / (const.m_e + const.m_p)).to(u.cm ** -3)

def tau_GP(z, x_HI, helium_mass_frac, cosmo = astropy_cosmo):
    n_HI = bg_HI_density(z, x_HI, helium_mass_frac, cosmo = cosmo)
    return (3 * ((wav_lyman_alpha * u.AA) ** 3) * lambda_alpha * n_HI / (8 * np.pi * cosmo.H(z))).to(u.dimensionless_unscaled)

def tau_DW(wav_rest, z_gal, R_b, x_HI, helium_mass_frac, z_re_end = 6., cosmo = astropy_cosmo):
    tau_0 = tau_GP(z_gal, x_HI, helium_mass_frac, cosmo = cosmo)
    z_bubble_near = (cosmo.comoving_distance(z_gal) - R_b.to(u.Mpc)).to(cu.redshift, cu.redshift_distance(cosmo, kind="comoving"))
    z_obs = (wav_rest * (1 + z_gal) / (wav_lyman_alpha * u.AA)) - 1
    integral = integral_result((1 + z_bubble_near) / (1 + z_obs)) - integral_result((1 + z_re_end) / (1 + z_obs))
    return tau_0 * R_alpha * (((1 + z_obs) / (1 + z_gal)) ** (3 / 2)) * integral / np.pi

# proximate DLA system
def Doppler_parameter(gas_temp):
    return np.sqrt(2 * const.k_B * gas_temp / const.m_p)

def delta_lambda_lyman_alpha_from_gas_temp(gas_temp):
    return delta_lambda_lyman_alpha_from_b(Doppler_parameter(gas_temp))

def delta_lambda_lyman_alpha_from_b(b):
    print(b.to(u.km / u.s))
    return (b / const.c) * wav_lyman_alpha * u.AA

def DLA_damping_param(delta_lambda):
    return (((wav_lyman_alpha * u.AA) ** 2) * lambda_alpha / (4 * np.pi * const.c * delta_lambda)).to(u.dimensionless_unscaled)

def full_voigt_profile(x, alpha, gamma):
    sigma = alpha / np.sqrt(2 * np.log(2))
    return np.real(wofz((x + 1j * gamma) / sigma / np.sqrt(2))) / sigma / np.sqrt(2 * np.pi)

def Tepper_Garcia06_voigt_profile(a, x):
    x_sq = x ** 2
    H_0 = np.exp(-x_sq)
    Q = 1.5 / x_sq
    return H_0 - (a / (np.sqrt(np.pi) * x_sq)) * (((H_0 ** 2) * (4 * x_sq ** 2 + 7 * x_sq + 4 + Q)) - 1 - Q)

def Tepper_Garcia06_lyman_alpha_voigt_profile(wav_rest, delta_lambda):
    a = DLA_damping_param(delta_lambda)
    #print(a)
    x = ((wav_rest - wav_lyman_alpha * u.AA) / delta_lambda).to(u.dimensionless_unscaled)
    return Tepper_Garcia06_voigt_profile(a, x)

def tau_proximate_DLA(wav_rest, N_HI, delta_lambda, voigt_method = "Tepper-Garcia+06"):
    if voigt_method == "Tepper-Garcia+06":
        H_a_x = Tepper_Garcia06_lyman_alpha_voigt_profile(wav_rest, delta_lambda)
    tau = N_HI * DLA_damping_param(delta_lambda) * lyman_alpha_photon_absorption_const * H_a_x
    #print(N_HI, a, lyman_alpha_photon_absorption_const)
    return tau

def get_transmission(tau_arr):
    return np.exp(-np.sum(tau_arr))


    
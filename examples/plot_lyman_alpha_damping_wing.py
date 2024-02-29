#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:04:23 2023

@author: u92876da
"""

# plot_lyman_alpha_damping_wing.py
from astropy.units import u
import matplotlib.pyplot as plt

from galfind.lyman_alpha_damping_wing import *
from galfind import wav_lyman_alpha, DLA
from galfind.Emission_lines import Emission_line

def plot_transmission(ax, ax2, wav_rest, trans, z, label = None, annotate = True, save = True, show = False, legend_kwargs = {}):
    ax.plot(wav_rest, trans, label = label)
    ax2.plot(wav_rest.to(u.km / u.s, equivalencies = u.doppler_relativistic(wav_lyman_alpha * u.AA)), trans)
    if annotate:
        ax.axvline(wav_lyman_alpha, 0, 1, c = "red", lw = 3, ls = "dotted")
        ax.set_xlabel(r"$\lambda_{\mathrm{rest}}~/~\mathrm{\AA}$")
        ax2.set_xlabel(r"$\Delta v_{\mathrm{Ly\alpha}}~/~\mathrm{kms}^{-1}$") #(r"$\lambda_{\mathrm{obs}}~/~\mathrm{\AA}$")
        ax.set_ylabel("Transmission")
    if show:
        ax.legend(**legend_kwargs)
        plt.show()  

def plot_lyman_alpha_damping_wing():
    z_arr = [8., 8., 8., 8.]
    x_HI_arr = [0.5, 0.5, 0.5, 0.5]
    helium_mass_frac_arr = [0.25, 0.25, 0.25, 0.25]
    R_b_arr = [0., 0., 0., 0.] * u.Mpc
    wav_rest = np.linspace(1_200., 1_268., 100) * u.AA
    velocity_offset_arr = [0., 200., 400., 800.] * u.km / u.s
    legend_kwargs = {"bbox_to_anchor": (0.5, -0.2), "loc": "upper center"}
    
    fig, ax = plt.subplots()
    ax2 = ax.twiny()
    for i, (z, R_b, x_HI, helium_mass_frac, velocity_offset) in enumerate(zip(z_arr, R_b_arr, x_HI_arr, helium_mass_frac_arr, velocity_offset_arr)):
        trans = [get_transmission(tau_DW(velocity_offset.to(u.AA, equivalencies = u.doppler_relativistic(wav)), z, R_b, x_HI, helium_mass_frac)) for wav in wav_rest]
        plot_transmission(ax, ax2, wav_rest, trans, z, label = f"z={z}, R_b={R_b}, x_HI={x_HI}, Y={helium_mass_frac}, Δv={velocity_offset}", show = True if i == len(z_arr) - 1 else False, legend_kwargs = legend_kwargs)
    
def plot_proximate_DLA():
    z_arr = [8., 8.]
    gas_temp_arr = [1e7, 1e7] * u.K
    N_HI_arr = [10 ** 21.5, 10 ** 23.] / (u.cm ** 2)
    x_HI_arr = [1., 1.]
    R_b_arr = [0., 0., 0., 0.] * u.Mpc
    helium_mass_frac_arr = [0.25, 0.25, 0.25, 0.]
    velocity_offset_arr = [0., 0.] * u.km / u.s
    wav_rest = np.linspace(1_216., 1_400., 1000) * u.AA
    legend_kwargs = {"bbox_to_anchor": (0.5, -0.2), "loc": "upper center"}
    
    fig, ax = plt.subplots()
    ax2 = ax.twiny()
    for i, (z, x_HI, gas_temp, N_HI, R_b, x_HI, helium_mass_frac, velocity_offset) in enumerate(zip(z_arr, x_HI_arr, gas_temp_arr, N_HI_arr, R_b_arr, x_HI_arr, helium_mass_frac_arr, velocity_offset_arr)):
        delta_lambda = delta_lambda_lyman_alpha_from_gas_temp(gas_temp)
        #delta_lambda = delta_lambda_lyman_alpha_from_b(250 * u.km / u.s)
        # , tau_DW(wav, z, R_b, x_HI, helium_mass_frac)
        trans = [get_transmission([tau_proximate_DLA(velocity_offset.to(u.AA, equivalencies = u.doppler_relativistic(wav)), N_HI, delta_lambda)]) for wav in wav_rest]
        plot_transmission(ax, ax2, wav_rest, trans, z, label = f"DLA: T={gas_temp}, N_HI={N_HI}, Δv={velocity_offset}", show = True if i == len(z_arr) - 1 else False, legend_kwargs = legend_kwargs)

def plot_voigt_profile():
    gas_temp = 1e7 * u.K # thermal + turbulent
    b = 100 * u.km / u.s
    wav_rest = np.linspace(1_213., 1_218., 1000) * u.AA
    #delta_lambda = delta_lambda_lyman_alpha_from_gas_temp(gas_temp)
    delta_lambda = delta_lambda_lyman_alpha_from_b(b)
    print(delta_lambda.to(u.AA))
    H_a_x = Tepper_Garcia06_lyman_alpha_voigt_profile(wav_rest, delta_lambda)
    plt.plot(wav_rest, H_a_x)
    
    #plt.xlabel("v / km/s")
    plt.show()

def plot_obj_DLA(log_N_HI_arr = [21., 22., 22.5, 23.]):
    fig, ax = plt.subplots()
    wav_rest = np.linspace(1216., 1400., 1_000) * u.AA
    for log_N_HI in log_N_HI_arr:
        dla = DLA(10 ** log_N_HI / (u.cm ** 2), 150 * u.km / u.s, 0. * u.km / u.s, 0.)
        dla.plot_transmission_profile(ax, wav_rest)

if __name__ == "__main__":
    # lya = Emission_line("Lya", Doppler_b = 200. * u.km / u.s)
    # fig, ax = plt.subplots()
    # lya.plot_profile(ax)
    
    plot_obj_DLA()
    
    #plot_voigt_profile()
    #plot_lyman_alpha_damping_wing()
    #plot_proximate_DLA()
# reionization.py (functional)

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const

# set up astropy cosmology object
from astropy.cosmology import FlatLambdaCDM, Planck18
global astropy_cosmo
astropy_cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3, Tcmb0 = 2.725, Ob0 = 0.05)
#astropy_cosmo = Planck18
global Hydrogen_abundance
Hydrogen_abundance = 0.75

# unit converter functions
def M_UV_to_L_UV(M_UV): # redshift independent
    return ((400. * np.pi * 10 ** ((M_UV - u.Jy.to(u.ABmag)) / -2.5)) * u.Jy * u.pc ** 2).to(u.erg / (u.s * u.Hz)).value

# power law parametrizations
def power_law(x, A, slope):
    return A + x * slope

# fesc(M_UV) parametrizations

def flat_fesc(z, M_UV, fesc = 0.2):
    return np.full(len(M_UV), fesc)

def EPOCHS_III_fesc(z, M_UV):
    pass

# ξ_ion(M_UV) parametrizations

def flat_xi_ion(z, M_UV, log_xi_ion = 25.2): # u.Hz / u.erg
    return np.full(len(M_UV), 10. ** log_xi_ion)

# ρ_UV parametrizations

def schechter_function(M_UV, M_star, phi_star, alpha):
    return (phi_star * np.log(10) / 2.5) * ((10. ** (0.4 * (M_star - M_UV))) \
        ** (alpha + 1.)) * np.exp(-(10. ** (0.4 * (M_star - M_UV))))

def Bouwens15_UVLF_evolution(z, M_UV):
    # calculate UVLF params
    M_star = -20.95 + 0.01 * (z - 6.)
    phi_star = 0.47 * 10 ** (-0.27 * (z - 6.) - 3.) # * u.Mpc ** -3
    alpha = -1.87 - 0.1 * (z - 6.)
    return schechter_function(M_UV, M_star, phi_star, alpha)

def plot_UVLF(ax, z, M_UV, parametrization = "Bouwens+15", legend = True, save = True):
    save_path = f"z={z}, {parametrization}"
    ax.plot(M_UV, np.log10(Bouwens15_UVLF_evolution(z, M_UV)), label = save_path)
    save_path = save_path + "_UVLF.png"
    ax.set_xlabel(r"$M_{UV}$")
    ax.set_ylabel(r"$\log_{10}(\Phi~/~N~\mathrm{mag}^{-1}~\mathrm{Mpc}^{-3})$")
    if legend:
        ax.legend()
    if save:
        plt.savefig(save_path)
        print(f"Written to {save_path}")

# C_HII parametrizations
        
def const_clumping_factor(z, clumping_factor = 3.):
    return clumping_factor

def Shull2012_clumping_factor(z):
    return 2.9 * ((1. + z) / 6.) ** -1.1

# α_Α/Β parametrizations

def case_B_recomb_coeff(electron_temp):
    if electron_temp == 1e4 * u.K:
        return 2.6e-13 * u.cm ** 3 / u.s
    else:
        raise(Exception())

# calculate n_dot_ion
def calc_n_dot_ion(z, M_UV_lim, fesc_func = flat_fesc, xi_ion_func = flat_xi_ion, UVLF_func = Bouwens15_UVLF_evolution):
    M_UV = np.linspace(-30., M_UV_lim, 1_000)
    integrand = fesc_func(z, M_UV) * xi_ion_func(z, M_UV) * UVLF_func(z, M_UV) * M_UV_to_L_UV(M_UV)
    return np.trapz(integrand, M_UV) / (u.s * u.Mpc ** 3)

# calculate n_dot_ion required to have stable Q_HII in Universe at redshift, z
def n_dot_ion_stable_reionization(z, Hydrogen_abundance, astropy_cosmo, Q_HII, electron_temp = 1e4 * u.K, \
        recomb_coeff_func = case_B_recomb_coeff, clumping_factor_func = Shull2012_clumping_factor):
    return (calc_n_H(z, Hydrogen_abundance, astropy_cosmo) * Q_HII / calc_trec(z, Hydrogen_abundance, \
        astropy_cosmo, electron_temp, recomb_coeff_func, clumping_factor_func)).to(1 / (u.s * u.Mpc ** 3)).value

def plot_n_dot_ion_vs_M_UV_lim(ax, z_arr, M_UV_lim_arr, Hydrogen_abundance, astropy_cosmo, stable_Q_HII_lines = [], \
        fesc_func = flat_fesc, xi_ion_func = flat_xi_ion, UVLF_func = Bouwens15_UVLF_evolution, electron_temp = 1e4 * u.K, \
        recomb_coeff_func = case_B_recomb_coeff, clumping_factor_func = Shull2012_clumping_factor, legend = True, save = True):
    save_path = f"z={z_arr}_n_dot_ion_vs_M_UV_lim.png"
    if type(z_arr) in [int, float]:
        z_arr = [z_arr]
    for z in z_arr:
        n_dot_ion_arr = [calc_n_dot_ion(z, M_UV_lim, fesc_func, xi_ion_func, UVLF_func).value for M_UV_lim in M_UV_lim_arr]
        plot = ax.plot(M_UV_lim_arr, np.log10(n_dot_ion_arr), label = f"z = {z}")
        for Q_HII in stable_Q_HII_lines:
            n_dot_stable_ion = n_dot_ion_stable_reionization(z, Hydrogen_abundance, astropy_cosmo, \
                Q_HII, electron_temp, recomb_coeff_func, clumping_factor_func)
            ax.axhline(np.log10(n_dot_stable_ion), 0, 1, c = plot[0].get_color(), ls = "--")
    ax.set_xlabel(r"$M_{UV}$ integration limit")
    ax.set_ylabel(r"$\log_{10}(\dot{n}_{\mathrm{ion}}~/~s^{-1}~\mathrm{Mpc}^{-3})$")
    if legend:
        ax.legend()
    if save:
        plt.savefig(save_path)
        print(f"Written to {save_path}")

# calculate hydrogen density
def calc_n_H(z, Hydrogen_abundance, astropy_cosmo):
    return (astropy_cosmo.Ob(z) * Hydrogen_abundance * astropy_cosmo.critical_density(z) / const.m_p).to(u.cm ** -3)

def calc_approx_n_H(z, Hydrogen_abundance, astropy_cosmo):
    return (((1. + z) ** 3) * (1.1e-5 * u.cm ** -3) * Hydrogen_abundance * astropy_cosmo.Ob0 * (astropy_cosmo.H0 / 100).value ** 2).to(u.cm ** -3)

def calc_approx_n_H_v2(z, astropy_cosmo):
    return (((1. + z) ** 3) * (1.6e-7 * u.cm ** -3) * (astropy_cosmo.Ob0 / 0.019) * (astropy_cosmo.H0 / 100).value ** 2).to(u.cm ** -3)

# calculate recombination timescale
def calc_trec(z, Hydrogen_abundance, astropy_cosmo, electron_temp = 1e4 * u.K, recomb_coeff_func \
        = case_B_recomb_coeff, clumping_factor_func = Shull2012_clumping_factor):
    return (1. / ((1. + (1. - Hydrogen_abundance) / (4. * Hydrogen_abundance)) * \
        calc_n_H(z, Hydrogen_abundance, astropy_cosmo) * recomb_coeff_func(electron_temp) * clumping_factor_func(z))).to(u.s)

def calc_stable_Q_HII_z(Q_HII, M_UV_lim, Hydrogen_abundance, astropy_cosmo, fesc_func = flat_fesc, \
        xi_ion_func = flat_xi_ion, UVLF_func = Bouwens15_UVLF_evolution, electron_temp = 1e4 * u.K, \
        recomb_coeff_func = case_B_recomb_coeff, clumping_factor_func = Shull2012_clumping_factor, z_start = 20.):
    # n_dot_ion * t_rec = n_H -> calculate z_end via Monte-Carlo
    z_arr = np.linspace(0., z_start, 1_000)
    stable_Q_HII_list = [(calc_n_dot_ion(z, M_UV_lim, fesc_func, xi_ion_func, UVLF_func) * \
        calc_trec(z, Hydrogen_abundance, astropy_cosmo, electron_temp, recomb_coeff_func, \
        clumping_factor_func) / calc_n_H(z, Hydrogen_abundance, astropy_cosmo)).to(u.dimensionless_unscaled).value for z in z_arr]
    return z_arr[int(min(((i, value) for i, value in enumerate(stable_Q_HII_list)), key = lambda x : abs(x[1] - Q_HII))[0])]

def main():
    # plot UV LF
    z_arr = [5., 6., 7., 8., 9., 10.]
    M_UV = np.linspace(-21., -15., 100)
    # parametrization = "Bouwens+15"
    # fig, ax = plt.subplots()
    # for i, z in enumerate(z_arr):
    #     plot_UVLF(ax, z, M_UV, parametrization = parametrization, legend = True if z == z_arr[-1] else False, save = False)
    # plt.savefig(f"z={z_arr}_{parametrization}_UVLF.png")

    fig, ax = plt.subplots()
    M_UV_lim_arr = np.linspace(-22., -10., 1_000)
    plot_n_dot_ion_vs_M_UV_lim(ax, z_arr, M_UV_lim_arr, Hydrogen_abundance, astropy_cosmo, stable_Q_HII_lines = [1.])

    print(calc_stable_Q_HII_z(1., -13., Hydrogen_abundance, astropy_cosmo))

if __name__ == "__main__":
    main()
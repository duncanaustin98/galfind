#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 19:06:11 2023

@author: u92876da
"""

# lyman_alpha_beta_bias.py
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from galfind import (
    Emission_line,
    Mock_SED_obs,
    Mock_SED_rest,
    NIRCam,
    Photometry_rest,
)


def moving_average(x, n=3):
    ret = np.cumsum(x, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_bias(beta=-3.0, m_UV=26.0):
    non_JADES_bands = [
        "f070W",
        "f140M",
        "f162M",
        "f182M",
        "f210M",
        "f250M",
        "f300M",
        "f335M",
        "f360M",
        "f430M",
        "f460M",
        "f480M",
    ]
    redshifts = [6.0]  # np.linspace(6.5, 13., 100)
    eta_arr = [
        0.0
    ]  # [-0.15, -0.1, -0.05] #, 0.1, 0.05, 0.] #np.linspace(-0.15, 0.15, 10)#[0., -0.15, 0.15]
    line_widths = (
        [6e-16] * u.erg / (u.s * u.cm**2)
    )  # 1e-16 == 200 AA EW at m_UV == 28. + linear scaling
    plot = False
    fig, ax = plt.subplots()
    for line_width in line_widths:
        lya = Emission_line("Lya", line_width, 150.0 * u.km / u.s)
        for eta in eta_arr:
            delta_beta_arr = []
            for redshift in tqdm(redshifts):
                mock_sed_rest = Mock_SED_rest.power_law_from_beta_m_UV(
                    beta, m_UV, wav_res=0.1
                )
                mock_sed_rest.add_emission_lines([lya])
                mock_sed_rest.calc_line_EW("Lya")
                print(
                    mock_sed_rest.line_EWs,
                    mock_sed_rest.line_cont,
                    mock_sed_rest.line_fluxes,
                )
                # mock_sed_rest.plot_SED(ax, mag_units = u.erg / (u.s * u.cm ** 2 * u.AA))
                EW = mock_sed_rest.calc_line_EW(
                    "Lya"
                )  # mock_sed_rest.line_cont["Lya"], mock_sed_rest.line_fluxes["Lya"],
                mock_sed_obs = Mock_SED_obs.from_Mock_SED_rest(
                    mock_sed_rest, redshift
                )
                mock_sed_obs.calc_line_EW("Lya")
                print(
                    mock_sed_obs.line_EWs["Lya"] / (1 + mock_sed_obs.z),
                    mock_sed_obs.line_cont,
                    mock_sed_obs.line_fluxes,
                )
                instrument = NIRCam(excl_bands=non_JADES_bands)
                depths = [30.0 for band in instrument]
                mock_sed_obs.create_mock_phot(instrument, depths)

                # change redshift by a factor eta
                mock_sed_obs.mock_photometry.z = redshift
                beta_orig = Photometry_rest.from_phot_obs(
                    mock_sed_obs.mock_photometry
                ).basic_beta_calc(conv_filt=False)  # should be == -2
                # print(beta_orig)
                mock_sed_obs.mock_photometry.z = (
                    (1 + eta) * (1 + redshift)
                ) - 1
                print(mock_sed_obs.mock_photometry.z)
                if plot:
                    mock_sed_obs.plot_SED(
                        ax,
                        wav_units=u.AA,
                        mag_units=u.erg / (u.s * u.cm**2 * u.AA),
                    )
                    # mock_sed_obs.mock_photometry.plot_phot(ax, wav_units = u.AA, mag_units = u.Jy, label = f"z = {redshift}")
                # print(mock_sed_obs.mock_photometry.z)
                beta_new = Photometry_rest.from_phot_obs(
                    mock_sed_obs.mock_photometry
                ).basic_beta_calc(conv_filt=False)
                # print(beta_new)
                delta_beta = beta_new - beta_orig
                delta_beta_arr.append(delta_beta)
            if not plot:
                plt.plot(
                    moving_average(redshifts, n=5),
                    moving_average(delta_beta_arr, n=5),
                    label=f"eta = {eta}, EW = {EW}",
                )
                # plt.plot(((1 + eta) * (1 + redshifts)) - 1, delta_beta_arr, label = f"eta = {eta}, line width = {line_width}")
                plt.xlim(6.5, 13.0)
    plt.legend()
    if plot:
        plt.ylim(0, 3e-8)
    plt.show()


if __name__ == "__main__":
    plot_bias()

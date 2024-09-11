#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:26:52 2024

@author: austind
"""

# BPASS_[NII]_Halpha_ratio.py

import glob

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table


def main(
    imf="imf135_300",
    log_ages=[7.0],
    m_UV_norm=26.0 * u.ABmag,
    logU_arr=[-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0],
    metallicity_arr=["zem4", "z001", "z010"],
):  # ["zem5", "zem4", "z001", "z002", "z003", "z004", "z006", "z008", "z010", "z020", "z030", "z040"]
    from galfind import (
        Mock_SED_rest_template_set,
    )

    NII_Halpha = {}
    OIII_Hbeta = {}
    for metallicity in metallicity_arr:
        NII_Halpha[metallicity] = []
        OIII_Hbeta[metallicity] = []
        print(f"{metallicity=}")
        for logU in logU_arr:
            BPASS_template_set_rest = (
                Mock_SED_rest_template_set.load_bpass_in_templates(
                    metallicity=metallicity,
                    logU=logU,
                    imf=imf,
                    log_ages=log_ages,
                    m_UV_norm=m_UV_norm,
                    bpass_version="2.2_cloudy",
                )
            )
            print(f"{logU=}")
            for template, log_age in zip(
                BPASS_template_set_rest.SED_arr, log_ages
            ):
                print(f"{log_age=}")
                # OIII_5007_EW = template.calc_line_EW("[OIII]-5007")
                # Hbeta_EW = template.calc_line_EW("Hbeta")

                Halpha_EW = template.calc_line_EW("Halpha")
                NII_EW = template.calc_line_EW("[NII]-6583")
                # print(NII_EW / Halpha_EW, NII_EW, Halpha_EW)
                # fig, ax = plt.subplots()
                # ax.axvline(5_003.8, c = "red")
                # ax.axvline(5_008.8, c = "red")
                # ax.set_xlim(4_999.8, 5_027.8)
                # ax.set_ylim(-17.735, -17.71)
                # template.plot_SED(ax, mag_units = u.erg / (u.s * u.cm ** 2 * u.AA), log_fluxes = True, save = True, save_name = f"{logU=},{log_age=}_test")

            NII_Halpha[metallicity].append(
                (NII_EW / Halpha_EW).to(u.dimensionless_unscaled).value
            )
        # get OIII_Hbeta
        OIII_Hbeta[metallicity] = OIII_Hbeta_from_EW_tab(
            logU_arr, log_ages[-1], metallicity
        )
        # OIII_Hbeta[metallicity].append((OIII_5007_EW / Hbeta_EW).to(u.dimensionless_unscaled).value)
    # save_NII_Halpha(logU_arr, NII_Halpha, save_name)
    plot_NII_Halpha(logU_arr, NII_Halpha, age=log_ages[-1])

    plot_BPT(logU_arr, NII_Halpha, OIII_Hbeta, age=log_ages[-1])


def save_NII_Halpha(logU_arr, NII_Halpha, save_name):
    pass


def plot_NII_Halpha(logU_arr, NII_Halpha_dict, age):
    fig, ax = plt.subplots()
    for metallicity, NII_Halpha in NII_Halpha_dict.items():
        ax.plot(logU_arr, NII_Halpha, label=metallicity)
    ax.set_title(f"Age = {((10 ** age) * u.yr).to(u.Myr).value} Myr")
    ax.set_xlabel("logU")
    ax.set_ylabel(r"[NII]$_{\lambda 6583}$/H$\alpha$")
    ax.legend()
    plt.savefig(f"NII_Halpha_1e{str(age)}_yr.png")


def plot_BPT_AGN_OIII_Hbeta_NII_Halpha(
    ax, author_year="Kauffmann2013", kwargs={"c": "black"}
):
    x_arr = np.linspace(-4.0, 0.0, 1_000)
    if author_year == "Kauffmann2013":
        y_arr = 0.61 / (x_arr - 0.05) + 1.3
    elif author_year == "Kewley2001":
        y_arr = 0.61 / (x_arr - 0.47) + 1.19
    ax.plot(x_arr, y_arr, **kwargs)


def plot_BPT(logU_arr, NII_Halpha_dict, OIII_Hbeta_dict, age):
    fig, ax = plt.subplots()
    for (metallicity, NII_Halpha), (metallicity_2, OIII_Hbeta) in zip(
        NII_Halpha_dict.items(), OIII_Hbeta_dict.items()
    ):
        assert metallicity == metallicity_2
        ax.plot(np.log10(NII_Halpha), np.log10(OIII_Hbeta), label=metallicity)
    plot_BPT_AGN_OIII_Hbeta_NII_Halpha(ax)
    ax.set_title(f"Age = {((10 ** age) * u.yr).to(u.Myr).value} Myr")
    ax.set_xlabel(r"$\log($[NII]$_{\lambda 6583}$/H$\alpha)$")
    ax.set_ylabel(r"$\log($[OIII]$_{\lambda 5007}$/H$\beta)$")
    ax.set_xlim(-2.0, 0.0)
    ax.set_ylim(-1.0, 1.5)
    ax.legend()
    plt.savefig(f"BPT_1e{str(age)}_yr.png")


def OIII_Hbeta_from_EW_tab(logU_arr, age, metallicity, grain_name="ng"):
    # load BPASS v2.2 + CLOUDY models
    line_EW_path = f"/raid/scratch/data/BPASS/BPASS_v2.2_cloudy/line_ews/line_ews_{metallicity}_{grain_name}.dat"
    # line_EW_paths = glob.glob(f"/raid/scratch/data/BPASS/BPASS_v2.2_cloudy/line_ews/line_ews_*_{grain_name}.dat")
    # for line_EW_path in reversed(line_EW_paths):
    # print(f"Z = {line_EW_path.split('_')[-2]}")
    tab = Table.read(line_EW_path, format="ascii")
    tab = tab[tab["logage"] == age]
    tab.sort("logU")
    logU = list(tab["logU"])
    assert logU == logU_arr
    OIII_Hbeta = tab["W_OIII5007"] / tab["W_Hbeta"]
    return OIII_Hbeta


def BPASS_BPT_v2_1(
    metallicities="All",
    log_nH=2.0,
    log_age=6.5,
    bpass_dir="/raid/scratch/data/BPASS/BPASSv2.1_nebula_release-02-18-Kiwi/Optical/bin",
):
    fig, ax = plt.subplots()
    if metallicities == "All":
        tab_paths = glob.glob(f"{bpass_dir}/Optical_data_bin_*.dat")
    else:
        tab_paths = [
            f"{bpass_dir}/Optical_data_bin_{metallicity}.dat"
            for metallicity in metallicities
        ]
    for tab_path in tab_paths:
        metallicity = tab_path.split("_")[-1].replace(".dat", "")
        tab = Table.read(tab_path, format="ascii.commented_header")
        selection_mask = (tab["#3)log(n_H/cm^-3)"] == log_nH) & (
            tab["#4)log(Age/yr)"] == log_age
        )
        tab = tab[selection_mask]
        assert all(row["#3)log(n_H/cm^-3)"] == log_nH for row in tab)
        assert all(row["#4)log(Age/yr)"] == log_age for row in tab)
        tab.sort("#2)log(U)")
        OIII_Hbeta = tab["#17)OIII[5007A]"] / tab["#21)Hb[4861A]"]
        NII_Halpha = tab["#7)NII[6584A]"] / tab["#19)Ha[6563A]"]
        ax.plot(
            np.log10(NII_Halpha),
            np.log10(OIII_Hbeta),
            label=f"Z={metallicity}",
        )

    plot_BPT_AGN_OIII_Hbeta_NII_Halpha(ax)
    ax.legend()
    ax.set_title(
        f"Age = {((10 ** log_age) * u.yr).to(u.Myr).value} Myr, nH = {10 ** log_nH} / cm3"
    )
    ax.set_xlabel(r"$\log($[NII]$_{\lambda 6583}$/H$\alpha)$")
    ax.set_ylabel(r"$\log($[OIII]$_{\lambda 5007}$/H$\beta)$")
    ax.set_xlim(-2.5, 0.0)
    ax.set_ylim(-2.0, 1.5)
    ax.legend()
    plt.savefig(f"BPT_1e{str(log_age)}yr_nH=1e{log_nH}.png")


def BPASS_NII_Halpha_v2_1(
    metallicities="All",
    log_nH=2.0,
    log_age=6.5,
    bpass_dir="/raid/scratch/data/BPASS/BPASSv2.1_nebula_release-02-18-Kiwi/Optical/bin",
):
    fig, ax = plt.subplots()
    if metallicities == "All":
        tab_paths = glob.glob(f"{bpass_dir}/Optical_data_bin_*.dat")
    else:
        tab_paths = [
            f"{bpass_dir}/Optical_data_bin_{metallicity}.dat"
            for metallicity in metallicities
        ]
    for tab_path in tab_paths:
        metallicity = tab_path.split("_")[-1].replace(".dat", "")
        tab = Table.read(tab_path, format="ascii.commented_header")
        selection_mask = (tab["#3)log(n_H/cm^-3)"] == log_nH) & (
            tab["#4)log(Age/yr)"] == log_age
        )
        tab = tab[selection_mask]
        assert all(row["#3)log(n_H/cm^-3)"] == log_nH for row in tab)
        assert all(row["#4)log(Age/yr)"] == log_age for row in tab)
        tab.sort("#2)log(U)")
        logU = tab["#2)log(U)"]
        NII_Halpha = (tab["#7)NII[6584A]"] + tab["#5)NII[6548A]"]) / tab[
            "#19)Ha[6563A]"
        ]
        ax.plot(logU, np.log10(NII_Halpha), label=f"Z={metallicity}")

    ax.legend()
    ax.set_title(
        f"Age = {((10 ** log_age) * u.yr).to(u.Myr).value} Myr, nH = {10 ** log_nH} / cm3"
    )
    ax.set_xlabel(r"$\log(U)$")
    ax.set_ylabel(r"$\log($[NII]$_{\lambda 6548 + \lambda 6583}$/H$\alpha)$")
    # ax.set_xlim(-3., 0.5)
    # ax.set_ylim(-3.5, 1.5)
    ax.legend()
    plt.savefig(f"NII_Halpha_1e{str(log_age)}yr_nH=1e{log_nH}.png")


if __name__ == "__main__":
    metallicities = ["zem4", "z001", "z006", "z010", "z020"]
    log_ages = [6.0, 6.5, 7.0]
    log_nHs = [1.0, 2.0, 3.0]
    for log_nH in log_nHs:
        for log_age in log_ages:
            BPASS_BPT_v2_1(metallicities, log_nH, log_age)
            BPASS_NII_Halpha_v2_1(metallicities, log_nH, log_age)
    # main()

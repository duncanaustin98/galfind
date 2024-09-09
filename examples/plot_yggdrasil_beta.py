#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 21:55:37 2023

@author: u92876da
"""

# plot_yggdrasil_beta.py
import glob
from pathlib import Path

import astropy.units as u
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii
from astropy.table import Table, vstack

from galfind import Mock_SED_rest_template_set, config

plt.style.use(f"{config['DEFAULT']['GALFIND_DIR']}/galfind_style.mplstyle")


def calc_yggdrasil_beta(grid_dir, imfs, fcovs, sfhs):
    for imf in imfs:
        for fcov in fcovs:
            for sfh in sfhs:
                print(imf, fcov, sfh)
                # load in yggdrasil SEDs
                yggdrasil_seds = Mock_SED_rest_template_set.load_Yggdrasil_popIII_in_templates(
                    imf, fcov, sfh
                )
                for sed in yggdrasil_seds:
                    # calculate beta for the SED template
                    beta = sed.calc_UV_slope()[1]
                    print(beta)
                    # save beta into template metadata
                    output_name = f"{grid_dir}/{imf}_fcov_{str(fcov)}_SFR_{sfh}_Spectra/{sed.template_name}"
                    data = Table.read(output_name)
                    data.meta = {**data.meta, **{"beta": beta}}
                    # print(data)
                    ascii.write(
                        data, output_name, format="ecsv", overwrite=True
                    )


def plot_yggdrasil_age_vs_beta(grid_dir, imfs, fcovs, sfhs, solar_Z=0.02):
    in_path = "Yggdrasil.ecsv"
    if not Path(in_path).is_file():
        out_tab = Table(
            {"IMF": [], "fesc": [], "sfh": [], "age": [], "beta": []}
        )
        for i, imf in enumerate(imfs):
            print(imf)
            for j, fcov in enumerate(fcovs):
                print(fcov)
                for k, sfh in enumerate(sfhs):
                    print(sfh)
                    label = f"{imf}_fcov_{str(fcov)}_SFR_{sfh}_Spectra"
                    sed_names = glob.glob(f"{grid_dir}/{label}/*.ecsv")
                    ages = []
                    beta = []
                    for name in sed_names:
                        data = Table.read(name)
                        ages.append(data.meta["age"])
                        beta.append(data.meta["beta"])
                    ages, beta = zip(*sorted(zip(ages, beta)))
                    label = (
                        label.replace(f"_SFR_{sfh}_Spectra", "")
                        .replace("_kroupa_IMF", "")
                        .split("_")[0]
                    )
                    x = np.log10((ages * u.Myr).to(u.yr).value)
                    y = np.array(beta)
                    tab = Table(
                        {
                            "IMF": np.full(len(x), imf),
                            "fesc": np.full(len(x), 1 - float(fcov)),
                            "sfh": np.full(len(x), sfh),
                            "age": x,
                            "beta": y,
                        }
                    )
                    if len(out_tab) == 0:
                        out_tab = tab
                        print("here")
                    else:
                        out_tab = vstack([out_tab, tab])
                    print(out_tab)
        out_tab.meta = {"citation": "Zackrisson+11"}
        out_tab.write(in_path, overwrite=True)
    else:
        out_tab = Table.read(in_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, imf in enumerate(imfs):
        print(imf)
        ls = ["-", "--", "dotted"]
        for j, fcov in enumerate(fcovs):
            if not ("PopIII" in imf and fcov == "1"):
                print(fcov)
                for sfh in sfhs:
                    # print(out_tab)
                    tab_ = out_tab[
                        (
                            (out_tab["IMF"] == imf)
                            & (out_tab["fesc"] == 1 - float(fcov))
                            & (out_tab["sfh"] == sfh)
                        )
                    ]
                    x = tab_["age"]
                    y = tab_["beta"]
                    # print(x, y)
                    if j == 0:
                        label = imf.replace("_kroupa_IMF", "")
                        print(label.split("="))
                        if len(label.split("=")) > 1:
                            label_split = label.split("=")
                            label_split[0] = r"$Z$"
                            label_split[1] = (
                                f"{float(label_split[1]) / solar_Z : .2f}"
                                + r"Z$_{\odot}$"
                            )
                            print(label_split)
                            label = "$=$".join(label_split).replace(" ", "")
                            print(label)
                        (plot,) = ax.plot(x, y, ls=ls[j], label=label, lw=3)
                        colour = plot.get_color()
                    else:
                        ax.plot(x, y, label=None, ls=ls[j], lw=3, color=colour)

    colours = ["navajowhite" for i in range(3)]
    beta_arr = [-2.1951, -2.4879, -2.7188]
    beta_l1_arr = [0.0147, 0.0630, 0.0580]
    beta_u1_arr = [0.0143, 0.0590, 0.0571]
    labels = [r"$6.5<z<8.5$", r"$8.5<z<11$", r"$11<z<13$"]
    age_lims = [6.0, 9.0]
    alpha_base = 0.8
    for i, (beta, beta_l1, beta_u1, colour, label) in enumerate(
        zip(beta_arr, beta_l1_arr, beta_u1_arr, colours, labels)
    ):
        ax.fill_between(
            age_lims,
            [beta - beta_l1] * 2,
            [beta + beta_u1] * 2,
            color=colour,
            alpha=alpha_base,
        )  # β(M_UV=-19)
        ax.text(
            age_lims[-1] - 0.5,
            beta,
            label,
            c="white",
            path_effects=[pe.withStroke(linewidth=3.0, foreground="black")],
            ha="center",
            va="center",
        )

    y_lims = ax.get_ylim()
    # make secondary legend with linestyles on
    lines = []
    labels = []
    for line, fcov in zip(ls, fcovs):
        (ax_line,) = plt.plot(-99.0, -99.0, ls=line, lw=2.0, c="black")
        lines.append(ax_line)
        labels.append(r"$f_{\mathrm{esc, LyC}}=%.1f$" % (1.0 - float(fcov)))
    ls_legend = plt.legend(
        lines, labels, loc="lower right", ncol=len(fcovs), frameon=False
    )
    imf_legend = plt.legend(
        ncol=3,
        bbox_to_anchor=(1.0, 1.0),
        loc="upper right",
        framealpha=1.0,
        frameon=False,
    )
    plt.gca().add_artist(ls_legend)
    plt.gca().add_artist(imf_legend)

    ax.set_xlim(6, 9)
    ax.set_ylim(*y_lims)
    ax.set_xlabel(r"$\log_{10}$(Age / yr)")
    ax.set_ylabel(r"$\beta$ (Calzetti+1994)")

    # open EPOCHS catalogue
    # epochs_tab = Table.read(f"{config['DEFAULT']['GALFIND_WORK']}/Catalogues/v9/ACS_WFC+NIRCam/Combined/EPOCHS_MASTER_Sel-f277W+f356W+f444W_v9_loc_depth_masked_10pc_EAZY_matched_selection_ext_src_UV_pipes_v2_final_EPOCHS_III.fits")
    # epochs_tab = epochs_tab[epochs_tab["EPOCHS_III_certain"]]
    # ax2 = ax.inset_axes([0.92, 0.11, 0.2, 0.77]) #, transform = ax.transAxes
    # z_range_arr = [[6.5, 8.5], [8.5, 11.], [11., 13.]]
    # for z_range in z_range_arr:
    #     zlabel = f"{z_range[0]}<z<{z_range[1]}"
    #     beta_zrange = epochs_tab[((epochs_tab["zbest_fsps_larson"] > z_range[0]) & (epochs_tab["zbest_fsps_larson"] < z_range[1]) & \
    #         (epochs_tab["Beta_1250-3000Angstrom_conv_filt_PL"] > ax.get_ylim()[0]) & (epochs_tab["Beta_1250-3000Angstrom_conv_filt_PL"] < ax.get_ylim()[1]))]["Beta_1250-3000Angstrom_conv_filt_PL"]
    #     ax2.hist(beta_zrange, orientation = "horizontal", histtype = "step", density = True, label = zlabel, bins = 15)
    # ax2.set_ylim(*ax.get_ylim())

    plt.savefig(f"Yggdrasil_beta_{sfhs[0]}.png")
    plt.show()


if __name__ == "__main__":
    grid_dir = "/nvme/scratch/work/austind/yggdrasil_grids"
    imfs = [
        "Z=0.0004_kroupa_IMF",
        "Z=0.004_kroupa_IMF",
        "Z=0.02_kroupa_IMF",
        "PopIII.1",
        "PopIII.2",
        "PopIII_kroupa_IMF",
    ]  # , "Z=0.008_kroupa_IMF",
    fcovs = ["0", "0.5", "1"]
    sfhs = ["inst", "10M_yr", "100M_yr"]

    # calc_yggdrasil_beta(grid_dir, imfs, fcovs, sfhs)

    plot_yggdrasil_age_vs_beta(grid_dir, imfs, fcovs, sfhs)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 21:55:37 2023

@author: u92876da
"""

# plot_yggdrasil_beta.py
from astropy.table import Table
from astropy.io import ascii
import glob
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from galfind import Mock_SED_rest_template_set, Mock_SED_rest, config

plt.style.use(f"{config['DEFAULT']['GALFIND_DIR']}/galfind_style.mplstyle")

def calc_yggdrasil_beta(grid_dir, imfs, fcovs, sfhs):
    for imf in imfs:
        for fcov in fcovs:
            for sfh in sfhs:
                print(imf, fcov, sfh)
                # load in yggdrasil SEDs
                yggdrasil_seds = Mock_SED_rest_template_set.load_Yggdrasil_popIII_in_templates(imf, fcov, sfh)
                for sed in yggdrasil_seds:
                    # calculate beta for the SED template
                    beta = sed.calc_UV_slope()[1]
                    print(beta)
                    # save beta into template metadata
                    output_name = f"{grid_dir}/{imf}_fcov_{str(fcov)}_SFR_{sfh}_Spectra/{sed.template_name}"
                    data = Table.read(output_name)
                    data.meta = {**data.meta, **{"beta": beta}}
                    #print(data)
                    ascii.write(data, output_name, format = "ecsv", overwrite = True)

def plot_yggdrasil_age_vs_beta(ax, grid_dir, imfs, fcovs, sfhs):
    for i, imf in enumerate(imfs):
        print(imf)
        ls = ["-", "--", "-."]
        for j, fcov in enumerate(fcovs):
            print(fcov)
            for sfh in sfhs:
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
                label = label.replace(f"_SFR_{sfh}_Spectra", "").replace("_kroupa_IMF", "")
                if j == 0:
                    plot, = ax.plot(np.log10((ages * u.Myr).to(u.yr).value), beta, label = label, lw = 3)
                    colour = plot.get_color()
                else:
                    ax.plot(np.log10((ages * u.Myr).to(u.yr).value), beta, label = label, ls = ls[j], lw = 3, color = colour)
    plt.savefig("Yggdrasil_popI_II.png")

if __name__ == "__main__":
    grid_dir = "/Users/user/Documents/PGR/yggdrasil_grids"
    imfs = ["Z=0.0004_kroupa_IMF", "Z=0.004_kroupa_IMF", "Z=0.008_kroupa_IMF", "Z=0.02_kroupa_IMF"] #["PopIII.1", "PopIII.2", "PopIII_kroupa_IMF", "Z=0.0004_kroupa_IMF"]
    fcovs = ["0", "0.5", "1"]
    sfhs = ["inst"] #, "10M_yr"] #, "100M_yr"]
    
    # calc_yggdrasil_beta(grid_dir, imfs, fcovs, sfhs)
    
    fig, ax = plt.subplots(figsize = (6, 8))
    plot_yggdrasil_age_vs_beta(ax, grid_dir, imfs, fcovs, sfhs)
    plt.legend(bbox_to_anchor = (1.05, 0.5), loc = "center left")
    plt.xlim(6, 8)
    plt.xlabel(r"$\log_{10}$(Age / Myr)")
    plt.ylabel(r"$\beta$ (Calzetti+1994)")
    plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:22:19 2023

@author: austind
"""

# cat_from_pipes_templates.py
import glob
import os
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from tqdm import tqdm

from galfind import Mock_SED_obs, Mock_SED_rest


def save_template(
    sed_obs,
    out_dir="/raid/scratch/work/austind/GALFIND_WORK/Bagpipes/template_sets/beta_bias_continuity_bursty_zgauss_no_Lya",
):
    try:
        _, beta = sed_obs.calc_UV_slope()
    except:
        beta = -99.0
    spectrum = Table()
    spectrum["wav_obs"] = sed_obs.wavs
    if sed_obs.mags.unit != u.nJy:
        sed_obs.convert_mag_units(u.nJy)
    spectrum["f_nu"] = sed_obs.mags
    spectrum.meta = {**sed_obs.meta, **{"beta_C94": beta}}

    os.makedirs(f"{out_dir}/spectra_obs", exist_ok=True)
    # os.makedirs(f"{out_dir}/photometry", exist_ok = True)
    out_name = f"spectra_obs/{sed_obs.template_name}.ecsv"
    spectrum.write(f"{out_dir}/{out_name}", overwrite=True)


def collate_beta(
    out_dir="/raid/scratch/work/austind/GALFIND_WORK/Bagpipes/template_sets/beta_bias_continuity_bursty_zgauss_no_Lya/spectra_obs",
):
    if Path(f"{out_dir}/beta_C94.ecsv").is_file():
        beta = np.array(Table.read(f"{out_dir}/beta_C94.ecsv")["beta_C94"])
    else:
        files = glob.glob(f"{out_dir}/*")
        beta = np.array(
            [
                Table.read(file).meta["beta_C94"]
                for file in tqdm(
                    files[:10], total=len(files), desc="loading beta"
                )
            ]
        )
        output_beta = Table([beta], names=["beta_C94"], dtype=[float]).write(
            f"{out_dir}/beta_C94.ecsv", overwrite=True
        )
    # beta = [val for val in beta if val > -3. and val < -1.5]
    print(beta)
    return beta


def plot_beta(
    out_dir="/raid/scratch/work/austind/GALFIND_WORK/Bagpipes/template_sets/beta_bias_continuity_bursty_zgauss_no_Lya/spectra_obs",
):
    beta = collate_beta(out_dir)
    beta = [val for val in beta if val < 1.0 and val > -5.0]
    plt.hist(beta, bins=100)


def main():
    # fieldname = "JADES-Deep-GS"
    # version = "v9"
    # excl_bands = ["f182M", "f210M", "f430M", "f460M", "f480M"]
    # aper_diam = 0.32 * u.arcsec
    plot = False

    # data = Data.from_pipeline(fieldname, version, excl_bands = excl_bands)
    pipes_template_dir = "beta_bias_continuity_bursty_zgauss_no_Lya/spectra"
    n_gals = 10_000
    mu, sigma, min_m_UV, max_m_UV = 0.0, 1.0, 24.0, 31.5
    mag_arr = np.random.lognormal(mu, sigma, n_gals * 3)
    m_UV_arr = np.array(
        [
            mag
            for mag in max_m_UV - (mag_arr - np.exp(-(sigma**2)))
            if mag > min_m_UV and mag < max_m_UV
        ]
    )

    if len(m_UV_arr) < n_gals:
        raise (Exception("Increase number of m_UVs extracted!"))
    for i in tqdm(range(n_gals)):
        try:
            sed_rest = Mock_SED_rest.load_pipes_in_template(
                m_UV_arr[i], pipes_template_dir, i
            )
            sed_obs = Mock_SED_obs.from_Mock_SED_rest(
                sed_rest, sed_rest.meta["redshift"], IGM=None
            )
            save_template(sed_obs)
            if plot:
                fig, ax = plt.subplots()
                sed_obs.plot_SED(ax, mag_units=u.nJy)
                # sed_obs.create_mock_phot(data.instrument, depths = data.load_depths(aper_diam), min_pc_err = 10.)
        except:
            pass
    if plot:
        sed_obs.mock_photometry.plot(ax, mag_units=u.Jy)
        ax.set_xlim(9_000, 50_000)
        ax.set_ylim(0.0, 0.5e-8)


if __name__ == "__main__":
    # main()
    plot_beta()

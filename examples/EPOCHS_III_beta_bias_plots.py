#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:47:10 2024

@author: austind
"""

# EPOCHS_III_beta_bias_plots.py
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from galfind import DLA, Mock_SED_obs, Mock_SED_rest, NIRCam
from galfind import useful_funcs_austind as funcs


def plot_DLA_mag_diff(
    ax,
    instrument,
    depths,
    DLA_obj,
    plot_bands=["f090W", "f115W", "f150W", "f200W"],
    z_arr=np.linspace(6.5, 13.0, 10),
    beta_intrinsic=-2.5,
    m_UV_intrinsic=26.0,
    n_sig_detection=2.0,
):
    mock_sed_rest = Mock_SED_rest.power_law_from_beta_m_UV(
        beta_intrinsic, m_UV_intrinsic
    )
    for i, band in enumerate(plot_bands):
        mag_diff = np.zeros(len(z_arr))
        for j, z in tqdm(
            enumerate(z_arr),
            total=len(z_arr),
            desc=f"Running DLA mag diff for {band}",
        ):
            mock_sed_obs = Mock_SED_obs.from_Mock_SED_rest(mock_sed_rest, z)
            mock_sed_obs.create_mock_phot(instrument, depths)
            orig_phot = mock_sed_obs.mock_photometry
            # mock_sed_obs.plot_SED(ax, mag_units = u.Jy)
            # orig_phot.plot_phot(ax)
            mock_sed_obs.add_DLA(DLA_obj)
            mock_sed_obs.create_mock_phot(instrument, depths)
            new_phot = mock_sed_obs.mock_photometry
            # mock_sed_obs.plot_SED(ax, mag_units = u.Jy)
            # new_phot.plot_phot(ax)
            mag_diff[j] = (
                -2.5 * np.log10(new_phot[band].value / orig_phot[band].value)
                if funcs.calc_1sigma_flux(30.0, 8.9) * n_sig_detection
                < new_phot[band].value
                else np.nan
            )
        # ax.axvline((instrument.band_wavelengths[band] / line_diagnostics["Lya"]["line_wav"]) - 1.)
        ax.plot(z_arr, mag_diff)


if __name__ == "__main__":
    fig, ax = plt.subplots()
    instrument = NIRCam(
        excl_bands=[
            "f070W",
            "f140M",
            "f162M",
            "f182M",
            "f250M",
            "f300M",
            "f335M",
            "f360M",
            "f430M",
            "f460M",
            "f480M",
        ]
    )
    depths = [30.0 for band in instrument]
    DLA_obj = DLA((10**23.0) / (u.cm**2), 150.0 * u.km / u.s)
    plot_DLA_mag_diff(ax, instrument, depths, DLA_obj)

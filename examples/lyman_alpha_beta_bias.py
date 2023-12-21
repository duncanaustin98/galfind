#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 19:06:11 2023

@author: u92876da
"""

# lyman_alpha_beta_bias.py
import matplotlib.pyplot as plt
import astropy.units as u

from galfind import SED, Mock_SED_rest, Mock_SED_obs, Emission_line, NIRCam

def main(beta_arr = [-3.0], m_UV = 28.):
    lya = Emission_line("Lya", 5e-18 * u.erg / (u.s * u.cm ** 2), 150. * u.km / u.s)
    non_JADES_bands = ["f070W", "f140M", "f162M", "f182M", "f210M", "f250M", "f300M", "f360M", "f430M", "f460M", "f480M"]
    redshift = 10.
    fig, ax = plt.subplots()
    for beta in beta_arr:
        mock_sed_rest = Mock_SED_rest.power_law_from_beta_m_UV(beta, m_UV, wav_res = 0.1)
        mock_sed_obs = Mock_SED_obs.from_Mock_SED_rest(mock_sed_rest, redshift)
        instrument = NIRCam(excl_bands = non_JADES_bands)
        depths = [30. for band in instrument]
        mock_sed_obs.create_mock_phot(instrument, depths)
        mock_sed_obs.plot_SED(ax, wav_units = u.AA, mag_units = u.Jy)
        mock_sed_obs.mock_photometry.plot_phot(ax, wav_units = u.AA, mag_units = u.Jy)
        #mock_sed_rest.plot_SED(ax)
        # mock_sed_rest.add_emission_lines([lya])
        # mock_sed_rest.calc_line_EW("Lya")
        # #mock_sed_rest.plot_SED(ax, mag_units = u.Jy)
        # print(mock_sed_rest.line_cont["Lya"], mock_sed_rest.line_fluxes["Lya"], mock_sed_rest.calc_line_EW("Lya"))
    pass

if __name__ == "__main__":
    main()
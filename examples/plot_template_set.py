#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 20:59:55 2023

@author: u92876da
"""

# plot_template_set.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.units as u

from galfind import SED, Mock_SED_rest, Mock_SED_obs, config, NIRCam, IGM_attenuation

plt.style.use(f"{config['DEFAULT']['GALFIND_DIR']}/galfind_style.mplstyle")

instrument = NIRCam(excl_bands = ["f070W", "f140M", "f162M", "f182M", "f210M", "f250M", "f300M", "f335M", "f360M", "f430M", "f460M", "f480M"])
depths = {band: 26. for band in instrument.bands}
min_pc_err = 10
z = 10.5
incl_IGM_att = True
fig, ax = plt.subplots()
for i in range(13, 19):
    sed_obj_rest = Mock_SED_rest.load_EAZY_in_template(25., "fsps_larson", i)
    if incl_IGM_att:
        IGM = IGM_attenuation.IGM()
    else:
        IGM = None
    sed_obj_obs = Mock_SED_obs.from_Mock_SED_rest(sed_obj_rest, z, IGM = IGM)
    sed_plot = sed_obj_obs.plot_SED(ax, u.um, u.Jy, label = sed_obj_rest.template_name, annotate = True, \
            plot_kwargs = {"alpha": 0.6}, legend_kwargs = {"bbox_to_anchor": [0., 0.], "loc": "lower left", "fontsize": 8.})
    sed_obj_obs.create_mock_photometry(instrument, depths, min_pc_err)
    sed_obj_obs.mock_photometry.plot_phot(ax, u.um, u.Jy, plot_errs = True, errorbar_kwargs = {"color": sed_plot[0].get_color()})
# plot rest frame UV
ax.axvline((1250. * u.AA).to(u.um).value * (1 + z), color = "darkblue")
ax.axvline((3000. * u.AA).to(u.um).value * (1 + z), color = "darkblue")
# plot emission lines
ax.axvline((1906. * u.AA).to(u.um).value * (1 + z), color = "black", ls = "--", label = "CIII]")
ax.axvline((1640. * u.AA).to(u.um).value * (1 + z), color = "black", ls = "--", label = "HeII")
ax.axvline((1660. * u.AA).to(u.um).value * (1 + z), color = "black", ls = "--", label = "OIII]")
plt.yscale("log")
#plt.ylim(32., 20.) # ABmag
#plt.ylim(1e-3, 1e0) # uJy
plt.xlim(0.1, 5.) # um
plt.ylim(1e-7, 1e-6) # Jy
#plt.ylim(1e-21, 1e-15) # erg/s/cm^2/AA
plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 20:59:55 2023

@author: u92876da
"""

# plot_template_set.py
import astropy.units as u
import matplotlib.pyplot as plt

from galfind import (
    IGM_attenuation,
    Mock_SED_obs,
    Mock_SED_rest,
    NIRCam,
    config,
)
from galfind.Emission_lines import line_diagnostics

plt.style.use(f"{config['DEFAULT']['GALFIND_DIR']}/galfind_style.mplstyle")

instrument = NIRCam(
    excl_bands=[
        "F070W",
        "F140M",
        "F162M",
        "F182M",
        "F250M",
        "F300M",
        "F335M",
        "F360M",
        "F430M",
        "F460M",
        "F480M",
    ]
)
depths = [30.0 for band_name in instrument.band_names]
min_pc_err = 10
z = 0.0
template_set = "fsps_larson"
lines = ["Lya", "CIV-1549", "HeII-1640", "OIII]-1665", "CIII]-1909"]
incl_IGM_att = True
fig, ax = plt.subplots()
for i in range(12, 15):
    sed_obj_rest = Mock_SED_rest.load_EAZY_in_template(22.0, template_set, i)
    if incl_IGM_att:
        IGM = IGM_attenuation.IGM()
    else:
        IGM = None
    sed_obj_obs = Mock_SED_obs.from_Mock_SED_rest(sed_obj_rest, z, IGM=IGM)
    sed_plot = sed_obj_obs.plot_SED(
        ax,
        u.um,
        u.Jy,
        label=sed_obj_rest.template_name,
        annotate=True,
        plot_kwargs={"alpha": 0.6},
        legend_kwargs={
            "bbox_to_anchor": [0.0, 0.0],
            "loc": "lower left",
            "fontsize": 8.0,
        },
    )
    sed_obj_obs.create_mock_phot(instrument, depths, min_pc_err)
    sed_obj_obs.mock_photometry.plot(
        ax,
        u.um,
        u.Jy,
        plot_errs=True,
        errorbar_kwargs={"color": sed_plot[0].get_color()},
    )
    print(sed_obj_obs.template_name)
    for line in lines:
        try:
            sed_obj_obs.calc_line_EW(line)
            print(line, sed_obj_obs.line_EWs[line] / (1 + z))
        except:
            print(f"FAILED TO CALCULATE EW of {line}")
# plot rest frame UV
# ax.axvline((1250. * u.AA).to(u.um).value * (1 + z), color = "darkblue")
# ax.axvline((3000. * u.AA).to(u.um).value * (1 + z), color = "darkblue")
# plot emission lines
for line in lines:
    ax.axvline(
        line_diagnostics[line]["line_wav"].to(u.um).value * (1 + z),
        c="black",
        ls="--",
        label=line,
    )
    # ax.axvline(line_diagnostics[line]["feature_wavs"][0].to(u.um).value * (1 + z))
    # ax.axvline(line_diagnostics[line]["feature_wavs"][1].to(u.um).value * (1 + z))
    # for wav in np.array(line_diagnostics[line]["cont_wavs"]).flatten():
    #     ax.axvline((wav * u.AA).to(u.um).value * (1 + z))

# plt.legend()
plt.yscale("log")
# plt.ylim(4e-6, 1e-5)
# ax.set_xlim((1250. * u.AA * (1 + z)).to(u.um).value, (1500. * u.AA * (1 + z)).to(u.um).value)
# plt.ylim(32., 20.) # ABmag
# plt.ylim(1e-3, 1e0) # uJy
# plt.xlim(0.1, 5.) # um
# plt.ylim(3e-8, 1e-6) # Jy
# plt.ylim(1e-21, 1e-15) # erg/s/cm^2/AA
plt.show()

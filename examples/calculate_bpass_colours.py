# calculate_bpass_colours.py

import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import os
import matplotlib.patheffects as pe

from galfind import SED, Mock_SED_rest_template_set, Mock_SED_obs_template_set, Instrument, NIRCam
from galfind import useful_funcs_austind as funcs

def plot_colours(ax, redshifts, colour_x_name, colour_y_name, line_colours, metallicities, ls_arr, alpha_enhancement = "a+06", path_effects = [pe.withStroke(linewidth = 3, foreground = 'black')]):
    # create NIRCam instrument excluding all bands other than those needed for colours
    instrument = NIRCam(excl_bands = [band for band in NIRCam().bands if band not in colour_x_name.split("-") + colour_y_name.split("-")])
    for metallicity, ls in zip(metallicities, ls_arr):
        mock_sed_rest_templates = Mock_SED_rest_template_set.load_bpass_in_templates(metallicity = metallicity, alpha_enhancement = alpha_enhancement)
        mock_sed_obs_templates_arr = []
        for i, (z, colour) in enumerate(zip(redshifts, line_colours)):
            mock_sed_obs_templates = Mock_SED_obs_template_set.from_Mock_SED_rest_template_set(mock_sed_rest_templates, z)
            mock_sed_obs_templates.create_mock_phot(instrument)
            mock_sed_obs_templates.get_colours([colour_x_name, colour_y_name])
            mock_sed_obs_templates.plot_colour_colour_tracks(ax, colour_x_name, colour_y_name, \
                line_kwargs = {"ls": ls, "c": colour, "path_effects": path_effects}, save = False, show = False)

if __name__ == "__main__":
    redshifts = [6.5, 8.5, 10.5, 12.5]
    line_colours = ["red", "blue", "green", "gold"]
    colour_x_name = "f150W-f277W"
    colour_y_name = "f277W-f444W"
    #log_ages = "all"
    metallicities = ["z010", "z001", "zem4"]
    ls_arr = ["-", "--", "dotted"]
    alpha_enhancement = "a+06"
    fig, ax = plt.subplots()
    plot_colours(ax, redshifts, colour_x_name, colour_y_name, line_colours, metallicities, ls_arr, alpha_enhancement)
    plt.savefig(f"/nvme/scratch/work/austind/EPOCHS_I_plots/{colour_x_name}_vs_{colour_y_name}_z{str(redshifts)}_Z{str(metallicities)}_{alpha_enhancement}.png")

# plot_dust_attenuation_curves.py

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from galfind import Dust_Attenuation, Calzetti00, Mock_SED_obs, NIRCam

def plot_dust_attenuation_curves(dust_laws = [Calzetti00]):
    wavs = np.linspace(0.13 * u.um, 2.19 * u.um, 1_000)
    fig, ax = plt.subplots()
    for dust_law in dust_laws:
        dust_law_ = dust_law()
        dust_law_.plot_attenuation_law(ax, wavs)
        print(str(dust_law_))
    plt.savefig("/nvme/scratch/work/austind/GALFIND/examples/galfind_dust_laws.png")
    plt.show()

def plot_color_color_dust_arrow(ax, z, colour_x_name, colour_y_name, arrow_pos, E_BminusV, dust_law = Calzetti00, beta = -2.2, M_UV = -22.):
    # create NIRCam instrument excluding all bands other than those needed for colours
    instrument = NIRCam(excl_bands = [band for band in NIRCam().bands if band not in colour_x_name.split("-") + colour_y_name.split("-")])
    # make basic power law SED at redshift z and calculate colors
    sed_obs = Mock_SED_obs.power_law_from_beta_M_UV(z, beta, M_UV)
    sed_obs.create_mock_phot(instrument)
    colour_x_0 = sed_obs.get_colour(colour_x_name)
    colour_y_0 = sed_obs.get_colour(colour_y_name)
    print(colour_x_0, colour_y_0)
    # dust attenuate this power law SED and calculate colors
    sed_obs.dust_attenuate(dust_law, E_BminusV) # not yet implemented in mock_SED_obs
    sed_obs.colours = {} # reset colours
    colour_x = sed_obs.get_colour(colour_x_name)
    colour_y = sed_obs.get_colour(colour_y_name)
    # subtract the colors
    arrow_len_x = colour_x - colour_x_0
    arrow_len_y = colour_y - colour_y_0
    # plot the arrow

if __name__ == "__main__":
    redshifts = [9.]
    fig, ax = plt.subplots()
    for z in redshifts:
        plot_color_color_dust_arrow(ax, z, "f150W-f277W", "f277W-f444W", [0., 0.], 0.25)
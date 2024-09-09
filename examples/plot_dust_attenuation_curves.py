# plot_dust_attenuation_curves.py

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arrow

from galfind import C00, Mock_SED_obs, NIRCam


def plot_dust_attenuation_curves(dust_laws=[C00]):
    wavs = np.linspace(0.13 * u.um, 2.19 * u.um, 1_000)
    fig, ax = plt.subplots()
    for dust_law in dust_laws:
        dust_law_ = dust_law()
        dust_law_.plot_attenuation_law(ax, wavs)
        print(str(dust_law_))
    plt.savefig(
        "/nvme/scratch/work/austind/GALFIND/examples/galfind_dust_laws.png"
    )
    plt.show()


def plot_color_color_dust_arrow(
    ax,
    z,
    colour_x_name,
    colour_y_name,
    arrow_pos,
    E_BminusV,
    dust_law=C00(),
    beta=-2.2,
    M_UV=-22.0,
    arrow_kwargs={"c": "black"},
):
    # create NIRCam instrument excluding all bands other than those needed for colours
    instrument = NIRCam(
        excl_bands=[
            band
            for band in NIRCam().bands
            if band not in colour_x_name.split("-") + colour_y_name.split("-")
        ]
    )
    # make basic power law SED at redshift z and calculate colors
    sed_obs = Mock_SED_obs.power_law_from_beta_M_UV(z, beta, M_UV)
    sed_obs.create_mock_phot(instrument)
    colour_x_0 = sed_obs.get_colour(colour_x_name)
    colour_y_0 = sed_obs.get_colour(colour_y_name)
    # dust attenuate this power law SED and calculate colors
    sed_obs.add_dust_attenuation(dust_law, E_BminusV)
    sed_obs.colours = {}  # reset colours
    sed_obs.create_mock_phot(instrument)
    colour_x = sed_obs.get_colour(colour_x_name)
    colour_y = sed_obs.get_colour(colour_y_name)
    # plot the arrow
    print([colour_x - colour_x_0, colour_y - colour_y_0])
    arrow = Arrow(
        arrow_pos[0],
        arrow_pos[1],
        dx=colour_x - colour_x_0,
        dy=colour_y - colour_y_0,
        width=0.15,
        color=arrow_kwargs["c"],
    )
    ax.add_patch(arrow)
    ax.annotate(
        f"E(B-V)={E_BminusV:.2f}",
        (1.0, 1.0),
        xycoords=arrow,
        ha="center",
        va="top",
        **arrow_kwargs,
    )


if __name__ == "__main__":
    redshifts = [6.5, 8.5, 10.5, 12.5]
    fig, ax = plt.subplots()
    for z in redshifts:
        plot_color_color_dust_arrow(
            ax, z, "f150W-f277W", "f277W-f444W", [0.5, 0.5], 0.5
        )


import pytest
import matplotlib.pyplot as plt
import astropy.units as u

from galfind import SED_obs

@pytest.fixture(scope="module")
def custom_lephare_sed(gal_custom_lephare_loaded, custom_lephare_sed_fitter, aper_diams):
    return gal_custom_lephare_loaded.aper_phot[aper_diams[0]] \
        .SED_results[custom_lephare_sed_fitter.label].SED

@pytest.mark.requires_data
def test_custom_lephare_sed(custom_lephare_sed):
    assert isinstance(custom_lephare_sed, SED_obs)

@pytest.fixture(
    scope = "module",
    params = [
        {
            "save_name": "test_sed_plot_Jy.png",
            "mag_units": u.Jy,
        },
        {
            "save_name": "test_sed_plot_um_Jy.png",
            "wav_units": u.um,
            "mag_units": u.Jy,
        },
        {
            "save_name": "test_sed_plot_ABmag.png",
            "mag_units": u.ABmag,
        },
        {
            "save_name": "test_sed_plot_flam.png",
            "mag_units": u.erg / (u.s * u.cm ** 2 * u.AA),
        },
        {
            "save_name": "test_sed_plot",
        },
        {
            "save_name": "test_sed_plot.jpeg",
        },
    ]
)
def sed_plot_params(request):
    return request.param

@pytest.mark.requires_data
def test_plot_gal_sed_custom_lephare(custom_lephare_sed, sed_plot_params):
    fig, ax = plt.subplots()
    wav_unit = sed_plot_params.get("wav_units", u.AA)
    xlims = ([0.0, 6.0] * u.um).to(wav_unit).value
    ax.set_xlim(xlims)
    custom_lephare_sed.plot(ax = ax, **sed_plot_params)
    fig.clf()
    
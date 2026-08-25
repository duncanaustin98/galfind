import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

from galfind.spectra.SED import (
    SED,
    SED_2D,
    Mock_SED_obs,
    Mock_SED_rest,
    SED_obs,
    SED_rest,
)
from galfind.utils.exceptions import (
    GalfindTypeError,
    InvalidUnitError,
    LengthMismatchError,
    RangeError,
)


@pytest.fixture(scope="module")
def custom_lephare_sed(
    gal_custom_lephare_loaded, custom_lephare_sed_fitter, aper_diams
):
    return (
        gal_custom_lephare_loaded.aper_phot[aper_diams[0]]
        .SED_results[custom_lephare_sed_fitter.label]
        .SED
    )


@pytest.mark.requires_data
def test_custom_lephare_sed(custom_lephare_sed):
    assert isinstance(custom_lephare_sed, SED_obs)


@pytest.fixture(
    scope="module",
    params=[
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
            "mag_units": u.erg / (u.s * u.cm**2 * u.AA),
        },
        {
            "save_name": "test_sed_plot",
        },
        {
            "save_name": "test_sed_plot.jpeg",
        },
    ],
)
def sed_plot_params(request):
    return request.param


@pytest.mark.requires_data
@pytest.mark.lephare
def test_plot_gal_sed_custom_lephare(custom_lephare_sed, sed_plot_params):
    fig, ax = plt.subplots()
    wav_unit = sed_plot_params.get("wav_units", u.AA)
    xlims = ([0.0, 6.0] * u.um).to(wav_unit).value
    ax.set_xlim(xlims)
    custom_lephare_sed.plot(ax=ax, **sed_plot_params)
    fig.clf()


# --- Validation-failure tests for lightweight SED objects (no fixtures) ---

_wavs = np.array([1_000.0, 2_000.0, 3_000.0])
_mags = np.array([1.0, 2.0, 3.0])


def test_convert_mag_units_invalid_units_raises():
    sed = SED(_wavs, _mags, u.AA, u.Jy)
    with pytest.raises(InvalidUnitError, match="physical type"):
        sed.convert_mag_units(u.m)


def test_calc_colour_wrong_type_raises():
    sed_obs = SED_obs(6.0, _wavs, _mags, u.AA, u.Jy)
    with pytest.raises(GalfindTypeError, match="filters"):
        sed_obs.calc_colour("not_a_list")


def test_calc_colour_wrong_length_raises():
    sed_obs = SED_obs(6.0, _wavs, _mags, u.AA, u.Jy)
    with pytest.raises(LengthMismatchError, match="length 2"):
        sed_obs.calc_colour([1, 2, 3])


def test_calc_mUV_invalid_wav_range_raises():
    sed_obs = SED_obs(6.0, _wavs, _mags, u.AA, u.Jy)
    with pytest.raises(RangeError, match="less than"):
        sed_obs.calc_mUV(wav_range=[1_550.0, 1_450.0] * u.AA)


def test_normalize_to_m_UV_invalid_type_raises():
    mock_rest = Mock_SED_rest(_wavs, _mags, u.AA, u.Jy)
    with pytest.raises(GalfindTypeError, match="m_UV"):
        mock_rest.normalize_to_m_UV("not_valid")


def test_renorm_at_wav_invalid_wav_type_raises():
    mock_rest = Mock_SED_rest(_wavs, _mags, u.AA, u.Jy)
    with pytest.raises(GalfindTypeError, match="wav"):
        mock_rest.renorm_at_wav("not_a_quantity", 1.0 * u.Jy)


def test_renorm_at_wav_invalid_wav_unit_raises():
    mock_rest = Mock_SED_rest(_wavs, _mags, u.AA, u.Jy)
    with pytest.raises(InvalidUnitError, match="length"):
        mock_rest.renorm_at_wav(1.0 * u.s, 1.0 * u.Jy)


def test_add_emission_lines_invalid_type_raises():
    mock_rest = Mock_SED_rest(_wavs, _mags, u.AA, u.Jy)
    with pytest.raises(GalfindTypeError, match="emission_lines"):
        mock_rest.add_emission_lines("not_a_list")


def test_attenuate_IGM_invalid_type_raises():
    mock_obs = Mock_SED_obs(6.0, _wavs, _mags, u.AA, u.Jy, IGM=None)
    with pytest.raises(GalfindTypeError, match="IGM"):
        mock_obs.attenuate_IGM("not_an_igm_object")


def test_SED_2D_mixed_classes_raises():
    sed_rest = SED_rest(_wavs, _mags, u.AA, u.Jy)
    sed_obs = SED_obs(6.0, _wavs, _mags, u.AA, u.Jy)
    with pytest.raises(GalfindTypeError, match="same class"):
        SED_2D([sed_rest, sed_obs])


def test_SED_2D_getitem_non_int_raises():
    sed_rest_1 = SED_rest(_wavs, _mags, u.AA, u.Jy)
    sed_rest_2 = SED_rest(_wavs, _mags, u.AA, u.Jy)
    sed_2d = SED_2D([sed_rest_1, sed_rest_2])
    with pytest.raises(GalfindTypeError, match="Indexing"):
        sed_2d[0:1]

import threading
from copy import deepcopy
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

from galfind.galaxy import Galaxy
from galfind.sed_fitting import EAZY
from galfind.selection.Selector import Band_SNR_Selector
from galfind.spectra.SED import SED_obs
from galfind.utils.exceptions import (
    GalfindError,
    InvalidOptionError,
    LengthMismatchError,
    MissingDataError,
    MissingKeyError,
    RangeError,
)

# @pytest.mark.requires_data
# def test_gal_lephare_loaded(gal_lephare_loaded):
#     assert isinstance(gal_lephare_loaded, Galaxy)


@pytest.fixture(
    scope="module",
    params=[
        {
            "overwrite": True,
            # dynamic (n_cutout_rows=None, the default) sizing is not
            # yet implemented (Galaxy.plot_phot_diagnostic raises
            # NotImplementedError); pass an explicit row count instead
            "n_cutout_rows": 1,
        },
    ],
)
def plot_phot_diagnostic_kwargs(request):
    return request.param


# TODO: Generalize to run with all sed fitters
@pytest.mark.requires_data
@pytest.mark.lephare
def test_gal_lephare_loaded_plot_phot_diagnostic(
    gal_custom_lephare_loaded,
    custom_lephare_sed_fitter,
    data,
    plot_phot_diagnostic_kwargs,
):
    # fig, ax = plt.subplots()
    gal_custom_lephare_loaded.plot_phot_diagnostic(
        data,
        SED_arr=custom_lephare_sed_fitter,
        zPDF_arr=custom_lephare_sed_fitter,
        **plot_phot_diagnostic_kwargs,
    )


@pytest.mark.requires_data
@pytest.mark.lephare
def test_gal_lephare_eazy_plot_phot_diagnostic(
    gal_custom_lephare_eazy_loaded,
    custom_lephare_sed_fitter,
    eazy_fsps_larson_sed_fitter,
    data,
    plot_phot_diagnostic_kwargs,
):
    # fig, ax = plt.subplots()
    gal_custom_lephare_eazy_loaded.plot_phot_diagnostic(
        data,
        SED_arr=[custom_lephare_sed_fitter, eazy_fsps_larson_sed_fitter],
        zPDF_arr=[custom_lephare_sed_fitter, eazy_fsps_larson_sed_fitter],
        **plot_phot_diagnostic_kwargs,
    )


# Not lephare-marked (unlike the two tests above): the only
# plot_phot_diagnostic coverage that runs in CI, since lephare isn't
# pip-installable and its tests are excluded there (see the `lephare`
# marker's description in pyproject.toml).
@pytest.mark.requires_data
def test_gal_eazy_loaded_plot_phot_diagnostic(
    gal_eazy_loaded,
    eazy_fsps_larson_sed_fitter,
    data,
    plot_phot_diagnostic_kwargs,
):
    # deepcopy: plot_phot_diagnostic caches a multi_band_cutout onto the
    # galaxy (via make_cutouts), and gal_eazy_loaded is a shared
    # session-scoped fixture other tests assume is otherwise pristine
    gal = deepcopy(gal_eazy_loaded)
    gal.plot_phot_diagnostic(
        data,
        SED_arr=eazy_fsps_larson_sed_fitter,
        zPDF_arr=eazy_fsps_larson_sed_fitter,
        **plot_phot_diagnostic_kwargs,
    )


#################################################
# Pure-logic Galaxy tests below -- these deliberately avoid the `data`
# fixture (STPSF/FITS-backed) so they run fast and in CI. Tests that
# mutate a galaxy always `deepcopy` a session-scoped fixture (`high_z_gal`
# from conftest.py) first, matching the pattern conftest.py itself uses
# for `cat_eazy_sex_params_loaded`, since session fixtures are shared
# across the whole test session.
#################################################


@pytest.fixture(scope="module")
def bare_gal():
    """A minimal `Galaxy` with no real photometry -- for exercising
    logic that only touches ID/sky_coord/selection_flags/selection_kwargs,
    without depending on Photometry_obs/SED fitting machinery."""
    return Galaxy(
        ID=42,
        sky_coord=SkyCoord(ra=150.1234 * u.deg, dec=2.5678 * u.deg),
        aper_phot={},
        selection_flags={"is_high_z": True},
        selection_kwargs={"colour_cut": {"m1_m2": 1.5}},
        survey="test",
        version="v0",
    )


def test_galaxy_repr(bare_gal):
    assert repr(bare_gal) == "Galaxy(42, [150.12340,2.56780]deg)"


def test_galaxy_str(bare_gal):
    out = str(bare_gal)
    assert repr(bare_gal) in out
    assert "PHOTOMETRY:" in out
    assert "SELECTION FLAGS:" in out
    assert "is_high_z: True" in out


def test_galaxy_str_includes_aper_phot_repr(high_z_gal):
    # bare_gal has an empty aper_phot, so this covers the loop body
    # (`repr(phot_obs)` per aperture diameter) that test_galaxy_str can't
    aper_diam = next(iter(high_z_gal.aper_phot))
    out = str(high_z_gal)
    assert repr(high_z_gal.aper_phot[aper_diam]) in out


def test_galaxy_init_defaults_selection_dicts_to_empty(bare_gal):
    gal = Galaxy(
        ID=1,
        sky_coord=bare_gal.sky_coord,
        aper_phot={},
    )
    assert gal.selection_flags == {}
    assert gal.selection_kwargs == {}


def test_galaxy_getattr_ra_dec(bare_gal):
    assert bare_gal.RA == bare_gal.sky_coord.ra.degree * u.deg
    assert bare_gal.DEC == bare_gal.sky_coord.dec.degree * u.deg
    # lookup is case-insensitive for RA/DEC
    assert bare_gal.ra == bare_gal.RA


def test_galaxy_getattr_selection_flag(bare_gal):
    assert bare_gal.is_high_z is True


def test_galaxy_getattr_selection_kwarg(bare_gal):
    assert bare_gal.colour_cut__m1_m2 == 1.5


def test_galaxy_getattr_missing_raises(bare_gal):
    with pytest.raises(AttributeError):
        bare_gal.not_a_real_attribute


def test_galaxy_deepcopy_independent(bare_gal):
    gal_copy = deepcopy(bare_gal)
    assert gal_copy is not bare_gal
    assert gal_copy.ID == bare_gal.ID
    gal_copy.selection_flags["is_high_z"] = False
    assert bare_gal.selection_flags["is_high_z"] is True


def test_galaxy_deepcopy_failure_wraps_GalfindError(bare_gal):
    gal_copy = deepcopy(bare_gal)
    # threading.Lock objects can't be deepcopied -- stand in for any
    # attribute that fails mid-copy, to exercise the wrapping GalfindError
    gal_copy.unpicklable = threading.Lock()
    with pytest.raises(GalfindError, match="unpicklable"):
        deepcopy(gal_copy)


def test_galaxy_load_property(bare_gal):
    gal_copy = deepcopy(bare_gal)
    gal_copy.load_property(99.5 * u.Jy, "flux_prop")
    assert gal_copy.flux_prop == 99.5 * u.Jy


def test_update_SED_results_missing_aper_diam_raises(high_z_gal):
    gal = deepcopy(high_z_gal)
    fake_result = SimpleNamespace(aper_diam=999.0 * u.arcsec, SED_code="fake")
    with pytest.raises(MissingDataError):
        gal.update_SED_results(fake_result)


def test_update_SED_results_success(high_z_gal):
    gal = deepcopy(high_z_gal)
    aper_diam = next(iter(gal.aper_phot))
    fake_result = SimpleNamespace(aper_diam=aper_diam, SED_code="fake_code")
    gal.update_SED_results(fake_result)
    assert gal.aper_phot[aper_diam].SED_results["fake_code"] is fake_result


def test_update_SED_results_accepts_list(high_z_gal):
    gal = deepcopy(high_z_gal)
    aper_diam = next(iter(gal.aper_phot))
    fake_results = [
        SimpleNamespace(aper_diam=aper_diam, SED_code="fake_a"),
        SimpleNamespace(aper_diam=aper_diam, SED_code="fake_b"),
    ]
    gal.update_SED_results(fake_results)
    assert "fake_a" in gal.aper_phot[aper_diam].SED_results
    assert "fake_b" in gal.aper_phot[aper_diam].SED_results


def test_update_SED_result_lowz_zmax_info(
    high_z_gal, eazy_fsps_larson_sed_fitter
):
    gal = deepcopy(high_z_gal)
    aper_diam = next(iter(gal.aper_phot))
    label = eazy_fsps_larson_sed_fitter.label
    new_zmax_info = {"4.0": {"zbest": 1.0, "chi2_best": 99.0}}
    gal.update_SED_result_lowz_zmax_info(aper_diam, label, new_zmax_info)
    assert (
        gal.aper_phot[aper_diam].SED_results[label].lowz_zmax_properties
        == new_zmax_info
    )


def test_load_fixz_SED_result(high_z_gal):
    gal = deepcopy(high_z_gal)
    aper_diam = next(iter(gal.aper_phot))
    gal.load_fixz_SED_result(aper_diam, 5.678)
    sed_result = gal.aper_phot[aper_diam].SED_results["z"]
    assert sed_result.z == 5.678 * u.dimensionless_unscaled


def test_set_Vmax_missing_columns_raises(high_z_gal):
    gal = deepcopy(high_z_gal)
    tab = Table({"aper_diam": [0.32], "SED_fit_code": ["x"]})
    with pytest.raises(MissingKeyError):
        gal.set_Vmax(tab, "test_survey")


def test_set_Vmax_sums_per_region(high_z_gal, eazy_fsps_larson_sed_fitter):
    gal = deepcopy(high_z_gal)
    aper_diam = next(iter(gal.aper_phot))
    label = eazy_fsps_larson_sed_fitter.label
    tab = Table(
        {
            "aper_diam": [aper_diam.to(u.arcsec).value] * 3,
            "SED_fit_code": [label] * 3,
            "region": ["deep", "deep", "shallow"],
            "Vmax_total": [10.0, 5.0, 2.0],
        }
    )
    gal.set_Vmax(tab, "test_survey")
    sed_result = gal.aper_phot[aper_diam].SED_results[label]
    assert sed_result.Vmax["test_survey"]["deep"] == pytest.approx(15.0)
    assert sed_result.Vmax["test_survey"]["shallow"] == pytest.approx(2.0)


def test_set_Vmax_skips_unknown_aper_diam(high_z_gal):
    # an aper_diam the galaxy never had -> logged and skipped, not raised
    gal = deepcopy(high_z_gal)
    tab = Table(
        {
            "aper_diam": [999.0],
            "SED_fit_code": ["some_code"],
            "region": ["deep"],
            "Vmax_total": [10.0],
        }
    )
    gal.set_Vmax(tab, "test_survey")  # should not raise


def test_set_Vmax_skips_unknown_SED_fit_code(
    high_z_gal, eazy_fsps_larson_sed_fitter
):
    # an aper_diam the galaxy has, but a SED_fit_code it doesn't -> also
    # just logged and skipped, not raised
    gal = deepcopy(high_z_gal)
    aper_diam = next(iter(gal.aper_phot))
    tab = Table(
        {
            "aper_diam": [aper_diam.to(u.arcsec).value],
            "SED_fit_code": ["not_a_real_code"],
            "region": ["deep"],
            "Vmax_total": [10.0],
        }
    )
    gal.set_Vmax(tab, "test_survey")  # should not raise
    label = eazy_fsps_larson_sed_fitter.label
    assert not hasattr(gal.aper_phot[aper_diam].SED_results[label], "Vmax")


def test_set_Vmax_missing_rows_for_code_combo_raises(
    high_z_gal, eazy_fsps_larson_sed_fitter
):
    gal = deepcopy(high_z_gal)
    aper_diam = next(iter(gal.aper_phot))
    label = eazy_fsps_larson_sed_fitter.label
    # Inject a second, fake SED_fit_code into this galaxy's SED_results
    # so set_Vmax has to look up its per-(aper_diam, SED_fit_code) rows
    # in ecsv_rows -- as opposed to an aper_diam/code the galaxy never
    # had, which set_Vmax silently skips rather than raising on.
    gal.aper_phot[aper_diam].SED_results["fake_code"] = SimpleNamespace()
    tab = Table(
        {
            "aper_diam": [aper_diam.to(u.arcsec).value, 999.0],
            "SED_fit_code": [label, "fake_code"],
            "region": ["deep", "deep"],
            "Vmax_total": [10.0, 5.0],
        }
    )
    with pytest.raises(MissingDataError):
        gal.set_Vmax(tab, "test_survey")


#################################################
# Cutout/RGB smoke tests -- these need real FITS imaging (the `data`
# fixture) but not LePhare, unlike the plot_phot_diagnostic tests above.
# Each mutates a deepcopy of the shared session-scoped `gal_eazy_loaded`
# fixture (caches RGBs/cutouts as new attributes) to avoid leaking state
# into other tests that reuse it.
#################################################


@pytest.mark.requires_data
def test_make_RGB_caches_and_plot_RGB(gal_eazy_loaded, data):
    import matplotlib.pyplot as plt

    gal = deepcopy(gal_eazy_loaded)
    rgb_bands = {"B": ["F090W"], "G": ["F200W"], "R": ["F444W"]}
    rgb_obj = gal.make_RGB(data, rgb_bands=rgb_bands)
    assert hasattr(gal, "RGBs")
    # cached: a second call with the same args reuses rather than rebuilds
    assert gal.make_RGB(data, rgb_bands=rgb_bands) is rgb_obj

    fig, ax = plt.subplots()
    try:
        gal.plot_RGB(ax, rgb_bands)
    finally:
        plt.close(fig)


@pytest.mark.requires_data
def test_make_cutouts_caches_and_make_band_cutout(gal_eazy_loaded, data):
    gal = deepcopy(gal_eazy_loaded)
    multi_cutout = gal.make_cutouts(data)
    assert gal.multi_band_cutout is multi_cutout
    # cached: a second call reuses rather than rebuilding
    assert gal.make_cutouts(data) is multi_cutout

    band_cutout = gal.make_band_cutout(data["F444W"])
    assert "F444W_0.96as" in gal.cutouts
    assert gal.cutouts["F444W_0.96as"] is band_cutout


@pytest.mark.requires_data
def test_make_band_cutout_without_prior_multi_band_cutout(
    gal_eazy_loaded, data
):
    # exercises the fresh-build branch (no cached multi_band_cutout to
    # reuse a cutout from), as opposed to
    # test_make_cutouts_caches_and_make_band_cutout's reuse branch
    gal = deepcopy(gal_eazy_loaded)
    assert not hasattr(gal, "multi_band_cutout")
    band_cutout = gal.make_band_cutout(data["F444W"])
    assert "F444W_0.96as" in gal.cutouts
    assert gal.cutouts["F444W_0.96as"] is band_cutout


@pytest.mark.requires_data
def test_from_data_id(data):
    # also exercises Galaxy_Creator.from_data/__call__
    gal = Galaxy.from_data_id(data, 23)
    assert isinstance(gal, Galaxy)
    assert gal.ID == 23


@pytest.mark.requires_data
def test_plot_cutouts(
    gal_eazy_loaded, data, eazy_fsps_larson_sed_fitter, aper_diams
):
    import matplotlib.pyplot as plt

    gal = deepcopy(gal_eazy_loaded)
    fig = plt.figure()
    try:
        gal.plot_cutouts(
            fig,
            data,
            eazy_fsps_larson_sed_fitter,
            aper_diam=aper_diams[0],
        )
    finally:
        plt.close(fig)


@pytest.mark.requires_data
def test_plot_cutouts_incl_nodata_cutouts(
    gal_eazy_loaded, data, eazy_fsps_larson_sed_fitter, aper_diams
):
    import matplotlib.pyplot as plt

    gal = deepcopy(gal_eazy_loaded)
    fig = plt.figure()
    try:
        # incl_nodata_cutouts=True skips deepcopy-and-crop of `data` to
        # the galaxy's own filterset (the default-False branch, exercised
        # by test_plot_cutouts above)
        gal.plot_cutouts(
            fig,
            data,
            eazy_fsps_larson_sed_fitter,
            aper_diam=aper_diams[0],
            incl_nodata_cutouts=True,
        )
    finally:
        plt.close(fig)


@pytest.mark.requires_data
def test_load_sextractor_ext_src_corrs_and_plot_cutouts_kron(
    cat, eazy_fsps_larson_sed_fitter, data, aper_diams
):
    import matplotlib.pyplot as plt

    # `cat` (unlike gal_eazy_loaded) has no SED fitting run on it, but
    # plot_cutouts's SED_code param is currently unused in its active
    # code path (only referenced in commented-out scalebar code), so
    # that's fine here -- what this test actually needs `cat` for is a
    # deepcopy-able Catalogue to call load_sextractor_params() on, which
    # test_Galfit.py's real end-to-end fit test also relies on but only
    # runs when the `galfit` binary is on PATH (skipped otherwise) --
    # unlike that test, this one doesn't need galfit at all.
    cat_copy = deepcopy(cat)
    cat_copy.load_sextractor_params()
    gal = cat_copy[0]
    aper_diam = next(iter(gal.aper_phot))

    assert hasattr(
        gal,
        f"ext_src_corr_{aper_diam.to(u.arcsec).value:.2f}as_F444W",
    )

    # sex_KRON_RADIUS/sex_A_IMAGE/sex_B_IMAGE/sex_THETA_IMAGE (also set
    # by load_sextractor_params, via load_sextractor_kron_radii) make
    # plot_cutouts draw Kron ellipses alongside the fixed apertures
    assert all(
        hasattr(gal, name)
        for name in (
            "sex_KRON_RADIUS",
            "sex_A_IMAGE",
            "sex_B_IMAGE",
            "sex_THETA_IMAGE",
        )
    )
    fig = plt.figure()
    try:
        gal.plot_cutouts(
            fig,
            data,
            eazy_fsps_larson_sed_fitter,
            aper_diam=aper_diams[0],
        )
    finally:
        plt.close(fig)


def test_load_spectra_sets_attribute_and_returns_self(bare_gal):
    gal_copy = deepcopy(bare_gal)
    spectra = {"wavs": [1.0, 2.0], "flux": [3.0, 4.0]}
    result = gal_copy.load_spectra(spectra)
    assert result is gal_copy
    assert gal_copy.spectra == spectra


def test_plot_spec_diagnostic_without_spectra_is_a_noop(bare_gal):
    import matplotlib.pyplot as plt

    gal_copy = deepcopy(bare_gal)
    assert not hasattr(gal_copy, "spectra")
    fig, ax = plt.subplots()
    try:
        # plot_spec_diagnostic is currently a stub for both branches
        # (with/without `spectra` loaded); just check it doesn't raise
        gal_copy.plot_spec_diagnostic(ax)
    finally:
        plt.close(fig)


def test_plot_spec_diagnostic_with_spectra_is_a_noop(bare_gal):
    import matplotlib.pyplot as plt

    gal_copy = deepcopy(bare_gal)
    gal_copy.load_spectra({"wavs": [1.0], "flux": [2.0]})
    fig, ax = plt.subplots()
    try:
        gal_copy.plot_spec_diagnostic(ax)
    finally:
        plt.close(fig)


def test_extract_lowz_codes(high_z_gal, eazy_fsps_larson_sed_fitter):
    gal = deepcopy(high_z_gal)
    aper_diam = next(iter(gal.aper_phot))
    phot_obs = gal.aper_phot[aper_diam]
    z_gal = phot_obs.SED_results[eazy_fsps_larson_sed_fitter.label].z

    # two lowz-capped variants of the same templates, fit at redshifts
    # far enough below z_gal (9.5, minus the default lowz_dz=0.5) to
    # count as "genuinely inconsistent" low-z solutions
    lowz_zmax_4 = EAZY({"templates": "fsps_larson", "lowz_zmax": 4.0})
    lowz_zmax_6 = EAZY({"templates": "fsps_larson", "lowz_zmax": 6.0})
    sed_result_4 = SimpleNamespace(
        SED_code=lowz_zmax_4, aper_diam=aper_diam, z=3.5
    )
    sed_result_6 = SimpleNamespace(
        SED_code=lowz_zmax_6, aper_diam=aper_diam, z=5.5
    )
    phot_obs.SED_results[lowz_zmax_4.label] = sed_result_4
    phot_obs.SED_results[lowz_zmax_6.label] = sed_result_6

    lowz_codes = gal._extract_lowz_codes(
        aper_diam, [eazy_fsps_larson_sed_fitter]
    )

    # picks the *highest* lowz_zmax variant that's still inconsistent
    # with z_gal (i.e. the strongest available evidence against a low-z
    # solution)
    assert lowz_codes == [lowz_zmax_6]
    assert z_gal > 6.0 + 0.5


def test_extract_lowz_codes_none_when_consistent_with_lowz(
    high_z_gal, eazy_fsps_larson_sed_fitter
):
    gal = deepcopy(high_z_gal)
    aper_diam = next(iter(gal.aper_phot))
    phot_obs = gal.aper_phot[aper_diam]

    # a lowz-capped variant whose zmax is *not* far enough below z_gal
    # to be considered inconsistent (z_gal - lowz_dz <= lowz_zmax)
    lowz_zmax_9 = EAZY({"templates": "fsps_larson", "lowz_zmax": 9.2})
    sed_result_9 = SimpleNamespace(
        SED_code=lowz_zmax_9, aper_diam=aper_diam, z=9.0
    )
    phot_obs.SED_results[lowz_zmax_9.label] = sed_result_9

    lowz_codes = gal._extract_lowz_codes(
        aper_diam, [eazy_fsps_larson_sed_fitter]
    )
    assert lowz_codes == []


#################################################
# calc_Vmax input validation -- these all exit before `data` is ever
# touched (either raising, or -- for the out-of-z_bin case -- returning
# an early Vmax=-1.0 without dispatching to calc_Vmax_split_region), so
# `data=None` is safe to pass and no `requires_data`/STPSF dependency
# is pulled in.
#################################################


def test_calc_Vmax_invalid_method_raises(
    high_z_gal, eazy_fsps_larson_sed_fitter
):
    gal = deepcopy(high_z_gal)
    aper_diam = next(iter(gal.aper_phot))
    with pytest.raises(InvalidOptionError):
        gal.calc_Vmax(
            None,
            [0.0, 20.0],
            aper_diam,
            eazy_fsps_larson_sed_fitter,
            crops=[],
            Vmax_method="does_not_exist",
        )


def test_calc_Vmax_bad_z_bin_length_raises(
    high_z_gal, eazy_fsps_larson_sed_fitter
):
    gal = deepcopy(high_z_gal)
    aper_diam = next(iter(gal.aper_phot))
    with pytest.raises(LengthMismatchError):
        gal.calc_Vmax(
            None,
            [0.0],
            aper_diam,
            eazy_fsps_larson_sed_fitter,
            crops=[],
            Vmax_method="split_region",
        )


def test_calc_Vmax_reversed_z_bin_raises(
    high_z_gal, eazy_fsps_larson_sed_fitter
):
    gal = deepcopy(high_z_gal)
    aper_diam = next(iter(gal.aper_phot))
    with pytest.raises(RangeError):
        gal.calc_Vmax(
            None,
            [20.0, 0.0],
            aper_diam,
            eazy_fsps_larson_sed_fitter,
            crops=[],
            Vmax_method="split_region",
        )


def test_calc_Vmax_empty_crops_raises(
    high_z_gal, eazy_fsps_larson_sed_fitter
):
    gal = deepcopy(high_z_gal)
    aper_diam = next(iter(gal.aper_phot))
    with pytest.raises(RangeError):
        gal.calc_Vmax(
            None,
            [0.0, 20.0],
            aper_diam,
            eazy_fsps_larson_sed_fitter,
            crops=[],
            Vmax_method="split_region",
        )


def test_calc_Vmax_out_of_zbin_returns_minus_one(
    high_z_gal, eazy_fsps_larson_sed_fitter
):
    gal = deepcopy(high_z_gal)
    aper_diam = next(iter(gal.aper_phot))
    # a purely-photometric crop (not a Data_Selector/SED_fit_Selector),
    # so it survives calc_Vmax's crop filtering
    crop = Band_SNR_Selector(
        aper_diam=aper_diam,
        band="F444W",
        detect_or_non_detect="detect",
        SNR_lim=5.0,
    )
    # high_z_gal is fit at z~9.5 (see conftest.high_z_gal_z_true); pick a
    # z_bin that excludes it entirely
    Vmax, Vmax_kwargs, meta = gal.calc_Vmax(
        None,
        [0.0, 1.0],
        aper_diam,
        eazy_fsps_larson_sed_fitter,
        crops=[crop],
        Vmax_method="split_region",
    )
    assert Vmax == {"all": -1.0}
    assert Vmax_kwargs == {"all": {"zmin": -1.0, "zmax": -1.0}}
    assert meta == {}


class _NoOpPSF:
    """Minimal PSF stand-in for `_compute_Vmax`: its internal mock
    `Photometry_obs` is built with `simulated` left at its default
    (`False`), so `Photometry_obs.SNR` always applies an aperture
    correction from `self.psfs` -- with `psfs=None` that correction is
    `NaN` (see `Photometry_obs.aper_corrs`), which makes every SNR `NaN`
    and every selector fail regardless of how bright the mock source is.
    A zero-mag (i.e. no-op) correction keeps SNR meaningful without
    needing a real STPSF-backed PSF."""

    def get_aper_corrs(self, aper_diam, out_type="mag"):
        return 0.0


def test_compute_Vmax(high_z_gal, eazy_fsps_larson_sed_fitter):
    # _compute_Vmax takes plain explicit args (filterset/depths/area) --
    # unlike calc_Vmax_split_region, it doesn't need the real Data
    # area/depth pipeline (data.area_depths/unmasked_area), so it can be
    # exercised directly against the synthetic high_z_gal
    gal = deepcopy(high_z_gal)
    aper_diam = next(iter(gal.aper_phot))
    label = eazy_fsps_larson_sed_fitter.label
    sed_result = gal.aper_phot[aper_diam].SED_results[label]
    filterset = gal.cat_filterset

    # a flat, bright (well-detected at all trial redshifts) continuum
    # spanning far enough to cover every filter redshifted across the
    # z_step grid below
    wavs = np.linspace(500.0, 60_000.0, 2000)
    mags = np.full(2000, 25.0)
    sed_result.SED = SED_obs(sed_result.z.value, wavs, mags, u.AA, u.ABmag)

    depths = np.full(len(filterset), 28.5) * u.ABmag
    crop = Band_SNR_Selector(
        aper_diam=aper_diam,
        band="F444W",
        detect_or_non_detect="detect",
        SNR_lim=5.0,
    )

    Vmax, Vmax_kwargs = gal._compute_Vmax(
        zmin=9.0,
        zmax=10.0,
        filterset=filterset,
        depths=depths,
        aper_diam=aper_diam,
        psfs={name: _NoOpPSF() for name in filterset.filt_names},
        SED_fit_code=eazy_fsps_larson_sed_fitter,
        Vmax_crops=[crop],
        area=1.0 * u.arcmin**2,
        z_step=0.25,
    )
    # bright source, generous SNR cut -> detected throughout -> a real
    # (positive) volume, not the -1.0 too-few-detections sentinel
    assert Vmax > 0.0
    assert float(Vmax_kwargs["Vmax"]) == pytest.approx(Vmax)


def test_compute_Vmax_too_few_detections_returns_minus_one(
    high_z_gal, eazy_fsps_larson_sed_fitter
):
    gal = deepcopy(high_z_gal)
    aper_diam = next(iter(gal.aper_phot))
    label = eazy_fsps_larson_sed_fitter.label
    sed_result = gal.aper_phot[aper_diam].SED_results[label]
    filterset = gal.cat_filterset

    # a very faint continuum -> never passes the SNR cut at any trial z
    wavs = np.linspace(500.0, 60_000.0, 2000)
    mags = np.full(2000, 35.0)
    sed_result.SED = SED_obs(sed_result.z.value, wavs, mags, u.AA, u.ABmag)

    depths = np.full(len(filterset), 28.5) * u.ABmag
    crop = Band_SNR_Selector(
        aper_diam=aper_diam,
        band="F444W",
        detect_or_non_detect="detect",
        SNR_lim=5.0,
    )

    Vmax, Vmax_kwargs = gal._compute_Vmax(
        zmin=9.0,
        zmax=10.0,
        filterset=filterset,
        depths=depths,
        aper_diam=aper_diam,
        psfs={name: _NoOpPSF() for name in filterset.filt_names},
        SED_fit_code=eazy_fsps_larson_sed_fitter,
        Vmax_crops=[crop],
        area=1.0 * u.arcmin**2,
        z_step=0.25,
    )
    assert Vmax == -1.0
    assert Vmax_kwargs["zmin"] == "-1.0"
    assert Vmax_kwargs["zmax"] == "-1.0"

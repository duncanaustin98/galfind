import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

from galfind import useful_funcs_austind as funcs


@pytest.fixture(
    scope="module",
    params=[
        {},  # default
    ],
)
def ext_src_corr_inputs(request):
    return request.param


@pytest.mark.requires_data
def test_get_ext_src_corr_pass(
    phot_rest_sex_params_loaded, ext_src_corr_inputs
):
    ext_src_corr = funcs.get_ext_src_corr(
        phot_rest_sex_params_loaded, **ext_src_corr_inputs
    )
    assert isinstance(ext_src_corr, float)


@pytest.mark.requires_data
def test_get_ext_src_corr_no_sex_params_fail(phot_rest, ext_src_corr_inputs):
    # phot_rest is a shared session fixture that other tests/fixtures may
    # have already populated ext_src_corrs onto by the time this runs, so
    # explicitly clear it on a copy to reliably exercise "not pre-loaded"
    # regardless of test order, rather than relying on phot_rest happening
    # to still be untouched
    from copy import deepcopy

    phot_rest_no_ext_src_corrs = deepcopy(phot_rest)
    phot_rest_no_ext_src_corrs.ext_src_corrs = {}
    with pytest.raises(AttributeError):
        funcs.get_ext_src_corr(
            phot_rest_no_ext_src_corrs, **ext_src_corr_inputs
        )


def test_blank_phot_rest_ext_src_corr_nan(
    blank_phot_rest, ext_src_corr_inputs
):
    ext_src_corr = funcs.get_ext_src_corr(
        blank_phot_rest, **ext_src_corr_inputs
    )
    assert np.isnan(ext_src_corr)


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "depth": 28.0,
                "zero_point": 8.9,
            },
            True,
        ),
        (
            {
                "depth": 28.0 * u.ABmag,
                "zero_point": 8.9,
            },
            True,
        ),
    ],
)
def calc_1sigma_flux_inputs(request):
    return request.param


def test_calc_1sigma_flux(calc_1sigma_flux_inputs):
    inputs, outcome = calc_1sigma_flux_inputs
    if outcome is not True:
        with pytest.raises(outcome):
            funcs.calc_1sigma_flux(**inputs)
    else:
        funcs.calc_1sigma_flux(**inputs)


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "five_sigma_depth": 28.0,
                "n": 2,
            },
            True,
        ),
        (
            {
                "five_sigma_depth": 28.0 * u.ABmag,
                "n": 2,
            },
            True,
        ),
        (
            {
                "five_sigma_depth": 28.0,
                "n": 0.4,
            },
            True,
        ),
        (
            {
                "five_sigma_depth": 28.0,
                "n": -1,
            },
            AssertionError,
        ),
    ],
)
def five_to_n_sigma_mag_inputs(request):
    return request.param


def test_five_to_n_sigma_mag(five_to_n_sigma_mag_inputs):
    inputs, outcome = five_to_n_sigma_mag_inputs
    if outcome is not True:
        with pytest.raises(outcome):
            funcs.five_to_n_sigma_mag(**inputs)
    else:
        funcs.five_to_n_sigma_mag(**inputs)


def test_to_scalar_plain_number():
    assert funcs.to_scalar(5) == 5.0
    assert isinstance(funcs.to_scalar(5), float)


def test_to_scalar_quantity():
    assert funcs.to_scalar(5.0 * u.Jy) == 5.0
    assert funcs.to_scalar(28.0 * u.ABmag) == 28.0


@pytest.fixture(
    scope="module",
    params=[
        (1.0, 8.9),
        (0.1, 8.9),
        (10.0, 25.0),
    ],
)
def mag_flux_roundtrip_inputs(request):
    return request.param


def test_flux_mag_roundtrip(mag_flux_roundtrip_inputs):
    flux, zero_point = mag_flux_roundtrip_inputs
    mag = funcs.flux_to_mag(flux, zero_point)
    assert funcs.mag_to_flux(mag, zero_point) == pytest.approx(flux)


def test_flux_to_mag_strips_quantity_value():
    flux = 1.0
    zero_point = 8.9
    mag_plain = funcs.flux_to_mag(flux, zero_point)
    mag_quantity = funcs.flux_to_mag(flux * u.Jy, zero_point)
    assert mag_quantity == pytest.approx(mag_plain)


@pytest.fixture(
    scope="module",
    params=[0.1, 1.0, 2.5],
)
def flux_ratio_inputs(request):
    return request.param


def test_flux_mag_ratio_roundtrip(flux_ratio_inputs):
    mag_ratio = funcs.flux_to_mag_ratio(flux_ratio_inputs)
    assert funcs.mag_to_flux_ratio(mag_ratio) == pytest.approx(
        flux_ratio_inputs
    )


def test_flux_to_mag_ratio_unity_is_zero():
    assert funcs.flux_to_mag_ratio(1.0) == pytest.approx(0.0)


def test_flux_pc_to_mag_err_zero():
    assert funcs.flux_pc_to_mag_err(0.0) == pytest.approx(0.0)


def test_flux_pc_to_mag_err_positive():
    assert funcs.flux_pc_to_mag_err(0.1) > 0.0


@pytest.fixture(
    scope="module",
    params=[
        (1e-9, 28.9),
        (5e-8, 25.0),
    ],
)
def loc_depth_roundtrip_inputs(request):
    return request.param


def test_flux_err_loc_depth_roundtrip(loc_depth_roundtrip_inputs):
    flux_err, zero_point = loc_depth_roundtrip_inputs
    loc_depth = funcs.flux_err_to_loc_depth(flux_err, zero_point)
    assert funcs.loc_depth_to_flux_err(
        loc_depth, zero_point
    ) == pytest.approx(flux_err)


@pytest.fixture(
    scope="module",
    params=[0.0, 1.0, 5.0, 13.0],
)
def wav_z_inputs(request):
    return request.param


def test_wav_obs_rest_roundtrip(wav_z_inputs):
    wav_obs = 1.0 * u.um
    wav_rest = funcs.wav_obs_to_rest(wav_obs, wav_z_inputs)
    wav_obs_recovered = funcs.wav_rest_to_obs(wav_rest, wav_z_inputs)
    assert wav_obs_recovered.to(u.um).value == pytest.approx(
        wav_obs.to(u.um).value
    )


def test_wav_obs_to_rest_z_zero_is_unchanged():
    wav_obs = 2.0 * u.um
    assert funcs.wav_obs_to_rest(wav_obs, 0.0) == wav_obs


@pytest.fixture(
    scope="module",
    params=[
        (1, "1st"),
        (2, "2nd"),
        (3, "3rd"),
        (4, "4th"),
        (11, "11th"),
        (12, "12th"),
        (13, "13th"),
        (21, "21st"),
        (22, "22nd"),
        (23, "23rd"),
        (101, "101st"),
        (111, "111th"),
        (112, "112th"),
        (113, "113th"),
    ],
)
def ordinal_inputs(request):
    return request.param


def test_ordinal(ordinal_inputs):
    n, expected = ordinal_inputs
    assert funcs.ordinal(n) == expected


@pytest.fixture(
    scope="module",
    params=[None, 4.0, 6.5],
)
def lowz_zmax_inputs(request):
    return request.param


def test_lowz_label_roundtrip(lowz_zmax_inputs):
    label = funcs.lowz_label(lowz_zmax_inputs)
    assert funcs.zmax_from_lowz_label(label) == lowz_zmax_inputs


def test_lowz_label_free():
    assert funcs.lowz_label(None) == "zfree"


def test_get_z_bin_name():
    assert funcs.get_z_bin_name([6.0, 7.5]) == "6.0<z<7.5"


def test_get_SED_fit_label_aper_diam_z_bin_name():
    label = funcs.get_SED_fit_label_aper_diam_z_bin_name(
        "EAZY_fsps_larson", 0.32 * u.arcsec, [6.0, 7.5]
    )
    assert label == "EAZY_fsps_larson_0.32as_6.0<z<7.5"


def test_aper_diams_to_str_single():
    assert funcs.aper_diams_to_str([0.32] * u.arcsec) == "(0.32)as"


def test_aper_diams_to_str_multiple():
    aper_diams = np.array([0.32, 0.5]) * u.arcsec
    assert funcs.aper_diams_to_str(aper_diams) == "(0.32,0.50)as"


def test_calc_unmasked_area():
    pixel_scale = 1.0 * u.arcsec
    mask = np.ones((10, 10), dtype=bool)
    area = funcs.calc_unmasked_area(mask, pixel_scale)
    expected = (100 * u.arcsec**2).to(u.arcmin**2)
    assert area.value == pytest.approx(expected.value)


def test_calc_unmasked_area_tuple_and():
    pixel_scale = 1.0 * u.arcsec
    mask_1 = np.array([[True, True], [False, False]])
    mask_2 = np.array([[True, False], [False, True]])
    area = funcs.calc_unmasked_area((mask_1, mask_2), pixel_scale)
    expected = (1 * u.arcsec**2).to(u.arcmin**2)
    assert area.value == pytest.approx(expected.value)


def test_poisson_interval_zero_count():
    low, high = funcs.poisson_interval(0)
    assert low == 0.0
    assert high > 0.0


def test_poisson_interval_positive_count_brackets_k():
    low, high = funcs.poisson_interval(10)
    assert low < 10.0 < high


def test_gauss_func_peak_at_mu():
    mu, sigma = 2.0, 1.5
    assert funcs.gauss_func(mu, mu, sigma) == pytest.approx(np.pi * sigma)


def test_gauss_func_symmetric_about_mu():
    mu, sigma = 0.0, 1.0
    assert funcs.gauss_func(1.0, mu, sigma) == pytest.approx(
        funcs.gauss_func(-1.0, mu, sigma)
    )


def test_power_law_func():
    assert funcs.power_law_func(2.0, 3.0, 2.0) == pytest.approx(12.0)


def test_simple_power_law_func():
    assert funcs.simple_power_law_func(2.0, 1.0, 3.0) == pytest.approx(7.0)


def test_truncate_colname_short_unchanged():
    col = "short_colname"
    assert funcs.truncate_colname(col) == col


def test_truncate_colname_long_is_bounded():
    col = "flux_" + "F444W" * 20
    assert len(col) > 68
    out = funcs.truncate_colname(col)
    assert len(out) <= 68


def test_all_subclasses():
    class Base:
        pass

    class Child(Base):
        pass

    class Grandchild(Child):
        pass

    assert set(funcs.all_subclasses(Base)) == {Child, Grandchild}


def test_all_subclasses_no_subclasses():
    class Leaf:
        pass

    assert funcs.all_subclasses(Leaf) == ()


def test_flux_image_to_Jy_scalar_matches_jy_zero_point():
    jy_zero_point = (1 * u.Jy).to(u.ABmag).value
    result = funcs.flux_image_to_Jy(2.5, jy_zero_point)
    assert result.unit == u.Jy
    assert result.value == pytest.approx(2.5)


def test_flux_image_to_Jy_array():
    jy_zero_point = (1 * u.Jy).to(u.ABmag).value
    result = funcs.flux_image_to_Jy([1.0, 2.0], jy_zero_point)
    assert result.unit == u.Jy
    np.testing.assert_allclose(result.value, [1.0, 2.0])


@pytest.fixture(
    scope="module",
    params=[0.5, 1.0, 2.0],
)
def flux_lambda_roundtrip_wav(request):
    return request.param


def test_flux_jy_lambda_roundtrip(flux_lambda_roundtrip_wav):
    wav = flux_lambda_roundtrip_wav * u.um
    flux_jy = 1.0 * u.Jy
    flux_lambda = funcs.flux_Jy_to_lambda(flux_jy, wav)
    recovered = funcs.flux_lambda_to_Jy(flux_lambda, wav)
    assert recovered.to(u.Jy).value == pytest.approx(flux_jy.value)


def test_lum_nu_lam_roundtrip():
    wav = 5000 * u.Angstrom
    lum_nu = 1e30 * u.erg / u.s / u.Hz
    lum_lam = funcs.lum_nu_to_lum_lam(lum_nu, wav)
    recovered = funcs.lum_lam_to_lum_nu(lum_lam, wav)
    assert recovered.to(lum_nu.unit).value == pytest.approx(lum_nu.value)


def test_parse_s_region_valid():
    s_region = "POLYGON ICRS 10.0 20.0 11.0 20.0 11.0 21.0 10.0 21.0"
    coords = funcs.parse_s_region(s_region)
    assert coords.shape == (4, 2)
    assert coords[0].tolist() == [10.0, 20.0]


def test_parse_s_region_invalid_format():
    assert funcs.parse_s_region("not a polygon string") is None


def test_parse_s_region_odd_number_of_values():
    assert funcs.parse_s_region("POLYGON ICRS 10.0 20.0 11.0") is None


def test_linear_fit_recovers_known_line():
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    y = 2.0 * x + 1.0
    slope, intercept = funcs.linear_fit(x, y)
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)


def test_interpolate_linear_fit():
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    y = 2.0 * x + 1.0
    assert funcs.interpolate_linear_fit(x, y, 5.0) == pytest.approx(11.0)


def test_residual_sum_of_squares_zero_for_perfect_fit():
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    y = 2.0 * x + 1.0
    params = np.array([2.0, 1.0], dtype=np.float64)
    assert funcs.residual_sum_of_squares(params, x, y) == pytest.approx(0.0)


def test_residual_sum_of_squares_positive_for_imperfect_fit():
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    y = np.array([0.0, 3.0, 3.0], dtype=np.float64)
    params = np.array([1.0, 0.0], dtype=np.float64)
    assert funcs.residual_sum_of_squares(params, x, y) > 0.0


def test_get_depth_dir():
    out = funcs.get_depth_dir("/work", "JADES", "v11", ["ACS_WFC", "NIRCam"])
    assert list(out) == [
        "/work/Depths/ACS_WFC/v11/JADES",
        "/work/Depths/NIRCam/v11/JADES",
    ]


def test_get_eazy_dir():
    out = funcs.get_eazy_dir("/work", "JADES", "v11", ["ACS_WFC", "NIRCam"])
    assert list(out) == [
        "/work/EAZY/input/ACS_WFC+NIRCam/v11/JADES",
        "/work/EAZY/output/ACS_WFC+NIRCam/v11/JADES",
    ]


def test_get_mask_dir():
    out = funcs.get_mask_dir("/work", "JADES")
    assert list(out) == ["/work/Masks/JADES"]


def test_get_sex_dir():
    out = funcs.get_sex_dir("/work", "JADES", "v11", ["NIRCam"])
    assert list(out) == ["/work/SExtractor/NIRCam/v11/JADES"]


def test_get_stacked_images_dir():
    out = funcs.get_stacked_images_dir("/work", "JADES", "v11", ["NIRCam"])
    assert list(out) == ["/work/Stacked_Images/v11/NIRCam/JADES"]


@pytest.fixture(
    scope="module",
    params=[
        ("Depths", ["/work/Depths/NIRCam/v11/JADES"]),
        (
            "EAZY",
            [
                "/work/EAZY/input/NIRCam/v11/JADES",
                "/work/EAZY/output/NIRCam/v11/JADES",
            ],
        ),
        ("Masks", ["/work/Masks/JADES"]),
        ("SExtractor", ["/work/SExtractor/NIRCam/v11/JADES"]),
        ("Stacked_Images", ["/work/Stacked_Images/v11/NIRCam/JADES"]),
        ("Unrecognised", ValueError),
    ],
)
def find_target_dir_inputs(request):
    return request.param


def test_find_target_dir(find_target_dir_inputs):
    keyword, expected = find_target_dir_inputs
    if expected is ValueError:
        with pytest.raises(ValueError):
            funcs.find_target_dir(
                "/work", "JADES", "v11", ["NIRCam"], keyword
            )
    else:
        out = funcs.find_target_dir(
            "/work", "JADES", "v11", ["NIRCam"], keyword
        )
        assert list(out) == expected


def test_n_sigma_detection_at_depth_is_5sigma():
    assert funcs.n_sigma_detection(
        25.0, 25.0, 28.9
    ) == pytest.approx(5.0)


def test_convert_wav_units_same_unit_returns_input():
    wavs = 1.0 * u.um
    assert funcs.convert_wav_units(wavs, u.um) is wavs


def test_convert_wav_units_converts():
    wavs = 1.0 * u.um
    out = funcs.convert_wav_units(wavs, u.AA)
    assert out.unit == u.AA
    assert out.value == pytest.approx(10000.0)


def test_convert_mag_units_same_unit_is_passthrough():
    mags = 1.0 * u.Jy
    assert funcs.convert_mag_units(1.0 * u.um, mags, u.Jy) is mags


def test_convert_mag_units_jy_abmag_roundtrip():
    wavs = 1.0 * u.um
    mags = 1.0 * u.Jy
    ab_mag = funcs.convert_mag_units(wavs, mags, u.ABmag)
    recovered = funcs.convert_mag_units(wavs, ab_mag, u.Jy)
    assert recovered.to(u.Jy).value == pytest.approx(mags.value)


def test_convert_mag_units_invalid_units_raises():
    with pytest.raises(Exception):
        funcs.convert_mag_units(1.0 * u.um, 1.0 * u.Jy, u.m)


def test_convert_mag_err_units_same_unit_is_passthrough():
    mags = np.array([1.0, 1.0]) * u.Jy
    errs = [np.array([0.1, 0.1]) * u.Jy, np.array([0.15, 0.15]) * u.Jy]
    assert funcs.convert_mag_err_units(1.0 * u.um, mags, errs, u.Jy) is errs


def test_convert_mag_err_units_jy_to_abmag_positive():
    wavs = np.array([1.0, 1.0]) * u.um
    mags = np.array([1.0, 1.0]) * u.Jy
    errs = [np.array([0.1, 0.1]) * u.Jy, np.array([0.15, 0.15]) * u.Jy]
    l1, u1 = funcs.convert_mag_err_units(wavs, mags, errs, u.ABmag)
    assert np.all(l1.value > 0.0)
    assert np.all(u1.value > 0.0)


def test_convert_mag_err_units_mismatched_units_raises():
    wavs = np.array([1.0, 1.0]) * u.um
    mags = np.array([1.0, 1.0]) * u.Jy
    errs = [
        np.array([0.1, 0.1]) * u.ABmag,
        np.array([0.15, 0.15]) * u.ABmag,
    ]
    with pytest.raises(AssertionError):
        funcs.convert_mag_err_units(wavs, mags, errs, u.Jy)


def test_log_scale_fluxes():
    fluxes = np.array([1.0, 10.0, 100.0]) * u.Jy
    np.testing.assert_allclose(
        funcs.log_scale_fluxes(fluxes), [0.0, 1.0, 2.0]
    )


def test_log_scale_flux_errors():
    fluxes = np.array([10.0]) * u.Jy
    flux_errs = [np.array([1.0]) * u.Jy, np.array([1.0]) * u.Jy]
    log_l1, log_u1 = funcs.log_scale_flux_errors(fluxes, flux_errs)
    assert log_l1[0] == pytest.approx(
        np.log10(10.0) - np.log10(9.0)
    )
    assert log_u1[0] == pytest.approx(
        np.log10(11.0) - np.log10(10.0)
    )


def test_log_scale_flux_errors_wrong_length_raises():
    fluxes = np.array([10.0]) * u.Jy
    with pytest.raises(AssertionError):
        funcs.log_scale_flux_errors(fluxes, [np.array([1.0]) * u.Jy])


def test_dust_correct_positive_mag_applies_correction():
    lum = np.array([1.0, 1.0]) * u.erg / u.s
    dust_mag = np.array([0.0, 1.0]) * u.mag
    corrected = funcs.dust_correct(lum, dust_mag)
    assert corrected[0].value == pytest.approx(1.0)
    assert corrected[1].value == pytest.approx(10 ** (1.0 / 2.5))


def test_label_log():
    assert funcs.label_log("x") == r"$\log_{10}($x$)$"


def test_errs_to_log_no_uplim():
    data = np.array([10.0])
    data_err = [np.array([1.0]), np.array([1.0])]
    log_data, (log_l1, log_u1), uplims = funcs.errs_to_log(data, data_err)
    assert log_data[0] == pytest.approx(np.log10(10.0))
    assert not np.any(uplims)


def test_errs_to_log_lower_bound_below_zero_uses_inf_val():
    data = np.array([1.0])
    data_err = [np.array([2.0]), np.array([1.0])]
    _, (log_l1, _), _ = funcs.errs_to_log(data, data_err, inf_val=1e6)
    assert log_l1[0] == 1e6


def test_split_dir_name_dir():
    assert funcs.split_dir_name("/a/b/c.txt", "dir") == "/a/b/"


def test_split_dir_name_name():
    assert funcs.split_dir_name("/a/b/c.txt", "name") == "c.txt"


def test_date_finder_iso():
    assert funcs.date_finder("run on 2024-01-15 please") == ["2024-01-15"]


def test_date_finder_slash():
    assert funcs.date_finder("run on 15/01/2024 please") == ["15/01/2024"]


def test_date_finder_no_match():
    assert funcs.date_finder("no dates here") == []


def test_validate_quantity_none_returns_none():
    assert funcs.validate_quantity(None, "length") is None


def test_validate_quantity_non_quantity_returns_none():
    assert funcs.validate_quantity(5.0, "length") is None


def test_validate_quantity_correct_type_passthrough():
    quant = 1.0 * u.um
    assert funcs.validate_quantity(quant, "length") is quant


def test_validate_quantity_wrong_type_raises():
    with pytest.raises(AssertionError):
        funcs.validate_quantity(1.0 * u.um, "time")


def test_beta_slope_power_law_func():
    assert funcs.beta_slope_power_law_func(
        10.0, 1.0, -2.0
    ) == pytest.approx(0.1)


def test_crop_to_Calzetti94_filters():
    # windows include (1268,1284) and (1342,1371); 1290 falls in neither
    wavs = np.array([1275.0, 1290.0, 1350.0]) * u.AA
    mags = np.array([1.0, 2.0, 3.0])
    cropped_wavs, cropped_mags = funcs.crop_to_Calzetti94_filters(
        wavs, mags
    )
    assert cropped_wavs.value.tolist() == [1275.0, 1350.0]
    assert cropped_mags.tolist() == [1.0, 3.0]


def test_ext_source_corr_log_data():
    assert funcs.ext_source_corr(
        1.0, 100.0, is_log_data=True
    ) == pytest.approx(3.0)


def test_ext_source_corr_linear_data():
    assert funcs.ext_source_corr(
        2.0, 3.0, is_log_data=False
    ) == pytest.approx(6.0)


def test_power_law_beta_func():
    assert funcs.power_law_beta_func(2.0, 3.0, 2.0) == pytest.approx(12.0)


def test_singleton_returns_same_instance():
    class MySingleton(funcs.Singleton):
        pass

    assert MySingleton() is MySingleton()


def test_rolling_average():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(
        funcs.rolling_average(y, 2), [1.5, 2.5, 3.5, 4.5]
    )


def test_group_positions_single_group():
    coords = SkyCoord(
        ra=[10.0, 10.0001] * u.deg, dec=[20.0, 20.0001] * u.deg
    )
    groups = funcs.group_positions(coords, match_radius=5.0 * u.arcsec)
    assert len(groups) == 1
    (indices,) = groups.values()
    assert sorted(indices) == [0, 1]


def test_group_positions_two_groups():
    coords = SkyCoord(
        ra=[10.0, 50.0] * u.deg, dec=[20.0, 60.0] * u.deg
    )
    groups = funcs.group_positions(coords, match_radius=5.0 * u.arcsec)
    assert len(groups) == 2


def test_source_separation_zero_for_identical_coords():
    coord = SkyCoord(ra=10.0 * u.deg, dec=20.0 * u.deg)
    sep = funcs.source_separation(coord, coord, z=6.0)
    assert sep.to(u.kpc).value == pytest.approx(0.0)


def test_calc_Vmax_positive():
    vmax = funcs.calc_Vmax(1.0 * u.sr, 6.0, 7.0)
    assert vmax.unit == u.Mpc**3
    assert vmax.value > 0.0


def test_get_ext_src_corr_label_default():
    assert funcs.get_ext_src_corr_label() == "_extsrc_UV<10"


def test_get_ext_src_corr_label_no_uplim():
    assert (
        funcs.get_ext_src_corr_label("UV", None) == "_extsrc_UV"
    )


def test_get_ext_src_corr_label_none_key():
    assert funcs.get_ext_src_corr_label(None) == ""


def test_truncate_colname_hard_truncated_when_no_filter_match():
    col = "some_very_long_column_name_with_no_filter_pattern_" * 2
    assert len(col) > 68
    out = funcs.truncate_colname(col)
    assert out == col[:68]
    assert len(out) == 68


def test_fits_cat_to_np_reshape_by_aper_diams():
    tab = Table(
        {
            "a": [[1.0, 2.0], [3.0, 4.0]],
            "b": [[5.0, 6.0], [7.0, 8.0]],
        }
    )
    arr = funcs.fits_cat_to_np(tab, ["a", "b"])
    assert arr.shape == (2, 2, 2)


def test_fits_cat_to_np_no_reshape():
    tab = Table({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    arr = funcs.fits_cat_to_np(tab, ["a", "b"], reshape_by_aper_diams=False)
    assert arr.shape == (2, 2)


def test_fits_cat_to_np_empty_table_raises():
    tab = Table({"a": [], "b": []})
    with pytest.raises(AssertionError):
        funcs.fits_cat_to_np(tab, ["a", "b"])


def test_symlink_creates_link(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("data")
    link = tmp_path / "nested" / "link.txt"
    funcs.symlink(str(target), str(link))
    assert link.is_symlink()
    assert link.resolve() == target.resolve()


def test_symlink_missing_target_does_not_raise(tmp_path):
    link = tmp_path / "link.txt"
    funcs.symlink(str(tmp_path / "missing.txt"), str(link))
    assert not link.exists()


def test_make_dirs_creates_parent(tmp_path):
    target = tmp_path / "a" / "b" / "file.txt"
    funcs.make_dirs(str(target))
    assert (tmp_path / "a" / "b").is_dir()


def test_change_file_permissions_single_path(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("data")
    funcs.change_file_permissions(str(target), permissions=0o644)
    assert (target.stat().st_mode & 0o777) == 0o644


def test_change_file_permissions_list_of_paths(tmp_path):
    targets = [tmp_path / "a.txt", tmp_path / "b.txt"]
    for t in targets:
        t.write_text("data")
    funcs.change_file_permissions(
        [str(t) for t in targets], permissions=0o600
    )
    for t in targets:
        assert (t.stat().st_mode & 0o777) == 0o600


def test_change_file_permissions_missing_file_does_not_raise(tmp_path):
    funcs.change_file_permissions(str(tmp_path / "missing.txt"))


def test_change_file_permissions_log_true(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("data")
    funcs.change_file_permissions(str(target), permissions=0o644, log=True)
    assert (target.stat().st_mode & 0o777) == 0o644


def test_symlink_existing_symlink_is_left_alone(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("data")
    link = tmp_path / "link.txt"
    funcs.symlink(str(target), str(link))
    # calling again should hit the FileExistsError branch, not raise
    funcs.symlink(str(target), str(link))
    assert link.resolve() == target.resolve()


def test_adjust_errs():
    data = np.array([5.0])
    data_err = [np.array([3.0]), np.array([8.0])]
    out_data, out_err = funcs.adjust_errs(data, data_err)
    assert out_data is data
    np.testing.assert_allclose(out_err, [[2.0], [3.0]])


def test_errs_to_log_uplim_sigma_branch():
    # data + data_err[1] < 0 makes log_u1 undefined, triggering the
    # upper-limit branch (while data + uplim_sigma * data_err[1] > 0
    # keeps the reduced-significance replacement finite)
    data = np.array([1.0])
    data_err = [np.array([0.5]), np.array([-1.5])]
    log_data, (log_l1, log_u1), uplims = funcs.errs_to_log(
        data, data_err, uplim_sigma=0.5
    )
    assert np.all(uplims)
    assert log_u1[0] == 0.0
    assert log_l1[0] == pytest.approx(0.2)
    assert log_data[0] == pytest.approx(np.log10(0.25))


def test_get_phot_cat_path():
    path = funcs.get_phot_cat_path(
        "test_survey",
        "v1",
        "NIRCam",
        [0.32] * u.arcsec,
        "F444W",
    )
    assert path.endswith(
        "NIRCam/test_survey/(0.32)as/"
        "test_survey_MASTER_Sel-F444W_v1.fits"
    )


def test_get_phot_cat_path_no_forced_phot_band():
    path = funcs.get_phot_cat_path(
        "test_survey", "v1", "NIRCam", [0.32] * u.arcsec, None
    )
    assert path.endswith("test_survey_v1.fits")


def test_fits_cat_to_np_scalar_cells_reshape():
    tab = Table({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    arr = funcs.fits_cat_to_np(tab, ["a", "b"])
    assert arr.shape == (2, 2, 1)


def test_tqdm_joblib_restores_callback_and_closes():
    import joblib

    original_callback = joblib.parallel.BatchCompletionCallBack

    class DummyTqdm:
        def __init__(self):
            self.n = 0
            self.closed = False

        def update(self, n=1):
            self.n += n

        def close(self):
            self.closed = True

    dummy = DummyTqdm()
    with funcs.tqdm_joblib(dummy):
        assert (
            joblib.parallel.BatchCompletionCallBack is not original_callback
        )
    assert joblib.parallel.BatchCompletionCallBack is original_callback
    assert dummy.closed


class FakeFilter:
    def __init__(self, name, lower, upper):
        self.filt_name = name
        self.WavelengthLower50 = lower
        self.WavelengthUpper50 = upper


@pytest.fixture(scope="module")
def fake_filterset():
    return [
        FakeFilter("F090W", 800.0 * u.nm, 1000.0 * u.nm),
        FakeFilter("F150W", 1300.0 * u.nm, 1700.0 * u.nm),
        FakeFilter("F200W", 1750.0 * u.nm, 2250.0 * u.nm),
    ]


def test_get_first_bluewards_band(fake_filterset):
    band = funcs.get_first_bluewards_band(
        0.0, fake_filterset, 1500.0 * u.nm
    )
    assert band == "F090W"


def test_get_first_bluewards_band_negative_z(fake_filterset):
    assert (
        funcs.get_first_bluewards_band(-1.0, fake_filterset, 1500.0 * u.nm)
        is None
    )


def test_get_first_bluewards_band_ignore_bands(fake_filterset):
    band = funcs.get_first_bluewards_band(
        0.0, fake_filterset, 2200.0 * u.nm, ignore_bands="F200W"
    )
    assert band == "F150W"


def test_get_first_redwards_band(fake_filterset):
    band = funcs.get_first_redwards_band(
        0.0, fake_filterset, 1200.0 * u.nm
    )
    assert band == "F150W"


def test_get_first_redwards_band_none_found(fake_filterset):
    assert (
        funcs.get_first_redwards_band(0.0, fake_filterset, 3000.0 * u.nm)
        is None
    )


def test_flux_luminosity_roundtrip_default_units():
    wav = 5000.0 * u.AA
    z = 2.0
    flux_in = 1.0 * u.Jy
    lum = funcs.flux_to_luminosity(flux_in, wav, z)
    flux_out = funcs.luminosity_to_flux(lum, wav, z)
    assert flux_out.to(u.Jy).value == pytest.approx(flux_in.value)


def test_luminosity_to_flux_from_L_lambda():
    wav = 5000.0 * u.AA
    lum_lam = 1e30 * u.erg / u.s / u.AA
    flux = funcs.luminosity_to_flux(lum_lam, wav, 2.0)
    assert flux.unit == u.Jy
    assert flux.value > 0.0


def test_flux_to_luminosity_from_ABmag():
    wav = 5000.0 * u.AA
    lum = funcs.flux_to_luminosity(25.0 * u.ABmag, wav, 2.0)
    assert lum.value > 0.0


def test_flux_to_luminosity_from_f_lambda():
    wav = 5000.0 * u.AA
    flux_lam = 1e-20 * u.erg / u.s / u.cm**2 / u.AA
    lum = funcs.flux_to_luminosity(flux_lam, wav, 2.0)
    assert lum.value > 0.0

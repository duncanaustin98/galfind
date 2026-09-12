from copy import deepcopy
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

from galfind.catalogues import Catalogue, Catalogue_Creator
from galfind.catalogues.Catalogue import (
    check_hdu_exists,
    galfind_depth_labels,
    galfind_mask_labels,
    galfind_phot_labels,
    galfind_selection_labels,
    galfind_snr_labels,
    jaguar_phot_labels,
    load_bool_Table,
    load_galfind_phot,
    load_phot,
    open_galfind_cat,
    open_galfind_hdr,
    phot_property_from_fits,
    phot_property_from_galfind_tab,
    scattered_depth_labels,
    scattered_phot_labels,
)
from galfind.catalogues.Catalogue_Base import Catalogue_Base
from galfind.catalogues.Multiple_Catalogue import (
    Combined_Catalogue,
    Combined_Catalogue_Creator,
)
from galfind.utils.exceptions import (
    EmptyCatalogueError,
    GalfindTypeError,
    IncompatibleKwargsError,
    InvalidOptionError,
    InvalidUnitError,
    LengthMismatchError,
    MissingDataError,
    MissingKeyError,
    RangeError,
)


@pytest.mark.requires_data
def test_cat_from_data(cat):
    assert isinstance(cat, Catalogue)


@pytest.mark.requires_data
def test_id_cropped_cat_creator_from_data(cat_creator_id_cropped):
    assert isinstance(cat_creator_id_cropped, Catalogue_Creator)


@pytest.mark.requires_data
def test_id_cropped_cat_creator_from_data_call(cat_creator_id_cropped):
    cat = cat_creator_id_cropped()
    assert isinstance(cat, Catalogue)


# -- Catalogue_Creator constructor validation -------------------------------


def test_catalogue_creator_aper_diams_not_quantity():
    with pytest.raises(GalfindTypeError, match="aper_diams"):
        Catalogue_Creator(
            "test", "v1", "fake/path.fits", None, aper_diams=0.32
        )


def test_catalogue_creator_aper_diams_value_not_list():
    # a scalar Quantity has a float .value, not a list/np.ndarray
    with pytest.raises(GalfindTypeError, match="aper_diams.value"):
        Catalogue_Creator(
            "test", "v1", "fake/path.fits", None, aper_diams=0.32 * u.arcsec
        )


# -- module-level photometry-loading helper validation -----------------------


def test_load_galfind_phot_mismatched_labels():
    with pytest.raises(LengthMismatchError, match="phot_labels"):
        load_galfind_phot(
            None,
            phot_labels={0.32 * u.arcsec: ["FLUX_F444W"]},
            err_labels={0.16 * u.arcsec: ["FLUXERR_F444W"]},
            ZP=28.9,
        )


def test_load_galfind_phot_missing_zp():
    with pytest.raises(MissingKeyError, match="ZP"):
        load_galfind_phot(
            None,
            phot_labels={0.32 * u.arcsec: ["FLUX_F444W"]},
            err_labels={0.32 * u.arcsec: ["FLUXERR_F444W"]},
        )


def test_load_phot_mismatched_labels():
    with pytest.raises(LengthMismatchError, match="phot_labels"):
        load_phot(
            None,
            phot_labels={0.32 * u.arcsec: ["F444W"]},
            err_labels={0.16 * u.arcsec: ["F444W_err"]},
            ZP=28.9,
        )


def test_load_phot_missing_zp():
    with pytest.raises(MissingKeyError, match="ZP"):
        load_phot(
            None,
            phot_labels={0.32 * u.arcsec: ["F444W"]},
            err_labels={0.32 * u.arcsec: ["F444W_err"]},
        )


def test_galfind_phot_labels_missing_min_flux_pc_err():
    with pytest.raises(MissingKeyError, match="min_flux_pc_err"):
        galfind_phot_labels(None, None)


def test_jaguar_phot_labels_missing_min_flux_pc_err():
    with pytest.raises(MissingKeyError, match="min_flux_pc_err"):
        jaguar_phot_labels(None, None)


def test_scattered_phot_labels_missing_min_flux_pc_err():
    with pytest.raises(MissingKeyError, match="min_flux_pc_err"):
        scattered_phot_labels(None, None)


# -- Catalogue_Base validation ------------------------------------------------


def _fake_cat_creator():
    # __repr__ (invoked when building an EmptyCatalogueError message) reads
    # survey/version/filterset.instrument_name off cat_creator via
    # Catalogue_Base.__getattr__, so a bare `None` cat_creator isn't enough.
    return SimpleNamespace(
        survey="test",
        version="v1",
        filterset=SimpleNamespace(instrument_name="NIRCam"),
    )


def test_catalogue_base_getitem_empty_catalogue():
    empty_cat = Catalogue_Base([], cat_creator=_fake_cat_creator())
    with pytest.raises(EmptyCatalogueError, match="0 galaxies"):
        empty_cat[0]


def test_catalogue_base_remove_gal_no_index_or_id():
    empty_cat = Catalogue_Base([], cat_creator=None)
    with pytest.raises(IncompatibleKwargsError, match="index.*id"):
        empty_cat.remove_gal()


def test_catalogue_base_cross_match_max_sep_none():
    empty_cat = Catalogue_Base([], cat_creator=None)
    with pytest.raises(GalfindTypeError, match="max_sep"):
        empty_cat.cross_match(empty_cat, None)


# -- Combined_Catalogue.from_cats validation ---------------------------------


class _FakeCat:
    def __init__(self, aper_diams):
        self.aper_diams = aper_diams


def test_combined_catalogue_from_cats_aper_diams_mismatch():
    cat_arr = [
        _FakeCat(aper_diams=[0.32] * u.arcsec),
        _FakeCat(aper_diams=[0.16] * u.arcsec),
    ]
    with pytest.raises(LengthMismatchError, match="aper_diams"):
        Combined_Catalogue.from_cats(cat_arr)


def test_combined_catalogue_from_cats_survey_not_str():
    cat_arr = [
        _FakeCat(aper_diams=[0.32] * u.arcsec),
        _FakeCat(aper_diams=[0.32] * u.arcsec),
    ]
    with pytest.raises(GalfindTypeError, match="survey"):
        Combined_Catalogue.from_cats(cat_arr, survey=123, version="v1")


def test_combined_catalogue_from_cats_version_not_str():
    cat_arr = [
        _FakeCat(aper_diams=[0.32] * u.arcsec),
        _FakeCat(aper_diams=[0.32] * u.arcsec),
    ]
    with pytest.raises(GalfindTypeError, match="version"):
        Combined_Catalogue.from_cats(cat_arr, survey="test", version=123)


# -- Catalogue_Base dunder / core behaviour ----------------------------------


def test_catalogue_base_getattr_from_cat_creator():
    cat = Catalogue_Base([], cat_creator=_fake_cat_creator())
    assert cat.survey == "test"
    assert cat.version == "v1"


def test_catalogue_base_getattr_missing_attribute():
    cat = Catalogue_Base([], cat_creator=_fake_cat_creator())
    with pytest.raises(AttributeError):
        cat.totally_bogus_property


def test_catalogue_base_getattr_mismatched_units():
    class _FakeGal:
        def __init__(self, val):
            self.foo = val

    gals = [_FakeGal(1.0 * u.arcsec), _FakeGal(2.0 * u.deg)]
    cat = Catalogue_Base(gals, cat_creator=_fake_cat_creator())
    with pytest.raises(InvalidUnitError, match="foo"):
        cat.foo


def test_catalogue_base_getattr_collects_matching_units():
    class _FakeGal:
        def __init__(self, val):
            self.foo = val

    gals = [_FakeGal(1.0 * u.arcsec), _FakeGal(2.0 * u.arcsec)]
    cat = Catalogue_Base(gals, cat_creator=_fake_cat_creator())
    result = cat.foo
    assert list(result.value) == [1.0, 2.0]
    assert result.unit == u.arcsec


def test_catalogue_base_cat_dir_cat_name():
    cat_creator = SimpleNamespace(cat_path="/some/dir/survey_v1.fits")
    cat = Catalogue_Base([], cat_creator=cat_creator)
    assert cat.cat_dir == "/some/dir/"
    assert cat.cat_name == "survey_v1.fits"


def test_catalogue_base_len_iter():
    gals = ["g0", "g1", "g2"]
    cat = Catalogue_Base(gals, cat_creator=_fake_cat_creator())
    assert len(cat) == 3
    assert list(cat) == gals


def test_catalogue_base_iter_stopiteration():
    cat = Catalogue_Base([], cat_creator=_fake_cat_creator())
    iter(cat)
    with pytest.raises(StopIteration):
        next(cat)


def test_catalogue_base_getitem_int_and_slice():
    gals = ["g0", "g1", "g2"]
    cat = Catalogue_Base(gals, cat_creator=_fake_cat_creator())
    assert cat[1] == "g1"
    assert cat[0:2] == ["g0", "g1"]


def test_catalogue_base_getitem_single_element_list():
    gals = ["g0", "g1", "g2"]
    cat = Catalogue_Base(gals, cat_creator=_fake_cat_creator())
    assert cat[[1]] == "g1"


def test_catalogue_base_getitem_multi_element_list():
    gals = ["g0", "g1", "g2"]
    cat = Catalogue_Base(gals, cat_creator=_fake_cat_creator())
    assert cat[[0, 2]] == ["g0", "g2"]


def test_catalogue_base_setitem():
    cat = Catalogue_Base(["g0", "g1"], cat_creator=_fake_cat_creator())
    cat[0] = "g0_new"
    assert cat.gals[0] == "g0_new"


def test_catalogue_base_setattr_gal_broadcast_list():
    class _FakeGal:
        pass

    gals = [_FakeGal(), _FakeGal()]
    cat = Catalogue_Base(gals, cat_creator=_fake_cat_creator())
    cat.__setattr__("foo", [1, 2], obj="gal")
    assert gals[0].foo == 1
    assert gals[1].foo == 2


def test_catalogue_base_remove_gal_by_index():
    class _FakeGal:
        def __init__(self, ID):
            self.ID = ID

    gals = [_FakeGal(1), _FakeGal(2), _FakeGal(3)]
    cat = Catalogue_Base(gals, cat_creator=_fake_cat_creator())
    cat.remove_gal(index=1)
    assert [g.ID for g in cat.gals] == [1, 3]


def test_catalogue_base_deepcopy(synthetic_test_cat):
    cat_copy = deepcopy(synthetic_test_cat)
    assert cat_copy is not synthetic_test_cat
    assert cat_copy.gals is not synthetic_test_cat.gals
    assert len(cat_copy) == len(synthetic_test_cat)


def test_catalogue_base_ra_dec_range(synthetic_test_cat):
    ra_range = synthetic_test_cat.ra_range
    dec_range = synthetic_test_cat.dec_range
    assert ra_range[0].value == pytest.approx(150.0)
    assert ra_range[1].value == pytest.approx(150.0)
    assert dec_range[0].value == pytest.approx(2.0)
    assert dec_range[1].value == pytest.approx(2.0)


def test_catalogue_base_cross_match_self(synthetic_test_cat):
    # both synthetic galaxies sit at the same sky position, so every
    # galaxy should cross-match every galaxy (itself included)
    matches = synthetic_test_cat.cross_match(
        synthetic_test_cat, 1.0 * u.arcsec
    )
    assert len(matches) == 2
    for match_list in matches.values():
        assert len(match_list) == 2
        for sep, _matched_gal in match_list:
            assert sep <= 1.0 * u.arcsec


# -- Catalogue_Base._calc_Vmax validation (fixture-free) ---------------------


def test_catalogue_base_calc_vmax_zbin_length_mismatch():
    empty_cat = Catalogue_Base([], cat_creator=_fake_cat_creator())
    with pytest.raises(LengthMismatchError, match="z_bin"):
        empty_cat._calc_Vmax(
            None, z_bin=[1.0], aper_diam=None, SED_fit_code=None
        )


def test_catalogue_base_calc_vmax_zbin_not_increasing():
    empty_cat = Catalogue_Base([], cat_creator=_fake_cat_creator())
    with pytest.raises(RangeError, match="z_bin"):
        empty_cat._calc_Vmax(
            None, z_bin=[5.0, 3.0], aper_diam=None, SED_fit_code=None
        )


def test_catalogue_base_calc_vmax_bad_sed_fit_code_type():
    empty_cat = Catalogue_Base([], cat_creator=_fake_cat_creator())
    with pytest.raises(GalfindTypeError, match="SED_fit_code"):
        empty_cat._calc_Vmax(
            None, z_bin=[3.0, 5.0], aper_diam=None, SED_fit_code="not_a_code"
        )


def test_catalogue_base_calc_vmax_bad_n_jobs_type(eazy_fsps_larson_sed_fitter):
    empty_cat = Catalogue_Base([], cat_creator=_fake_cat_creator())
    with pytest.raises(GalfindTypeError, match="n_jobs"):
        empty_cat._calc_Vmax(
            None,
            z_bin=[3.0, 5.0],
            aper_diam=None,
            SED_fit_code=eazy_fsps_larson_sed_fitter,
            n_jobs="bad",
        )


def test_catalogue_base_calc_vmax_n_jobs_below_one(
    eazy_fsps_larson_sed_fitter,
):
    empty_cat = Catalogue_Base([], cat_creator=_fake_cat_creator())
    with pytest.raises(RangeError, match="n_jobs"):
        empty_cat._calc_Vmax(
            None,
            z_bin=[3.0, 5.0],
            aper_diam=None,
            SED_fit_code=eazy_fsps_larson_sed_fitter,
            n_jobs=0,
        )


# -- Catalogue.py module-level pure functions --------------------------------


def test_phot_property_from_galfind_tab_empty_cat_returns_empty_dict():
    result = phot_property_from_galfind_tab(Table(), {0.32 * u.arcsec: ["a"]})
    assert result == {}


def test_phot_property_from_galfind_tab_cat_aper_diams_not_quantity():
    tab = Table({"a": [1]})
    with pytest.raises(GalfindTypeError, match="cat_aper_diams"):
        phot_property_from_galfind_tab(
            tab, {0.32 * u.arcsec: ["a"]}, cat_aper_diams=5.0
        )


def test_phot_property_from_galfind_tab_cat_aper_diams_value_not_list():
    tab = Table({"a": [1]})
    with pytest.raises(GalfindTypeError, match="cat_aper_diams"):
        phot_property_from_galfind_tab(
            tab,
            {0.32 * u.arcsec: ["a"]},
            cat_aper_diams=0.32 * u.arcsec,
        )


def test_phot_property_from_galfind_tab_no_matching_aper_diam():
    tab = Table({"a": [1]})
    with pytest.raises(MissingDataError, match="aper_diam"):
        phot_property_from_galfind_tab(
            tab,
            {0.32 * u.arcsec: ["a"]},
            cat_aper_diams=[1.0] * u.arcsec,
        )


def test_phot_property_from_galfind_tab_mismatched_labels():
    tab = Table({"a": [1], "b": [2]})
    labels = {0.32 * u.arcsec: ["a"], 0.16 * u.arcsec: ["b"]}
    with pytest.raises(LengthMismatchError, match="labels"):
        phot_property_from_galfind_tab(
            tab, labels, cat_aper_diams=[0.32, 0.16] * u.arcsec
        )


def test_phot_property_from_fits_cat_aper_diams_not_quantity():
    tab = Table({"a": [1]})
    with pytest.raises(GalfindTypeError, match="cat_aper_diams"):
        phot_property_from_fits(
            tab, {0.32 * u.arcsec: ["a"]}, cat_aper_diams=5.0
        )


def test_phot_property_from_fits_cat_aper_diams_value_not_list():
    tab = Table({"a": [1]})
    with pytest.raises(GalfindTypeError, match="cat_aper_diams"):
        phot_property_from_fits(
            tab,
            {0.32 * u.arcsec: ["a"]},
            cat_aper_diams=0.32 * u.arcsec,
        )


def test_phot_property_from_fits_mismatched_labels():
    tab = Table({"a": [1], "b": [2]})
    labels = {0.32 * u.arcsec: ["a"], 0.16 * u.arcsec: ["b"]}
    with pytest.raises(LengthMismatchError, match="labels"):
        phot_property_from_fits(
            tab, labels, cat_aper_diams=[0.32, 0.16] * u.arcsec
        )


def test_galfind_phot_labels_happy_path(nircam_multi_filter, aper_diams):
    phot_labels, err_labels = galfind_phot_labels(
        nircam_multi_filter, aper_diams, min_flux_pc_err=10.0
    )
    assert set(phot_labels.keys()) == set(aper_diams)
    for cols in phot_labels.values():
        assert len(cols) == len(nircam_multi_filter)
        assert all(col.startswith("FLUX_APER_") for col in cols)
    for cols in err_labels.values():
        assert all("10pc" in col for col in cols)


def test_jaguar_phot_labels_happy_path(nircam_multi_filter, aper_diams):
    phot_labels, err_labels = jaguar_phot_labels(
        nircam_multi_filter, aper_diams, min_flux_pc_err=10.0
    )
    for cols in phot_labels.values():
        assert all(col.startswith("NRC_") for col in cols)
    # JAGUAR catalogues have no per-band error columns
    for cols in err_labels.values():
        assert cols == []


def test_scattered_phot_labels_happy_path(nircam_multi_filter, aper_diams):
    phot_labels, err_labels = scattered_phot_labels(
        nircam_multi_filter, aper_diams, min_flux_pc_err=10.0
    )
    for cols in phot_labels.values():
        assert all(col.endswith("_scattered") for col in cols)
    for cols in err_labels.values():
        assert all(col.endswith("_err") for col in cols)


def test_galfind_mask_labels(nircam_multi_filter):
    labels = galfind_mask_labels(nircam_multi_filter)
    assert labels == [
        f"unmasked_{name}" for name in nircam_multi_filter.filt_names
    ]


def test_galfind_depth_labels(nircam_multi_filter, aper_diams):
    labels = galfind_depth_labels(nircam_multi_filter, aper_diams)
    for cols in labels.values():
        assert cols == [
            f"loc_depth_{name}" for name in nircam_multi_filter.filt_names
        ]


def test_scattered_depth_labels(nircam_multi_filter, aper_diams):
    labels = scattered_depth_labels(nircam_multi_filter, aper_diams)
    for aper_diam, cols in labels.items():
        assert all(
            col.startswith("loc_depth_") and name in col
            for col, name in zip(cols, nircam_multi_filter.filt_names)
        )


def test_galfind_snr_labels(nircam_multi_filter, aper_diams):
    labels = galfind_snr_labels(nircam_multi_filter, aper_diams)
    assert set(labels.keys()) == set(aper_diams)
    for cols in labels.values():
        assert len(cols) == len(nircam_multi_filter)


def test_load_bool_Table():
    tab = Table({"SEL_A": [True, False], "SEL_B": [False, False]})
    result = load_bool_Table(tab, ["SEL_A", "SEL_B"])
    assert result == {"SEL_A": [True, False], "SEL_B": [False, False]}


def test_galfind_selection_labels():
    tab = Table(
        {
            "SEL_A": np.array([True, False]),
            "flux": np.array([1.0, 2.0]),
        }
    )
    assert galfind_selection_labels(tab) == ["SEL_A"]


def test_check_hdu_exists_open_galfind_cat_hdr(synthetic_eazy_cat_path):
    # "ID" is handled as a first_ext_keys convention by open_galfind_cat
    # rather than a literal HDU name, so no HDU is actually named "ID"
    assert check_hdu_exists(synthetic_eazy_cat_path, "not_a_real_hdu") is False
    assert check_hdu_exists(synthetic_eazy_cat_path, "SELECTION") is False

    tab = open_galfind_cat(synthetic_eazy_cat_path, "ID")
    assert len(tab) == 2

    missing = open_galfind_cat(synthetic_eazy_cat_path, "not_a_real_hdu")
    assert missing is None

    hdr = open_galfind_hdr(synthetic_eazy_cat_path, "ID")
    assert hasattr(hdr, "keys")


# -- Catalogue_Creator additional validation ---------------------------------


def test_catalogue_creator_survey_mismatch(
    tmp_path, nircam_multi_filter, aper_diams
):
    tab = Table({"NUMBER": [1]})
    tab.meta["SURVEY"] = "other_survey"
    cat_path = str(tmp_path / "survey_mismatch.fits")
    tab.write(cat_path)
    with pytest.raises(InvalidOptionError, match="SURVEY"):
        Catalogue_Creator(
            "test",
            "v1",
            cat_path,
            nircam_multi_filter,
            aper_diams,
            apply_gal_instr_mask=False,
        )


def test_catalogue_creator_version_mismatch(
    tmp_path, nircam_multi_filter, aper_diams
):
    tab = Table({"NUMBER": [1]})
    tab.meta["VERSION"] = "v_other"
    cat_path = str(tmp_path / "version_mismatch.fits")
    tab.write(cat_path)
    with pytest.raises(InvalidOptionError, match="VERSION"):
        Catalogue_Creator(
            "test",
            "v1",
            cat_path,
            nircam_multi_filter,
            aper_diams,
            apply_gal_instr_mask=False,
        )


def test_apply_gal_instr_mask_length_mismatch():
    with pytest.raises(LengthMismatchError, match="gal_instr_mask"):
        Catalogue_Creator._apply_gal_instr_mask([1, 2, 3], [True, False])


def test_apply_gal_instr_mask_filters_bands():
    arr = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    mask = [
        np.array([True, False, True]),
        np.array([False, True, True]),
    ]
    result = Catalogue_Creator._apply_gal_instr_mask(arr, mask)
    assert list(result[0]) == [1, 3]
    assert list(result[1]) == [5, 6]


# -- Catalogue class validation reachable without a real SED fit ------------


def test_catalogue_repr_and_str(synthetic_eazy_cat):
    r = repr(synthetic_eazy_cat)
    assert r.startswith("CATALOGUE(")
    s = str(synthetic_eazy_cat)
    assert "TOTAL GALAXIES" in s
    assert "RA RANGE" in s


def test_catalogue_update_sed_results_length_mismatch(synthetic_test_cat):
    with pytest.raises(LengthMismatchError, match="cat_SED_results"):
        synthetic_test_cat.update_SED_results([1])


def test_catalogue_update_sed_result_lowz_zmax_info_length_mismatch(
    synthetic_test_cat,
):
    with pytest.raises(LengthMismatchError, match="zmax_info_arr"):
        synthetic_test_cat.update_SED_result_lowz_zmax_info(
            0.32 * u.arcsec, "key", [1]
        )


def test_catalogue_load_sextractor_re_missing_data(synthetic_test_cat):
    with pytest.raises(MissingDataError, match="data"):
        synthetic_test_cat.load_sextractor_Re()


def test_catalogue_load_sextractor_auto_mags_missing_data(synthetic_test_cat):
    with pytest.raises(MissingDataError, match="data"):
        synthetic_test_cat.load_sextractor_auto_mags()


def test_catalogue_load_sextractor_auto_fluxes_missing_data(
    synthetic_test_cat,
):
    with pytest.raises(MissingDataError, match="data"):
        synthetic_test_cat.load_sextractor_auto_fluxes()


def test_catalogue_load_sextractor_auto_fluxes_bad_multiply_factor_type(
    synthetic_test_cat,
):
    with pytest.raises(GalfindTypeError, match="multiply_factor"):
        synthetic_test_cat.load_sextractor_auto_fluxes(multiply_factor=5.0)


def test_catalogue_load_sextractor_auto_fluxes_bad_multiply_factor_keys(
    synthetic_test_cat,
):
    with pytest.raises(InvalidOptionError, match="multiply_factor"):
        synthetic_test_cat.load_sextractor_auto_fluxes(
            multiply_factor={"NOT_A_REAL_BAND": 1.0}
        )


def test_catalogue_load_band_properties_from_cat_bad_dest(synthetic_test_cat):
    with pytest.raises(InvalidOptionError, match="dest"):
        synthetic_test_cat.load_band_properties_from_cat(
            "FLUX_RADIUS", "sex_Re", dest="not_gal_or_phot_obs"
        )


def test_catalogue_load_fixz_sed_results_length_mismatch(synthetic_test_cat):
    with pytest.raises(LengthMismatchError, match="z_arr"):
        synthetic_test_cat.load_fixz_SED_results(0.32 * u.arcsec, z_arr=[1.0])


def test_catalogue_calc_vmax_missing_data(synthetic_test_cat):
    with pytest.raises(MissingDataError, match="data"):
        synthetic_test_cat.calc_Vmax([3.0, 5.0], 0.32 * u.arcsec, None)


def test_catalogue_scatter_missing_aper_diam(synthetic_test_cat):
    with pytest.raises(MissingDataError, match="aper_diam"):
        synthetic_test_cat.scatter(5.0 * u.arcsec)


def test_catalogue_update_depths_from_data_missing_data(synthetic_eazy_cat):
    # unlike synthetic_test_cat, this catalogue has a real cat_path (so
    # the MissingDataError message's self.cat_name access succeeds) but
    # still has no 'data' attribute, since it wasn't built via from_data
    with pytest.raises(MissingDataError, match="data"):
        synthetic_eazy_cat._update_depths_from_data(0.32 * u.arcsec)


def test_readme_info_from_fits_tab_returns_dict(synthetic_eazy_cat_path):
    result = Catalogue._readme_info_from_fits_tab(synthetic_eazy_cat_path)
    assert result == {}


# -- Multiple_Catalogue: Combined_Catalogue_Creator --------------------------


def test_combined_catalogue_creator_repr_no_crops(
    nircam_multi_filter, aper_diams
):
    creator = Combined_Catalogue_Creator(
        "test", "v1", nircam_multi_filter, aper_diams
    )
    assert repr(creator) == "Combined_Catalogue_Creator(test, v1)"
    assert creator.crops == []
    assert creator.crop_name == ""


def test_combined_catalogue_creator_repr_with_crops(
    nircam_multi_filter, aper_diams
):
    creator = Combined_Catalogue_Creator(
        "test", "v1", nircam_multi_filter, aper_diams, crops=["EPOCHS"]
    )
    assert repr(creator) == "Combined_Catalogue_Creator(test, v1, ['EPOCHS'])"


def test_combined_catalogue_creator_str(nircam_multi_filter, aper_diams):
    creator = Combined_Catalogue_Creator(
        "test",
        "v1",
        nircam_multi_filter,
        aper_diams,
        cat_path="/tmp/fake_combined.fits",
        crops=["EPOCHS"],
    )
    s = str(creator)
    assert "Survey: test" in s
    assert "Version: v1" in s
    assert "Catalogue path: /tmp/fake_combined.fits" in s
    assert "Crops applied: EPOCHS" in s


# -- Multiple_Catalogue: Combined_Catalogue -----------------------------------


def test_combined_catalogue_load_sextractor_ext_src_corrs_missing_data():
    class _FakeSubCat:
        def load_sextractor_ext_src_corrs(self):
            # constituent catalogues loaded fine, but the combined
            # catalogue's own galaxies were never updated
            pass

    aper_diam = 0.32 * u.arcsec
    combined = Combined_Catalogue.__new__(Combined_Catalogue)
    combined.cat_arr = [_FakeSubCat()]
    combined.gals = [SimpleNamespace(aper_phot={aper_diam: SimpleNamespace()})]
    combined.cat_creator = SimpleNamespace(aper_diams=[aper_diam])
    with pytest.raises(MissingDataError, match="ext_src_corrs"):
        combined.load_sextractor_ext_src_corrs()


def test_combined_catalogue_plot_phot_diagnostics_delegates():
    calls = []

    class _FakeSubCat:
        def plot_phot_diagnostics(self, *args, **kwargs):
            calls.append((args, kwargs))

    combined = Combined_Catalogue.__new__(Combined_Catalogue)
    combined.cat_arr = [_FakeSubCat(), _FakeSubCat()]
    combined.plot_phot_diagnostics(1, key="value")
    assert calls == [((1,), {"key": "value"}), ((1,), {"key": "value"})]

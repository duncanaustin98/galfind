from types import SimpleNamespace

import astropy.units as u
import pytest

from galfind.catalogues import Catalogue, Catalogue_Creator
from galfind.catalogues.Catalogue import (
    galfind_phot_labels,
    jaguar_phot_labels,
    load_galfind_phot,
    load_phot,
    scattered_phot_labels,
)
from galfind.catalogues.Catalogue_Base import Catalogue_Base
from galfind.catalogues.Multiple_Catalogue import Combined_Catalogue
from galfind.utils.exceptions import (
    EmptyCatalogueError,
    GalfindTypeError,
    IncompatibleKwargsError,
    LengthMismatchError,
    MissingKeyError,
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

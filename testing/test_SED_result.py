import pytest

from galfind.sed_fitting.SED_result import Catalogue_SED_results
from galfind.utils.exceptions import (
    IncompatibleKwargsError,
    LengthMismatchError,
    MissingKeyError,
)


def test_from_fits_cat_missing_code_key_raises_missing_key_error():
    with pytest.raises(MissingKeyError, match="'code'"):
        Catalogue_SED_results.from_fits_cat([None], [{}])


def test_from_fits_cat_mismatched_lengths_raises_length_mismatch():
    with pytest.raises(LengthMismatchError, match="SED_fit_cats"):
        Catalogue_SED_results.from_fits_cat([None, None], [{"code": None}])


def test_from_fits_cat_no_phot_source_raises_incompatible_kwargs():
    with pytest.raises(IncompatibleKwargsError, match="phot_arr"):
        Catalogue_SED_results.from_fits_cat([], [])

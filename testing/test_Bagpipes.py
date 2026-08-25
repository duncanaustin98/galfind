import astropy.units as u
import pytest

from galfind.sed_fitting.Bagpipes import Bagpipes, calculate_bins
from galfind.utils.exceptions import (
    GalfindTypeError,
    IncompatibleKwargsError,
    InvalidOptionError,
    InvalidUnitError,
    LengthMismatchError,
    MissingKeyError,
    RangeError,
)


def test_bagpipes_fit_instructions_not_dict_raises_type_error():
    with pytest.raises(GalfindTypeError, match="fit_instructions"):
        Bagpipes({"fit_instructions": "not_a_dict"})


def test_bagpipes_fit_instructions_without_custom_label_raises():
    with pytest.raises(IncompatibleKwargsError, match="custom_label"):
        Bagpipes({"fit_instructions": {}})


def test_bagpipes_fit_instructions_with_custom_label_constructs():
    bagpipes = Bagpipes({"fit_instructions": {}}, custom_label="test_label")
    assert bagpipes.label == "test_label"


def test_extract_seds_mismatched_lengths_raises_length_mismatch():
    bagpipes = Bagpipes({"fit_instructions": {}}, custom_label="test_label")
    with pytest.raises(LengthMismatchError, match="len\\(IDs\\)"):
        bagpipes.extract_SEDs([1, 2], ["a"], cat=None, aper_diam=None)


def test_extract_seds_missing_required_kwargs_raises_missing_key():
    bagpipes = Bagpipes({"fit_instructions": {}}, custom_label="test_label")
    with pytest.raises(MissingKeyError, match="'cat' and 'aper_diam'"):
        bagpipes.extract_SEDs([1], ["a"])


def test_move_files_invalid_direction_raises_invalid_option():
    bagpipes = Bagpipes({"fit_instructions": {}}, custom_label="test_label")
    with pytest.raises(InvalidOptionError, match="from_temp.*to_temp"):
        bagpipes._move_files(None, None, direction="bogus")


def test_load_pipes_spec_abmag_raises_invalid_unit():
    with pytest.raises(InvalidUnitError, match="ABmag"):
        Bagpipes._load_pipes_spec(1, None, spec_units=u.ABmag)


def test_calculate_bins_non_ascending_ages_raises_range_error():
    with pytest.raises(RangeError, match="ascending order"):
        calculate_bins(
            redshift=5.0,
            fixed_bin_ages=[100.0, 10.0] * u.Myr,
            return_flat=True,
        )


def test_calculate_bins_log_time_raises_invalid_option():
    with pytest.raises(InvalidOptionError, match="log_time"):
        calculate_bins(redshift=5.0, log_time=True, return_flat=True)


def test_calculate_bins_non_flat_raises_invalid_option():
    with pytest.raises(InvalidOptionError, match="return_flat"):
        calculate_bins(redshift=5.0, log_time=False, return_flat=False)

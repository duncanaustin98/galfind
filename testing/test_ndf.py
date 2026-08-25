import astropy.units as u
import numpy as np
import pytest

from galfind.number_density_functions.ndf import (
    Base_Number_Density_Function,
    Number_Density_Function,
)
from galfind.utils.exceptions import (
    GalfindTypeError,
    InvalidOptionError,
    LengthMismatchError,
    RangeError,
)


def _make_base_ndf():
    return Base_Number_Density_Function(
        "M_UV",
        np.array([1.0, 2.0]),
        9.0,
        np.array([1e-4, 1e-5]),
        np.array([[1e-5, 1e-5], [1e-5, 1e-5]]),
        "TestAuthor2024",
    )


def test_from_flags_repo_bad_obs_or_models_raises_invalid_option():
    with pytest.raises(InvalidOptionError, match="obs_or_models"):
        Base_Number_Density_Function.from_flags_repo(
            "M_UV",
            [8.0, 9.0],
            "Finkelstein2016",
            obs_or_models="not_a_valid_option",
        )


def test_base_ndf_add_bad_type_raises_type_error():
    ndf = _make_base_ndf()
    with pytest.raises(GalfindTypeError, match="other"):
        ndf + 5


def test_base_ndf_plot_bad_x_lims_length_raises_length_mismatch():
    ndf = _make_base_ndf()
    with pytest.raises(LengthMismatchError, match="x_lims"):
        ndf.plot(x_lims=[1.0, 2.0, 3.0])


def test_crop_to_xbin_bad_ordering_raises_range_error():
    ndf = Number_Density_Function.__new__(Number_Density_Function)
    ndf.x_bins = np.array([[1.0, 2.0], [2.0, 3.0]]) * u.dimensionless_unscaled
    ndf.x_mid_bins = np.array([1.5, 2.5]) * u.dimensionless_unscaled
    ndf.crop_name = "test_crop"
    with pytest.raises(RangeError, match="x_bin"):
        ndf.crop_to_xbin(
            [3.0 * u.dimensionless_unscaled, 1.0 * u.dimensionless_unscaled]
        )

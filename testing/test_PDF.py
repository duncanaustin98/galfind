"""Lightweight validation-failure tests for `galfind.visualization.PDF`.

Deliberately avoid needing real SED-fitting output: these exercise `PDF`,
`SED_fit_PDF`, and `PDF_nD` construction/validation directly with small
synthetic astropy Quantity arrays.
"""

import astropy.units as u
import numpy as np
import pytest

from galfind.utils.exceptions import (
    GalfindTypeError,
    InvalidOptionError,
    LengthMismatchError,
    MissingDataError,
    RangeError,
)
from galfind.visualization.PDF import PDF, PDF_nD, SED_fit_PDF


def _simple_pdf(n=5):
    arr = np.linspace(1.0, 5.0, n) * u.dimensionless_unscaled
    return PDF.from_1D_arr("z", arr, Nbins=3)


def test_pdf_init_invalid_x_type_raises():
    with pytest.raises(GalfindTypeError):
        PDF("z", [1, 2, 3], np.array([0.2, 0.3, 0.5]))


def test_from_1D_arr_non_quantity_arr_raises():
    with pytest.raises(GalfindTypeError):
        PDF.from_1D_arr("z", np.array([1.0, 2.0, 3.0]))


def test_from_1D_arr_all_nan_raises_range_error():
    arr = np.array([np.nan, np.nan, np.nan]) * u.dimensionless_unscaled
    with pytest.raises(RangeError):
        PDF.from_1D_arr("z", arr)


def test_get_percentile_non_float_raises():
    pdf_obj = _simple_pdf()
    with pytest.raises(GalfindTypeError):
        pdf_obj.get_percentile(50)  # int, not float


def test_add_mismatched_property_name_raises():
    pdf_a = _simple_pdf()
    pdf_b = PDF.from_1D_arr(
        "mass", np.linspace(1.0, 5.0, 5) * u.dimensionless_unscaled, Nbins=3
    )
    with pytest.raises(LengthMismatchError):
        pdf_a.__add__(pdf_b)


def test_load_peaks_from_sed_result_nth_peak_type_raises():
    sed_fit_pdf = SED_fit_PDF.from_1D_arr(
        "z",
        np.linspace(1.0, 5.0, 5) * u.dimensionless_unscaled,
        {"code": "test"},
        Nbins=3,
    )
    with pytest.raises(GalfindTypeError):
        sed_fit_pdf.load_peaks_from_SED_result(SED_result=None, nth_peak="0")


def test_load_peaks_from_sed_result_nth_peak_not_zero_raises():
    sed_fit_pdf = SED_fit_PDF.from_1D_arr(
        "z",
        np.linspace(1.0, 5.0, 5) * u.dimensionless_unscaled,
        {"code": "test"},
        Nbins=3,
    )
    with pytest.raises(InvalidOptionError):
        sed_fit_pdf.load_peaks_from_SED_result(SED_result=None, nth_peak=1)


def test_pdf_nd_missing_input_arr_raises():
    x = np.array([1.0, 2.0, 3.0]) * u.dimensionless_unscaled
    p_x = np.array([0.2, 0.3, 0.5])
    pdf_no_input_arr = PDF("z", x, p_x)
    with pytest.raises(MissingDataError):
        PDF_nD([pdf_no_input_arr])


def test_pdf_nd_from_matrix_length_mismatch_raises():
    matrix = np.zeros((2, 5))
    with pytest.raises(LengthMismatchError):
        PDF_nD.from_matrix(["only_one_name"], matrix)

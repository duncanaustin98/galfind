
import pytest
import numpy as np
import astropy.units as u

from galfind import Multiple_Filter
from galfind import useful_funcs_austind as funcs


@pytest.fixture(
    scope = "module",
    params = [
        {}, # default
    ]
)
def ext_src_corr_inputs(request):
    return request.param

@pytest.mark.requires_data
def test_get_ext_src_corr_pass(phot_rest_sex_params_loaded, ext_src_corr_inputs):
    ext_src_corr = funcs.get_ext_src_corr(phot_rest_sex_params_loaded, **ext_src_corr_inputs)
    assert isinstance(ext_src_corr, float)

@pytest.mark.requires_data
def test_get_ext_src_corr_no_sex_params_fail(phot_rest, ext_src_corr_inputs):
    # expect this to fail due to not having extended source corrections pre-loaded
    with pytest.raises(AttributeError):
        funcs.get_ext_src_corr(phot_rest, **ext_src_corr_inputs)

def test_blank_phot_rest_ext_src_corr_nan(blank_phot_rest, ext_src_corr_inputs):
    ext_src_corr = funcs.get_ext_src_corr(blank_phot_rest, **ext_src_corr_inputs)
    assert np.isnan(ext_src_corr)


@pytest.fixture(
    scope = "module",
    params = [
        (
            {
                "depth": 28.0,
                "zero_point": 8.9,
            }, True
        ),
        (
            {
                "depth": 28.0 * u.ABmag,
                "zero_point": 8.9,
            }, True
        )
    ]
)
def calc_1sigma_flux_inputs(request):
    return request.param

def test_calc_1sigma_flux(calc_1sigma_flux_inputs):
    inputs, outcome = calc_1sigma_flux_inputs
    if outcome != True:
        with pytest.raises(outcome):
            funcs.calc_1sigma_flux(**inputs)
    else:
        result = funcs.calc_1sigma_flux(**inputs)


@pytest.fixture(
    scope = "module",
    params = [
        (
            {
                "five_sigma_depth": 28.0,
                "n": 2,
            }, True
        ),
        (
            {
                "five_sigma_depth": 28.0 * u.ABmag,
                "n": 2,
            }, True
        ),
        (
            {
                "five_sigma_depth": 28.0,
                "n": 0.4,
            }, True
        ),
        (
            {
                "five_sigma_depth": 28.0,
                "n": -1,
            }, AssertionError
        )
    ]
)
def five_to_n_sigma_mag_inputs(request):
    return request.param

def test_five_to_n_sigma_mag(five_to_n_sigma_mag_inputs):
    inputs, outcome = five_to_n_sigma_mag_inputs
    if outcome != True:
        with pytest.raises(outcome):
            funcs.five_to_n_sigma_mag(**inputs)
    else:
        result = funcs.five_to_n_sigma_mag(**inputs)


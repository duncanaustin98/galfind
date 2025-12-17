
import pytest
import numpy as np

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
    # make fake ext_src_corrs
    
    ext_src_corr = funcs.get_ext_src_corr(blank_phot_rest, **ext_src_corr_inputs)
    assert np.isnan(ext_src_corr)




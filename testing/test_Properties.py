
import pytest
from pytest_lazy_fixtures import lf
import os
import sys
import inspect
import numpy as np
from copy import copy, deepcopy
import astropy.units as u

os.environ["GALFIND_CONFIG_DIR"] = f"{os.getcwd()}/testing"
os.environ["GALFIND_CONFIG_NAME"] = "test_galfind_config.ini"

import galfind
from galfind import MUV_Calculator, Catalogue, Galaxy
from galfind.Property_calculator import MUV_SED_Property_Calculator

# @pytest.fixture(
#     scope = "module",
#     params = [
#         ({"IDs": 1}, True),
#     ]
# )
# def call_MUV_phot_calculator(request):
#     return MUV_Calculator, *request.param

# @pytest.fixture(
#     scope = "module",
#     params = [
#         ({"IDs": "invalid"}, Exception),
#     ]
# )
# def fail_MUV_phot_calculator(request):
#     return MUV_Calculator, *request.param

@pytest.fixture(
    scope = "module",
    params = [
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
            }, True
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "ext_src_corrs": None,
            }, True
        ),
    ]
)
def call_MUV_SED_property_calculator(request):
    inputs, outcome = request.param
    inputs["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return MUV_SED_Property_Calculator, inputs, outcome


@pytest.fixture(
    scope = "module",
    params = [
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fit_label": "eazy_fsps_larson_sed_fitter",
            }, True
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fit_label": "eazy_fsps_larson_sed_fitter",
                "ext_src_corrs": None,
            }, True
        ),
    ]
)
def call_MUV_phot_property_calculator(request):
    inputs, outcome = request.param
    inputs["SED_fit_label"] = request.getfixturevalue(inputs["SED_fit_label"]).label
    return MUV_Calculator, inputs, outcome


##################################################

def get_call_property_calculator_fixtures():
    module = sys.modules[__name__]
    return [
        lf(name) for name, obj in inspect.getmembers(module)
        if name.endswith("_property_calculator") and name.startswith("call_")
    ]

@pytest.fixture(
    scope = "module",
    params = get_call_property_calculator_fixtures()
)
def call_property_calculator(request):
    return request.param

#################################################

def test_pass_property_calculator_init(call_property_calculator):
    property_calculator_cls, inputs, _ = call_property_calculator
    # instantiate selector_cls with inputs
    property_calculator_inst = property_calculator_cls(**inputs)
    assert isinstance(property_calculator_inst, property_calculator_cls)

# def test_fail_selector_init(fail_selector):
#     selector_cls, inputs, outcome = fail_selector
#     # instantiate selector_cls with inputs
#     with pytest.raises(outcome):
#         selector_cls(**inputs)

# TODO: Determine expected __call__ failures due to 
# objects not containing required information

# @pytest.mark.requires_data
# def test_selector_call_gal(call_selector, gal):
#     selector_cls, inputs, outcome = call_selector
#     outcome_ = copy(outcome)
#     if isinstance(outcome_, dict):
#         outcome_ = outcome_["gal"]
#     if outcome_ != True:
#         with pytest.raises(outcome_):
#             selector_inst = selector_cls(**inputs)
#             selector_inst(gal)
#     else:
#         selector_inst = selector_cls(**inputs)
#         out_gal = selector_inst(gal)
#         assert isinstance(out_gal, Galaxy)


@pytest.mark.requires_data
def test_property_calculator_call_cat(call_property_calculator, cat_eazy_loaded):
    property_calculator_cls, inputs, outcome = call_property_calculator
    outcome_ = copy(outcome)
    if isinstance(outcome_, dict):
        outcome_ = outcome_["cat"]
    if outcome_ != True:
        with pytest.raises(outcome_):
            property_calculator_inst = property_calculator_cls(**inputs)
            property_calculator_inst(cat_eazy_loaded)
    else:
        property_calculator_inst = property_calculator_cls(**inputs)
        out_cat = property_calculator_inst(cat_eazy_loaded)
        assert isinstance(out_cat, Catalogue)

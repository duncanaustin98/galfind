import inspect
import os
import sys
from copy import copy

import astropy.units as u
import numpy as np
import pytest
from pytest_lazy_fixtures import lf

# anchor to this file's own location, not the process cwd, so
# GALFIND_WORK/GALFIND_DATA always land under <repo_root>/testing/
# test_work regardless of the directory tests are invoked from
os.environ["GALFIND_CONFIG_DIR"] = os.path.dirname(os.path.abspath(__file__))
os.environ["GALFIND_CONFIG_NAME"] = "test_galfind_config.ini"

from galfind.catalogues import Catalogue
from galfind.properties import (
    Fesc_From_Beta_Calculator,
    LUV_Calculator,
    Lya_Break_Strength_Calculator,
    MUV_Calculator,
    SFR_UV_Calculator,
    UV_Beta_Calculator,
    UV_Dust_Attenuation_Calculator,
    mUV_Calculator,
)
from galfind.properties.Property_calculator import MUV_SED_Property_Calculator

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
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
            },
            True,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "ext_src_corrs": None,
            },
            True,
        ),
    ],
)
def call_MUV_SED_property_calculator(request):
    inputs, outcome = request.param
    inputs["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return MUV_SED_Property_Calculator, inputs, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fit_label": "eazy_fsps_larson_sed_fitter",
            },
            True,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fit_label": "eazy_fsps_larson_sed_fitter",
                "ext_src_corrs": None,
            },
            True,
        ),
    ],
)
def call_MUV_phot_property_calculator(request):
    inputs, outcome = request.param
    inputs["SED_fit_label"] = request.getfixturevalue(
        inputs["SED_fit_label"]
    ).label
    return MUV_Calculator, inputs, outcome


##################################################


def get_call_property_calculator_fixtures():
    module = sys.modules[__name__]
    return [
        lf(name)
        for name, obj in inspect.getmembers(module)
        if name.endswith("_property_calculator") and name.startswith("call_")
    ]


@pytest.fixture(scope="module", params=get_call_property_calculator_fixtures())
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

# TODO: Determine expected __call__ failures due to
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
def test_property_calculator_call_cat(
    call_property_calculator, cat_eazy_loaded
):
    property_calculator_cls, inputs, outcome = call_property_calculator
    outcome_ = copy(outcome)
    if isinstance(outcome_, dict):
        outcome_ = outcome_["cat"]
    if outcome_ is not True:
        with pytest.raises(outcome_):
            property_calculator_inst = property_calculator_cls(**inputs)
            property_calculator_inst(cat_eazy_loaded)
    else:
        property_calculator_inst = property_calculator_cls(**inputs)
        out_cat = property_calculator_inst(cat_eazy_loaded)
        assert isinstance(out_cat, Catalogue)


#################################################
# Regression tests: UV_Dust_Attenuation_Calculator, SFR_UV_Calculator,
# and Dust_Attenuation_From_UV_Calculator (and its
# Line_Dust_Attenuation_From_UV_Calculator subclass) were all missing
# their required `_kwarg_assertions` implementation, making them
# uninstantiable (`TypeError: Can't instantiate abstract class ...`).
# UV_Dust_Attenuation_Calculator's validation had been misplaced
# inside `plot_name` instead; SFR_UV_Calculator's was simply absent.
#################################################


class TestKwargAssertionsRegression:
    def test_uv_dust_attenuation_calculator_constructs(
        self, eazy_fsps_larson_sed_fitter
    ):
        calc = UV_Dust_Attenuation_Calculator(
            0.32 * u.arcsec, eazy_fsps_larson_sed_fitter
        )
        assert isinstance(calc, UV_Dust_Attenuation_Calculator)
        assert isinstance(calc.plot_name, str)

    def test_sfr_uv_calculator_constructs(self, eazy_fsps_larson_sed_fitter):
        calc = SFR_UV_Calculator(0.32 * u.arcsec, eazy_fsps_larson_sed_fitter)
        assert isinstance(calc, SFR_UV_Calculator)

    def test_sfr_uv_calculator_invalid_conv_raises(
        self, eazy_fsps_larson_sed_fitter
    ):
        with pytest.raises(AssertionError):
            SFR_UV_Calculator(
                0.32 * u.arcsec,
                eazy_fsps_larson_sed_fitter,
                SFR_conv="not_a_real_conversion",
            )


#################################################
# Hand-built z~9.5 Lyman-break "galaxy" (`high_z_gal`) and undetected
# contaminant (`garbage_gal`), from conftest.py, run through the UV
# continuum property calculator chain. Uses real (small) PDF chains
# and lets the calculators' own caching mechanism save outputs under
# testing/test_work/RestPhot_PDFs/..., without any real FITS/Gaia data.
#################################################


class TestHighZSyntheticGalaxyProperties:
    """`high_z_gal` has a real, fittable UV continuum, so every
    calculator in the chain should return a finite value; `garbage_gal`
    is undetected in every band, so the UV slope fit fails and NaN
    should propagate through every downstream (dependent) property."""

    def _get_calculators(self, sed_fitter):
        aper_diam = 0.32 * u.arcsec
        return [
            UV_Beta_Calculator(aper_diam, sed_fitter),
            mUV_Calculator(aper_diam, sed_fitter, ext_src_corrs=None),
            MUV_Calculator(aper_diam, sed_fitter, ext_src_corrs=None),
            LUV_Calculator(aper_diam, sed_fitter, ext_src_corrs=None),
            SFR_UV_Calculator(aper_diam, sed_fitter, ext_src_corrs=None),
            UV_Dust_Attenuation_Calculator(aper_diam, sed_fitter),
            Fesc_From_Beta_Calculator(aper_diam, sed_fitter),
            Lya_Break_Strength_Calculator(aper_diam, sed_fitter),
        ]

    def test_high_z_gal_properties_are_finite(
        self, high_z_gal, eazy_fsps_larson_sed_fitter
    ):
        aper_diam = 0.32 * u.arcsec
        for calc in self._get_calculators(eazy_fsps_larson_sed_fitter):
            out_gal = calc(high_z_gal, n_chains=200, overwrite=True)
            phot_rest = out_gal.aper_phot[aper_diam].SED_results[
                eazy_fsps_larson_sed_fitter.label
            ].phot_rest
            value = phot_rest.properties[calc.name]
            value = value.value if hasattr(value, "value") else value
            assert np.isfinite(value), (
                f"{calc.__class__.__name__} ({calc.name}) returned "
                f"non-finite value {value!r} for a genuine high-z source"
            )

    def test_garbage_gal_properties_are_nan(
        self, garbage_gal, eazy_fsps_larson_sed_fitter
    ):
        aper_diam = 0.32 * u.arcsec
        for calc in self._get_calculators(eazy_fsps_larson_sed_fitter):
            out_gal = calc(garbage_gal, n_chains=200, overwrite=True)
            phot_rest = out_gal.aper_phot[aper_diam].SED_results[
                eazy_fsps_larson_sed_fitter.label
            ].phot_rest
            value = phot_rest.properties[calc.name]
            value = value.value if hasattr(value, "value") else value
            assert np.isnan(value), (
                f"{calc.__class__.__name__} ({calc.name}) returned "
                f"non-nan value {value!r} for an undetected source"
            )

    def test_synthetic_cat_sources_distinguished(
        self, synthetic_test_cat, eazy_fsps_larson_sed_fitter
    ):
        # sanity check the calculator chain distinguishes the two
        # sources in the catalogue the same way as the standalone
        # galaxy fixtures (run per-galaxy: property calculators need
        # cat.open_cat()/write_hdu() for genuine catalogue-level
        # dispatch, which needs a real FITS-backed catalogue -- see
        # TestSyntheticCatalogue in test_Selectors.py)
        aper_diam = 0.32 * u.arcsec
        calc = UV_Beta_Calculator(aper_diam, eazy_fsps_larson_sed_fitter)
        results = {}
        for gal in synthetic_test_cat:
            out_gal = calc(gal, n_chains=200, overwrite=True)
            phot_rest = out_gal.aper_phot[aper_diam].SED_results[
                eazy_fsps_larson_sed_fitter.label
            ].phot_rest
            value = phot_rest.properties[calc.name]
            results[gal.ID] = value.value if hasattr(value, "value") else value
        assert np.isfinite(results[1])
        assert np.isnan(results[2])

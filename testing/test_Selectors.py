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


from conftest import (
    test_aper_diams,
    test_bands_,
    test_forced_phot_band_,
    test_survey,
)
from galfind import config
from galfind.catalogues import Catalogue
from galfind.galaxy import Galaxy
from galfind.imaging import NIRCam
from galfind.imaging.Instrument import JWST
from galfind.properties import UV_Beta_Calculator
from galfind.selection.Selector import (
    Band_Mag_Selector,
    Band_SNR_Selector,
    Bluewards_Lya_Non_Detect_Selector,
    Bluewards_LyLim_Non_Detect_Selector,
    Brown_Dwarf_Selector,
    Chi_Sq_Diff_Selector,
    Chi_Sq_Lim_Selector,
    Chi_Sq_Template_Diff_Selector,
    Colour_Selector,
    Compactness_Selector,
    COSMOS_Web_Selector,
    Depth_Region_Selector,
    Ds9_Region_Selector,
    EPOCHS_Selector,
    EPOCHS_unmasked_criteria,
    Hainline24_TY_Brown_Dwarf_Selector_1,
    Hainline24_TY_Brown_Dwarf_Selector_2,
    ID_Selector,
    Kokorev24_LRD_red1_Selector,
    Kokorev24_LRD_red2_Selector,
    Kokorev24_LRD_Selector,
    Lya_Band_Selector,
    Min_Band_Selector,
    Min_Instrument_Unmasked_Band_Selector,
    Min_Unmasked_Band_Selector,
    Re_Selector,
    Redshift_Bin_Selector,
    Redshift_Limit_Selector,
    Redwards_Lya_Detect_Selector,
    Rest_Frame_Property_Bin_Selector,
    Rest_Frame_Property_Kwarg_Selector,
    Rest_Frame_Property_Limit_Selector,
    Robust_zPDF_Selector,
    Selector,
    Sextractor_Band_Radius_Selector,  # up to here
    Sextractor_Bands_Radius_Selector,
    Sextractor_Instrument_Radius_PSF_FWHM_Selector,
    Sextractor_Instrument_Radius_Selector,
    Stacked_Blue_Lya_Non_Detect_Selector,
    Unmasked_Band_Selector,
    Unmasked_Bands_Selector,
    Unmasked_Bluewards_Lya_Selector,
    Unmasked_Instrument_Selector,
    Unmasked_Redwards_Lya_Selector,
    zPDF_High_Tail_Selector,
)


@pytest.fixture(
    scope="module",
    params=[
        ({"IDs": 1}, True),
        ({"IDs": [1, 2]}, True),
        ({"IDs": np.array([1, 2])}, True),
        ({"IDs": 123_456_789}, {"gal": True, "cat": Exception}),
        ({"IDs": np.arange(1, 50, 1), "name": "large_named"}, True),
    ],
)
def call_ID_selector(request):
    return ID_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        ({"IDs": "invalid"}, Exception),
        ({"IDs": 1.5}, Exception),
        ({"IDs": [1, "2"]}, Exception),
        ({"name": "test"}, Exception),
        ({"IDs": [0, 1]}, Exception),
    ],
)
def fail_ID_selector(request):
    return ID_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "region_path": (
                    f"{config['Masking']['MASK_DIR']}/{test_survey}"
                    f"/reg/stellar/{test_bands_[-1]}_stellar.reg"
                )
            },
            True,
        ),
        (
            {
                "region_path": (
                    f"{config['Masking']['MASK_DIR']}/{test_survey}"
                    f"/reg/stellar/{test_bands_[-1]}_stellar.reg"
                ),
                "region_name": f"{test_bands_[-1]}_stellar",
            },
            True,
        ),
        (
            {
                "region_path": (
                    f"{config['Masking']['MASK_DIR']}/{test_survey}"
                    f"/reg/stellar/{test_bands_[-1]}_stellar.reg"
                ),
                "region_name": f"{test_bands_[-1]}_stellar",
                "fail_name": f"not_{test_bands_[-1]}_stellar",
            },
            True,
        ),
    ],
)
def call_ds9_region_selector(request):
    return Ds9_Region_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        ({"region_path": "invalid_path.reg"}, Exception),
    ],
)
def fail_ds9_region_selector(request):
    return Ds9_Region_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": test_aper_diams[0],
                "filt_name": test_forced_phot_band_[0],
                "region_label": 0,
            },
            True,
        ),
        (
            {
                "aper_diam": 0.1 * u.arcsec,
                "filt_name": test_forced_phot_band_[0],
                "region_label": 0,
            },
            True,
        ),
        (
            {
                "aper_diam": test_aper_diams[0],
                "filt_name": test_forced_phot_band_,
                "region_label": 0,
            },
            True,
        ),
    ],
)
def call_depth_region_selector(request):
    return Depth_Region_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": test_aper_diams[0],
                "filt_name": 0,
                "region_label": 0,
            },
            Exception,
        ),
        (
            {
                "aper_diam": test_aper_diams[0].value,
                "filt_name": test_forced_phot_band_[0],
                "region_label": 0,
            },
            Exception,
        ),
    ],
)
def fail_depth_region_selector(request):
    return Depth_Region_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "z_lim": 1.0,
                "gtr_or_less": "gtr",
            },
            True,
        ),
        (
            {
                "aper_diam": 0.1 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "z_lim": 1.0,
                "gtr_or_less": "gtr",
            },
            Exception,
        ),
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "z_lim": 1.0,
                "gtr_or_less": "less",
            },
            True,
        ),
    ],
)
def call_redshift_limit_selector(request):
    inputs, outcome = request.param
    inputs["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return Redshift_Limit_Selector, inputs, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "z_lim": 1.0,
                "gtr_or_less": "random",
            },
            Exception,
        ),
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "z_lim": -1.0,
                "gtr_or_less": "gtr",
            },
            Exception,
        ),
    ],
)
def fail_redshift_limit_selector(request):
    inputs, outcome = request.param
    inputs["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return Redshift_Limit_Selector, inputs, outcome


@pytest.fixture(scope="module")
def uv_beta_calculator(eazy_fsps_larson_sed_fitter):
    return UV_Beta_Calculator(test_aper_diams[0], eazy_fsps_larson_sed_fitter)


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "property_calculator": "uv_beta_calculator",
                "property_lim": -2.0 * u.dimensionless_unscaled,
                "gtr_or_less": "less",
            },
            True,
        ),
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "property_calculator": "uv_beta_calculator",
                "property_lim": -2.0 * u.dimensionless_unscaled,
                "gtr_or_less": "gtr",
            },
            True,
        ),
    ],
)
def call_rest_frame_property_limit_selector(request):
    inputs, outcome = request.param
    inputs["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    inputs["property_calculator"] = request.getfixturevalue(
        inputs["property_calculator"]
    )
    return Rest_Frame_Property_Limit_Selector, inputs, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "property_calculator": "uv_beta_calculator",
                "property_lim": -2.0 * u.dimensionless_unscaled,
                "gtr_or_less": "invalid",
            },
            Exception,
        ),
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "property_calculator": "uv_beta_calculator",
                "property_lim": -2.0,
                "gtr_or_less": "less",
            },
            Exception,
        ),
        (
            {
                "aper_diam": 0.1 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "property_calculator": "uv_beta_calculator",
                "property_lim": -2.0 * u.dimensionless_unscaled,
                "gtr_or_less": "less",
            },
            Exception,
        ),
    ],
)
def fail_rest_frame_property_limit_selector(request):
    inputs, outcome = request.param
    inputs["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    inputs["property_calculator"] = request.getfixturevalue(
        inputs["property_calculator"]
    )
    return Rest_Frame_Property_Limit_Selector, inputs, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "z_bin": [0.5, 1.0],
            },
            True,
        ),
        (
            {
                "aper_diam": 0.1 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "z_bin": [0.5, 1.0],
            },
            Exception,
        ),
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "z_bin": [1, 2],
            },
            True,
        ),
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "z_bin": [1.2, 2],
            },
            True,
        ),
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "z_bin": np.array([1.5, 2.5]),
            },
            True,
        ),
    ],
)
def call_redshift_bin_selector(request):
    inputs, outcome = request.param
    inputs["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return Redshift_Bin_Selector, inputs, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "z_bin": [1.0, 0.5],
            },
            Exception,
        ),
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "z_bin": [0.5, "1.0"],
            },
            Exception,
        ),
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "z_bin": 1.0,
            },
            Exception,
        ),
    ],
)
def fail_redshift_bin_selector(request):
    inputs, outcome = request.param
    inputs["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return Redshift_Bin_Selector, inputs, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "property_calculator": "uv_beta_calculator",
                "property_bin": [-3.0, -1.0] * u.dimensionless_unscaled,
            },
            True,
        ),
    ],
)
def call_rest_frame_property_bin_selector(request):
    inputs, outcome = request.param
    inputs["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    inputs["property_calculator"] = request.getfixturevalue(
        inputs["property_calculator"]
    )
    return Rest_Frame_Property_Bin_Selector, inputs, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "property_calculator": "uv_beta_calculator",
                "property_bin": [-1.0, -3.0] * u.dimensionless_unscaled,
            },
            Exception,
        ),
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "property_calculator": "uv_beta_calculator",
                "property_bin": [-3.0, -2.0, -1.0]
                * u.dimensionless_unscaled,
            },
            Exception,
        ),
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "property_calculator": "uv_beta_calculator",
                "property_bin": [-3.0, -1.0],
            },
            Exception,
        ),
    ],
)
def fail_rest_frame_property_bin_selector(request):
    inputs, outcome = request.param
    inputs["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    inputs["property_calculator"] = request.getfixturevalue(
        inputs["property_calculator"]
    )
    return Rest_Frame_Property_Bin_Selector, inputs, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": test_aper_diams[0],
                "colour_bands": [
                    test_forced_phot_band_[0],
                    test_forced_phot_band_[1],
                ],
                "bluer_or_redder": "bluer",
                "colour_val": 0.5,
            },
            True,
        ),
        (
            {
                "aper_diam": 0.1 * u.arcsec,
                "colour_bands": [
                    test_forced_phot_band_[0],
                    test_forced_phot_band_[1],
                ],
                "bluer_or_redder": "bluer",
                "colour_val": 0.5,
            },
            True,
        ),
        (
            {
                "aper_diam": test_aper_diams[0],
                "colour_bands": (
                    f"{test_forced_phot_band_[0]}-{test_forced_phot_band_[1]}"
                ),
                "bluer_or_redder": "bluer",
                "colour_val": 0.5,
            },
            True,
        ),
        (
            {
                "aper_diam": test_aper_diams[0],
                "colour_bands": np.array(
                    [test_forced_phot_band_[0], test_forced_phot_band_[1]]
                ),
                "bluer_or_redder": "bluer",
                "colour_val": 0.5,
            },
            True,
        ),
        (
            {
                "aper_diam": test_aper_diams[0],
                "colour_bands": [
                    test_forced_phot_band_[0],
                    test_forced_phot_band_[1],
                ],
                "bluer_or_redder": "redder",
                "colour_val": 0.5,
            },
            True,
        ),
        (
            {
                "aper_diam": test_aper_diams[0],
                "colour_bands": (
                    f"{test_forced_phot_band_[0]}-{test_forced_phot_band_[1]}"
                ),
                "bluer_or_redder": "bluer",
                "colour_val": 2,
            },
            True,
        ),
        (
            {
                "aper_diam": test_aper_diams[0],
                "colour_bands": [
                    test_forced_phot_band_[0],
                    test_forced_phot_band_[1],
                ],
                "bluer_or_redder": "bluer",
                "colour_val": -0.5,
            },
            True,
        ),
    ],
)
def call_colour_selector(request):
    return Colour_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": test_aper_diams[0],
                "colour_bands": [
                    test_forced_phot_band_[0],
                    test_forced_phot_band_[1],
                ],
                "bluer_or_redder": "random",
                "colour_val": 0.5,
            },
            Exception,
        ),
        (
            {
                "aper_diam": test_aper_diams[0].value,
                "colour_bands": [
                    test_forced_phot_band_[0],
                    test_forced_phot_band_[1],
                ],
                "bluer_or_redder": "bluer",
                "colour_val": 0.5,
            },
            Exception,
        ),
    ],
)
def fail_colour_selector(request):
    return Colour_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        ({"min_bands": 2}, True),
    ],
)
def call_min_band_selector(request):
    return Min_Band_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        ({"min_bands": 2.0}, Exception),
        ({"min_bands": -1}, Exception),
        ({"min_bands": "invalid"}, Exception),
        ({"min_bands": [2]}, Exception),
        ({"min_bands": np.array([2])}, Exception),
    ],
)
def fail_min_band_selector(request):
    return Min_Band_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "filt_name": "F444W",
                "compare_radii": [0.16, 0.32] * u.arcsec,
                "compactness_lim": 1.2,
            },
            True,
        ),
        (
            {
                "filt_name": "F444W",
                "compare_radii": [0.16, 0.32] * u.arcsec,
                "compactness_lim": 1.2,
                "psf_normed": False,
            },
            True,
        ),
    ],
)
def call_compactness_selector(request):
    return Compactness_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "filt_name": "not_a_filter",
                "compare_radii": [0.16, 0.32] * u.arcsec,
                "compactness_lim": 1.2,
            },
            Exception,
        ),
        (
            {
                "filt_name": "F444W",
                "compare_radii": [0.32, 0.16] * u.arcsec,
                "compactness_lim": 1.2,
            },
            Exception,
        ),
        (
            {
                "filt_name": "F444W",
                "compare_radii": [0.16] * u.arcsec,
                "compactness_lim": 1.2,
            },
            Exception,
        ),
        (
            {
                "filt_name": "F444W",
                "compare_radii": [0.16, 0.32],
                "compactness_lim": 1.2,
            },
            Exception,
        ),
        (
            {
                "filt_name": "F444W",
                "compare_radii": [0.16, 0.32] * u.arcsec,
                "compactness_lim": "invalid",
            },
            Exception,
        ),
        (
            {
                "filt_name": "F444W",
                "compare_radii": [0.16, 0.32] * u.arcsec,
                "compactness_lim": 1.2,
                "psf_normed": "invalid",
            },
            Exception,
        ),
    ],
)
def fail_compactness_selector(request):
    return Compactness_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        ({"filt_name": "F444W"}, True),
    ],
)
def call_unmasked_band_selector(request):
    return Unmasked_Band_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        ({"filt_name": "invalid"}, Exception),
        ({"filt_name": 0}, Exception),
        ({"filt_name": 0.32 * u.arcsec}, Exception),
        ({"filt_name": [0, 1]}, Exception),
    ],
)
def fail_unmasked_band_selector(request):
    return Unmasked_Band_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        ({"min_bands": 2}, True),
    ],
)
def call_min_unmasked_band_selector(request):
    return Min_Unmasked_Band_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        ({"min_bands": 2.0}, Exception),
        ({"min_bands": -1}, Exception),
        ({"min_bands": "invalid"}, Exception),
        ({"min_bands": [2]}, Exception),
        ({"min_bands": np.array([2])}, Exception),
    ],
)
def fail_min_unmasked_band_selector(request):
    return Min_Unmasked_Band_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        ({"min_bands": 1, "instrument": NIRCam()}, True),
        ({"min_bands": 1, "instrument": "NIRCam"}, True),
    ],
)
def call_min_instrument_unmasked_band_selector(request):
    return Min_Instrument_Unmasked_Band_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        ({"min_bands": 2.0, "instrument": NIRCam()}, Exception),
        ({"min_bands": -1, "instrument": NIRCam()}, Exception),
        ({"min_bands": 1, "instrument": "invalid"}, Exception),
        ({"min_bands": 1, "instrument": JWST()}, Exception),
        ({"min_bands": 0.32 * u.arcsec, "instrument": NIRCam()}, Exception),
    ],
)
def fail_min_instrument_unmasked_band_selector(request):
    return Min_Instrument_Unmasked_Band_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "SNR_lim": 2.0,
            },
            True,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "SNR_lim": 2.0,
            },
            True,
        ),
    ],
)
def call_bluewards_lylim_non_detect_selector(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()
    inputs_["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return Bluewards_LyLim_Non_Detect_Selector, inputs_, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "SNR_lim": 2.0,
            },
            Exception,
        ),
    ],
)
def fail_bluewards_lylim_non_detect_selector(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()  # monkeypatch?
    inputs_["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return Bluewards_LyLim_Non_Detect_Selector, inputs_, outcome


@pytest.fixture(scope="module")
def call_bluewards_lya_non_detect_selector(
    call_bluewards_lylim_non_detect_selector,
):
    call_cls, inputs, _ = call_bluewards_lylim_non_detect_selector
    return Bluewards_Lya_Non_Detect_Selector, inputs, True


@pytest.fixture(scope="module")
def fail_bluewards_lya_non_detect_selector(
    fail_bluewards_lylim_non_detect_selector,
):
    fail_cls, inputs, outcome = fail_bluewards_lylim_non_detect_selector
    return Bluewards_Lya_Non_Detect_Selector, inputs, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "SNR_lims": 2.0,
                "widebands_only": True,
            },
            True,
        ),
    ],
)
def call_redwards_lya_detect_selector(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()
    inputs_["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return Redwards_Lya_Detect_Selector, inputs_, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "SNR_lims": -1.0,
                "widebands_only": True,
            },
            Exception,
        ),
    ],
)
def fail_redwards_lya_detect_selector(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()  # monkeypatch?
    inputs_["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return Redwards_Lya_Detect_Selector, inputs_, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "SNR_lim": 2.0,
                "detect_or_non_detect": "detect",
                "widebands_only": True,
            },
            True,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "SNR_lim": 2.0,
                "detect_or_non_detect": "non_detect",
                "widebands_only": True,
            },
            True,
        ),
    ],
)
def call_lya_band_selector(request):
    inputs, outcome = request.param
    inputs["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return Lya_Band_Selector, inputs, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "SNR_lim": 2.0,
                "detect_or_non_detect": "invalid",
                "widebands_only": True,
            },
            Exception,
        ),
    ],
)
def fail_lya_band_selector(request):
    inputs, outcome = request.param
    inputs["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return Lya_Band_Selector, inputs, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "n_bands": 2,
                "widebands_only": True,
            },
            True,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "n_bands": 2,
                "widebands_only": False,
            },
            True,
        ),
    ],
)
def call_unmasked_bluewards_lya_selector(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()
    inputs_["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return Unmasked_Bluewards_Lya_Selector, inputs_, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "n_bands": 2.0,
                "widebands_only": True,
            },
            Exception,
        ),
    ],
)
def fail_unmasked_bluewards_lya_selector(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()  # monkeypatch?
    inputs_["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return Unmasked_Bluewards_Lya_Selector, inputs_, outcome


@pytest.fixture(scope="module")
def call_unmasked_redwards_lya_selector(call_unmasked_bluewards_lya_selector):
    call_cls, inputs, _ = call_unmasked_bluewards_lya_selector
    return Unmasked_Redwards_Lya_Selector, inputs, True


@pytest.fixture(scope="module")
def fail_unmasked_redwards_lya_selector(fail_unmasked_bluewards_lya_selector):
    fail_cls, inputs, outcome = fail_unmasked_bluewards_lya_selector
    return Unmasked_Redwards_Lya_Selector, inputs, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "SNR_lim": 2.0,
            },
            True,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "SNR_lim": 2.0,
                "dz": 0.2,
            },
            True,
        ),
    ],
)
def call_stacked_blue_lya_non_detect_selector(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()
    inputs_["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return Stacked_Blue_Lya_Non_Detect_Selector, inputs_, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32,  # missing units
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "SNR_lim": 2.0,
            },
            Exception,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "SNR_lim": 2.0,
                "dz": -1.0,
            },
            Exception,
        ),
    ],
)
def fail_stacked_blue_lya_non_detect_selector(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()
    inputs_["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return Stacked_Blue_Lya_Non_Detect_Selector, inputs_, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "band": "F444W",
                "detect_or_non_detect": "detect",
                "SNR_lim": 5.0,
            },
            True,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "band": "F444W",
                "detect_or_non_detect": "non_detect",
                "SNR_lim": 5.0,
            },
            True,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "band": 0,
                "detect_or_non_detect": "detect",
                "SNR_lim": 5.0,
            },
            True,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "band": -1,
                "detect_or_non_detect": "detect",
                "SNR_lim": 5.0,
            },
            True,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "band": 1,
                "detect_or_non_detect": "detect",
                "SNR_lim": 5.0,
            },
            True,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "band": -2,
                "detect_or_non_detect": "detect",
                "SNR_lim": 5.0,
            },
            True,
        ),
    ],
)
def call_band_snr_selector(request):
    return Band_SNR_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "band": "invalid",
                "detect_or_non_detect": "detect",
                "SNR_lim": 5.0,
            },
            Exception,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "band": "F444W",
                "detect_or_non_detect": "invalid",
                "SNR_lim": 5.0,
            },
            Exception,
        ),
    ],
)
def fail_band_snr_selector(request):
    return Band_SNR_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "band": "F444W",
                "detect_or_non_detect": "detect",
                "mag_lim": 28.0,
            },
            True,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "band": "F444W",
                "detect_or_non_detect": "non_detect",
                "mag_lim": 28.0,
            },
            True,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "band": 1,
                "detect_or_non_detect": "non_detect",
                "mag_lim": 28.0,
            },
            True,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "band": 0,
                "detect_or_non_detect": "non_detect",
                "mag_lim": 28.0,
            },
            True,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "band": -1,
                "detect_or_non_detect": "non_detect",
                "mag_lim": 28.0,
            },
            True,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "band": -2,
                "detect_or_non_detect": "non_detect",
                "mag_lim": 28.0,
            },
            True,
        ),
    ],
)
def call_band_mag_selector(request):
    return Band_Mag_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "band": "F444W",
                "detect_or_non_detect": "invalid",
                "mag_lim": 28.0,
            },
            Exception,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "band": "invalid",
                "detect_or_non_detect": "detect",
                "mag_lim": 28.0,
            },
            Exception,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "band": "F444W",
                "detect_or_non_detect": "detect",
                "mag_lim": 28.0 * u.ABmag,  # shouldn't really fail here!
            },
            Exception,
        ),
    ],
)
def fail_band_mag_selector(request):
    return Band_Mag_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "chi_sq_lim": 4.0,
                "reduced": True,
            },
            True,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "chi_sq_lim": 4.0,
                "reduced": False,
            },
            True,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "chi_sq_lim": 2,
                "reduced": True,
            },
            True,
        ),
    ],
)
def call_chi_sq_lim_selector(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()
    inputs_["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return Chi_Sq_Lim_Selector, inputs_, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "chi_sq_lim": 4.0,
                "reduced": "invalid",
            },
            Exception,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "chi_sq_lim": -1.0,
                "reduced": True,
            },
            Exception,
        ),
    ],
)
def fail_chi_sq_lim_selector(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()
    inputs_["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return Chi_Sq_Lim_Selector, inputs_, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "chi_sq_diff": 2.0,
                "dz": 0.5,
                "lowz_zmax_arr": [4.0, 6.0],
            },
            True,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "chi_sq_diff": 2,
                "dz": 0.5,
                "lowz_zmax_arr": [4.0, 6.0],
            },
            True,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "chi_sq_diff": 2.0,
                "dz": 1,
                "lowz_zmax_arr": [4.0, 6.0],
            },
            True,
        ),
    ],
)
def call_chi_sq_diff_selector(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()
    inputs_["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return Chi_Sq_Diff_Selector, inputs_, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "chi_sq_diff": -1.0,
                "dz": 0.5,
            },
            Exception,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "chi_sq_diff": 2.0,
                "dz": -0.5,
            },
            Exception,
        ),
    ],
)
def fail_chi_sq_diff_selector(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()
    inputs_["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return Chi_Sq_Diff_Selector, inputs_, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "chi_sq_diff": 2.0,
                "secondary_SED_fit_label": "eazy_sfhz_sed_fitter",
            },
            True,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "chi_sq_diff": 2,
                "secondary_SED_fit_label": "eazy_sfhz_sed_fitter",
                "reduced": True,
            },
            True,
        ),
    ],
)
def call_chi_sq_template_diff_selector(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()
    inputs_["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    inputs_["secondary_SED_fit_label"] = request.getfixturevalue(
        inputs["secondary_SED_fit_label"]
    )
    return Chi_Sq_Template_Diff_Selector, inputs_, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "chi_sq_diff": -1.0,
                "secondary_SED_fit_label": "eazy_sfhz_sed_fitter",
            },
            Exception,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "chi_sq_diff": 2.0,
                "secondary_SED_fit_label": "eazy_sfhz_sed_fitter",
                "reduced": "invalid",
            },
            Exception,
        ),
    ],
)
def fail_chi_sq_template_diff_selector(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()
    inputs_["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    inputs_["secondary_SED_fit_label"] = request.getfixturevalue(
        inputs["secondary_SED_fit_label"]
    )
    return Chi_Sq_Template_Diff_Selector, inputs_, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "integral_lim": 0.6,
                "dz_over_z": 0.1,
            },
            True,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "integral_lim": 0.6,
                "dz_over_z": 1.5,
            },
            True,
        ),
    ],
)
def call_robust_zPDF_selector(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()
    inputs_["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return Robust_zPDF_Selector, inputs_, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "integral_lim": 1.1,
                "dz_over_z": 0.1,
            },
            Exception,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "integral_lim": 1.234_567_890,
                "dz_over_z": 0.1,
            },
            Exception,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "integral_lim": 0.60,
                "dz_over_z": 0.0,
            },
            Exception,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "integral_lim": 0.6,
                "dz_over_z": -1.0,
            },
            Exception,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "integral_lim": 1.1,
                "dz_over_z": 0.0,
            },
            Exception,
        ),
    ],
)
def fail_robust_zPDF_selector(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()
    inputs_["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return Robust_zPDF_Selector, inputs_, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fit_label": "eazy_fsps_larson_sed_fitter",
                "z_thr": 8.0,
                "p_lim": 0.8,
            },
            True,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fit_label": "eazy_fsps_larson_sed_fitter",
                "z_thr": 0.1,
                "p_lim": 0.1,
            },
            True,
        ),
    ],
)
def call_zpdf_high_tail_selector(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()
    inputs_["SED_fit_label"] = request.getfixturevalue(
        inputs["SED_fit_label"]
    )
    return zPDF_High_Tail_Selector, inputs_, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fit_label": "eazy_fsps_larson_sed_fitter",
                "z_thr": -1.0,
                "p_lim": 0.8,
            },
            Exception,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fit_label": "eazy_fsps_larson_sed_fitter",
                "z_thr": 8.0,
                "p_lim": 1,  # must be float, not int
            },
            Exception,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fit_label": "eazy_fsps_larson_sed_fitter",
                "z_thr": 8.0,
                "p_lim": 1.5,
            },
            Exception,
        ),
        (
            {
                "aper_diam": 0.32 * u.arcsec,
                "SED_fit_label": "eazy_fsps_larson_sed_fitter",
                "z_thr": 8.0,
                "p_lim": 0.0,
            },
            Exception,
        ),
    ],
)
def fail_zpdf_high_tail_selector(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()
    inputs_["SED_fit_label"] = request.getfixturevalue(
        inputs["SED_fit_label"]
    )
    return zPDF_High_Tail_Selector, inputs_, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "filt_name": "F444W",
                "gtr_or_less": "gtr",
                "lim": 50.0 * u.marcsec,
            },
            True,
        ),
        (
            {
                "filt_name": "F444W",
                "gtr_or_less": "less",
                "lim": 100.0 * u.marcsec,
            },
            True,
        ),
        (
            {
                "filt_name": "F200W",
                "gtr_or_less": "gtr",
                "lim": 45.0 * u.marcsec,
            },
            True,
        ),
    ],
)
def call_sextractor_band_radius_selector(request):
    return Sextractor_Band_Radius_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "filt_name": "invalid_band",
                "gtr_or_less": "gtr",
                "lim": 50.0 * u.marcsec,
            },
            Exception,
        ),
        (
            {
                "filt_name": "F444W",
                "gtr_or_less": "invalid",
                "lim": 50.0 * u.marcsec,
            },
            Exception,
        ),
        (
            {
                "filt_name": "F444W",
                "gtr_or_less": "gtr",
                "lim": -50.0 * u.marcsec,
            },
            Exception,
        ),
        (
            {
                "filt_name": "F444W",
                "gtr_or_less": "gtr",
                "lim": 50.0,  # missing units
            },
            Exception,
        ),
    ],
)
def fail_sextractor_band_radius_selector(request):
    return Sextractor_Band_Radius_Selector, *request.param


# Re_Selector requires a morph_fitter which is complex to instantiate
# in tests without full data. We'll skip these fixtures as they require
# Morphology_Fitter objects that need actual PSF and cutout data.


@pytest.fixture(
    scope="module",
    params=[
        ({"aper_diam": test_aper_diams[0]}, True),
    ],
)
def call_kokorev24_lrd_red1_selector(request):
    return Kokorev24_LRD_red1_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        ({"aper_diam": 0.32}, Exception),  # missing units
        ({"aper_diam": -0.32 * u.arcsec}, Exception),  # negative aperture
    ],
)
def fail_kokorev24_lrd_red1_selector(request):
    return Kokorev24_LRD_red1_Selector, *request.param


@pytest.fixture(scope="module")
def call_kokorev24_lrd_red2_selector(call_kokorev24_lrd_red1_selector):
    call_cls, inputs, _ = call_kokorev24_lrd_red1_selector
    return Kokorev24_LRD_red2_Selector, inputs, True


@pytest.fixture(scope="module")
def fail_kokorev24_lrd_red2_selector(fail_kokorev24_lrd_red1_selector):
    fail_cls, inputs, outcome = fail_kokorev24_lrd_red1_selector
    return Kokorev24_LRD_red2_Selector, inputs, outcome


@pytest.fixture(scope="module")
def call_kokorev24_lrd_selector(call_kokorev24_lrd_red1_selector):
    call_cls, inputs, _ = call_kokorev24_lrd_red1_selector
    return Kokorev24_LRD_Selector, inputs, True


@pytest.fixture(scope="module")
def fail_kokorev24_lrd_selector(fail_kokorev24_lrd_red1_selector):
    fail_cls, inputs, outcome = fail_kokorev24_lrd_red1_selector
    return Kokorev24_LRD_Selector, inputs, outcome


@pytest.fixture(
    scope="module",
    params=[
        ({"filt_names": "F444W"}, True),
        ({"filt_names": ["F444W"]}, True),
        ({"filt_names": ["F200W", "F444W"]}, True),
        ({"filt_names": "F200W+F444W"}, True),
        ({"filt_names": "F277W+F444W"}, Exception),
    ],
)
def call_unmasked_bands_selector(request):
    return Unmasked_Bands_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        ({"filt_names": "invalid_band"}, Exception),
        ({"filt_names": ["invalid1", "invalid2"]}, Exception),
    ],
)
def fail_unmasked_bands_selector(request):
    return Unmasked_Bands_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        ({"instrument": NIRCam()}, True),
        ({"instrument": "NIRCam"}, True),
    ],
)
def call_unmasked_instrument_selector(request):
    return Unmasked_Instrument_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        ({"instrument": "invalid_instrument"}, Exception),
        # JWST is a facility, not an instrument
        ({"instrument": JWST()}, Exception),
    ],
)
def fail_unmasked_instrument_selector(request):
    return Unmasked_Instrument_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "filt_names": ["F277W", "F356W", "F444W"],
                "gtr_or_less": "gtr",
                "lim": 45.0 * u.marcsec,
            },
            Exception,
        ),
        (
            {
                "filt_names": ["F444W"],
                "gtr_or_less": "less",
                "lim": 100.0 * u.marcsec,
            },
            True,
        ),
    ],
)
def call_sextractor_bands_radius_selector(request):
    return Sextractor_Bands_Radius_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "filt_names": ["invalid_band"],
                "gtr_or_less": "gtr",
                "lim": 45.0 * u.marcsec,
            },
            Exception,
        ),
        (
            {
                "filt_names": ["F444W"],
                "gtr_or_less": "invalid",
                "lim": 45.0 * u.marcsec,
            },
            Exception,
        ),
    ],
)
def fail_sextractor_bands_radius_selector(request):
    return Sextractor_Bands_Radius_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "instrument": "NIRCam",
                "gtr_or_less": "gtr",
                "lim": 45.0 * u.marcsec,
            },
            {"cat": True, "gal": Exception},
        ),
        (
            {
                "instrument": "NIRCam",
                "gtr_or_less": "less",
                "lim": 100.0 * u.marcsec,
            },
            {"cat": True, "gal": Exception},
        ),
    ],
)
def call_sextractor_instrument_radius_selector(request):
    return Sextractor_Instrument_Radius_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "instrument": "invalid_instrument",
                "gtr_or_less": "gtr",
                "lim": 45.0 * u.marcsec,
            },
            Exception,
        ),
        (
            {
                "instrument": "NIRCam",
                "gtr_or_less": "invalid",
                "lim": 45.0 * u.marcsec,
            },
            Exception,
        ),
    ],
)
def fail_sextractor_instrument_radius_selector(request):
    return Sextractor_Instrument_Radius_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "instrument": "NIRCam",
                "gtr_or_less": "gtr",
                "scaling": 1.0,
            },
            {"cat": True, "gal": Exception},
        ),
        (
            {
                "instrument": "NIRCam",
                "gtr_or_less": "less",
                "scaling": 0.5,
            },
            {"cat": True, "gal": Exception},
        ),
        (
            {
                "instrument": "NIRCam",
                "gtr_or_less": "gtr",
                "scaling": 1.2,
            },
            {"cat": True, "gal": Exception},
        ),
    ],
)
def call_sextractor_instrument_radius_psf_fwhm_selector(request):
    return Sextractor_Instrument_Radius_PSF_FWHM_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "instrument": "invalid_instrument",
                "gtr_or_less": "gtr",
                "scaling": 1.0,
            },
            Exception,
        ),
        (
            {
                "instrument": "ACS_WFC",  # Not NIRCam, which is required
                "gtr_or_less": "gtr",
                "scaling": 1.0,
            },
            Exception,
        ),
    ],
)
def fail_sextractor_instrument_radius_psf_fwhm_selector(request):
    return Sextractor_Instrument_Radius_PSF_FWHM_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "red_chi_sq_diff": 4.0,
                "gal_SED_fitter": "EAZY_fsps_larson",
                "ref_filt": "F444W",
            },
            True,
        ),
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "red_chi_sq_diff": 4.0,
                "gal_SED_fitter": "EAZY_fsps_larson",
                "ref_filt": "F444W",
                "SNR_lim": 3.0,
                "red_chi_sq_lim": 3.0,
            },
            True,
        ),
    ],
)
def call_brown_dwarf_selector(request):
    inputs, outcome = request.param
    inputs["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return Brown_Dwarf_Selector, inputs, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "red_chi_sq_diff": 4.0,
                "gal_SED_fitter": "EAZY_fsps_larson",
                "ref_filt": "not_a_filter",
            },
            Exception,
        ),
    ],
)
def fail_brown_dwarf_selector(request):
    inputs, outcome = request.param
    inputs["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return Brown_Dwarf_Selector, inputs, outcome


# Re_Selector requires a morph_fitter (Morphology_Fitter) which is complex to
# instantiate in tests without full data. Adding placeholder fixtures.
@pytest.fixture(scope="module", params=[])
def call_re_selector(request):
    return Re_Selector, *request.param


@pytest.fixture(scope="module", params=[])
def fail_re_selector(request):
    return Re_Selector, *request.param


@pytest.fixture(
    scope="module",
    params=[
        ({"aper_diam": test_aper_diams[0]}, True),
    ],
)
def call_hainline24_ty_brown_dwarf_selector_1(request):
    return Hainline24_TY_Brown_Dwarf_Selector_1, *request.param


@pytest.fixture(
    scope="module",
    params=[
        ({"aper_diam": 0.32}, Exception),  # missing units
        ({"aper_diam": -0.32 * u.arcsec}, Exception),  # negative aperture
    ],
)
def fail_hainline24_ty_brown_dwarf_selector_1(request):
    return Hainline24_TY_Brown_Dwarf_Selector_1, *request.param


@pytest.fixture(
    scope="module",
    params=[
        ({"aper_diam": test_aper_diams[0]}, True),
    ],
)
def call_hainline24_ty_brown_dwarf_selector_2(request):
    return Hainline24_TY_Brown_Dwarf_Selector_2, *request.param


@pytest.fixture(
    scope="module",
    params=[
        ({"aper_diam": 0.32}, Exception),  # missing units
        ({"aper_diam": -0.32 * u.arcsec}, Exception),  # negative aperture
    ],
)
def fail_hainline24_ty_brown_dwarf_selector_2(request):
    return Hainline24_TY_Brown_Dwarf_Selector_2, *request.param


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
            },
            True,
        ),
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "forced_phot_band": ["F277W", "F356W", "F444W"],
            },
            Exception,
        ),
    ],
)
def call_epochs_unmasked_criteria(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()
    inputs_["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return EPOCHS_unmasked_criteria, inputs_, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32,  # missing units
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
            },
            Exception,
        ),
    ],
)
def fail_epochs_unmasked_criteria(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()
    inputs_["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return EPOCHS_unmasked_criteria, inputs_, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
            },
            Exception,
        ),
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "simulated": True,
            },
            Exception,
        ),
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "forced_phot_band": ["F277W", "F356W", "F444W"],
            },
            Exception,
        ),
    ],
)
def call_epochs_selector(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()
    inputs_["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return EPOCHS_Selector, inputs_, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": 0.32,  # missing units
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
            },
            Exception,
        ),
    ],
)
def fail_epochs_selector(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()
    inputs_["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    return EPOCHS_Selector, inputs_, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "band_names": ["F200W", "F444W"],
                "aper_diam": test_aper_diams[0],
                "SED_fit_label": "eazy_fsps_larson_sed_fitter",
            },
            # F277W is hardcoded internally (colour/hot-pixel checks)
            # but isn't in the local test filterset
            Exception,
        ),
        (
            {
                "band_names": ["F200W", "F444W"],
                "aper_diam": test_aper_diams[0],
                "SED_fit_label": "eazy_fsps_larson_sed_fitter",
                "allow_lowz": True,
            },
            Exception,
        ),
        (
            {
                "band_names": ["F200W", "F444W"],
                "aper_diam": test_aper_diams[0],
                "SED_fit_label": "eazy_fsps_larson_sed_fitter",
                "simulated": True,
            },
            Exception,
        ),
    ],
)
def call_cosmos_web_selector(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()
    inputs_["SED_fit_label"] = request.getfixturevalue(
        inputs["SED_fit_label"]
    )
    return COSMOS_Web_Selector, inputs_, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "band_names": ["F200W", "F444W"],
                "aper_diam": 0.32,  # missing units
                "SED_fit_label": "eazy_fsps_larson_sed_fitter",
            },
            Exception,
        ),
    ],
)
def fail_cosmos_web_selector(request):
    inputs, outcome = request.param
    inputs_ = inputs.copy()
    inputs_["SED_fit_label"] = request.getfixturevalue(
        inputs["SED_fit_label"]
    )
    return COSMOS_Web_Selector, inputs_, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "property_calculator": "uv_beta_calculator",
                "kwarg_name": "rest_UV_wav_lims",
                "kwarg_val": [1_250.0, 3_000.0],
            },
            True,
        ),
    ],
)
def call_rest_frame_property_kwarg_selector(request):
    inputs, outcome = request.param
    inputs["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    inputs["property_calculator"] = request.getfixturevalue(
        inputs["property_calculator"]
    )
    return Rest_Frame_Property_Kwarg_Selector, inputs, outcome


@pytest.fixture(
    scope="module",
    params=[
        (
            {
                "aper_diam": test_aper_diams[0],
                "SED_fitter": "eazy_fsps_larson_sed_fitter",
                "property_calculator": "uv_beta_calculator",
                "kwarg_name": 123,
                "kwarg_val": [1_250.0, 3_000.0],
            },
            Exception,
        ),
    ],
)
def fail_rest_frame_property_kwarg_selector(request):
    inputs, outcome = request.param
    inputs["SED_fitter"] = request.getfixturevalue(inputs["SED_fitter"])
    inputs["property_calculator"] = request.getfixturevalue(
        inputs["property_calculator"]
    )
    return Rest_Frame_Property_Kwarg_Selector, inputs, outcome


##################################################


def get_call_selector_fixtures():
    module = sys.modules[__name__]
    return [
        lf(name)
        for name, obj in inspect.getmembers(module)
        if name.endswith("_selector") and name.startswith("call_")
    ]


@pytest.fixture(scope="module", params=get_call_selector_fixtures())
def call_selector(request):
    return request.param


def get_fail_selector_fixtures():
    module = sys.modules[__name__]
    return [
        lf(name)
        for name, obj in inspect.getmembers(module)
        if name.endswith("_selector") and name.startswith("fail_")
    ]


@pytest.fixture(scope="module", params=get_fail_selector_fixtures())
def fail_selector(request):
    selector_cls, inputs, outcome = request.param
    return selector_cls, inputs, outcome


#################################################


def test_pass_selector_init(call_selector):
    selector_cls, inputs, _ = call_selector
    # instantiate selector_cls with inputs
    selector_inst = selector_cls(**inputs)
    assert isinstance(selector_inst, selector_cls)


def test_fail_selector_init(fail_selector):
    selector_cls, inputs, outcome = fail_selector
    # instantiate selector_cls with inputs
    with pytest.raises(outcome):
        selector_cls(**inputs)


# TODO: Determine expected __call__ failures due to
# objects not containing required information


@pytest.mark.requires_data
def test_selector_call_gal(call_selector, gal):
    selector_cls, inputs, outcome = call_selector
    outcome_ = copy(outcome)
    if isinstance(outcome_, dict):
        outcome_ = outcome_["gal"]
    if outcome_ is not True:
        with pytest.raises(outcome_):
            selector_inst = selector_cls(**inputs)
            selector_inst(gal)
    else:
        selector_inst = selector_cls(**inputs)
        out_gal = selector_inst(gal)
        assert isinstance(out_gal, Galaxy)


@pytest.mark.requires_data
def test_selector_call_cat(call_selector, cat):
    selector_cls, inputs, outcome = call_selector
    outcome_ = copy(outcome)
    if isinstance(outcome_, dict):
        outcome_ = outcome_["cat"]
    if outcome_ is not True:
        with pytest.raises(outcome_):
            selector_inst = selector_cls(**inputs)
            selector_inst(cat)
    else:
        selector_inst = selector_cls(**inputs)
        out_cat = selector_inst(cat)
        assert isinstance(out_cat, Catalogue)


def test_selector_repr_str_and_metadata(call_selector):
    # repr/str/name/select_kwarg_names_dtypes are pure metadata derived
    # only from the selector's own kwargs at construction time, so these
    # don't need a Galaxy/Catalogue to exercise
    selector_cls, inputs, _ = call_selector
    selector_inst = selector_cls(**inputs)
    assert isinstance(repr(selector_inst), str)
    assert isinstance(str(selector_inst), str)
    assert isinstance(selector_inst.name, str)
    names, dtypes = selector_inst.select_kwarg_names_dtypes
    assert isinstance(names, list)
    assert isinstance(dtypes, list)
    assert len(names) == len(dtypes)


def test_shorten_kwarg_colname_short_unchanged():
    assert Selector.shorten_kwarg_colname("short_name") == "short_name"


def test_shorten_kwarg_colname_strips_filter_suffix():
    long_name = "a" * 64 + "F444W"
    assert len(long_name) > 68
    out = Selector.shorten_kwarg_colname(long_name)
    assert "F444W" not in out
    assert len(out) <= 68


def test_shorten_kwarg_colname_still_too_long_raises():
    with pytest.raises(AssertionError):
        Selector.shorten_kwarg_colname("x" * 100)


class TestHighZSyntheticGalaxy:
    """Run selectors' actual runtime logic against the hand-built
    z~9.5 dropout `high_z_gal`, asserting concrete, reasoned selection
    outcomes -- independent of any real imaging/masking/EAZY data."""

    def test_redshift_limit_selector(
        self, high_z_gal, eazy_fsps_larson_sed_fitter
    ):
        selector = Redshift_Limit_Selector(
            test_aper_diams[0],
            eazy_fsps_larson_sed_fitter,
            z_lim=8.5,
            gtr_or_less="gtr",
        )
        out = selector(high_z_gal)
        assert out.selection_flags[selector.name]

    def test_robust_zPDF_selector(
        self, high_z_gal, eazy_fsps_larson_sed_fitter
    ):
        selector = Robust_zPDF_Selector(
            test_aper_diams[0],
            eazy_fsps_larson_sed_fitter,
            integral_lim=0.6,
            dz_over_z=0.1,
        )
        out = selector(high_z_gal)
        assert out.selection_flags[selector.name]

    @pytest.mark.parametrize(
        "z_thr,expected",
        [
            (8.0, True),  # well below the z=9.5 peak
            (10.5, False),  # above the peak, in the PDF's tail
        ],
    )
    def test_zPDF_high_tail_selector(
        self, high_z_gal, eazy_fsps_larson_sed_fitter, z_thr, expected
    ):
        selector = zPDF_High_Tail_Selector(
            test_aper_diams[0],
            eazy_fsps_larson_sed_fitter,
            z_thr=z_thr,
            p_lim=0.8,
        )
        out = selector(high_z_gal)
        assert bool(out.selection_flags[selector.name]) == expected

    def test_chi_sq_lim_selector(
        self, high_z_gal, eazy_fsps_larson_sed_fitter
    ):
        selector = Chi_Sq_Lim_Selector(
            test_aper_diams[0],
            eazy_fsps_larson_sed_fitter,
            chi_sq_lim=3.0,
            reduced=True,
        )
        out = selector(high_z_gal)
        assert out.selection_flags[selector.name]

    def test_chi_sq_diff_selector(
        self, high_z_gal, eazy_fsps_larson_sed_fitter
    ):
        selector = Chi_Sq_Diff_Selector(
            test_aper_diams[0],
            eazy_fsps_larson_sed_fitter,
            chi_sq_diff=4.0,
            dz=0.5,
            lowz_zmax_arr=[4.0, 6.0],
        )
        out = selector(high_z_gal)
        assert out.selection_flags[selector.name]

    def test_bluewards_lya_non_detect_selector(
        self, high_z_gal, eazy_fsps_larson_sed_fitter
    ):
        selector = Bluewards_Lya_Non_Detect_Selector(
            test_aper_diams[0], eazy_fsps_larson_sed_fitter, SNR_lim=2.0
        )
        out = selector(high_z_gal)
        assert out.selection_flags[selector.name]

    def test_redwards_lya_detect_selector(
        self, high_z_gal, eazy_fsps_larson_sed_fitter
    ):
        selector = Redwards_Lya_Detect_Selector(
            test_aper_diams[0],
            eazy_fsps_larson_sed_fitter,
            SNR_lims=5.0,
            widebands_only=False,
        )
        out = selector(high_z_gal)
        assert out.selection_flags[selector.name]

    def test_stacked_blue_lya_non_detect_selector(
        self, high_z_gal, eazy_fsps_larson_sed_fitter
    ):
        selector = Stacked_Blue_Lya_Non_Detect_Selector(
            test_aper_diams[0], eazy_fsps_larson_sed_fitter, SNR_lim=2.0
        )
        out = selector(high_z_gal)
        assert out.selection_flags[selector.name]

    def test_band_snr_selector_non_detect(self, high_z_gal):
        selector = Band_SNR_Selector(
            test_aper_diams[0],
            band="F090W",
            detect_or_non_detect="non_detect",
            SNR_lim=2.0,
        )
        out = selector(high_z_gal)
        assert out.selection_flags[selector.name]

    def test_band_mag_selector_detect(self, high_z_gal):
        selector = Band_Mag_Selector(
            test_aper_diams[0],
            band="F444W",
            detect_or_non_detect="detect",
            mag_lim=30.0,
        )
        out = selector(high_z_gal)
        assert out.selection_flags[selector.name]

    def test_colour_selector(self, high_z_gal):
        selector = Colour_Selector(
            test_aper_diams[0],
            colour_bands=["F277W", "F444W"],
            bluer_or_redder="bluer",
            colour_val=1.0,
        )
        out = selector(high_z_gal)
        assert out.selection_flags[selector.name]

    def test_epochs_selector(self, high_z_gal, eazy_fsps_larson_sed_fitter):
        selector = EPOCHS_Selector(
            test_aper_diams[0], eazy_fsps_larson_sed_fitter
        )
        out = selector(high_z_gal)
        assert out.selection_flags[selector.name]

    def test_cosmos_web_selector(
        self, high_z_gal, eazy_fsps_larson_sed_fitter
    ):
        # COSMOS-Web's stricter Redwards_Lya_Detect_Selector requires
        # SNR > 10 in the first two redwards-of-Lya bands; high_z_gal is
        # built at SNR = 8, so this composite selection is correctly
        # *not* satisfied even though the galaxy is a genuine z~9.5
        # dropout -- demonstrating COSMOS_Web_Selector now runs
        # end-to-end (construction previously raised unconditionally,
        # see call_cosmos_web_selector) without erroring.
        selector = COSMOS_Web_Selector(
            ["F200W", "F444W"],
            test_aper_diams[0],
            eazy_fsps_larson_sed_fitter,
        )
        out = selector(high_z_gal)
        assert not out.selection_flags[selector.name]


class TestSyntheticCatalogue:
    """A 2-source hand-built `Catalogue` (`synthetic_test_cat`: one
    genuine z~9.5 dropout, `high_z_gal`, plus one undetected/poorly-fit
    contaminant, `garbage_gal`) run through selectors per-galaxy. Note
    `synthetic_test_cat` has no real FITS file backing it, so selectors
    are applied to each galaxy directly rather than to the catalogue
    object itself -- Catalogue-level dispatch (`selector(cat)`) needs
    `cat.open_cat()` to persist results, which requires a real FITS
    catalogue (this is also why the existing `call_*_selector`-driven
    `test_selector_call_cat` tests are marked `requires_data`).
    """

    def test_catalogue_has_both_sources(self, synthetic_test_cat):
        assert len(synthetic_test_cat) == 2
        assert [gal.ID for gal in synthetic_test_cat] == [1, 2]

    @pytest.mark.parametrize(
        "gal_fixture,expected",
        [
            (lf("high_z_gal"), True),
            (lf("garbage_gal"), False),
        ],
    )
    def test_epochs_selector_distinguishes_sources(
        self,
        synthetic_test_cat,
        eazy_fsps_larson_sed_fitter,
        gal_fixture,
        expected,
    ):
        # sanity check the fixture galaxy is actually the one held in
        # the catalogue, not just an equivalent copy
        assert any(gal_fixture is gal for gal in synthetic_test_cat)
        selector = EPOCHS_Selector(
            test_aper_diams[0], eazy_fsps_larson_sed_fitter
        )
        out = selector(gal_fixture)
        assert bool(out.selection_flags[selector.name]) == expected

    @pytest.mark.parametrize(
        "gal_fixture,expected",
        [
            (lf("high_z_gal"), True),
            (lf("garbage_gal"), False),
        ],
    )
    def test_robust_zPDF_selector_distinguishes_sources(
        self,
        synthetic_test_cat,
        eazy_fsps_larson_sed_fitter,
        gal_fixture,
        expected,
    ):
        selector = Robust_zPDF_Selector(
            test_aper_diams[0],
            eazy_fsps_larson_sed_fitter,
            integral_lim=0.6,
            dz_over_z=0.1,
        )
        out = selector(gal_fixture)
        assert bool(out.selection_flags[selector.name]) == expected

    @pytest.mark.parametrize(
        "gal_fixture,expected",
        [
            (lf("high_z_gal"), True),
            (lf("garbage_gal"), False),
        ],
    )
    def test_chi_sq_diff_selector_distinguishes_sources(
        self,
        synthetic_test_cat,
        eazy_fsps_larson_sed_fitter,
        gal_fixture,
        expected,
    ):
        # the garbage source's best fit (z~1.3) is already below both
        # candidate lowz_zmax_arr caps (4.0, 6.0), so neither is even
        # an applicable "low-z alternative" to compare against -> the
        # chi-sq-difference dropout test isn't meaningful here and
        # correctly does not select it
        selector = Chi_Sq_Diff_Selector(
            test_aper_diams[0],
            eazy_fsps_larson_sed_fitter,
            chi_sq_diff=4.0,
            dz=0.5,
            lowz_zmax_arr=[4.0, 6.0],
        )
        out = selector(gal_fixture)
        assert bool(out.selection_flags[selector.name]) == expected

    @pytest.mark.parametrize(
        "gal_fixture,expected",
        [
            (lf("high_z_gal"), True),
            (lf("garbage_gal"), False),
        ],
    )
    def test_bluewards_lya_non_detect_selector_distinguishes_sources(
        self,
        synthetic_test_cat,
        eazy_fsps_larson_sed_fitter,
        gal_fixture,
        expected,
    ):
        # the garbage source has no bands redwards of its (low, ~1.3)
        # best-fit redshift's Lyman-alpha, so nothing is bluewards of
        # it either -> not selected
        selector = Bluewards_Lya_Non_Detect_Selector(
            test_aper_diams[0], eazy_fsps_larson_sed_fitter, SNR_lim=2.0
        )
        out = selector(gal_fixture)
        assert bool(out.selection_flags[selector.name]) == expected

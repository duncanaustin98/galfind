
from typing import List
import pytest
import os

os.environ["GALFIND_CONFIG_DIR"] = os.getcwd()
os.environ["GALFIND_CONFIG_NAME"] = "test_galfind_config.ini"

from galfind import Data

from test_config import (
    survey,
    version,
    instrument_names,
)

@pytest.fixture(scope="session")
def data_from_survey_version(
    survey: str,
    version: str,
    instrument_names: List[str],
):
    return Data.from_survey_version(
        survey = survey,
        version = version,
        instrument_names = instrument_names,
        im_str = "test",
        rms_err_ext_name = "RMS_ERR",
    )


def test_data_from_pipeline(
    data_from_survey_version: Data,
    survey: str,
    version: str,
    instrument_names: str,
    aper_diams: List[float],
    forced_phot_band: List[str],
):

    data_from_pipeline = Data.pipeline(
        survey = survey,
        version = version,
        instrument_names = instrument_names,
        im_str = "test",
        rms_err_ext_name = "RMS_ERR",
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
    )
    assert data.survey == data_from_survey_version.survey
    assert data.version == data_from_survey_version.version
    assert data.instrument_names == data_from_survey_version.instrument_names
    assert data.aper_diams == data_from_survey_version.aper_diams
    assert data.forced_phot_band == data_from_survey_version.forced_phot_band
    #assert data_from_pipeline == data_from_survey_version


# @pytest.fixture(scope="session")
# def data_from_pipeline(survey, version, instrument_name):
#     """
#     Retrieve data from the pipeline based on the given survey, version, and instrument name.

#     Args:
#         survey (str): The name of the survey to retrieve data from.
#         version (str): The version of the data to retrieve.
#         instrument_name (str): The name of the instrument(s) used, separated by '+' if multiple.

#     Returns:
#         Data: An instance of the Data class containing the retrieved data.
#     """
#     return Data.from_pipeline(
#         survey=survey,
#         version=version,
#         instrument_names=instrument_name.split("+"),
#     )


# # Ensure the Data.from_pipeline() classmethod works correctly
# def test_data_from_pipeline(galfind_data: tuple[Data, dict[str, Any]]):
#     # Determine expected outputs
#     output_key = (
#         galfind_data.survey,
#         galfind_data.version,
#         galfind_data.instrument.name,
#     )
#     expected = data_test_output[output_key]
#     # Test that the correct bands are loaded
#     band_names = galfind_data.instrument.band_names
#     assert band_names == expected["band_names"]
#     assert (
#         len(galfind_data.im_paths)
#         == len(galfind_data.wht_paths)
#         == len(galfind_data.rms_err_paths)
#         == len(band_names)
#     )
#     assert all(
#         band_name in expected["band_names"]
#         for band_name in galfind_data.im_paths.keys()
#     )
#     assert all(
#         band_name in expected["band_names"]
#         for band_name in galfind_data.wht_paths.keys()
#     )
#     assert all(
#         band_name in expected["band_names"]
#         for band_name in galfind_data.rms_err_paths_paths.keys()
#     )
#     # Test that the length of the sci/wht/rms_err exts are correct
#     assert (
#         len(galfind_data.im_exts)
#         == len(galfind_data.wht_exts)
#         == len(galfind_data.rms_err_exts)
#         == len(band_names)
#     )
#     # Test that the alignment band is correct
#     assert galfind_data.alignment_band == expected["alignment_band"]
#     # Test common directories have been loaded appropriately
#     # Test RGB is created appropriately if asked to plot

#     # # These are tested below
#     # # Test segmentation maps have been loaded appropriately
#     # sex_dir = f"{config['Sextractor']['SEX_DIR']}/{instrument.name}/{version}/{survey}"
#     # path_to_seg = lambda band_name: f"{sex_dir}/{survey}/{survey}_{band_name}_{band_name}_sel_cat_{version}"
#     # assert all(path == path_to_seg(band_name) for band_name, path in galfind_data.seg_paths.items())
#     # # Test mask paths have been loaded appropriately


# def test_data_load_data(galfind_data: tuple[Data, dict[str, Any]]):
#     # Determine expected outputs
#     output_key = (
#         galfind_data.survey,
#         galfind_data.version,
#         galfind_data.instrument.name,
#     )
#     expected = data_test_output[output_key]
#     # Attempt to load data for every band
#     for i, band_name in enumerate(galfind_data.instrument.band_names):
#         # runs Data.load_im, Data.load_seg, and Data.load_mask
#         im_data, im_header, seg_data, seg_header, mask = (
#             galfind_data.load_data(band_name, incl_mask=True)
#         )
#         # Test that the sci shapes/pixel scales/ZPs are correct
#         assert (
#             im_data.shape
#             == galfind_data.im_shapes[band_name]
#             == expected["im_shapes"][i]
#         )
#         assert (
#             Data.load_pix_scale(
#                 im_header, band_name, galfind_data.instrument.name
#             )
#             == galfind_data.im_pixel_scales[band_name]
#             == expected["im_pixel_scales"][i]
#         )
#         assert (
#             Data.load_ZP(im_header, band_name, galfind_data.instrument.name)
#             == galfind_data.im_pixel_scales[band_name]
#             == expected["im_zps"][i]
#         )
#         # Test segmentation map shapes
#         assert seg_data.shape == im_data.shape
#         # Test mask shapes
#         assert mask.shape == im_data.shape

# # Should also test loading data via Data.load_data() for stacked bands
# # runs Data.combine_band_names if not isinstance(band, str)

# # Ensure all Data methods work as advertised
# # Data.__add__
# # Data.__sub__ -> doesn't exist just yet
# # Data.full_name


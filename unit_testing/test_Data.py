from typing import Any

from conftest import data_test_output

from galfind.Data import Data


# Ensure the Data.from_pipeline() classmethod works correctly
def test_data_from_pipeline(galfind_data: tuple[Data, dict[str, Any]]):
    # Determine expected outputs
    output_key = (
        galfind_data.survey,
        galfind_data.version,
        galfind_data.instrument.name,
    )
    expected = data_test_output[output_key]
    # Test that the correct bands are loaded
    band_names = galfind_data.instrument.band_names
    assert band_names == expected["band_names"]
    assert (
        len(galfind_data.im_paths)
        == len(galfind_data.wht_paths)
        == len(galfind_data.rms_err_paths)
        == len(band_names)
    )
    assert all(
        band_name in expected["band_names"]
        for band_name in galfind_data.im_paths.keys()
    )
    assert all(
        band_name in expected["band_names"]
        for band_name in galfind_data.wht_paths.keys()
    )
    assert all(
        band_name in expected["band_names"]
        for band_name in galfind_data.rms_err_paths_paths.keys()
    )
    # Test that the length of the sci/wht/rms_err exts are correct
    assert (
        len(galfind_data.im_exts)
        == len(galfind_data.wht_exts)
        == len(galfind_data.rms_err_exts)
        == len(band_names)
    )
    # Test that the alignment band is correct
    assert galfind_data.alignment_band == expected["alignment_band"]
    # Test common directories have been loaded appropriately
    # Test RGB is created appropriately if asked to plot

    # # These are tested below
    # # Test segmentation maps have been loaded appropriately
    # sex_dir = f"{config['Sextractor']['SEX_DIR']}/{instrument.name}/{version}/{survey}"
    # path_to_seg = lambda band_name: f"{sex_dir}/{survey}/{survey}_{band_name}_{band_name}_sel_cat_{version}"
    # assert all(path == path_to_seg(band_name) for band_name, path in galfind_data.seg_paths.items())
    # # Test mask paths have been loaded appropriately


def test_data_load_data(galfind_data: tuple[Data, dict[str, Any]]):
    # Determine expected outputs
    output_key = (
        galfind_data.survey,
        galfind_data.version,
        galfind_data.instrument.name,
    )
    expected = data_test_output[output_key]
    # Attempt to load data for every band
    for i, band_name in enumerate(galfind_data.instrument.band_names):
        # runs Data.load_im, Data.load_seg, and Data.load_mask
        im_data, im_header, seg_data, seg_header, mask = (
            galfind_data.load_data(band_name, incl_mask=True)
        )
        # Test that the sci shapes/pixel scales/ZPs are correct
        assert (
            im_data.shape
            == galfind_data.im_shapes[band_name]
            == expected["im_shapes"][i]
        )
        assert (
            Data.load_pix_scale(
                im_header, band_name, galfind_data.instrument.name
            )
            == galfind_data.im_pixel_scales[band_name]
            == expected["im_pixel_scales"][i]
        )
        assert (
            Data.load_ZP(im_header, band_name, galfind_data.instrument.name)
            == galfind_data.im_pixel_scales[band_name]
            == expected["im_zps"][i]
        )
        # Test segmentation map shapes
        assert seg_data.shape == im_data.shape
        # Test mask shapes
        assert mask.shape == im_data.shape


# Should also test loading data via Data.load_data() for stacked bands
# runs Data.combine_band_names if not isinstance(band, str)

# Ensure all Data methods work as advertised
# Data.__add__
# Data.__sub__ -> doesn't exist just yet
# Data.full_name

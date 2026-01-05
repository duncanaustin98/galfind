
import pytest

from galfind import SED_code, Catalogue, Multiple_Filter, LePhare

# def test_sed_fitter_init(sed_fitter):
#     assert isinstance(sed_fitter, SED_code)

# def test_lephare_init(lephare_sed_fitter):
#     assert isinstance(lephare_sed_fitter, LePhare)

# @pytest.mark.requires_data
# def test_lephare_compile(lephare_sed_fitter, multi_filter_test_bands):
#     lephare_sed_fitter.compile(multi_filter_test_bands)

# @pytest.mark.requires_data
# def test_cat_lephare_loaded(cat_lephare_loaded):
#     assert isinstance(cat_lephare_loaded, Catalogue)
#     assert len(cat_lephare_loaded) > 0

@pytest.mark.requires_data
def test_cat_custom_lephare_loaded(cat_custom_lephare_loaded):
    assert isinstance(cat_custom_lephare_loaded, Catalogue)

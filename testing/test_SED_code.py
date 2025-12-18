
from galfind import SED_code

# def test_sed_fitter_init(sed_fitter):
#     assert isinstance(sed_fitter, SED_code)

def test_lephare_init(lephare_sed_fitter):
    assert isinstance(lephare_sed_fitter, SED_code)

def test_lephare_compile(lephare_sed_fitter, multi_filter_test_bands):
    lephare_sed_fitter.compile(multi_filter_test_bands)

def test_lephare_call(lephare_sed_fitter, cat, aper_diams):
    lephare_sed_fitter(cat, aper_diams[0])
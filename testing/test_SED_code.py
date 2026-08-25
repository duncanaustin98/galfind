import threading
from copy import deepcopy

import pytest

from galfind.catalogues import Catalogue
from galfind.sed_fitting.Brown_Dwarf_Fitter import Template_Fitter
from galfind.sed_fitting.EAZY import EAZY
from galfind.utils.exceptions import GalfindError, MissingKeyError

SED_FIT_PARAMS = {"templates": "fsps_larson", "lowz_zmax": None}


def test_missing_required_sed_fit_param_raises_missing_key_error():
    with pytest.raises(MissingKeyError, match="templates"):
        EAZY({"lowz_zmax": None})


def test_excl_bands_label_missing_key_raises_missing_key_error():
    eazy = EAZY({**SED_FIT_PARAMS, "excl_bands": [["F444W"]]})
    with pytest.raises(MissingKeyError, match="excl_bands_label"):
        eazy.excl_bands_label


def test_deepcopy_failure_raises_galfind_error():
    eazy = EAZY(SED_FIT_PARAMS)
    eazy.unpicklable_attr = threading.Lock()
    with pytest.raises(GalfindError, match="deepcopy"):
        deepcopy(eazy)


def test_template_fitter_missing_required_key_raises_missing_key_error():
    with pytest.raises(MissingKeyError, match="templates"):
        Template_Fitter({})


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


@pytest.mark.slow
def test_real_eazy_fit_recovers_high_z_dropout(
    synthetic_eazy_cat, eazy_fsps_larson_sed_fitter, aper_diams
):
    """Run the *real* EAZY fitting pipeline (actual templates/binary --
    not a hand-attached fake SED_result) on `synthetic_eazy_cat`, a
    minimal hand-authored FITS catalogue (see `synthetic_eazy_cat_path`
    in conftest.py) holding the same two synthetic sources as
    `high_z_gal`/`garbage_gal`: one genuine z~9.5 Lyman-break dropout
    and one all-noise contaminant. Unlike Selectors/property
    calculators, SED fitting needs a real FITS-backed Catalogue at
    multiple stages (SED_code.__call__ checks/writes results via
    cat.open_cat()), so it can't run on the in-memory-only
    high_z_gal/garbage_gal/synthetic_test_cat fixtures directly.

    Checks that EAZY's own, independently-derived best fit recovers
    the dropout's input redshift with a good fit, and does not find a
    high-z solution for the noise source.
    """
    aper_diam = aper_diams[0]
    out_cat = eazy_fsps_larson_sed_fitter(
        synthetic_eazy_cat, aper_diam, update=True, overwrite=True, fit=True
    )
    results = {}
    for gal in out_cat:
        sed_result = gal.aper_phot[aper_diam].SED_results[
            eazy_fsps_larson_sed_fitter.label
        ]
        z = sed_result.z
        chi_sq = sed_result.chi_sq
        results[gal.ID] = {
            "z": z.value if hasattr(z, "value") else z,
            "chi_sq": chi_sq.value if hasattr(chi_sq, "value") else chi_sq,
        }
    # genuine dropout (ID=1): EAZY should independently recover a
    # redshift close to the true input value (9.5), with a good
    # (low) fit chi-squared
    assert abs(results[1]["z"] - 9.5) < 1.0, results[1]
    assert results[1]["chi_sq"] < 10.0, results[1]
    # pure noise contaminant (ID=2): EAZY should not claim a
    # high-z solution
    assert results[2]["z"] < 5.0, results[2]

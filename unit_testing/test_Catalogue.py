import pytest
import json
import astropy.units as u

from galfind import config
from galfind.Catalogue_Creator import Galfind_Catalogue_Creator
from galfind.Catalogue import Catalogue


def mark_params(all_params, unmarked_params):
    params = [
        pytest.params(_param)
        if _param in unmarked_params
        else pytest.params(_param, marks=pytest.mark.skip)
        for _param in all_params
    ]
    return params


@pytest.fixture(
    scope="session",
    params=mark_params(
        json.loads(config.get("SExtractor", "APER_DIAMS")), [0.32]
    ),
)
def aper_diam(request):
    return request.param * u.arcsec


@pytest.fixture(
    scope="session", params=mark_params(["sex", "loc_depth"], ["loc_depth"])
)
def cat_type(request):
    return request.param


@pytest.fixture(scope="session", params=mark_params(["10", 10.0, 10], [10]))
def min_flux_pc_err(request):
    return request.param


@pytest.fixture(scope="session", params=mark_params(["flux", "mag"], ["flux"]))
def flux_or_mag(request):
    return request.param


@pytest.fixture(scope="session")
def galfind_cat_creator(cat_type, aper_diam, min_flux_pc_err, flux_or_mag):
    # Set up
    cat_creator = Galfind_Catalogue_Creator(
        cat_type, aper_diam, min_flux_pc_err, flux_or_mag
    )
    yield cat_creator
    # Tear down
    del cat_creator


@pytest.fixture(
    scope="session",
    params=mark_params(
        ["F444W", ["F277W", "F356W", "F444W"]], [["F277W", "F356W", "F444W"]]
    ),
)
def forced_phot_band(request):
    return request.param


@pytest.fixture(scope="session")
def galfind_catalogue_from_pipeline(
    temp_galfind_work,
    survey,
    version,
    galfind_cat_creator,
    instrument_name,
    forced_phot_band,
    excl_bands,
    crop_by,
    sex_prefer,
    n_depth_reg,
):
    # Set up
    cat = Catalogue.from_pipeline(
        survey,
        version,
        galfind_cat_creator,
        SED_fit_params_arr=[],
        instrument_name=instrument_name,
        forced_phot_band=forced_phot_band,
        excl_bands=excl_bands,
        crop_by=crop_by,
        load_SED_rest_properties=False,
        sex_prefer=sex_prefer,
        n_depth_reg=n_depth_reg,
        load_ext_src_corrs=False,
    )
    yield cat
    # Tear down
    del cat

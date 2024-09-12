import pytest
import json
import astropy.units as u

from conftest import data_args

from galfind import config
from galfind.Catalogue_Creator import GALFIND_Catalogue_Creator


def mark_params(all_params, marked_params):
    params = [
        pytest.params(_param)
        if _param in marked_params
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
    return GALFIND_Catalogue_Creator(
        cat_type, aper_diam, min_flux_pc_err, flux_or_mag
    )


# @pytest.fixture(scope = "session", params = mark_params())
# def galfind_catalogue_from_pipeline(survey, version, galfind_cat_creator, SED_fit_params_arr):
#     pass

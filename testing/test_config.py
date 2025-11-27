
import pytest
import astropy.units as u

test_galfind_data_dir = "test_data"
test_survey = "test"
test_version = "v0"
test_instrument_names = ["ACS_WFC", "NIRCam"]
test_bands = ["F814W", "F090W", "F200W", "F444W"]
test_aper_diams = [0.32] * u.arcsec
test_forced_phot_band = ["F200W", "F444W"]


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(param, marks=pytest.mark.requires_data)
        for param in [test_survey]
    ],
)
def survey(request):
    return request.param


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(param, marks=pytest.mark.requires_data)
        for param in [test_version]
    ],
)
def version(request):
    return request.param


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(param, marks=pytest.mark.requires_data)
        for param in [test_instrument_names]
    ],
)
def instrument_names(request):
    return request.param


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(param, marks=pytest.mark.requires_data)
        for param in [test_aper_diams]
    ],
)
def aper_diams(request):
    return request.param


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(param, marks=pytest.mark.requires_data)
        for param in [test_forced_phot_band]
    ],
)
def forced_phot_band(request):
    return request.param
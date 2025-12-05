
import pytest
from pytest_lazy_fixtures import lf
import astropy.units as u
import os

os.environ["GALFIND_CONFIG_DIR"] = f"{os.getcwd()}/testing"
os.environ["GALFIND_CONFIG_NAME"] = "test_galfind_config.ini"

import galfind
from galfind import (
    config,
    Facility,
    Instrument,
    JWST,
    NIRCam,
    Filter,
    EAZY,
    Data,
    Band_Data,
    Stacked_Band_Data,
    Catalogue,
)
from galfind.Data import morgan_version_to_dir

test_galfind_data_dir = "test_data"
test_survey = "test"
test_version = "v0"
test_instrument_names = ["ACS_WFC", "NIRCam"]
test_bands_ = ["F814W", "F090W", "F200W", "F444W"]
test_aper_diams = [0.32] * u.arcsec
test_forced_phot_band_ = ["F200W", "F444W"]


@pytest.fixture
def test_bands():
    return test_bands_

@pytest.fixture(scope = "session")
def test_forced_phot_band():
    return test_forced_phot_band_

@pytest.fixture(scope="session")
def survey():
    return test_survey

@pytest.fixture(scope="session")
def version():
    return test_version

@pytest.fixture(scope="session")
def instrument_names():
    return test_instrument_names

@pytest.fixture(scope="session")
def aper_diams():
    return test_aper_diams

@pytest.fixture(scope="session")
def nircam_pix_scale():
    return 0.03 * u.arcsec


def pytest_generate_tests(metafunc):
    slow_marker = metafunc.definition.get_closest_marker("slow")
    if "facility" in metafunc.fixturenames:
        if slow_marker:
            params = [
                pytest.param(param)
                for param in Facility.__subclasses__()
            ]
        else:
            params = [JWST]
        metafunc.parametrize("facility", params, indirect=True)
    elif "instrument" in metafunc.fixturenames:
        if slow_marker:
            params = [
                pytest.param(param)
                for param in Instrument.__subclasses__()
            ]
        else:
            params = [NIRCam]
        metafunc.parametrize("instrument", params, indirect=True)
    elif "filter" in metafunc.fixturenames:
        if slow_marker:
            multi_filter = Multiple_Filter.from_facilities(
                [facility for facility in Facility.__subclasses__()]
            )
            params = list(multi_filter)
        else:
            params = [Filter.from_SVO("JWST", "NIRCam", "F444W")]
        metafunc.parametrize("filter", params, indirect=True)


@pytest.fixture(scope = "session")
def facility(request):
    return request.param

@pytest.fixture(scope = "session")
def facility_inst(facility):
    return facility()

@pytest.fixture(scope = "session")
def instrument(request):
    return request.param

@pytest.fixture(scope = "session")
def instrument_inst(instrument):
    return instrument()

@pytest.fixture(scope = "session")
def filter(request):
    return request.param

@pytest.fixture(scope = "session")
def f444w():
    return Filter.from_SVO("JWST", "NIRCam", "F444W")

# @pytest.fixture(scope = "session")
# def cat():
#     return 

# @pytest.fixture(scope = "session")
# def gal():
#     return 

@pytest.fixture(scope="session")
def eazy_fsps_larson_sed_fitter():
    return EAZY({"templates": "fsps_larson", "lowz_zmax": None})

@pytest.fixture(scope="session")
def eazy_sfhz_sed_fitter():
    return EAZY({"templates": "sfhz", "lowz_zmax": None})

@pytest.fixture(
    scope = "session",
    params = [
        lf(eazy_fsps_larson_sed_fitter),
        lf(eazy_sfhz_sed_fitter),
    ]
)
def sed_fitter(request):
    return request.param

# @pytest.fixture(autouse = True, scope = "session")
# def galfind_work_dir(tmp_path_factory):
#     tmp = tmp_path_factory.mktemp("session_workdir")
#     config.set("DEFAULT", "GALFIND_WORK", str(tmp))
#     # monkeypatch.("GALFIND_WORK_DIR", str(tmpdir))

@pytest.fixture(scope = "session")
def data_dir_nircam(survey, version, nircam_pix_scale):
    return Data._get_data_dir(
        survey,
        version,
        pix_scale = nircam_pix_scale,
        instrument = NIRCam(),
        data_dir = galfind.config["DEFAULT"]["GALFIND_DATA"],
    )

@pytest.fixture(scope="session")
def forced_phot_stacked_band_data_from_arr(
    survey,
    version,
    data_dir_nircam,
    aper_diams,
    test_forced_phot_band
):
    band_data_arr = [
        Band_Data(
            filt = Filter.from_SVO("JWST", "NIRCam", band_name),
            survey = survey,
            version = version,
            im_path = f"{data_dir_nircam}/{band_name}_{survey}.fits",
            im_ext = 1,
            rms_err_path = f"{data_dir_nircam}/{band_name}_{survey}.fits",
            rms_err_ext = 3,
            rms_err_ext_name = "RMS_ERR",
            wht_path = f"{data_dir_nircam}/{band_name}_{survey}.fits",
            wht_ext = 4,
            aper_diams = aper_diams,
        ) for band_name in test_forced_phot_band
    ]
    return Stacked_Band_Data.from_band_data_arr(band_data_arr)

@pytest.fixture(scope="session")
def data(
    survey,
    version,
    instrument_names,
    aper_diams,
    forced_phot_stacked_band_data_from_arr,
):
    return Data.pipeline(
        survey,
        version,
        instrument_names,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_stacked_band_data_from_arr,
        im_str = ["test"],
    )

@pytest.fixture(scope="session")
def cat(data):
    return Catalogue.from_data(data)

@pytest.fixture(scope="session")
def gal(cat):
    return cat[0]
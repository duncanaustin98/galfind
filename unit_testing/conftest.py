import pytest

from galfind import EAZY
from galfind.Data import Data

test_surveys = ["JOF"]
test_versions = ["v11"]
test_instrument_names = ["ACS_WFC+NIRCam"]

SED_fit_params_to_test = [
    {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": 4.0},
    {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": 6.0},
    {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None},
]

data_test_output = {
    ("JOF", "v11", "ACS_WFC+NIRCam"): {
        "band_names": [
            "F435W",
            "F606W",
            "F775W",
            "F814W",
            "F850LP",
            "F090W",
            "F115W",
            "F150W",
            "F162M",
            "F182M",
            "F200W",
            "F210M",
            "F250M",
            "F277W",
            "F300M",
            "F335M",
            "F356W",
            "F410M",
            "F444W",
        ],
        "im_exts": [],
        "im_shapes": [(2000, 3800)],
        "wht_exts": [],
        "rms_err_exts": [],
        "im_pixel_scales": [],
        "im_zps": [],
        "is_blank": True,
        "alignment_band": "F444W",
    }
}


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(param, marks=pytest.mark.requires_data)
        for param in test_surveys
    ],
)
def survey(request):
    return request.param


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(param, marks=pytest.mark.requires_data)
        for param in test_versions
    ],
)
def version(request):
    return request.param


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(param, marks=pytest.mark.requires_data)
        for param in test_instrument_names
    ],
)
def instrument_name(request):
    return request.param


def data_ids(args):
    return f"{args[0]},{args[1]},{args[2]}"


@pytest.fixture(scope="session")
def galfind_data(survey, version, instrument_name):
    return Data.from_pipeline(
        survey=survey,
        version=version,
        instrument_names=instrument_name.split("+"),
    )

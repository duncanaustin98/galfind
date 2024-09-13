import pytest

from galfind import config
from galfind.EAZY import EAZY
from galfind.Bagpipes import Bagpipes
from galfind.Data import Data


@pytest.fixture(
    params=[
        pytest.param(_bool, marks=pytest.mark.requires_data)
        for _bool in [True, False]
    ]
)
def temp_galfind_work(tmp_path, request):
    """
    A pytest fixture that temporarily changes the GALFIND_WORK directory for testing purposes.
    This fixture creates a temporary directory and sets it as the GALFIND_WORK directory in the
    configparser's DEFAULT section. After the test, it resets the GALFIND_WORK directory to its
    original value and removes the temporary directory.
    Args:
        tmp_path (pathlib.Path): A pytest fixture that provides a temporary directory unique to the test invocation.
        request (pytest.FixtureRequest): A pytest fixture that provides information about the requesting test function.
    Yields:
        Any: The boolean parameter passed to the fixture.
    """

    if request.param:
        # Create a temporary directory for testing methods that create files
        temp_dir = tmp_path.mktemp("temp")
        # Note the original GALFIND_WORK directory
        original_galfind_work = config["DEFAULT"]["GALFIND_WORK"]
        # Set as configparser DEFAULT GALFIND_WORK directory
        config.set("DEFAULT", "GALFIND_WORK", temp_dir)
        yield request.param
        # Remove the temporary directory
        temp_dir.rmdir()
        # Reset the GALFIND_WORK directory
        config.set("DEFAULT", "GALFIND_WORK", original_galfind_work)
    else:
        yield request.param


# @pytest.fixture(scope="session")

test_surveys = ["JOF"]
test_versions = ["v11"]
test_instrument_names = ["ACS_WFC+NIRCam"]

SED_fit_params_to_test = [
    {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": 4.0},
    {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": 6.0},
    {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None},
    {
        "code": Bagpipes(),
        "dust": "Cal",
        "dust_prior": "log_10",
        "metallicity_prior": "log_10",
        "sps_model": "BC03",
        "fix_z": False,
        "z_range": (0.0, 25.0),
        "sfh": "continuity_bursty",
    },
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


@pytest.fixture(scope="session")
def galfind_data(survey, version, instrument_name):
    """
    Retrieve data from the pipeline based on the given survey, version, and instrument name.

    Args:
        survey (str): The name of the survey to retrieve data from.
        version (str): The version of the data to retrieve.
        instrument_name (str): The name of the instrument(s) used, separated by '+' if multiple.

    Returns:
        Data: An instance of the Data class containing the retrieved data.
    """
    return Data.from_pipeline(
        survey=survey,
        version=version,
        instrument_names=instrument_name.split("+"),
    )

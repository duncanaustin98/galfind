
import pytest
from copy import copy, deepcopy
from galfind import Instrument, Facility, ACS_WFC, NIRCam, JWST


def test_facility_str(facility_inst):
    s = str(facility_inst)
    assert isinstance(s, str)

def test_facility_repr(facility_inst):
    r = repr(facility_inst)
    assert isinstance(r, str)

def test_facility_singleton(facility_inst):
    # test singleton identity
    shallow = copy(facility_inst)
    assert shallow is facility_inst
    deep = deepcopy(facility_inst)
    assert deep is facility_inst
    facility_inst.test = {}
    facility_inst.test['test'] = 123
    assert shallow.test['test'] == 123
    assert deep.test['test'] == 123

def test_instrument_str(instrument_inst):
    s = str(instrument_inst)
    assert isinstance(s, str)

def test_instrument_repr(instrument_inst):
    r = repr(instrument_inst)
    assert isinstance(r, str)

def test_instrument_singleton(instrument_inst):
    # test singleton identity
    shallow = copy(instrument_inst)
    assert shallow is instrument_inst
    deep = deepcopy(instrument_inst)
    assert deep is instrument_inst
    instrument_inst.align_params['test'] = 123
    assert shallow.align_params['test'] == 123
    assert deep.align_params['test'] == 123


@pytest.fixture(
    scope="module",
    params=[
        (NIRCam, False),
        (NIRCam(), True),
        (ACS_WFC(), False),
        ("NIRCam", False),
        (JWST(), False),
        (123, False),
    ],
)
def nircam_eq_case(request):
    return request.param

@pytest.fixture(scope="module")
def nircam():
    return NIRCam()

def test_instrument_eq(nircam, nircam_eq_case):
    other, expected = nircam_eq_case
    if expected:
        assert nircam == other
    else:
        assert nircam != other


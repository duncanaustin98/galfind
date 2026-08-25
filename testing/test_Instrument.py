from copy import copy, deepcopy

import pytest

from galfind.imaging import NIRCam
from galfind.imaging.Instrument import ACS_SBC, ACS_WFC, IRAC, JWST, VISTA
from galfind.utils.exceptions import (
    InvalidOptionError,
    MissingFileError,
    MissingKeyError,
)


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
    facility_inst.test["test"] = 123
    assert shallow.test["test"] == 123
    assert deep.test["test"] == 123


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
    instrument_inst.align_params["test"] = 123
    assert shallow.align_params["test"] == 123
    assert deep.align_params["test"] == 123


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


class _FakeBandData:
    """Minimal stand-in for `Band_Data`, providing only what `calc_ZP`
    needs (`filt_name` and `load_im`), to test header-validation failure
    paths without constructing a real `Band_Data` (which requires real
    FITS files on disk)."""

    def __init__(self, filt_name, header):
        self.filt_name = filt_name
        self._header = header

    def load_im(self):
        return None, self._header


class TestInstrumentValidation:
    """Focused tests for constructor/kwarg validation failure paths."""

    def test_make_psf_invalid_method(self, nircam):
        with pytest.raises(InvalidOptionError, match="method"):
            nircam.make_psf(band_data=None, method="bogus")

    def test_nircam_get_psf_norm_missing_file(self, nircam, monkeypatch):
        monkeypatch.setattr(
            nircam,
            "get_psf_norm_path",
            lambda band_data: "/nonexistent/psf_norm_file.txt",
        )
        with pytest.raises(MissingFileError, match="not found"):
            nircam.get_psf_norm(band_data=None)

    def test_acs_sbc_calc_zp_missing_keys(self):
        acs_sbc = ACS_SBC()
        fake_band_data = _FakeBandData("F125LP", {})
        with pytest.raises(MissingKeyError, match="ZEROPNT"):
            acs_sbc.calc_ZP(fake_band_data)

    def test_acs_wfc_calc_zp_missing_keys(self):
        acs_wfc = ACS_WFC()
        fake_band_data = _FakeBandData("F606W", {})
        with pytest.raises(MissingKeyError, match="ZEROPNT"):
            acs_wfc.calc_ZP(fake_band_data)

    def test_vista_make_model_psf_not_implemented(self):
        vista = VISTA()
        with pytest.raises(NotImplementedError, match="VISTA"):
            vista.make_model_psf(band_data=None)

    def test_irac_make_model_psf_not_implemented(self):
        irac = IRAC()
        with pytest.raises(NotImplementedError, match="IRAC"):
            irac.make_model_psf(band_data=None)

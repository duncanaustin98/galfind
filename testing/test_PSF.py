"""Tests for `galfind.imaging.PSF` -- focused validation-failure tests for
`PSF_Base`/`PSF_Cutout` that don't require real STPSF calibration data or
band imaging fixtures.
"""

import threading
from copy import deepcopy

import astropy.units as u
import h5py
import numpy as np
import pytest

from galfind.imaging.PSF import PSF_Base, PSF_Cutout
from galfind.utils.exceptions import (
    GalfindError,
    InvalidOptionError,
    MissingFileError,
)


def _make_eec_file(path):
    """Write a minimal, valid encircled-energy-curve HDF5 file."""
    radii = np.linspace(0.03, 2.0, 50)
    eec = np.linspace(0.1, 1.0, 50)
    with h5py.File(path, "w") as f:
        f.create_dataset("radii", data=radii)
        f.create_dataset("eec", data=eec)


class TestPSFBaseInit:
    def test_missing_eec_path_raises(self, tmp_path):
        missing_path = str(tmp_path / "does_not_exist.h5")
        with pytest.raises(MissingFileError, match="eec_path"):
            PSF_Base(missing_path, name="test_psf")

    def test_valid_eec_path_constructs(self, tmp_path):
        eec_path = str(tmp_path / "eec.h5")
        _make_eec_file(eec_path)
        psf = PSF_Base(eec_path, name="test_psf")
        assert psf.eec_path == eec_path
        assert psf.name == "test_psf"


class TestPSFBaseAperCorrs:
    def test_invalid_out_type_raises(self, tmp_path):
        eec_path = str(tmp_path / "eec.h5")
        _make_eec_file(eec_path)
        psf = PSF_Base(eec_path, name="test_psf")
        with pytest.raises(InvalidOptionError, match="out_type"):
            psf.get_aper_corrs(0.32 * u.arcsec, out_type="not_a_type")


class TestPSFBaseDeepcopy:
    def test_uncopyable_attribute_raises_galfind_error(self, tmp_path):
        eec_path = str(tmp_path / "eec.h5")
        _make_eec_file(eec_path)
        psf = PSF_Base(eec_path, name="test_psf")
        # threading.Lock objects cannot be deep-copied and raise
        # TypeError from copy.deepcopy, which __deepcopy__ should
        # convert into a GalfindError rather than silently continuing
        # (dropped via a bare `breakpoint()`).
        psf.bad_attr = threading.Lock()
        with pytest.raises(GalfindError, match="bad_attr"):
            deepcopy(psf)


class TestPSFCutoutFromFits:
    def test_missing_fits_path_raises(self, tmp_path):
        missing_path = str(tmp_path / "does_not_exist.fits")
        # `from_fits` checks `fits_path` exists before touching `filt`,
        # so no Filter fixture is needed to reach the MissingFileError.
        with pytest.raises(MissingFileError, match="fits_path"):
            PSF_Cutout.from_fits(missing_path, filt=None, origin="test")

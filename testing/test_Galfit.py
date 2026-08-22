"""Tests for `galfind.properties.Morphology.Galfit_Fitter` -- the GALFIT
counterpart to `PySersic_Fitter` (`test_PySersic.py`), which this suite
mirrors: constraint/prior-parsing coverage using the same style of
assertions, plus a real end-to-end fit using the shared lightweight
`survey="test", version="v0"` fixture (see root `conftest.py`) rather
than synthetic arrays, since GALFIT operates on real FITS files via an
external binary and has no array-level entry point analogous to
`PySersic_Fitter._fit_array`.
"""

import shutil
from pathlib import Path

import astropy.units as u
import pytest

from galfind.properties.Morphology import Galfit_Fitter

GALFIT_AVAILABLE = shutil.which("galfit") is not None


def _make_fitter(**kwargs):
    kwargs.setdefault("psf", None)
    kwargs.setdefault("model", "sersic")
    return Galfit_Fitter(**kwargs)


class TestLoadConstraints:
    """Mirrors `test_PySersic.py::TestLoadPriors` -- same scalar-fixed /
    `[lo, hi]`-bounded validation semantics, exercised against
    `Galfit_Fitter`'s actual constructor instead of a standalone
    `_load_priors` staticmethod."""

    def test_default_constraints(self):
        fitter = _make_fitter()
        assert fitter.primary_constraints["sersic"]["n"] == [0.01, 10]

    def test_fixed_param_recorded(self):
        fitter = _make_fitter(
            primary_constraints={
                "sersic": {"x": [-2, 2], "y": [-2, 2], "n": 1.0}
            },
        )
        assert fitter.fixed_params["sersic"]["n"] == 1.0

    def test_bad_bound_order_raises(self):
        with pytest.raises(AssertionError):
            _make_fitter(
                primary_constraints={
                    "sersic": {"x": [-2, 2], "y": [-2, 2], "n": [10, 1]}
                },
            )

    def test_bad_bound_length_raises(self):
        with pytest.raises(AssertionError):
            _make_fitter(
                primary_constraints={
                    "sersic": {"x": [-2, 2], "y": [-2, 2], "n": [1, 2, 3]},
                },
            )

    def test_unknown_param_raises(self):
        with pytest.raises(AssertionError):
            _make_fitter(
                primary_constraints={"sersic": {"not_a_param": 1.0}},
            )

    def test_unknown_model_raises(self):
        with pytest.raises(AssertionError):
            _make_fitter(primary_constraints={"not_a_model": {"n": 1.0}})

    def test_fixed_param_overrides_fid_params(self):
        fitter = _make_fitter(
            primary_constraints={
                "sersic": {"x": [-2, 2], "y": [-2, 2], "n": 2.0}
            },
            fid_params={"sersic": {"n": 1, "axr": 1, "pa": 0}},
        )
        assert fitter.fid_params["sersic"]["n"] == 2.0

    def test_fid_params_missing_required_key_raises(self):
        with pytest.raises(AssertionError):
            _make_fitter(fid_params={"sersic": {"n": 1}})  # missing axr, pa

    def test_combined_model_string_splits_on_plus(self):
        # "sersic+psf" validates against _available_models=["sersic","psf"]
        # by splitting on "+" (Morphology_Fitter.__init__) -- same
        # convention pysersic's "sersic_pointsource" name maps onto.
        fitter = _make_fitter(model="sersic+psf")
        assert fitter.model == "sersic+psf"


@pytest.mark.requires_data
@pytest.mark.skipif(
    not GALFIT_AVAILABLE, reason="galfit binary not found on PATH"
)
class TestRealFit:
    """A real, end-to-end GALFIT run (writes FITS/constraint/mask files,
    shells out to the actual `galfit` binary) against the shared
    `survey="test", version="v0"` fixture (`data`/`cat`/`gal` fixtures
    in root `conftest.py`) -- the same lightweight fixture data used
    elsewhere for `requires_data` tests, not the real GS-z14-1 survey
    (that's reserved for the documentation notebooks)."""

    def test_fits_gal(self, gal, cat, tmp_path):
        band_data = cat.data["F444W"]
        psf = band_data.psf
        cat.load_sextractor_params()
        # Safety net only -- `Catalogue.load_sextractor_kron_radii` sets
        # sex_A_IMAGE_AS/sex_B_IMAGE_AS (needed by
        # `Galfit_Fitter._fit_cutout`'s `cutout.meta["A_IMAGE_AS"]`/
        # `"B_IMAGE_AS"`, which `meta` reads back from the cutout FITS
        # file's PRIMARY header) as part of `load_sextractor_params()`
        # above, so this should be a no-op on a fresh Catalogue. Note
        # `sex_A_IMAGE`/`sex_B_IMAGE` are plain scalars (one
        # representative band, see `Galaxy.plot`'s identical usage), not
        # per-band dicts like `sex_KRON_RADIUS`.
        if not hasattr(gal, "sex_A_IMAGE_AS"):
            gal.sex_A_IMAGE_AS = {
                "F444W": gal.sex_KRON_RADIUS["F444W"]
                * gal.sex_A_IMAGE
                * band_data.pix_scale
            }
            gal.sex_B_IMAGE_AS = {
                "F444W": gal.sex_KRON_RADIUS["F444W"]
                * gal.sex_B_IMAGE
                * band_data.pix_scale
            }
        cutout = gal.make_band_cutout(
            band_data,
            cutout_size=1.5 * u.arcsec,
            overwrite=True,
        )
        fitter = _make_fitter(psf=psf, neighbours_model=None)
        # plot=True with cat=None regression-tests a real bug: the
        # auto-plot branch used to call `self._get_neighbours(cutout,
        # in_dir, cat=cat)` unconditionally for the (purely cosmetic)
        # neighbour overlay, crashing with
        # "AttributeError: 'NoneType' object has no attribute 'data'"
        # whenever no Catalogue was supplied -- even though `cat=None`
        # is `_fit_cutout`'s own documented default.
        result = fitter._fit_cutout(
            cutout,
            fid_mag=gal.sex_MAG_AUTO["F444W"],
            fid_re=gal.sex_Re["F444W"],
            in_dir=str(tmp_path / "in") + "/",
            out_dir=str(tmp_path / "out") + "/",
            cat=None,
            plot=True,
        )
        assert result is not None
        assert "n" in result.properties
        assert Path(result.plot_path).is_file()

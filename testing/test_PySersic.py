"""Tests for `galfind.properties.PySersic` -- straightforward pysersic
prior customization (fixed/bounded primary and neighbour priors), the
three model options ("sersic", "pointsource", "sersic_pointsource"), and
simultaneous neighbour fitting.

These are unit tests against `PySersic_Fitter._fit_array`/`_load_priors`
directly, using synthetic numpy arrays -- no galfind `Catalogue`/
`Galaxy`/`PSF_Base` objects are needed, since building those for a real
morphology fit is heavy (see the existing note on this in
`test_Selectors.py`).
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pysersic = pytest.importorskip("pysersic")

import astropy.units as u
from _synthetic_morphology import (
    make_pointsource_stamp,
    make_two_source_stamp,
)
from _synthetic_morphology import (
    make_sersic_stamp as _make_sersic_image,
)

from galfind.properties.PySersic import PySersic_Fitter


def _make_fitter(
    model="sersic",
    primary_priors=None,
    neighbour_priors=None,
    neighbours_model=None,
    method="laplace",
):
    return PySersic_Fitter(
        psf=None,
        model=model,
        primary_priors=primary_priors,
        neighbour_priors=neighbour_priors,
        neighbours_model=neighbours_model,
        method=method,
    )


class TestLoadPriors:
    def test_no_priors(self):
        fixed, bounded = PySersic_Fitter._load_priors(None, "sersic")
        assert fixed == {} and bounded == {}

    def test_fixed_and_bounded(self):
        fixed, bounded = PySersic_Fitter._load_priors(
            {"sersic": {"n": 1.0, "r_eff": [0.5, 10.0]}}, "sersic"
        )
        assert fixed == {"n": 1.0}
        assert bounded == {"r_eff": (0.5, 10.0)}

    def test_unknown_param_raises(self):
        with pytest.raises(AssertionError):
            PySersic_Fitter._load_priors(
                {"sersic": {"not_a_param": 1.0}}, "sersic"
            )

    def test_unknown_model_raises(self):
        with pytest.raises(AssertionError):
            PySersic_Fitter._load_priors(
                {"not_a_model": {"n": 1.0}}, "not_a_model"
            )

    def test_bad_bound_order_raises(self):
        with pytest.raises(AssertionError):
            PySersic_Fitter._load_priors(
                {"sersic": {"r_eff": [10.0, 0.5]}}, "sersic"
            )

    def test_bad_bound_length_raises(self):
        with pytest.raises(AssertionError):
            PySersic_Fitter._load_priors(
                {"sersic": {"r_eff": [0.5, 1.0, 2.0]}}, "sersic"
            )

    def test_bad_value_type_raises(self):
        with pytest.raises(TypeError):
            PySersic_Fitter._load_priors(
                {"sersic": {"n": "not a number"}}, "sersic"
            )


class TestFitArray:
    def test_free_sersic_recovers_size(self):
        sci, rms, mask, psf = _make_sersic_image(scale=3.0)
        fitter = _make_fitter(model="sersic")
        map_params, _ = fitter._fit_array(sci, rms, mask, psf)
        # b_1(n=1) ~= 1.678, so r_eff ~= 1.678 * scale ~= 5.0 pix
        assert 3.0 < map_params["r_eff"][1] < 7.5

    def test_fixed_n_is_exact(self):
        sci, rms, mask, psf = _make_sersic_image(scale=3.0)
        fitter = _make_fitter(
            model="sersic",
            primary_priors={"sersic": {"n": 1.0}},
        )
        map_params, _ = fitter._fit_array(sci, rms, mask, psf)
        n_lo, n_med, n_hi = map_params["n"]
        assert n_lo == n_med == n_hi == pytest.approx(1.0)

    def test_bounded_r_eff_caps_the_fit(self):
        # deliberately broad, low-contrast blob that an unbounded fit
        # would run away to a much larger r_eff on
        sci, rms, mask, psf = _make_sersic_image(
            ny=61,
            nx=61,
            amp=1.0,
            scale=15.0,
            noise=0.05,
            seed=1,
        )
        fitter = _make_fitter(
            model="sersic",
            primary_priors={"sersic": {"r_eff": [0.5, 10.0]}},
        )
        map_params, _ = fitter._fit_array(sci, rms, mask, psf)
        assert map_params["r_eff"][1] <= 10.0 + 1e-6
        assert map_params["r_eff"][1] > 9.0  # pinned against the cap

    def test_pointsource_model(self):
        sci, rms, mask, psf = make_pointsource_stamp(amp=10.0)
        fitter = _make_fitter(model="pointsource")
        map_params, _ = fitter._fit_array(sci, rms, mask, psf)
        assert map_params["flux"][1] == pytest.approx(10.0, rel=0.5)
        assert "r_eff" not in map_params

    def test_sersic_pointsource_model(self):
        sci, rms, mask, psf = _make_sersic_image(scale=3.0, seed=3)
        ny, nx = sci.shape
        sci[ny // 2, nx // 2] += 5.0
        fitter = _make_fitter(model="sersic_pointsource")
        map_params, _ = fitter._fit_array(sci, rms, mask, psf)
        assert "f_ps" in map_params
        assert 0.0 <= map_params["f_ps"][1] <= 1.0

    def test_neighbour_fit_improves_primary(self):
        true_primary_scale = 3.0
        (
            sci,
            rms,
            mask,
            psf,
            (primary_x, primary_y),
            (neighbour_x, neighbour_y),
        ) = make_two_source_stamp(primary_scale=true_primary_scale)

        single_fitter = _make_fitter(model="sersic")
        single_map_params, _ = single_fitter._fit_array(sci, rms, mask, psf)

        multi_fitter = _make_fitter(model="sersic", neighbours_model="sersic")
        neighbour_catalog = {
            "x": np.array([primary_x, neighbour_x]),
            "y": np.array([primary_y, neighbour_y]),
            "flux": np.array([100.0, 60.0]),
            "r": np.array([3.0, 2.5]),
            "type": ["sersic", "sersic"],
        }
        multi_map_params, _ = multi_fitter._fit_array(
            sci,
            rms,
            mask,
            psf,
            neighbour_catalog=neighbour_catalog,
        )

        true_r_eff = 1.678 * true_primary_scale
        single_err = abs(single_map_params["r_eff"][1] - true_r_eff)
        multi_err = abs(multi_map_params["r_eff_0"][1] - true_r_eff)
        assert multi_err < single_err


@pytest.mark.requires_data
class TestRealFit:
    """A real end-to-end `PySersic_Fitter._fit_cutout` run against the
    shared `survey="test", version="v0"` fixture (`data`/`cat`/`gal`
    fixtures in root `conftest.py`), mirroring
    `test_Galfit.py::TestRealFit`. Also checks that a second
    `_fit_cutout` call against the same `out_dir` reloads the cached
    `.h5` fit instead of re-running `_fit_array`, and that the
    residual/model/image and corner plots actually get written to
    disk."""

    def test_fits_gal_and_caches(self, gal, cat, tmp_path):
        band_data = cat.data["F444W"]
        psf = band_data.psf
        cutout = gal.make_band_cutout(
            band_data,
            cutout_size=1.5 * u.arcsec,
            overwrite=True,
        )
        fitter = PySersic_Fitter(
            psf=psf,
            model="sersic",
            primary_priors={"sersic": {"n": 1.0}},
            method="laplace",
        )
        out_dir = str(tmp_path)

        with patch.object(
            PySersic_Fitter, "_fit_array", wraps=fitter._fit_array
        ) as mocked_fit_array:
            result = fitter._fit_cutout(
                cutout,
                cat=None,
                plot=True,
                out_dir=out_dir,
            )
            assert result is not None
            assert "n" in result.properties
            assert result.properties["n"].value == pytest.approx(1.0)
            assert mocked_fit_array.call_count == 1

            png_files = list(Path(out_dir).rglob("*.png"))
            assert any("residual" in p.name for p in png_files)
            assert any("corner" in p.name for p in png_files)

            # a second fit against the same out_dir should reload the
            # cached .h5 rather than re-running the fit
            result2 = fitter._fit_cutout(
                cutout,
                cat=None,
                plot=False,
                out_dir=out_dir,
            )
            assert mocked_fit_array.call_count == 1
            assert result2.properties["r_eff"].value == pytest.approx(
                result.properties["r_eff"].value
            )

"""Lightweight validation-failure tests for `galfind.visualization.Cutout`.

These deliberately avoid needing real FITS/imaging data: `Band_Cutout_Base`
and `RGB_Base` are concrete (all of `Cutout_Base`'s abstract methods are
implemented on the base classes themselves), so they can be constructed
directly with small stand-in objects to exercise their input validation.
"""

from types import SimpleNamespace

import pytest

from galfind.utils.exceptions import (
    InvalidOptionError,
    LengthMismatchError,
    MissingFileError,
    MissingKeyError,
)
from galfind.visualization.Cutout import Band_Cutout_Base, RGB_Base


class _FakeBandCutout:
    """Minimal stand-in exposing only the `band_data.filt_name` /
    `band_data.filt.filt_name` attributes that `RGB_Base` needs for its
    constructor-time validation and `get_colour_filt_names`."""

    def __init__(self, filt_name: str):
        self.band_data = SimpleNamespace(
            filt_name=filt_name,
            filt=SimpleNamespace(filt_name=filt_name),
        )


def _rgb_cutouts(b="F090W", g="F150W", r="F200W"):
    return {
        "B": [_FakeBandCutout(b)],
        "G": [_FakeBandCutout(g)],
        "R": [_FakeBandCutout(r)],
    }


def test_band_cutout_base_missing_cutout_path_raises(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.fits")
    with pytest.raises(MissingFileError):
        Band_Cutout_Base(missing_path, band_data=None, cutout_size=None)


def test_rgb_base_missing_bgr_key_raises():
    cutouts = {
        "B": [_FakeBandCutout("F090W")],
        "G": [_FakeBandCutout("F150W")],
    }
    with pytest.raises(MissingKeyError):
        RGB_Base(cutouts)


def test_rgb_base_duplicate_filter_across_channels_raises():
    cutouts = _rgb_cutouts(b="F090W", g="F090W", r="F200W")
    with pytest.raises(LengthMismatchError):
        RGB_Base(cutouts)


def test_rgb_base_constructs_with_distinct_filters():
    rgb = RGB_Base(_rgb_cutouts())
    assert rgb.get_colour_filt_names("B") == ["F090W"]


def test_get_colour_filt_names_invalid_colour_raises():
    rgb = RGB_Base(_rgb_cutouts())
    with pytest.raises(InvalidOptionError):
        rgb.get_colour_filt_names("X")

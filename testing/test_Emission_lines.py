import astropy.units as u
import numpy as np
import pytest

from galfind.spectra.Emission_lines import Emission_line, line_diagnostics
from galfind.utils.exceptions import InvalidOptionError


@pytest.fixture(scope="module")
def lya_line():
    return Emission_line(
        "Lya", 1e-17 * u.erg / u.s / u.cm**2, Doppler_b=100.0 * u.km / u.s
    )


@pytest.fixture(scope="module")
def civ_line():
    # CIV-1549 has rel_lambda=None and no oscillator_strength entry
    return Emission_line(
        "CIV-1549",
        1e-17 * u.erg / u.s / u.cm**2,
        Doppler_b=100.0 * u.km / u.s,
    )


def test_emission_line_repr(lya_line):
    assert repr(lya_line) == (
        f"Emission_Line(Lya, {line_diagnostics['Lya']['line_wav']})"
    )


def test_emission_line_str_contains_expected_fields(lya_line):
    out = str(lya_line)
    assert "Lya" in out
    assert "Rest wavelength" in out
    assert "Doppler parameter" in out
    assert "Oscillator strength" in out


def test_emission_line_str_omits_oscillator_strength_when_undefined(
    civ_line,
):
    assert "Oscillator strength" not in str(civ_line)


def test_delta_lambda_positive_for_nonzero_doppler_b(lya_line):
    assert lya_line.delta_lambda.unit.is_equivalent(u.AA)
    assert lya_line.delta_lambda.to(u.AA).value > 0.0


def test_delta_lambda_zero_for_zero_doppler_b():
    line = Emission_line("Lya", 1e-17 * u.erg / u.s / u.cm**2)
    assert line.delta_lambda.value == 0.0


def test_R_zero_when_rel_lambda_undefined(civ_line):
    assert civ_line.R == 0.0


def test_R_positive_when_rel_lambda_defined(lya_line):
    assert lya_line.R > 0.0


def test_a_zero_when_rel_lambda_undefined(civ_line):
    assert civ_line.a == 0.0


def test_a_infinite_for_zero_doppler_b():
    # Lya has rel_lambda defined, but a divides by delta_lambda,
    # which is zero for the default Doppler_b
    line = Emission_line("Lya", 1e-17 * u.erg / u.s / u.cm**2)
    assert np.isinf(line.a)


def test_line_profile_normalized_to_line_flux(lya_line):
    profile = lya_line.line_profile
    integrated = np.trapezoid(profile["flux"], profile["wavs"])
    assert integrated.to(lya_line.line_flux.unit).value == pytest.approx(
        lya_line.line_flux.value, rel=1e-6
    )


def test_line_profile_unsupported_voigt_type_raises():
    line = Emission_line(
        "Lya",
        1e-17 * u.erg / u.s / u.cm**2,
        Doppler_b=100.0 * u.km / u.s,
        voigt_type="unsupported",
    )
    with pytest.raises(InvalidOptionError, match="unsupported"):
        line.line_profile


def test_line_width_within_feature_window(lya_line):
    feature_wavs = line_diagnostics["Lya"]["feature_wavs"]
    width = lya_line.line_width(lim=1e-4)
    assert len(width) > 0
    assert np.all(width >= feature_wavs[0])
    assert np.all(width <= feature_wavs[1])


def test_line_width_shrinks_for_higher_threshold(lya_line):
    narrow = lya_line.line_width(lim=0.5)
    wide = lya_line.line_width(lim=1e-4)
    assert len(narrow) <= len(wide)

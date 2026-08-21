import astropy.units as u
import numpy as np
import pytest

from galfind.utils.lyman_alpha_damping_wing import (
    DLA_damping_param,
    Doppler_parameter,
    Tepper_Garcia06_lyman_alpha_voigt_profile,
    Tepper_Garcia06_voigt_profile,
    bg_HI_density,
    delta_lambda_lyman_alpha_from_b,
    delta_lambda_lyman_alpha_from_gas_temp,
    full_voigt_profile,
    get_transmission,
    integral_result,
    tau_DW,
    tau_GP,
    tau_proximate_DLA,
)


def test_integral_result_matches_reference():
    assert integral_result(0.5) == pytest.approx(4.49986859486237, rel=1e-8)
    assert integral_result(0.1) == pytest.approx(1.66716389860176, rel=1e-8)


def test_integral_result_array_input():
    result = integral_result(np.array([0.1, 0.5]))
    np.testing.assert_allclose(
        result, [1.66716389860176, 4.49986859486237], rtol=1e-8
    )


def test_bg_HI_density_scales_with_neutral_fraction():
    full = bg_HI_density(6.0, 1.0, 0.245)
    half = bg_HI_density(6.0, 0.5, 0.245)
    assert half.unit.physical_type == "number density"
    assert half.to(u.cm**-3).value == pytest.approx(
        0.5 * full.to(u.cm**-3).value, rel=1e-8
    )


def test_bg_HI_density_zero_when_neutral_fraction_zero():
    density = bg_HI_density(6.0, 0.0, 0.245)
    assert density.to(u.cm**-3).value == 0.0


def test_bg_HI_density_matches_reference():
    density = bg_HI_density(6.0, 1.0, 0.245)
    assert density.to(u.cm**-3).value == pytest.approx(
        7.121106675813832e-05, rel=1e-6
    )


def test_bg_HI_density_increases_with_redshift():
    low_z = bg_HI_density(0.0, 1.0, 0.245).to(u.cm**-3).value
    high_z = bg_HI_density(6.0, 1.0, 0.245).to(u.cm**-3).value
    assert high_z > low_z


def test_tau_GP_zero_when_neutral_fraction_zero():
    assert tau_GP(6.0, 0.0, 0.245).to(u.dimensionless_unscaled).value == 0.0


def test_tau_GP_matches_reference():
    tau = tau_GP(6.0, 1.0, 0.245)
    assert tau.unit.physical_type == "dimensionless"
    assert tau.to(u.dimensionless_unscaled).value == pytest.approx(
        413725.0696350503, rel=1e-6
    )


def test_tau_DW_positive_and_finite():
    tau = tau_DW(
        1220.0 * u.AA,
        z_gal=8.0,
        R_b=1.0 * u.Mpc,
        x_HI=0.5,
        helium_mass_frac=0.245,
    )
    assert np.isfinite(tau.to(u.dimensionless_unscaled).value)
    assert tau.to(u.dimensionless_unscaled).value > 0.0


def test_tau_DW_zero_when_neutral_fraction_zero():
    tau = tau_DW(
        1220.0 * u.AA,
        z_gal=8.0,
        R_b=1.0 * u.Mpc,
        x_HI=0.0,
        helium_mass_frac=0.245,
    )
    assert tau.to(u.dimensionless_unscaled).value == pytest.approx(
        0.0, abs=1e-10
    )


def test_Doppler_parameter_positive_velocity():
    b = Doppler_parameter(1e4 * u.K)
    assert b.unit.physical_type == "speed"
    assert b.to(u.km / u.s).value == pytest.approx(12.84865731940276, rel=1e-6)


def test_Doppler_parameter_increases_with_temperature():
    b_cool = Doppler_parameter(1e3 * u.K).to(u.km / u.s).value
    b_hot = Doppler_parameter(1e5 * u.K).to(u.km / u.s).value
    assert b_hot > b_cool


def test_delta_lambda_from_b_matches_from_gas_temp():
    b = Doppler_parameter(1e4 * u.K)
    from_b = delta_lambda_lyman_alpha_from_b(b)
    from_temp = delta_lambda_lyman_alpha_from_gas_temp(1e4 * u.K)
    assert from_b.to(u.AA).value == pytest.approx(
        from_temp.to(u.AA).value, rel=1e-10
    )
    assert from_b.to(u.AA).value == pytest.approx(
        0.05210180185212782, rel=1e-6
    )


def test_DLA_damping_param_dimensionless_and_positive():
    delta_lambda = delta_lambda_lyman_alpha_from_gas_temp(1e4 * u.K)
    a = DLA_damping_param(delta_lambda)
    assert a.unit.physical_type == "dimensionless"
    assert a.to(u.dimensionless_unscaled).value == pytest.approx(
        0.0004714544484044383, rel=1e-6
    )


def test_DLA_damping_param_decreases_with_larger_delta_lambda():
    small_delta = delta_lambda_lyman_alpha_from_gas_temp(1e3 * u.K)
    large_delta = delta_lambda_lyman_alpha_from_gas_temp(1e5 * u.K)
    a_small = DLA_damping_param(small_delta).value
    a_large = DLA_damping_param(large_delta).value
    assert a_large < a_small


def test_full_voigt_profile_matches_reference():
    assert full_voigt_profile(0.0, 1.0, 1.0) == pytest.approx(
        0.22455546962575995, rel=1e-6
    )


def test_full_voigt_profile_peaks_at_line_center():
    center = full_voigt_profile(0.0, 1.0, 0.1)
    offset = full_voigt_profile(2.0, 1.0, 0.1)
    assert center > offset


def test_Tepper_Garcia06_voigt_profile_matches_reference():
    result = Tepper_Garcia06_voigt_profile(0.001, 1e-6)
    assert result == pytest.approx(0.999999999999, rel=1e-8)


def test_Tepper_Garcia06_voigt_profile_decreases_with_offset():
    near = Tepper_Garcia06_voigt_profile(0.001, 0.1)
    far = Tepper_Garcia06_voigt_profile(0.001, 3.0)
    assert near > far


def test_Tepper_Garcia06_lyman_alpha_voigt_profile_matches_reference():
    delta_lambda = delta_lambda_lyman_alpha_from_gas_temp(1e4 * u.K)
    profile = Tepper_Garcia06_lyman_alpha_voigt_profile(
        1220.0 * u.AA, delta_lambda
    )
    assert profile.value == pytest.approx(3.85202240666328e-08, rel=1e-6)


def test_tau_proximate_DLA_matches_reference():
    delta_lambda = delta_lambda_lyman_alpha_from_gas_temp(1e4 * u.K)
    tau = tau_proximate_DLA(1220.0 * u.AA, 1e20 / u.cm**2, delta_lambda)
    assert tau.to(u.dimensionless_unscaled).value == pytest.approx(
        0.22713092213429464, rel=1e-6
    )


def test_tau_proximate_DLA_positive():
    delta_lambda = delta_lambda_lyman_alpha_from_gas_temp(1e4 * u.K)
    tau = tau_proximate_DLA(1220.0 * u.AA, 1e20 / u.cm**2, delta_lambda)
    assert tau.to(u.dimensionless_unscaled).value > 0.0


def test_get_transmission_zero_tau_gives_unit_transmission():
    assert get_transmission([0.0, 0.0]) == 1.0


def test_get_transmission_matches_reference():
    assert get_transmission([1.0, 2.0]) == pytest.approx(
        0.049787068367863944, rel=1e-8
    )


def test_get_transmission_sums_tau_before_exponentiating():
    combined = get_transmission([0.5, 0.5])
    single = get_transmission([1.0])
    assert combined == pytest.approx(single, rel=1e-10)

import astropy.units as u
import numpy as np
import pytest

from galfind.spectra.DLA import DLA


@pytest.fixture
def dla():
    return DLA(N_HI=1e20 / u.cm**2, Doppler_b=30 * u.km / u.s)


def test_init_stores_attributes():
    dla = DLA(
        N_HI=1e21 / u.cm**2,
        Doppler_b=50 * u.km / u.s,
        vel_offset=10 * u.km / u.s,
        z_offset=0.5,
        voigt_method="Tepper-Garcia+06",
    )
    assert dla.N_HI == 1e21 / u.cm**2
    assert dla.Doppler_b == 50 * u.km / u.s
    assert dla.vel_offset == 10 * u.km / u.s
    assert dla.z_offset == 0.5
    assert dla.voigt_method == "Tepper-Garcia+06"


def test_init_defaults():
    dla = DLA(N_HI=1e20 / u.cm**2, Doppler_b=30 * u.km / u.s)
    assert dla.vel_offset == 0.0 * u.km / u.s
    assert dla.z_offset == 0.0
    assert dla.voigt_method == "Tepper-Garcia+06"


def test_delta_lambda_positive_and_scales_with_doppler_b():
    dla_slow = DLA(N_HI=1e20 / u.cm**2, Doppler_b=10 * u.km / u.s)
    dla_fast = DLA(N_HI=1e20 / u.cm**2, Doppler_b=30 * u.km / u.s)
    assert dla_slow.delta_lambda.to(u.AA).value > 0.0
    assert dla_fast.delta_lambda.to(u.AA).value == pytest.approx(
        3.0 * dla_slow.delta_lambda.to(u.AA).value, rel=1e-8
    )


def test_delta_lambda_zero_for_zero_doppler_b():
    dla = DLA(N_HI=1e20 / u.cm**2, Doppler_b=0 * u.km / u.s)
    assert dla.delta_lambda.to(u.AA).value == 0.0


def test_photon_absorption_const_matches_reference(dla):
    assert dla.photon_absorption_const.to(u.cm**2).value == pytest.approx(
        1.2506226996250567e-10, rel=1e-6
    )


def test_a_positive_and_matches_reference(dla):
    assert dla.a.unit.physical_type == "dimensionless"
    assert dla.a.value == pytest.approx(0.00020192851151738336, rel=1e-6)


def test_a_decreases_with_larger_doppler_b():
    dla_slow = DLA(N_HI=1e20 / u.cm**2, Doppler_b=10 * u.km / u.s)
    dla_fast = DLA(N_HI=1e20 / u.cm**2, Doppler_b=30 * u.km / u.s)
    assert dla_fast.a.value < dla_slow.a.value


def test_z_vel_offset_zero_when_z_offset_zero(dla):
    assert dla.z_vel_offset.to(u.km / u.s).value == 0.0


def test_z_vel_offset_positive_for_positive_z_offset():
    dla = DLA(N_HI=1e20 / u.cm**2, Doppler_b=30 * u.km / u.s, z_offset=0.001)
    assert dla.z_vel_offset.to(u.km / u.s).value == pytest.approx(
        299.6425618458279, rel=1e-6
    )


def test_tau_near_line_center_matches_reference(dla):
    wav_rest = 1220.0 * u.AA
    tau = dla.tau(wav_rest)
    assert tau.unit.physical_type == "dimensionless"
    assert tau.value == pytest.approx(0.22736167731405535, rel=1e-6)


def test_tau_far_from_line_center_is_small(dla):
    wav_rest = 1300.0 * u.AA
    tau = dla.tau(wav_rest)
    assert tau.value == pytest.approx(0.0005987097299041641, rel=1e-6)


def test_tau_decreases_away_from_line_center(dla):
    tau_near = dla.tau(1220.0 * u.AA).value
    tau_far = dla.tau(1300.0 * u.AA).value
    assert tau_near > tau_far


def test_transmission_between_zero_and_one(dla):
    transmission = dla.transmission(1220.0 * u.AA).value
    assert 0.0 < transmission < 1.0
    assert transmission == pytest.approx(0.7966326062385976, rel=1e-6)


def test_transmission_approaches_one_far_from_line(dla):
    transmission = dla.transmission(1300.0 * u.AA).value
    assert transmission == pytest.approx(0.9994014694610033, rel=1e-6)


def test_transmission_equals_exp_minus_tau(dla):
    wav_rest = 1220.0 * u.AA
    assert dla.transmission(wav_rest).value == pytest.approx(
        np.exp(-dla.tau(wav_rest).value), rel=1e-10
    )


def test_transmission_lower_for_larger_column_density():
    wav_rest = 1220.0 * u.AA
    weak = DLA(N_HI=1e20 / u.cm**2, Doppler_b=30 * u.km / u.s)
    strong = DLA(N_HI=1e22 / u.cm**2, Doppler_b=30 * u.km / u.s)
    assert strong.transmission(wav_rest) < weak.transmission(wav_rest)


def test_plot_transmission_profile_plots_on_axes(dla):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    wav_rest = np.linspace(1210.0, 1230.0, 20) * u.AA
    dla.plot_transmission_profile(ax, wav_rest)
    assert len(ax.lines) == 1
    plotted_y = ax.lines[0].get_ydata()
    np.testing.assert_allclose(plotted_y, dla.transmission(wav_rest).value)
    plt.close(fig)


def test_plot_voigt_profile_plots_on_axes(dla):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    wav_rest = np.linspace(1210.0, 1230.0, 20) * u.AA
    dla.plot_voigt_profile(ax, wav_rest)
    assert len(ax.lines) == 1
    plt.close(fig)

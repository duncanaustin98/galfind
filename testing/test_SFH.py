import sys
import types

import astropy.units as u
import numpy as np
import pytest

from galfind.SFH import SFH


@pytest.fixture
def ages():
    return np.linspace(1e6, 1e9, 50) * u.yr


@pytest.fixture
def sfh_post(ages):
    rng = np.random.default_rng(42)
    return rng.uniform(0.1, 10.0, size=(20, len(ages)))


@pytest.fixture
def sfh(ages, sfh_post):
    return SFH(z=6.0, ages=ages, sfh_post=sfh_post)


@pytest.fixture
def fake_bagpipes(monkeypatch):
    """Install a minimal stand-in `bagpipes` module.

    `SFH.age_of_universe` and the z-axis plotting code import the real
    `bagpipes` package, which is not installed in the test environment.
    This fixture provides the two attributes those code paths need
    (`bagpipes.utils.z_array` / `age_at_z`) so the surrounding GALFIND
    logic can be exercised without depending on bagpipes itself.
    """
    module = types.ModuleType("bagpipes")
    module.utils = types.SimpleNamespace(
        z_array=np.linspace(0.0, 30.0, 1000),
        age_at_z=np.linspace(13.8, 0.05, 1000),
    )
    monkeypatch.setitem(sys.modules, "bagpipes", module)
    return module


def test_init_stores_attributes(ages, sfh_post):
    sfh = SFH(z=7.5, ages=ages, sfh_post=sfh_post, type="delayed")
    assert sfh.z == 7.5
    assert sfh.ages is ages
    assert sfh.sfh_post is sfh_post
    assert sfh.type == "delayed"


def test_init_default_type(ages, sfh_post):
    sfh = SFH(z=6.0, ages=ages, sfh_post=sfh_post)
    assert sfh.type == "continuity_bursty"


def test_repr(sfh):
    assert repr(sfh) == "SFH(z=6.00, type=continuity_bursty)"


def test_age_of_universe_interpolates_bagpipes_grid(sfh, fake_bagpipes):
    expected = (
        1.0e9
        * np.interp(
            sfh.z,
            fake_bagpipes.utils.z_array,
            fake_bagpipes.utils.age_at_z,
        )
        * u.yr
    )
    assert sfh.age_of_universe.to(u.yr).value == pytest.approx(
        expected.to(u.yr).value, rel=1e-8
    )


def test_plot_invalid_plot_type_raises(sfh):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        sfh.plot(ax, plot_type="bogus")
    plt.close(fig)


def test_plot_lookback_draws_median_line_and_envelope(sfh):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    result_ax, z_ax, zoom_ax = sfh.plot(
        ax, plot_type="lookback", annotate=False
    )
    assert result_ax is ax
    assert z_ax is None
    assert zoom_ax is None
    assert len(ax.lines) == 1
    assert len(ax.collections) == 1  # fill_between envelope
    plotted_median = ax.lines[0].get_ydata()
    expected_median = np.percentile(sfh.sfh_post, 50, axis=0)
    np.testing.assert_allclose(plotted_median, expected_median, rtol=1e-8)
    plt.close(fig)


def test_plot_lookback_sets_xlim_and_labels(sfh, fake_bagpipes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    sfh.plot(ax, plot_type="lookback")
    assert ax.get_xlim()[0] == 0
    assert "Lookback Time" in ax.get_xlabel()
    plt.close(fig)


def test_plot_lookback_annotates_redshift_text(sfh, fake_bagpipes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    sfh.plot(ax, plot_type="lookback", label_z=True)
    texts = [t.get_text() for t in ax.texts]
    assert any("6.00" in text for text in texts)
    plt.close(fig)


def test_plot_absolute_sets_xlim_and_redshift_axis(sfh, fake_bagpipes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    result_ax, z_ax, zoom_ax = sfh.plot(ax, plot_type="absolute")
    assert z_ax is not None
    assert zoom_ax is None
    assert "Age of Universe" in ax.get_xlabel()
    assert "Redshift" in z_ax.get_xlabel()
    plt.close(fig)


def test_plot_absolute_without_z_axis_returns_none(sfh, fake_bagpipes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    result_ax, z_ax, zoom_ax = sfh.plot(ax, plot_type="absolute", z_axis=False)
    assert z_ax is None
    plt.close(fig)


def test_plot_lookback_without_annotate_skips_labels(sfh):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    sfh.plot(ax, plot_type="lookback", annotate=False)
    assert ax.get_xlabel() == ""
    plt.close(fig)


def test_plot_crop_ages_restricts_plotted_range(sfh):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    sfh.plot(
        ax,
        plot_type="lookback",
        annotate=False,
        crop_ages=[0.0, 5.0e2] * u.Myr,
    )
    plotted_x = ax.lines[0].get_xdata()
    assert len(plotted_x) < len(sfh.ages)
    assert np.all(plotted_x >= 0.0)
    assert np.all(plotted_x <= 5.0e2)
    plt.close(fig)


def test_plot_crop_ages_wrong_length_raises(sfh):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    with pytest.raises(Exception):
        sfh.plot(ax, plot_type="lookback", crop_ages=[0.0] * u.Myr)
    plt.close(fig)


def test_plot_crop_ages_wrong_unit_raises(sfh):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    with pytest.raises(Exception):
        sfh.plot(ax, plot_type="lookback", crop_ages=[0.0, 1.0] * u.AA)
    plt.close(fig)


def test_plot_save_true_raises_not_implemented(sfh):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    with pytest.raises(NotImplementedError):
        sfh.plot(ax, plot_type="lookback", annotate=False, save=True)
    plt.close(fig)


def test_plot_mismatched_sfh_post_shape_raises(ages):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bad_sfh_post = np.ones((5, len(ages) - 1))
    sfh = SFH(z=6.0, ages=ages, sfh_post=bad_sfh_post)
    fig, ax = plt.subplots()
    with pytest.raises(Exception):
        sfh.plot(ax, plot_type="lookback")
    plt.close(fig)


def test_plot_custom_colours_used(sfh):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    sfh.plot(
        ax,
        plot_type="lookback",
        annotate=False,
        primary_colour="red",
        secondary_colour="blue",
    )
    assert ax.lines[0].get_color() == "red"
    plt.close(fig)


def test_plot_zoom_time_creates_inset_axes(sfh, fake_bagpipes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    result_ax, z_ax, zoom_ax = sfh.plot(
        ax, plot_type="lookback", zoom_time=500.0 * u.Myr
    )
    assert zoom_ax is not None
    assert zoom_ax.get_xlim()[1] == pytest.approx(500.0)
    plt.close(fig)

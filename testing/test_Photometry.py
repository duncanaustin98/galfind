import astropy.units as u
import numpy as np
import pytest

from galfind.photometry.Photometry import Mock_Photometry, Photometry
from galfind.photometry.Photometry_obs import Photometry_obs
from galfind.photometry.Photometry_rest import Photometry_rest


@pytest.mark.requires_data
def test_phot_rest(phot_rest):
    assert isinstance(phot_rest, Photometry_rest)


def test_blank_phot_rest(blank_phot_rest):
    assert isinstance(blank_phot_rest, Photometry_rest)
    assert len(blank_phot_rest) == 0


@pytest.fixture(scope="module")
def simple_flux(test_bands):
    return np.linspace(1.0, 4.0, len(test_bands)) * u.uJy


@pytest.fixture(scope="module")
def simple_flux_errs(test_bands):
    return np.linspace(0.1, 0.4, len(test_bands)) * u.uJy


@pytest.fixture(scope="module")
def simple_depths(test_bands):
    return np.linspace(27.0, 28.5, len(test_bands)) * u.ABmag


@pytest.fixture(scope="module")
def simple_phot(
    multi_filter_test_bands, simple_flux, simple_flux_errs, simple_depths
):
    return Photometry(
        multi_filter_test_bands, simple_flux, simple_flux_errs, simple_depths
    )


class TestPhotometry:
    """Simplest-possible construction and behaviour checks for the
    `Photometry` base class, none of which need any imaging/catalogue
    data on disk."""

    def test_construction(self, simple_phot, test_bands):
        assert isinstance(simple_phot, Photometry)
        assert len(simple_phot) == len(test_bands)

    def test_repr(self, simple_phot, test_bands):
        assert repr(simple_phot) == f"Photometry({len(test_bands)} bands)"

    def test_wav(self, simple_phot, test_bands):
        wav = simple_phot.wav
        assert wav.unit == u.AA
        assert len(wav) == len(test_bands)

    def test_depths_from_dict(
        self,
        multi_filter_test_bands,
        simple_flux,
        simple_flux_errs,
        test_bands,
    ):
        depths_dict = {
            band: (27.0 + i * 0.1) * u.ABmag
            for i, band in enumerate(test_bands)
        }
        phot = Photometry(
            multi_filter_test_bands,
            simple_flux,
            simple_flux_errs,
            depths_dict,
        )
        assert phot.depths.unit == u.ABmag
        assert len(phot.depths) == len(test_bands)

    def test_depths_dict_missing_band_raises(
        self,
        multi_filter_test_bands,
        simple_flux,
        simple_flux_errs,
        test_bands,
    ):
        depths_dict = {band: 27.0 * u.ABmag for band in test_bands[:-1]}
        with pytest.raises(AssertionError):
            Photometry(
                multi_filter_test_bands,
                simple_flux,
                simple_flux_errs,
                depths_dict,
            )

    def test_depths_wrong_units_raises(
        self,
        multi_filter_test_bands,
        simple_flux,
        simple_flux_errs,
        test_bands,
    ):
        bad_depths = np.linspace(27.0, 28.5, len(test_bands)) * u.mag
        with pytest.raises(AssertionError):
            Photometry(
                multi_filter_test_bands,
                simple_flux,
                simple_flux_errs,
                bad_depths,
            )

    def test_mismatched_length_raises(
        self, multi_filter_test_bands, simple_flux_errs, simple_depths
    ):
        bad_flux = np.array([1.0, 2.0]) * u.uJy
        with pytest.raises(AssertionError):
            Photometry(
                multi_filter_test_bands,
                bad_flux,
                simple_flux_errs,
                simple_depths,
            )

    def test_getitem_int(self, simple_phot):
        single = simple_phot[0]
        assert isinstance(single, Photometry)
        assert len(single) == 1

    def test_getitem_str(self, simple_phot, test_bands):
        single = simple_phot[test_bands[0]]
        assert len(single) == 1
        assert single.filterset.filt_names == [test_bands[0]]

    def test_scatter_fluxes_shape(self, simple_phot, test_bands):
        np.random.seed(0)
        scattered = simple_phot.scatter_fluxes(n_scatter=3)
        assert scattered.shape == (3, len(test_bands))

    def test_scatter_single(self, simple_phot):
        np.random.seed(0)
        scattered_phot = simple_phot.scatter(n_scatter=1)
        assert isinstance(scattered_phot, Photometry)
        assert len(scattered_phot) == len(simple_phot)


class TestMockPhotometry:
    """Simplest-possible construction checks for `Mock_Photometry`."""

    def test_construction(
        self, multi_filter_test_bands, simple_flux, simple_depths
    ):
        mock = Mock_Photometry(
            multi_filter_test_bands,
            simple_flux,
            simple_depths,
            min_flux_pc_err=10.0,
        )
        assert isinstance(mock, Mock_Photometry)
        assert isinstance(mock, Photometry)
        assert mock.min_flux_pc_err == 10.0
        assert len(mock.flux_errs) == len(mock)

    def test_length_mismatch_raises(
        self, multi_filter_test_bands, simple_flux
    ):
        bad_depths = np.array([27.0, 28.0]) * u.ABmag
        with pytest.raises(AssertionError):
            Mock_Photometry(
                multi_filter_test_bands,
                simple_flux,
                bad_depths,
                min_flux_pc_err=10.0,
            )


class TestPhotometryObs:
    """Simplest-possible construction checks for `Photometry_obs`."""

    def test_construction(
        self,
        multi_filter_test_bands,
        simple_flux,
        simple_flux_errs,
        simple_depths,
        aper_diams,
    ):
        phot_obs = Photometry_obs(
            multi_filter_test_bands,
            simple_flux,
            simple_flux_errs,
            simple_depths,
            aper_diams[0],
        )
        assert isinstance(phot_obs, Photometry_obs)
        assert isinstance(phot_obs, Photometry)
        assert phot_obs.aper_diam == aper_diams[0]
        assert phot_obs.psfs is None
        assert phot_obs.SED_results == {}

    def test_repr_contains_aper_diam(
        self,
        multi_filter_test_bands,
        simple_flux,
        simple_flux_errs,
        simple_depths,
        aper_diams,
    ):
        phot_obs = Photometry_obs(
            multi_filter_test_bands,
            simple_flux,
            simple_flux_errs,
            simple_depths,
            aper_diams[0],
        )
        assert "Photometry_obs" in repr(phot_obs)
        assert str(aper_diams[0]) in repr(phot_obs)


class TestPhotometryRestFromPhot:
    """`Photometry_rest.from_phot` needs only a plain `Photometry` and a
    redshift, no fitted catalogue data."""

    def test_from_phot(self, simple_phot):
        phot_rest = Photometry_rest.from_phot(simple_phot, z=5.0)
        assert isinstance(phot_rest, Photometry_rest)
        assert phot_rest.z == 5.0
        assert len(phot_rest) == len(simple_phot)

    def test_repr(self, simple_phot):
        phot_rest = Photometry_rest.from_phot(simple_phot, z=5.0)
        assert repr(phot_rest) == (
            f"Photometry_rest({simple_phot.filterset.instrument_name}, "
            f"z=5.000)"
        )

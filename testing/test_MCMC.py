import pytest

from galfind.utils.exceptions import (
    GalfindError,
    InvalidOptionError,
    LengthMismatchError,
    MissingKeyError,
    RangeError,
)
from galfind.utils.MCMC import (
    Flat_Prior,
    Gaussian_Prior,
    Linear_Fitter,
    Priors,
)


def test_flat_prior_wrong_number_of_limits_raises_length_mismatch():
    with pytest.raises(LengthMismatchError, match="exactly two limits"):
        Flat_Prior("z", [0.0, 1.0, 2.0], fiducial=0.5)


def test_flat_prior_descending_limits_raises_range_error():
    with pytest.raises(RangeError, match="ascending order"):
        Flat_Prior("z", [5.0, 1.0], fiducial=2.0)


def test_gaussian_prior_missing_mu_sigma_raises_missing_key():
    with pytest.raises(MissingKeyError, match="'mu' and 'sigma'"):
        Gaussian_Prior("z", {"lower_lim": 0.0, "upper_lim": 1.0}, fiducial=0.5)


def test_priors_duplicate_name_raises_galfind_error():
    p1 = Flat_Prior("m", [0.0, 1.0], fiducial=0.5)
    p2 = Flat_Prior("m", [0.0, 1.0], fiducial=0.5)
    with pytest.raises(GalfindError, match="unique"):
        p1 + p2


def test_priors_getitem_missing_name_raises_missing_key():
    priors = Priors([Flat_Prior("m", [0.0, 1.0], fiducial=0.5)])
    with pytest.raises(MissingKeyError, match="not found"):
        priors["nonexistent"]


def test_linear_fitter_unrecognised_fixed_param_raises_invalid_option():
    priors = Priors(
        [
            Flat_Prior("m", [0.0, 1.0], fiducial=0.5),
            Flat_Prior("c", [0.0, 1.0], fiducial=0.5),
        ]
    )
    with pytest.raises(InvalidOptionError, match="bogus_param"):
        Linear_Fitter(
            priors,
            None,
            None,
            None,
            nwalkers=10,
            fixed_params={"bogus_param": 1.0},
        )

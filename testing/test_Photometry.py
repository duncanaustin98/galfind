
import pytest

from galfind import Photometry_rest

@pytest.mark.requires_data
def test_phot_rest(phot_rest):
    assert isinstance(phot_rest, Photometry_rest)

def test_blank_phot_rest(blank_phot_rest):
    assert isinstance(blank_phot_rest, Photometry_rest)
    assert len(blank_phot_rest) == 0
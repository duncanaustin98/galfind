
import pytest

from galfind import (
    Catalogue
)

def test_cat_from_data(cat):
    assert isinstance(cat, Catalogue)

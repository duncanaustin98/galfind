
import pytest

from galfind import (
    Catalogue,
    Catalogue_Creator,
)

@pytest.mark.requires_data
def test_cat_from_data(cat):
    assert isinstance(cat, Catalogue)

@pytest.mark.requires_data
def test_id_cropped_cat_creator_from_data(cat_creator_id_cropped):
    assert isinstance(cat_creator_id_cropped, Catalogue_Creator)

@pytest.mark.requires_data
def test_id_cropped_cat_creator_from_data_call(cat_creator_id_cropped):
    cat = cat_creator_id_cropped()
    assert isinstance(cat, Catalogue)

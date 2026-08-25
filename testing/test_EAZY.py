import pytest

from galfind.sed_fitting.EAZY import EAZY
from galfind.utils.exceptions import (
    GalfindTypeError,
    IncompatibleKwargsError,
    LengthMismatchError,
    MissingFileError,
)

SED_FIT_PARAMS = {"templates": "fsps_larson", "lowz_zmax": None}


def test_extract_seds_mismatched_lengths_raises_length_mismatch():
    eazy = EAZY(SED_FIT_PARAMS)
    with pytest.raises(LengthMismatchError, match="len\\(IDs\\)"):
        eazy.extract_SEDs([1, 2], ["a.h5"])


def test_extract_seds_inconsistent_paths_raises_incompatible_kwargs():
    eazy = EAZY(SED_FIT_PARAMS)
    with pytest.raises(IncompatibleKwargsError, match="SED_paths"):
        eazy.extract_SEDs([1, 2], ["a.h5", "b.h5"])


def test_extract_pdfs_non_list_paths_raises_type_error():
    eazy = EAZY(SED_FIT_PARAMS)
    with pytest.raises(GalfindTypeError, match="PDF_paths"):
        eazy.extract_PDFs("z", [1, 2], {"a.h5": 1})


def test_extract_pdfs_non_list_ids_raises_type_error():
    eazy = EAZY(SED_FIT_PARAMS)
    with pytest.raises(GalfindTypeError, match="IDs"):
        eazy.extract_PDFs("z", {"not": "a_list"}, ["a.h5"])


def test_extract_pdfs_mismatched_lengths_raises_length_mismatch():
    eazy = EAZY(SED_FIT_PARAMS)
    with pytest.raises(LengthMismatchError, match="len\\(IDs\\)"):
        eazy.extract_PDFs("z", [1, 2], ["a.h5"])


def test_extract_pdfs_inconsistent_paths_raises_incompatible_kwargs():
    eazy = EAZY(SED_FIT_PARAMS)
    with pytest.raises(IncompatibleKwargsError, match="PDF_paths"):
        eazy.extract_PDFs("z", [1, 2], ["a.h5", "b.h5"])


def test_extract_pdfs_wrong_extension_raises_missing_file_error():
    eazy = EAZY(SED_FIT_PARAMS)
    with pytest.raises(MissingFileError, match="\\.h5"):
        eazy.extract_PDFs("z", [1, 2], ["a.spec", "a.spec"])

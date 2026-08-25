import pytest

from galfind.sed_fitting.LePhare import LePhare
from galfind.utils.exceptions import (
    GalfindTypeError,
    IncompatibleKwargsError,
    InvalidOptionError,
    LengthMismatchError,
)

GAL_TEMPLATES = "BC03_Chabrier2003_Zm42m62"


def test_lephare_init_filterset_not_multiple_filter_raises_type_error():
    with pytest.raises(GalfindTypeError, match="Multiple_Filter"):
        LePhare(
            {"GAL_TEMPLATES": GAL_TEMPLATES},
            filterset="not_a_filterset",
        )


def test_lephare_compile_without_filterset_raises_incompatible_kwargs():
    lephare = LePhare({"GAL_TEMPLATES": GAL_TEMPLATES})
    with pytest.raises(IncompatibleKwargsError, match="filterset"):
        lephare.compile()


def test_get_lib_name_invalid_type_raises_invalid_option():
    lephare = LePhare({"GAL_TEMPLATES": GAL_TEMPLATES})
    with pytest.raises(InvalidOptionError, match="STAR.*QSO.*GAL"):
        lephare.get_lib_name("BOGUS")


def test_extract_seds_mismatched_lengths_raises_length_mismatch():
    lephare = LePhare({"GAL_TEMPLATES": GAL_TEMPLATES})
    with pytest.raises(LengthMismatchError, match="len\\(IDs\\)"):
        lephare.extract_SEDs([1, 2], ["a.spec"])


def test_sed_fit_params_from_label_bad_format_raises_length_mismatch():
    lephare = LePhare({"GAL_TEMPLATES": GAL_TEMPLATES})
    with pytest.raises(LengthMismatchError, match="'_'-separated parts"):
        lephare.SED_fit_params_from_label("too_many_underscore_parts")


def test_extract_pdfs_non_list_paths_raises_type_error():
    lephare = LePhare({"GAL_TEMPLATES": GAL_TEMPLATES})
    with pytest.raises(GalfindTypeError, match="PDF_paths"):
        lephare.extract_PDFs("z", [1, 2], {"a.spec": 1})


def test_extract_pdfs_mismatched_lengths_raises_length_mismatch():
    lephare = LePhare({"GAL_TEMPLATES": GAL_TEMPLATES})
    with pytest.raises(LengthMismatchError, match="len\\(IDs\\)"):
        lephare.extract_PDFs("z", [1, 2], ["a.spec"])

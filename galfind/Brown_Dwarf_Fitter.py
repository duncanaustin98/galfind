
from __future__ import annotations

from BDFit import StarFit
import astropy.units as u
from typing import TYPE_CHECKING, Dict, List, NoReturn
if TYPE_CHECKING:
    from . import Catalogue
try:
    from typing import Self, Type, Any  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import SED_code

class Template_Fitter(SED_code):

    def __init__(
        self: Self,
        SED_fit_params: Dict[str, Any],
    ):
        super().__init__(SED_fit_params)

    @classmethod
    def from_label(cls, label: str) -> Type[SED_code]:
        pass

    @property
    def ID_label(self) -> str:
        return "ID"

    @property
    def label(self) -> str:
        return f"{self.__class__.__name__}_{self.SED_fit_params['templates']}"

    @property
    def hdu_name(self) -> str:
        return f"{self.__class__.__name__}_{self.SED_fit_params['templates']}"

    @property
    def tab_suffix(self) -> str:
        return self.SED_fit_params["templates"]

    @property
    def required_SED_fit_params(self) -> List[str]:
        return ["templates"]

    @property
    def are_errs_percentiles(self) -> bool:
        return False # not sure here

    # BELOW TWO METHODS SHOULD BE IN PARENT CLASS

    #@abstractmethod
    def _load_gal_property_labels(self, gal_property_labels: Dict[str, str]) -> NoReturn:
        self.gal_property_labels = {key: f"{item}_{self.tab_suffix}" 
            for key, item in gal_property_labels.items()}

    #@abstractmethod
    def _load_gal_property_err_labels(self, gal_property_err_labels: Dict[str, List[str, str]]) -> NoReturn:
        self.gal_property_err_labels = {key: [f"{item[0]}_{self.tab_suffix}", f"{item[1]}_{self.tab_suffix}"]
            for key, item in gal_property_err_labels.items()}

    def _load_gal_property_units(self) -> NoReturn:
        pass

    def make_in(
        self, 
        cat: Catalogue, 
        aper_diam: u.Quantity, 
        overwrite: bool = False
    ) -> str:
        pass

    def fit(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity,
        save_SEDs: bool = True,
        save_PDFs: bool = True,
        overwrite: bool = False,
        **kwargs: Dict[str, Any],
    ) -> NoReturn:
        # sort out templates
        StarFit(libraries = self.SED_fit_params["templates"])
        pass

    def make_fits_from_out(self, out_path):
        pass

    def _get_out_paths(
        self: Self, 
        cat: Catalogue, 
        aper_diam: u.Quantity
    ) -> Tuple[str, str, str, Dict[str, List[str]], List[str]]:
        pass

    def extract_SEDs(
        self: Self, 
        IDs: List[int], 
        SED_paths: Union[str, List[str]]
    ) -> List[SED_obs]:
        pass

    def extract_PDFs(
        self: Self, 
        gal_property: str, 
        IDs: List[int], 
        PDF_paths: Union[str, List[str]], 
    ) -> List[Type[PDF]]:
        pass

    def load_cat_property_PDFs(
        self: Self, 
        PDF_paths: Union[List[str], List[Dict[str, str]]],
        IDs: List[int]
    ) -> List[Dict[str, Optional[Type[PDF]]]]:
        pass

    def _assert_SED_fit_params(self) -> NoReturn:
        for key in self.required_SED_fit_params:
            assert key in self.SED_fit_params.keys(), galfind_logger.critical(
                f"'{key}' not in SED_fit_params keys = {list(self.SED_fit_params.keys())}"
            )

    pass


class Brown_Dwarf_Fitter(Template_Fitter):

    pass
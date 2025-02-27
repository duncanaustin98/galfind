
from __future__ import annotations

from BDFit import StarFit
import numpy as np
import astropy.units as u
from astropy.table import Table
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, NoReturn, Union
if TYPE_CHECKING:
    from . import Catalogue, SED_obs
try:
    from typing import Self, Type, Any  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import useful_funcs_austind as funcs
from . import config, galfind_logger
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
        return f"{self.__class__.__name__}_{self.tab_suffix}"

    @property
    def hdu_name(self) -> str:
        return f"{self.__class__.__name__}_{self.tab_suffix}"

    @property
    def tab_suffix(self) -> str:
        if len(self.SED_fit_params["templates"]) > 1:
            return "+".join(self.SED_fit_params["templates"])
        else:
            return self.SED_fit_params["templates"][0]

    @property
    def required_SED_fit_params(self) -> List[str]:
        return ["templates"]

    @property
    def are_errs_percentiles(self) -> bool:
        return False # not sure here

    # BELOW TWO METHODS SHOULD BE HADNLED BETTER IN SED_code

    #@abstractmethod
    def _load_gal_property_labels(self) -> NoReturn:
        super()._load_gal_property_labels({})

    #@abstractmethod
    def _load_gal_property_err_labels(self) -> NoReturn:
        super()._load_gal_property_err_labels({})

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
        out_path = self._get_out_paths(cat, aper_diam)[1]
        if not Path(out_path).is_file() or overwrite:
            # convert cat.filterset to bands used in StarFit
            facilities_to_search = {}
            for filt in cat.filterset:
                if filt.facility_name not in facilities_to_search.keys():
                    facilities_to_search[filt.facility_name] = []
                facilities_to_search[filt.facility_name].append(filt.instrument.SVO_name)
            bands = [
                f"{filt.instrument_name}.{filt.band_name}" 
                if filt.instrument_name != "NIRCam" else filt.band_name 
                for filt in cat.filterset
            ]
            starfit = StarFit(
                facilities_to_search = facilities_to_search,
                libraries = self.SED_fit_params["templates"],
                compile_bands = bands,
            )
            starfit.fit_catalog(
                photometry_function = self._load_phot,
                bands = bands,
                photometry_function_kwargs = {
                    "cat": cat,
                    "aper_diam": aper_diam,
                    "out_units": u.nJy,
                    "no_data_val": np.nan,
                    "incl_units": True,
                },
                sys_err = None, 
                filter_mask = None,
                subset = None,
            )
            self.starfit = starfit
            tab = starfit.make_cat()
            tab["ID"] = np.array(cat.ID)
            # save as a .fits file
            funcs.make_dirs(out_path)
            tab.write(out_path, overwrite = overwrite)
            galfind_logger.info(f"Saved {self.tab_suffix} fits to {out_path}")
            # save the best fitting SEDs
            SED_save_path = out_path.replace(".fits", "_SEDs")
            starfit.save_best_fit_SEDs(SED_save_path)

    def make_fits_from_out(self, out_path):
        pass

    def _get_out_paths(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity
    ) -> Tuple[str, str, str, Dict[str, List[str]], List[str]]:
        aper_diams_str = funcs.aper_diams_to_str(np.array([aper_diam.to(u.arcsec).value]) * u.arcsec)
        out_path = f"{config['TemplateFitting']['BROWN_DWARF_OUT_DIR']}/{cat.version}/" + \
            f"{cat.filterset.instrument_name}/{cat.survey}/{aper_diams_str}/{self.hdu_name}.fits"
        return None, out_path, out_path, None, out_path

    def extract_SEDs(
        self: Self, 
        IDs: List[int], 
        SED_paths: str,
    ) -> List[SED_obs]:
        # open .fits table
        tab = Table.read(SED_paths)
        # extract SEDs

        pass

    def extract_PDFs(
        self: Self, 
        gal_property: str, 
        IDs: List[int], 
        PDF_paths: Union[str, List[str]], 
    ) -> Optional[List[Type[PDF]]]:
        return None

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


class Brown_Dwarf_Fitter(Template_Fitter):

    def __init__(self: Self):
        SED_fit_params = {"templates": ["sonora_bobcat", "sonora_cholla", "sonora_elf_owl", "sonora_diamondback", "low-z"]}
        super().__init__(SED_fit_params)

    @property
    def label(self) -> str:
        return self.__class__.__name__

    @property
    def hdu_name(self) -> str:
        return self.__class__.__name__

    @property
    def tab_suffix(self) -> str:
        return "brown_dwarf"
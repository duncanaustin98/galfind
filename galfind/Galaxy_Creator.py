from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.table import Table
from numpy.typing import NDArray
from typing import Union, Tuple, Any, List, Dict, Callable, Optional, NoReturn, TYPE_CHECKING
if TYPE_CHECKING:
    from . import (
        Filter,
        Band_Data_Base,
        Selector,
        Multiple_Filter,
        Data,
        Property_Calculator_Base,
        Band_Cutout_Base,
        Band_Cutout,
        Region_Selector,
        Mask_Selector,
        PSF_Base,
    )
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import galfind_logger
from . import Catalogue_Creator, Catalogue
from .Catalogue import (
    open_galfind_cat,
    open_galfind_hdr,
    load_IDs_Table,
    load_skycoords_Table,
    load_galfind_phot,
    galfind_phot_labels,
    load_galfind_mask,
    galfind_mask_labels,
    load_galfind_depths,
    galfind_depth_labels,
    load_bool_Table,
    galfind_selection_labels,
)

class Galaxy_Creator(Catalogue_Creator):

    def __init__(
        self: Self,
        survey: str,
        version: str,
        id: int,
        cat_path: str,
        filterset: Multiple_Filter,
        aper_diams: u.Quantity,
        open_cat: Callable[[str, str], Any] = open_galfind_cat,
        open_hdr: Callable[[Any], Dict[str, str]] = open_galfind_hdr,
        load_ID_func: Optional[Callable] = load_IDs_Table,
        ID_label: str = "NUMBER",
        load_ID_kwargs: Dict[str, Any] = {},
        load_skycoords_func: Optional[Callable] = load_skycoords_Table,
        skycoords_labels: Dict[str, str] = {"RA": "ALPHA_J2000", "DEC": "DELTA_J2000"},
        skycoords_units: Dict[str, u.Unit] = {"RA": u.deg, "DEC": u.deg},
        load_skycoords_kwargs: Dict[str, Any] = {},
        load_phot_func: Callable = load_galfind_phot,
        get_phot_labels: Callable[[Multiple_Filter], Dict[str, str]] = galfind_phot_labels,
        load_phot_kwargs: Dict[str, Any] = {"ZP": u.Jy.to(u.ABmag), "min_flux_pc_err": 10.},
        load_mask_func: Optional[Callable] = load_galfind_mask,
        get_mask_labels: Callable[[Multiple_Filter], Dict[str, str]] = galfind_mask_labels,
        load_mask_kwargs: Dict[str, Any] = {},
        load_depth_func: Optional[Callable] = load_galfind_depths,
        get_depth_labels: Callable[[Multiple_Filter], Dict[str, str]] = galfind_depth_labels,
        load_depth_kwargs: Dict[str, Any] = {},
        load_selection_func: Optional[Callable[[], Dict[u.Quantity, Dict[str, List[Any]]]]] = load_bool_Table,
        get_selection_labels: Callable[[Table, List[str]], List[str]] = galfind_selection_labels,
        load_selection_kwargs: Dict[str, Any] = {},
        load_SED_result_func: Optional[Callable] = None,
        apply_gal_instr_mask: bool = True,
        simulated: bool = False,
    ) -> None:
        self.id = id
        super().__init__(
            survey=survey,
            version=version,
            cat_path=cat_path,
            filterset=filterset,
            aper_diams=aper_diams,
            crops=None,
            open_cat=open_cat,
            open_hdr=open_hdr,
            load_ID_func=load_ID_func,
            ID_label=ID_label,
            load_ID_kwargs=load_ID_kwargs,
            load_skycoords_func=load_skycoords_func,
            skycoords_labels=skycoords_labels,
            skycoords_units=skycoords_units,
            load_skycoords_kwargs=load_skycoords_kwargs,
            load_phot_func=load_phot_func,
            get_phot_labels=get_phot_labels,
            load_phot_kwargs=load_phot_kwargs,
            load_mask_func=load_mask_func,
            get_mask_labels=get_mask_labels,
            load_mask_kwargs=load_mask_kwargs,
            load_depth_func=load_depth_func,
            get_depth_labels=get_depth_labels,
            load_depth_kwargs=load_depth_kwargs,
            load_selection_func=load_selection_func,
            get_selection_labels=get_selection_labels,
            load_selection_kwargs=load_selection_kwargs,
            load_SED_result_func=load_SED_result_func,
            apply_gal_instr_mask=apply_gal_instr_mask,
            simulated=simulated
        )

    @classmethod
    def from_data(
        cls: Type[Self],
        data: Data,
        id: int,
        **kwargs,
    ) -> Self:
        gal_creator = cls(
            data.survey,
            data.version,
            id,
            data._get_phot_cat_path(),
            data.filterset,
            data.aper_diams,
            **kwargs,
        )
        gal_creator.data = data
        return gal_creator

    def __call__(
        self: Self,
        psfs: Optional[
            Union[
                List[Optional[Type[PSF_Base]]],
                NDArray[Optional[Type[PSF_Base]]],
                Dict[str, Optional[Type[PSF_Base]]],
            ]
        ] = None,
    ) -> Galaxy:
        from . import Photometry_obs, Galaxy
        galfind_logger.info(
            f"Loading {repr(self)} galaxy!"
        )
        # make array of Photometry_obs for each aperture diameter
        galfind_logger.debug(
            f"Loading {repr(self)} photometry!"
        )
        IDs = self.load_IDs(cropped = True)
        sky_coords = self.load_skycoords(cropped = True)
        phot, phot_err = self.load_phot(cropped = True)
        depths = self.load_depths(cropped = True)
        for aper_diam in self.aper_diams:
            for arr, label in zip([phot, phot_err, depths], ["photometry", "photometry error", "depths"]):
                assert aper_diam in arr, galfind_logger.critical(
                    f"{label.capitalize()} for {repr(self)} is missing aperture diameter {aper_diam}!"
                )
        selection_flags_arr, selection_kwargs_arr = self.load_selection_flags_kwargs(cropped = True)
        filterset_arr = self.load_gal_filtersets(length = len(IDs), cropped = True)
        assert len(IDs) == len(sky_coords) == len(selection_flags_arr) == len(selection_kwargs_arr) == len(filterset_arr) == 1, \
            galfind_logger.critical(
                f"{len(IDs)=} != {len(sky_coords)=} != {len(selection_flags_arr)=} != {len(selection_kwargs_arr)=} != {len(filterset_arr)=} != 1"
            )
        sky_coord = sky_coords[0]
        phot = {aper_diam: phot[aper_diam][0] for aper_diam in self.aper_diams}
        phot_err = {aper_diam: phot_err[aper_diam][0] for aper_diam in self.aper_diams}
        depths = {aper_diam: depths[aper_diam][0] for aper_diam in self.aper_diams}
        selection_flags = selection_flags_arr[0]
        selection_kwargs = selection_kwargs_arr[0]
        filterset = filterset_arr[0]
        SED_results = {}
        phot_obs = {
            aper_diam: Photometry_obs(
                filterset,
                phot[aper_diam],
                phot_err[aper_diam],
                depths[aper_diam],
                aper_diam,
                SED_results = SED_results,
                psfs = psfs,
                simulated = self.simulated,
            ) for aper_diam in self.aper_diams
        }
        # make an array of galaxy objects to be stored in the catalogue
        #, origin_survey = self.survey
        gal = Galaxy(
            self.id,
            sky_coord,
            phot_obs,
            selection_flags = selection_flags,
            selection_kwargs = selection_kwargs,
            cat_filterset = self.filterset,
            survey = self.survey,
            simulated = self.simulated,
        )
        # if hasattr(self, "data"):
        #     gal.data = self.data
        galfind_logger.info(f"Made {self.cat_path} galaxy!")
        return gal

    def load_crops(
        self: Self,
        *args,
        **kwargs,
    ):
        super().load_crops(crops = None)
        # load table
        tab = self.open_cat(self.cat_path, "ID")
        crop_mask = np.full(len(tab), False)
        crop_mask[tab[self.ID_label] == self.id] = True
        self.crop_mask = crop_mask
        assert np.sum(crop_mask) == 1, galfind_logger.critical(
            f"Galaxy ID {self.id} not found in catalogue {self.cat_path}!"
        )

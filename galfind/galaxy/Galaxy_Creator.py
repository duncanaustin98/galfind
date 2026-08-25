"""Factory for creating individual Galaxy objects from catalogue FITS data.

Specializes in loading single-galaxy data with customizable loading
functions for
photometry, masks, depths, and optional SED results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import astropy.units as u
import numpy as np
from astropy.table import Table
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..imaging import Data, PSF_Base
    from . import (
        Galaxy,
        Multiple_Filter,
    )
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from .. import galfind_logger
from ..catalogues.Catalogue import (
    Catalogue_Creator,
    galfind_depth_labels,
    galfind_mask_labels,
    galfind_phot_labels,
    galfind_selection_labels,
    load_bool_Table,
    load_galfind_depths,
    load_galfind_mask,
    load_galfind_phot,
    load_IDs_Table,
    load_skycoords_Table,
    open_galfind_cat,
    open_galfind_hdr,
)
from ..utils.exceptions import LengthMismatchError, MissingDataError


class Galaxy_Creator(Catalogue_Creator):
    """Factory for creating individual Galaxy objects from catalogue data.

    Inherits from `Catalogue_Creator` and specializes in loading data for
    a single galaxy (identified by ID) from a FITS catalogue, including
    photometry, masks, depths, and optional SED fit results.

    Parameters
    ----------
    survey : `str`
        Survey name.
    version : `str`
        Data release version.
    id : `int`
        Unique galaxy identifier in the catalogue.
    cat_path : `str`
        Path to the FITS catalogue file.
    filterset : `Multiple_Filter`
        Set of filters for the survey.
    aper_diams : `astropy.units.Quantity`
        Aperture diameters for photometry.
    open_cat : callable, optional
        Function to open the catalogue file. Default is `open_galfind_cat`.
    open_hdr : callable, optional
        Function to extract the catalogue header. Default is
        `open_galfind_hdr`.
    load_ID_func : callable or `None`, optional
        Function to load galaxy ID. Default is `load_IDs_Table`.
    load_skycoords_func : callable or `None`, optional
        Function to load RA/DEC. Default is `load_skycoords_Table`.
    load_phot_func : callable, optional
        Function to load photometry. Default is `load_galfind_phot`.
    load_mask_func : callable or `None`, optional
        Function to load masks. Default is `load_galfind_mask`.
    load_depth_func : callable or `None`, optional
        Function to load depth maps. Default is `load_galfind_depths`.
    load_selection_func : callable or `None`, optional
        Function to load selection/boolean flags. Default is `load_bool_Table`.
    load_SED_result_func : callable or `None`, optional
        Function to load SED fit results. Default is `None`.
    apply_gal_instr_mask : `bool`, optional
        Apply instrument masks to the galaxy. Default is `True`.
    cache_fits_handle : `bool`, optional
        Whether to keep a single FITS file handle open and reuse it for all
        extension reads. Significantly speeds up loading many galaxies from
        the same catalogue. Falls back to standard reading if any errors occur.
        Default is `True`.
    simulated : `bool`, optional
        Whether the galaxy is from a simulation. Default is `False`.
    **kwargs
        Additional keyword arguments passed to parent class.
    """

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
        skycoords_labels: Dict[str, str] = {
            "RA": "ALPHA_J2000",
            "DEC": "DELTA_J2000",
        },
        skycoords_units: Dict[str, u.Unit] = {"RA": u.deg, "DEC": u.deg},
        load_skycoords_kwargs: Dict[str, Any] = {},
        load_phot_func: Callable = load_galfind_phot,
        get_phot_labels: Callable[
            [Multiple_Filter], Dict[str, str]
        ] = galfind_phot_labels,
        load_phot_kwargs: Dict[str, Any] = {
            "ZP": u.Jy.to(u.ABmag),
            "min_flux_pc_err": 10.0,
        },
        load_mask_func: Optional[Callable] = load_galfind_mask,
        get_mask_labels: Callable[
            [Multiple_Filter], Dict[str, str]
        ] = galfind_mask_labels,
        load_mask_kwargs: Dict[str, Any] = {},
        load_depth_func: Optional[Callable] = load_galfind_depths,
        get_depth_labels: Callable[
            [Multiple_Filter], Dict[str, str]
        ] = galfind_depth_labels,
        load_depth_kwargs: Dict[str, Any] = {},
        load_selection_func: Optional[
            Callable[[], Dict[u.Quantity, Dict[str, List[Any]]]]
        ] = load_bool_Table,
        get_selection_labels: Callable[
            [Table, List[str]], List[str]
        ] = galfind_selection_labels,
        load_selection_kwargs: Dict[str, Any] = {},
        load_SED_result_func: Optional[Callable] = None,
        apply_gal_instr_mask: bool = True,
        cache_fits_handle: bool = True,
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
            cache_fits_handle=cache_fits_handle,
            simulated=simulated,
        )

    def __repr__(self: Self) -> str:
        return (
            f"{self.__class__.__name__}({self.survey},"
            f"{self.version},{self.id})"
        )

    @classmethod
    def from_data(
        cls: Type[Self],
        data: Data,
        id: int,
        **kwargs,
    ) -> Self:
        """Create a galaxy creator from Data object and ID.

        Parameters
        ----------
        data : `Data`
            Data object containing survey, version, and filter information.
        id : `int`
            Galaxy ID to create a creator for.
        **kwargs
            Additional keyword arguments passed to constructor.

        Returns
        -------
        `Galaxy_Creator`
            Galaxy creator instance initialized from data.
        """
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
        from ..photometry.Photometry_obs import Photometry_obs
        from .Galaxy import Galaxy

        galfind_logger.info(f"Loading {repr(self)} galaxy!")
        # make array of Photometry_obs for each aperture diameter
        galfind_logger.debug(f"Loading {repr(self)} photometry!")
        IDs = self.load_IDs(cropped=True)
        sky_coords = self.load_skycoords(cropped=True)
        phot, phot_err = self.load_phot(cropped=True)
        depths = self.load_depths(cropped=True)
        for aper_diam in self.aper_diams:
            for arr, label in zip(
                [phot, phot_err, depths],
                ["photometry", "photometry error", "depths"],
            ):
                if aper_diam not in arr:
                    raise MissingDataError(
                        f"{label.capitalize()} for {repr(self)} is "
                        f"missing aper_diam={aper_diam!r}."
                    )
        selection_flags_arr, selection_kwargs_arr = (
            self.load_selection_flags_kwargs(cropped=True)
        )
        filterset_arr = self.load_gal_filtersets(length=len(IDs), cropped=True)
        lengths = {
            "IDs": len(IDs),
            "sky_coords": len(sky_coords),
            "selection_flags_arr": len(selection_flags_arr),
            "selection_kwargs_arr": len(selection_kwargs_arr),
            "filterset_arr": len(filterset_arr),
        }
        if not all(length == 1 for length in lengths.values()):
            raise LengthMismatchError(
                f"{repr(self)} loaded mismatched lengths (all must be "
                f"1 for a single galaxy): lengths={lengths!r}."
            )
        sky_coord = sky_coords[0]
        phot = {aper_diam: phot[aper_diam][0] for aper_diam in self.aper_diams}
        phot_err = {
            aper_diam: phot_err[aper_diam][0] for aper_diam in self.aper_diams
        }
        depths = {
            aper_diam: depths[aper_diam][0] for aper_diam in self.aper_diams
        }
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
                SED_results=SED_results,
                psfs=psfs,
                simulated=self.simulated,
            )
            for aper_diam in self.aper_diams
        }
        # make an array of galaxy objects to be stored in the catalogue
        # , origin_survey = self.survey
        gal = Galaxy(
            self.id,
            sky_coord,
            phot_obs,
            selection_flags=selection_flags,
            selection_kwargs=selection_kwargs,
            cat_filterset=self.filterset,
            survey=self.survey,
            version=self.version,
            simulated=self.simulated,
        )
        if not hasattr(gal, "gal_creator"):
            setattr(gal, "gal_creator", self)
        galfind_logger.info(f"Made {repr(gal)} from {repr(self)}!")
        return gal

    def load_crops(
        self: Self,
        *args,
        **kwargs,
    ) -> None:
        """Load crop mask for this galaxy from catalogue.

        Loads the photometric catalogue and creates a boolean mask
        identifying rows corresponding to this galaxy's ID.
        """
        super().load_crops(crops=None)
        # load table
        tab = self.open_cat(self.cat_path, "ID")
        crop_mask = np.full(len(tab), False)
        crop_mask[tab[self.ID_label] == self.id] = True
        self.crop_mask = crop_mask
        if np.sum(crop_mask) != 1:
            raise MissingDataError(
                f"Galaxy ID={self.id} not found (or not unique) in "
                f"catalogue cat_path={self.cat_path!r}; matched "
                f"{int(np.sum(crop_mask))} row(s), expected exactly 1."
            )

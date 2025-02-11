from __future__ import annotations

import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from abc import abstractmethod
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, List, Dict
if TYPE_CHECKING:
    from . import Catalogue, SED_code, Selector, Multiple_Filter
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from .. import galfind_logger
from ..Catalogue import galfind_depth_labels, scattered_phot_labels, load_galfind_depths
from .. import useful_funcs_austind as funcs


class Grid_2D:

    def __init__(
        self: Self,
        x: u.Quantity,
        y: u.Quantity,
        z: NDArray[float],
    ):
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def from_sim_cat(
        cls: Type[Self],
        sim_cat: Catalogue,
        SED_fitter_arr: List[SED_code],
        sample: Type[Selector],
        aper_diam: u.Quantity,
        mode: str = "n_nearest",
        depth_region: str = "all",
    ) -> Self:
        #assert sim_cat.cat_creator.load_mask_func is None 
        assert not sim_cat.cat_creator.apply_gal_instr_mask
        # determine scattered catalogue path
        scattered_cat_path = funcs.get_phot_cat_path(
            sim_cat.survey,
            sim_cat.version,
            sim_cat.filterset.instrument_name,
            sim_cat.data.aper_diams,
            forced_phot_band_name = None,
        ).replace(".fits", f"_reg={depth_region}.fits") #_{sim_cat.cat_path.split('/')[-1]}

        # construct catalogue creator for scattered catalogue
        scattered_cat_creator = deepcopy(sim_cat.cat_creator)
        scattered_cat_creator.cat_path = scattered_cat_path
        # define new photometry and photometry error labels
        scattered_cat_creator.get_phot_labels = scattered_phot_labels
        # define ZP to be from Jy
        load_phot_kwargs = scattered_cat_creator.load_phot_kwargs
        load_phot_kwargs["ZP"] = u.Jy.to(u.ABmag)
        load_phot_kwargs["incl_errs"] = True
        scattered_cat_creator.load_phot_kwargs = load_phot_kwargs
        # define new depth labels and load in function
        scattered_cat_creator.get_depth_labels = galfind_depth_labels
        scattered_cat_creator.load_depth_func = load_galfind_depths

        # make scattered catalogue if it doesn't already exist
        if not Path(scattered_cat_path).is_file():
            galfind_logger.info(
                f"Making {scattered_cat_path.split('/')[-1]} scattered catalogue"
            )
            # make a new catalogue from the scattered photometry of the original
            scattered_sim_cat = deepcopy(sim_cat)
            scattered_sim_cat.scatter(aper_diam, mode, depth_region)
            scattered_tab = scattered_sim_cat.open_cat() # old table
            # update cat creator with the updated one
            scattered_sim_cat.cat_creator = scattered_cat_creator
            # add new scattered flux columns to the old table
            for i, filt in tqdm(enumerate(scattered_sim_cat.data.filterset), 
                desc = "Adding scattered flux/err/depth columns to the table",
                total = len(scattered_sim_cat.data.filterset)
            ):
                scattered_tab[f"{filt.band_name}_scattered"] = np.array(
                    [
                        gal.aper_phot[aper_diam].flux[i].value
                        for gal in scattered_sim_cat
                    ]
                )
                scattered_tab[f"{filt.band_name}_err"] = np.array(
                    [
                        gal.aper_phot[aper_diam].flux_errs[i].value
                        for gal in scattered_sim_cat
                    ]
                )
                scattered_tab[f"loc_depth_{filt.band_name}"] = np.array(
                    [
                        gal.aper_phot[aper_diam].depths[i].value
                        for gal in scattered_sim_cat
                    ]
                )
            # save the new scattered catalogue
            scattered_tab.write(scattered_cat_path, overwrite = True)
            galfind_logger.info(
                f"Scattered catalogue saved at {scattered_cat_path}"
            )
        else:
            # load the scattered catalogue
            scattered_sim_cat = scattered_cat_creator()

        # Run SED fitting on the scattered catalogue
        [
            SED_fitter(
                scattered_sim_cat,
                aper_diam,
                load_PDFs = False, #True,
                load_SEDs = False, #True,
                update = True
            ) for SED_fitter in SED_fitter_arr
        ]
        # # perform sample selection
        # sample_cat = sample(scattered_sim_cat, return_copy = True)
        # breakpoint()
        # return cls.from_sim_cat_select_cat(sim_cat, sample_cat)

    @classmethod
    def from_sim_cat_select_cat(
        cls: Type[Self],
        sim_cat: Catalogue,
        select_cat: Catalogue,
    ) -> Self:

        x, y, z = cls._make_grid(sim_cat, select_cat)
        obj = cls(x, y, z)
        obj._sim_cat = sim_cat
        obj._select_cat = select_cat
        return obj
    
    @staticmethod
    @abstractmethod
    def _make_grid(
        sim_cat: Catalogue,
        select_cat: Catalogue,
    ) -> Tuple[u.Quantity, u.Quantity, NDArray[float, float]]:
        # make grid from the catalogues
        pass

    def __call__(
        self: Self,
        x: u.Quantity,
        y: u.Quantity,
    ) -> float:
        # interpolate grid
        pass


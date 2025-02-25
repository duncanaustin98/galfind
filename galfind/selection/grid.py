from __future__ import annotations

import astropy.units as u
import numpy as np
import h5py
from numpy.typing import NDArray
from abc import abstractmethod
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, List, Dict, Optional
if TYPE_CHECKING:
    from . import Catalogue, SED_code, Selector, Multiple_Filter, Property_Calculator_Base
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from .. import galfind_logger, config
from ..Catalogue import galfind_depth_labels, scattered_phot_labels, load_galfind_depths
from .. import useful_funcs_austind as funcs


class Grid:

    def __init__(
        self: Self,
        x: u.Quantity,
        y: u.Quantity,
        z: NDArray[float],
        x_name: str,
        y_name: str,
    ):
        self.x = x
        self.y = y
        self.z = z
        self.x_name = x_name
        self.y_name = y_name

    @classmethod
    def from_select_sim_xy(
        cls: Type[Self],
        x_select: u.Quantity,
        y_select: u.Quantity,
        x_sim: u.Quantity,
        y_sim: u.Quantity,
        x_arr: NDArray[float],
        y_arr: NDArray[float],
        x_name: str,
        y_name: str,
    ) -> Self:
        assert x_select.unit == x_arr.unit
        assert y_select.unit == y_arr.unit
        assert len(x_select) == len(y_select) == len(x_sim) == len(y_sim)
        # determine bins that each x and y selection value falls into
        x_select_bins = np.digitize(x_select, x_arr)
        y_select_bins = np.digitize(y_select, y_arr)
        # determine bins that each x and y simulated value falls into
        x_sim_bins = np.digitize(x_sim, x_arr)
        y_sim_bins = np.digitize(y_sim, y_arr)
        # determine which galaxies are selected
        selected = np.full(len(x_select_bins), False)
        for i, (x_select_bin, y_select_bin, x_sim_bin, y_sim_bin) in enumerate(zip(x_select_bins, y_select_bins, x_sim_bin, y_sim_bin)):
            selected[i] = cls._select(x_select_bin, y_select_bin, x_sim_bin, y_sim_bin)
        # make a grid from the selected and simulated galaxies
        z, _, _ = np.histogram2d(
            x_sim[selected],
            y_sim[selected],
            bins = (x_arr.value, y_arr.value)
        )
        return cls(x_sim, y_sim, z, x_name, y_name)

    @staticmethod
    @abstractmethod
    def _select(
        x_select_bin: int,
        y_select_bin: int,
        x_sim_bin: int,
        y_sim_bin: int,
    ) -> bool:
        pass

    # @classmethod
    # def from_cat_xy(
    #     cls: Type[Self],
    #     cat: Catalogue,
    #     x_calculator: Type[Property_Calculator_Base],
    #     y_calculator: Type[Property_Calculator_Base],
    #     x_arr: NDArray[float],
    #     y_arr: NDArray[float],
    #     grid_type: str,
    # ) -> None:
    #     assert grid_type.lower() in ["simulated", "selected"]
    #     save_path = f"{config['DEFAULT']['GALFIND_WORK']}/Grids/{grid_type.lower()}" + \
    #         f"{cat.version}/{cat.filterset.instrument_name}/{cat.survey}/" + \
    #         f"{y_calculator.name}_vs_{x_calculator.name}.h5"
    #     funcs.make_dirs(save_path)
    #     breakpoint()
    #     if Path(save_path).is_file():
    #         hf = h5py.File(save_path, "r")
    #         x_arr = hf["x"][:]
    #         x_name = hf["x"].attrs["x_name"]
    #         x_arr *= u.Unit(hf["x"].attrs["x_unit"])
    #         y_arr = hf["y"][:]
    #         y_name = hf["y"].attrs["y_name"]
    #         y_arr *= u.Unit(hf["y"].attrs["y_unit"])
    #         z = hf["z"][:]

    #         hf.close()
    #         galfind_logger.info(
    #             f"Loaded {grid_type} grid from {save_path}. " + \
    #             "Faster to not re-compute the catalogues."
    #         )
    #     else:
    #         x_calculator(cat)
    #         y_calculator(cat)
    #         # make grid from the catalogues
    #         x = x_calculator.extract_vals(cat).to(x_arr.unit).value
    #         x_name = x_calculator.name
    #         y = y_calculator.extract_vals(cat).to(y_arr.unit).value
    #         y_name = y_calculator.name
    #         z, _, _ = np.histogram2d(x, y, bins = (x_arr.value, y_arr.value))
    #         Grid_2D._save_grid(x_arr, y_arr, z, save_path, x_name, y_name)
    #     return cls(x, y, z, x_name, y_name)

    @staticmethod
    def _save_grid(
        x_arr: NDArray[float],
        y_arr: NDArray[float],
        z: NDArray[float, float],
        save_path: str,
        x_name: str,
        y_name: str,
    ) -> None:
        # save grid as .h5 file
        hf = h5py.File(save_path, "w")
        hf_x = hf.create_dataset("x", data = x_arr.value)
        hf_x.attrs["x_name"] = x_name
        hf_x.attrs["x_unit"] = x_arr.unit.to_string()
        hf_y = hf.create_dataset("y", data = y_arr.value)
        hf_y.attrs["y_name"] = y_name
        hf_y.attrs["y_unit"] = y_arr.unit.to_string()
        hf.create_dataset("z", data = z)
        hf.close()


class Correct_Grid(Grid):

    def __init__(
        self: Self,
        x: u.Quantity,
        y: u.Quantity,
        z: NDArray[float],
        x_name: str,
        y_name: str,
    ):
        super().__init__(x, y, z, x_name, y_name)

    @classmethod
    def from_sim_cat_select_cat(
        cls: Type[Self],
        sim_cat: Catalogue,
        select_cat: Catalogue,
        x_calculator: Type[Property_Calculator_Base],
        y_calculator: Type[Property_Calculator_Base],
        x_arr: NDArray[float],
        y_arr: NDArray[float],
        sim_cat_x_colname: Optional[str] = None,
        sim_cat_y_colname: Optional[str] = None,
    ) -> Self:
        # extract x and y properties from selection catalogue
        x_calculator(select_cat)
        y_calculator(select_cat)
        x_select = x_calculator.extract_vals(select_cat)
        y_select = y_calculator.extract_vals(select_cat)
        # extract x and y properties from simulated catalogue
        if sim_cat_x_colname is None:
            x_calculator(sim_cat)
            x_sim = x_calculator.extract_vals(sim_cat)
        else:
            x_sim = sim_cat.open_cat()[sim_cat_x_colname]
        if sim_cat_y_colname is None:
            y_calculator(sim_cat)
            y_sim = y_calculator.extract_vals(sim_cat)
        else:
            y_sim = sim_cat.open_cat()[sim_cat_y_colname]
        # make grid from the extracted properties
        return cls.from_select_sim_xy(
            x_select,
            y_select,
            x_sim,
            y_sim,
            x_arr,
            y_arr,
        )


class Grid_2D:

    def __init__(
        self: Self,
        sim_grid: Grid,
        select_grid: Grid,
    ):
        self.sim_grid = sim_grid
        self.select_grid = select_grid

    @classmethod
    def from_sim_cat(
        cls: Type[Self],
        sim_cat: Catalogue,
        SED_fitter_arr: List[SED_code],
        sampler: Type[Selector],
        aper_diam: u.Quantity,
        x_calculator: Type[Property_Calculator_Base],
        y_calculator: Type[Property_Calculator_Base],
        x_arr: NDArray[float],
        y_arr: NDArray[float],
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

        # run/load SED fitting on the scattered catalogue
        [
            SED_fitter(
                scattered_sim_cat,
                aper_diam,
                load_PDFs = True,
                load_SEDs = True,
                update = True
            ) for SED_fitter in SED_fitter_arr
        ]
        # perform sample selection
        select_cat = deepcopy(scattered_sim_cat)
        for _sampler in sampler:
            _sampler(scattered_sim_cat)
            #select_cat = sampler(select_cat, return_copy = True)
        raise Exception()
        return cls.from_sim_cat_select_cat(
            sim_cat, 
            select_cat,
            x_calculator,
            y_calculator,
            x_arr,
            y_arr,
        )

    @classmethod
    def from_sim_cat_select_cat(
        cls: Type[Self],
        sim_cat: Catalogue,
        select_cat: Catalogue,
        x_calculator: Type[Property_Calculator_Base],
        y_calculator: Type[Property_Calculator_Base],
        x_arr: NDArray[float],
        y_arr: NDArray[float],
        sim_cat_x_colname: Optional[str] = None,
        sim_cat_y_colname: Optional[str] = None,
    ) -> Self:
        # make grids from the catalogues or load if already made
        # sim_grid = Grid.from_cat_xy(
        #     sim_cat,
        #     x_calculator,
        #     y_calculator,
        #     x_arr,
        #     y_arr,
        #     grid_type = "simulated"
        # )
        correct_grid = Correct_Grid.from_sim_cat_select_cat(
            sim_cat,
            select_cat,
            x_calculator,
            y_calculator,
            x_arr,
            y_arr,
            sim_cat_x_colname,
            sim_cat_y_colname,
        )
        incorrect_grid = Incorrect_Grid.from_sim_cat_select_cat(
            sim_cat,
            select_cat,
            x_calculator,
            y_calculator,
            x_arr,
            y_arr,
            sim_cat_x_colname,
            sim_cat_y_colname,
        )
        breakpoint()
        #return cls(sim_grid, select_grid)

    def __call__(
        self: Self,
        x: u.Quantity,
        y: u.Quantity,
    ) -> float:
        # interpolate grid
        pass


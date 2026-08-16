from __future__ import annotations

import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.patheffects as pe
from numpy.typing import NDArray
import logging
from numpy.typing import NDArray
from abc import abstractmethod
from copy import deepcopy
from astropy.table import Table
from tqdm import tqdm
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, List, Dict, Optional, Callable
if TYPE_CHECKING:
    from . import Catalogue, SED_code, Selector, Multiple_Filter, Property_Calculator_Base
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from .. import galfind_logger
from ..Catalogue import galfind_depth_labels, scattered_phot_labels, load_galfind_depths, scattered_depth_labels#, scattered_phot_labels_inst
from .. import useful_funcs_austind as funcs


class Grid:

    def __init__(
        self: Self,
        x: u.Quantity,
        y: u.Quantity,
        N: NDArray[float],
        x_name: str,
        y_name: str,
    ):
        self.x = x
        self.y = y
        self.N = N
        self.x_name = x_name
        self.y_name = y_name

    # @classmethod
    # def from_select_sim_xy(
    #     cls: Type[Self],
    #     x_select: u.Quantity,
    #     y_select: u.Quantity,
    #     x_sim: u.Quantity,
    #     y_sim: u.Quantity,
    #     x_arr: NDArray[float],
    #     y_arr: NDArray[float],
    #     x_name: str,
    #     y_name: str,
    # ) -> Self:
    #     assert x_select.unit == x_arr.unit
    #     assert y_select.unit == y_arr.unit
    #     assert len(x_select) == len(y_select) == len(x_sim) == len(y_sim)
    #     # determine bins that each x and y selection value falls into
    #     x_select_bins = np.digitize(x_select, x_arr)
    #     y_select_bins = np.digitize(y_select, y_arr)
    #     # determine bins that each x and y simulated value falls into
    #     x_sim_bins = np.digitize(x_sim, x_arr)
    #     y_sim_bins = np.digitize(y_sim, y_arr)
    #     # determine which galaxies are selected
    #     selected = np.full(len(x_select_bins), False)
    #     for i, (x_select_bin, y_select_bin, x_sim_bin, y_sim_bin) in enumerate(zip(x_select_bins, y_select_bins, x_sim_bins, y_sim_bins)):
    #         selected[i] = cls._select(x_select_bin, y_select_bin, x_sim_bin, y_sim_bin)
    #     # make a grid from the selected and simulated galaxies
    #     z, _, _ = np.histogram2d(
    #         x_sim[selected],
    #         y_sim[selected],
    #         bins = (x_arr.value, y_arr.value)
    #     )
    #     return cls(x_sim, y_sim, z, x_name, y_name)

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

    @classmethod
    def from_fits_cat(
        cls: Type[Self],
        cat_path: str,
        x_arr: NDArray[float],
        y_arr: NDArray[float],
        x_name: str,
        y_name: str,
        x_hdu: Optional[str] = None,
        y_hdu: Optional[str] = None,
        select_colnames: Optional[List[str]] = None,
        select_hdu: Optional[str] = None,
    ):
        x_tab = Table.read(cat_path, hdu = x_hdu)
        y_tab = Table.read(cat_path, hdu = y_hdu)
        if select_colnames is not None:
            select_tab = Table.read(cat_path, hdu = select_hdu)
            select_mask = np.full(len(select_tab), True)
            for colname in select_colnames:
                select_mask &= select_tab[colname]
            if len(x_tab) != len(select_tab):
                raise ValueError("Length of x_tab and select_tab do not match")
            else:
                x_tab = x_tab[select_mask]
                galfind_logger.debug(f"Applied selection mask to {cat_path} for column(s) {select_colnames}")
            if len(y_tab) != len(select_tab):
                raise ValueError("Length of y_tab and select_tab do not match")
            else:
                y_tab = y_tab[select_mask]
                galfind_logger.debug(f"Applied selection mask to {cat_path} for column(s) {select_colnames}")
        x = x_tab[x_name]
        y = y_tab[y_name]
        N = np.histogram2d(x, y, bins=[x_arr, y_arr])[0]
        return cls(x_arr, y_arr, N, x_name, y_name)

    @classmethod
    def from_h5(
        cls: Type[Self],
        save_path: str
    ) -> Self:
        hf = h5py.File(save_path, "r")
        x_arr = hf["x"][:]
        x_name = hf["x"].attrs["x_name"]
        #x_arr *= u.Unit(hf["x"].attrs["x_unit"])
        y_arr = hf["y"][:]
        y_name = hf["y"].attrs["y_name"]
        #y_arr *= u.Unit(hf["y"].attrs["y_unit"])
        N = hf["N"][:]
        hf.close()
        galfind_logger.info(f"Loaded grid from {save_path}!")
        grid = cls(x_arr, y_arr, N, x_name, y_name)
        grid.h5_path = save_path
        return grid
        
    def save_h5(
        self: Self,
        save_path: str,
    ) -> None:
        if not hasattr(self, "h5_path"):
            galfind_logger.info(f"Saving grid to {save_path}!")
            self.h5_path = save_path
            funcs.make_dirs(save_path)
            # save grid as .h5 file
            hf = h5py.File(save_path, "w")
            hf_x = hf.create_dataset("x", data = self.x)
            hf_x.attrs["x_name"] = self.x_name
            # hf_x.attrs["x_unit"] = x_arr.unit.to_string()
            hf_y = hf.create_dataset("y", data = self.y)
            hf_y.attrs["y_name"] = self.y_name
            #hf_y.attrs["y_unit"] = y_arr.unit.to_string()
            hf.create_dataset("N", data = self.N)
            hf.close()
        else:
            galfind_logger.warning(
                f"Grid already has an associated h5 file at {self.h5_path}!"
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
        x_arr: NDArray[float],
        y_arr: NDArray[float],
        x_simname: str,
        y_simname: str,
        x_selectname: str,
        y_selectname: str,
        xsim_hdu: Optional[str] = None,
        ysim_hdu: Optional[str] = None,
        xselect_hdu: Optional[str] = None,
        yselect_hdu: Optional[str] = None,
        mode: str = "n_nearest",
        depth_region: str = "all",
        sim_filterset: Optional[Multiple_Filter] = None,
        data_filterset: Optional[Multiple_Filter] = None,
        aper_diams: Optional[List[u.Quantity]] = None,
        depth_labels_func: Optional[Callable] = None,
        phot_labels_func: Optional[Callable] = None,
        save_PDFs: bool = True,
        save_SEDs: bool = True,
    ) -> Self:
        #assert sim_cat.cat_creator.load_mask_func is None 
        assert not sim_cat.cat_creator.apply_gal_instr_mask

        if hasattr(sim_cat, 'data') and sim_cat.data is not None:
            if data_filterset is not None or aper_diams is not None or sim_filterset is not None:
                galfind_logger.critical("Don't provide filterset and aper_diams if you have a Data object")
            sim_filterset = sim_cat.data.filterset
            aper_diams = sim_cat.data.aper_diams
            data_filterset = sim_cat.data.filterset
            
        elif not hasattr(sim_cat, 'data') or sim_cat.data is None:
            if data_filterset is None or aper_diams is None or sim_filterset is None:
                galfind_logger.critical("filterset and aper_diams must be provided if not in the Catalogue object")

        # determine scattered catalogue path
        scattered_cat_path = funcs.get_phot_cat_path(
            sim_cat.survey,
            sim_cat.version,
            sim_filterset.instrument_name,
            aper_diams,
            forced_phot_filt_name = None,
        ).replace(".fits", f"_reg={depth_region}.fits") #_{sim_cat.cat_path.split('/')[-1]}

        # construct catalogue creator for scattered catalogue
        scattered_cat_creator = deepcopy(sim_cat.cat_creator)
        scattered_cat_creator.cat_path = scattered_cat_path
        # define new photometry and photometry error labels
        scattered_cat_creator.get_phot_labels = scattered_phot_labels if phot_labels_func is None else phot_labels_func
        # define ZP to be from Jy
        load_phot_kwargs = scattered_cat_creator.load_phot_kwargs
        load_phot_kwargs["ZP"] = u.Jy.to(u.ABmag)
        load_phot_kwargs["incl_errs"] = True
        scattered_cat_creator.load_phot_kwargs = load_phot_kwargs
        # define new depth labels and load in function
        scattered_cat_creator.get_depth_labels = galfind_depth_labels if depth_labels_func is None else depth_labels_func
        scattered_cat_creator.load_depth_func = load_galfind_depths
        scattered_cat_creator.simulated = True

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
            for i, filt in tqdm(enumerate(data_filterset), 
                desc = "Adding scattered flux/err/depth columns to the table",
                total = len(data_filterset),
                disable = galfind_logger.getEffectiveLevel() > logging.INFO
            ):
                scattered_tab[f"{filt.instrument_name}.{filt.filt_name}_scattered"] = np.array(
                    [
                        gal.aper_phot[aper_diam].flux[i].value
                        for gal in scattered_sim_cat
                    ]
                )
                scattered_tab[f"{filt.instrument_name}.{filt.filt_name}_err"] = np.array(
                    [
                        gal.aper_phot[aper_diam].flux_errs[i].value
                        for gal in scattered_sim_cat
                    ]
                )
                scattered_tab[f"loc_depth_{filt.instrument_name}.{filt.filt_name}"] = np.array(
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
                save_PDFs = save_PDFs,
                save_SEDs = save_SEDs,
                load_SEDs = False,
                update = True,
            ) for SED_fitter in SED_fitter_arr #[SED_fitter_arr[1]]
        ]
        #raise Exception()
        # perform sample selection
        if sampler is not None:
            select_cat = deepcopy(scattered_sim_cat)
            for _sampler in sampler:
                _sampler(scattered_sim_cat)
                select_cat = _sampler(select_cat, return_copy = True)
        return cls.from_fits_tabs(
            sim_cat.cat_path,
            select_cat.cat_path,
            x_arr,
            y_arr,
            x_simname,
            y_simname,
            x_selectname,
            y_selectname,
            xsim_hdu,
            ysim_hdu,
            xselect_hdu,
            yselect_hdu,
            select_colnames = [_sampler.name for _sampler in sampler],
            select_hdu = "SELECTION",
        )
    
    @classmethod
    def from_fits_tabs(
        cls: Type[Self],
        sim_cat_path: str,
        select_cat_path: str,
        x_arr: NDArray[float],
        y_arr: NDArray[float],
        x_simname: str,
        y_simname: str,
        x_selectname: str,
        y_selectname: str,
        xsim_hdu: Optional[str] = None,
        ysim_hdu: Optional[str] = None,
        xselect_hdu: Optional[str] = None,
        yselect_hdu: Optional[str] = None,
        select_colnames: Optional[List[str]] = None,
        select_hdu: Optional[str] = "SELECTION",
    ) -> Self:
        sim_grid = Grid.from_fits_cat(
            sim_cat_path,
            x_arr,
            y_arr,
            x_simname,
            y_simname,
            xsim_hdu,
            ysim_hdu,
        )
        select_grid = Grid.from_fits_cat(
            select_cat_path,
            x_arr,
            y_arr,
            x_selectname,
            y_selectname,
            xselect_hdu,
            yselect_hdu,
            select_colnames = select_colnames,
            select_hdu = select_hdu,
        )
        return cls(sim_grid, select_grid)

    # @classmethod
    # def from_sim_cat_select_cat(
    #     cls: Type[Self],
    #     sim_cat: Catalogue,
    #     select_cat: Catalogue,
    #     x_calculator: Type[Property_Calculator_Base],
    #     y_calculator: Type[Property_Calculator_Base],
    #     x_arr: NDArray[float],
    #     y_arr: NDArray[float],
    #     sim_cat_x_colname: Optional[str] = None,
    #     sim_cat_y_colname: Optional[str] = None,
    # ) -> Self:
    #     # make grids from the catalogues or load if already made
    #     # sim_grid = Grid.from_cat_xy(
    #     #     sim_cat,
    #     #     x_calculator,
    #     #     y_calculator,
    #     #     x_arr,
    #     #     y_arr,
    #     #     grid_type = "simulated"
    #     # )
    #     correct_grid = Correct_Grid.from_sim_cat_select_cat(
    #         sim_cat,
    #         select_cat,
    #         x_calculator,
    #         y_calculator,
    #         x_arr,
    #         y_arr,
    #         sim_cat_x_colname,
    #         sim_cat_y_colname,
    #     )
    #     incorrect_grid = Incorrect_Grid.from_sim_cat_select_cat(
    #         sim_cat,
    #         select_cat,
    #         x_calculator,
    #         y_calculator,
    #         x_arr,
    #         y_arr,
    #         sim_cat_x_colname,
    #         sim_cat_y_colname,
    #     )
    #     breakpoint()
    #     #return cls(sim_grid, select_grid)

    @classmethod
    def from_h5(cls: Type[Self], sim_path: str, select_path: str) -> Self:
        sim_grid = Grid.from_h5(sim_path)
        select_grid = Grid.from_h5(select_path)
        return cls(sim_grid, select_grid)

    def save_h5(self: Self, sim_path: str, select_path: str) -> None:
        self.sim_grid.save_h5(sim_path)
        self.select_grid.save_h5(select_path)

    def __call__(
        self: Self,
        x: Union[int, float, List[float], NDArray[float]], #u.Quantity,
        y: Union[int, float, List[float], NDArray[float]], #u.Quantity,
    ) -> float:
        # scipy regular grid interpolator, since our grid is regular in x and y (but not necessarily in N)
        from scipy.interpolate import RegularGridInterpolator
        x_arr = self.sim_grid.x
        xmid = 0.5 * (x_arr[:-1] + x_arr[1:])
        y_arr = self.sim_grid.y
        ymid = 0.5 * (y_arr[:-1] + y_arr[1:])
        N_arr = self.select_grid.N / self.sim_grid.N
        # mask out bins with no simulated galaxies to avoid extrapolation
        mask = self.sim_grid.N > 0
        xmid_masked = xmid[mask.any(axis = 1)]
        ymid_masked = ymid[mask.any(axis = 0)]
        N_arr_masked = N_arr[np.ix_(mask.any(axis = 1), mask.any(axis = 0))]
        interpolator = RegularGridInterpolator((xmid_masked, ymid_masked), N_arr_masked, bounds_error = False, fill_value = None)
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(y, list):
            y = np.array(y)
        interpolated = interpolator((x, y))
        if any(np.isnan(interpolated)):
            err_message = f"{sum(np.isnan(interpolated))} interpolated values are NaN for ({x=}, {y=})!"
            galfind_logger.critical(err_message)
            raise ValueError(err_message)
        return interpolated


    def pcolormesh(
        self: Self,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        cmap: str = "viridis",
        cbar_label: str = "Completeness",
        cmesh_kwargs: Dict[str, Any] = {},
        survey_version_label: Optional[str] = None,
        save_path: Optional[str] = None,
        close: bool = True,
    ) -> None:
        if "cmap" not in cmesh_kwargs.keys():
            cmesh_kwargs["cmap"] = cmap
        if fig is None and ax is None:
            fig, ax = plt.subplots()
        c = ax.pcolormesh(self.sim_grid.x, self.sim_grid.y, (self.select_grid.N / self.sim_grid.N).T, **cmesh_kwargs)
        cb = plt.colorbar(c, ax = ax)
        cb.set_label(cbar_label)
        ax.set_xlabel(r"Redshift, $z$")
        ax.invert_yaxis()
        ax.set_ylabel(r"$M_{\rm UV}$")
        ax.set_xticks(self.sim_grid.x)
        ax.set_yticks(self.sim_grid.y)
        # label survey and version in upper right
        if survey_version_label is not None:
            ax.text(0.975, 1.025, survey_version_label, #f"{survey}, {version}",
                transform = ax.transAxes,
                ha = "right", va = "bottom", fontsize = 16, color = "black",
                path_effects = [pe.withStroke(linewidth = 3, foreground = "white")]
            )
        # label each box with selected and simulated number counts
        label_kwargs = {
            "ha": "center",
            "va": "center",
            "color": "red",
            "fontsize": 7.0,
            #"fontweight": "bold",
            "path_effects": [pe.withStroke(linewidth = 0.5, foreground = "white")]
        }
        for i in tqdm(range(len(self.sim_grid.x)-1), desc = "Labelling bins", total = len(self.sim_grid.x)-1):
            for j in range(len(self.sim_grid.y)-1):
                n_sim = int(self.sim_grid.N[i, j])
                n_sel = int(self.select_grid.N[i, j])
                if n_sim > 0: # only label bins with data
                    ax.text(
                        0.5 * (self.sim_grid.x[i] + self.sim_grid.x[i+1]),
                        0.5 * (self.sim_grid.y[j] + self.sim_grid.y[j+1]),
                        f"{n_sel}/{n_sim}",
                        **label_kwargs,
                    )
        if save_path is not None:
            funcs.make_dirs(save_path)
            plt.savefig(save_path, dpi = 300)
            galfind_logger.info(f"Saved 2D grid plot at {save_path}")
        if close:
            plt.close(fig)



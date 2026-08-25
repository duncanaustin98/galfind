from __future__ import annotations

import logging
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import astropy.units as u
import h5py
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from numpy.typing import NDArray
from tqdm import tqdm

if TYPE_CHECKING:
    from . import Catalogue, Multiple_Filter, SED_code, Selector
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from .. import galfind_logger
from ..catalogues.Catalogue import (  # , scattered_phot_labels_inst
    galfind_depth_labels,
    load_galfind_depths,
    scattered_phot_labels,
)
from ..utils import useful_funcs_austind as funcs
from ..utils.exceptions import (
    IncompatibleKwargsError,
    LengthMismatchError,
    RangeError,
)


class Grid:
    """2D histogram grid of galaxy counts over two binned properties
    (e.g. redshift and M_UV).

    Stores a 2D count array `N` binned in `x` and `y`, together with the
    names of the properties/columns used to construct it. Pairs of `Grid`
    instances (a "simulated" grid and a "selected" grid) are combined by
    `Grid_2D` to build selection-function/completeness grids.

    Parameters
    ----------
    x : `astropy.units.Quantity`
        Bin edges (or centres) along the x-axis.
    y : `astropy.units.Quantity`
        Bin edges (or centres) along the y-axis.
    N : `numpy.ndarray`
        2D histogram of galaxy counts binned in `x` and `y`.
    x_name : `str`
        Name of the property/column binned along the x-axis.
    y_name : `str`
        Name of the property/column binned along the y-axis.

    Attributes
    ----------
    x : `astropy.units.Quantity`
        Bin edges (or centres) along the x-axis.
    y : `astropy.units.Quantity`
        Bin edges (or centres) along the y-axis.
    N : `numpy.ndarray`
        2D histogram of galaxy counts.
    x_name : `str`
        Name of the property binned along the x-axis.
    y_name : `str`
        Name of the property binned along the y-axis.
    h5_path : `str`
        Path to this grid's saved ``.h5`` file. Only set once the grid
        has been loaded via `from_h5` or saved via `save_h5`.
    """

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
    #     for i, (x_select_bin, y_select_bin, x_sim_bin, y_sim_bin) in \
    #             enumerate(zip(x_select_bins, y_select_bins,
    #             x_sim_bins, y_sim_bins)):
    #         selected[i] = cls._select(
    #             x_select_bin, y_select_bin, x_sim_bin, y_sim_bin
    #         )
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
    #     save_path = f"{config['DEFAULT']['GALFIND_WORK']}/Grids/" + \
    #         f"{grid_type.lower()}" + \
    # f"{cat.version}/{cat.filterset.instrument_name}/{cat.survey}/" +
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
        """Construct a `Grid` by histogramming two columns of a FITS catalogue.

        Parameters
        ----------
        cat_path : `str`
            Path to the FITS catalogue to read.
        x_arr : `numpy.ndarray`
            Bin edges along the x-axis, passed to `numpy.histogram2d`.
        y_arr : `numpy.ndarray`
            Bin edges along the y-axis, passed to `numpy.histogram2d`.
        x_name : `str`
            Name of the column to bin along the x-axis.
        y_name : `str`
            Name of the column to bin along the y-axis.
        x_hdu : `str`, optional
            FITS HDU/extension name containing the x column. Default is
            `None` (the table's default HDU).
        y_hdu : `str`, optional
            FITS HDU/extension name containing the y column. Default is
            `None` (the table's default HDU).
        select_colnames : `list` of `str`, optional
            Names of boolean selection columns to combine (via logical
            AND) and apply as a row mask before histogramming. Default is
            `None` (no selection applied).
        select_hdu : `str`, optional
            FITS HDU/extension name containing the `select_colnames`
            columns. Only used if `select_colnames` is given. Default is
            `None`.

        Returns
        -------
        `Grid`
            The grid of counts binned in `x_name` and `y_name`.

        Raises
        ------
        LengthMismatchError
            If `select_colnames` is given and the selection table has a
            different number of rows than the x or y table.
        """
        x_tab = Table.read(cat_path, hdu=x_hdu)
        y_tab = Table.read(cat_path, hdu=y_hdu)
        if select_colnames is not None:
            select_tab = Table.read(cat_path, hdu=select_hdu)
            select_mask = np.full(len(select_tab), True)
            for colname in select_colnames:
                select_mask &= select_tab[colname]
            if len(x_tab) != len(select_tab):
                raise LengthMismatchError(
                    f"len(x_tab)={len(x_tab)} != len(select_tab)="
                    f"{len(select_tab)} for cat_path={cat_path!r}."
                )
            else:
                x_tab = x_tab[select_mask]
                galfind_logger.debug(
                    f"Applied selection mask to {cat_path} for "
                    + f"column(s) {select_colnames}"
                )
            if len(y_tab) != len(select_tab):
                raise LengthMismatchError(
                    f"len(y_tab)={len(y_tab)} != len(select_tab)="
                    f"{len(select_tab)} for cat_path={cat_path!r}."
                )
            else:
                y_tab = y_tab[select_mask]
                galfind_logger.debug(
                    f"Applied selection mask to {cat_path} for "
                    + f"column(s) {select_colnames}"
                )
        x = x_tab[x_name]
        y = y_tab[y_name]
        N = np.histogram2d(x, y, bins=[x_arr, y_arr])[0]
        return cls(x_arr, y_arr, N, x_name, y_name)

    @classmethod
    def from_h5(cls: Type[Self], save_path: str) -> Self:
        """Load a `Grid` from a saved ``.h5`` file.

        Parameters
        ----------
        save_path : `str`
            Path to the ``.h5`` file previously written by `save_h5`.

        Returns
        -------
        `Grid`
            The loaded grid, with `h5_path` set to `save_path`.
        """
        hf = h5py.File(save_path, "r")
        x_arr = hf["x"][:]
        x_name = hf["x"].attrs["x_name"]
        # x_arr *= u.Unit(hf["x"].attrs["x_unit"])
        y_arr = hf["y"][:]
        y_name = hf["y"].attrs["y_name"]
        # y_arr *= u.Unit(hf["y"].attrs["y_unit"])
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
        """Save this grid's `x`, `y`, and `N` arrays to an ``.h5`` file.

        Does nothing (other than logging a warning) if this grid already
        has an associated ``.h5`` file.

        Parameters
        ----------
        save_path : `str`
            Path to write the ``.h5`` file to.

        Returns
        -------
        `None`
        """
        if not hasattr(self, "h5_path"):
            galfind_logger.info(f"Saving grid to {save_path}!")
            self.h5_path = save_path
            funcs.make_dirs(save_path)
            # save grid as .h5 file
            hf = h5py.File(save_path, "w")
            hf_x = hf.create_dataset("x", data=self.x)
            hf_x.attrs["x_name"] = self.x_name
            # hf_x.attrs["x_unit"] = x_arr.unit.to_string()
            hf_y = hf.create_dataset("y", data=self.y)
            hf_y.attrs["y_name"] = self.y_name
            # hf_y.attrs["y_unit"] = y_arr.unit.to_string()
            hf.create_dataset("N", data=self.N)
            hf.close()
        else:
            galfind_logger.warning(
                f"Grid already has an associated h5 file at {self.h5_path}!"
            )


class Grid_2D:
    """Pair of `Grid` instances used to compute a 2D selection
    fraction/completeness.

    Combines a "simulated" grid (counts of all simulated galaxies) with a
    "selected" grid (counts of galaxies passing some selection), binned
    identically in the same two properties, so that the ratio
    `select_grid.N / sim_grid.N` gives the selection fraction
    (completeness) per bin.

    Parameters
    ----------
    sim_grid : `Grid`
        Grid of counts for all simulated galaxies.
    select_grid : `Grid`
        Grid of counts for galaxies passing the selection.

    Attributes
    ----------
    sim_grid : `Grid`
        Grid of counts for all simulated galaxies.
    select_grid : `Grid`
        Grid of counts for galaxies passing the selection.
    """

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
        """Construct a `Grid_2D` from a simulated catalogue by
        scattering, fitting, and selecting.

        Builds (or loads, if already saved) a "scattered" version of
        `sim_cat` with fluxes, flux errors, and depths drawn according to
        `mode` and `depth_region`, runs each SED fitting code in
        `SED_fitter_arr` on it, applies `sampler` to obtain a selected
        sub-catalogue, and then delegates to `from_fits_tabs` to
        histogram the simulated and selected catalogues into a `Grid_2D`.

        Parameters
        ----------
        sim_cat : `Catalogue`
            The (unscattered) simulated input catalogue. Its `cat_creator`
            must not apply a galaxy/instrument mask.
        SED_fitter_arr : `list` of `SED_code`
            SED fitting codes to run on the scattered simulated catalogue.
        sampler : `Selector`
            Iterable of selector(s) applied to the scattered catalogue
            (and a copy of it) to determine the selected sub-sample. If
            `None`, no selection is applied.
        aper_diam : `astropy.units.Quantity`
            Aperture diameter of the photometry to scatter and fit.
        x_arr : `numpy.ndarray`
            Bin edges along the x-axis.
        y_arr : `numpy.ndarray`
            Bin edges along the y-axis.
        x_simname : `str`
            Name of the column to bin along the x-axis for the simulated
            grid.
        y_simname : `str`
            Name of the column to bin along the y-axis for the simulated
            grid.
        x_selectname : `str`
            Name of the column to bin along the x-axis for the selected
            grid.
        y_selectname : `str`
            Name of the column to bin along the y-axis for the selected
            grid.
        xsim_hdu : `str`, optional
            FITS HDU containing `x_simname`. Default is `None`.
        ysim_hdu : `str`, optional
            FITS HDU containing `y_simname`. Default is `None`.
        xselect_hdu : `str`, optional
            FITS HDU containing `x_selectname`. Default is `None`.
        yselect_hdu : `str`, optional
            FITS HDU containing `y_selectname`. Default is `None`.
        mode : `str`, optional
            Photometric scattering mode passed to `Catalogue.scatter`.
            Default is `"n_nearest"`.
        depth_region : `str`, optional
            Depth region to use when scattering the photometry. Default
            is `"all"`.
        sim_filterset : `Multiple_Filter`, optional
            Filterset of the simulated catalogue. Required if `sim_cat`
            has no associated `Data` object. Default is `None`.
        data_filterset : `Multiple_Filter`, optional
            Filterset used to add the scattered flux/error/depth columns.
            Required if `sim_cat` has no associated `Data` object.
            Default is `None`.
        aper_diams : `list` of `astropy.units.Quantity`, optional
            Aperture diameters available in the catalogue. Required if
            `sim_cat` has no associated `Data` object. Default is `None`.
        depth_labels_func : `callable`, optional
            Function used to derive depth column labels for the scattered
            catalogue creator. Defaults to `galfind_depth_labels` if
            `None`.
        phot_labels_func : `callable`, optional
            Function used to derive photometry column labels for the
            scattered catalogue creator. Defaults to
            `scattered_phot_labels` if `None`.
        save_PDFs : `bool`, optional
            Whether to save SED-fitting PDFs when running
            `SED_fitter_arr`. Default is `True`.
        save_SEDs : `bool`, optional
            Whether to save best-fit SEDs when running `SED_fitter_arr`.
            Default is `True`.

        Returns
        -------
        `Grid_2D`
            The grid pair built from the scattered simulated and selected
            catalogues.

        Raises
        ------
        IncompatibleKwargsError
            If `sim_cat.cat_creator.apply_gal_instr_mask` is `True`, if
            `sim_cat` has an associated `Data` object and any of
            `data_filterset`, `aper_diams`, `sim_filterset` are also
            given, or if `sim_cat` has no associated `Data` object and
            any of `data_filterset`, `aper_diams`, `sim_filterset` are
            missing.
        """
        # assert sim_cat.cat_creator.load_mask_func is None
        if sim_cat.cat_creator.apply_gal_instr_mask:
            raise IncompatibleKwargsError(
                "sim_cat.cat_creator.apply_gal_instr_mask=True; "
                "Grid_2D.from_sim_cat requires a sim_cat whose "
                "cat_creator does not apply a galaxy/instrument mask."
            )

        if hasattr(sim_cat, "data") and sim_cat.data is not None:
            if (
                data_filterset is not None
                or aper_diams is not None
                or sim_filterset is not None
            ):
                raise IncompatibleKwargsError(
                    "sim_cat has an associated Data object, but one or "
                    f"more of data_filterset={data_filterset!r}, "
                    f"aper_diams={aper_diams!r}, "
                    f"sim_filterset={sim_filterset!r} was also given; "
                    "don't provide filterset and aper_diams if sim_cat "
                    "has a Data object."
                )
            sim_filterset = sim_cat.data.filterset
            aper_diams = sim_cat.data.aper_diams
            data_filterset = sim_cat.data.filterset

        elif not hasattr(sim_cat, "data") or sim_cat.data is None:
            if (
                data_filterset is None
                or aper_diams is None
                or sim_filterset is None
            ):
                raise IncompatibleKwargsError(
                    "sim_cat has no associated Data object, but one or "
                    f"more of data_filterset={data_filterset!r}, "
                    f"aper_diams={aper_diams!r}, "
                    f"sim_filterset={sim_filterset!r} is missing; all "
                    "three must be provided if sim_cat has no Data "
                    "object."
                )

        # determine scattered catalogue path
        scattered_cat_path = funcs.get_phot_cat_path(
            sim_cat.survey,
            sim_cat.version,
            sim_filterset.instrument_name,
            aper_diams,
            forced_phot_filt_name=None,
        ).replace(
            ".fits", f"_reg={depth_region}.fits"
        )  # _{sim_cat.cat_path.split('/')[-1]}

        # construct catalogue creator for scattered catalogue
        scattered_cat_creator = deepcopy(sim_cat.cat_creator)
        scattered_cat_creator.cat_path = scattered_cat_path
        # define new photometry and photometry error labels
        scattered_cat_creator.get_phot_labels = (
            scattered_phot_labels
            if phot_labels_func is None
            else phot_labels_func
        )
        # define ZP to be from Jy
        load_phot_kwargs = scattered_cat_creator.load_phot_kwargs
        load_phot_kwargs["ZP"] = u.Jy.to(u.ABmag)
        load_phot_kwargs["incl_errs"] = True
        scattered_cat_creator.load_phot_kwargs = load_phot_kwargs
        # define new depth labels and load in function
        scattered_cat_creator.get_depth_labels = (
            galfind_depth_labels
            if depth_labels_func is None
            else depth_labels_func
        )
        scattered_cat_creator.load_depth_func = load_galfind_depths
        scattered_cat_creator.simulated = True

        # make scattered catalogue if it doesn't already exist
        if not Path(scattered_cat_path).is_file():
            galfind_logger.info(
                f"Making {scattered_cat_path.split('/')[-1]} "
                + "scattered catalogue"
            )
            # make a new catalogue from the scattered photometry of the
            # original
            scattered_sim_cat = deepcopy(sim_cat)
            scattered_sim_cat.scatter(aper_diam, mode, depth_region)
            scattered_tab = scattered_sim_cat.open_cat()  # old table
            # update cat creator with the updated one
            scattered_sim_cat.cat_creator = scattered_cat_creator
            # add new scattered flux columns to the old table
            for i, filt in tqdm(
                enumerate(data_filterset),
                desc="Adding scattered flux/err/depth columns to the table",
                total=len(data_filterset),
                disable=galfind_logger.getEffectiveLevel() > logging.INFO,
            ):
                scattered_tab[
                    f"{filt.instrument_name}.{filt.filt_name}_scattered"
                ] = np.array(
                    [
                        gal.aper_phot[aper_diam].flux[i].value
                        for gal in scattered_sim_cat
                    ]
                )
                scattered_tab[
                    f"{filt.instrument_name}.{filt.filt_name}_err"
                ] = np.array(
                    [
                        gal.aper_phot[aper_diam].flux_errs[i].value
                        for gal in scattered_sim_cat
                    ]
                )
                scattered_tab[
                    f"loc_depth_{filt.instrument_name}.{filt.filt_name}"
                ] = np.array(
                    [
                        gal.aper_phot[aper_diam].depths[i].value
                        for gal in scattered_sim_cat
                    ]
                )
            # save the new scattered catalogue
            scattered_tab.write(scattered_cat_path, overwrite=True)
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
                save_PDFs=save_PDFs,
                save_SEDs=save_SEDs,
                load_SEDs=False,
                update=True,
            )
            for SED_fitter in SED_fitter_arr  # [SED_fitter_arr[1]]
        ]
        # raise Exception()
        # perform sample selection
        if sampler is not None:
            select_cat = deepcopy(scattered_sim_cat)
            for _sampler in sampler:
                _sampler(scattered_sim_cat)
                select_cat = _sampler(select_cat, return_copy=True)
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
            select_colnames=[_sampler.name for _sampler in sampler],
            select_hdu="SELECTION",
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
        """Construct a `Grid_2D` by histogramming simulated and
        selected FITS catalogues.

        Parameters
        ----------
        sim_cat_path : `str`
            Path to the FITS catalogue of all simulated galaxies.
        select_cat_path : `str`
            Path to the FITS catalogue of galaxies passing selection.
        x_arr : `numpy.ndarray`
            Bin edges along the x-axis.
        y_arr : `numpy.ndarray`
            Bin edges along the y-axis.
        x_simname : `str`
            Name of the column to bin along the x-axis in `sim_cat_path`.
        y_simname : `str`
            Name of the column to bin along the y-axis in `sim_cat_path`.
        x_selectname : `str`
            Name of the column to bin along the x-axis in
            `select_cat_path`.
        y_selectname : `str`
            Name of the column to bin along the y-axis in
            `select_cat_path`.
        xsim_hdu : `str`, optional
            FITS HDU containing `x_simname`. Default is `None`.
        ysim_hdu : `str`, optional
            FITS HDU containing `y_simname`. Default is `None`.
        xselect_hdu : `str`, optional
            FITS HDU containing `x_selectname`. Default is `None`.
        yselect_hdu : `str`, optional
            FITS HDU containing `y_selectname`. Default is `None`.
        select_colnames : `list` of `str`, optional
            Names of boolean selection columns to apply as a row mask
            before histogramming `select_cat_path`. Default is `None`.
        select_hdu : `str`, optional
            FITS HDU containing `select_colnames`. Default is
            `"SELECTION"`.

        Returns
        -------
        `Grid_2D`
            The grid pair built from `sim_cat_path` and `select_cat_path`.
        """
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
            select_colnames=select_colnames,
            select_hdu=select_hdu,
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
        """Load a `Grid_2D` from a pair of saved ``.h5`` files.

        Parameters
        ----------
        sim_path : `str`
            Path to the saved simulated `Grid` ``.h5`` file.
        select_path : `str`
            Path to the saved selected `Grid` ``.h5`` file.

        Returns
        -------
        `Grid_2D`
            The loaded grid pair.
        """
        sim_grid = Grid.from_h5(sim_path)
        select_grid = Grid.from_h5(select_path)
        return cls(sim_grid, select_grid)

    def save_h5(self: Self, sim_path: str, select_path: str) -> None:
        """Save this grid pair's `sim_grid` and `select_grid` to ``.h5`` files.

        Parameters
        ----------
        sim_path : `str`
            Path to write the simulated `Grid` ``.h5`` file to.
        select_path : `str`
            Path to write the selected `Grid` ``.h5`` file to.

        Returns
        -------
        `None`
        """
        self.sim_grid.save_h5(sim_path)
        self.select_grid.save_h5(select_path)

    def __call__(
        self: Self,
        x: Union[int, float, List[float], NDArray[float]],  # u.Quantity,
        y: Union[int, float, List[float], NDArray[float]],  # u.Quantity,
    ) -> float:
        # scipy regular grid interpolator, since our grid is regular
        # in x and y (but not necessarily in N)
        from scipy.interpolate import RegularGridInterpolator

        x_arr = self.sim_grid.x
        xmid = 0.5 * (x_arr[:-1] + x_arr[1:])
        y_arr = self.sim_grid.y
        ymid = 0.5 * (y_arr[:-1] + y_arr[1:])
        N_arr = self.select_grid.N / self.sim_grid.N
        # mask out bins with no simulated galaxies to avoid extrapolation
        mask = self.sim_grid.N > 0
        xmid_masked = xmid[mask.any(axis=1)]
        ymid_masked = ymid[mask.any(axis=0)]
        N_arr_masked = N_arr[np.ix_(mask.any(axis=1), mask.any(axis=0))]
        interpolator = RegularGridInterpolator(
            (xmid_masked, ymid_masked),
            N_arr_masked,
            bounds_error=False,
            fill_value=None,
        )
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(y, list):
            y = np.array(y)
        interpolated = interpolator((x, y))
        if any(np.isnan(interpolated)):
            raise RangeError(
                f"{sum(np.isnan(interpolated))} interpolated values "
                f"are NaN for (x={x!r}, y={y!r}); these lie outside "
                "the completeness grid's populated bins."
            )
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
        """Plot the selection fraction grid as a pcolormesh,
        labelling each bin's counts.

        Plots `select_grid.N / sim_grid.N` as a colour mesh over the
        shared `x`/`y` bins, annotating each populated bin with
        `"{n_selected}/{n_simulated}"`.

        Parameters
        ----------
        fig : `matplotlib.figure.Figure`, optional
            Figure to plot on. A new figure/axes pair is created if both
            `fig` and `ax` are `None`. Default is `None`.
        ax : `matplotlib.axes.Axes`, optional
            Axes to plot on. Default is `None`.
        cmap : `str`, optional
            Colormap name used if `"cmap"` is not already present in
            `cmesh_kwargs`. Default is `"viridis"`.
        cbar_label : `str`, optional
            Label for the colorbar. Default is `"Completeness"`.
        cmesh_kwargs : `dict`, optional
            Additional keyword arguments passed to `Axes.pcolormesh`.
            Default is `{}`.
        survey_version_label : `str`, optional
            If given, annotated in the upper-right corner of the plot.
            Default is `None`.
        save_path : `str`, optional
            If given, the path to save the figure to. Default is `None`.
        close : `bool`, optional
            Whether to close the figure after (optionally) saving it.
            Default is `True`.

        Returns
        -------
        `None`
        """
        if "cmap" not in cmesh_kwargs.keys():
            cmesh_kwargs["cmap"] = cmap
        if fig is None and ax is None:
            fig, ax = plt.subplots()
        c = ax.pcolormesh(
            self.sim_grid.x,
            self.sim_grid.y,
            (self.select_grid.N / self.sim_grid.N).T,
            **cmesh_kwargs,
        )
        cb = plt.colorbar(c, ax=ax)
        cb.set_label(cbar_label)
        ax.set_xlabel(r"Redshift, $z$")
        ax.invert_yaxis()
        ax.set_ylabel(r"$M_{\rm UV}$")
        ax.set_xticks(self.sim_grid.x)
        ax.set_yticks(self.sim_grid.y)
        # label survey and version in upper right
        if survey_version_label is not None:
            ax.text(
                0.975,
                1.025,
                survey_version_label,  # f"{survey}, {version}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=16,
                color="black",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            )
        # label each box with selected and simulated number counts
        label_kwargs = {
            "ha": "center",
            "va": "center",
            "color": "red",
            "fontsize": 7.0,
            # "fontweight": "bold",
            "path_effects": [pe.withStroke(linewidth=0.5, foreground="white")],
        }
        for i in tqdm(
            range(len(self.sim_grid.x) - 1),
            desc="Labelling bins",
            total=len(self.sim_grid.x) - 1,
        ):
            for j in range(len(self.sim_grid.y) - 1):
                n_sim = int(self.sim_grid.N[i, j])
                n_sel = int(self.select_grid.N[i, j])
                if n_sim > 0:  # only label bins with data
                    ax.text(
                        0.5 * (self.sim_grid.x[i] + self.sim_grid.x[i + 1]),
                        0.5 * (self.sim_grid.y[j] + self.sim_grid.y[j + 1]),
                        f"{n_sel}/{n_sim}",
                        **label_kwargs,
                    )
        if save_path is not None:
            funcs.make_dirs(save_path)
            plt.savefig(save_path, dpi=300)
            galfind_logger.info(f"Saved 2D grid plot at {save_path}")
        if close:
            plt.close(fig)

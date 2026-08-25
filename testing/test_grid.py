import os
from types import SimpleNamespace

# anchor to this file's own location, not the process cwd, so
# GALFIND_WORK/GALFIND_DATA always land under <repo_root>/testing/
# test_work regardless of the directory tests are invoked from
os.environ["GALFIND_CONFIG_DIR"] = os.path.dirname(os.path.abspath(__file__))
os.environ["GALFIND_CONFIG_NAME"] = "test_galfind_config.ini"

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from galfind.selection.completeness import Completeness
from galfind.selection.grid import Grid, Grid_2D
from galfind.utils.exceptions import (
    IncompatibleKwargsError,
    LengthMismatchError,
    RangeError,
)


class TestGrid:
    def test_from_fits_cat_raises_on_select_tab_length_mismatch(
        self, tmp_path
    ):
        # x/y table has 5 rows, "SELECT" table has only 3 -> mismatch
        data_tab = Table(
            {"x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)}
        )
        select_tab = Table({"flag": np.array([True, False, True])})
        cat_path = str(tmp_path / "sim.fits")
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.BinTableHDU(data_tab, name="DATA"),
                fits.BinTableHDU(select_tab, name="SELECT"),
            ]
        ).writeto(cat_path, overwrite=True)

        with pytest.raises(LengthMismatchError):
            Grid.from_fits_cat(
                cat_path,
                x_arr=np.arange(6, dtype=float),
                y_arr=np.arange(6, dtype=float),
                x_name="x",
                y_name="y",
                x_hdu="DATA",
                y_hdu="DATA",
                select_colnames=["flag"],
                select_hdu="SELECT",
            )


class TestGrid2D:
    def test_call_raises_rangeerror_for_nan_interpolation(self):
        # one bin has zero simulated galaxies (0/0 -> NaN) while its row
        # and column both still have data elsewhere, so it isn't masked
        # out of the interpolator; querying exactly at that grid node
        # returns NaN and should raise.
        x_edges = np.array([0.0, 1.0, 2.0])
        y_edges = np.array([0.0, 1.0, 2.0])
        sim_N = np.array([[5.0, 5.0], [5.0, 0.0]])
        select_N = np.array([[2.0, 2.0], [2.0, 0.0]])
        grid_2d = Grid_2D(
            Grid(x_edges, y_edges, sim_N, "x", "y"),
            Grid(x_edges, y_edges, select_N, "x", "y"),
        )
        # pass list (not scalar) inputs: Grid_2D.__call__ does
        # `any(np.isnan(interpolated))`, which requires an iterable
        # result rather than a 0-d array.
        with pytest.raises(RangeError):
            with np.errstate(invalid="ignore"):
                grid_2d([1.5], [1.5])

    def test_from_sim_cat_raises_incompatiblekwargs_when_mask_applied(self):
        fake_sim_cat = SimpleNamespace(
            cat_creator=SimpleNamespace(apply_gal_instr_mask=True)
        )
        with pytest.raises(IncompatibleKwargsError):
            Grid_2D.from_sim_cat(
                sim_cat=fake_sim_cat,
                SED_fitter_arr=[],
                sampler=None,
                aper_diam=None,
                x_arr=np.arange(3, dtype=float),
                y_arr=np.arange(3, dtype=float),
                x_simname="x",
                y_simname="y",
                x_selectname="x",
                y_selectname="y",
            )


class TestCompleteness:
    def test_from_simulated_fits_cat_raises_on_length_mismatch(self, tmp_path):
        # "DATA" table has 5 rows, "SELECTION" table has only 3 -> mismatch
        data_tab = Table(
            {
                "z": np.linspace(5.0, 10.0, 5),
                "x": np.linspace(-22.0, -18.0, 5),
            }
        )
        select_tab = Table({"selected": np.array([True, False, True])})
        cat_path = str(tmp_path / "sim_compl.fits")
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.BinTableHDU(data_tab, name="DATA"),
                fits.BinTableHDU(select_tab, name="SELECTION"),
            ]
        ).writeto(cat_path, overwrite=True)

        with pytest.raises(LengthMismatchError):
            Completeness.from_simulated_fits_cat(
                cat_path=cat_path,
                selection_column="selected",
                z_bin=[5.0, 10.0],
                x_bins=[-22.0, -20.0, -18.0],
                z_colname="z",
                x_colname="x",
            )

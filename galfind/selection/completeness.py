
from __future__ import annotations

from galfind.imaging.Data import Data
import numpy as np
import h5py as h5
from numpy.typing import NDArray
from astropy.table import Table
from scipy.interpolate import interp1d
import astropy.units as u
from typing import TYPE_CHECKING, Callable, Dict, Union, Optional, List
if TYPE_CHECKING:
    from . import Galaxy, Catalogue, Property_Calculator
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from .grid import Grid_2D
from .. import galfind_logger
from ..Catalogue_Base import Catalogue_Base

class Completeness:
    """Completeness curve of a selection as a function of one property.

    Stores a 1D completeness curve (fraction of simulated sources
    recovered by a selection) tabulated on a grid of a single derived
    property (e.g. magnitude or redshift), and provides interpolation
    of the completeness at arbitrary values of that property for a
    given galaxy or catalogue via `__call__`.

    Parameters
    ----------
    x : `numpy.ndarray` of `float`
        Grid of values of the property against which completeness is
        tabulated.
    x_calculator : `Type[Property_Calculator]`
        Calculator class used to compute the value of the relevant
        property for a given `Galaxy`.
    completeness : `numpy.ndarray` of `float`
        Completeness fraction tabulated at each value of `x`.
    x_completeness_lim : `Callable`, optional
        Function of `(x_gal, compl_gal)` returning `bool`, used to flag
        galaxies whose derived property lies beyond the regime where the
        tabulated completeness curve is reliable; when it returns
        `True` the completeness is replaced by `high_x_val`. Default is
        `None`.
    origin : `str`, optional
        Label identifying the survey/depth region this completeness
        curve was derived from (e.g. used to look up the correct curve
        in `Catalogue_Completeness`). Default is `None`.

    Attributes
    ----------
    x : `numpy.ndarray` of `float`
        Grid of property values.
    x_calculator : `Type[Property_Calculator]`
        Calculator used to compute the property for a galaxy.
    completeness : `numpy.ndarray` of `float`
        Tabulated completeness fractions.
    x_completeness_lim : `Callable` or `None`
        Limit-flagging function, if provided.
    origin : `str` or `None`
        Origin label of this completeness curve.
    """

    def __init__(
        self: Self,
        x: NDArray[float],
        x_calculator: Type[Property_Calculator],
        completeness: NDArray[float],
        x_completeness_lim: Optional[Callable[[float], bool]] = None,
        origin: Optional[str] = None,
    ):
        self.x = x
        self.x_calculator = x_calculator
        self.completeness = completeness
        self.x_completeness_lim = x_completeness_lim
        self.origin = origin

    @classmethod
    def from_h5(
        cls: Type[Completeness],
        h5_path: str,
        x_calculator: Type[Property_Calculator],
        x_label: str = "x",
        completeness_label: str = "completeness",
        x_completeness_lim: Optional[Callable[[float, float], bool]] = None,
        origin: Optional[str] = None,
    ):
        """Construct a `Completeness` curve from a saved HDF5 file.

        Parameters
        ----------
        h5_path : `str`
            Path to the HDF5 file containing the tabulated `x` and
            completeness datasets.
        x_calculator : `Type[Property_Calculator]`
            Calculator class used to compute the relevant property for
            a given `Galaxy`.
        x_label : `str`, optional
            Name of the HDF5 dataset containing the `x` grid values.
            Default is `"x"`.
        completeness_label : `str`, optional
            Name of the HDF5 dataset containing the completeness
            values. Default is `"completeness"`.
        x_completeness_lim : `Callable`, optional
            Function used to flag galaxies outside the reliable regime
            of the completeness curve. Default is `None`.
        origin : `str`, optional
            Label identifying the survey/depth region this curve came
            from. Default is `None`.

        Returns
        -------
        `Completeness`
            A new `Completeness` instance built from the HDF5 data.
        """
        # TODO: Extract origin from h5 path
        # open h5 file
        hf = h5.File(h5_path, "r")
        # read x and y datasets
        x = hf[x_label][:]
        compl = hf[completeness_label][:]
        # close h5 file
        hf.close()
        return cls(x, x_calculator, compl, x_completeness_lim, origin = origin)

    @classmethod
    def from_simulated_fits_cat(
        cls: Type[Completeness],
        cat_path: str,
        selection_column: str,
        z_bin: List[float],
        x_bins: List[float],
        z_colname: str,
        x_colname: str,
        x_calculator: Optional[Type[Property_Calculator]] = None,
        x_completeness_lim: Optional[Callable[[float, float], bool]] = None,
        origin: Optional[str] = None,
        # n_z_bins: int = 10,
        # survey: str,
        # version: str,
        # realization: int = 10,
        # z_arr: NDArray[float],
        # depth_region: str = "all",
        # MUV_arr: NDArray[float] = np.arange(-22.5, -16.5, 0.5),
        # cmap: str = "viridis",
    ):
        """Construct a `Completeness` curve from a simulated FITS catalogue.

        Reads a simulated catalogue and its associated `"SELECTION"`
        HDU, bins sources by redshift and the property of interest, and
        computes the completeness in each `x` bin as the ratio of
        selected to simulated source counts (marginalised over the
        single provided redshift bin).

        Parameters
        ----------
        cat_path : `str`
            Path to the simulated FITS catalogue. Must also contain a
            `"SELECTION"` HDU with the same number of rows.
        selection_column : `str`
            Name of the boolean column in the `"SELECTION"` HDU used to
            mask selected sources.
        z_bin : `list` of `float`
            Two-element list giving the redshift bin edges to include.
        x_bins : `list` of `float`
            Bin edges for the property of interest.
        z_colname : `str`
            Name of the redshift column in the catalogue.
        x_colname : `str`
            Name of the property column in the catalogue.
        x_calculator : `Type[Property_Calculator]`, optional
            Calculator class used to compute the property for a given
            `Galaxy`. Default is `None`.
        x_completeness_lim : `Callable`, optional
            Function used to flag galaxies outside the reliable regime
            of the completeness curve. Default is `None`.
        origin : `str`, optional
            Label identifying the survey/depth region this curve came
            from. Default is `None`.

        Returns
        -------
        `Completeness`
            A new `Completeness` instance with `x` set to the bin
            midpoints of `x_bins` and `completeness` set to the
            selected/simulated ratio in each bin.

        Raises
        ------
        AssertionError
            If the simulated catalogue and `"SELECTION"` HDU do not
            have the same number of rows.
        """
        tab = Table.read(cat_path)
        select_tab = Table.read(cat_path, hdu = "SELECTION")
        assert len(tab) == len(select_tab)

        z = tab[z_colname]
        x = tab[x_colname]

        simulated = np.histogram2d(z, x, bins=[z_bin, x_bins])[0]
        select_mask = (select_tab[selection_column])
        selected = np.histogram2d(z[select_mask], x[select_mask], bins=[z_bin, x_bins])[0]
        completeness = selected / simulated
        x_mid_bins = (x_bins[1:] + x_bins[:-1]) / 2
        return cls(
            x = x_mid_bins,
            x_calculator = x_calculator,
            completeness = completeness[0],
            x_completeness_lim = x_completeness_lim,
            origin = origin,
        )


    @property
    def high_x_val(self: Self) -> float:
        """`float`: Completeness value assumed beyond the tabulated limit.

        Value substituted for the interpolated completeness when
        `x_completeness_lim` flags a galaxy's property value as being
        beyond the regime where the tabulated curve is reliable.
        Currently always returns `1.0`.
        """
        return 1.0
        # # calculate the mean of the completeness values above the completeness limit
        # if self.x_completeness_lim is None:
        #     return None
        # high_x_mask = self.x > self.x_completeness_lim
        # return np.mean(self.completeness[high_x_mask])

    # @classmethod
    # def from_simulated_fits_cat(
    #     cls: Type[Completeness],
    #     cat_path: str,
    #     z_bin: List[float],
    #     z_intr_label: str,
    #     z_obs_label: str,
    #     #x_intr_label_arr: NDArray[str],
    #     #x_obs_label_arr: NDArray[str],
    #     x_calculator: Type[Property_Calculator],
    #     SNR_bins: Union[u.Quantity, u.Magnitude, u.Dex],
    # ):
    #     # TODO: Generalize this!

    #     cat = Table.read(cat_path)

    #     band = "F115W"


    #     # convert SNR bins into mag bins
    #     mag_bins = np.sort(-2.5 * np.log10(SNR_bins / 5.0) + cat[f"loc_depth_{band}"][0])
    #     mag_mid_bins = (mag_bins[1:] + mag_bins[:-1]) / 2
    #     SNR_mid_bins = 5.0 * 10 ** ((mag_mid_bins - cat[f"loc_depth_{band}"][0]) / -2.5)

    #     for band_ in ["F606W", "F115W"]:
    #         if band_ in ["F435W", "F606W"]:
    #             jag_flux_colname = f"HST_{band_}_fnu"
    #         else:
    #             jag_flux_colname = f"NRC_{band_}_fnu"
    #         cat[f"intr_{band_}_mag"] = -2.5 * np.log10(cat[jag_flux_colname]) + 31.4
    #         cat[f"obs_{band_}_mag"] = -2.5 * np.log10(cat[f"{band_}_scattered"]) + 8.9
    #         cat[f"intr_{band_}_mag_indices"] = np.digitize(cat[f"intr_{band_}_mag"], mag_bins)
    #         cat[f"obs_{band_}_mag_indices"] = np.digitize(cat[f"obs_{band_}_mag"], mag_bins)
    #         cat[f"SNR_{band_}"] = cat[f"{band_}_scattered"] / ((10 ** ((cat[f"loc_depth_{band_}"] - 8.9) / -2.5)) / 5.0)

    #     eazy_cat = Table.read(cat_path, hdu = "EAZY_FSPS_LARSON")
    #     selection_cat = Table.read(cat_path, hdu = "SELECTION")
    #     intr_z_mask = np.logical_and(cat["redshift"] > z_bin[0], cat["redshift"] < z_bin[1])
    #     obs_z_mask = np.logical_and(eazy_cat["zbest_fsps_larson_zfree"] > z_bin[0], eazy_cat["zbest_fsps_larson_zfree"] < z_bin[1])

    #     # if sample is None:
    #     #     sample_mask = cat[f"SNR_{band}"] > sigma_lim
    #     #     sample_name = f"{band}>{sigma_lim:.1f}σ"
    #     # else:
    #     #     sample_name = sample(aper_diams[0], SED_fitter).name
    #     #     sample_mask = selection_cat[sample_name]
    #     sample_name = "8sig_detect"
    #     sample_mask = cat["SNR_F115W"] > 8.0

    #     hist_intr = np.histogram(cat[intr_z_mask][f"intr_{band}_mag"], bins = mag_bins)[0]
    #     hist_obs = np.histogram(cat[np.logical_and.reduce([sample_mask, intr_z_mask, obs_z_mask])][f"intr_{band}_mag"], bins = mag_bins)[0]
    #     compl = hist_obs / hist_intr

    def __call__(
        self: Self,
        obj: Union[Galaxy, Type[Catalogue_Base]],
    ) -> float:
        from .. import Galaxy, Catalogue, Combined_Catalogue
        if isinstance(obj, Galaxy):
            return self._call_gal(obj)
        else:
            return np.array([self._call_gal(gal) for gal in obj])

    def _call_gal(
        self: Self,
        gal: Galaxy,
    ) -> float:
        self.x_calculator(gal)
        x_gal = self.x_calculator.extract_vals(gal).value
        #x_gal = x_gal * 0.76983916192 # aperture correction
        # compl_index = (np.abs(self.x - x_gal)).argmin()
        # # interpolate curve
        # compl_gal = self.completeness[compl_index]
        compl_gal = interp1d(self.x, self.completeness, fill_value = "extrapolate")(x_gal)
        if self.x_completeness_lim is not None and self.x_completeness_lim(x_gal, compl_gal):
            compl_gal = self.high_x_val 
        print(x_gal, compl_gal)
        return compl_gal

    def plot():
        """Plot the completeness curve. Not yet implemented."""
        pass


class Catalogue_Completeness: #(Completeness)
    """Collection of `Completeness` curves keyed by origin.

    Aggregates several `Completeness` objects (typically one per
    survey/depth-region combination) and dispatches a lookup for a
    given galaxy or catalogue to the appropriate curve based on its
    `origin`.

    Parameters
    ----------
    compl_arr : `list` of `Completeness`
        The individual completeness curves to combine.

    Attributes
    ----------
    compl_arr : `list` of `Completeness`
        The stored completeness curves.
    origins : `list` of `str`
        Origin labels of each curve in `compl_arr` (see `origins`
        property).
    """

    def __init__(
        self: Self,
        compl_arr = List[Completeness],
    ):
        self.compl_arr = compl_arr

    @property
    def origins(self: Self) -> List[str]:
        """`numpy.ndarray` of `str`: Origin label of each stored completeness curve."""
        return np.array([compl.origin for compl in self.compl_arr])
    
    def __call__(
        self: Self,
        obj: Union[Type[Catalogue_Base], Galaxy],
        data: Optional[Data] = None,
        depth_region: Optional[str] = None,
    ) -> float:
        from .. import Galaxy
        if isinstance(obj, Galaxy):
            return self._call_gal(obj, data=data, depth_region=depth_region)
        else: # Catalogue or Combined_Catalogue
            return np.array([self._call_gal(gal, data=data, depth_region=depth_region) for gal in obj])
        # if isinstance(obj, Catalogue):
        #     data_arr = [obj.data]
        # else: # Combined_Catalogue
        #     data_arr = [cat.data for cat in obj.cat_arr]
        # if isinstance(cat, Combined_Catalogue):
        #     cat_arr = cat.cat_arr
        # else:
        #     cat_arr = [cat]
        # for cat in cat_arr:
        #     origin_indices = np.array([i for i, origin in enumerate(self.origins) if origin.startswith(cat.survey)])
        #     try:
        #         assert len(origin_indices) > 0, galfind_logger.critical(
        #             f"{cat.survey=} not in {self.origins=}"
        #         )
        #     except:
        #         breakpoint()
            #breakpoint()
            #[completeness.x_calculator._get_IDs_properties(cat) for completeness in np.array(self.compl_arr)[origin_indices]]

    def _call_gal(
        self: Self,
        gal: Galaxy,
        data: Optional[Data] = None,
        depth_region: Optional[str] = None,
    ) -> Union[float, Dict[str, float]]:
        if data is not None and depth_region is not None:
            origin = f"{data.survey}_{depth_region}"
            compl = self.compl_arr[np.where(origin == self.origins)[0][0]]
            return compl(gal)
        else:
            return {compl.origin: compl(gal) for compl in self.compl_arr}
        # if data_arr is None:
        #     if hasattr(gal, "region"):
        #         region = gal.region[0]
        #     else:
        #         region = "all"
        #     origin = f"{gal.survey}_{region}"
        #     completeness = self.compl_arr[np.where(origin == self.origins)[0][0]]
        #     compl_gal = completeness(gal)
        # else:
        #     assert 
        # return compl_gal


class Completeness_2D(Grid_2D):
    """Grid of completeness as a function of two properties.

    `Grid_2D` subclass intended for characterising completeness (the
    ratio of selected to simulated source counts) as a function of two
    derived properties (e.g. redshift and absolute UV magnitude), using
    the interpolation and plotting behaviour inherited from `Grid_2D`.
    """
    pass

from __future__ import annotations

import numpy as np
import h5py as h5
from numpy.typing import NDArray
from astropy.table import Table
from scipy.interpolate import interp1d
import astropy.units as u
from typing import TYPE_CHECKING, Union, Optional, List
if TYPE_CHECKING:
    from . import Galaxy, Catalogue, Property_Calculator
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from .grid import Grid_2D
from .. import galfind_logger

class Completeness:

    def __init__(
        self: Self,
        x: NDArray[float],
        x_calculator: Type[Property_Calculator],
        completeness: NDArray[float],
        x_completeness_lim: float,
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
        x_completeness_lim: Optional[float] = None,
        origin: Optional[str] = None,
    ):
        # TODO: Extract origin from h5 path
        # open h5 file
        hf = h5.File(h5_path, "r")
        # read x and y datasets
        x = hf[x_label][:]
        compl = hf[completeness_label][:]
        # close h5 file
        hf.close()
        return cls(x, x_calculator, compl, x_completeness_lim, origin = origin)

    @property
    def high_x_val(self: Self) -> float:
        return 1.0
        # calculate the mean of the completeness values above the completeness limit
        if self.x_completeness_lim is None:
            return None
        high_x_mask = self.x > self.x_completeness_lim
        return np.mean(self.completeness[high_x_mask])

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
        x_gal = self.x_calculator(gal).value
        #x_gal = x_gal * 0.76983916192 # aperture correction
        # compl_index = (np.abs(self.x - x_gal)).argmin()
        # # interpolate curve
        # compl_gal = self.completeness[compl_index]
        if self.x_completeness_lim is not None and x_gal > self.x_completeness_lim:
            compl_gal = self.high_x_val
        else:
            compl_gal = interp1d(self.x, self.completeness)(x_gal)
        print(x_gal, compl_gal)
        return compl_gal

    def plot():
        pass


class Catalogue_Completeness: #(Completeness)

    def __init__(
        self: Self,
        compl_arr = List[Completeness],
    ):
        self.compl_arr = compl_arr

    @property
    def origins(self: Self) -> List[str]:
        return np.array([compl.origin for compl in self.compl_arr])
    
    def __call__(
        self: Self,
        cat: Type[Catalogue_Base],
    ) -> float:
        from .. import Catalogue, Combined_Catalogue
        if isinstance(cat, Combined_Catalogue):
            cat_arr = cat.cat_arr
        else:
            cat_arr = [cat]
        for cat in cat_arr:
            origin_indices = np.array([i for i, origin in enumerate(self.origins) if origin.startswith(cat.survey)])
            try:
                assert len(origin_indices) > 0, galfind_logger.critical(
                    f"{cat.survey=} not in {self.origins=}"
                )
            except:
                breakpoint()
            [completeness.x_calculator._get_IDs_properties(cat) for completeness in np.array(self.compl_arr)[origin_indices]]
        return np.array([self._call_gal(gal) for gal in cat])

    def _call_gal(
        self: Self,
        gal: Galaxy,
    ) -> float:
        # get the completeness for the galaxy
        if hasattr(gal, "region"):
            region = gal.region[0]
        else:
            region = "all"
        origin = f"{gal.survey}_{region}"
        completeness = self.compl_arr[np.where(origin == self.origins)[0][0]]
        compl_gal = completeness(gal)
        return compl_gal


class Completeness_2D(Grid_2D):
    pass
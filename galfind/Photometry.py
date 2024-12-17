#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:14:30 2023

@author: austind
"""

from __future__ import annotations

from copy import deepcopy
import time
import matplotlib.pyplot as plt
from abc import ABC
import astropy.units as u
import matplotlib.patheffects as pe
import numpy as np
from numpy.typing import NDArray
from typing import Union, Optional, List, Dict, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from . import Instrument
    from astropy.utils.masked import Masked
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import galfind_logger
from . import useful_funcs_austind as funcs
from . import Filter, Multiple_Filter


class Photometry:
    def __init__(
        self: Self,
        filterset: Multiple_Filter,
        flux: Union[Masked, u.Quantity],
        flux_errs: Union[Masked, u.Quantity],
        depths: Union[Dict[str, float], List[float]],
    ):
        self.filterset = filterset
        self.flux = flux
        self.flux_errs = flux_errs
        if isinstance(depths, dict):
            assert all(filt_name in depths.keys() for filt_name in filterset.band_names), \
                galfind_logger.critical(
                    f"not all {filterset.band_names} in {depths.keys()=}"
                )
            depths = [depths[filt.filt_name] for filt in filterset]
        self.depths = depths
        try:
            assert (
                len(self.filterset)
                == len(self.flux)
                == len(self.flux_errs)
                == len(self.depths)
            )
        except:
            breakpoint()

    def __str__(self) -> str:
        output_str = funcs.line_sep
        output_str += f"{self.__class__.__name__.upper()}:\n"
        output_str += funcs.band_sep
        #if print_instrument:
        output_str += str(self.instrument)
        # if print_fluxes:
        fluxes_str = [
            "%.1f ± %.1f nJy"
            % (flux.to(u.nJy).value, flux_err.to(u.nJy).value)
            for flux, flux_err in zip(
                self.flux.filled(fill_value=np.nan),
                self.flux_errs.filled(fill_value=np.nan),
            )
        ]
        output_str += f"FLUXES: {fluxes_str}\n"
        output_str += f"MAGS: {[np.round(mag, 2) for mag in self.flux.filled(fill_value = np.nan).to(u.ABmag).value]}\n"
        # if print_depths:
        output_str += (
            f"DEPTHS: {[np.round(depth, 2) for depth in self.depths.value]}\n"
        )
        output_str += funcs.line_sep
        return output_str

    def __getitem__(self: Self, i: Any):
        if isinstance(i, int):
            indices = np.array([i])
        elif isinstance(i, (np.ndarray, slice)):
            indices = i
        elif isinstance(i, list):
            indices = np.array(i)
        elif isinstance(i, str):
            i = i.split("+")
            indices = np.sort(np.array([np.where( \
                np.array(self.filterset.band_names) \
                == i_)[0][0] for i_ in i]))
        else:
            raise (
                TypeError(
                    f"{i=} in {__class__.__name__}.__getitem__ has invalid {type(i)=}"
                )
            )
        copy = deepcopy(self)
        copy.filterset = copy.filterset[indices]
        copy.flux = copy.flux[indices]
        copy.flux_errs = copy.flux_errs[indices]
        copy.depths = copy.depths[indices]
        return copy

    def __len__(self):
        return len(self.flux)

    # def __getattr__(
    #     self, property_name: str, origin: Union[str, dict] = "phot"
    # ) -> Union[None, u.Quantity, u.Magnitude, u.Dex]:
    #     assert origin in ["phot", "instrument"], galfind_logger.critical(
    #         f"{origin=} not in ['phot', 'instrument']"
    #     )
    #     if origin == "phot":
    #         split_property_name = property_name.split("_")
    #         if split_property_name[0] in self.instrument.band_names:
    #             band = split_property_name[0]
    #             property_name = "_".join(split_property_name[1:])
    #             if property_name in [
    #                 "flux_nu",
    #                 "flux_lambda",
    #                 "mag",
    #                 "flux_nu_errs",
    #                 "flux_nu_l1",
    #                 "flux_nu_u1",
    #                 "flux_lambda_l1",
    #                 "flux_lambda_errs",
    #                 "mag_l1",
    #                 "mag_u1",
    #                 "mag_errs",
    #             ] or ("depth" == property_name.split("_")[0]):
    #                 index = self.instrument.index_from_band_name(band)
    #                 wav = self.instrument[index].WavelengthCen
    #                 units = {
    #                     "flux_nu": u.Jy,
    #                     "flux_lambda": u.erg / (u.s * (u.cm**2) * u.AA),
    #                     "mag": u.ABmag,
    #                 }
    #                 if property_name in ["flux_nu", "flux_lambda", "mag"]:
    #                     return funcs.convert_mag_units(
    #                         wav, self.flux[index], units[property_name]
    #                     )
    #                 elif property_name in [
    #                     "flux_nu_errs",
    #                     "flux_lambda_errs",
    #                     "mag_errs",
    #                 ]:
    #                     flux = self.flux[index]
    #                     return funcs.convert_mag_err_units(
    #                         wav,
    #                         flux,
    #                         self.flux_errs[index],
    #                         units[property_name],
    #                     )
    #                 elif property_name in [
    #                     "flux_nu_l1",
    #                     "flux_lambda_l1",
    #                     "mag_l1",
    #                 ]:
    #                     flux = self.flux[index]
    #                     return funcs.convert_mag_err_units(
    #                         wav,
    #                         flux,
    #                         self.flux_errs[index],
    #                         units[property_name],
    #                     )[0]
    #                 elif property_name in [
    #                     "flux_nu_u1",
    #                     "flux_lambda_u1",
    #                     "mag_u1",
    #                 ]:
    #                     flux = self.flux[index]
    #                     return funcs.convert_mag_err_units(
    #                         wav,
    #                         flux,
    #                         self.flux_errs[index],
    #                         units[property_name],
    #                     )[1]
    #                 else:
    #                     depth = self.depths[index]
    #                     if property_name[-5:] == "sigma":
    #                         # calculate n sigma depths in ABmag
    #                         return funcs.five_to_n_sigma_mag(
    #                             depth, int(property_name[:-5].split("_")[-1])
    #                         )
    #                     else:
    #                         # return standard 5 sigma depths in ABmag
    #                         return self.depths[index]
    #             else:
    #                 err_message = (
    #                     f"{property_name=} not available in Photometry object!"
    #                 )
    #                 galfind_logger.critical(err_message)
    #                 raise AttributeError(err_message)
    #         else:
    #             raise AttributeError
    #     else:  # origin == "instrument":
    #         return self.instrument.__getattr__(property_name, origin)
    #         # elif name == "full_mask":
    #         #     return np.array([getattr(gal, "phot").mask for gal in self])

    @property
    def wav(self):
        return np.array([filt.WavelengthCen.to(u.AA).value \
            for filt in self.filterset]) * u.AA

    # @classmethod
    # def from_fits_cat(cls, fits_cat_row, instrument, cat_creator):
    #     fluxes, flux_errs = cat_creator.load_photometry(
    #         fits_cat_row, instrument.band_names
    #     )
    #     try:
    #         # local depths only currently works for one aperture diameter
    #         loc_depths = np.array(
    #             [
    #                 fits_cat_row[f"loc_depth_{band_name}"].T[
    #                     cat_creator.aper_diam_index
    #                 ]
    #                 for band_name in instrument.band_names
    #             ]
    #         )
    #     except:
    #         # print("local depths not loaded")
    #         loc_depths = None
    #     return cls(instrument, fluxes[0], flux_errs[0], loc_depths)

    def __add__(
        self: Self,
        other: Union[
            str,
            Filter,
            Multiple_Filter,
            Self,
            List[Union[str, Filter, Multiple_Filter, Self]],
        ],
    ) -> Self:
        pass

    def __sub__(
        self: Self,
        other: Union[
            str,
            Filter,
            Multiple_Filter,
            Self,
            List[Union[str, Filter, Multiple_Filter, Self]],
        ],
    ) -> Self:
        pass

    def crop(self: Type[Self], indices: Union[List[int], NDArray[int]]) -> Self:
        copy = deepcopy(self)
        indices = np.array(indices).astype(int)
        copy.filterset = copy.filterset[indices]
        for index in reversed(indices):
            self.instrument -= self.instrument[index]
        self.flux = np.delete(self.flux, indices)
        self.flux_errs = np.delete(self.flux_errs, indices)
        self.depths = np.delete(self.depths, indices)

    def plot(
        self: Self,
        ax: Optional[plt.Axes] = None,
        wav_units: u.Unit = u.AA,
        mag_units: u.Unit = u.Jy,
        plot_errs={"x": True, "y": True},
        annotate=True,
        uplim_sigma=2.0,
        auto_scale=True,
        errorbar_kwargs={
            "ls": "",
            "marker": "o",
            "ms": 4.0,
            "zorder": 100.0,
            "path_effects": [pe.withStroke(linewidth=2.0, foreground="white")],
        },
        filled=True,
        colour="black",
        label="Photometry",
        return_extra=False,
    ):
        wavs_to_plot = funcs.convert_wav_units(self.wav, wav_units).value
        mags_to_plot = funcs.convert_mag_units(self.wav, self.flux, mag_units)

        if uplim_sigma == None:
            uplims = list(np.full(len(self.flux), False))
        else:
            # work out optimal size of error bar in terms of sigma
            if mag_units == u.ABmag:
                uplim_sigma_arrow = 1.5
            else:
                uplim_sigma_arrow = {
                    "power density/spectral flux density wav": 1.5,
                    "ABmag/spectral flux density": 1.5,
                    "spectral flux density": 1.5,
                }[str(u.get_physical_type(mag_units))]
            assert uplim_sigma_arrow < uplim_sigma, galfind_logger.critical(
                f"uplim_sigma_arrow = {uplim_sigma_arrow} < uplim_sigma = {uplim_sigma}"
            )
            # calculate upper limits based on depths
            galfind_logger.warning(
                f"This will not work if {self.__class__.__name__ =} != 'Photometry_obs'"
            )
            uplims = [True if SNR < uplim_sigma else False for SNR in self.SNR]
            # set photometry to uplim_sigma for the data to be plotted as upper limits
            uplim_indices = [
                i for i, is_uplim in enumerate(uplims) if is_uplim
            ]
            uplim_vals = [
                funcs.convert_mag_units(
                    self.wav,
                    funcs.convert_mag_units(self.wav, self.depths[i], u.Jy)
                    * uplim_sigma
                    / 5.0,
                    mag_units,
                ).value
                for i in uplim_indices
            ] * mag_units
            mags_to_plot.put(uplim_indices, uplim_vals)
            galfind_logger.debug(
                "Should test whether upper plotting limits preserves the mask!"
            )
        self.non_detected_indices = uplims

        if plot_errs["y"]:
            mag_errs_new_units = funcs.convert_mag_err_units(
                self.wav,
                self.flux,
                self.flux_errs,
                mag_units,
            )
            # update with upper limit errors
            uplim_l1_vals = [
                funcs.convert_mag_units(
                    self.wav,
                    funcs.convert_mag_units(self.wav, self.depths[i], u.Jy)
                    * uplim_sigma_arrow
                    / 5.0,
                    mag_units,
                ).value
                for i in uplim_indices
            ] * mag_units

            if mag_units == u.ABmag:
                # swap l1 / u1 errors
                uplim_u1_vals = (uplim_l1_vals - uplim_vals).value
                uplim_l1_vals = [np.nan for i in uplim_indices]
            else:
                uplim_l1_vals = (uplim_vals - uplim_l1_vals).value
                uplim_u1_vals = [np.nan for i in uplim_indices]
            yerr = []
            for i, uplim_errs in enumerate([uplim_l1_vals, uplim_u1_vals]):
                mag_errs = mag_errs_new_units[i].value
                mag_errs.put(uplim_indices, uplim_errs)
                yerr.append(mag_errs * mag_units)
        else:
            yerr = None

        # log scale y axis if not in units of ABmag
        if mag_units != u.ABmag:
            if plot_errs["y"]:
                yerr = np.array(
                    funcs.log_scale_flux_errors(mags_to_plot, yerr)
                )
            mags_to_plot = np.array(
                funcs.log_scale_fluxes(mags_to_plot)
            )  # called 'mags_to_plot' but in this case are fluxes
        else:
            if plot_errs["y"]:
                yerr = np.array([yerr[0].value, yerr[1].value])
            mags_to_plot = np.array(mags_to_plot.value)

        if plot_errs["x"]:
            xerr = np.array(
                [
                    [
                        funcs.convert_wav_units(
                            filt.WavelengthCen - filt.WavelengthLower50,
                            wav_units,
                        ).value
                        for filt in self.filterset
                    ],
                    [
                        funcs.convert_wav_units(
                            filt.WavelengthUpper50 - filt.WavelengthCen,
                            wav_units,
                        ).value
                        for filt in self.filterset
                    ],
                ]
            )
        else:
            xerr = None

        # update errorbar kwargs - not quite general
        if filled:
            errorbar_kwargs["mfc"] = colour
        else:
            errorbar_kwargs["mfc"] = "none"
        errorbar_kwargs["color"] = colour
        errorbar_kwargs["label"] = label

        if auto_scale:
            # auto-scale the x-axis
            lower_xlim = np.min(wavs_to_plot - xerr[0]) * 0.95
            upper_xlim = np.max(wavs_to_plot + xerr[1]) * 1.05
            ax.set_xlim(lower_xlim, upper_xlim)
            # auto-scale the y-axis based on plotting units
            if mag_units == u.ABmag:
                if plot_errs["y"]:
                    lower_ylim = np.nanmax(mags_to_plot + yerr[1]) + 0.25
                    upper_ylim = np.nanmin(mags_to_plot - yerr[0]) - 0.75
                else:
                    lower_ylim = np.max(mags_to_plot) + 0.25
                    upper_ylim = np.min(mags_to_plot) - 0.75
            else:  # auto-scale flux units
                if plot_errs["y"]:
                    lower_ylim = np.nanmin(mags_to_plot - yerr[0]) - 0.15
                    upper_ylim = np.nanmax(mags_to_plot + yerr[1]) + 0.35
                else:
                    lower_ylim = np.min(mags_to_plot) - 0.15
                    upper_ylim = np.max(mags_to_plot) + 0.35
            ax.set_ylim(lower_ylim, upper_ylim)

        if mag_units == u.ABmag:
            plot_limits = {"lolims": uplims}
        else:
            plot_limits = {"uplims": uplims}

        plot = ax.errorbar(
            wavs_to_plot,
            mags_to_plot,
            xerr=xerr,
            yerr=yerr,
            **plot_limits,
            **errorbar_kwargs,
        )

        if return_extra:
            return plot, wavs_to_plot, mags_to_plot, yerr, uplims
        else:
            return plot
    
    def scatter_fluxes(
        self: Self, 
        n_scatter: int = 1
    ) -> Union[u.Quantity, Masked[u.Quantity]]:
        assert self.flux.unit != u.ABmag, \
            galfind_logger.critical(
                f"{self.flux.unit=} == 'ABmag'"
            )
        galfind_logger.debug("Finished assertion")
        scattered_fluxes = np.array(
            [
                np.random.normal(flux, err, n_scatter)
                for flux, err in zip(self.flux.value, self.flux_errs.value)
            ]
        ).T * self.flux.unit
        return scattered_fluxes

    def scatter(
        self: Self, 
        n_scatter: int = 1
    ) -> List[Photometry]:
        scattered_fluxes = self.scatter_fluxes(n_scatter)
        galfind_logger.debug("Made phot matrix")
        scattered_phot = self._make_phot_from_scattered_fluxes(scattered_fluxes, n_scatter)
        galfind_logger.debug("Constructed Photometry objects")
        if len(scattered_phot) == 1:
            return scattered_phot[0]
        else:
            return scattered_phot

    def _make_phot_from_scattered_fluxes(
        self: Self,
        scattered_fluxes: Masked[NDArray[float]], 
        n_scatter: int
    ) -> List[Photometry]:
        return [
            Photometry(
                self.filterset,
                scattered_fluxes[:, i],
                self.flux_errs,
                self.depths,
            )
            for i in range(n_scatter)
        ]

class Multiple_Photometry(ABC):
    def __init__(
        self, instrument_arr, flux_arr, flux_errs_arr, loc_depths_arr
    ):
        self.phot_arr = [
            Photometry(instrument, flux, flux_errs, loc_depths)
            for instrument, flux, flux_errs, loc_depths in zip(
                instrument_arr, flux_arr, flux_errs_arr, loc_depths_arr
            )
        ]

    @classmethod
    def from_fits_cat(cls, fits_cat, instrument, cat_creator):
        flux_arr, flux_errs_arr, gal_bands = cat_creator.load_photometry(
            fits_cat, instrument.band_names
        )
        # local depths not yet loaded in
        depths_arr = cat_creator.load_depths(
            fits_cat, instrument.band_names, gal_bands
        )
        instrument_arr = [
            deepcopy(instrument).remove_bands(
                [band for band in instrument if band.band_name not in bands]
            )
            for bands in gal_bands
        ]
        return cls(instrument_arr, flux_arr, flux_errs_arr, depths_arr)


class Mock_Photometry(Photometry):
    def __init__(
        self, instrument, flux, depths, min_flux_pc_err
    ):  # these depths should be 5σ and in units of ABmag
        assert len(flux) == len(depths)
        # add astropy units of ABmag if depths are not already
        try:
            assert depths.unit == u.ABmag
        except:
            depths *= u.ABmag
        # calculate errors from ABmag depths
        flux_errs = self.flux_errs_from_depths(flux, depths, min_flux_pc_err)
        self.min_flux_pc_err = min_flux_pc_err
        super().__init__(instrument, flux, flux_errs, depths)

    @staticmethod
    def flux_errs_from_depths(flux, depths, min_flux_pc_err):
        # calculate 1σ depths to Jy
        one_sig_depths_Jy = depths.to(u.Jy) / 5
        # apply min_flux_pc_err criteria
        flux_errs = (
            np.array(
                [
                    depth
                    if depth > flux * min_flux_pc_err / 100
                    else flux * min_flux_pc_err / 100
                    for flux, depth in zip(flux.value, one_sig_depths_Jy.value)
                ]
            )
            * u.Jy
        )
        return flux_errs

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Photometry across a set of filters.

Stores per-filter flux densities, flux uncertainties and optional depths, with
utilities for indexing, error propagation, Monte Carlo flux scattering, and plotting.
"""

from __future__ import annotations

from copy import deepcopy
import matplotlib.pyplot as plt
from abc import ABC
import astropy.units as u
from astropy.utils.masked import Masked
import matplotlib.patheffects as pe
import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Union, Optional, List, Dict, Any, NoReturn
if TYPE_CHECKING:
    pass
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from .. import galfind_logger
from ..utils import useful_funcs_austind as funcs
from ..imaging.Filter import Filter, Multiple_Filter


class Photometry:
    """Observed photometry of a single source across a set of filters.

    Stores per-filter flux densities, flux uncertainties and (optional)
    depths, and provides utilities for indexing, error propagation,
    Monte Carlo flux scattering, and plotting.

    Parameters
    ----------
    filterset : `Multiple_Filter`
        The set of filters (bands) the photometry is measured in.
    flux : `astropy.units.Quantity` or `astropy.utils.masked.Masked`
        Flux density measured in each filter, one element per filter in
        `filterset`.
    flux_errs : `astropy.units.Quantity` or `astropy.utils.masked.Masked`
        Uncertainty on `flux` for each filter.
    depths : `dict` of {`str`: `astropy.units.Magnitude`}, `astropy.units.Magnitude`, or `None`
        Limiting depth for each filter, in ABmag. If given as a `dict`
        keyed by filter name, it is converted internally to an array
        ordered to match `filterset`. May be `None` if depths are not
        available.

    Attributes
    ----------
    filterset : `Multiple_Filter`
        The set of filters the photometry is measured in.
    flux : `astropy.units.Quantity` or `astropy.utils.masked.Masked`
        Flux density in each filter.
    flux_errs : `astropy.units.Quantity` or `astropy.utils.masked.Masked`
        Flux density uncertainty in each filter.
    depths : `astropy.units.Magnitude` or `None`
        Limiting depth (in ABmag) for each filter.
    wav : `astropy.units.Quantity`
        Central wavelength of each filter in `filterset`, in Angstrom
        (property).

    Raises
    ------
    AssertionError
        If `depths` is a `dict` missing one of the filter names in
        `filterset`, if `depths` is not in ABmag units, or if `flux`,
        `flux_errs` and `depths` do not all have the same length as
        `filterset` (where not `None`).
    """

    def __init__(
        self: Self,
        filterset: Multiple_Filter,
        flux: Union[Masked, u.Quantity],
        flux_errs: Union[Masked, u.Quantity],
        depths: Optional[Union[Dict[str, u.Magnitude], u.Magnitude]],
    ):
        self.filterset = filterset
        self.flux = flux
        self.flux_errs = flux_errs
        if isinstance(depths, dict):
            assert all(filt_name in depths.keys() for filt_name in filterset.filt_names), \
                galfind_logger.critical(
                    f"not all {filterset.filt_names} in {depths.keys()=}"
                )
           
            depths = [depths[filt.filt_name] for filt in filterset]
            assert all(depth.unit == u.ABmag for depth in depths), \
                galfind_logger.critical(
                    f"not all depths are in ABmag: {depths=}"
                )
            depths = np.array([depth.value for depth in depths]) * u.ABmag
            
        self.depths = depths
        if self.depths is not None:
            assert self.depths.unit == u.ABmag, \
                galfind_logger.critical(
                    f"{depths.unit=} != 'ABmag'"
                )
        assert all(
            len(self.filterset) == len(getattr(self, name))
            for name in ["flux", "flux_errs", "depths"] 
            if getattr(self, name) is not None), \
                galfind_logger.critical(
                    f"Not all ['flux', 'flux_errs', 'depths'] are {len(self.filterset)=} or None!"
                )

    def __repr__(self) -> str:
        """Return the official string representation of the Photometry object."""
        n_bands = len(self.filterset)
        return f"{self.__class__.__name__}({n_bands} bands)"

    def __str__(self, print_cls_name: bool = True) -> str:
        output_str = ""
        if print_cls_name:
            output_str += funcs.line_sep
            output_str += f"{self.__class__.__name__.upper()}:\n"
            output_str += funcs.band_sep
        else:
            output_str += funcs.band_sep
        #if print_instrument:
        if hasattr(self, "instrument"):
            output_str += str(self.instrument)
        # Show brief summary for speed - avoid expensive min/max operations
        output_str += f"N FILTERS: {len(self.filterset)}\n"
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
                np.array(self.filterset.filt_names) \
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
    #         if split_property_name[0] in self.instrument.filt_names:
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
    #                 index = self.instrument.index_from_filt_name(band)
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
        """Central wavelength of each filter in `filterset`.

        Returns
        -------
        `astropy.units.Quantity`
            Array of per-filter central wavelengths, in Angstrom.
        """
        return np.array([filt.WavelengthCen.to(u.AA).value \
            for filt in self.filterset]) * u.AA

    # @classmethod
    # def from_fits_cat(cls, fits_cat_row, instrument, cat_creator):
    #     fluxes, flux_errs = cat_creator.load_photometry(
    #         fits_cat_row, instrument.filt_names
    #     )
    #     try:
    #         # local depths only currently works for one aperture diameter
    #         loc_depths = np.array(
    #             [
    #                 fits_cat_row[f"loc_depth_{filt_name}"].T[
    #                     cat_creator.aper_diam_index
    #                 ]
    #                 for filt_name in instrument.filt_names
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
            Type[Filter],
            Filter,
            Multiple_Filter,
            Self,
            List[Union[str, Filter, Multiple_Filter, Self]],
        ],
    ) -> Self:
        if isinstance(other, tuple(Filter.__subclasses__())) or isinstance(other, Filter):
            filterset_copy = deepcopy(self.filterset)
            filterset_copy -= other
            old_index = np.where(np.array(self.filterset.filt_names) == other.filt_name)[0][0]
            self.filterset = filterset_copy
            new_fluxes = np.delete(self.flux, old_index)
            new_flux_errs = np.delete(self.flux_errs, old_index)
            new_depths = np.delete(self.depths, old_index)
            self.flux = new_fluxes
            self.flux_errs = new_flux_errs
            self.depths = new_depths
        return self

    def crop(self: Type[Self], indices: Union[List[int], NDArray[int]]) -> Self:
        """Remove the filters at the given indices from the photometry.

        Builds a deep copy of `filterset` with the specified indices
        removed, and deletes the corresponding entries from `flux`,
        `flux_errs`, and `depths` on `self`.

        Parameters
        ----------
        indices : `list` of `int` or `numpy.ndarray` of `int`
            Indices of the filters/bands to remove.

        Returns
        -------
        `None`
            This method mutates `self` in place and does not return a
            value.
        """
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
        plot_errs: Dict[str, bool] = {"x": False, "y": True},
        #annotate=True,
        uplim_sigma: Optional[float] = 2.0,
        auto_scale: bool = True,
        errorbar_kwargs: Dict[str, Any] = {
            "ls": "",
            "marker": "o",
            "ms": 4.0,
            "zorder": 100.0,
            "elinewidth": 1.5,
            "path_effects": [pe.withStroke(linewidth=2.0, foreground="white")],
        },
        filled: bool = True,
        colour: str = "black",
        label: Optional[str] = "Photometry",
        return_extra: bool = False,
        log_scale: bool = False,
    ):
        """Plot the photometry as an errorbar plot of flux (or magnitude) vs. wavelength.

        Converts the stored flux and central wavelengths to the requested
        plotting units, optionally treats low-significance bands as upper
        limits (replacing their plotted value and error bar with an
        upper-limit arrow derived from the depth), optionally log-scales
        non-ABmag flux units, and optionally auto-scales the axes.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`, optional
            Axes to draw the errorbar plot on. Default is `None`.
        wav_units : `astropy.units.Unit`, optional
            Units to convert the plotted wavelengths to. Default is
            `astropy.units.AA`.
        mag_units : `astropy.units.Unit`, optional
            Units to convert the plotted flux/magnitude values to.
            Default is `astropy.units.Jy`.
        plot_errs : `dict` of {`str`: `bool`}, optional
            Whether to plot x ("x") and/or y ("y") error bars. Default
            is `{"x": False, "y": True}`.
        uplim_sigma : `float`, optional
            SNR threshold below which a band is plotted as a non-detection
            (upper limit) rather than a normal data point, with the
            plotted value derived from `uplim_sigma / 5` times the depth.
            If `None`, no band is treated as an upper limit. Default is
            `2.0`.
        auto_scale : `bool`, optional
            Whether to automatically set the x- and y-axis limits from
            the plotted data. Default is `True`.
        errorbar_kwargs : `dict`, optional
            Keyword arguments passed through to `ax.errorbar`. The
            `"mfc"`, `"color"`, and `"label"` entries are overwritten
            internally based on `filled`, `colour`, and `label`. Default
            sets `ls`, `marker`, `ms`, `zorder`, `elinewidth`, and a
            white-stroke `path_effects` outline.
        filled : `bool`, optional
            Whether markers are filled with `colour` (`True`) or drawn
            unfilled with `mfc="none"` (`False`). Default is `True`.
        colour : `str`, optional
            Marker and error bar colour. Default is `"black"`.
        label : `str`, optional
            Legend label for the errorbar series. Default is
            `"Photometry"`.
        return_extra : `bool`, optional
            If `True`, also return the intermediate wavelength, flux,
            error, and upper-limit arrays used to make the plot. Default
            is `False`.
        log_scale : `bool`, optional
            Whether to log10-scale the plotted flux values and errors.
            Only valid when `mag_units != astropy.units.ABmag`. Default
            is `False`.

        Returns
        -------
        `matplotlib.container.ErrorbarContainer`
            The object returned by `ax.errorbar`, if `return_extra` is
            `False`.
        `tuple`
            ``(plot, wavs_to_plot, mags_to_plot, yerr, uplims)`` if
            `return_extra` is `True`, where `plot` is the
            `ax.errorbar` return value, `wavs_to_plot` and
            `mags_to_plot` are the plotted wavelength/flux arrays,
            `yerr` is the plotted y-error array (or `None`), and
            `uplims` is a `list` of `bool` flagging which bands were
            plotted as upper limits.

        Raises
        ------
        AssertionError
            If `log_scale` is `True` and `mag_units` is
            `astropy.units.ABmag`, or if the fixed upper-limit arrow
            size (in sigma) is not smaller than `uplim_sigma`.
        """
        #breakpoint()
        if log_scale:
            assert mag_units != u.ABmag, galfind_logger.critical(
                f"Cannot log scale {mag_units=} == 'ABmag'"
            )
        # if not isinstance(self.flux.value, Masked):
        #     breakpoint()
        wavs_to_plot = funcs.convert_wav_units(self.wav, wav_units).value
        mags_to_plot = funcs.convert_mag_units(self.wav, self.flux, mag_units)
        # if not isinstance(mags_to_plot.value, Masked):
        #     mags_to_plot = np.ma.masked_array(mags_to_plot.value, mask=False)

        if uplim_sigma is None:
            uplims = list(np.full(len(self.flux), False))
            if plot_errs["y"]:
                yerr = np.array(
                    funcs.convert_mag_err_units(
                        self.wav,
                        self.flux,
                        [self.flux_errs, self.flux_errs],
                        mag_units,
                    )
                ) * mag_units
            else:
                yerr = None
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

            if plot_errs["y"]:
                mag_errs_new_units = funcs.convert_mag_err_units(
                    self.wav,
                    self.flux,
                    [self.flux_errs, self.flux_errs],
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
                    # 0.0 (not nan) since ax.errorbar() multiplies this side by
                    # zero for lolims/uplims points; nan * 0 = nan, which
                    # breaks the line connecting the point to the arrow
                    uplim_l1_vals = [0.0 for i in uplim_indices]
                else:
                    uplim_l1_vals = (uplim_vals - uplim_l1_vals).value
                    uplim_u1_vals = [0.0 for i in uplim_indices]
                yerr = []
                for i, uplim_errs in enumerate([uplim_l1_vals, uplim_u1_vals]):
                    mag_errs = mag_errs_new_units[i].value
                    mag_errs.put(uplim_indices, uplim_errs)
                    yerr.append(mag_errs * mag_units)
            else:
                yerr = None

        self.non_detected_indices = uplims

        # log scale y axis if not in units of ABmag
        if mag_units != u.ABmag:
            if log_scale:
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
            xlim_pad = funcs.convert_wav_units(0.15 * u.um, wav_units).value
            if plot_errs["x"]:
                lower_xlim = np.min(wavs_to_plot - xerr[0]) - xlim_pad
                upper_xlim = np.max(wavs_to_plot + xerr[1]) + xlim_pad
            else:
                lower_xlim = np.min(wavs_to_plot) - xlim_pad
                upper_xlim = np.max(wavs_to_plot) + xlim_pad
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
                if log_scale:
                    if plot_errs["y"]:
                        lower_ylim = np.nanmin(mags_to_plot - yerr[0]) - 0.15
                        upper_ylim = np.nanmax(mags_to_plot + yerr[1]) + 0.35
                    else:
                        lower_ylim = np.min(mags_to_plot) - 0.15
                        upper_ylim = np.max(mags_to_plot) + 0.35
                else:
                    if plot_errs["y"]:
                        lower_ylim = np.nanmin(mags_to_plot - yerr[0]) * 0.95
                        upper_ylim = np.nanmax(mags_to_plot + yerr[1]) * 1.05
                    else:
                        lower_ylim = np.min(mags_to_plot) * 0.95
                        upper_ylim = np.max(mags_to_plot) * 1.05
            try:
                ax.set_ylim(lower_ylim, upper_ylim)
            except:
                pass

        if mag_units == u.ABmag:
            plot_limits = {"lolims": uplims}
        else:
            plot_limits = {"uplims": uplims}

        # TODO: make nans very large and exclude from autoscaling
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
        n_scatter: int = 1,
        update: bool = False,
    ) -> Union[u.Quantity, Masked[u.Quantity]]:
        """Draw Monte Carlo flux realisations by Gaussian scattering each band.

        For each filter, draws `n_scatter` samples from a normal
        distribution with mean `flux` and standard deviation `flux_errs`.

        Parameters
        ----------
        n_scatter : `int`, optional
            Number of scattered flux realisations to draw per filter.
            Default is `1`.
        update : `bool`, optional
            If `True`, replace `self.flux` with the first scattered
            realisation. Default is `False`.

        Returns
        -------
        `astropy.units.Quantity` or `astropy.utils.masked.Masked`
            Array of scattered fluxes with shape ``(n_scatter,
            n_filters)``, in the same units as `flux`.

        Raises
        ------
        AssertionError
            If `flux` is in ABmag units.
        """
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
        if update:
            self.flux = scattered_fluxes[0]
        return scattered_fluxes

    def scatter(
        self: Self,
        n_scatter: int = 1,
    ) -> List[Photometry]:
        """Build new `Photometry` object(s) with Monte Carlo scattered fluxes.

        Parameters
        ----------
        n_scatter : `int`, optional
            Number of scattered flux realisations to generate. Default
            is `1`.

        Returns
        -------
        `Photometry`
            A single `Photometry` object with scattered fluxes, if
            `n_scatter` is `1`.
        `list` of `Photometry`
            A list of `Photometry` objects, one per scattered flux
            realisation, if `n_scatter` is greater than `1`.
        """
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
    
    def _update_errs_from_depths(
        self: Self,
        min_flux_pc_err: float,
    ) -> NoReturn:
        # calculate 1σ depths and convert to Jy
        one_sig_depths_Jy = self.depths.to(u.Jy) / 5

        assert min_flux_pc_err >= 0., galfind_logger.critical(f"Negative {min_flux_pc_err=}<0")
        # apply min_flux_pc_err criteria
        # TODO: Retain the flux mask in flux errors if there is one
        self.flux_errs = np.array(
            [
                depth
                if depth > flux * min_flux_pc_err / 100
                else flux * min_flux_pc_err / 100
                for flux, depth in zip(self.flux.value, one_sig_depths_Jy.value)
            ]
        ) * u.Jy


class Multiple_Photometry(ABC):
    """Collection of `Photometry` objects built from parallel per-source input arrays.

    Parameters
    ----------
    instrument_arr : `list` of `Instrument`
        Instrument/filterset object for each source.
    flux_arr : `list` of `astropy.units.Quantity`
        Per-filter flux values for each source.
    flux_errs_arr : `list` of `astropy.units.Quantity`
        Per-filter flux error values for each source.
    loc_depths_arr : `list`
        Per-filter local depths for each source.

    Attributes
    ----------
    phot_arr : `list` of `Photometry`
        One `Photometry` object per source, built from the
        corresponding elements of the input arrays.
    """

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
        """Construct a `Multiple_Photometry` from a FITS catalogue.

        Parameters
        ----------
        fits_cat : `astropy.table.Table`
            Catalogue table to load photometry from.
        instrument : `Instrument`
            Instrument definition used to determine available
            filters/bands.
        cat_creator : object
            Catalogue creator object providing `load_photometry` and
            `load_depths` methods.

        Returns
        -------
        `Multiple_Photometry`
            New instance populated with one `Photometry` per catalogue
            row, with per-row instrument objects restricted to the
            bands present for that row.
        """
        flux_arr, flux_errs_arr, gal_bands = cat_creator.load_photometry(
            fits_cat, instrument.filt_names
        )
        # local depths not yet loaded in
        depths_arr = cat_creator.load_depths(
            fits_cat, instrument.filt_names, gal_bands
        )
        instrument_arr = [
            deepcopy(instrument).remove_bands(
                [band for band in instrument if band.filt_name not in bands]
            )
            for bands in gal_bands
        ]
        return cls(instrument_arr, flux_arr, flux_errs_arr, depths_arr)


class Mock_Photometry(Photometry):
    """Simulated (mock) photometry for a single source, with flux errors derived from specified depths.

    Parameters
    ----------
    filterset : `Multiple_Filter`
        The set of filters the mock photometry is defined in.
    flux : `astropy.units.Quantity` or `astropy.units.Magnitude`
        Input (noiseless) flux for each filter.
    depths : `list` of `float`, `numpy.ndarray` of `float`, or `astropy.units.Magnitude`
        5σ limiting depths for each filter. Assumed to be in ABmag; if
        not already an `astropy.units.Magnitude` in ABmag, ABmag units
        are applied directly.
    min_flux_pc_err : `float` or `None`
        Minimum fractional flux error to apply, expressed as a
        percentage of the flux; used as a floor on the depth-derived
        error (see `flux_errs_from_depths`).

    Attributes
    ----------
    min_flux_pc_err : `float` or `None`
        The minimum percentage flux error applied when deriving
        `flux_errs`.

    Raises
    ------
    AssertionError
        If `flux` and `depths` do not have the same length.
    """

    def __init__(
        self: Self,
        filterset: Multiple_Filter,
        flux: Union[u.Quantity, u.Magnitude],
        depths: Union[List[float], NDArray[float], u.Magnitude],
        min_flux_pc_err: Optional[float],
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
        super().__init__(filterset, flux, flux_errs, depths)

    @staticmethod
    def flux_errs_from_depths(flux, depths, min_flux_pc_err):
        """Derive per-filter flux errors from 5σ ABmag depths, with a minimum fractional error floor.

        Parameters
        ----------
        flux : `astropy.units.Quantity`
            Flux value in each filter (used to evaluate the
            `min_flux_pc_err` floor).
        depths : `astropy.units.Magnitude`
            5σ limiting depths in ABmag for each filter.
        min_flux_pc_err : `float`
            Minimum flux error, as a percentage of `flux`; the 1σ
            depth-derived error is replaced by this floor wherever the
            floor is larger.

        Returns
        -------
        `astropy.units.Quantity`
            1σ flux error for each filter, in Jy.
        """
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

"""Spectral energy distribution (SED) representation and manipulation.

Provides classes for storing and manipulating SEDs as wavelength/magnitude/flux pairs
with unit conversion, plotting, and photometric summary calculations.
"""

from __future__ import annotations

import glob
import os
from abc import ABC
import astropy.io.ascii as ascii
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from typing import TYPE_CHECKING, Optional, Dict, Any, List, Union
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11
if TYPE_CHECKING:
    from . import Multiple_Filter
from . import (
    IGM_attenuation,
    Mock_Photometry,
    astropy_cosmo,
    config,
    galfind_logger,
)
from . import useful_funcs_austind as funcs
from .Emission_lines import line_diagnostics


class SED:
    """Base class representing a spectral energy distribution (SED).

    Stores paired arrays of wavelength and magnitude/flux values as
    :class:`astropy.units.Quantity` objects and provides unit conversion,
    plotting, and photometric summary calculations (bandpass-averaged
    fluxes, equivalent widths, UV slope inputs, UVJ colours) that are
    shared by both rest-frame and observed-frame subclasses.

    Parameters
    ----------
    wavs : array-like
        Wavelength values (without units attached yet).
    mags : array-like
        Magnitude or flux density values (without units attached yet).
    wav_units : `astropy.units.Unit`
        Units to attach to `wavs`.
    mag_units : `astropy.units.Unit`
        Units to attach to `mags`.

    Attributes
    ----------
    wavs : `astropy.units.Quantity`
        Wavelength array with units.
    mags : `astropy.units.Quantity`
        Magnitude/flux array with units.
    """

    # should include mag errors here
    def __init__(self, wavs, mags, wav_units, mag_units):
        self.wavs = wavs * wav_units
        self.mags = mags * mag_units
        # self.mag_units = mag_units

    def __repr__(self) -> str:
        """Return the official string representation of the SED object."""
        # Use first/last instead of min/max to avoid scanning entire array
        wav_range = f"{self.wavs[0]:.1f}-{self.wavs[-1]:.1f}"
        return f"SED({wav_range}, {len(self.wavs)} points)"

    def __str__(self):
        """Return a detailed string representation of the SED."""
        output_str = f"SED Object:\n"
        # Use first/last values instead of min/max to avoid scanning entire array
        output_str += f"  Wavelength range: {self.wavs[0]:.3f} - {self.wavs[-1]:.3f} {self.wavs.unit}\n"
        output_str += f"  Number of points: {len(self.wavs)}\n"
        if hasattr(self, 'file'):
            output_str += f"  File: {self.file}\n"
        if hasattr(self, 'z'):
            output_str += f"  Redshift: {self.z:.4f}\n"
        return output_str

    def convert_wav_units(self, units, update=True):
        """Convert the wavelength array to a new set of units.

        Parameters
        ----------
        units : `astropy.units.Unit`
            Target wavelength units.
        update : `bool`, optional
            Whether to overwrite `self.wavs` with the converted array.
            Default is `True`.

        Returns
        -------
        `astropy.units.Quantity`
            The wavelength array converted to `units`.
        """
        wavs = self.wavs.to(units)
        if update:
            self.wavs = wavs
        return wavs

    def convert_mag_units(self, units, update=True):
        """Convert the magnitude/flux array to a new set of units.

        Handles conversion between AB magnitudes, spectral flux density
        (f_nu, e.g. Jy) and power density per unit wavelength (f_lambda,
        e.g. erg / s / cm^2 / AA), using `self.wavs` to apply the
        appropriate spectral density equivalency where required.

        Parameters
        ----------
        units : `astropy.units.Unit`
            Target magnitude/flux units. Must be `astropy.units.ABmag` or
            have physical type 'spectral flux density',
            'ABmag/spectral flux density', or
            'power density/spectral flux density wav'.
        update : `bool`, optional
            Whether to overwrite `self.mags` with the converted array.
            Default is `True`.

        Returns
        -------
        `astropy.units.Quantity` or `astropy.units.Magnitude`
            The magnitude/flux array converted to `units`.

        Raises
        ------
        Exception
            If `units` is not `ABmag` and does not have a recognised
            spectral flux density physical type.
        """
        if units == self.mags.unit:
            mags = self.mags
        elif units == u.ABmag:
            if u.get_physical_type(self.mags.unit) in [
                "ABmag/spectral flux density",
                "spectral flux density",
            ]:  # f_ν -> derivative of u.Jy
                mags = self.mags.to(u.ABmag)
            elif (
                u.get_physical_type(self.mags.unit)
                == "power density/spectral flux density wav"
            ):  # f_λ -> derivative of u.erg / (u.s * (u.cm ** 2) * u.AA)
                mags = self.mags.to(
                    u.ABmag, equivalencies=u.spectral_density(self.wavs)
                )
        elif u.get_physical_type(units) in [
            "ABmag/spectral flux density",
            "spectral flux density",
        ]:  # f_ν -> derivative of u.Jy
            if self.mags.unit == u.ABmag:
                mags = self.mags.to(units)
            elif (
                u.get_physical_type(self.mags.unit)
                == "power density/spectral flux density wav"
                or u.get_physical_type(self.mags.unit)
                == "ABmag/spectral flux density"
            ):
                mags = self.mags.to(
                    units, equivalencies=u.spectral_density(self.wavs)
                )
        elif (
            u.get_physical_type(units)
            == "power density/spectral flux density wav"
        ):  # f_λ -> derivative of u.erg / (u.s * (u.cm ** 2) * u.AA):
            # if self.mags.unit == u.ABmag:
            mags = self.mags.to(
                units, equivalencies=u.spectral_density(self.wavs)
            )
            # elif u.get_physical_type(self.mags.unit) == "spectral flux density": # f_ν -> derivative of u.Jy
            # mags = self.mags.to(units, equivalencies = u.spectral_density(self.wavs))
        else:
            raise (
                Exception(
                    "Units must be either ABmag or have physical units of 'spectral flux density' or 'power density/spectral flux density wav'!"
                )
            )
        if update:
            self.mags = mags
        return mags

    def plot(
        self: Self,
        ax: Optional[plt.Axes] = None,
        wav_units: u.Unit = u.AA,
        mag_units: u.Unit = u.ABmag,
        label: Optional[str] = None,
        annotate: bool = True,
        save_name: Optional[str] = None,
        save_dir: Optional[str] = f"{config['Other']['PLOT_DIR']}/SEDs",
        log_fluxes: bool = True,
        plot_kwargs: Dict[str, Any] = {},
        legend_kwargs: Dict[str, Any] = {},
    ):
        """Plot the SED as flux/magnitude versus wavelength.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`, optional
            Axes to plot on. If `None`, a new figure and axes are created.
            Default is `None`.
        wav_units : `astropy.units.Unit`, optional
            Units to convert the wavelength axis to. Default is `astropy.units.AA`.
        mag_units : `astropy.units.Unit`, optional
            Units to convert the flux/magnitude axis to. Default is `astropy.units.ABmag`.
        label : `str`, optional
            Legend label for the plotted line. Overridden by
            `self.template_name` if the SED has one. Default is `None`.
        annotate : `bool`, optional
            Whether to set axis labels and draw the legend. Default is `True`.
        save_name : `str`, optional
            If given, the filename (png or pdf) to save the plot to.
            Default is `None`.
        save_dir : `str`, optional
            Directory to save the plot in when `save_name` is given.
            Default is `f"{config['Other']['PLOT_DIR']}/SEDs"`.
        log_fluxes : `bool`, optional
            Whether to log-scale flux values when `mag_units` is not
            `ABmag`. Default is `True`.
        plot_kwargs : `dict`, optional
            Extra keyword arguments passed to `ax.plot`. Default is `{}`.
        legend_kwargs : `dict`, optional
            Extra keyword arguments passed to `ax.legend`. Default is `{}`.

        Returns
        -------
        `list` of `matplotlib.lines.Line2D`
            The line artist(s) returned by `ax.plot`.
        """
        if ax is None:
            fig, ax = plt.subplots()

        wavs = funcs.convert_wav_units(self.wavs, wav_units)
        mags = funcs.convert_mag_units(self.wavs, self.mags, mag_units)

        if mag_units != u.ABmag and log_fluxes:
            mags = funcs.log_scale_fluxes(mags)
        else:
            mags = mags.value

        if label is not None and hasattr(self, "template_name"):
            label = self.template_name

        plot = ax.plot(wavs.value, mags, label=label, **plot_kwargs)
        if annotate:
            ax.set_xlabel(
                funcs.label_wavelengths(
                    wav_units,
                    False,
                    ""
                    if self.__class__.__name__.split("_")[-1] == "Photometry"
                    else self.__class__.__name__.split("_")[-1],
                )
            )
            ax.set_ylabel(
                funcs.label_fluxes(
                    mag_units, True if mag_units != u.ABmag and log_fluxes else False
                )
            )
            ax.legend(**legend_kwargs)
        if save_name is not None:
            # save png by default
            if save_name.split(".")[-1] not in ["png", "pdf"]:
                save_name = f"{'.'.join(save_name.split('.')[-1:])}.png"
            if save_dir is not None:
                save_path = f"{save_dir}/{save_name}"
            else:
                save_path = save_name
            funcs.make_dirs(save_path)
            plt.savefig(save_path)
            funcs.change_file_permissions(save_path)
            galfind_logger.info(f"Saved {repr(self)} plot to {save_path}")
        return plot

    def calc_bandpass_averaged_flux(
        self: Self,
        filter_wavs: Union[u.Quantity, u.Dex],
        filter_trans: Union[u.Quantity, u.Magnitude, u.Dex],
        detector_type: str = "photon",
    ):
        """Calculate the bandpass-averaged flux of the SED through a filter.

        Interpolates the SED (in f_lambda) onto the filter's wavelength
        grid and computes the transmission-weighted average flux, using
        either photon-counting or energy-integrating detector weighting.

        Parameters
        ----------
        filter_wavs : `astropy.units.Quantity` or `astropy.units.Dex`
            Wavelength grid of the filter transmission profile.
        filter_trans : `astropy.units.Quantity`, `astropy.units.Magnitude`, or `astropy.units.Dex`
            Filter transmission values corresponding to `filter_wavs`.
        detector_type : `str`, optional
            Either `"photon"` (weights by lambda * T(lambda)) or
            `"energy"` (weights by T(lambda) only). Falls back to
            `"photon"` with a warning if an unrecognised value is given.
            Default is `"photon"`.

        Returns
        -------
        `float`
            The bandpass-averaged flux in f_lambda units (erg / s / cm^2 / AA).
        """
        detector_type = detector_type.lower()
        if detector_type not in ["photon", "energy"]:
            galfind_logger.warning(
                f"Cannot calculate bandpass-averaged flux for {detector_type=}, assuming 'photon'"
            )
            detector_type = "photon"
        wavs = funcs.convert_wav_units(self.wavs, u.AA).value
        mags = funcs.convert_mag_units(
            self.wavs, self.mags, u.erg / (u.s * u.AA * u.cm**2)
        ).value
        # update filter wavelengths to correct units
        filter_wavs = funcs.convert_wav_units(filter_wavs, u.AA).value
        # interpolate SED to be on same grid as filter_profile
        sed_interp = interp1d(wavs, mags, fill_value="extrapolate")(filter_wavs) # in f_lambda
        if detector_type == "photon":
            # calculate integral(λ * f(λ) * T(λ) dλ)
            numerator = np.trapz(filter_wavs * sed_interp * filter_trans, x = filter_wavs)
            # calculate integral(λ * T(λ) dλ)
            denominator = np.trapz(filter_wavs * filter_trans, x = filter_wavs)
        else: # detector_type == "energy"
            # calculate integral(f(λ) * T(λ) dλ)
            numerator = np.trapz(sed_interp * filter_trans, x = filter_wavs)
            # calculate integral(T(λ) dλ)
            denominator = np.trapz(filter_trans, x = filter_wavs)
        # calculate bandpass-averaged flux in Jy
        return numerator / denominator

    def calc_line_EW(
        self: Self,
        line_name: str,
        plot: bool = False,
    ):  # ONLY WORKS FOR EMISSION LINES, NOT ABSORPTION AT THIS POINT
        """Calculate the rest-frame equivalent width of an emission line.

        Only works for emission (not absorption) features. The continuum
        is estimated by interpolating across the line-adjacent continuum
        windows defined in `line_diagnostics`, and the line flux is
        integrated only where it exceeds the continuum (negative
        excursions are clipped to zero). Results are cached on `self`
        under `line_EWs`, `line_fluxes`, and `line_cont`.

        Parameters
        ----------
        line_name : `str`
            Name of the emission line, must be a key in
            `galfind.Emission_lines.line_diagnostics`.
        plot : `bool`, optional
            If `True`, plot the masked line-feature flux before returning.
            Default is `False`.

        Returns
        -------
        `astropy.units.Quantity`
            The (rest-frame) equivalent width of the line.

        Raises
        ------
        Exception
            If called on an SED whose class name does not contain
            `"rest"` or `"obs"`.
        """
        wavs_AA = self.convert_wav_units(u.AA, update=True)
        flux_lambda = self.convert_mag_units(
            u.erg / (u.s * u.cm**2 * u.AA), update=False
        )
        # calculate line + continuum flux
        if "rest" in type(self).__name__:
            line_lims = line_diagnostics[line_name]["feature_wavs"]
            cont_lims = line_diagnostics[line_name]["cont_wavs"]
        elif "obs" in type(self).__name__:
            line_lims = line_diagnostics[line_name]["feature_wavs"] * (
                1 + self.z
            )
            cont_lims = line_diagnostics[line_name]["cont_wavs"] * (1 + self.z)
        else:
            raise (
                Exception(
                    "Attempted EW calculation in a class not containing 'rest' or 'obs' in the class name!"
                )
            )
        # mask everything but the line of interest
        feature_mask = (wavs_AA > line_lims[0].to(u.AA)) & (
            wavs_AA < line_lims[1].to(u.AA)
        )
        if plot:
            plt.plot(wavs_AA[feature_mask], flux_lambda[feature_mask])
            plt.show()
        # line_plus_cont_flux = np.trapz(
        #     flux_lambda[feature_mask], x=wavs_AA[feature_mask]
        # )
        # calculate continuum flux and mean continuum level
        cont_mask = np.logical_or.reduce(
            np.array(
                [
                    (
                        (wavs_AA > lims[0].to(u.AA))
                        & (wavs_AA < lims[1].to(u.AA))
                    )
                    for lims in cont_lims
                ]
            )
        )
        # mask everything but the line of interest
        cont_flux = (
            interp1d(
                wavs_AA[cont_mask],
                flux_lambda[cont_mask],
                fill_value="extrapolate",
            )(wavs_AA[feature_mask])
            * u.erg
            / (u.s * u.cm**2 * u.AA)
        )
        mean_cont = np.trapz(
            cont_flux, x=wavs_AA[feature_mask]
        )  # not 100% correct here in the case of cont > line flux
        # calculate line flux
        # if line flux goes negative at any point, set to zero (only works for emission and not absorption)
        line_flux_integrand = (
            np.array(
                [
                    (flux_lambda[feature_mask][i] - cont_flux[i]).value
                    if (flux_lambda[feature_mask][i] - cont_flux[i]).value
                    > 0.0
                    else 0.0
                    for i, x in enumerate(wavs_AA[feature_mask])
                ]
            )
            * u.erg
            / (u.s * u.cm**2 * u.AA)
        )
        line_flux = np.trapz(
            line_flux_integrand, x=wavs_AA[feature_mask]
        )  # (line_plus_cont_flux - cont_flux) * feature_width # emission == positive EW
        # calculate line EW
        line_EW = np.trapz(
            line_flux_integrand / cont_flux, x=wavs_AA[feature_mask]
        )
        # save result in self
        if not hasattr(self, "line_EWs"):
            self.line_EWs = {line_name: line_EW}
            self.line_fluxes = {line_name: line_flux}
            self.line_cont = {line_name: mean_cont}
        else:
            self.line_EWs[line_name] = line_EW
            self.line_fluxes[line_name] = line_flux
            self.line_cont[line_name] = mean_cont
        return line_EW

    def calc_xi_ion(
        self: Self,
        line_name: str = "Halpha"
    ) -> u.Quantity:
        """Calculate the ionizing photon production efficiency, xi_ion.

        Derives xi_ion = L(line) / (L_UV * 1.36e-12) assuming case B
        recombination at T = 10^4 K and n_e = 10^2 cm^-3, using the
        line luminosity (from `calc_line_EW`) and the UV continuum
        luminosity at rest-frame 1500 AA (from `calc_mUV`). No dust
        correction is applied.

        Parameters
        ----------
        line_name : `str`, optional
            Name of the recombination line to use. Default is `"Halpha"`.

        Returns
        -------
        `astropy.units.Quantity`
            The ionizing photon production efficiency, in Hz / erg.
        """
        # Note that no dust correction is applied here
        self.calc_line_EW(line_name, plot = False)
        Ha_flux = self.line_fluxes[line_name]
        # calculate mUV
        mUV = self.calc_mUV()
        # convert mUV to L_UV in erg/s/Hz
        dL = astropy_cosmo.luminosity_distance(self.z).to(u.cm)
        L_UV = 4 * np.pi * dL ** 2 \
            * funcs.convert_mag_units(
                funcs.wav_rest_to_obs(1_500.0 * u.AA, self.z),
                mUV,
                u.erg / (u.s * u.Hz * u.cm**2)
            ) * (1.0 + self.z)
        # convert Ha_flux to L_Ha in erg/s
        L_Ha = 4 * np.pi * dL ** 2 * Ha_flux / (1.0 + self.z)
        xi_ion = (L_Ha / (L_UV * 1.36e-12)).value * u.Hz / u.erg # assuming case B recombination, T = 10^4 K, n_e = 10^2 cm^-3
        return xi_ion

    def calc_UVJ_colours(self, resolution=1.0 * u.AA):
        """Calculate rest-frame U-V and V-J colours using top-hat filters.

        Builds synthetic top-hat U, V and J filters (fixed rest-frame
        effective wavelengths and FWHMs), computes the bandpass-averaged
        flux of the SED through each, and stores the resulting fluxes
        and colours on `self`.

        Parameters
        ----------
        resolution : `astropy.units.Quantity`, optional
            Wavelength resolution used to sample each top-hat filter.
            Default is `1.0 * astropy.units.AA`.

        Returns
        -------
        `None`
            Results are stored as `self.UVJ_fluxes` and `self.UVJ_colours`.
        """
        UVJ_filters = {
            "U": {"lam_eff": 3_650.0 * u.AA, "lam_fwhm": 660.0 * u.AA},
            "V": {"lam_eff": 5_510.0 * u.AA, "lam_fwhm": 880.0 * u.AA},
            "J": {"lam_eff": 12_200.0 * u.AA, "lam_fwhm": 2130.0 * u.AA},
        }  # these are rest frame wavelengths

        bp_averaged_fluxes = np.zeros(3)
        for i, band in enumerate(["U", "V", "J"]):
            galfind_logger.info(f"Calculating {band} fluxes")
            band_wavs = (
                np.linspace(
                    (
                        UVJ_filters[band]["lam_eff"]
                        - UVJ_filters[band]["lam_fwhm"] / 2.0
                    ),
                    (
                        UVJ_filters[band]["lam_eff"]
                        + UVJ_filters[band]["lam_fwhm"] / 2.0
                    ),
                    int(
                        np.round(
                            (UVJ_filters[band]["lam_fwhm"] / resolution)
                            .to(u.dimensionless_unscaled)
                            .value
                        )
                    ),
                )
                * u.AA
            )
            filter_profile = {
                "Wavelength": band_wavs,
                "Transmission": np.ones(len(band_wavs)),
            }
            bp_averaged_fluxes[i] = self.calc_bandpass_averaged_flux(filter_profile)
        # convert bp_averaged_fluxes to Jy
        bp_averaged_fluxes_Jy = funcs.convert_mag_units(
            [UVJ_filters[band]["lam_eff"] for band in ["U", "V", "J"]],
            bp_averaged_fluxes * u.erg / (u.s * (u.cm**2) * u.AA),
            u.Jy,
        )

        self.UVJ_fluxes = {
            band: flux
            for band, flux in zip(["U", "V", "J"], bp_averaged_fluxes_Jy)
        }
        self.UVJ_colours = {
            "U-V": -2.5
            * np.log10(self.UVJ_fluxes["U"] / self.UVJ_fluxes["V"]),
            "V-J": -2.5
            * np.log10(self.UVJ_fluxes["V"] / self.UVJ_fluxes["J"]),
        }


class SED_rest(SED):
    """A rest-frame spectral energy distribution.

    Parameters
    ----------
    wavs : array-like
        Rest-frame wavelength values (without units attached yet).
    mags : array-like
        Magnitude or flux density values (without units attached yet).
    wav_units : `astropy.units.Unit`
        Units to attach to `wavs`.
    mag_units : `astropy.units.Unit`
        Units to attach to `mags`.
    wav_range : `astropy.units.Quantity`, optional
        Rest-frame wavelength range to crop `wavs`/`mags` to.
        Default is `[0, 10_000] * astropy.units.AA`.
    """

    # should include mag errors here
    def __init__(
        self, wavs, mags, wav_units, mag_units, wav_range=[0, 10_000] * u.AA
    ):
        try:
            wavs = wavs.value # if wavs is in Angstrom
        except:
            pass
        mags = mags[
            (wavs > wav_range.to(wav_units).value[0])
            & (wavs < wav_range.to(wav_units).value[1])
        ]
        wavs = wavs[
            (wavs > wav_range.to(wav_units).value[0])
            & (wavs < wav_range.to(wav_units).value[1])
        ]
        super().__init__(wavs, mags, wav_units, mag_units)
    
    @classmethod
    def from_SED_obs(cls, SED_obs, out_wav_units=u.AA, out_mag_units=u.ABmag):
        """Construct a rest-frame SED from an observed-frame SED.

        De-redshifts the wavelength array of `SED_obs` using its `z`
        attribute.

        Parameters
        ----------
        SED_obs : `SED_obs`
            The observed-frame SED to de-redshift.
        out_wav_units : `astropy.units.Unit`, optional
            Wavelength units of the returned SED. Default is `astropy.units.AA`.
        out_mag_units : `astropy.units.Unit`, optional
            Magnitude/flux units of the returned SED. Default is `astropy.units.ABmag`.

        Returns
        -------
        `SED_rest`
            The corresponding rest-frame SED.
        """
        wavs = funcs.convert_wav_units(
            SED_obs.wavs / (1 + SED_obs.z), out_wav_units
        )
        mags = funcs.convert_mag_units(wavs, SED_obs.mags, out_mag_units)
        return cls(
            wavs.value,
            mags.value,
            wavs.unit,
            mags.unit,
            wav_range=[0, 10_000] * u.AA,
        )


class SED_obs(SED):
    """An observed-frame spectral energy distribution.

    Parameters
    ----------
    z : `float`
        Redshift of the SED, used to relate rest- and observed-frame
        quantities.
    wavs : array-like
        Observed-frame wavelength values (without units attached yet).
    mags : array-like
        Magnitude or flux density values (without units attached yet).
    wav_units : `astropy.units.Unit`
        Units to attach to `wavs`.
    mag_units : `astropy.units.Unit`
        Units to attach to `mags`.

    Attributes
    ----------
    z : `float`
        Redshift of the SED.
    """

    # should include mag errors here
    def __init__(self, z, wavs, mags, wav_units, mag_units):
        self.z = z
        super().__init__(wavs, mags, wav_units, mag_units)

    @classmethod
    def from_SED_rest(cls, z_int, SED_rest):
        """Construct an observed-frame SED from a rest-frame SED.

        Redshifts the wavelength array of `SED_rest` to the observed
        frame.

        Parameters
        ----------
        z_int : `float`
            Redshift to place the rest-frame SED at.
        SED_rest : `SED_rest`
            The rest-frame SED to redshift.

        Returns
        -------
        `SED_obs`
            The corresponding observed-frame SED, with magnitudes in
            AB mag.
        """
        wav_obs = funcs.wav_rest_to_obs(SED_rest.wavs, z_int)
        mag_obs = funcs.convert_mag_units(
            SED_rest.wavs, SED_rest.mags, u.ABmag
        )
        return cls(
            z_int, wav_obs.value, mag_obs.value, SED_rest.wavs.unit, u.ABmag
        )

    def create_mock_photometry(
        self: Self,
        filterset: Multiple_Filter,
        depths: Optional[u.Quantity] = None,
        min_flux_pc_err: float = 10.0,
    ) -> Mock_Photometry:
        """Create mock photometry by integrating the SED through a filterset.

        Computes the bandpass-averaged flux of the SED in each filter of
        `filterset` and wraps the result in a `Mock_Photometry` object,
        which is also cached on `self.mock_photometry`.

        Parameters
        ----------
        filterset : `Multiple_Filter`
            The set of filters to synthesize photometry through.
        depths : `astropy.units.Quantity`, optional
            Per-band depths to assume for the mock photometry. If `None`,
            depths of 99 ABmag are assumed (i.e. very well detected).
            Default is `None`.
        min_flux_pc_err : `float`, optional
            Minimum fractional flux error (percent) to assume in the mock
            photometry. Default is `10.0`.

        Returns
        -------
        `Mock_Photometry`
            The synthesized mock photometry, also stored as
            `self.mock_photometry`.
        """
        # if depths not given, expect the galaxy to be very well detected
        if depths is None:
            depths = np.full(len(filterset), 99.0) * u.ABmag
        # elif isinstance(depths, dict):
        #     depths = [depth for (band, depth) in depths.items()]
        detector_types = [
            getattr(filt, "DetectorType").split(" ")[0]
            if hasattr(filt, "DetectorType") else "photon" 
            for filt in filterset
        ]
        bp_averaged_fluxes = (
            [
                self.calc_bandpass_averaged_flux(filt.wav, filt.trans, detector_type)
                for filt, detector_type in zip(filterset, detector_types)
            ]
            * u.erg
            / (u.s * (u.cm**2) * u.AA)
        )
        # convert bp_averaged_fluxes to Jy
        band_wavs = (
            np.array([filt.WavelengthCen.value for filt in filterset]) * u.AA
        )
        bp_averaged_fluxes_Jy = funcs.convert_mag_units(
            band_wavs, bp_averaged_fluxes, u.Jy
        )
        self.mock_photometry = Mock_Photometry(
            filterset, bp_averaged_fluxes_Jy, depths, min_flux_pc_err
        )
        return self.mock_photometry

    def calc_colour(self, filters, depths=[]):
        """Calculate the AB magnitude colour between two filters.

        Parameters
        ----------
        filters : `list`
            A two-element list/array of filter objects (blue, red) to
            compute the bandpass-averaged flux/magnitude in.
        depths : `list` or `dict`, optional
            Per-band depths; unused beyond validating the interface.
            If empty, depths of 99.0 are assumed. Default is `[]`.

        Returns
        -------
        `astropy.units.Magnitude`
            The colour (blue filter magnitude minus red filter
            magnitude).

        Raises
        ------
        AssertionError
            If `filters` is not a list/array of length 2.
        """
        assert type(filters) in [np.array, list]
        assert len(filters) == 2
        if type(depths) == dict:
            depths = [depth for (band, depth) in depths.items()]
        if (
            depths == []
        ):  # if depths not given, expect the galaxy to be very well detected
            depths = [99.0 for band in filters]

        if hasattr(filters[0], "DetectorType"):
            blue_detector_type = getattr(filters[0], "DetectorType").split(" ")[0]
        else:
            blue_detector_type = "photon"
        if hasattr(filters[1], "DetectorType"):
            red_detector_type = getattr(filters[1], "DetectorType").split(" ")[0]
        else:
            red_detector_type = "photon"
        if blue_detector_type != red_detector_type:
            galfind_logger.warning(
                "Detector types for blue and red filters are different. Assuming photon counting for both."
            )
            blue_detector_type = "photon"
            red_detector_type = "photon"

        blue_flux = (
            self.calc_bandpass_averaged_flux(filters[0].wav, filters[0].trans, blue_detector_type)
            * u.erg
            / (u.s * (u.cm**2) * u.AA)
        )
        blue_flux_mAB = funcs.convert_mag_units(
            np.array(filters[0].WavelengthCen.value) * u.AA, blue_flux, u.ABmag
        )
        red_flux = (
            self.calc_bandpass_averaged_flux(filters[1].wav, filters[1].trans, red_detector_type)
            * u.erg
            / (u.s * (u.cm**2) * u.AA)
        )
        red_flux_mAB = funcs.convert_mag_units(
            np.array(filters[1].WavelengthCen.value) * u.AA, red_flux, u.ABmag
        )
        return blue_flux_mAB - red_flux_mAB

    def calc_mUV(
        self: Self,
        wav_range: u.Quantity = [1_450.0, 1_550.0] * u.AA,
        wav_resolution: u.Quantity = 1.0 * u.AA,
    ):
        """Calculate the observed-frame apparent UV magnitude, m_UV.

        Builds a rest-frame top-hat filter spanning `wav_range`, redshifts
        it to the observed frame using `self.z`, and computes the
        bandpass-averaged AB magnitude of the SED through it.

        Parameters
        ----------
        wav_range : `astropy.units.Quantity`, optional
            Rest-frame wavelength range defining the top-hat filter.
            Default is `[1_450.0, 1_550.0] * astropy.units.AA`.
        wav_resolution : `astropy.units.Quantity`, optional
            Wavelength resolution used to sample the top-hat filter.
            Default is `1.0 * astropy.units.AA`.

        Returns
        -------
        `astropy.units.Magnitude`
            The apparent UV magnitude, m_UV.

        Raises
        ------
        AssertionError
            If `wav_range[0]` is not less than `wav_range[1]`.
        """
        assert wav_range[0] < wav_range[1], \
            galfind_logger.critical(
                f"{wav_range[0]=}!<{wav_range[1]=}"
            )
        # create tophat filter in rest frame
        from galfind import Tophat_Filter
        obs_wav_range = wav_range * (1. + self.z)
        #wavs = np.arange(obs_wav_range[0].value, obs_wav_range[1].value, wav_resolution.value) * wav_range.unit
        mUV_filter = Tophat_Filter("mUV", obs_wav_range[0], obs_wav_range[1], wav_resolution)
        UV_flux = self.calc_bandpass_averaged_flux(mUV_filter.wav, mUV_filter.trans) * u.erg / (u.s * (u.cm**2) * u.AA)
        # convert to m_AB
        return funcs.convert_mag_units(
            mUV_filter.WavelengthCen,
            UV_flux,
            u.ABmag,
        )

    def calc_MUV(
        self: Self,
        wav_range: u.Quantity = [1_450.0, 1_550.0] * u.AA,
        wav_resolution: u.Quantity = 1.0 * u.AA,
    ):
        """Calculate the rest-frame absolute UV magnitude, M_UV.

        Converts the apparent UV magnitude from `calc_mUV` to an absolute
        magnitude using the luminosity distance at `self.z`.

        Parameters
        ----------
        wav_range : `astropy.units.Quantity`, optional
            Rest-frame wavelength range defining the top-hat filter used
            in `calc_mUV`. Default is `[1_450.0, 1_550.0] * astropy.units.AA`.
        wav_resolution : `astropy.units.Quantity`, optional
            Wavelength resolution used to sample the top-hat filter.
            Default is `1.0 * astropy.units.AA`.

        Returns
        -------
        `float`
            The absolute UV magnitude, M_UV.
        """
        mUV = self.calc_mUV(wav_range, wav_resolution)
        dL = astropy_cosmo.luminosity_distance(self.z).to(u.pc).value
        MUV = mUV.value - 5 * np.log10(dL / 10.0) + 2.5 * np.log10(1.0 + self.z)
        return MUV # * u.ABmag


class Mock_SED_rest(SED_rest):  # , Mock_SED):
    """A rest-frame SED representing a mock/template (rather than fitted) spectrum.

    Extends `SED_rest` with a template name and metadata, and adds
    classmethod loaders for various theoretical template libraries
    (EAZY, Bagpipes, Yggdrasil Pop III, BPASS) as well as power-law
    and normalization utilities.

    Parameters
    ----------
    wavs : array-like
        Rest-frame wavelength values (without units attached yet).
    mags : array-like
        Magnitude or flux density values (without units attached yet).
    wav_units : `astropy.units.Unit`
        Units to attach to `wavs`.
    mag_units : `astropy.units.Unit`
        Units to attach to `mags`.
    template_name : `str`, optional
        Name/identifier of the template. Default is `None`.
    meta : `dict`, optional
        Arbitrary metadata associated with the template (e.g. age,
        metallicity). Default is `None`.

    Attributes
    ----------
    template_name : `str` or `None`
        Name/identifier of the template.
    meta : `dict` or `None`
        Metadata associated with the template.
    """

    def __init__(
        self, wavs, mags, wav_units, mag_units, template_name=None, meta=None
    ):
        self.template_name = template_name
        self.meta = meta
        super().__init__(wavs, mags, wav_units, mag_units)

    @classmethod
    def from_Mock_SED_obs(
        cls, mock_SED_obs, out_wav_units=u.AA, out_mag_units=u.ABmag, IGM=None
    ):
        """Construct a rest-frame mock SED from an observed-frame mock SED.

        Parameters
        ----------
        mock_SED_obs : `Mock_SED_obs`
            The observed-frame mock SED to de-redshift.
        out_wav_units : `astropy.units.Unit`, optional
            Wavelength units of the returned SED. Default is `astropy.units.AA`.
        out_mag_units : `astropy.units.Unit`, optional
            Magnitude/flux units of the returned SED. Default is `astropy.units.ABmag`.
        IGM : `galfind.IGM_attenuation.IGM`, optional
            Unused placeholder for future IGM un-attenuation support.
            Default is `None`.

        Returns
        -------
        `Mock_SED_rest`
            The corresponding rest-frame mock SED, inheriting the
            template name of `mock_SED_obs`.
        """
        wavs = funcs.convert_wav_units(
            mock_SED_obs.wavs / (1 + mock_SED_obs.z), out_wav_units
        )
        mags = funcs.convert_mag_units(wavs, mock_SED_obs.mags, out_mag_units)
        # ensure IGM output is of the correct type
        mock_sed_rest_obj = cls(
            wavs.value,
            mags.value,
            wavs.unit,
            mags.unit,
            mock_SED_obs.template_name,
        )
        # if IGM_out == None:
        #     mock_sed_rest_obj.un_attenuate_IGM(mock_SED_obs.z, mock_SED_obs.IGM)
        # elif isinstance(IGM_out, IGM_attenuation.IGM):
        #     if IGM_out.prescription != mock_SED_obs.IGM.prescription:
        #         raise(Exception("Not currently included the functionality to swap IGM attenuation whilst creating new Mock_SED_rest object from Mock_SED_obs object yet"))
        # else:
        #     raise(Exception(f"'IGM_out' = {IGM_out} must be either 'None' or 'IGM' class"))
        return mock_sed_rest_obj

    @classmethod
    def power_law_from_beta_m_UV(
        cls,
        beta,
        m_UV,
        wav_range=[912.0, 10_000.0] * u.AA,
        wav_res=1.0 * u.AA,
        template_name=None,
    ):
        """Construct a power-law mock rest-frame SED, f_lambda ~ lambda^beta.

        Parameters
        ----------
        beta : `float`
            The UV continuum power-law slope index.
        m_UV : `astropy.units.Quantity` or `astropy.units.Magnitude`
            Apparent UV magnitude to normalize the resulting SED to.
        wav_range : `astropy.units.Quantity`, optional
            Rest-frame wavelength range to construct the SED over.
            Default is `[912.0, 10_000.0] * astropy.units.AA`.
        wav_res : `astropy.units.Quantity`, optional
            Wavelength resolution/step size. Default is `1.0 * astropy.units.AA`.
        template_name : `str`, optional
            Name/identifier for the resulting SED. Default is `None`.

        Returns
        -------
        `Mock_SED_rest`
            The normalized power-law rest-frame SED.
        """
        wavs = (
            np.linspace(
                wav_range[0].value,
                wav_range[1].value,
                int(
                    ((wav_range[1] - wav_range[0]) / wav_res)
                    .to(u.dimensionless_unscaled)
                    .value
                ),
            )
            * wav_range.unit
        )
        mags = funcs.convert_wav_units(wavs, u.AA).value ** beta
        mock_sed = cls(
            wavs.value,
            mags,
            u.AA,
            u.erg / (u.s * u.AA * u.cm**2),
            template_name=template_name,
        )
        mock_sed.normalize_to_m_UV(m_UV)
        return mock_sed

    @classmethod
    def load_SED_in_template(
        cls, code_name, m_UV, template_set, template_number
    ):
        """Load a single rest-frame template SED for a given fitting code.

        Dispatches to `load_EAZY_in_template` or `load_pipes_in_template`
        depending on `code_name`.

        Parameters
        ----------
        code_name : `str`
            Name of the SED-fitting code the template belongs to, either
            `"EAZY"` or `"Bagpipes"`.
        m_UV : `astropy.units.Quantity` or `astropy.units.Magnitude`
            Apparent UV magnitude to normalize the template to.
        template_set : `str`
            Name of the template set to load from.
        template_number : `int` or `str`
            Index or filename of the template within the set.

        Returns
        -------
        `Mock_SED_rest`
            The loaded and normalized template SED.

        Raises
        ------
        Exception
            If `code_name` is not `"EAZY"` or `"Bagpipes"`.
        """
        if code_name == "EAZY":
            return cls.load_EAZY_in_template(
                m_UV, template_set, template_number
            )
        elif code_name == "Bagpipes":
            return cls.load_pipes_in_template()
        else:
            raise (
                Exception(
                    f"Rest frame template load in currently unavailable for code_name = {code_name}"
                )
            )

    @classmethod
    def load_EAZY_in_template(cls, m_UV, template_set, template_filename):
        """Load a single EAZY rest-frame template SED from disk.

        Parameters
        ----------
        m_UV : `astropy.units.Quantity` or `astropy.units.Magnitude`
            Apparent UV magnitude to normalize the template to.
        template_set : `str`
            Name of the EAZY template set (e.g. `"fsps_larson"`); also
            used to look up the units for that set and the `.txt` file
            listing available template filenames.
        template_filename : `str` or `int`
            Filename of the template ascii file, or an integer index
            into the template set's `.txt` listing file.

        Returns
        -------
        `Mock_SED_rest`
            The loaded template SED, converted to Jy/AA and normalized
            to `m_UV`.
        """
        EAZY_template_units = {
            "fsps_larson": {
                "wavs": u.AA,
                "mags": u.erg / (u.s * (u.cm**2) * u.AA),
            }
        }
        if isinstance(template_filename, int):
            template_labels = open(
                f"{config['EAZY']['EAZY_TEMPLATE_DIR'].replace('/templates', '')}/{template_set}.txt",
                "r",
            )
            template_filename = template_labels.readlines()[
                template_filename
            ].replace("\n", "")
            template_labels.close()
        template = Table.read(
            f"{config['EAZY']['EAZY_TEMPLATE_DIR']}/{template_filename}",
            names=["Wavelength", "SED"],
            format="ascii",
        )
        # restrict template to appropriate wavelength range
        template_obj = cls(
            template["Wavelength"],
            template["SED"],
            EAZY_template_units[template_set]["wavs"],
            EAZY_template_units[template_set]["mags"],
            template_filename.split("/")[1],
        )
        template_obj.convert_mag_units(u.Jy, update=True)
        template_obj.convert_wav_units(u.AA, update=True)
        template_obj.normalize_to_m_UV(m_UV)
        return template_obj

    @classmethod
    def load_pipes_in_template(cls, m_UV, template_set, template_filename):
        """Load a single Bagpipes rest-frame template SED from disk.

        Parameters
        ----------
        m_UV : `astropy.units.Quantity` or `astropy.units.Magnitude`
            Apparent UV magnitude to normalize the template to.
        template_set : `str`
            Subdirectory of `config['Bagpipes']['BAGPIPES_TEMPLATE_DIR']`
            containing the template.
        template_filename : `int`
            Integer index used to glob for the matching `.ecsv` file
            within the template set directory.

        Returns
        -------
        `Mock_SED_rest`
            The loaded template SED, converted to Jy/AA and normalized
            to `m_UV`, with `meta` populated from the ecsv file's
            metadata.
        """
        pipes_template_units = {
            "wavs": u.AA,
            "mags": u.erg / (u.s * (u.cm**2)),
        }  # rest frame f_lambda
        if isinstance(template_filename, int):
            template_name = glob.glob(
                f"{config['Bagpipes']['BAGPIPES_TEMPLATE_DIR']}/{template_set}/*_{str(template_filename)}.ecsv"
            )[0]
        template = Table.read(
            template_name, names=["Wavelength", "SED"], format="ascii.ecsv"
        )
        # restrict template to appropriate wavelength range
        template_obj = cls(
            template["Wavelength"].value,
            template["SED"].value,
            pipes_template_units["wavs"],
            pipes_template_units["mags"] / u.AA,
            template_name.split("/")[-1].replace(".ecsv", ""),
            meta=dict(template.meta),
        )
        template_obj.convert_mag_units(u.Jy, update=True)
        template_obj.convert_wav_units(u.AA, update=True)
        template_obj.normalize_to_m_UV(m_UV)
        return template_obj

    @classmethod
    def load_Yggdrasil_popIII_in_template(
        cls, imf, fcov, sfh, template_filename
    ):
        """Load a single Yggdrasil Pop III rest-frame template SED from disk.

        Parameters
        ----------
        imf : `str`
            Initial mass function identifier used in the grid directory name.
        fcov : `float` or `str`
            Covering fraction used in the grid directory name.
        sfh : `str`
            Star formation history identifier used in the grid directory
            name.
        template_filename : `str`
            Filename of the template ascii file within the grid directory.

        Returns
        -------
        `Mock_SED_rest`
            The loaded template SED, converted to Jy/AA, with `age`
            (from the file's metadata) attached as an attribute.
        """
        # print("Incorrect normalization for yggdrasil input templates!")
        yggdrasil_dir = f"/Users/user/Documents/PGR/yggdrasil_grids/{imf}_fcov_{str(fcov)}_SFR_{sfh}_Spectra"
        # if isinstance(template_filename, int):
        #     SED_arr = glob.glob(yggdrasil_dir)
        #     print(SED_arr)
        template = Table.read(
            f"{yggdrasil_dir}/{template_filename}", format="ascii"
        )
        # convert template fluxes to appropriate units
        #  * (astropy_cosmo.luminosity_distance(z) ** 2 / (1 + z))
        fluxes = (
            (
                template["flux"]
                * (u.erg / (u.s * u.AA))
                / (4 * np.pi * u.pc**2)
            ).to(u.erg / (u.s * (u.cm**2) * u.AA))
        ).value
        template_obj = cls(
            template["wav"],
            fluxes,
            u.AA,
            u.erg / (u.s * (u.cm**2) * u.AA),
            template_filename,
        )
        template_obj.convert_mag_units(u.Jy, update=True)
        template_obj.convert_wav_units(u.AA, update=True)
        template_obj.age = template.meta["age"] * u.Myr
        return template_obj

    def normalize_to_m_UV(self, m_UV):
        """Rescale the SED in place so its flux at rest-frame 1500 AA matches m_UV.

        Parameters
        ----------
        m_UV : `astropy.units.Quantity` or `astropy.units.Magnitude` or `None`
            Target apparent UV magnitude at rest-frame 1500 AA. If `None`,
            no normalization is performed.

        Returns
        -------
        `None`
            `self.mags` is rescaled in place.

        Raises
        ------
        AssertionError
            If `m_UV` is not `None` and not a `Quantity`/`Magnitude`.
        """
        if m_UV is not None:
            assert type(m_UV) in [u.Quantity, u.Magnitude]
            norm = (
                funcs.convert_mag_units(1_500.0 * u.AA, m_UV, u.Jy).value
                / funcs.convert_mag_units(self.wavs, self.mags, u.Jy).value[
                    np.abs(self.wavs.to(u.AA).value - 1_500.0).argmin()
                ]
            )
            self.mags = (
                np.array([norm * mag for mag in self.mags.value])
                * self.mags.unit
            )

    def renorm_at_wav(
        self, wav, mag
    ):  # this mag can also be a flux, but must have astropy units
        """Rescale the SED in place so its flux at a given wavelength matches `mag`.

        Parameters
        ----------
        wav : `astropy.units.Quantity`
            Wavelength (with units of length) at which to match the flux.
        mag : `astropy.units.Quantity` or `astropy.units.Magnitude`
            Target magnitude or flux at `wav`.

        Returns
        -------
        `None`
            `self.mags` is rescaled in place.

        Raises
        ------
        AssertionError
            If `wav` is not a `Quantity` with physical type 'length', or
            `mag` is not a `Quantity`/`Magnitude`.
        """
        assert isinstance(wav, u.Quantity)
        assert isinstance(mag, (u.Quantity, u.Magnitude))
        assert u.get_physical_type(wav.unit) == "length"
        norm = (
            funcs.convert_mag_units(wav, mag, u.Jy).value
            / funcs.convert_mag_units(self.wavs, self.mags, u.Jy).value[
                np.abs(self.wavs.to(u.AA).value - wav.to(u.AA).value).argmin()
            ]
        )
        self.mags = (
            np.array([norm * mag for mag in self.mags.value]) * self.mags.unit
        )

    def calc_UV_slope(self, output_errs=False, method="Calzetti+94"):
        """Fit the UV continuum power-law slope (beta) of the SED.

        Crops the SED to the Calzetti+94 UV continuum windows, converts
        to f_lambda if necessary, and fits `A * lambda^beta` via
        `scipy.optimize.curve_fit`.

        Parameters
        ----------
        output_errs : `bool`, optional
            If `True`, also return the 1-sigma uncertainties on the
            fitted amplitude and slope. Default is `False`.
        method : `str`, optional
            Method used to select the continuum windows. Only
            `"Calzetti+94"` is currently supported. Default is `"Calzetti+94"`.

        Returns
        -------
        `tuple` of `float`
            `(A, beta)`, or `(A, beta, A_err, beta_err)` if `output_errs`
            is `True`.
        """
        if method == "Calzetti+94":
            # crop to Calzetti+94 filters
            wavs, mags = funcs.crop_to_Calzetti94_filters(self.wavs, self.mags)
            # convert self.mags to f_λ if needed
            if mags.unit == u.ABmag:
                mags = funcs.convert_mag_units(
                    wavs, mags, u.erg / (u.s * (u.cm**2) * u.AA)
                )
            elif (
                u.get_physical_type(mags.unit)
                != "power density/spectral flux density wav"
            ):
                mags = funcs.convert_mag_units(
                    wavs, mags, u.erg / (u.s * (u.cm**2) * u.AA)
                )
        popt, pcov = curve_fit(
            funcs.beta_slope_power_law_func,
            wavs.value,
            mags.value,
            maxfev=1_000,
        )
        A, beta = popt[0], popt[1]
        if output_errs:
            A_err = np.sqrt(pcov[0][0])
            beta_err = np.sqrt(pcov[1][1])
            return A, beta, A_err, beta_err
        else:
            return A, beta

    def add_emission_lines(self, emission_lines):
        """Add emission line profiles to the SED spectrum in place.

        Each line's profile is interpolated onto the SED's wavelength
        grid (zero outside the profile's own range) and added directly
        to `self.mags`.

        Parameters
        ----------
        emission_lines : `list`
            List of emission line objects, each exposing a
            `line_profile` dict with `"wavs"` and `"flux"` `Quantity`
            entries.

        Returns
        -------
        `None`
            `self.mags` is updated in place.

        Raises
        ------
        AssertionError
            If `emission_lines` is not a list or `numpy.array`.
        """
        assert type(emission_lines) in [list, np.array]
        for emission_line in emission_lines:
            # update units
            orig_wav_unit = self.wavs.unit
            orig_mag_unit = self.mags.unit
            self.convert_wav_units(
                emission_line.line_profile["wavs"].unit, update=True
            )
            self.convert_mag_units(
                emission_line.line_profile["flux"].unit, update=True
            )
            # interpolate emission line to be on the same wavelength grid as the spectrum
            interp_line_profile = interp1d(
                emission_line.line_profile["wavs"],
                emission_line.line_profile["flux"].value,
                bounds_error=False,
                fill_value=0.0,
            )(self.wavs)
            # correct for line flux difference between interped and non-interped line profile
            # self_copy = deepcopy(self)
            # self_copy.mags += interp_line_profile * self_copy.mags.unit
            # self_copy.calc_line_EW(emission_line.line_name)
            # line_flux = (self_copy.line_fluxes[emission_line.line_name] / u.AA).to(u.erg / (u.s * u.AA * u.cm ** 2), \
            #                 equivalencies = u.spectral_density(line_diagnostics[emission_line.line_name]["line_wav"])) * u.AA
            # add normalized line profile to spectrum
            self.mags += (
                interp_line_profile * self.mags.unit
            )  # * emission_line.line_flux / line_flux

    def add_dust_attenuation(self, dust_attenuation, E_BminusV):
        """Apply dust attenuation to the SED spectrum in place.

        Converts the SED to flux density (Jy) and multiplies by the
        attenuation curve evaluated at `E_BminusV`.

        Parameters
        ----------
        dust_attenuation : object
            A dust attenuation law object exposing an
            `attenuate(wavs, E_BminusV)` method returning A(lambda) in
            magnitudes.
        E_BminusV : `float`
            Colour excess, E(B-V), to apply.

        Returns
        -------
        `None`
            `self.mags` is updated in place; `self.colours` is reset.
        """
        self.colours = {}  # reset colours
        self.dust_attenuation = dust_attenuation
        self.E_BminusV = E_BminusV
        self.wavs = funcs.convert_wav_units(self.wavs, u.AA)
        self.mags = funcs.convert_mag_units(self.wavs, self.mags, u.Jy)
        # attenuate flux in Jy (named self.mags)
        #print(dust_attenuation)
        self.mags *= 10 ** (
            -0.4 * dust_attenuation.attenuate(self.wavs, E_BminusV)
        )

class Mock_SED_obs(SED_obs):
    """An observed-frame SED representing a mock/template (rather than fitted) spectrum.

    Extends `SED_obs` with a template name, metadata, and (optionally)
    IGM attenuation applied automatically at construction time.

    Parameters
    ----------
    z : `float`
        Redshift of the SED.
    wavs : array-like
        Observed-frame wavelength values (without units attached yet).
    mags : array-like
        Magnitude or flux density values (without units attached yet).
    wav_units : `astropy.units.Unit`
        Units to attach to `wavs`.
    mag_units : `astropy.units.Unit`
        Units to attach to `mags`.
    template_name : `str`, optional
        Name/identifier of the template. Default is `None`.
    IGM : `galfind.IGM_attenuation.IGM`, optional
        IGM attenuation prescription to apply immediately after
        construction. If `None`, no attenuation is applied.
        Default is a new `IGM_attenuation.IGM()` instance.
    meta : `dict`, optional
        Arbitrary metadata associated with the template. Default is `None`.

    Attributes
    ----------
    template_name : `str` or `None`
        Name/identifier of the template.
    meta : `dict` or `None`
        Metadata associated with the template.
    IGM : `galfind.IGM_attenuation.IGM` or `None`
        The IGM attenuation prescription that has been applied, if any.
    """

    def __init__(
        self,
        z,
        wavs,
        mags,
        wav_units,
        mag_units,
        template_name=None,
        IGM=IGM_attenuation.IGM(),
        meta=None,
    ):
        self.template_name = template_name
        self.meta = meta
        super().__init__(z, wavs, mags, wav_units, mag_units)
        if IGM != None:
            self.attenuate_IGM(IGM)

    @classmethod
    def from_Mock_SED_rest(
        cls,
        mock_SED_rest,
        z,
        out_wav_units=u.AA,
        out_mag_units=u.ABmag,
        IGM=IGM_attenuation.IGM(),
    ):
        """Construct an observed-frame mock SED from a rest-frame mock SED.

        Parameters
        ----------
        mock_SED_rest : `Mock_SED_rest`
            The rest-frame mock SED to redshift.
        z : `float`
            Redshift to place the SED at.
        out_wav_units : `astropy.units.Unit`, optional
            Wavelength units of the returned SED. Default is `astropy.units.AA`.
        out_mag_units : `astropy.units.Unit`, optional
            Magnitude/flux units of the returned SED. Default is `astropy.units.ABmag`.
        IGM : `galfind.IGM_attenuation.IGM`, optional
            IGM attenuation prescription to apply. Default is a new
            `IGM_attenuation.IGM()` instance.

        Returns
        -------
        `Mock_SED_obs`
            The corresponding observed-frame mock SED.
        """
        mags = mock_SED_rest.convert_mag_units(out_mag_units)
        wavs = (mock_SED_rest.wavs * (1 + z)).to(out_wav_units)
        mock_SED_obs_obj = cls(
            z,
            wavs.value,
            mags.value,
            out_wav_units,
            out_mag_units,
            mock_SED_rest.template_name,
            IGM,
            meta=mock_SED_rest.meta,
        )
        return mock_SED_obs_obj

    @classmethod
    def power_law_from_beta_M_UV(
        cls, z, beta, M_UV, template_name=None, IGM=IGM_attenuation.IGM()
    ):
        """Construct a power-law mock observed-frame SED from beta and M_UV.

        Converts the absolute UV magnitude to an apparent magnitude at
        `z`, builds a rest-frame power-law SED via
        `Mock_SED_rest.power_law_from_beta_m_UV`, and redshifts and
        IGM-attenuates it.

        Parameters
        ----------
        z : `float`
            Redshift to place the SED at.
        beta : `float`
            The UV continuum power-law slope index.
        M_UV : `astropy.units.Quantity` or `astropy.units.Magnitude`
            Absolute UV magnitude.
        template_name : `str`, optional
            Name/identifier for the resulting SED. Default is `None`.
        IGM : `galfind.IGM_attenuation.IGM`, optional
            IGM attenuation prescription to apply. Default is a new
            `IGM_attenuation.IGM()` instance.

        Returns
        -------
        `Mock_SED_obs`
            The normalized, redshifted, IGM-attenuated power-law SED.
        """
        lum_distance = astropy_cosmo.luminosity_distance(z).to(u.pc)
        m_UV = (
            M_UV.to(u.ABmag).value
            - 2.5 * np.log10(1 + z)
            + 5 * np.log10(lum_distance.value / 10)
        ) * u.ABmag
        mock_SED_rest = Mock_SED_rest.power_law_from_beta_m_UV(
            beta, m_UV, template_name=template_name
        )
        obs_SED = cls.from_SED_rest(z, mock_SED_rest)
        obs_SED.attenuate_IGM(IGM)
        return obs_SED

    def attenuate_IGM(self, IGM=IGM_attenuation.IGM()):
        """Apply IGM attenuation to the SED spectrum in place.

        A no-op if the SED has already been attenuated (`self.IGM` is
        already set to a non-`None` `IGM` object).

        Parameters
        ----------
        IGM : `galfind.IGM_attenuation.IGM`, optional
            IGM attenuation prescription to apply. Default is a new
            `IGM_attenuation.IGM()` instance.

        Returns
        -------
        `None`
            `self.mags` is updated in place, and `self.IGM` is set to
            `IGM`.

        Raises
        ------
        Exception
            If `IGM` is not an instance of `galfind.IGM_attenuation.IGM`.
        """
        if not hasattr(self, "IGM"):
            self.IGM = None
        if isinstance(IGM, IGM_attenuation.IGM):
            if self.IGM == None:  # not already been attenuated
                # attenuate SED for IGM absorption
                IGM_transmission = IGM.interp_transmission(
                    self.z, self.wavs / (1 + self.z)
                )
                if self.mags.unit == u.ABmag:
                    self.mags = (
                        self.mags.value - 2.5 * np.log10(IGM_transmission)
                    ) * u.ABmag
                else:
                    self.mags *= IGM_transmission
                # save IGM object after attenuating
                self.IGM = IGM
            else:
                # print("SED has already been attenuated! Ignoring")
                pass
        else:
            raise (
                Exception(f"Could not attenuate by a non IGM object = {IGM}")
            )

    def calc_UV_slope(self, output_errs=False, method="Calzetti+94"):
        """Fit the UV continuum power-law slope (beta) of the SED.

        De-redshifts to a rest-frame `Mock_SED_rest` and delegates the
        fit to `Mock_SED_rest.calc_UV_slope`.

        Parameters
        ----------
        output_errs : `bool`, optional
            If `True`, also return the 1-sigma uncertainties on the
            fitted amplitude and slope. Default is `False`.
        method : `str`, optional
            Method used to select the continuum windows. Only
            `"Calzetti+94"` is currently supported. Default is `"Calzetti+94"`.

        Returns
        -------
        `tuple` of `float`
            `(A, beta)`, or `(A, beta, A_err, beta_err)` if `output_errs`
            is `True`.
        """
        # create rest frame mock SED object
        mock_sed_rest = Mock_SED_rest.from_Mock_SED_obs(self)
        # calculate amplitude and beta of power law fit
        A, beta = mock_sed_rest.calc_UV_slope(
            output_errs=output_errs, method=method
        )
        return A, beta

    def get_colour(self, colour_name):
        """Calculate an AB magnitude colour from the mock photometry.

        Requires `create_mock_photometry` to have been called first.
        Caches the result on `self.colours`.

        Parameters
        ----------
        colour_name : `str`
            Colour to compute, formatted as `"<band1>-<band2>"`, where
            both bands must exist in `self.mock_photometry`.

        Returns
        -------
        `float`
            The colour in AB magnitudes (`<band1>` minus `<band2>`).

        Raises
        ------
        AssertionError
            If `colour_name` does not split into exactly two bands, or
            if the mock photometry fluxes are not in Jy.
        """
        if "mock_photometry" in self.__dict__:
            bands = colour_name.split("-")
            assert len(bands) == 2
            # requires colour to exist in the mock photometry
            for band in bands:
                if band not in self.mock_photometry.filterset.filt_names:
                    galfind_logger.critical(
                        f"self.mock_photometry includes the bands = {self.mock_photometry.filterset.filt_names}, and {band} is not included!"
                    )
                assert self.mock_photometry[band].flux.unit == u.Jy
            # calculate colour in mags
            colour = -2.5 * np.log10(
                self.mock_photometry[bands[0]].flux / self.mock_photometry[bands[1]].flux
            ) * u.ABmag
            # save colour in Mock_SED object
            if "colours" in self.__dict__:
                self.colours = {**self.colours, **{colour_name: colour.value}}
            else:
                self.colours = {colour_name: colour.value}
            return colour.value
        else:
            galfind_logger.critical(
                "self.mock_photometry does not exist! Please first create photometry from SED template!"
            )

    def add_DLA(self, DLA_obj):
        """Apply damped Lyman-alpha (DLA) absorption to the SED in place.

        Converts the SED to flux density (Jy) and multiplies by the
        DLA transmission curve evaluated at the rest-frame wavelengths.

        Parameters
        ----------
        DLA_obj : object
            A DLA object exposing a `transmission(rest_wavs)` method.

        Returns
        -------
        `None`
            `self.mags` is updated in place; `self.colours` is reset.
        """
        self.colours = {}  # reset colours
        self.DLA = DLA_obj
        self.wavs = funcs.convert_wav_units(self.wavs, u.AA)
        self.mags = funcs.convert_mag_units(self.wavs, self.mags, u.Jy)
        self.mags *= DLA_obj.transmission(self.wavs / (1 + self.z))

    def add_dust_attenuation(self, dust_attenuation, E_BminusV):
        """Apply dust attenuation to the SED spectrum in place.

        Converts the SED to flux density (Jy) and multiplies by the
        attenuation curve evaluated at `E_BminusV`.

        Parameters
        ----------
        dust_attenuation : object
            A dust attenuation law object exposing an
            `attenuate(wavs, E_BminusV)` method returning A(lambda) in
            magnitudes.
        E_BminusV : `float`
            Colour excess, E(B-V), to apply.

        Returns
        -------
        `None`
            `self.mags` is updated in place; `self.colours` is reset.
        """
        self.colours = {}  # reset colours
        self.dust_attenuation = dust_attenuation
        self.E_BminusV = E_BminusV
        self.wavs = funcs.convert_wav_units(self.wavs, u.AA)
        self.mags = funcs.convert_mag_units(self.wavs, self.mags, u.Jy)
        # attenuate flux in Jy (named self.mags)
        print(dust_attenuation)
        self.mags *= 10 ** (
            -0.4 * dust_attenuation.attenuate(self.wavs, E_BminusV)
        )

    def add_emission_lines(self, line_diagnostics):
        """Not yet implemented for observed-frame mock SEDs.

        Parameters
        ----------
        line_diagnostics : `dict`
            Emission line diagnostic information (currently unused).

        Returns
        -------
        `None`
            This method is currently a no-op.
        """
        pass


class Mock_SED_template_set(ABC):
    """Abstract base class for an ordered, iterable collection of mock SEDs.

    Concrete subclasses (`Mock_SED_rest_template_set`,
    `Mock_SED_obs_template_set`) provide classmethod constructors that
    load a whole template library into a list of `Mock_SED_rest`/
    `Mock_SED_obs` objects; this base class provides the shared sequence
    protocol (`len`, iteration, indexing) and bulk photometry creation.

    Parameters
    ----------
    mock_SED_arr : `list`
        List of `Mock_SED_rest` or `Mock_SED_obs` objects making up the
        template set.

    Attributes
    ----------
    SED_arr : `list`
        The underlying list of mock SED objects.
    """

    def __init__(self, mock_SED_arr):
        self.SED_arr = mock_SED_arr

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter > len(self) - 1:
            raise StopIteration
        else:
            template = self[self.iter]
            self.iter += 1
            return template

    def __getitem__(self, index):
        return self.SED_arr[index]

    def __len__(self):
        return len(self.SED_arr)

    def create_mock_photometry(self, filterset):
        """Create mock photometry for every SED in the template set.

        Parameters
        ----------
        filterset : `Multiple_Filter`
            The set of filters to synthesize photometry through.

        Returns
        -------
        `None`
            Each SED's `mock_photometry` attribute is populated in
            place; results are not collected or returned.
        """
        [sed.create_mock_photometry(filterset) for sed in self.SED_arr]

    # @abstractmethod
    # def calc_UV_slope():
    #     pass


class Mock_SED_rest_template_set(Mock_SED_template_set):
    """A collection of rest-frame mock SED templates.

    Parameters
    ----------
    mock_SED_rest_arr : `list`
        List of `Mock_SED_rest` objects making up the template set.
    """

    def __init__(self, mock_SED_rest_arr):
        super().__init__(mock_SED_rest_arr)

    @classmethod
    def load_SED_in_templates():
        """Not yet implemented."""
        pass

    @classmethod
    def load_EAZY_in_templates(cls, m_UV, template_set):
        """Load an entire EAZY rest-frame template set from disk.

        Reads the `<template_set>.txt` listing file and loads each
        listed template via `Mock_SED_rest.load_EAZY_in_template`.

        Parameters
        ----------
        m_UV : `astropy.units.Quantity` or `astropy.units.Magnitude`
            Apparent UV magnitude to normalize each template to.
        template_set : `str`
            Name of the EAZY template set.

        Returns
        -------
        `Mock_SED_rest_template_set`
            The loaded set of template SEDs.
        """
        mock_SED_rest_arr = []
        # read in .txt file if it exists
        template_labels = open(
            f"{config['EAZY']['EAZY_TEMPLATE_DIR']}/{template_set}.txt", "r"
        )
        for name in template_labels.readlines():
            mock_SED_rest_arr.append(
                Mock_SED_rest.load_EAZY_in_template(
                    m_UV, template_set, name.replace("\n", "")
                )
            )
        template_labels.close()
        return cls(mock_SED_rest_arr)

    @classmethod
    def load_Yggdrasil_popIII_in_templates(cls, imf, fcov, sfh):
        """Load an entire Yggdrasil Pop III rest-frame template grid from disk.

        Parameters
        ----------
        imf : `str`
            Initial mass function identifier used in the grid directory name.
        fcov : `float` or `str`
            Covering fraction used in the grid directory name.
        sfh : `str`
            Star formation history identifier used in the grid directory
            name.

        Returns
        -------
        `Mock_SED_rest_template_set`
            The loaded set of template SEDs (one per `.ecsv` file found).
        """
        mock_SED_rest_arr = []
        yggdrasil_dir = f"/Users/user/Documents/PGR/yggdrasil_grids/{imf}_fcov_{str(fcov)}_SFR_{sfh}_Spectra"
        SED_filenames = glob.glob(f"{yggdrasil_dir}/*.ecsv")
        for name in SED_filenames:
            mock_SED_rest_arr.append(
                Mock_SED_rest.load_Yggdrasil_popIII_in_template(
                    imf, fcov, sfh, name.split("/")[-1]
                )
            )
        return cls(mock_SED_rest_arr)

    @classmethod
    def load_bpass_in_templates(
        cls,
        metallicity="z020",
        imf="imf135_300",
        model_type="bin",
        alpha_enhancement="a+06",
        log_ages=np.linspace(6.0, 9.0, int(np.round(3 / 0.1)) + 1),
        bpass_version="2.3",
        m_UV_norm=26.0 * u.ABmag,
        grain_name="ng",
        logU=-1.0,
    ):
        """Load an entire BPASS rest-frame template grid at a set of ages.

        Parameters
        ----------
        metallicity : `str`, optional
            BPASS metallicity identifier. Default is `"z020"`.
        imf : `str`, optional
            BPASS IMF identifier. Default is `"imf135_300"`.
        model_type : `str`, optional
            `"bin"` for binary or `"sin"` for single-star models.
            Default is `"bin"`.
        alpha_enhancement : `str`, optional
            Alpha enhancement identifier (BPASS v2.3 only). Default is `"a+06"`.
        log_ages : array-like or `str`, optional
            Array of log10(age / yr) values to load, or `"all"` to load
            every available age. Default is
            `np.linspace(6.0, 9.0, int(np.round(3 / 0.1)) + 1)`.
        bpass_version : `str`, optional
            BPASS release version, `"2.3"` or `"2.2_cloudy"`. Default is `"2.3"`.
        m_UV_norm : `astropy.units.Quantity` or `astropy.units.Magnitude`, optional
            Apparent UV magnitude to normalize each loaded template to.
            If `None`, no normalization is applied. Default is `26.0 * astropy.units.ABmag`.
        grain_name : `str`, optional
            Dust grain type for `"2.2_cloudy"` models, `"ng"` or `"gr"`.
            Default is `"ng"`.
        logU : `float`, optional
            Ionization parameter for `"2.2_cloudy"` models, must be one
            of the keys of the internal `logU_dict`. Default is `-1.0`.

        Returns
        -------
        `Mock_SED_rest_template_set`
            The loaded set of template SEDs, one per requested age.

        Raises
        ------
        AssertionError
            If `bpass_version == "2.2_cloudy"` and `grain_name` or
            `logU` is not a recognised value.
        """
        meta = {
            "metallicity": metallicity,
            "imf": imf,
            "model_type": model_type,
            "alpha_enhancement": alpha_enhancement,
            "bpass_version": bpass_version,
        }
        bpass_Lsun = 3.848e26 * u.J / u.s
        bpass_version_dir_dict = {
            "2.3": f"BPASS_v2.3_release/bpass_v2.3.{alpha_enhancement}",
            "2.2_cloudy": "BPASS_v2.2_cloudy/spec",
        }
        bpass_version_name_dict = {
            "2.3": f"spectra-{model_type}-{imf}.{alpha_enhancement}.{metallicity}.dat",
        }
        if bpass_version == "2.2_cloudy":
            assert grain_name in ["ng", "gr"]
            v2_2_model_version = "11" if grain_name == "ng" else "10"
            logU_dict = {
                -1.0: "a",
                -1.5: "b",
                -2.0: "c",
                -2.5: "d",
                -3.0: "e",
                -3.5: "f",
                -4.0: "g",
            }
            assert float(logU) in logU_dict.keys()
            v2_2_model_version += logU_dict[float(logU)]
            bpass_version_name_dict["2.2_cloudy"] = (
                f"cloudyspec_{imf}_{metallicity}_{model_type}_v{v2_2_model_version}.sed"
            )

        bpass_dir = (
            f"/raid/scratch/data/BPASS/{bpass_version_dir_dict[bpass_version]}"
        )
        bpass_name = bpass_version_name_dict[bpass_version]
        SED_file = f"{bpass_dir}/{bpass_name}"
        mock_SED_rest_arr = []
        spectra = ascii.read(SED_file)
        if bpass_version == "2.3":
            age_range_indices = [2, 53]
        if bpass_version == "2.2_cloudy":
            age_range_indices = [2, 22]
            spectra = spectra[:-1]
        rest_wavs = np.array(spectra["col1"]).astype(float) * u.AA
        for i in range(*age_range_indices):
            age = 10 ** (0.1 * (i - 2)) * u.Myr
            load_spectrum = False
            if log_ages == "all":
                load_spectrum = True
            elif np.isclose(log_ages, np.log10(age.value) + 6.0).any():
                load_spectrum = True
            else:
                load_spectrum = False
            if load_spectrum:
                spectrum = (np.array(spectra[f"col{i}"])) * bpass_Lsun / u.AA
                spectrum_Jy = funcs.luminosity_to_flux(
                    spectrum, rest_wavs, out_units=u.Jy
                )
                mock_sed_rest = Mock_SED_rest(
                    rest_wavs.value,
                    spectrum_Jy.value,
                    u.AA,
                    u.Jy,
                    template_name=bpass_name.replace(
                        ".dat", f"{age.value:.1f}Myr"
                    ),
                    meta={
                        **meta,
                        **{
                            "age": age,
                            "log_age_yr": np.log10(age.value) + 6.0,
                        },
                    },
                )
                if m_UV_norm is not None:
                    mock_sed_rest.normalize_to_m_UV(m_UV_norm)
                mock_SED_rest_arr.append(mock_sed_rest)
        return cls(mock_SED_rest_arr)

    def calc_mock_beta_phot(self, m_UV, template_set, instrument, depths):
        """Not yet implemented."""
        pass


class Mock_SED_obs_template_set(Mock_SED_template_set):
    """A collection of observed-frame mock SED templates.

    Parameters
    ----------
    mock_SED_obs_arr : `list`
        List of `Mock_SED_obs` objects making up the template set.
    """

    def __init__(self, mock_SED_obs_arr):
        super().__init__(mock_SED_obs_arr)

    @classmethod
    def from_Mock_SED_rest_template_set(
        cls, Mock_SED_rest_template_set, z_arr
    ):
        """Redshift every SED in a rest-frame template set to build an observed-frame set.

        Parameters
        ----------
        Mock_SED_rest_template_set : `Mock_SED_rest_template_set`
            The rest-frame template set to redshift.
        z_arr : `float`, `int`, or array-like
            Redshift(s) to apply. If a scalar, the same redshift is
            applied to every SED in the set.

        Returns
        -------
        `Mock_SED_obs_template_set`
            The corresponding observed-frame template set.
        """
        if type(z_arr) in [float, int]:
            z_arr = np.full(len(Mock_SED_rest_template_set), z_arr)
        return cls(
            [
                Mock_SED_obs.from_Mock_SED_rest(mock_sed_rest, z)
                for mock_sed_rest, z in zip(Mock_SED_rest_template_set, z_arr)
            ]
        )

    def get_colours(self, colour_names):
        """Calculate colours for every SED in the template set.

        Parameters
        ----------
        colour_names : `list` of `str`
            Colours to compute for each SED, e.g. `["U-V", "V-J"]`.

        Returns
        -------
        `list` of `float`
            Flattened list of colours (SED-major, colour-minor order).
        """
        return [
            sed.get_colour(colour)
            for sed in self.SED_arr
            for colour in colour_names
        ]

    def plot_colour_colour_tracks(
        self,
        ax,
        colour_x_name,
        colour_y_name,
        shown_log_ages=[7.0, 8.0, 9.0],
        line_kwargs={},
        save=False,
        show=False,
        save_dir="/nvme/scratch/work/austind/EPOCHS_I_plots",
    ):
        """Plot an evolutionary colour-colour track for the template set.

        Assumes the SEDs in the template set are already sorted by age,
        starting with the youngest. The youngest SED is marked with a
        star; SEDs at `shown_log_ages` are marked with squares.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes to plot the track on.
        colour_x_name : `str`
            Name of the colour (previously computed via `get_colour`) to
            plot on the x-axis.
        colour_y_name : `str`
            Name of the colour (previously computed via `get_colour`) to
            plot on the y-axis.
        shown_log_ages : `list` of `float`, optional
            log10(age / yr) values at which to mark square symbols along
            the track. Default is `[7.0, 8.0, 9.0]`.
        line_kwargs : `dict`, optional
            Keyword arguments passed to `ax.plot` and used for marker
            colour/path effects (`"c"` and `"path_effects"` keys are
            required for the marker calls). Default is `{}`.
        save : `bool`, optional
            Whether to save the resulting figure to `save_dir`.
            Default is `False`.
        show : `bool`, optional
            Whether to call `plt.show()`. Default is `False`.
        save_dir : `str`, optional
            Directory to save the figure in when `save` is `True`.
            Default is `"/nvme/scratch/work/austind/EPOCHS_I_plots"`.

        Returns
        -------
        `None`
        """
        # assumes the SEDs are already sorted by age, starting with the youngest
        colour_x = np.array(
            [sed.colours[colour_x_name] for sed in self.SED_arr]
        )
        colour_y = np.array(
            [sed.colours[colour_y_name] for sed in self.SED_arr]
        )
        ax.plot(colour_x, colour_y, **line_kwargs)

        log_ages = [sed.meta["log_age_yr"] for sed in self.SED_arr]
        plot_indices = np.array(
            [
                True
                if np.isclose(shown_log_ages, log_age).any() and i != 0
                else False
                for i, log_age in enumerate(log_ages)
            ]
        )
        # plot youngest age SED as star
        ax.scatter(
            colour_x[0],
            colour_y[0],
            color=line_kwargs["c"],
            marker="*",
            zorder=999.0,
            s=50.0,
            path_effects=line_kwargs["path_effects"],
        )
        ax.scatter(
            colour_x[plot_indices],
            colour_y[plot_indices],
            color=line_kwargs["c"],
            marker="s",
            s=10.0,
            zorder=999.0,
            path_effects=line_kwargs["path_effects"],
        )
        if save:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                f"{save_dir}/{colour_x_name}_vs_{colour_y_name}.png", dpi=400
            )
            funcs.change_file_permissions(
                f"{save_dir}/{colour_x_name}_vs_{colour_y_name}.png"
            )
        if show:
            plt.show()


class SED_2D:
    """An ensemble of SED posterior draws (e.g. from Bagpipes) sharing a class.

    Represents a collection of individual `SED` objects (all of the same
    subclass, e.g. all `SED_obs`) that together sample a posterior
    distribution of possible spectra — for example, one draw per
    posterior sample from a Bagpipes fit. Since each draw generally has
    its own redshift, the draws are not simply stacked; `plot` handles
    combining them carefully (see its docstring) to avoid smearing
    artifacts near sharp spectral features such as the Lyman break.

    Parameters
    ----------
    SED_arr : `list` of `SED`
        List of SED objects (all instances of the same class) making up
        the ensemble.

    Attributes
    ----------
    SED_arr : `list` of `SED`
        The underlying list of SED draws.

    Raises
    ------
    AssertionError
        If the SEDs in `SED_arr` are not all of the same class.
    """

    def __init__(
        self: Self,
        SED_arr: List[Type[SED]],
    ):
        sed_classes = np.unique([SED.__class__.__name__ for SED in SED_arr])
        assert len(sed_classes) == 1, \
            galfind_logger.critical(
                f"SED_2D can only be created from a list of SEDs of the same class! Found {sed_classes}."
            )
        self.SED_arr = SED_arr

    def __repr__(self: Self) -> str:
        """Return the official string representation of the SED_2D collection."""
        return f"SED_2D({len(self)} SEDs)"

    def __str__(self: Self) -> str:
        """Return a detailed string representation of the SED_2D collection."""
        output_str = f"SED_2D Collection:\n"
        output_str += f"  Number of SEDs: {len(self)}\n"
        if len(self) > 0:
            first_sed = self.SED_arr[0] if isinstance(self.SED_arr, (list, np.ndarray)) else next(iter(self.SED_arr))
            output_str += f"  First SED wavelength range: {first_sed.wavs.min():.1f} - {first_sed.wavs.max():.1f}\n"
        return output_str

    def __len__(self) -> int:
        return len(self.SED_arr)

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter > len(self) - 1:
            raise StopIteration
        else:
            sed = self[self.iter]
            self.iter += 1
            return sed
        
    def __getitem__(self: Self, index: Any) -> Type[SED]:
        if len(self) == 0:
            raise IndexError(f"No SEDs in {self}!")
        if isinstance(index, int):
            return self.SED_arr[index]
        else:
            raise TypeError(
                f"Indexing {self} with {type(index)} is not supported! Use an integer index."
            )
        
    @property
    def frame(self: Self) -> Optional[str]:
        """Returns the frame of the SEDs in the SED_2D."""
        possible_frames = ["rest", "obs"]
        frame = self.SED_arr[0].__class__.__name__.split("_")[-1]
        if frame in possible_frames:
            return frame
        else:
            return None

    def plot(
        self: Self,
        #fig: Optional[plt.Figure],
        ax: Optional[plt.Axes] = None,
        wav_units: u.Unit = u.AA,
        mag_units: u.Unit = u.ABmag,
        label: Optional[str] = None,
        annotate: bool = True,
        save_name: Optional[str] = None,
        log_fluxes: bool = True,
        plot_chains: bool = False,
        plot_kwargs: Dict[str, Any] = {},
        legend_kwargs: Dict[str, Any] = {},
    ): # -> plt.Axes: #Tuple[plt.Figure, plt.Axes]:
        """Plot the median SED of the ensemble with a 16th-84th percentile band.

        Each draw in the ensemble generally has its own redshift, so the
        draws cannot simply be averaged point-by-point on a shared
        wavelength grid: a draw's observed-frame coverage shifts with its
        own redshift, and extrapolating a draw across a steep feature
        such as the Lyman break would fabricate flux it never
        constrained. To avoid this, near-zero/negative model flux is
        first floored to a tiny positive value (so the AB mag conversion
        does not produce +inf/nan for physically-zero flux, e.g. from
        near-total Lyman-limit/IGM attenuation), then each draw's
        spectrum is interpolated onto the first draw's wavelength grid
        with `bounds_error=False, fill_value=np.nan` so it is never
        extrapolated beyond its own native wavelength coverage. The 16th,
        50th and 84th percentiles are then taken across draws with
        `numpy.nanpercentile`, which ignores the `nan`-filled gaps on a
        per-wavelength basis rather than smearing out-of-coverage draws
        into the combined statistic.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`, optional
            Axes to plot on. If `None`, a new figure and axes are created.
            Default is `None`.
        wav_units : `astropy.units.Unit`, optional
            Units to convert the wavelength axis to. Default is `astropy.units.AA`.
        mag_units : `astropy.units.Unit`, optional
            Units to convert the flux/magnitude axis to. Default is `astropy.units.ABmag`.
        label : `str`, optional
            Legend label for the plotted median line. Overridden by
            `self.template_name` if the ensemble has one. Default is `None`.
        annotate : `bool`, optional
            Whether to set axis labels and draw the legend. Default is `True`.
        save_name : `str`, optional
            If given, the filename (png or pdf) to save the plot to.
            Default is `None`.
        log_fluxes : `bool`, optional
            Whether to log-scale flux values when `mag_units` is not
            `ABmag`. Default is `True`.
        plot_chains : `bool`, optional
            Currently unused. Default is `False`.
        plot_kwargs : `dict`, optional
            Extra keyword arguments passed to `ax.plot` for the median
            line. Default is `{}`.
        legend_kwargs : `dict`, optional
            Extra keyword arguments passed to `ax.legend`. Default is `{}`.

        Returns
        -------
        `list` of `matplotlib.lines.Line2D`
            The line artist(s) returned by `ax.plot` for the median SED.
        """
        if ax is None:
            fig, ax = plt.subplots()

        wavs_arr = [
            funcs.convert_wav_units(sed.wavs, wav_units)
            for sed in self
        ]
        # floor zero/negative model flux (e.g. from near-total Lyman-limit/
        # IGM attenuation) to a tiny positive value; log10 of a <=0 flux
        # (via the AB mag conversion or log_scale_fluxes) gives +inf/nan
        sed_mags = [
            np.maximum(sed.mags.value, 1e-10) * sed.mags.unit
            if sed.mags.unit != u.ABmag else sed.mags
            for sed in self
        ]
        mags_arr = [
            funcs.log_scale_fluxes(
                funcs.convert_mag_units(sed.wavs, mags, mag_units)
            ) if mag_units != u.ABmag and log_fluxes else
            funcs.convert_mag_units(sed.wavs, mags, mag_units).value
            for sed, mags in zip(self, sed_mags)
        ]

        if label is not None and hasattr(self, "template_name"):
            label = self.template_name

        # interpolate mags onto common wavelength grid
        #all_wavs = np.concatenate([wavs_arr_ for wavs_arr_ in wavs_arr])
        wavs_interp = wavs_arr[0] #np.linspace(np.min(all_wavs), np.max(all_wavs), 10_000)
        # each draw generally has its own redshift, so its observed-frame
        # wavelength grid doesn't exactly span wavs_interp; don't extrapolate
        # past a draw's own coverage (extrapolating across the steep Lyman
        # break slope shoots to zero/negative flux, i.e. +inf/nan AB mag at
        # the edges) - mark those points nan instead and combine nan-safely
        mags_interp = np.array([
            interp1d(
                wavs,
                mags,
                bounds_error=False,
                fill_value=np.nan,
            )(wavs_interp)
            for wavs, mags in zip(wavs_arr, mags_arr)
        ])

        # determine 16th, 50th and 84th percentiles of the interpolated mags
        mags_16 = np.nanpercentile(mags_interp, 16, axis=0)
        mags_50 = np.nanpercentile(mags_interp, 50, axis=0)
        mags_84 = np.nanpercentile(mags_interp, 84, axis=0)
        #breakpoint()
        plot = ax.plot(wavs_interp.value, mags_50, label=label, **plot_kwargs)
        ax.fill_between(wavs_interp.value, mags_16, mags_84, alpha=0.5, color=plot[0].get_color())

        if annotate:
            ax.set_xlabel(
                funcs.label_wavelengths(
                    wav_units,
                    False,
                    "" if self.frame is None else self.frame,
                )
            )
            ax.set_ylabel(
                funcs.label_fluxes(
                    mag_units, True if mag_units != u.ABmag and log_fluxes else False
                )
            )
            ax.legend(**legend_kwargs)

        if save_name is not None:
            # save png by default
            if save_name.split(".")[-1] not in ["png", "pdf"]:
                save_name = f"{'.'.join(save_name.split('.')[-1:])}.png"
            funcs.make_dirs(save_name)
            plt.savefig(save_name)
            funcs.change_file_permissions(save_name)

        return plot

    def create_mock_photometry(self: Self, *args, **kwargs):
        """Create ensemble-median mock photometry across all SED draws.

        Creates mock photometry independently for each draw (must yield
        the same length, depths, `min_flux_pc_err`, and flux unit across
        all draws), then takes the median flux in each band to produce a
        single representative `Mock_Photometry` object.

        Parameters
        ----------
        *args
            Positional arguments forwarded to each draw's
            `create_mock_photometry` (typically a `Multiple_Filter`
            filterset).
        **kwargs
            Keyword arguments forwarded to each draw's
            `create_mock_photometry`.

        Returns
        -------
        `Mock_Photometry`
            The median mock photometry across all draws, also stored as
            `self.mock_photometry`.

        Raises
        ------
        AssertionError
            If the per-draw mock photometry lengths, depths,
            `min_flux_pc_err`, or flux units are not all identical.
        """
        mock_phot_arr = [sed.create_mock_photometry(*args, **kwargs) for sed in self.SED_arr]
        assert all([len(mock_phot) == len(mock_phot_arr[0]) for mock_phot in mock_phot_arr]), \
            galfind_logger.critical(
                f"Mock photometry lengths are not the same for all SEDs in {self}!"
            )
        assert all([all([mock_phot.depths[i] == mock_phot_arr[0].depths[i] for i in range(len(mock_phot))]) for mock_phot in mock_phot_arr]), \
            galfind_logger.critical(
                f"Mock photometry depths are not the same for all SEDs in {self}!"
            )
        assert all([getattr(mock_phot, "min_flux_pc_err") == getattr(mock_phot_arr[0], "min_flux_pc_err") for mock_phot in mock_phot_arr]), \
            galfind_logger.critical(
                f"Mock photometry min_flux_pc_err are not the same for all SEDs in {self}!"
            )
        assert all([getattr(mock_phot, "flux").unit == getattr(mock_phot_arr[0], "flux").unit for mock_phot in mock_phot_arr]), \
            galfind_logger.critical(
                f"Mock photometry flux units are not the same for all SEDs in {self}!"
            )
        # weighted stack of the mock photometry fluxes and errors
        mock_phot_fluxes = np.array([
            mock_phot.flux.value for mock_phot in mock_phot_arr
        ])
        # make median flux array
        median_fluxes = np.median(mock_phot_fluxes, axis=0)
        self.mock_photometry = Mock_Photometry(
            mock_phot_arr[0].filterset,
            median_fluxes * mock_phot_arr[0].flux.unit,
            mock_phot_arr[0].depths,
            mock_phot_arr[0].min_flux_pc_err,
        )
        return self.mock_photometry
    
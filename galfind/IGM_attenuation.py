#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Intergalactic medium attenuation calculations.

Implements Inoue et al. (2014) Lyman series LAF and DLA optical depth functions
for IGM absorption in galaxy spectra.
"""

# IGM_attenuation.py
from pathlib import Path

import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import logging 

from . import config, wav_lyman_lim, galfind_logger
from .Emission_lines import wav_lyman_alpha


def calc_Inoue14_LS_LAF_optical_depth(lyman_series, wav_obs_arr, z):
    """Compute the Lyman series Lyman-alpha forest (LAF) optical depth of Inoue et al. (2014).

    Implements the piecewise power-law fits to the Lyman-alpha forest
    absorption (equations 8-9, 21 of Inoue et al. 2014) summed over all
    Lyman series transitions in `lyman_series`.

    Parameters
    ----------
    lyman_series : `astropy.table.Table`
        Table of Lyman series transition data, must contain columns
        ``"lambda_j"`` (rest wavelength), ``"A_j1_LAF"``, ``"A_j2_LAF"``
        and ``"A_j3_LAF"`` (Inoue+14 LAF coefficients).
    wav_obs_arr : array-like
        Observed-frame wavelengths, in the same units as
        ``lyman_series["lambda_j"]``.
    z : `float`
        Redshift of the source.

    Returns
    -------
    `numpy.ndarray`
        Lyman series LAF optical depth at each wavelength in `wav_obs_arr`,
        summed over all Lyman series transitions.
    """
    tau_arr = np.zeros((len(lyman_series), len(wav_obs_arr)))
    for j, wav_j in enumerate(np.array(lyman_series["lambda_j"])):
        valid_series_indices = (wav_obs_arr > wav_j) & (
            wav_obs_arr < wav_j * (1 + z)
        )
        wav_indices_1 = (wav_obs_arr < 2.2 * wav_j) & (valid_series_indices)
        wav_indices_2 = (
            (wav_obs_arr < 5.7 * wav_j)
            & (~wav_indices_1)
            & (valid_series_indices)
        )
        wav_indices_3 = (
            (~wav_indices_1) & (~wav_indices_2) & (valid_series_indices)
        )
        tau_arr[j, wav_indices_1] = (
            lyman_series["A_j1_LAF"][j]
            * (wav_obs_arr[wav_indices_1] / wav_j) ** 1.2
        )
        tau_arr[j, wav_indices_2] = (
            lyman_series["A_j2_LAF"][j]
            * (wav_obs_arr[wav_indices_2] / wav_j) ** 3.7
        )
        tau_arr[j, wav_indices_3] = (
            lyman_series["A_j3_LAF"][j]
            * (wav_obs_arr[wav_indices_3] / wav_j) ** 5.5
        )
    return np.sum(tau_arr, axis=0)


def calc_Inoue14_LS_DLA_optical_depth(lyman_series, wav_obs_arr, z):
    """Compute the Lyman series damped Lyman-alpha (DLA) optical depth of Inoue et al. (2014).

    Implements the piecewise power-law fits to the DLA absorption
    (equations 10-11, 22 of Inoue et al. 2014) summed over all Lyman
    series transitions in `lyman_series`.

    Parameters
    ----------
    lyman_series : `astropy.table.Table`
        Table of Lyman series transition data, must contain columns
        ``"lambda_j"`` (rest wavelength), ``"A_j1_DLA"`` and
        ``"A_j2_DLA"`` (Inoue+14 DLA coefficients).
    wav_obs_arr : array-like
        Observed-frame wavelengths, in the same units as
        ``lyman_series["lambda_j"]``.
    z : `float`
        Redshift of the source.

    Returns
    -------
    `numpy.ndarray`
        Lyman series DLA optical depth at each wavelength in `wav_obs_arr`,
        summed over all Lyman series transitions.
    """
    tau_arr = np.zeros((len(lyman_series), len(wav_obs_arr)))
    for j, wav_j in enumerate(np.array(lyman_series["lambda_j"])):
        valid_series_indices = (wav_obs_arr > wav_j) & (
            wav_obs_arr < wav_j * (1 + z)
        )
        wav_indices_1 = (wav_obs_arr < 3 * wav_j) & (valid_series_indices)
        wav_indices_2 = (~wav_indices_1) & (valid_series_indices)
        tau_arr[j, wav_indices_1] = (
            lyman_series["A_j1_DLA"][j]
            * (wav_obs_arr[wav_indices_1] / wav_j) ** 2
        )
        tau_arr[j, wav_indices_2] = (
            lyman_series["A_j2_DLA"][j]
            * (wav_obs_arr[wav_indices_2] / wav_j) ** 3
        )
    return np.sum(tau_arr, axis=0)


def calc_Inoue14_LC_LAF_optical_depth(wav_obs_arr, z):
    """Compute the Lyman continuum Lyman-alpha forest (LAF) optical depth of Inoue et al. (2014).

    Implements the redshift-dependent piecewise power-law fits to the
    Lyman continuum forest absorption (equations 12-15, 21 of Inoue et al.
    2014), with separate expressions for ``z < 1.2``, ``1.2 <= z < 4.7``
    and ``z >= 4.7``. Returns zero everywhere if ``z <= 0``.

    Parameters
    ----------
    wav_obs_arr : array-like
        Observed-frame wavelengths, in the same units as `wav_lyman_lim`.
    z : `float`
        Redshift of the source.

    Returns
    -------
    `numpy.ndarray`
        Lyman continuum LAF optical depth at each wavelength in
        `wav_obs_arr`.
    """
    tau = np.zeros(len(wav_obs_arr))
    gtr_lyman_lim_indices = wav_obs_arr > wav_lyman_lim
    if z > 0.0:
        if z < 1.2:
            wav_indices = (wav_obs_arr < wav_lyman_lim * (1 + z)) & (
                gtr_lyman_lim_indices
            )
            tau[wav_indices] = 0.325 * (
                ((wav_obs_arr[wav_indices] / wav_lyman_lim) ** 1.2)
                - ((1 + z) ** -0.9)
                * ((wav_obs_arr[wav_indices] / wav_lyman_lim) ** 2.1)
            )
        elif z < 4.7:
            wav_indices_1 = (wav_obs_arr < 2.2 * wav_lyman_lim) & (
                gtr_lyman_lim_indices
            )
            wav_indices_2 = (
                (wav_obs_arr < wav_lyman_lim * (1 + z))
                & (~wav_indices_1)
                & (gtr_lyman_lim_indices)
            )
            tau[wav_indices_1] = (
                (
                    2.55e-2
                    * ((1 + z) ** 1.6)
                    * ((wav_obs_arr[wav_indices_1] / wav_lyman_lim) ** 2.1)
                )
                + (
                    0.325
                    * ((wav_obs_arr[wav_indices_1] / wav_lyman_lim) ** 1.2)
                )
                - (
                    0.25
                    * ((wav_obs_arr[wav_indices_1] / wav_lyman_lim) ** 2.1)
                )
            )
            tau[wav_indices_2] = 2.55e-2 * (
                (
                    ((1 + z) ** 1.6)
                    * ((wav_obs_arr[wav_indices_2] / wav_lyman_lim) ** 2.1)
                )
                - ((wav_obs_arr[wav_indices_2] / wav_lyman_lim) ** 3.7)
            )
        else:
            wav_indices_1 = (wav_obs_arr < 2.2 * wav_lyman_lim) & (
                gtr_lyman_lim_indices
            )
            wav_indices_2 = (
                (wav_obs_arr < 5.7 * wav_lyman_lim)
                & (~wav_indices_1)
                & (gtr_lyman_lim_indices)
            )
            wav_indices_3 = (
                (wav_obs_arr < wav_lyman_lim * (1 + z))
                & (~wav_indices_1)
                & (~wav_indices_2)
                & (gtr_lyman_lim_indices)
            )
            tau[wav_indices_1] = (
                (
                    5.22e-4
                    * ((1 + z) ** 3.4)
                    * (wav_obs_arr[wav_indices_1] / wav_lyman_lim) ** 2.1
                )
                + (0.325 * (wav_obs_arr[wav_indices_1] / wav_lyman_lim) ** 1.2)
                - (
                    3.14e-2
                    * (wav_obs_arr[wav_indices_1] / wav_lyman_lim) ** 2.1
                )
            )
            tau[wav_indices_2] = (
                (
                    5.22e-4
                    * ((1 + z) ** 3.4)
                    * (wav_obs_arr[wav_indices_2] / wav_lyman_lim) ** 2.1
                )
                + (0.218 * (wav_obs_arr[wav_indices_2] / wav_lyman_lim) ** 2.1)
                - (
                    2.55e-2
                    * (wav_obs_arr[wav_indices_2] / wav_lyman_lim) ** 3.7
                )
            )
            tau[wav_indices_3] = 5.22e-4 * (
                (
                    ((1 + z) ** 3.4)
                    * (wav_obs_arr[wav_indices_3] / wav_lyman_lim) ** 2.1
                )
                - ((wav_obs_arr[wav_indices_3] / wav_lyman_lim) ** 5.5)
            )
    return tau


def calc_Inoue14_LC_DLA_optical_depth(wav_obs_arr, z):
    """Compute the Lyman continuum damped Lyman-alpha (DLA) optical depth of Inoue et al. (2014).

    Implements the redshift-dependent piecewise power-law fits to the
    Lyman continuum DLA absorption (equations 16-17, 22 of Inoue et al.
    2014), with separate expressions for ``z < 2.0`` and ``z >= 2.0``.
    Returns zero everywhere if ``z <= 0``.

    Parameters
    ----------
    wav_obs_arr : array-like
        Observed-frame wavelengths, in the same units as `wav_lyman_lim`.
    z : `float`
        Redshift of the source.

    Returns
    -------
    `numpy.ndarray`
        Lyman continuum DLA optical depth at each wavelength in
        `wav_obs_arr`.
    """
    tau = np.zeros(len(wav_obs_arr))
    gtr_lyman_lim_indices = wav_obs_arr > wav_lyman_lim
    if z > 0.0:
        if z < 2.0:
            wav_indices = (wav_obs_arr < wav_lyman_lim * (1 + z)) & (
                gtr_lyman_lim_indices
            )
            tau[wav_indices] = (
                (0.211 * (1 + z) ** 2)
                - (7.66e-2 * (1 + z) ** 2.3)
                * ((wav_obs_arr[wav_indices] / wav_lyman_lim) ** -1.3)
                - (0.135 * (wav_obs_arr[wav_indices] / wav_lyman_lim) ** 2)
            )
        else:
            wav_indices_1 = (
                wav_obs_arr < 3 * wav_lyman_lim
            ) & gtr_lyman_lim_indices
            wav_indices_2 = (
                (wav_obs_arr < wav_lyman_lim * (1 + z))
                & (~wav_indices_1)
                & (gtr_lyman_lim_indices)
            )
            tau[wav_indices_1] = (
                0.634
                + (4.7e-2 * (1 + z) ** 3)
                - (
                    1.78e-2
                    * ((1 + z) ** 3.3)
                    * ((wav_obs_arr[wav_indices_1] / wav_lyman_lim) ** -0.3)
                )
            )
            tau[wav_indices_2] = (
                4.7e-2 * (1 + z) ** 3
                - 1.78e-2
                * ((1 + z) ** 3.3)
                * ((wav_obs_arr[wav_indices_2] / wav_lyman_lim) ** -0.3)
                - 2.92e-2 * ((wav_obs_arr[wav_indices_2] / wav_lyman_lim) ** 3)
            )
    return tau


def calc_IGM_transmission(
    lyman_series,
    wav_rest_arr,
    z,
    prescription=config["MockSEDs"]["IGM_PRESCRIPTION"],
):
    """Compute the intergalactic medium (IGM) transmission spectrum for a source at redshift `z`.

    Converts `wav_rest_arr` to observed-frame wavelengths using `z`, sums
    the Lyman series and Lyman continuum LAF/DLA optical depths (for the
    ``"Inoue+14"`` prescription), and returns the corresponding
    transmission ``exp(-optical_depth)``.

    Parameters
    ----------
    lyman_series : `astropy.table.Table`
        Table of Lyman series transition data, as used by
        `calc_Inoue14_LS_LAF_optical_depth` and
        `calc_Inoue14_LS_DLA_optical_depth`.
    wav_rest_arr : `float`, `list` or `numpy.ndarray`
        Rest-frame wavelength(s), in the same units as `wav_lyman_lim`.
    z : `float`
        Redshift of the source.
    prescription : `str`, optional
        IGM attenuation prescription to use. Only ``"Inoue+14"`` is
        currently implemented. Default is
        ``config["MockSEDs"]["IGM_PRESCRIPTION"]``.

    Returns
    -------
    `numpy.ndarray`
        IGM transmission at each wavelength in `wav_rest_arr`.

    Raises
    ------
    Exception
        If `prescription` is not ``"Inoue+14"``.
    """
    if isinstance(wav_rest_arr, float):
        wav_rest_arr = np.array([wav_rest_arr])
    elif isinstance(wav_rest_arr, list):
        wav_rest_arr = np.array(wav_rest_arr)
    wav_obs_arr = wav_rest_arr * (1 + z)
    if prescription == "Inoue+14":
        optical_depth = np.array(
            calc_Inoue14_LC_DLA_optical_depth(wav_obs_arr, z)
            + calc_Inoue14_LC_LAF_optical_depth(wav_obs_arr, z)
            + calc_Inoue14_LS_DLA_optical_depth(lyman_series, wav_obs_arr, z)
            + calc_Inoue14_LS_LAF_optical_depth(lyman_series, wav_obs_arr, z)
        )
    else:
        raise (
            Exception(
                f"IGM attenuation not available for prescription = {prescription}. Please choose one of ['Inoue+14']"
            )
        )
    transmission = np.exp(-optical_depth)
    return transmission


def make_IGM_transmission_grid(
    wav_rest_arr, z_arr, prescription=config["MockSEDs"]["IGM_PRESCRIPTION"]
):
    """Compute and save a 2D grid of IGM transmission over rest wavelength and redshift.

    Loads the Lyman series absorption table from
    ``config['MockSEDs']['IGM_DIR']/LS_absorption.fits``, computes
    `calc_IGM_transmission` for every wavelength in `wav_rest_arr` at
    every redshift in `z_arr`, and saves the resulting grid (along with
    `z_arr` and `wav_rest_arr`) to an HDF5 file named
    ``"{prescription}_IGM_grid.h5"`` under ``config['MockSEDs']['IGM_DIR']``.

    Parameters
    ----------
    wav_rest_arr : array-like
        Rest-frame wavelengths at which to evaluate the transmission grid.
    z_arr : array-like
        Redshifts at which to evaluate the transmission grid.
    prescription : `str`, optional
        IGM attenuation prescription to use, passed to
        `calc_IGM_transmission` and used to name the output file. Default
        is ``config["MockSEDs"]["IGM_PRESCRIPTION"]``.
    """
    # allocate 2d IGM transmission grid memory
    IGM_transmission = np.zeros((len(z_arr), len(wav_rest_arr)))
    # load lyman series .fits file
    lyman_series = Table.read(
        f"{config['MockSEDs']['IGM_DIR']}/LS_absorption.fits"
    )
    # calculate 2d IGM transmission grid
    for i, z in tqdm(
        enumerate(z_arr),
        total=len(z_arr),
        desc=f"Making {prescription} IGM grid",
        disable=galfind_logger.getEffectiveLevel() > logging.INFO
    ):
        IGM_transmission[i, :] = calc_IGM_transmission(
            lyman_series, wav_rest_arr, z, prescription=prescription
        )
    with h5py.File(
        f"{config['MockSEDs']['IGM_DIR']}/{prescription}_IGM_grid.h5", "w"
    ) as IGM_grid:
        IGM_grid.create_dataset("IGM_transmission", data=IGM_transmission)
        IGM_grid.create_dataset("Redshifts", data=z_arr)
        IGM_grid.create_dataset("Rest_wavelengths", data=wav_rest_arr)
        IGM_grid.close()


class IGM:
    """Intergalactic medium (IGM) transmission model, backed by a precomputed grid.

    On construction, loads (creating if necessary) a 2D grid of IGM
    transmission as a function of rest-frame wavelength and redshift,
    spanning the Lyman limit to Lyman-alpha, which can then be
    interpolated to give transmission spectra at arbitrary redshift.

    Parameters
    ----------
    prescription : `str`, optional
        IGM attenuation prescription used to compute/load the
        transmission grid. Default is
        ``config["MockSEDs"]["IGM_PRESCRIPTION"]``.
    max_z : `float`, optional
        Maximum redshift of the transmission grid, only used if the grid
        needs to be created. Default is `25.0`.
    delta_z : `float`, optional
        Redshift grid spacing, only used if the grid needs to be created.
        Default is `0.01`.
    n_wav_rest : `int`, optional
        Number of rest-frame wavelength grid points between the Lyman
        limit and Lyman-alpha, only used if the grid needs to be created.
        Default is `10_000`.

    Attributes
    ----------
    prescription : `str`
        IGM attenuation prescription used for this transmission grid.
    transmission_grid : `numpy.ndarray`
        2D IGM transmission grid, indexed as ``[z_index, wav_rest_index]``,
        set by `load_IGM_transmission_grid`.
    z_arr : `numpy.ndarray`
        Redshift grid points of `transmission_grid`, set by
        `load_IGM_transmission_grid`.
    wav_rest_arr : `numpy.ndarray`
        Rest-frame wavelength grid points of `transmission_grid`, set by
        `load_IGM_transmission_grid`.
    """

    def __init__(
        self,
        prescription=config["MockSEDs"]["IGM_PRESCRIPTION"],
        max_z=25.0,
        delta_z=0.01,
        n_wav_rest=10_000,
    ):
        """Initialize the IGM transmission grid.

        Creates or loads a precomputed IGM transmission grid for the specified
        prescription and redshift/wavelength ranges.
        """
        self.prescription = prescription
        # make IGM grid if it doesn't exist, else load it
        if not Path(
            f"{config['MockSEDs']['IGM_DIR']}/{config['MockSEDs']['IGM_PRESCRIPTION']}_IGM_grid.h5"
        ).is_file():
            make_IGM_transmission_grid(
                np.linspace(wav_lyman_lim, wav_lyman_alpha, n_wav_rest),
                np.linspace(0.0, max_z, int(max_z / delta_z)),
            )
        self.load_IGM_transmission_grid()

    @property
    def interpolator(self):
        """`scipy.interpolate.RegularGridInterpolator`: Interpolator over `(z_arr, wav_rest_arr)` for `transmission_grid`, extrapolating outside its bounds."""
        return RegularGridInterpolator(
            (self.z_arr, self.wav_rest_arr),
            self.transmission_grid,
            bounds_error=False,
            fill_value=None,
        )  # extrapolate points too

    def load_IGM_transmission_grid(self):
        """Load the precomputed IGM transmission grid from disk into `transmission_grid`, `z_arr` and `wav_rest_arr`.

        Reads the HDF5 file at
        ``config['MockSEDs']['IGM_DIR']/{self.prescription}_IGM_grid.h5``,
        as created by `make_IGM_transmission_grid`.
        """
        with h5py.File(
            f"{config['MockSEDs']['IGM_DIR']}/{self.prescription}_IGM_grid.h5",
            "r",
        ) as IGM_grid:
            self.transmission_grid = IGM_grid["IGM_transmission"][()]
            self.z_arr = IGM_grid["Redshifts"][()]
            self.wav_rest_arr = IGM_grid["Rest_wavelengths"][()]
            IGM_grid.close()

    def plot_IGM_transmission_grid(
        self,
        ax,
        imshow_kwargs={},
        cbar_kwargs={},
        annotate=True,
        save=False,
        show=False,
    ):
        """Plot the full 2D IGM transmission grid as an image.

        Displays `transmission_grid` on `ax` using `ax.imshow`, with the
        wavelength range on the x-axis and redshift range on the y-axis.

        Parameters
        ----------
        ax : `matplotlib.pyplot.Axes`
            Axes to plot the transmission grid on.
        imshow_kwargs : `dict`, optional
            Additional keyword arguments passed to `ax.imshow`. Default is
            `{}`.
        cbar_kwargs : `dict`, optional
            Additional keyword arguments passed to `plt.colorbar`, only
            used if `annotate` is `True`. Default is `{}`.
        annotate : `bool`, optional
            Whether to add a colourbar and axis labels. Default is `True`.
        save : `bool`, optional
            Currently unused (no-op). Default is `False`.
        show : `bool`, optional
            Whether to call `plt.show()` after plotting. Default is `False`.
        """
        grid = ax.imshow(
            self.transmission_grid,
            aspect=(np.max(self.wav_rest_arr) - np.min(self.wav_rest_arr))
            / (np.max(self.z_arr) - np.min(self.z_arr)),
            extent=[
                np.min(self.wav_rest_arr),
                np.max(self.wav_rest_arr),
                np.min(self.z_arr),
                np.max(self.z_arr),
            ],
            **imshow_kwargs,
        )
        if annotate:
            plt.colorbar(grid, label="Transmission", **cbar_kwargs)
            ax.set_xlabel(r"$\lambda_{\mathrm{rest}}~/~\mathrm{\AA}$")
            ax.set_ylabel(r"Redshift, $z$")
        if save:
            # plt.savefig()
            pass
        if show:
            plt.show()

    def interp_transmission(
        self, z, wav_rest_arr
    ):  # wav_rest_arr should have units
        """Interpolate the IGM transmission grid at a given redshift and rest wavelengths.

        Wavelengths below `wav_lyman_lim` are assigned zero transmission,
        wavelengths above `wav_lyman_alpha` are assigned unit transmission,
        and wavelengths in between are interpolated from `interpolator`.

        Parameters
        ----------
        z : `float`
            Redshift at which to evaluate the transmission.
        wav_rest_arr : `astropy.units.Quantity`
            Rest-frame wavelengths at which to evaluate the transmission.

        Returns
        -------
        `list`
            Transmission values corresponding to each wavelength in
            `wav_rest_arr`.
        """
        wav_rest_arr = wav_rest_arr.to(u.AA).value
        # calculate rest wavelengths from self in the appropriate wavelength range between wav_lyman_lim and wav_lyman_alpha
        attenuated_indices = (wav_rest_arr > wav_lyman_lim) & (
            wav_rest_arr <= wav_lyman_alpha
        )
        interp_pts = [
            [z, wav_rest] for wav_rest in wav_rest_arr[attenuated_indices]
        ]
        transmission_arr = (
            list(
                np.zeros(len(wav_rest_arr[wav_rest_arr <= wav_lyman_lim] - 1))
            )
            + list(self.interpolator(interp_pts))
            + list(np.ones(len(wav_rest_arr[wav_rest_arr > wav_lyman_alpha])))
        )
        return transmission_arr

    def plot_slice(
        self,
        ax,
        z,
        wav_rest_arr,
        frame="rest",
        plot_kwargs={},
        legend_kwargs={},
        annotate=True,
        save=False,
        show=False,
    ):
        """Plot a single IGM transmission spectrum at fixed redshift.

        Computes the transmission at `z` via `interp_transmission` and
        plots it against rest-frame or observed-frame wavelength.

        Parameters
        ----------
        ax : `matplotlib.pyplot.Axes`
            Axes to annotate with title/labels/limits if `annotate` is
            `True`. The plotting itself is done via `plt.plot`.
        z : `float`
            Redshift at which to evaluate the transmission.
        wav_rest_arr : `astropy.units.Quantity`
            Rest-frame wavelengths at which to evaluate the transmission.
        frame : `str`, optional
            Frame to plot the wavelength axis in, either ``"rest"`` or
            ``"obs"``. Default is ``"rest"``.
        plot_kwargs : `dict`, optional
            Additional keyword arguments passed to `plt.plot`. Default is
            `{}`.
        legend_kwargs : `dict`, optional
            Additional keyword arguments passed to `plt.legend`, only used
            if `annotate` is `True`. Default is `{}`.
        annotate : `bool`, optional
            Whether to set the axis title, labels and limits, and draw a
            legend. Default is `True`.
        save : `bool`, optional
            Currently unused (no-op). Default is `False`.
        show : `bool`, optional
            Whether to call `plt.show()` after plotting. Default is `False`.

        Raises
        ------
        Exception
            If `frame` is not ``"rest"`` or ``"obs"``.
        """
        transmission_arr = self.interp_transmission(z, wav_rest_arr)
        if frame == "rest":
            plt.plot(wav_rest_arr.value, transmission_arr, **plot_kwargs)
        elif frame == "obs":
            plt.plot(
                wav_rest_arr.value * (1 + z), transmission_arr, **plot_kwargs
            )
        else:
            raise (
                Exception(
                    f"frame = {frame} is invalid! Please choose either 'rest' or 'obs'"
                )
            )
        if annotate:
            ax.set_title(f"{self.prescription} IGM attenuation")
            if frame == "rest":
                ax.set_xlabel(
                    r"$\lambda_{\mathrm{rest}}~/~\mathrm{%s}$"
                    % wav_rest_arr.unit
                )
                ax.set_xlim(wav_lyman_lim, wav_lyman_alpha)
            else:
                ax.set_xlabel(
                    r"$\lambda_{\mathrm{obs}}~/~\mathrm{%s}$"
                    % wav_rest_arr.unit
                )
            ax.set_ylabel("Transmission")
            ax.set_ylim(0.0, 1.0)
            plt.legend(**legend_kwargs)
        if save:
            # plt.savefig()
            pass
        if show:
            plt.show()

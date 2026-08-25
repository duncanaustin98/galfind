"""Number density function containers and statistical fitting.

Provides classes for storing binned number densities (UVLFs, mass
functions) with
uncertainties and includes MCMC/Schechter fitters for function fitting.
"""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    NoReturn,
    Optional,
    Tuple,
    Union,
)

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from numpy.typing import NDArray

if TYPE_CHECKING:
    from . import (
        Catalogue_Base,
        Mask_Selector,
        Property_Calculator,
    )
    from .selection import Completeness, Completeness_2D
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from .. import config, galfind_logger
from ..sed_fitting.SED_codes import SED_code
from ..utils import useful_funcs_austind as funcs
from ..utils.exceptions import (
    ExternalToolError,
    GalfindError,
    GalfindTypeError,
    InvalidOptionError,
    InvalidUnitError,
    LengthMismatchError,
    MissingDataError,
    MissingKeyError,
    RangeError,
)
from ..utils.MCMC import MCMC_Fitter, Priors


class Base_Number_Density_Function:
    """Base class for number density functions (UVLFs, mass functions, etc.).

    Stores binned number density measurements at a reference redshift,
    including
    measurements and their uncertainties. Subclasses add additional
    functionality
    for computing densities and handling multiple functions.

    Parameters
    ----------
    x_name : `str`
        Property name (e.g., "UVLF", "stellar_mass").
    x_mid_bins : `numpy.ndarray`
        Bin centers for the property x.
    z_ref : `str`, `int`, or `float`
        Reference redshift for this measurement.
    phi : `numpy.ndarray`
        Number density measurements.
    phi_errs_cv : `numpy.ndarray`
        Uncertainties on number densities (including cosmic variance).
    author_year : `str`
        Citation identifier (e.g., "Finkelstein2016").

    Attributes
    ----------
    x_name : `str`
        Property name.
    x_mid_bins : `numpy.ndarray`
        Bin centers.
    z_ref : `float`
        Reference redshift.
    phi : `numpy.ndarray`
        Number densities.
    phi_errs_cv : `numpy.ndarray`
        Density uncertainties.
    author_year : `str`
        Citation.
    """

    def __init__(
        self: Self,
        x_name: str,
        x_mid_bins: NDArray[float],
        z_ref: Union[str, int, float],
        phi: NDArray[float],
        phi_errs_cv: NDArray[float],
        author_year: str,
    ):
        self.x_name = x_name
        self.x_mid_bins = x_mid_bins
        self.z_ref = z_ref
        self.phi = phi
        self.phi_errs_cv = phi_errs_cv
        self.author_year = author_year

    # obsolete after Base_Number_Density_Function.from_flags_repo()
    @classmethod
    def from_ecsv(
        cls, x_name: str, z_ref: Union[str, int, float], author_year: str
    ) -> Self:
        """Load a number density function from an ECSV data file.

        Parameters
        ----------
        x_name : `str`
            Property name (e.g., ``"M1500"``, ``"M_UV"``, ``"stellar_mass"``).
        z_ref : `str`, `int`, or `float`
            Reference redshift.
        author_year : `str`
            Author and year label for the publication.

        Returns
        -------
        `Base_Number_Density_Function`
            A number density function object loaded from ECSV.
        """
        if x_name in ["M1500", "M_UV", "MUV"]:
            x_name = "UVLF"
        if isinstance(z_ref, (str, int)):
            z_ref = float(z_ref)
        x_name_config_key = f"{x_name}_LIT_DIR"
        ecsv_data_path = (
            f"{config['NumberDensityFunctions'][x_name_config_key]}"
            f"/z={z_ref:.1f}/{author_year}.ecsv"
        )
        tab = Table.read(ecsv_data_path)
        x_mid_bins = np.array(tab["M_UV"])
        phi = np.array(tab["phi"])
        phi_errs_cv = np.array([tab["phi_l1"], tab["phi_u1"]])
        return cls(x_name, x_mid_bins, z_ref, phi, phi_errs_cv, author_year)

    @classmethod
    def from_flags_repo(
        cls,
        x_name: str,
        z_bin: Union[list, np.array],
        author_year: str,
        obs_or_models: str = "obs",
    ) -> Optional[Self]:
        """Load a number density function from the FLAGS data repository.

        Parameters
        ----------
        x_name : `str`
            Property name.
        z_bin : `list` or `numpy.ndarray`
            Redshift bin edges ``[z_min, z_max]``.
        author_year : `str`
            Author and year label for the publication.
        obs_or_models : `str`, optional
            Data type: ``"obs"`` for observations or ``"models/binned"``
            for models.
            Default is ``"obs"``.

        Returns
        -------
        `Base_Number_Density_Function` or `None`
            A number density function object, or `None` if not found.

        Raises
        ------
        InvalidOptionError
            If `obs_or_models` is not one of ``"obs"`` or
            ``"models/binned"``.
        ExternalToolError
            If ``flags_data.distribution_functions`` cannot be imported,
            or if the requested `x_name` has no matching entry in the
            loaded dataset.
        """
        if obs_or_models not in ["obs", "models/binned"]:
            raise InvalidOptionError(
                f"obs_or_models={obs_or_models!r} not recognised; must be "
                "one of ['obs', 'models/binned']."
            )
        sys.path.insert(1, config["NumberDensityFunctions"]["FLAGS_DATA_DIR"])
        try:
            from flags_data import distribution_functions
        except Exception as e:
            raise ExternalToolError(
                "Could not import flags_data.distribution_functions from "
                f"config['NumberDensityFunctions']['FLAGS_DATA_DIR']="
                f"{config['NumberDensityFunctions']['FLAGS_DATA_DIR']!r}: {e}"
            ) from e

        flags_property_name_conv = {
            "M1500": "LUV",
            "M1500_[1250,3000]AA": "LUV",
            "M1500_[1250,3000]AA_extsrc": "LUV",
            "M1500_[1250,3000]AA_extsrc_UV<10": "LUV",
            "M_UV": "LUV",
            "stellar_mass": "Mstar",
        }

        datasets = distribution_functions.list_datasets(
            f"{flags_property_name_conv[x_name]}/{obs_or_models}"
        )

        # num_obs = np.linspace(0, 1, len(datasets))
        z_ref = (z_bin[0] + z_bin[1]) / 2.0
        for pos, path in enumerate(datasets):
            ds = distribution_functions.read(path, verbose=False)
            if all(string in ds.name for string in author_year.split("+")):
                # choose closest redshift to bin centre
                z = None
                deltaz = 100
                for z_i in ds.redshifts:
                    delta_z_i = np.abs(z_i - z_ref)
                    # must be within redshift bin
                    if delta_z_i <= (z_bin[1] - z_bin[0]) / 2.0:
                        if delta_z_i < deltaz:
                            deltaz = delta_z_i
                            z = float(z_i)
                if z is None:
                    galfind_logger.warning(
                        f"No available redshift for {author_year=} "
                        f"in {z_bin=}!"
                        + f" Available redshifts are {ds.redshifts}"
                    )
                    return None
                else:
                    label = (
                        ds.slabel.replace(r"\rm", "")
                        .replace("$", "")
                        .replace("\\", "")
                        .replace(" ", "")
                    )

                    label = f"{label},z={z}"

                    try:
                        if x_name in [
                            key
                            for (key, val) in flags_property_name_conv.items()
                            if val == "LUV"
                        ]:
                            x = ds.M[z]
                        else:
                            x = ds.log10X[z]
                    except Exception as e:
                        attr_name = (
                            "M"
                            if flags_property_name_conv[x_name] == "LUV"
                            else "log10X"
                        )
                        raise ExternalToolError(
                            f"flags_data dataset {ds.name!r} has no "
                            f"{attr_name} entry for z={z} (requested "
                            f"x_name={x_name!r}): {e}"
                        ) from e

                    if flags_property_name_conv[x_name] == "LUV":
                        if obs_or_models == "obs":
                            phi_err = np.array(ds.log10phi_mag_err[z])
                        log10phi = ds.log10phi_mag[z]
                    else:
                        if obs_or_models == "obs":
                            phi_err = np.array(ds.log10phi_err[z])
                        log10phi = ds.log10phi[z]
                    if obs_or_models == "obs":
                        if len(np.shape(phi_err)) > 1:
                            low = np.array(phi_err[0])
                            high = np.array(phi_err[1])
                        else:
                            low = high = phi_err
                        err_high = 10 ** (log10phi + high) - 10**log10phi
                        err_low = (10**log10phi) - 10 ** (log10phi - low)
                        phi_err = np.array([err_low, err_high])
                    else:  # obs_or_models == "models/binned"
                        phi_err = np.zeros((2, len(log10phi)))

                    # x = ds.log10X[z] - np.log10(
                    #     1. / funcs.imf_mass_factor[ds.imf]
                    # ) stellar mass only
                    return cls(
                        x_name,
                        x,
                        z,
                        10**log10phi,
                        phi_err,
                        author_year,
                    )
        galfind_logger.info(
            f"No {author_year=} in {obs_or_models} for {x_name=}"
        )
        return None  # if no author_year in flags_data

    def __add__(
        self: Self,
        other: Union[
            Type[Base_Number_Density_Function],
            List[Type[Base_Number_Density_Function]],
            Multiple_Number_Density_Function,
        ],
    ) -> Multiple_Number_Density_Function:
        base_ndf_subcls = tuple(Base_Number_Density_Function.__subclasses__())
        if isinstance(other, list):
            if not all(isinstance(ndf, base_ndf_subcls) for ndf in other):
                raise GalfindTypeError(
                    f"other={other!r} contains an element that is not an "
                    f"instance of {base_ndf_subcls}."
                )
            number_density_funcs = [self] + other
        elif isinstance(other, base_ndf_subcls):
            number_density_funcs = [self, other]
        elif isinstance(other, Multiple_Number_Density_Function):
            if not all(
                isinstance(ndf, base_ndf_subcls)
                for ndf in other.number_density_functions
            ):
                raise GalfindTypeError(
                    f"other.number_density_functions="
                    f"{other.number_density_functions!r} contains an "
                    f"element that is not an instance of {base_ndf_subcls}."
                )
            number_density_funcs = [self] + other.number_density_functions
        else:
            raise GalfindTypeError(
                f"other={other!r} has type {type(other)}; must be one of "
                f"{base_ndf_subcls}, List[{base_ndf_subcls}], or "
                "Multiple_Number_Density_Function."
            )
        # TODO: Ensure no duplicates
        multiple_ndf = Multiple_Number_Density_Function(number_density_funcs)
        return multiple_ndf

    def __len__(self):
        return len(self.phi)

    def get_z_bin_name(self) -> str:
        """Get the redshift bin name label.

        Returns
        -------
        `str`
            Label of the form ``"z=<z_ref>"``.
        """
        return f"z={float(self.z_ref):.1f}"

    def plot(
        self,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        log_x: bool = False,
        log_y: bool = False,
        annotate: bool = False,
        save: bool = False,
        show: bool = False,
        plot_kwargs: dict = {},
        legend_kwargs: dict = {},
        x_lims: Optional[Union[List[float], str]] = "default",
        y_lims: Optional[List[float]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        plot_cv_errs: bool = True,
        offset: float = 0.0,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the number density function.

        Parameters
        ----------
        fig : `matplotlib.figure.Figure`, optional
            Figure to plot on. Default is `None`.
        ax : `matplotlib.axes.Axes`, optional
            Axes to plot on. Default is `None`.
        log_x : `bool`, optional
            Whether to use log scale for x-axis. Default is `False`.
        log_y : `bool`, optional
            Whether to use log scale for y-axis. Default is `False`.
        annotate : `bool`, optional
            Whether to add annotations. Default is `False`.
        save : `bool`, optional
            Whether to save the figure. Default is `False`.
        show : `bool`, optional
            Whether to display the figure. Default is `False`.
        plot_kwargs : `dict`, optional
            Keyword arguments for plotting. Default is empty dict.
        legend_kwargs : `dict`, optional
            Keyword arguments for legend. Default is empty dict.
        x_lims : `list` of `float` or `str`, optional
            X-axis limits or ``"default"``. Default is ``"default"``.
        y_lims : `list` of `float`, optional
            Y-axis limits. Default is `None`.
        title : `str`, optional
            Plot title. Default is `None`.
        save_path : `str`, optional
            Path to save figure. Default is `None`.
        plot_cv_errs : `bool`, optional
            Whether to plot cosmic variance errors. Default is `True`.
        offset : `float`, optional
            X-axis offset. Default is 0.0.

        Returns
        -------
        `tuple` of (`matplotlib.figure.Figure`, `matplotlib.axes.Axes`)
            Figure and axes used for plotting.

        Raises
        ------
        LengthMismatchError
            If `x_lims` or `y_lims` is given and does not have length 2.
        """

        if all(i is None for i in [fig, ax]):
            fig_, ax_ = plt.subplots()
        else:
            fig_, ax_ = fig, ax

        # don't plot empty bins
        if isinstance(self.x_mid_bins, (u.Quantity, u.Magnitude, u.Dex)):
            x_mid_bins = np.array(
                [
                    _x
                    for _x, _y in zip(self.x_mid_bins.value, self.phi)
                    if _y != 0.0
                ]
            )
        else:
            x_mid_bins = np.array(
                [_x for _x, _y in zip(self.x_mid_bins, self.phi) if _y != 0.0]
            )
        x_mid_bins += offset
        phi = np.array([_y for _y in self.phi if _y != 0.0])
        if (
            not plot_cv_errs
            and hasattr(self, "phi_errs")
            and isinstance(self, Number_Density_Function)
        ):
            phi_errs_ = self.phi_errs
        else:
            phi_errs_ = self.phi_errs_cv
        phi_errs = np.array(
            [
                [
                    _yerr
                    for _yerr, _y in zip(phi_errs_[0], self.phi)
                    if _y != 0.0
                ],
                [
                    _yerr
                    for _yerr, _y in zip(phi_errs_[1], self.phi)
                    if _y != 0.0
                ],
            ]
        )
        if log_y:
            y = np.log10(phi)
            y_errs = np.array(
                [
                    np.log10(_phi / (_phi - _phi_err))
                    for _phi, _phi_err in zip(phi, phi_errs)
                ],
                [
                    np.log10(1.0 + (_phi_err / _phi))
                    for _phi, _phi_err in zip(phi, phi_errs)
                ],
            )
        else:
            y = phi
            y_errs = phi_errs

        # sort out plot_kwargs
        default_plot_kwargs = {
            "ls": "",
            "marker": "o",
            "label": f"{self.author_year}, {self.get_z_bin_name()}",
        }
        # overwrite default with input for duplicate kwargs
        for key in plot_kwargs.keys():
            if key in default_plot_kwargs.keys():
                default_plot_kwargs.pop(key)
                default_plot_kwargs[key] = plot_kwargs[key]
        _plot_kwargs = {**plot_kwargs, **default_plot_kwargs}

        galfind_logger.info(f"Plotting {default_plot_kwargs['label']}")
        if all([y_err == 0.0 for y_err in y_errs.flatten()]):
            galfind_logger.debug(
                f"No errors to plot for {default_plot_kwargs['label']}"
            )
            line = ax_.plot(x_mid_bins, y, **_plot_kwargs)
        else:
            # turn nans into upper limits
            if any(np.isnan(y_errs[0])):
                upper_limit_indices = np.where(np.isnan(y_errs[0]))[0]
                y_errs[0][upper_limit_indices] = 0.5 * y[upper_limit_indices]
                _plot_kwargs["uplims"] = [
                    True if i in upper_limit_indices else False
                    for i in range(len(y))
                ]
            line = ax_.errorbar(x_mid_bins, y, yerr=y_errs, **_plot_kwargs)

        if annotate:
            y_label = r"$\Phi$ / N dex$^{-1}$Mpc$^{-3}$"
            if log_x:
                x_label = r"$\log_{10}($" + self.x_name + r"$)$"
                ax_.set_xscale("log")
            else:
                x_label = self.x_name
            if log_y:
                y_label = r"$\log_{10}($" + y_label + r"$)$"
            else:
                ax_.set_yscale("log")
            ax_.set_xlabel(x_label)
            ax_.set_ylabel(y_label)
            if title is not None:
                ax_.set_title(title)
            # sort out legend_kwargs
            default_legend_kwargs = {
                "loc": "best",
                # "bbox_to_anchor": (1.05, 0.5),
            }
            # overwrite default with input for duplicate kwargs
            for key in legend_kwargs.keys():
                if key in default_legend_kwargs.keys():
                    default_legend_kwargs.pop(key)
            _legend_kwargs = {**legend_kwargs, **default_legend_kwargs}
            ax_.legend(**_legend_kwargs)

        if x_lims is not None:
            if isinstance(x_lims, str):
                if x_lims == "default":
                    x_lims = self.x_name
                ax_.set_xlim(*funcs.default_lims[x_lims])
            else:
                if len(x_lims) != 2:
                    raise LengthMismatchError(
                        f"x_lims={x_lims!r} has length {len(x_lims)}; must "
                        "have length 2 ([x_min, x_max])."
                    )
                ax_.set_xlim(*x_lims)
        if y_lims is not None:
            if len(y_lims) != 2:
                raise LengthMismatchError(
                    f"y_lims={y_lims!r} has length {len(y_lims)}; must "
                    "have length 2 ([y_min, y_max])."
                )
            ax.set_ylim(*y_lims)

        if save:
            if save_path is None:
                save_path = self.get_plot_path()
                # if save_name is not None:
                #     plot_path = "/".join(
                #         plot_path.split("/")[:-1]
                #     ) + f"/{save_name}.png"
            funcs.make_dirs(save_path)
            plt.savefig(save_path, bbox_inches="tight")
            funcs.change_file_permissions(save_path)
            galfind_logger.info(f"Saved plot to {save_path}")
        if show:
            plt.show()
        return fig_, ax_, line


class Number_Density_Function(Base_Number_Density_Function):
    """Number density function computed from a galaxy sample.

    Computes and stores binned number densities for a property (e.g., UV
    luminosity,
    stellar mass) from a galaxy catalogue, including cosmic variance and
    completeness
    corrections.

    Parameters
    ----------
    x_name : `str`
        Property name (e.g., "M_UV", "log_M_*").
    x_bins : array-like
        Bin edges for the property.
    x_origin : `str`
        Data source for the property.
    z_bin : `tuple` of (`float`, `float`)
        Redshift bin (z_min, z_max).
    Ngals : `int`
        Number of galaxies in the sample.
    phi : `numpy.ndarray`
        Number density per bin.
    phi_errs : `numpy.ndarray`
        Poisson uncertainties (2 elements: lower, upper).
    cv_errs : `float`
        Cosmic variance as a fractional uncertainty.
    origin_surveys : `str`
        Comma-separated survey names.
    crop_name : `str`
        Name of the spatial crop/region.
    cv_origin : `str`
        Source of cosmic variance estimates.
    completeness : `Completeness` or `None`, optional
        Completeness correction object. Default is `None`.
    Vmax_method : `str`, optional
        Method for volume calculation. Default is "uniform_depth".
    """

    def __init__(
        self,
        x_name: str,
        x_bins,
        x_origin,
        z_bin,
        Ngals: int,
        phi,
        phi_errs,
        cv_errs,
        origin_surveys,
        crop_name: str,
        cv_origin,
        completeness: Optional[Completeness] = None,
        Vmax_method: str = "uniform_depth",
    ):
        self.crop_name = crop_name
        self.x_bins = x_bins
        self.x_origin = x_origin
        self.z_bin = z_bin
        self.Ngals = Ngals
        self.phi_errs = phi_errs  # poisson only
        self.cv_errs = cv_errs  # cosmic variance % errs / 100
        self.origin_surveys = origin_surveys
        self.cv_origin = cv_origin
        self.completeness = completeness
        self.Vmax_method = Vmax_method
        x_mid_bins = (
            np.array(
                [(x_bin[1].value + x_bin[0].value) / 2.0 for x_bin in x_bins]
            )
            * x_bins[0].unit
        )

        z_ref = float((z_bin[1] + z_bin[0]) / 2.0)
        phi_errs_cv = np.array(
            [
                np.sqrt(phi_errs[i] ** 2.0 + (cv_errs * phi) ** 2.0)
                for i in range(2)
            ]
        )
        super().__init__(
            x_name,
            x_mid_bins,
            z_ref,
            phi,
            phi_errs_cv,
            "This work",
        )

    @classmethod
    def from_ecsv(
        cls: Type[Number_Density_Function],
        save_path: str,
        completeness: Optional[Completeness] = None,
    ) -> Self:
        """Load a number density function from an ECSV file.

        Parameters
        ----------
        save_path : `str`
            Path to the ECSV file.
        completeness : `Completeness`, optional
            Completeness correction object. Default is `None`.

        Returns
        -------
        `Number_Density_Function`
            Loaded number density function.
        """
        tab = Table.read(save_path)
        x_bins_up = np.array(tab["x_bins_up"])
        x_bins_low = np.array(tab["x_bins_low"])
        x_unit = tab["x_bins_up"].unit
        x_bins = (
            np.array(
                [
                    [x_bin_low, x_bin_up]
                    for x_bin_low, x_bin_up in zip(x_bins_low, x_bins_up)
                ]
            )
            * x_unit
        )
        Ngals = np.array(tab["Ngals"])
        phi = np.array(tab["phi"])
        phi_l1 = np.array(tab["phi_l1"])
        phi_u1 = np.array(tab["phi_u1"])
        cv_errs = np.array(tab["cv_errs"])
        # SED_fit_params_key, x_name, origin_surveys, z_bin = \
        #     Number_Density_Function.extract_info_from_save_path(
        #         save_path
        #     )
        cv_origin = tab.meta["cv_origin"]
        x_origin = tab.meta["x_origin"]
        x_name = tab.meta["x_name"]
        origin_surveys = tab.meta["origin_surveys"]
        crop_name = tab.meta["crop_name"]
        z_bin = tab.meta["z_bin"]
        if "Vmax_method" in tab.meta.keys():
            Vmax_method = tab.meta["Vmax_method"]
        else:
            Vmax_method = None
        return cls(
            x_name,
            x_bins,
            x_origin,
            z_bin,
            Ngals,
            phi,
            np.array([phi_l1, phi_u1]),
            cv_errs,
            origin_surveys,
            crop_name,
            cv_origin,
            completeness,
            Vmax_method,
        )

    @classmethod
    def from_cat(
        cls,
        cat: Type[Catalogue_Base],
        x_calculator: Type[Property_Calculator],
        x_bin_edges: List[float],
        z_bin: List[float],
        aper_diam: u.Quantity,
        SED_fit_code: SED_code,
        x_origin: str = "phot_rest",
        z_step: float = 0.01,
        cv_origin: Union[str, None] = "Driver2010",
        completeness: Optional[Dict[str, Dict[str, Completeness_2D]]] = None,
        unmasked_area: Union[
            str, List[str], u.Quantity, Type[Mask_Selector]
        ] = "selection",
        plot: bool = True,
        save: bool = True,
        timed: bool = False,
        Vmax_method: str = "uniform_depth",
        n_Vmax_jobs: int = 1,
    ) -> Optional[Self]:
        """Compute a number density function from a catalogue.

        Parameters
        ----------
        cat : `Catalogue_Base`
            Source catalogue.
        x_calculator : `Property_Calculator`
            Calculator for the property binned in the function.
        x_bin_edges : `list` of `float`
            Bin edges for the property.
        z_bin : `list` of `float`
            Redshift bin ``[z_min, z_max]``.
        aper_diam : `astropy.units.Quantity`
            Aperture diameter.
        SED_fit_code : `SED_code`
            SED-fitter code.
        x_origin : `str`, optional
            Origin label for the property. Default is ``"phot_rest"``.
        z_step : `float`, optional
            Redshift step for z-dependent corrections. Default is 0.01.
        cv_origin : `str`, optional
            Source of cosmic variance estimates. Default is
            ``"Driver2010"``.
        completeness : `dict`, optional
            Completeness corrections. Default is `None`.
        unmasked_area : `str`, `list`, `Quantity`, or `Mask_Selector`, optional
            Unmasked area specification. Default is ``"selection"``.
        plot : `bool`, optional
            Whether to plot the result. Default is `True`.
        save : `bool`, optional
            Whether to save the result. Default is `True`.
        timed : `bool`, optional
            Whether to time the computation. Default is `False`.
        Vmax_method : `str`, optional
            Method for volume calculation. Default is ``"uniform_depth"``.
        n_Vmax_jobs : `int`, optional
            Number of parallel jobs for Vmax. Default is 1.

        Returns
        -------
        `Number_Density_Function` or `None`
            The computed number density function, or `None` if calculation
            fails.

        Raises
        ------
        LengthMismatchError
            If `z_bin` does not have length 2.
        RangeError
            If ``z_bin[0]`` is not less than ``z_bin[1]``, if `x_bin_edges`
            has fewer than 2 elements, or if `x_bin_edges` is not sorted
            in ascending order.
        InvalidOptionError
            If `cv_origin` is not ``"Driver2010"``, or if `x_origin` is
            not one of ``"phot_rest"``/``"SED_result"``.
        GalfindTypeError
            If `SED_fit_code` is not an instance of a registered
            `SED_code` subclass, or if `cat` is not a `Catalogue` or
            `Combined_Catalogue`.
        MissingDataError
            If not every galaxy in `cat` has been fit with
            `SED_fit_code`.
        InvalidUnitError
            If the extracted x values do not all share the same unit.
        """
        from ..catalogues.Multiple_Catalogue import Combined_Catalogue

        if isinstance(cat, Combined_Catalogue):
            plot = False
        # input assertions
        if len(z_bin) != 2:
            raise LengthMismatchError(
                f"z_bin={z_bin!r} has length {len(z_bin)}; must have "
                "length 2 ([z_min, z_max])."
            )
        if not z_bin[0] < z_bin[1]:
            raise RangeError(
                f"z_bin[0]={z_bin[0]} must be less than z_bin[1]={z_bin[1]}."
            )
        if len(x_bin_edges) < 2:
            raise RangeError(
                f"x_bin_edges={x_bin_edges!r} has length "
                f"{len(x_bin_edges)}; must have at least 2 elements to "
                "define at least one bin."
            )
        # ensure x_bin_edges are sorted from lower to higher x
        # values in every z bin
        if not all(
            _x == _sorted_x
            for _x, _sorted_x in zip(
                np.sort(np.array(x_bin_edges)), np.array(x_bin_edges)
            )
        ):
            raise RangeError(
                f"x_bin_edges={x_bin_edges!r} must be sorted in "
                "ascending order."
            )
        # TODO: ensure x_bin_edges are evenly spaced?

        if cv_origin not in ["Driver2010"]:
            raise InvalidOptionError(
                f"cv_origin={cv_origin!r} not recognised; must be "
                "'Driver2010'."
            )
        # SED fit label assertions
        if not isinstance(SED_fit_code, tuple(SED_code.__subclasses__())):
            raise GalfindTypeError(
                f"SED_fit_code has type {type(SED_fit_code)}; must be an "
                f"instance of one of {tuple(SED_code.__subclasses__())}."
            )
        if not all(
            SED_fit_code.label in gal.aper_phot[aper_diam].SED_results.keys()
            for gal in cat
        ):
            raise MissingDataError(
                f"Not every galaxy in {cat!r} has SED_fit_code.label="
                f"{SED_fit_code.label!r} in its aper_phot[{aper_diam!r}]"
                ".SED_results; run this SED fit first."
            )
        # x_origin assertions
        if x_origin not in ["phot_rest", "SED_result"]:
            raise InvalidOptionError(
                f"x_origin={x_origin!r} not recognised; must be one of "
                "['phot_rest', 'SED_result']."
            )

        # extract x values
        # TODO: Generalize this to exclude x_origin dependence
        if x_origin == "phot_rest":
            x = [
                gal.aper_phot[aper_diam]
                .SED_results[SED_fit_code.label]
                .phot_rest.properties[x_calculator.name]
                for gal in cat
            ]
        else:  # x_origin == "SED_result":
            x = [
                gal.aper_phot[aper_diam]
                .SED_results[SED_fit_code.label]
                .properties[x_calculator.name]
                for gal in cat
            ]
        # remove nans
        x = [x_ for x_ in x if not np.isnan(x_)]
        if not all(x_.unit == x[0].unit for x_ in x):
            raise InvalidUnitError(
                f"Not all extracted x values share the same unit; "
                f"expected {x[0].unit!r} for every element of {x!r}."
            )
        x = np.array([x_.value for x_ in x]) * x[0].unit

        # crop catalogue to this redshift bin
        from . import Redshift_Bin_Selector

        # TODO: Implement Redshift_Limit_Selector in case of np.nan z_bin entry
        z_bin_selector = Redshift_Bin_Selector(aper_diam, SED_fit_code, z_bin)
        z_bin_cat = deepcopy(cat).crop(z_bin_selector)
        # ensure every galaxy in this redshift bin has
        # the relevant property already calculated
        if (
            len(
                [
                    i
                    for i, gal in enumerate(z_bin_cat)
                    if np.isnan(
                        gal.aper_phot[aper_diam]
                        .SED_results[SED_fit_code.label]
                        .phot_rest.properties[x_calculator.name]
                    )
                ]
            )
            != 0
        ):
            nan_gals = [
                gal
                for gal in z_bin_cat
                if np.isnan(
                    gal.aper_phot[aper_diam]
                    .SED_results[SED_fit_code.label]
                    .phot_rest.properties[x_calculator.name]
                )
            ]
            galfind_logger.warning(
                f"{len(nan_gals)} {repr(x_calculator)} nans for {z_bin=}!"
            )
            for gal in nan_gals:
                galfind_logger.warning(
                    f"{gal.ID}: "
                    f"(z={gal.aper_phot[aper_diam].SED_results[SED_fit_code.label].z:.2f}"
                    + f",{gal.aper_phot[aper_diam].filterset.filt_names})"
                )
            # remove nan_gals from z_bin_cat
            z_bin_cat.gals = [gal for gal in z_bin_cat if gal not in nan_gals]

        if len(z_bin_cat) == 0:
            galfind_logger.warning(f"No galaxies in {z_bin=}")
            return None

        # determine save_path
        # full_survey_name = funcs.get_full_survey_name(
        #     cat.survey,
        #     cat.version,
        #     cat.filterset
        # )
        save_path = Number_Density_Function.get_save_path(
            cat.survey,
            x_origin,
            x_calculator.name,
            z_bin_cat.crop_name,
            completeness=completeness,
            Vmax_method=Vmax_method,
        )

        if not Path(save_path).is_file():
            # create x_bins from x_bin_edges (must include start
            # and end values here too)
            x_bins = [
                [x_bin_edges[i].value, x_bin_edges[i + 1].value]
                * x_bin_edges.unit
                for i in range(len(x_bin_edges) - 1)
                if i != len(x_bin_edges) - 1
            ]
            # calculate Vmax for each galaxy in catalogue within z bin
            z_bin_cat.calc_Vmax(
                z_bin,
                aper_diam,
                SED_fit_code,
                z_step,
                unmasked_area=unmasked_area,
                Vmax_method=Vmax_method,
                n_jobs=n_Vmax_jobs,
            )

            if plot:
                # overall_fig, fig_axs_ = figs.make_phot_diagnostic_fig(
                #     len(z_bin_cat.filterset)
                # )
                z_bin_cat.plot_phot_diagnostics(
                    aper_diam,
                    SED_arr=SED_fit_code,
                    zPDF_arr=SED_fit_code,
                    # fig_axs = fig_axs_
                )
                # plt.close(fig_axs_[0].figure)

            Ngals = np.zeros(len(x_bins))
            phi = np.zeros(len(x_bins))
            phi_l1 = np.zeros(len(x_bins))
            phi_u1 = np.zeros(len(x_bins))
            cv_errs = np.zeros(len(x_bins))
            # phi_errs_cv = np.zeros(len(x_bins))
            # loop through each mass bin in the given redshift bin
            for i, x_bin in enumerate(x_bins):
                if len(z_bin_cat) == 0:
                    Ngals[i] = 0
                else:
                    if plot:
                        # plot histogram
                        hist_fig, hist_ax = plt.subplots()
                        z_bin_cat.hist(
                            x_calculator,
                            hist_fig,
                            hist_ax,
                            from_pdf=True,
                            save=False,
                            overwrite=True,
                            density=True,
                        )
                        z_bin_cat.hist(
                            x_calculator,
                            hist_fig,
                            hist_ax,
                            from_pdf=False,
                            save=True,
                            overwrite=True,
                            density=True,
                        )
                        plt.close(hist_fig)

                    # crop to galaxies in the x bin - not the
                    # bootstrapping method
                    from . import Rest_Frame_Property_Bin_Selector

                    # TODO: Implement Rest_Frame_Property_Limit_Selector
                    # in case of np.nan x_bin entry
                    x_bin_selector = Rest_Frame_Property_Bin_Selector(
                        aper_diam, SED_fit_code, x_calculator, x_bin
                    )
                    try:
                        z_bin_x_bin_cat = deepcopy(z_bin_cat).crop(
                            x_bin_selector
                        )
                    except Exception as e:
                        raise GalfindError(
                            f"Failed to crop {z_bin_cat.crop_name!r} to "
                            f"x_bin={x_bin!r} using "
                            f"{x_bin_selector!r}: {e}"
                        ) from e
                    Ngals[i] = len(z_bin_x_bin_cat)

                    # plot cutouts
                    if plot and Ngals[i] > 0:
                        # fig_axs_ = figs.make_phot_diagnostic_fig(
                        #     len(z_bin_x_bin_cat.filterset)
                        # )
                        z_bin_x_bin_cat.plot_phot_diagnostics(
                            aper_diam,
                            SED_arr=SED_fit_code,
                            zPDF_arr=SED_fit_code,
                            # fig_axs = fig_axs_,
                        )
                        # plt.close(fig_axs_[0].figure)
                        # plot histogram
                        hist_fig, hist_ax = plt.subplots()
                        z_bin_x_bin_cat.hist(
                            x_calculator,
                            hist_fig,
                            hist_ax,
                            from_pdf=True,
                            save=False,
                            overwrite=True,
                            density=True,
                        )
                        z_bin_x_bin_cat.hist(
                            x_calculator,
                            hist_fig,
                            hist_ax,
                            from_pdf=False,
                            save=True,
                            overwrite=True,
                            density=True,
                        )
                        plt.close(hist_fig)

                # if there are galaxies in the z,x bin
                if int(Ngals[i]) != 0:
                    # plot histogram
                    # z_bin_x_bin_cat.hist(x_calculator, hist_fig, hist_ax)
                    dx = x_bin[1].value - x_bin[0].value
                    # extract Vmax's
                    Vmax_arr = [
                        getattr(
                            gal.aper_phot[aper_diam].SED_results[
                                SED_fit_code.label
                            ],
                            "Vmax",
                        )
                        for gal in z_bin_x_bin_cat
                    ]
                    # V_max[z_bin_cat.crop_name.split("/")[-1]]
                    # [full_survey_name].value
                    # V_max = np.array(
                    #     [
                    #         gal.aper_phot[aper_diam]. \
                    #         SED_results[SED_fit_code.label]. \

                    #         for gal in z_bin_x_bin_cat
                    #     ]
                    # )
                    if isinstance(cat, Combined_Catalogue):
                        cat_arr = cat.cat_arr
                    else:
                        cat_arr = [cat]
                    Vmax_reg_compl = []
                    for cat_ in cat_arr:
                        if completeness is not None:
                            if cat_.data.survey not in completeness.keys():
                                raise MissingKeyError(
                                    f"cat_.data.survey={cat_.data.survey!r} "
                                    "not in completeness.keys()="
                                    f"{list(completeness.keys())!r}."
                                )
                        regions = np.unique(
                            [
                                reg
                                for Vmax in Vmax_arr
                                for reg in Vmax[cat_.survey].keys()
                            ]
                        )
                        # detectable_gals = np.full(len(regions), Ngals[i])
                        for region in regions:
                            # TODO: FINISH THIS FOR MULTI-REGION SURVEYS
                            Vmax_ = np.array(
                                [
                                    Vmax[cat_.survey][region]
                                    for Vmax in Vmax_arr
                                ]
                            )
                            keep_indices = Vmax_ != -1.0
                            # Vmax = Vmax_[keep_indices] # * u.Mpc ** 3
                            if len(Vmax_[keep_indices]) != Ngals[i]:
                                galfind_logger.warning(
                                    f"{Ngals[i] - len(Vmax_[keep_indices])} "
                                    f"galaxies not detected in {region=}"
                                )
                                # detectable_gals[j] = len(Vmax_[keep_indices])
                            if completeness is None:
                                compl_bin = np.ones(len(z_bin_x_bin_cat))
                            else:
                                if (
                                    region
                                    not in completeness[cat_.survey].keys()
                                ):
                                    raise MissingKeyError(
                                        f"region={region!r} not in "
                                        "completeness[cat_.survey].keys()="
                                        f"{list(completeness[cat_.survey].keys())!r}."
                                    )
                                redshifts = np.zeros(len(z_bin_x_bin_cat))
                                xvals = np.zeros(len(z_bin_x_bin_cat))
                                for k, gal in enumerate(z_bin_x_bin_cat):
                                    redshifts[k] = (
                                        gal.aper_phot[aper_diam]
                                        .SED_results[SED_fit_code.label]
                                        .z.value
                                    )
                                    xvals[k] = (
                                        gal.aper_phot[aper_diam]
                                        .SED_results[SED_fit_code.label]
                                        .phot_rest.properties[
                                            x_calculator.name
                                        ]
                                        .value
                                    )
                                compl_bin = completeness[cat_.survey][region](
                                    redshifts, xvals
                                )
                            if len(compl_bin) != len(Vmax_):
                                raise LengthMismatchError(
                                    f"len(compl_bin)={len(compl_bin)} != "
                                    f"len(Vmax_)={len(Vmax_)} for "
                                    f"{z_bin_x_bin_cat.crop_name}."
                                )
                            # import matplotlib.pyplot as plt
                            # from scipy.interpolate import interp1d
                            # fig, ax = plt.subplots()
                            # ax.scatter(
                            #     completeness.compl_arr[0].x_calculator(
                            #         z_bin_x_bin_cat
                            #     )[keep_indices],
                            #     compl_bin, label = str(x_bin)
                            # )
                            # ax.plot(
                            #     completeness.compl_arr[0].x,
                            #     completeness.compl_arr[0].completeness,
                            #     label = "Completeness"
                            # )
                            # ax.plot(
                            #     completeness.compl_arr[0].x,
                            #     interp1d(
                            #         completeness.compl_arr[0].x,
                            #         completeness.compl_arr[0].completeness
                            #     )(completeness.compl_arr[0].x),
                            #     label = "Interpolated Completeness"
                            # )
                            # ax.legend()
                            # plt.savefig("test_compl_NEP.png")
                            np.mean(compl_bin)
                            Vmax_reg_compl.append(Vmax_ * compl_bin)

                    Vmax_reg_compl = np.array(Vmax_reg_compl)
                    Vmax_tot = np.clip(Vmax_reg_compl, 0.0, None).sum(axis=0)
                    # np.sum(Vmax_reg_compl, axis = 1)
                    Vmax_tot = Vmax_tot[Vmax_tot > 0.0]
                    phi[i] = np.sum(Vmax_tot**-1.0) / dx
                    # use standard Poisson errors if number of
                    # galaxies in bin is not small
                    detected_gals = len(Vmax_tot)
                    if detected_gals >= 4:
                        phi_errs = np.sqrt(np.sum(Vmax_tot**-2.0)) / dx
                        phi_l1[i] = phi_errs
                        phi_u1[i] = phi_errs
                    else:
                        poisson_int = funcs.poisson_interval(
                            detected_gals, 0.32
                        )
                        phi_l1[i] = phi[i] * np.min(
                            np.abs((np.array(poisson_int[0]) - detected_gals))
                            / detected_gals
                        )
                        phi_u1[i] = phi[i] * np.min(
                            np.abs((np.array(poisson_int[1]) - detected_gals))
                            / detected_gals
                        )

                    from ..catalogues.Catalogue import Catalogue
                    from ..catalogues.Catalogue_Base import Catalogue_Base
                    from ..catalogues.Multiple_Catalogue import (
                        Combined_Catalogue,
                    )

                    if isinstance(cat, Combined_Catalogue):
                        data_arr = [cat_.data for cat_ in cat.cat_arr]
                    elif isinstance(cat, Catalogue):
                        data_arr = [cat.data]
                    else:
                        valid_names = ", ".join(
                            c.__name__ for c in Catalogue_Base.__subclasses__()
                        )
                        raise GalfindTypeError(
                            f"cat={cat!r} has type {type(cat)}; must be an "
                            f"instance of one of {valid_names}."
                        )
                    if cv_origin is None:
                        pass
                    elif cv_origin == "Driver2010":
                        cv_errs[i] = funcs.calc_cv_proper(
                            z_bin,
                            data_arr=data_arr,
                            masked_selector=unmasked_area,
                            z=np.sum(z_bin) / 2.0,
                        )
                    else:
                        raise InvalidOptionError(
                            f"cv_origin={cv_origin!r} not recognised; must "
                            "be None or 'Driver2010'."
                        )
            number_density_func = cls(
                x_calculator.name,
                x_bins,
                x_origin,
                z_bin,
                Ngals,
                phi,
                np.array([phi_l1, phi_u1]),
                cv_errs,
                cat.survey,
                z_bin_cat.crop_name,
                cv_origin,
                completeness=completeness,
                Vmax_method=Vmax_method,
            )
            if save:
                number_density_func.save()
            return number_density_func

        else:  # load results
            return cls.from_ecsv(save_path)

    # @staticmethod
    # def get_origin_surveys(data_arr) -> str:
    #     return "+".join([data.full_name for data in data_arr])

    # cv_origin == "Driver2010"
    @staticmethod
    def get_save_path(
        origin_surveys: str,
        SED_fit_params_key: str,
        x_name: str,
        crop_name: str,
        ext: str = ".ecsv",
        completeness: Optional[Completeness] = None,
        Vmax_method: str = "uniform_depth",
    ) -> str:
        """Get file path for saving number density function data.

        Parameters
        ----------
        origin_surveys : `str`
            Survey name(s) that contributed to the data.
        SED_fit_params_key : `str`
            Key for SED fitting parameters.
        x_name : `str`
            Name of the property (x-axis).
        crop_name : `str`
            Redshift bin or region identifier.
        ext : `str`, optional
            File extension. Default is ".ecsv".
        completeness : `Completeness`, optional
            Completeness corrections applied. Default is None.
        Vmax_method : `str`, optional
            Vmax calculation method. Default is "uniform_depth".

        Returns
        -------
        `str`
            Full path for the data file.
        """
        if completeness is None:
            compl_name = ""
        else:
            compl_name = "_compl_corr"
        if Vmax_method is None:
            Vmax_method_str = ""
        else:
            Vmax_method_str = f"/{Vmax_method}"
        save_path = (
            config["NumberDensityFunctions"]["NUMBER_DENSITY_FUNC_DIR"]
            + f"/Data/{SED_fit_params_key}/{x_name}/{origin_surveys}"
            + f"{Vmax_method_str}/{crop_name}{compl_name}{ext}"
        )
        funcs.make_dirs(save_path)
        return save_path

    @staticmethod
    def extract_info_from_save_path(
        save_path,
    ) -> Tuple[str, str, str, NDArray[float]]:
        """Extract NDF metadata from file path.

        Parameters
        ----------
        save_path : `str`
            Path to NDF data file.

        Returns
        -------
        `tuple`
            (SED_fit_params_key, x_name, origin_surveys, z_bin)
        """
        split_save_path = save_path.split("/")
        SED_fit_params_key = split_save_path[-4]
        x_name = split_save_path[-3]
        origin_surveys = split_save_path[-2]
        z_bin = np.array(
            [
                float(split_save_path[-1].split("<")[0]),
                float(split_save_path[-1].split("<")[2]),
            ]
        )
        return SED_fit_params_key, x_name, origin_surveys, z_bin

    def crop_to_xbin(self: Type[Self], x_bin: List[float]) -> Optional[Self]:
        """Create a new NDF cropped to a specific x-axis range.

        Parameters
        ----------
        x_bin : `list` of `float`
            [x_min, x_max] range to crop to.

        Returns
        -------
        `Self` or `None`
            Cropped NDF object, or None if no data in range.

        Raises
        ------
        RangeError
            If both `x_bin` bounds are finite and ``x_bin[0]`` is not
            less than ``x_bin[1]``.
        """
        # check if x_bin is within self.x_bins
        if x_bin[0] < self.x_bins[0][0] and x_bin[1] > self.x_bins[-1][1]:
            galfind_logger.warning(
                f"{x_bin=} not within {self.x_bins[0][0]} "
                f"and {self.x_bins[-1][1]}!"
            )
            return None
        # find indices of x bins in self that are entirely within
        # the output crop x_bin
        if np.isfinite(x_bin[0]):
            lower_mask = self.x_mid_bins >= x_bin[0]
        else:
            lower_mask = np.full(len(self.x_mid_bins), True)
        if np.isfinite(x_bin[1]):
            upper_mask = self.x_mid_bins <= x_bin[1]
        else:
            upper_mask = np.full(len(self.x_mid_bins), True)
        if np.isfinite(x_bin[0]) and np.isfinite(x_bin[1]):
            if not x_bin[0] < x_bin[1]:
                raise RangeError(
                    f"x_bin[0]={x_bin[0]} must be less than x_bin[1]="
                    f"{x_bin[1]}."
                )
            xbin_str = f"{x_bin[0].value:.1f}<=x<={x_bin[1].value:.1f}"
        elif np.isfinite(x_bin[0]) and not np.isfinite(x_bin[1]):
            xbin_str = f"x>={x_bin[0].value:.1f}"
        elif not np.isfinite(x_bin[0]) and np.isfinite(x_bin[1]):
            xbin_str = f"x<={x_bin[1].value:.1f}"
        indices = np.where(lower_mask & upper_mask)[0]
        if len(indices) == 0:
            galfind_logger.warning(
                f"No x bins within {x_bin=} in {self.x_bins}!"
            )
            return None
        else:
            new_crop_name = f"{self.crop_name}_{xbin_str}"
            new_ndf = self.__class__(
                self.x_name,
                self.x_bins[indices],
                self.x_origin,
                self.z_bin,
                self.Ngals[indices],
                self.phi[indices],
                self.phi_errs[:, indices],
                self.cv_errs[indices],
                self.origin_surveys,
                new_crop_name,
                cv_origin=self.cv_origin,
                completeness=self.completeness,
                Vmax_method=self.Vmax_method,
            )
            return new_ndf

    # def get_z_bin_name(self) -> str:
    #     return f"{self.z_bin[0]:.1f}<z<{self.z_bin[1]:.1f}"

    def get_plot_path(self) -> str:
        """Get file path for saving NDF plots.

        Returns
        -------
        `str`
            Path for the plot file.
        """
        plot_path = self.get_save_path(
            self.origin_surveys,
            self.x_origin,
            self.x_name,
            self.crop_name,
            ext=".png",
            completeness=self.completeness,
            Vmax_method=self.Vmax_method,
        ).replace("/Data/", "/Plots/")

        if os.access(plot_path, os.W_OK):
            funcs.make_dirs(plot_path)
        else:
            galfind_logger.warning(f"Cannot write to {plot_path}!")
        return plot_path

    def fit(
        self: Self,
        fit_type: Type[MCMC_Fitter],
        priors: Priors,
        fixed_params: Dict[str, float],
        n_walkers: int,
        n_steps: int,
        n_processes: int = 1,
        backend_filename: Optional[str] = None,
        incl_cv_errs: bool = True,
    ) -> NoReturn:
        """Fit a model to the number density function using MCMC.

        Parameters
        ----------
        fit_type : `Type[MCMC_Fitter]`
            MCMC fitter class to use.
        priors : `Priors`
            Prior distributions for model parameters.
        fixed_params : `dict`
            Parameters to fix during fitting.
        n_walkers : `int`
            Number of MCMC walkers.
        n_steps : `int`
            Number of MCMC steps.
        n_processes : `int`, optional
            Number of parallel processes. Default is 1.
        backend_filename : `str`, optional
            Path for MCMC state file. Auto-generated if None.
        incl_cv_errs : `bool`, optional
            Whether to include cosmic variance errors. Default is True.
        """
        if backend_filename is None:
            backend_filename = self.get_save_path(
                self.origin_surveys,
                self.x_origin,
                self.x_name,
                self.crop_name,
                completeness=self.completeness,
                Vmax_method=self.Vmax_method,
            )
            fixed_params_str = "_".join(
                [f"{key}={val:.3f}" for key, val in fixed_params.items()]
            )
            if fixed_params_str != "":
                fixed_params_str = f"_{fixed_params_str}"
            if not incl_cv_errs:
                incl_cv_str = "_no_cv"
                phi_errs_ptr = self.phi_errs
            else:
                incl_cv_str = ""
                phi_errs_ptr = self.phi_errs_cv
            backend_filename = backend_filename.replace(
                "/Data/", f"/{fit_type.__name__.replace('Fitter', 'Fits')}/"
            ).replace(".ecsv", f"{fixed_params_str}{incl_cv_str}.h5")
            funcs.make_dirs(backend_filename)
        # remove 0s from x_mid_bins, phi, and phi_errs
        zero_indices = np.where(self.phi == 0.0)[0]
        x_mid_bins = np.delete(self.x_mid_bins.value, zero_indices)
        phi = np.delete(self.phi, zero_indices)
        phi_errs = np.array(
            [
                np.delete(phi_errs_ptr[0], zero_indices),
                np.delete(phi_errs_ptr[1], zero_indices),
            ]
        )
        self.fitter = fit_type(
            priors,
            x_mid_bins,
            phi,
            phi_errs,
            n_walkers,
            backend_filename,
            fixed_params,
        )
        # run fitter
        self.fitter(n_steps, n_processes)

    def save(self, save_path: Optional[str] = None) -> NoReturn:
        """Save the number density function to a file.

        Parameters
        ----------
        save_path : `str`, optional
            Path for saving the NDF data. Auto-generated if None.

        Raises
        ------
        InvalidUnitError
            If the lower or upper `x_bins` edges do not all share the
            same unit, or if the lower and upper edge units differ from
            each other.
        """
        if save_path is None:
            save_path = self.get_save_path(
                self.origin_surveys,
                self.x_origin,
                self.x_name,
                self.crop_name,
                completeness=self.completeness,
                Vmax_method=self.Vmax_method,
            )
        if not all(
            x_bin[0].unit == self.x_bins[0][0].unit for x_bin in self.x_bins
        ):
            raise InvalidUnitError(
                "Not all lower x_bins edges share the same unit; "
                f"expected {self.x_bins[0][0].unit!r} for every element "
                f"of {self.x_bins!r}."
            )
        if not all(
            x_bin[1].unit == self.x_bins[0][1].unit for x_bin in self.x_bins
        ):
            raise InvalidUnitError(
                "Not all upper x_bins edges share the same unit; "
                f"expected {self.x_bins[0][1].unit!r} for every element "
                f"of {self.x_bins!r}."
            )
        if self.x_bins[0][0].unit != self.x_bins[0][1].unit:
            raise InvalidUnitError(
                f"self.x_bins[0][0].unit={self.x_bins[0][0].unit!r} != "
                f"self.x_bins[0][1].unit={self.x_bins[0][1].unit!r}; "
                "lower and upper x_bin edges must share the same unit."
            )
        x_bins_low = (
            np.array([x_bin[0].value for x_bin in self.x_bins])
            * self.x_bins[0][0].unit
        )
        x_bins_up = (
            np.array([x_bin[1].value for x_bin in self.x_bins])
            * self.x_bins[0][1].unit
        )
        tab = Table(
            {
                "x_bins_low": x_bins_low,
                "x_bins_up": x_bins_up,
                "Ngals": self.Ngals,
                "phi": self.phi,
                "phi_l1": self.phi_errs[0],
                "phi_u1": self.phi_errs[1],
                "cv_errs": self.cv_errs,
            },
            dtype=[float, float, int, float, float, float, float],
        )
        tab.meta = {
            "x_origin": self.x_origin,
            "x_name": self.x_name,
            "origin_surveys": self.origin_surveys,
            "z_bin": self.z_bin,
            "cv_origin": self.cv_origin,
            "crop_name": self.crop_name,
            "Vmax_method": self.Vmax_method,
        }
        funcs.make_dirs(save_path)
        tab.write(save_path, overwrite=True)
        galfind_logger.info(
            f"Saved {self.x_name} {self.z_bin} "
            + f"{self.origin_surveys} to {save_path}"
        )

    def plot(
        self: Type[Self],
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        log_x: bool = False,
        log_y: bool = False,
        annotate: bool = True,
        save: bool = True,
        show: bool = False,
        plot_kwargs: Dict[str, Any] = {},
        legend_kwargs: Dict[str, Any] = {},
        x_lims: Optional[Union[List[float], str]] = "default",
        y_lims: Optional[List[float]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        plot_cv_errs: bool = True,
        offset: float = 0.0,
        obs_author_years: Dict[str, Dict[str, Any]] = {},
        sim_author_years: Dict[str, Dict[str, Any]] = {},
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the number density function.

        Parameters
        ----------
        fig : `matplotlib.figure.Figure`, optional
            Figure to plot on. Created if None. Default is None.
        ax : `matplotlib.axes.Axes`, optional
            Axes to plot on. Created if None. Default is None.
        log_x : `bool`, optional
            Whether to use log scale on x-axis. Default is False.
        log_y : `bool`, optional
            Whether to use log scale on y-axis. Default is False.
        annotate : `bool`, optional
            Whether to add axis labels. Default is True.
        save : `bool`, optional
            Whether to save the figure. Default is True.
        show : `bool`, optional
            Whether to display the figure. Default is False.
        plot_kwargs : `dict`, optional
            Kwargs for plot styling. Default is empty dict.
        legend_kwargs : `dict`, optional
            Kwargs for legend. Default is empty dict.
        x_lims : `list` or `str`, optional
            X-axis limits or "default". Default is "default".
        y_lims : `list`, optional
            Y-axis limits. Default is None.
        title : `str`, optional
            Plot title. Default is None.
        save_path : `str`, optional
            Path to save figure. Default is None.
        plot_cv_errs : `bool`, optional
            Whether to plot cosmic variance errors. Default is True.
        offset : `float`, optional
            X-axis offset. Default is 0.0.
        obs_author_years : `dict`, optional
            Mapping of observational literature author-years to their
            plot kwargs, overplotted via `Base_Number_Density_Function
            .from_flags_repo`. Default is empty dict.
        sim_author_years : `dict`, optional
            Mapping of simulation literature author-years to their plot
            kwargs, overplotted via `Base_Number_Density_Function
            .from_flags_repo`. Default is empty dict.

        Returns
        -------
        `tuple`
            (fig, ax) matplotlib objects.
        """
        if all(_x is None for _x in [fig, ax]):
            fig_, ax_ = plt.subplots()
        else:
            fig_, ax_ = fig, ax

        if title is None:
            title = self.crop_name

        lit_lines = []
        for author_year, author_year_kwargs in obs_author_years.items():
            author_year_func_from_flags_data = (
                Base_Number_Density_Function.from_flags_repo(
                    self.x_name,
                    self.z_bin,
                    author_year,
                    "obs",
                )
            )
            if author_year_func_from_flags_data is not None:
                lit_lines.append(
                    author_year_func_from_flags_data.plot(
                        fig_,
                        ax_,
                        log_x,
                        log_y,
                        annotate=False,
                        save=False,
                        show=False,
                        plot_kwargs=author_year_kwargs,
                        x_lims=None,
                    )[-1]
                )
        for author_year, author_year_kwargs in sim_author_years.items():
            author_year_func_from_flags_data = (
                Base_Number_Density_Function.from_flags_repo(
                    self.x_name,
                    self.z_bin,
                    author_year,
                    "models/binned",
                )
            )
            if author_year_func_from_flags_data is not None:
                lit_lines.append(
                    author_year_func_from_flags_data.plot(
                        fig_,
                        ax_,
                        log_x,
                        log_y,
                        annotate=False,
                        save=False,
                        show=False,
                        plot_kwargs=author_year_kwargs,
                        x_lims=None,
                    )[-1]
                )
        fig_, ax_, line = super().plot(
            fig_,
            ax_,
            log_x,
            log_y,
            annotate,
            save,
            show,
            plot_kwargs,
            legend_kwargs,
            x_lims,
            y_lims,
            title,
            save_path,
            plot_cv_errs=plot_cv_errs,
            offset=offset,
        )
        return fig_, ax_, line, lit_lines


#         def mass_function(catalog, fields, z_bins, mass_bins,
#  rerun=False, out_directory = '/nvme/scratch/work/tharvey/masses/',
#  mass_keyword='MASS_BEST',mass_form='log', z_keyword='Z_BEST',
#  sed_tool='LePhare', template='', z_step=0.01,
#   n_jobs=2, cat_version='v7', do_muv=False, use_vmax_simple = False,
#   field_keyword='field',
# other_name = '',
other_sed_path = ("/nvme/scratch/work/austind/Bagpipes/pipes/seds/",)
#   use_base=True, base_cat='/nvme/scratch/work/tharvey/catalogs/
#   robust_and_good_gal_all_criteria_3sigma_all_fields_masses.fits',
#   id_keyword='NUMBER',  use_new_zloop=True, select_444=False,
#   use_bootstrap=True, rerun_other_pdfs = True,
#     other_appended=False, flag_ids=[], base_cat_filter=None,
#     zgauss=False):

# calculate optimal redshift bin size?
# try:
#     zs = np.array([i[0] for i in catalog[z_keyword]])
#     for z_bin in z_bins:

#         mask = (zs > z_bin[0]) & (zs < z_bin[1])
#         mass_test_z = mass_test[mask]
#         iqr = np.subtract(*np.percentile(mass_test_z, [75, 25]))

#         print(f'Optimal bin size: {2*iqr/np.cbrt(len(mass_test_z)):.2f}')
# except:
#     pass

# if use_bootstrap:
#     bootstrap_bins(catalog, fields, z_bins,mass_bins, len_array,
#     rerun_other_pdfs, out_directory, z_keyword=z_keyword,
#     mass_keyword=mass_keyword, mass_form=mass_form,
#     field_keyword=field_keyword, other_name=other_name,
#     other_sed_path=other_sed_path, load_duncans=load_duncans,
#     id_keyword=id_keyword, other_h5_path=other_h5_path, muv=muv,
#     vmax_keyword=vmax_keyword, name_444=name_444, zgauss=zgauss,
#     sed_tool=sed_tool)

#     plot_bins_pdfs(catalog, fields, z_bins, mass_bins, rerun=False,
#     out_directory = out_directory,
#     mass_form=mass_form,mass_keyword=mass_keyword,
#     z_keyword=z_keyword, zgauss=zgauss)
# else:


# Specific class written to allow for fitting of galaxy bias
class Multiple_Number_Density_Function(Number_Density_Function):
    """Combine multiple number density functions for the same property.

    Aggregates several independently-computed number density functions
    (e.g., from different surveys or data releases) into one dataset,
    handling differences in sample sizes and errors.

    Parameters
    ----------
    number_density_functions : `list` of `Base_Number_Density_Function`
        List of number density functions to combine. All must have the
        same binning, redshift, and other key properties.

    Attributes
    ----------
    number_density_functions : `list`
        The input list of density functions.
    """

    def __init__(
        self: Self,
        number_density_functions: List[Type[Base_Number_Density_Function]],
    ):
        self.number_density_functions = number_density_functions
        self._assertions(number_density_functions)
        super().__init__(
            x_name=getattr(number_density_functions[0], "x_name"),
            x_bins=getattr(number_density_functions[0], "x_bins"),
            x_origin=getattr(number_density_functions[0], "x_origin"),
            z_bin=getattr(number_density_functions[0], "z_bin"),
            Ngals=np.array([ndf.Ngals for ndf in number_density_functions]).T,
            phi=np.array([ndf.phi for ndf in number_density_functions]).T,
            phi_errs=np.array(
                [ndf.phi_errs for ndf in number_density_functions]
            ).transpose((1, 2, 0)),
            cv_errs=np.array(
                [ndf.cv_errs for ndf in number_density_functions]
            ).T,
            origin_surveys="+".join(
                [ndf.origin_surveys for ndf in number_density_functions]
            ),
            crop_name=getattr(number_density_functions[0], "crop_name"),
            cv_origin=getattr(number_density_functions[0], "cv_origin"),
            # completeness = getattr(
            #     number_density_functions[0], "completeness"
            # ),
        )

    @staticmethod
    def _assertions(
        number_density_functions: List[Type[Number_Density_Function]],
    ):
        same_attr_labels = [
            "x_name",
            "x_bins",
            "x_origin",
            "z_bin",
            "crop_name",
            "cv_origin",
            # "completeness",
        ]
        if not all(
            np.array_equal(
                getattr(ndf, attr_label),
                getattr(number_density_functions[0], attr_label),
            )
            for ndf in number_density_functions
            for attr_label in same_attr_labels
        ):
            raise GalfindError(
                f"Not all of {', '.join(same_attr_labels)} are the same "
                "for all number_density_functions="
                f"{number_density_functions!r}; every element must share "
                "these attributes to be combined."
            )
        if not all(
            getattr(ndf, "completeness") is None
            for ndf in number_density_functions
        ):
            raise GalfindError(
                "Not all completeness are None for all "
                f"number_density_functions={number_density_functions!r}; "
                "combining number density functions with completeness "
                "corrections applied is not currently supported."
            )
        # ensure all number density functions have different Ngals arrays
        if not all(
            np.unique(
                [getattr(ndf, "phi") for ndf in number_density_functions],
                axis=0,
                return_counts=True,
            )[1]
            == 1
        ):
            raise GalfindError(
                f"number_density_functions={number_density_functions!r} "
                "contains duplicate 'phi' arrays; every element must "
                "have a distinct phi array."
            )

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter > len(self) - 1:
            raise StopIteration
        else:
            ndf = self[self.iter]
            self.iter += 1
            return ndf

    def __getitem__(
        self, index: Any
    ) -> Optional[
        Union[
            Type[Number_Density_Function], List[Type[Number_Density_Function]]
        ]
    ]:
        if len(self) == 0:
            raise IndexError("No number density functions in object!")
        if isinstance(index, int):
            return self.number_density_functions[index]
        else:
            raise IndexError(f"{repr(index)} not an int!")

    def __add__(
        self: Self,
        other: Union[
            Type[Base_Number_Density_Function],
            List[Type[Base_Number_Density_Function]],
            Multiple_Number_Density_Function,
        ],
    ) -> Multiple_Number_Density_Function:
        base_ndf_subcls = tuple(Base_Number_Density_Function.__subclasses__())
        if isinstance(other, list):
            if not all(isinstance(ndf, base_ndf_subcls) for ndf in other):
                raise GalfindTypeError(
                    f"other={other!r} contains an element that is not an "
                    f"instance of {base_ndf_subcls}."
                )
            new_ndfs = other
        elif isinstance(other, base_ndf_subcls):
            new_ndfs = [other]
        elif isinstance(other, Multiple_Number_Density_Function):
            if not all(
                isinstance(ndf, base_ndf_subcls)
                for ndf in other.number_density_functions
            ):
                raise GalfindTypeError(
                    f"other.number_density_functions="
                    f"{other.number_density_functions!r} contains an "
                    f"element that is not an instance of {base_ndf_subcls}."
                )
            new_ndfs = other.number_density_functions
        else:
            raise GalfindTypeError(
                f"other={other!r} has type {type(other)}; must be one of "
                f"{base_ndf_subcls}, List[{base_ndf_subcls}], or "
                "Multiple_Number_Density_Function."
            )
        self._assertions(new_ndfs)
        self.number_density_functions += new_ndfs
        for hstack_label in ["Ngals", "phi", "cv_errs"]:
            setattr(
                self,
                hstack_label,
                np.hstack(
                    [
                        getattr(self, hstack_label),
                        np.array(
                            [getattr(ndf, hstack_label) for ndf in new_ndfs]
                        ).T,
                    ]
                ),
            )
        phi_errs_l1 = np.hstack(
            [
                getattr(self, "phi_errs")[0],
                np.array([getattr(ndf, "phi_errs")[0] for ndf in new_ndfs]).T,
            ]
        )
        phi_errs_u1 = np.hstack(
            [
                getattr(self, "phi_errs")[1],
                np.array([getattr(ndf, "phi_errs")[1] for ndf in new_ndfs]).T,
            ]
        )
        self.phi_errs = np.array([phi_errs_l1, phi_errs_u1])
        self.origin_surveys = "+".join(
            [ndf.origin_surveys for ndf in self.number_density_functions]
        )
        return self

    # @classmethod
    # def from_cat(
    #     cls,
    #     cat: Catalogue,
    #     x_name: str,
    #     x_bin_edges_arr: Union[list, np.array],
    #     z_bins: Union[list, np.array],
    #     x_origin: Union[str, dict] = "EAZY_fsps_larson_zfree",
    #     z_step: float = 0.1,
    #     use_vmax_simple: bool = False,
    #     unmasked_area: Union[str, List[str], u.Quantity] = "selection",
    #     timed: bool = False,
    # ) -> "Number_Density_Function":
    #     # input assertions
    #     assert all(len(z_bin) == 2 for z_bin in z_bins)
    #     assert all(z_bin[0] < z_bin[1] for z_bin in z_bins)
    #     assert len(x_bin_edges_arr) == len(z_bins)
    #     assert all(len(x_bin_edges) >= 2 for x_bin_edges in x_bin_edges_arr)
    # # ensure x_bin_edges are sorted from lower to higher x
    # # values in every z bin
    #     assert all(
    #         np.sort(np.array(x_bin_edges)) == np.array(x_bin_edges)
    #         for x_bin_edges in x_bin_edges_arr
    #     )
    #     # ensure x_bin_edges are evenly spaced?
    #     # extract x_name values from catalogue
    #     if isinstance(x_origin, dict):
    #         assert "code" in x_origin.keys()
    #         assert x_origin["code"].__class__.__name__ in [
    #             code.__name__ for code in SED_code.__subclasses__()
    #         ]
    #         SED_fit_params = x_origin  # redshifts must come from
    #         same SED fitting as x values
    #     elif isinstance(x_origin, str):
    #         # convert to SED_fit_params
    #         SED_fit_params = x_origin.split("_")[0]
    #     else:
    #         galfind_logger.critical(
    #             f"{x_origin=} with {type(x_origin)=} not in [dict, str]!"
    #         )

    #     x = getattr(cat, x_name, x_origin)

    #     # calculate mass function in each redshift bin
    #     for i, (z_bin, x_bin_edges) in enumerate(
    #         zip(z_bins, x_bin_edges_arr)
    #     ):
    #         # create x_bins from x_bin_edges (must include start
    #         # and end values here too)
    #         x_bins = [
    #             [x_bin_edges[i], x_bin_edges[i + 1]]
    #             for i in range(len(x_bin_edges) - 1)
    #             if i != len(x_bin_edges) - 1
    #         ]
    #         # extract z_bin_name
    #         assert isinstance(x_origin, str)
    #         z_bin_name = f"{x_origin}_{z_bin[0]:.1f}<z<{z_bin[1]:.1f}"
    #         # calculate Vmax for each galaxy in catalogue within z bin
    #         # in general call Vmax_multifield
    #         cat.calc_Vmax(
    #             cat.data, z_bin, x_origin, z_step,
    #             unmasked_area = unmasked_area, timed=timed
    #         )
    #         # crop catalogue to this redshift bin
    #         z_bin_cat = cat.crop(z_bin, "z", x_origin)

    #         Ngals = np.zeros(len(x_bins))
    #         phi = np.zeros(len(x_bins))
    #         phi_errs = np.zeros(len(x_bins))
    #         cv_errs = np.zeros(len(x_bins))
    #         phi_errs_cv = np.zeros(len(x_bins))
    #         # loop through each mass bin in the given redshift bin
    #         for j, x_bin in enumerate(x_bins):
    #             # crop to galaxies in the x bin - not the bootstrapping
    #             method
    #             z_bin_x_bin_cat = z_bin_cat.crop(
    #                 x_bin, x_name, SED_fit_params)

    #             Ngals[j] = len(z_bin_x_bin_cat)
    #             # if there are galaxies in the z, mx bin
    #             if Ngals[j] != 0:
    #                 dx = x_bin[1] - x_bin[0]
    #                 V_max = np.array(
    #                     [
    #                         gal.V_max[z_bin_name][cat.data.full_name]
    #                         for gal in cat
    #                     ]
    #                 )
    #                 phi[j] = (np.sum(V_max**-1.0) / dx).value
    #                 # use standard Poisson errors if number of
    #                 # galaxies in bin is not small
    #                 if len(V_max) >= 4:
    #                     phi_errs[j] = (
    #                         np.sqrt(np.sum(V_max**-2.0)) / dx
    #                     ).value
    #                 else:
    #                     # using minimum is a minor cheat for symmetric errors
    #                     phi_errs[j] = phi[j] * np.min(
    #                         np.abs(
    #                             (
    #                                 np.array(
    #                                     funcs.poisson_interval(
    #                                         len(V_max), 0.32
    #                                     )
    #                                 )
    #                                 - len(V_max)
    #                             )
    #                         )
    #                         / len(V_max)
    #                     )
    #                 cv_errs[j] = funcs.calc_cv_proper(
    #                     float(z_bin[0]),
    #                     float(z_bin[1]),
    #                     fields_used=fields_used,
    #                     **kwargs,
    #                 )
    #                 phi_errs_cv[j] = np.sqrt(
    #                     phi_errs[j] ** 2.0 + (cv_errs[j] * phi[j]) ** 2.0
    #                 )

    def __len__(self):
        return len(self.number_density_functions)

    def fit(
        self: Self,
        fit_type: Type[MCMC_Fitter],
        priors: Priors,
        fixed_params: Dict[str, float],
        n_walkers: int,
        n_steps: int,
        n_processes: int = 1,
        backend_filename: Optional[str] = None,
        incl_cv_errs: bool = True,
    ) -> NoReturn:
        # instantiate fitter
        if backend_filename is None:
            backend_filename = self.get_save_path(
                self.origin_surveys,
                self.x_origin,
                self.x_name,
                self.crop_name,
                completeness=self.completeness,
                Vmax_method=self.Vmax_method,
            )
            fixed_params_str = "_".join(
                [f"{key}={val:.3f}" for key, val in fixed_params.items()]
            )
            if fixed_params_str != "":
                fixed_params_str = f"_{fixed_params_str}"
            if not incl_cv_errs:
                incl_cv_str = "_no_cv"
                phi_errs_ptr = self.phi_errs
            else:
                incl_cv_str = ""
                phi_errs_ptr = self.phi_errs_cv
            backend_filename = backend_filename.replace(
                "/Data/", f"/{fit_type.__name__.replace('Fitter', 'Fits')}/"
            ).replace(".ecsv", f"{fixed_params_str}{incl_cv_str}.h5")
            funcs.make_dirs(backend_filename)
        # mask nans and zeros
        phi_mask = np.isnan(self.phi) | (self.phi == 0)
        x_mid_bins = np.ma.array(
            np.tile(self.x_mid_bins.value, (len(self), 1)).T, mask=phi_mask
        ).compressed()
        phi = np.ma.array(self.phi, mask=phi_mask).compressed()
        phi_errs = np.array(
            [
                np.ma.array(phi_errs_ptr[0], mask=phi_mask).compressed(),
                np.ma.array(phi_errs_ptr[1], mask=phi_mask).compressed(),
            ]
        )
        surveys_arr = np.ma.array(
            np.tile(
                [ndf.origin_surveys for ndf in self.number_density_functions],
                (len(self[0]), 1),
            ),
            mask=phi_mask,
        ).compressed()
        self.fitter = fit_type(
            surveys_arr=surveys_arr,
            priors=priors,
            x_data=x_mid_bins,
            y_data=phi,
            y_data_errs=phi_errs,
            nwalkers=n_walkers,
            backend_filename=backend_filename,
            fixed_params=fixed_params,
        )
        # run fitter
        self.fitter(n_steps, n_processes)

    def plot(self):
        pass

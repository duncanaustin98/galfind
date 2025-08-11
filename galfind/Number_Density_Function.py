from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from astropy.table import Table
from typing import NoReturn, Optional, Union, Tuple, Dict, List, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from . import (
        Catalogue_Base,
        Catalogue,
        Combined_Catalogue,
        Rest_Frame_Property_Calculator,
        Property_Calculator,
        Mask_Selector,
    )
    from .selection import Completeness
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import useful_funcs_austind as funcs
from . import config, galfind_logger
from . import MCMC_Fitter, Priors, Schechter_Mag_Fitter, Schechter_Lum_Fitter
from .SED_codes import SED_code


class Base_Number_Density_Function:
    def __init__(
        self: Self,
        x_name: str,
        x_mid_bins: NDArray[float],
        z_ref: Union[str, int, float], 
        phi: NDArray[float],
        phi_errs_cv: NDArray[float],
        author_year: str
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
        cls, 
        x_name: str, 
        z_ref: Union[str, int, float], 
        author_year: str
    ) -> Self:
        if x_name in ["M1500", "M_UV", "MUV"]:
            x_name = "UVLF"
        if isinstance(z_ref, (str, int)):
            z_ref = float(z_ref)
        x_name_config_key = f"{x_name}_LIT_DIR"
        ecsv_data_path = f"{config['NumberDensityFunctions'][x_name_config_key]}/z={z_ref:.1f}/{author_year}.ecsv"
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
        assert obs_or_models in ["obs", "models"]
        sys.path.insert(1, config["NumberDensityFunctions"]["FLAGS_DATA_DIR"])
        try:
            from flags_data import distribution_functions
        except:
            galfind_logger.critical(
                "Could not import flags_data.distribution_functions"
            )

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

        num_obs = np.linspace(0, 1, len(datasets))
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
                        f"No available redshift for {author_year=} in {z_bin=}!" + \
                        f" Available redshifts are {ds.redshifts}"
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

                    if flags_property_name_conv[x_name] == "LUV":
                        phi_err = np.array([ds.log10phi_mag_err[z][0][::-1], ds.log10phi_mag_err[z][1][::-1]])
                        log10phi = ds.log10phi_mag[z][::-1]
                    else:
                        phi_err = np.array(ds.log10phi_err[z])
                        log10phi = ds.log10phi[z]
                    if len(np.shape(phi_err)) > 1:
                        low = np.array(phi_err[0])
                        high = np.array(phi_err[1])
                    else:
                        low = high = phi_err
                    err_high = 10 ** (log10phi + high) - 10**log10phi
                    err_low = (10**log10phi) - 10 ** (log10phi - low)
                    phi_err = np.array([err_low, err_high])
                    
                    if x_name in [key for (key, val) in \
                            flags_property_name_conv.items() if val == "LUV"]:
                        x = ds.M[z]
                    else:
                        x = ds.log10X[z]
                    # x = ds.log10X[z] - np.log10(1. / funcs.imf_mass_factor[ds.imf]) stellar mass only
                    return cls(
                        x_name, x, z, 10**log10phi, phi_err, author_year
                    )
        galfind_logger.info(f"No {author_year=} in {obs_or_models} for {x_name=}")
        return None  # if no author_year in flags_data

    def get_z_bin_name(self) -> str:
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
    ) -> Tuple[plt.Figure, plt.Axes]:

        if all(i is None for i in [fig, ax]):
            fig_, ax_ = plt.subplots()
        else:
            fig_, ax_ = fig, ax

        # don't plot empty bins
        if isinstance(self.x_mid_bins, (u.Quantity, u.Magnitude, u.Dex)):
            x_mid_bins = np.array(
                [_x for _x, _y in zip(self.x_mid_bins.value, self.phi) if _y != 0.0]
            )
        else:
            x_mid_bins = np.array(
                [_x for _x, _y in zip(self.x_mid_bins, self.phi) if _y != 0.0]
            )
        phi = np.array([_y for _y in self.phi if _y != 0.0])
        if not plot_cv_errs and hasattr(self, "phi_errs") and \
                isinstance(self, Number_Density_Function):
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
        ax_.errorbar(x_mid_bins, y, yerr=y_errs, **_plot_kwargs)
        galfind_logger.info(f"Plotting {default_plot_kwargs['label']}")

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
                #"bbox_to_anchor": (1.05, 0.5),
            }
            # overwrite default with input for duplicate kwargs
            for key in plot_kwargs.keys():
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
                assert len(x_lims) == 2
                ax_.set_xlim(*x_lims)
        if y_lims is not None:
            assert len(y_lims) == 2
            ax.set_ylim(*y_lims)

        if save:
            # if self.__class__.__name__ != "Number_Density_Function":
            #     raise NotImplementedError
            #     # assert save_path is not None
            #     # save_path = config['NumberDensityFunctions']['NUMBER_DENSITY_FUNC_DIR'] + \
            #     #     f"/Plots/Literature/{save_name}"
            # else:
            if save_path is None:
                save_path = self.get_plot_path()
                # if save_name is not None:
                #     plot_path = "/".join(plot_path.split("/")[:-1]) + f"/{save_name}.png"
            funcs.make_dirs(save_path)
            plt.savefig(save_path, bbox_inches="tight")
            funcs.change_file_permissions(save_path)
            galfind_logger.info(f"Saved plot to {save_path}")
        if show:
            plt.show()
        return fig_, ax_


class Number_Density_Function(Base_Number_Density_Function):
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
    ) -> Self:
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
        x_mid_bins = np.array(
            [(x_bin[1].value + x_bin[0].value) / 2.0 for x_bin in x_bins]
        ) * x_bins[0].unit
        
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
        completeness: Optional[Completeness] = None
    ) -> Self:
        tab = Table.read(save_path)
        x_bins_up = np.array(tab["x_bins_up"])
        x_bins_low = np.array(tab["x_bins_low"])
        x_unit = tab["x_bins_up"].unit
        x_bins = np.array(
            [
                [x_bin_low, x_bin_up]
                for x_bin_low, x_bin_up in zip(x_bins_low, x_bins_up)
            ]
        ) * x_unit
        Ngals = np.array(tab["Ngals"])
        phi = np.array(tab["phi"])
        phi_l1 = np.array(tab["phi_l1"])
        phi_u1 = np.array(tab["phi_u1"])
        cv_errs = np.array(tab["cv_errs"])
        # SED_fit_params_key, x_name, origin_surveys, z_bin = Number_Density_Function.extract_info_from_save_path(save_path)
        cv_origin = tab.meta["cv_origin"]
        x_origin = tab.meta["x_origin"]
        x_name = tab.meta["x_name"]
        origin_surveys = tab.meta["origin_surveys"]
        crop_name = tab.meta["crop_name"]
        z_bin = tab.meta["z_bin"]
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
        completeness: Optional[Catalogue_Completeness] = None,
        unmasked_area: Union[str, List[str], u.Quantity, Type[Mask_Selector]] = "selection",
        plot: bool = True,
        save: bool = True,
        timed: bool = False,
    ) -> Optional[Self]:
        # input assertions
        assert len(z_bin) == 2
        assert z_bin[0] < z_bin[1]
        assert len(x_bin_edges) >= 2
        # ensure x_bin_edges are sorted from lower to higher x values in every z bin
        assert all(
            _x == _sorted_x
            for _x, _sorted_x in zip(
                np.sort(np.array(x_bin_edges)), np.array(x_bin_edges)
            )
        )
        # TODO: ensure x_bin_edges are evenly spaced?

        assert cv_origin in ["Driver2010"]
        # SED fit label assertions
        assert isinstance(SED_fit_code, tuple(SED_code.__subclasses__()))
        assert all(SED_fit_code.label in gal.aper_phot[aper_diam].SED_results.keys() for gal in cat)
        # x_origin assertions
        assert x_origin in ["phot_rest", "SED_result"]

        # extract x values
        # TODO: Generalize this to exclude x_origin dependence
        if x_origin == "phot_rest":
            x = [gal.aper_phot[aper_diam].SED_results[SED_fit_code.label].phot_rest.properties[x_calculator.name] for gal in cat]
        else: # x_origin == "SED_result":
            x = [gal.aper_phot[aper_diam].SED_results[SED_fit_code.label].properties[x_calculator.name] for gal in cat]
        # remove nans
        x = [x_ for x_ in x if not np.isnan(x_)]
        assert all(x_.unit == x[0].unit for x_ in x)
        x = np.array([x_.value for x_ in x]) * x[0].unit

        # crop catalogue to this redshift bin
        from . import Redshift_Limit_Selector, Redshift_Bin_Selector
        # TODO: Implement Redshift_Limit_Selector in case of np.nan z_bin entry
        z_bin_selector = Redshift_Bin_Selector(aper_diam, SED_fit_code, z_bin)
        z_bin_cat = deepcopy(cat).crop(z_bin_selector)
        # ensure every galaxy in this redshift bin has 
        # the relevant property already calculated
        if len(z_bin_cat) == 0:
            galfind_logger.warning(
                f"No galaxies in {z_bin=}"
            )
            return None
        elif len([i for i, gal in enumerate(z_bin_cat) if \
                np.isnan(gal.aper_phot[aper_diam].SED_results \
                [SED_fit_code.label].phot_rest.properties \
                [x_calculator.name])]) != 0:
            nan_gals = [gal for gal in z_bin_cat if np.isnan(gal.aper_phot[aper_diam].SED_results[SED_fit_code.label].phot_rest.properties[x_calculator.name])]
            galfind_logger.warning(
                f"{len(nan_gals)} {repr(x_calculator)} nans for {z_bin=}!"
            )
            for gal in nan_gals:
                galfind_logger.warning(
                    f"{gal.ID}: (z={gal.aper_phot[aper_diam].SED_results[SED_fit_code.label].z:.2f}" + \
                    f",{gal.aper_phot[aper_diam].filterset.band_names})"
                )
            breakpoint()
            return None
        
        # determine save_path
        full_survey_name = funcs.get_full_survey_name(cat.survey, cat.version, cat.filterset)
        save_path = Number_Density_Function.get_save_path(
            cat.survey,
            x_origin,
            x_calculator.name,
            z_bin_cat.crop_name,
            completeness = completeness,
        )

        if not Path(save_path).is_file():
            # create x_bins from x_bin_edges (must include start and end values here too)
            x_bins = [
                [x_bin_edges[i].value, x_bin_edges[i + 1].value] * x_bin_edges.unit
                for i in range(len(x_bin_edges) - 1)
                if i != len(x_bin_edges) - 1
            ]
            # calculate Vmax for each galaxy in catalogue within z bin
            z_bin_cat.calc_Vmax(
                z_bin, 
                aper_diam, 
                SED_fit_code, 
                z_step,
                unmasked_area = unmasked_area,
            )

            if plot:
                z_bin_cat.plot_phot_diagnostics(
                    aper_diam, 
                    SED_arr = SED_fit_code,
                    zPDF_arr = SED_fit_code,
                )

            Ngals = np.zeros(len(x_bins))
            phi = np.zeros(len(x_bins))
            phi_l1 = np.zeros(len(x_bins))
            phi_u1 = np.zeros(len(x_bins))
            cv_errs = np.zeros(len(x_bins))
            #phi_errs_cv = np.zeros(len(x_bins))
            # loop through each mass bin in the given redshift bin
            for i, x_bin in enumerate(x_bins):
                if len(z_bin_cat) == 0:
                    Ngals[i] = 0
                else:

                    if plot:
                        # plot histogram
                        hist_fig, hist_ax = plt.subplots()
                        z_bin_cat.hist(x_calculator, hist_fig, hist_ax)
                        plt.close()
                    
                    # crop to galaxies in the x bin - not the bootstrapping method
                    from . import Rest_Frame_Property_Limit_Selector, Rest_Frame_Property_Bin_Selector
                    # TODO: Implement Rest_Frame_Property_Limit_Selector in case of np.nan x_bin entry
                    x_bin_selector = Rest_Frame_Property_Bin_Selector(aper_diam, SED_fit_code, x_calculator, x_bin)
                    z_bin_x_bin_cat = deepcopy(z_bin_cat).crop(x_bin_selector)
                    Ngals[i] = len(z_bin_x_bin_cat)

                    # plot cutouts
                    if plot and Ngals[i] > 0:
                        z_bin_x_bin_cat.plot_phot_diagnostics(
                            aper_diam, 
                            SED_arr = SED_fit_code,
                            zPDF_arr = SED_fit_code,
                        )
                    
                # if there are galaxies in the z,x bin
                if int(Ngals[i]) != 0:
                    # plot histogram
                    #z_bin_x_bin_cat.hist(x_calculator, hist_fig, hist_ax)
                    dx = x_bin[1].value - x_bin[0].value
                    # extract Vmax's
                    V_max = np.array(
                        [
                            gal.aper_phot[aper_diam].SED_results[SED_fit_code.label]. \
                            V_max[z_bin_cat.crop_name.split("/")[-1]][full_survey_name].value
                            for gal in z_bin_x_bin_cat
                        ]
                    )
                    remove_indices = V_max != -1.0
                    V_max = V_max[remove_indices] #* u.Mpc ** 3
                    if len(V_max) != Ngals[i]:
                        galfind_logger.warning(
                            f"{Ngals[i] - len(V_max)} galaxies not detected"
                        )
                    if completeness is None:
                        compl_bin = np.ones(len(z_bin_x_bin_cat))
                    else:
                        compl_bin = completeness(z_bin_x_bin_cat)
                    try:
                        compl_bin = compl_bin[remove_indices]
                    except:
                        breakpoint()
                    assert len(compl_bin) == len(V_max), \
                        galfind_logger.critical(
                            f"{len(compl_bin)=} != {len(V_max)=} for {z_bin_x_bin_cat.crop_name}"
                        )
                    # import matplotlib.pyplot as plt
                    # from scipy.interpolate import interp1d
                    # fig, ax = plt.subplots()
                    # ax.scatter(completeness.compl_arr[0].x_calculator(z_bin_x_bin_cat)[remove_indices], compl_bin, label = str(x_bin))
                    # ax.plot(completeness.compl_arr[0].x, completeness.compl_arr[0].completeness, label = "Completeness")
                    # ax.plot(completeness.compl_arr[0].x, interp1d(completeness.compl_arr[0].x, completeness.compl_arr[0].completeness)(completeness.compl_arr[0].x), label = "Interpolated Completeness")
                    # ax.legend()
                    # plt.savefig("test_compl_NEP.png")
                    # breakpoint()
                    phi[i] = np.sum((V_max * compl_bin) ** - 1.0) / dx
                    # use standard Poisson errors if number of galaxies in bin is not small
                    if len(V_max) >= 4:
                        phi_errs = np.sqrt(np.sum((V_max * compl_bin) ** -2.0)) / dx
                        phi_l1[i] = phi_errs
                        phi_u1[i] = phi_errs
                    else:
                        poisson_int = funcs.poisson_interval(len(V_max), 0.32)
                        phi_l1[i] = phi[i] * np.min(
                            np.abs((np.array(poisson_int[0]) - len(V_max)))
                            / len(V_max)
                        )
                        phi_u1[i] = phi[i] * np.min(
                            np.abs((np.array(poisson_int[1]) - len(V_max)))
                            / len(V_max)
                        )
                    from . import Catalogue_Base, Catalogue, Combined_Catalogue
                    if isinstance(cat, Combined_Catalogue):
                        data_arr = [cat_.data for cat_ in cat.cat_arr]
                    elif isinstance(cat, Catalogue):
                        data_arr = [cat.data]
                    else:
                        err_message = f"{repr(cat)=} not in {', '.join(Catalogue_Base.__subclasses__())}!"
                        raise ValueError(err_message)
                    if cv_origin is None:
                        pass
                    elif cv_origin == "Driver2010":
                        cv_errs[i] = funcs.calc_cv_proper(
                            z_bin, 
                            data_arr = data_arr,
                            masked_selector = unmasked_area,
                            z = np.sum(z_bin) / 2.0,
                        )
                    else:
                        raise NotImplementedError
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
                completeness = completeness,
            )
            if save:
                number_density_func.save()
            return number_density_func

        else: # load results
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
    ) -> str:
        if completeness is None:
            compl_name = ""
        else:
            compl_name = "_compl_corr"
        save_path = config['NumberDensityFunctions']['NUMBER_DENSITY_FUNC_DIR'] + \
            f"/Data/{SED_fit_params_key}/{x_name}/" + \
            f"{origin_surveys}/{crop_name}{compl_name}{ext}"
        funcs.make_dirs(save_path)
        return save_path

    # cv_origin == "Driver2010"
    @staticmethod
    def extract_info_from_save_path(save_path):
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

    # def get_z_bin_name(self) -> str:
    #     return f"{self.z_bin[0]:.1f}<z<{self.z_bin[1]:.1f}"

    def get_plot_path(self) -> str:
        plot_path = self.get_save_path(
            self.origin_surveys,
            self.x_origin,
            self.x_name,
            self.crop_name,
            ext = ".png",
            completeness = self.completeness,
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
        backend_filename: Optional[str] = None
    ) -> NoReturn:
        if backend_filename is None:
            backend_filename = self.get_save_path(
                self.origin_surveys,
                self.x_origin,
                self.x_name,
                self.crop_name,
                completeness = self.completeness,
            )
            backend_filename = backend_filename\
                .replace("/Data/", f"/{fit_type.__name__.replace('Fitter', 'Fits')}/")\
                .replace(".ecsv", ".h5")
            funcs.make_dirs(backend_filename)
        # remove 0s from x_mid_bins, phi, and phi_errs
        zero_indices = np.where(self.phi == 0.0)[0]
        x_mid_bins = np.delete(self.x_mid_bins.value, zero_indices)
        phi = np.delete(self.phi, zero_indices)
        phi_errs_cv = np.array([np.delete(self.phi_errs_cv[0], zero_indices), 
            np.delete(self.phi_errs_cv[1], zero_indices)])
        self.fitter = fit_type(
            priors, 
            x_mid_bins,
            phi,
            phi_errs_cv,
            n_walkers,
            backend_filename,
            fixed_params
        )
        # run fitter
        self.fitter(n_steps, n_processes)

    def save(self, save_path: Optional[str] = None) -> NoReturn:
        if save_path is None:
            save_path = self.get_save_path(
                self.origin_surveys,
                self.x_origin,
                self.x_name,
                self.crop_name,
                completeness = self.completeness,
            )
        assert all(x_bin[0].unit == self.x_bins[0][0].unit for x_bin in self.x_bins)
        assert all(x_bin[1].unit == self.x_bins[0][1].unit for x_bin in self.x_bins)
        assert self.x_bins[0][0].unit == self.x_bins[0][1].unit
        x_bins_low = np.array([x_bin[0].value for x_bin in self.x_bins]) * self.x_bins[0][0].unit
        x_bins_up = np.array([x_bin[1].value for x_bin in self.x_bins]) * self.x_bins[0][1].unit
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
            "crop_name": self.crop_name
        }
        funcs.make_dirs(save_path)
        tab.write(save_path, overwrite=True)
        galfind_logger.info(
            f"Saved {self.x_name} {self.z_bin} " + \
            f"{self.origin_surveys} to {save_path}"
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
        obs_author_years: Dict[str, Any] = {},
        sim_author_years: Dict[str, Any] = {},
        save_path: Optional[str] = None,
        plot_cv_errs: bool = False,
    ) -> Tuple[plt.Figure, plt.Axes]:
        
        if all(_x is None for _x in [fig, ax]):
            fig_, ax_ = plt.subplots()
        else:
            fig_, ax_ = fig, ax

        if title is None:
            title = self.crop_name

        for author_year, author_year_kwargs in obs_author_years.items():
            author_year_func_from_flags_data = (
                Base_Number_Density_Function.from_flags_repo(
                    self.x_name, self.z_bin, author_year, "obs"
                )
            )
            if author_year_func_from_flags_data is not None:
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
                )
        for author_year, author_year_kwargs in sim_author_years.items():
            author_year_func_from_flags_data = (
                Base_Number_Density_Function.from_flags_repo(
                    self.x_name, self.z_bin, author_year, "models"
                )
            )
            if author_year_func_from_flags_data is not None:
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
                )

        fig_, ax_ = super().plot(
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
            plot_cv_errs = plot_cv_errs,
        )
                
        return fig_, ax_


#         def mass_function(catalog, fields, z_bins, mass_bins, rerun=False, out_directory = '/nvme/scratch/work/tharvey/masses/',
#  mass_keyword='MASS_BEST',mass_form='log', z_keyword='Z_BEST', sed_tool='LePhare', template='', z_step=0.01,
#   n_jobs=2, cat_version='v7', do_muv=False, use_vmax_simple = False, field_keyword='field',
#   other_name = '', other_sed_path='/nvme/scratch/work/austind/Bagpipes/pipes/seds/',
#   use_base=True, base_cat='/nvme/scratch/work/tharvey/catalogs/robust_and_good_gal_all_criteria_3sigma_all_fields_masses.fits',
#   id_keyword='NUMBER',  use_new_zloop=True, select_444=False, use_bootstrap=True, rerun_other_pdfs = True,
#     other_appended=False, flag_ids=[], base_cat_filter=None, zgauss=False):

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
#     vmax_keyword=vmax_keyword, name_444=name_444, zgauss=zgauss, sed_tool=sed_tool)

#     plot_bins_pdfs(catalog, fields, z_bins, mass_bins, rerun=False, out_directory = out_directory,
#     mass_form=mass_form,mass_keyword=mass_keyword, z_keyword=z_keyword, zgauss=zgauss)
# else:


class Multiple_Number_Density_Function:
    def __init__(
        self, 
        number_density_function_arr: Union[list, np.array]
    ):
        self.number_density_function_arr = number_density_function_arr

    @classmethod
    def from_cat(
        cls,
        cat: Catalogue,
        x_name: str,
        x_bin_edges_arr: Union[list, np.array],
        z_bins: Union[list, np.array],
        x_origin: Union[str, dict] = "EAZY_fsps_larson_zfree",
        z_step: float = 0.1,
        use_vmax_simple: bool = False,
        unmasked_area: Union[str, List[str], u.Quantity] = "selection",
        timed: bool = False,
    ) -> "Number_Density_Function":
        # input assertions
        assert all(len(z_bin) == 2 for z_bin in z_bins)
        assert all(z_bin[0] < z_bin[1] for z_bin in z_bins)
        assert len(x_bin_edges_arr) == len(z_bins)
        assert all(len(x_bin_edges) >= 2 for x_bin_edges in x_bin_edges_arr)
        # ensure x_bin_edges are sorted from lower to higher x values in every z bin
        assert all(
            np.sort(np.array(x_bin_edges)) == np.array(x_bin_edges)
            for x_bin_edges in x_bin_edges_arr
        )
        # ensure x_bin_edges are evenly spaced?
        # extract x_name values from catalogue
        if isinstance(x_origin, dict):
            assert "code" in x_origin.keys()
            assert x_origin["code"].__class__.__name__ in [
                code.__name__ for code in SED_code.__subclasses__()
            ]
            SED_fit_params = x_origin  # redshifts must come from same SED fitting as x values
        elif isinstance(x_origin, str):
            # convert to SED_fit_params
            SED_fit_params = x_origin.split("_")[0]
        else:
            galfind_logger.critical(
                f"{x_origin=} with {type(x_origin)=} not in [dict, str]!"
            )

        x = getattr(cat, x_name, x_origin)

        # calculate mass function in each redshift bin
        for i, (z_bin, x_bin_edges) in enumerate(zip(z_bins, x_bin_edges_arr)):
            # create x_bins from x_bin_edges (must include start and end values here too)
            x_bins = [
                [x_bin_edges[i], x_bin_edges[i + 1]]
                for i in range(len(x_bin_edges) - 1)
                if i != len(x_bin_edges) - 1
            ]
            # extract z_bin_name
            assert isinstance(x_origin, str)
            z_bin_name = f"{x_origin}_{z_bin[0]:.1f}<z<{z_bin[1]:.1f}"
            # calculate Vmax for each galaxy in catalogue within z bin
            # in general call Vmax_multifield
            cat.calc_Vmax(cat.data, z_bin, x_origin, z_step, unmasked_area = unmasked_area, timed=timed)
            # crop catalogue to this redshift bin
            z_bin_cat = cat.crop(z_bin, "z", x_origin)

            Ngals = np.zeros(len(x_bins))
            phi = np.zeros(len(x_bins))
            phi_errs = np.zeros(len(x_bins))
            cv_errs = np.zeros(len(x_bins))
            phi_errs_cv = np.zeros(len(x_bins))
            # loop through each mass bin in the given redshift bin
            for j, x_bin in enumerate(x_bins):
                # crop to galaxies in the x bin - not the bootstrapping method
                z_bin_x_bin_cat = z_bin_cat.crop(x_bin, x_name, SED_fit_params)

                Ngals[j] = len(z_bin_x_bin_cat)
                # if there are galaxies in the z, mx bin
                if Ngals[j] != 0:
                    dx = x_bin[1] - x_bin[0]
                    V_max = np.array(
                        [
                            gal.V_max[z_bin_name][cat.data.full_name]
                            for gal in cat
                        ]
                    )
                    phi[j] = (np.sum(V_max**-1.0) / dx).value
                    # use standard Poisson errors if number of galaxies in bin is not small
                    if len(V_max) >= 4:
                        phi_errs[j] = (np.sqrt(np.sum(V_max**-2.0)) / dx).value
                    else:
                        # using minimum is a minor cheat for symmetric errors?
                        phi_errs[j] = phi[j] * np.min(
                            np.abs(
                                (
                                    np.array(
                                        funcs.poisson_interval(
                                            len(V_max), 0.32
                                        )
                                    )
                                    - len(V_max)
                                )
                            )
                            / len(V_max)
                        )
                    cv_errs[j] = funcs.calc_cv_proper(
                        float(z_bin[0]),
                        float(z_bin[1]),
                        fields_used=fields_used,
                        **kwargs,
                    )
                    phi_errs_cv[j] = np.sqrt(
                        phi_errs[j] ** 2.0 + (cv_errs[j] * phi[j]) ** 2.0
                    )

    def __len__(self):
        return len(self.number_density_function_arr)

    def plot(self):
        pass

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import NoReturn, Union
import astropy.units as u
import os
import sys
from pathlib import Path
from astropy.table import Table

from . import useful_funcs_austind as funcs
from . import galfind_logger, config, astropy_cosmo, sed_code_to_name_dict
from . import Galaxy, Photometry_obs
from .SED import SED_obs
from .SED_codes import SED_code


class Base_Number_Density_Function:
    def __init__(self, x_name, x_mid_bins, z_ref, phi, phi_errs_cv, author_year):
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
    ) -> "Base_Number_Density_Function":  # literature
        if x_name in ["M1500", "M_UV", "MUV"]:
            x_name = "UVLF"
        if type(z_ref) in [str, int]:
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
    ) -> Union[None, "Base_Number_Density_Function"]:
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
            if ds.name == author_year:
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
                if type(z) == type(None):
                    galfind_logger.warning(
                        f"No available redshift for {author_year=} in {z_bin=}! Available redshifts are {ds.redshifts}"
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

                    if x_name in ["M1500", "M_UV"]:
                        x = ds.M[z]
                    else:
                        x = ds.log10X[z]
                    # x = ds.log10X[z] - np.log10(1. / funcs.imf_mass_factor[ds.imf]) stellar mass only
                    return cls(x_name, x, z, 10**log10phi, phi_err, author_year)
        return None  # if no author_year in flags_data

    def get_z_bin_name(self) -> str:
        return f"z={float(self.z_ref):.1f}"

    def plot(
        self,
        fig=None,
        ax=None,
        log: bool = False,
        annotate: bool = False,
        save: bool = False,
        show: bool = False,
        plot_kwargs: dict = {},
        legend_kwargs: dict = {},
        x_lims: Union[list, np.array, str, None] = "default",
        save_name: Union[str, None] = None,
    ) -> None:
        if all(type(i) == type(None) for i in [fig, ax]):
            fig, ax = plt.subplots()

        # don't plot empty bins
        x_mid_bins = np.array(
            [_x for _x, _y in zip(self.x_mid_bins, self.phi) if _y != 0.0]
        )
        phi = np.array([_y for _y in self.phi if _y != 0.0])
        phi_errs_cv = np.array(
            [
                [
                    _yerr
                    for _yerr, _y in zip(self.phi_errs_cv[0], self.phi)
                    if _y != 0.0
                ],
                [
                    _yerr
                    for _yerr, _y in zip(self.phi_errs_cv[1], self.phi)
                    if _y != 0.0
                ],
            ]
        )
        if log:
            y = np.log10(phi)
            y_errs = np.array(
                [
                    np.log10(_phi / (_phi - _phi_err))
                    for _phi, _phi_err in zip(phi, phi_errs_cv)
                ],
                [
                    np.log10(1.0 + (_phi_err / _phi))
                    for _phi, _phi_err in zip(phi, phi_errs_cv)
                ],
            )
        else:
            y = phi
            y_errs = phi_errs_cv

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
        _plot_kwargs = {**plot_kwargs, **default_plot_kwargs}
        ax.errorbar(x_mid_bins, y, yerr=y_errs, **_plot_kwargs)
        galfind_logger.info(f"Plotting {default_plot_kwargs['label']}")

        if annotate:
            y_label = r"$\Phi$ / N dex$^{-1}$Mpc$^{-3}$"
            if log:
                y_label = r"$\log_{10}($" + y_label + ")"
            else:
                ax.set_yscale("log")
            ax.set_xlabel(self.x_name)
            ax.set_ylabel(y_label)
            if type(x_lims) != type(None):
                if type(x_lims) in [str]:
                    ax.set_xlim(*funcs.default_lims[x_lims])
                else:
                    assert len(x_lims) == 2
                    ax.set_xlim(*x_lims)
            # sort out legend_kwargs
            default_legend_kwargs = {
                "loc": "center left",
                "bbox_to_anchor": (1.05, 0.5),
            }
            # overwrite default with input for duplicate kwargs
            for key in plot_kwargs.keys():
                if key in default_legend_kwargs.keys():
                    default_legend_kwargs.pop(key)
            _legend_kwargs = {**legend_kwargs, **default_legend_kwargs}
            ax.legend(**_legend_kwargs)
        if save:
            if self.__class__.__name__ != "Number_Density_Function":
                assert type(save_name) != type(None)
                plot_path = f"{config['NumberDensityFunctions']['NUMBER_DENSITY_FUNC_DIR']}/Plots/Literature/{save_name}"
            else:
                plot_path = self.get_plot_path()
            funcs.make_dirs(plot_path)
            plt.savefig(plot_path)
        if show:
            plt.show()


class Number_Density_Function(Base_Number_Density_Function):
    def __init__(
        self,
        x_name,
        x_bins,
        x_origin,
        z_bin,
        Ngals,
        phi,
        phi_errs,
        cv_errs,
        origin_surveys,
        cv_origin,
    ) -> "Number_Density_Function":
        self.x_bins = x_bins
        self.x_origin = x_origin
        self.z_bin = z_bin
        self.Ngals = Ngals
        self.phi_errs = phi_errs  # poisson only
        self.cv_errs = cv_errs  # cosmic variance % errs / 100
        self.origin_surveys = origin_surveys
        self.cv_origin = cv_origin
        x_mid_bins = np.array([(x_bin[1] + x_bin[0]) / 2.0 for x_bin in x_bins])
        z_ref = float((z_bin[1] + z_bin[0]) / 2.0)
        phi_errs_cv = np.array(
            [np.sqrt(phi_errs[i] ** 2.0 + (cv_errs * phi) ** 2.0) for i in range(2)]
        )
        super().__init__(x_name, x_mid_bins, z_ref, phi, phi_errs_cv, "This work")

    @classmethod
    def from_ecsv(cls, save_path: str):
        tab = Table.read(save_path)
        x_bins_up = np.array(tab["x_bins_up"])
        x_bins_low = np.array(tab["x_bins_low"])
        x_bins = np.array(
            [
                [x_bin_low, x_bin_up]
                for x_bin_low, x_bin_up in zip(x_bins_low, x_bins_up)
            ]
        )
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
            cv_origin,
        )

    @classmethod
    def from_multiple_cat(
        cls,
        multiple_cat,
        x_name: str,
        x_bin_edges: Union[list, np.array],
        z_bin: Union[list, np.array],
        x_origin: Union[str, dict] = "EAZY_fsps_larson_zfree",
        z_step: float = 0.01,
        cv_origin: Union[str, None] = "Driver2010",
        save: bool = True,
        timed: bool = False,
    ) -> "Number_Density_Function":
        pass

    @classmethod
    def from_single_cat(
        cls,
        cat,
        x_name: str,
        x_bin_edges: Union[list, np.array],
        z_bin: Union[list, np.array],
        x_origin: Union[str, dict] = "EAZY_fsps_larson_zfree",
        z_step: float = 0.01,
        cv_origin: Union[str, None] = "Driver2010",
        save: bool = True,
        timed: bool = False,
    ) -> "Number_Density_Function":
        return cls.from_cat(
            cat,
            [cat.data],
            x_name,
            x_bin_edges,
            z_bin,
            x_origin,
            z_step,
            cv_origin,
            save,
            timed,
        )

    @classmethod
    def from_cat(
        cls,
        cat,
        data_arr: Union[list, np.array],
        x_name: str,
        x_bin_edges: Union[list, np.array],
        z_bin: Union[list, np.array],
        x_origin: Union[str, dict] = "EAZY_fsps_larson_zfree_REST_PROPERTY",
        z_step: float = 0.01,
        cv_origin: Union[str, None] = "Driver2010",
        save: bool = True,
        timed: bool = False,
    ) -> "Number_Density_Function":
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
        # ensure x_bin_edges are evenly spaced?
        assert cv_origin in ["Driver2010"]

        if type(x_origin) in [dict]:
            assert "code" in x_origin.keys()
            assert x_origin["code"].__class__.__name__ in [
                code.__name__ for code in SED_code.__subclasses__()
            ]
            SED_fit_params = (
                x_origin  # redshifts must come from same SED fitting as x values
            )
            SED_fit_params_key = x_origin["code"].label_from_SED_fit_params(
                SED_fit_params
            )
            x_origin = SED_fit_params_key
        elif type(x_origin) in [str]:
            galfind_logger.debug("Won't work for rest frame properties currently")
            # convert to SED_fit_params
            if x_origin.endswith("_REST_PROPERTY"):
                SED_fit_params_key = x_origin.replace("_REST_PROPERTY", "")
            else:
                SED_fit_params_key = x_origin
            SED_fit_params = sed_code_to_name_dict[
                SED_fit_params_key.split("_")[0]
            ].SED_fit_params_from_label(SED_fit_params_key)
        else:
            galfind_logger.critical(
                f"{x_origin=} with {type(x_origin)=} not in [dict, str]!"
            )

        # determine save_path
        origin_surveys = Number_Density_Function.get_origin_surveys(data_arr)
        save_path = Number_Density_Function.get_save_path(
            origin_surveys, SED_fit_params_key, z_bin, x_name
        )

        if not Path(
            save_path
        ).is_file():  # perform calculation if not previously computed
            # extract z_bin_name
            z_bin_name = funcs.get_SED_fit_params_z_bin_name(SED_fit_params_key, z_bin)
            # crop catalogue to this redshift bin
            z_bin_cat = cat.crop(z_bin, "z", SED_fit_params)

            # extract photometry type from x_origin
            phot_type = "rest" if x_origin.endswith("_REST_PROPERTY") else "obs"
            # create x_bins from x_bin_edges (must include start and end values here too)
            x_bins = [
                [x_bin_edges[i], x_bin_edges[i + 1]]
                for i in range(len(x_bin_edges) - 1)
                if i != len(x_bin_edges) - 1
            ]

            # calculate Vmax for each galaxy in catalogue within z bin
            z_bin_cat.calc_Vmax(
                data_arr, z_bin, SED_fit_params_key, z_step, timed=timed
            )

            Ngals = np.zeros(len(x_bins))
            phi = np.zeros(len(x_bins))
            phi_l1 = np.zeros(len(x_bins))
            phi_u1 = np.zeros(len(x_bins))
            cv_errs = np.zeros(len(x_bins))
            phi_errs_cv = np.zeros(len(x_bins))
            # loop through each mass bin in the given redshift bin
            for i, x_bin in enumerate(x_bins):
                # crop to galaxies in the x bin - not the bootstrapping method
                z_bin_x_bin_cat = z_bin_cat.crop(
                    x_bin, x_name, SED_fit_params, phot_type
                )

                Ngals[i] = len(z_bin_x_bin_cat)
                # if there are galaxies in the z,x bin
                if int(Ngals[i]) != 0:
                    dx = x_bin[1] - x_bin[0]
                    # calculate Vmax's
                    V_max = np.zeros(int(Ngals[i])).astype(float)
                    for data in data_arr:
                        V_max += np.array(
                            [
                                gal.V_max[z_bin_name][data.full_name]
                                if gal.V_max[z_bin_name][data.full_name] != -1.0
                                else 0.0
                                for gal in z_bin_x_bin_cat
                            ]
                        )
                    V_max = np.array([_V_max for _V_max in V_max if _V_max != 0.0])
                    if len(V_max) != Ngals[i]:
                        galfind_logger.warning(
                            f"{Ngals[i] - len(V_max)} galaxies not detected"
                        )
                    phi[i] = np.sum(V_max**-1.0) / dx
                    # use standard Poisson errors if number of galaxies in bin is not small
                    if len(V_max) >= 4:
                        phi_errs = np.sqrt(np.sum(V_max**-2.0)) / dx
                        phi_l1[i] = phi_errs
                        phi_u1[i] = phi_errs
                    else:
                        poisson_int = funcs.poisson_interval(len(V_max), 0.32)
                        phi_l1[i] = phi[i] * np.min(
                            np.abs((np.array(poisson_int[0]) - len(V_max))) / len(V_max)
                        )
                        phi_u1[i] = phi[i] * np.min(
                            np.abs((np.array(poisson_int[1]) - len(V_max))) / len(V_max)
                        )
                    if type(cv_origin) == type(None):
                        pass
                    elif (
                        cv_origin == "Driver2010"
                    ):  # could open this up to more cosmic variance calculators
                        cv_errs[i] = funcs.calc_cv_proper(z_bin, data_arr=data_arr)
                    else:
                        raise NotImplementedError

            number_density_func = cls(
                x_name,
                x_bins,
                x_origin,
                z_bin,
                Ngals,
                phi,
                np.array([phi_l1, phi_u1]),
                cv_errs,
                origin_surveys,
                cv_origin,
            )

            if save and not Path(save_path).is_file():
                number_density_func.save()

            return number_density_func

        else:  # load results
            return cls.from_ecsv(save_path)

    @staticmethod
    def get_origin_surveys(data_arr) -> str:
        return "+".join([data.full_name for data in data_arr])

    # cv_origin == "Driver2010"
    @staticmethod
    def get_save_path(
        origin_surveys: str,
        SED_fit_params_key: str,
        z_bin: Union[list, np.array],
        x_name: str,
        ext: str = ".ecsv",
    ) -> str:
        save_path = f"{config['NumberDensityFunctions']['NUMBER_DENSITY_FUNC_DIR']}/Data/{SED_fit_params_key}/{x_name}/{origin_surveys}/{z_bin[0]}<z<{z_bin[1]}{ext}"
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

    def get_z_bin_name(self) -> str:
        return f"{self.z_bin[0]:.1f}<z<{self.z_bin[1]:.1f}"

    def get_plot_path(self) -> str:
        plot_path = self.get_save_path(
            self.origin_surveys,
            self.x_origin.replace("_REST_PROPERTY", ""),
            self.z_bin,
            self.x_name,
            ext=".png",
        ).replace("/Data/", "/Plots/")
        funcs.make_dirs(plot_path)
        return plot_path

    def save(self, save_path: Union[str, None] = None) -> None:
        if type(save_path) == type(None):
            save_path = self.get_save_path(
                self.origin_surveys,
                self.x_origin.replace("_REST_PROPERTY", ""),
                self.z_bin,
                self.x_name,
            )
        x_bins_low = np.array([x_bin[0] for x_bin in self.x_bins])
        x_bins_up = np.array([x_bin[1] for x_bin in self.x_bins])
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
        }
        tab.write(save_path, overwrite=True)

    def plot(
        self,
        fig=None,
        ax=None,
        log: bool = False,
        annotate: bool = True,
        save: bool = True,
        show: bool = False,
        plot_kwargs: dict = {},
        legend_kwargs: dict = {},
        x_lims: Union[list, np.array, str, None] = "default",
        obs_author_years: dict = {},
        sim_author_years: dict = {},
    ) -> None:
        if all(type(_x) == type(None) for _x in [fig, ax]):
            fig, ax = plt.subplots()
        for author_year, author_year_kwargs in obs_author_years.items():
            author_year_func_from_flags_data = (
                Base_Number_Density_Function.from_flags_repo(
                    self.x_name, self.z_bin, author_year, "obs"
                )
            )
            if type(author_year_func_from_flags_data) != type(None):
                author_year_func_from_flags_data.plot(
                    fig,
                    ax,
                    log,
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
            if type(author_year_func_from_flags_data) != type(None):
                author_year_func_from_flags_data.plot(
                    fig,
                    ax,
                    log,
                    annotate=False,
                    save=False,
                    show=False,
                    plot_kwargs=author_year_kwargs,
                    x_lims=None,
                )
        # plot this work
        super().plot(
            fig, ax, log, annotate, save, show, plot_kwargs, legend_kwargs, x_lims
        )


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
    def __init__(self, number_density_function_arr: Union[list, np.array]) -> None:
        self.number_density_function_arr = number_density_function_arr

    @classmethod
    def from_cat(
        cls,
        cat,
        x_name: str,
        x_bin_edges_arr: Union[list, np.array],
        z_bins: Union[list, np.array],
        x_origin: Union[str, dict] = "EAZY_fsps_larson_zfree",
        z_step: float = 0.1,
        use_vmax_simple: bool = False,
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
        if type(x_origin) in [dict]:
            assert "code" in x_origin.keys()
            assert x_origin["code"].__class__.__name__ in [
                code.__name__ for code in SED_code.__subclasses__()
            ]
            SED_fit_params = (
                x_origin  # redshifts must come from same SED fitting as x values
            )
        elif type(x_origin) in [str]:
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
            assert type(x_origin) == str  # NOT GENERAL!
            z_bin_name = f"{x_origin}_{z_bin[0]:.1f}<z<{z_bin[1]:.1f}"
            # calculate Vmax for each galaxy in catalogue within z bin
            # in general call Vmax_multifield
            cat.calc_Vmax(cat.data, z_bin, x_origin, z_step, timed=timed)
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
                        [gal.V_max[z_bin_name][cat.data.full_name] for gal in cat]
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
                                    np.array(funcs.poisson_interval(len(V_max), 0.32))
                                    - len(V_max)
                                )
                            )
                            / len(V_max)
                        )
                    cv_errs[j] = funcs.calc_cv_proper(
                        float(z_bin[0]), float(z_bin[1]), fields_used=fields_used
                    )
                    phi_errs_cv[j] = np.sqrt(
                        phi_errs[j] ** 2.0 + (cv_errs[j] * phi[j]) ** 2.0
                    )

    def __len__(self):
        return len(self.number_density_function_arr)

    def plot(self):
        pass

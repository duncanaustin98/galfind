import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import NoReturn, Union
import astropy.units as u
import os
from pathlib import Path
from astropy.table import Table

from . import useful_funcs_austind as funcs
from . import galfind_logger, config, astropy_cosmo, sed_code_to_name_dict
from . import Galaxy, Photometry_obs
from .SED import SED_obs
from .SED_codes import SED_code

class Number_Density_Function:

    def __init__(self, x_name, x_bins, x_origin, z_bin, Ngals, \
            phi, phi_errs, cv_errs, origin_surveys, cv_origin, \
            author_year: str = "This work") -> "Number_Density_Function":
        self.x_name = x_name
        self.x_bins = x_bins
        self.x_origin = x_origin
        self.z_bin = z_bin
        self.Ngals = Ngals
        self.phi = phi
        self.phi_errs = phi_errs
        self.cv_errs = cv_errs
        self.origin_surveys = origin_surveys
        self.cv_origin = cv_origin
        self.author_year = author_year

    @classmethod
    def from_save_path(cls, save_path: str):
        tab = Table.read(save_path)
        x_bins_up = np.array(tab["x_bins_up"])
        x_bins_low = np.array(tab["x_bins_low"])
        x_bins = np.array([[x_bin_low, x_bin_up] for x_bin_low, x_bin_up in zip(x_bins_low, x_bins_up)])
        Ngals = np.array(tab["Ngals"])
        phi = np.array(tab["phi"])
        phi_errs = np.array(tab["phi_errs"])
        cv_errs = np.array(tab["cv_errs"])
        #SED_fit_params_key, x_name, origin_surveys, z_bin = Number_Density_Function.extract_info_from_save_path(save_path)
        cv_origin = tab.meta["cv_origin"]
        x_origin = tab.meta["x_origin"]
        x_name = tab.meta["x_name"]
        origin_surveys = tab.meta["origin_surveys"]
        z_bin = tab.meta["z_bin"]
        return cls(x_name, x_bins, x_origin, z_bin, Ngals, phi, phi_errs, cv_errs, origin_surveys, cv_origin)

    @classmethod
    def from_multiple_cat(cls, multiple_cat, x_name: str, x_bin_edges: Union[list, np.array], \
            z_bin: Union[list, np.array], x_origin: Union[str, dict] = "EAZY_fsps_larson_zfree", \
            z_step: float = 0.01, cv_origin: Union[str, None] = "Driver2010", save: bool = True, \
            timed: bool = False) -> "Number_Density_Function":
        pass

    @classmethod
    def from_single_cat(cls, cat, x_name: str, x_bin_edges: Union[list, np.array], \
            z_bin: Union[list, np.array], x_origin: Union[str, dict] = "EAZY_fsps_larson_zfree", \
            z_step: float = 0.01, cv_origin: Union[str, None] = "Driver2010", save: bool = True, \
            timed: bool = False) -> "Number_Density_Function":
        return cls.from_cat(cat, [cat.data], x_name, x_bin_edges, z_bin, x_origin, z_step, cv_origin, save, timed)
    
    @classmethod
    def from_cat(cls, cat, data_arr: Union[list, np.array], x_name: str, x_bin_edges: Union[list, np.array], \
            z_bin: Union[list, np.array], x_origin: Union[str, dict] = "EAZY_fsps_larson_zfree_REST_PROPERTY", \
            z_step: float = 0.01, cv_origin: Union[str, None] = "Driver2010", save: bool = True, \
            timed: bool = False) -> "Number_Density_Function":
        
        # input assertions
        assert len(z_bin) == 2
        assert z_bin[0] < z_bin[1]
        assert len(x_bin_edges) >= 2
        # ensure x_bin_edges are sorted from lower to higher x values in every z bin
        assert all(_x == _sorted_x for _x, _sorted_x in zip(np.sort(np.array(x_bin_edges)), np.array(x_bin_edges)))
        # ensure x_bin_edges are evenly spaced?
        assert cv_origin in ["Driver2010"]

        if type(x_origin) in [dict]:
            assert "code" in x_origin.keys()
            assert x_origin["code"].__class__.__name__ in [code.__name__ for code in SED_code.__subclasses__()]
            SED_fit_params = x_origin # redshifts must come from same SED fitting as x values
            SED_fit_params_key = x_origin["code"].label_from_SED_fit_params(SED_fit_params)
        elif type(x_origin) in [str]:
            galfind_logger.debug("Won't work for rest frame properties currently")
            # convert to SED_fit_params
            if x_origin.endswith("_REST_PROPERTY"):
                SED_fit_params_key = x_origin.replace("_REST_PROPERTY", "")
            else:
                SED_fit_params_key = x_origin
            SED_fit_params = sed_code_to_name_dict[SED_fit_params_key.split("_")[0]].SED_fit_params_from_label(SED_fit_params_key)
        else:
            galfind_logger.critical(f"{x_origin=} with {type(x_origin)=} not in [dict, str]!")

        # determine save_path
        origin_surveys = Number_Density_Function.get_origin_surveys(data_arr)
        save_path = Number_Density_Function.get_save_path(origin_surveys, SED_fit_params_key, z_bin, x_name)
        
        if not Path(save_path).is_file(): # perform calculation if not previously computed
            # extract z_bin_name
            z_bin_name = funcs.get_SED_fit_params_z_bin_name(SED_fit_params_key, z_bin)
            # crop catalogue to this redshift bin
            z_bin_cat = cat.crop(z_bin, "z", SED_fit_params)

            # extract photometry type from x_origin
            phot_type = "rest" if x_origin.endswith("_REST_PROPERTY") else "obs"
            # create x_bins from x_bin_edges (must include start and end values here too)
            x_bins = [[x_bin_edges[i], x_bin_edges[i + 1]] \
                for i in range(len(x_bin_edges) - 1) if i != len(x_bin_edges) - 1]
            
            # calculate Vmax for each galaxy in catalogue within z bin
            z_bin_cat.calc_Vmax(data_arr, z_bin, SED_fit_params_key, z_step, timed = timed)

            Ngals = np.zeros(len(x_bins))
            phi = np.zeros(len(x_bins))
            phi_errs = np.zeros(len(x_bins))
            cv_errs = np.zeros(len(x_bins))
            phi_errs_cv = np.zeros(len(x_bins))
            # loop through each mass bin in the given redshift bin
            for i, x_bin in enumerate(x_bins):
                # crop to galaxies in the x bin - not the bootstrapping method
                z_bin_x_bin_cat = z_bin_cat.crop(x_bin, x_name, SED_fit_params, phot_type)
                Ngals[i] = len(z_bin_x_bin_cat)
                # if there are galaxies in the z,x bin
                if int(Ngals[i]) != 0:
                    dx = x_bin[1] - x_bin[0]
                    # calculate Vmax's
                    V_max = np.zeros(int(Ngals[i])).astype(float)
                    for data in data_arr:
                        V_max += np.array([gal.V_max[z_bin_name][data.full_name] \
                            if gal.V_max[z_bin_name][data.full_name] != -1. else 0. for gal in z_bin_x_bin_cat])
                    V_max = np.array([_V_max for _V_max in V_max if _V_max != 0.])
                    if len(V_max) != Ngals[i]:
                        galfind_logger.warning(f"{Ngals[i] - len(V_max)} galaxies not detected")
                    phi[i] = np.sum(V_max ** -1.) / dx
                    # use standard Poisson errors if number of galaxies in bin is not small
                    if len(V_max) >= 4:
                        phi_errs[i] = np.sqrt(np.sum(V_max ** -2.)) / dx
                    else:
                        # using minimum is a minor cheat for symmetric errors?
                        phi_errs[i] = phi[i] * np.min(np.abs((np.array(funcs.poisson_interval(len(V_max), 0.32)) - len(V_max))) / len(V_max))
                    if type(cv_origin) == type(None):
                        pass
                    elif cv_origin == "Driver2010": # could open this up to more cosmic variance calculators
                        cv_errs[i] = funcs.calc_cv_proper(z_bin, data_arr = data_arr)
                    else:
                        raise NotImplementedError
            
            number_density_func = cls(x_name, x_bins, x_origin, z_bin, Ngals, phi, phi_errs, cv_errs, origin_surveys, cv_origin)
            
            if save and not Path(save_path).is_file():
                number_density_func.save()
            
            return number_density_func
        
        else: # load results
            return cls.from_save_path(save_path)

    @property
    def x_mid_bins(self):
        return np.array([(x_bin[1] + x_bin[0]) / 2. for x_bin in self.x_bins])

    @property
    def phi_errs_cv(self):
        return np.sqrt(self.phi_errs ** 2. + (self.cv_errs * self.phi) ** 2.)

    @staticmethod
    def get_origin_surveys(data_arr) -> str:
        return "+".join([data.full_name for data in data_arr])

    # cv_origin == "Driver2010"
    @staticmethod
    def get_save_path(origin_surveys: str, SED_fit_params_key: str, z_bin: Union[list, np.array], x_name: str, ext: str = ".ecsv") -> str:
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
        z_bin = np.array([float(split_save_path[-1].split("<")[0]), float(split_save_path[-1].split("<")[2])])
        return SED_fit_params_key, x_name, origin_surveys, z_bin
    
    def get_plot_path(self) -> str:
        plot_path = self.get_save_path(self.origin_surveys, self.x_origin.replace("_REST_PROPERTY", ""), \
            self.z_bin, self.x_name, ext = ".png").replace("/Data/", "/Plots/")
        funcs.make_dirs(plot_path)
        return plot_path

    def save(self, save_path: Union[str, None] = None) -> None:
        if type(save_path) == type(None):
            save_path = self.get_save_path(self.origin_surveys, self.x_origin.replace("_REST_PROPERTY", ""), self.z_bin, self.x_name)
        x_bins_low = np.array([x_bin[0] for x_bin in self.x_bins])
        x_bins_up = np.array([x_bin[1] for x_bin in self.x_bins])
        tab = Table({"x_bins_low": x_bins_low, "x_bins_up": x_bins_up, "Ngals": self.Ngals, "phi": self.phi, \
            "phi_errs": self.phi_errs, "cv_errs": self.cv_errs}, dtype = [float, float, int, float, float, float])
        tab.meta = {"x_origin": self.x_origin, "x_name": self.x_name, "origin_surveys": self.origin_surveys, \
            "z_bin": self.z_bin, "cv_origin": self.cv_origin}
        tab.write(save_path, overwrite = True)

    def plot(self, fig = None, ax = None, log: bool = False, annotate: bool = True, \
            save: bool = True, show: bool = False, plot_kwargs: dict = {}, \
            x_lims: Union[list, np.array, str, None] = "default", use_galfind_style: bool = True) -> None:
        
        if use_galfind_style: # (oppan gangnam style)
            plt.style.use(f"{config['DEFAULT']['GALFIND_DIR']}/galfind_style.mplstyle")

        if all(type(i) == type(None) for i in [fig, ax]):
            fig, ax = plt.subplots()
        
        # don't plot empty bins
        x_mid_bins = np.array([_x for _x, _y in zip(self.x_mid_bins, self.phi) if _y != 0.])
        phi = np.array([_y for _y in self.phi if _y != 0.])
        phi_errs_cv = np.array([_yerr for _yerr, _y in zip(self.phi_errs_cv, self.phi) if _y != 0.])
        if log:
            y = np.log10(phi)
            y_errs = np.array([np.log10(_phi / (_phi - _phi_err)) for _phi, _phi_err in zip(phi, phi_errs_cv)], \
                [np.log10(1. + (_phi_err / _phi)) for _phi, _phi_err in zip(phi, phi_errs_cv)])
        else:
            y = phi
            y_errs = phi_errs_cv

        # sort out plot_kwargs
        default_plot_kwargs = {"ls": "", "fmt": "o", "label": f"{self.author_year},{funcs.get_z_bin_name(self.z_bin)}"}
        # overwrite default with input for duplicate kwargs
        for key in plot_kwargs.keys():
            if key in default_plot_kwargs.keys():
                default_plot_kwargs.pop(key)
        plot_kwargs = {**plot_kwargs, **default_plot_kwargs}

        ax.errorbar(x_mid_bins, y, yerr = y_errs, **plot_kwargs)

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
        if save:
            plot_path = self.get_plot_path()
            plt.savefig(plot_path)
        if show:
            plt.show()

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
    def from_cat(cls, cat, x_name: str, x_bin_edges_arr: Union[list, np.array], \
            z_bins: Union[list, np.array], x_origin: Union[str, dict] = "EAZY_fsps_larson_zfree", \
            z_step: float = 0.1, use_vmax_simple: bool = False, timed: bool = False) -> "Number_Density_Function":

        # input assertions
        assert all(len(z_bin) == 2 for z_bin in z_bins)
        assert all(z_bin[0] < z_bin[1] for z_bin in z_bins)
        assert len(x_bin_edges_arr) == len(z_bins)
        assert all(len(x_bin_edges) >= 2 for x_bin_edges in x_bin_edges_arr)
        # ensure x_bin_edges are sorted from lower to higher x values in every z bin
        assert all(np.sort(np.array(x_bin_edges)) == np.array(x_bin_edges) for x_bin_edges in x_bin_edges_arr)
        # ensure x_bin_edges are evenly spaced?
        # extract x_name values from catalogue
        if type(x_origin) in [dict]:
            assert "code" in x_origin.keys()
            assert x_origin["code"].__class__.__name__ in [code.__name__ for code in SED_code.__subclasses__()]
            SED_fit_params = x_origin # redshifts must come from same SED fitting as x values
        elif type(x_origin) in [str]:
            # convert to SED_fit_params
            SED_fit_params = x_origin.split("_")[0]
        else:
            galfind_logger.critical(f"{x_origin=} with {type(x_origin)=} not in [dict, str]!")

        x = getattr(cat, x_name, x_origin)

        # calculate mass function in each redshift bin
        for i, (z_bin, x_bin_edges) in enumerate(zip(z_bins, x_bin_edges_arr)):
            # create x_bins from x_bin_edges (must include start and end values here too)
            x_bins = [[x_bin_edges[i], x_bin_edges[i + 1]] \
                for i in range(len(x_bin_edges) - 1) if i != len(x_bin_edges) - 1]
            # extract z_bin_name
            assert type(x_origin) == str # NOT GENERAL!
            z_bin_name = f"{x_origin}_{z_bin[0]:.1f}<z<{z_bin[1]:.1f}"
            # calculate Vmax for each galaxy in catalogue within z bin
            # in general call Vmax_multifield
            cat.calc_Vmax(cat.data, z_bin, x_origin, z_step, timed = timed)
            # crop catalogue to this redshift bin
            z_bin_cat = cat.crop(z_bin, "z", x_origin)

            Ngals = np.zeros(len(x_bins))
            phi = np.zeros(len(x_bins))
            phi_errs = np.zeros(len(x_bins))
            cv_errs = np.zeros(len(x_bins))
            phi_errs_cv = np.zeros(len(x_bins))
            # loop through each mass bin in the given redshift bin
            for j, x_bin in enumerate(x_bins):
                
                # crop to galaxies in the x bin - not the bootstrapping method
                z_bin_x_bin_cat = z_bin_cat.crop(x_bin, x_name, SED_fit_params)

                Ngals[j] = len(z_bin_x_bin_cat)
                # if there are galaxies in the z, mx bin
                if Ngals[j] != 0:
                    dx = x_bin[1] - x_bin[0]
                    V_max = np.array([gal.V_max[z_bin_name][cat.data.full_name] for gal in cat])
                    phi[j] = (np.sum(V_max ** -1.) / dx).value
                    # use standard Poisson errors if number of galaxies in bin is not small
                    if len(V_max) >= 4:
                        phi_errs[j] = (np.sqrt(np.sum(V_max ** -2.)) / dx).value
                    else:
                        # using minimum is a minor cheat for symmetric errors?
                        phi_errs[j] = phi[j] * np.min(np.abs((np.array(funcs.poisson_interval(len(V_max), 0.32)) - len(V_max))) / len(V_max))
                    cv_errs[j] = funcs.calc_cv_proper(float(z_bin[0]), float(z_bin[1]), fields_used = fields_used)
                    phi_errs_cv[j] = np.sqrt(phi_errs[j] ** 2. + (cv_errs[j] * phi[j]) ** 2.)
                

    def __len__(self):
        return len(self.number_density_function_arr)

    def plot(self):
        pass
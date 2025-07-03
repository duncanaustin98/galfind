#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:55:57 2023

@author: austind
"""

from __future__ import annotations

import types
import itertools
import importlib
import h5py
import sys
import os
import glob
import astropy.constants as const
import shutil
from pathlib import Path
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from astropy.table import Table
from tqdm import tqdm
import logging
from typing import Union, Dict, Any, List, Tuple, Optional, NoReturn, TYPE_CHECKING
if TYPE_CHECKING:
    from . import Catalogue, PDF, Multiple_Filter
    from bagpipes.filters import filter_set
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import Redshift_PDF, SED_code, SED_fit_PDF, config, galfind_logger
from . import useful_funcs_austind as funcs
from .useful_funcs_austind import astropy_cosmo as cosmo
from .decorators import run_in_dir
from .SED import SED_obs
from .Filter import Filter

pipes_unit_dict = {
    "z": u.dimensionless_unscaled,
    "redshift": u.dimensionless_unscaled,
    "chisq": u.dimensionless_unscaled,
    "mass": u.dex(u.solMass),
    "dust": u.ABmag,
    "fwhm": u.Gyr,
    "metallicity": u.dimensionless_unscaled, # is this actually in units of solar metallicity?
    "mass_weighted_zmet": u.dimensionless_unscaled, # check!
    "tmax": u.Gyr,
    "logU": u.dimensionless_unscaled, # ???
    "fesc": u.dimensionless_unscaled,
    "sfr": u.solMass / u.yr,
    "sfr_10myr": u.solMass / u.yr,
    "ssfr": u.dex(u.yr**-1),
    "ssfr_10myr": u.dex(u.yr**-1),
    "nsfr": u.dimensionless_unscaled, # check what this is!
    "nsfr_10myr": u.dimensionless_unscaled, # check what this is!
    "mass_weighted_age": u.Gyr,
    "tform": u.Gyr,
    "tquench": u.Gyr,
    "xi_ion_caseB": u.Hz / u.erg,
    "ndot_ion_caseB": u.Hz,
    "UV_colour": u.ABmag,
    "VJ_colour": u.ABmag,
    "beta_C94": u.dimensionless_unscaled,
    "burstiness": u.dimensionless_unscaled,
    "m_UV": u.ABmag,
    "M_UV": u.ABmag,
    "flux": u.erg / u.s / u.cm**2 / u.AA, # check!
    "EWrest": u.AA,
}

def get_pipes_unit(label: str) -> u.Unit:
    if label in pipes_unit_dict.keys():
        return pipes_unit_dict[label]
    elif any(key in label for key in pipes_unit_dict.keys()):
        relevant_keys = [key for key in pipes_unit_dict.keys() if key in label]
        if len(relevant_keys) == 1:
            return pipes_unit_dict[relevant_keys[0]]
        else:
            galfind_logger.critical(
                f"Multiple {relevant_keys=} found for {label=}! Bagpipes unit_dict requires updating"
            )
    else:
        galfind_logger.debug(
            f"No keys found in {label=} for Bagpipes unit_dict! Returning dimensionless"
        )
        return u.dimensionless_unscaled


class Bagpipes(SED_code):

    def __init__(
        self: Self,
        SED_fit_params: Dict[str, Any],
        custom_label: Optional[str] = None,
        sampler: str = "multinest",
    ) -> Self:
        if custom_label is not None:
            self.custom_label = custom_label
        self.sampler = sampler
        # start off empty
        self.gal_property_labels = {}
        self.gal_property_err_labels = {}
        super().__init__(SED_fit_params)

    @classmethod
    def from_label(cls, label: str) -> Type[SED_code]:
        # TODO: For continuity SFH, add z used to calculate bins 
        # into the SED fit params from_label extractor
        breakpoint()
        SED_fit_params = {}
        SED_fit_params["sfh"] = label.split("_sfh_")[1].split("_dust_")[0]
        dust_label = label.split("_dust_")[1].split("_Z_")[0]
        if "log_10" in dust_label:
            assert dust_label[-6:] == "log_10"
            SED_fit_params["dust_prior"] = "log_10"
            SED_fit_params["dust"] = dust_label[:-7]
        elif "uniform" in dust_label:
            assert dust_label[-7:] == "uniform"
            SED_fit_params["dust_prior"] = "uniform"
            SED_fit_params["dust"] = dust_label[:-8]
        else:
            galfind_logger.critical(
                f"Invalid dust prior from {dust_label=}! Must be in ['log_10', 'uniform']"
            )
            breakpoint()
        split_metallicity_label = label.split("_Z_")[1].split("_")
        if (
            split_metallicity_label[0] == "log"
            and split_metallicity_label[1] == "10"
        ):
            SED_fit_params["metallicity_prior"] = "log_10"
        elif split_metallicity_label[0] == "uniform":
            SED_fit_params["metallicity_prior"] = "uniform"
        else:
            galfind_logger.critical(
                f"Invalid metallicity prior from {split_metallicity_label=}! Must be in ['log_10', 'uniform']"
            )
            breakpoint()
        # easier if BC03 read properly
        if "BPASS" in label:
            SED_fit_params["sps_model"] = "BPASS"
            redshift_label = label.split(SED_fit_params["sps_model"])[1][1:]
        else:
            SED_fit_params["sps_model"] = "BC03"
            redshift_label = label.split(SED_fit_params["metallicity_prior"])[-1][1:]
        if redshift_label == "zfix":
            SED_fit_params["fix_z"] = True
        else:
            split_zlabel = redshift_label.split("_z_")
            SED_fit_params["z_range"] = (
                float(split_zlabel[0]),
                float(split_zlabel[1]),
            )
            SED_fit_params["fix_z"] = False
        return cls(SED_fit_params)

    @property
    def ID_label(self) -> str:
        return "#ID"

    @property
    def label(self) -> str:
        # should be generalized more here including e.g. SED_fit_params assertions
        if hasattr(self, "custom_label"):
            return self.custom_label
        else:
            # sort redshift label
            if self.SED_fit_params["fix_z"]:
                redshift_label = "zfix"
            else:
                if "z_sigma" in self.SED_fit_params.keys():
                    assert isinstance(self.SED_fit_params["z_sigma"], float), \
                        galfind_logger.critical(
                            f"{type(self.SED_fit_params['z_sigma'])=}!=int!"
                        )
                    redshift_label = f"zgauss_{self.SED_fit_params['z_sigma']:.1f}sig"
                else:
                    redshift_label = "zfree"
                # if "z_range" in self.SED_fit_params.keys():
                #     assert len(self.SED_fit_params["z_range"]) == 2
                #     redshift_label = f"{int(self.SED_fit_params['z_range'][0])}z{int(self.SED_fit_params['z_range'][1])}"
                # else:
                #     galfind_logger.critical(
                #         f"Bagpipes {self.SED_fit_params=} must include either " + \
                #         "'z_range' if 'fix_z' == False or not included!"
                #     )
                #     breakpoint()
            # sort SPS label
            if self.SED_fit_params["sps_model"].upper() in ["BC03", "BPASS"]:
                sps_label = self.SED_fit_params["sps_model"].upper()
            else:
                galfind_logger.critical(
                    f"Bagpipes {self.SED_fit_params=} must include " + \
                    "'sps_model' with .upper() in ['BC03', 'BPASS']"
                )
                breakpoint()
            # sfh label
            if "continuity" in self.SED_fit_params["sfh"]:
                assert "z_calculator" in self.SED_fit_params.keys(), \
                    galfind_logger.critical(
                        f"Bagpipes {self.SED_fit_params=} must include " + \
                        "'z_calculator' for 'continuity' SFH"
                    )
                from . import Redshift_Extractor
                if isinstance(self.SED_fit_params["z_calculator"], Redshift_Extractor):
                    z_label = f"z{self.SED_fit_params['z_calculator'].SED_fit_label.replace('_', '').replace('zfree', '')}"
                else:
                    z_label = f"z{self.SED_fit_params['z_calculator']:0.1f}"
                sfh_label = f"{self.SED_fit_params['sfh']}_{z_label}"
                if "fixed_bin_ages" in self.SED_fit_params.keys():
                    # ensure fixed_bin_ages is a Quantity
                    assert isinstance(self.SED_fit_params["fixed_bin_ages"], u.Quantity), \
                        galfind_logger.critical(
                            "fixed_bin_ages must be a Quantity!"
                        )
                    # ensure fixed_bin_ages has dimensions of time
                    assert self.SED_fit_params["fixed_bin_ages"].unit.is_equivalent(u.Myr), \
                        galfind_logger.critical(
                            "fixed_bin_ages must be in Myr!"
                        )
                    if not (len(self.SED_fit_params["fixed_bin_ages"]) == 1 and \
                            self.SED_fit_params["fixed_bin_ages"].to(u.Myr).value[0] == 10.0):
                        sfh_label += "_"
                        for i, age in enumerate(self.SED_fit_params["fixed_bin_ages"].to(u.Myr).value):
                            if i != 0:
                                sfh_label += ","
                            sfh_label += f"{age:.1f}"
                        sfh_label += "Myr"
                if "num_bins" in self.SED_fit_params.keys():
                    # ensure num_bins is an integer
                    assert isinstance(self.SED_fit_params["num_bins"], int), \
                        galfind_logger.critical(
                            "num_bins must be an integer!"
                        )
                    sfh_label += f"_{self.SED_fit_params['num_bins']}bins"
                sfh_label = sfh_label.replace("continuity", "cont")
            else:
                sfh_label = self.SED_fit_params["sfh"]
            # '_dust' label
            return (
                f"Bagpipes_sfh_{sfh_label}_{self.SED_fit_params['dust']}_"
                + f"{self.SED_fit_params['dust_prior']}_Z_{self.SED_fit_params['metallicity_prior']}"
                + f"_{sps_label}_{redshift_label}{self.excl_bands_label}"
            )

    @property
    def hdu_name(self) -> str:
        # TODO: Copied from EAZY
        #return f"{self.__class__.__name__}_{self.SED_fit_params['templates']}"
        return self.label

    @property
    def tab_suffix(self) -> str:
        # TODO: Copied from EAZY
        # return f"{self.SED_fit_params['templates']}_" + \
        #     f"{funcs.lowz_label(self.SED_fit_params['lowz_zmax'])}"
        return self.label.replace("Bagpipes_", "")

    @property
    def required_SED_fit_params(self) -> List[str]:
        return ["sps_model", "IMF", "fix_z"]

    @property
    def are_errs_percentiles(self) -> bool:
        return True

    @property
    def display_label(self) -> str:
        sps = self.SED_fit_params["sps_model"]
        sfh = self.SED_fit_params["sfh"].split("_")[0] #replace("_", " ")#.capitalize()
        if "continuity" in sfh and "fixed_bin_ages" in self.SED_fit_params.keys():
            fixed_bin_ages = self.SED_fit_params["fixed_bin_ages"].to(u.Myr).value
            sfh += f" ({', '.join([f'{age:.1f}' for age in fixed_bin_ages])} Myr)"
        return f"{sps} {sfh}"
    
    def _assert_SED_fit_params(self) -> NoReturn:
        # add defaults required whether fit_instructions are included or not
        for name, default in zip(["sps_model", "IMF", "fix_z"], ["BC03", "", False]):
            if name not in self.SED_fit_params.keys():
                self.SED_fit_params[name] = default
        if "fit_instructions" in self.SED_fit_params.keys():
            assert isinstance(self.SED_fit_params["fit_instructions"], dict)
            assert hasattr(self, "custom_label")
        else:
            defaults_dict = {
                "sfh": "lognorm",
                "age_prior": "log_10",
                "metallicity_prior": "log_10",
                "dust": "Calzetti",
                "dust_prior": "log_10",
                "dust_eta": 1.0,
                "t_bc": 10 * u.Myr,
                "logU": (-4.0, -1.0), 
                "logU_prior": "uniform",
                "fesc": None, #(1.e-4, 1.0),
                "fesc_prior": "log_10"
            }
            for name, default in defaults_dict.items():
                if name not in self.SED_fit_params.keys():
                    self.SED_fit_params[name] = default
        super()._assert_SED_fit_params()

    def reload(self: Self) -> types.ModuleType:
        # re-load to take into account sps model
        if self.SED_fit_params["sps_model"] == "BPASS":
            # set environment variable to use BPASS
            os.environ["use_bpass"] = "1"
        else:
            # set environment variable to use BC03
            os.environ["use_bpass"] = "0"
        sys.modules.pop("bagpipes", None)
        import bagpipes
        return bagpipes

    def __call__(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity,
        save_PDFs: bool = True,
        save_SEDs: bool = True,
        load_PDFs: bool = True,
        load_SEDs: bool = True,
        timed: bool = True,
        overwrite: bool = False,
        update: bool = False,
        **fit_kwargs
    ):
        try:
            from mpi4py import MPI
            self.rank = MPI.COMM_WORLD.Get_rank()
            self.size = MPI.COMM_WORLD.Get_size()
            from mpi4py.futures import MPIPoolExecutor
        except ImportError:
            self.rank = 0
            self.size = 1
        
        self._load_fit_instructions()
        if "continuity" in self.SED_fit_params["sfh"]:
            self._update_continuity_sfh_fit_instructions(cat)
        if not self.SED_fit_params["fix_z"] and "z_sigma" in self.SED_fit_params.keys():
            self._update_redshift_fit_instructions(cat)
        return super().__call__(
            cat,
            aper_diam,
            save_PDFs,
            save_SEDs,
            load_PDFs,
            load_SEDs,
            timed,
            overwrite,
            update,
            **fit_kwargs
        )

    def _load_gal_property_labels(self):
        return super()._load_gal_property_labels(self.gal_property_labels)

    def _load_gal_property_err_labels(self):
        return super()._load_gal_property_err_labels(self.gal_property_err_labels)

    def _load_gal_property_units(self) -> NoReturn:
        # TODO: Copied from EAZY
        self.gal_property_units = {}

    def extract_priors(
        self: Self,
        filterset: Multiple_Filter,
        redshift: float,
        n_draws: int = 10_000,
        plot: bool = True,
    ) -> str:
        save_path = f"{config['Bagpipes']['PIPES_OUT_DIR']}/priors/{self.label}_z{redshift:.1f}.h5"
        funcs.make_dirs(save_path)
        if not Path(save_path).is_file():
            self.reload()
            if not hasattr(self, "fit_instructions"):
                self._load_fit_instructions()
                if "continuity" in self.SED_fit_params["sfh"]:
                    self._update_continuity_sfh_fit_instructions(cat = None)
                    self.fit_instructions = self.fit_instructions[0]
            if "redshift" in self.fit_instructions.keys():
                if isinstance(self.fit_instructions["redshift"], float):
                    fit_instructions = self.fit_instructions
                else:
                    fit_instructions = self.fit_instructions
                    assert redshift < fit_instructions["redshift"][1] and redshift > fit_instructions["redshift"][0]
                    fit_instructions["redshift"] = redshift
            else:
                fit_instructions = self.fit_instructions
                fit_instructions["redshift"] = redshift
            filt_list = [self._get_filt_path(filt) for filt in filterset]
            galfind_logger.info(f"Extracting priors for {self.label} at z={redshift:.1f}")
            bagpipes = self.reload()
            priors = bagpipes.fitting.check_priors(fit_instructions, filt_list = filt_list, n_draws = n_draws)
            hf = h5py.File(save_path, "w")
            for key, vals in priors.samples.items():
                hf.create_dataset(key, data=np.array(vals).flatten())
            hf.close()
            if plot:
                for i, (key, vals) in enumerate(priors.samples.items()):
                    try:
                        fig, ax = plt.subplots()
                        ax.hist(vals, bins = int(n_draws / 100), histtype = "step", label = key)
                        ax.set_xlabel(key)
                        out_path = f"{save_path.replace('.h5', f'/{key}.png')}"
                        funcs.make_dirs(out_path)
                        fig.savefig(out_path)
                        fig.clf()
                    except:
                        pass
        return save_path

    def make_in(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity,
        overwrite: bool = False
    ) -> str:
        # no need for bagpipes input catalogue
        pass

    def _temp_out_subdir(
        self: Self,
        cat: Catalogue
    ) -> str:
        # /{self.label}
        temp_subdir = f"{cat.version}/{cat.survey}/{cat.filterset.instrument_name}/temp"
        while len(glob.glob(f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/posterior/{temp_subdir}/*")) > 0:
            temp_subdir += "_"
        return temp_subdir
    
    def _new_subdir(
        self: Self,
        cat: Catalogue
    ) -> str:
        return f"{cat.version}/{cat.survey}/{cat.filterset.instrument_name}/{self.label}"

    def _move_files(
        self,
        cat: Catalogue,
        direction = "from_temp"
    ) -> NoReturn:
        assert direction in ["from_temp", "to_temp"]
        temp_out_subdir = self._temp_out_subdir(cat)
        temp_post_dir = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/posterior/{temp_out_subdir}"
        temp_plots_dir = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/plots/{temp_out_subdir}"
        temp_sed_dir = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/seds/{temp_out_subdir}"
        temp_sfr_dir = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/sfr/{temp_out_subdir}"
        new_subdir = self._new_subdir(cat)
        new_post_dir = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/posterior/{new_subdir}"
        new_plots_dir = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/plots/{new_subdir}"
        new_sed_dir = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/seds/{new_subdir}"
        new_sfr_dir = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/sfr/{new_subdir}"
        if direction == "from_temp":
            # move files from temp directory to main directory
            from_post_dir = temp_post_dir
            to_post_dir = new_post_dir
            from_plots_dir = temp_plots_dir
            to_plots_dir = new_plots_dir
            from_sed_dir = temp_sed_dir
            to_sed_dir = new_sed_dir
            from_sfr_dir = temp_sfr_dir
            to_sfr_dir = new_sfr_dir
        elif direction == "to_temp":
            # move files from main directory to temp directory
            from_post_dir = new_post_dir
            to_post_dir = temp_post_dir
            from_plots_dir = new_plots_dir
            to_plots_dir = temp_plots_dir
            from_sed_dir = new_sed_dir
            to_sed_dir = temp_sed_dir
            from_sfr_dir = new_sfr_dir
            to_sfr_dir = temp_sfr_dir
        funcs.make_dirs(f"{to_post_dir}/")
        funcs.make_dirs(f"{to_plots_dir}/")
        funcs.make_dirs(f"{to_sed_dir}/")
        funcs.make_dirs(f"{to_sfr_dir}/")
        for from_dir, to_dir in zip(
            [from_post_dir, from_plots_dir, from_sed_dir, from_sfr_dir],
            [to_post_dir, to_plots_dir, to_sed_dir, to_sfr_dir]
        ):
            #breakpoint()
            for path in glob.glob(f"{from_dir}/*"):
                if not Path(f"{to_dir}/{path.split('/')[-1]}").is_file():
                    os.rename(path, f"{to_dir}/{path.split('/')[-1]}")
                    galfind_logger.info(
                        f"Moved {path.split('/')[-1]} to {to_dir}"
                    )
                else:
                    galfind_logger.info(
                        f"{path.split('/')[-1]} already exists in {to_dir}, skipping!"
                    )
        self._move_fits_cat(cat, direction = direction)

    def _move_fits_cat(
        self: Self,
        cat: Catalogue,
        direction: str = "from_temp"
    ) -> NoReturn:
        assert direction in ["from_temp", "to_temp"]
        fits_dir = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/cats"
        funcs.make_dirs(f"{fits_dir}/")
        temp_out_subdir = self._temp_out_subdir(cat)
        new_subdir = self._new_subdir(cat)
        if direction == "from_temp":
            # move files from temp directory to main directory
            from_fits_path = f"{fits_dir}/{temp_out_subdir}.fits"
            to_fits_path = f"{fits_dir}/{new_subdir}.fits"
        elif direction == "to_temp":
            # move files from main directory to temp directory
            from_fits_path = f"{fits_dir}/{new_subdir}.fits"
            to_fits_path = f"{fits_dir}/{temp_out_subdir}.fits"
        # Move fits catalogue
        if Path(from_fits_path).is_file() and not Path(to_fits_path).is_file():
            os.rename(from_fits_path, to_fits_path)
            galfind_logger.info(
                f"Moved {from_fits_path} to {to_fits_path}"
            )
        

    @run_in_dir(path=config["Bagpipes"]["PIPES_OUT_DIR"])
    def fit(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity,
        save_SEDs: bool = True,
        save_PDFs: bool = True,
        overwrite: bool = False,
        **kwargs: Dict[str, Any],
    ) -> NoReturn:
        # determine temp directories
        out_subdir = self._temp_out_subdir(cat)
        path_post = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/posterior/{out_subdir}"
        path_plots = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/plots/{out_subdir}"
        path_sed = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/seds/{out_subdir}"
        path_sfr = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/sfr/{out_subdir}"
        path_fits = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/cats/{out_subdir}"
        new_subdir = self._new_subdir(cat)
        new_path_post = path_post.replace(out_subdir, new_subdir)
        funcs.make_dirs(new_path_post)
        new_path_plots = path_plots.replace(out_subdir, new_subdir)
        funcs.make_dirs(new_path_plots)
        new_path_sed = path_sed.replace(out_subdir, new_subdir)
        funcs.make_dirs(new_path_sed)
        new_path_sfr = path_sfr.replace(out_subdir, new_subdir)
        funcs.make_dirs(new_path_sfr)
        new_path_fits = path_fits.replace(out_subdir, f"{new_subdir}.fits")
        funcs.make_dirs(new_path_fits)
        
        if self.rank == 0:
            # if overwrite:
            #     shutil.rmtree(path_post, ignore_errors = True)
            # make appropriate temp directories
            for path in [path_post, path_plots, path_sed, path_fits]:
                funcs.make_dirs(path)
                funcs.change_file_permissions(path)
        phot_tab = cat.open_cat()
        os.environ["total_to_fit"] = str(len(phot_tab))
        os.environ["num_loaded"] = "0"

        if "rerun" in kwargs.keys():
            rerun = kwargs["rerun"]
        else:
            rerun = False
        # only run for galaxies that haven't been run yet
        #breakpoint()
        #rerun = True
        if rerun:
            to_run_arr = np.ones(len(cat), dtype=bool)
            # move stuff back to temp directory
            self._move_files(cat, direction = "to_temp")
        else:
            to_run_arr = np.ones(len(cat), dtype=bool)
            for i, gal in enumerate(cat):
                save_path = f"{new_path_post}/{gal.ID}.h5"
                # if Path(save_path).is_file() and not overwrite:
                #     to_run_arr[i] = False
            if all(not to_run for to_run in to_run_arr):
                galfind_logger.info("All objects run and not rerun/overwrite.")
                breakpoint()
                return
        self._move_fits_cat(cat, direction = "to_temp")

        run_cat = deepcopy(cat)
        run_cat.gals = run_cat[to_run_arr]
        # remove filters without a depth measurement
        if self.SED_fit_params["excl_bands"] == []:
            excl_bands_arr = np.array([[] for _ in range(len(run_cat.gals))])
        elif isinstance(self.SED_fit_params["excl_bands"][0], list):
            excl_bands_arr = self.SED_fit_params["excl_bands"]
        else:
            excl_bands_arr = np.full(len(run_cat.gals), self.SED_fit_params["excl_bands"])
        assert len(excl_bands_arr) == len(run_cat.gals), \
            galfind_logger.critical(
                f"Bagpipes {excl_bands_arr=} must be a (ragged) list of lists with length {len(run_cat.gals)}!"
            )

        gals_arr = []
        for gal, excl_bands in tqdm(zip(run_cat.gals, excl_bands_arr), "Removing filters without depth measurements", disable = galfind_logger.getEffectiveLevel() > logging.INFO):
            remove_filt = []
            for i, (depth, filt) in enumerate(zip(gal.aper_phot[aper_diam].depths, gal.aper_phot[aper_diam].filterset)):
                if np.isnan(depth) or filt.band_name in excl_bands:
                    remove_filt.extend([filt])
            for filt in remove_filt:
                gal.aper_phot[aper_diam] -= filt
                galfind_logger.warning(
                    f"Removed {filt.band_name} from {gal.ID} for bagpipes fitting."
                )
            gals_arr.extend([gal])

        run_cat.gals = gals_arr
        IDs = [gal.ID for gal in gals_arr]
        filters = self._load_filters(run_cat, aper_diam)

        # if fix_z is not False
        if not (not self.SED_fit_params["fix_z"]):
            if isinstance(self.SED_fit_params["fix_z"], str):
                redshifts = np.array([getattr(gal, self.SED_fit_params["fix_z"]) for gal in gals_arr]).astype(float)
            else:
                redshifts = np.array([gal.aper_phot[aper_diam].SED_results \
                    [self.SED_fit_params["fix_z"]].z for gal in gals_arr]).astype(float)
        else:
            redshifts = None

        # # if use_redshift_sigma:
        # #     if set_redshift_sigma is None:
        # #         redshift_err_low = np.array(np.ravel(catalog[f"{zcol_name}{photoz_template}{extra_append}"])-np.ravel(catalog[f'{zcol_low_name}{photoz_template}{extra_append}']))
        # #         redshift_err_high = np.array(np.ravel(catalog[f'{zcol_up_name}{photoz_template}{extra_append}'])-np.ravel(catalog[f"{zcol_name}{photoz_template}{extra_append}"]))
        # #         redshift_sigma = np.mean([redshift_err_low, redshift_err_high], axis=0)
        # #         redshift_sigma[redshift_sigma < 0] = 3 # Replaces values where redshift_sigma is negative with 3 (have seen -99s before)
        # #     elif len(set_redshift_sigma) == len(ids):
        # #         redshift_sigma = np.array(set_redshift_sigma)

        # # if fix_redshifts:
        # #     np.save(path, np.vstack((ids,redshifts)))
        # # else:
        # #     np.save(path, np.vstack((ids,np.ones(len(ids))*-1)))
        
        # #	use_redshift_sigma = False
        # #	fix_redshifts = False
        # #	ids = [set_ID]
        # # Log fit parameters to json
        # warnings.simplefilter(action='ignore', category=FutureWarning) 

        # if overwrite:
        #     print(f'Removing {out_subdir}')
        #     shutil.rmtree('pipes/posterior/'+out_subdir, ignore_errors=True)
        #     shutil.rmtree('pipes/seds/'+out_subdir, ignore_errors=True)
        #     shutil.rmtree('pipes/cats/'+out_subdir, ignore_errors=True)
        #     shutil.rmtree('pipes/plots/'+out_subdir, ignore_errors=True)

        
        # if sfh in ['continuity', 'continuity_bursty']:
        #     fit_instructions_list = []
        #     for z in redshifts:
        #         fit_instructions_i = deepcopy(fit_instructions)
        #         fit_instructions_i['continuity']['bin_edges'] = list(calculate_bins(redshift = z, num_bins=cont_nbins, first_bin=first_bin, second_bin=second_bin, return_flat=True, output_unit='Myr', log_time=False))
        #         fit_instructions_list.append(fit_instructions_i)
        #     fit_instructions = fit_instructions_list
        #     print('Continuity model detected. Setting custom fit_instructions list for each galaxy.')
        
        # # if self.rank == 0:
        # #     fit_instructions_write = deepcopy(fit_instructions)
        # #     # Convert numpy arrays to lists for json
        # #     if type(fit_instructions_write) == list:
        # #         fit_instructions_write = fit_instructions_write[0]
        # #     for key in fit_instructions_write.keys():
        # #         if type(fit_instructions_write[key]) == np.ndarray:
        # #             fit_instructions_write[key] = fit_instructions_write[key].tolist()
        # #         elif type(fit_instructions_write[key]) == dict:
        # #             for subkey in fit_instructions_write[key].keys():
        # #                 if type(fit_instructions_write[key][subkey]) == np.ndarray:
        # #                     fit_instructions_write[key][subkey] = fit_instructions_write[key][subkey].tolist()

        # #     json_file = json.dumps(fit_instructions_write)
        # #     f = open(f'{path_overall}/posterior/{out_subdir}/config.json',"w")
        # #     f.write(json_file)
        # #     f.close()

        if all(hasattr(gal, "aper_phot") for gal in gals_arr):
            photometry_exists = True
            load_func = self._load_pipes_phot
        # TODO: spectroscopic fitting
        spectrum_exists = False
        if "plot" in kwargs.keys():
            plot = kwargs["plot"]
        else:
            plot = True
        bagpipes = self.reload()
        fit_cat = bagpipes.fit_catalogue(
            IDs,
            self.fit_instructions,
            load_func,
            spectrum_exists = spectrum_exists,
            photometry_exists = photometry_exists,
            run = out_subdir,
            make_plots = False, #plot,
            cat_filt_list = filters,
            redshifts = redshifts, 
            redshift_sigma = None, #redshift_sigma if use_redshift_sigma else None, 
            analysis_function = None, #custom_plotting if plot else None, 
            vary_filt_list = True,
            full_catalogue = True,
            save_pdf_txts = save_PDFs,
            n_posterior = 500,
            #time_calls = time_calls
            load_data_kwargs = {"cat": run_cat, "aper_diam": aper_diam}
        )
        #breakpoint()
        #galfind_logger.info(f"Fitting bagpipes with {self.fit_instructions=}")
        try:
            run_parallel = False
            fit_cat.fit(
                verbose = False,
                mpi_serial = run_parallel,
                sampler = self.sampler,
                overwrite_h5 = True, #overwrite,
            )
        except Exception as e:
            raise e
        # rename files and move to appropriate directories

        if self.rank == 0:
            galfind_logger.info(f"Renaming and moving {self.label} output files on rank 0.")
            self._move_files(cat, direction = "from_temp")

    def make_fits_from_out(
        self: Self, 
        out_path: str,
        overwrite: bool = True
    ) -> NoReturn:
        # update properties from bagpipes output table column names
        tab = Table.read(out_path)
        self._update_gal_properties(tab.colnames)
        pass
        # fits_out_path = self.get_galfind_fits_path(out_path)
        # breakpoint()
        # if not Path(fits_out_path).is_file() or overwrite:
        #     tab = Table.read(out_path)
        #     breakpoint()
        #     self._update_gal_proeprties(tab.colnames)
        #     tab[self.ID_label] = np.array(
        #         [id.split("_")[0] for id in tab["#ID"]]
        #     ).astype(int)
        #     tab.remove_column("#ID")
        #     if "input_redshift" in tab.colnames:
        #         if all(z == 0.0 for z in tab["input_redshift"]):
        #             tab.remove_column("input_redshift")
        #     for name in tab.colnames:
        #         if name != self.ID_label:
        #             tab.rename_column(
        #                 name, self.galaxy_property_labels(name, self.SED_fit_params)
        #             )
        #     tab.write(fits_out_path, overwrite=True)

    def _update_gal_properties(self, labels: List[str]) -> NoReturn:
        ignore_labels = ["#ID"]
        labels = [label for label in labels if label not in ignore_labels]
        gal_property_labels = {label: f"{label}_50" for label in list(np.unique(["_".join(label.split("_")[:-1]) for label in labels if label.split("_")[-1] == "50"]))}
        symmetric_err_labels = {label: [f"{label}_err", f"{label}_err"] for label in labels if f"{label}_err" in labels}
        self.gal_property_err_labels = {**{label: [f"{label}_16", f"{label}_84"] for label in gal_property_labels.keys()}, **symmetric_err_labels}
        non_PDF_property_labels = {label: label for label in labels if label.split("_")[-1] not in ["err", "16", "50", "84"]}
        self.gal_property_labels = {**gal_property_labels, **{label: label for label in non_PDF_property_labels.keys()}}
        for key, val in self.gal_property_labels.items():
            if "redshift" in key:
                self.gal_property_labels["z"] = val
                del self.gal_property_labels[key]
                if key in self.gal_property_err_labels.keys():
                    self.gal_property_err_labels["z"] = self.gal_property_err_labels[key]
                    del self.gal_property_err_labels[key]
                break
        for key, val in self.gal_property_labels.items():
            if "chisq_phot" in key:
                self.gal_property_labels["chi_sq"] = val
                del self.gal_property_labels[key]
                break
        self.gal_property_units = {label: get_pipes_unit(label) for label in self.gal_property_labels.keys()}

    @staticmethod
    def _get_filt_path(filt: Filter) -> str:
        return f"{config['Bagpipes']['PIPES_FILT_DIR']}/" + \
            f"{filt.instrument_name}/{filt.band_name}.txt"

    def _generate_filters(
        self: Self,
        filterset: Multiple_Filter,
    ) -> NoReturn:
        for i, filt in enumerate(filterset):
            filt_path = self._get_filt_path(filt)
            if not Path(filt_path).is_file():
                funcs.make_dirs(filt_path)
                wavs = filt.wav.to(u.AA).value
                trans = filt.trans
                np.savetxt(filt_path, np.array([wavs, trans]).T, header = filt.band_name)
                galfind_logger.info(
                    f"Generated Bagpipes input filter for {filt.band_name}"
                )

    def _load_filters(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity,
    ) -> List[str]:
        self._generate_filters(cat.filterset)
        cat_filt_paths = np.zeros(len(cat), dtype=object)

        if self.SED_fit_params["excl_bands"] == []:
            excl_bands_arr = np.array([[] for _ in range(len(cat.gals))])
        elif isinstance(self.SED_fit_params["excl_bands"][0], list):
            excl_bands_arr = self.SED_fit_params["excl_bands"]
        else:
            excl_bands_arr = np.full(len(cat.gals), self.SED_fit_params["excl_bands"])
        assert len(excl_bands_arr) == len(cat.gals), \
            galfind_logger.critical(
                f"Bagpipes {excl_bands_arr=} must be a (ragged) list of lists with length {len(cat.gals)}!"
            )
        
        for i, (gal, excl_bands) in enumerate(zip(cat, excl_bands_arr)):
            gal_filt_paths = []
            for filt in gal.aper_phot[aper_diam].filterset:
                if filt.band_name not in excl_bands:
                    gal_filt_paths.extend([self._get_filt_path(filt)])
            cat_filt_paths[i] = gal_filt_paths
        return list(cat_filt_paths)
    
    def _get_out_paths(
        self: Self, 
        cat: Catalogue, 
        aper_diam: u.Quantity
    ) -> Tuple[str, str, str, Dict[str, List[str]], List[str]]:
    # @staticmethod
    # def get_out_paths(
    #     cat,
    #     SED_fit_params,
    #     IDs,
    #     load_properties=[
    #         "stellar_mass",
    #         "formed_mass",
    #         "dust:Av",
    #         "beta_C94",
    #         "m_UV",
    #         "M_UV",
    #         "sfr",
    #         "sfr_10myr",
    #     ],
    # ):  # , "Halpha_EWrest", "xi_ion_caseB"
        #pipes_name = Bagpipes.label_from_SED_fit_params(self.SED_fit_params)
        in_path = None
        out_path = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/cats/{cat.version}/" + \
            f"{cat.survey}/{cat.filterset.instrument_name}/{self.label}.fits"
        fits_out_path = Bagpipes.get_galfind_fits_path(out_path)
        h5_dir = out_path.replace(".fits", "").replace("cats", "posterior")
        h5_paths = [f"{h5_dir}/{ID}.h5" for ID in cat.ID]
        # PDF_dir = out_path.replace(".fits", "").replace("cats", "posterior")
        # SED_dir = out_path.replace(".fits", "").replace("cats", "posterior")
        # open h5
        # load_properties = []
        # PDF_paths = {
        #     gal_property if "redshift" not in gal_property else "z": [
        #         f"{h5_dir}/{gal_property}/{str(int(ID))}_{cat.survey}.txt"
        #         if Path(
        #             f"{h5_dir}/{gal_property}/{str(int(ID))}_{cat.survey}.txt"
        #         ).is_file()
        #         else None
        #         for ID in cat.ID
        #     ]
        #     for gal_property in load_properties
        # }
        # determine SED paths
        #SED_paths = []
        # SED_paths = [
        #     f"{h5_dir}/{str(int(ID))}_{cat.survey}.dat"
        #     if Path(f"{h5_dir}/{str(int(ID))}_{cat.survey}.dat").is_file()
        #     else None
        #     for ID in cat.ID
        # ]
        return in_path, out_path, fits_out_path, h5_paths, h5_paths #PDF_paths, SED_paths

    def extract_SEDs(
        self: Self,
        IDs: List[int], 
        SED_paths: Union[str, List[str]],
        *args,
        **kwargs,
    ) -> List[SED_obs]:
        # ensure this works if only extracting 1 galaxy
        if isinstance(IDs, (str, int, float)):
            IDs = np.array([int(IDs)])
        if isinstance(SED_paths, str):
            SED_paths = [SED_paths]
        assert len(IDs) == len(SED_paths), galfind_logger.critical(
            f"len(IDs) = {len(IDs)} != len(data_paths) = {len(SED_paths)}!"
        )
        assert all(name in kwargs.keys() for name in ["cat", "aper_diam"]), \
            galfind_logger.critical(
                f"Bagpipes.extract_SEDs() with {kwargs.keys()=} requires 'cat' and 'aper_diam'!"
            )
        cat = kwargs["cat"]
        aper_diam = kwargs["aper_diam"]
        
        # extract redshifts
        if self.SED_fit_params["fix_z"]:
            z_arr = self.SED_fit_params["z_calculator"](cat)
        else:
            raise NotImplementedError
        # extract observed frame wavelengths
        rest_wavs = self._extract_SED_wavelengths(cat, aper_diam)
        assert len(rest_wavs) == len(z_arr), \
            galfind_logger.critical(f"{len(rest_wavs)=}!={len(z_arr)=}")
        wavs = [wavs * (1. + z) * u.um for wavs, z in zip(rest_wavs, z_arr)]
        # if isinstance(rest_wavs, np.ndarray):
        #     wavs = [wavs_ * u.um for wavs_ in rest_wavs * (1. + z_arr[:, np.newaxis])]
        # else:
        #     wavs = 
        #wavs = self._extract_SED_wavelengths(cat, aper_diam) * (1. + z_arr[:, np.newaxis]) * u.um
        # extract in erg/s/cm^2/AA
        if "spectrum_type" in kwargs.keys():
            spectrum_type = kwargs["spectrum_type"]
        else:
            spectrum_type = "spectrum_full"
        flambda = [self._extract_SED_fluxes(SED_path, spectrum_type = spectrum_type) * u.erg / u.s / u.cm ** 2 / u.AA for SED_path in SED_paths]
        # convert fluxes to uJy
        fnu = [(flambda_ * wavs_ ** 2 / const.c).to(u.uJy) for flambda_, wavs_ in zip(flambda, wavs)]

        SED_obs_arr = [
            SED_obs(z, wav.value, flux.value, u.um, u.uJy)
            if all(i is not None for i in [z, wav, flux])
            else None
            for z, wav, flux in tqdm(
                zip(z_arr, wavs, fnu),
                desc="Constructing pipes SEDs",
                total=len(wavs),
                disable = galfind_logger.getEffectiveLevel() > logging.INFO
            )
        ]
        return SED_obs_arr
    
    def _extract_SED_wavelengths(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity,
    ) -> List[u.Quantity]:
        run_cat = deepcopy(cat)
        # remove filters without a depth measurement
        if self.SED_fit_params["excl_bands"] == []:
            excl_bands_arr = np.array([[] for _ in range(len(cat.gals))])
        elif isinstance(self.SED_fit_params["excl_bands"][0], list):
            excl_bands_arr = self.SED_fit_params["excl_bands"]
        else:
            excl_bands_arr = np.full(len(cat.gals), self.SED_fit_params["excl_bands"])
        assert len(excl_bands_arr) == len(cat.gals), \
            galfind_logger.critical(
                f"Bagpipes {excl_bands_arr=} must be a (ragged) list of lists with length {len(cat.gals)}!"
            )
        gals_arr = []
        for gal, excl_bands in tqdm(zip(cat.gals, excl_bands_arr), "Removing filters without depth measurements", disable = galfind_logger.getEffectiveLevel() > logging.INFO):
            remove_filt = []
            for i, (depth, filt) in enumerate(zip(gal.aper_phot[aper_diam].depths, gal.aper_phot[aper_diam].filterset)):
                if np.isnan(depth) or filt.band_name in excl_bands:
                    remove_filt.extend([filt])
            for filt in remove_filt:
                gal.aper_phot[aper_diam] -= filt
                galfind_logger.warning(
                    f"Removed {filt.band_name} from {gal.ID} for bagpipes fitting."
                )
            gals_arr.extend([gal])
        run_cat.gals = gals_arr
        filters = self._load_filters(run_cat, aper_diam)

        bagpipes = self.reload()
        from bagpipes.filters import filter_set

        ft_arr = [filter_set(gal_filters) for gal_filters in filters]
        wavs_arr = [self._get_wavs(ft) for ft in ft_arr]
        return wavs_arr
    
    def _get_wavs(self: Self, ft: Type[filter_set]) -> u.Quantity:

        bagpipes = self.reload()
        if self.SED_fit_params["sps_model"] == "BPASS":
            from bagpipes import config_bpass as pipes_config
        else:
            from bagpipes import config as pipes_config

        min_wav = ft.min_phot_wav
        max_wav = ft.max_phot_wav
        max_z = pipes_config.max_redshift

        max_wavs = [(min_wav / (1.0 + max_z)), 1.01 * max_wav, 10**8]

        x = [1.0]

        R = [pipes_config.R_other, pipes_config.R_phot, pipes_config.R_other]

        for i in range(len(R)):
            if i == len(R) - 1 or R[i] > R[i + 1]:
                while x[-1] < max_wavs[i]:
                    x.append(x[-1] * (1.0 + 0.5 / R[i]))

            else:
                while x[-1] * (1.0 + 0.5 / R[i]) < max_wavs[i]:
                    x.append(x[-1] * (1.0 + 0.5 / R[i]))

        wavs = (np.array(x) * u.AA).to(u.um).value

        return wavs

    def _extract_SED_fluxes(self: Self, SED_path: str, spectrum_type: str = "spectrum_full") -> u.Quantity:
        f = h5py.File(SED_path, "r")
        spectrum_full = np.array(f["advanced_quantities"][spectrum_type])
        spectrum_percentiles = np.percentile(spectrum_full, [16, 50, 84], axis = 0)
        #spectrum_l1 = spectrum_percentiles[0]
        spectrum_med = spectrum_percentiles[1] # erg / s / cm**2 / AA
        #spectrum_u1 = spectrum_percentiles[2]
        f.close()
        return spectrum_med

    def extract_PDFs(
        self: Self, 
        gal_property: str, 
        IDs: List[int], 
        PDF_paths: str, 
    ) -> List[Type[PDF]]:
        pass
        # breakpoint()
        # pdfs = PDF.from_1D_arr(
        #     gal_property,
        #     np.array(Table.read(PDF_paths, format="ascii.fast_no_header")["col1"]),
        #     self.SED_fit_params,
        # )
        # # ensure this works if only extracting 1 galaxy
        # if isinstance(IDs, (str, int, float)):
        #     IDs = np.array([int(IDs)])
        # if isinstance(PDF_paths, str):
        #     PDF_paths = [PDF_paths]
        # # # return list of None's if gal_property not in the PDF_paths, else load the PDFs
        # # if gal_property not in PDF_paths.keys():
        # #     return list(np.full(len(IDs), None))
        # # else:
        # if gal_property not in Bagpipes.gal_property_unit_dict.keys():
        #     Bagpipes.gal_property_unit_dict[gal_property] = (
        #         u.dimensionless_unscaled
        #     )
        # pdf_arrs = [
        #     np.array(Table.read(path, format="ascii.fast_no_header")["col1"])
        #     if type(path) != type(None)
        #     else None
        #     for path in tqdm(
        #         PDF_paths,
        #         desc=f"Loading {gal_property} PDFs",
        #         total=len(PDF_paths),
        #     )
        # ]
        # if gal_property == "z":
        #     pdfs = [
        #         Redshift_PDF.from_1D_arr(
        #             pdf
        #             * u.Unit(Bagpipes.gal_property_unit_dict[gal_property]),
        #             SED_fit_params,
        #             timed=timed,
        #         )
        #         if type(pdf) != type(None)
        #         else None
        #         for pdf in tqdm(
        #             pdf_arrs,
        #             desc=f"Constructing {gal_property} PDFs",
        #             total=len(pdf_arrs),
        #         )
        #     ]
        # else:
        #     pdfs = [
        #         SED_fit_PDF.from_1D_arr(
        #             gal_property,
        #             pdf
        #             * u.Unit(Bagpipes.gal_property_unit_dict[gal_property]),
        #             SED_fit_params,
        #             timed=timed,
        #         )
        #         if type(pdf) != type(None)
        #         else None
        #         for pdf in tqdm(
        #             pdf_arrs,
        #             desc=f"Constructing {gal_property} PDFs",
        #             total=len(pdf_arrs),
        #         )
        #     ]
        # # add save path to PDF
        # pdfs = [
        #     pdf.add_save_path(path) if type(pdf) != type(None) else None
        #     for path, pdf in zip(PDF_paths, pdfs)
        # ]
        # return pdfs
    
    def load_cat_property_PDFs(
        self: Self, 
        PDF_paths: List[str],
        IDs: List[int]
    ) -> List[Dict[str, Optional[Type[PDF]]]]:
        from . import PDF
        assert len(IDs) == len(PDF_paths), \
            galfind_logger.critical(
                f"{len(IDs)=} != {len(PDF_paths)=}"
            )
        ignore_labels = ["dust_curve", "photometry", "spectrum_full", "uvj", "sfh", "mass_weighted_zmet", "chisq_phot", "ndot_ion_caseB_rest", "ndot_ion_caseB_obs"]
        cat_property_PDFs = []
        for h5_path, ID in tqdm(zip(PDF_paths, IDs), desc=f"Loading {self.label} PDFs", total=len(IDs), disable=galfind_logger.getEffectiveLevel() > logging.INFO):
            gal_property_PDFs = {}
            if Path(h5_path).is_file():
                with h5py.File(h5_path, "r") as h5:
                    # TODO: Make appropriate redshift PDF should it be left free in the fitting procedure
                    for quantity_type in ["basic_quantities", "advanced_quantities"]:
                        quantities = np.array(h5[quantity_type])
                        for name in quantities:
                            if name not in ignore_labels:
                                if "redshift" in name:
                                    pdf = Redshift_PDF.from_1D_arr(np.array(h5[quantity_type][name]) * u.dimensionless_unscaled, self.SED_fit_params)
                                    name = "z"
                                else:
                                    pdf = SED_fit_PDF.from_1D_arr(name, np.array(h5[quantity_type][name]) * get_pipes_unit(name), self.SED_fit_params)
                                gal_property_PDFs[name] = pdf
                    h5.close()
            else:
                raise FileNotFoundError(f"{h5_path} not found!")
            cat_property_PDFs.extend([gal_property_PDFs])
        return cat_property_PDFs

    @staticmethod
    def get_galfind_fits_path(path):
        return path #.replace(".fits", "_galfind.fits")

    def load_pipes_fit_obj(self):
        pass

    def make_templates(self):
        pass

    @staticmethod
    def _load_pipes_phot(
        ID: int,
        cat: Catalogue,
        aper_diam: u.Quantity,
    ) -> np.NDArray[float, float]:
        from . import ID_Selector
        # get appropriate galaxy photometry from catalogue
        aper_phot = cat[ID_Selector(int(ID))][0].aper_phot[aper_diam]
        # extract fluxes and errors in uJy
        band_wavs = np.array([filt.WavelengthCen.to(u.AA).value for filt in aper_phot.filterset]) * u.AA
        assert all(u.get_physical_type(f_nu) in [
                "ABmag/spectral flux density",
                "spectral flux density",
            ] for f_nu in [aper_phot.flux, aper_phot.flux_errs]
        )
        # TODO: remove bands if they are masked
        flux = funcs.convert_mag_units(band_wavs, aper_phot.flux, u.uJy)
        flux_errs = aper_phot.flux_errs.to(u.uJy)
        try:
            flux = flux.unmasked
            flux_errs = flux_errs.unmasked
        except:
            pass
        flux = flux.value
        flux_errs = flux_errs.value
        #funcs.convert_mag_err_units(
        #     band_wavs, aper_phot.flux, aper_phot.flux_errs, u.uJy
        # )
        assert len(flux_errs) == len(aper_phot)
        # if flux < 1e19 and flux != -99 and flux != 0:
        #     pass
        pipes_input = np.vstack((np.array(flux), np.array(flux_errs))).T

        galfind_logger.debug(
            f"{cat.survey} {ID}: \n {pipes_input}, \n " + \
            f"bands = {','.join(aper_phot.filterset.band_names)}"
        )
        # TODO: append to bagpipes log file for survey/version/instrument
        return pipes_input

    def _load_fit_instructions(self: Self) -> None:
        if "fit_instructions" in self.SED_fit_params.keys():
            fit_instructions = self.SED_fit_params["fit_instructions"]
        else:
            fit_instructions = {}
            # Max age of birth clouds: Gyr
            fit_instructions["t_bc"] = self.SED_fit_params["t_bc"].to(u.Gyr).value

            # star formation history
            exp = {}
            const = {}
            delayed = {}
            burst = {}
            lognorm = {}

            # exponential SF history
            # Automatically adjusts for age of the universe
            exp["age"] = (0.001, 15.0)
            exp["age_prior"] = self.SED_fit_params["age_prior"]
            exp["tau"] = (0.01, 15.0)
            
            exp["massformed"] = (5.0, 15.0)  # Change this?

            exp["metallicity_prior"] = self.SED_fit_params["metallicity_prior"]
            if self.SED_fit_params["metallicity_prior"] == "log_10":
                exp["metallicity"] = (1e-03, 3)
            elif self.SED_fit_params["metallicity_prior"] == "const":
                exp["metallicity"] = (0, 3)

            # 1e-4 1e1
            const["age_max"] = (0.01, 15.0)  # Gyr
            const["age_min"] = 0.001  # Gyr
            const["age_prior"] = self.SED_fit_params["age_prior"]
            # Log_10 total stellar mass formed: M_Solar
            const["massformed"] = (5.0, 15.0)

            const["metallicity_prior"] = self.SED_fit_params["metallicity_prior"]
            if self.SED_fit_params["metallicity_prior"] == "log_10":
                const["metallicity"] = (1.e-3, 3.0)
            elif self.SED_fit_params["metallicity_prior"] == "uniform":
                const["metallicity"] = (0, 3)

            delayed["tau"] = (0.01, 15.0)  # `Gyr`
            # Log_10 total stellar mass formed: M_Solar
            delayed["massformed"] = (5.0, 15.0)

            delayed["age"] = (0.001, 15.0) # Gyr
            delayed["age_prior"] = self.SED_fit_params["age_prior"]
            delayed["metallicity_prior"] = self.SED_fit_params["metallicity_prior"]
            if self.SED_fit_params["metallicity_prior"] == "log_10":
                delayed["metallicity"] = (1.e-3, 3)
            elif self.SED_fit_params["metallicity_prior"] == "uniform":
                delayed["metallicity"] = (0, 3)

            burst["age"] = (0.01, 15.0)  # Gyr time since burst
            burst["age_prior"] = self.SED_fit_params["age_prior"]
            # Log_10 total stellar mass formed: M_Solar
            burst["massformed"] = (0.0, 15.0)

            burst["metallicity_prior"] = self.SED_fit_params["metallicity_prior"]
            if self.SED_fit_params["metallicity_prior"] == "log_10":
                burst["metallicity"] = (1e-03, 3)
            elif self.SED_fit_params["metallicity_prior"] == "uniform":
                burst["metallicity"] = (0, 3)

            # 1e-4 1e1
            # lognorm["tstart"] = (0.001, 15) # Gyr THIS NEVER DID ANYTHING!
            # lognorm["tstart_prior"] = age_prior
            lognorm["tmax"] = (0.01, 15) 
            lognorm["fwhm"] = (0.01, 15)
            # Log_10 total stellar mass formed: M_Solar
            lognorm["massformed"] = (5.0, 15.0)

            lognorm["metallicity_prior"] = self.SED_fit_params["metallicity_prior"]
            if self.SED_fit_params["metallicity_prior"] == "log_10":
                lognorm["metallicity"] = (1e-03, 3.0)
            elif self.SED_fit_params["metallicity_prior"] == "uniform":
                lognorm["metallicity"] = (0.0, 3.0)

            # DPL
            dblplaw = {}  # double-power-law
            # Vary the time of peak star-formation between
            # the Big Bang at 0 Gyr and 15 Gyr later. In
            # practice the code automatically stops this
            # exceeding the age of the universe at the
            # observed redshift.
            dblplaw["tau"] = (0.0, 15.0)
            dblplaw["tau_prior"] = self.SED_fit_params["age_prior"]
            # Vary the falling power law slope from 0.01 to 1000.
            dblplaw["alpha"] = (0.01, 1000.0)
            # Vary the rising power law slope from 0.01 to 1000.
            dblplaw["beta"] = (0.01, 1000.0)
            dblplaw["alpha_prior"] = "log_10"
            dblplaw["beta_prior"] = "log_10"
            # above as in Carnall et al. (2017).
            dblplaw["massformed"] = (5.0, 15.0)
            # dblplaw["metallicity"] = (0., 2.5)
            if self.SED_fit_params["metallicity_prior"] == "log_10":
                dblplaw["metallicity"] = (1.e-3, 3.0)
            elif self.SED_fit_params["metallicity_prior"] == "uniform":
                dblplaw["metallicity"] = (0.0, 3.0)

            # Leja et al. 2019 continuity SFH
            continuity = {}
            # continuity["age"] = (0.01, 15) # Gyr
            # continuity['age_prior'] = age_prior
            continuity["massformed"] = (5.0, 15.0)
            continuity["metallicity_prior"] = self.SED_fit_params["metallicity_prior"]
            if self.SED_fit_params["metallicity_prior"] == "log_10":
                continuity["metallicity"] = (1.e-3, 3.0)
            elif self.SED_fit_params["metallicity_prior"] == "uniform":
                continuity["metallicity"] = (0.0, 3.0)

            # Iyer et al. (2019) Non-parametric SFH
            nbins = 6
            iyer = {}  # The model of Iyer et al. (2019)
            iyer["sfr"] = (1e-3, 1e3)
            iyer["sfr_prior"] = "uniform"  # Solar masses per year
            iyer["bins"] = nbins # integer
            # This prior distribution must be used
            iyer["bins_prior"] = "dirichlet"
            # The Dirichlet prior has a single tunable parameter α that specifies how correlated the values are. In our case, values of this parameter α<1 result in values that can be arbitrarily close, leading to extremely spiky SFHs because galaxies have to assemble a significant fraction of their mass in a very short period of time, while α>1 leads to smoother SFHs with more evenly spaced values that never- theless have considerable diversity. In practice, we use a value of α=5, which leads to a distribution of parameters that is similar to what we find in SAM and MUFASA.
            iyer["alpha"] = 5.0
            # Log_10 total stellar mass formed: M_Solar
            iyer["massformed"] = (5.0, 15.0)

            iyer["metallicity_prior"] = self.SED_fit_params["metallicity_prior"]
            if self.SED_fit_params["metallicity_prior"] == "log_10":
                iyer["metallicity"] = (1.e-3, 3.0)
            elif self.SED_fit_params["metallicity_prior"] == "uniform":
                iyer["metallicity"] = (0.0, 3.0)

            # Put prior dictionaries in main fit_instructions dictionary
            if self.SED_fit_params["sfh"] == "exp":
                fit_instructions["exponential"] = exp
            elif self.SED_fit_params["sfh"] == "const":
                fit_instructions["constant"] = const
            elif self.SED_fit_params["sfh"] == "burst":
                fit_instructions["burst"] = burst
            elif self.SED_fit_params["sfh"] == "delayed":
                fit_instructions["delayed"] = delayed
            elif self.SED_fit_params["sfh"] == "delayed+burst":
                fit_instructions["delayed"] = delayed
                fit_instructions["burst"] = burst
            elif self.SED_fit_params["sfh"] == "exp+burst":
                fit_instructions["exponential"] = exp
                fit_instructions["burst"] = burst
            elif self.SED_fit_params["sfh"] == "const+burst":
                fit_instructions["const"] = const
                fit_instructions["burst"] = burst
            elif self.SED_fit_params["sfh"] == "rising":
                delayed["tau"] = (0.5, 15.0)
                fit_instructions["delayed"] = delayed
            elif self.SED_fit_params["sfh"] == "lognorm":
                fit_instructions["lognormal"] = lognorm
            elif self.SED_fit_params["sfh"] == "iyer":
                fit_instructions["iyer"] = iyer
            elif self.SED_fit_params["sfh"] == "continuity":
                fit_instructions["continuity"] = continuity
            elif self.SED_fit_params["sfh"] == "continuity_bursty":
                fit_instructions["continuity"] = continuity
            elif self.SED_fit_params["sfh"] == "dblplaw":
                fit_instructions["dblplaw"] = dblplaw
            else:
                err_message = f"{self.SED_fit_params['sfh']} SFH not found."
                galfind_logger.critical(err_message)
                raise Exception(err_message)

            # nebular emission (lines/continuum)
            nebular = {}
            if all(name is not None for name in [self.SED_fit_params["logU"], self.SED_fit_params["logU_prior"]]):
                nebular["logU"] = self.SED_fit_params["logU"]
                nebular["logU_prior"] = self.SED_fit_params["logU_prior"]
            if all(name is not None for name in [self.SED_fit_params["fesc"], self.SED_fit_params["fesc_prior"]]):
                nebular["fesc"] = self.SED_fit_params["fesc"]
                nebular["fesc_prior"] = self.SED_fit_params["fesc_prior"]
            fit_instructions["nebular"] = nebular

            # dust
            if all(name is not None for name in [self.SED_fit_params["dust"], self.SED_fit_params["dust_prior"]]):
                dust = {}
                # Multiplicative factor on Av for stars in birth clouds
                if self.SED_fit_params["dust_eta"] is not None:
                    dust["eta"] = self.SED_fit_params["dust_eta"]
                if self.SED_fit_params["dust"].lower() == "salim":
                    dust["type"] = "Salim"  # Salim
                    # Deviation from Calzetti slope ("Salim" type only)
                    dust["delta"] = (-0.3, 0.3)
                    dust["delta_prior"] = "Gaussian"
                    # This is Calzetti (approx)
                    dust["delta_prior_mu"] = 0.0
                    dust["delta_prior_sigma"] = 0.1
                    dust["B"] = (0.0, 5.0)
                    dust["B_prior"] = "uniform"
                elif self.SED_fit_params["dust"].lower() == "calzetti":
                    dust["type"] = "Calzetti"
                elif self.SED_fit_params["dust"].lower() == "cf00":
                    # Below taken from Tacchella+2022
                    # This is taken from Example 5 in the bagpipes documentation
                    dust["type"] = "CF00"
                    # dust["eta"] = 2.
                    # dust["Av"] = (0., 2.0)
                    dust["n"] = (0.3, 2.5)
                    dust["n_prior"] = "Gaussian"
                    # This is Calzetti (approx)
                    dust["n_prior_mu"] = 0.7
                    dust["n_prior_sigma"] = 0.3
                    # dust['n'] = (-1.0, 0.4) # 0.7088 is slope of calzetti - so deviation is - 0.3 < n < 1.1
                    # I think as it is offset from -1 (see Tachella, and is not given as negative, we want (0, 1.4), to represent (-1, 04))
                    dust["n_prior"] = "uniform"
                    # eta - 1 is done in code. Make eta be (1, 3) to represent (0, 2)
                    dust["eta"] = (1, 3)
                    dust["eta_prior"] = "Gaussian"
                    dust["eta_prior_mu"] = 2.0
                    dust["eta_prior_sigma"] = 0.3

                dust["Av_prior"] = self.SED_fit_params["dust_prior"]
                if self.SED_fit_params["dust_prior"] == "log_10":
                    dust["Av"] = (1e-4, 10.0)
                elif self.SED_fit_params["dust_prior"] == "uniform":
                    dust["Av"] = (0.0, 6.0)
                
                fit_instructions["dust"] = dust

            if not self.SED_fit_params["fix_z"]:
                if "z_sigma" not in self.SED_fit_params.keys():
                    fit_instructions["redshift"] = (0.0, 25.0)
        
        self.fit_instructions = fit_instructions

    def _update_continuity_sfh_fit_instructions(
        self: Self,
        cat: Optional[Catalogue],
    ) -> Dict[str, Any]:
        z_calculator = self.SED_fit_params["z_calculator"]
        if cat is None:
            assert isinstance(self.SED_fit_params["z_calculator"], float)
            z_arr = [self.SED_fit_params["z_calculator"]]
        else:
            if isinstance(z_calculator, float):
                z_arr = np.full(len(cat), z_calculator)
            else:
                z_arr = z_calculator(cat)
        fit_instructions_arr = []
        for z in z_arr:
            fit_instructions_i = deepcopy(self.fit_instructions)
            if "fixed_bin_ages" in self.SED_fit_params.keys():
                fixed_bin_ages = self.SED_fit_params["fixed_bin_ages"].to(u.Myr)
            else:
                fixed_bin_ages = [10.0] * u.Myr  # Default to 10 Myr bins
            if "num_bins" in self.SED_fit_params.keys():
                num_bins = self.SED_fit_params["num_bins"]
            else:
                num_bins = 6
            bin_edges = list(calculate_bins(
                redshift = z,
                fixed_bin_ages = fixed_bin_ages,
                num_bins=num_bins,
                return_flat=True, 
                output_unit='Myr', 
                log_time=False
            ))
            fit_instructions_i["continuity"]["bin_edges"] = bin_edges
            fit_instructions_arr.extend([fit_instructions_i])

        sfr_bins = {}
        if self.SED_fit_params["sfh"] == "continuity":
            scale = 0.3
        if self.SED_fit_params["sfh"] == "continuity_bursty":
            scale = 1.0
        for i in range(1, len(fit_instructions_arr[0]["continuity"]["bin_edges"]) - 1):
            sfr_bins["dsfr" + str(i)] = (-10.0, 10.0)
            sfr_bins["dsfr" + str(i) + "_prior"] = "student_t"
            # Defaults to this value as in Leja19, but can be set
            sfr_bins["dsfr" + str(i) + "_prior_scale"] = scale
            # Defaults to this value as in Leja19, but can be set
            sfr_bins["dsfr" + str(i) + "_prior_df"] = 2
        for fit_instructions in fit_instructions_arr:
            fit_instructions["continuity"] = {**fit_instructions["continuity"], **sfr_bins}
        self.fit_instructions = fit_instructions_arr

    def _update_redshift_fit_instructions(
        self: Self,
        cat: Type[Catalogue_Base],
    ):
        fit_instructions_arr = deepcopy(self.fit_instructions)
        if isinstance(fit_instructions_arr, dict):
            fit_instructions_arr = [fit_instructions_arr] * len(cat)
        # load catalogue redshifts
        z_calculator = self.SED_fit_params["z_calculator"]
        z_pdfs = z_calculator.extract_PDFs(cat)
        for fit_instructions, z_pdf in zip(fit_instructions_arr, z_pdfs):
            fit_instructions["redshift"] = (0.0, 25.0)
            fit_instructions["redshift_prior"] = "Gaussian"
            fit_instructions["redshift_prior_mu"] = z_pdf.median.value
            fit_instructions["redshift_prior_sigma"] = self.SED_fit_params["z_sigma"] * np.mean(z_pdf.errs.value)
        self.fit_instructions = fit_instructions_arr

def calculate_bins(redshift, redshift_sfr_start=20, log_time=True, output_unit = 'yr', return_flat = False, num_bins=6, fixed_bin_ages = [10.0] * u.Myr): #, cosmo = funcs.astropy_cosmo):
    time_observed = cosmo.lookback_time(redshift)
    time_sfr_start = cosmo.lookback_time(redshift_sfr_start)
    time_dif = abs(time_observed - time_sfr_start)

    # ensure fixed_bin_ages are ascending
    assert np.all(np.diff(fixed_bin_ages.value) > 0), \
        galfind_logger.critical(
            "fixed_bin_ages must be in ascending order!"
        )
    assert not log_time, \
        galfind_logger.critical(
            "log_time=True not implemented for calculate_bins!"
        ) 

    # if second_bin is not None:
    #     assert second_bin > first_bin, "Second bin must be greater than first bin"
    # first_bin = fixed_bin_ages[0] #* u.Myr
    # second_bin = fixed_bin_ages[1] if len(fixed_bin_ages) > 1 else None

    diff = np.linspace(np.log10(fixed_bin_ages[-1].to(output_unit).value), np.log10(time_dif.to(output_unit).value), num_bins - (len(fixed_bin_ages) - 1))
    #breakpoint()
    # if second_bin is None:
    #     diff = np.linspace(np.log10(first_bin.to(output_unit).value), np.log10(time_dif.to(output_unit).value), num_bins)
    # else:
    #     diff = np.linspace(np.log10(second_bin.to(output_unit).value), np.log10(time_dif.to(output_unit).value), num_bins-1)
    if not log_time:
        diff = 10**diff

    if return_flat:
        init_bins = [0.0]
        if len(fixed_bin_ages) > 1:
            init_bins.extend(fixed_bin_ages[:-1].to(output_unit).value)
        return np.concatenate([init_bins, diff])
        # if second_bin is None:
        #     breakpoint()
        #     return np.concatenate(([0],diff))
        # else:
        #     # if log_time:
        #     #     breakpoint()
        #     #     return np.concatenate([[0, np.log10(first_bin.to(output_unit).value)], diff])
        #     # else:
        #     breakpoint()
        #     return np.concatenate([[0, first_bin.to(output_unit).value], diff])
    else:
        raise NotImplementedError(
            "return_flat=False not implemented for calculate_bins!"
        )
        bins = []
        bins.append([0, np.log10(first_bin.to('year').value) if log_time else first_bin.to('year').value])
        if second_bin is not None:
            bins.append([np.log10(first_bin.to('year').value) if log_time else first_bin.to('year').value, np.log10(second_bin.to('year').value) if log_time else second_bin.to('year').value])
        for i in range(1, len(diff)):
            bins.append([diff[i-1], diff[i]])
        
        return  bins
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:55:57 2023

@author: austind
"""

from __future__ import annotations

import bagpipes
import itertools
import importlib
import os
import glob
import shutil
from pathlib import Path
import astropy.units as u
import numpy as np
from copy import deepcopy
from astropy.table import Table
from tqdm import tqdm
from typing import Union, Dict, Any, List, Tuple, Optional, NoReturn, TYPE_CHECKING
if TYPE_CHECKING:
    from . import Catalogue, PDF, Multiple_Filter
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import Redshift_PDF, SED_code, SED_fit_PDF, config, galfind_logger
from . import useful_funcs_austind as funcs
from .decorators import run_in_dir
from .SED import SED_obs

pipes_unit_dict = {
    "z": u.dimensionless_unscaled,
    "redshift": u.dimensionless_unscaled,
    "mass": u.dex(u.solMass),
    "dust": u.ABmag,
    "fwhm": u.Gyr,
    "metallicity": u.dimensionless_unscaled, # is this actually in units of solar metallicity?
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
    "xi_ion_caseB": u.dex(u.Hz / u.erg),
    "UV_colour": u.ABmag,
    "VJ_colour": u.ABmag,
    "beta_C94": u.dimensionless_unscaled,
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
        galfind_logger.info(
            f"No keys found in {label=} for Bagpipes unit_dict! Returning dimensionless"
        )
        return u.dimensionless_unscaled


class Bagpipes(SED_code):
    # # these should be made on runtime!
    # galaxy_properties = [
    #     "redshift",
    #     "stellar_mass",
    #     "formed_mass",
    #     "dust:Av",
    #     "beta_C94",
    #     "m_UV",
    #     "M_UV",
    #     "sfr",
    #     "sfr_10myr",
    #     "ssfr",
    #     "ssfr_10myr",
    # ]  # "Halpha_EWrest", "xi_ion_caseB",
    # gal_property_fmt_dict = {
    #     "z": "Redshift, z",
    #     "stellar_mass": r"$M_{\star}$",
    #     "formed_mass": r"$M_{\star, \mathrm{formed}}$",
    #     "dust:Av": r"$A_V$",
    #     "beta_C94": r"$\beta_{\mathrm{C94}}$",
    #     "m_UV": r"$m_{\mathrm{UV}}$",
    #     "M_UV": r"$M_{\mathrm{UV}}$",
    #     "Halpha_EWrest": r"EW$_{\mathrm{rest}}$(H$\alpha$)",
    #     "xi_ion_caseB": r"$\xi_{\mathrm{ion}}$",
    #     "sfr": r"SFR$_{\mathrm{100Myr}}$",
    #     "sfr_10myr": r"SFR$_{\mathrm{10Myr}}$",
    #     "ssfr": r"sSFR$_{\mathrm{100Myr}}$",
    #     "ssfr_10myr": r"sSFR$_{\mathrm{10Myr}}$",
    # }
    # gal_property_unit_dict = {
    #     "z": u.dimensionless_unscaled,
    #     "stellar_mass": "dex(solMass)",
    #     "formed_mass": "dex(solMass)",
    #     "dust:Av": u.ABmag,
    #     "beta_C94": u.dimensionless_unscaled,
    #     "m_UV": u.ABmag,
    #     "M_UV": u.ABmag,
    #     "Halpha_EWrest": u.AA,
    #     "xi_ion_caseB": u.Hz / u.erg,
    #     "sfr": u.solMass / u.yr,
    #     "sfr_10myr": u.solMass / u.yr,
    #     "ssfr": u.yr**-1,
    #     "ssfr_10myr": u.yr**-1,
    # }
    # galaxy_property_dict = {
    #     **{
    #         gal_property
    #         if "redshift" not in gal_property
    #         else "z": f"{gal_property}_50"
    #         for gal_property in galaxy_properties
    #     },
    #     **{"chi_sq": "chisq_phot"},
    # }
    # galaxy_property_errs_dict = {
    #     gal_property if "redshift" not in gal_property else "z": [
    #         f"{gal_property}_16",
    #         f"{gal_property}_84",
    #     ]
    #     for gal_property in galaxy_properties
    # }
    # available_templates = ["BC03", "BPASS"]
    # ext_src_corr_properties = [
    #     "stellar_mass",
    #     "formed_mass",
    #     "m_UV",
    #     "M_UV",
    #     "sfr",
    #     "sfr_10myr",
    # ]

    def __init__(
        self: Self,
        SED_fit_params: Dict[str, Any], 
        custom_label: Optional[str] = None,
        sampler: str = "multinest",
    ) -> Self:
        # determine which bagpipes import is required
        #importlib.reload(bagpipes)
        if custom_label is not None:
            self.custom_label = custom_label
        self.sampler = sampler
        # start off empty
        self.gal_property_labels = {}
        self.gal_property_err_labels = {}
        super().__init__(SED_fit_params)

    @classmethod
    def from_label(cls, label: str) -> Type[SED_code]:
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
            redshift_label = label.split(SED_fit_params["metallicity_prior"])[
                -1
            ][1:]
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
                if "z_range" in self.SED_fit_params.keys():
                    assert len(self.SED_fit_params["z_range"]) == 2
                    redshift_label = f"{int(self.SED_fit_params['z_range'][0])}_z_{int(self.SED_fit_params['z_range'][1])}"
                else:
                    galfind_logger.critical(
                        f"Bagpipes {self.SED_fit_params=} must include either " + \
                        "'z_range' if 'fix_z' == False or not included!"
                    )
                    breakpoint()
            # sort SPS label
            if self.SED_fit_params["sps_model"].upper() == "BC03":
                sps_label = ""  # should change this probably to read BC03
            elif self.SED_fit_params["sps_model"].upper() == "BPASS":
                sps_label = f"_{self.SED_fit_params['sps_model'].lower()}"
            else:
                galfind_logger.critical(
                    f"Bagpipes {self.SED_fit_params=} must include " + \
                    "'sps_model' with .upper() in ['BC03', 'BPASS']"
                )
                breakpoint()
            return (
                f"Bagpipes_sfh_{self.SED_fit_params['sfh']}_dust_{self.SED_fit_params['dust']}_"
                + f"{self.SED_fit_params['dust_prior']}_Z_{self.SED_fit_params['metallicity_prior']}"
                + f"{sps_label}_{redshift_label}"
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
                "logU": (-3.0, -1.0), 
                "logU_prior": "uniform",
                "fesc": (1.e-4, 1.0),
                "fesc_prior": "log_10"
            }
            for name, default in defaults_dict.items():
                if name not in self.SED_fit_params.keys():
                    self.SED_fit_params[name] = default
        super()._assert_SED_fit_params()

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
        super()._load_gal_property_labels(self.gal_property_labels)

    def _load_gal_property_err_labels(self):
        super()._load_gal_property_err_labels(self.gal_property_err_labels)

    def _load_gal_property_units(self) -> NoReturn:
        # TODO: Copied from EAZY
        self.gal_property_units = {}

    def make_in(
        self,
        cat: Catalogue,
        aper_diam: u.Quantity,
        overwrite: bool = False
    ) -> str:
        # no need for bagpipes input catalogue
        pass

    def _temp_out_subdir(self, cat: Catalogue) -> str:
        return f"{cat.version}/{cat.survey}/{cat.filterset.instrument_name}/{self.label}/temp"
    
    def _new_subdir(self, cat: Catalogue) -> str:
        return f"{cat.version}/{cat.survey}/{cat.filterset.instrument_name}/{self.label}"

    def _move_files(self, cat: Catalogue, direction = "from_temp") -> NoReturn:
        assert direction in ["from_temp", "to_temp"]
        temp_out_subdir = f"{cat.survey}/temp"#self._temp_out_subdir(cat)
        temp_post_dir = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/posterior/{temp_out_subdir}"
        temp_plots_dir = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/plots/{temp_out_subdir}"
        temp_sed_dir = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/seds/{temp_out_subdir}"

        temp_sfr_dir = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/sfr/{temp_out_subdir}"
        temp_fits_dir = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/cats/{temp_out_subdir}"
        new_subdir = self._new_subdir(cat)
        new_post_dir = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/posterior/{new_subdir}"
        new_plots_dir = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/plots/{new_subdir}"
        new_sed_dir = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/seds/{new_subdir}"
        new_sfr_dir = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/sfr/{new_subdir}"
        new_fits_dir = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/cats/{new_subdir}"
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
            from_fits_dir = temp_fits_dir
            to_fits_dir = new_fits_dir
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
            from_fits_dir = new_fits_dir
            to_fits_dir = temp_fits_dir
        funcs.make_dirs(f"{to_post_dir}/")
        funcs.make_dirs(f"{to_plots_dir}/")
        funcs.make_dirs(f"{to_sed_dir}/")
        funcs.make_dirs(f"{to_sfr_dir}/")
        funcs.make_dirs(f"{to_fits_dir}/")
        for from_dir, to_dir in zip(
            [from_post_dir, from_plots_dir, from_sed_dir, from_sfr_dir],
            [to_post_dir, to_plots_dir, to_sed_dir, to_sfr_dir]
        ):
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
        # TODO: fits catalogue
        

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
        # out_subdir = self._temp_out_subdir(cat)
        out_subdir = f"{cat.survey}/temp"
        path_post = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/posterior/{out_subdir}"
        path_plots = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/plots/{out_subdir}"
        path_sed = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/seds/{out_subdir}"
        path_sfr = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/sfr/{out_subdir}"
        path_fits = f"{config['Bagpipes']['PIPES_OUT_DIR']}/pipes/cats/{out_subdir}"

        new_subdir = f"{cat.version}/{cat.survey}/{cat.filterset.instrument_name}/{self.label}"
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
        if rerun:
            to_run_arr = np.ones(len(cat), dtype=bool)
            # move stuff back to temp directory
            self._move_files(cat, direction = "to_temp")
        else:
            to_run_arr = np.ones(len(cat), dtype=bool)
            for i, gal in enumerate(cat):
                save_path = f"{new_path_post}/{gal.ID}.h5"
                if Path(save_path).is_file() and not overwrite:
                    to_run_arr[i] = False
            if all(not to_run for to_run in to_run_arr):
                galfind_logger.info("All objects run and not rerun/overwrite.")
                return

        run_cat = deepcopy(cat)
        run_cat.gals = run_cat[to_run_arr]
        # remove filters without a depth measurement
        gals_arr = []
        for gal in tqdm(run_cat.gals, "Removing filters without depth measurements"):
            remove_filt = []
            for i, (depth, filt) in enumerate(zip(gal.aper_phot[aper_diam].depths, gal.aper_phot[aper_diam].filterset)):
                if np.isnan(depth):
                    remove_filt.extend([filt])
            for filt in remove_filt:
                gal.aper_phot[aper_diam] -= filt
                galfind_logger.warning(
                    f"Removed {filt.band_name} from {gal.ID} for bagpipes fitting due to NaN depth."
                )
            gals_arr.extend([gal])
        run_cat.gals = gals_arr
        IDs = [gal.ID for gal in gals_arr]
        filters = self._load_filters(run_cat, aper_diam)

        # if fix_z is not False
        if not (not self.SED_fit_params["fix_z"]):
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

        fit_instructions = self._load_fit_instructions()
        #print(fit_instructions)
        if all(hasattr(gal, "aper_phot") for gal in gals_arr):
            photometry_exists = True
            load_func = self._load_pipes_phot
        # TODO: spectroscopic fitting
        spectrum_exists = False
        if "plot" in kwargs.keys():
            plot = kwargs["plot"]
        else:
            plot = True
        fit_cat = bagpipes.fit_catalogue(
            IDs,
            fit_instructions,
            load_func,
            spectrum_exists = spectrum_exists,
            photometry_exists = photometry_exists,
            run = out_subdir,
            make_plots = plot,
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
        galfind_logger.info(f"Fitting bagpipes with {fit_instructions=}")
        try:
            run_parallel = False
            fit_cat.fit(verbose = True, mpi_serial = run_parallel, sampler = self.sampler)
        except Exception as e:
            raise e

        # rename files and move to appropriate directories

        if self.rank == 0:
            galfind_logger.info(f"Renaming and moving {self.label} output files on rank 0.")
            self._move_files(cat, direction = "from_temp")
            # # try:
            #     # if overwrite:
            #     #     shutil.rmtree(f'{path_overall}/posterior/{field}/{new_path}', ignore_errors=True)
            #     #     shutil.rmtree(f'{path_overall}/plots/{field}/{new_path}', ignore_errors=True)

            # new_post_names = [path.split("/")[-1] for path in glob.glob(f"{path_post}/*")]
            # for post_name in new_post_names:
            #     if not Path(f"{new_path_post}/{post_name}").is_file():
            #         os.rename(f"{path_post}/{post_name}", f"{new_path_post}/{post_name}")
            #         galfind_logger.info(
            #             f"Moved {post_name} to new_path_post"
            #         )
            #     else:
            #         galfind_logger.info(
            #             f"{post_name} already exists in new_path_post, skipping!"
            #         )

            # new_plot_names = [path.split("/")[-1] for path in glob.glob(f"{path_plots}/*")]
            # for plot_name in new_plot_names:
            #     if not Path(f"{new_path_plots}/{plot_name}").is_file():
            #         os.rename(f"{path_plots}/{plot_name}", f"{new_path_plots}/{plot_name}")
            #         galfind_logger.info(
            #             f"Moved {plot_name} to new_path_plots"
            #         )
            #     else:
            #         galfind_logger.info(
            #             f"{plot_name} already exists in new_path_plots, skipping!"
            #         )

            # new_sed_names = [path.split("/")[-1] for path in glob.glob(f"{path_sed}/*")]
            # for sed_name in new_sed_names:
            #     if not Path(f"{new_path_sed}/{sed_name}").is_file():
            #         os.rename(f"{path_sed}/{sed_name}", f"{new_path_sed}/{sed_name}")
            #         galfind_logger.info(
            #             f"Moved {sed_name} to new_path_sed"
            #         )
            #     else:
            #         galfind_logger.info(
            #             f"{sed_name} already exists in new_path_sed, skipping!"
            #         )

            # new_sfh_names = [path.split("/")[-1] for path in glob.glob(f"{path_sfr}/*")]
            # for sfh_name in new_sfh_names:
            #     if not Path(f"{new_path_sfr}/{sfh_name}").is_file():
            #         os.rename(f"{path_sfr}/{sfh_name}", f"{new_path_sfr}/{sfh_name}")
            #         galfind_logger.info(
            #             f"Moved {sfh_name} to new_path_sfr"
            #         )
            #     else:
            #         galfind_logger.info(
            #             f"{sfh_name} already exists in new_path_sfr, skipping!"
            #         )
            
            # if not Path(new_path_fits).is_file():
            #     os.rename(f"{path_fits}.fits", new_path_fits)
            #     galfind_logger.info(
            #         f"Moved {path_fits}.fits to {new_path_fits}"
            #     )
            # else:
            #     # merge catalogues
            #     raise NotImplementedError
            
            # except FileExistsError as e:
            #     print(e)
            #     print('Path to rename already exists, skipping output.')
            #     print('Moving files instead')
            #     # Moving plots 
            #     files = glob.glob(path_plots+'/*')
            #     new_plot_path = f'{path_overall}/plots/{field}/{new_path}'
            
            #     for file in files:
            #         filename = file.split('/')[-1]
            #         shutil.move(file, new_plot_path+f'/{filename}')
            #         print('Moving', file, 'to', new_plot_path+f'/{filename}')
            #     # Moving .h5 files
            #     files = glob.glob(path_post+'/*')
            #     new_post_path = f'{path_overall}/posterior/{field}/{new_path}'
                
            #     for file in files:
            #         print(file)
            #         filename = file.split('/')[-1]
                    
            #         shutil.move(file, new_post_path+f'/{filename}')
            #         print('Moving', file, 'to', new_post_path+f'/{filename}')
                
            #     files = glob.glob(path_sed+'/*')
            #     new_sed_path = f'{path_overall}/seds/{field}/{new_path}'
                
            #     for file in files:
            #         print(file)
            #         filename = file.split('/')[-1]
            #         shutil.move(file, new_sed_path+f'/{filename}')
            #         print('Moving', file, 'to', new_sed_path+f'/{filename}')

            #     # Merging catalogue
            #     try:
            #         orig_table = Table.read(f'{path_overall}/cats/{field}/{new_path}.fits')

            #         run_table = Table.read(f'{path_overall}/cats/{out_subdir}.fits')
            #         if reset:
            #             new_table = run_table
            #         else:
            #             new_table = vstack([orig_table, run_table])
                    
            #         new_table.write(f'{path_overall}/cats/{field}/{new_path}.fits', overwrite=True)
            #     except FileNotFoundError:
            #         shutil.move(f'{path_overall}/cats/{out_subdir}.fits', f'{path_overall}/cats/{field}/{new_path}.fits')
            
            # shutil.rmtree('pipes/posterior/'+out_subdir, ignore_errors=True)
            # shutil.rmtree('pipes/seds/'+out_subdir, ignore_errors=True)
            # try:
            #     os.remove(f'pipes/cats/{field}/temp.fits')
            # except FileNotFoundError:
            #     pass
            # shutil.rmtree('pipes/plots/'+out_subdir, ignore_errors=True)

            # try:
            #     os.remove(path)
            # except FileNotFoundError: 
            #     pass

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
                # delete the key
                del self.gal_property_labels[key]
                # only 1 redshift key should exist
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
    ) -> NDArray[str]:
        self._generate_filters(cat.filterset)
        cat_filt_paths = np.zeros(len(cat), dtype=object)
        for i, gal in enumerate(cat):
            gal_filt_paths = []
            for filt in gal.aper_phot[aper_diam].filterset:
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
        PDF_dir = out_path.replace(".fits", "").replace("cats", "pdfs")
        SED_dir = out_path.replace(".fits", "").replace("cats", "seds")

        load_properties = []
        PDF_paths = {
            gal_property if "redshift" not in gal_property else "z": [
                f"{PDF_dir}/{gal_property}/{str(int(ID))}_{cat.survey}.txt"
                if Path(
                    f"{PDF_dir}/{gal_property}/{str(int(ID))}_{cat.survey}.txt"
                ).is_file()
                else None
                for ID in cat.ID
            ]
            for gal_property in load_properties
        }
        # determine SED paths
        SED_paths = [
            f"{SED_dir}/{str(int(ID))}_{cat.survey}.dat"
            if Path(f"{SED_dir}/{str(int(ID))}_{cat.survey}.dat").is_file()
            else None
            for ID in cat.ID
        ]
        return in_path, out_path, fits_out_path, PDF_paths, SED_paths

    def extract_SEDs(
        self: Self,
        IDs: List[int], 
        SED_paths: Union[str, List[str]]
    ) -> List[SED_obs]:
        # ensure this works if only extracting 1 galaxy
        if isinstance(IDs, (str, int, float)):
            IDs = np.array([int(IDs)])
        if isinstance(SED_paths, str):
            SED_paths = [SED_paths]
        assert len(IDs) == len(SED_paths), galfind_logger.critical(
            f"len(IDs) = {len(IDs)} != len(data_paths) = {len(SED_paths)}!"
        )
        z_arr = np.zeros(len(IDs))
        for i, path in enumerate(SED_paths):
            if path is not None:
                f = open(path)
                header = f.readline()
                z_arr[i] = float(header.replace("\n", "").split("z=")[-1])
                f.close()
        data_arr = [
            np.loadtxt(path) if path is not None else None
            for path in SED_paths
        ]
        wavs = [
            data[:, 0] if data is not None else None
            for data in data_arr
        ]
        fluxes = [
            data[:, 2] if data is not None else None
            for data in data_arr
        ]
        SED_obs_arr = [
            SED_obs(z, wav, flux, u.um, u.uJy)
            if all(i is not None for i in [z, wav, flux])
            else None
            for z, wav, flux in tqdm(
                zip(z_arr, wavs, fluxes),
                desc="Constructing pipes SEDs",
                total=len(wavs),
            )
        ]
        return SED_obs_arr

    def extract_PDFs(
        self: Self, 
        gal_property: str, 
        IDs: List[int], 
        PDF_paths: Union[str, List[str]], 
    ) -> List[Type[PDF]]:
        # ensure this works if only extracting 1 galaxy
        if isinstance(IDs, (str, int, float)):
            IDs = np.array([int(IDs)])
        if isinstance(PDF_paths, str):
            PDF_paths = [PDF_paths]
        # # return list of None's if gal_property not in the PDF_paths, else load the PDFs
        # if gal_property not in PDF_paths.keys():
        #     return list(np.full(len(IDs), None))
        # else:
        if gal_property not in Bagpipes.gal_property_unit_dict.keys():
            Bagpipes.gal_property_unit_dict[gal_property] = (
                u.dimensionless_unscaled
            )
        pdf_arrs = [
            np.array(Table.read(path, format="ascii.fast_no_header")["col1"])
            if type(path) != type(None)
            else None
            for path in tqdm(
                PDF_paths,
                desc=f"Loading {gal_property} PDFs",
                total=len(PDF_paths),
            )
        ]
        if gal_property == "z":
            pdfs = [
                Redshift_PDF.from_1D_arr(
                    pdf
                    * u.Unit(Bagpipes.gal_property_unit_dict[gal_property]),
                    SED_fit_params,
                    timed=timed,
                )
                if type(pdf) != type(None)
                else None
                for pdf in tqdm(
                    pdf_arrs,
                    desc=f"Constructing {gal_property} PDFs",
                    total=len(pdf_arrs),
                )
            ]
        else:
            pdfs = [
                SED_fit_PDF.from_1D_arr(
                    gal_property,
                    pdf
                    * u.Unit(Bagpipes.gal_property_unit_dict[gal_property]),
                    SED_fit_params,
                    timed=timed,
                )
                if type(pdf) != type(None)
                else None
                for pdf in tqdm(
                    pdf_arrs,
                    desc=f"Constructing {gal_property} PDFs",
                    total=len(pdf_arrs),
                )
            ]
        # add save path to PDF
        pdfs = [
            pdf.add_save_path(path) if type(pdf) != type(None) else None
            for path, pdf in zip(PDF_paths, pdfs)
        ]
        return pdfs

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
    ) -> NDArray[float, float]:
        # get appropriate galaxy photometry from catalogue
        aper_phot = cat[{"ID": ID}].aper_phot[aper_diam]
        # extract fluxes and errors in uJy
        band_wavs = np.array([filt.WavelengthCen.to(u.AA).value for filt in aper_phot.filterset]) * u.AA
        assert all(u.get_physical_type(f_nu) in [
                "ABmag/spectral flux density",
                "spectral flux density",
            ] for f_nu in [aper_phot.flux, aper_phot.flux_errs]
        )
        # TODO: remove bands if they are masked
        flux = funcs.convert_mag_units(band_wavs, aper_phot.flux, u.uJy).unmasked.value
        flux_errs = aper_phot.flux_errs.to(u.uJy).unmasked.value
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

    def _load_fit_instructions(self: Self) -> Dict[str, Any]:
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
            
            exp["massformed"] = (5.0, 12.0)  # Change this?

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
            const["massformed"] = (5.0, 12.0)

            const["metallicity_prior"] = self.SED_fit_params["metallicity_prior"]
            if self.SED_fit_params["metallicity_prior"] == "log_10":
                const["metallicity"] = (1.e-3, 3.0)
            elif self.SED_fit_params["metallicity_prior"] == "uniform":
                const["metallicity"] = (0, 3)

            delayed["tau"] = (0.01, 15.0)  # `Gyr`
            # Log_10 total stellar mass formed: M_Solar
            delayed["massformed"] = (5.0, 12.0)

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
            burst["massformed"] = (0.0, 12.0)

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
            lognorm["massformed"] = (5.0, 12.0)

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
            dblplaw["massformed"] = (5.0, 12.0)
            # dblplaw["metallicity"] = (0., 2.5)
            if self.SED_fit_params["metallicity_prior"] == "log_10":
                dblplaw["metallicity"] = (1.e-3, 3.0)
            elif self.SED_fit_params["metallicity_prior"] == "uniform":
                dblplaw["metallicity"] = (0.0, 3.0)

            # Leja et al. 2019 continuity SFH
            continuity = {}
            # continuity["age"] = (0.01, 15) # Gyr
            # continuity['age_prior'] = age_prior
            continuity["massformed"] = (5.0, 12.0)
            continuity["metallicity_prior"] = self.SED_fit_params["metallicity_prior"]
            if self.SED_fit_params["metallicity_prior"] == "log_10":
                continuity["metallicity"] = (1.e-3, 3.0)
            elif self.SED_fit_params["metallicity_prior"] == "uniform":
                continuity["metallicity"] = (0.0, 3.0)

            # if not self.len(fix_z_SED_fit_params) != 0:
            #     if self.SED_fit_params["sfh"] == "continuity":
            #         raise Exception(
            #             "Continuity model not compatible with varying redshift range."
            #         )
            #         continuity["redshift"] = redshift_range

            # # This is a filler - real one is generated below when catalogue is loaded in
            # cont_nbins = num_bins
            # continuity["bin_edges"] = list(
            #     calculate_bins(
            #         redshift=8,
            #         num_bins=cont_nbins,
            #         first_bin=first_bin,
            #         second_bin=second_bin,
            #         return_flat=True,
            #         output_unit="Myr",
            #         log_time=False,
            #     )
            # )
            # scale = 0
            # if self.SED_fit_params["sfh"] == "continuity":
            #     scale = 0.3
            # if self.SED_fit_params["sfh"] == "continuity_bursty":
            #     scale = 1.0

            # for i in range(1, len(continuity["bin_edges"]) - 1):
            #     continuity["dsfr" + str(i)] = (-10.0, 10.0)
            #     continuity["dsfr" + str(i) + "_prior"] = "student_t"
            #     # Defaults to this value as in Leja19, but can be set
            #     continuity["dsfr" + str(i) + "_prior_scale"] = scale
            #     # Defaults to this value as in Leja19, but can be set
            #     continuity["dsfr" + str(i) + "_prior_df"] = 2

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
            iyer["massformed"] = (5.0, 12.0)

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
            # elif self.SED_fit_params["sfh"] == "continuity":
            #     fit_instructions["continuity"] = continuity
            # elif self.SED_fit_params["sfh"] == "continuity_bursty":
            #     fit_instructions["continuity"] = continuity
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
                fit_instructions["redshift"] = (0.0, 25.0)

        return fit_instructions

def calculate_bins(redshift, redshift_sfr_start=20, log_time=True, output_unit = 'yr', return_flat = False, num_bins=6, first_bin=10*u.Myr, second_bin=None, cosmo = funcs.astropy_cosmo):
    time_observed = cosmo.lookback_time(redshift)
    time_sfr_start = cosmo.lookback_time(redshift_sfr_start)
    time_dif = abs(time_observed - time_sfr_start)
    if second_bin is not None:
        assert second_bin > first_bin, "Second bin must be greater than first bin"

    if second_bin is None:
        diff = np.linspace(np.log10(first_bin.to(output_unit).value), np.log10(time_dif.to(output_unit).value), num_bins)
    else:
        diff = np.linspace(np.log10(second_bin.to(output_unit).value), np.log10(time_dif.to(output_unit).value), num_bins-1)
    
    if not log_time:
        diff = 10**diff

    if return_flat:
        if second_bin is None:
            return np.concatenate(([0],diff))
        else:
            if log_time:
                return np.concatenate([[0, np.log10(first_bin.to(output_unit).value)], diff])
            else:
                return np.concatenate([[0, first_bin.to(output_unit).value], diff])
    bins = []
    bins.append([0, np.log10(first_bin.to('year').value) if log_time else first_bin.to('year').value])
    if second_bin is not None:
        bins.append([np.log10(first_bin.to('year').value) if log_time else first_bin.to('year').value, np.log10(second_bin.to('year').value) if log_time else second_bin.to('year').value])
     
    for i in range(1, len(diff)):
        bins.append([diff[i-1], diff[i]])
    
    return  bins
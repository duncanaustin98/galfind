#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:44:23 2023

@author: austind
"""

from __future__ import annotations

import itertools
import subprocess
from pathlib import Path
import os
import json
import logging
from tqdm import tqdm
import astropy.units as u
import numpy as np
from astropy.io import fits
from numpy.typing import NDArray
from astropy.table import Table
from typing import Any, Dict, List, Union, NoReturn, Optional, TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from . import Catalogue, Filter, Multiple_Filter, PDF, SED_obs
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11
from . import Multiple_Filter, SED_code, Redshift_PDF, config, galfind_logger
from . import useful_funcs_austind as funcs
from .decorators import run_in_dir

class LePhare(SED_code):
    ID_label = "IDENT"
    ext_src_corr_properties = ["MASS_BEST", "SFR_BEST"]

    def __init__(self, SED_fit_params: Dict[str, Any]):
        super().__init__(SED_fit_params)

    @classmethod
    def from_label(cls, label: str):
        raise NotImplementedError
        #return super().from_label(label)

    # @property
    # def ID_label(self) -> str:
    #     return "IDENT"
    
    @property
    def label(self) -> str:
        # TODO: Full name should probably be different to this, depending on template set used
        return f"{self.__class__.__name__}_{self.SED_fit_params['GAL_TEMPLATES']}"
    
    @property
    def hdu_name(self) -> str:
        return f"{self.__class__.__name__}_{self.SED_fit_params['GAL_TEMPLATES']}"
    
    @property
    def tab_suffix(self) -> str:
        return f"{self.SED_fit_params['GAL_TEMPLATES']}"
    
    @property
    def required_SED_fit_params(self) -> List[str]:
        return ["GAL_TEMPLATES"]
    
    @property
    def are_errs_percentiles(self) -> bool:
        # TODO: Check this is correct!
        return False
    
    def _load_gal_property_labels(self) -> NoReturn:
        self.gal_property_labels = {
            "z": "Z_BEST",
            "Mstar": "MASS_BEST",
            "chi_sq": "CHI_BEST",
        }
    
    def _load_gal_property_err_labels(self) -> NoReturn:
        self.gal_property_err_labels = {}

    def _load_gal_property_units(self) -> NoReturn:
        # still need to double check the UBVJ units
        self.gal_property_units = {
            **{gal_property: u.dimensionless_unscaled 
               for gal_property in ["z", "chi_sq"]},
            **{"Mstar": u.solMass},
        }

    def _assert_SED_fit_params(self) -> NoReturn:
        default_strings = [
            "GAL_AGES", 
            "STAR_TEMPLATES", 
            "QSO_TEMPLATES", 
            "COMPILE_SURVEY_FILTERS",
            "Z_STEP",
            "COSMOLOGY",
            "MOD_EXTINC",
            "EXTINC_LAW",
            "EB_V",
            "EM_LINES",
        ]
        default_types = [str, str, str, bool, [float], [float], [int], [str], [float], bool]
        names_dict = {
            "Z_STEP": ["DELTA_Z_LOW_Z", "Z_MAX", "DELTA_Z_HIGH_Z"],
            "COSMOLOGY": ["H0", "OMEGA_M", "OMEGA_L"]
        }
        # TODO: Add this into the SED_code class, called by super()
        for default_str, default_type in zip(default_strings, default_types):
            if default_str not in self.SED_fit_params.keys():
                if default_type is bool:
                    self.SED_fit_params[default_str] = config.getboolean("LePhare", default_str)
                elif default_type is int:
                    self.SED_fit_params[default_str] = config.getint("LePhare", default_str)
                elif default_type is float:
                    self.SED_fit_params[default_str] = config.getfloat("LePhare", default_str)
                elif default_type is str or isinstance(default_type, list):
                    config_str = config.get("LePhare", default_str)
                    if default_type == str:
                        self.SED_fit_params[default_str] = config_str
                    else:
                        assert len(default_type) == 1
                        params = [default_type[0](i) for i in config_str.split(",")]
                        if default_str in names_dict.keys():
                            names = names_dict[default_str]
                            for param, name in zip(params, names):
                                if name not in self.SED_fit_params.keys():
                                    self.SED_fit_params[name] = param
                                else:
                                    galfind_logger.info(
                                        f"{name} already in LePhare " + \
                                        "SED_fit_params, skipping config value"
                                    )
                        else:
                            self.SED_fit_params[default_str] = params
                else:
                    raise ValueError(f"Invalid default_type: {default_type}")
            elif default_str in names_dict.keys():
                assert isinstance(self.SED_fit_params[default_str], list)
                assert len(self.SED_fit_params[default_str]) == len(names_dict[default_str]), \
                    galfind_logger.critical(
                        f"{default_str=} must have {len(names_dict[default_str])} elements"
                    )
                assert all(name not in self.SED_fit_params.keys() for name in names_dict[default_str]), \
                    galfind_logger.critical(
                        f"{names_dict[default_str]} already in SED_fit_params"
                    )
                for param, name in zip(self.SED_fit_params[default_str], names_dict[default_str]):
                    self.SED_fit_params[name] = param

        return super()._assert_SED_fit_params()

    def make_in(
        self,
        cat: Catalogue, 
        aper_diam: u.Quantity,
        overwrite: bool = False,
    ) -> str:  # from FITS_organiser.py
        in_path = self.get_in_path(cat, aper_diam)
        if not Path(in_path).is_file() or overwrite:
            # 1) obtain input data
            IDs = np.array([gal.ID for gal in cat.gals])
            redshifts = np.array([-99.0 for i in range(len(cat))]) # TODO: Load spec-z's
            # load photometry (STILL SHOULD BE MORE GENERAL!!!)
            if self.SED_fit_params["COMPILE_SURVEY_FILTERS"]:
                input_filterset = cat.filterset
            else:
                instr_name = cat.filterset.instrument_name
                input_filterset = Multiple_Filter.from_instruments(instr_name.split("+"))
            
            phot, phot_err = self._load_phot(
                cat,
                aper_diam,
                u.erg / (u.s * u.cm ** 2 * u.Hz), #u.Jy, #u.ABmag,
                -99.0,
                None, #{"threshold": 2.0, "value": 3.0},
                input_filterset,
            )

            # calculate context
            contexts = self._calc_context(cat, cat.filterset, aper_diam)

            # 2) make and save LePhare .in catalogue
            in_data = np.array(
                [
                    np.concatenate(
                        (
                            [IDs[i]],
                            list(itertools.chain(*zip(phot[i], phot_err[i]))),
                            [contexts[i]],
                            [redshifts[i]],
                        ),
                        axis=None,
                    )
                    for i in range(len(IDs))
                ]
            )
            in_names = (
                ["ID"]
                + list(
                    itertools.chain(
                        *zip(
                            input_filterset.band_names,
                            [f"{filt.band_name}_err" for filt in input_filterset],
                        )
                    )
                )
                + ["context", "z"]
            )
            in_types = (
                [int]
                + list(np.full(len(input_filterset) * 2, float))
                + [int, float]
            )
            in_tab = Table(in_data, dtype=in_types, names=in_names)
            funcs.make_dirs(in_path)
            in_tab.write(
                in_path,
                format="ascii.no_header",
                delimiter=" ",
                overwrite=True,
            )
        return in_path
    
    def compile(
        self: Self,
        filterset: Multiple_Filter, 
        types: List[str] = ["STAR", "QSO", "GAL"],
        template_save_suffix: str = ""
    ) -> None:
        # determine appropriate input filterset
        #input_filterset = self.get_input_filterset(filterset)
        self.compile_filters(filterset)
        for type in types:
            self.compile_binary(type)
            self.compile_templates(filterset, type, save_suffix = template_save_suffix)

    @run_in_dir(path=config["LePhare"]["LEPHARE_CONFIG_DIR"])
    def compile_binary(self, _type: str) -> None:
        assert _type in ["STAR", "QSO", "GAL"], \
        galfind_logger.critical(
            f"{_type=} not in ['STAR', 'QSO', 'GAL']"
        )
        save_dir = f"{config['LePhare']['LEPHARE_SED_DIR']}/{_type}"
        save_name = self.SED_fit_params[f"{_type}_TEMPLATES"]
        save_path = f"{save_dir}/{save_name}.list"
        assert Path(save_path).is_file(), \
            galfind_logger.critical(
                f"{save_path=} not found"
            )
        output_bin_name = self._get_bin_out_path(_type)
        if not Path(output_bin_name).is_file():
            if _type == "GAL":
                age_path = f"{self.SED_fit_params['GAL_AGES']}.list"
            else:
                age_path = ""
            config_name = config["LePhare"]["LEPHARE_CONFIG_FILE"].split("/")[-1]
            input = [
                f"{config['DEFAULT']['GALFIND_DIR']}/compile_lephare_binaries.sh",
                config_name,
                _type,
                save_path,
                save_name,
                age_path
            ]
            # SExtractor bash script python wrapper
            galfind_logger.debug(input)
            galfind_logger.info(f"Compiling {_type} LePhare binaries")
            process = subprocess.Popen(input) #, shell = True)
            process.wait()
            galfind_logger.info(
                f"Finished compiling {_type} LePhare " + \
                f"binaries, saved to {output_bin_name}"
            )
            funcs.change_file_permissions(output_bin_name)
            funcs.change_file_permissions(output_bin_name.replace(".bin", ".doc"))
            if Path(output_bin_name.replace(".bin", ".phys")).is_file():
                funcs.change_file_permissions(output_bin_name.replace(".bin", ".phys"))
        else:
            galfind_logger.debug(f"{output_bin_name} already exists")

    def _get_bin_out_path(self: Self, type: str) -> str:
        return f"{os.environ['LEPHAREWORK']}/lib_bin/" + \
            self.SED_fit_params[f"{type}_TEMPLATES"] + ".bin"

    @run_in_dir(path=config["LePhare"]["LEPHARE_CONFIG_DIR"])
    def compile_filters(
        self: Self,
        input_filterset: Multiple_Filter,
    ) -> None:
        save_dir = f"{os.environ['LEPHAREWORK']}/filt"
        save_name = self._get_save_filterset_name(input_filterset)
        save_path = f"{save_dir}/{save_name}"
        if not Path(save_path).is_file():
            [self._make_filt_txt(filt) for filt in input_filterset]
            config_name = config["LePhare"]["LEPHARE_CONFIG_FILE"].split("/")[-1]
            in_filt_name = self._get_input_filterset_name(input_filterset)
            input = [
                f"{config['DEFAULT']['GALFIND_DIR']}/compile_lephare_filters.sh",
                config_name,
                in_filt_name,
                save_name
            ]
            # SExtractor bash script python wrapper
            galfind_logger.debug(input)
            if self.SED_fit_params["COMPILE_SURVEY_FILTERS"]:
                extra_log_out = "survey"
            else:
                extra_log_out = f"{input_filterset.instrument_name=}"
            galfind_logger.info(f"Compiling LePhare filters for {extra_log_out}!")
            process = subprocess.Popen(input)
            process.wait()
            galfind_logger.info(
                f"Finished compiling LePhare " + \
                f"filters for {extra_log_out}, saved to {save_path}"
            )
            funcs.change_file_permissions(save_path)
            funcs.change_file_permissions(save_path.replace(".filt", ".doc"))
        else:
            galfind_logger.debug(f"{save_path} already exists")

    def _make_filt_txt(self: Self, filt: Filter) -> None:
        save_path = f"{os.environ['LEPHAREDIR']}/filt/" + \
            f"{filt.facility_name}/{filt.instrument_name}/" + \
            f"{filt.band_name}.txt"
        funcs.make_dirs(save_path)
        if Path(save_path).is_file():
            galfind_logger.debug(f"LePhare filter for {repr(filt)} already exists")
        else:
            galfind_logger.info(f"Making LePhare filter for {repr(filt)}")
            #np.vstack([np.array(filt.wav.to(u.AA).value), np.array(filt.trans)]).T
            out_filt = np.column_stack((np.array(filt.wav.to(u.AA).value), np.array(filt.trans)))
            np.savetxt(save_path, out_filt, header = \
                f"{filt.instrument_name}/{filt.band_name}", comments = "# ")
            galfind_logger.info(f"Saved LePhare filters for {repr(filt)} to {save_path}")
            funcs.change_file_permissions(save_path)

    def _get_input_filterset_name(self, input_filterset: Multiple_Filter) -> str:
        return ",".join([f"{filt.facility_name}/{filt.instrument_name}" + \
            f"/{filt.band_name}.txt" for filt in input_filterset])
    
    def _get_save_filterset_name(self, input_filterset: Multiple_Filter) -> str:
        if self.SED_fit_params["COMPILE_SURVEY_FILTERS"]:
            return "+".join([filt.band_name for filt in input_filterset]) + ".filt"
        else:
            return input_filterset.instrument_name + ".filt"

    @run_in_dir(path=config["LePhare"]["LEPHARE_CONFIG_DIR"])
    def compile_templates(
        self, 
        input_filterset: Multiple_Filter, 
        type: str,
        save_suffix: str = ""
    ) -> NoReturn:
        assert type in ["STAR", "QSO", "GAL"], \
            galfind_logger.critical(
                f"{type=} not in ['STAR', 'QSO', 'GAL']"
            )
        
        # TODO: Neaten up path loading, although the below should still work
        temp_save_dir = f"{config['LePhare']['LEPHARE_SED_DIR']}/{type}"
        temp_save_name = self.SED_fit_params[f"{type}_TEMPLATES"]
        temp_save_path = f"{temp_save_dir}/{temp_save_name}.list"
        assert Path(temp_save_path).is_file(), \
            galfind_logger.critical(
                f"TEMPLATES at {temp_save_path=} not found"
            )
        out_bin_path = self._get_bin_out_path(type)
        assert Path(out_bin_path).is_file(), \
            galfind_logger.critical(
                f"BINARIES at {out_bin_path=} not found"
            )

        save_dir = f"{os.environ['LEPHAREWORK']}/lib_mag"
        filt_save_name = self._get_save_filterset_name(input_filterset).replace(".filt", "")
        bin_save_name = self.SED_fit_params[f"{type}_TEMPLATES"]
        if save_suffix != "":
            if save_suffix[0] != "_":
                save_suffix = f"_{save_suffix}"
        save_name = f"{bin_save_name}_{filt_save_name}{save_suffix}"
        save_path = f"{save_dir}/{save_name}.bin"
        if not Path(save_path).is_file():
            config_name = config["LePhare"]["LEPHARE_CONFIG_FILE"].split("/")[-1]
            in_filt_name = self._get_input_filterset_name(input_filterset)
            input = [
                f"{config['DEFAULT']['GALFIND_DIR']}/compile_lephare_templates.sh",
                config_name,
                type,
                in_filt_name,
                filt_save_name,
                bin_save_name,
                save_suffix,
            ]
            # SExtractor bash script python wrapper
            galfind_logger.debug(input)
            if self.SED_fit_params["COMPILE_SURVEY_FILTERS"]:
                extra_log_out = "survey"
            else:
                extra_log_out = f"{input_filterset.instrument_name=}"
            galfind_logger.info(
                "Compiling LePhare templates for " + \
                f"{extra_log_out} {bin_save_name}!"
            )
            process = subprocess.Popen(input)
            process.wait()
            galfind_logger.info(
                f"Finished compiling LePhare templates " + \
                f"for {extra_log_out} {bin_save_name}, saved to {save_path}!"
            )
            funcs.change_file_permissions(save_path)
            funcs.change_file_permissions(save_path.replace(".bin", ".doc"))
        else:
            galfind_logger.debug(f"{save_path} already exists")

    def _calc_context(
        self: Self,
        cat: Catalogue,
        filterset: Multiple_Filter,
        aper_diam: u.Quantity,
        # phot: NDArray[float]
    ) -> List[int]:
        # TODO: Update for the case where some galaxies have no data for a specific band!
        contexts = np.zeros(len(cat), dtype=int)
        for i, gal in tqdm(
            enumerate(cat),
            total = len(cat),
            desc = f"Calculating {repr(self)} {repr(cat)} contexts",
            disable = galfind_logger.getEffectiveLevel() > logging.INFO
        ):
            gal_context = 2 * (2 ** (len(filterset) - 1)) - 1
            for j, band in enumerate(filterset):
                if band not in gal.aper_phot[aper_diam].filterset:
                    band_context = 2 ** j
                    gal_context = gal_context - band_context
            contexts[i] = gal_context
        return contexts

    # Currently black box fitting from the lephare config path. Need to make this function more general
    def fit(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity,
        save_SEDs: bool = True,
        save_PDFs: bool = True,
        overwrite: bool = False,
        update: bool = False,
        **kwargs: Dict[str, Any],
    ) -> None:

        fits_out_path = self.get_fits_out_path(cat, aper_diam)

        if not Path(fits_out_path).is_file() or overwrite:
            galfind_logger.info(f"Fitting {repr(cat)} with {repr(self)}")

            in_path = self.get_in_path(cat, aper_diam)
            out_path = self.get_out_path(cat, aper_diam)
            lephare_config_path = f"{config['LePhare']['LEPHARE_CONFIG_DIR']}/default.para"

            if "save_suffix" in kwargs.keys():
                save_suffix = kwargs["save_suffix"]
            else:
                save_suffix = ""
            gal_lib_name = self.get_lib_name(cat, "GAL", save_suffix = save_suffix)
            star_lib_name = self.get_lib_name(cat, "STAR", save_suffix = save_suffix)
            qso_lib_name = self.get_lib_name(cat, "QSO", save_suffix = save_suffix)

            run_dir = fits_out_path.replace(".fits", "_SEDs")
            funcs.make_dirs(f"{run_dir}/*")

            # temp_save_dir = f"{config['LePhare']['LEPHARE_SED_DIR']}/{type}"
            # temp_save_name = self.SED_fit_params[f"{type}_TEMPLATES"]
            # temp_save_path = f"{temp_save_dir}/{temp_save_name}.list"
            # assert Path(temp_save_path).is_file(), \
            #     galfind_logger.critical(
            #         f"TEMPLATES at {temp_save_path=} not found"
            #     )
            # out_bin_path = self._get_bin_out_path(type)
            # assert Path(out_bin_path).is_file(), \
            #     galfind_logger.critical(
            #         f"BINARIES at {out_bin_path=} not found"
            #     )

            process = subprocess.Popen(
                [
                    f"{config['DEFAULT']['GALFIND_DIR']}/run_lephare.sh",
                    lephare_config_path,
                    in_path,
                    out_path,
                    config["DEFAULT"]["GALFIND_DIR"],
                    gal_lib_name,
                    star_lib_name,
                    qso_lib_name,
                    run_dir,
                ]
            )
            process.wait()
        else:
            galfind_logger.info(
                f"{repr(cat)} already fitted with " + \
                f"{repr(self)} at {fits_out_path}, skipping fit"
            )

    def SED_fit_params_from_label(self, label):
        label_arr = label.split("_")
        assert len(label_arr) == 2
        assert label_arr[1] in self.available_templates
        return {"code": self, "templates": label_arr[1]}

    def make_fits_from_out(
        self: Self,
        out_path: str,
        *args: Any,
        **kwargs: Dict[str, Any],
    ) -> None:
        fits_out_path = self.fits_out_path_from_out_path(out_path, *args, **kwargs)

        # read in the data from the .out table
        txt_in = np.genfromtxt(out_path, comments="#")

        # store the column labels
        column_labels = []
        reached_output_format = False
        with open(out_path) as open_file:
            while True:
                line = open_file.readline()
                # break while statement if it is not a comment line
                # i.e. does not start with #
                if not line.startswith("#"):
                    break

                if line.startswith("#  IDENT"):
                    reached_output_format = True

                if reached_output_format:
                    if line.startswith("#########################"):
                        break
                    params_numbers = line.split(", ")
                    params_numbers[0] = params_numbers[0].replace("#  ", "")
                    params_numbers.remove(params_numbers[-1])
                    # print(params_numbers)
                    for param_number in params_numbers:
                        output_param = param_number.split("  ")
                        # print(output_param)
                        column_labels.append(output_param[0])
            open_file.close()

        funcs.change_file_permissions(fits_out_path)
        # write data to a .fits file
        fits_columns = []
        for i in range(len(column_labels)):
            loc_col = fits.Column(
                name=column_labels[i],
                array=np.array((txt_in.T)[i]),
                format="D",
            )
            fits_columns.append(loc_col)
        fits_table = fits.BinTableHDU.from_columns(fits_columns)
        fits_table.writeto(fits_out_path, overwrite=True)
        funcs.change_file_permissions(fits_out_path)
        return fits_out_path

    def get_in_path(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity,
    ) -> str:
        in_dir = f"{config['LePhare']['LEPHARE_DIR']}/input/{cat.filterset.instrument_name}/{cat.version}/{cat.survey}"
        in_name = cat.cat_name.replace('.fits', f"_{aper_diam.to(u.arcsec).value:.2f}as.in")
        return f"{in_dir}/{in_name}"

    def get_out_path(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity,
    ) -> str:
        in_path = self.get_in_path(cat, aper_diam)
        out_folder = funcs.split_dir_name(
            in_path.replace("input", "output"), "dir"
        )
        out_path = f"{out_folder}/{funcs.split_dir_name(in_path, 'name').replace('.in', '.out')}"
        return out_path

    def fits_out_path_from_out_path(
        self: Self,
        out_path: str,
        *args: Any,
        **kwargs: Dict[str, Any],
    ) -> str:
        return out_path.replace(
            ".out",
            f"_LePhare_{self.SED_fit_params['GAL_TEMPLATES']}.fits"
        )

    def get_fits_out_path(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity,
    ) -> str:
        return self.fits_out_path_from_out_path(
            self.get_out_path(cat, aper_diam)
        )

    @staticmethod
    def get_spec_path(
        fits_out_path: str,
        ID: int,
    ) -> str:
        return fits_out_path.replace(
            ".fits",
            f"_SEDs/Id{'0' * (9 - len(str(ID))) + str(ID)}.spec"
        )

    def get_PDF_paths(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity,
    ) -> Dict[str, List[str]]:
        fits_out_path = self.get_fits_out_path(cat, aper_diam)
        PDF_paths = {
            "z": [
                self.get_spec_path(fits_out_path, gal.ID)
                for gal in cat
            ]
        }
        return PDF_paths
    
    def get_SED_paths(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity
    ) -> Dict[str, List[str]]:
        fits_out_path = self.get_fits_out_path(cat, aper_diam)
        return [self.get_spec_path(fits_out_path, gal.ID) for gal in cat]

    def _get_out_paths(
        self: Self, 
        cat: Catalogue, 
        aper_diam: u.Quantity
    ) -> Tuple[str, str, str, Dict[str, List[str]], List[str]]:
        in_path = self.get_in_path(cat, aper_diam)
        out_path = self.get_out_path(cat, aper_diam)
        fits_out_path = self.get_fits_out_path(cat, aper_diam)
        PDF_paths = self.get_PDF_paths(cat, aper_diam)
        SED_paths = self.get_SED_paths(cat, aper_diam)
        return in_path, out_path, fits_out_path, PDF_paths, SED_paths

    def get_lib_name(
        self: Self,
        cat: Catalogue,
        type: str,
        save_suffix: str = "",
    ) -> str:
        assert type in ["STAR", "QSO", "GAL"], \
            galfind_logger.critical(
                f"{type=} not in ['STAR', 'QSO', 'GAL']"
            )
        #save_dir = f"{os.environ['LEPHAREWORK']}/lib_mag"
        filt_save_name = self._get_save_filterset_name(cat.filterset).replace(".filt", "")
        bin_save_name = self.SED_fit_params[f"{type}_TEMPLATES"]
        if save_suffix != "":
            if save_suffix[0] != "_":
                save_suffix = f"_{save_suffix}"
        save_name = f"{bin_save_name}_{filt_save_name}{save_suffix}"
        #save_path = f"{save_dir}/{save_name}.bin"
        return save_name

    def extract_SEDs(
        self: Self, 
        IDs: List[int], 
        SED_paths: Union[str, List[str]],
        *args,
        **kwargs,
    ) -> List[SED_obs]:
        # TODO: Generalize. Currently a near-carbon copy of EAZY method

        # ensure this works if only extracting 1 galaxy
        if isinstance(IDs, (str, int, float)):
            IDs = np.array([int(IDs)])
        if isinstance(SED_paths, str):
            SED_paths = [SED_paths]
        assert len(IDs) == len(SED_paths), \
            galfind_logger.critical(
                f"{len(IDs)=} != {len(SED_paths)=}"
            )
        # extract redshift PDF for each ID
        SEDs = [
            self._extract_SED(SED_path, type = "GAL")
            for SED_path in tqdm(
                SED_paths,
                total = len(IDs),
                desc = f"Constructing {repr(self)} galaxy SEDs",
                disable = galfind_logger.getEffectiveLevel() > logging.INFO
            )
        ]
        return SEDs

    def _extract_SED(
        self: Self,
        SED_path: str,
        type: str = "GAL",
    ) -> SED_obs:
        from galfind import SED_obs
        assert type in ["GAL", "STAR", "QSO"], \
            galfind_logger.critical(
                f"{type=} not in ['GAL', 'STAR', 'QSO']"
            )
        wav = []
        flux = []
        reached_pdf = False
        SED_fmt = []
        with open(SED_path) as f:
            i = 0
            j = 0
            while True:
                line = f.readline()
                # determine PDF length
                if line.startswith("PDF"):
                    PDF_len = int(line.split(" ")[-1])
                # determine SED lengths
                if not reached_pdf and any(
                    line.startswith(type) for type in ["GAL", "STAR", "QSO"]
                ):
                    SED_name = line.split(" ")[0]
                    SED_len = int([ele for ele in line.split(" ") if ele != ""][1])
                    SED_z = float([ele for ele in line.split(" ") if ele != ""][5])
                    SED_fmt.append((SED_name, SED_len, SED_z))
                # locate beginning of redshift PDF
                if line.startswith("  0.00000"):
                    reached_pdf = True
                    wanted_SED = [fmt for fmt in SED_fmt if fmt[0].startswith(type)][0]
                    wanted_SED_name = wanted_SED[0]
                    wanted_SED_len = wanted_SED[1]
                    wanted_z = wanted_SED[2]
                    skip_len = PDF_len
                    for fmt in SED_fmt:
                        if fmt[0] != wanted_SED_name:
                            skip_len += fmt[1]
                        else:
                            break
                # skip over len(PDF_len) lines
                if reached_pdf:
                    i += 1
                    if i > skip_len:
                        j += 1
                        # started SEDs
                        if j > wanted_SED_len:
                            break
                        else:
                            line = line.replace("  "," ")
                            line = line.replace("\n", "")
                            SED_ = line.split(" ")
                            SED_.remove(SED_[0])
                            wav.append(float(SED_[0]))
                            flux.append(float(SED_[1]))
            f.close()
        sed_obs = SED_obs(wanted_z, wav, flux, u.AA, u.ABmag)
        return sed_obs

    def extract_PDFs(
        self: Self, 
        gal_property: str, 
        IDs: List[int], 
        PDF_paths: Union[str, List[str]], 
    ) -> List[Redshift_PDF]:
        # TODO: Generalize. Again the majority is carbon copy of eazy method

        # ensure this works if only extracting 1 galaxy
        if isinstance(IDs, (str, int, float)):
            IDs = np.array([int(IDs)])
        if isinstance(PDF_paths, str):
            PDF_paths = [PDF_paths]

        # EAZY only has redshift PDFs
        if gal_property != "z":
            return np.array(list(itertools.repeat(None, len(IDs))))
        else:
            # ensure the correct type
            assert isinstance(PDF_paths, (list, np.ndarray)), \
                galfind_logger.critical(
                    f"type(data_paths) = {type(PDF_paths)} not in [list, np.array]!"
                )
            assert isinstance(IDs, (list, np.ndarray)), \
                galfind_logger.critical(
                    f"type(IDs) = {type(IDs)} not in [list, np.array]!"
                )
            # ensure the correct array size
            assert len(IDs) == len(PDF_paths), \
                galfind_logger.critical(
                    f"{len(IDs)=} != {len(PDF_paths)=}!"
                )
            # extract redshift PDF for each ID
            redshift_PDFs = [
                self._extract_PDF(PDF_path)
                for PDF_path in tqdm(
                    PDF_paths,
                    total = len(IDs),
                    desc = f"Constructing {repr(self)} redshift PDFs",
                    disable = galfind_logger.getEffectiveLevel() > logging.INFO
                )
            ]
            return redshift_PDFs

    def _extract_PDF(
        self: Self,
        PDF_path: str,
    ) -> Redshift_PDF:
        z = []
        PDF = []
        reached_output_format = False
        with open(PDF_path) as f:
            i = 0
            while True:
                line = f.readline()
                if line.startswith("PDF"):
                    PDF_len = int(line.split(" ")[-1])
                # start at z = 0 (i.e. beginning of PDF)
                if line.startswith("  0.00000"):
                    reached_output_format = True
                if reached_output_format:
                    line = line.replace("  "," ")
                    line = line.replace("\n", "")
                    z_PDF = line.split(" ")
                    z_PDF.remove(z_PDF[0])
                    z.append(float(z_PDF[0]))
                    PDF.append(float(z_PDF[1]))
                    i += 1
                    if i == PDF_len: #line.startswith(f" {zmax:.5f}"): # end at zmax
                        break
            f.close()
        
        pdf_out = Redshift_PDF(
            np.array(z) * u.dimensionless_unscaled,
            np.array(PDF),
            self.SED_fit_params,
            normed=False
        )
        return pdf_out

    def load_cat_property_PDFs(
        self: Self, 
        PDF_paths: Union[List[str], List[Dict[str, str]]],
        IDs: List[int],
    ) -> List[Dict[str, Optional[Type[PDF]]]]:
        # TODO: Put this in parent class (carbon copy of eazy method)
        cat_property_PDFs_ = {
            gal_property: self.extract_PDFs(
                gal_property,
                IDs,
                PDF_path,
            )
            for gal_property, PDF_path in PDF_paths.items()
        }
        cat_property_PDFs_ = [
            {
                gal_property: PDF_arr[i]
                for gal_property, PDF_arr in cat_property_PDFs_.items()
                if PDF_arr[i] is not None
            }
            for i in range(len(IDs))
        ]
        # set to None if no PDFs are found
        cat_property_PDFs = [
            None if len(cat_property_PDF) == 0 else cat_property_PDF
            for cat_property_PDF in cat_property_PDFs_
        ]
        return cat_property_PDFs

# def calc_LePhare_errs(cat, col_name):
#     if col_name == "Z_BEST":
#         data = np.array(cat[col_name])
#         data_err = np.array([np.array(cat[col_name + "68_LOW"]), np.array(cat[col_name + "68_HIGH"])])
#         data, data_err = adjust_errs(data, data_err)
#         return data_err

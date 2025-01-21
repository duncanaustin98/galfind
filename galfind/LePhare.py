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
import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import Table
from typing import Any, Dict, List, Union, NoReturn, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from . import Catalogue, Filter, Multiple_Filter, PDF
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11
from . import Multiple_Filter, SED_code, config, galfind_logger
from . import useful_funcs_austind as funcs
from .decorators import run_in_dir

class LePhare(SED_code):
    ext_src_corr_properties = ["MASS_BEST", "SFR_BEST"]

    def __init__(self, SED_fit_params: Dict[str, Any]):
        super().__init__(SED_fit_params)

    @classmethod
    def from_label(cls, label: str):
        raise NotImplementedError
        #return super().from_label(label)

    @property
    def ID_label(self) -> str:
        return "IDENT"
    
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
                if default_type == bool:
                    self.SED_fit_params[default_str] = config.getboolean("LePhare", default_str)
                elif default_type == int:
                    self.SED_fit_params[default_str] = config.getint("LePhare", default_str)
                elif default_type == float:
                    self.SED_fit_params[default_str] = config.getfloat("LePhare", default_str)
                elif default_type == str or isinstance(default_type, list):
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
        
        in_dir = f"{config['LEPHARE']['LEPHARE_DIR']}/input/{cat.filterset.instrument_name}/{cat.version}/{cat.survey}"
        in_name = cat.cat_name.replace('.fits', f"_{aper_diam.to(u.arcsec).value:.2f}as.in")
        in_path = f"{in_dir}/{in_name}"
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
                u.ABmag,
                -99.0,
                {"threshold": 2.0, "value": 3.0},
                input_filterset,
            )

            # calculate context
            contexts = self._calc_context(phot)

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
        self, 
        filterset: Multiple_Filter, 
        types: List[str] = ["STAR", "QSO", "GAL"],
        template_save_suffix: str = ""
    ) -> NoReturn:
        # determine appropriate input filterset
        input_filterset = self.get_input_filterset(filterset)
        self.compile_filters(input_filterset)
        for _type in types:
            self.compile_binary(_type)
            self.compile_templates(input_filterset, _type, save_suffix = template_save_suffix)

    @run_in_dir(path=config["LePhare"]["LEPHARE_CONFIG_DIR"])
    def compile_binary(self, _type: str) -> NoReturn:
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

    def _get_bin_out_path(self, _type: str) -> str:
        return f"{os.environ['LEPHAREWORK']}/lib_bin/" + \
            self.SED_fit_params[f"{_type}_TEMPLATES"] + ".bin"

    @run_in_dir(path=config["LePhare"]["LEPHARE_CONFIG_DIR"])
    def compile_filters(self, input_filterset: Multiple_Filter) -> NoReturn:
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

    def _make_filt_txt(self, filt: Filter) -> NoReturn:
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

    def _calc_context(self, phot: np.ndarray) -> List[int]:
        # TODO: Update for the case where some galaxies have no data for a specific band!
        contexts = []
        for i, gal in enumerate(cat):
            gal_context = 2 * (2 ** (len(SED_input_bands) - 1)) - 1
            for j, band in enumerate(SED_input_bands):
                if band not in gal.aper_phot[aper_diam].instrument.band_names:
                    band_context = 2**j
                    gal_context = gal_context - band_context
            contexts.append(gal_context)
        return np.array(contexts).astype(int)

    # Currently black box fitting from the lephare config path. Need to make this function more general
    def fit(self, in_path, out_path, SED_folder, instrument):
        template_name = f"{instrument.name}_MedWide"
        lephare_config_path = f"{self.code_dir}/default.para"
        # LePhare bash script python wrapper
        process = subprocess.Popen(
            [
                f"{config['DEFAULT']['GALFIND_DIR']}/run_lephare.sh",
                lephare_config_path,
                in_path,
                out_path,
                config["DEFAULT"]["GALFIND_DIR"],
                SED_folder,
                template_name,
            ]
        )
        process.wait()

    def SED_fit_params_from_label(self, label):
        label_arr = label.split("_")
        assert len(label_arr) == 2
        assert label_arr[1] in self.available_templates
        return {"code": self, "templates": label_arr[1]}

    def make_fits_from_out(
        self, out_path, *args, **kwargs
    ):  # from TXT_to_FITS_converter.py
        fits_out_path = self.out_fits_name(out_path, *args, **kwargs)

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

    def _get_out_paths(
        self: Self, 
        cat: Catalogue, 
        aper_diam: u.Quantity
    ) -> Tuple[str, str, str, Dict[str, List[str]], List[str]]:
        return NotImplementedError
        # return out_path.replace(".out", "_LePhare.fits")

    @staticmethod
    def extract_SEDs(IDs, SED_paths):
        pass

    @staticmethod
    def extract_PDFs(gal_property, IDs, PDF_paths, SED_fit_params):
        pass
        # str_ID = "0" * (9 - len(str(ID))) + str(ID)
        # print("ID = " + str_ID)

        # z = []
        # PDF = []
        # reached_output_format = False
        # with open(self.z_PDF_path_from_cat_path(cat_path, ID, low_z_run)) as open_file:
        #     while True:
        #         line = open_file.readline()
        #         # start at z = 0
        #         if line.startswith("  0.00000"):
        #             reached_output_format = True
        #         if reached_output_format:
        #             line = line.replace("  "," ")
        #             line = line.replace("\n", "")
        #             z_PDF = line.split(" ")
        #             z_PDF.remove(z_PDF[0])
        #             z.append(float(z_PDF[0]))
        #             PDF.append(float(z_PDF[1]))
        #         # end at z = 15
        #         if line.startswith(" 25.00000"):
        #             break
        #     open_file.close()
        # return z, PDF

    def load_cat_property_PDFs(
        self: Self, 
        PDF_paths: Union[List[str], List[Dict[str, str]]],
        IDs: List[int]
    ) -> List[Dict[str, Optional[Type[PDF]]]]:
        pass

# def calc_LePhare_errs(cat, col_name):
#     if col_name == "Z_BEST":
#         data = np.array(cat[col_name])
#         data_err = np.array([np.array(cat[col_name + "68_LOW"]), np.array(cat[col_name + "68_HIGH"])])
#         data, data_err = adjust_errs(data, data_err)
#         return data_err

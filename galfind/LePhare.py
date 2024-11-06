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

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import Table
from typing import Any, Dict, List, NoReturn, TYPE_CHECKING
if TYPE_CHECKING:
    from . import Catalogue

from . import Multiple_Filter, SED_code, config, galfind_logger
from . import useful_funcs_austind as funcs

class LePhare(SED_code):
    available_templates = ["BC03"]
    ext_src_corr_properties = ["MASS_BEST", "SFR_BEST"]

    def __init__(self, SED_fit_params: Dict[str, Any]):
        super().__init__(SED_fit_params)

    @property
    def ID_label(self) -> str:
        return "IDENT"
    
    @property
    def label(self) -> str:
        # TODO: Full name should probably be different to this, depending on template set used
        return f"{self.__class__.__name__}_{self.SED_fit_params['templates']}"
    
    @property
    def hdu_name(self) -> str:
        return f"{self.__class__.__name__}_{self.SED_fit_params['templates']}"
    
    @property
    def tab_suffix(self) -> str:
        return f"{self.SED_fit_params['templates']}"
    
    @property
    def required_SED_fit_params(self) -> List[str]:
        return ["templates"]
    
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
            if self.SED_fit_params["survey_in_filt"]:
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
    
    def _get_bin_path(self, cat: Catalogue) -> str:
        return f"{config['LEPHARE']['LEPHARE_DIR']}/templates/{self.SED_fit_params['templates']}.bin"

    def _get_filt_path(self, cat: Catalogue) -> str:
        if self.SED_fit_params["survey_in_filt"]:
            unique_filterset_name = f"{cat.survey}_{cat.filterset.instrument_name}"
            return f"{config['LEPHARE']['LEPHARE_DIR']}/filt/{unique_filterset_name}.bin"
        else:
            return f"{config['LEPHARE']['LEPHARE_DIR']}/filt/"

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
        lephare_config_path = f"{self.code_dir}/Photo_z.para"
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

    @staticmethod
    def get_out_paths(out_path, SED_fit_params, IDs):
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

# def calc_LePhare_errs(cat, col_name):
#     if col_name == "Z_BEST":
#         data = np.array(cat[col_name])
#         data_err = np.array([np.array(cat[col_name + "68_LOW"]), np.array(cat[col_name + "68_HIGH"])])
#         data, data_err = adjust_errs(data, data_err)
#         return data_err

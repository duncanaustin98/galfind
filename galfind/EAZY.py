#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:52:36 2023

@author: austind
"""

from __future__ import annotations

# EAZY.py
import itertools
import os
import time
import warnings
from pathlib import Path

import astropy.units as u
import eazy
import h5py
import numpy as np
from astropy.table import Table
from eazy import hdf5
from scipy.linalg import LinAlgWarning
from tqdm import tqdm
from typing import TYPE_CHECKING, List, Any, Dict, Optional, NoReturn, Type, Union, Tuple
if TYPE_CHECKING:
    from . import Catalogue, PDF
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

warnings.filterwarnings("ignore", category=LinAlgWarning)

from . import Redshift_PDF, SED_code, config, galfind_logger
from . import useful_funcs_austind as funcs
from .decorators import run_in_dir
from .SED import SED_obs

# %% EAZY SED fitting code

# TODO: update these at runtime
EAZY_FILTER_CODES = {
    "NIRCam": {
        "F070W": 36,
        "F090W": 1,
        "F115W": 2,
        "F140M": 37,
        "F150W": 3,
        "F162M": 38,
        "F182M": 39,
        "F200W": 4,
        "F210M": 40,
        "F250M": 41,
        "F277W": 5,
        "F300M": 42,
        "F335M": 43,
        "F356W": 6,
        "F360M": 44,
        "F410M": 7,
        "F430M": 45,
        "F444W": 8,
        "F460M": 46,
        "F480M": 47,
    },
    "ACS_WFC": {
        "F435W": 22,
        "F606W": 23,
        "F625W": 48,
        "F775W": 49,
        "F850LP": 50,
        "F814W": 24,
        "F105W": 25,
        "F125W": 26,
        "F140W": 27,
        "F150W": 28,
    },
    "MIRI": {
        "F560W": 13,
        "F770W": 14,
        "F1000W": 15,
        "F1130W": 16,
        "F1280W": 17,
        "F1500W": 18,
        "F1800W": 19,
        "F2100W": 20,
        "F2550W": 21,
    },
}


class EAZY(SED_code):
    #ext_src_corr_properties = []
    def __init__(self: Self, SED_fit_params: Dict[str, Any]):
        super().__init__(SED_fit_params)

    @classmethod
    def from_label(cls, label: str) -> Type[SED_code]:
        label_arr = label.split("_")
        templates = "_".join(
            label_arr[1:-1]
        )  # templates may contain underscore
        SED_fit_params = {"templates": templates,
            "lowz_zmax": funcs.zmax_from_lowz_label(label_arr[-1])}
        return cls(SED_fit_params)

    @property
    def ID_label(self) -> str:
        return "IDENT"

    @property
    def label(self) -> str:
        # first write the code name, next write the template name, finish off with lowz_zmax
        return f"{self.__class__.__name__}_{self.SED_fit_params['templates']}" + \
            f"_{funcs.lowz_label(self.SED_fit_params['lowz_zmax'])}"

    @property
    def hdu_name(self) -> str:
        return f"{self.__class__.__name__}_{self.SED_fit_params['templates']}"
    
    @property
    def tab_suffix(self) -> str:
        return f"{self.SED_fit_params['templates']}_" + \
            f"{funcs.lowz_label(self.SED_fit_params['lowz_zmax'])}"
    
    @property
    def required_SED_fit_params(self) -> List[str]:
        return ["templates", "lowz_zmax"]
    
    @property
    def are_errs_percentiles(self) -> bool:
        return False

    def _load_gal_property_labels(self):
        gal_property_labels = {
            **{"z": "zbest", "chi_sq": "chi2_best"},
            **{
                f"{ubvj_filt}_flux": f"{ubvj_filt}_rf_flux"
                for ubvj_filt in ["U", "B", "V", "J"]
            },
        }
        super()._load_gal_property_labels(gal_property_labels)

    def _load_gal_property_err_labels(self):
        gal_property_err_labels = {
            f"{ubvj_filt}_flux": [
                f"{ubvj_filt}_rf_flux_err",
                f"{ubvj_filt}_rf_flux_err",
            ]
            for ubvj_filt in ["U", "B", "V", "J"]
        }
        super()._load_gal_property_err_labels(gal_property_err_labels)

    def _load_gal_property_units(self) -> NoReturn:
        # still need to double check the UBVJ units
        self.gal_property_units = {
            **{gal_property: u.dimensionless_unscaled for gal_property in ["z", "chi_sq"]},
            **{f"{ubvj_filt}_flux": u.nJy for ubvj_filt in ["U", "B", "V", "J"]},
        }

    def _assert_SED_fit_params(self):
        default_strings = ["N_PROC", "Z_STEP", "Z_MIN", "Z_MAX", "SAVE_UBVJ"]
        default_types = [int, float, float, float, bool]
        for default_str, default_type in zip(default_strings, default_types):
            if default_str == "Z_MAX":
                if self.SED_fit_params["lowz_zmax"] is None:
                    self.SED_fit_params["Z_MAX"] = config.getfloat("EAZY", "Z_MAX")
                else:
                    self.SED_fit_params["Z_MAX"] = self.SED_fit_params["lowz_zmax"]
            else:
                if default_str not in self.SED_fit_params.keys():
                    if default_type == bool:
                        self.SED_fit_params[default_str] = config.getboolean("EAZY", default_str)
                    elif default_type == int:
                        self.SED_fit_params[default_str] = config.getint("EAZY", default_str)
                    elif default_type == float:
                        self.SED_fit_params[default_str] = config.getfloat("EAZY", default_str)
        return super()._assert_SED_fit_params()

    def make_in(
        self, 
        cat: Catalogue, 
        aper_diam: u.Quantity, 
        overwrite: bool = False
    ) -> str:
        in_dir = f"{config['EAZY']['EAZY_DIR']}/input/{cat.filterset.instrument_name}/{cat.version}/{cat.survey}"
        in_name = cat.cat_name.replace('.fits', f"_{aper_diam.to(u.arcsec).value:.2f}as.in")
        in_path = f"{in_dir}/{in_name}"
        if not Path(in_path).is_file() or overwrite:
            # 1) obtain input data
            IDs = np.array([gal.ID for gal in cat.gals])  # load IDs
            redshifts = np.array([-99.0 for i in range(len(cat))]) # TODO: Load spec-z's
            # load photometry
            phot, phot_err = self._load_phot(cat, aper_diam, u.uJy, -99.0, None)
            # Get filter codes (referenced to GALFIND/EAZY/jwst_nircam_FILTER.RES.info) for the given instrument and bands
            filt_codes = [
                EAZY_FILTER_CODES[band.instrument_name][band.band_name]
                for band in cat.filterset
            ]
            # Make input file
            in_data = np.array(
                [
                    np.concatenate(
                        (
                            [IDs[i]],
                            list(itertools.chain(*zip(phot[i], phot_err[i]))),
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
                            [f"F{filt_code}" for filt_code in filt_codes],
                            [f"E{filt_code}" for filt_code in filt_codes],
                        )
                    )
                )
                + ["z_spec"]
            )
            in_types = (
                [int]
                + list(np.full(len(cat.filterset.band_names) * 2, float))
                + [float]
            )
            in_tab = Table(in_data, dtype=in_types, names=in_names)
            funcs.make_dirs(in_path)
            in_tab.write(
                in_path,
                format="ascii.commented_header",
                delimiter=" ",
                overwrite=True,
            )
            funcs.change_file_permissions(in_path)
        return in_path

    @run_in_dir(path=config["EAZY"]["EAZY_DIR"])
    def fit(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity,
        save_SEDs: bool = True,
        save_PDFs: bool = True,
        overwrite: bool = False,
        update: bool = False,
        **kwargs: Dict[str, Any],
    ) -> NoReturn:
        """
        z_step  - redshift step size - default 0.01
        z_min - minimum redshift to fit - default 0
        z_max - maximum redshift to fit - default 25.
        save_SEDs - whether to write out best-fitting SEDs. Default True.
        save_PDFs - Whether to write out redshift PDF. Default True.
        save_plots - whether to save SED plots - default False. Use in conjunction with plot_ids to plot SEDS of specific ids.
        save_ubvj - whether to save restframe UBVJ fluxes -default True.
        **kwargs - additional arguments to pass to EAZY to overide defaults
        """
        # Change this to config file path
        # This if/else tree chooses which template file to use based on 'templates' argument
        # FSPS - default EAZY templates, good allrounders
        # fsps_larson - default here, optimized for high redshift (see Larson et al. 2023)
        # HOT_45K - modified IMF high-z templates for use between 8 < z < 12
        # HOT_60K - modified IMF high-z templates for use at z > 12
        # Nakajima - unobscured AGN templates

        in_path, out_path, fits_out_path, PDF_paths, SED_paths = self._get_out_paths(cat, aper_diam)

        templates = self.SED_fit_params["templates"]

        os.makedirs("/".join(fits_out_path.split("/")[:-1]), exist_ok=True)
        h5_path = fits_out_path.replace(".fits", ".h5")
        zPDF_path = h5_path.replace(".h5", "_zPDFs.h5")
        SED_path = h5_path.replace(".h5", "_SEDs.h5")
        lowz_label = funcs.lowz_label(self.SED_fit_params["lowz_zmax"])

        eazy_templates_path = config["EAZY"]["EAZY_TEMPLATE_DIR"]
        default_param_path = (
            f"{config['EAZY']['EAZY_CONFIG_DIR']}/zphot.param.default"
        )
        translate_file = (
            f"{config['EAZY']['EAZY_CONFIG_DIR']}/zphot_jwst.translate"
        )

        params = {}
        z_min = self.SED_fit_params["Z_MIN"]
        z_max = self.SED_fit_params["Z_MAX"]
        if templates == "fsps_larson":
            params["TEMPLATES_FILE"] = (
                f"{eazy_templates_path}/LarsonTemplates/tweak_fsps_QSF_12_v3_newtemplates.param"
            )
        elif templates == "BC03":
            # This path is broken
            params["TEMPLATES_FILE"] = (
                f"{eazy_templates_path}/bc03_chabrier_2003.param"
            )
        elif templates == "fsps":
            params["TEMPLATES_FILE"] = (
                f"{eazy_templates_path}/fsps_full/tweak_fsps_QSF_12_v3.param"
            )
        elif templates == "nakajima_full":
            params["TEMPLATES_FILE"] = (
                f"{eazy_templates_path}/Nakajima2022/tweak_fsps_QSF_12_v3_larson_nakajima_all.param"
            )
        elif templates == "nakajima_subset":
            params["TEMPLATES_FILE"] = (
                f"{eazy_templates_path}/Nakajima2022/tweak_fsps_QSF_12_v3_larson_nakajima_subset.param"
            )
        elif templates == "fsps_jades":
            params["TEMPLATES_FILE"] = (
                f"{eazy_templates_path}/jades/jades.param"
            )
        elif templates == "HOT_45K":
            params["TEMPLATES_FILE"] = (
                f"{eazy_templates_path}/fsps-hot/45k/fsps_45k.param"
            )
            z_min = 8
            z_max = 12
        elif templates == "HOT_60K":
            params["TEMPLATES_FILE"] = (
                f"{eazy_templates_path}/fsps-hot/60k/fsps_60k.param"
            )
            z_min = 12
            z_max = 25
        # Redshift limits
        params["Z_MIN"] = z_min # Setting minimum Z
        params["Z_MAX"] = z_max # Setting maximum Z
        params["Z_STEP"] = self.SED_fit_params["Z_STEP"] # Setting photo-z step

        # Next section deals with passing config parameters into EAZY config dictionary
        # JWST filter_file
        params["FILTERS_RES"] = (
            f"{config['EAZY']['EAZY_CONFIG_DIR']}/jwst_nircam_FILTER.RES"
        )

        # Errors
        params["WAVELENGTH_FILE"] = (
            f"{eazy_templates_path}/lambda.def"  # Wavelength grid definition file
        )
        params["TEMP_ERR_FILE"] = (
            f"{eazy_templates_path}/TEMPLATE_ERROR.eazy_v1.0"  # Template error definition file
        )

        # Priors
        # TODO: Load in and fix specific galaxies to spec-z's
        #params["FIX_ZSPEC"] = fix_z

        # Input files
        # -------------------------------------------------------------------------------------------------------------

        # Defining in/out files
        params["CATALOG_FILE"] = in_path
        params["MAIN_OUTPUT_FILE"] = fits_out_path
        params["OUTPUT_DIRECTORY"] = "/".join(fits_out_path.split("/")[:-1])

        # Pass in optional arguments
        params.update(kwargs)

        if not Path(h5_path).is_file() or overwrite:
            # Initialize photo-z object with above parameters
            galfind_logger.info(
                f"Running {self.__class__.__name__} {templates} {lowz_label}"
            )
            fit = eazy.photoz.PhotoZ(
                param_file=default_param_path,
                zeropoint_file=None,
                params=params,
                load_prior=False,
                load_products=False,
                translate_file=translate_file,
                n_proc=self.SED_fit_params["N_PROC"],
            )
            
            fit.fit_catalog(n_proc=self.SED_fit_params["N_PROC"], get_best_fit=True)
            # Save backup of fit in hdf5 file
            hdf5.write_hdf5(
                fit,
                h5file=h5_path,
                include_fit_coeffs=False,
                include_templates=True,
                verbose=False,
            )
            galfind_logger.info(
                f"Finished running {self.__class__.__name__} {templates} {lowz_label}"
            )
        elif (
            not Path(fits_out_path).is_file()
            or not Path(zPDF_path).is_file()
            or not Path(SED_path).is_file()
        ):
            # load in .h5 file
            fit = hdf5.initialize_from_hdf5(h5file=h5_path, verbose=True)
        else:
            fit = None

        if not Path(fits_out_path).is_file() and fit is not None:
            # If not using Fsps larson, use standard saving output. Otherwise generate own fits file.
            if templates == "HOT_45K" or templates == "HOT_60K":
                fit.standard_output(
                    UBVJ=(9, 10, 11, 12),
                    absmag_filters=[9, 10, 11, 12],
                    extra_rf_filters=[9, 10, 11, 12],
                    n_proc=self.SED_fit_params["N_PROC"],
                    save_fits=1,
                    get_err=True,
                    simple=False,
                )
            else:
                colnames = [
                    "IDENT",
                    "zbest",
                    "zbest_16",
                    "zbest_84",
                    "chi2_best",
                ]
                data = [
                    fit.OBJID,
                    fit.zbest,
                    fit.pz_percentiles([16]),
                    fit.pz_percentiles([84]),
                    fit.chi2_best,
                ]

                table = Table(data=data, names=colnames)

                # Get rest frame colors
                if self.SED_fit_params["SAVE_UBVJ"]:
                    # This is all duplicated from base code.
                    rf_tempfilt, lc_rest, ubvj = fit.rest_frame_fluxes(
                        f_numbers=[9, 10, 11, 12], simple=False, n_proc=self.SED_fit_params["N_PROC"]
                    )
                    for i, ubvj_filt in enumerate(["U", "B", "V", "J"]):
                        table[f"{ubvj_filt}_rf_flux"] = ubvj[:, i, 2]
                        # symmetric errors
                        table[f"{ubvj_filt}_rf_flux_err"] = (
                            ubvj[:, i, 3] - ubvj[:, i, 1]
                        ) / 2.0
                    galfind_logger.info(
                        f"Finished calculating UBVJ fluxes for {self.__class__.__name__} {templates} {lowz_label}"
                    )

                # add the template name to the column labels except for IDENT
                for col_name in table.colnames:
                    if col_name != self.ID_label:
                        table.rename_column(
                            col_name,
                            f"{col_name}_{self.tab_suffix}",
                        )
                # Write fits file
                table.write(fits_out_path, overwrite=True)
                funcs.change_file_permissions(fits_out_path)
                galfind_logger.info(
                    f"Written {self.__class__.__name__} {templates} {lowz_label} fits out file to: {fits_out_path}"
                )
        else:
            table = Table.read(fits_out_path)

        # save PDFs in .h5 file
        if save_PDFs and not Path(zPDF_path).is_file():
            self.save_zPDFs(zPDF_path, fit)
            galfind_logger.info(
                f"Finished saving z-PDFs for {self.__class__.__name__} {templates} {lowz_label}"
            )

        # Save best-fitting SEDs
        if save_SEDs and not Path(SED_path).is_file():
            z_arr = np.array(table[f"zbest_{templates}_{lowz_label}"]).astype(
                float
            )
            self.save_SEDs(SED_path, fit, z_arr, u.AA, u.nJy)
            galfind_logger.info(
                f"Finished saving SEDs for {self.__class__.__name__} {templates} {lowz_label}"
            )

        # Write used parameters
        if fit is not None:
            fit.param.write(fits_out_path.replace(".fits", "_params.csv"))
            funcs.change_file_permissions(
                fits_out_path.replace(".fits", "_params.csv")
            )
            galfind_logger.info(
                f"Written output pararmeters for {self.__class__.__name__} {templates} {lowz_label}"
            )

    @staticmethod
    def save_zPDFs(zPDF_path: str, fit) -> NoReturn:
        fit_pz = 10 ** (fit.lnp)
        fit_zgrid = fit.zgrid
        hf = h5py.File(zPDF_path, "w")
        hf.create_dataset("z", data=np.array(fit.zgrid).astype(np.float32), compression="gzip", dtype="f4")
        pz_arr = np.array(
            [
                np.array(
                    [
                        np.array(fit_pz[pos_obj][pos])
                        for pos, z in enumerate(fit_zgrid)
                    ]
                )
                for pos_obj, ID in tqdm(
                    enumerate(fit.OBJID),
                    total=len(fit.OBJID),
                    desc="Saving z-PDFs",
                )
            ]
        )
        hf.create_dataset("p_z_arr", data=pz_arr, compression="gzip")
        hf.close()

    @staticmethod
    def save_SEDs(SED_path: str, fit, z_arr: List[float], wav_unit: u.Unit = u.AA, flux_unit: u.Unit = u.nJy) -> NoReturn:
        hf = h5py.File(SED_path, "w")
        hf.create_dataset("wav_unit", data=str(wav_unit))
        hf.create_dataset("flux_unit", data=str(flux_unit))
        hf.create_dataset("z_arr", data=z_arr, compression="gzip")
        # Load best-fitting SEDs
        fit_data_arr = [
            fit.show_fit(
                ID,
                id_is_idx=False,
                show_components=False,
                show_prior=False,
                logpz=False,
                get_spec=True,
                show_fnu=1,
            )
            for ID in tqdm(
                fit.OBJID,
                desc="Creating fit_data_arr for SED saving",
                total=len(fit.OBJID),
            )
        ]
        wav_flux_arr = [
            [
                (np.array(fit_data["templz"]) * fit_data["wave_unit"]).to(
                    wav_unit
                ),
                (np.array(fit_data["templf"]) * fit_data["flux_unit"]).to(
                    flux_unit
                ),
            ]
            for fit_data in tqdm(
                fit_data_arr,
                desc="Creating wav_flux_arr",
                total=len(fit_data_arr),
            )
        ]
        wav_flux_arr = np.array(wav_flux_arr).astype(np.float32)
        hf.create_dataset("wav_flux_arr", data=wav_flux_arr, compression="gzip", dtype="f4")
        hf.close()

    def make_fits_from_out(self, out_path: str) -> NoReturn:
        pass

    def _get_out_paths(
        self: Self, 
        cat: Catalogue, 
        aper_diam: u.Quantity
    ) -> Tuple[str, str, str, Dict[str, List[str]], List[str]]:
        in_dir = f"{config['EAZY']['EAZY_DIR']}/input/{cat.filterset.instrument_name}/{cat.version}/{cat.survey}"
        in_name = cat.cat_name.replace('.fits', f"_{aper_diam.to(u.arcsec).value:.2f}as.in")
        in_path = f"{in_dir}/{in_name}"
        out_folder = funcs.split_dir_name(
            in_path.replace("input", "output"), "dir"
        )
        out_path = f"{out_folder}/{funcs.split_dir_name(in_path, 'name').replace('.in', '.out')}"
        fits_out_path = f"{out_path.replace('.out', '')}_EAZY_{self.SED_fit_params['templates']}_{funcs.lowz_label(self.SED_fit_params['lowz_zmax'])}.fits"
        IDs = [gal.ID for gal in cat.gals]
        PDF_paths = {
            "z": list(
                np.full(len(IDs), fits_out_path.replace(".fits", "_zPDFs.h5"))
            )
        }
        SED_paths = list(
            np.full(len(IDs), fits_out_path.replace(".fits", "_SEDs.h5"))
        )
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
        assert len(IDs) == len(SED_paths), \
            galfind_logger.critical(
                f"{len(IDs)=} != {len(SED_paths)=}"
            )
        # ensure that for EAZY all the SED_paths are the same
        assert all(
            SED_path == SED_paths[0] for SED_path in SED_paths
        ), galfind_logger.critical(
            f"SED_paths must all be the same for {__class__.__name__}"
        )
        # open .h5 file
        # return np.ones(len(IDs))
        hf = h5py.File(SED_paths[0], "r")
        IDs_np = np.array(IDs)
        z_arr = hf["z_arr"][IDs_np - 1].astype(np.float32)
        wav_flux_arr = hf["wav_flux_arr"][IDs_np - 1].astype(np.float32)
        wav_arr = wav_flux_arr[:, 0].astype(np.float32)
        flux_arr = wav_flux_arr[:, 1].astype(np.float32)
        wav_unit = u.Unit(hf["wav_unit"][()].decode())
        flux_unit = u.Unit(hf["flux_unit"][()].decode())
        hf.close()
        SED_obs_arr = [
            SED_obs(z, wav, flux, wav_unit, flux_unit)
            for z, wav, flux in tqdm(
                zip(z_arr, wav_arr, flux_arr),
                total=len(z_arr),
                desc="Constructing SEDs",
            )
        ]
        # close .h5 file
        return SED_obs_arr

    def extract_PDFs(
        self: Self, 
        gal_property: str, 
        IDs: List[int], 
        PDF_paths: Union[str, List[str]], 
    ) -> List[Redshift_PDF]:
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
                    f"len(IDs) = {len(IDs)} != len(data_paths) = {len(PDF_paths)}!"
                )
            # ensure all data_paths are the same and are of .h5 type
            assert all(
                PDF_path == PDF_paths[0] for PDF_path in PDF_paths
            ), galfind_logger.critical("All data_paths must be the same!")
            assert PDF_paths[0][-3:] == ".h5", galfind_logger.critical(
                f"{PDF_paths[0]} must have .h5 file extension"
            )
            # open .h5 file
            hf = h5py.File(PDF_paths[0], "r")
            hf_z = np.array(hf["z"]) * u.dimensionless_unscaled
            pz_arr = hf["p_z_arr"][np.array(IDs) - 1]
            # extract redshift PDF for each ID
            redshift_PDFs = [
                Redshift_PDF(hf_z, pz, self.SED_fit_params, normed=False)
                for ID, pz in tqdm(
                    zip(IDs, pz_arr),
                    total=len(IDs),
                    desc="Constructing redshift PDFs",
                )
            ]
            # close .h5 file
            hf.close()
            return redshift_PDFs

    def load_cat_property_PDFs(
            self: Self, 
            PDF_paths: List[Dict[str, str]],
            IDs: List[int]
        ) -> List[Dict[str, Optional[Type[PDF]]]]:
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
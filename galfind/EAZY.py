#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:52:36 2023

@author: austind
"""

# EAZY.py
import numpy as np
import astropy.units as u
import itertools
from astropy.table import Table
from pathlib import Path
import eazy
import os
import time
import warnings
from tqdm import tqdm
import h5py
from eazy import hdf5
from scipy.linalg import LinAlgWarning

warnings.filterwarnings("ignore", category=LinAlgWarning)

from . import SED_code, Redshift_PDF
from .SED import SED_obs
from . import useful_funcs_austind as funcs
from . import config, galfind_logger
from .decorators import run_in_dir

# %% EAZY SED fitting code

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
    gal_property_fmt_dict = {
        **{"z": "Redshift, z"},
        **{f"{ubvj_filt}_flux": ubvj_filt for ubvj_filt in ["U", "B", "V", "J"]},
    }
    # still need to double check the UBVJ units
    gal_property_unit_dict = {
        **{"z": u.dimensionless_unscaled},
        **{f"{ubvj_filt}_flux": u.nJy for ubvj_filt in ["U", "B", "V", "J"]},
    }
    galaxy_property_dict = {
        **{"z": "zbest", "chi_sq": "chi2_best"},
        **{
            f"{ubvj_filt}_flux": f"{ubvj_filt}_rf_flux"
            for ubvj_filt in ["U", "B", "V", "J"]
        },
    }
    galaxy_property_errs_dict = {
        f"{ubvj_filt}_flux": [f"{ubvj_filt}_rf_flux_err", f"{ubvj_filt}_rf_flux_err"]
        for ubvj_filt in ["U", "B", "V", "J"]
    }
    available_templates = ["fsps", "fsps_larson", "fsps_jades"]
    ext_src_corr_properties = []
    ID_label = "IDENT"
    are_errs_percentiles = False

    def __init__(self, SED_fit_params=None):
        # EAZY specific SED fit params assertions here
        super().__init__(
            SED_fit_params,
            self.galaxy_property_dict,
            self.galaxy_property_errs_dict,
            self.available_templates,
            self.ID_label,
            self.are_errs_percentiles,
        )

    def galaxy_property_labels(self, gal_property, SED_fit_params, is_err=False):
        assert (
            "templates" in SED_fit_params.keys()
            and "lowz_zmax" in SED_fit_params.keys()
        )
        if SED_fit_params["templates"] not in self.available_templates:
            raise (
                Exception(
                    f"templates = {SED_fit_params['templates']} are not in {self.available_templates}, and hence are not yet encorporated for galfind EAZY SED fitting"
                )
            )

        return_suffix = f"{SED_fit_params['templates']}_{funcs.lowz_label(SED_fit_params['lowz_zmax'])}"
        if gal_property in self.galaxy_property_dict.keys() and not is_err:
            return f"{self.galaxy_property_dict[gal_property]}_{return_suffix}"
        elif gal_property in self.galaxy_property_errs_dict.keys() and is_err:
            return [
                f"{self.galaxy_property_errs_dict[gal_property][0]}_{return_suffix}",
                f"{self.galaxy_property_errs_dict[gal_property][1]}_{return_suffix}",
            ]
            # if gal_property in property_dict.keys():
            #     return f"{property_dict[gal_property]}_{return_suffix}"
            # else:
            #    raise(Exception(f"{self.__class__.__name__}.galaxy_property_{'errs_' if is_err else ''}dict = {property_dict} does not include key for gal_property = {gal_property}!"))
        else:
            return f"{gal_property}_{return_suffix}"

    def make_in(self, cat, fix_z=False):  # , *args, **kwargs):
        eazy_in_dir = f"{config['EAZY']['EAZY_DIR']}/input/{cat.instrument.name}/{cat.version}/{cat.survey}"
        eazy_in_path = f"{eazy_in_dir}/{cat.cat_name.replace('.fits', '.in')}"
        if not Path(eazy_in_path).is_file():
            # 1) obtain input data
            IDs = np.array([gal.ID for gal in cat.gals])  # load IDs
            # load redshifts
            if not fix_z:
                redshifts = np.array([-99.0 for gal in cat.gals])
            else:
                redshifts = None
            # load photometry
            phot, phot_err = self.load_photometry(cat, u.uJy, -99.0, None)
            # Get filter codes (referenced to GALFIND/EAZY/jwst_nircam_FILTER.RES.info) for the given instrument and bands
            filt_codes = [
                EAZY_FILTER_CODES[band.instrument][band.band_name]
                for band in cat.instrument
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
                + list(np.full(len(cat.instrument.band_names) * 2, float))
                + [float]
            )
            in_tab = Table(in_data, dtype=in_types, names=in_names)
            funcs.make_dirs(eazy_in_path)
            in_tab.write(
                eazy_in_path,
                format="ascii.commented_header",
                delimiter=" ",
                overwrite=True,
            )
            funcs.change_file_permissions(eazy_in_path)

    @run_in_dir(path=config["EAZY"]["EAZY_DIR"])
    def run_fit(
        self,
        in_path,
        fits_out_path,
        instrument,
        SED_fit_params,
        fix_z=False,
        n_proc=1,
        z_step=0.01,
        z_min=0,
        z_max=25,
        save_best_seds=config.getboolean("EAZY", "SAVE_SEDS"),
        save_PDFs=config.getboolean("EAZY", "SAVE_PDFS"),
        write_hdf=True,
        save_plots=False,
        plot_ids=None,
        plot_all=False,
        save_ubvj=True,
        run_lowz=True,
        wav_unit=u.AA,
        flux_unit=u.nJy,
        overwrite=False,
        *args,
        **kwargs,
    ):
        """
        in_path - input EAZY catalogue path
        fits_out_path - output EAZY catalogue path
        template - which EAZY template to use - see below for list
        fix_z - whether to fix photo-z or not
        z_step  - redshift step size - default 0.01
        z_min - minimum redshift to fit - default 0
        z_max - maximum redshift to fit - default 25.
        save_best_seds - whether to write out best-fitting SEDs. Default True.
        save_PDFs - Whether to write out redshift PDF. Default True.
        write_hdf - whether to backup output to hdf5 - default True
        save_plots - whether to save SED plots - default False. Use in conjunction with plot_ids to plot SEDS of specific ids.
        plot_ids - list of ids to plot if save_plots is True.
        plot_all - whether to plot all SEDs. Default False.
        save_ubvj - whether to save restframe UBVJ fluxes -default True.
        run_lowz - whether to run low-z fit. Default True.
        lowz_zmax_arr - maximum redshifts to fit in low-z fits. Default [4., 6., None]
        **kwargs - additional arguments to pass to EAZY to overide defaults
        """
        # Change this to config file path
        # This if/else tree chooses which template file to use based on 'templates' argument
        # FSPS - default EAZY templates, good allrounders
        # fsps_larson - default here, optimized for high redshift (see Larson et al. 2023)
        # HOT_45K - modified IMF high-z templates for use between 8 < z < 12
        # HOT_60K - modified IMF high-z templates for use at z > 12
        # Nakajima - unobscured AGN templates

        assert "lowz_zmax" in SED_fit_params.keys(), galfind_logger.critical(
            f"'lowz_zmax' not in SED_fit_params keys = {np.array(SED_fit_params.keys())}"
        )

        templates = SED_fit_params["templates"]
        lowz_zmax = SED_fit_params["lowz_zmax"]

        os.makedirs("/".join(fits_out_path.split("/")[:-1]), exist_ok=True)
        h5_path = fits_out_path.replace(".fits", ".h5")
        zPDF_path = h5_path.replace(".h5", "_zPDFs.h5")
        SED_path = h5_path.replace(".h5", "_SEDs.h5")
        lowz_label = funcs.lowz_label(lowz_zmax)

        eazy_templates_path = config["EAZY"]["EAZY_TEMPLATE_DIR"]
        default_param_path = (
            f"{config['DEFAULT']['GALFIND_DIR']}/configs/zphot.param.default"
        )
        translate_file = (
            f"{config['DEFAULT']['GALFIND_DIR']}/configs/zphot_jwst.translate"
        )
        params = {}
        if templates == "fsps_larson":
            params["TEMPLATES_FILE"] = (
                f"{eazy_templates_path}/LarsonTemplates/tweak_fsps_QSF_12_v3_newtemplates.param"
            )
        elif templates == "BC03":
            # This path is broken
            params["TEMPLATES_FILE"] = f"{eazy_templates_path}/bc03_chabrier_2003.param"
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
            params["TEMPLATES_FILE"] = f"{eazy_templates_path}/jades/jades.param"
        elif templates == "HOT_45K":
            params["TEMPLATES_FILE"] = (
                f"{eazy_templates_path}/fsps-hot/45k/fsps_45k.param"
            )
            z_min = 8
            z_max = 12
            # print(f'Running HOT 45K with fixed redshift = {fix_z}')
            if not fix_z:
                # print('Fixing 8<z<12')
                pass
        elif templates == "HOT_60K":
            params["TEMPLATES_FILE"] = (
                f"{eazy_templates_path}/fsps-hot/60k/fsps_60k.param"
            )
            z_min = 12
            z_max = 25
            # print(f'Running HOT 45K with fixed redshift = {fix_z}')
            if not fix_z:
                # print('Fixing 12<z<25')
                pass

        # Next section deals with passing config parameters into EAZY config dictionary
        # JWST filter_file
        params["FILTERS_RES"] = (
            f"{config['DEFAULT']['GALFIND_DIR']}/configs/jwst_nircam_FILTER.RES"
        )

        # Redshift limits
        params["Z_STEP"] = z_step  # Setting photo-z step
        params["Z_MIN"] = z_min  # Setting minimum Z
        if lowz_zmax == None:
            params["Z_MAX"] = z_max  # Setting maximum Z
        else:
            params["Z_MAX"] = lowz_zmax  # Setting maximum Z

        # Errors
        params["WAVELENGTH_FILE"] = (
            f"{eazy_templates_path}/lambda.def"  # Wavelength grid definition file
        )
        params["TEMP_ERR_FILE"] = (
            f"{eazy_templates_path}/TEMPLATE_ERROR.eazy_v1.0"  # Template error definition file
        )

        # Priors
        params["FIX_ZSPEC"] = fix_z  # Fix redshift to catalog zspec

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
                n_proc=n_proc,
            )
            fit.fit_catalog(n_proc=n_proc, get_best_fit=True)
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

        if not Path(fits_out_path).is_file() and fit != None:
            # If not using Fsps larson, use standard saving output. Otherwise generate own fits file.
            if templates == "HOT_45K" or templates == "HOT_60K":
                fit.standard_output(
                    UBVJ=(9, 10, 11, 12),
                    absmag_filters=[9, 10, 11, 12],
                    extra_rf_filters=[9, 10, 11, 12],
                    n_proc=n_proc,
                    save_fits=1,
                    get_err=True,
                    simple=False,
                )
            else:
                colnames = ["IDENT", "zbest", "zbest_16", "zbest_84", "chi2_best"]
                data = [
                    fit.OBJID,
                    fit.zbest,
                    fit.pz_percentiles([16]),
                    fit.pz_percentiles([84]),
                    fit.chi2_best,
                ]

                table = Table(data=data, names=colnames)

                # Get rest frame colors
                if save_ubvj:
                    # This is all duplicated from base code.
                    rf_tempfilt, lc_rest, ubvj = fit.rest_frame_fluxes(
                        f_numbers=[9, 10, 11, 12], simple=False, n_proc=n_proc
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
                    if col_name != "IDENT":
                        table.rename_column(
                            col_name,
                            self.galaxy_property_labels(col_name, SED_fit_params),
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
            # [self.save_zPDF(pos_obj, ID, hf, fit.zgrid, pz) for pos_obj, ID in \
            #    tqdm(enumerate(fit.OBJID), total = len(fit.OBJID), \
            #    desc = f"Saving z-PDFs for {self.__class__.__name__} {templates} {lowz_label}")]
            galfind_logger.info(
                f"Finished saving z-PDFs for {self.__class__.__name__} {templates} {lowz_label}"
            )

        # Save best-fitting SEDs
        if save_best_seds and not Path(SED_path).is_file():
            z_arr = np.array(table[f"zbest_{templates}_{lowz_label}"]).astype(float)
            self.save_SEDs(SED_path, fit, z_arr, wav_unit, flux_unit)
            # [self.save_SED(ID, z, hf, fit, wav_unit = wav_unit, flux_unit = flux_unit) \
            #    for ID, z in tqdm(zip(fit.OBJID, np.array(table[f"zbest_{templates}_{lowz_label}"]).astype(float)), total = len(fit.OBJID), \
            #    desc = f"Saving best-fit template SEDs for {self.__class__.__name__} {templates} {lowz_label}")]
            galfind_logger.info(
                f"Finished saving SEDs for {self.__class__.__name__} {templates} {lowz_label}"
            )

        # Write used parameters
        if fit != None:
            fit.param.write(fits_out_path.replace(".fits", "_params.csv"))
            funcs.change_file_permissions(fits_out_path.replace(".fits", "_params.csv"))
            galfind_logger.info(
                f"Written output pararmeters for {self.__class__.__name__} {templates} {lowz_label}"
            )

    # @staticmethod
    # def save_zPDF(pos_obj, ID, hf, fit_zgrid, fit_pz):
    #    #hf.create_dataset(f"ID={int(ID)}_p(z)", data = np.array([fit_pz[pos_obj][pos] for pos, z in enumerate(fit_zgrid)]))
    #    hf.create_dataset(f"ID={int(ID)}_p(z)", data = np.array([np.array(fit_pz[pos_obj][pos]) for pos, z in enumerate(fit_zgrid)]))

    @staticmethod
    def save_zPDFs(zPDF_path, fit):
        fit_pz = 10 ** (fit.lnp)
        fit_zgrid = fit.zgrid
        hf = h5py.File(zPDF_path, "w")
        hf.create_dataset("z", data=np.array(fit.zgrid))
        pz_arr = np.array(
            [
                np.array(
                    [np.array(fit_pz[pos_obj][pos]) for pos, z in enumerate(fit_zgrid)]
                )
                for pos_obj, ID in tqdm(
                    enumerate(fit.OBJID), total=len(fit.OBJID), desc="Saving z-PDFs"
                )
            ]
        )
        hf.create_dataset("p_z_arr", data=pz_arr)
        hf.close()

    @staticmethod
    def save_SEDs(SED_path, fit, z_arr, wav_unit=u.AA, flux_unit=u.nJy):
        hf = h5py.File(SED_path, "w")
        hf.create_dataset("wav_unit", data=str(wav_unit))
        hf.create_dataset("flux_unit", data=str(flux_unit))
        hf.create_dataset("z_arr", data=z_arr)
        # assert(fit.OBJID == np.sort(fit.OBJID))
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
                (np.array(fit_data["templz"]) * fit_data["wave_unit"]).to(wav_unit),
                (np.array(fit_data["templf"]) * fit_data["flux_unit"]).to(flux_unit),
            ]
            for fit_data in tqdm(
                fit_data_arr, desc="Creating wav_flux_arr", total=len(fit_data_arr)
            )
        ]
        # flux_arr = [(np.array(fit_data['templf']) * fit_data['flux_unit']).to(flux_unit) for fit_data in fit_data_arr]
        # wav_flux_arr = [[wav, flux] for wav, flux in zip(wav_arr, flux_arr)]
        hf.create_dataset("wav_flux_arr", data=wav_flux_arr)
        # gal_SED = hf.create_group(f"ID={int(fit_data['id'])}")
        # gal_SED.create_dataset("z", data = z)
        # gal_SED.create_dataset("wav", data = wav)
        # gal_SED.create_dataset("flux", data = flux)
        hf.close()

    @staticmethod
    def label_from_SED_fit_params(SED_fit_params):
        assert (
            "code" in SED_fit_params.keys()
            and "templates" in SED_fit_params.keys()
            and "lowz_zmax" in SED_fit_params.keys()
        )
        # first write the code name, next write the template name, finish off with lowz_zmax
        return f"{SED_fit_params['code'].__class__.__name__}_{SED_fit_params['templates']}_{funcs.lowz_label(SED_fit_params['lowz_zmax'])}"

    @staticmethod
    def hdu_from_SED_fit_params(SED_fit_params):
        return EAZY.label_from_SED_fit_params(SED_fit_params).replace(
            f"_{funcs.lowz_label(SED_fit_params['lowz_zmax'])}", ""
        )  # remove lowz_zmax label from suffix

    def SED_fit_params_from_label(self, label):
        label_arr = label.split("_")
        templates = "_".join(label_arr[1:-1])  # templates may contain underscore
        assert templates in self.available_templates
        return {
            "code": self,
            "templates": templates,
            "lowz_zmax": funcs.zmax_from_lowz_label(label_arr[-1]),
        }

    def make_fits_from_out(self, out_path, SED_fit_params):  # *args, **kwargs):
        pass

    @staticmethod
    def get_out_paths(cat, SED_fit_params, IDs):  # *args, **kwargs):
        in_dir = f"{config['EAZY']['EAZY_DIR']}/input/{cat.instrument.name}/{cat.version}/{cat.survey}"
        in_path = f"{in_dir}/{cat.cat_name.replace('.fits', '.in')}"
        out_folder = funcs.split_dir_name(in_path.replace("input", "output"), "dir")
        out_path = f"{out_folder}/{funcs.split_dir_name(in_path, 'name').replace('.in', '.out')}"
        fits_out_path = f"{out_path.replace('.out', '')}_EAZY_{SED_fit_params['templates']}_{funcs.lowz_label(SED_fit_params['lowz_zmax'])}.fits"
        PDF_paths = {
            "z": list(np.full(len(IDs), fits_out_path.replace(".fits", "_zPDFs.h5")))
        }
        SED_paths = list(np.full(len(IDs), fits_out_path.replace(".fits", "_SEDs.h5")))
        return in_path, out_path, fits_out_path, PDF_paths, SED_paths

    @staticmethod
    def extract_SEDs(IDs, SED_paths):
        # ensure this works if only extracting 1 galaxy
        if type(IDs) in [str, int, float]:
            IDs = np.array([int(IDs)])
        if type(SED_paths) == str:
            SED_paths = [SED_paths]
        assert len(IDs) == len(SED_paths), galfind_logger.critical(
            f"len(IDs) = {len(IDs)} != len(data_paths) = {len(SED_paths)}!"
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
        z_arr = hf["z_arr"][IDs - 1]
        wav_flux_arr = hf["wav_flux_arr"][IDs - 1]
        wav_arr = wav_flux_arr[:, 0]
        flux_arr = wav_flux_arr[:, 1]
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

    @staticmethod
    def extract_PDFs(gal_property, IDs, PDF_paths, SED_fit_params, timed=True):
        # ensure this works if only extracting 1 galaxy
        if type(IDs) in [str, int, float]:
            IDs = np.array([int(IDs)])
        if type(PDF_paths) == str:
            PDF_paths = [PDF_paths]

        # EAZY only has redshift PDFs
        if gal_property != "z":
            return list(np.full(len(IDs), None))
        else:
            # ensure the correct type
            assert type(PDF_paths) in [list, np.ndarray], galfind_logger.critical(
                f"type(data_paths) = {type(PDF_paths)} not in [list, np.array]!"
            )
            assert type(IDs) in [list, np.ndarray], galfind_logger.critical(
                f"type(IDs) = {type(IDs)} not in [list, np.array]!"
            )
            # ensure the correct array size
            assert len(IDs) == len(PDF_paths), galfind_logger.critical(
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
            # extract redshift PDF for each ID
            if timed:
                start = time.time()
            pz_arr = hf["p_z_arr"][IDs - 1]
            if timed:
                mid = time.time()
            redshift_PDFs = [
                Redshift_PDF(hf_z, pz, SED_fit_params, normed=False)
                for ID, pz in tqdm(
                    zip(IDs, pz_arr), total=len(IDs), desc="Constructing redshift PDFs"
                )
            ]
            if timed:
                end = time.time()
                print(mid - start, end - mid)
            # close .h5 file
            hf.close()
            return redshift_PDFs

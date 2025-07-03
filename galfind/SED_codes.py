#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 00:17:39 2023

@author: austind
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import astropy.units as u
import itertools
import logging
import numpy as np
from astropy.table import Table, join
from tqdm import tqdm
from typing import TYPE_CHECKING, NoReturn, Tuple, Union, List, Dict, Any, Optional
if TYPE_CHECKING:
    from . import Catalogue, Multiple_Filter, SED_obs, PDF
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import SED_result, config, galfind_logger
from . import useful_funcs_austind as funcs

class SED_code(ABC):
    def __init__(
        self,
        SED_fit_params: Dict[str, Any],
    ):
        # assert isinstance(SED_fit_params, dict), \
        #     galfind_logger.critical(
        #         f"{SED_fit_params=} with {type(SED_fit_params)=}!=dict"
        #     )
        self.SED_fit_params = SED_fit_params
        self._assert_SED_fit_params()
        self._load_gal_property_labels()
        self._load_gal_property_err_labels()
        self._load_gal_property_units()

    @classmethod
    @abstractmethod
    def from_label(cls, label: str) -> Type[SED_code]:
        pass

    @property
    @abstractmethod
    def ID_label(self) -> str:
        pass

    @property
    @abstractmethod
    def label(self) -> str:
        pass

    @property
    @abstractmethod
    def hdu_name(self) -> str:
        pass

    @property
    @abstractmethod
    def tab_suffix(self) -> str:
        pass

    @property
    @abstractmethod
    def required_SED_fit_params(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def are_errs_percentiles(self) -> bool:
        pass

    @property
    def excl_bands_label(self) -> str:
        if self.SED_fit_params["excl_bands"] == []:
            return ""
        if isinstance(self.SED_fit_params["excl_bands"][0], list):
            assert "excl_bands_label" in self.SED_fit_params.keys(), \
                galfind_logger.critical(
                    f"excl_bands_label not in SED_fit_params keys = {list(self.SED_fit_params.keys())}"
                )
            return f"_{self.SED_fit_params['excl_bands_label']}"
        else:
            return f"_{'_'.join(self.SED_fit_params['excl_bands'])}"

    #@abstractmethod
    def _load_gal_property_labels(self, gal_property_labels: Dict[str, str]) -> NoReturn:
        self.gal_property_labels = {key: f"{item}_{self.tab_suffix}" 
            for key, item in gal_property_labels.items()}

    #@abstractmethod
    def _load_gal_property_err_labels(self, gal_property_err_labels: Dict[str, List[str, str]]) -> NoReturn:
        self.gal_property_err_labels = {key: [f"{item[0]}_{self.tab_suffix}", f"{item[1]}_{self.tab_suffix}"]
            for key, item in gal_property_err_labels.items()}

    @abstractmethod
    def _load_gal_property_units(self) -> NoReturn:
        pass

    @abstractmethod
    def make_in(
        self, 
        cat: Catalogue, 
        aper_diam: u.Quantity, 
        overwrite: bool = False
    ) -> str:
        pass

    @abstractmethod
    def fit(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity,
        save_SEDs: bool = True,
        save_PDFs: bool = True,
        overwrite: bool = False,
        **kwargs: Dict[str, Any],
    ) -> NoReturn:
        pass

    @abstractmethod
    def make_fits_from_out(self, out_path):
        pass

    @abstractmethod
    def _get_out_paths(
        self: Self, 
        cat: Catalogue, 
        aper_diam: u.Quantity
    ) -> Tuple[str, str, str, Dict[str, List[str]], List[str]]:
        pass

    @abstractmethod
    def extract_SEDs(
        self: Self, 
        IDs: List[int], 
        SED_paths: Union[str, List[str]],
        *args,
        **kwargs
    ) -> List[SED_obs]:
        pass

    @abstractmethod
    def extract_PDFs(
        self: Self, 
        gal_property: str, 
        IDs: List[int], 
        PDF_paths: Union[str, List[str]], 
    ) -> List[Type[PDF]]:
        pass

    @abstractmethod
    def load_cat_property_PDFs(
        self: Self, 
        PDF_paths: Union[List[str], List[Dict[str, str]]],
        IDs: List[int]
    ) -> List[Dict[str, Optional[Type[PDF]]]]:
        pass

    def _assert_SED_fit_params(self) -> NoReturn:
        for key in self.required_SED_fit_params:
            assert key in self.SED_fit_params.keys(), galfind_logger.critical(
                f"'{key}' not in SED_fit_params keys = {list(self.SED_fit_params.keys())}"
            )
        if "excl_bands" not in self.SED_fit_params.keys():
            self.SED_fit_params["excl_bands"] = []
        if isinstance(self.SED_fit_params["excl_bands"], str):
            self.SED_fit_params["excl_bands"] = [self.SED_fit_params["excl_bands"]]
        assert isinstance(self.SED_fit_params["excl_bands"], list), \
            galfind_logger.critical(
                f"{self.SED_fit_params['excl_bands']=} != list"
            )

    
    def __str__(self) -> str:
        output_str = funcs.line_sep
        output_str += f"{self.__class__.__name__.upper()}\n"
        output_str += funcs.band_sep
        output_str += f"LABEL: {self.label}\n"
        output_str += f"HDU_NAME: {self.hdu_name}\n"
        output_str += f"TAB_SUFFIX: {self.tab_suffix}\n"
        output_str += funcs.band_sep
        output_str += "SED_FIT_PARAMS:\n"
        for key, value in self.SED_fit_params.items():
            output_str += f"{key}: {value}\n"
        output_str += funcs.band_sep
        output_str += "GAL_PROPERTY_LABELS:\n"
        for key, label in self.gal_property_labels.items():
            output_str += f"{key}: {label}\n"
        output_str += funcs.band_sep
        output_str += "GAL_PROPERTY_ERR_LABELS:\n"
        for key, labels in self.gal_property_err_labels.items():
            output_str += f"{key}: {labels}\n"
        output_str += funcs.band_sep
        output_str += "GAL_PROPERTY_UNITS:\n"
        for key, unit in self.gal_property_units.items():
            output_str += f"{key}: {unit}\n"
        output_str += funcs.line_sep
        return output_str

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
    ) -> List[SED_result]:
        if timed:
            start = time.time()

        galfind_logger.info(
            f"Making .in file for {self.label} SED fitting for " + \
            f"{cat.survey} {cat.version} {cat.filterset.instrument_name}"
        )
        self.make_in(cat, aper_diam, overwrite)
        galfind_logger.info(
            f"Made .in file for {self.label} SED fitting for " + \
            f"{cat.survey} {cat.version} {cat.filterset.instrument_name}. "
        )

        in_path, out_path, fits_out_path, PDF_paths, SED_paths = self._get_out_paths(cat, aper_diam)
        # run the SED fitting if not already done so or if wanted overwriting
        fits_cat = cat.open_cat(hdu=self) # could be cached
        if "z" not in self.gal_property_labels.keys():
            fit = True
        elif fits_cat is None or self.gal_property_labels["z"] \
                not in fits_cat.colnames:
            fit = True
        else:
            fit = False

        if fit:
            self.fit(
                cat,
                aper_diam,
                save_SEDs=save_SEDs,
                save_PDFs=save_PDFs,
                overwrite=overwrite,
                **fit_kwargs
            )
            self.make_fits_from_out(out_path)
            self.update_fits_cat(cat, fits_out_path)

        if timed:
            mid = time.time()
            print(f"Running SED fitting took {(mid - start):.1f}s")

        SED_fit_cat = cat.open_cat(cropped=True, hdu=self)
        aper_phot_IDs = [gal.ID for gal in cat]
        phot_arr = [gal.aper_phot[aper_diam] for gal in cat]
        # assume properties all come from the same table if only a single catalogue is given
        SED_fit_IDs = SED_fit_cat[self.ID_label]
        assert all(aper_phot_ID == SED_fit_ID for aper_phot_ID, SED_fit_ID \
            in zip(aper_phot_IDs, SED_fit_IDs)), galfind_logger.critical(
            f"IDs in SED_fit_cat do not match those in the catalogue"
            )
        cat_properties = [{gal_property: SED_fit_cat[label][i] * \
            self.gal_property_units[gal_property] if gal_property in \
            self.gal_property_units.keys() else SED_fit_cat[label][i] * \
            u.dimensionless_unscaled for gal_property, label in \
            self.gal_property_labels.items()} for i in range(len(aper_phot_IDs))]
        
        # TODO: When instantiating the class, ensure that all errors have an associated property
        assert all(
            err_key in self.gal_property_labels.keys()
            for err_key in self.gal_property_err_labels.keys()
        )

        # adjust errors if required (i.e. if 16th and 84th percentiles rather than errors)
        cat_property_errs = {
            gal_property: list(
                funcs.adjust_errs(
                    np.array(SED_fit_cat[self.gal_property_labels[gal_property]]),
                    np.array(
                        [
                            np.array(SED_fit_cat[err_labels[0]]),
                            np.array(SED_fit_cat[err_labels[1]]),
                        ]
                    ),
                )[1]
            )
            if self.are_errs_percentiles
            else [list(SED_fit_cat[err_labels[0]]), list(SED_fit_cat[err_labels[1]])]
            for gal_property, err_labels in self.gal_property_err_labels.items()
        }

        cat_property_errs = [
            {
                key: [value[0][i], value[1][i]]
                for key, value in cat_property_errs.items()
            }
            for i in range(len(aper_phot_IDs))
        ]

        if timed:
            mid = time.time()
            print(
                f"Loading properties and associated errors took {(mid - start):.1f}s"
            )

        # save PDF and SED paths in galfind catalogue object
        #cat.save_phot_PDF_paths(PDF_paths)
        #cat.save_phot_SED_paths(SED_paths)

        if load_PDFs:
            galfind_logger.info(
                f"Loading {self.hdu_name} property PDFs into " + \
                f"{cat.survey} {cat.version} {cat.filterset.instrument_name}"
            )
            cat_property_PDFs = self.load_cat_property_PDFs(PDF_paths, aper_phot_IDs)
            galfind_logger.info(
                f"Finished loading {self.hdu_name} property PDFs into " + \
                f"{cat.survey} {cat.version} {cat.filterset.instrument_name}"
            )
        else:
            galfind_logger.info(
                f"Not loading {self.hdu_name} property PDFs into " + \
                f"{cat.survey} {cat.version} {cat.filterset.instrument_name}"
            )
            cat_property_PDFs = np.array(
                list(itertools.repeat(None, len(cat_properties)))
            )

        if load_SEDs:
            galfind_logger.info(
                f"Loading {self.hdu_name} SEDs into " + \
                f"{cat.survey} {cat.version} {cat.filterset.instrument_name}"
            )
            cat_SEDs = self.extract_SEDs(aper_phot_IDs, SED_paths, cat = cat, aper_diam = aper_diam)
            galfind_logger.info(
                f"Finished loading {self.hdu_name} SEDs into " + \
                f"{cat.survey} {cat.version} {cat.filterset.instrument_name}"
            )
        else:
            galfind_logger.info(
                f"Not loading {self.hdu_name} SEDs into " + \
                f"{cat.survey} {cat.version} {cat.filterset.instrument_name}"
            )
            cat_SEDs = np.array(
                list(itertools.repeat(None, len(cat_properties)))
            )

        cat_SED_results = [SED_result(self, phot, properties, property_errs, property_PDFs, SED) \
            for phot, properties, property_errs, property_PDFs, SED in \
            zip(phot_arr, cat_properties, cat_property_errs, cat_property_PDFs, cat_SEDs)]

        if update:
            cat.update_SED_results(cat_SED_results, timed=timed)
        return cat_SED_results

    def _load_phot(self,
        cat: Catalogue,
        aper_diam: u.Quantity,
        out_units: u.Unit,
        no_data_val: Any,
        upper_sigma_lim: Optional[Dict[str, Union[float, int]]] = None,
        input_filterset: Optional[Multiple_Filter] = None,
        incl_units: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if input_filterset is None:
            input_filterset = cat.filterset
        breakpoint()
        input_filterset.filters = np.array([filt for filt in input_filterset if filt.band_name not in self.SED_fit_params["excl_bands"]])
        galfind_logger.info(f"Excluded bands: {self.excl_bands_label}")
        #breakpoint()
        # load in raw photometry from the galaxies in the catalogue and convert to appropriate units
        phot = np.array(
            [gal.aper_phot[aper_diam].flux.to(out_units) for gal in cat], dtype=object
        )  # [:, :, 0]
        phot_shape = phot.shape
        if out_units != u.ABmag:
            phot_err = np.array(
                [gal.aper_phot[aper_diam].flux_errs.to(out_units) for gal in cat],
                dtype=object,
            )  # [:, :, 0]
        else:
            # Not correct in general! Only for high S/N! Fails to scale mag errors asymetrically from flux errors
            phot_err = np.array(
                [
                    funcs.flux_pc_to_mag_err(
                        gal.aper_phot[aper_diam].flux_errs / gal.aper_phot[aper_diam].flux
                    )
                    for gal in cat
                ],
                dtype=object,
            )  # [:, :, 0]
        breakpoint()
        # include upper limits if wanted
        if upper_sigma_lim is not None:
            # determine relevant indices
            upper_lim_indices = [
                [i, j]
                for i, gal in enumerate(cat)
                for j, depth in enumerate(gal.phot[0].depths)
                if funcs.n_sigma_detection(
                    depth,
                    (phot[i][j] * out_units).to(u.ABmag).value
                    + gal.aper_phot[aper_diam].aper_corrs[j],
                    u.Jy.to(u.ABmag),
                )
                < upper_sigma_lim["threshold"]
            ]
            phot = np.array(
                [
                    funcs.five_to_n_sigma_mag(
                        loc_depth, upper_sigma_lim["value"]
                    )
                    if [i, j] in upper_lim_indices
                    else phot[i][j]
                    for i, gal in enumerate(cat)
                    for j, loc_depth in enumerate(gal.aper_phot[aper_diam].depths)
                ]
            ).reshape(phot_shape)
            phot_err = np.array(
                [
                    -1.0 if [i, j] in upper_lim_indices else phot_err[i][j]
                    for i, gal in enumerate(cat)
                    for j, loc_depth in enumerate(gal.aper_phot[aper_diam].depths)
                ]
            ).reshape(phot_shape)

        # insert 'no_data_val' from SED_input_bands with no data in the catalogue
        phot_in = np.zeros((len(cat), len(input_filterset)))
        phot_err_in = np.zeros((len(cat), len(input_filterset)))
        load_message = f"Loading photometry for {cat.survey} {cat.version} for {self.label} SED fitting"
        galfind_logger.info(load_message)
        for i, gal in tqdm(
            enumerate(cat), desc=load_message, total=len(cat), disable=galfind_logger.getEffectiveLevel() > logging.INFO
        ):
            for j, (band_name, band_instrument) in enumerate(zip(input_filterset.band_names, input_filterset.instrument_names)):
                if band_name in gal.aper_phot[aper_diam].filterset.band_names:  # Check mask?
                    index = np.where(
                        (band_name == np.array(gal.aper_phot[aper_diam].filterset.band_names)) \
                            & (band_instrument == np.array(gal.aper_phot[aper_diam].filterset.instrument_names))
                    )[0]
                    assert len(index) == 1, galfind_logger.critical(
                        f"Multiple indices found for {band_name} in {gal.aper_phot[aper_diam].filterset.band_names}"
                    )
                    index = index[0]

                    phot_in[i, j] = np.array(phot[i].data)[index]
                    phot_err_in[i, j] = np.array(phot_err[i].data)[index]
                else:
                    phot_in[i, j] = no_data_val
                    phot_err_in[i, j] = no_data_val
        if incl_units:
            phot_in = phot_in * out_units
            phot_err_in = phot_err_in * out_units
        return phot_in, phot_err_in

    # should be catalogue method
    def update_fits_cat(
        self: Self,
        cat: Catalogue,
        fits_out_path: str,
    ) -> None:
        # open relevant catalogue hdu extension
        orig_tab = cat.open_cat(hdu=self)
        SED_fitting_tab = Table.read(fits_out_path)
        # if table has not already been made
        if orig_tab is None:
            cat.write_hdu(SED_fitting_tab, hdu=self.hdu_name.upper())
        else:
            # if any of the column names are the same
            if any(name in orig_tab.colnames for name in SED_fitting_tab.colnames if name != self.ID_label):
                galfind_logger.warning(
                    f"Column names in {self.hdu_name=} table already exist in catalogue table. " + \
                    "Will not update catalogue table."
                )
                # TODO: merge tables appropriately
            else:
                # combine catalogues
                combined_tab = join(
                    orig_tab,
                    SED_fitting_tab,
                    keys=self.ID_label,
                    join_type="outer",
                )
                cat.write_hdu(combined_tab, hdu=self.hdu_name.upper())

    # def update_lowz_zmax(SED_results):
    #     if "dz" in SED_fit_params.keys():
    #         assert (
    #             "lowz_zmax" not in SED_fit_params.keys()
    #         ), galfind_logger.critical(
    #             "Cannot have both 'dz' and 'lowz_zmax' in SED_fit_params"
    #         )
    #         available_SED_fit_params = [
    #             SED_fit_params["code"].SED_fit_params_from_label(label)
    #             for label in SED_results.keys()
    #             if label.split("_")[0]
    #             == SED_fit_params["code"].__class__.__name__
    #         ]
    #         # extract sorted (low -> high) lowz_zmax's (excluding 'None') from available SED_fit_params
    #         available_lowz_zmax = sorted(
    #             filter(
    #                 lambda lowz_zmax: lowz_zmax is not None,
    #                 [
    #                     SED_fit_params_["lowz_zmax"]
    #                     for SED_fit_params_ in available_SED_fit_params
    #                     if "lowz_zmax" in SED_fit_params_.keys()
    #                 ],
    #             )
    #         )
    #         # calculate z
    #         z = SED_results[
    #             SED_fit_params["code"].label_from_SED_fit_params(
    #                 {**SED_fit_params, "lowz_zmax": None}
    #             )
    #         ].z
    #         # add appropriate 'lowz_zmax' to dict
    #         lowz_zmax = [
    #             lowz_zmax
    #             for lowz_zmax in reversed(available_lowz_zmax)
    #             if lowz_zmax < z - SED_fit_params["dz"]
    #         ]
    #         if len(lowz_zmax) > 0:
    #             SED_fit_params["lowz_zmax"] = lowz_zmax[0]
    #         else:
    #             galfind_logger.warning(
    #                 f"No appropriate lowz_zmax run for z = {z}, dz = {SED_fit_params['dz']}. Available runs are: lowz_zmax = {', '.join(np.array(available_lowz_zmax).astype(str))}"
    #             )
    #             SED_fit_params["lowz_zmax"] = None
    #         # remove 'dz' from dict
    #         SED_fit_params.pop("dz")
    #     return SED_fit_params
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
import numpy as np
from astropy.table import Table, join
from tqdm import tqdm
from typing import TYPE_CHECKING, NoReturn, Tuple, Union, List, Dict, Any, Optional, Type
if TYPE_CHECKING:
    from . import Catalogue, Multiple_Filter

from . import SED_result, config, galfind_logger
from . import useful_funcs_austind as funcs

# %% SED_code class


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

    @abstractmethod
    def _load_gal_property_labels(self) -> NoReturn:
        pass

    @abstractmethod
    def _load_gal_property_err_labels(self) -> NoReturn:
        pass

    @abstractmethod
    def _load_gal_property_units(self) -> NoReturn:
        pass

    @abstractmethod
    def make_in(self, cat: Catalogue):
        pass

    @abstractmethod
    def fit(self, cat: Catalogue, overwrite: bool = False):
        pass

    @abstractmethod
    def make_fits_from_out(self, out_path):
        pass

    @abstractmethod
    def _get_out_paths(out_path, IDs):
        pass

    @abstractmethod
    def extract_SEDs(IDs, data_paths):
        pass

    @abstractmethod
    def extract_PDFs(gal_property, IDs, data_paths):
        pass

    def _assert_SED_fit_params(self) -> NoReturn:
        for key in self.required_SED_fit_params:
            assert key in self.SED_fit_params.keys(), galfind_logger.critical(
                f"'{key}' not in SED_fit_params keys = {list(self.SED_fit_params.keys())}"
            )

    def __call__(
        self,
        cat: Catalogue,
        aper_diam: u.Quantity,
        save_SEDs: bool = True,
        save_PDFs: bool = True,
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
        fits_cat = cat.open_cat(hdu=self.hdu_name) # should be cached
        breakpoint()
        if fits_cat is None or self.gal_property_labels["z"] \
                not in fits_cat.colnames:
            self.fit(
                cat,
                aper_diam,
                save_SEDs=save_SEDs,
                save_PDFs=save_PDFs,
                overwrite=overwrite,
                **fit_kwargs
            )
            self.make_fits_from_out(out_path)
            breakpoint()
            self.update_fits_cat(cat, fits_out_path)
        # save PDF and SED paths in galfind catalogue object
        #cat.save_phot_PDF_paths(PDF_paths)
        #cat.save_phot_SED_paths(SED_paths)

        if timed:
            mid = time.time()
            print(f"Running SED fitting took {(mid - start):.1f}s")

        SED_fit_cat = cat.open_cat(
            hdu=self.hdu_name, cropped=True,
        )
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
            self.galaxy_property_labels.items()} for i in range(len(aper_phot_IDs))]
        # 
        # TODO: ensure that all errors have an associated property - when instantiating the class
        # assert all(
        #     err_key in self.galaxy_property_dict.keys()
        #     for err_key in self.galaxy_property_errs_dict.keys()
        # )
        # adjust errors if required (i.e. if 16th and 84th percentiles rather than errors)
        cat_property_errs = {
            gal_property: list(
                funcs.adjust_errs(
                    np.array(SED_fit_cat[self.galaxy_property_labels[gal_property]]),
                    np.array(
                        [
                            np.array(SED_fit_cat[err_labels[0]]),
                            np.array(SED_fit_cat[err_labels[1]]),
                        ]
                    ),
                )[1]
            )
            if self.are_errs_percentiles
            else [
                list(SED_fit_cat[err_labels[0]]),
                list(SED_fit_cat[err_labels[1]]),
            ]
            for gal_property, err_labels in self.galaxy_property_err_labels.items()
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

        if load_PDFs:
            galfind_logger.info(
                f"Loading {self.hdu_name} property PDFs into " + \
                f"{cat.survey} {cat.version} {cat.instrument_name}"
            )
            # construct PDF objects, type = array of len(fits_cat), 
            # each element a dict of {gal_property: PDF object} excluding None PDFs
            cat_property_PDFs_ = {
                gal_property: self.extract_PDFs(
                    gal_property,
                    aper_phot_IDs,
                    PDF_paths[gal_property],
                )
                for gal_property in PDF_paths.keys()
            }
            cat_property_PDFs_ = [
                {
                    gal_property: PDF_arr[i]
                    for gal_property, PDF_arr in cat_property_PDFs_.items()
                    if PDF_arr[i] is not None
                }
                for i in range(len(aper_phot_IDs))
            ]
            # set to None if no PDFs are found
            cat_property_PDFs = [
                None if len(cat_property_PDF) == 0 else cat_property_PDF
                for cat_property_PDF in cat_property_PDFs_
            ]
            galfind_logger.info(
                f"Finished loading {self.hdu_name} property PDFs into " + \
                f"{cat.survey} {cat.version} {cat.instrument_name}"
            )
        else:
            galfind_logger.info(
                f"Not loading {self.hdu_name} property PDFs into " + \
                f"{cat.survey} {cat.version} {cat.instrument_name}"
            )
            out_shape = np.array(cat_properties).shape
            cat_property_PDFs = np.array(
                list(
                    itertools.repeat(
                        list(itertools.repeat(None, out_shape[1])),
                        out_shape[0],
                    )
                )
            )

        if load_SEDs:
            galfind_logger.info(
                f"Loading {self.hdu_name} SEDs into " + \
                f"{cat.survey} {cat.version} {cat.instrument_name}"
            )
            cat_SEDs = self.extract_SEDs(aper_phot_IDs, SED_paths)
            galfind_logger.info(
                f"Finished loading {self.hdu_name} SEDs into " + \
                f"{cat.survey} {cat.version} {cat.instrument_name}"
            )
        else:
            galfind_logger.info(
                f"Not loading {self.hdu_name} SEDs into " + \
                f"{cat.survey} {cat.version} {cat.instrument_name}"
            )
            out_shape = np.array(cat_properties).shape
            cat_SEDs = np.array(
                list(
                    itertools.repeat(
                        list(itertools.repeat(None, out_shape[1])),
                        out_shape[0],
                    )
                )
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        if input_filterset is None:
            input_filterset = cat.filterset
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
        for i, gal in tqdm(
            enumerate(cat), desc="Making .in file", total=len(cat)
        ):
            for j, band_name in enumerate(input_filterset.band_names):
                if band_name in gal.aper_phot[aper_diam].filterset.band_names:  # Check mask?
                    index = np.where(
                        band_name == np.array(gal.aper_phot[aper_diam].filterset.band_names)
                    )[0][0]
                    phot_in[i, j] = np.array(phot[i].data)[index]
                    phot_err_in[i, j] = np.array(phot_err[i].data)[index]
                else:
                    phot_in[i, j] = no_data_val
                    phot_err_in[i, j] = no_data_val
        return phot_in, phot_err_in

    # should be catalogue method
    def update_fits_cat(
        self, cat: Catalogue, fits_out_path: str
    ) -> None:  # *args, **kwargs):
        # open relevant catalogue hdu extension
        orig_tab = cat.open_cat(hdu=self.hdu_name)
        SED_fitting_tab = Table.read(fits_out_path)
        # if table has not already been made
        if orig_tab is None:
            cat.write_hdu(SED_fitting_tab, hdu=self.hdu_name)
        else:
            # combine catalogues
            combined_tab = join(
                orig_tab,
                SED_fitting_tab,
                keys=self.ID_label,
                join_type="outer",
            )
            cat.write_hdu(combined_tab, hdu=self.hdu_name)

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
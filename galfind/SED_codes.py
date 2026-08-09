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
from copy import deepcopy
from numpy.typing import NDArray
from astropy.table import Table, vstack
from tqdm import tqdm
from typing import TYPE_CHECKING, NoReturn, Tuple, Union, List, Dict, Any, Optional
if TYPE_CHECKING:
    from . import Galaxy, Catalogue, Spectral_Catalogue, Multiple_Filter, SED_obs, PDF
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import SED_result, config, galfind_logger
from . import useful_funcs_austind as funcs

class SED_code(ABC):
    """Abstract base class defining the common interface for external SED-fitting codes.

    Subclasses (e.g. `LePhare`, `EAZY`) wrap an external photometric
    redshift / SED-fitting tool, providing a uniform interface for
    building input catalogues, running the fit, and parsing the results
    back into GALFIND-native `SED_result`, `SED_obs` and PDF objects.
    Instances are also callable (see `__call__`), which drives the full
    pipeline of pre-fitting, input catalogue creation, fitting, and
    result loading for a `Galaxy`, `Catalogue` or `Spectral_Catalogue`.

    Parameters
    ----------
    SED_fit_params : `dict`
        Dictionary of SED fitting parameters/options for this run. Must
        contain the keys named in `required_SED_fit_params`.
    **kwargs : `dict`
        Additional keyword arguments. Each key/value pair is set as an
        instance attribute.

    Attributes
    ----------
    SED_fit_params : `dict`
        SED fitting parameters/options for this run.
    gal_property_labels : `dict`
        Mapping from galaxy property name to output catalogue column name.
    gal_property_err_labels : `dict`
        Mapping from galaxy property name to a two-element list of
        lower/upper output catalogue error column names.
    gal_property_units : `dict`
        Mapping from galaxy property name to `astropy.units.Unit`.
    """

    def __init__(
        self,
        SED_fit_params: Dict[str, Any],
        **kwargs: Dict[str, Any],
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
        for key, val in kwargs.items():
            setattr(self, key, val)

    @classmethod
    @abstractmethod
    def from_label(cls, label: str) -> Type[SED_code]:
        """Construct an `SED_code` instance from a saved catalogue label.

        Parameters
        ----------
        label : `str`
            Label string identifying a previous fitting run.

        Returns
        -------
        `SED_code`
            An instance of the concrete `SED_code` subclass reconstructed
            from `label`.
        """
        pass

    # @property
    # @abstractmethod
    # def ID_label(self) -> str:
    #     pass

    @property
    @abstractmethod
    def label(self) -> str:
        """`str`: Unique label identifying this fitting run.

        Must be implemented by each `SED_code` subclass.
        """
        pass

    @property
    @abstractmethod
    def hdu_name(self) -> str:
        """`str`: Name of the FITS HDU/extension this code's output is stored under.

        Must be implemented by each `SED_code` subclass.
        """
        pass

    @property
    @abstractmethod
    def tab_suffix(self) -> str:
        """`str`: Column-name suffix used to distinguish this run's output columns.

        Must be implemented by each `SED_code` subclass.
        """
        pass

    @property
    @abstractmethod
    def required_SED_fit_params(self) -> List[str]:
        """`list` of `str`: Names of the `SED_fit_params` keys required by this code.

        Must be implemented by each `SED_code` subclass.
        """
        pass

    @property
    @abstractmethod
    def are_errs_percentiles(self) -> bool:
        """`bool`: Whether output property errors are stored as percentiles rather than 1-sigma values.

        Must be implemented by each `SED_code` subclass.
        """
        pass

    @property
    def excl_bands_label(self) -> str:
        """`str`: Suffix encoding any bands excluded from this fitting run.

        Returns an empty string if no bands are excluded
        (``SED_fit_params["excl_bands"] == []``); if
        ``SED_fit_params["excl_bands"]`` is a list of lists, returns
        ``SED_fit_params["excl_bands_label"]`` prefixed with ``"_"``;
        otherwise returns the excluded band names joined with ``"_"``,
        prefixed with ``"_"``.
        """
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.tab_suffix})"

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
    def pre_fitting(
        self: Type[Self],
        cat: Union[Catalogue, Spectral_Catalogue],
        aper_diam: u.Quantity,
        overwrite: bool = False,
        save_name: Optional[str] = None,
    ) -> None:
        """Perform any pre-fitting setup required before running the fit.

        Parameters
        ----------
        cat : `Catalogue` or `Spectral_Catalogue`
            Catalogue of galaxies about to be fitted.
        aper_diam : `astropy.units.Quantity`
            Aperture diameter of the photometry to be fitted.
        overwrite : `bool`, optional
            Whether to overwrite any existing pre-fitting products. Default
            is `False`.
        save_name : `str`, optional
            Optional custom name used when saving pre-fitting products.
            Default is `None`.
        """
        pass

    @abstractmethod
    def make_in(
        self: Type[Self],
        cat: Catalogue,
        aper_diam: u.Quantity,
        overwrite: bool = False
    ) -> str:
        """Build the input photometric catalogue required by the external code.

        Parameters
        ----------
        cat : `Catalogue`
            Catalogue of galaxies to build the input file for.
        aper_diam : `astropy.units.Quantity`
            Aperture diameter of the photometry to extract.
        overwrite : `bool`, optional
            Whether to remake the input file if one already exists. Default
            is `False`.

        Returns
        -------
        `str`
            Path to the (possibly newly written) input catalogue.
        """
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
    ) -> None:
        """Run the external SED-fitting code on a catalogue's input file.

        Parameters
        ----------
        cat : `Catalogue`
            Catalogue of galaxies to fit.
        aper_diam : `astropy.units.Quantity`
            Aperture diameter of the photometry being fitted.
        save_SEDs : `bool`, optional
            Whether to save the best-fit SEDs. Default is `True`.
        save_PDFs : `bool`, optional
            Whether to save the property PDFs. Default is `True`.
        overwrite : `bool`, optional
            Whether to re-run the fit even if the output already exists.
            Default is `False`.
        **kwargs : `dict`
            Additional keyword arguments passed through to the concrete
            implementation.
        """
        pass

    @abstractmethod
    def make_fits_from_out(self, out_path):
        """Convert the raw output of the external code into a FITS binary table.

        Parameters
        ----------
        out_path : `str`
            Path to the raw output catalogue produced by `fit`.
        """
        pass

    @abstractmethod
    def _get_out_paths(
        self: Self, 
        cat: Catalogue, 
        aper_diam: u.Quantity,
        save_name: Optional[str] = None,
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
        """Extract best-fit SEDs from the external code's output files.

        Parameters
        ----------
        IDs : `list` of `int`
            Galaxy IDs to extract SEDs for.
        SED_paths : `str` or `list` of `str`
            Path(s) to the file(s) containing each galaxy's best-fit SED.
        *args : `tuple`
            Additional positional arguments, implementation-specific.
        **kwargs : `dict`
            Additional keyword arguments, implementation-specific.

        Returns
        -------
        `list` of `SED_obs`
            Best-fit galaxy SED for each requested ID, in the same order
            as `IDs`.
        """
        pass

    @abstractmethod
    def extract_PDFs(
        self: Self,
        gal_property: str,
        IDs: List[int],
        PDF_paths: Union[str, List[str]],
    ) -> List[Type[PDF]]:
        """Extract posterior PDFs for a galaxy property from the external code's output files.

        Parameters
        ----------
        gal_property : `str`
            Name of the galaxy property to extract PDFs for.
        IDs : `list` of `int`
            Galaxy IDs to extract PDFs for.
        PDF_paths : `str` or `list` of `str`
            Path(s) to the file(s) containing each galaxy's PDF for
            `gal_property`.

        Returns
        -------
        `list` of `PDF`
            PDF object for `gal_property`, one per ID in `IDs`.
        """
        pass

    @abstractmethod
    def load_cat_property_PDFs(
        self: Self,
        PDF_paths: Union[List[str], List[Dict[str, str]]],
        IDs: List[int]
    ) -> List[Dict[str, Optional[Type[PDF]]]]:
        """Load per-galaxy property PDFs from a set of PDF file paths.

        Parameters
        ----------
        PDF_paths : `list` of `str` or `list` of `dict`
            PDF file path(s) for each requested property, keyed
            appropriately by the concrete implementation.
        IDs : `list` of `int`
            Galaxy IDs to load PDFs for.

        Returns
        -------
        `list` of `dict`
            One dictionary per galaxy (in the order of `IDs`), mapping
            property name to its PDF object, or `None` for galaxies with
            no available PDFs.
        """
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
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            try:
                setattr(result, key, deepcopy(value, memo))
            except:
                galfind_logger.critical(
                    f"deepcopy({repr(self)}) {key}: {value} FAIL!"
                )
                breakpoint()
        return result

    def __call__(
        self: Self,
        target: Union[Galaxy, Catalogue, Spectral_Catalogue],
        aper_diam: u.Quantity,
        save_PDFs: bool = True,
        save_SEDs: bool = True,
        load_PDFs: bool = True,
        load_SEDs: bool = True,
        timed: bool = True,
        overwrite: bool = False,
        update: bool = True,
        save_name: Optional[str] = None,
        **fit_kwargs,
    ) -> List[SED_result]:
        from . import Galaxy
        # Make length 1 catalogue if required
        if isinstance(target, Galaxy):
            cat = funcs.cat_from_gal(
                target,
                data = fit_kwargs.get("data", None),
                gal_creator_kwargs = fit_kwargs.get("gal_creator_kwargs", {}),
            )
            # TODO: check loading from cache
            fits_cat = cat.open_cat(hdu=self)
            if fits_cat is None:
                fit = True
            elif target.ID in np.array(fits_cat[self.ID_label], dtype=int):
                fit = False
            else:
                fit = True
                if save_name is None:
                    save_name = ""
                else:
                    save_name += "_"
                save_name += f"ID={target.ID}"
            cat = self(
                cat,
                aper_diam,
                save_PDFs = save_PDFs,
                save_SEDs = save_SEDs,
                load_PDFs = load_PDFs,
                load_SEDs = load_SEDs,
                overwrite = overwrite,
                update = update,
                save_name = save_name,
                fit = fit,
                **fit_kwargs,
            )
            # return galaxy rather than catalogue if input was a galaxy
            return cat[0]

        if timed:
            start = time.time()

        in_path, out_path, fits_out_path, PDF_paths, SED_paths = \
            self._get_out_paths(
                target,
                aper_diam,
                save_name = save_name,
            )
        # run the SED fitting if not already done so or if wanted overwriting
        self.pre_fitting(target, aper_diam, overwrite, save_name = save_name)
        # TODO: check loading from cache
        if fit_kwargs.get("fit", None) is not None:
            fit = fit_kwargs["fit"]
        else:
            fits_cat = target.open_cat(hdu=self)
            if "z" not in self.gal_property_labels.keys():
                fit = True
            elif fits_cat is None or self.gal_property_labels["z"] not in fits_cat.colnames:
                fit = True
            else:
                fit = False
        if fit:
            galfind_logger.info(
                f"Making .in file for {self.label} SED fitting for {repr(target)}!"
            )
            self.make_in(target, aper_diam, overwrite, save_name = save_name)
            galfind_logger.info(
                f"Made .in file for {self.label} SED fitting for {repr(target)}!"
            )
            self.fit(
                target,
                aper_diam,
                save_SEDs=save_SEDs,
                save_PDFs=save_PDFs,
                overwrite=overwrite,
                save_name=save_name,
                **fit_kwargs
            )
            self.make_fits_from_out(out_path, save_name = save_name)
            self.update_fits_cat(target, fits_out_path)

        if timed:
            mid = time.time()
            print(f"Running SED fitting took {(mid - start):.1f}s")

        if update:
            SED_fit_cat = target.open_cat(cropped=True, hdu=self)
            aper_phot_IDs = [gal.ID for gal in target]
            phot_arr = [gal.aper_phot[aper_diam] for gal in target]
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

            if load_PDFs:
                galfind_logger.info(
                    f"Loading {self.hdu_name} property PDFs into {repr(target)}!"
                )
                cat_property_PDFs = self.load_cat_property_PDFs(PDF_paths, aper_phot_IDs)
                galfind_logger.info(
                    f"Finished loading {self.hdu_name} property PDFs into {repr(target)}!"
                )
            else:
                galfind_logger.info(
                    f"Not loading {self.hdu_name} property PDFs into {repr(target)}!"
                )
                cat_property_PDFs = np.array(
                    list(itertools.repeat(None, len(cat_properties)))
                )

            if load_SEDs:
                galfind_logger.info(
                    f"Loading {self.hdu_name} SEDs into {repr(target)}!"
                )
                if all("z" in pdf.keys() if pdf is not None else False for pdf in cat_property_PDFs):
                    zPDFs = [pdf["z"] for pdf in cat_property_PDFs]
                else:
                    zPDFs = None
                cat_SEDs = self.extract_SEDs(aper_phot_IDs, SED_paths, cat = target, aper_diam = aper_diam, zPDFs = zPDFs)
                galfind_logger.info(
                    f"Finished loading {self.hdu_name} SEDs into {repr(target)}!"
                )
            else:
                galfind_logger.info(
                    f"Not loading {self.hdu_name} SEDs into {repr(target)}!"
                )
                cat_SEDs = np.array(
                    list(itertools.repeat(None, len(cat_properties)))
                )
            cat_SED_results = [
                SED_result(
                    deepcopy(self),
                    phot,
                    properties,
                    property_errs,
                    property_PDFs,
                    SED,
                ) for phot, properties, property_errs, property_PDFs, SED in tqdm(
                    zip(
                        phot_arr,
                        cat_properties,
                        cat_property_errs,
                        cat_property_PDFs,
                        cat_SEDs
                    ),
                    desc=f"Creating {repr(self)} cat_SED_results for {repr(target)}",
                    total=len(target),
                    disable=galfind_logger.getEffectiveLevel() > logging.INFO,
                )
            ]

            target.update_SED_results(cat_SED_results, timed = timed)
            if not hasattr(target, "SED_results"):
                target.SED_results = {}
            if not aper_diam in target.SED_results.keys():
                target.SED_results[aper_diam] = []
            target.SED_results[aper_diam].append(self)

        return target
            

    def _load_phot(self,
        cat: Catalogue,
        aper_diam: u.Quantity,
        out_units: u.Unit,
        no_data_val: Any,
        upper_sigma_lim: Optional[Dict[str, Union[float, int]]] = None,
        input_filterset: Optional[Multiple_Filter] = None,
        incl_units: bool = False,
    ) -> Tuple[NDArray[float], NDArray[float]]:
        if input_filterset is None:
            input_filterset = cat.filterset

        input_filterset.filters = np.array([filt for filt in input_filterset if filt.filt_name not in self.SED_fit_params["excl_bands"]])
        galfind_logger.info(f"Excluded bands: {self.excl_bands_label}")
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
            galfind_logger.warning(
                "Photometry is in ABmag, but errors are not scaled asymmetrically. " +
                "This is not recommended!"
            )
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
                for j, depth in enumerate(gal.aper_phot[aper_diam].depths)
                if funcs.n_sigma_detection(
                    depth,
                    (phot[i][j] * out_units).to(u.ABmag).value
                    + gal.aper_phot[aper_diam].aper_corrs[j],
                    u.Jy.to(u.ABmag),
                )
                < upper_sigma_lim["threshold"]
                or not np.isfinite(phot[i][j])
            ]
            phot = np.array(
                [
                    funcs.five_to_n_sigma_mag(
                        loc_depth,
                        upper_sigma_lim["value"]
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
            enumerate(cat),
            desc=load_message,
            total=len(cat),
            disable=galfind_logger.getEffectiveLevel() > logging.INFO,
        ):
            for j, (filt_name, band_instrument) in enumerate(zip(input_filterset.filt_names, input_filterset.instrument_names)):
                if filt_name in gal.aper_phot[aper_diam].filterset.filt_names:  # Check mask?
                    index = np.where(
                        (filt_name == np.array(gal.aper_phot[aper_diam].filterset.filt_names)) \
                            & (band_instrument == np.array(gal.aper_phot[aper_diam].filterset.instrument_names))
                    )[0]
                    assert len(index) == 1, galfind_logger.critical(
                        f"Multiple indices found for {filt_name} in {gal.aper_phot[aper_diam].filterset.filt_names}"
                    )
                    index = index[0]
                    
                    new_phot_in = np.array(phot[i].data)[index]
                    new_phot_err_in = np.array(phot_err[i].data)[index]
                    if not (np.isfinite(new_phot_in) and np.isfinite(new_phot_err_in)):
                        new_phot_in = no_data_val
                        new_phot_err_in = no_data_val

                    phot_in[i, j] = new_phot_in
                    phot_err_in[i, j] = new_phot_err_in
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
            if all(
                name in orig_tab.colnames for name in SED_fitting_tab.colnames
            ) and all(
                name in SED_fitting_tab.colnames for name in orig_tab.colnames
            ):
                # combine catalogues
                combined_tab = vstack(
                    [orig_tab, SED_fitting_tab],
                    metadata_conflicts = "silent",
                )
                # sort by ID
                combined_tab = combined_tab[np.argsort(combined_tab[self.ID_label].astype(int))]
                # write out ID
                cat.write_hdu(combined_tab, hdu=self.hdu_name.upper())
            # if any of the column names are the same
            elif any(name in orig_tab.colnames for name in SED_fitting_tab.colnames if name != self.ID_label):
                galfind_logger.warning(
                    f"Column names in {self.hdu_name=} table already exist in catalogue table. " + \
                    "Will not update catalogue table."
                )
                # TODO: merge tables appropriately
                raise NotImplementedError(
                    "Merging tables with non-totally overlapping column names is not yet implemented."
                )
            else:
                err_msg = f"Column names in {self.hdu_name=} != those in {fits_out_path}. " + \
                    "Will not update catalogue table."
                galfind_logger.critical(err_msg)
                raise Exception(err_msg)

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
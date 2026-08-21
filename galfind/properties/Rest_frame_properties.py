"""Rest-frame property calculators for galaxies.

Derives rest-frame properties from rest-frame photometry with flux scattering,
uncertainty propagation, and PDF caching for efficient computation.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    NoReturn,
    Optional,
    Tuple,
    Union,
)

import astropy.units as u
import numpy as np
from astropy.table import Table
from joblib import Parallel, delayed, parallel_config
from numba import njit
from scipy.stats import norm
from tqdm import tqdm

if TYPE_CHECKING:
    from . import Multiple_Filter
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from .. import all_filt_names, astropy_cosmo, config, galfind_logger
from ..catalogues.Catalogue import Catalogue
from ..catalogues.Catalogue_Base import Catalogue_Base
from ..galaxy.Galaxy import Galaxy
from ..photometry.Photometry_rest import Photometry_rest
from ..sed_fitting.SED_codes import SED_code
from ..spectra.Dust_Attenuation import M99, AUV_from_beta, Calzetti00, Dust_Law
from ..spectra.Emission_lines import line_diagnostics, strong_optical_lines
from ..utils import useful_funcs_austind as funcs
from ..utils.decorators import ignore_warnings
from ..visualization.PDF import PDF
from .Property_calculator import Property_Calculator

# Rest optical line property naming functions

# def get_rest_optical_flux_contam_label(
#     line_names: list, flux_contamination_params: dict
# ):
#     assert all(
#         line_name in line_diagnostics.keys() for line_name in line_names
#     )
#     assert type(flux_contamination_params) == dict
#     flux_cont_keys = flux_contamination_params.keys()
#     if "mu" in flux_cont_keys and "sigma" in flux_cont_keys:
#         return (
#             f"{line_names[0]}_cont_G("
#             f"{flux_contamination_params['mu']:.1f},"
#             f"{flux_contamination_params['sigma']:.1f})"
#         )
#         # _{'+'.join(line_names[1:])}"
#     elif "mu" in flux_cont_keys and "sigma" not in flux_cont_keys:
#         return (
#             f"{line_names[0]}_cont_"
#             f"{flux_contamination_params['mu']:.1f}"
#         )
#         # _{'+'.join(line_names[1:])}"
#     elif len(flux_contamination_params) == 0:
#         return "+".join(line_names)
#     else:
#         raise NotImplementedError

# def get_rest_optical_flux_contam_scaling(
#     flux_contamination_params: dict, iters: int
# ):
#     assert type(flux_contamination_params) == dict
#     flux_cont_keys = flux_contamination_params.keys()
#     if "mu" in flux_cont_keys and "sigma" in flux_cont_keys:
#         return np.random.normal(
#             1.0 - flux_contamination_params["mu"],
#             flux_contamination_params["sigma"],
#             iters,
#         )
#     elif "mu" in flux_cont_keys and "sigma" not in flux_cont_keys:
#         return 1.0 - flux_contamination_params["mu"]
#     elif len(flux_contamination_params) == 0:
#         return 1.0
#     else:
#         raise NotImplementedError

# def _get_wav_line_precision(self, line_name: str, dz: float):
#     assert line_name in line_diagnostics.keys()
#     wav_rest = line_diagnostics[line_name]["line_wav"]
#     dlambda = dz * wav_rest / (1.0 + self.z)
#     return dlambda


class Rest_Frame_Property_Calculator(Property_Calculator):
    """Abstract base class for calculators deriving rest-frame
    properties from SED-fit photometry.

    Concrete subclasses compute quantities (e.g. UV continuum slope,
    absolute magnitude, dust attenuation, line fluxes) from the
    rest-frame photometry (`Photometry_rest`) associated with a given
    SED-fitting run, optionally propagating uncertainties by scattering
    fluxes into `n_chains` posterior samples and caching the result as a
    `PDF`. Any `pre_req_properties` are calculated first and their
    results are looked up (via `phot_rest.properties`/`property_PDFs`)
    inside `_calculate`.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used to compute this property.
    SED_fit_label : `str` or `Type[SED_code]`
        Label (or `SED_code` instance, from which the label is taken)
        identifying the SED-fitting run whose `Photometry_rest` this
        property is calculated from.
    pre_req_properties : `list` of `Rest_Frame_Property_Calculator`, optional
        Other rest-frame property calculators that must be run before
        this one, since this calculator's `_calculate` depends on their
        results. Default is `[]`.
    **global_kwargs : `dict`
        Additional keyword arguments specific to the subclass, stored on
        `global_kwargs` and validated by `_kwarg_assertions`.

    Attributes
    ----------
    SED_fit_label : `str`
        Label identifying the SED-fitting run associated with this
        calculator.
    pre_req_properties : `list` of `Rest_Frame_Property_Calculator`
        Prerequisite property calculators run before this one.
    global_kwargs : `dict`
        Subclass-specific keyword arguments supplied at construction.
    aper_diam : `astropy.units.Quantity`
        Aperture diameter associated with this calculator, set by
        `Property_Calculator.__init__`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        pre_req_properties: List[Rest_Frame_Property_Calculator] = [],
        **global_kwargs,
    ) -> None:
        # self.aper_diam = aper_diam
        if isinstance(SED_fit_label, SED_code):
            SED_fit_label = SED_fit_label.label
        self.SED_fit_label = SED_fit_label
        self.pre_req_properties = pre_req_properties
        self.global_kwargs = global_kwargs
        self._kwarg_assertions()
        super().__init__(aper_diam)

    def __call__(
        self: Self,
        object: Union[Type[Catalogue_Base], Galaxy, Photometry_rest],
        n_chains: int = 10_000,
        output: bool = True,
        overwrite: bool = False,
        n_jobs: int = 1,
    ) -> Optional[Union[Type[Catalogue_Base], Galaxy, Photometry_rest]]:
        """Calculate and cache rest-frame properties for
        galaxy/catalogue objects.

        Computes prerequisite properties first, then calculates this property.
        PDFs are cached to disk when applicable.

        Parameters
        ----------
        object : `Catalogue_Base`, `Galaxy`, or `Photometry_rest`
            Object to process (catalogue, single galaxy, or photometry).
        n_chains : `int`, optional
            Number of PDF chains for uncertainty estimation. Default is
            `10_000`.
        output : `bool`, optional
            Whether to return the modified object. Default is `True`.
        overwrite : `bool`, optional
            Whether to overwrite existing cached properties. Default is
            `False`.
        n_jobs : `int`, optional
            Number of parallel jobs for catalogue processing. Default is `1`.

        Returns
        -------
        Modified object or `None`
            If `output=True`, returns the modified object. Otherwise
            returns `None`.
        """
        # calculate pre-requisite properties first
        [
            rest_frame_property(
                object,
                n_chains,
                output=False,
                overwrite=overwrite,
                n_jobs=n_jobs,
            )
            for rest_frame_property in self.pre_req_properties
        ]
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
            obj = self._call_cat(
                object, n_chains, output, overwrite, n_jobs=n_jobs
            )
        elif isinstance(object, Galaxy):
            obj = self._call_gal(object, n_chains, output, overwrite)
        elif isinstance(object, Photometry_rest):
            obj = self._call_phot_rest(object, n_chains, output, overwrite)
        else:
            err_message = (
                f"{object=} with {type(object)=} "
                + "not in [Catalogue, Galaxy, Photometry_rest]"
            )
            galfind_logger.critical(err_message)
            raise TypeError(err_message)
        if output:
            return obj

    def _call_cat(
        self: Self,
        cat: Catalogue,
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        n_jobs: int = 1,
        dtype: np.dtype = np.float32,
    ) -> Optional[Catalogue]:
        """Calculate and cache properties for all galaxies in a catalogue.

        Parameters
        ----------
        cat : `Catalogue`
            Catalogue of galaxies to process.
        n_chains : `int`, optional
            Number of PDF chains. Default is `10_000`.
        output : `bool`, optional
            Whether to return the modified catalogue. Default is `False`.
        overwrite : `bool`, optional
            Whether to overwrite existing properties. Default is `False`.
        n_jobs : `int`, optional
            Number of parallel jobs. Default is `1`.
        dtype : `numpy.dtype`, optional
            Floating-point precision for saved arrays. Default is
            `numpy.float32`.

        Returns
        -------
        `Catalogue` or `None`
            Modified catalogue if `output=True`, else `None`.
        """
        assert isinstance(n_jobs, int), galfind_logger.critical(
            f"{n_jobs=} with {type(n_jobs)=} != int"
        )
        try:
            save_dir = (
                f"{config['PhotProperties']['PDF_SAVE_DIR']}/"
                f"{cat.version}/{cat.survey}/{cat.filterset.instrument_name}/"
                + f"{self.aper_diam.to(u.arcsec).value:.2f}as"
                + f"/{self.SED_fit_label}/{self.name}"
            )
        except Exception:
            breakpoint()
        if n_jobs <= 1:
            # update properties for each galaxy in the catalogue
            [
                self._call_gal(
                    gal,
                    n_chains=n_chains,
                    output=False,
                    overwrite=overwrite,
                    save_dir=save_dir,
                )
                for gal in tqdm(
                    cat,
                    total=len(cat),
                    desc=f"Calculating {self.name}",
                    disable=galfind_logger.getEffectiveLevel() > logging.INFO,
                )
            ]
        else:
            # TODO: should be set when serializing the object
            for gal in tqdm(
                cat,
                total=len(cat),
                disable=galfind_logger.getEffectiveLevel() > logging.INFO,
            ):
                for label in gal.aper_phot[self.aper_diam].SED_results.keys():
                    try:
                        gal.aper_phot[self.aper_diam].flux = gal.aper_phot[
                            self.aper_diam
                        ].flux.unmasked
                    except Exception:
                        pass
                    try:
                        gal.aper_phot[
                            self.aper_diam
                        ].flux_errs = gal.aper_phot[
                            self.aper_diam
                        ].flux_errs.unmasked
                    except Exception:
                        pass
                    try:
                        gal.aper_phot[self.aper_diam].SED_results[
                            label
                        ].phot_rest.flux = (
                            gal.aper_phot[self.aper_diam]
                            .SED_results[label]
                            .phot_rest.flux.unmasked
                        )
                    except Exception:
                        pass
                    try:
                        gal.aper_phot[self.aper_diam].SED_results[
                            label
                        ].phot_rest.flux_errs = (
                            gal.aper_phot[self.aper_diam]
                            .SED_results[label]
                            .phot_rest.flux_errs.unmasked
                        )
                    except Exception:
                        pass
            # multi-process with joblib
            # sort input params
            params_arr = [
                (self, gal, n_chains, overwrite, save_dir, dtype)
                for gal in cat
            ]
            # run in parallel
            with funcs.tqdm_joblib(
                tqdm(
                    desc=f"Calculating {self.name} for "
                    f"{cat.survey} {cat.version} "
                    f"{cat.filterset.instrument_name}",
                    total=len(cat),
                )
            ):
                with parallel_config(backend="loky", n_jobs=n_jobs):
                    gals = Parallel()(
                        delayed(self._call_gal_multi_process)(params)
                        for params in params_arr
                    )
            cat.gals = gals
        # if cat.cat_creator.crops == []:
        self._update_fits_cat(cat)
        if output:
            return cat

    @staticmethod
    def _call_gal_multi_process(params: Dict[str, Any]) -> NoReturn:
        self, gal, n_chains, overwrite, save_dir, dtype = params
        return self._call_gal(
            gal,
            n_chains=n_chains,
            output=True,
            overwrite=overwrite,
            save_dir=save_dir,
            dtype=dtype,
        )

    def _update_fits_cat(
        self: Self,
        cat: Catalogue,
    ) -> NoReturn:
        # TODO: generalize this funciton further
        # determine appropriate hdu and name to save properties as
        # TODO: generalize hdu name for non-EAZY SED fitting labels
        property_hdu = (
            f"PROPERTIES_{'_'.join(self.SED_fit_label.split('_')[:-1])}"
        ).upper()
        property_name = (
            f"{self.name}_{self.aper_diam.to(u.arcsec).value:.2f}as"
        )
        # open fits catalogue
        tab = cat.open_cat(hdu=property_hdu)
        if tab is None:
            write = True
            tab = Table()
            tab["ID"] = np.array([gal.ID for gal in cat])
        elif property_name not in tab.colnames:
            if len(tab) == len(cat):
                write = True
            else:
                write = False
        else:
            write = False

        if write:
            if all(
                [
                    self.SED_fit_label
                    in gal.aper_phot[self.aper_diam].SED_results.keys()
                    for gal in cat
                ]
            ):
                property_vals = np.zeros(len(cat), dtype=np.float32)
                property_l1 = np.zeros(len(cat), dtype=np.float32)
                property_u1 = np.zeros(len(cat), dtype=np.float32)
                for i, gal in enumerate(cat):
                    phot_rest_ = (
                        gal.aper_phot[self.aper_diam]
                        .SED_results[self.SED_fit_label]
                        .phot_rest
                    )
                    if np.isnan(phot_rest_.properties[self.name]):
                        property_vals[i] = np.nan
                    else:
                        property_vals[i] = phot_rest_.properties[
                            self.name
                        ].value
                    if phot_rest_.property_PDFs[self.name] is None:
                        property_l1[i] = np.nan
                        property_u1[i] = np.nan
                    else:
                        # extract errors from PDFs storing chains
                        property_l1[i] = (
                            property_vals[i]
                            - phot_rest_.property_PDFs[self.name]
                            .get_percentile(16.0)
                            .value
                        )
                        property_u1[i] = (
                            phot_rest_.property_PDFs[self.name]
                            .get_percentile(84.0)
                            .value
                            - property_vals[i]
                        )

                kwarg_names = np.unique(
                    [
                        key
                        for gal in cat
                        for key in gal.aper_phot[self.aper_diam]
                        .SED_results[self.SED_fit_label]
                        .phot_rest.property_kwargs[self.name]
                        .keys()
                    ]
                )
                for kwarg_name in kwarg_names:
                    tab[f"{property_name}_{kwarg_name}"] = [
                        gal.aper_phot[self.aper_diam]
                        .SED_results[self.SED_fit_label]
                        .phot_rest.property_kwargs[self.name][kwarg_name]
                        if kwarg_name
                        in gal.aper_phot[self.aper_diam]
                        .SED_results[self.SED_fit_label]
                        .phot_rest.property_kwargs[self.name]
                        .keys()
                        else np.nan
                        for gal in cat
                    ]

            else:
                raise (
                    ValueError(
                        galfind_logger.critical(
                            f"{self.SED_fit_label=} not in all "
                            f"galaxies in {cat.survey} {cat.version}"
                        )
                    )
                )
            # update fits catalogue
            tab[property_name] = property_vals
            tab[f"{property_name}_l1"] = property_l1
            tab[f"{property_name}_u1"] = property_u1
            cat.write_hdu(tab, hdu=property_hdu)

    def _call_gal(
        self: Self,
        gal: Galaxy,
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        save_dir: Optional[str] = None,
        dtype: np.dtype = np.float32,
    ) -> Optional[Galaxy]:
        """Update the relevant Photometry_rest object stored in the Galaxy.

        Parameters
        ----------
        gal : `Galaxy`
            Galaxy object to process.
        n_chains : `int`, optional
            Number of PDF chains. Default is `10_000`.
        output : `bool`, optional
            Whether to return the modified galaxy. Default is `False`.
        overwrite : `bool`, optional
            Whether to overwrite existing properties. Default is `False`.
        save_dir : `str` or `None`, optional
            Directory to save PDFs. If `None`, auto-constructs path from
            config.
            If provided, should already include property name in path.
            Default is `None`.
        dtype : `numpy.dtype`, optional
            Floating-point precision for saved arrays. Default is
            `numpy.float32`.

        Returns
        -------
        `Galaxy` or `None`
            Modified galaxy if `output=True`, else `None`.
        """
        # update the relevant Photometry_rest object stored in the Galaxy
        assert self.aper_diam in gal.aper_phot.keys(), galfind_logger.critical(
            f"{self.aper_diam=} not in {'+'.join(list(gal.aper_phot.keys()))}"
        )
        assert (
            self.SED_fit_label
            in gal.aper_phot[self.aper_diam].SED_results.keys()
        ), galfind_logger.critical(
            f"{self.SED_fit_label=} not in "
            + "+".join(list(gal.aper_phot[self.aper_diam].SED_results.keys()))
        )
        # Auto-construct save_dir if not provided
        if save_dir is None:
            # Use full path structure if survey and version are
            # available, else use simpler fallback
            if gal.survey is not None and gal.version is not None:
                save_dir = (
                    f"{config['PhotProperties']['PDF_SAVE_DIR']}/"
                    + f"{gal.version}/{gal.survey}/"
                    + f"{gal.cat_filterset.instrument_name}/"
                    + f"{self.aper_diam.to(u.arcsec).value:.2f}as"
                    + f"/{self.SED_fit_label}/{self.name}"
                )
            else:
                # Fallback path for galaxies without survey/version metadata
                save_dir = (
                    f"{config['PhotProperties']['PDF_SAVE_DIR']}/"
                    + f"uncatalogued/{gal.cat_filterset.instrument_name}/"
                    + f"{self.aper_diam.to(u.arcsec).value:.2f}as"
                    + f"/{self.SED_fit_label}/{self.name}"
                )
        else:
            save_dir = save_dir.rstrip("/")
        save_path = f"{save_dir}/{gal.ID}.npy"
        self._call_phot_rest(
            gal.aper_phot[self.aper_diam]
            .SED_results[self.SED_fit_label]
            .phot_rest,
            n_chains=n_chains,
            output=False,
            overwrite=overwrite,
            save_path=save_path,
            dtype=dtype,
        )

        if output:
            return gal

    def _call_phot_rest(
        self: Self,
        phot_rest: Photometry_rest,
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        save_path: Optional[str] = None,
        save_scattered_fluxes: bool = False,
        dtype: np.dtype = np.float32,
    ) -> Optional[Photometry_rest]:
        """Calculate and cache rest-frame properties for photometry.

        Parameters
        ----------
        phot_rest : `Photometry_rest`
            Rest-frame photometry object to calculate properties for.
        n_chains : `int`, optional
            Number of PDF chains for uncertainty estimation. Default is
            `10_000`.
        output : `bool`, optional
            Whether to return the modified photometry. Default is `False`.
        overwrite : `bool`, optional
            Whether to overwrite existing properties. Default is `False`.
        save_path : `str` or `None`, optional
            Full path (including filename) to save PDF. If `None`, PDFs are not
            saved to disk. Default is `None`.
        save_scattered_fluxes : `bool`, optional
            Whether to save scattered fluxes alongside PDF. Default is `False`.
        dtype : `numpy.dtype`, optional
            Floating-point precision for saved arrays. Default is
            `numpy.float32`.

        Returns
        -------
        `Photometry_rest` or `None`
            Modified photometry if `output=True`, else `None`.
        """
        property_name = self.name
        calculated = False
        # if any pre-requisite properties are NaN, set this property to NaN
        if property_name in phot_rest.properties.keys():
            properties_to_nan_check = [self] + self.pre_req_properties
        else:
            properties_to_nan_check = self.pre_req_properties
        if any(
            np.isnan(phot_rest.properties[property.name])
            for property in properties_to_nan_check
        ):
            phot_rest.properties[property_name] = np.nan
            if n_chains > 1:
                phot_rest.property_errs[property_name] = np.array(
                    [np.nan, np.nan]
                )
                phot_rest.property_PDFs[property_name] = None
                phot_rest.property_kwargs[property_name] = {}
        else:
            if n_chains <= 1:
                if (
                    property_name not in phot_rest.properties.keys()
                    or overwrite
                ):
                    galfind_logger.debug(
                        "Calculating basic "
                        + f"{property_name} for {repr(phot_rest)}"
                    )
                    self.obj_kwargs = self._calc_obj_kwargs(phot_rest)
                    if self._fail_criteria(phot_rest):
                        phot_rest.properties[property_name] = np.nan
                    else:
                        # breakpoint()
                        value = self._calculate(
                            np.array(
                                [
                                    phot_rest.flux[
                                        self.obj_kwargs["keep_indices"]
                                    ].value
                                ]
                            )
                            * phot_rest.flux.unit,
                            phot_rest,
                        )[0]
                        if value is None:
                            phot_rest.properties[property_name] = np.nan
                        else:
                            phot_rest.properties[property_name] = value
                        calculated = True
                else:
                    galfind_logger.debug(
                        "Already calculated basic "
                        + f"{property_name} for {repr(phot_rest)}"
                    )
            else:
                # if PDF does not already exist in the object
                # but has been run before, load it if not wanting to overwrite
                if save_path is not None and Path(save_path).is_file():
                    if (
                        property_name not in phot_rest.property_PDFs.keys()
                        and not overwrite
                    ):
                        PDF_obj = PDF.from_npy(save_path)
                        galfind_logger.debug(
                            f"Loading {len(PDF_obj)=} {property_name}"
                            + f" PDF in {repr(phot_rest)}"
                        )
                        phot_rest.property_PDFs[property_name] = PDF_obj
                        phot_rest.properties[property_name] = PDF_obj.median
                        phot_rest.property_errs[property_name] = PDF_obj.errs
                        phot_rest.property_kwargs[property_name] = (
                            PDF_obj.kwargs
                        )
                    elif (
                        property_name in phot_rest.property_PDFs.keys()
                        and not overwrite
                    ):
                        galfind_logger.debug(
                            f"Already loaded {property_name} PDF "
                            f"in {repr(phot_rest)}"
                        )
                        if output:
                            return phot_rest
                        else:
                            return
                    # else: overwrite=True -> fall through and
                    # recompute below, regardless of what's cached on
                    # disk or already loaded in memory

                if (
                    property_name not in phot_rest.property_PDFs.keys()
                    or overwrite
                ):
                    n_new_chains = n_chains
                    galfind_logger.debug(
                        f"Creating {property_name} PDF in "
                        f"{repr(phot_rest)} from "
                        f"n={n_new_chains} chains"
                    )
                elif len(phot_rest.property_PDFs[property_name]) < n_chains:
                    n_new_chains = n_chains - len(
                        phot_rest.property_PDFs[property_name]
                    )
                    galfind_logger.debug(
                        f"Adding n={n_new_chains} {property_name} "
                        f"chains to {repr(phot_rest)}"
                    )
                else:
                    # len(phot_rest.property_PDFs[property_name])
                    # >= n_chains
                    n_new_chains = 0
                    galfind_logger.debug(
                        f"Already calculated "
                        f"n={len(phot_rest.property_PDFs[property_name])}"
                        + f" {property_name} chains to {repr(phot_rest)}"
                    )
                # breakpoint()
                if n_new_chains > 0:
                    self.obj_kwargs = self._calc_obj_kwargs(phot_rest)
                    if self._fail_criteria(phot_rest):
                        phot_rest.property_PDFs[property_name] = None
                        phot_rest.properties[property_name] = np.nan
                        phot_rest.property_errs[property_name] = np.array(
                            [np.nan, np.nan]
                        )
                        phot_rest.property_kwargs[property_name] = {}
                    else:  # phot_rest has not failed
                        galfind_logger.debug("Making PDF")
                        PDF_obj, scattered_fluxes = self._make_PDF(
                            phot_rest, n_new_chains, dtype=dtype
                        )
                        galfind_logger.debug("PDF made")
                        if PDF_obj is None:
                            phot_rest.property_PDFs[property_name] = None
                            phot_rest.properties[property_name] = np.nan
                            phot_rest.property_errs[property_name] = np.array(
                                [np.nan, np.nan]
                            )
                            phot_rest.property_kwargs[property_name] = {}
                        else:
                            if n_new_chains != n_chains:
                                if save_scattered_fluxes:
                                    # load old scattered fluxes
                                    old_scattered_fluxes = 0.0
                                    scattered_fluxes = np.concatenate(
                                        [
                                            old_scattered_fluxes,
                                            scattered_fluxes,
                                        ]
                                    )
                                PDF_obj = (
                                    phot_rest.property_PDFs[property_name]
                                    + PDF_obj
                                )
                            phot_rest.property_PDFs[property_name] = PDF_obj
                            phot_rest.properties[property_name] = (
                                PDF_obj.median
                            )
                            phot_rest.property_errs[property_name] = (
                                PDF_obj.errs
                            )
                            # update saved PDF
                            if save_path is not None:
                                funcs.make_dirs(save_path)
                                if (
                                    phot_rest.property_PDFs[property_name]
                                    is not None
                                    and save_path is not None
                                ):
                                    if save_scattered_fluxes:
                                        np.save(
                                            save_path.replace(
                                                ".npy", "_scattered_fluxes.npy"
                                            ),
                                            scattered_fluxes.value,
                                        )
                                        galfind_logger.debug(
                                            "Scattered fluxes saved"
                                        )
                                    phot_rest.property_PDFs[
                                        property_name
                                    ].save(save_path)
                                    file_stem = save_path.split("/")[
                                        -1
                                    ].replace(".npy", "")
                                    galfind_logger.debug(
                                        f"PDF saved for {file_stem}"
                                    )
                            calculated = True
            if calculated:
                phot_rest.property_kwargs[property_name] = (
                    self._get_output_kwargs(phot_rest)
                )

        if output:
            return phot_rest

    def _make_PDF(
        self: Self,
        phot_rest: Photometry_rest,
        n_chains: int,
        dtype: np.dtype = np.float32,
    ) -> Tuple[PDF, u.Quantity]:
        # ensure the type is a float
        assert "float" in dtype.__name__, galfind_logger(
            f"{dtype=} is not a float type"
        )
        # try:
        # scatter relevant photometric data points n_chains times
        if "keep_indices" in self.obj_kwargs.keys():
            cropped_phot_rest = phot_rest[self.obj_kwargs["keep_indices"]]
        else:
            cropped_phot_rest = deepcopy(phot_rest)
        scattered_fluxes = cropped_phot_rest.scatter_fluxes(n_chains)
        # calculate chain
        galfind_logger.debug(f"Calculating {self.name} chains")
        vals = self._calculate(scattered_fluxes, phot_rest)

        if vals is not None:
            # increase the floating point precision of saved array if required
            while any(
                val.value > np.finfo(dtype).max
                or val.value < np.finfo(dtype).min
                for val in vals
            ):
                new_dtype_precision = (
                    int(dtype.__name__.replace("float", "")) * 2
                )
                dtype = getattr(np, f"float{new_dtype_precision}")
            # update datatype of vals
            vals = vals.astype(dtype)
        galfind_logger.debug(f"{self.name} chains calculated")
        # construct PDF object
        try:
            if vals is None:
                PDF_obj = None
            else:
                PDF_obj = PDF.from_1D_arr(
                    self.name, vals, kwargs=self._get_output_kwargs(phot_rest)
                )
        except Exception:
            breakpoint()
        return PDF_obj, scattered_fluxes

    def extract_vals(
        self: Self,
        object: Union[Type[Catalogue_Base], Galaxy, Photometry_rest],
    ) -> Union[u.Quantity, u.Magnitude, u.Dex]:
        """Extract the previously-calculated property value(s) from an object.

        Parameters
        ----------
        object : `Catalogue_Base`, `Galaxy`, or `Photometry_rest`
            The object to extract this property's value(s) from.

        Returns
        -------
        `astropy.units.Quantity`, `astropy.units.Magnitude`, or
        `astropy.units.Dex`
            The property value for a single `Galaxy` or `Photometry_rest`,
            or an array of values (one per galaxy, with consistent units)
            for a `Catalogue_Base` subclass instance.

        Raises
        ------
        AssertionError
            If the per-galaxy property units are not all consistent.
        TypeError
            If `object` is not a `Catalogue_Base` subclass instance,
            `Galaxy`, or `Photometry_rest`.
        """
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
            cat_vals = [
                gal.aper_phot[self.aper_diam]
                .SED_results[self.SED_fit_label]
                .phot_rest.properties[self.name]
                for gal in object
            ]
            cat_vals_no_nans = [val for val in cat_vals if not np.isnan(val)]
            if not all(isinstance(val, float) for val in cat_vals_no_nans):
                assert all(
                    val.unit == cat_vals[0].unit for val in cat_vals_no_nans
                ), galfind_logger.critical(
                    f"Units of {self.name} in {object} are not consistent"
                )
                cat_vals = (
                    np.array(
                        [
                            val.value if not np.isnan(val) else val
                            for val in cat_vals
                        ]
                    )
                    * cat_vals[0].unit
                )
            else:
                cat_vals = np.array(cat_vals)
            return cat_vals
        elif isinstance(object, Galaxy):
            return (
                object.aper_phot[self.aper_diam]
                .SED_results[self.SED_fit_label]
                .phot_rest.properties[self.name]
            )
        elif isinstance(object, Photometry_rest):
            return object.properties[self.name]
        else:
            err_message = (
                f"{object=} with {type(object)=} "
                + "not in ["
                + f"{', '.join(Catalogue_Base.__subclasses__())}, "
                + "Galaxy, Photometry_rest]"
            )
            galfind_logger.critical(err_message)
            raise TypeError(err_message)

    # TODO: Propagate from parent class
    def extract_errs(
        self: Self,
        object: Union[Type[Catalogue_Base], Galaxy, Photometry_rest],
    ) -> Union[u.Quantity, u.Magnitude, u.Dex]:
        """Extract the previously-calculated property
        uncertainty/uncertainties from an object.

        Parameters
        ----------
        object : `Catalogue_Base`, `Galaxy`, or `Photometry_rest`
            The object to extract this property's uncertainty/uncertainties
            from.

        Returns
        -------
        `astropy.units.Quantity`, `astropy.units.Magnitude`, or
        `astropy.units.Dex`
            The (lower, upper) uncertainty for a single `Galaxy` or
            `Photometry_rest`, or an array of uncertainties (one per
            galaxy, with consistent units) for a `Catalogue_Base`
            subclass instance.

        Raises
        ------
        AssertionError
            If the per-galaxy property error units are not all consistent.
        TypeError
            If `object` is not a `Catalogue_Base` subclass instance,
            `Galaxy`, or `Photometry_rest`.
        """
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
            cat_errs = [
                gal.aper_phot[self.aper_diam]
                .SED_results[self.SED_fit_label]
                .phot_rest.property_errs[self.name]
                for gal in object
            ]
            if all(
                isinstance(val, tuple([u.Quantity, u.Magnitude, u.Dex]))
                for val in cat_errs
            ):
                assert all(
                    val.unit == cat_errs[0].unit for val in cat_errs
                ), galfind_logger.critical(
                    f"Units of {self.name} in {object} are not consistent"
                )
                cat_errs = (
                    np.array([val.value for val in cat_errs])
                    * cat_errs[0].unit
                )
            else:
                cat_errs = np.array(cat_errs)
            return cat_errs
        elif isinstance(object, Galaxy):
            return (
                object.aper_phot[self.aper_diam]
                .SED_results[self.SED_fit_label]
                .phot_rest.property_errs[self.name]
            )
        elif isinstance(object, Photometry_rest):
            return object.property_errs[self.name]
        else:
            err_message = (
                f"{object=} with {type(object)=} "
                + "not in [Catalogue, Galaxy, Photometry_rest]"
            )
            galfind_logger.critical(err_message)
            raise TypeError(err_message)

    def extract_PDFs(
        self: Self,
        object: Union[Type[Catalogue_Base], Galaxy, Photometry_rest],
    ) -> Union[Type[PDF], List[Type[PDF]]]:
        """Extract the previously-calculated property PDF(s) from an object.

        Parameters
        ----------
        object : `Catalogue_Base`, `Galaxy`, or `Photometry_rest`
            The object to extract this property's PDF(s) from.

        Returns
        -------
        `PDF` or `list` of `PDF`
            The PDF (or `None` if unavailable) for a single `Galaxy` or
            `Photometry_rest`, or a list of PDFs (one per galaxy) for a
            `Catalogue_Base` subclass instance.

        Raises
        ------
        TypeError
            If `object` is not a `Catalogue_Base` subclass instance,
            `Galaxy`, or `Photometry_rest`.
        """
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
            return [
                gal.aper_phot[self.aper_diam]
                .SED_results[self.SED_fit_label]
                .phot_rest.property_PDFs[self.name]
                for gal in object
            ]
        elif isinstance(object, Galaxy):
            return (
                object.aper_phot[self.aper_diam]
                .SED_results[self.SED_fit_label]
                .phot_rest.property_PDFs[self.name]
            )
        elif isinstance(object, Photometry_rest):
            return object.property_PDFs[self.name]
        else:
            err_message = (
                f"{object=} with {type(object)=} "
                + "not in ["
                + f"{', '.join(Catalogue_Base.__subclasses__())}, "
                + "Galaxy, Photometry_rest]"
            )
            galfind_logger.critical(err_message)
            raise TypeError(err_message)

    @abstractmethod
    def _kwarg_assertions(self: Self) -> None:
        pass

    @abstractmethod
    def _calc_obj_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        pass

    @abstractmethod
    def _calculate(
        self: Self, fluxes_arr: u.Quantity, phot_rest: Photometry_rest
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        pass

    @abstractmethod
    def _get_output_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        pass


class beta_fit:
    """Callable model function for fitting a UV power-law (beta)
    spectrum to photometry.

    Pre-computes, for each filter in `filterset`, the rest-frame
    wavelength grid, transmission curve and normalisation (integral of
    the transmission curve over the rest-frame wavelength grid) needed
    to synthesise power-law photometry via `get_fluxes`. Instances are
    callable with a signature compatible with e.g.
    `scipy.optimize.curve_fit`.

    Parameters
    ----------
    z : `float`
        Redshift used to convert observed-frame filter wavelengths to
        the rest frame.
    filterset : `Multiple_Filter`
        Set of filters to synthesise power-law photometry through.

    Attributes
    ----------
    filterset : `Multiple_Filter`
        Filters associated with this fit.
    wavelength_rest : `dict`
        Rest-frame wavelength grid (`numpy.ndarray`, in Angstrom) for
        each filter, keyed by filter name.
    mid_wav_rest : `dict`
        Rest-frame central wavelength (in Angstrom) for each filter,
        keyed by filter name.
    transmission : `dict`
        Filter transmission curve (`numpy.ndarray`) for each filter,
        keyed by filter name.
    norm : `dict`
        Normalisation (integral of the transmission curve over the
        rest-frame wavelength grid) for each filter, keyed by filter name.
    """

    def __init__(self: Self, z: float, filterset: Multiple_Filter) -> NoReturn:
        self.filterset = filterset
        self.wavelength_rest = {}
        self.mid_wav_rest = {}
        self.transmission = {}
        self.norm = {}
        max_length = np.max([len(filt.wav) for filt in filterset])
        for filt in filterset:
            wav_rest = np.array(
                funcs.convert_wav_units(filt.wav, u.AA).value / (1.0 + z)
            )
            trans = np.array(filt.trans)
            length = len(wav_rest)
            if length != max_length:
                wav_rest = np.concatenate(
                    [wav_rest, np.full(max_length - length, wav_rest[-1])]
                )
                trans = np.concatenate(
                    [trans, np.full(max_length - length, trans[-1])]
                )
            self.mid_wav_rest[filt.filt_name] = filt.WavelengthCen.to(
                u.AA
            ).value
            self.wavelength_rest[filt.filt_name] = wav_rest
            self.transmission[filt.filt_name] = trans
            self.norm[filt.filt_name] = np.trapezoid(
                self.transmission[filt.filt_name],
                x=self.wavelength_rest[filt.filt_name],
            )

    def __call__(self, _, A, beta):
        return self.get_fluxes(
            A,
            beta,
            self.wavelength_rest,
            self.transmission,
            self.norm,
            self.filterset.filt_names,
        )


@njit
def get_fluxes(wav_rest, A, beta, trans, norm):
    """Synthesise per-filter fluxes for a UV power-law spectrum.

    For each filter, integrates the power-law spectrum
    :math:`f_\\lambda = 10^{A} \\lambda_{\\mathrm{rest}}^{\\beta}` against
    the filter's rest-frame transmission curve and divides by the
    filter's normalisation, giving the transmission-weighted mean flux
    density in that band.

    Parameters
    ----------
    wav_rest : `numpy.ndarray`
        Rest-frame wavelength grid for each filter, one row per filter.
    A : `float`
        Power-law amplitude (in log10 space).
    beta : `float`
        Power-law (UV continuum slope) index.
    trans : `numpy.ndarray`
        Filter transmission curve for each filter, one row per filter,
        matching the shape of `wav_rest`.
    norm : `numpy.ndarray`
        Normalisation (integral of the transmission curve over
        `wav_rest`) for each filter.

    Returns
    -------
    `numpy.ndarray`
        Synthesised flux density in each filter.
    """
    return np.array(
        [
            np.trapezoid(
                (10**A) * (wav_rest[i] ** beta) * trans[i],
                x=wav_rest[i],
            )
            / norm[i]
            for i in range(len(wav_rest))
        ]
    )


@njit
def fit_beta_gradient_descent(
    wav_rest,
    mid_wav_rest,
    flux,
    trans,
    norm,
    init_A,
    init_beta,
    learning_rate=1e-6,
    max_iter=1000,
    tol=1e-6,
):
    """Perform gradient descent to minimize the residual sum of squares.

    Fits power-law parameters A and beta to synthesized photometry by
    minimizing the residual sum of squares via gradient descent.

    Parameters
    ----------
    wav_rest : `numpy.ndarray`
        Rest-frame wavelength grid for each filter, one row per filter.
    mid_wav_rest : `numpy.ndarray`
        Mid-wavelength (in Angstrom) for each filter.
    flux : `numpy.ndarray`
        Observed photometric fluxes for each filter.
    trans : `numpy.ndarray`
        Filter transmission curve for each filter, one row per filter,
        matching the shape of `wav_rest`.
    norm : `numpy.ndarray`
        Normalisation (integral of the transmission curve over
        `wav_rest`) for each filter.
    init_A : `float`
        Initial power-law amplitude (in log10 space).
    init_beta : `float`
        Initial power-law (UV continuum slope) index.
    learning_rate : `float`, optional
        Learning rate for gradient descent updates. Default is `1e-6`.
    max_iter : `int`, optional
        Maximum number of gradient descent iterations. Default is `1000`.
    tol : `float`, optional
        Convergence tolerance for gradient magnitude. Default is `1e-6`.

    Returns
    -------
    `float`
        The fitted power-law slope (beta).
    """
    A = init_A
    beta = init_beta
    for i in range(max_iter):
        model_flux = get_fluxes(wav_rest, A, beta, trans, norm)
        residuals = flux - model_flux
        n = len(residuals)
        # Compute sums for least squares
        sum_res = 0.0
        sum_wav_res = 0.0
        for i in range(n):
            sum_res += residuals[i]
            sum_wav_res += mid_wav_rest[i] * residuals[i]
        # print(residuals)
        # print(sum_res, sum_wav_res)
        # Compute gradients
        grad_A = -2 * sum_res
        grad_beta = -2 * sum_wav_res

        # Update parameters
        A -= learning_rate * grad_A
        beta -= learning_rate * grad_beta

        # Check for convergence
        if np.sqrt(grad_A**2 + grad_beta**2) < tol:
            break

    return (
        beta  # A, beta, i  # Return optimized parameters and iterations taken
    )


def rest_UV_wavs_name(rest_UV_wav_lims):
    """Build a compact label string for a pair of rest-frame UV
    wavelength limits.

    Parameters
    ----------
    rest_UV_wav_lims : `astropy.units.Quantity`
        Two-element array of rest-frame UV wavelength limits.

    Returns
    -------
    `str`
        The wavelength limits (converted to integer Angstrom), formatted
        as e.g. ``"[1250,3000]AA"``.
    """
    rest_UV_wav_lims = [
        int(
            funcs.convert_wav_units(
                rest_UV_wav_lim * rest_UV_wav_lims.unit, u.AA
            ).value
        )
        for rest_UV_wav_lim in rest_UV_wav_lims.value
    ]
    return f"{str(rest_UV_wav_lims).replace(' ', '')}AA"


class UV_Beta_Calculator(Rest_Frame_Property_Calculator):
    """Calculates the rest-frame UV continuum slope, beta.

    Fits a power-law :math:`f_\\lambda \\propto \\lambda^{\\beta}` (via a
    linear fit in log-log space) to the photometric bands that fall
    entirely within the rest-frame UV wavelength limits, giving the UV
    continuum slope beta.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used to compute this property.
    SED_fit_label : `str` or `Type[SED_code]`
        Label (or `SED_code` instance) identifying the SED-fitting run
        whose `Photometry_rest` this property is calculated from.
    rest_UV_wav_lims : `astropy.units.Quantity`, optional
        Two-element rest-frame wavelength range defining the UV
        continuum. Default is `[1_250.0, 3_000.0] * u.AA`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
    ) -> NoReturn:
        global_kwargs = {"rest_UV_wav_lims": rest_UV_wav_lims}
        super().__init__(aper_diam, SED_fit_label, [], **global_kwargs)

    @property
    def name(self: Self) -> str:
        """`str`: Short identifier, ``"beta_{rest_UV_wav_lims}"``."""
        return (
            f"beta_{rest_UV_wavs_name(self.global_kwargs['rest_UV_wav_lims'])}"
        )

    @property
    def plot_name(self: Self) -> str:
        """`str`: Human-readable plot label for the UV continuum slope."""
        return r"$\beta_{\mathrm{UV}}$"

    def _kwarg_assertions(self: Self) -> None:
        assert (
            u.get_physical_type(self.global_kwargs["rest_UV_wav_lims"])
            == "length"
        )
        assert len(self.global_kwargs["rest_UV_wav_lims"]) == 2
        assert (
            self.global_kwargs["rest_UV_wav_lims"][0]
            < self.global_kwargs["rest_UV_wav_lims"][1]
        )
        assert self.global_kwargs["rest_UV_wav_lims"][0] > 1_216.0 * u.AA
        assert self.global_kwargs["rest_UV_wav_lims"][1] < 3_646.0 * u.AA

    def _calc_obj_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        # determine bands that fall within rest frame UV wavelength limits
        # remove any nans from rest_UV_SNRs
        nan_SNR_indices = np.isnan(phot_rest.flux / phot_rest.flux_errs)
        rest_frame_UV_indices = [
            i
            for i, filt in enumerate(phot_rest.filterset)
            if filt.WavelengthLower50
            > self.global_kwargs["rest_UV_wav_lims"][0]
            * (1.0 + phot_rest.z.value)
            and filt.WavelengthUpper50
            < self.global_kwargs["rest_UV_wav_lims"][1]
            * (1.0 + phot_rest.z.value)
            and not nan_SNR_indices[i]
        ]
        if len(rest_frame_UV_indices) < 2:
            failure = True
        else:
            rest_UV_band_wavs = (
                np.array(
                    [
                        funcs.convert_wav_units(
                            filt.WavelengthCen / (1.0 + phot_rest.z.value),
                            u.AA,
                        ).value
                        for filt in phot_rest.filterset[rest_frame_UV_indices]
                    ]
                )
                * u.AA
            )
            phot_rest_UV = phot_rest[rest_frame_UV_indices]
            rest_UV_SNRs = phot_rest_UV.flux / phot_rest_UV.flux_errs
            failure = False
            # determine what percentage of scatters fall into the
            # negative flux region
            negative_flux_pc = (
                1.0
                - np.prod(
                    [
                        1.0 - norm.cdf(0.0, loc=mu, scale=std)
                        for mu, std in zip(
                            phot_rest_UV.flux.value,
                            phot_rest_UV.flux_errs.value,
                        )
                    ]
                )
            ) * 100.0
        if failure:
            rest_frame_UV_indices = None
            rest_UV_band_wavs = None
            rest_UV_SNRs = None
            negative_flux_pc = None

        return {
            "keep_indices": rest_frame_UV_indices,
            "rest_UV_band_wavs": rest_UV_band_wavs,
            "rest_UV_SNRs": rest_UV_SNRs,
            "negative_flux_pc": negative_flux_pc,
        }

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        if (
            any(value is None for value in self.obj_kwargs.values())
            or self.obj_kwargs["negative_flux_pc"] > 99.0
        ):
            return True
        else:
            return False

    @ignore_warnings
    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # appropriately convert flux units
        fluxes_arr = np.log10(
            funcs.convert_mag_units(
                self.obj_kwargs["rest_UV_band_wavs"],
                fluxes_arr,
                u.erg / (u.s * u.AA * u.cm**2),
            ).value
        )
        # ensure fluxes_arr is always 2D (n_chains, n_bands)
        if fluxes_arr.ndim == 1:
            fluxes_arr = fluxes_arr[np.newaxis, :]
        beta_arr = (
            np.array(
                [
                    funcs.linear_fit(
                        np.log10(
                            self.obj_kwargs["rest_UV_band_wavs"].value,
                            dtype=np.float64,
                        ),
                        fluxes,
                    )[0]
                    for fluxes in fluxes_arr
                ]
            )
            * u.dimensionless_unscaled
        )
        return beta_arr

    def _get_output_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {
            "rest_UV_filt_names": "+".join(
                np.array(phot_rest.filterset.filt_names)[
                    self.obj_kwargs["keep_indices"]
                ]
            ),
            "n_UV_bands": len(self.obj_kwargs["keep_indices"]),
            "negative_flux_pc": self.obj_kwargs["negative_flux_pc"],
        }

    # save scattered fluxes to retain access to fit amplitudes
    def _call_phot_rest(
        self: Self,
        phot_rest: Photometry_rest,
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        save_path: Optional[str] = None,
        dtype: np.dtype = np.float32,
    ) -> Optional[Photometry_rest]:
        return super()._call_phot_rest(
            phot_rest,
            n_chains,
            output,
            overwrite,
            save_path,
            save_scattered_fluxes=True,
            dtype=dtype,
        )


class UV_Dust_Attenuation_Calculator(Rest_Frame_Property_Calculator):
    """Calculates the UV dust attenuation,
    A(ref_wav), from the UV continuum slope.

    Requires a `UV_Beta_Calculator` as a prerequisite, then converts the
    fitted UV continuum slope (beta) to a dust attenuation at `ref_wav`
    using the supplied `beta_dust_conv` beta-to-A(UV) conversion.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used to compute this property.
    SED_fit_label : `str` or `Type[SED_code]`
        Label (or `SED_code` instance) identifying the SED-fitting run
        whose `Photometry_rest` this property is calculated from.
    rest_UV_wav_lims : `astropy.units.Quantity`, optional
        Two-element rest-frame wavelength range defining the UV
        continuum used to fit beta. Default is `[1_250.0, 3_000.0] * u.AA`.
    beta_dust_conv : `str` or `Type[AUV_from_beta]`, optional
        Beta-to-A(UV) conversion to use, either the name of an
        `AUV_from_beta` subclass or an instance/class thereof. Default
        is `M99`.
    ref_wav : `astropy.units.Quantity`, optional
        Rest-frame reference wavelength at which the dust attenuation is
        computed. Default is `1_500.0 * u.AA`.
    keep_valid : `bool`, optional
        Whether to clip negative attenuations to zero. Default is `True`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        beta_dust_conv: Union[str, Type[AUV_from_beta]] = M99,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        keep_valid: bool = True,
    ) -> NoReturn:
        pre_req_properties = [
            UV_Beta_Calculator(aper_diam, SED_fit_label, rest_UV_wav_lims)
        ]
        if isinstance(beta_dust_conv, str):
            beta_dust_conv = [
                beta_dust_conv_cls()
                for beta_dust_conv_cls in AUV_from_beta.__subclasses__()
                if beta_dust_conv_cls.__name__ == beta_dust_conv
            ][0]
        elif not isinstance(beta_dust_conv, AUV_from_beta):
            beta_dust_conv = beta_dust_conv()
        global_kwargs = {
            "ref_wav": ref_wav,
            "beta_dust_conv": beta_dust_conv,
            "keep_valid": keep_valid,
        }
        super().__init__(
            aper_diam, SED_fit_label, pre_req_properties, **global_kwargs
        )

    @property
    def name(self: Self) -> str:
        """`str`: Short identifier,
        ``"A{ref_wav}_{beta_dust_conv}_{rest_UV_wav_lims}[_A>0]"``.
        """
        label = (
            f"A{self.global_kwargs['ref_wav'].to(u.AA).value:.0f}"
            + f"_{self.global_kwargs['beta_dust_conv'].__class__.__name__}"
            + "_"
            + rest_UV_wavs_name(
                self.pre_req_properties[0].global_kwargs["rest_UV_wav_lims"]
            )
        )
        if self.global_kwargs["keep_valid"]:
            label += "_A>0"
        return label

    @property
    def plot_name(self: Self) -> str:
        """`str`: Human-readable plot label for the dust
        attenuation at `ref_wav`.
        """
        ref_wav_label = f"{self.global_kwargs['ref_wav'].to(u.AA).value:.0f}"
        return rf"$A_{{{ref_wav_label}}}$"

    def _kwarg_assertions(self: Self) -> None:
        assert (
            self.global_kwargs["ref_wav"]
            > self.pre_req_properties[0].global_kwargs["rest_UV_wav_lims"][0]
        )
        assert (
            self.global_kwargs["ref_wav"]
            < self.pre_req_properties[0].global_kwargs["rest_UV_wav_lims"][1]
        )
        assert (
            self.global_kwargs["beta_dust_conv"].__class__
            in AUV_from_beta.__subclasses__()
        )
        assert isinstance(self.global_kwargs["keep_valid"], bool)

    def _calc_obj_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        # always pass
        return False

    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # calculate beta
        if len(fluxes_arr) > 1:
            beta_arr = phot_rest.property_PDFs[
                self.pre_req_properties[0].name
            ].input_arr
            assert len(fluxes_arr) == len(beta_arr)
        else:
            beta_arr = phot_rest.properties[self.pre_req_properties[0].name]
        # calculate A_UV
        A_UV_arr = self.global_kwargs["beta_dust_conv"](beta_arr)
        # limit to A_UV > 0
        if self.global_kwargs["keep_valid"]:
            A_UV_arr[A_UV_arr < 0.0] = 0.0
        return A_UV_arr

    def _get_output_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}


class Fesc_From_Beta_Calculator(Rest_Frame_Property_Calculator):
    """Calculates the Lyman-continuum escape fraction, fesc, from
    the UV continuum slope.

    Requires a `UV_Beta_Calculator` as a prerequisite, then converts the
    fitted UV continuum slope (beta) to an escape fraction using the
    `fesc_conv` conversion registered in
    `useful_funcs_austind.fesc_from_beta_conversions`.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used to compute this property.
    SED_fit_label : `str` or `Type[SED_code]`
        Label (or `SED_code` instance) identifying the SED-fitting run
        whose `Photometry_rest` this property is calculated from.
    rest_UV_wav_lims : `astropy.units.Quantity`, optional
        Two-element rest-frame wavelength range defining the UV
        continuum used to fit beta. Default is `[1_250.0, 3_000.0] * u.AA`.
    fesc_conv : `str`, optional
        Name of the beta-to-fesc conversion to use, a key into
        `useful_funcs_austind.fesc_from_beta_conversions`. Default is
        `"Chisholm22"`.
    keep_valid : `bool`, optional
        Whether to clip the escape fraction to the physical range
        `[0, 1]`. Default is `False`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        fesc_conv: str = "Chisholm22",
        keep_valid: bool = False,
    ) -> NoReturn:
        pre_req_properties = [
            UV_Beta_Calculator(aper_diam, SED_fit_label, rest_UV_wav_lims)
        ]
        global_kwargs = {"fesc_conv": fesc_conv, "keep_valid": keep_valid}
        super().__init__(
            aper_diam, SED_fit_label, pre_req_properties, **global_kwargs
        )

    @property
    def name(self: Self) -> str:
        """`str`: Short identifier,
        ``"fesc={fesc_conv}_{rest_UV_wav_lims}[_0<fesc<1]"``.
        """
        # if isinstance(self.global_kwargs["fesc_conv"], str):
        label = f"fesc={self.global_kwargs['fesc_conv']}_" + rest_UV_wavs_name(
            self.pre_req_properties[0].global_kwargs["rest_UV_wav_lims"]
        )
        # else: # float
        #    label = f"fesc={self.global_kwargs['fesc_conv']:.2f}"
        if self.global_kwargs["keep_valid"]:
            label += "_0<fesc<1"
        return label

    @property
    def plot_name(self: Self) -> str:
        """`str`: Human-readable plot label for the escape fraction."""
        return r"$f_{\mathrm{esc}}$"  # type of fesc here too

    def _kwarg_assertions(self: Self) -> None:
        # if isinstance(self.global_kwargs["fesc_conv"], str):
        assert (
            self.global_kwargs["fesc_conv"]
            in funcs.fesc_from_beta_conversions.keys()
        )
        # elif isinstance(self.global_kwargs["fesc_conv"], float):
        #     assert self.global_kwargs["fesc_conv"] >= 0.0
        #     assert self.global_kwargs["fesc_conv"] <= 1.0
        # else:
        #     raise ValueError("fesc_conv must be a string or float")
        assert isinstance(self.global_kwargs["keep_valid"], bool)

    def _calc_obj_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        # always pass
        return False

    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # calculate beta
        if len(fluxes_arr) > 1:
            beta_arr = phot_rest.property_PDFs[
                self.pre_req_properties[0].name
            ].input_arr
            assert len(fluxes_arr) == len(beta_arr)
        else:
            beta_arr = phot_rest.properties[self.pre_req_properties[0].name]
        # if isinstance(self.global_kwargs["fesc_conv"], str):
        fesc_arr = funcs.fesc_from_beta_conversions[
            self.global_kwargs["fesc_conv"]
        ](beta_arr)
        # else:
        #    fesc_arr = np.full_like(beta_arr, self.global_kwargs["fesc_conv"])
        if self.global_kwargs["keep_valid"]:
            fesc_arr[fesc_arr < 0.0] = 0.0
            fesc_arr[fesc_arr > 1.0] = 1.0
        return fesc_arr

    def _get_output_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}


class mUV_Calculator(Rest_Frame_Property_Calculator):
    """Calculates the apparent UV magnitude, mUV, at a rest-frame
    reference wavelength.

    Requires a `UV_Beta_Calculator` as a prerequisite, re-creates the
    fitted UV power-law flux over a top-hat window centred on `ref_wav`,
    and takes the median AB magnitude across that window (converted to
    the observed frame). Optionally applies an extended-source
    correction.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used to compute this property.
    SED_fit_label : `str` or `Type[SED_code]`
        Label (or `SED_code` instance) identifying the SED-fitting run
        whose `Photometry_rest` this property is calculated from.
    rest_UV_wav_lims : `astropy.units.Quantity`, optional
        Two-element rest-frame wavelength range defining the UV
        continuum used to fit beta. Default is `[1_250.0, 3_000.0] * u.AA`.
    ref_wav : `astropy.units.Quantity`, optional
        Rest-frame reference wavelength at which mUV is computed.
        Default is `1_500.0 * u.AA`.
    top_hat_width : `astropy.units.Quantity`, optional
        Width of the top-hat wavelength window (centred on `ref_wav`)
        averaged over. Default is `100.0 * u.AA`.
    resolution : `astropy.units.Quantity`, optional
        Wavelength spacing of the top-hat window grid. Default is
        `1.0 * u.AA`.
    ext_src_corrs : `str` or `None`, optional
        Key identifying the extended-source correction to apply (either
        `"UV"` or a filter name), or `None` to skip the correction.
        Default is `"UV"`.
    ext_src_uplim : `int`, `float`, or `None`, optional
        Upper limit applied to the extended-source correction, or
        `None` for no limit. Default is `10.0`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        top_hat_width: u.Quantity = 100.0 * u.AA,
        resolution: u.Quantity = 1.0 * u.AA,
        ext_src_corrs: Optional[str] = "UV",
        ext_src_uplim: Optional[Union[int, float]] = 10.0,
    ) -> NoReturn:
        pre_req_properties = [
            UV_Beta_Calculator(aper_diam, SED_fit_label, rest_UV_wav_lims)
        ]
        global_kwargs = {
            "ref_wav": ref_wav,
            "top_hat_width": top_hat_width,
            "resolution": resolution,
            "ext_src_corrs": ext_src_corrs,
            "ext_src_uplim": ext_src_uplim,
        }
        super().__init__(
            aper_diam, SED_fit_label, pre_req_properties, **global_kwargs
        )

    @property
    def name(self: Self) -> str:
        """`str`: Short identifier,
        ``"m{ref_wav}_{rest_UV_wav_lims}{ext_src_label}"``.
        """
        ext_src_label = funcs.get_ext_src_corr_label(
            ext_src_key=self.global_kwargs["ext_src_corrs"],
            ext_src_uplim=self.global_kwargs["ext_src_uplim"],
        )
        return (
            f"m{self.global_kwargs['ref_wav'].to(u.AA).value:.0f}_"
            + rest_UV_wavs_name(
                self.pre_req_properties[0].global_kwargs["rest_UV_wav_lims"]
            )
            + ext_src_label
        )

    @property
    def plot_name(self: Self) -> str:
        """`str`: Human-readable plot label for the apparent UV magnitude."""
        return r"$m_{\mathrm{UV}}$"

    def _kwarg_assertions(self: Self) -> None:
        assert all(
            u.get_physical_type(self.global_kwargs[name]) == "length"
            for name in ["ref_wav", "top_hat_width", "resolution"]
        )
        assert (
            self.global_kwargs["ref_wav"]
            > self.pre_req_properties[0].global_kwargs["rest_UV_wav_lims"][0]
        )
        assert (
            self.global_kwargs["ref_wav"]
            < self.pre_req_properties[0].global_kwargs["rest_UV_wav_lims"][1]
        )
        assert self.global_kwargs["top_hat_width"] > 0.0 * u.AA
        assert self.global_kwargs["resolution"] > 0.0 * u.AA
        if self.global_kwargs["ext_src_corrs"] is not None:
            assert (
                self.global_kwargs["ext_src_corrs"] in ["UV"] + all_filt_names
            )
        if self.global_kwargs["ext_src_uplim"] is not None:
            assert isinstance(
                self.global_kwargs["ext_src_uplim"], (int, float)
            )
            assert self.global_kwargs["ext_src_uplim"] > 0.0

    def _calc_obj_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        # always pass
        return False

    @ignore_warnings
    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # This doesn't technically require the array of scattered
        # fluxes as input!
        if len(fluxes_arr) > 1:
            save_path = phot_rest.property_PDFs[
                self.pre_req_properties[0].name
            ].save_path.replace(".npy", "_scattered_fluxes.npy")
            # load scattered fluxes
            scattered_fluxes = np.load(save_path) * u.Jy
            assert len(fluxes_arr) == len(scattered_fluxes)
            fluxes_arr = scattered_fluxes
            beta_arr = phot_rest.property_PDFs[
                self.pre_req_properties[0].name
            ].input_arr
        else:  # fluxes are unscattered
            beta_arr = phot_rest.properties[self.pre_req_properties[0].name]

        band_wavs = self.pre_req_properties[0]._calc_obj_kwargs(phot_rest)[
            "rest_UV_band_wavs"
        ]
        # appropriately convert flux units to rest frame
        fluxes_arr = np.log10(
            funcs.convert_mag_units(
                band_wavs, fluxes_arr, u.erg / (u.s * u.AA * u.cm**2)
            ).value
        )
        # calculate fit amplitudes - ensure fluxes_arr is always 2D
        # (n_chains, n_bands)
        if fluxes_arr.ndim == 1:
            fluxes_arr = fluxes_arr[np.newaxis, :]
        amplitude_arr = (
            np.array(
                [
                    funcs.linear_fit(
                        np.log10(band_wavs.value, dtype=np.float64), fluxes
                    )[1]
                    for fluxes in fluxes_arr
                ]
            )
            * u.erg
            / (u.s * u.AA * u.cm**2)
        )
        assert len(amplitude_arr) == len(beta_arr)

        # re-create linear fit(s) to calculate mUV's
        rest_wavelengths = funcs.convert_wav_units(
            np.linspace(
                self.global_kwargs["ref_wav"]
                - self.global_kwargs["top_hat_width"] / 2,
                self.global_kwargs["ref_wav"]
                + self.global_kwargs["top_hat_width"] / 2,
                int(
                    np.round(
                        (
                            self.global_kwargs["top_hat_width"]
                            / self.global_kwargs["resolution"]
                        )
                        .to(u.dimensionless_unscaled)
                        .value,
                        0,
                    )
                ),
            ),
            u.AA,
        )
        mUV_arr = np.median(
            funcs.convert_mag_units(
                rest_wavelengths,
                10
                ** (
                    np.full(
                        (len(beta_arr), len(rest_wavelengths)),
                        np.log10(rest_wavelengths.value),
                    )
                    * beta_arr[:, np.newaxis].value
                    + amplitude_arr[:, np.newaxis].value
                )
                * u.erg
                / (u.s * u.AA * u.cm**2),
                u.ABmag,
            ),
            axis=1,
        )
        # TODO: speed up implementation of extended source corrections
        if self.global_kwargs["ext_src_corrs"] is not None:
            ext_src_corr = funcs.get_ext_src_corr(
                phot_rest,
                ext_src_key=self.global_kwargs["ext_src_corrs"],
                ext_src_uplim=self.global_kwargs["ext_src_uplim"],
                ref_wav=self.global_kwargs["ref_wav"],
            )
            # apply extended source corrections
            mUV_arr = (
                mUV_arr.value + funcs.flux_to_mag_ratio(ext_src_corr)
            ) * u.ABmag
        return mUV_arr

    def _get_output_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}

    def _call_cat(
        self: Self,
        cat: Catalogue,
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        n_jobs: int = 1,
    ) -> Optional[Catalogue]:
        if self.global_kwargs["ext_src_corrs"]:
            # load extended source corrections
            cat.load_sextractor_ext_src_corrs()
        return super()._call_cat(cat, n_chains, output, overwrite, n_jobs)

    def _call_phot_rest(
        self: Self,
        phot_rest: Photometry_rest,
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        save_path: Optional[str] = None,
        save_scattered_fluxes: bool = False,
        dtype: np.dtype = np.float32,
    ) -> Optional[Photometry_rest]:
        if self.global_kwargs["ext_src_corrs"]:
            # assert that extended source corrections have been loaded
            assert hasattr(
                phot_rest, "ext_src_corrs"
            ), galfind_logger.critical(
                "Extended source corrections must be pre-loaded!"
            )
        return super()._call_phot_rest(
            phot_rest,
            n_chains,
            output,
            overwrite,
            save_path,
            save_scattered_fluxes,
            dtype,
        )


class MUV_Calculator(Rest_Frame_Property_Calculator):
    """Calculates the absolute UV magnitude,
    MUV, from the apparent UV magnitude.

    Requires an `mUV_Calculator` as a prerequisite and converts the
    apparent magnitude to an absolute magnitude using the standard
    distance modulus, including a K-correction for the redshifted
    bandpass.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used to compute this property.
    SED_fit_label : `str` or `Type[SED_code]`
        Label (or `SED_code` instance) identifying the SED-fitting run
        whose `Photometry_rest` this property is calculated from.
    rest_UV_wav_lims : `astropy.units.Quantity`, optional
        Two-element rest-frame wavelength range defining the UV
        continuum used to fit beta. Default is `[1_250.0, 3_000.0] * u.AA`.
    ref_wav : `astropy.units.Quantity`, optional
        Rest-frame reference wavelength at which MUV is computed.
        Default is `1_500.0 * u.AA`.
    top_hat_width : `astropy.units.Quantity`, optional
        Width of the top-hat wavelength window (centred on `ref_wav`)
        used when computing mUV. Default is `100.0 * u.AA`.
    resolution : `astropy.units.Quantity`, optional
        Wavelength spacing of the top-hat window grid used when
        computing mUV. Default is `1.0 * u.AA`.
    ext_src_corrs : `str` or `None`, optional
        Key identifying the extended-source correction applied when
        computing mUV, or `None` to skip the correction. Default is
        `"UV"`.
    ext_src_uplim : `int`, `float`, or `None`, optional
        Upper limit applied to the extended-source correction, or
        `None` for no limit. Default is `10.0`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        top_hat_width: u.Quantity = 100.0 * u.AA,
        resolution: u.Quantity = 1.0 * u.AA,
        ext_src_corrs: Optional[str] = "UV",
        ext_src_uplim: Optional[Union[int, float]] = 10.0,
    ) -> NoReturn:
        mUV_calculator = mUV_Calculator(
            aper_diam,
            SED_fit_label,
            rest_UV_wav_lims,
            ref_wav,
            top_hat_width,
            resolution,
            ext_src_corrs,
            ext_src_uplim,
        )
        super().__init__(aper_diam, SED_fit_label, [mUV_calculator])

    @property
    def name(self: Self) -> str:
        """`str`: Short identifier,
        ``"M{ref_wav}_{rest_UV_wav_lims}[_extsrc_...]"``.
        """
        ext_src_label = (
            f"_extsrc_"
            f"{self.pre_req_properties[0].global_kwargs['ext_src_corrs']}"
            if self.pre_req_properties[0].global_kwargs["ext_src_corrs"]
            is not None
            else ""
        )
        ext_src_lim_label = (
            f"<{self.pre_req_properties[0].global_kwargs['ext_src_uplim']:.0f}"
            if self.pre_req_properties[0].global_kwargs["ext_src_uplim"]
            is not None
            and self.pre_req_properties[0].global_kwargs["ext_src_corrs"]
            is not None
            else ""
        )
        return (
            f"M{self.pre_req_properties[0].global_kwargs['ref_wav'].to(u.AA).value:.0f}_"
            + rest_UV_wavs_name(
                self.pre_req_properties[0]
                .pre_req_properties[0]
                .global_kwargs["rest_UV_wav_lims"]
            )
            + ext_src_label
            + ext_src_lim_label
        )

    @property
    def plot_name(self: Self) -> str:
        """`str`: Human-readable plot label for the absolute UV magnitude."""
        return r"$M_{\mathrm{UV}}$"

    def _kwarg_assertions(self: Self) -> NoReturn:
        pass

    def _calc_obj_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        # always pass
        return False

    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # load mUVs
        if len(fluxes_arr) > 1:
            mUV_arr = phot_rest.property_PDFs[
                self.pre_req_properties[0].name
            ].input_arr
        else:
            mUV_arr = phot_rest.properties[self.pre_req_properties[0].name]
        # calculate M_UV from m_UV
        d_L = (
            astropy_cosmo.luminosity_distance(phot_rest.z.value).to(u.pc).value
        )
        return (
            mUV_arr.value
            - 5.0 * np.log10(d_L / 10.0)
            + 2.5 * np.log10(1.0 + phot_rest.z.value)
        ) * u.ABmag

    def _get_output_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}


class LUV_Calculator(Rest_Frame_Property_Calculator):
    """Calculates the rest-frame UV luminosity,
    LUV, from the apparent UV magnitude.

    Requires an `mUV_Calculator` as a prerequisite and converts mUV to a
    luminosity using the observed-frame reference wavelength and
    redshift. If `beta_dust_conv` is not `None`, also runs a
    `UV_Dust_Attenuation_Calculator` as a further prerequisite and
    dust-corrects the resulting luminosity.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used to compute this property.
    SED_fit_label : `str` or `Type[SED_code]`
        Label (or `SED_code` instance) identifying the SED-fitting run
        whose `Photometry_rest` this property is calculated from.
    rest_UV_wav_lims : `astropy.units.Quantity`, optional
        Two-element rest-frame wavelength range defining the UV
        continuum used to fit beta. Default is `[1_250.0, 3_000.0] * u.AA`.
    ref_wav : `astropy.units.Quantity`, optional
        Rest-frame reference wavelength at which LUV is computed.
        Default is `1_500.0 * u.AA`.
    beta_dust_conv : `str`, `Type[AUV_from_beta]`, or `None`, optional
        Beta-to-A(UV) conversion used to dust-correct the luminosity, or
        `None` to skip dust correction. Default is `M99`.
    top_hat_width : `astropy.units.Quantity`, optional
        Width of the top-hat wavelength window used when computing mUV.
        Default is `100.0 * u.AA`.
    resolution : `astropy.units.Quantity`, optional
        Wavelength spacing of the top-hat window grid used when
        computing mUV. Default is `1.0 * u.AA`.
    ext_src_corrs : `str` or `None`, optional
        Key identifying the extended-source correction applied when
        computing mUV, or `None` to skip the correction. Default is
        `"UV"`.
    ext_src_uplim : `int`, `float`, or `None`, optional
        Upper limit applied to the extended-source correction, or
        `None` for no limit. Default is `10.0`.

    Attributes
    ----------
    dust_calculator : `UV_Dust_Attenuation_Calculator` or `None`
        Prerequisite dust attenuation calculator, or `None` if
        `beta_dust_conv` was `None`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        # frame: str = "obs",
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        beta_dust_conv: Optional[Union[str, Type[AUV_from_beta]]] = M99,
        top_hat_width: u.Quantity = 100.0 * u.AA,
        resolution: u.Quantity = 1.0 * u.AA,
        ext_src_corrs: Optional[str] = "UV",
        ext_src_uplim: Optional[Union[int, float]] = 10.0,
    ) -> NoReturn:
        mUV_calculator = mUV_Calculator(
            aper_diam,
            SED_fit_label,
            rest_UV_wav_lims,
            ref_wav,
            top_hat_width,
            resolution,
            ext_src_corrs,
            ext_src_uplim,
        )
        pre_req_properties = [mUV_calculator]
        if beta_dust_conv is None:
            self.dust_calculator = None
        else:
            if isinstance(beta_dust_conv, str):
                beta_dust_conv = [
                    beta_dust_conv_cls()
                    for beta_dust_conv_cls in AUV_from_beta.__subclasses__()
                    if beta_dust_conv_cls.__name__ == beta_dust_conv
                ][0]
            elif not isinstance(beta_dust_conv, AUV_from_beta):
                beta_dust_conv = beta_dust_conv()
            self.dust_calculator = UV_Dust_Attenuation_Calculator(
                aper_diam,
                SED_fit_label,
                rest_UV_wav_lims,
                beta_dust_conv,
                ref_wav,
                keep_valid=True,
            )
            pre_req_properties.append(self.dust_calculator)
        global_kwargs = {}  # "frame": frame}
        super().__init__(
            aper_diam, SED_fit_label, pre_req_properties, **global_kwargs
        )

    @property
    def name(self: Self) -> str:
        """`str`: Short identifier,
        ``"L{ref_wav}[_{dust}dust]_{rest_UV_wav_lims}[_extsrc_...]"``.
        """
        if self.dust_calculator is not None:
            dust_label = (
                "_"
                + "_".join(self.dust_calculator.name.split("_")[1:2])
                + "dust"
            )
        else:
            dust_label = ""
        rest_wavs_label = rest_UV_wavs_name(
            self.pre_req_properties[0]
            .pre_req_properties[0]
            .global_kwargs["rest_UV_wav_lims"]
        )
        ext_src_label = (
            f"_extsrc_{self.pre_req_properties[0].global_kwargs['ext_src_corrs']}"
            if self.pre_req_properties[0].global_kwargs["ext_src_corrs"]
            is not None
            else ""
        )
        ext_src_lim_label = (
            f"<{self.pre_req_properties[0].global_kwargs['ext_src_uplim']:.0f}"
            if self.pre_req_properties[0].global_kwargs["ext_src_uplim"]
            is not None
            and self.pre_req_properties[0].global_kwargs["ext_src_corrs"]
            is not None
            else ""
        )
        # {self.global_kwargs['frame']}
        return (
            f"L{self.pre_req_properties[0].global_kwargs['ref_wav'].to(u.AA).value:.0f}"
            f"{dust_label}_{rest_wavs_label}{ext_src_label}{ext_src_lim_label}"
        )

    @property
    def plot_name(self: Self) -> str:
        """`str`: Human-readable plot label for the UV luminosity."""
        return r"$L_{\mathrm{UV}}$"  # frame and units here too

    def _kwarg_assertions(self: Self) -> NoReturn:
        pass
        # assert self.global_kwargs["frame"] in ["rest", "obs"]

    def _calc_obj_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        # always pass
        return False

    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # load mUVs
        if len(fluxes_arr) > 1:
            mUV_arr = phot_rest.property_PDFs[
                self.pre_req_properties[0].name
            ].input_arr
        else:
            mUV_arr = phot_rest.properties[self.pre_req_properties[0].name]
        # convert mUVs to LUVs
        # if self.global_kwargs["frame"] == "rest":
        #     z = 0.0
        #     wavs = np.full(len(fluxes_arr), \
        #         self.pre_req_properties[0].global_kwargs["ref_wav"])
        # else: # frame == "obs"
        # use observed frame wavelengths
        wavs = np.full(
            len(fluxes_arr),
            self.pre_req_properties[0].global_kwargs["ref_wav"]
            * (1.0 + phot_rest.z.value),
        )
        LUV_arr = funcs.flux_to_luminosity(mUV_arr, wavs, phot_rest.z.value)

        # extract dust chains/value if required
        if self.dust_calculator is not None:
            if len(fluxes_arr) > 1:
                AUV_arr = phot_rest.property_PDFs[
                    self.dust_calculator.name
                ].input_arr
                assert len(fluxes_arr) == len(AUV_arr)
            else:
                AUV_arr = phot_rest.properties[self.dust_calculator.name]
            LUV_arr = funcs.dust_correct(LUV_arr, AUV_arr)
        # output luminosities
        return LUV_arr

    def _get_output_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}


class SFR_UV_Calculator(Rest_Frame_Property_Calculator):
    """Calculates the UV-derived star formation rate,
    SFR_UV, from the UV luminosity.

    Requires an `LUV_Calculator` as a prerequisite and converts LUV to a
    star formation rate using the `SFR_conv` conversion registered in
    `useful_funcs_austind.SFR_conversions`.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used to compute this property.
    SED_fit_label : `str` or `Type[SED_code]`
        Label (or `SED_code` instance) identifying the SED-fitting run
        whose `Photometry_rest` this property is calculated from.
    rest_UV_wav_lims : `astropy.units.Quantity`, optional
        Two-element rest-frame wavelength range defining the UV
        continuum used to fit beta. Default is `[1_250.0, 3_000.0] * u.AA`.
    ref_wav : `astropy.units.Quantity`, optional
        Rest-frame reference wavelength at which LUV is computed.
        Default is `1_500.0 * u.AA`.
    beta_dust_conv : `str`, `Type[AUV_from_beta]`, or `None`, optional
        Beta-to-A(UV) conversion used to dust-correct LUV, or `None` to
        skip dust correction. Default is `M99`.
    SFR_conv : `str`, optional
        Name of the LUV-to-SFR conversion to use, a key into
        `useful_funcs_austind.SFR_conversions`. Default is `"MD14"`.
    top_hat_width : `astropy.units.Quantity`, optional
        Width of the top-hat wavelength window used when computing mUV.
        Default is `100.0 * u.AA`.
    resolution : `astropy.units.Quantity`, optional
        Wavelength spacing of the top-hat window grid used when
        computing mUV. Default is `1.0 * u.AA`.
    ext_src_corrs : `str` or `None`, optional
        Key identifying the extended-source correction applied when
        computing mUV, or `None` to skip the correction. Default is
        `"UV"`.
    ext_src_uplim : `int`, `float`, or `None`, optional
        Upper limit applied to the extended-source correction, or
        `None` for no limit. Default is `10.0`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        beta_dust_conv: Optional[Union[str, Type[AUV_from_beta]]] = M99,
        SFR_conv: str = "MD14",
        top_hat_width: u.Quantity = 100.0 * u.AA,
        resolution: u.Quantity = 1.0 * u.AA,
        ext_src_corrs: Optional[str] = "UV",
        ext_src_uplim: Optional[Union[int, float]] = 10.0,
    ) -> NoReturn:
        LUV_calculator = LUV_Calculator(
            aper_diam,
            SED_fit_label,
            # "obs",
            rest_UV_wav_lims,
            ref_wav,
            beta_dust_conv,
            top_hat_width,
            resolution,
            ext_src_corrs,
            ext_src_uplim,
        )
        pre_req_properties = [LUV_calculator]
        global_kwargs = {"SFR_conv": SFR_conv}
        super().__init__(
            aper_diam, SED_fit_label, pre_req_properties, **global_kwargs
        )

    @property
    def name(self: Self) -> str:
        """`str`: Short identifier,
        ``"SFR{ref_wav}[_{dust}dust]_{rest_UV_wav_lims}_{SFR_conv}[_extsrc_...]"``.
        """
        if self.pre_req_properties[0].dust_calculator is not None:
            dust_label = (
                "_"
                + "_".join(
                    self.pre_req_properties[0].dust_calculator.name.split("_")[
                        1:2
                    ]
                )
                + "dust"
            )
        else:
            dust_label = ""
        _ref_wav = (
            self.pre_req_properties[0]
            .pre_req_properties[0]
            .global_kwargs["ref_wav"]
        )
        ref_wav_label = f"{_ref_wav.to(u.AA).value:.0f}"
        rest_wavs_label = rest_UV_wavs_name(
            self.pre_req_properties[0]
            .pre_req_properties[0]
            .pre_req_properties[0]
            .global_kwargs["rest_UV_wav_lims"]
        )
        _ext_src_corrs = (
            self.pre_req_properties[0]
            .pre_req_properties[0]
            .global_kwargs["ext_src_corrs"]
        )
        ext_src_label = (
            f"_extsrc_{_ext_src_corrs}"
            if self.pre_req_properties[0]
            .pre_req_properties[0]
            .global_kwargs["ext_src_corrs"]
            is not None
            else ""
        )
        _ext_src_uplim = (
            self.pre_req_properties[0]
            .pre_req_properties[0]
            .global_kwargs["ext_src_uplim"]
        )
        ext_src_lim_label = (
            f"<{_ext_src_uplim:.0f}"
            if self.pre_req_properties[0]
            .pre_req_properties[0]
            .global_kwargs["ext_src_uplim"]
            is not None
            and self.pre_req_properties[0]
            .pre_req_properties[0]
            .global_kwargs["ext_src_corrs"]
            is not None
            else ""
        )
        return (
            f"SFR{ref_wav_label}{dust_label}_"
            + f"{rest_wavs_label}_{self.global_kwargs['SFR_conv']}"
            + ext_src_label
            + ext_src_lim_label
        )

    @property
    def plot_name(self: Self) -> str:
        """`str`: Human-readable plot label for the UV-derived star
        formation rate."""
        return r"$\mathrm{SFR}_{\mathrm{UV}}$"

    def _kwarg_assertions(self: Self) -> None:
        assert self.global_kwargs["SFR_conv"] in funcs.SFR_conversions.keys()

    def _calc_obj_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        # always pass
        return False

    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # load LUVs
        if len(fluxes_arr) > 1:
            LUV_arr = phot_rest.property_PDFs[
                self.pre_req_properties[0].name
            ].input_arr
        else:
            LUV_arr = phot_rest.properties[self.pre_req_properties[0].name]
        # convert LUVs to SFRs
        SFR_arr = (
            funcs.SFR_conversions[self.global_kwargs["SFR_conv"]] * LUV_arr
        )
        return SFR_arr

    def _get_output_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}


class Optical_Continuum_Calculator(Rest_Frame_Property_Calculator):
    """Calculates the rest-frame optical continuum flux density
    near a strong emission line.

    Identifies the photometric band nearest to (and containing) the
    first of `strong_line_names`, then finds the other bands within
    `rest_optical_wavs` that are free of strong optical emission lines
    to use as continuum bands. If there are two or more continuum
    bands, the continuum flux at the emission line band is estimated by
    linear interpolation; if there is only one, its flux is used
    directly.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used to compute this property.
    SED_fit_label : `str` or `Type[SED_code]`
        Label (or `SED_code` instance) identifying the SED-fitting run
        whose `Photometry_rest` this property is calculated from.
    strong_line_names : `str` or `list`
        Name(s) of the strong optical emission line(s) (keys of
        `Emission_lines.line_diagnostics`) whose continuum is being
        calculated. A `"+"`-delimited string is split into a list.
    rest_optical_wavs : `astropy.units.Quantity`, optional
        Two-element rest-frame wavelength range within which continuum
        bands are searched for. Default is `[4_200.0, 10_000.0] * u.AA`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        strong_line_names: Union[str, list],
        rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
    ) -> None:
        if isinstance(strong_line_names, str):
            strong_line_names = strong_line_names.split("+")
        global_kwargs = {
            "strong_line_names": strong_line_names,
            "rest_optical_wavs": rest_optical_wavs,
        }
        super().__init__(aper_diam, SED_fit_label, [], **global_kwargs)

    @property
    def name(self: Self) -> str:
        """`str`: Short identifier, ``"cont_{strong_line_names}"``."""
        return f"cont_{'+'.join(self.global_kwargs['strong_line_names'])}"

    @property
    def plot_name(self: Self) -> str:
        """`str`: Human-readable plot label for the continuum flux density."""
        return (
            f"{'+'.join(self.global_kwargs['strong_line_names'])}"
            " continuum / nJy"
        )

    def _kwarg_assertions(self: Self) -> None:
        assert all(
            line_name in strong_optical_lines
            for line_name in self.global_kwargs["strong_line_names"]
        )

    def _calc_obj_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        # determine the nearest band to the first line
        wavelength = line_diagnostics[
            self.global_kwargs["strong_line_names"][0]
        ]["line_wav"] * (1.0 + phot_rest.z.value)
        if len(phot_rest.filterset) == 0:
            return {
                "emission_band": None,
                "cont_bands": None,
                "keep_indices": None,
            }
        nearest_band = phot_rest.filterset[
            int(
                np.abs(
                    [
                        funcs.convert_wav_units(filt.WavelengthCen, u.AA).value
                        for filt in phot_rest.filterset
                    ]
                    - funcs.convert_wav_units(wavelength, u.AA).value
                ).argmin()
            )
        ]
        # ensure emission line actually falls within this band
        emission_bands = [
            filt.filt_name
            for filt in phot_rest.filterset
            if wavelength > filt.WavelengthLower50
            and wavelength < filt.WavelengthUpper50
        ]
        if nearest_band.filt_name not in emission_bands:
            emission_band = None
            cont_bands = None
            cont_band_indices = None
        else:
            emission_band = nearest_band
            cont_bands = []
            cont_band_indices = []
            for i, filt in enumerate(phot_rest.filterset):
                # get continuum bands which are entirely within the
                # rest frame optical and do not contain any strong optical
                # lines
                if (
                    filt.WavelengthUpper50
                    < self.global_kwargs["rest_optical_wavs"][1]
                    * (1.0 + phot_rest.z.value)
                    and filt.WavelengthLower50
                    > self.global_kwargs["rest_optical_wavs"][0]
                    * (1.0 + phot_rest.z.value)
                    and not any(
                        line_diagnostics[line_name]["line_wav"]
                        * (1.0 + phot_rest.z.value)
                        < filt.WavelengthUpper50
                        and line_diagnostics[line_name]["line_wav"]
                        * (1.0 + phot_rest.z.value)
                        > filt.WavelengthLower50
                        for line_name in strong_optical_lines
                    )
                ):
                    cont_bands.extend([filt])
                    cont_band_indices.extend([i])
            if len(cont_bands) == 0 or any(
                np.isnan(phot_rest.depths[i]) for i in cont_band_indices
            ):
                cont_bands = None
                cont_band_indices = None
        return {
            "emission_band": emission_band,
            "cont_bands": cont_bands,
            "keep_indices": cont_band_indices,
        }

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        if any(value is None for value in self.obj_kwargs.values()):
            return True
        else:
            # ensure all lines lie within this band (defined by 50%
            # throughput boundaries)
            # and that there are no other strong optical lines in this band
            # and that there is more than 1 relevant continuum band
            if (
                not all(
                    (line_diagnostics[line_name]["line_wav"])
                    * (1.0 + phot_rest.z.value)
                    < self.obj_kwargs["emission_band"].WavelengthUpper50
                    and (line_diagnostics[line_name]["line_wav"])
                    * (1.0 + phot_rest.z.value)
                    > self.obj_kwargs["emission_band"].WavelengthLower50
                    for line_name in self.global_kwargs["strong_line_names"]
                )
                or any(
                    line_diagnostics[line_name]["line_wav"]
                    * (1.0 + phot_rest.z.value)
                    < self.obj_kwargs["emission_band"].WavelengthUpper50
                    and line_diagnostics[line_name]["line_wav"]
                    * (1.0 + phot_rest.z.value)
                    > self.obj_kwargs["emission_band"].WavelengthLower50
                    for line_name in strong_optical_lines
                    if line_name not in self.global_kwargs["strong_line_names"]
                )
                or len(self.obj_kwargs["cont_bands"]) == 0
            ):
                return True
            else:
                return False

    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # funcs.convert_mag_units(wavs, phot_rest.flux[
        #     self.obj_kwargs["cont_band_indices"]], u.nJy)
        # TODO: Generalize in the instance of flux units not convertible to nJy
        flux_unit = fluxes_arr.unit
        cont_fluxes = np.array(fluxes_arr.value, dtype=np.float64)

        if len(self.obj_kwargs["cont_bands"]) == 1:
            cont_chains = cont_fluxes[:, 0]
        elif len(self.obj_kwargs["cont_bands"]) >= 2:
            # calculate continuum from interpolation to
            # middle of the emission band if two continuum bands
            cont_wavs = [
                (band.WavelengthCen.to(u.AA) / (1.0 + phot_rest.z.value)).value
                for band in self.obj_kwargs["cont_bands"]
            ]  # in Angstrom
            em_wav = (
                self.obj_kwargs["emission_band"].WavelengthCen.to(u.AA)
                / (1.0 + phot_rest.z.value)
            ).value  # in Angstrom
            cont_chains = np.array(
                [
                    funcs.interpolate_linear_fit(
                        np.array(cont_wavs, dtype=np.float64),
                        cont_fluxes_,
                        em_wav,
                    )
                    for cont_fluxes_ in cont_fluxes
                ]
            )
        # set negative fluxes to NaNs
        valid_chains = cont_chains[cont_chains > 0.0]
        if len(fluxes_arr) > 1:
            self.obj_kwargs["negative_flux_pc"] = 100.0 * (
                1 - len(valid_chains) / len(cont_chains)
            )
            if (
                len(valid_chains) < 50
                or self.obj_kwargs["negative_flux_pc"] > 99.0
            ):
                return None
        else:
            if len(valid_chains) < 1:
                return None
        cont_chains[cont_chains < 0.0] = np.nan
        return (cont_chains * flux_unit).to(u.nJy)

    def _get_output_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {
            "bands": "+".join(
                [band.filt_name for band in self.obj_kwargs["cont_bands"]]
            ),
            "negative_flux_pc": self.obj_kwargs["negative_flux_pc"],
        }


class Optical_Line_EW_Calculator(Rest_Frame_Property_Calculator):
    """Calculates the equivalent width of a strong rest-frame optical
    emission line.

    Requires an `Optical_Continuum_Calculator` as a prerequisite, then
    computes the equivalent width as
    ``(band_flux / continuum_flux - 1) * bandwidth``, where `bandwidth`
    is the 50%-throughput width of the emission line's photometric band
    (optionally de-redshifted if `frame` is `"rest"`).

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used to compute this property.
    SED_fit_label : `str` or `Type[SED_code]`
        Label (or `SED_code` instance) identifying the SED-fitting run
        whose `Photometry_rest` this property is calculated from.
    strong_line_names : `str` or `list`
        Name(s) of the strong optical emission line(s) (keys of
        `Emission_lines.line_diagnostics`) whose equivalent width is
        being calculated. A `"+"`-delimited string is split into a list.
    frame : `str`, optional
        Either `"rest"` or `"obs"`, determining whether the emission
        band's bandwidth is de-redshifted before use. Default is
        `"rest"`.
    rest_optical_wavs : `astropy.units.Quantity`, optional
        Two-element rest-frame wavelength range within which continuum
        bands are searched for. Default is `[4_200.0, 10_000.0] * u.AA`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        strong_line_names: Union[str, list],
        frame: str = "rest",
        rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
    ) -> None:
        if isinstance(strong_line_names, str):
            strong_line_names = strong_line_names.split("+")
        global_kwargs = {
            "strong_line_names": strong_line_names,
            "frame": frame,
            "rest_optical_wavs": rest_optical_wavs,
        }
        pre_req_properties = [
            Optical_Continuum_Calculator(
                aper_diam, SED_fit_label, strong_line_names, rest_optical_wavs
            )
        ]
        super().__init__(
            aper_diam, SED_fit_label, pre_req_properties, **global_kwargs
        )

    @property
    def name(self: Self) -> str:
        """`str`: Short identifier, ``"EW{frame}_{strong_line_names}"``."""
        return (
            f"EW{self.global_kwargs['frame']}_"
            + f"{'+'.join(self.global_kwargs['strong_line_names'])}"
        )

    @property
    def plot_name(self: Self) -> str:
        """`str`: Human-readable plot label for the emission line
        equivalent width."""
        line_label = "+".join(self.global_kwargs["strong_line_names"])
        return rf"$\mathrm{{EW}}_{{\mathrm{{{line_label}}}}}$"

    def _kwarg_assertions(self: Self) -> None:
        assert all(
            line_name in strong_optical_lines
            for line_name in self.global_kwargs["strong_line_names"]
        )
        assert self.global_kwargs["frame"] in ["rest", "obs"]

    def _calc_obj_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        # determine the nearest band to the first line
        wavelength = line_diagnostics[
            self.global_kwargs["strong_line_names"][0]
        ]["line_wav"] * (1.0 + phot_rest.z.value)
        nearest_band = phot_rest.filterset[
            int(
                np.abs(
                    [
                        funcs.convert_wav_units(filt.WavelengthCen, u.AA).value
                        for filt in phot_rest.filterset
                    ]
                    - funcs.convert_wav_units(wavelength, u.AA).value
                ).argmin()
            )
        ]
        # ensure emission line actually falls within this band
        emission_bands = [
            filt.filt_name
            for filt in phot_rest.filterset
            if wavelength > filt.WavelengthLower50
            and wavelength < filt.WavelengthUpper50
        ]

        if nearest_band.filt_name not in emission_bands:
            failure = True
        else:
            emission_band = nearest_band
            emission_band_index = int(
                np.where(
                    np.array(phot_rest.filterset.filt_names)
                    == emission_band.filt_name
                )[0][0]
            )
            if np.isnan(phot_rest.depths[emission_band_index]):
                failure = True
            else:
                failure = False
                emission_band_wavelength = emission_band.WavelengthCen
                bandwidth = (
                    emission_band.WavelengthUpper50
                    - emission_band.WavelengthLower50
                )
                if self.global_kwargs["frame"] == "rest":
                    bandwidth /= 1.0 + phot_rest.z.value
                bandwidth = bandwidth.to(u.AA)
        if failure:
            emission_band = None
            emission_band_index = None
            emission_band_wavelength = None
            bandwidth = None
        return {
            "emission_band": emission_band,
            "keep_indices": emission_band_index,
            "emission_band_wav": emission_band_wavelength,
            "bandwidth": bandwidth,
        }

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        if any(value is None for value in self.obj_kwargs.values()):
            return True
        else:
            # ensure all lines lie within this band (defined by 50%
            # throughput boundaries)
            # and that there are no other strong optical lines in this band
            if not all(
                (line_diagnostics[line_name]["line_wav"])
                * (1.0 + phot_rest.z.value)
                < self.obj_kwargs["emission_band"].WavelengthUpper50
                and (line_diagnostics[line_name]["line_wav"])
                * (1.0 + phot_rest.z.value)
                > self.obj_kwargs["emission_band"].WavelengthLower50
                for line_name in self.global_kwargs["strong_line_names"]
            ) or any(
                line_diagnostics[line_name]["line_wav"]
                * (1.0 + phot_rest.z.value)
                < self.obj_kwargs["emission_band"].WavelengthUpper50
                and line_diagnostics[line_name]["line_wav"]
                * (1.0 + phot_rest.z.value)
                > self.obj_kwargs["emission_band"].WavelengthLower50
                for line_name in strong_optical_lines
                if line_name not in self.global_kwargs["strong_line_names"]
            ):
                return True
            else:
                return False

    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # load previously computed continuum chains
        if len(fluxes_arr) > 1:
            cont_fluxes = phot_rest.property_PDFs[
                self.pre_req_properties[0].name
            ].input_arr
            assert len(fluxes_arr) == len(cont_fluxes)
        else:
            cont_fluxes = phot_rest.properties[self.pre_req_properties[0].name]
        return (
            (fluxes_arr.flatten() / cont_fluxes).to(u.dimensionless_unscaled)
            - 1.0
        ) * self.obj_kwargs["bandwidth"]

    def _get_output_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        # calculate potential contaminant lines
        contam_lines = [
            name
            for name, line in line_diagnostics.items()
            if name not in strong_optical_lines
            and line["line_wav"] * (1.0 * phot_rest.z.value)
            < self.obj_kwargs["emission_band"].WavelengthUpper50
            and line["line_wav"] * (1.0 * phot_rest.z.value)
            > self.obj_kwargs["emission_band"].WavelengthLower50
        ]
        return {
            "band": self.obj_kwargs["emission_band"].filt_name,
            "contam_lines": "+".join(contam_lines),
        }


class Dust_Attenuation_From_UV_Calculator(Rest_Frame_Property_Calculator):
    """Calculates the dust attenuation at an arbitrary wavelength,
    scaled from the UV attenuation.

    Requires a `UV_Dust_Attenuation_Calculator` as a prerequisite
    (giving A(UV_ref_wav)), then rescales it to `calc_wav` using the
    ratio of the `dust_law` attenuation curve evaluated at `calc_wav`
    and at `UV_ref_wav`.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used to compute this property.
    SED_fit_label : `str` or `Type[SED_code]`
        Label (or `SED_code` instance) identifying the SED-fitting run
        whose `Photometry_rest` this property is calculated from.
    calc_wav : `astropy.units.Quantity`
        Rest-frame wavelength at which the dust attenuation is computed.
    dust_law : `str` or `Type[Dust_Law]`, optional
        Dust attenuation law to use, either the name of a `Dust_Law`
        subclass or an instance/class thereof. Default is `Calzetti00`.
    beta_dust_conv : `str` or `Type[AUV_from_beta]`, optional
        Beta-to-A(UV) conversion used to obtain the UV attenuation.
        Default is `M99`.
    UV_ref_wav : `astropy.units.Quantity`, optional
        Rest-frame reference wavelength at which the UV attenuation is
        computed. Default is `1_500.0 * u.AA`.
    UV_wav_lims : `astropy.units.Quantity`, optional
        Two-element rest-frame wavelength range defining the UV
        continuum used to fit beta. Default is `[1_250.0, 3_000.0] * u.AA`.
    keep_valid : `bool`, optional
        Whether to clip negative attenuations to zero. Default is `False`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        calc_wav: u.Quantity,
        dust_law: Union[str, Type[Dust_Law]] = Calzetti00,
        beta_dust_conv: Union[str, Type[AUV_from_beta]] = M99,
        UV_ref_wav: u.Quantity = 1_500.0 * u.AA,
        UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        keep_valid: bool = False,
    ) -> NoReturn:
        dust_atten_calculator = UV_Dust_Attenuation_Calculator(
            aper_diam,
            SED_fit_label,
            UV_wav_lims,
            beta_dust_conv,
            UV_ref_wav,
        )
        pre_req_properties = [dust_atten_calculator]
        if isinstance(dust_law, str):
            dust_law = [
                dust_law_cls()
                for dust_law_cls in Dust_Law.__subclasses__()
                if dust_law_cls.__name__ == dust_law
            ][0]
        elif not isinstance(dust_law, Dust_Law):
            dust_law = dust_law()
        global_kwargs = {
            "calc_wav": calc_wav,
            "dust_law": dust_law,
            "keep_valid": keep_valid,
        }
        super().__init__(
            aper_diam, SED_fit_label, pre_req_properties, **global_kwargs
        )

    @property
    def name(self: Self) -> str:
        """`str`: Short identifier,
        ``"A{calc_wav}_{beta_dust_conv}_{dust_law}[_A>0]"``."""
        beta_dust_conv_name = (
            self.pre_req_properties[0]
            .global_kwargs["beta_dust_conv"]
            .__class__.__name__
        )
        label = (
            f"A{self.global_kwargs['calc_wav'].to(u.AA).value:.0f}"
            + f"_{beta_dust_conv_name}"
            + f"_{self.global_kwargs['dust_law'].__class__.__name__}"
        )
        if self.global_kwargs["keep_valid"]:
            label += "_A>0"
        return label

    @property
    def plot_name(self: Self) -> str:
        """`str`: Human-readable plot label for the dust attenuation
        at `calc_wav`."""
        return f"Dust attenuation (E(B-V)) @ {self.global_kwargs['calc_wav']}"

    def _kwarg_assertions(self: Self) -> None:
        assert u.get_physical_type(self.global_kwargs["calc_wav"]) == "length"
        assert (
            self.global_kwargs["dust_law"].__class__
            in Dust_Law.__subclasses__()
        )
        assert isinstance(self.global_kwargs["keep_valid"], bool)

    def _calc_obj_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        if any(value is None for value in self.obj_kwargs.values()):
            return True
        else:
            return False

    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # calculate AUV
        if len(fluxes_arr) > 1:
            AUV_arr = phot_rest.property_PDFs[
                self.pre_req_properties[0].name
            ].input_arr
            assert len(fluxes_arr) == len(AUV_arr)
        else:
            AUV_arr = phot_rest.properties[self.pre_req_properties[0].name]
        AUV_arr = AUV_arr.to(u.ABmag).value
        # calculate A_lambda
        A_lambda = (
            AUV_arr
            * self.global_kwargs["dust_law"].k_lambda(
                self.global_kwargs["calc_wav"].to(u.AA)
            )
            / self.global_kwargs["dust_law"].k_lambda(
                self.pre_req_properties[0].global_kwargs["ref_wav"].to(u.AA)
            )
        ) * u.ABmag
        if self.global_kwargs["keep_valid"]:
            A_lambda[A_lambda < 0.0] = 0.0
        return A_lambda

    def _get_output_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}


class Line_Dust_Attenuation_From_UV_Calculator(
    Dust_Attenuation_From_UV_Calculator
):
    """Calculates the dust attenuation at the wavelength of a given
    emission line, scaled from the UV attenuation.

    Convenience subclass of `Dust_Attenuation_From_UV_Calculator` that
    looks up `calc_wav` from `line_name`'s rest-frame wavelength in
    `Emission_lines.line_diagnostics`.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used to compute this property.
    SED_fit_label : `str` or `Type[SED_code]`
        Label (or `SED_code` instance) identifying the SED-fitting run
        whose `Photometry_rest` this property is calculated from.
    line_name : `str`
        Name of the emission line (a key of
        `Emission_lines.line_diagnostics`) at whose rest-frame
        wavelength the dust attenuation is computed.
    dust_law : `str` or `Type[Dust_Law]`, optional
        Dust attenuation law to use, either the name of a `Dust_Law`
        subclass or an instance/class thereof. Default is `Calzetti00`.
    beta_dust_conv : `str` or `Type[AUV_from_beta]`, optional
        Beta-to-A(UV) conversion used to obtain the UV attenuation.
        Default is `M99`.
    UV_ref_wav : `astropy.units.Quantity`, optional
        Rest-frame reference wavelength at which the UV attenuation is
        computed. Default is `1_500.0 * u.AA`.
    UV_wav_lims : `astropy.units.Quantity`, optional
        Two-element rest-frame wavelength range defining the UV
        continuum used to fit beta. Default is `[1_250.0, 3_000.0] * u.AA`.
    keep_valid : `bool`, optional
        Whether to clip negative attenuations to zero. Default is `False`.

    Raises
    ------
    AssertionError
        If `line_name` is not a key of `Emission_lines.line_diagnostics`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        line_name: str,
        dust_law: Union[str, Type[Dust_Law]] = Calzetti00,
        beta_dust_conv: Union[str, Type[AUV_from_beta]] = M99,
        UV_ref_wav: u.Quantity = 1_500.0 * u.AA,
        UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        keep_valid: bool = False,
    ) -> NoReturn:
        assert line_name in line_diagnostics.keys(), galfind_logger.critical(
            f"{line_name=} not in {line_diagnostics.keys()}"
        )
        super().__init__(
            aper_diam,
            SED_fit_label,
            line_diagnostics[line_name]["line_wav"].to(u.AA),
            dust_law,
            beta_dust_conv,
            UV_ref_wav,
            UV_wav_lims,
            keep_valid,
        )


class Optical_Line_Flux_Calculator(Rest_Frame_Property_Calculator):
    """Calculates the flux of a strong rest-frame optical emission line.

    Requires an `Optical_Continuum_Calculator` and an
    `Optical_Line_EW_Calculator` as prerequisites, and computes the line
    flux as ``EW * continuum_flux_density``. If dust-law/conversion
    arguments are all given, also runs a
    `Line_Dust_Attenuation_From_UV_Calculator` as a further prerequisite
    and dust-corrects the resulting flux.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used to compute this property.
    SED_fit_label : `str` or `Type[SED_code]`
        Label (or `SED_code` instance) identifying the SED-fitting run
        whose `Photometry_rest` this property is calculated from.
    strong_line_names : `str` or `list`
        Name(s) of the strong optical emission line(s) (keys of
        `Emission_lines.line_diagnostics`) whose flux is being
        calculated. A `"+"`-delimited string is split into a list.
    frame : `str`, optional
        Either `"rest"` or `"obs"`, passed through to the prerequisite
        `Optical_Line_EW_Calculator`. Default is `"rest"`.
    rest_optical_wavs : `astropy.units.Quantity`, optional
        Two-element rest-frame wavelength range within which continuum
        bands are searched for. Default is `[4_200.0, 10_000.0] * u.AA`.
    dust_law : `str`, `Type[Dust_Law]`, or `None`, optional
        Dust attenuation law used to dust-correct the line flux, or
        `None` (together with the other dust arguments) to skip dust
        correction. Default is `Calzetti00`.
    beta_dust_conv : `str`, `Type[AUV_from_beta]`, or `None`, optional
        Beta-to-A(UV) conversion used to obtain the UV attenuation for
        dust correction. Default is `M99`.
    UV_ref_wav : `astropy.units.Quantity` or `None`, optional
        Rest-frame reference wavelength at which the UV attenuation is
        computed. Default is `1_500.0 * u.AA`.
    UV_wav_lims : `astropy.units.Quantity` or `None`, optional
        Two-element rest-frame wavelength range defining the UV
        continuum used to fit beta. Default is `[1_250.0, 3_000.0] * u.AA`.

    Attributes
    ----------
    dust_calculator : `Line_Dust_Attenuation_From_UV_Calculator` or `None`
        Prerequisite dust attenuation calculator, or `None` if any of
        `dust_law`, `beta_dust_conv`, `UV_ref_wav`, or `UV_wav_lims`
        was `None`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        strong_line_names: Union[str, list],
        frame: str = "rest",
        rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
        dust_law: Optional[Union[str, Type[Dust_Law]]] = Calzetti00,
        beta_dust_conv: Optional[Union[str, Type[AUV_from_beta]]] = M99,
        UV_ref_wav: Optional[u.Quantity] = 1_500.0 * u.AA,
        UV_wav_lims: Optional[u.Quantity] = [1_250.0, 3_000.0] * u.AA,
    ) -> NoReturn:
        cont_calculator = Optical_Continuum_Calculator(
            aper_diam, SED_fit_label, strong_line_names, rest_optical_wavs
        )
        EW_calculator = Optical_Line_EW_Calculator(
            aper_diam,
            SED_fit_label,
            strong_line_names,
            frame,
            rest_optical_wavs,
        )
        pre_req_properties = [cont_calculator, EW_calculator]
        if any(
            dust_arg is None
            for dust_arg in [dust_law, beta_dust_conv, UV_ref_wav, UV_wav_lims]
        ):
            self.dust_calculator = None
        else:
            if isinstance(strong_line_names, str):
                strong_line_names = strong_line_names.split("+")
            self.dust_calculator = Line_Dust_Attenuation_From_UV_Calculator(
                aper_diam,
                SED_fit_label,
                strong_line_names[0],
                dust_law,
                beta_dust_conv,
                UV_ref_wav,
                UV_wav_lims,
                keep_valid=True,
            )
            pre_req_properties.append(self.dust_calculator)
        super().__init__(aper_diam, SED_fit_label, pre_req_properties)

    @property
    def name(self: Self) -> str:
        """`str`: Short identifier,
        ``"flux_{frame}_{strong_line_names}[_{dust}_...]"``."""
        if self.dust_calculator is not None:
            dust_label = (
                "_"
                + "_".join(self.dust_calculator.name.split("_")[1:2])
                + "_"
                + self.dust_calculator.name.split("_")[-1]
            )
        else:
            dust_label = ""
        strong_line_names = self.pre_req_properties[1].global_kwargs[
            "strong_line_names"
        ]
        return (
            f"flux_{self.pre_req_properties[1].global_kwargs['frame']}_"
            + f"{'+'.join(strong_line_names)}{dust_label}"
        )

    @property
    def plot_name(self: Self) -> str:
        """`str`: Human-readable plot label for the emission line flux."""
        strong_line_names = self.pre_req_properties[1].global_kwargs[
            "strong_line_names"
        ]
        return (
            f"{'+'.join(strong_line_names)} flux / "
            + r"$\mathrm{erg\,s^{-1}\,cm^{-2}}$"
        )

    def _kwarg_assertions(self: Self) -> NoReturn:
        pass

    def _calc_obj_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        emission_band = phot_rest.property_kwargs[
            self.pre_req_properties[1].name
        ]["band"]
        if emission_band in phot_rest.filterset.filt_names:
            emission_band_index = int(
                np.where(
                    np.array(phot_rest.filterset.filt_names) == emission_band
                )[0][0]
            )
            if np.isnan(phot_rest.depths[emission_band_index]):
                band_wav = None
            else:
                band_wav = deepcopy(
                    phot_rest.filterset[emission_band].WavelengthCen
                )
                if band_wav is not None:
                    if (
                        self.pre_req_properties[1].global_kwargs["frame"]
                        == "rest"
                    ):
                        band_wav /= 1.0 + phot_rest.z.value
        else:
            band_wav = None
        return {"band_wav": band_wav}

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        if any(value is None for value in self.obj_kwargs.values()):
            return True
        else:
            return False

    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # extract continuum and EW chains/value
        if len(fluxes_arr) > 1:
            cont_arr = phot_rest.property_PDFs[
                self.pre_req_properties[0].name
            ].input_arr
            EW_arr = phot_rest.property_PDFs[
                self.pre_req_properties[1].name
            ].input_arr
            assert len(fluxes_arr) == len(cont_arr) == len(EW_arr)
        else:
            cont_arr = phot_rest.properties[self.pre_req_properties[0].name]
            EW_arr = phot_rest.properties[self.pre_req_properties[1].name]
        # calculate line fluxes
        line_flux_arr = (
            EW_arr
            * funcs.convert_mag_units(
                self.obj_kwargs["band_wav"],
                cont_arr,
                u.erg / (u.s * u.AA * u.cm**2),
            )
        ).to(u.erg / (u.s * u.cm**2))
        # extract dust chains/value if required
        if self.dust_calculator is not None:
            if len(fluxes_arr) > 1:
                A_arr = phot_rest.property_PDFs[
                    self.dust_calculator.name
                ].input_arr
                assert len(fluxes_arr) == len(A_arr)
            else:
                A_arr = phot_rest.properties[self.dust_calculator.name]
            # correct for dust attenuation
            line_flux_arr = funcs.dust_correct(line_flux_arr, A_arr)
        # output line fluxes in appropriate frame
        return line_flux_arr

    def _get_output_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}


class Optical_Line_Luminosity_Calculator(Rest_Frame_Property_Calculator):
    """Calculates the luminosity of a strong rest-frame optical emission line.

    Requires an observed-frame `Optical_Line_Flux_Calculator` as a
    prerequisite and converts the line flux to a luminosity assuming
    isotropic emission, ``L = 4 pi d_L^2 F``.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used to compute this property.
    SED_fit_label : `str` or `Type[SED_code]`
        Label (or `SED_code` instance) identifying the SED-fitting run
        whose `Photometry_rest` this property is calculated from.
    strong_line_names : `str` or `list`
        Name(s) of the strong optical emission line(s) (keys of
        `Emission_lines.line_diagnostics`) whose luminosity is being
        calculated. A `"+"`-delimited string is split into a list.
    rest_optical_wavs : `astropy.units.Quantity`, optional
        Two-element rest-frame wavelength range within which continuum
        bands are searched for. Default is `[4_200.0, 10_000.0] * u.AA`.
    dust_law : `str`, `Type[Dust_Law]`, or `None`, optional
        Dust attenuation law used to dust-correct the underlying line
        flux, or `None` to skip dust correction. Default is `Calzetti00`.
    beta_dust_conv : `str`, `Type[AUV_from_beta]`, or `None`, optional
        Beta-to-A(UV) conversion used to obtain the UV attenuation for
        dust correction. Default is `M99`.
    UV_ref_wav : `astropy.units.Quantity` or `None`, optional
        Rest-frame reference wavelength at which the UV attenuation is
        computed. Default is `1_500.0 * u.AA`.
    UV_wav_lims : `astropy.units.Quantity` or `None`, optional
        Two-element rest-frame wavelength range defining the UV
        continuum used to fit beta. Default is `[1_250.0, 3_000.0] * u.AA`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        strong_line_names: Union[str, list],
        # frame: str = "rest",
        rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
        dust_law: Optional[Union[str, Type[Dust_Law]]] = Calzetti00,
        beta_dust_conv: Optional[Union[str, Type[AUV_from_beta]]] = M99,
        UV_ref_wav: Optional[u.Quantity] = 1_500.0 * u.AA,
        UV_wav_lims: Optional[u.Quantity] = [1_250.0, 3_000.0] * u.AA,
    ) -> NoReturn:
        pre_req_properties = [
            Optical_Line_Flux_Calculator(
                aper_diam,
                SED_fit_label,
                strong_line_names,
                "obs",
                rest_optical_wavs,
                dust_law,
                beta_dust_conv,
                UV_ref_wav,
                UV_wav_lims,
            )
        ]
        super().__init__(aper_diam, SED_fit_label, pre_req_properties)

    @property
    def name(self: Self) -> str:
        """`str`: Short identifier,
        ``"lum_{strong_line_names}[_{dust}_...]"``."""
        if self.pre_req_properties[0].dust_calculator is not None:
            dust_label = (
                "_"
                + "_".join(
                    self.pre_req_properties[0].dust_calculator.name.split("_")[
                        1:2
                    ]
                )
                + "_"
                + self.pre_req_properties[0].dust_calculator.name.split("_")[
                    -1
                ]
            )
        else:
            dust_label = ""
        strong_line_names = (
            self.pre_req_properties[0]
            .pre_req_properties[1]
            .global_kwargs["strong_line_names"]
        )
        return f"lum_{'+'.join(strong_line_names)}{dust_label}"

    @property
    def plot_name(self: Self) -> str:
        """`str`: Human-readable plot label for the emission line
        luminosity."""
        strong_line_names = self.global_kwargs["strong_line_names"]
        return f"{'+'.join(strong_line_names)} luminosity / erg/s"

    def _kwarg_assertions(self: Self) -> None:
        pass

    def _calc_obj_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {
            "lum_distance": astropy_cosmo.luminosity_distance(
                phot_rest.z.value
            )
        }

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        if any(value is None for value in self.obj_kwargs.values()):
            return True
        else:
            return False

    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # extract line flux chains/value
        if len(fluxes_arr) > 1:
            line_flux_arr = phot_rest.property_PDFs[
                self.pre_req_properties[0].name
            ].input_arr
            assert len(fluxes_arr) == len(line_flux_arr)
        else:
            line_flux_arr = phot_rest.properties[
                self.pre_req_properties[0].name
            ]
        # if len(line_flux_arr[np.isfinite(line_flux_arr)]) == 0:
        #     breakpoint()
        # calculate line luminosities
        line_lum_arr = (
            4 * np.pi * line_flux_arr * self.obj_kwargs["lum_distance"] ** 2
        ).to(u.erg / u.s)
        # output line luminosities
        return line_lum_arr

    def _get_output_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}


class Ndot_Ion_Calculator(Rest_Frame_Property_Calculator):
    """Calculates the ionizing photon production rate, Ndot_ion,
    from the H-alpha luminosity.

    Requires an `Optical_Line_Luminosity_Calculator` for H-alpha (and,
    if `fesc_conv` is a string, a `Fesc_From_Beta_Calculator`) as
    prerequisites. Assuming Case B recombination, converts the H-alpha
    luminosity to an ionizing photon production rate, correcting for
    any Lyman-continuum escape fraction if supplied.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used to compute this property.
    SED_fit_label : `str` or `Type[SED_code]`
        Label (or `SED_code` instance) identifying the SED-fitting run
        whose `Photometry_rest` this property is calculated from.
    rest_optical_wavs : `astropy.units.Quantity`, optional
        Two-element rest-frame wavelength range within which continuum
        bands for the H-alpha line are searched for. Default is
        `[4_200.0, 10_000.0] * u.AA`.
    dust_law : `str`, `Type[Dust_Law]`, or `None`, optional
        Dust attenuation law used to dust-correct the H-alpha flux, or
        `None` to skip dust correction. Default is `Calzetti00`.
    beta_dust_conv : `str`, `Type[AUV_from_beta]`, or `None`, optional
        Beta-to-A(UV) conversion used to obtain the UV attenuation for
        dust correction. Default is `M99`.
    UV_wav_lims : `astropy.units.Quantity` or `None`, optional
        Two-element rest-frame wavelength range defining the UV
        continuum used to fit beta (and, if `fesc_conv` is a string, the
        escape fraction). Default is `[1_250.0, 3_000.0] * u.AA`.
    fesc_conv : `str`, `float`, or `None`, optional
        Either the name of a beta-to-fesc conversion (triggering a
        `Fesc_From_Beta_Calculator` prerequisite), a fixed escape
        fraction value, or `None` for zero escape fraction. Default is
        `None`.
    logged : `bool`, optional
        Whether to return `log10(Ndot_ion / Hz)`. Default is `True`.

    Attributes
    ----------
    fesc_calculator : `Fesc_From_Beta_Calculator`, `float`, or `None`
        Escape fraction calculator/value derived from `fesc_conv`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        # frame: str = "rest",
        rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
        dust_law: Optional[Union[str, Type[Dust_Law]]] = Calzetti00,
        beta_dust_conv: Optional[Union[str, Type[AUV_from_beta]]] = M99,
        UV_wav_lims: Optional[u.Quantity] = [1_250.0, 3_000.0] * u.AA,
        fesc_conv: Optional[Union[str, float]] = None,
        logged: bool = True,
    ) -> NoReturn:
        line_lum_calculator = Optical_Line_Luminosity_Calculator(
            aper_diam,
            SED_fit_label,
            "Halpha",
            rest_optical_wavs,
            dust_law,
            beta_dust_conv,
        )
        pre_req_properties = [line_lum_calculator]
        if fesc_conv is None:
            self.fesc_calculator = None
        elif isinstance(fesc_conv, str):
            self.fesc_calculator = Fesc_From_Beta_Calculator(
                aper_diam,
                SED_fit_label,
                UV_wav_lims,
                fesc_conv,
                keep_valid=True,
            )
            pre_req_properties.append(self.fesc_calculator)
        else:  # float
            self.fesc_calculator = fesc_conv
        global_kwargs = {"logged": logged}
        super().__init__(
            aper_diam, SED_fit_label, pre_req_properties, **global_kwargs
        )

    @property
    def name(self: Self) -> str:
        """`str`: Short identifier,
        ``"[log_]ndot_ion_{line}[_{dust}dust]_{fesc}"``."""
        if (
            self.pre_req_properties[0].pre_req_properties[0].dust_calculator
            is not None
        ):
            dust_label = (
                "_"
                + "_".join(
                    self.pre_req_properties[0]
                    .pre_req_properties[0]
                    .dust_calculator.name.split("_")[1:2]
                )
                + "dust"
            )
        else:
            dust_label = ""
        if self.fesc_calculator is None:
            fesc_label = "fesc=0"
        elif isinstance(self.fesc_calculator, Fesc_From_Beta_Calculator):
            fesc_label = self.fesc_calculator.name.split("_")[0]
            if dust_label == "":
                fesc_label += "_".join(fesc_label.split("_")[1:])
        else:  # isinstance(fesc_conv, float)
            fesc_label = f"fesc={self.fesc_calculator:.2f}"
        line_label = "+".join(
            self.pre_req_properties[0]
            .pre_req_properties[0]
            .pre_req_properties[1]
            .global_kwargs["strong_line_names"]
        )
        # try:
        #     ext_src_label = f"_extsrc_{self.pre_req_properties[0]
        #         .pre_req_properties[0].global_kwargs['ext_src_corrs']}" \
        #         if self.pre_req_properties[0].pre_req_properties[0]
        #         .global_kwargs["ext_src_corrs"] is not None else ""
        #     ext_src_lim_label = f"<{self.pre_req_properties[0]
        #         .pre_req_properties[0]
        #         .global_kwargs['ext_src_uplim']:.0f}" if \
        #         self.pre_req_properties[0].pre_req_properties[0]
        #         .global_kwargs["ext_src_uplim"] is not None and \
        #         self.pre_req_properties[0].pre_req_properties[0]
        #         .global_kwargs["ext_src_corrs"] is not None else ""
        # except:
        #     breakpoint()
        label = (
            f"ndot_ion_{line_label}{dust_label}_{fesc_label}"
            # {ext_src_label}{ext_src_lim_label}"
        )
        if self.global_kwargs["logged"]:
            label = f"log_{label}"
        return label

    @property
    def plot_name(self: Self) -> str:
        """`str`: Human-readable plot label for the ionizing photon
        production rate."""
        if self.global_kwargs["logged"]:
            return r"$\log(\dot{n}_{\mathrm{ion}}~/~\mathrm{s}^{-1})$"
        else:
            return r"$\dot{n}_{\mathrm{ion}}~/~\mathrm{s}^{-1}$"

    def _kwarg_assertions(self: Self) -> NoReturn:
        if self.fesc_calculator is not None:
            assert isinstance(
                self.fesc_calculator, (Fesc_From_Beta_Calculator, float)
            )
        if isinstance(self.fesc_calculator, float):
            assert self.fesc_calculator >= 0.0
            assert self.fesc_calculator <= 1.0

    def _calc_obj_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        # always pass
        return False

    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        if (
            self.fesc_calculator is None
            or self.fesc_calculator == 0.0
            or self.fesc_calculator == 0
        ):
            ndot_0 = True
        else:
            ndot_0 = False
        # extract line and UV luminosity (and fesc is required) chains/value
        if len(fluxes_arr) > 1:
            line_lum_arr = phot_rest.property_PDFs[
                self.pre_req_properties[0].name
            ].input_arr
            assert len(fluxes_arr) == len(line_lum_arr)
            if self.fesc_calculator is None:
                fesc_arr = np.full(len(fluxes_arr), 0.0)
            elif isinstance(self.fesc_calculator, Fesc_From_Beta_Calculator):
                fesc_arr = phot_rest.property_PDFs[
                    self.fesc_calculator.name
                ].input_arr
            else:  # isinstance(fesc_conv, float)
                fesc_arr = np.full(
                    len(fluxes_arr), float(self.fesc_calculator)
                )
            assert len(fluxes_arr) == len(fesc_arr)
        else:
            line_lum_arr = phot_rest.properties[
                self.pre_req_properties[0].name
            ]
            if self.fesc_calculator is None:
                fesc_arr = 0.0
            elif isinstance(self.fesc_calculator, Fesc_From_Beta_Calculator):
                fesc_arr = phot_rest.properties[self.fesc_calculator.name]
            else:  # isinstance(fesc_conv, float)
                fesc_arr = float(self.fesc_calculator)

        # calculate ndot_ion values
        # under assumption of Case B recombination
        if ndot_0:
            ndot_ion_arr = line_lum_arr / (1.36e-12 * u.erg)
        else:
            ndot_ion_arr = (
                line_lum_arr * fesc_arr / (1.36e-12 * u.erg * (1.0 - fesc_arr))
            )
        ndot_ion_arr = ndot_ion_arr.to(u.Hz)
        ndot_ion_arr[~np.isfinite(ndot_ion_arr)] = np.nan
        if self.global_kwargs["logged"]:
            ndot_ion_arr = np.log10(ndot_ion_arr.value) * u.Unit(
                f"dex({ndot_ion_arr.unit.to_string()})"
            )
        finite_ndot_ion_arr = ndot_ion_arr[np.isfinite(ndot_ion_arr)]
        if len(fluxes_arr) > 1:
            self.obj_kwargs["negative_ndot_ion_pc"] = 100.0 * (
                1 - len(finite_ndot_ion_arr) / len(ndot_ion_arr)
            )
            if (
                len(finite_ndot_ion_arr) < 50
                or self.obj_kwargs["negative_ndot_ion_pc"] > 99.0
            ):
                return None
        else:
            if len(finite_ndot_ion_arr) < 1:
                return None
        return ndot_ion_arr

    def _get_output_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}


# class Ndot_Ion_Fesc_Calculator(Rest_Frame_Property_Calculator):

#     def __init__(
#         self: Self,
#         aper_diam: u.Quantity,
#         SED_fit_label: Union[str, Type[SED_code]],
#         #frame: str = "rest",
#         rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
#         dust_law: Optional[Union[str, Type[Dust_Law]]] = Calzetti00,
#         beta_dust_conv: Optional[Union[str, Type[AUV_from_beta]]] = M99,
#         UV_wav_lims: Optional[u.Quantity] = [1_250.0, 3_000.0] * u.AA,
#         fesc_conv: Union[str, float] = "Chisholm22",
#         logged: bool = True,
#     ) -> NoReturn:
#         ndot_ion_calculator = \
#             Ndot_Ion_Calculator(
#                 aper_diam,
#                 SED_fit_label,
#                 #frame,
#                 rest_optical_wavs,
#                 dust_law,
#                 beta_dust_conv,
#                 UV_wav_lims,
#                 fesc_conv,
#                 logged = False,
#             )

#         pre_req_properties = [ndot_ion_calculator]
#         if isinstance(fesc_conv, str):
#             self.fesc_calculator = Fesc_From_Beta_Calculator(
#                 aper_diam,
#                 SED_fit_label,
#                 UV_wav_lims,
#                 fesc_conv,
#                 keep_valid = True
#             )
#             pre_req_properties.append(self.fesc_calculator)
#         else: # float
#             self.fesc_calculator = fesc_conv
#         global_kwargs = {"logged": logged}
#         super().__init__(
#             aper_diam, SED_fit_label, pre_req_properties, **global_kwargs
#         )

#     @property
#     def name(self: Self) -> str:
#         if self.pre_req_properties[0].pre_req_properties[0] \
#                 .pre_req_properties[0].dust_calculator is not None:
#             dust_label = "_" + "_".join(
#                 self.pre_req_properties[0].pre_req_properties[0]
#                 .dust_calculator.name.split("_")[1:2]
#             ) + "dust"
#         else:
#             dust_label = ""
#         if isinstance(self.fesc_calculator, Fesc_From_Beta_Calculator):
#             fesc_label = self.fesc_calculator.name.split("_")[0]
#             if dust_label == "":
#                 fesc_label += "_".join(fesc_label.split("_")[1:])
#         else: # isinstance(fesc_conv, float)
#             fesc_label = f"fesc={self.fesc_calculator:.2f}"
#         line_label = "+".join(
#             self.pre_req_properties[0].pre_req_properties[0]
#             .pre_req_properties[0].pre_req_properties[1]
#             .global_kwargs["strong_line_names"]
#         )
#         # try:
#         #     ext_src_label = f"_extsrc_{self.pre_req_properties[0]
#         #         .pre_req_properties[0].global_kwargs['ext_src_corrs']}" \
#         #         if self.pre_req_properties[0].pre_req_properties[0]
#         #         .global_kwargs["ext_src_corrs"] is not None else ""
#         #     ext_src_lim_label = f"<{self.pre_req_properties[0]
#         #         .pre_req_properties[0]
#         #         .global_kwargs['ext_src_uplim']:.0f}" if \
#         #         self.pre_req_properties[0].pre_req_properties[0]
#         #         .global_kwargs["ext_src_uplim"] is not None and \
#         #         self.pre_req_properties[0].pre_req_properties[0]
#         #         .global_kwargs["ext_src_corrs"] is not None else ""
#         # except:
#         #     breakpoint()
#         label = f"fesc_ndot_ion_{line_label}{dust_label}_{fesc_label}"
#         #{ext_src_label}{ext_src_lim_label}"
#         if self.global_kwargs["logged"]:
#             label = f"log_{label}"
#         return label

#     @property
#     def plot_name(self: Self) -> str:
#         if self.global_kwargs["logged"]:
#             return (
#                 r"$\log(\dot{n}_{\mathrm{ion}}f_{\mathrm{esc}}"
#                 r"~/~\mathrm{s}^{-1})$"
#             )
#         else:
#             return
r"$\dot{n}_{\mathrm{ion}}f_{\mathrm{esc}}~/~\mathrm{s}^{-1}$"

#     def _kwarg_assertions(self: Self) -> NoReturn:
#         if isinstance(self.fesc_calculator, float):
#             assert self.fesc_calculator >= 0.0
#             assert self.fesc_calculator <= 1.0
#         else:
#             assert isinstance(
#                 self.fesc_calculator, Fesc_From_Beta_Calculator
#             )

#     def _calc_obj_kwargs(
#         self: Self,
#         phot_rest: Photometry_rest
#     ) -> Dict[str, Any]:
#         return {}

#     def _fail_criteria(
#         self: Self,
#         phot_rest: Photometry_rest,
#     ) -> bool:
#         # always pass
#         return False

#     def _calculate(
#         self: Self,
#         fluxes_arr: u.Quantity,
#         phot_rest: Photometry_rest,
#     ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
#         # extract line and UV luminosity (and fesc is required) chains/value
#         if len(fluxes_arr) > 1:
#             ndot_ion_arr = phot_rest.property_PDFs[
#                 self.pre_req_properties[0].name
#             ].input_arr
#             assert len(fluxes_arr) == len(ndot_ion_arr)
#             if self.fesc_calculator is None:
#                 fesc_arr = np.full(len(fluxes_arr), 0.0)
#             elif isinstance(self.fesc_calculator, Fesc_From_Beta_Calculator):
#                 fesc_arr = phot_rest.property_PDFs[
#                     self.fesc_calculator.name
#                 ].input_arr
#             else: # isinstance(fesc_conv, float)
#                 fesc_arr = np.full(len(fluxes_arr), self.fesc_calculator)
#             assert len(fluxes_arr) == len(fesc_arr)
#         else:
#             ndot_ion_arr = phot_rest.properties[
#                 self.pre_req_properties[0].name
#             ]
#             if self.fesc_calculator is None:
#                 fesc_arr = 0.0
#             elif isinstance(self.fesc_calculator, Fesc_From_Beta_Calculator):
#                 fesc_arr = phot_rest.properties[self.fesc_calculator.name]
#             else: # isinstance(fesc_conv, float)
#                 fesc_arr = self.fesc_calculator
#         # calculate fesc_ndot_ion values
#         fesc_ndot_ion_arr = fesc_arr * ndot_ion_arr
#         return fesc_ndot_ion_arr

#     def _get_output_kwargs(
#         self: Self,
#         phot_rest: Photometry_rest
#     ) -> Dict[str, Any]:
#         return {}


class Xi_Ion_Calculator(Rest_Frame_Property_Calculator):
    """Calculates the ionizing photon production efficiency, xi_ion,
    from H-alpha and UV luminosities.

    Requires an `Optical_Line_Luminosity_Calculator` for H-alpha and an
    `LUV_Calculator` (and, if `fesc_conv` is a string, a
    `Fesc_From_Beta_Calculator`) as prerequisites. Assuming Case B
    recombination, computes
    ``xi_ion = L(Halpha) / (1.36e-12 erg * (1 - fesc) * LUV)``.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used to compute this property.
    SED_fit_label : `str` or `Type[SED_code]`
        Label (or `SED_code` instance) identifying the SED-fitting run
        whose `Photometry_rest` this property is calculated from.
    rest_optical_wavs : `astropy.units.Quantity`, optional
        Two-element rest-frame wavelength range within which continuum
        bands for the H-alpha line are searched for. Default is
        `[4_200.0, 10_000.0] * u.AA`.
    dust_law : `str`, `Type[Dust_Law]`, or `None`, optional
        Dust attenuation law used to dust-correct the H-alpha flux and
        UV luminosity, or `None` to skip dust correction. Default is
        `Calzetti00`.
    beta_dust_conv : `str`, `Type[AUV_from_beta]`, or `None`, optional
        Beta-to-A(UV) conversion used to obtain the UV attenuation for
        dust correction. Default is `M99`.
    UV_ref_wav : `astropy.units.Quantity` or `None`, optional
        Rest-frame reference wavelength at which LUV is computed.
        Default is `1_500.0 * u.AA`.
    UV_wav_lims : `astropy.units.Quantity` or `None`, optional
        Two-element rest-frame wavelength range defining the UV
        continuum used to fit beta (and, if `fesc_conv` is a string, the
        escape fraction). Default is `[1_250.0, 3_000.0] * u.AA`.
    top_hat_width : `astropy.units.Quantity`, optional
        Width of the top-hat wavelength window used when computing mUV
        (for LUV). Default is `100.0 * u.AA`.
    resolution : `astropy.units.Quantity`, optional
        Wavelength spacing of the top-hat window grid used when
        computing mUV (for LUV). Default is `1.0 * u.AA`.
    fesc_conv : `str`, `float`, or `None`, optional
        Either the name of a beta-to-fesc conversion (triggering a
        `Fesc_From_Beta_Calculator` prerequisite), a fixed escape
        fraction value, or `None` for zero escape fraction. Default is
        `None`.
    logged : `bool`, optional
        Whether to return `log10(xi_ion / (Hz / erg))`. Default is `True`.

    Attributes
    ----------
    fesc_calculator : `Fesc_From_Beta_Calculator`, `float`, or `None`
        Escape fraction calculator/value derived from `fesc_conv`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        # frame: str = "rest",
        rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
        dust_law: Optional[Union[str, Type[Dust_Law]]] = Calzetti00,
        beta_dust_conv: Optional[Union[str, Type[AUV_from_beta]]] = M99,
        UV_ref_wav: Optional[u.Quantity] = 1_500.0 * u.AA,
        UV_wav_lims: Optional[u.Quantity] = [1_250.0, 3_000.0] * u.AA,
        top_hat_width: u.Quantity = 100.0 * u.AA,
        resolution: u.Quantity = 1.0 * u.AA,
        fesc_conv: Optional[Union[str, float]] = None,
        logged: bool = True,
        # ext_src_corrs: Optional[str] = "UV",
        # ext_src_uplim: Optional[Union[int, float]] = 10.0,
    ) -> NoReturn:
        line_lum_calculator = Optical_Line_Luminosity_Calculator(
            aper_diam,
            SED_fit_label,
            "Halpha",
            # frame,
            rest_optical_wavs,
            dust_law,
            beta_dust_conv,
            UV_ref_wav,
            UV_wav_lims,
        )
        LUV_calculator = LUV_Calculator(
            aper_diam,
            SED_fit_label,
            # frame,
            UV_wav_lims,
            UV_ref_wav,
            beta_dust_conv,
            top_hat_width,
            resolution,
            ext_src_corrs=None,
            ext_src_uplim=None,
        )
        pre_req_properties = [line_lum_calculator, LUV_calculator]
        if fesc_conv is None:
            self.fesc_calculator = None
        elif isinstance(fesc_conv, str):
            self.fesc_calculator = Fesc_From_Beta_Calculator(
                aper_diam,
                SED_fit_label,
                UV_wav_lims,
                fesc_conv,
                keep_valid=True,
            )
            pre_req_properties.append(self.fesc_calculator)
        else:  # float
            self.fesc_calculator = fesc_conv
        global_kwargs = {"logged": logged}
        super().__init__(
            aper_diam, SED_fit_label, pre_req_properties, **global_kwargs
        )

    @property
    def name(self: Self) -> str:
        """`str`: Short identifier,
        ``"[log_]xi_ion_{line}[_{dust}dust]_{fesc}"``."""
        if (
            self.pre_req_properties[0].pre_req_properties[0].dust_calculator
            is not None
        ):
            dust_label = (
                "_"
                + "_".join(
                    self.pre_req_properties[0]
                    .pre_req_properties[0]
                    .dust_calculator.name.split("_")[1:2]
                )
                + "dust"
            )
        else:
            dust_label = ""
        if self.fesc_calculator is None:
            fesc_label = "fesc=0"
        elif isinstance(self.fesc_calculator, Fesc_From_Beta_Calculator):
            fesc_label = self.fesc_calculator.name.split("_")[0]
            if dust_label == "":
                fesc_label += "_".join(fesc_label.split("_")[1:])
        else:  # isinstance(fesc_conv, float)
            fesc_label = f"fesc={self.fesc_calculator:.2f}"
        line_label = "+".join(
            self.pre_req_properties[0]
            .pre_req_properties[0]
            .pre_req_properties[1]
            .global_kwargs["strong_line_names"]
        )
        # ext_src_label = "_extsrc" if self.pre_req_properties[1]. \
        #     pre_req_properties[0].global_kwargs["ext_src_corrs"] else ""
        label = (
            f"xi_ion_{line_label}{dust_label}_{fesc_label}"  # {ext_src_label}"
        )
        if self.global_kwargs["logged"]:
            label = f"log_{label}"
        return label

    @property
    def plot_name(self: Self) -> str:
        """`str`: Human-readable plot label for the ionizing photon
        production efficiency."""
        if self.global_kwargs["logged"]:
            return r"$\log(\xi_{\mathrm{ion}}~/~\mathrm{Hz erg}^{-1})$"
        else:
            return r"$\xi_{\mathrm{ion}}~/~\mathrm{Hz erg}^{-1}$"

    def _kwarg_assertions(self: Self) -> NoReturn:
        if self.fesc_calculator is not None:
            assert isinstance(
                self.fesc_calculator, (Fesc_From_Beta_Calculator, float)
            )
        if isinstance(self.fesc_calculator, float):
            assert self.fesc_calculator >= 0.0
            assert self.fesc_calculator <= 1.0

    def _calc_obj_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        # always pass
        return False

    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # extract line and UV luminosity (and fesc is required) chains/value
        if len(fluxes_arr) > 1:
            line_lum_arr = phot_rest.property_PDFs[
                self.pre_req_properties[0].name
            ].input_arr
            LUV_arr = phot_rest.property_PDFs[
                self.pre_req_properties[1].name
            ].input_arr
            assert len(fluxes_arr) == len(line_lum_arr) == len(LUV_arr)
            if self.fesc_calculator is None:
                fesc_arr = np.full(len(fluxes_arr), 0.0)
            elif isinstance(self.fesc_calculator, Fesc_From_Beta_Calculator):
                fesc_arr = phot_rest.property_PDFs[
                    self.fesc_calculator.name
                ].input_arr
            else:  # isinstance(fesc_conv, float)
                fesc_arr = np.full(len(fluxes_arr), self.fesc_calculator)
            assert len(fluxes_arr) == len(fesc_arr)
        else:
            line_lum_arr = phot_rest.properties[
                self.pre_req_properties[0].name
            ]
            LUV_arr = phot_rest.properties[self.pre_req_properties[1].name]
            if self.fesc_calculator is None:
                fesc_arr = 0.0
            elif isinstance(self.fesc_calculator, Fesc_From_Beta_Calculator):
                fesc_arr = phot_rest.properties[self.fesc_calculator.name]
            else:  # isinstance(fesc_conv, float)
                fesc_arr = self.fesc_calculator
        # calculate xi_ion values
        # under assumption of Case B recombination
        xi_ion_arr = (
            line_lum_arr / (1.36e-12 * u.erg * (1.0 - fesc_arr) * LUV_arr)
        ).to(u.Hz / u.erg)
        xi_ion_arr[~np.isfinite(xi_ion_arr)] = np.nan
        if self.global_kwargs["logged"]:
            xi_ion_arr = np.log10(xi_ion_arr.value) * u.Unit(
                f"dex({xi_ion_arr.unit.to_string()})"
            )
        finite_xi_ion_arr = xi_ion_arr[np.isfinite(xi_ion_arr)]
        if len(fluxes_arr) > 1:
            self.obj_kwargs["negative_xi_ion_pc"] = 100.0 * (
                1 - len(finite_xi_ion_arr) / len(xi_ion_arr)
            )
            if (
                len(finite_xi_ion_arr) < 50
                or self.obj_kwargs["negative_xi_ion_pc"] > 99.0
            ):
                return None
        else:
            if len(finite_xi_ion_arr) < 1:
                return None
        return xi_ion_arr

    def _get_output_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}


class SFR_Halpha_Calculator(Rest_Frame_Property_Calculator):
    """Calculates the H-alpha-derived star formation rate, SFR_Halpha.

    Requires an `Optical_Line_Luminosity_Calculator` for H-alpha as a
    prerequisite and converts the (optionally dust-corrected) H-alpha
    luminosity to a star formation rate using the Kennicutt (1998)
    calibration (Salpeter 1955 IMF, 0.1-100 Msun),
    ``SFR = 7.9e-42 * L(Halpha) [erg/s]``.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used to compute this property.
    SED_fit_label : `str` or `Type[SED_code]`
        Label (or `SED_code` instance) identifying the SED-fitting run
        whose `Photometry_rest` this property is calculated from.
    rest_optical_wavs : `astropy.units.Quantity`, optional
        Two-element rest-frame wavelength range within which continuum
        bands for the H-alpha line are searched for. Default is
        `[4_200.0, 10_000.0] * u.AA`.
    dust_law : `str`, `Type[Dust_Law]`, or `None`, optional
        Dust attenuation law used to dust-correct the H-alpha flux, or
        `None` to skip dust correction. Default is `Calzetti00`.
    beta_dust_conv : `str`, `Type[AUV_from_beta]`, or `None`, optional
        Beta-to-A(UV) conversion used to obtain the UV attenuation for
        dust correction. Default is `M99`.
    UV_ref_wav : `astropy.units.Quantity` or `None`, optional
        Rest-frame reference wavelength at which the UV attenuation is
        computed. Default is `1_500.0 * u.AA`.
    UV_wav_lims : `astropy.units.Quantity` or `None`, optional
        Two-element rest-frame wavelength range defining the UV
        continuum used to fit beta. Default is `[1_250.0, 3_000.0] * u.AA`.
    logged : `bool`, optional
        Whether to return `log10(SFR_Halpha / (Msun / yr))`. Default is
        `True`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
        dust_law: Optional[Union[str, Type[Dust_Law]]] = Calzetti00,
        beta_dust_conv: Optional[Union[str, Type[AUV_from_beta]]] = M99,
        UV_ref_wav: Optional[u.Quantity] = 1_500.0 * u.AA,
        UV_wav_lims: Optional[u.Quantity] = [1_250.0, 3_000.0] * u.AA,
        logged: bool = True,
    ) -> NoReturn:
        line_lum_calculator = Optical_Line_Luminosity_Calculator(
            aper_diam,
            SED_fit_label,
            "Halpha",
            rest_optical_wavs,
            dust_law,
            beta_dust_conv,
            UV_ref_wav,
            UV_wav_lims,
        )
        pre_req_properties = [line_lum_calculator]
        global_kwargs = {"logged": logged}
        super().__init__(
            aper_diam,
            SED_fit_label,
            pre_req_properties,
            **global_kwargs,
        )

    @property
    def name(self: Self) -> str:
        """`str`: Short identifier, ``"[log_]SFR_Halpha[_{dust}dust]"``."""
        if (
            self.pre_req_properties[0].pre_req_properties[0].dust_calculator
            is not None
        ):
            dust_label = (
                "_"
                + "_".join(
                    self.pre_req_properties[0]
                    .pre_req_properties[0]
                    .dust_calculator.name.split("_")[1:2]
                )
                + "dust"
            )
        else:
            dust_label = ""
        label = f"SFR_Halpha{dust_label}"
        if self.global_kwargs["logged"]:
            label = f"log_{label}"
        return label

    @property
    def plot_name(self: Self) -> str:
        """`str`: Human-readable plot label for the H-alpha-derived
        star formation rate."""
        if self.global_kwargs["logged"]:
            return r"$\log(SFR_{\mathrm{H}\alpha})$"
        else:
            return r"$SFR_{\mathrm{H}\alpha}$"

    def _kwarg_assertions(self: Self) -> NoReturn:
        pass

    def _calc_obj_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        # always pass
        return False

    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # extract line and UV luminosity (and fesc is required) chains/value
        if len(fluxes_arr) > 1:
            line_lum_arr = phot_rest.property_PDFs[
                self.pre_req_properties[0].name
            ].input_arr
            assert len(fluxes_arr) == len(line_lum_arr)
        else:
            line_lum_arr = phot_rest.properties[
                self.pre_req_properties[0].name
            ]
        # calculate SFR_Halpha values
        # from Kennicutt 1998 (Salpeter 1955 IMF, 0.1-100 Msun)
        SFR_Halpha_arr = (
            (7.9e-42 * line_lum_arr.to(u.erg / u.s)).value * u.Msun / u.yr
        )
        finite_SFR_Halpha_arr = SFR_Halpha_arr[np.isfinite(SFR_Halpha_arr)]
        if len(fluxes_arr) > 1:
            self.obj_kwargs["negative_SFR_Halpha_pc"] = 100.0 * (
                1 - len(finite_SFR_Halpha_arr) / len(SFR_Halpha_arr)
            )
            if (
                len(finite_SFR_Halpha_arr) < 50
                or self.obj_kwargs["negative_SFR_Halpha_pc"] > 99.0
            ):
                return None
        else:
            if len(finite_SFR_Halpha_arr) < 1:
                return None
        SFR_Halpha_arr[~np.isfinite(SFR_Halpha_arr)] = np.nan
        if self.global_kwargs["logged"]:
            SFR_Halpha_arr = np.log10(SFR_Halpha_arr.value) * u.Unit(
                f"dex({SFR_Halpha_arr.unit.to_string()})"
            )
        return SFR_Halpha_arr

    def _get_output_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}


class Lya_Break_Strength_Calculator(Rest_Frame_Property_Calculator):
    """Calculates the Lyman-alpha break strength as the color across
    the Ly-alpha break.

    Identifies the reddest rest-frame filter that is entirely blueward of the
    Lyman-alpha wavelength and the bluest rest-frame filter that is entirely
    redward of the Lyman-alpha wavelength, then computes the color (magnitude
    difference) between them. The break strength is defined as
    mag_blue - mag_red in rest-frame AB magnitudes.

    For non-detections in the blue band (SNR < snr_lolim), the blue flux is
    clamped to the detection limit flux to calculate a lower-limit break
    strength, with the lower error bar set to nan to indicate this. To
    disable this behavior and use all measurements regardless of SNR, set
    snr_lolim to None.

    Parameters
    ----------
    aper_diam : `astropy.units.Quantity`
        Aperture diameter of the photometry used to compute this property.
    SED_fit_label : `str` or `Type[SED_code]`
        Label (or `SED_code` instance) identifying the SED-fitting run
        whose `Photometry_rest` this property is calculated from.
    snr_lolim : `float` or `None`, optional
        SNR threshold below which the blueward band is treated as a
        non-detection and the break strength is computed as a lower
        limit. Set to `None` to disable lower-limit handling. Default
        is `2.0`.
    """

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        snr_lolim: Optional[float] = None,
        **kwargs,
    ) -> NoReturn:
        if "lya_wav" not in kwargs:
            from ..spectra.Emission_lines import line_diagnostics

            kwargs["lya_wav"] = line_diagnostics["Lya"]["line_wav"]
        kwargs["snr_lolim"] = snr_lolim
        super().__init__(aper_diam, SED_fit_label, [], **kwargs)

    @property
    def name(self: Self) -> str:
        """`str`: Short identifier, ``"lya_break_strength"``."""
        name_ = "lya_break_strength"
        if self.global_kwargs["snr_lolim"] is not None:
            name_ += f"_snr_lolim={self.global_kwargs['snr_lolim']:.2f}"
        return name_

    @property
    def plot_name(self: Self) -> str:
        """`str`: Human-readable plot label for the Lyman-alpha break
        strength."""
        return r"$\Delta m_{\rm AB}(\rm Ly\alpha)$"

    def _kwarg_assertions(self: Self) -> None:
        pass

    def _calc_obj_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        """Identify filters entirely blueward and redward of the
        Lyman-alpha wavelength."""
        from ..spectra.Emission_lines import line_diagnostics

        # Get rest-frame filter edges and centers
        rest_filt_centers = []
        rest_filt_upper_edges = []
        rest_filt_lower_edges = []

        for filt in phot_rest.filterset:
            center = funcs.convert_wav_units(
                filt.WavelengthCen / (1.0 + phot_rest.z.value), u.AA
            ).value
            upper = funcs.convert_wav_units(
                filt.WavelengthUpper50 / (1.0 + phot_rest.z.value), u.AA
            ).value
            lower = funcs.convert_wav_units(
                filt.WavelengthLower50 / (1.0 + phot_rest.z.value), u.AA
            ).value
            rest_filt_centers.append(center)
            rest_filt_upper_edges.append(upper)
            rest_filt_lower_edges.append(lower)

        rest_filt_centers = np.array(rest_filt_centers)
        rest_filt_upper_edges = np.array(rest_filt_upper_edges)
        rest_filt_lower_edges = np.array(rest_filt_lower_edges)

        # Find filters entirely blueward (upper edge < Ly-alpha) and
        # entirely redward (lower edge > Ly-alpha)
        lya_wavelength = line_diagnostics["Lya"]["line_wav"].to(u.AA).value
        blue_indices = np.where(rest_filt_upper_edges < lya_wavelength)[0]
        red_indices = np.where(rest_filt_lower_edges > lya_wavelength)[0]

        # Select the reddest blue filter (highest center) and the
        # bluest red filter (lowest center)
        if len(blue_indices) == 0 or len(red_indices) == 0:
            return {
                "blue_idx": None,
                "red_idx": None,
                "blue_filt_name": None,
                "red_filt_name": None,
                "blue_wav": None,
                "red_wav": None,
                "is_limit": False,
            }

        blue_idx = blue_indices[np.argmax(rest_filt_centers[blue_indices])]
        red_idx = red_indices[np.argmin(rest_filt_centers[red_indices])]

        # Check if blue filter is below SNR limit (only if snr_lolim
        # is not None)
        snr_lolim = self.global_kwargs["snr_lolim"]
        # TODO: Un-aperture correct the phot_rest.flux!!!
        blue_flux_obs = phot_rest.flux[blue_idx].value
        blue_flux_err_obs = phot_rest.flux_errs[blue_idx].value
        blue_snr = blue_flux_obs / blue_flux_err_obs
        is_limit = (snr_lolim is not None) and (blue_snr < snr_lolim)

        return {
            "blue_idx": blue_idx,
            "red_idx": red_idx,
            "blue_filt_name": phot_rest.filterset.filt_names[blue_idx],
            "red_filt_name": phot_rest.filterset.filt_names[red_idx],
            "blue_wav": rest_filt_centers[blue_idx] * u.AA,
            "red_wav": rest_filt_centers[red_idx] * u.AA,
            "is_limit": is_limit,
        }

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        """Fail if no valid filter pair brackets the break."""
        if (
            self.obj_kwargs["blue_idx"] is None
            or self.obj_kwargs["red_idx"] is None
        ):
            return True

        # Check for valid flux measurements
        blue_SNR = (
            phot_rest.flux[self.obj_kwargs["blue_idx"]]
            / phot_rest.flux_errs[self.obj_kwargs["blue_idx"]]
        )
        red_SNR = (
            phot_rest.flux[self.obj_kwargs["red_idx"]]
            / phot_rest.flux_errs[self.obj_kwargs["red_idx"]]
        )

        if np.isnan(blue_SNR) or np.isnan(red_SNR):
            return True

        return False

    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        """Calculate break strength as mag_blue - mag_red."""
        blue_idx = self.obj_kwargs["blue_idx"]
        red_idx = self.obj_kwargs["red_idx"]
        blue_wav = self.obj_kwargs["blue_wav"]
        red_wav = self.obj_kwargs["red_wav"]

        # Extract fluxes for the blue and red filters
        if fluxes_arr.ndim == 1:
            # Single measurement - reshape to 2D for consistent handling
            blue_flux = fluxes_arr[blue_idx : blue_idx + 1].copy()
            red_flux = fluxes_arr[red_idx : red_idx + 1]
        else:
            # Array of chains
            blue_flux = fluxes_arr[:, blue_idx].copy()
            red_flux = fluxes_arr[:, red_idx]

        # If blue flux is below SNR limit, use limit flux (flux at snr_lolim)
        if self.obj_kwargs["is_limit"]:
            snr_lolim = self.global_kwargs["snr_lolim"]
            limit_mag = (
                phot_rest.depths[blue_idx]
                - 2.5 * np.log10(snr_lolim / 5.0) * u.ABmag
            )
            blue_mag = limit_mag * np.ones_like(blue_flux.value)

        # Convert fluxes to AB magnitudes using galfind's conversion utilities
        with np.errstate(divide="ignore", invalid="ignore"):
            if not self.obj_kwargs["is_limit"]:
                blue_mag = funcs.convert_mag_units(
                    blue_wav, blue_flux, u.ABmag
                )
            red_mag = funcs.convert_mag_units(red_wav, red_flux, u.ABmag)

        # Calculate break strength as color (blue - red)
        break_strength = blue_mag - red_mag

        return break_strength

    def _get_output_kwargs(
        self: Self, phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {
            "blue_filter": self.obj_kwargs["blue_filt_name"],
            "red_filter": self.obj_kwargs["red_filt_name"],
            "blue_wav_rest": self.obj_kwargs["blue_wav"],
            "red_wav_rest": self.obj_kwargs["red_wav"],
            "is_lower_limit": self.obj_kwargs["is_limit"],
        }

    def _call_phot_rest(
        self: Self,
        phot_rest: Photometry_rest,
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        save_path: Optional[str] = None,
        dtype: np.dtype = np.float32,
    ) -> Optional[Photometry_rest]:
        """Override to handle lower-limit cases by setting lower
        error to nan."""
        property_name = self.name

        result = super()._call_phot_rest(
            phot_rest,
            n_chains,
            output=True,
            overwrite=overwrite,
            save_path=save_path,
            dtype=dtype,
        )

        if not hasattr(self, "obj_kwargs") or self.obj_kwargs is None:
            self.obj_kwargs = self._calc_obj_kwargs(phot_rest)

        if self.obj_kwargs.get("is_limit", False):
            lower_err, upper_err = phot_rest.property_errs[property_name]
            if hasattr(upper_err, "unit"):
                phot_rest.property_errs[property_name] = (
                    np.array([np.nan, upper_err.value]) * upper_err.unit
                )
            else:
                phot_rest.property_errs[property_name] = np.array(
                    [np.nan, upper_err]
                )

        if output:
            return result

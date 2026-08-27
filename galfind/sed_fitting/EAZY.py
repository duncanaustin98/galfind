#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wrapper around EAZY-py photo-z and SED-fitting tool.

Handles input catalogue construction, filter response file setup, PhotoZ fitter
execution, and result parsing into GALFIND-native objects.
"""

from __future__ import annotations

# EAZY.py
import itertools
import logging
import os
import warnings
from copy import deepcopy
from pathlib import Path
from shutil import copy
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    NoReturn,
    Optional,
    Tuple,
    Type,
    Union,
)

import astropy.units as u
import eazy
import h5py
import numpy as np
from astropy.table import Table
from eazy import hdf5
from scipy.linalg import LinAlgWarning
from tqdm import tqdm

if TYPE_CHECKING:
    from ..visualization import Redshift_PDF
    from . import (
        PDF,
        Catalogue,
        Galaxy,
        Multiple_Filter,
        SED_obs,
        SED_Result,
        Spectral_Catalogue,
    )
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

warnings.filterwarnings("ignore", category=LinAlgWarning)

from .. import config, galfind_logger
from ..spectra.SED import SED_obs
from ..utils import useful_funcs_austind as funcs
from ..utils.decorators import run_in_dir
from ..utils.exceptions import (
    GalfindTypeError,
    IncompatibleKwargsError,
    LengthMismatchError,
    MissingFileError,
    RangeError,
)
from .SED_codes import SED_code

# %% EAZY SED fitting code

# TODO: update these at runtime
"""EAZY_FILTER_CODES = {
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
}"""


def _resolve_templates_file(
    templates_file_path: str, eazy_templates_path: str
) -> str:
    """Rewrite a ``.param`` template list to use the local template dir.

    Some checked-in template list files (e.g. the Larson/Nakajima/sfhz/hot
    sets, unlike EAZY's own bundled fsps templates) list their constituent
    ``.dat``/``.fits`` template paths as machine-specific absolute paths
    (from whichever machine they were generated on) rather than paths
    relative to `eazy_templates_path`. Since these directories aren't
    shipped with eazy-py, such paths break on any other machine (e.g. CI).

    This locates the last ``/templates/`` segment in each such path and
    replaces everything up to and including it with `eazy_templates_path`,
    writing a resolved copy into the current working directory (the
    per-run EAZY_DIR) if any line needed rewriting.

    Parameters
    ----------
    templates_file_path : `str`
        Path to the original ``.param`` template list file.
    eazy_templates_path : `str`
        Local, correctly resolved ``EAZY_TEMPLATE_DIR`` for this machine.

    Returns
    -------
    `str`
        Path to the resolved ``.param`` file (the original path if no
        line needed rewriting).
    """
    with open(templates_file_path) as f:
        lines = f.readlines()

    changed = False
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue
        tokens = line.split()
        path_token = tokens[1]
        marker = "/templates/"
        if path_token.startswith("/") and marker in path_token:
            suffix = path_token.rsplit(marker, 1)[1]
            resolved_path = f"{eazy_templates_path}/{suffix}"
            if resolved_path != path_token:
                tokens[1] = resolved_path
                line = " ".join(tokens) + "\n"
                changed = True
        new_lines.append(line)

    if not changed:
        return templates_file_path

    resolved_file_path = os.path.abspath(
        f"resolved_{os.path.basename(templates_file_path)}"
    )
    with open(resolved_file_path, "w") as f:
        f.writelines(new_lines)
    return resolved_file_path


class EAZY(SED_code):
    """`SED_code` wrapper around the external EAZY-py photo-z /
    SED-fitting tool.

    Handles construction of EAZY ASCII input catalogues and filter
    response files, running the ``eazy-py`` `PhotoZ` fitter, and parsing
    the resulting best-fit redshifts, rest-frame UBVJ fluxes, SEDs and
    redshift PDFs back into GALFIND-native `SED_obs` and `Redshift_PDF`
    objects.

    Parameters
    ----------
    SED_fit_params : `dict`
        Dictionary of SED fitting parameters/options for this run. Must
        contain (or will be populated with defaults for) the keys
        required by EAZY, including ``"templates"`` and ``"lowz_zmax"``.
        Passed on to `SED_code.__init__`.
    **kwargs : `dict`
        Additional keyword arguments passed on to `SED_code.__init__`,
        where each key/value pair is set as an instance attribute.

    Attributes
    ----------
    SED_fit_params : `dict`
        SED fitting parameters/options, set by `SED_code.__init__`.
    """

    ID_label = "IDENT"

    # ext_src_corr_properties = []
    def __init__(
        self: Self,
        SED_fit_params: Dict[str, Any],
        **kwargs: Dict[str, Any],
    ):
        super().__init__(SED_fit_params, **kwargs)

    @classmethod
    def from_label(cls, label: str) -> Type[SED_code]:
        """Construct an `EAZY` instance from a saved catalogue label.

        Parameters
        ----------
        label : `str`
            Label string of the form
            ``"<code>_<templates>_<lowz_zmax_label>"``, as produced by
            `EAZY` output columns.

        Returns
        -------
        `EAZY`
            A new `EAZY` instance configured with the `SED_fit_params`
            parsed from `label`.
        """
        label_arr = label.split("_")
        templates = "_".join(
            label_arr[1:-1]
        )  # templates may contain underscore
        SED_fit_params = {
            "templates": templates,
            "lowz_zmax": funcs.zmax_from_lowz_label(label_arr[-1]),
        }
        return cls(SED_fit_params)

    # @property
    # def ID_label(self) -> str:
    #     return "IDENT"

    @property
    def label(self) -> str:
        """`str`: Unique label for this fitting run, implementing the
        `SED_code.label` interface.

        Combines the class name, template set name, and low-z zmax label.
        """
        # first write the code name, next write the template name,
        # finish off with lowz_zmax
        return (
            f"{self.__class__.__name__}_{self.SED_fit_params['templates']}"
            + f"_{funcs.lowz_label(self.SED_fit_params['lowz_zmax'])}"
        )

    @property
    def hdu_name(self) -> str:
        """`str`: Name of the FITS HDU/extension this code's output is
        stored under.

        Implements the `SED_code.hdu_name` interface. Combines the class
        name with the template set name.
        """
        return f"{self.__class__.__name__}_{self.SED_fit_params['templates']}"

    @property
    def tab_suffix(self) -> str:
        """`str`: Column-name suffix used to distinguish this run's
        output columns.

        Implements the `SED_code.tab_suffix` interface. Combines the
        template set name with the low-z zmax label.
        """
        return (
            f"{self.SED_fit_params['templates']}_"
            + f"{funcs.lowz_label(self.SED_fit_params['lowz_zmax'])}"
        )

    @property
    def required_SED_fit_params(self) -> List[str]:
        """`list` of `str`: Names of the `SED_fit_params` keys
        required by `EAZY`.

        Implements the `SED_code.required_SED_fit_params` interface.
        """
        return ["templates", "lowz_zmax"]

    @property
    def are_errs_percentiles(self) -> bool:
        """`bool`: Whether output property errors are stored as
        percentiles rather than 1-sigma values.

        Implements the `SED_code.are_errs_percentiles` interface.
        """
        return False

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
        lowz_zmax_arr: Optional[List[float]] = None,
        save_name: Optional[str] = None,
        **fit_kwargs,
    ) -> Union[Galaxy, Catalogue, Spectral_Catalogue]:
        target = super().__call__(
            target,
            aper_diam,
            save_PDFs,
            save_SEDs,
            load_PDFs,
            load_SEDs,
            timed,
            overwrite,
            update,
            save_name=save_name,
            **fit_kwargs,
        )
        from ..catalogues.Catalogue import Catalogue, Spectral_Catalogue

        if isinstance(target, (Catalogue, Spectral_Catalogue)):
            # BUG: save_name doesn't propagate
            # - don't think it makes a difference in the function though
            self._update_lowz_zmax(
                target, aper_diam, lowz_zmax_arr, save_name=save_name
            )
        return target

    def _update_lowz_zmax(
        self: Self,
        cat: Union[Catalogue, Spectral_Catalogue],
        aper_diam: u.Quantity,
        lowz_zmax_arr: List[float],
        save_name: Optional[str] = None,
    ) -> Optional[List[SED_Result]]:
        # cat_SED_results = [
        #     deepcopy(gal).aper_phot[aper_diam].SED_results[self.label]
        #     for gal in cat
        # ]
        if (
            lowz_zmax_arr is not None
            and self.SED_fit_params["lowz_zmax"] is None
        ):
            # update cat_SED_results with lowz_zmax info
            h5_path = self._get_out_paths(cat, aper_diam, save_name=save_name)[
                2
            ].replace(".fits", ".h5")
            # eazy-py's hdf5.initialize_from_hdf5() -> cat_from_hdf5()
            # reconstructs an identity translate table (rows "F5"->"F5",
            # "E5"->"E5", etc.) for catalogues that already use eazy's own
            # native F<n>/E<n> column naming, as GALFIND's saved catalogues
            # do. PhotoZ's read_catalog() then matches each such column
            # *twice* -- once via that redundant translate table, once via
            # its own native F<n> column-name recognition -- doubling
            # NFILT and producing a reloaded template grid shape
            # ((NZ, NTEMP, 2*NFILT)) that doesn't match the real one,
            # below. The native-column pass alone is correct and
            # sufficient here, so temporarily empty the translate table's
            # rows (an astropy Table, not a TranslateFile) that
            # cat_from_hdf5() hands back before initialize_from_hdf5()
            # builds the PhotoZ object with it.
            _orig_cat_from_hdf5 = hdf5.cat_from_hdf5

            def _cat_from_hdf5_no_dup_translate(h5file):
                cat_, trans_ = _orig_cat_from_hdf5(h5file)
                trans_.remove_rows(slice(None))
                # eazy's TranslateFile.__init__ does `tr["error"] = 1.0`
                # when "error" isn't already a column, which recent
                # astropy versions reject as a scalar assignment to a
                # zero-length table. Pre-empt that by adding the column
                # ourselves (also zero-length, so still a no-op).
                trans_["error"] = np.array([], dtype=float)
                return cat_, trans_

            hdf5.cat_from_hdf5 = _cat_from_hdf5_no_dup_translate
            try:
                fit = hdf5.initialize_from_hdf5(h5file=h5_path, verbose=False)
            finally:
                hdf5.cat_from_hdf5 = _orig_cat_from_hdf5
            lowz_zmax_arr = np.sort(lowz_zmax_arr)
            save_dict_arr = np.full(len(cat), deepcopy({}))
            zbest_arr = {}
            chi2_best_arr = {}
            for lowz_zmax in lowz_zmax_arr:
                if lowz_zmax > self.SED_fit_params["Z_MAX"]:
                    raise RangeError(
                        f"lowz_zmax={lowz_zmax} cannot be greater than "
                        f"SED_fit_params['Z_MAX']="
                        f"{self.SED_fit_params['Z_MAX']}."
                    )
                fit_copy = deepcopy(fit)
                zgrid_mask = np.array(
                    [
                        i
                        for i in range(len(fit_copy.zgrid))
                        if fit_copy.zgrid[i] <= lowz_zmax
                    ]
                )
                fit_copy.chi2_fit = fit_copy.chi2_fit[:, zgrid_mask]
                fit_copy.fit_coeffs = fit_copy.fit_coeffs[:, zgrid_mask, :]
                fit_copy.tef_lnp = fit_copy.tef_lnp[:, zgrid_mask]
                fit_copy.zgrid = fit_copy.zgrid[zgrid_mask]
                fit_copy.trdz = fit_copy.trdz[zgrid_mask]
                fit_copy.lnp = fit_copy.lnp[:, zgrid_mask]
                fit_copy.fit_at_zbest()
                idx = np.array(cat.ID) - 1
                zbest_arr[f"{lowz_zmax:.1f}"] = fit_copy.zbest[idx]
                chi2_best_arr[f"{lowz_zmax:.1f}"] = fit_copy.chi2_best[idx]

                # cat_SED_results = [
                #     deepcopy(gal).aper_phot[aper_diam]
                #     .SED_results[self.label] for gal in cat
                # ]
                # assert len(cat) == len(zbest_arr) == len(chi2_best_arr), \
                #     galfind_logger.critical(
                #         f"ARRAY LENGTH MISMATCH: {len(cat)=}, " +
                #         f"{len(zbest_arr)=}, {len(chi2_best_arr)=}"
                #     )
                # cat_SED_results = [
                #     deepcopy(gal).aper_phot[aper_diam]
                #     .SED_results[self.label].update_lowz_zmax_properties(
                #         f"{lowz_zmax:.1f}", {}
                #     ) for gal in cat
                # ]
                # cat_SED_results = np.full(len(cat), None)
            for i in range(len(cat)):
                # SED_result = deepcopy(
                #     cat[i].aper_phot[aper_diam].SED_results[self.label]
                # )
                save_dict = {
                    f"{lowz_zmax:.1f}": {
                        "zbest": zbest_arr[f"{lowz_zmax:.1f}"][i],
                        "chi2_best": chi2_best_arr[f"{lowz_zmax:.1f}"][i],
                    }
                    for lowz_zmax in lowz_zmax_arr
                }
                # }
                save_dict_arr[i] = save_dict  # .update(save_dict)
                # SED_result.update_lowz_zmax_properties(
                #     f"{lowz_zmax:.1f}", save_dict
                # )
                # cat_SED_results[i] = SED_result
                # cat_SED_result = deepcopy(cat[i]).aper_phot[aper_diam]
                # .SED_results[self.label]
                # cat_SED_results[i] = deepcopy(cat_SED_result)
            # cat.update_SED_results(cat_SED_results)
            cat.update_SED_result_lowz_zmax_info(
                aper_diam, self.label, save_dict_arr
            )
            # print(save_dict_arr)
            # for i, gal in enumerate(cat[:5]):
            # print(i, id(gal.aper_phot[aper_diam].SED_results[self.label]))
            # # cat_SED_results = [
            # #     gal.aper_phot[aper_diam].SED_results[self.label].\
            # #     update_lowz_zmax_properties(save_dict_arr[i])
            # #     for i, gal in enumerate(cat)
            # # ]
            # cat.update_SED_results(cat_SED_results)
            return [
                gal.aper_phot[aper_diam].SED_results[self.label] for gal in cat
            ]

    def _load_gal_property_labels(self):
        gal_property_labels = {"z": "zbest", "chi_sq": "chi2_best"}
        if self.SED_fit_params.get("SAVE_UBVJ", True):
            gal_property_labels.update(
                {
                    f"{ubvj_filt}_flux": f"{ubvj_filt}_rf_flux"
                    for ubvj_filt in ["U", "B", "V", "J"]
                }
            )
        super()._load_gal_property_labels(gal_property_labels)

    def _load_gal_property_err_labels(self):
        if self.SED_fit_params.get("SAVE_UBVJ", True):
            gal_property_err_labels = {
                f"{ubvj_filt}_flux": [
                    f"{ubvj_filt}_rf_flux_err",
                    f"{ubvj_filt}_rf_flux_err",
                ]
                for ubvj_filt in ["U", "B", "V", "J"]
            }
        else:
            gal_property_err_labels = {}
        super()._load_gal_property_err_labels(gal_property_err_labels)

    def _load_gal_property_units(self) -> NoReturn:
        self.gal_property_units = {
            gal_property: u.dimensionless_unscaled
            for gal_property in ["z", "chi_sq"]
        }
        if self.SED_fit_params.get("SAVE_UBVJ", True):
            self.gal_property_units.update(
                {
                    f"{ubvj_filt}_flux": u.nJy
                    for ubvj_filt in ["U", "B", "V", "J"]
                }
            )

    def _assert_SED_fit_params(self):
        default_strings = ["N_PROC", "Z_STEP", "Z_MIN", "Z_MAX", "SAVE_UBVJ"]
        default_types = [int, float, float, float, bool]
        for default_str, default_type in zip(default_strings, default_types):
            if default_str == "Z_MAX":
                if self.SED_fit_params["lowz_zmax"] is None:
                    self.SED_fit_params["Z_MAX"] = config.getfloat(
                        "EAZY", "Z_MAX"
                    )
                else:
                    self.SED_fit_params["Z_MAX"] = self.SED_fit_params[
                        "lowz_zmax"
                    ]
            else:
                if default_str not in self.SED_fit_params.keys():
                    if default_type is bool:
                        self.SED_fit_params[default_str] = config.getboolean(
                            "EAZY", default_str
                        )
                    elif default_type is int:
                        self.SED_fit_params[default_str] = config.getint(
                            "EAZY", default_str
                        )
                    elif default_type is float:
                        self.SED_fit_params[default_str] = config.getfloat(
                            "EAZY", default_str
                        )
        return super()._assert_SED_fit_params()

    def pre_fitting(
        self: Type[Self],
        cat: Catalogue,
        aper_diam: u.Quantity,
        overwrite: bool = False,
        save_name: Optional[str] = None,
    ) -> None:
        """Perform any pre-fitting setup required before running EAZY.

        Implements the `SED_code.pre_fitting` interface. Currently a
        no-op for `EAZY`.

        Parameters
        ----------
        cat : `Catalogue`
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

    def make_in(
        self,
        cat: Catalogue,
        aper_diam: u.Quantity,
        overwrite: bool = False,
        save_name: Optional[str] = None,
    ) -> str:
        """Build the EAZY ASCII input photometric catalogue and filter file.

        Implements the `SED_code.make_in` interface. Loads photometry for
        each galaxy in `cat`, writes the corresponding EAZY filter
        response file via `_make_filter_file`, and writes an ASCII
        catalogue of ID, flux/flux-error pairs per filter and
        spectroscopic redshift, in the format EAZY expects.

        Parameters
        ----------
        cat : `Catalogue`
            Catalogue of galaxies to build the input file for.
        aper_diam : `astropy.units.Quantity`
            Aperture diameter of the photometry to extract.
        overwrite : `bool`, optional
            Whether to remake the input file if one already exists. Default
            is `False`.
        save_name : `str`, optional
            Optional custom name to append when constructing the input
            file path. Default is `None`.

        Returns
        -------
        `str`
            Path to the (possibly newly written) EAZY ``.in`` input
            catalogue.
        """
        if save_name is None:
            save_name = ""
        else:
            save_name = f"_{save_name}"
        in_dir = (
            f"{config['EAZY']['EAZY_DIR']}/input/"
            f"{cat.filterset.instrument_name}/{cat.version}/{cat.survey}"
        )
        in_name = cat.cat_name.replace(
            ".fits", f"_{aper_diam.to(u.arcsec).value:.2f}as{save_name}.in"
        )
        in_path = f"{in_dir}/{in_name}"
        in_filt_name = f"{in_path.replace('.in', '_filters.RES')}"
        if not Path(in_path).is_file() or overwrite:
            # 1) obtain input data
            IDs = np.array([gal.ID for gal in cat.gals])  # load IDs
            redshifts = np.array(
                [-99.0 for i in range(len(cat))]
            )  # TODO: Load spec-z's
            # load photometry
            phot, phot_err = self._load_phot(
                cat, aper_diam, u.uJy, -99.0, None
            )

            funcs.make_dirs(in_path)

            # Make filter file
            filt_codes = self._make_filter_file(
                cat.filterset,
                in_filt_name,
                default_param_path=f"{config['EAZY']['EAZY_CONFIG_DIR']}/EAZY_UVJ.RES",
            )
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
                + list(np.full(len(cat.filterset.filt_names) * 2, float))
                + [float]
            )
            in_tab = Table(in_data, dtype=in_types, names=in_names)

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
        save_name: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ) -> NoReturn:
        """
        z_step  - redshift step size - default 0.01
        z_min - minimum redshift to fit - default 0
        z_max - maximum redshift to fit - default 25.
        save_SEDs - whether to write out best-fitting SEDs. Default True.
        save_PDFs - Whether to write out redshift PDF. Default True.
        save_plots - whether to save SED plots - default False. Use in
        conjunction with plot_ids to plot SEDS of specific ids.
        save_ubvj - whether to save restframe UBVJ fluxes -default True.
        **kwargs - additional arguments to pass to EAZY to overide defaults
        """
        # Change this to config file path
        # This if/else tree chooses which template file to use based
        # on 'templates' argument
        # FSPS - default EAZY templates, good allrounders
        # fsps_larson - default here, optimized for high redshift
        # (see Larson et al. 2023)
        # HOT_45K - modified IMF high-z templates for use between 8 < z < 12
        # HOT_60K - modified IMF high-z templates for use at z > 12
        # Nakajima - unobscured AGN templates

        in_path, out_path, fits_out_path, PDF_paths, SED_paths = (
            self._get_out_paths(
                cat,
                aper_diam,
                save_name=save_name,
            )
        )

        in_filt_path = f"{in_path.replace('.in', '_filters.RES')}"

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
        elif templates == "sfhz":
            params["TEMPLATES_FILE"] = (
                f"{eazy_templates_path}/sfhz/corr_sfhz_13_galfind.param"
            )
        elif templates == "sfhz+carnall_eelg":
            params["TEMPLATES_FILE"] = (
                f"{eazy_templates_path}/sfhz/carnall_sfhz_13_galfind.param"
            )
        elif templates == "sfhz_blue_agn":  # "sfhz+carnall_eelg+agn":
            params["TEMPLATES_FILE"] = (
                f"{eazy_templates_path}/sfhz/sorted_agn_blue_sfhz_13_galfind.param"
            )
        elif templates == "pegase":
            params["TEMPLATE_COMBOS"] = "1"
            params["TEMPLATES_FILE"] = (
                f"{eazy_templates_path}/pegase13.spectra_galfind.param"
            )

        params["TEMPLATES_FILE"] = _resolve_templates_file(
            params["TEMPLATES_FILE"], eazy_templates_path
        )

        # Redshift limits
        params["Z_MIN"] = z_min  # Setting minimum Z
        params["Z_MAX"] = z_max  # Setting maximum Z
        params["Z_STEP"] = self.SED_fit_params[
            "Z_STEP"
        ]  # Setting photo-z step

        # Next section deals with passing config parameters into EAZY
        # config dictionary
        # JWST filter_file
        params["FILTERS_RES"] = in_filt_path
        # f"{config['EAZY']['EAZY_CONFIG_DIR']}/jwst_nircam_FILTER.RES"

        # Errors
        params["WAVELENGTH_FILE"] = (
            f"{eazy_templates_path}/lambda.def"
            # Wavelength grid definition file
        )
        params["TEMP_ERR_FILE"] = (
            f"{eazy_templates_path}/TEMPLATE_ERROR.eazy_v1.0"
            # Template error definition file
        )

        # Priors
        # TODO: Load in and fix specific galaxies to spec-z's
        # params["FIX_ZSPEC"] = fix_z

        # Input files
        # ------------------------------------------------------------

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
            fit.fit_catalog(
                n_proc=self.SED_fit_params["N_PROC"], get_best_fit=True
            )
            # Save backup of fit in hdf5 file
            hdf5.write_hdf5(
                fit,
                h5file=h5_path,
                include_fit_coeffs=False,
                include_templates=True,
                verbose=False,
            )
            galfind_logger.info(
                f"Finished running {self.__class__.__name__} "
                f"{templates} {lowz_label}"
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
            # If not using Fsps larson, use standard saving output.
            # Otherwise generate own fits file.
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
                    # This is duplicated from base code.
                    # TODO: add n_proc option to rest_frame_fluxes
                    # function and use it here. n_proc != 0 spwans
                    # many threads
                    rf_tempfilt, lc_rest, ubvj = fit.rest_frame_fluxes(
                        f_numbers=[1, 2, 3, 4],
                        simple=False,
                        percentiles=[16, 50, 84],
                        n_proc=self.SED_fit_params["N_PROC"],  # 0
                    )
                    for i, ubvj_filt in enumerate(["U", "B", "V", "J"]):
                        table[f"{ubvj_filt}_rf_flux"] = ubvj[:, i, 1]
                        # symmetric errors
                        table[f"{ubvj_filt}_rf_flux_err"] = (
                            ubvj[:, i, 2] - ubvj[:, i, 0]
                        ) / 2.0
                    galfind_logger.info(
                        f"Finished calculating UBVJ fluxes for {repr(self)}"
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
                    f"Written {repr(self)} fits out file to: {fits_out_path}"
                )
        else:
            table = Table.read(fits_out_path)

        # save PDFs in .h5 file
        if save_PDFs and not Path(zPDF_path).is_file():
            self.save_zPDFs(zPDF_path, fit)
            galfind_logger.info(f"Finished saving z-PDFs for {repr(self)}")

        # Save best-fitting SEDs
        if save_SEDs and not Path(SED_path).is_file():
            z_arr = np.array(table[f"zbest_{templates}_{lowz_label}"]).astype(
                float
            )
            self.save_SEDs(SED_path, fit, z_arr, u.AA, u.nJy)
            galfind_logger.info(f"Finished saving SEDs for {repr(self)}")

        # Write used parameters
        if fit is not None:
            fit.param.write(fits_out_path.replace(".fits", "_params.csv"))
            funcs.change_file_permissions(
                fits_out_path.replace(".fits", "_params.csv")
            )
            galfind_logger.info(f"Written output pararmeters for {repr(self)}")

    @staticmethod
    def save_zPDFs(zPDF_path: str, fit) -> NoReturn:
        """Save per-galaxy redshift PDFs from an EAZY fit to an HDF5 file.

        Parameters
        ----------
        zPDF_path : `str`
            Output path for the HDF5 file of redshift PDFs.
        fit : `eazy.photoz.PhotoZ`
            Fitted EAZY `PhotoZ` object to extract redshift PDFs from.
        """
        fit_pz = 10 ** (fit.lnp)
        fit_zgrid = fit.zgrid
        hf = h5py.File(zPDF_path, "w")
        hf.create_dataset(
            "z",
            data=np.array(fit.zgrid).astype(np.float32),
            compression="gzip",
            dtype="f4",
        )
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
                    disable=galfind_logger.getEffectiveLevel() > logging.INFO,
                    desc="Saving z-PDFs",
                )
            ]
        )
        hf.create_dataset("p_z_arr", data=pz_arr, compression="gzip")
        hf.close()

    @staticmethod
    def save_SEDs(
        SED_path: str,
        fit,
        z_arr: List[float],
        wav_unit: u.Unit = u.AA,
        flux_unit: u.Unit = u.nJy,
    ) -> NoReturn:
        """Save per-galaxy best-fit template SEDs from an EAZY fit to
        an HDF5 file.

        Parameters
        ----------
        SED_path : `str`
            Output path for the HDF5 file of best-fit SEDs.
        fit : `eazy.photoz.PhotoZ`
            Fitted EAZY `PhotoZ` object to extract best-fit SEDs from.
        z_arr : `list` of `float`
            Best-fit redshift for each galaxy in `fit.OBJID`, stored
            alongside the SEDs.
        wav_unit : `astropy.units.Unit`, optional
            Unit to convert best-fit SED wavelengths to before saving.
            Default is `astropy.units.AA`.
        flux_unit : `astropy.units.Unit`, optional
            Unit to convert best-fit SED fluxes to before saving. Default
            is `astropy.units.nJy`.
        """
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
                disable=galfind_logger.getEffectiveLevel() > logging.INFO,
            )
        ]
        wav_flux_arr = np.array(wav_flux_arr).astype(np.float32)
        hf.create_dataset(
            "wav_flux_arr", data=wav_flux_arr, compression="gzip", dtype="f4"
        )
        hf.close()

    def make_fits_from_out(
        self, out_path: str, **kwargs: Dict[str, Any]
    ) -> NoReturn:
        """Convert the raw output of the fit into a FITS binary table.

        Implements the `SED_code.make_fits_from_out` interface. Currently
        a no-op for `EAZY`, since `fit` already writes the FITS output
        table directly.

        Parameters
        ----------
        out_path : `str`
            Path to the output catalogue produced by `fit`.
        **kwargs : `dict`
            Accepted for interface compatibility with
            `SED_code.make_fits_from_out`; unused.
        """
        pass

    def _get_out_paths(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity,
        save_name: Optional[str] = None,
    ) -> Tuple[str, str, str, Dict[str, List[str]], List[str]]:
        if save_name is None:
            save_name = ""
        else:
            save_name = f"_{save_name}"
        in_dir = (
            f"{config['EAZY']['EAZY_DIR']}/input/"
            f"{cat.filterset.instrument_name}/{cat.version}/{cat.survey}"
        )
        in_name = cat.cat_name.replace(
            ".fits", f"_{aper_diam.to(u.arcsec).value:.2f}as{save_name}.in"
        )
        in_path = f"{in_dir}/{in_name}"

        out_folder = funcs.split_dir_name(
            in_path.replace("input", "output"), "dir"
        )
        out_path = (
            f"{out_folder}/"
            f"{funcs.split_dir_name(in_path, 'name').replace('.in', '.out')}"
        )
        fits_out_path = (
            f"{out_path.replace('.out', '')}_EAZY_"
            f"{self.SED_fit_params['templates']}_"
            f"{funcs.lowz_label(self.SED_fit_params['lowz_zmax'])}.fits"
        )
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
        SED_paths: Union[str, List[str]],
        *args,
        **kwargs,
    ) -> List[SED_obs]:
        """Extract best-fit SEDs from the EAZY-generated HDF5 SED file.

        Implements the `SED_code.extract_SEDs` interface.

        Parameters
        ----------
        IDs : `list` of `int`
            Galaxy IDs to extract SEDs for.
        SED_paths : `str` or `list` of `str`
            Path(s) to the HDF5 (``.h5``) file containing best-fit SEDs
            for the parent catalogue, as returned by `_get_out_paths`.
            All entries must be identical, since EAZY stores all
            galaxies' SEDs for a run in a single file.
        *args : `tuple`
            Unused; accepted for interface compatibility.
        **kwargs : `dict`
            Unused; accepted for interface compatibility.

        Returns
        -------
        `list` of `SED_obs`
            Best-fit galaxy SED for each requested ID, in the same order
            as `IDs`.

        Raises
        ------
        LengthMismatchError
            If `IDs` and `SED_paths` have different lengths.
        IncompatibleKwargsError
            If the elements of `SED_paths` are not all identical.
        """
        # ensure this works if only extracting 1 galaxy
        if isinstance(IDs, (str, int, float)):
            IDs = np.array([int(IDs)])
        if isinstance(SED_paths, str):
            SED_paths = [SED_paths]
        if len(IDs) != len(SED_paths):
            raise LengthMismatchError(
                f"len(IDs)={len(IDs)} != len(SED_paths)={len(SED_paths)}; "
                "IDs and SED_paths must have the same length."
            )
        # ensure that for EAZY all the SED_paths are the same
        if not all(SED_path == SED_paths[0] for SED_path in SED_paths):
            raise IncompatibleKwargsError(
                f"SED_paths={SED_paths!r} are not all identical; "
                f"{self.__class__.__name__} stores every galaxy's best-fit "
                "SED in a single per-catalogue .h5 file, so all SED_paths "
                "entries must be the same path."
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
        # close .h5 file
        SED_obs_arr = [
            SED_obs(z, wav, flux, wav_unit, flux_unit)
            for z, wav, flux in tqdm(
                zip(z_arr, wav_arr, flux_arr),
                total=len(z_arr),
                desc="Constructing SEDs",
                disable=galfind_logger.getEffectiveLevel() > logging.INFO,
            )
        ]
        return SED_obs_arr

    def extract_PDFs(
        self: Self,
        gal_property: str,
        IDs: List[int],
        PDF_paths: Union[str, List[str]],
    ) -> List[Redshift_PDF]:
        """Extract posterior redshift PDFs from EAZY output files.

        Implements the `SED_code.extract_PDFs` interface. EAZY only
        stores a PDF for redshift, so requesting any other property
        returns an array of `None`.

        Parameters
        ----------
        gal_property : `str`
            Name of the galaxy property to extract PDFs for. Only
            ``"z"`` yields non-`None` PDFs.
        IDs : `list` of `int`
            Galaxy IDs to extract PDFs for.
        PDF_paths : `str` or `list` of `str`
            Path(s) to the HDF5 (``.h5``) file containing redshift PDFs
            for the parent catalogue. All entries must be identical,
            since EAZY stores all galaxies' PDFs for a run in a single
            file.

        Returns
        -------
        `list`
            List of `Redshift_PDF` objects (one per ID) if
            `gal_property` is ``"z"``, otherwise a list of `None` of the
            same length.

        Raises
        ------
        GalfindTypeError
            If `PDF_paths` or `IDs` is not a `list`/`numpy.ndarray`, when
            `gal_property` is ``"z"``.
        LengthMismatchError
            If `PDF_paths` and `IDs` differ in length, when
            `gal_property` is ``"z"``.
        IncompatibleKwargsError
            If the elements of `PDF_paths` are not all identical, when
            `gal_property` is ``"z"``.
        MissingFileError
            If `PDF_paths[0]` does not have a ``.h5`` extension, when
            `gal_property` is ``"z"``.
        """
        from ..visualization import Redshift_PDF

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
            if not isinstance(PDF_paths, (list, np.ndarray)):
                raise GalfindTypeError(
                    f"PDF_paths has type {type(PDF_paths)}; must be a "
                    "list or numpy.ndarray."
                )
            if not isinstance(IDs, (list, np.ndarray)):
                raise GalfindTypeError(
                    f"IDs has type {type(IDs)}; must be a list or "
                    "numpy.ndarray."
                )
            # ensure the correct array size
            if len(IDs) != len(PDF_paths):
                raise LengthMismatchError(
                    f"len(IDs)={len(IDs)} != len(PDF_paths)="
                    f"{len(PDF_paths)}; IDs and PDF_paths must have the "
                    "same length."
                )
            # ensure all data_paths are the same and are of .h5 type
            if not all(PDF_path == PDF_paths[0] for PDF_path in PDF_paths):
                raise IncompatibleKwargsError(
                    f"PDF_paths={PDF_paths!r} are not all identical; "
                    f"{self.__class__.__name__} stores every galaxy's "
                    "redshift PDF in a single per-catalogue .h5 file, so "
                    "all PDF_paths entries must be the same path."
                )
            if PDF_paths[0][-3:] != ".h5":
                raise MissingFileError(
                    f"PDF_paths[0]={PDF_paths[0]!r} does not have a "
                    "'.h5' file extension; EAZY redshift PDFs must be "
                    "read from an HDF5 (.h5) output file."
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
                    disable=galfind_logger.getEffectiveLevel() > logging.INFO,
                )
            ]
            # close .h5 file
            hf.close()
            return redshift_PDFs

    def load_cat_property_PDFs(
        self: Self, PDF_paths: List[Dict[str, str]], IDs: List[int]
    ) -> List[Dict[str, Optional[Type[PDF]]]]:
        """Load per-galaxy property PDFs from a set of PDF file paths.

        Implements the `SED_code.load_cat_property_PDFs` interface. Calls
        `extract_PDFs` for each requested property and reorganises the
        result into one dictionary of non-`None` PDFs per galaxy.

        Parameters
        ----------
        PDF_paths : `dict`
            Mapping from galaxy property name to the PDF file path(s)
            for that property, as returned by `_get_out_paths`.
        IDs : `list` of `int`
            Galaxy IDs to load PDFs for.

        Returns
        -------
        `list` of `dict`
            One dictionary per galaxy (in the order of `IDs`), mapping
            property name to its PDF object. `None` is used in place of
            a dictionary for galaxies with no available PDFs.
        """
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

    def _make_filter_file(
        self: Self,
        filterset: Multiple_Filter,
        filter_file: str,
        default_param_path: str,
    ) -> NoReturn:
        """
        Write a filter file for EAZY from a Multiple_Filter object

        """
        # Need to write a filterset file for EAZY
        # Two files one with list of filters and one with the transmission
        # curves
        # format of list is

        # 1 len(transmission_curve) name1 lambda_c = pivot_wav
        # 2 len(transmission_curve) name2 lambda_c = pivot_wav

        # format of transmission curve is
        # len(transmission_curve) name1 lambda_c = pivot_wav
        # 1 0.1 0.0
        # 2 0.2 0.1
        # 3 0.3 0.2
        # ...
        # len(transmission_curve) name1 lambda_c = pivot_wav
        # 1 0.1 0.0
        # 2 0.2 0.1

        # copy default filter file and append to it

        copy(default_param_path, filter_file)
        copy(f"{default_param_path}.INFO", f"{filter_file}.INFO")
        # count lines in .INFO

        with open(f"{filter_file}.INFO", "r") as f:
            current_lines = f.readlines()
            nexisting = len(current_lines)
            last_line = current_lines[-1]

        with open(filter_file, "a") as f:
            with open(f"{filter_file}.INFO", "a") as f_info:
                # work out whether we need to move to the next line -
                # i.e is the current line got anything in it

                if not last_line.endswith("\n"):
                    f_info.write("\n")

                f.write("\n")

                # count lines in file
                for i, filt in enumerate(filterset):
                    code = i + nexisting + 1
                    wav_cent = (
                        filt.properties["WavelengthEff"].to(u.Angstrom).value
                    )
                    f_info.write(
                        f"{code}  {len(filt.trans)} "
                        f"{filt.facility_name}/{filt.instrument_name}."
                        f"{filt.filt_name} lambda_c= {wav_cent}\n"
                    )
                    f.write(
                        f" {len(filt.trans)} "
                        f"{filt.facility_name}/{filt.instrument_name}."
                        f"{filt.filt_name} lambda_c= {wav_cent}\n"
                    )

                    for pos, (wav, trans) in enumerate(
                        zip(filt.wav, filt.trans)
                    ):
                        f.write(
                            f"{pos + 1} {wav.to(u.Angstrom).value} {trans}\n"
                        )

        return np.arange(nexisting + 1, nexisting + 1 + len(filterset))

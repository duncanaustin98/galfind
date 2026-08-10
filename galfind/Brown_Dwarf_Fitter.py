
from __future__ import annotations

import numpy as np
import astropy.units as u
import h5py
from astropy.table import Table
from pathlib import Path
import itertools
from typing import TYPE_CHECKING, Dict, List, NoReturn, Union
if TYPE_CHECKING:
    from . import Catalogue
try:
    from typing import Self, Type, Any  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import useful_funcs_austind as funcs
from . import config, galfind_logger
from . import SED_code
from .SED import SED_obs

class Template_Fitter(SED_code):
    """`SED_code` wrapper around external stellar/brown-dwarf template fitting (`BDFit.StarFit`).

    Fits observed photometry against pre-computed template grids (e.g.
    Sonora atmosphere models) using the external ``BDFit`` package's
    ``StarFit`` fitter, and parses the results back into GALFIND-native
    tables and SEDs.

    Parameters
    ----------
    SED_fit_params : `dict`
        Dictionary of SED fitting parameters/options for this run. Must
        contain the key ``"templates"``, a list of template library names
        to fit against. Passed on to `SED_code.__init__`.
    fit_instrument : `str`, optional
        Name of the instrument whose bands are used for fitting. Default
        is ``"NIRCam"``.

    Attributes
    ----------
    fit_instrument : `str`
        Name of the instrument whose bands are used for fitting.
    SED_fit_params : `dict`
        SED fitting parameters/options, set by `SED_code.__init__`.
    """

    ID_label = "ID"

    def __init__(
        self: Self,
        SED_fit_params: Dict[str, Any],
        fit_instrument: str = "NIRCam",
    ):
        """Initialize a Template_Fitter instance.

        Parameters
        ----------
        SED_fit_params : `dict`
            Dictionary of SED fitting parameters/options. Must contain
            the key ``"templates"``.
        fit_instrument : `str`, optional
            Name of the instrument whose bands are used for fitting.
            Default is ``"NIRCam"``.
        """
        self.fit_instrument = fit_instrument
        super().__init__(SED_fit_params)

    @classmethod
    def from_label(cls, label: str) -> Type[SED_code]:
        """Construct a `Template_Fitter` instance from a saved catalogue label.

        Parameters
        ----------
        label : `str`
            Label string identifying a previous fitting run.

        Raises
        ------
        NotImplementedError
            Always; this method is not yet implemented for `Template_Fitter`.
        """
        raise NotImplementedError()

    # @property
    # def ID_label(self) -> str:
    #     return "ID"

    @property
    def label(self) -> str:
        """`str`: Unique label for this fitting run, implementing the `SED_code.label` interface.

        Combines the class name with `tab_suffix`.
        """
        return f"{self.__class__.__name__}_{self.tab_suffix}"

    @property
    def hdu_name(self) -> str:
        """`str`: Name of the FITS HDU/extension this code's output is stored under.

        Implements the `SED_code.hdu_name` interface. Combines the class
        name with `tab_suffix`.
        """
        return f"{self.__class__.__name__}_{self.tab_suffix}"

    @property
    def tab_suffix(self) -> str:
        """`str`: Column-name suffix used to distinguish this run's output columns.

        Implements the `SED_code.tab_suffix` interface. A ``"+"``-joined
        string of the requested template library names
        (``SED_fit_params["templates"]``) if more than one is given,
        otherwise the single template library name.
        """
        if len(self.SED_fit_params["templates"]) > 1:
            return "+".join(self.SED_fit_params["templates"])
        else:
            return self.SED_fit_params["templates"][0]

    @property
    def required_SED_fit_params(self) -> List[str]:
        """`list` of `str`: Names of the `SED_fit_params` keys required by `Template_Fitter`.

        Implements the `SED_code.required_SED_fit_params` interface.
        """
        return ["templates"]

    @property
    def are_errs_percentiles(self) -> bool:
        """`bool`: Whether output property errors are stored as percentiles rather than 1-sigma values.

        Implements the `SED_code.are_errs_percentiles` interface.
        """
        return False # not sure here

    # BELOW TWO METHODS SHOULD BE HADNLED BETTER IN SED_code

    #@abstractmethod
    def _load_gal_property_labels(self) -> NoReturn:
        """Load mappings from internal property names to output column names.

        Implements the `SED_code._load_gal_property_labels` interface.
        """
        self.gal_property_labels = {
            "chi_sq": "chi2",
            "red_chi_sq": "red_chi2",
            "temp": "temp",
            "log_g": "log_g",
            "kzz": "kzz",
            #"met": "met",
            #"co": "co",
            #"f": "f",
        }
        #super()._load_gal_property_labels(gal_property_labels)

    #@abstractmethod
    def _load_gal_property_err_labels(self) -> NoReturn:
        """Load mappings for property error column names.

        Implements the `SED_code._load_gal_property_err_labels` interface.
        Sets empty dict since this fitter does not provide error estimates.
        """
        super()._load_gal_property_err_labels({})

    def _load_gal_property_units(self) -> NoReturn:
        """Load the astropy units for each fitted property.

        Implements the `SED_code._load_gal_property_units` interface.
        """
        self.gal_property_units = {
            **{gal_property: u.dimensionless_unscaled for gal_property in ["chi_sq", "red_chi_sq"]},
            "temp": u.K,
            "log_g": u.dimensionless_unscaled,
            "kzz": u.dimensionless_unscaled,
        }
        # , "met", "co", "f"

    def make_in(
        self,
        cat: Catalogue,
        aper_diam: u.Quantity,
        overwrite: bool = False
    ) -> str:
        """Build the input photometric catalogue required by the fit.

        Implements the `SED_code.make_in` interface. Currently a no-op for
        `Template_Fitter`; photometry is instead loaded directly within
        `fit` via `SED_code._load_phot`.

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
        `None`
            This no-op implementation returns `None`, despite the
            annotated `str` return type.
        """
        pass

    def fit(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity,
        save_SEDs: bool = True,
        save_PDFs: bool = True,
        overwrite: bool = False,
        verbose: bool = True,
        **kwargs: Dict[str, Any],
    ) -> NoReturn:
        """Fit a catalogue's photometry against template libraries using `BDFit.StarFit`.

        Implements the `SED_code.fit` interface. Selects the bands
        belonging to `self.fit_instrument`, builds a `StarFit` fitter for
        the requested template libraries (``SED_fit_params["templates"]``),
        fits the catalogue's photometry, and writes the resulting best-fit
        table (and, if `save_SEDs` is `True`, best-fit SEDs) to disk.
        Skips fitting entirely if the output FITS file already exists and
        `overwrite` is `False`.

        Parameters
        ----------
        cat : `Catalogue`
            Catalogue of galaxies to fit.
        aper_diam : `astropy.units.Quantity`
            Aperture diameter of the photometry being fitted.
        save_SEDs : `bool`, optional
            Whether to save the best-fit SEDs to an accompanying ``.h5``
            file. Default is `True`.
        save_PDFs : `bool`, optional
            Currently unused within this method body; kept for interface
            compatibility. Default is `True`.
        overwrite : `bool`, optional
            Whether to re-run the fit even if the output file already
            exists. Default is `False`.
        verbose : `bool`, optional
            Whether `StarFit` should print verbose fitting output. Default
            is `True`.
        **kwargs : `dict`
            Additional keyword arguments. Currently unused.

        Raises
        ------
        AssertionError
            If `cat.filterset` contains no filters with
            ``instrument_name == self.fit_instrument``.
        """
        from BDFit import StarFit
        out_path = self._get_out_paths(cat, aper_diam)[1]
        if not Path(out_path).is_file() or overwrite:
            # convert cat.filterset to bands used in StarFit
            fit_instrument_indices = np.array([i for i, filt in enumerate(cat.filterset) if filt.instrument_name == self.fit_instrument])
            fit_instrument_filterset = cat.filterset[fit_instrument_indices]
            assert len(fit_instrument_filterset) > 0, \
                galfind_logger.critical(
                    f"No filters in cat.filterset with instrument_name = {self.fit_instrument=}"
                )
            facilities_to_search = {}
            for filt in fit_instrument_filterset:
                if filt.facility_name not in facilities_to_search.keys():
                    facilities_to_search[filt.facility_name] = []
                facilities_to_search[filt.facility_name].append(filt.instrument.SVO_name)
            bands = [
                f"{filt.instrument_name}.{filt.filt_name}" 
                if filt.instrument_name != "NIRCam" else filt.filt_name 
                for filt in fit_instrument_filterset
            ]
            starfit = StarFit(
                facilities_to_search = facilities_to_search,
                libraries = self.SED_fit_params["templates"],
                compile_bands = bands,
                verbose = verbose,
            )
            starfit.fit_catalog(
                photometry_function = self._load_phot,
                bands = bands,
                photometry_function_kwargs = {
                    "cat": cat,
                    "aper_diam": aper_diam,
                    "out_units": u.nJy,
                    "no_data_val": np.nan,
                    "incl_units": True,
                    "input_filterset": fit_instrument_filterset,
                },
                sys_err = None,
                filter_mask = None,
                subset = None,
            )
            #self.starfit = starfit
            tab = starfit.make_cat()
            tab["ID"] = np.array(cat.ID)
            # save as a .fits file
            funcs.make_dirs(out_path)
            tab.write(out_path, overwrite = overwrite)
            galfind_logger.info(f"Saved {self.tab_suffix} fits to {out_path}")
            # save the best fitting SEDs
            if save_SEDs:
                SED_save_path = out_path.replace(".fits", "_SEDs.h5")
                starfit.make_best_fit_SEDs(SED_save_path, IDs = cat.ID)
                galfind_logger.info(f"Saved {self.tab_suffix} SEDs to {SED_save_path}")

    def make_fits_from_out(self, out_path):
        """Convert the raw output of the fit into a FITS binary table.

        Implements the `SED_code.make_fits_from_out` interface. Currently
        a no-op for `Template_Fitter`, since `fit` already writes the
        output table directly to `out_path`.

        Parameters
        ----------
        out_path : `str`
            Path to the output catalogue produced by `fit`.
        """
        pass

    def _get_out_paths(
        self: Self,
        cat: Catalogue,
        aper_diam: u.Quantity
    ) -> Tuple[str, str, str, Dict[str, List[str]], List[str]]:
        """Construct output file paths for this fitting run.

        Implements the `SED_code._get_out_paths` interface.

        Parameters
        ----------
        cat : `Catalogue`
            Catalogue being fit.
        aper_diam : `astropy.units.Quantity`
            Aperture diameter used in the fit.

        Returns
        -------
        `tuple`
            ``(None, out_path, out_path, None, SED_path)`` where ``out_path``
            is the FITS catalogue output file and ``SED_path`` is the HDF5 SED file.
        """
        aper_diams_str = funcs.aper_diams_to_str(np.array([aper_diam.to(u.arcsec).value]) * u.arcsec)
        out_path = f"{config['TemplateFitting']['BROWN_DWARF_OUT_DIR']}/{cat.version}/" + \
            f"{cat.filterset.instrument_name}/{cat.survey}/{aper_diams_str}/" + \
            f"{self.hdu_name}_{self.fit_instrument}.fits"
        SED_path = out_path.replace(".fits", "_SEDs.h5")
        return None, out_path, out_path, None, SED_path

    def extract_SEDs(
        self: Self,
        IDs: List[int],
        SED_paths: str,
        *args,
        **kwargs,
    ) -> List[SED_obs]:
        """Extract best-fit SEDs from the `StarFit`-generated HDF5 SED file.

        Implements the `SED_code.extract_SEDs` interface. Reads each
        template library group from the HDF5 file and matches stored
        galaxy IDs against `IDs`.

        Parameters
        ----------
        IDs : `list` of `int`
            Galaxy IDs to extract SEDs for.
        SED_paths : `str`
            Path to the HDF5 (``.h5``) file containing best-fit SEDs for
            all fitted template libraries, as produced by `fit`.
        *args : `tuple`
            Unused; accepted for interface compatibility.
        **kwargs : `dict`
            Unused; accepted for interface compatibility.

        Returns
        -------
        `list` of `SED_obs` or `None`
            Best-fit galaxy SED for each requested ID, in the same order
            as `IDs`. `None` is returned in place of an `SED_obs` for any
            ID not found in `SED_paths`.
        """
        # open .h5 table
        SED_h5 = h5py.File(SED_paths, "r")
        libraries = list(SED_h5.keys())
        # loop over each library, extracting SEDs for relevant IDs
        sed_obs_arr = {}
        for library in libraries:
            lib_IDs = SED_h5[library]["IDs"][:]
            wavs = SED_h5[library]["wavs"][:]
            fluxes_arr = SED_h5[library]["fluxes"][:]
            wav_unit = u.Unit(SED_h5[library].attrs["wav_unit"])
            flux_unit = u.Unit(SED_h5[library].attrs["flux_unit"])
            sed_obs_arr_ = {ID: SED_obs(0.0, wavs, fluxes, wav_unit, flux_unit)
                for ID, fluxes in zip(lib_IDs, fluxes_arr) if ID in IDs}
            sed_obs_arr.update(sed_obs_arr_)
        return [sed_obs_arr[ID] if ID in sed_obs_arr.keys() else None for ID in IDs]

    def extract_PDFs(
        self: Self,
        gal_property: str,
        IDs: List[int],
        PDF_paths: Union[str, List[str]],
    ) -> Optional[List[Type[PDF]]]:
        """Extract posterior PDFs for a galaxy property.

        Implements the `SED_code.extract_PDFs` interface. Currently a
        no-op for `Template_Fitter`, since `StarFit` does not produce
        property PDFs; always returns `None`.

        Parameters
        ----------
        gal_property : `str`
            Name of the galaxy property to extract PDFs for.
        IDs : `list` of `int`
            Galaxy IDs to extract PDFs for.
        PDF_paths : `str` or `list` of `str`
            Path(s) to PDF file(s).

        Returns
        -------
        `None`
            No PDFs are available for `Template_Fitter`.
        """
        pass

    def load_cat_property_PDFs(
        self: Self,
        PDF_paths: Union[List[str], List[Dict[str, str]]],
        IDs: List[int]
    ) -> List[Dict[str, Optional[Type[PDF]]]]:
        """Load per-galaxy property PDFs.

        Implements the `SED_code.load_cat_property_PDFs` interface.
        `Template_Fitter` does not produce property PDFs, so this simply
        returns an array of `None`, one per galaxy.

        Parameters
        ----------
        PDF_paths : `list` of `str` or `list` of `dict`
            Unused; accepted for interface compatibility.
        IDs : `list` of `int`
            Galaxy IDs to load PDFs for.

        Returns
        -------
        `numpy.ndarray`
            Array of `None`, one per galaxy in `IDs`.
        """
        return np.array(
            list(itertools.repeat(None, len(IDs)))
        )

    def _assert_SED_fit_params(self) -> NoReturn:
        """Validate that required SED fitting parameters are present.

        Implements the `SED_code._assert_SED_fit_params` interface. Checks
        that all required keys are present in `SED_fit_params` and initializes
        ``excl_bands`` if not already present.
        """
        for key in self.required_SED_fit_params:
            assert key in self.SED_fit_params.keys(), galfind_logger.critical(
                f"'{key}' not in SED_fit_params keys = {list(self.SED_fit_params.keys())}"
            )
        if "excl_bands" not in self.SED_fit_params.keys():
            self.SED_fit_params["excl_bands"] = []


class Brown_Dwarf_Fitter(Template_Fitter):
    """`Template_Fitter` preconfigured to fit the Sonora brown-dwarf template grids.

    Fixes ``SED_fit_params["templates"]`` to the full set of Sonora
    atmosphere model libraries (``sonora_bobcat``, ``sonora_cholla``,
    ``sonora_elf_owl``, ``sonora_diamondback``) plus a ``"low-z"`` library.

    Parameters
    ----------
    fit_instrument : `str`, optional
        Name of the instrument whose bands are used for fitting. Default
        is ``"NIRCam"``.

    Attributes
    ----------
    fit_instrument : `str`
        Name of the instrument whose bands are used for fitting.
    SED_fit_params : `dict`
        SED fitting parameters/options, fixed to the Sonora + ``"low-z"``
        template libraries.
    """

    def __init__(
        self: Self,
        fit_instrument: str = "NIRCam"
    ):
        """Initialize a Brown_Dwarf_Fitter with Sonora template libraries.

        Parameters
        ----------
        fit_instrument : `str`, optional
            Name of the instrument whose bands are used for fitting.
            Default is ``"NIRCam"``.
        """
        SED_fit_params = {"templates": ["sonora_bobcat", "sonora_cholla", "sonora_elf_owl", "sonora_diamondback", "low-z"]}
        super().__init__(SED_fit_params, fit_instrument)

    @property
    def label(self) -> str:
        """`str`: Unique label for this fitting run, implementing the `SED_code.label` interface.

        Overrides `Template_Fitter.label`, always returning ``"bd"``.
        """
        return "bd"
        #return self.__class__.__name__

    @property
    def hdu_name(self) -> str:
        """`str`: Name of the FITS HDU/extension this code's output is stored under.

        Implements the `SED_code.hdu_name` interface. Overrides
        `Template_Fitter.hdu_name`, returning the class name alone.
        """
        return self.__class__.__name__

    @property
    def tab_suffix(self) -> str:
        """`str`: Column-name suffix used to distinguish this run's output columns.

        Implements the `SED_code.tab_suffix` interface. Overrides
        `Template_Fitter.tab_suffix`, always returning an empty string.
        """
        return ""
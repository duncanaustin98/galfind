"""Spectral data and spectroscopic instrument configuration.

Provides classes for handling spectral data and NIRSpec grating configuration,
including dispersion, resolution, and transmission curves for named gratings.
"""

from __future__ import annotations

import csv
import os
from abc import ABC, abstractmethod
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

from astropy.wcs import WCS
from scipy.optimize import curve_fit

try:
    from typing import Self  # , Type  # python 3.11+
except ImportError:
    from typing_extensions import (
        Self,  # , Type  # python > 3.7 AND python < 3.11
    )

if TYPE_CHECKING:
    from ..imaging import Multiple_Filter
    from ..visualization import PDF
import logging
from copy import deepcopy

import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.utils.masked import Masked
from lmfit import Parameters, fit_report, minimize
from numpy.typing import NDArray
from tqdm import tqdm

from .. import astropy_cosmo as cosmo
from .. import config, galfind_logger
from ..utils import useful_funcs_austind as funcs
from ..utils.exceptions import (
    GalfindError,
    GalfindTypeError,
    InvalidOptionError,
    InvalidUnitError,
    LengthMismatchError,
    MissingDataError,
    MissingKeyError,
    RangeError,
)


class Spectral_Grating:  # disperser
    """NIRSpec spectral grating (disperser) configuration.

    On construction, loads the dispersion, resolution and transmission
    curves associated with the named grating.

    Parameters
    ----------
    name : `str`
        Name of the grating (e.g. ``"PRISM"``, ``"G140M"``, ``"G395H"``).

    Attributes
    ----------
    name : `str`
        Name of the grating.
    nominal_resolution : `float`
        Nominal spectral resolution, set by `load_resolution_curve`.
    resolution_curve_path : `str`
        Path to the FITS file containing the resolution curve, set by
        `load_resolution_curve`.
    """

    def __init__(self, name: str) -> NoReturn:
        self.name = name
        self.load_dispersion_curve()
        self.load_resolution_curve()
        self.load_transmission_curve()

    def __repr__(self) -> str:
        """Return a string representation of the `Spectral_Grating`
        instance."""
        return f"Spectral_Grating({self.name})"

    def load_dispersion_curve(self):
        """Load the dispersion curve for this grating. Not yet implemented."""
        pass

    def get_dispersion(self, wavs):
        """Return the dispersion at the given wavelength(
            s). Not yet implemented.

        Parameters
        ----------
        wavs : array-like
            Wavelength(s) at which to evaluate the dispersion.
        """
        pass

    def load_resolution_curve(self):
        """Set the nominal spectral resolution and resolution curve path
        for this grating.

        The nominal resolution is set to 100 for the ``"PRISM"`` grating,
        1000 for medium-resolution gratings (name ending in ``"M"``), and
        2700 otherwise (high-resolution gratings). Also sets
        `resolution_curve_path` to the corresponding resolution curve FITS
        file under ``config['Spectra']['R_CURVE_DIR']``.
        """
        self.nominal_resolution = (
            100.0
            if self.name == "PRISM"
            else 1_000.0
            if self.name[-1] == "M"
            else 2_700.0
        )
        self.resolution_curve_path = (
            f"{config['Spectra']['R_CURVE_DIR']}"
            "/NIRSpec/jwst_nirspec_prism_disp.fits"
        )

    def get_resolution(self, wavs):
        """Return the spectral resolution at the given wavelength(s). Not
        yet implemented.

        Parameters
        ----------
        wavs : array-like
            Wavelength(s) at which to evaluate the resolution.
        """
        pass

    def load_transmission_curve(self):
        """Load the transmission curve for this grating. Not yet
        implemented."""
        pass

    def get_transmission(self, wavs):
        """Return the transmission at the given wavelength(
            s). Not yet implemented.

        Parameters
        ----------
        wavs : array-like
            Wavelength(s) at which to evaluate the transmission.
        """
        pass


class Spectral_Filter:
    """NIRSpec blocking filter configuration used alongside a grating.

    Parameters
    ----------
    name : `str`
        Name of the filter (e.g. ``"CLEAR"``, ``"F070LP"``).

    Attributes
    ----------
    name : `str`
        Name of the filter.
    """

    def __init__(self, name: str) -> NoReturn:
        self.name = name
        self.load_transmission_curve()

    def __repr__(self) -> str:
        """Return a string representation of the `Spectral_Filter` instance."""
        return f"Spectral_Filter({self.name})"

    def load_transmission_curve(self):
        """Load the transmission curve for this filter. Not yet implemented."""
        pass

    def get_transmission(self, wavs):
        """Return the transmission at the given wavelength(
            s). Not yet implemented.

        Parameters
        ----------
        wavs : array-like
            Wavelength(s) at which to evaluate the transmission.
        """
        pass


class Spectral_Instrument(ABC):
    """Abstract base class for a spectrograph configuration.

    Combines a `Spectral_Grating` and a `Spectral_Filter` into a single
    instrument configuration.

    Parameters
    ----------
    grating : `Spectral_Grating`
        The grating (disperser) used by this instrument configuration.
    filter : `Spectral_Filter`
        The blocking filter used by this instrument configuration.

    Attributes
    ----------
    grating : `Spectral_Grating`
        The grating associated with this instrument configuration.
    filter : `Spectral_Filter`
        The filter associated with this instrument configuration.
    """

    def __init__(
        self,
        grating: Spectral_Grating,
        filter: Spectral_Filter,
    ) -> None:
        self.grating = grating
        self.filter = filter

    @abstractmethod
    def load_sensitivity(self):
        """Load the sensitivity curve for this instrument configuration.
        Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_sensitivity(self):
        """Return the sensitivity of this instrument configuration. Must
        be implemented by subclasses."""
        pass


# average_resolution: u.Quantity,
# wavelengths: u.Quantity,
# sensitivity: Callable[..., u.Quantity],


class NIRSpec(Spectral_Instrument):
    """JWST NIRSpec instrument configuration (grating + filter pair).

    Parameters
    ----------
    grating_name : `str`
        Name of the NIRSpec grating (e.g. ``"PRISM"``, ``"G395H"``).
    filter_name : `str`
        Name of the NIRSpec blocking filter (e.g. ``"CLEAR"``, ``"F290LP"``).
        Combined with ``grating_name`` as ``"{grating_name}/{filter_name}"``,
        this must be one of `available_grating_filters`.

    Attributes
    ----------
    grating_filter_name : `str`
        Combined ``"{grating_name}/{filter_name}"`` configuration name.
    grating : `Spectral_Grating`
        The grating associated with this instrument configuration.
    filter : `Spectral_Filter`
        The filter associated with this instrument configuration.
    """

    available_grating_filters = [
        "G140M/F070LP",
        "G140M/F100LP",
        "G235M/F170LP",
        "G395M/F290LP",
        "G140H/F070LP",
        "G140H/F100LP",
        "G235H/F170LP",
        "G395H/F290LP",
        "PRISM/CLEAR",
    ]

    def __init__(self, grating_name: str, filter_name: str) -> NoReturn:
        grating_filter_name = f"{grating_name}/{filter_name}"
        self.grating_filter_name = grating_filter_name
        if grating_filter_name not in self.available_grating_filters:
            raise InvalidOptionError(
                f"grating_filter_name={grating_filter_name!r} not in "
                f"available_grating_filters={self.available_grating_filters!r}."
            )
        super().__init__(
            Spectral_Grating(grating_name),
            Spectral_Filter(filter_name),
        )

    def load_sensitivity(self):
        """Load the sensitivity curve for this NIRSpec configuration
        (e.g. from pandeia). Not yet implemented."""
        # load from pandeia
        pass

    def get_sensitivity(self):
        """Return the sensitivity of this NIRSpec configuration. Not yet
        implemented."""
        # determine from self.sensitivity
        pass


instrument_conv_dict = {"NIRSPEC": NIRSpec}

#: Mapping of ``author_year`` strings to the rest-frame
#: ``[blue_range, red_range]`` continuum windows either side of the
#: 4000 Angstrom break, for use in `Spectrum.fit_D4000_break`.
D4000_WAV_RANGES: Dict[str, u.Quantity] = {
    # original/"wide" definition
    "Bruzual+83": [[3_750.0, 3_950.0], [4_050.0, 4_200.0]] * u.AA,
    # narrow definition, D4000_n
    "Balogh+99": [[3_850.0, 3_950.0], [4_000.0, 4_100.0]] * u.AA,
    "Wang+25": [[3_620.0, 3_720.0], [4_000.0, 4_100.0]] * u.AA,
}


class Spectrum:
    """A single reduced 1D spectrum of a source, with associated metadata.

    Parameters
    ----------
    wavs : `astropy.units.Quantity`
        Observed-frame wavelengths of the spectrum.
    fluxes : `astropy.units.Quantity` or `astropy.units.Magnitude`
        Flux (or magnitude) values corresponding to `wavs`.
    flux_errs : `astropy.units.Quantity` or `astropy.units.Magnitude`
        Uncertainties on `fluxes`.
    sky_coord : `astropy.coordinates.SkyCoord`
        Sky position of the source.
    z : `float`
        Redshift of the source.
    z_method : `str`
        Method/origin used to determine `z` (e.g. ``"cat"``).
    instrument : `Spectral_Instrument`
        Instrument configuration used to take this spectrum.
    reduction_name : `str`
        Name/version of the data reduction pipeline used to produce this
        spectrum.
    MSA_metafile_name : `str`
        Path to the associated MSA metafile, if any.
    author_years : `dict`, optional
        Mapping of ``author_year`` strings to redshift values from the
        literature. Default is `{}`.
    meta : `dict`, optional
        Additional metadata associated with the spectrum, typically the
        FITS header of the originating exposure. Default is `{}`.
    **kwargs
        Additional keyword arguments set as attributes on the instance.

    Attributes
    ----------
    wavs : `astropy.units.Quantity`
        Observed-frame wavelengths of the spectrum.
    fluxes : `astropy.units.Quantity` or `astropy.units.Magnitude`
        Flux (or magnitude) values corresponding to `wavs`.
    flux_errs : `astropy.units.Quantity` or `astropy.units.Magnitude`
        Uncertainties on `fluxes`.
    sky_coord : `astropy.coordinates.SkyCoord`
        Sky position of the source.
    RA : `float`
        Right ascension of `sky_coord`, in degrees.
    DEC : `float`
        Declination of `sky_coord`, in degrees.
    z : `float`
        Redshift of the source.
    z_method : `str`
        Method/origin used to determine `z`.
    instrument : `Spectral_Instrument`
        Instrument configuration used to take this spectrum.
    reduction_name : `str`
        Name/version of the data reduction pipeline used to produce this
        spectrum.
    MSA_metafile_name : `str`
        Path to the associated MSA metafile, if any.
    author_years : `dict`
        Mapping of ``author_year`` strings to redshift values from the
        literature.
    meta : `dict`
        Additional metadata associated with the spectrum.
    """

    def __init__(
        self,
        wavs: u.Quantity,
        fluxes: Union[u.Quantity, u.Magnitude],
        flux_errs: Union[u.Quantity, u.Magnitude],
        sky_coord: SkyCoord,
        z: float,
        z_method: str,
        instrument: Spectral_Instrument,
        reduction_name: str,
        MSA_metafile_name: str,
        author_years: dict = {},  # {author_year: z}
        meta: dict = {},
        **kwargs,
    ) -> NoReturn:
        self.wavs = wavs
        self.fluxes = fluxes
        self.flux_errs = flux_errs
        self.sky_coord = sky_coord
        self.RA = sky_coord.ra.deg
        self.DEC = sky_coord.dec.deg
        self.z = z
        self.z_method = z_method
        self.instrument = instrument
        self.reduction_name = reduction_name
        self.MSA_metafile_name = MSA_metafile_name
        self.author_years = author_years
        self.meta = meta
        # rest-frame wavelength ranges cached by e.g. fit_D4000_break,
        # highlighted on subsequent calls to plot()
        self._plot_wav_highlights: Dict[str, Dict[str, Any]] = {}
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        """Return a string representation of the `Spectrum` instance."""
        return (
            f"Spectrum({self.src_name}, z={self.z}, "
            f"{self.instrument.grating_filter_name})"
        )

    def _cache_wav_highlight(
        self: Self,
        key: str,
        rest_wav_ranges: u.Quantity,
        colors: Union[str, List[str]] = "grey",
    ) -> NoReturn:
        """Cache rest-frame wavelength range(s) to highlight on `plot()`.

        Stores `rest_wav_ranges` (and associated `colors`) under `key` in
        `self._plot_wav_highlights`, overwriting any existing entry with
        the same key. `plot()` converts these to the frame and units it
        is called with and draws them as shaded spans (not added to the
        legend).

        Parameters
        ----------
        key : `str`
            Name identifying this set of highlighted ranges (e.g.
            ``"D4000"``). Overwrites any previously cached entry with the
            same key.
        rest_wav_ranges : `astropy.units.Quantity`
            Rest-frame wavelength ranges to highlight, as an ``(N, 2)``
            array of ``[low, high]`` pairs.
        colors : `str` or `list` of `str`, optional
            Colour(s) to shade each range with. If a single `str`, used
            for every range. Default is ``"grey"``.
        """
        if isinstance(colors, str):
            colors = [colors] * len(rest_wav_ranges)
        self._plot_wav_highlights[key] = {
            "rest_wav_ranges": rest_wav_ranges,
            "colors": colors,
        }

    @property
    def PID(self) -> Union[int, None]:
        """`int` or `None`: JWST program ID, taken from
        `meta["PROGRAM"]` or the leading component of `meta["SRCNAM1"]`
        (cached after first access)."""
        try:
            return self._PID
        except AttributeError:
            if "PROGRAM" in self.meta.keys():
                self._PID = int(self.meta["PROGRAM"])
            elif "SRCNAM1" in self.meta.keys():
                self._PID = str(self.meta["SRCNAM1"].split("_")[0])
            else:
                raise MissingKeyError(
                    f"{repr(self)}.meta is missing both 'PROGRAM' and "
                    "'SRCNAM1'; cannot determine PID."
                )
            return self._PID

    # @property
    # def root(self) -> Union[str, None]:
    #     """`str` or `None`: Root name of the source, taken from
    #     `meta["SRCNAM1"]` (cached after first access)."""
    #     try:
    #         return self._root
    #     except AttributeError:
    #         if "SRCNAM1" in self.meta.keys():
    #             self._root = str(self.meta["SRCNAM1"].split("_")[0])
    #         else:
    #             raise (Exception())
    #         return self._root

    @property
    def src_ID(self) -> Union[int, None]:
        """`int` or `None`: Source ID, taken from `meta["SOURCEID"]` or
        the second component of `meta["SRCNAM1"]` (cached after first
        access)."""
        try:
            return self._src_ID
        except AttributeError:
            if "SOURCEID" in self.meta.keys():
                self._src_ID = int(self.meta["SOURCEID"])
            elif "SRCNAM1" in self.meta.keys():
                self._src_ID = int(self.meta["SRCNAM1"].split("_")[1])
            else:
                raise MissingKeyError(
                    f"{repr(self)}.meta is missing both 'SOURCEID' and "
                    "'SRCNAM1'; cannot determine src_ID."
                )
            return self._src_ID

    @property
    def src_name(self):
        """`str`: Unique source name, combining `PID` and `src_ID` as
        ``"{PID}_{src_ID}"``."""
        _src_name = f"{self.PID}_{self.src_ID}"
        if hasattr(self, "root"):
            _src_name = f"{self.root}_{_src_name}"
        return _src_name

    @property
    def MSA_ID(self):
        """`int`: MSA metadata ID, taken from `meta["MSAMETID"]`
        (cached after first access)."""
        try:
            return self._meta_ID
        except AttributeError:
            if "MSAMETID" in self.meta.keys():
                self._meta_ID = int(self.meta["MSAMETID"])
            else:
                raise MissingKeyError(
                    f"{repr(self)}.meta is missing 'MSAMETID'; cannot "
                    "determine MSA_ID."
                )
            return self._meta_ID

    @property
    def dither_pt(self):
        """`int`: Dither point/pattern index, taken from
        `meta["PATT_NUM"]` (cached after first access)."""
        try:
            return self._dither_pt
        except AttributeError:
            if "PATT_NUM" in self.meta.keys():
                self._dither_pt = int(self.meta["PATT_NUM"])
            else:
                raise MissingKeyError(
                    f"{repr(self)}.meta is missing 'PATT_NUM'; cannot "
                    "determine dither_pt."
                )
            return self._dither_pt

        # meta = {"PID": int(header["PROGRAM"]), "src_ID": int(header(
        #    "SOURCEID")), "slit_ID": int(header["SLITID"]), "exp_time":
        #    float(header["DURATION"]), "readout_pattern": \
        #    "nod_type": str(header["NOD_TYPE"]).replace(" ", ""),
        #    "src_slit_pos": [float(header["SRCXPOS"]),
        #    float(header["SRCYPOS"])]}
        #    str(header["READPATT"]).replace(" ", ""), "n_integrations":
        #    int(header["NINTS"]), "n_groups": int(header["NGROUPS"]), \

    @classmethod
    def from_DJA(
        cls,
        url_path: str,
        save: bool = True,
        version: str = "v4_4",
        z: Union[float, None] = None,
        *args,
        **kwargs,
    ) -> Self:
        """Construct a `Spectrum` from a DAWN JWST Archive (DJA) 2D spectrum.

        Downloads (or loads a local cache of) the 2D spectrum FITS file at
        `url_path`, extracts the source position and instrument
        configuration from its header, extracts the 1D spectrum using
        ``msaexp`` (caching the result as a local ``.h5`` file), and
        downloads the associated MSA metafile. The extracted 1D wavelength,
        flux and flux error arrays are passed to the `Spectrum` constructor.

        Parameters
        ----------
        url_path : `str`
            URL or local path to the DJA 2D spectrum FITS file.
        save : `bool`, optional
            Whether to save a local copy of the downloaded 2D spectrum FITS
            file. Default is `True`.
        version : `str`, optional
            DJA reduction version (e.g. ``"v2"``, ``"v3"``, ``"v4_2"``,
            ``"v4_4"``), used to determine how the flux errors and MSA
            metafile name are extracted. Default is ``"v4_4"``.
        z : `float` or `None`, optional
            Redshift of the source. If given, `z_method` is set to
            ``"cat"``, otherwise `z_method` is `None`. Default is `None`.
        *args
            Additional positional arguments passed to the `Spectrum`
            constructor.
        **kwargs
            Additional keyword arguments passed to the `Spectrum`
            constructor.

        Returns
        -------
        `Spectrum`
            A new `Spectrum` instance built from the DJA data, with an
            additional `origin` attribute set to the local path of the 2D
            spectrum FITS file.
        """
        # open 2D spectrum
        loc_2d_path = url_path.replace(
            config["Spectra"]["DJA_WEB_DIR"],
            config["Spectra"]["DJA_2D_SPECTRA_DIR"],
        )
        if not Path(loc_2d_path).is_file():
            funcs.make_dirs(loc_2d_path)
            img = fits.open(url_path, cache=False)
            if save:
                img.writeto(loc_2d_path)
                funcs.change_file_permissions(loc_2d_path)
        else:
            img = fits.open(loc_2d_path)
        # extract info from img header
        header = img["SCI"].header
        sky_coord = SkyCoord(
            ra=float(header["SRCRA"]) * u.deg,
            dec=float(header["SRCDEC"]) * u.deg,
        )
        # make Spectral_Instrument object
        grating_name = str(header["GRATING"]).replace(" ", "")
        filter_name = str(header["FILTER"]).replace(" ", "")
        try:
            instrument = instrument_conv_dict[
                str(header["INSTRUME"]).replace(" ", "")
            ]
        except Exception:
            instrument = NIRSpec
        instrument = instrument(grating_name, filter_name)

        # extract 1D spectrum from 2D fits image using msaexp
        loc_1d_path = url_path.replace(
            config["Spectra"]["DJA_WEB_DIR"],
            config["Spectra"]["DJA_1D_SPECTRA_DIR"],
        )
        if not Path(loc_1d_path).is_file():
            import msaexp.spectrum

            spectrum_1D = msaexp.spectrum.SpectrumSampler(loc_2d_path)
            # could also extract resolution here
            mask = ~spectrum_1D.spec["valid"]
            wavs = spectrum_1D.spec["wave"]
            fluxes = spectrum_1D.spec[
                "flux"
            ]  # Masked( * flux_unit, mask = mask)
            if version == "v2":
                # determine number of exposures
                N_exposures = int(header["NOUTPUTS"]) * int(header["NFRAMES"])
                flux_errs = spectrum_1D.spec["full_err"] * (N_exposures**-0.25)
            elif version in ["v3", "v4_2", "v4_4"]:
                flux_errs = spectrum_1D.spec["full_err"]
            else:
                flux_errs = spectrum_1D.spec["full_err"]
            # breakpoint()
            # save as local .h5 file
            funcs.make_dirs(loc_1d_path)
            hf = h5py.File(loc_1d_path, "w")
            for name, data in zip(
                ["mask", "wavs", "fluxes", "flux_errs"],
                [mask, wavs, fluxes, flux_errs],
            ):
                hf.create_dataset(name, data=data)
            wav_unit = u.um  # NOT GENERAL!
            flux_unit = u.Unit(str(header["BUNIT"].replace(" ", "")))
            hf.attrs["wav_unit"] = (u.um).to_string()
            hf.attrs["flux_unit"] = flux_unit.to_string()
            hf.close()
        else:
            hf = h5py.File(loc_1d_path, "r")
            mask = np.array(hf["mask"])
            wavs = np.array(hf["wavs"])
            wav_unit = u.Unit(hf.attrs["wav_unit"])
            flux_unit = u.Unit(hf.attrs["flux_unit"])
            fluxes = np.array(hf["fluxes"])
            flux_errs = np.array(hf["flux_errs"])

        wavs *= wav_unit
        fluxes = Masked(fluxes * flux_unit, mask=mask)
        flux_errs = Masked(np.array(flux_errs) * flux_unit, mask=mask)

        if version in ["v2"]:
            msa_metafile = str(header["MSAMETFL"]).replace(" ", "")
        elif version in ["v3", "v4_2", "v4_4"]:
            msa_metafile = str(header["MSAMETFL"]).replace(" ", "")
        else:
            msa_metafile = str(header["MSAMET1"]).replace(" ", "")

        meta_uri_dir = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product"
        meta_in_path = f"{meta_uri_dir}/{msa_metafile}"

        try:
            out_dir = config["Spectra"]["DJA_2D_SPECTRA_DIR"].replace(
                "2D",
                "MSA_metafiles",
            )
            meta_out_path = f"{out_dir}/{msa_metafile}"
            if not Path(meta_out_path).is_file():
                meta = fits.open(meta_in_path, cache=False)
                funcs.make_dirs(meta_out_path)
                meta.writeto(meta_out_path)
                funcs.change_file_permissions(meta_out_path)
            MSA_metafile_name = meta_out_path
        except Exception:
            MSA_metafile_name = None

        if z is None:
            z_method = None
        else:
            z_method = "cat"
        reduction_name = f"DJA_{version}"

        spec_obj = cls(
            wavs,
            fluxes,
            flux_errs,
            sky_coord,
            z,
            z_method,
            instrument,
            reduction_name,
            MSA_metafile_name,
            meta={name: header[name] for name in header},
            **kwargs,
        )
        spec_obj.origin = loc_2d_path
        return spec_obj

    def load_MSA_metafile(self):
        """Load the MSA metafile referenced by `MSA_metafile_name` into
        `MSA_metafile`.

        If loading fails (e.g. `MSA_metafile_name` is invalid or `None`),
        `MSA_metafile` is set to `None`. Does nothing if `MSA_metafile`
        is already set.
        """
        from msaexp import msa

        if not hasattr(self, "MSA_metafile"):
            try:
                self.MSA_metafile = msa.MSAMetafile(self.MSA_metafile_name)
            except Exception:
                self.MSA_metafile = None

    def plot_slitlet(
        self: Self,
        ax: plt.Axes,
        wcs: WCS,
        add_labels: bool = True,
        colour: str = "magenta",
        nod_colour: str = "lightpink",
        **plot_kwargs,
    ):
        """Overplot the NIRSpec MSA slitlet outlines for this source's
        dither point on an image.

        Loads the MSA metafile if not already loaded, retrieves the slit
        regions for this source's dither point/MSA metadata ID, converts
        their sky-coordinate corners to pixel coordinates using `wcs`, and
        draws each slit that overlaps the field of view onto `ax`. The
        primary source slit is drawn in `colour`, nod slits in `nod_colour`.

        Parameters
        ----------
        ax : `matplotlib.pyplot.Axes`
            Axes to plot the slitlet outlines on.
        wcs : `astropy.wcs.WCS`
            WCS used to convert slit sky corners to pixel coordinates.
        add_labels : `bool`, optional
            Whether to annotate the axes with the dither point, MSA
            metafile name and source ID. Default is `True`.
        colour : `str`, optional
            Colour used to draw the primary source slit. Default is
            ``"magenta"``.
        nod_colour : `str`, optional
            Colour used to draw non-source (nod) slits. Default is
            ``"lightpink"``.
        **plot_kwargs
            Additional keyword arguments passed to `ax.plot` when drawing
            each slit outline.
        """
        # mostly copied from msaexp MSAMetafile base code
        self.load_MSA_metafile()
        if self.MSA_metafile is None:
            raise MissingDataError(
                f"{repr(self)}.MSA_metafile failed to load from "
                f"MSA_metafile_name={self.MSA_metafile_name!r}; cannot "
                "plot slitlet."
            )
        slits = self.MSA_metafile.regions_from_metafile(
            dither_point_index=self.dither_pt,
            as_string=False,
            with_bars=True,
            msa_metadata_id=self.MSA_ID,
        )
        for s in slits:
            xy = np.array(s.xy[0])  # shape (4, 2) - RA/Dec corners
            # convert corners to pixel coordinates
            pixels = wcs.world_to_pixel_values(xy[:, 0], xy[:, 1])
            x_pix = np.append(pixels[0], pixels[0][0])  # close the rectangle
            y_pix = np.append(pixels[1], pixels[1][0])
            if s.meta["is_source"]:
                colour_ = colour
                lw = plot_kwargs.get("lw", 2.0) * 1.5
            else:
                colour_ = nod_colour
                lw = plot_kwargs.get("lw", 2.0)
            plot_kwargs_ = deepcopy(plot_kwargs)
            plot_kwargs_.pop("lw", None)
            # plot only slits that enter the field of view
            if np.any(
                (x_pix >= 0)
                & (x_pix < wcs.pixel_shape[0])
                & (y_pix >= 0)
                & (y_pix < wcs.pixel_shape[1])
            ):
                ax.plot(x_pix, y_pix, color=colour_, lw=lw, **plot_kwargs_)

        if add_labels:
            ax.text(
                0.03,
                0.07,
                f"Dither #{self.dither_pt}",
                ha="left",
                va="bottom",
                transform=ax.transAxes,
                color="magenta",  # color
                fontsize=8,
            )
            ax.text(
                0.03,
                0.03,
                f"{os.path.basename(self.MSA_metafile.metafile)}",
                ha="left",
                va="bottom",
                transform=ax.transAxes,
                color="magenta",  # color,
                fontsize=8,
            )
            ax.text(
                0.97,
                0.07,
                f"{self.src_ID}",
                ha="right",
                va="bottom",
                transform=ax.transAxes,
                color="magenta",  # colour,
                fontsize=8,
            )
            # ax.text(
            #     0.97,
            #     0.03,
            #     f"({self.sky_coord.ra.deg:.6f}, "
            #     f"{self.sky_coord.dec.deg:.6f})",
            #     ha = "right",
            #     va = "bottom",
            #     transform = ax.transAxes,
            #     color = colour,
            #     fontsize = 8,
            # )

    def calc_SNR_cont(
        self: Self,
        rest_cont_wav: u.Quantity,
        delta_wav: u.Quantity = 100 * u.AA,
    ):
        """Compute the median signal-to-noise ratio in a rest-frame
        continuum window.

        Selects data points within ``delta_wav / 2`` of `rest_cont_wav` in
        the rest frame, computes the per-pixel flux/flux_err ratio, and
        takes the median as the mean SNR. Also caches the result on `self.SNR`.

        Parameters
        ----------
        rest_cont_wav : `astropy.units.Quantity`
            Central rest-frame wavelength of the continuum window.
        delta_wav : `astropy.units.Quantity`, optional
            Full width of the rest-frame continuum window. Default is
            ``100 * u.AA``.

        Returns
        -------
        `float`
            Median SNR of the data points within the continuum window.
        """
        if not hasattr(self, "z"):
            raise MissingDataError(
                f"{repr(self)} does not have a redshift (z) attribute!"
            )
        rest_wavs = self.wavs / (1.0 + self.z)
        wav_mask = (rest_wavs > rest_cont_wav - delta_wav / 2.0) & (
            rest_wavs < rest_cont_wav + delta_wav / 2.0
        )
        fluxes = self.fluxes[wav_mask]
        flux_errs = self.flux_errs[wav_mask]
        SNR_arr = [flux / err for flux, err in zip(fluxes, flux_errs)]
        mean_SNR = np.nanmedian(SNR_arr)
        # HACK: This is not general!
        self.SNR = mean_SNR
        return mean_SNR

    def plot(
        self: Self,
        frame: str = "obs",
        src: str = "manual",
        out_dir: Optional[
            str
        ] = f"{config['DEFAULT']['GALFIND_WORK']}/DJA_spec_plots/",
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        wav_units: u.Unit = u.um,
        flux_units: u.Unit = u.uJy,
        annotate: bool = True,
        log_fluxes: bool = False,
        rest_wav_range: Optional[u.Quantity] = None,
        plot_masked: bool = True,
        **fit_kwargs: Dict[str, Any],
    ) -> NoReturn:
        """Plot the spectrum,
        either via ``msaexp``'s plotting routine or manually.

        If ``src == "msaexp"``, uses `msaexp.spectrum.plot_spectrum` on
        `self.origin` and saves the figure under `out_dir`. If
        ``src == "manual"``, plots the (mask-cropped) flux density against
        wavelength on `ax` (creating `fig`/`ax` if not given), converting
        units as requested and optionally plotting in log flux space.

        Parameters
        ----------
        frame: `str`, optional
            Frame to plot the spectrum in, either ``"obs"`` or ``"rest".
            Default is ``"obs"``.
        src : `str`, optional
            Plotting method to use, either ``"msaexp"`` or ``"manual"``.
            Default is ``"manual"``.
        out_dir : `str` or `None`, optional
            Output directory for the saved figure when ``src == "msaexp"``.
            Default is
            ``f"{config['DEFAULT']['GALFIND_WORK']}/DJA_spec_plots/"``.
        fig : `matplotlib.pyplot.Figure` or `None`, optional
            Figure to plot on when ``src == "manual"``. Created if `None`.
            Default is `None`.
        ax : `matplotlib.pyplot.Axes` or `None`, optional
            Axes to plot on when ``src == "manual"``. Created if `None`.
            Default is `None`.
        wav_units : `astropy.units.Unit`, optional
            Units to convert the wavelength axis to. Default is `u.um`.
        flux_units : `astropy.units.Unit`, optional
            Units to convert the flux axis to. Default is `u.uJy`.
        annotate : `bool`, optional
            Whether to annotate the axes with labels, title and legend.
            Default is `True`.
        log_fluxes : `bool`, optional
            Whether to plot the flux axis in log10 space. Default is `False`.
        rest_wav_range : `astropy.units.Quantity` or `None`, optional
            Rest-frame wavelength range to plot, as ``(min, max)``, in any
            units convertible to Angstrom. If given, the spectrum is
            cropped to this range before plotting. Default is `None`.
        plot_masked : `bool`, optional
            Whether to plot masked data points. Default is `True`.
        **fit_kwargs : `dict`
            Additional keyword arguments passed to the plotting calls
            (e.g. line colour, alpha).
        """
        if src not in ["msaexp", "manual"]:
            raise InvalidOptionError(
                f"src={src!r} not in ['msaexp', 'manual']."
            )
        if frame not in ["obs", "rest"]:
            raise InvalidOptionError(
                f"frame={frame!r} not in ['obs', 'rest']."
            )
        if src == "msaexp":
            raise NotImplementedError()
            import msaexp.spectrum

            fig, spec, data = msaexp.spectrum.plot_spectrum(
                self.origin, z=self.z
            )
            self.fit_data = data
            if out_dir is None:
                out_dir = ""
            save_path = (
                f"{out_dir}{self.instrument.grating_filter_name}/"
                f"{self.src_name}_spec.png"
            )
            funcs.make_dirs(save_path)
            fig.savefig(save_path)
        elif src == "manual":
            if fig is None or ax is None:
                fig, ax = plt.subplots()
            # unit conversions
            mask = ~self.fluxes.mask
            wavs = funcs.convert_wav_units(self.wavs[mask], wav_units)
            if frame == "rest":
                wavs /= 1 + self.z
            if rest_wav_range is None:
                wav_range_mask = np.ones_like(wavs, dtype=bool)
            else:
                if frame == "rest":
                    wav_range = np.array(rest_wav_range.to(u.AA).value)
                else:  # frame == "obs":
                    wav_range = np.array(rest_wav_range.to(u.AA).value) * (
                        1 + self.z
                    )
                wav_range_mask = np.array(
                    [
                        wav > wav_range[0] and wav < wav_range[1]
                        for wav in wavs.to(u.AA).value
                    ]
                ).astype(bool)
            wavs = wavs[wav_range_mask]
            fluxes = funcs.convert_mag_units(
                wavs,
                self.fluxes[mask][wav_range_mask].filled(np.nan),
                flux_units,
            )
            flux_errs = funcs.convert_mag_err_units(
                wavs,
                self.fluxes[mask][wav_range_mask].filled(np.nan),
                np.array(
                    [
                        self.flux_errs[mask][wav_range_mask]
                        .filled(np.nan)
                        .value,
                        self.flux_errs[mask][wav_range_mask]
                        .filled(np.nan)
                        .value,
                    ]
                )
                * self.flux_errs.unit,
                flux_units,
            )
            wavs = wavs.value
            fluxes = fluxes.value
            if log_fluxes:
                flux_errs_l1 = np.log10(fluxes / (fluxes - flux_errs[0].value))
                flux_errs_u1 = np.log10((fluxes + flux_errs[1].value) / fluxes)
                fluxes = np.log10(fluxes)
            else:
                flux_errs_l1 = flux_errs[0].value
                flux_errs_u1 = flux_errs[1].value
            flux_errs = [flux_errs_l1, flux_errs_u1]
            ax.plot(wavs, fluxes, label=self.src_name, **fit_kwargs)
            alpha = deepcopy(fit_kwargs).pop("alpha", 1.0) * 0.5
            ax.fill_between(
                wavs,
                fluxes - flux_errs[0],
                fluxes + flux_errs[1],
                alpha=alpha,
                **fit_kwargs,
            )
            # highlight any cached wavelength ranges
            # (e.g. from fit_D4000_break), excluded from the legend
            for highlight in self._plot_wav_highlights.values():
                for rest_wav_range, color in zip(
                    highlight["rest_wav_ranges"],
                    highlight["colors"],
                ):
                    plot_wav_range = funcs.convert_wav_units(
                        rest_wav_range, wav_units
                    )
                    if frame == "obs":
                        plot_wav_range = plot_wav_range * (1.0 + self.z)
                    ax.axvspan(
                        plot_wav_range[0].value,
                        plot_wav_range[1].value,
                        color=color,
                        alpha=0.2,
                    )
        if annotate:
            # label x and y axes
            ax.set_xlabel(
                f"{frame.capitalize()} wavelength [{wav_units.to_string()}]"
            )
            if log_fluxes:
                ax.set_ylabel(f"log10(flux [{flux_units.to_string()}])")
            else:
                ax.set_ylabel(f"flux [{flux_units.to_string()}]")
            # add title with source name and redshift
            ax.set_title(f"{self.src_name} (z={self.z:.3f})")
            # add legend
            ax.legend()

    def make_mock_phot(
        self: Self,
        filterset: Multiple_Filter,
        depths: Optional[Dict[str, float]] = None,
    ):
        """Create mock photometry in a given filterset from this spectrum.

        Builds an `SED_obs` from the spectrum's wavelength/flux arrays and
        redshift, then creates mock photometry by convolving it with
        `filterset`.

        Parameters
        ----------
        filterset : `Multiple_Filter`
            Set of filters to create mock photometry for.
        depths : `dict` of `str` to `float`, optional
            Per-band depths used when generating the mock photometry.
            Default is `None`.

        Returns
        -------
        Mock photometry created from the spectrum's `SED_obs` representation
        via `SED_obs.create_mock_photometry`.
        """
        from . import SED_obs

        # TODO: Link SED and Spectrum objects
        # make SED object from self
        if self.z is None:
            raise MissingDataError(
                f"{repr(self)} does not have a redshift (z) attribute!"
            )
        sed_obs = SED_obs(
            self.z,
            self.wavs.value,
            self.fluxes.value,
            self.wavs.unit,
            self.fluxes.unit,
        )
        return sed_obs.create_mock_photometry(
            filterset,
            depths=depths,
        )

    def fit_UV_slope(
        self: Self,
        wav_range: Union[str, u.Quantity] = "Calzetti+94",
    ) -> Tuple[float, List[float]]:
        """Fit the rest-frame UV continuum slope, beta, of the spectrum.

        Crops the rest-frame spectrum to the Calzetti et al. (1994)
        continuum windows, converts fluxes to f_lambda, and fits a
        power-law ``f(wav) = 10**A * wav**beta`` via
        `scipy.optimize.curve_fit`.

        Parameters
        ----------
        wav_range : `str` or `astropy.units.Quantity`, optional
            Wavelength range/window definition to fit over. Only
            ``"Calzetti+94"`` is currently implemented. Default is
            ``"Calzetti+94"``.

        Returns
        -------
        `tuple` of (`float`, `list` of `float`)
            The fitted UV slope ``beta`` and its symmetric
            ``[beta_err, beta_err]`` uncertainty. Returns ``(nan, [nan, nan])``
            if there are not enough valid data points or if the fit fails.

        Raises
        ------
        MissingDataError
            If the spectrum does not have a redshift (`z`) attribute.
        InvalidOptionError
            If `wav_range` is not ``"Calzetti+94"``.
        """
        if not hasattr(self, "z"):
            raise MissingDataError(
                f"{repr(self)} does not have a redshift (z) attribute!"
            )
        # convert wavs to rest frame
        rest_wavs = self.wavs / (1.0 + self.z)
        if wav_range == "Calzetti+94":
            wavs, fluxes = funcs.crop_to_Calzetti94_filters(
                rest_wavs, self.fluxes
            )
        else:
            raise InvalidOptionError(
                f"wav_range={wav_range!r} not in ['Calzetti+94']; only "
                "'Calzetti+94' is currently implemented for fit_UV_slope."
            )
        if len([mask for mask in fluxes.mask if not mask]) <= 1:
            galfind_logger.debug(
                "Not enough valid data points to fit UV slope for "
                f"{repr(self)}"
            )
            beta = np.nan
            beta_err = np.nan
        else:
            # convert fluxes to f_lambda in rest frame
            fluxes = funcs.convert_mag_units(
                wavs, fluxes, u.erg / u.s / u.cm**2 / u.AA
            )
            try:
                popt, pcov = curve_fit(
                    funcs.beta_slope_power_law_func,
                    wavs.value,
                    fluxes.value,
                    maxfev=1_000,
                )
                # A = popt[0]
                beta = popt[1]
                # A_err = np.sqrt(pcov[0][0])
                beta_err = np.sqrt(pcov[1][1])
            except Exception as e:
                galfind_logger.debug(
                    f"Failed to fit UV slope for {repr(self)}: {e}"
                )
                beta = np.nan
                beta_err = np.nan
                breakpoint()
        return beta, [beta_err, beta_err]

    def fit_Muv(
        self: Self,
        wav_range: u.Quantity = [1_450.0, 1_550.0] * u.AA,
        size=10_000,
    ):
        """Compute the absolute UV magnitude, M_UV, of the spectrum.

        Takes the inverse-variance weighted mean rest-frame f_lambda flux
        density within `wav_range`, propagates its uncertainty via Monte
        Carlo sampling of size `size`, converts to apparent AB magnitude at
        1500 Angstrom, and applies the distance and cosmological-dimming
        corrections to obtain M_UV. Also caches `flambda_1500_chains`,
        `MUV_arr`, `MUV`, `MUV_l1` and `MUV_u1` on `self`.

        Parameters
        ----------
        wav_range : `astropy.units.Quantity`, optional
            Rest-frame wavelength range to average the continuum flux
            density over. Default is ``[1450.0, 1550.0] * u.AA``.
        size : `int`, optional
            Number of Monte Carlo samples used to propagate the flux
            uncertainty into M_UV. Default is `10_000`.

        Returns
        -------
        `tuple` of (`astropy.units.Magnitude`, `list`)
            The median M_UV and its ``[l1, u1]`` 16th/84th percentile
            uncertainties. Returns ``(nan, [nan, nan])`` if there is no
            valid data or the unit conversion fails.
        """
        if not hasattr(self, "z"):
            raise MissingDataError(
                f"{repr(self)} does not have a redshift (z) attribute!"
            )
        rest_wavs = funcs.convert_wav_units(self.wavs, u.AA) / (1.0 + self.z)
        wav_range_AA = wav_range.to(u.AA)
        valid = (
            ~self.fluxes.mask
            & (rest_wavs < wav_range_AA[1])
            & (rest_wavs > wav_range_AA[0])
        )
        rest_wavs = rest_wavs[valid]
        fluxes = self.fluxes.filled(np.nan)[valid]
        flux_errs = self.flux_errs.filled(np.nan)[valid]

        if len(rest_wavs) > 0:
            try:
                flux_errs = funcs.convert_mag_err_units(
                    rest_wavs,
                    fluxes,
                    [flux_errs, flux_errs],
                    u.erg / u.s / u.cm**2 / u.AA,
                )[0]  # symmetric in flux space
            except Exception as e:
                galfind_logger.debug(
                    f"Failed to convert mag err units for {repr(self)}: {e}"
                )
                return np.nan, [np.nan, np.nan]
        else:
            galfind_logger.debug(f"No valid data for {self.src_name}")
            return np.nan, [np.nan, np.nan]
        fluxes = funcs.convert_mag_units(
            rest_wavs, fluxes, u.erg / u.s / u.cm**2 / u.AA
        )
        # fluxes *= (1. + z) ** 2
        # flux_errs *= (1. + z) ** 2
        # weighted mean
        weights = flux_errs**-2.0
        flambda_1500 = np.sum(fluxes * weights) / (
            np.sum(weights) * len(fluxes)
        )
        flambda_1500_err = np.sqrt(1.0 / np.sum(weights))
        # convert to MUV
        flambda_1500_chains = (
            np.random.normal(flambda_1500.value, flambda_1500_err.value, size)
            * u.erg
            / (u.s * (u.cm**2) * u.AA)
        )
        self.flambda_1500_chains = flambda_1500_chains
        fnu = funcs.convert_mag_units(
            1_500.0 * u.AA, flambda_1500_chains, u.Jy
        )
        mUV = -2.5 * np.log10(fnu.value) + u.Jy.to(u.ABmag)
        # mUV += 2.5 * np.log10(self.norm_factor)
        MUV_arr = (
            mUV
            - 5.0
            * np.log10(cosmo.luminosity_distance(self.z).to(u.pc).value / 10.0)
            + 2.5 * np.log10(1.0 + self.z)
        )
        self.MUV_arr = MUV_arr
        self.MUV = np.nanmedian(MUV_arr)
        self.MUV_l1 = self.MUV - np.nanpercentile(MUV_arr, 16)
        self.MUV_u1 = np.nanpercentile(MUV_arr, 84) - self.MUV
        return self.MUV, [self.MUV_l1, self.MUV_u1]

    def fit_D4000_break(
        self: Self,
        wav_ranges: Union[str, u.Quantity] = "Bruzual+83",
        size: int = 10_000,
    ) -> Optional[PDF]:
        """Compute the D4000 spectral break strength of the spectrum.

        Computes the median flux in the two rest-frame wavelength windows
        given by `wav_ranges` (blue and red side of the 4000 Angstrom
        break), forms their flux ratio, propagates its uncertainty via
        Monte Carlo sampling of size `size`, and converts to the
        magnitude-like D4000 index ``blue_mag - red_mag =
        2.5 * log10(f_red / f_blue)``, so a real break (``f_red >
        f_blue``) gives a positive D4000.

        Also caches `wav_ranges` (as rest-frame wavelengths) under the
        ``"D4000"`` key, so that subsequent calls to `plot()` highlight
        the blue/red continuum windows used here.

        Parameters
        ----------
        wav_ranges : `str` or `astropy.units.Quantity`, optional
            Either an ``author_year`` string identifying a literature
            definition of the two rest-frame wavelength ranges either side
            of the break (see `D4000_WAV_RANGES` for the available
            options, currently ``"Bruzual+83"``, ``"Balogh+99"`` and
            ``"Wang+25"``), or
            the two rest-frame wavelength ranges
            ``[blue_range, red_range]`` themselves, used to measure the
            continuum either side of the break. The first range must be
            entirely blueward of the second. Default is ``"Bruzual+83"``.
        size : `int`, optional
            Number of Monte Carlo samples used to propagate the flux ratio
            uncertainty. Default is `10_000`.

        Returns
        -------
        `PDF` or `None`
            A `PDF` (in `astropy.units.ABmag`) wrapping the Monte Carlo
            chain of `size` D4000 samples (its `median`/`errs` give the
            usual point estimate and ``[l1, u1]`` uncertainty). Also
            cached on `self.D4000_PDF`. Returns (and caches) `None` if
            either continuum window has a negative median flux.

        Raises
        ------
        InvalidOptionError
            If `wav_ranges` is a `str` not in `D4000_WAV_RANGES`.
        """
        from ..visualization.PDF import PDF

        if not hasattr(self, "z"):
            raise MissingDataError(
                f"{repr(self)} does not have a redshift (z) attribute!"
            )
        if isinstance(wav_ranges, str):
            author_year = wav_ranges
            if author_year not in D4000_WAV_RANGES:
                raise InvalidOptionError(
                    f"wav_ranges={author_year!r} not in "
                    f"{list(D4000_WAV_RANGES.keys())}."
                )
            wav_ranges = D4000_WAV_RANGES[author_year]
        else:
            author_year = None
        if len(wav_ranges) != 2:
            raise LengthMismatchError(
                f"wav_ranges={wav_ranges!r} has length {len(wav_ranges)}; "
                "must be a list of two wavelength ranges."
            )
        if not all(len(wav_range) == 2 for wav_range in wav_ranges):
            raise LengthMismatchError(
                f"Each element of wav_ranges={wav_ranges!r} must be a "
                "wavelength range (i.e. a list of two wavelengths)."
            )
        if not all(wav_range[0] < wav_range[1] for wav_range in wav_ranges):
            raise RangeError(
                f"In each wavelength range in wav_ranges={wav_ranges!r}, "
                "the first wavelength must be less than the second "
                "wavelength."
            )
        if not np.mean(wav_ranges[0]) < np.mean(wav_ranges[1]):
            raise RangeError(
                f"The first wavelength range in wav_ranges={wav_ranges!r} "
                "must be blueshifted relative to the second."
            )
        self._cache_wav_highlight(
            key="D4000",
            rest_wav_ranges=wav_ranges,
            colors=["tab:blue", "tab:red"],
        )
        rest_wavs = funcs.convert_wav_units(self.wavs, u.AA) / (1.0 + self.z)
        D4000_fluxes = {}
        D4000_flux_errs = {}
        for i, wav_range in enumerate(wav_ranges):
            wav_range_AA = wav_range.to(u.AA)
            valid = (
                ~self.fluxes.mask
                & (rest_wavs < wav_range_AA[1])
                & (rest_wavs > wav_range_AA[0])
            )
            rest_wavs_ = rest_wavs[valid]
            # median flux in wav_range
            if len(rest_wavs_) > 0:
                fluxes = self.fluxes.filled(np.nan)[valid]
                flux_errs = self.flux_errs.filled(np.nan)[valid]
                D4000_fluxes[i] = np.nanmedian(fluxes).value
                D4000_flux_errs[i] = np.nanmedian(flux_errs).value
            else:
                galfind_logger.debug(
                    f"No valid data for {self.src_name} in D4000 "
                    f"wav_range {wav_range=}"
                )
                D4000_fluxes[i] = np.nan
                D4000_flux_errs[i] = np.nan
        if any(col < 0 for col in D4000_fluxes.values()) or any(
            col < 0 for col in D4000_flux_errs.values()
        ):
            galfind_logger.debug(
                f"Negative fluxes for {self.src_name} in D4000 "
                f"wav_ranges: {D4000_fluxes=}"
            )
            self.D4000_PDF = None
            return None
        else:
            # compute D4000 and error
            D4000_flux_ratio = D4000_fluxes[1] / D4000_fluxes[0]
            D4000_flux_ratio_err = D4000_flux_ratio * np.sqrt(
                (D4000_flux_errs[1] / D4000_fluxes[1]) ** 2
                + (D4000_flux_errs[0] / D4000_fluxes[0]) ** 2
            )
            D4000_flux_ratio_arr = np.random.normal(
                D4000_flux_ratio, D4000_flux_ratio_err, size
            )
            # D4000 = blue_mag - red_mag = -2.5*log10(f_blue/f_red)
            #        = +2.5*log10(f_red/f_blue), so a real break
            #        (f_red > f_blue) gives a positive D4000
            D4000_arr = 2.5 * np.log10(D4000_flux_ratio_arr)
            self.D4000_PDF = PDF.from_1D_arr(
                "D4000",
                D4000_arr * u.ABmag,
                kwargs={"wav_ranges": wav_ranges, "author_year": author_year},
            )
            return self.D4000_PDF

    def fit_Ha(
        self: Self,
        wav_range=[6_200.0, 6_900.0] * u.AA,
        Halpha_wav: u.Quantity = 6562.8 * u.AA,
        frame: str = "rest",
        plot: bool = True,
        size: int = 10_000,
    ):
        """Fit an H-alpha Gaussian-plus-continuum model to the spectrum.

        Crops the rest-frame spectrum to `wav_range`, fits a Gaussian plus
        constant continuum model (`Halpha_gauss`/`Halpha_residual`) via
        ``lmfit.minimize``, propagates the fitted parameter uncertainties
        via Monte Carlo sampling of size `size`, and derives the H-alpha
        flux, equivalent width and integrated SNR. Caches
        `Halpha_flux_arr`, `Ha_EWrest`/`Ha_EWobs`, `Ha_cont`, `Ha_flux` and
        `Ha_SNR` on `self`. If `plot` is `True`, also saves a plot of the
        data and best-fit model.

        Parameters
        ----------
        wav_range : `astropy.units.Quantity`, optional
            Rest-frame wavelength range to fit over. Default is
            ``[6200.0, 6900.0] * u.AA``.
        Halpha_wav : `astropy.units.Quantity`, optional
            Rest-frame wavelength of the H-alpha line, used to define the
            feature mask for the SNR calculation. Default is
            ``6562.8 * u.AA``.
        frame : `str`, optional
            Frame in which to compute the equivalent width, either
            ``"rest"`` or ``"obs"``. Default is ``"rest"``.
        plot : `bool`, optional
            Whether to plot and save the data and best-fit model. Default
            is `True`.
        size : `int`, optional
            Number of Monte Carlo samples used to propagate parameter
            uncertainties. Default is `10_000`.

        Returns
        -------
        `None`
            Returns `None` if there is no valid data in `wav_range` or if
            the fit fails (in which case `failed_Ha_fit` is set to `True`).
            Otherwise returns `None` implicitly after setting the
            attributes described above.

        Raises
        ------
        InvalidOptionError
            If `frame` is not ``"rest"`` or ``"obs"``.
        """
        rest_wavs = funcs.convert_wav_units(self.wavs, u.AA) / (1.0 + self.z)
        wav_range_AA = wav_range.to(u.AA)
        valid = (
            ~self.fluxes.mask
            & (rest_wavs < wav_range_AA[1])
            & (rest_wavs > wav_range_AA[0])
        )
        rest_wavs = rest_wavs[valid]
        fluxes = self.fluxes.filled(np.nan)[valid]
        flux_errs = self.flux_errs.filled(np.nan)[valid]
        if len(rest_wavs) > 0:
            flux_errs = funcs.convert_mag_err_units(
                rest_wavs,
                fluxes,
                [flux_errs, flux_errs],
                u.erg / u.s / u.cm**2 / u.AA,
            )[0]  # symmetric in flux space
        else:
            print(f"No valid data for {self.src_name}")
            return None
        fluxes = funcs.convert_mag_units(
            rest_wavs, fluxes, u.erg / u.s / u.cm**2 / u.AA
        )

        # TODO: This doesn't actually constrain the width - something
        # wrong with the fitting here!
        params = Parameters()
        params.add(
            "A",
            value=np.max(fluxes.value) - np.median(fluxes.value),
            min=0.0,
            max=1e-12,
        )
        params.add("c", value=np.median(fluxes.value), min=0.0, max=1e-12)
        params.add("sigma", value=15.0, min=1.0, max=50.0)
        # dmodel = Model(gauss_model)
        # result = dmodel.fit(fluxes.value, params, wavs=wavs.value)
        # print(result.fit_report())

        out = minimize(
            Halpha_residual,
            params,
            args=(rest_wavs.value,),
            kws={"y": fluxes.value, "y_err": flux_errs.value},
        )
        print(fit_report(out))

        sigma = out.params["sigma"].value
        cont = out.params["c"].value

        try:
            A_arr = np.random.normal(
                loc=out.params["A"].value,
                scale=out.params["A"].stderr,
                size=size,
            )
            sigma_arr = np.random.normal(
                loc=params["sigma"].value,
                scale=out.params["sigma"].stderr,
                size=size,
            )
            cont_arr = np.random.normal(
                loc=params["c"].value, scale=out.params["c"].stderr, size=size
            )
        except Exception:
            breakpoint()
            self.failed_Ha_fit = True
            return

        Halpha_flux_arr = A_arr * sigma_arr * np.sqrt(2 * np.pi)
        self.Halpha_flux_arr = Halpha_flux_arr * u.erg / u.s / u.cm**2
        # Halpha_flux_arr = [self._Halpha_flux(A, sigma, c) *
        self.norm_factor
        #     for A, sigma, c in zip(self.A_arr, self.sigma_arr, self.c_arr)]
        if frame == "rest":
            EW_arr = Halpha_flux_arr / cont_arr
        elif frame == "obs":
            EW_arr = Halpha_flux_arr * (1.0 + self.z) / cont_arr
        else:
            raise InvalidOptionError(
                f"frame={frame!r} not in ['rest', 'obs']."
            )
        EW_percentiles = np.percentile(EW_arr, [16, 50, 84])
        if frame == "rest":
            self.Ha_EWrest = {
                "16": EW_percentiles[0],
                "50": EW_percentiles[1],
                "84": EW_percentiles[2],
            }
            # return EW_percentiles[0], EW_percentiles[1], EW_percentiles[2]
        elif frame == "obs":
            self.Ha_EWobs = {
                "16": EW_percentiles[0],
                "50": EW_percentiles[1],
                "84": EW_percentiles[2],
            }
            # return EW_percentiles[0], EW_percentiles[1], EW_percentiles[2]
        else:
            raise InvalidOptionError(
                f"frame={frame!r} not in ['rest', 'obs']."
            )

        cont_percentiles = np.percentile(cont_arr, [16, 50, 84])
        self.Ha_cont = {
            "16": cont_percentiles[0],
            "50": cont_percentiles[1],
            "84": cont_percentiles[2],
        }

        Halpha_flux_percentiles = np.percentile(Halpha_flux_arr, [16, 50, 84])
        self.Ha_flux = {
            "16": Halpha_flux_percentiles[0],
            "50": Halpha_flux_percentiles[1],
            "84": Halpha_flux_percentiles[2],
        }

        feature_mask = (rest_wavs.value > Halpha_wav.value - 5.0 * sigma) & (
            rest_wavs.value < Halpha_wav.value + 5.0 * sigma
        )
        SNRs = (fluxes[feature_mask].value - cont) / flux_errs.value[
            feature_mask
        ]
        integrated_SNR = np.sum(SNRs) / np.sqrt(len(SNRs))
        print(f"Integrated SNR: {integrated_SNR}")
        if np.isnan(integrated_SNR):
            raise RangeError(
                f"{repr(self)} integrated_SNR for the H-alpha fit is NaN."
            )
        self.Ha_SNR = integrated_SNR

        fig, ax = plt.subplots()
        if plot:
            ax.plot(
                rest_wavs.value, fluxes.value, c="black", label="NIRSpec/PRISM"
            )
            ax.fill_between(
                rest_wavs.value,
                fluxes.value - flux_errs.value,
                fluxes.value + flux_errs.value,
                alpha=0.5,
                color="black",
            )
            median_chains = [
                np.median(Halpha_gauss(wav, A_arr, sigma_arr, cont_arr))
                for wav in rest_wavs.value
            ]
            ax.plot(
                rest_wavs.value, median_chains, c="red", label="Halpha model"
            )
            model_l1 = [
                np.percentile(
                    Halpha_gauss(wav, A_arr, sigma_arr, cont_arr), 16
                )
                for wav in rest_wavs.value
            ]
            model_u1 = [
                np.percentile(
                    Halpha_gauss(wav, A_arr, sigma_arr, cont_arr), 84
                )
                for wav in rest_wavs.value
            ]
            ax.fill_between(
                rest_wavs.value, model_l1, model_u1, alpha=0.5, color="red"
            )
            # make rf string containing EW width and errors
            # plt.text(0.05, 0.95, r"EW$_{\mathrm{rest}}$(H$\alpha$)=" +
            # f"{Halpha_EWrest_50:.2f}" + r"$^{+" +
            # f"{Halpha_EWrest_84 - Halpha_EWrest_50:.2f}" + r"}_{-" +
            # f"{Halpha_EWrest_50 - Halpha_EWrest_16:.2f}" +
            # r"}~\mathrm{\AA}$", transform = plt.gca().transAxes)
            # make rf string containing flux and errors
            Halpha_flux_err_hi = (
                Halpha_flux_percentiles[2] - Halpha_flux_percentiles[1]
            )
            Halpha_flux_err_lo = (
                Halpha_flux_percentiles[1] - Halpha_flux_percentiles[0]
            )
            ax.text(
                0.05,
                0.95,
                r"$F_{\mathrm{H}\alpha}$="
                + f"{Halpha_flux_percentiles[1]:.2e}"
                + r"$^{+"
                f"{Halpha_flux_err_hi:.2e}" + r"}_{-"
                f"{Halpha_flux_err_lo:.2e}" + r"}~\mathrm{erg/s/cm^2}$",
                transform=plt.gca().transAxes,
            )
            # make rf string to show the SNR
            ax.text(
                0.05,
                0.9,
                f"SNR={self.Ha_SNR:.2f}",
                transform=plt.gca().transAxes,
            )
            out_path = f"../plots/Halpha_spec_fits/manual/{self.file}.png"
            funcs.make_dirs(out_path)
            ax.set_xlabel("Wavelength (AA)")
            ax.set_ylabel("Flux (erg/s/cm^2/AA)")
            ax.legend(loc="upper right")
            plt.savefig(out_path)
            plt.clf()

    def fit_xi_ion(self: Self, plot: bool = False):
        """Compute the ionizing photon production efficiency,
        xi_ion, of the source.

        Runs `fit_Muv` and `fit_Ha` to obtain the rest-frame UV and
        H-alpha luminosities, converts the H-alpha luminosity to an
        ionizing photon production rate using the Kennicutt (1998)-derived
        calibration ``ndot_ion = 7.28e11 * L(Halpha)``, and divides by the
        UV luminosity to obtain xi_ion. Caches `ndot_ion_arr`, `ndot_ion`,
        `ndot_ion_l1`, `ndot_ion_u1`, `xi_ion_arr`, `xi_ion`, `xi_ion_l1`
        and `xi_ion_u1` on `self`.

        Parameters
        ----------
        plot : `bool`, optional
            Whether `fit_Ha` should plot and save the H-alpha fit. Default
            is `False`.
        """
        self.fit_Muv()
        self.fit_Ha(plot=plot)
        LUV_arr = funcs.flux_to_luminosity(
            self.flambda_1500_chains, 1_500.0 * u.AA, self.z
        )
        LHa_arr = funcs.flux_to_luminosity(
            self.Halpha_flux_arr / (1.0 + self.z),
            6_562.8 * u.AA,
            self.z,
            out_units=u.erg / u.s,
        )
        self.ndot_ion_arr = 7.28e11 * LHa_arr.value
        self.ndot_ion = np.median(self.ndot_ion_arr)
        self.ndot_ion_l1 = np.percentile(self.ndot_ion_arr, 16)
        self.ndot_ion_u1 = np.percentile(self.ndot_ion_arr, 84)
        self.xi_ion_arr = self.ndot_ion_arr / LUV_arr.value
        self.xi_ion = np.median(self.xi_ion_arr)
        self.xi_ion_l1 = np.percentile(self.xi_ion_arr, 16)
        self.xi_ion_u1 = np.percentile(self.xi_ion_arr, 84)


def Halpha_gauss(x, A, sigma, c):
    """Evaluate a Gaussian-plus-constant model of the H-alpha emission line.

    Parameters
    ----------
    x : array-like
        Rest-frame wavelength(s) in Angstrom.
    A : `float`
        Gaussian amplitude.
    sigma : `float`
        Gaussian standard deviation, in Angstrom.
    c : `float`
        Constant continuum level.

    Returns
    -------
    array-like
        Model flux values at `x`, centred on the H-alpha rest wavelength
        (6562.8 Angstrom).
    """
    return A * np.exp(-0.5 * ((x - 6562.8) / sigma) ** 2) + c


def Halpha_residual(params, x, y, y_err):
    """Compute the weighted residual of `Halpha_gauss` for use with
    `lmfit.minimize`.

    Parameters
    ----------
    params : `lmfit.Parameters`
        Fit parameters, must contain ``'A'``, ``'sigma'`` and ``'c'``.
    x : array-like
        Rest-frame wavelength(s) in Angstrom.
    y : array-like
        Observed flux values at `x`.
    y_err : array-like
        Uncertainties on `y`.

    Returns
    -------
    array-like
        ``(model - y) / y_err`` evaluated at `x`.
    """
    # gaussian plus a constant
    model = Halpha_gauss(x, params["A"], params["sigma"], params["c"])
    return (model - y) / y_err


# should inherit from Catalogue_Base
class Spectral_Catalogue:
    """A collection of `Spectrum` objects, grouped by unique source.

    Spectra sharing the same `Spectrum.src_name` are grouped together
    under a single entry, so that a single `Spectral_Catalogue` element
    may contain multiple spectra (e.g. from different observations) of
    the same source.

    Parameters
    ----------
    spectrum_arr : `numpy.typing.NDArray` of `Spectrum`
        Array of `Spectrum` objects to group into the catalogue.

    Attributes
    ----------
    spectrum_arr : `list` of `list` of `Spectrum`
        Spectra grouped by unique source name; each element is a list of
        the `Spectrum` objects belonging to one unique source.
    """

    def __init__(self, spectrum_arr: NDArray[Spectrum]) -> None:
        # check if any of the sources are the same
        orig_src_names = [spec.src_name for spec in spectrum_arr]
        unique_src_names = np.unique(orig_src_names)
        self.spectrum_arr = [
            [spec for spec in spectrum_arr if spec.src_name == src_name]
            for src_name in unique_src_names
        ]
        # self.sky_coords = np.array(
        #     [spec[0].sky_coord for spec in self.spectrum_arr]
        # )

    def __len__(self):
        return len(self.spectrum_arr)

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter > len(self) - 1:
            raise StopIteration
        else:
            gal_spectra = self[self.iter]
            self.iter += 1
            return gal_spectra

    def __getitem__(self, index):
        return self.spectrum_arr[index]

    def __getattr__(self, name):
        if hasattr(self[0][0], name):
            return [getattr(spec[0], name) for spec in self]
        else:
            raise AttributeError

    def __add__(self, cat):
        if cat.__class__.__name__ != "Spectral_Catalogue":
            raise GalfindTypeError(
                f"cat has type {type(cat).__name__}; must be a "
                "Spectral_Catalogue object."
            )
        spectra_arr = np.array(
            [spectrum for gal in self for spectrum in gal]
            + [spectrum for gal in cat for spectrum in gal]
        )
        return Spectral_Catalogue(spectra_arr)

    def __deepcopy__(self, memo):
        galfind_logger.debug(f"deepcopy({self.__class__.__name__})")
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            try:
                setattr(result, key, deepcopy(value, memo))
            except Exception as e:
                raise GalfindError(
                    f"deepcopy({self.__class__.__name__}) failed for "
                    f"attribute {key!r}={value!r}: {e}"
                ) from e
        return result

    @classmethod
    def from_DJA(
        cls,
        ra_range: Union[list, np.array, u.Quantity] = None,
        dec_range: Union[list, np.array, u.Quantity] = None,
        PID: Union[int, None] = None,
        z_cat_range: Union[list, np.array, None] = None,
        grating_filter: Union[str, None] = None,
        grade: int = 3,
        filename_arr: Optional[List[str]] = None,
        save: bool = True,
        z_from_cat: bool = False,
        version: str = "v4_4",
        zlabel: str = "z",
    ):
        """Construct a `Spectral_Catalogue` by querying the DAWN JWST
        Archive (DJA) catalogue.

        Loads the DJA source catalogue for the given `version`, optionally
        filters it (by `filename_arr`, or by any combination of `ra_range`,
        `dec_range`, `grade`, `grating_filter`, `z_cat_range`, `PID`), and
        loads a `Spectrum` for each remaining row via `Spectrum.from_DJA`.

        Parameters
        ----------
        ra_range : `list`, `numpy.array` or `astropy.units.Quantity`, optional
            Two-element right ascension range to filter the catalogue to.
            Ignored if `filename_arr` is given. Default is `None`.
        dec_range : `list`, `numpy.array` or `astropy.units.Quantity`, optional
            Two-element declination range to filter the catalogue to.
            Ignored if `filename_arr` is given. Default is `None`.
        PID : `int` or `None`, optional
            JWST program ID to filter the catalogue to. Ignored if
            `filename_arr` is given. Default is `None`.
        z_cat_range : `list`, `numpy.array` or `None`, optional
            Two-element catalogue redshift range to filter the catalogue
            to. If given, `z_from_cat` is set to `True`. Ignored if
            `filename_arr` is given. Default is `None`.
        grating_filter : `str` or `None`, optional
            Grating/filter combination (from
            `NIRSpec.available_grating_filters`) to filter the catalogue
            to. Ignored if `filename_arr` is given. Default is `None`.
        grade : `int`, optional
            DJA redshift quality grade to filter the catalogue to. Ignored
            if `filename_arr` is given. Default is `3`.
        filename_arr : `list` of `str`, optional
            Explicit list of DJA ``"file"`` values to select from the
            catalogue, bypassing all other filtering criteria. Default is
            `None`.
        save : `bool`, optional
            Whether to save a local copy of each downloaded 2D spectrum
            FITS file, passed to `Spectrum.from_DJA`. Default is `True`.
        z_from_cat : `bool`, optional
            Whether to pass the catalogue redshift to `Spectrum.from_DJA`
            for each source. Automatically set to `True` if `z_cat_range`
            is given. Default is `False`.
        version : `str`, optional
            DJA reduction version to query. Default is ``"v4_4"``.
        zlabel : `str`, optional
            Name of the redshift column in the DJA catalogue. Default is
            ``"z"``.

        Returns
        -------
        `Spectral_Catalogue`
            A new `Spectral_Catalogue` containing a `Spectrum` for each
            selected DJA catalogue row.
        """
        if grating_filter is not None:
            if grating_filter not in NIRSpec.available_grating_filters:
                raise InvalidOptionError(
                    f"grating_filter={grating_filter!r} not in "
                    "NIRSpec.available_grating_filters="
                    f"{NIRSpec.available_grating_filters!r}."
                )
        available_versions = ["v1", "v2", "v3", "v4_2", "v4_4"]
        if version not in available_versions:
            raise InvalidOptionError(
                f"version={version!r} not in {available_versions!r}."
            )
        # open and crop catalogue
        # DJA_cat = utils.read_catalog(
        #     config['Spectra']['DJA_CAT_PATH'], format = "ascii.ecsv"
        # )
        cat_path = config["Spectra"]["DJA_CAT_PATH"].replace("v4_4", version)
        if filename_arr is not None:
            # only a handful of rows are typically needed here, so avoid
            # paying the cost of parsing the full (~80k row, ~350MB)
            # catalogue: scan the raw text for the "file" column and only
            # fully parse the matching lines
            filename_set = set(filename_arr)
            with open(cat_path, newline="") as f:
                header_line = f.readline()
                header = next(csv.reader([header_line]))
                file_idx = header.index("file")
                matched_lines = [
                    line
                    for line in f
                    if line.split(",", file_idx + 1)[file_idx]
                    .strip()
                    .strip('"')
                    in filename_set
                ]
            DJA_cat = Table.read(
                header_line + "".join(matched_lines), format="csv"
            )
            mask = np.isin(np.array(DJA_cat["file"]), np.array(filename_arr))
            DJA_cat = DJA_cat[mask]
            # TODO: assertions that these follow the other rules
        else:
            # only read the columns this call actually needs, rather than
            # all 500+ columns in the full catalogue
            needed_cols = {"root", "file"}
            if ra_range is not None:
                needed_cols.add("ra")
            if dec_range is not None:
                needed_cols.add("dec")
            if grade is not None:
                needed_cols.add("grade")
            if grating_filter is not None:
                needed_cols.update({"grating", "filter"})
            if z_cat_range is not None:
                needed_cols.add(zlabel)
            if PID is not None:
                needed_cols.add("PID")
            if z_from_cat:
                needed_cols.add(zlabel)
            DJA_cat = Table.read(
                cat_path,
                include_names=sorted(needed_cols),
                fast_reader=True,
            )
            if ra_range is not None:
                if len(ra_range) != 2:
                    raise LengthMismatchError(
                        f"ra_range={ra_range!r} has length "
                        f"{len(ra_range)}; must have length 2."
                    )
                if type(ra_range) in [list, np.array]:
                    if ra_range[0].unit != ra_range[1].unit:
                        raise InvalidUnitError(
                            f"ra_range={ra_range!r} elements have "
                            f"mismatched units {ra_range[0].unit!r} and "
                            f"{ra_range[1].unit!r}."
                        )
                    ra_range = [
                        ra_range[0].value,
                        ra_range[1].value,
                    ] * ra_range[0].unit
                ra_range = sorted(ra_range.to(u.deg).value)
                galfind_logger.info(
                    f"Filtering DJA_{version} catalogue to RA range "
                    f"{ra_range}. Original size: {len(DJA_cat)}"
                )
                DJA_cat = DJA_cat[
                    (
                        (DJA_cat["ra"] > ra_range[0])
                        & (DJA_cat["ra"] < ra_range[1])
                    )
                ]
                galfind_logger.info(
                    f"Filtered DJA_{version} catalogue to size: {len(DJA_cat)}"
                )

            if dec_range is not None:
                if len(dec_range) != 2:
                    raise LengthMismatchError(
                        f"dec_range={dec_range!r} has length "
                        f"{len(dec_range)}; must have length 2."
                    )
                if type(dec_range) in [list, np.array]:
                    if dec_range[0].unit != dec_range[1].unit:
                        raise InvalidUnitError(
                            f"dec_range={dec_range!r} elements have "
                            f"mismatched units {dec_range[0].unit!r} and "
                            f"{dec_range[1].unit!r}."
                        )
                    dec_range = [
                        dec_range[0].value,
                        dec_range[1].value,
                    ] * dec_range[0].unit
                dec_range = sorted(dec_range.to(u.deg).value)
                galfind_logger.info(
                    f"Filtering DJA_{version} catalogue to Dec range "
                    f"{dec_range}. Original size: {len(DJA_cat)}"
                )
                DJA_cat = DJA_cat[
                    (
                        (DJA_cat["dec"] > dec_range[0])
                        & (DJA_cat["dec"] < dec_range[1])
                    )
                ]
                galfind_logger.info(
                    f"Filtered DJA_{version} catalogue to size: {len(DJA_cat)}"
                )

            if grade is not None:
                galfind_logger.info(
                    f"Filtering DJA_{version} catalogue to grade "
                    f"{grade} sources. Original size: {len(DJA_cat)}"
                )
                DJA_cat = DJA_cat[DJA_cat["grade"] == grade]
                galfind_logger.info(
                    f"Filtered DJA_{version} catalogue to size: {len(DJA_cat)}"
                )

            if grating_filter is not None:
                galfind_logger.info(
                    f"Filtering DJA_{version} catalogue to "
                    f"grating/filter {grating_filter}. Original size: "
                    f"{len(DJA_cat)}"
                )
                if "grating" in DJA_cat.colnames:
                    # TODO: Generalize this!
                    if version in ["v4_2"]:
                        if grating_filter == "PRISM/CLEAR":
                            grating_filter = "PRISM_CLEAR"
                    DJA_cat = DJA_cat[
                        DJA_cat["grating"] == grating_filter.split("/")[0]
                    ]
                if "filter" in DJA_cat.colnames:
                    DJA_cat = DJA_cat[
                        DJA_cat["filter"] == grating_filter.split("/")[1]
                    ]
                galfind_logger.info(
                    f"Filtered DJA_{version} catalogue to size: {len(DJA_cat)}"
                )

            if z_cat_range is not None:
                galfind_logger.info(
                    f"Filtering DJA_{version} catalogue to z range "
                    f"{z_cat_range}. Original size: {len(DJA_cat)}"
                )
                DJA_cat = DJA_cat[
                    (
                        (DJA_cat[zlabel] > z_cat_range[0])
                        & (DJA_cat[zlabel] < z_cat_range[1])
                    )
                ]
                galfind_logger.info(
                    f"Filtered DJA_{version} catalogue to size: {len(DJA_cat)}"
                )
                z_from_cat = True

            if PID is not None:
                galfind_logger.info(
                    f"Filtering DJA_{version} catalogue to PID {PID}. "
                    f"Original size: {len(DJA_cat)}"
                )
                if "PID" in DJA_cat.colnames:
                    DJA_cat = DJA_cat[DJA_cat["PID"] == PID]
                galfind_logger.info(
                    f"Filtered DJA_{version} catalogue to size: {len(DJA_cat)}"
                )
        if z_from_cat:
            return cls(
                [
                    Spectrum.from_DJA(
                        f"{config['Spectra']['DJA_WEB_DIR']}/{root}/{file}",
                        save=save,
                        version=version,
                        z=z,
                        root=root,
                        file=file,
                    )
                    for root, file, z in tqdm(
                        zip(DJA_cat["root"], DJA_cat["file"], DJA_cat[zlabel]),
                        total=len(DJA_cat),
                        desc=f"Loading DJA_{version} catalogue",
                        disable=galfind_logger.getEffectiveLevel()
                        > logging.INFO,
                    )
                ]
            )
        else:
            return cls(
                [
                    Spectrum.from_DJA(
                        f"{config['Spectra']['DJA_WEB_DIR']}/{root}/{file}",
                        save=save,
                        version=version,
                        root=root,
                        file=file,
                    )
                    for root, file in tqdm(
                        zip(DJA_cat["root"], DJA_cat["file"]),
                        total=len(DJA_cat),
                        desc=f"Loading DJA_{version} catalogue",
                        disable=galfind_logger.getEffectiveLevel()
                        > logging.INFO,
                    )
                ]
            )

    def plot(
        self: Self,
        src: str = "msaexp",
    ):
        """Plot every spectrum in the catalogue.

        Calls `Spectrum.plot` for every spectrum belonging to every
        source in the catalogue.

        Parameters
        ----------
        src : `str`, optional
            Plotting method passed to `Spectrum.plot`, either ``"msaexp"``
            or ``"manual"``. Default is ``"msaexp"``.
        """
        for gal in tqdm(self, desc="Plotting spectra"):
            for spec in gal:
                spec.plot(src=src)

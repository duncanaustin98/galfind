#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single astronomical source with photometry,
SED fits, morphology, and cutouts.

Represents one galaxy (or other source) identified in a GALFIND catalogue
with sky position, aperture photometry for multiple diameters, and methods
for creating image cutouts, RGB images, and diagnostic plots.
"""

from __future__ import annotations

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
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.utils.masked import Masked
from matplotlib.patches import Ellipse
from scipy.interpolate import interp1d

if TYPE_CHECKING:
    from ..imaging import Band_Data_Base, PSF_Base
    from . import (
        Band_Cutout,
        Mask_Selector,
        Multiple_Filter,
        SED_code,
        Selector,
    )
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from .. import (
    astropy_cosmo,
    config,
    figs,
    galfind_logger,
)
from ..imaging import Data
from ..photometry import Photometry_obs
from ..sed_fitting.SED_result import SED_result
from ..spectra.SED import SED_obs
from ..utils import useful_funcs_austind as funcs
from ..visualization.Cutout import RGB, Multiple_Band_Cutout

# should be exhaustive
select_func_to_type = {
    "select_min_bands": "data",
    "select_min_unmasked_bands": "data",
    "select_unmasked_instrument": "data",
    "select_phot_galaxy_property_bin": "SED",
    "select_phot_galaxy_property": "SED",
    "phot_bluewards_Lya_non_detect": "phot_rest",
    "phot_redwards_Lya_detect": "phot_rest",
    "phot_Lya_band": "phot_rest",
    "phot_SNR_crop": "phot_obs",
    "select_rest_UV_line_emitters_dmag": "phot_rest",
    "select_rest_UV_line_emitters_sigma": "phot_rest",
    "select_colour": "phot_obs",
    "select_colour_colour": "phot_obs",
    "select_UVJ": "SED",
    "select_Kokorev24_LRDs": "phot_obs",
    "select_depth_region": "data",
    "select_chi_sq_lim": "SED",
    "select_chi_sq_diff": "SED",
    "select_robust_zPDF": "SED",
    "select_band_flux_radius": "morphology",
    "select_EPOCHS": "combined",
    "select_combined": "combined",
}


class Galaxy:
    """A single astronomical source with photometry, SED fits, morphology.

    Includes cutouts for different apertures.

    Represents one galaxy (or other source) identified in a GALFIND
    catalogue. Stores the sky position and aperture photometry (with any
    associated SED fitting results) for one or more aperture diameters,
    and provides methods to make image cutouts and RGB images, and to
    produce diagnostic plots of the photometry, SED fits, and redshift
    PDFs. Selection flags and keyword arguments produced by `Selector`
    objects are attached dynamically and exposed as ordinary attributes
    through `__getattr__` (as are `RA`/`DEC`, derived from `sky_coord`).

    Parameters
    ----------
    ID : `int`
        Unique galaxy identifier within its catalogue.
    sky_coord : `astropy.coordinates.SkyCoord`
        On-sky position of the galaxy.
    aper_phot : `Dict[astropy.units.Quantity, Photometry_obs]`
        Aperture photometry for the galaxy, keyed by aperture diameter.
    selection_flags : `Dict[astropy.units.Quantity, Dict[str, bool]]`, optional
        Boolean selection results keyed by selection name, recording
        whether the galaxy passes each selection criterion. Default is
        `None`, in which case an empty `dict` is used.
    selection_kwargs : `Dict[astropy.units.Quantity, Dict[str, ...]]`, optional
        Keyword arguments associated with each selection, keyed the same
        way as `selection_flags`. Default is `None`, in which case an
        empty `dict` is used.
    cat_filterset : `Multiple_Filter`, optional
        The full filterset of the catalogue the galaxy was loaded from.
        Default is `None`.
    survey : `str`, optional
        Name of the survey/field the galaxy belongs to. Default is `None`.
    simulated : `bool`, optional
        Whether the galaxy originates from simulated (mock) data rather
        than real observations. Default is `False`.

    Attributes
    ----------
    ID : `int`
        Unique galaxy identifier.
    sky_coord : `astropy.coordinates.SkyCoord`
        On-sky position of the galaxy.
    aper_phot : `Dict[astropy.units.Quantity, Photometry_obs]`
        Aperture photometry keyed by aperture diameter.
    selection_flags : `dict`
        Boolean selection results keyed by selection name.
    selection_kwargs : `dict`
        Keyword arguments used to produce each selection.
    cat_filterset : `Multiple_Filter` or `None`
        Filterset of the parent catalogue.
    survey : `str` or `None`
        Survey/field name.
    simulated : `bool`
        Whether this is a simulated source.
    RA : `astropy.units.Quantity`
        Right ascension in degrees, derived from `sky_coord` (accessed
        dynamically via `__getattr__`).
    DEC : `astropy.units.Quantity`
        Declination in degrees, derived from `sky_coord` (accessed
        dynamically via `__getattr__`).
    """

    def __init__(
        self: Self,
        ID: int,
        sky_coord: SkyCoord,
        aper_phot: Dict[u.Quantity, Photometry_obs],
        selection_flags: Optional[Dict[u.Quantity, Dict[str, bool]]] = None,
        selection_kwargs: Optional[
            Dict[u.Quantity, Dict[str, Dict[str, Any]]]
        ] = None,
        cat_filterset: Optional[Multiple_Filter] = None,
        survey: Optional[str] = None,
        version: Optional[str] = None,
        simulated: bool = False,
    ):
        """Initialize a Galaxy instance.

        Parameters
        ----------
        ID : `int`
            Unique identifier for this galaxy.
        sky_coord : `astropy.coordinates.SkyCoord`
            Sky coordinates of the galaxy.
        aper_phot : `dict`
            Dictionary mapping aperture diameters (`astropy.units.Quantity`)
            to `Photometry_obs` objects.
        selection_flags : `dict`, optional
            Selection flags (e.g. detection status) per aperture diameter.
            Default is `None` (empty dict).
        selection_kwargs : `dict`, optional
            Additional selection metadata per aperture diameter. Default is
            `None` (empty dict).
        cat_filterset : `Multiple_Filter`, optional
            Filter set of the source catalogue. Default is `None`.
        survey : `str`, optional
            Name of the survey. Default is `None`.
        version : `str`, optional
            Catalogue version. Default is `None`.
        simulated : `bool`, optional
            Whether this is a simulated galaxy. Default is `False`.
        """
        self.ID = int(ID)
        self.sky_coord = sky_coord
        self.aper_phot = aper_phot
        if selection_flags is None:
            selection_flags = {}
        self.selection_flags = selection_flags
        if selection_kwargs is None:
            selection_kwargs = {}
        self.selection_kwargs = selection_kwargs
        self.cat_filterset = cat_filterset
        self.survey = survey
        self.version = version
        self.simulated = simulated

    @classmethod
    def from_data_id(
        cls,
        data: Data,
        id: int,
        **gal_creator_kwargs,
    ) -> Galaxy:
        """Construct a `Galaxy` for a single ID from a `Data` object.

        Builds a `Galaxy_Creator` for the given catalogue `Data` and
        `id`, then calls it to load the galaxy's photometry, sky
        coordinates, and selection flags from the underlying catalogue.

        Parameters
        ----------
        data : `Data`
            The `Data` object describing the observations/catalogue from
            which to load the galaxy.
        id : `int`
            The catalogue ID of the galaxy to load.
        **gal_creator_kwargs
            Additional keyword arguments forwarded to
            `Galaxy_Creator.from_data`.

        Returns
        -------
        `Galaxy`
            The loaded galaxy, including its PSFs from `data.psfs`.
        """
        from .Galaxy_Creator import Galaxy_Creator

        gal_creator = Galaxy_Creator.from_data(
            data,
            id=int(id),
            **gal_creator_kwargs,
        )
        return gal_creator(psfs=data.psfs)

    def __repr__(self):
        """Return a concise string representation of the galaxy.

        Returns
        -------
        `str`
            String showing class name, ID, and sky coordinates.
        """
        return (
            f"{self.__class__.__name__}({self.ID}, "
            + f"[{self.RA.to(u.deg).value:.5f},"
            + f"{self.DEC.to(u.deg).value:.5f}]deg)"
        )

    def __str__(self):
        """Return a detailed multi-line string representation of the galaxy.

        Returns
        -------
        `str`
            Formatted string showing photometry and selection information.
        """
        output_str = funcs.line_sep
        output_str += f"{repr(self)}\n"
        output_str += funcs.line_sep
        output_str += "PHOTOMETRY:\n"
        output_str += funcs.band_sep
        for phot_obs in self.aper_phot.values():
            output_str += f"{repr(phot_obs)}\n"
        output_str += funcs.band_sep
        output_str += "SELECTION FLAGS:\n"
        output_str += funcs.band_sep
        for i, (select_name, is_selected) in enumerate(
            self.selection_flags.items()
        ):
            output_str += f"{select_name}: {is_selected}\n"
            if i == len(self.selection_flags) - 1:
                output_str += funcs.band_sep
        output_str += funcs.line_sep
        return output_str

    # def __setattr__(self, name, value, obj = "gal"):
    #     if obj == "gal":
    #         if type(name) != list and type(name) != np.array:
    #             super().__setattr__(name, value)
    #         else:
    #             # use setattr to set values within Galaxy dicts (e.g. properties)  # noqa: E501
    #             self.globals()[name[0]][name[1]] = value
    #     else:
    #         raise(Exception(f"obj = {obj} must be 'gal'!"))

    def __getattr__(self, property_name: str) -> Any:
        """Dynamically retrieve galaxy attributes.

        Provides access to RA/DEC from sky coordinate and selection flags
        and kwargs stored in the dictionaries.

        Parameters
        ----------
        property_name : `str`
            Name of the property to retrieve.

        Returns
        -------
        `Any`
            The requested property value.

        Raises
        ------
        AttributeError
            If the property is not found or is a special pickling method.
        """
        # if property_name in self.__dict__.keys():
        #    return self.__getattribute__(property_name)

        # Avoid recursion for pickling-related attributes
        if property_name in {"__getstate__", "__setstate__"}:
            raise AttributeError(property_name)

        if property_name.upper() == "RA":
            return self.sky_coord.ra.degree * u.deg
        elif property_name.upper() == "DEC":
            return self.sky_coord.dec.degree * u.deg
        if property_name in self.selection_flags:
            return self.selection_flags[property_name]
        elif (
            property_name.split("__")[0] in self.selection_kwargs
            and property_name.split("__")[1]
            in self.selection_kwargs[property_name.split("__")[0]]
        ):
            return self.selection_kwargs[property_name.split("__")[0]][
                property_name.split("__")[1]
            ]
        else:
            # if property_name not in [
            #     "__array_struct__",
            #     "__array_interface__",
            #     "__array__",
            # ]:
            #     pass
            # galfind_logger.critical(
            #     f"Galaxy {self.ID=} has no {property_name=}!"
            # )
            raise AttributeError(property_name)
        # else:
        #    return self.phot.__getattr__(property_name, origin)

    def __deepcopy__(self, memo):
        """Create a deep copy of this galaxy.

        Parameters
        ----------
        memo : `dict`
            Dictionary tracking already-copied objects for recursive copying.

        Returns
        -------
        `Galaxy`
            A new Galaxy instance with all attributes deep-copied.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            try:
                setattr(result, key, deepcopy(value, memo))
            except Exception:
                galfind_logger.critical(
                    f"deepcopy({repr(self)}) {key}: {value} FAIL!"
                )
                breakpoint()
        return result

    def update_SED_results(
        self: Self, gal_SED_results: Union[SED_result, List[SED_result]]
    ) -> NoReturn:
        """Attach one or more `SED_result` objects to this galaxy's photometry.

        Each `SED_result` is dispatched to the `Photometry_obs` object in
        `aper_phot` matching its aperture diameter.

        Parameters
        ----------
        gal_SED_results : `SED_result` or `List[SED_result]`
            The SED fitting result(s) to store. A single result is
            wrapped in a list internally.

        Raises
        ------
        AssertionError
            If any `gal_SED_result.aper_diam` is not a key of
            `self.aper_phot`.
        """
        if not isinstance(gal_SED_results, list):
            gal_SED_results = [gal_SED_results]
        missing_aper_diams = [
            gal_SED_result.aper_diam
            for gal_SED_result in gal_SED_results
            if gal_SED_result.aper_diam not in self.aper_phot.keys()
        ]
        assert len(missing_aper_diams) == 0, galfind_logger.critical(
            f"Galaxy {self.ID=} missing "
            + f"{missing_aper_diams} aperture photometry."
        )
        [
            self.aper_phot[gal_SED_result.aper_diam].update_SED_result(
                gal_SED_result
            )
            for gal_SED_result in gal_SED_results
        ]

    def update_SED_result_lowz_zmax_info(
        self, aper_diam, SED_result_key, zmax_info
    ):
        """Forward low-z zmax metadata to the relevant `Photometry_obs` object.

        Parameters
        ----------
        aper_diam : `astropy.units.Quantity`
            Aperture diameter key selecting which `Photometry_obs` object
            in `aper_phot` to update.
        SED_result_key : `str`
            Label of the `SED_result` to update.
        zmax_info : `Any`
            Low-z maximum-redshift information to store on the
            `SED_result`.

        Returns
        -------
        `Any`
            The return value of the underlying
            `Photometry_obs.update_SED_result_lowz_zmax_info` call.
        """
        return self.aper_phot[aper_diam].update_SED_result_lowz_zmax_info(
            SED_result_key, zmax_info
        )

    def load_fixz_SED_result(
        self: Self,
        aper_diam: u.Quantity,
        z_value: float,
        z_label: str = "z",
    ) -> NoReturn:
        """Load a fixed-redshift SED fitting result for one aperture diameter.

        Parameters
        ----------
        aper_diam : `astropy.units.Quantity`
            Aperture diameter key selecting which `Photometry_obs` object
            in `aper_phot` to update.
        z_value : `float`
            The fixed redshift at which the SED fit was performed.
        z_label : `str`, optional
            Label identifying the redshift used, forwarded to
            `Photometry_obs.load_fixz_SED_result`. Default is `"z"`.
        """
        self.aper_phot[aper_diam].load_fixz_SED_result(z_value, z_label)

    def load_property(
        self, gal_property: Union[dict, u.Quantity], save_name: str
    ) -> None:
        """Attach an arbitrary property to the galaxy as a new attribute.

        Parameters
        ----------
        gal_property : `dict` or `astropy.units.Quantity`
            The property value to store.
        save_name : `str`
            Name of the attribute under which `gal_property` is stored.
        """
        setattr(self, save_name, gal_property)

    def make_RGB(
        self: Type[Self],
        data: Data,
        rgb_bands: Dict[str, List[str]] = {
            "B": ["F090W"],
            "G": ["F200W"],
            "R": ["F444W"],
        },
        cutout_size: u.Quantity = 0.96 * u.arcsec,
    ) -> RGB:
        """Build (or retrieve a cached) `RGB` image cutout for this galaxy.

        Results are cached on `self.RGBs`, keyed by cutout size and by the
        band combination used for each colour channel, so repeated calls
        with the same arguments reuse the previously built `RGB` object.

        Parameters
        ----------
        data : `Data`
            The `Data` object providing the band images from which the
            cutouts are drawn.
        rgb_bands : `Dict[str, List[str]]`, optional
            Mapping of colour channel (`"B"`, `"G"`, `"R"`) to the list of
            filter names to combine into that channel. Default combines
            F090W (blue), F200W (green), and F444W (red).
        cutout_size : `astropy.units.Quantity`, optional
            Angular size of the cutout. Default is `0.96 * u.arcsec`.

        Returns
        -------
        `RGB`
            The RGB cutout object for this galaxy and band combination.
        """
        if not hasattr(self, "RGBs"):
            self.RGBs = {}
        cutout_size_str = f"{cutout_size.to(u.arcsec).value:.2f}as"
        if cutout_size_str not in self.RGBs.keys():
            self.RGBs[cutout_size_str] = {}
        rgb_key = ",".join(
            f"{colour}={'+'.join(rgb_bands[colour])}"
            for colour in ["B", "G", "R"]
        )
        if (
            rgb_key
            not in self.RGBs[f"{cutout_size.to(u.arcsec).value:.2f}as"].keys()
        ):
            RGB_obj = RGB.from_gal_data(self, data, rgb_bands, cutout_size)
            assert RGB_obj.name == rgb_key
            self.RGBs[cutout_size_str][rgb_key] = RGB_obj
        return self.RGBs[cutout_size_str][rgb_key]

    def plot_RGB(
        self,
        ax: Optional[plt.Axes],
        rgb_bands: Dict[str, List[str]],
        cutout_size: u.Quantity = 0.96 * u.arcsec,
        method: str = "trilogy",
        rgb_unit: u.Unit = u.uJy,
        rgb_kwargs: Dict[str, Any] = {},
        imshow_kwargs: Dict[str, Any] = {},
    ) -> Optional[List[plt.Text]]:
        """Plot a previously-made RGB cutout for this galaxy on a
        matplotlib axis.

        Requires that `make_RGB` has already been called for the same
        `rgb_bands` and `cutout_size`, since the cached `RGB` object is
        looked up from `self.RGBs` rather than rebuilt.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes` or `None`
            Axis to plot the RGB image on.
        rgb_bands : `Dict[str, List[str]]`
            Mapping of colour channel (`"B"`, `"G"`, `"R"`) to the list of
            filter names combined into that channel, used to look up the
            cached `RGB` object.
        cutout_size : `astropy.units.Quantity`, optional
            Angular size of the cutout, used to look up the cached `RGB`
            object. Default is `0.96 * u.arcsec`.
        method : `str`, optional
            RGB scaling/combination method passed to `RGB.plot`. Default
            is `"trilogy"`.
        rgb_unit : `astropy.units.Unit`, optional
            Flux unit used when combining the bands. Default is `u.uJy`.
        rgb_kwargs : `Dict[str, Any]`, optional
            Extra keyword arguments passed to `RGB.plot`. Default is `{}`.
        imshow_kwargs : `Dict[str, Any]`, optional
            Extra keyword arguments passed through to the underlying
            `imshow` call. Default is `{}`.

        Returns
        -------
        `List[matplotlib.text.Text]` or `None`
            The text label artists returned by `RGB.plot`, if any.
        """
        cutout_size_str = f"{cutout_size.to(u.arcsec).value:.2f}as"
        rgb_key = ",".join(
            f"{colour}={'+'.join(rgb_bands[colour])}"
            for colour in ["B", "G", "R"]
        )
        RGB_obj = self.RGBs[cutout_size_str][rgb_key]
        rgb_labels = RGB_obj.plot(
            ax,
            method,
            unit=rgb_unit,
            rgb_kwargs=rgb_kwargs,
            imshow_kwargs=imshow_kwargs,
        )
        return rgb_labels

    def make_cutouts(
        self: Type[Self],
        data: Data,
        cutout_size: u.Quantity = 0.96 * u.arcsec,
        overwrite: bool = False,
    ) -> Multiple_Band_Cutout:
        """Build (or retrieve a cached) multi-band cutout set for this galaxy.

        The result is cached on `self.multi_band_cutout`; subsequent calls
        return the cached object without regenerating the cutouts.

        Parameters
        ----------
        data : `Data`
            The `Data` object providing the band images from which the
            cutouts are drawn.
        cutout_size : `astropy.units.Quantity`, optional
            Angular size of the cutouts. Default is `0.96 * u.arcsec`.
        overwrite : `bool`, optional
            Whether to overwrite any existing cutout files on disk when
            building the cutouts. Default is `False`.

        Returns
        -------
        `Multiple_Band_Cutout`
            The multi-band cutout object for this galaxy.
        """
        if not hasattr(self, "multi_band_cutout"):
            self.multi_band_cutout = Multiple_Band_Cutout.from_gal_data(
                self, data, cutout_size, overwrite=overwrite
            )
        else:
            galfind_logger.debug(
                f"{self.__class__.__name__} {self.ID=} already has cutouts!"
            )
        return self.multi_band_cutout

    def make_band_cutout(
        self: Type[Self],
        band_data_base: Type[Band_Data_Base],
        cutout_size: u.Quantity = 0.96 * u.arcsec,
        overwrite: bool = False,
    ) -> Band_Cutout:
        """Build (or retrieve a cached) single-band cutout for this galaxy.

        If a `multi_band_cutout` already exists and contains a cutout for
        the requested band and cutout size, that cutout is reused;
        otherwise a new `Band_Cutout` is created. The result is cached in
        `self.cutouts`, keyed by filter name (and cutout size, plus a
        `"_native"` suffix for native-resolution data).

        Parameters
        ----------
        band_data_base : `Band_Data_Base`
            The band data object describing the image to cut out from.
        cutout_size : `astropy.units.Quantity`, optional
            Angular size of the cutout. Default is `0.96 * u.arcsec`.
        overwrite : `bool`, optional
            Whether to overwrite any existing cutout file on disk when
            building the cutout. Default is `False`.

        Returns
        -------
        `Band_Cutout`
            The single-band cutout object for this galaxy and band.
        """
        if not hasattr(self, "cutouts"):
            self.cutouts = {}
        cutout_ = None
        cutout_size_str = f"{cutout_size.to(u.arcsec).value:.2f}as"
        label = f"{band_data_base.filt_name}_{cutout_size_str}"
        if band_data_base.is_native:
            label += "_native"
        if hasattr(self, "multi_band_cutout"):
            # point to relevant cutout that has already been made
            # (multi_band_cutout is a single Multiple_Band_Cutout -- a
            # list-like collection of Band_Cutouts for one cutout_size --
            # not a dict keyed by size, so match on each cutout's own
            # filt_name/cutout_size directly)
            cutout_ = next(
                (
                    cutout
                    for cutout in self.multi_band_cutout
                    if cutout.filt_name == band_data_base.filt_name
                    and cutout.cutout_size == cutout_size
                ),
                None,
            )
        if cutout_ is None:
            # if isinstance(band_data_base, Stacked_Band_Data):
            #     from . import Stacked_Band_Cutout
            #     cutout_ = Stacked_Band_Cutout.from_data_skycoords(
            #         gal.data,
            #         filt,
            #         sky_coords,
            #         cutout_size,
            #         save_path = None,
            #     )
            # else:
            from ..visualization.Cutout import Band_Cutout

            # make the cutout from the relevant band_data
            cutout_ = Band_Cutout.from_gal_band_data(
                self,
                band_data_base,
                cutout_size,
                overwrite=overwrite,
            )
        self.cutouts[label] = cutout_
        return cutout_

    def plot_cutouts(
        self: Type[Self],
        fig: plt.Figure,
        data: Data,
        SED_code: SED_code,
        # hide_masked_cutouts: bool = True,
        cutout_size: u.Quantity = 0.96 * u.arcsec,
        aper_diam: u.Quantity = 0.32 * u.arcsec,
        imshow_kwargs: Dict[str, Any] = {},
        norm_kwargs: Dict[str, Any] = {},
        aper_kwargs: Dict[str, Any] = {},
        kron_kwargs: Dict[str, Any] = {},
        cutout_label_kwargs: Dict[str, Any] = {},
        cutout_gridspec_kwargs: Dict[str, Any] = {},
        n_rows: int = 2,
        incl_nodata_cutouts: bool = False,
        split_by_instr: bool = True,
        split_by_instr_cmap: str = "plasma",
    ):
        """Plot per-band image cutouts for this galaxy in a grid on a figure.

        Builds (or reuses) a `Multiple_Band_Cutout` for the galaxy, adds
        fixed circular aperture regions (and, if SExtractor Kron
        parameters are available, elliptical Kron apertures) to each
        panel, and delegates the actual plotting to
        `Multiple_Band_Cutout.plot`.

        Parameters
        ----------
        fig : `matplotlib.figure.Figure`
            Figure on which the cutout grid is drawn.
        data : `Data`
            The `Data` object providing the band images to cut out and
            plot. If `incl_nodata_cutouts` is `False`, this is first
            deep-copied and cropped to the bands present in the galaxy's
            photometry filterset.
        SED_code : `SED_code`
            SED fitting code/result used when building the cutouts (e.g.
            for redshift-dependent scale bars in `make_cutouts`).
        cutout_size : `astropy.units.Quantity`, optional
            Angular size of each cutout. Default is `0.96 * u.arcsec`.
        aper_diam : `astropy.units.Quantity`, optional
            Aperture diameter used both to select the relevant
            `Photometry_obs` object and to draw the fixed circular
            aperture region on each cutout. Default is `0.32 * u.arcsec`.
        imshow_kwargs : `Dict[str, Any]`, optional
            Extra keyword arguments passed through to the underlying
            `imshow` calls. Default is `{}`.
        norm_kwargs : `Dict[str, Any]`, optional
            Extra keyword arguments controlling the image normalization
            of each cutout. Default is `{}`.
        aper_kwargs : `Dict[str, Any]`, optional
            Style keyword arguments for the fixed circular aperture
            patches (e.g. `linestyle`, `color`). Defaults are filled in
            for `linestyle`, `color`, and `path_effects` if not supplied.
        kron_kwargs : `Dict[str, Any]`, optional
            Style keyword arguments for the Kron ellipse patches, used
            only if the galaxy has SExtractor Kron shape parameters
            loaded. Default is `{}`.
        cutout_label_kwargs : `Dict[str, Any]`, optional
            Extra keyword arguments controlling the per-cutout band-name
            labels, forwarded to `Multiple_Band_Cutout.plot` as
            `label_kwargs`. Default is `{}`.
        cutout_gridspec_kwargs : `Dict[str, Any]`, optional
            Extra keyword arguments controlling the `GridSpec` layout of
            the cutout grid. Default is `{}`.
        n_rows : `int`, optional
            Number of rows to arrange the band cutouts into. Default is 2.
        incl_nodata_cutouts : `bool`, optional
            Whether to include bands with no data for this galaxy rather
            than restricting `data` to bands present in the galaxy's
            filterset. Default is `False`.
        split_by_instr : `bool`, optional
            Whether to visually group/colour cutout panels by instrument.
            Default is `True`.
        split_by_instr_cmap : `str`, optional
            Name of the matplotlib colormap used to distinguish
            instruments when `split_by_instr` is `True`. Default is
            `"plasma"`.

        Returns
        -------
        `numpy.ndarray` of `matplotlib.axes.Axes`
            The array of axes used for the cutout grid, as returned by
            `Multiple_Band_Cutout.plot`.
        """
        if incl_nodata_cutouts:
            data_ = data
        else:
            data_ = deepcopy(data)
            gal_filterset_mask = np.array(
                [
                    True
                    if band_data.filt_name
                    in self.aper_phot[aper_diam].filterset.filt_names
                    else False
                    for band_data in data_
                ]
            )
            data_.band_data_arr = data_[gal_filterset_mask]

        multi_band_cutout = self.make_cutouts(
            data_,
            cutout_size,
        )
        # # make intructions for radii to plot
        # sex_radius = None
        # aper_kwargs = {
        #     "fill": False,
        #     "linestyle": "--",
        #     "lw": 1,
        #     "color": "white",
        #     "zorder": 20,
        # }
        # sex_rad_kwargs = {
        #     "fill": False,
        #     "linestyle": "--",
        #     "lw": 1,
        #     "color": "blue",
        #     "zorder": 20,
        # }
        # breakpoint()
        # if sex_radius is None:
        #     plot_radii = [
        #         [{"radius": aper_diam, "kwargs": aper_kwargs}]
        #         for cutout in multi_band_cutout
        #     ]
        # else:
        #     plot_radii = [
        #         [
        #             {"radius": aper_diam, "kwargs": aper_kwargs},
        #             {
        #                 "radius": self.sex_radius[cutout.filt.filt_name],
        #                 "kwargs": sex_rad_kwargs,
        #             },
        #         ]
        #         for cutout in multi_band_cutout
        #     ]

        # # make instructions for scalebars to plot
        # physical_scalebar_kwargs = {
        #     "loc": "upper left",
        #     "pad": 0.3,
        #     "color": "white",
        #     "frameon": False,
        #     "size_vertical": 1.5,
        # }
        # angular_scalebar_kwargs = {
        #     "loc": "lower right",
        #     "pad": 0.3,
        #     "color": "white",
        #     "frameon": False,
        #     "size_vertical": 2,
        # }
        # z = self.aper_phot[aper_diam].SED_results[SED_code.label].z
        # scalebars = [
        #     {
        #         "physical": {
        #             **physical_scalebar_kwargs,
        #             "z": z,
        #             "pix_length": 10,
        #         }
        #     }
        #     if i == 0
        #     else {"angular": {**angular_scalebar_kwargs, "as_length": 0.3}}
        #     if i == len(multi_band_cutout) - 1
        #     else {}
        #     for i, cutout in enumerate(multi_band_cutout)
        # ]

        # make fixed aperture regions to plot
        aper_kwargs.setdefault("linestyle", "-")
        aper_kwargs.setdefault("color", "green")
        aper_kwargs.setdefault(
            "path_effects", [pe.withStroke(linewidth=3.0, foreground="white")]
        )
        fixed_apertures = np.full(
            len(self.aper_phot[aper_diam]),
            {"aper_diam": aper_diam, **aper_kwargs},
        )
        # make kron aperture regions to plot
        if all(
            hasattr(self, property_name)
            for property_name in [
                "sex_KRON_RADIUS",
                "sex_A_IMAGE",
                "sex_B_IMAGE",
                "sex_THETA_IMAGE",
            ]
        ):
            kron_kwargs = {
                "linestyle": "--",
                "color": "blue",
                "path_effects": [
                    pe.withStroke(linewidth=3.0, foreground="white")
                ],
            }
            kron_apertures = [
                Ellipse(
                    (-99.0, -99.0),
                    width=(self.sex_KRON_RADIUS[filt_name] * self.sex_A_IMAGE)
                    .to(u.dimensionless_unscaled)
                    .value,
                    height=(self.sex_KRON_RADIUS[filt_name] * self.sex_B_IMAGE)
                    .to(u.dimensionless_unscaled)
                    .value,
                    angle=self.sex_THETA_IMAGE.to(u.deg).value,
                    **kron_kwargs,
                )
                for filt_name in self.aper_phot[aper_diam].filterset.filt_names
            ]
            plot_regions = {
                filt_name: [aper, kron]
                for filt_name, aper, kron in zip(
                    self.aper_phot[aper_diam].filterset.filt_names,
                    fixed_apertures,
                    kron_apertures,
                )
            }
        else:
            plot_regions = {
                filt_name: [aper]
                for filt_name, aper in zip(
                    self.aper_phot[aper_diam].filterset.filt_names,
                    fixed_apertures,
                )
            }
        ax_arr = multi_band_cutout.plot(
            fig=fig,
            n_rows=n_rows,
            # fig_scaling: float = 1.5,
            split_by_instr=split_by_instr,
            split_by_instr_cmap=split_by_instr_cmap,
            imshow_kwargs=imshow_kwargs,
            norm_kwargs=norm_kwargs,
            plot_regions=plot_regions,
            label_kwargs=cutout_label_kwargs,
            # scalebars: Optional[Dict] = [],
            # mask: Optional[List[bool]] = None,
            gridspec_kwargs=cutout_gridspec_kwargs,
            show=False,
            save=False,
            close_fig=False,
        )
        return ax_arr

    def _extract_lowz_codes(
        self: Type[Self],
        aper_diam: u.Quantity,
        SED_arr: List[SED_code],
        lowz_dz: float = 0.5,
    ) -> List[SED_code]:
        """Extract low-redshift SED codes compatible with the galaxy's
        redshift.

        Identifies SED fitting codes with restricted maximum redshifts that are
        consistent with this galaxy's fitted redshift, allowing for a tolerance
        specified by ``lowz_dz``.

        Parameters
        ----------
        aper_diam : `astropy.units.Quantity`
            Aperture diameter used to access the galaxy's SED results.
        SED_arr : `list` of `SED_code`
            Array of SED fitting codes to search for low-z variants.
        lowz_dz : `float`, optional
            Redshift tolerance for matching (in units of redshift).
            Default is 0.5.

        Returns
        -------
        `list` of `SED_code`
            Low-redshift SED codes compatible with this galaxy.
        """
        lowz_codes = []
        none_zmax_codes = [
            code
            for code in SED_arr
            if code.SED_fit_params.get("lowz_zmax", False) is None
        ]
        for none_zmax_code in none_zmax_codes:
            none_zmax_code_SED_fit_params = deepcopy(
                none_zmax_code.SED_fit_params
            )
            lowz_zmax_codes = []
            for sed_result in self.aper_phot[aper_diam].SED_results.values():
                code = sed_result.SED_code
                if code.SED_fit_params.get("lowz_zmax", False) is None:
                    continue
                code_SED_fit_params = deepcopy(code.SED_fit_params)
                remove_keys = [
                    key
                    for key in code_SED_fit_params.keys()
                    if any(
                        substr in key.lower() for substr in ["z_max", "zmax"]
                    )
                ]
                for key in remove_keys:
                    del code_SED_fit_params[key]
                    if key in none_zmax_code_SED_fit_params.keys():
                        del none_zmax_code_SED_fit_params[key]
                if code_SED_fit_params == none_zmax_code_SED_fit_params:
                    lowz_zmax_codes.append(code)
            z_gal = (
                self.aper_phot[aper_diam].SED_results[none_zmax_code.label].z
            )
            lowz_zmax = np.array(
                sorted(
                    [
                        lowz_zmax_code.SED_fit_params["lowz_zmax"]
                        for lowz_zmax_code in lowz_zmax_codes
                    ]
                )
            )
            if any(lowz_zmax + lowz_dz < z_gal):
                lowz_codes.extend(
                    [
                        lowz_zmax_codes[
                            np.where(lowz_zmax + lowz_dz < z_gal)[0][-1]
                        ]
                    ]
                )
        return lowz_codes

    def plot_phot_diagnostic(
        self: Type[Self],
        data: Data,
        SED_arr: List[Union[str, SED_code]],
        zPDF_arr: List[Union[str, SED_code]],
        fig: Optional[plt.Figure] = None,
        ax: Optional[Tuple[plt.Figure, List[plt.Axes], List[plt.Axes]]] = None,
        plot_lowz: bool = True,
        lowz_dz: float = 0.5,
        n_cutout_rows: Optional[int] = None,
        wav_unit=u.um,
        flux_unit=u.ABmag,
        log_fluxes: bool = False,
        # hide_masked_cutouts=True,
        cutout_size: u.Quantity = 0.96 * u.arcsec,
        # high_dyn_rng=False,
        annotate_PDFs=True,
        # plot_rejected_reasons=False,
        aper_diam: u.Quantity = 0.32 * u.arcsec,
        imshow_kwargs: Dict[str, Any] = {},
        norm_kwargs: Dict[str, Any] = {},
        aper_kwargs: Dict[str, Any] = {},
        kron_kwargs: Dict[str, Any] = {},
        cutout_label_kwargs: Dict[str, Any] = {},
        sed_plot_kwargs: Dict[str, Any] = {},
        cutout_gridspec_kwargs: Dict[str, Any] = {},
        legend_kwargs: Dict[str, Any] = {"loc": "best"},
        errorbar_kwargs: Dict[str, Any] = {},
        SNR_label_kwargs: Dict[str, Any] = {},
        phot_axes_labels_kwargs: Dict[str, Any] = {},
        phot_axes_ticks_kwargs: Dict[str, Any] = {},
        PDF_axes_ticks_kwargs: Dict[str, Any] = {},
        PDF_axes_labels_kwargs: Dict[str, Any] = {},
        PDF_axes_title_kwargs: Dict[str, Any] = {},
        title_kwargs: Dict[str, Any] = {},
        title: Optional[str] = None,
        split_by_instr: bool = True,
        split_by_instr_cmap: str = "plasma",
        incl_nodata_cutouts: bool = False,
        SED_cmap: Optional[str] = None,
        overwrite: bool = False,
        save: bool = True,
        show: bool = False,
        dpi: int = 300,
        ext: str = "png",
    ) -> Optional[
        Tuple[plt.Figure, plt.Figure, List[plt.Axes], List[plt.Axes]]
    ]:
        """Make and save/show a full diagnostic plot for this galaxy.

        Combines band-image cutouts, best-fit SED curve(s), observed
        photometry, and redshift PDF panel(s) for one or more SED fitting
        codes into a single figure, and writes it to
        `{PLOT_DIR}/{version}/{instrument}/{survey}/SED_plots/{aper_diam}as/{ID}.{ext}`
        unless the file already exists and `overwrite` and `save` do not
        request otherwise.

        Parameters
        ----------
        data : `Data`
            The `Data` object providing the band images and survey/version
            metadata used for the cutouts and output path/title.
        SED_arr : `List[str or SED_code]`
            SED fitting code(s) (or their labels) whose best-fit SEDs and
            mock photometry are plotted on the photometry axis.
        zPDF_arr : `List[str or SED_code]`
            SED fitting code(s) (or their labels) whose redshift PDFs are
            plotted, one per PDF axis.
        fig : `matplotlib.figure.Figure`, optional
            Unused placeholder for a pre-existing figure; if `None` (or
            `ax` is `None`) a new figure/axes layout is created via
            `figs.make_phot_diagnostic_fig`. Default is `None`.
        ax : `tuple`, optional
            Pre-existing `(cutout_fig, phot_ax, PDF_ax)` layout to plot
            onto; if `None`, a new layout is created. Default is `None`.
        plot_lowz : `bool`, optional
            Whether to also include any low-z (`lowz_zmax`) fits related
            to the codes in `SED_arr`/`zPDF_arr`, found via
            `_extract_lowz_codes`. Default is `True`.
        lowz_dz : `float`, optional
            Minimum redshift offset from the primary fit's redshift for a
            low-z fit to be included when `plot_lowz` is `True`. Default
            is 0.5.
        n_cutout_rows : `int`, optional
            Number of rows to arrange the band cutouts into. Must
            currently be given explicitly.
        wav_unit : `astropy.units.Unit`, optional
            Wavelength unit for the x-axis of the photometry/SED plot.
            Default is `u.um`.
        flux_unit : `astropy.units.Unit`, optional
            Flux unit for the y-axis of the photometry/SED plot. Default
            is `u.ABmag`.
        log_fluxes : `bool`, optional
            Whether to plot fluxes on a logarithmic scale. Default is
            `False`.
        cutout_size : `astropy.units.Quantity`, optional
            Angular size of each band cutout. Default is `0.96 * u.arcsec`.
        annotate_PDFs : `bool`, optional
            Whether to annotate the redshift PDF panels (e.g. with peak
            values). Default is `True`.
        aper_diam : `astropy.units.Quantity`, optional
            Aperture diameter selecting which `Photometry_obs` object to
            plot. Default is `0.32 * u.arcsec`.
        imshow_kwargs : `Dict[str, Any]`, optional
            Extra keyword arguments passed through to the cutout `imshow`
            calls. Default is `{}`.
        norm_kwargs : `Dict[str, Any]`, optional
            Extra keyword arguments controlling the cutout image
            normalization. Default is `{}`.
        aper_kwargs : `Dict[str, Any]`, optional
            Style keyword arguments for the fixed circular aperture
            patches on the cutouts. Default is `{}`.
        kron_kwargs : `Dict[str, Any]`, optional
            Style keyword arguments for the Kron ellipse patches on the
            cutouts. Default is `{}`.
        cutout_label_kwargs : `Dict[str, Any]`, optional
            Extra keyword arguments controlling the per-cutout band-name
            labels. Default is `{}`.
        sed_plot_kwargs : `Dict[str, Any]`, optional
            Extra keyword arguments passed to `SED_obs.plot` for the
            best-fit SED curve(s). Default is `{}`.
        cutout_gridspec_kwargs : `Dict[str, Any]`, optional
            Extra keyword arguments controlling the `GridSpec` layout of
            the cutout grid. Default is `{}`.
        legend_kwargs : `Dict[str, Any]`, optional
            Keyword arguments passed to `ax.legend(**legend_kwargs)` for
            the photometry axis legend. Defaults for `loc`, `fontsize`,
            and `frameon` are filled in if not supplied. Default is
            `{"loc": "best"}`.
        errorbar_kwargs : `Dict[str, Any]`, optional
            Keyword arguments passed through to the observed-photometry
            errorbar plot call (`Photometry_obs.plot`). Defaults for
            `ls`, `marker`, `ms`, `zorder`, and `path_effects` are filled
            in if not supplied. Default is `{}`.
        SNR_label_kwargs : `Dict[str, Any]`, optional
            Keyword arguments passed through to `Photometry_obs.plot` for
            styling the per-band SNR annotation labels. Default is `{}`.
        phot_axes_labels_kwargs : `Dict[str, Any]`, optional
            Keyword arguments passed to `ax.set_xlabel`/`ax.set_ylabel` on
            the photometry axis. Default is `{}`.
        phot_axes_ticks_kwargs : `Dict[str, Any]`, optional
            Keyword arguments passed to
            `ax.tick_params(**phot_axes_ticks_kwargs)` on the photometry
            axis. Default is `{}`.
        PDF_axes_ticks_kwargs : `Dict[str, Any]`, optional
            Keyword arguments passed to
            `ax.tick_params(**PDF_axes_ticks_kwargs)` on each redshift
            PDF axis. Default is `{}`.
        PDF_axes_labels_kwargs : `Dict[str, Any]`, optional
            Keyword arguments passed through to the PDF `plot` method as
            `label_kwargs`, controlling the PDF axis labels. Default is
            `{}`.
        PDF_axes_title_kwargs : `Dict[str, Any]`, optional
            Keyword arguments passed to
            `ax.set_title(label, **PDF_axes_title_kwargs)` for each
            redshift PDF panel title. A default `fontsize` of `"medium"`
            is filled in if not supplied. Default is `{}`.
        title_kwargs : `Dict[str, Any]`, optional
            Keyword arguments passed to
            `phot_ax.set_title(title, **title_kwargs)` for the overall
            photometry axis title. Default is `{}`.
        title : `str`, optional
            Title of the photometry axis. If `None`, defaults to
            `"{survey} {ID} ({version})"`. Default is `None`.
        split_by_instr : `bool`, optional
            Whether to visually group/colour cutout panels by instrument.
            Default is `True`.
        split_by_instr_cmap : `str`, optional
            Name of the matplotlib colormap used to distinguish
            instruments when `split_by_instr` is `True`. Default is
            `"plasma"`.
        incl_nodata_cutouts : `bool`, optional
            Whether to include bands with no data for this galaxy in the
            cutout grid. Default is `False`.
        SED_cmap : `str`, optional
            Name of a matplotlib colormap used to assign consistent
            colours to each SED fit in `SED_arr`. If `None`, colours are
            instead taken from each SED line's own default colour.
            Default is `None`.
        overwrite : `bool`, optional
            Whether to regenerate and overwrite the plot if the output
            file already exists. Default is `False`.
        save : `bool`, optional
            Whether to save the figure to `out_path`. Default is `True`.
        show : `bool`, optional
            Whether to display the figure interactively (only used if
            `save` is `True`). Default is `False`.
        dpi : `int`, optional
            Resolution (dots per inch) used when saving the figure.
            Default is 300.
        ext : `str`, optional
            File extension/format used when saving the figure. Default is
            `"png"`.

        Returns
        -------
        `tuple` of (`Figure`, `Figure`, `List[Axes]`, `List[Axes]`)
            The overall figure, the cutout sub-figure, the photometry
            axis, and the list of redshift PDF axes, respectively.

        Raises
        ------
        NotImplementedError
            If `n_cutout_rows` is `None`, since dynamic determination of
            the cutout row count is not yet implemented.
        """

        out_path = (
            f"{config['Other']['PLOT_DIR']}/{data.version}/"
            + f"{data.filterset.instrument_name}/{data.survey}/SED_plots/"
            + f"{aper_diam.to(u.arcsec).value:.2f}as/{self.ID}.{ext}"
        )

        if not Path(out_path).is_file() or overwrite or not save:
            if fig is None or ax is None:
                fig, ax = figs.make_phot_diagnostic_fig(len(data))

            # unpack tuple
            cutout_fig, phot_ax, PDF_ax = ax

            if not isinstance(SED_arr, list):
                SED_arr = [SED_arr]
            if not isinstance(zPDF_arr, list):
                zPDF_arr = [zPDF_arr]
            # extract lowz_zmax from SED_arr if required
            if plot_lowz:
                SED_arr.extend(
                    self._extract_lowz_codes(aper_diam, SED_arr, lowz_dz)
                )
                zPDF_arr.extend(
                    self._extract_lowz_codes(aper_diam, zPDF_arr, lowz_dz)
                )

            zPDF_labels = [
                code.display_label
                if hasattr(code, "display_label")
                else code.label
                for code in zPDF_arr
            ]
            # reset parameters
            default_PDF_axes_title_kwargs = {"fontsize": "medium"}
            for key, value in default_PDF_axes_title_kwargs.items():
                PDF_axes_title_kwargs.setdefault(key, value)
            for ax_, label in zip(PDF_ax, zPDF_labels):
                ax_.set_yticks([])
                ax_.tick_params(**PDF_axes_ticks_kwargs)
                ax_.set_title(label, **PDF_axes_title_kwargs)

            # plot cutouts (assuming reference SED_fit_params is at 0th index)
            if n_cutout_rows is None:
                # determine n_cutout_rows dynamically
                raise NotImplementedError(
                    "Dynamic n_cutout_rows not implemented yet!"
                )
            self.plot_cutouts(
                cutout_fig,
                data,
                SED_arr[0],
                # hide_masked_cutouts=hide_masked_cutouts,
                cutout_size=cutout_size,
                # high_dyn_rng=high_dyn_rng,
                aper_diam=aper_diam,
                imshow_kwargs=imshow_kwargs,
                norm_kwargs=norm_kwargs,
                aper_kwargs=aper_kwargs,
                kron_kwargs=kron_kwargs,
                cutout_label_kwargs=cutout_label_kwargs,
                cutout_gridspec_kwargs=cutout_gridspec_kwargs,
                n_rows=n_cutout_rows,
                incl_nodata_cutouts=incl_nodata_cutouts,
                split_by_instr=split_by_instr,
                split_by_instr_cmap=split_by_instr_cmap,
            )

            # plot specified SEDs and save colours
            if SED_cmap is None:
                SED_colours = {}
            else:
                SED_cmap_ = plt.get_cmap(SED_cmap)
                SED_colours = {
                    code.label: SED_cmap_(
                        i / (len(SED_arr) - 1) if len(SED_arr) > 1 else 0.5
                    )
                    for i, code in enumerate(SED_arr)
                    for i, code in enumerate(SED_arr)
                }
            mock_errorbar_kwargs = {
                "ls": "",
                "marker": "o",
                "ms": 8.0,
                "zorder": 100.0,
                "path_effects": [
                    pe.withStroke(linewidth=2.0, foreground="white")
                ],
            }
            for code in reversed(SED_arr):
                if (
                    self.aper_phot[aper_diam].SED_results[code.label].SED
                    is not None
                ):
                    if code.label in SED_colours.keys():
                        sed_plot_kwargs["color"] = SED_colours.get(code.label)
                    SED_plot = (
                        self.aper_phot[aper_diam]
                        .SED_results[code.label]
                        .SED.plot(
                            phot_ax,
                            wav_unit,
                            flux_unit,
                            log_fluxes=log_fluxes,
                            label=code.display_label
                            if hasattr(code, "display_label")
                            else code.label,
                            plot_kwargs=sed_plot_kwargs,
                        )
                    )
                    if SED_cmap is None:
                        SED_colours[code.label] = SED_plot[0].get_color()
                    # plot the mock photometry
                    self.aper_phot[aper_diam].SED_results[
                        code.label
                    ].SED.create_mock_photometry(
                        self.aper_phot[aper_diam].filterset,
                        depths=self.aper_phot[aper_diam].depths,
                        # min flux pc err = 10.0
                    )
                    self.aper_phot[aper_diam].SED_results[
                        code.label
                    ].SED.mock_photometry.plot(
                        phot_ax,
                        wav_unit,
                        flux_unit,
                        uplim_sigma=None,
                        auto_scale=False,
                        plot_errs={"x": False, "y": False},
                        errorbar_kwargs=mock_errorbar_kwargs,
                        label=None,
                        filled=False,
                        colour=SED_colours[code.label],
                        log_scale=log_fluxes,
                    )
                    # ax_photo.scatter(band_wavs_lowz, band_mags_lowz, edgecolors=eazy_color_lowz, marker='o', facecolor='none', s=80, zorder=4.5)  # noqa: E501

            default_errorbar_kwargs = {
                "ls": "",
                "marker": "o",
                "ms": 4.0,
                "zorder": 100.0,
                "path_effects": [
                    pe.withStroke(linewidth=2.0, foreground="white")
                ],
            }
            for key, value in default_errorbar_kwargs.items():
                errorbar_kwargs.setdefault(key, value)
            self.aper_phot[aper_diam].plot(
                phot_ax,
                wav_unit,
                flux_unit,
                annotate=False,
                # uplim_sigma = 2.0,
                # auto_scale = True,
                # label_SNRs = True,
                errorbar_kwargs=errorbar_kwargs,
                # filled = True,
                # colour = "black",
                # label = "Photometry",
                SNR_labelsize=7.5,
                SNR_label_kwargs=SNR_label_kwargs,
                log_scale=log_fluxes,
            )

            # photometry axis x/y labels and ticks
            phot_ax.set_xlabel(phot_ax.get_xlabel(), **phot_axes_labels_kwargs)
            phot_ax.set_ylabel(phot_ax.get_ylabel(), **phot_axes_labels_kwargs)
            phot_ax.tick_params(**phot_axes_ticks_kwargs)

            # photometry axis title
            if title is None:
                title = f"{data.survey} {self.ID} ({data.version})"
            phot_ax.set_title(title, **title_kwargs)
            # plot rejected reasons somewhere
            # if plot_rejected_reasons:
            #     rejected = str(row[f'rejected_reasons{col_ext}'][0])
            #     if rejected != '':
            #         phot_ax.annotate(rejected, (0.9, 0.95), ha='center', fontsize='small', xycoords = 'axes fraction', zorder=5)  # noqa: E501
            # photometry axis legend
            default_legend_kwargs = {
                "loc": "best",
                "fontsize": 8.0,
                "frameon": True,
            }
            for key, value in default_legend_kwargs.items():
                legend_kwargs.setdefault(key, value)
            phot_ax.legend(**legend_kwargs)
            for text in phot_ax.get_legend().get_texts():
                text.set_path_effects(
                    [pe.withStroke(linewidth=3, foreground="white")]
                )
                text.set_zorder(12)

            # plot PDF on relevant axis
            # again, this is not totally generalized and should be == 2 for now
            # assert (len(zPDF_arr) == len(PDF_ax))
            # could extend to plotting multiple PDFs on the same axis
            for ax, code in zip(PDF_ax, zPDF_arr):
                if code.label in SED_colours.keys():
                    colour = SED_colours[code.label]
                else:
                    colour = "black"
                # load peak value/chi sq from SED_result into redshift PDF
                # TODO: Load this into the PDF when instantiated from
                SED_result
                self.aper_phot[aper_diam].SED_results[
                    code.label
                ].property_PDFs["z"].load_peaks_from_SED_result(
                    self.aper_phot[aper_diam].SED_results[code.label]
                )
                # plot the PDF
                self.aper_phot[aper_diam].SED_results[
                    code.label
                ].property_PDFs["z"].plot(
                    ax,
                    annotate=annotate_PDFs,
                    colour=colour,
                    label_kwargs=PDF_axes_labels_kwargs,
                )

            if save:
                funcs.make_dirs(out_path)
                plt.savefig(out_path, dpi=dpi)  # , bbox_inches="tight")
                funcs.change_file_permissions(out_path)
                if show:
                    plt.show()
                else:
                    # for ax in [phot_ax] + PDF_ax:
                    #     ax.clear()
                    plt.close(fig)

        return fig, cutout_fig, phot_ax, PDF_ax

    # Spectroscopy

    def load_spectra(self, spectra):
        """Attach spectroscopic data to this galaxy.

        Parameters
        ----------
        spectra : `Any`
            Spectroscopic data to store on the galaxy.

        Returns
        -------
        `Galaxy`
            This galaxy instance, with `spectra` set.
        """
        self.spectra = spectra
        return self

    def plot_spec_diagnostic(
        self, ax, grating_filter="PRISM/CLEAR", overwrite=True
    ):
        """Plot a spectroscopic diagnostic for this galaxy, if available.

        Not all galaxies have spectroscopic data attached (via
        `load_spectra`); this method is currently a stub and does not yet
        perform any plotting.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axis intended for the spectroscopic diagnostic plot.
        grating_filter : `str`, optional
            Name of the spectrograph grating/filter combination to plot.
            Default is `"PRISM/CLEAR"`.
        overwrite : `bool`, optional
            Whether to overwrite an existing plot. Default is `True`.
        """
        # bare in mind that not all galaxies have spectroscopic data
        if hasattr(self, "spectra"):
            # plot spectral diagnostic
            pass
        else:
            pass

    # %% Selection methods

    # def is_selected(
    #     self,
    #     selectors: List[Type[Selector]],
    #     aper_diam: u.Quantity,
    #     SED_fit_label: str,
    #     timed: bool = False,
    # ) -> bool:
    #     # # input assertions
    #     # assert type(crop_names) in [str, np.array, list, dict]
    #     # if type(crop_names) in [str]:
    #     #     crop_names = [crop_names]
    #     # if type(incl_selection_types) in [str]:
    #     #     incl_selection_types = [incl_selection_types]
    #     # if incl_selection_types != ["All"]:
    #     #     assert all(
    #     #         fit_type in select_func_to_type.values()
    #     #         for fit_type in incl_selection_types
    #     #     )
    #     # # perform selections if required
    #     # selection_names = []
    #     # if timed:
    #     #     start = time.time()
    #     # for i, crop_name in enumerate(crop_names):
    #     #     func, kwargs, func_type = (
    #     #         Galaxy._get_selection_func_from_output_name(
    #     #             crop_name, SED_fit_params
    #     #         )
    #     #     )
    # # if func_type in incl_selection_types or incl_selection_types == [
    #     #         "All"
    #     #     ]:
    #     #         selection_name = func(self, **kwargs)[1]
    #     #         if selection_name != crop_name:
    #     #             breakpoint()  # see what's wrong
    #     #             assert selection_name == crop_name  # break code
    #     #         selection_names.append(selection_name)
    #     #         run = True
    #     #     else:
    #     #         run = False
    #     #     if timed:
    #     #         print(
    #     #             f"{crop_name=} was {'run' if run else 'skipped'} and took:"  # noqa: E501
    #     #         )
    #     #         if i == 0:
    #     #             mid = time.time()
    #     #             print(f"{mid - start}s")
    #     #         else:
    #     #             print(f"{time.time() - mid}s")
    #     #             mid = time.time()
    #     # breakpoint()
    #     breakpoint()
    #     [selector(self, aper_diam, SED_fit_label, return_copy = False) for selector in selectors]  # noqa: E501
    #     # determine whether galaxy is selected
    #     if all(self.selection_flags[name] for name in selectors):
    #         selected = True
    #     else:
    #         selected = False
    #     # clear selection flags
    #     # self.selection_flags = {}
    #     return selected

    # @staticmethod
    # def _get_selection_func_from_output_name(
    #     name: str, SED_fit_params: Union[str, dict]
    # ):
    #     # only currently works for standard EPOCHS selection!
    #     if type(SED_fit_params) in [str]:
    #         SED_fit_params_key = SED_fit_params
    #         SED_fit_params = globals()[
    #             SED_fit_params_key.split("_")[0]
    #         ]().SED_fit_params_from_label(SED_fit_params_key)
    #         galfind_logger.warning(
    #             f"Galaxy._get_selection_func_from_output_name faster with {type(SED_fit_params)=} == 'dict'"  # noqa: E501
    #         )
    #     else:
    #         SED_fit_params_key = SED_fit_params[
    #             "code"
    #         ].label_from_SED_fit_params(SED_fit_params)
    #     # simplest case
    #     if hasattr(Galaxy, f"select_{name}"):
    #         func = getattr(Galaxy, f"select_{name}")
    #         # default arguments
    #         kwargs = {}
    #     # phot_SNR_crop
    #     elif "bluest_band_SNR" in name or "reddest_band_SNR" in name:
    #         # not yet exhaustive! - have not done band SNR cropping
    #         func = Galaxy.phot_SNR_crop
    #         split_name = name.split("_band_SNR")
    #         sign = split_name[1][0]
    #         assert sign in ["<", ">"]
    #         detect_or_non_detect = "detect" if sign == ">" else "non_detect"
    #         SNR_lim = float(name.split(sign)[1])
    #         if split_name[0] == "bluest":
    #             filt_name_or_index = 0
    #         elif split_name[0] == "reddest":
    #             filt_name_or_index = -1
    #         else:
    #             filt_name_or_index = int(split_name[0].split("_")[0][:-2])
    #             if "bluest" in split_name[0]:
    #                 filt_name_or_index -= 1
    #             else:  # reddest in split_name[0]
    #                 filt_name_or_index *= -1
    #         kwargs = {
    #             "filt_name_or_index": filt_name_or_index,
    #             "SNR_lim": SNR_lim,
    #             "detect_or_non_detect": detect_or_non_detect,
    #         }
    #     # phot_bluewards_Lya_non_detect
    #     elif "bluewards_Lya_SNR<" in name:
    #         func = Galaxy.phot_bluewards_Lya_non_detect
    #         kwargs = {"SNR_lim": float(name.split("bluewards_Lya_SNR<")[1])}
    #     # phot_redwards_Lya_detect
    #     elif "redwards_Lya_SNR>" in name:
    #         func = Galaxy.phot_redwards_Lya_detect
    #         SNR_str = name.split(">")[1].split("_")
    #         if name.split("_")[0] == "ALL":
    #             SNR_lims = float(SNR_str[0])
    #         else:  # name.split("_") == "redwards"
    #             SNR_lims = [float(SNR) for SNR in SNR_str[0].split(",")]
    #         if len(SNR_str) == 1:
    #             widebands_only = False
    #         else:
    #             assert len(SNR_str) == 2
    #             assert SNR_str[1] == "widebands"
    #             widebands_only = True
    #         kwargs = {"SNR_lims": SNR_lims, "widebands_only": widebands_only}
    #     # select_chi_sq_lim
    #     elif "red_chi_sq<" in name:
    #         func = Galaxy.select_chi_sq_lim
    #         kwargs = {"chi_sq_lim": float(name.split("red_chi_sq<")[1])}
    #     # select_chi_sq_diff
    #     elif "chi_sq_diff" in name and ",dz>" in name:
    #         func = Galaxy.select_chi_sq_diff
    #         split_name = name.split(",dz>")
    #         kwargs = {
    #             "chi_sq_diff": float(split_name[0].split(">")[1]),
    #             "delta_z_low_z": float(split_name[1]),
    #         }
    #     # select_robust_zPDF
    #     elif "zPDF>" in name and "%,|dz|/z<" in name:
    #         func = Galaxy.select_robust_zPDF
    #         split_name = name.split("%,|dz|/z<")
    #         kwargs = {
    #             "integral_lim": int(split_name[0].split(">")[1]) / 100,
    #             "delta_z_over_z": float(split_name[1]),
    #         }
    #     # select_min_unmasked_bands
    #     elif "unmasked_bands>" in name:
    #         func = Galaxy.select_min_unmasked_bands
    #         kwargs = {"min_bands": int(name.split(">")[1]) + 1}
    #     # select_unmasked_instrument
    #     elif "unmasked_" in name:
    #         func = Galaxy.select_unmasked_instrument
    #         instrument_name = name.split("_")[1]
    #         assert instrument_name in [
    #             subcls.__name__
    #             for subcls in Instrument.__subclasses__()
    #             if subcls.__name__ != "Combined_Instrument"
    #         ]
    #         kwargs = {"instrument": instr_to_name_dict[instrument_name]}
    #     # select_band_flux_radius
    #     elif "Re_" in name:
    #         func = Galaxy.select_band_flux_radius
    #         split_name = name.split(">")
    #         if len(split_name) == 1:
    #             # no ">" exists
    #             split_name = name.split("<")
    #             gtr_or_less = "less"
    #         else:  # ">" exists
    #             gtr_or_less = "gtr"
    #         band = split_name[0].split("_")[1]
    #         lim_str = split_name[1]
    #         if "pix" in lim_str:
    #             lim = float(lim_str.split("pix")[0]) * u.dimensionless_unscaled  # noqa: E501
    #         else:  # lim in as
    #             lim = float(lim_str.split("as")[0]) * u.arcsec
    #         kwargs = {"band": band, "gtr_or_less": gtr_or_less, "lim": lim}
    #     else:
    #         galfind_logger.critical(
    #             f"Galaxy._get_selection_func_from_output_name could not determine {name=}!"  # noqa: E501
    #         )
    #         raise NotImplementedError
    #     assert func.__name__ in select_func_to_type.keys()
    #     func_type = select_func_to_type[func.__name__]
    #     if func_type in ["SED", "phot_rest", "combined"]:
    #         kwargs["SED_fit_params"] = SED_fit_params
    #     return func, kwargs, func_type

    # Rest-frame SED photometric properties

    # def _calc_SED_rest_property(
    #     self,
    #     SED_rest_property_function,
    #     SED_fit_params_label,
    #     save_dir,
    #     iters,
    #     **kwargs,
    # ):
    #     phot_rest_obj = self.phot.SED_results[SED_fit_params_label].phot_rest
    #     if type(save_dir) == type(None):
    #         save_path = None
    #     else:
    #         save_path = f"{save_dir}/{SED_fit_params_label}/property_name/{self.ID}.ecsv"  # noqa: E501
    #     phot_rest_obj._calc_property(
    #         SED_rest_property_function, iters, save_path=save_path, **kwargs
    #     )[1]
    #     return self

    # def _load_SED_rest_properties(
    #     self,
    #     PDF_dir,
    #     property_names,
    #     SED_fit_params_label=EAZY({"templates": "fsps_larson", "lowz_zmax": None}).label,  # noqa: E501
    # ):
    #     # determine which properties have already been calculated
    #     property_names_to_load = [
    #         property_name
    #         for property_name in property_names
    #         if Path(f"{PDF_dir}/{property_name}/{self.ID}.ecsv").is_file()
    #     ]
    #     PDF_paths = [
    #         f"{PDF_dir}/{property_name}/{self.ID}.ecsv"
    #         for property_name in property_names_to_load
    #     ]
    #     for PDF_path, property_name in zip(PDF_paths, property_names_to_load):  # noqa: E501
    #         self.phot.SED_results[
    #             SED_fit_params_label
    #         ].phot_rest.property_PDFs[property_name] = PDF.from_ecsv(PDF_path)  # noqa: E501
    #         self.phot.SED_results[
    #             SED_fit_params_label
    #         ].phot_rest._update_properties_from_PDF(property_name)
    #     return self

    # def _del_SED_rest_properties(
    #     self,
    #     property_names,
    #     SED_fit_params_label=EAZY({"templates": "fsps_larson", "lowz_zmax": None}).label,  # noqa: E501
    # ):
    #     for property_name in property_names:
    #         self.phot.SED_results[
    #             SED_fit_params_label
    #         ].phot_rest.property_PDFs.pop(property_name)
    #         self.phot.SED_results[
    #             SED_fit_params_label
    #         ].phot_rest.properties.pop(property_name)
    #         self.phot.SED_results[
    #             SED_fit_params_label
    #         ].phot_rest.property_errs.pop(property_name)
    #     return self

    # def _get_SED_rest_property_names(self, PDF_dir):
    #     PDF_paths = glob.glob(f"{PDF_dir}/*/{self.ID}.ecsv")
    #     return [path.split("/")[-2] for path in PDF_paths]

    def load_sextractor_ext_src_corrs(
        self: Self,
        filt_names: Optional[List[str]] = None,
        aper_corrs: Optional[Dict[str, float]] = None,
    ) -> None:
        """Load SExtractor extended-source aperture corrections for
        this galaxy.

        Requires that `FLUX_AUTO` has already been loaded (as
        `sex_FLUX_AUTO`). For each aperture diameter in `aper_phot`, the
        aperture corrections are forwarded to the relevant
        `Photometry_obs.load_sextractor_ext_src_corrs`, and per-filter
        correction values are additionally stored on the galaxy as
        `ext_src_corr_{aper_diam}as_{filter}` attributes.

        Parameters
        ----------
        filt_names : `List[str]`, optional
            Filter names to restrict the correction to. If `None`, all
            filters in `cat_filterset` are used. Default is `None`.
        aper_corrs : `Dict[str, float]`, optional
            Aperture correction values keyed by filter name. Default is
            `None`.

        Raises
        ------
        AttributeError
            If `self.sex_FLUX_AUTO` has not been loaded.
        """
        # FLUX_AUTO must already be loaded
        if not hasattr(self, "sex_FLUX_AUTO"):
            galfind_logger.critical(
                f"Galaxy {self.ID=} has no {self.FLUX_AUTO=}!"
            )
            raise AttributeError("sex_FLUX_AUTO")
        # load ext_src_corrs into aper_phot
        for aper_diam in self.aper_phot.keys():
            self.aper_phot[aper_diam].load_sextractor_ext_src_corrs(
                filt_names, aper_corrs=aper_corrs
            )
            for filt in self.cat_filterset:
                if filt_names is None or filt.filt_name in filt_names:
                    setattr(
                        self,
                        f"ext_src_corr_{aper_diam.to(u.arcsec).value:.2f}as_{filt.filt_name}",
                        getattr(
                            self.aper_phot[aper_diam],
                            f"ext_src_corr_{filt.filt_name}",
                            np.nan,
                        ),
                    )

    def set_Vmax(
        self: Self,
        ecsv_rows: Table,
        data_survey: str,
    ) -> None:
        """Load precomputed V_max values for this galaxy from an ecsv table.

        For each combination of aperture diameter and SED fitting code
        present in `ecsv_rows` that also exists in this galaxy's
        `aper_phot`/`SED_results`, sums the per-row `Vmax_total` values
        for each region and stores them on the corresponding
        `SED_result.Vmax[data_survey][region]`.

        Parameters
        ----------
        ecsv_rows : `astropy.table.Table`
            Table of rows containing (at least) the `aper_diam`,
            `SED_fit_code`, `region`, and `Vmax_total` columns for this
            galaxy.
        data_survey : `str`
            Name of the survey/field the Vmax values were computed in,
            used as the key under which they are stored.

        Raises
        ------
        AssertionError
            If `ecsv_rows` is missing any of the required columns, or if
            no rows exist in `ecsv_rows` for a given aperture diameter /
            SED fitting code combination present in the galaxy.
        """
        try:
            assert all(
                colname in ecsv_rows.colnames
                for colname in [
                    "aper_diam",
                    "SED_fit_code",
                    "region",
                    "Vmax_total",
                ]
            ), galfind_logger.critical(
                f"{repr(self)=} ecsv_rows missing cols for Vmax! "
                f"{ecsv_rows.colnames=}"
            )
        except Exception as e:
            print(e)
            breakpoint()
            raise e
        # TODO: Currently only stores Vmax for a single redshift / selection
        aper_diam_arr = np.unique(ecsv_rows["aper_diam"].data) * u.arcsec
        SED_fit_code_arr = np.unique(ecsv_rows["SED_fit_code"].data)
        for aper_diam in aper_diam_arr:
            if aper_diam in self.aper_phot.keys():
                aper_phot_obj = self.aper_phot[aper_diam]
                for SED_fit_code in SED_fit_code_arr:
                    if SED_fit_code in aper_phot_obj.SED_results.keys():
                        SED_result_obj = aper_phot_obj.SED_results[
                            SED_fit_code
                        ]
                        if not hasattr(SED_result_obj, "Vmax"):
                            # SED_result_obj.zmin = {}
                            # SED_result_obj.zmax = {}
                            SED_result_obj.Vmax = {}
                        if data_survey not in SED_result_obj.Vmax.keys():
                            SED_result_obj.Vmax[data_survey] = {}
                        # crop ecsv tab to just the relevant rows for this
                        # aper_diam and SED_fit_code combination
                        ecsv_row = ecsv_rows[
                            (
                                (
                                    ecsv_rows["aper_diam"]
                                    == aper_diam.to(u.arcsec).value
                                )
                                & (ecsv_rows["SED_fit_code"] == SED_fit_code)
                            )
                        ]
                        assert len(ecsv_row) > 0, galfind_logger.critical(
                            f"Galaxy {self.ID=} has no rows in "
                            + f"ecsv for {aper_diam:.2f} {SED_fit_code=}"
                        )
                        regions = np.unique(ecsv_row["region"].data)
                        for region in regions:
                            region_row = ecsv_row[ecsv_row["region"] == region]
                            # TODO: save the Vmax kwargs in the object
                            SED_result_obj.Vmax[data_survey][region] = np.sum(
                                region_row["Vmax_total"].data
                            )
                    else:
                        galfind_logger.debug(
                            f"Galaxy {self.ID=} has no SED_result for "
                            f"{aper_diam:.2f} {SED_fit_code=}, "
                            f"cannot append Vmax!"
                        )
            else:
                galfind_logger.debug(
                    f"Galaxy {self.ID=} has no aper_phot for "
                    + f"{aper_diam:.2f}, cannot append Vmax!"
                )

    # Vmax calculation in a single field
    def calc_Vmax(
        self: Self,
        data: Data,
        z_bin: List[float],
        aper_diam: u.Quantity,
        SED_fit_code: SED_code,
        crops: List[Type[Selector]],
        z_step: float = 0.01,
        depth_mode: str = "n_nearest",
        unmasked_area: Union[
            str, List[str], u.Quantity, Type[Mask_Selector]
        ] = "selection",
        Vmax_method: str = "uniform_depth",
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Calculate the maximum observable volume (
            V_max) for this galaxy in a field.

        Dispatches to a `calc_Vmax_{Vmax_method}` method (e.g.
        `calc_Vmax_split_region`) after filtering out selection `crops`
        that depend on data/morphology or on SED fitting that would need
        to be redone at each trial redshift, since only crops applicable
        purely from the (redshifted) photometry are meaningful for a
        V_max calculation.

        Parameters
        ----------
        data : `Data`
            The `Data` object providing the depths and area information
            for the field the V_max is computed in.
        z_bin : `List[float]`
            Two-element `[zmin, zmax]` redshift bin to compute the volume
            over.
        aper_diam : `astropy.units.Quantity`
            Aperture diameter selecting which `Photometry_obs`/
            `SED_result` to use.
        SED_fit_code : `SED_code`
            SED fitting code whose result (and redshift) is used.
        crops : `List[Selector]`
            Selection criteria the galaxy must continue to satisfy at
            each trial redshift; crops based on data/morphology or on
            SED-fit-dependent properties are automatically excluded.
        z_step : `float`, optional
            Redshift step size used when scanning the trial redshift
            range. Default is 0.01.
        depth_mode : `str`, optional
            Depth-measurement mode to use (e.g. `"n_nearest"`). Default is
            `"n_nearest"`.
        unmasked_area : `str` or `Mask_Selector`, optional
            Specification of the unmasked area to use; either a
            precomputed `astropy.units.Quantity` area, a `Mask_Selector`,
            or `"selection"` to use the forced photometry band's mask.
            Default is `"selection"`.
        Vmax_method : `str`, optional
            Name suffix of the `calc_Vmax_*` method to dispatch to.
            Default is `"uniform_depth"`.

        Returns
        -------
        `tuple` of (`dict`, `dict`, `dict`)
            The V_max value(s), the keyword arguments used to compute
            them, and metadata about the calculation, as returned by the
            dispatched `calc_Vmax_{Vmax_method}` method. If the galaxy's
            redshift falls outside `z_bin`, V_max is `-1.0` for all
            regions.

        Raises
        ------
        AssertionError
            If there is no `calc_Vmax_{Vmax_method.lower()}` method, if
            `z_bin` does not have exactly two elements with
            `z_bin[0] < z_bin[1]`, if `crops` is empty, or if no crops
            remain after excluding data/morphology/SED-fit-dependent
            selectors.
        """
        assert hasattr(
            self, f"calc_Vmax_{Vmax_method.lower()}"
        ), galfind_logger.critical(
            f"Galaxy.calc_Vmax has no method for {Vmax_method.lower()=}! "
            + f"Please implement Galaxy.calc_Vmax_{Vmax_method.lower()}"
        )

        # TODO: remove dependence on full_survey_name input
        # input assertions
        assert len(z_bin) == 2
        assert z_bin[0] < z_bin[1]
        assert crops != []
        from . import (
            Data_Selector,
            Multiple_Selector,
            Rest_Frame_Property_Limit_Selector,
            SED_fit_Selector,
        )

        sed_result = self.aper_phot[aper_diam].SED_results[SED_fit_code.label]
        # flatten multiple selectors and remove
        # SED_fit_selectors that require SED fitting from crops
        multiple_selectors = tuple(Multiple_Selector.__subclasses__())
        while any(isinstance(crop, multiple_selectors) for crop in crops):
            crops_ = []
            for crop in crops:
                if isinstance(crop, multiple_selectors):
                    crops_.extend([i for i in crop])
                else:
                    crops_.extend([crop])
            crops = crops_
        # TODO: generalize this! i.e. re-perform new morphological fits
        # and SED fitting with redshifted photometry
        Vmax_crops = []
        for crop in crops:
            # skip all selection based on data and morphology
            if isinstance(crop, tuple(Data_Selector.__subclasses__())):
                continue
            # skip all selection requiring SED fitting
            elif isinstance(crop, tuple(SED_fit_Selector.__subclasses__())):
                if crop.requires_SED_fit or isinstance(
                    crop, Rest_Frame_Property_Limit_Selector
                ):
                    continue
            Vmax_crops.extend([crop])
        assert Vmax_crops != []
        # calculate V_max
        if sed_result.z > z_bin[1] or sed_result.z < z_bin[0]:
            Vmax = {"all": -1.0}
            Vmax_kwargs = {"all": {"zmin": -1.0, "zmax": -1.0}}
            meta = {}
        else:
            calc_Vmax_func = getattr(self, f"calc_Vmax_{Vmax_method.lower()}")
            Vmax, Vmax_kwargs, meta = calc_Vmax_func(
                data,
                z_bin,
                aper_diam,
                SED_fit_code,
                Vmax_crops,
                z_step,
                depth_mode,
                unmasked_area,
            )
        return Vmax, Vmax_kwargs, meta

    def calc_Vmax_split_region(
        self: Self,
        data: Data,
        z_bin: List[float],
        aper_diam: u.Quantity,
        SED_fit_code: SED_code,
        Vmax_crops: List[Type[Selector]],
        z_step: float = 0.01,
        depth_mode: str = "n_nearest",
        unmasked_area: Union[
            str, List[str], u.Quantity, Type[Mask_Selector]
        ] = "selection",
    ) -> Tuple[float, float, float]:
        """Calculate V_max for this galaxy separately in each field region.

        For each region in `data.regions` (or a single `"all"` region if
        `data` has none defined) and each redshift sub-bin over which the
        unmasked area has been tabulated in `data.area_depths`, computes
        the contribution to V_max using `_compute_Vmax` with the median
        per-band depths for that region.

        Parameters
        ----------
        data : `Data`
            The `Data` object providing per-band median depths
            (`med_depth`), region definitions, and area-vs-depth
            information (`area_depths`).
        z_bin : `List[float]`
            Two-element `[zmin, zmax]` redshift bin to compute the volume
            over.
        aper_diam : `astropy.units.Quantity`
            Aperture diameter used to look up per-band median depths.
        SED_fit_code : `SED_code`
            SED fitting code passed through to `_compute_Vmax`.
        Vmax_crops : `List[Selector]`
            Pre-filtered selection criteria to re-apply at each trial
            redshift.
        z_step : `float`, optional
            Redshift step size used when scanning the trial redshift
            range. Default is 0.01.
        depth_mode : `str`, optional
            Depth-measurement mode (currently unused directly in this
            method but accepted for interface consistency). Default is
            `"n_nearest"`.
        unmasked_area : `str`, `astropy.units.Quantity`, or
        `Mask_Selector`, optional
            Specification of the unmasked area to use for each redshift
            sub-bin; either a fixed `astropy.units.Quantity`, a
            `Mask_Selector`, or `"selection"` to use the forced photometry
            band's mask. Default is `"selection"`.

        Returns
        -------
        `tuple` of (`dict`, `dict`, `dict`)
            Nested dictionaries (keyed by region, then by redshift
            sub-bin label) of the total V_max, the V_max keyword
            arguments, and metadata (depths and area used), respectively.

        Raises
        ------
        AssertionError
            If any band in `data` is missing `med_depth` attributes or
            median depths for a given region.
        NotImplementedError
            If `unmasked_area` is not a recognised type/value.
        """
        from ..selection import Mask_Selector

        if hasattr(data, "regions"):
            galfind_logger.debug(
                f"Calculating split Vmax for {repr(self)} in {data.regions=}"
            )
            regions = data.regions
        else:
            galfind_logger.debug(
                f"Calculating Vmax for {repr(self)} in {data.full_name=}"
            )
            regions = ["all"]
            setattr(data, "regions", regions)

        Vmax_output = {}  # total Vmax outputs
        Vmax_kwargs_output = {}  # galaxy-dependent Vmax kwargs output

        meta = {
            "z_bin": z_bin,
        }  # data shared by all galaxies in this field

        if isinstance(unmasked_area, u.Quantity):
            area = unmasked_area
            calc_area = False
        else:
            calc_area = True

        for region in regions:
            assert all(
                hasattr(band_data, "med_depth") for band_data in data
            ), galfind_logger.critical(
                f"Not all bands in {repr(data)=} have med_depth attributes!"
            )
            assert all(
                region in band_data.med_depth[aper_diam].keys()
                for band_data in data
            ), galfind_logger.critical(
                f"Not all bands in {repr(data)=} have med_depths for {region=}"
            )

            Vmax_output[region] = {}
            Vmax_kwargs_output[region] = {}
            meta[region] = {}

            # extract zbins from data.area_depths keys
            area_zbins = []
            area_zbin_labels = []
            for key in data.area_depths[region].keys():
                split_key = key.split("<z<")
                if len(split_key) == 2:
                    area_zbins.append(
                        [float(split_key[0]), float(split_key[1])]
                    )
                    area_zbin_labels.append(key)

            for area_zbin, area_zbin_label in zip(
                area_zbins, area_zbin_labels
            ):
                if calc_area:
                    if issubclass(unmasked_area.__class__, Mask_Selector):
                        mask_selector = unmasked_area
                    elif unmasked_area == "selection":
                        mask_selector = data.forced_phot_band.filt_name
                    else:
                        raise NotImplementedError(
                            f"{unmasked_area=} not implemented for "
                            + "Galaxy.calc_Vmax_split_region!"
                        )
                    mask_selector_name = data._get_mask_selector_name(
                        mask_selector, region, area_zbin
                    )
                    area = data.unmasked_area[mask_selector_name]["MASK"]
                depths = [
                    band_data.med_depth[aper_diam][region]
                    for band_data in data
                ]
                # determine appropriate min and max redshift to compute the Vmax over  # noqa: E501
                # in this redshift bin given the zbin the area has been computed over  # noqa: E501
                zmin = np.max([z_bin[0], area_zbin[0]])
                zmax = np.min([z_bin[1], area_zbin[1]])
                # compute Vmax for this galaxy
                Vmax_, Vmax_kwargs_ = self._compute_Vmax(
                    zmin,
                    zmax,
                    filterset=data.filterset,
                    depths=depths,
                    aper_diam=aper_diam,
                    psfs=data.psfs,
                    SED_fit_code=SED_fit_code,
                    Vmax_crops=Vmax_crops,
                    area=area,
                    z_step=z_step,
                )
                Vmax_output[region][area_zbin_label] = np.clip(
                    Vmax_, 0.0, None
                )
                Vmax_kwargs_output[region][area_zbin_label] = Vmax_kwargs_
                meta[region][area_zbin_label] = {
                    "depths": depths,
                    "area": area,
                }
        return Vmax_output, Vmax_kwargs_output, meta

    #             if self.survey == data.survey and self.region == region:
    #                 # use local depths
    #                 data_depths = self.aper_phot[aper_diam].depths.value
    #             else:
    #                 # use median depths in field region
    #                 data_depths = [band_data.med_depth[aper_diam][region] for band_data in data]  # noqa: E501
    #             # calculate z_range
    #             # z_test for other fields should be lower than starting z
    #             z_detect = []
    #             z_arr = np.arange(z_bin[0], z_bin[1] + z_step, z_step)
    #             for z in z_arr: #, tqdm(
    #             #     np.arange(z_bin[0], z_bin[1] + z_step, z_step),
    #             #     desc=f"Calculating z_max and z_min for ID={self.ID}",
    #             # ):
    #                 galfind_logger.debug(
    #                     "ΙGM attenuation is ignored when redshifting the best-fit galaxy SED!"  # noqa: E501
    #                 )
    #                 wav_z = sed_obs.wavs * ((1.0 + z) / (1.0 + sed_obs.z))
    #                 distance_test = astropy_cosmo.luminosity_distance(z)
    #                 mag_z = (
    #                     funcs.convert_mag_units(
    #                         sed_obs.wavs, sed_obs.mags, u.ABmag
    #                     ).value
    #                     - (
    #                         5
    #                         * np.log10(
    #                             (distance_detect / distance_test)
    #                             .to(u.dimensionless_unscaled)
    #                             .value
    #                         )
    #                     )
    #                     + (2.5 * np.log10((1.0 + sed_obs.z) / (1.0 + z)))
    #                 )
    #                 # construct galaxy at new redshift with average depths of new field  # noqa: E501
    #                 test_sed_obs = SED_obs(
    #                     z, wav_z.value, mag_z, wav_z.unit, u.ABmag
    #                 )
    #                 galfind_logger.debug("Not propagating min_flux_pc_err! Setting to 10.0%")  # noqa: E501
    #                 test_mock_phot = test_sed_obs.create_mock_photometry(
    #                     data.filterset,
    #                     depths=data_depths,
    #                     min_flux_pc_err=10.0,
    #                 )
    #                 test_phot_obs = Photometry_obs(
    #                     test_mock_phot.filterset,
    #                     Masked(
    #                         test_mock_phot.flux,
    #                         mask=np.full(len(data.filterset), False),
    #                     ),
    #                     Masked(
    #                         test_mock_phot.flux_errs,
    #                         mask=np.full(len(data.filterset), False),
    #                     ),
    #                     test_mock_phot.depths,
    #                     aper_diam,
    #                 )
    #                 sed_result = SED_result(
    #                     SED_fit_code,
    #                     test_phot_obs,
    #                     {"z": z},
    #                     property_errs = {},
    #                     property_PDFs = {},
    #                     SED = None,
    #                 )
    #                 test_phot_obs.SED_results = {
    #                     SED_fit_code.label: sed_result
    #                 }
    #                 test_gal = Galaxy(
    #                     self.ID,
    #                     self.sky_coord,
    #                     {aper_diam: test_phot_obs},
    #                     selection_flags = {},
    #                     selection_kwargs = {},
    #                 )
    #                 # run selection methods on new galaxy
    #                 [selector(test_gal, return_copy = False) for selector in Vmax_crops]  # noqa: E501
    #                 goodz = all(
    #                     test_gal.selection_flags[selector.name]
    #                     for selector in Vmax_crops
    #                 )
    #                 if goodz:
    #                     z_detect.append(z)

    #             z_detect = np.array(z_detect)
    #             if len(z_detect) < 2:
    #                 z_max = -1.0
    #                 z_min = -1.0
    #                 # V_max_simple = -1.
    #                 V_max = -1.0
    #             else:
    #                 z_max = np.round(z_detect[-1], 3)
    #                 z_min = np.round(z_detect[0], 3)
    #
    #                 # compute cumulative area vs depth
    #                     V_max = funcs.calc_Vmax(
    #                         area,
    #                         z_min_used,
    #                         z_max_used
    #                     ).to(u.Mpc**3).value

    #             z_min_used = np.max([z_min, z_bin[0]])
    #             z_max_used = np.min([z_max, z_bin[1]])

    #         return z_min_used, z_max_used, V_max

    def calc_Vmax_area_depth_scaled_split_region(
        self: Self,
        data: Data,
        z_bin: List[float],
        aper_diam: u.Quantity,
        SED_fit_code: SED_code,
        Vmax_crops: List[Type[Selector]],
        z_step: float = 0.01,
        depth_mode: str = "n_nearest",
        unmasked_area: Union[
            str, List[str], u.Quantity, Type[Mask_Selector]
        ] = "selection",
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Calculate V_max for this galaxy,
        scaling depths across the unmasked area.

        Similar to `calc_Vmax_split_region`, but rather than using a
        single median depth per region/redshift sub-bin, interpolates the
        forced-photometry-band depth across `n_steps` steps of the
        cumulative unmasked area, derives the corresponding depths of the
        other bands via their own cumulative-area curves, and sums the
        V_max contribution from each depth step (each with its own area
        increment).

        Parameters
        ----------
        data : `Data`
            The `Data` object providing per-band median depths
            (`med_depth`), region definitions, and area-vs-depth curves
            (`area_depths`, with `"cum_dist"` and `"total_depths"` per
            band).
        z_bin : `List[float]`
            Two-element `[zmin, zmax]` redshift bin to compute the volume
            over.
        aper_diam : `astropy.units.Quantity`
            Aperture diameter used to look up per-band median depths.
        SED_fit_code : `SED_code`
            SED fitting code passed through to `_compute_Vmax`.
        Vmax_crops : `List[Selector]`
            Pre-filtered selection criteria to re-apply at each trial
            redshift.
        z_step : `float`, optional
            Redshift step size used when scanning the trial redshift
            range. Default is 0.01.
        depth_mode : `str`, optional
            Depth-measurement mode (currently unused directly in this
            method but accepted for interface consistency). Default is
            `"n_nearest"`.
        unmasked_area : `str` or `Mask_Selector`, optional
            Specification of the unmasked area (currently unused directly
            in this method, since area is instead derived from
            `data.area_depths`). Default is `"selection"`.

        Returns
        -------
        `tuple` of (`dict`, `dict`, `dict`)
            Nested dictionaries (keyed by region, then by redshift
            sub-bin label) of the total V_max summed over depth steps,
            the per-step V_max keyword arguments array, and metadata
            (per-step depths and cumulative area), respectively.

        Raises
        ------
        AssertionError
            If any band in `data` is missing `med_depth` attributes or
            median depths for a given region.
        """
        # load appropriate depths for each data object in data_arr
        galfind_logger.debug(
            "Should use local depth if the data.full_name "
            + "is the same as the catalogue the galaxy is measured in!"
        )
        if hasattr(data, "regions"):
            galfind_logger.debug(
                f"Calculating split Vmax for {repr(self)} in {data.regions=}"
            )
            regions = data.regions
        else:
            galfind_logger.debug(
                f"Calculating Vmax for {repr(self)} in {data.full_name=}"
            )
            regions = ["all"]
            setattr(data, "regions", regions)

        Vmax_output = {}  # total Vmax outputs
        Vmax_kwargs_output = {}  # galaxy-dependent Vmax kwargs

        n_steps = 10
        meta = {
            "z_bin": z_bin,
            "z_step": z_step,
            "n_depth_steps": n_steps,
        }  # data shared by all galaxies in this field

        for region in regions:
            assert all(
                hasattr(band_data, "med_depth") for band_data in data
            ), galfind_logger.critical(
                f"Not all bands in {repr(data)=} have med_depth attributes!"
            )
            assert all(
                region in band_data.med_depth[aper_diam].keys()
                for band_data in data
            ), galfind_logger.critical(
                f"Not all bands in {repr(data)=} have med_depths for {region=}"
            )

            Vmax_output[region] = {}
            Vmax_kwargs_output[region] = {}
            meta[region] = {}

            # extract zbins from data.area_depths keys
            area_zbins = []
            area_zbin_labels = []
            for key in data.area_depths[region].keys():
                split_key = key.split("<z<")
                if len(split_key) == 2:
                    area_zbins.append(
                        [float(split_key[0]), float(split_key[1])]
                    )
                    area_zbin_labels.append(key)

            for area_zbin, area_zbin_label in zip(
                area_zbins, area_zbin_labels
            ):
                # interpolate depths of forced_phot_band n_steps times
                forced_phot_band_depths = np.linspace(
                    np.max(
                        data.area_depths[region][area_zbin_label][
                            "total_depths"
                        ][data.forced_phot_band.filt_name]
                    ),
                    np.min(
                        data.area_depths[region][area_zbin_label][
                            "total_depths"
                        ][data.forced_phot_band.filt_name]
                    ),
                    n_steps,
                )
                # depths[data.forced_phot_band.filt_name] =
                forced_phot_band_depths
                cum_area_arr = interp1d(
                    data.area_depths[region][area_zbin_label]["total_depths"][
                        data.forced_phot_band.filt_name
                    ],
                    data.area_depths[region][area_zbin_label]["cum_dist"][
                        data.forced_phot_band.filt_name
                    ],
                )(forced_phot_band_depths)
                # use interpolated forced_phot_band area to get depths of other bands at each step  # noqa: E501
                # for band in data.filterset:
                # if band.filt_name == data.forced_phot_band.filt_name:
                #     continue
                depths = {
                    filt.filt_name: interp1d(
                        data.area_depths[region][area_zbin_label]["cum_dist"][
                            filt.filt_name
                        ],
                        data.area_depths[region][area_zbin_label][
                            "total_depths"
                        ][filt.filt_name],
                        fill_value="extrapolate",
                    )(cum_area_arr)
                    for filt in data.filterset
                }
                # determine appropriate min and max redshift to compute the Vmax over  # noqa: E501
                # in this redshift bin given the zbin the area has been computed over  # noqa: E501
                zmin = np.max([z_bin[0], area_zbin[0]])
                zmax = np.min([z_bin[1], area_zbin[1]])
                Vmax_arr = np.zeros(n_steps)
                Vmax_kwargs_arr = np.full((n_steps,), {})
                cum_area_arr = cum_area_arr * u.arcmin**2
                depth_step_arr = np.zeros((n_steps, len(data.filterset)))
                for i in range(n_steps):
                    depth_step = [
                        depths[filt.filt_name][i] for filt in data.filterset
                    ]
                    if i == 0:
                        area_step = cum_area_arr[i]
                    else:
                        area_step = cum_area_arr[i] - cum_area_arr[i - 1]
                    # compute Vmax for this galaxy using these depths and area
                    Vmax_, Vmax_kwargs_ = self._compute_Vmax(
                        zmin,
                        zmax,
                        filterset=data.filterset,  # TODO: use aper_phot?
                        depths=depth_step,
                        aper_diam=aper_diam,
                        psfs=data.psfs,
                        SED_fit_code=SED_fit_code,
                        Vmax_crops=Vmax_crops,
                        area=area_step,
                        z_step=z_step,
                    )
                    Vmax_arr[i] = Vmax_
                    Vmax_kwargs_arr[i] = Vmax_kwargs_
                    depth_step_arr[i] = depth_step
                    # print(f"Depth step {i+1}/{n_steps}: Vmax = {Vmax_arr[i]:.2f} Mpc^3")  # noqa: E501
                # print(f"Vmax steps for {area_zbin_label}: {Vmax_arr}")
                # print(f"Total Vmax for {area_zbin_label} = {np.sum(np.clip(Vmax_arr, 0.0, None)):.2f} Mpc^3")  # noqa: E501
                Vmax_output[region][area_zbin_label] = np.sum(
                    np.clip(Vmax_arr, 0.0, None)
                )
                Vmax_kwargs_output[region][area_zbin_label] = Vmax_kwargs_arr
                meta[region][area_zbin_label] = {
                    "depth_step_arr": depth_step_arr,
                    "cum_area_arr": cum_area_arr,
                }
        return Vmax_output, Vmax_kwargs_output, meta

    def _compute_Vmax(
        self: Self,
        zmin: float,
        zmax: float,
        filterset: Multiple_Filter,
        depths: List[float],
        aper_diam: u.Quantity,
        psfs: Optional[Dict[str, Type[PSF_Base]]],
        SED_fit_code: SED_code,
        Vmax_crops: List[Type[Selector]],
        area: u.Quantity,
        z_step: float = 0.01,
    ) -> float:
        """Compute comoving volume (Vmax) for galaxy detection.

        Redshifts the galaxy's best-fit SED to different redshifts within
        the specified range, creates mock photometry, and applies selection
        criteria to determine the volume in which the galaxy would be visible.

        Parameters
        ----------
        zmin : `float`
            Minimum redshift for Vmax calculation.
        zmax : `float`
            Maximum redshift for Vmax calculation.
        filterset : `Multiple_Filter`
            Filter set used for the observations.
        depths : `list` of `float`
            5-sigma limiting depths for each filter (ABmag).
        aper_diam : `astropy.units.Quantity`
            Aperture diameter used for photometry.
        psfs : `dict` or `None`
            PSF dictionary for aperture corrections.
        SED_fit_code : `SED_code`
            SED fitting code instance providing the fit.
        Vmax_crops : `list` of `Selector`
            Selection criteria applied to determine detectability.
        area : `astropy.units.Quantity`
            Sky area (solid angle) over which the galaxy could be detected.
        z_step : `float`, optional
            Redshift grid spacing for volume calculation. Default is 0.01.

        Returns
        -------
        `float`
            Comoving volume (Mpc^3) within which the galaxy would be detected.
        """
        sed_result = self.aper_phot[aper_diam].SED_results[SED_fit_code.label]
        distance_detect = astropy_cosmo.luminosity_distance(sed_result.z)
        sed_obs = sed_result.SED

        z_detect = []
        z_arr = np.linspace(
            zmin, zmax, int(np.round((zmax - zmin), 4) / z_step) + 1
        )

        for z in z_arr:
            galfind_logger.debug("IGM attenuation ignored in SED redshift")
            wav_z = sed_obs.wavs * ((1.0 + z) / (1.0 + sed_obs.z))
            distance_test = astropy_cosmo.luminosity_distance(z)
            mag_z = (
                funcs.convert_mag_units(
                    sed_obs.wavs, sed_obs.mags, u.ABmag
                ).value
                - (
                    5
                    * np.log10(
                        (distance_detect / distance_test)
                        .to(u.dimensionless_unscaled)
                        .value
                    )
                )
                + (2.5 * np.log10((1.0 + sed_obs.z) / (1.0 + z)))
            )
            # construct galaxy at new redshift with average depths of new field
            test_sed_obs = SED_obs(
                z,
                wav_z.value,
                mag_z,
                wav_z.unit,
                u.ABmag,
            )
            galfind_logger.debug(
                "Not propagating min_flux_pc_err! Setting to 10.0%"
            )
            test_mock_phot = test_sed_obs.create_mock_photometry(
                filterset,
                depths=depths,
                min_flux_pc_err=10.0,
            )
            test_phot_obs = Photometry_obs(
                test_mock_phot.filterset,
                Masked(
                    test_mock_phot.flux,
                    mask=np.full(len(filterset), False),
                ),
                Masked(
                    test_mock_phot.flux_errs,
                    mask=np.full(len(filterset), False),
                ),
                test_mock_phot.depths,
                aper_diam,
                psfs=psfs,
            )
            sed_result = SED_result(
                SED_fit_code,
                test_phot_obs,
                {"z": z},
                property_errs={},
                property_PDFs={},
                SED=None,
            )
            test_phot_obs.SED_results = {SED_fit_code.label: sed_result}
            test_gal = Galaxy(
                self.ID,
                self.sky_coord,
                {aper_diam: test_phot_obs},
                selection_flags={},
                selection_kwargs={},
                cat_filterset=self.cat_filterset,
                survey=self.survey,
                version=self.version,
                simulated=self.simulated,
            )
            # run selection methods on new galaxy
            [selector(test_gal, return_copy=False) for selector in Vmax_crops]
            goodz = all(
                test_gal.selection_flags[selector.name]
                for selector in Vmax_crops
            )
            if goodz:
                z_detect.append(z)

        z_detect = np.array(z_detect)
        if len(z_detect) < 2:
            Vmax = -1.0
            Vmax_kwargs = {
                "zskip": str([]),
                "zmin": str(-1.0),
                "zmax": str(-1.0),
                "area": str(area.to(u.arcmin**2).value),
            }
        else:
            z_detect_mid_bins = (z_detect[:-1] + z_detect[1:]) / 2
            zbin_use_idx = np.where(np.diff(z_detect) <= z_step)[0]
            if len(zbin_use_idx) + 1 != len(z_detect):
                galfind_logger.debug(
                    "Some z bins were skipped during Vmax calculation! "
                    + f"{z_detect=}, {zbin_use_idx=}"
                )
            z_detect_mid_bins_use = z_detect_mid_bins[zbin_use_idx]
            Vmax = np.sum(
                np.array(
                    [
                        funcs.calc_Vmax(
                            area,
                            z_detect_ - (z_step / 2),
                            z_detect_ + (z_step / 2),
                        )
                        .to(u.Mpc**3)
                        .value
                        for z_detect_ in z_detect_mid_bins_use
                    ]
                )
            )
            ignore_z_mask = np.full(len(z_detect_mid_bins), True)
            ignore_z_mask[zbin_use_idx] = False
            if len(z_detect_mid_bins_use) > 0:
                zmin = np.round(
                    np.min(z_detect_mid_bins_use - (z_step / 2)), 2
                )
                zmax = np.round(
                    np.max(z_detect_mid_bins_use + (z_step / 2)), 2
                )
            else:
                zmin = -1.0
                zmax = -1.0
            Vmax_kwargs = {
                "zskip": str(
                    list(np.round(z_detect_mid_bins[ignore_z_mask], 2))
                ).replace("\n", ""),
                "zmin": str(zmin),
                "zmax": str(zmax),
                "area": str(area.to(u.arcmin**2).value),
            }
        Vmax_kwargs["Vmax"] = str(Vmax)
        return Vmax, Vmax_kwargs

    # def calc_Vmax_multifield(self, detect_cat_name: str, data_arr: Union[list, np.array], z_bin: Union[list, np.array], \  # noqa: E501
    #         SED_fit_params_key: str = "EAZY_fsps_larson_zfree", z_step: float = 0.01) -> None:  # noqa: E501
    #     z_bin_name = f"{SED_fit_params_key}_{z_bin[0]:.1f}<z<{z_bin[1]:.1f}"
    #     joint_survey_name = "+".join([data.full_name for data in data_arr])
    #     V_max = 0.
    #     V_max_simple = 0.
    #     fields_used = []
    #     for data in data_arr:
    #         # calculate Vmax in each field
    #         self.calc_Vmax(detect_cat_name, data, z_bin, SED_fit_params_key, z_step)  # noqa: E501
    #         if self.V_max[z_bin_name][data.full_name] >= 0.:
    #             V_max += self.V_max[z_bin_name][data.full_name]
    #             fields_used.append(data.full_name)
    #         #V_max_simple += self.V_max_simple[z_bin_name][data.full_name]

    #     self.V_max_fields_used[z_bin_name][joint_survey_name] = fields_used
    #     #self.V_max_simple[z_bin_name][joint_survey_name] = V_max_simple
    #     self.V_max[z_bin_name][joint_survey_name] = V_max
    #     return self

    # def save_Vmax(
    #     self: Self,
    #     Vmax: float,
    #     z_bin_name: str,
    #     full_survey_name: str,
    #     is_simple_Vmax: bool = False,
    # ) -> NoReturn:
    #     # if is_simple_Vmax:
    #     #     if not hasattr(self, "V_max_simple"):
    #     #         self.V_max_simple = {}
    #     #     if not z_bin_name in self.V_max_simple.keys():
    #     #         self.V_max_simple[z_bin_name] = {}
    #     #     self.V_max_simple[z_bin_name][full_survey_name] = Vmax
    #     # else:
    #     if not hasattr(self, "V_max"):
    #         self.V_max = {}
    #     if z_bin_name not in self.V_max.keys():
    #         self.V_max[z_bin_name] = {}
    #     self.V_max[z_bin_name][full_survey_name] = Vmax


# class Multiple_Galaxy:
#     def __init__(
#         self,
#         sky_coords,
#         IDs,
#         phots,
#         mask_flags_arr,
#         selection_flags_arr,
#         timed=True,
#     ):
#         if timed:
#             self.gals = [
#                 Galaxy(sky_coord, ID, phot, mask_flags, selection_flags)
#                 for sky_coord, ID, phot, mask_flags, selection_flags in tqdm(
#                     zip(
#                         sky_coords,
#                         IDs,
#                         phots,
#                         mask_flags_arr,
#                         selection_flags_arr,
#                     ),
#                     desc="Initializing galaxy objects",
#                     total=len(sky_coords),
#                 )
#             ]
#         else:
#             self.gals = [
#                 Galaxy(sky_coord, ID, phot, mask_flags, selection_flags)
#                 for sky_coord, ID, phot, mask_flags, selection_flags in zip(
#                     sky_coords, IDs, phots, mask_flags_arr,
#                     selection_flags_arr
#                 )
#             ]

#     def __len__(self):
#         return len(self.gals)

#     def __iter__(self):
#         self.iter = 0
#         return self

#     def __next__(self):
#         if self.iter > len(self) - 1:
#             raise StopIteration
#         else:
#             gal = self[self.iter]
#             self.iter += 1
#             return gal

#     def __getitem__(self, index):
#         return self.gals[index]

#     @classmethod
#     def from_fits_cat(
#         cls,
#         fits_cat: Union[Table, list, np.array],
#         instrument,
#         cat_creator,
#         SED_fit_params_arr,
#         timed=True,
#     ):
#         # load photometries from catalogue
#         phots = Multiple_Photometry_obs.from_fits_cat(
#             fits_cat, instrument, cat_creator, SED_fit_params_arr,
#             timed=timed
#         ).phot_obs_arr
#         # load the ID and Sky Coordinate from the source catalogue
#         IDs = np.array(fits_cat[cat_creator.ID_label]).astype(int)
#         # load sky co-ordinates
#         RAs = (
#             np.array(fits_cat[cat_creator.ra_dec_labels["RA"]])
#             * cat_creator.ra_dec_units["RA"]
#         )
#         Decs = (
#             np.array(fits_cat[cat_creator.ra_dec_labels["DEC"]])
#             * cat_creator.ra_dec_units["DEC"]
#         )
#         sky_coords = SkyCoord(RAs, Decs, frame="icrs")
#         # mask flags should come from cat_creator
#         # mask_flags_arr = [{f"unmasked_{band}": cat_creator.load_flag(fits_cat_row, f"unmasked_{band}") for band in instrument.filt_names} for fits_cat_row in fits_cat]  # noqa: E501
#         mask_flags_arr = [
#             {} for fits_cat_row in fits_cat
#         ]  # f"unmasked_{band}": None for band in instrument.filt_names
#         selection_flags_arr = [
#             {
#                 selection_flag: bool(fits_cat_row[selection_flag])
#                 for selection_flag in cat_creator.selection_labels(fits_cat)
#             }
#             for fits_cat_row in fits_cat
#         ]
# return cls(sky_coords, IDs, phots, mask_flags_arr, selection_flags_arr)

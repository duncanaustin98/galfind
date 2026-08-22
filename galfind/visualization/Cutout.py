"""Image cutout classes for single bands and RGB combinations.

Provides cutout image functionality including loading, visualization with
size scales
and contours, and support for stacked and multiple-cutout operations.
"""

from __future__ import annotations

import itertools
import sys
from abc import ABC, abstractmethod
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
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import (
    ImageNormalize,
    LinearStretch,
    LogStretch,
    ManualInterval,
    make_lupton_rgb,
)
from matplotlib.artist import ArtistInspector
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from galfind.imaging.Data import Band_Data_Base

if TYPE_CHECKING:
    from . import (
        Band_Data,
        Catalogue,
        Data,
        Filter,
        Galaxy,
        Morphology_Result,
        Multiple_Catalogue,
        Multiple_Data,
        Multiple_Filter,
    )
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from .. import astropy_cosmo, config, figs, galfind_logger
from ..imaging.Data import Band_Data, Stacked_Band_Data
from ..imaging.Filter import Filter
from ..utils import Depths
from ..utils import useful_funcs_austind as funcs


class Cutout_Base(ABC):
    """Abstract base class defining the common interface for cutout objects.

    Declares the properties and methods that any cutout-like object (single
    band cutouts, stacked cutouts, RGB combinations, and collections
    thereof) must implement, and provides shared plotting utilities.
    """

    @property
    @abstractmethod
    def ID(self) -> str:
        """`str`: Unique identifier for the object the cutout was made from.

        Must be implemented by subclasses.
        """
        pass

    @property
    @abstractmethod
    def meta(self) -> dict:
        """`dict`: Metadata associated with the cutout (e.g. sky position,
        size).

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def load(
        self: Self,
        hdu_name: str = "SCI",
    ) -> Union[
        Dict[str, Tuple[Dict[str, Any], np.ndarray]],
        Tuple[Dict[str, Any], np.ndarray],
    ]:
        """Load cutout data (and header) from the saved cutout file.

        Parameters
        ----------
        hdu_name : `str`, optional
            Name of the FITS extension to load. If `None`, all extensions
            are loaded. Default is ``"SCI"``.

        Returns
        -------
        `dict` or `tuple`
            If `hdu_name` is `None`, a dictionary mapping extension name to
            ``(header, data)`` tuples for every extension. Otherwise, the
            ``(header, data)`` tuple for the requested extension.
        """
        pass

    @abstractmethod
    def plot(self) -> plt.Axes:
        """Plot the cutout.

        Must be implemented by subclasses.

        Returns
        -------
        `matplotlib.axes.Axes`
            The axes the cutout was plotted on.
        """
        pass

    def _plot_regions(
        self: Self,
        ax: plt.Axes,
        plot_regions: List[Dict[str, Any]] = [],
        def_plot_region_kwargs: Dict[str, Any] = {
            "fill": False,
            "color": "white",
            "linestyle": "--",
            "linewidth": 1,
            "zorder": 20,
        },
    ) -> NoReturn:
        if len(plot_regions) > 0:
            # add circles to show extraction aperture and sextractor
            # FLUX_RADIUS
            xpos = np.mean(ax.get_xlim())
            ypos = np.mean(ax.get_ylim())
            for plot_region in plot_regions:
                skip_region = False
                if isinstance(plot_region, dict):
                    assert "aper_diam" in plot_region.keys()
                    pix_scale = (
                        self.meta["SIZE_AS"] * u.arcsec / self.meta["SIZE_PIX"]
                    )
                    radius = (
                        (plot_region["aper_diam"] / (2.0 * pix_scale))
                        .to(u.dimensionless_unscaled)
                        .value
                    )
                    # add region kwargs to default values
                    plot_region_kwargs = deepcopy(plot_region)
                    plot_region_kwargs.pop("aper_diam")

                    for key, value in plot_region_kwargs.items():
                        def_plot_region_kwargs[key] = value
                    # make circular region with given radius
                    region = patches.Circle(
                        (xpos, ypos),
                        radius,
                        **def_plot_region_kwargs,
                    )
                elif isinstance(
                    plot_region,
                    tuple(
                        [patches.Ellipse] + patches.Ellipse.__subclasses__()
                    ),
                ):
                    region = plot_region
                    if region.center == (-99.0, -99.0):
                        region.set_center((xpos, ypos))
                    # update default kwargs with pre-set ones
                    blank_patch = Patch()
                    kwarg_names = ArtistInspector(blank_patch).get_setters()
                    kwarg_names.remove("transform")
                    blank_kwargs = {
                        key: value
                        for key, value in ArtistInspector(blank_patch)
                        .properties()
                        .items()
                        if key in kwarg_names
                    }
                    reg_kwargs = {
                        key: value
                        for key, value in ArtistInspector(region)
                        .properties()
                        .items()
                        if key in kwarg_names
                    }
                    assert len(blank_kwargs) == len(reg_kwargs)
                    added_reg_kwargs = {
                        key: value
                        for key, value in reg_kwargs.items()
                        if value != blank_kwargs[key]
                    }
                    for key, value in added_reg_kwargs.items():
                        def_plot_region_kwargs[key] = value
                    # set region kwargs
                    region.set(**def_plot_region_kwargs)
                else:
                    skip_region = True
                    galfind_logger.warning(
                        f"{plot_region=} does not contain "
                        + f"'aper_diam' or {type(plot_region)=} not in "
                        + tuple(
                            [patches.Ellipse]
                            + patches.Ellipse.__subclasses__()
                        )
                        + ", skipping!"
                    )
                if not skip_region:
                    ax.add_patch(region)


class Band_Cutout_Base(Cutout_Base, ABC):
    """Abstract base class for a single-band image cutout.

    Wraps a saved cutout ``.fits`` file for one filter/band, providing
    access to its data/metadata extensions (e.g. ``SCI``, ``RMS_ERR``,
    ``WHT``, ``SEG``) and shared plotting functionality. Concrete
    subclasses (`Band_Cutout`, `Stacked_Band_Cutout`) implement the
    construction of the underlying cutout file.

    Parameters
    ----------
    cutout_path : `str`
        Path to the saved cutout ``.fits`` file. Must already exist.
    band_data : `Band_Data`
        Band data object the cutout was extracted from.
    cutout_size : `astropy.units.Quantity`
        Angular size of the cutout.

    Attributes
    ----------
    cutout_path : `str`
        Path to the saved cutout ``.fits`` file.
    band_data : `Band_Data`
        Band data object the cutout was extracted from.
    cutout_size : `astropy.units.Quantity`
        Angular size of the cutout.
    """

    def __init__(
        self: Self,
        cutout_path: str,
        band_data: Band_Data,
        cutout_size: u.Quantity,
    ) -> Self:
        assert Path(cutout_path).is_file(), galfind_logger.critical(
            f"Cutout path {cutout_path} does not exist!"
        )
        self.cutout_path = cutout_path
        self.band_data = band_data
        self.cutout_size = cutout_size

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__.upper()}({self.ID}"
            + f",{self.filt_name}"
            + f",{self.cutout_size.to(u.arcsec).value:.2f}as)"
        )

    def __str__(self) -> str:
        output_str = funcs.line_sep
        output_str += f"{repr(self)}:\n"
        output_str += funcs.line_sep
        output_str += f"Cutout path: {self.cutout_path}\n"
        if hasattr(self, "morph_fits"):
            if len(self.morph_fits) > 0:
                output_str += "Morphology fits:\n"
                output_str += f"{repr(self.morph_fits)}\n"
        output_str += "Meta:\n"
        output_str += funcs.band_sep
        for key, val in self.meta.items():
            output_str += f"{key}: {val}\n"
        output_str += funcs.line_sep
        return output_str

    def __copy__(self) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, deepcopy(value, memo))
        return result

    # ensure this is in the correct class
    @property
    def ID(self) -> str:
        """`str`: Unique identifier for the cutout, derived from its
        metadata."""
        return self._get_ID(self.meta)

    @staticmethod
    def _get_ID(meta: Dict[str, Any]) -> str:
        if "ID" in meta:
            return meta["ID"]
        else:
            return f"({meta['RA']:.5f},{meta['DEC']:.5f})"

    @property
    def instr_name(self) -> str:
        """`str`: Name of the instrument the cutout's band belongs
        to, derived from its metadata."""
        return self._get_instr_name(self.meta)

    @staticmethod
    def _get_instr_name(meta: Dict[str, Any]) -> str:
        if "INSTR" in meta.keys():
            return meta["INSTR"]
        else:
            return None

    @property
    def meta(self) -> dict:
        """`dict`: Metadata stored in the cutout's ``PRIMARY`` FITS header."""
        return dict(self.load("PRIMARY")[0])

    # sky_coord, survey, version may need to be stored in Cutout_Base
    @property
    def sky_coord(self) -> SkyCoord:
        """`astropy.coordinates.SkyCoord`: Sky position the cutout is
        centred on."""
        return SkyCoord(
            ra=self.meta["RA"] * u.deg,
            dec=self.meta["DEC"] * u.deg,
        )

    @property
    def survey(self) -> str:
        """`str`: Survey the cutout's band data belongs to."""
        return self.band_data.survey

    @property
    def version(self) -> str:
        """`str`: Data reduction version of the cutout's band data."""
        return self.band_data.version

    @property
    def filt_name(self) -> Filter:
        """`str`: Name of the filter the cutout was made in, taken
        from `band_data`."""
        return self.band_data.filt_name

    @staticmethod
    def _get_save_path(
        band_data_base: Type[Band_Data_Base],
        cutout_size: u.Quantity,
        ID: str,
        instr_name: Optional[str],
        data_type: str,
    ) -> str:
        assert data_type in [
            "data",
            "png",
            "svg",
            "pdf",
        ], galfind_logger.critical(f"Invalid {data_type=}")
        if data_type == "data":
            ext = ".fits"
        elif data_type == "png":
            ext = ".png"
        elif data_type == "svg":
            ext = ".svg"
        elif data_type == "pdf":
            ext = ".pdf"
        if instr_name is None:
            instr_name = ""
        else:
            instr_name = f"{instr_name}/"
        # get forced phot subdir
        if not hasattr(band_data_base, "aper_diams") or not hasattr(
            band_data_base, "forced_phot_args"
        ):
            galfind_logger.debug(
                f"{band_data_base=} does not have aper_diams or "
                + "forced_phot_args attributes needed to get "
                + "forced phot subdir!"
            )
            subdir = ""
        else:
            forced_phot_subdir = Depths.get_forced_phot_subdir(
                band_data_base.aper_diams,
                band_data_base.forced_phot_args,
            )
            subdir = f"/{forced_phot_subdir}"
        save_dir = (
            f"{config['Cutouts']['CUTOUT_DIR']}/{band_data_base.version}/"
            + f"{band_data_base.survey}/{instr_name}"
            + f"{cutout_size.to(u.arcsec).value:.2f}as/"
            + f"{band_data_base.filt_name}{subdir}/{data_type}"
        )
        if band_data_base.is_native:
            save_dir += "_native"
        save_name = f"{ID}{ext}"
        save_path = f"{save_dir}/{save_name}"
        funcs.make_dirs(save_path)
        return save_path

    def load(
        self: Self,
        hdu_name: str = "SCI",
    ) -> Union[
        Dict[str, Tuple[Dict[str, Any], np.ndarray]],
        Tuple[Dict[str, Any], np.ndarray],
    ]:
        """Load cutout data (and header) from `cutout_path`.

        Parameters
        ----------
        hdu_name : `str`, optional
            Name of the FITS extension to load (e.g. ``"SCI"``,
            ``"RMS_ERR"``, ``"WHT"``, ``"SEG"``, ``"PRIMARY"``). If `None`,
            all extensions in the file are loaded. Default is ``"SCI"``.

        Returns
        -------
        `dict` or `tuple`
            If `hdu_name` is `None`, a dictionary mapping extension name to
            ``(header, data)`` tuples for every extension in the cutout
            file. Otherwise, the ``(header, data)`` tuple for the requested
            extension.
        """
        if hdu_name is None:
            hdul = fits.open(self.cutout_path, ignore_missing_simple=True)
            return {hdu.name: (dict(hdu.header), hdu.data) for hdu in hdul}
        else:
            hdu = fits.open(self.cutout_path, ignore_missing_simple=True)[
                hdu_name
            ]
            return dict(hdu.header), hdu.data

    def update_morph_fits(
        self: Self,
        morph_results: Union[Morphology_Result, List[Morphology_Result]],
        overwrite: bool = False,
    ) -> NoReturn:
        """Store one or more morphology fitting results on this cutout.

        Results are stored in the `morph_fits` dictionary, keyed by the
        fitter's name.

        Parameters
        ----------
        morph_results : `Morphology_Result` or `list` of `Morphology_Result`
            Morphology fit result(s) to add.
        overwrite : `bool`, optional
            If `True` (or if `morph_fits` does not yet exist), any existing
            `morph_fits` dictionary is reset before adding the new results.
            Default is `False`.
        """
        from ..properties.Morphology import Morphology_Result

        if overwrite or not hasattr(self, "morph_fits"):
            self.morph_fits = {}
        if isinstance(morph_results, Morphology_Result):
            morph_results = [morph_results]
        self.morph_fits = {
            **self.morph_fits,
            **{result.fitter.name: result for result in morph_results},
        }

    def plot(
        self: Self,
        ax: Optional[plt.Axes] = None,
        plot_type: str = "SCI",
        imshow_kwargs: Dict[str, Any] = {},
        norm_kwargs: Dict[str, Any] = {},
        label_kwargs: Dict[str, Any] = {},
        plot_regions: List[Dict[str, Any]] = [],
        scalebars: Optional[Dict] = [],
        show: bool = False,
        save: bool = True,
        *args,
        **kwargs,
    ) -> NoReturn:
        """Plot the cutout image on a matplotlib axis.

        Loads the requested extension and displays it with `Axes.imshow`.
        Segmentation maps (``plot_type="SEG"``) are remapped so that unique
        segment IDs are evenly spread over ``[0, 1]`` to improve contrast.
        Supports an ``"EPOCHS"`` normalisation preset (via
        `_EPOCHS_cutout_scaling`), a text label, plotted regions (via
        `_plot_regions`), and angular/physical scalebars, and can save the
        resulting figure as an SVG.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`, optional
            Axis to plot on. If `None`, a new figure and axis are created.
            Default is `None`.
        plot_type : `str`, optional
            Name of the FITS extension to plot (e.g. ``"SCI"``, ``"SEG"``).
            Default is ``"SCI"``.
        imshow_kwargs : `dict`, optional
            Keyword arguments passed to `Axes.imshow`. If ``"norm"`` is the
            string ``"EPOCHS"``, `_EPOCHS_cutout_scaling` is used to rescale
            the cutout data and compute the normalisation instead. Default
            is ``{}``.
        norm_kwargs : `dict`, optional
            Keyword arguments passed to `_EPOCHS_cutout_scaling` when the
            ``"EPOCHS"`` normalisation is requested. Default is ``{}``.
        label_kwargs : `dict`, optional
            Keyword arguments controlling an optional text label (e.g.
            ``"label"``, ``"xpos"``, ``"ypos"``, ``"c"``, ``"fontsize"``)
            drawn in axis-fraction coordinates. Default is ``{}``.
        plot_regions : `list` of `dict`, optional
            Regions to overlay on the cutout, passed to `_plot_regions`.
            Default is ``[]``.
        scalebars : `dict`, optional
            Mapping of scalebar type (``"angular"`` or ``"physical"``) to
            keyword arguments for that scalebar (e.g. ``"as_length"``,
            ``"z"``, ``"pix_length"``, ``"loc"``). Default is ``[]``.
        show : `bool`, optional
            Whether to call `matplotlib.pyplot.show` after plotting.
            Default is `False`.
        save : `bool`, optional
            Whether to save the figure as an SVG to the standard cutout
            path. Default is `True`.
        *args, **kwargs
            Additional positional/keyword arguments (currently unused).
        """
        #        high_dyn_range: bool = False,
        #        SNR: Optional[float] = None,
        if ax is None:
            fig, ax = plt.subplots()
        # load cutout
        cutout_data = self.load(plot_type)[1]
        def_imshow_kwargs = {
            "norm": "linear",
            "cmap": "magma",
            "origin": "lower",
        }
        if "norm" in imshow_kwargs.keys():
            # scale cutout
            if isinstance(imshow_kwargs["norm"], str):
                if imshow_kwargs["norm"] == "EPOCHS":
                    cutout_data, norm = self._EPOCHS_cutout_scaling(
                        cutout_data, **norm_kwargs
                    )
                    imshow_kwargs["norm"] = norm
        for key, value in imshow_kwargs.items():
            def_imshow_kwargs[key] = value
        # plot cutout
        if plot_type == "SEG":
            # colour by unique values set to be between 0 and 1 to
            # increase the imshow contrast
            unique_ids = np.unique(cutout_data.copy())
            assert unique_ids[0] == 0, galfind_logger.warning(
                "Unique IDs in SEG cutout should start at 0, "
                + f"but got {unique_ids[0]}!"
            )
            mapping = {
                uid: i / (len(unique_ids) - 1)
                for i, uid in enumerate(unique_ids)
            }
            cutout_data = np.vectorize(mapping.get)(cutout_data)
        ax.imshow(cutout_data, **def_imshow_kwargs)
        # sort label kwargs
        def_label_kwargs = {
            "xpos": 0.95,
            "ypos": 0.95,
            "fontsize": "medium",
            "c": "white",
            "ha": "right",
            "va": "top",
            "zorder": 10,
            "fontweight": "bold",
        }
        for key, value in def_label_kwargs.items():
            label_kwargs.setdefault(key, value)
            # def_label_kwargs[key] = value
        label = label_kwargs.pop("label", None)
        if label is not None:
            text_unpack_kwargs = deepcopy(label_kwargs)
            text_unpack_kwargs.pop("xpos")
            text_unpack_kwargs.pop("ypos")
            # plot text for band label
            ax.text(
                def_label_kwargs["xpos"],
                def_label_kwargs["ypos"],
                label,
                transform=ax.transAxes,
                **text_unpack_kwargs,
            )

        # plot any regions wanted
        self._plot_regions(ax, plot_regions)

        # add scalebars
        if len(scalebars) > 0:
            pix_scale = self.meta["SIZE_AS"] * u.arcsec / self.meta["SIZE_PIX"]
            for key, scalebar_kwargs in scalebars.items():
                plot_scalebar = True
                if key == "angular":
                    assert all(
                        [
                            key in scalebar_kwargs.keys()
                            for key in ["as_length"]
                        ]
                    )
                    size = scalebar_kwargs["as_length"] / pix_scale.value
                    label = f'{str(scalebar_kwargs["as_length"]):.1f}"'
                    [scalebar_kwargs.pop(key) for key in ["as_length"]]
                elif key == "physical":
                    assert all(
                        [
                            key in scalebar_kwargs.keys()
                            for key in ["z", "pix_length"]
                        ]
                    )
                    d_A = astropy_cosmo.angular_diameter_distance(
                        scalebar_kwargs["z"]
                    )
                    pix_scale = u.pixel_scale(pix_scale / u.pixel)
                    re_as = (scalebar_kwargs["pix_length"] * u.pixel).to(
                        u.arcsec, pix_scale
                    )
                    re_kpc = (re_as * d_A).to(u.kpc, u.dimensionless_angles())
                    size = scalebar_kwargs["pix_length"]
                    label = f"{re_kpc:.1f}"
                    [scalebar_kwargs.pop(key) for key in ["z", "pix_length"]]
                else:
                    plot_scalebar = False
                    galfind_logger.warning(f"Invalid scalebar key: {key}")

                if plot_scalebar:
                    assert "loc" in scalebar_kwargs.keys()
                    scalebar = AnchoredSizeBar(
                        ax.transData, size, label, **scalebar_kwargs
                    )
                    ax.add_artist(scalebar)
        # option to save here
        if save:
            save_path = self._get_save_path(
                self.band_data,
                self.cutout_size,
                self.ID,
                self.instr_name,
                "svg",
            )
            funcs.make_dirs(save_path)
            plt.savefig(save_path)
            funcs.change_file_permissions(save_path)
            galfind_logger.info(f"Saved png cutout to: {save_path}")
        if show:
            plt.show()

    def _EPOCHS_cutout_scaling(
        self: Self,
        cutout_data: np.ndarray,
        high_dyn_range: Optional[bool] = False,
        SNR: Optional[float] = None,
        *args,
        **kwargs,
    ) -> Tuple[np.ndarray, ImageNormalize]:
        # Set top value based on central 10x10 pixel region
        # TODO: GENERALIZE!
        top = np.max(cutout_data[:20, 10:20])
        cutout_size_pix = self.meta["SIZE_PIX"]
        top = np.max(
            cutout_data[
                int(cutout_size_pix // 2 - 0.3 * cutout_size_pix) : int(
                    cutout_size_pix // 2 + 0.3 * cutout_size_pix
                ),
                int(cutout_size_pix // 2 - 0.3 * cutout_size_pix) : int(
                    cutout_size_pix // 2 + 0.3 * cutout_size_pix
                ),
            ]
        )
        bottom_val = top / 10**5
        if high_dyn_range:
            a = 300.0
        else:
            a = 0.1
        stretch = LogStretch(a=a)
        if SNR is not None:
            if SNR < 100.0:
                bottom_val = top * 1e-3
                # a = 100
            if SNR <= 15.0:
                bottom_val = top * 1e-2
                # a = 0.1
            if SNR < 8.0:
                bottom_val = top / 100_000
                stretch = LinearStretch()

        cutout_data = np.clip(
            cutout_data * 0.9999, bottom_val * 1.000001, top
        )  # why?
        norm = ImageNormalize(
            cutout_data,
            interval=ManualInterval(bottom_val, top),
            clip=True,
            stretch=stretch,
        )
        return cutout_data, norm


class Band_Cutout(Band_Cutout_Base):
    """A single-band image cutout for one galaxy/position and filter.

    Instances are constructed by making (or loading a previously made)
    cutout ``.fits`` file for a `Band_Data` object at a given sky position,
    via the `from_gal_band_data` or `from_data_skycoord` class methods.
    """

    @classmethod
    def from_gal_band_data(
        cls: Type[Self],
        gal: Galaxy,
        band_data: Band_Data,
        cutout_size: u.Quantity,
        overwrite: bool = False,
    ) -> Self:
        """Construct a `Band_Cutout` centred on a `Galaxy`'s sky position.

        Builds cutout metadata from available SExtractor-derived galaxy
        properties (e.g. size, magnitude, flux, Kron radius, image shape
        parameters) in addition to the galaxy's ID and instrument name, then
        delegates cutout creation to `from_data_skycoord`.

        Parameters
        ----------
        gal : `Galaxy`
            Galaxy to centre the cutout on.
        band_data : `Band_Data`
            Band data to extract the cutout from.
        cutout_size : `astropy.units.Quantity`
            Angular size of the cutout.
        overwrite : `bool`, optional
            Whether to overwrite an existing cutout file. Default is
            `False`.

        Returns
        -------
        `Band_Cutout`
            The constructed (or loaded) cutout.
        """
        # TODO: ensure in some way that the galaxy arises from the data
        # extract the position of the galaxy
        sky_coord = gal.sky_coord
        meta = {"ID": gal.ID, "INSTR": gal.cat_filterset.instrument_name}
        meta_keys = [
            "Re",
            "FLUX_AUTO",
            "MAG_AUTO",
            "KRON_RADIUS",
            "A_IMAGE",
            "B_IMAGE",
            "THETA_IMAGE",
            "A_IMAGE_AS",
            "B_IMAGE_AS",
        ]
        suffixes = ["_AS", "_JY", "", "", "", "", "", "", ""]
        filt_name = band_data.filt.filt_name
        for meta_key, suffix in zip(meta_keys, suffixes):
            meta_key = f"sex_{meta_key}"
            if hasattr(gal, meta_key):
                attr = getattr(gal, meta_key)
                if isinstance(attr, dict):
                    attr = attr[filt_name]
                attr = attr.value
                if len(meta_key) > 8:
                    meta_key = f"HIERARCH {meta_key}"
                meta = {
                    **meta,
                    **{
                        f"{meta_key.replace('sex_', '').upper()}"
                        f"{suffix.upper()}": attr
                    },
                }
            else:
                galfind_logger.debug(f"No {meta_key} found for {repr(gal)}!")
        return cls.from_data_skycoord(
            band_data,
            sky_coord,
            cutout_size,
            overwrite=overwrite,
            **meta,
        )

    @classmethod
    def from_data_skycoord(
        cls: Type[Self],
        band_data: Band_Data,
        sky_coord: SkyCoord,
        cutout_size: u.Quantity,
        overwrite: bool = False,
        **meta: Any,
    ) -> Self:
        """Construct a `Band_Cutout` at a given sky position.

        Assembles cutout metadata (survey, version, position, size), makes
        (or reuses) the cutout ``.fits`` file via `_make_cutout`, updates
        the associated `Band_Data` object to point at the cutout, and
        returns the resulting `Band_Cutout` instance.

        Parameters
        ----------
        band_data : `Band_Data`
            Band data to extract the cutout from.
        sky_coord : `astropy.coordinates.SkyCoord`
            Sky position to centre the cutout on.
        cutout_size : `astropy.units.Quantity`
            Angular size of the cutout.
        overwrite : `bool`, optional
            Whether to overwrite an existing cutout file. Default is
            `False`.
        **meta : `Any`
            Additional metadata to store in the cutout's ``PRIMARY`` FITS
            header.

        Returns
        -------
        `Band_Cutout`
            The constructed (or loaded) cutout.
        """
        # make cutout from data at the sky co-ordinate and save
        meta = {
            **meta,
            "SURVEY": band_data.survey,
            "VERSION": band_data.version,
            "RA": sky_coord.ra.value,
            "DEC": sky_coord.dec.value,
            "SIZE_AS": cutout_size.to(u.arcsec).value,
            "SIZE_PIX": (cutout_size / band_data.pix_scale)
            .to(u.dimensionless_unscaled)
            .value,
        }
        ID = cls._get_ID(meta)
        instr_name = cls._get_instr_name(meta)
        save_path = cls._get_save_path(
            band_data,
            cutout_size,
            ID,
            instr_name,
            data_type="data",
        )
        cls._make_cutout(
            band_data, sky_coord, cutout_size, save_path, meta, overwrite
        )
        band_data = cls._update_band_data_base(band_data, save_path)
        return cls(save_path, band_data, cutout_size)

    # def set_cutout_size(
    #     self: Self,
    #     cutout_size: u.Quantity,
    #     overwrite: bool = True
    # ) -> NoReturn:
    #     sky_coord = self.sky_coord
    #     meta = self.meta
    #     meta["SIZE_AS"] = cutout_size.to(u.arcsec).value
    #     meta["SIZE_PIX"] = (cutout_size / self.band_data.pix_scale) \
    #         .to(u.dimensionless_unscaled).value
    #     self.cutout_size = cutout_size
    #     self.cutout_path = self._get_save_path(
    #         self.band_data, cutout_size, self.ID, "data"
    #     )
    #     self._make_cutout(
    #         self.band_data,
    #         sky_coord,
    #         cutout_size,
    #         self.cutout_path,
    #         meta,
    #         overwrite=overwrite
    #         )
    # self.band_data = self._update_band_data(self.band_data, self.cutout_path)

    @staticmethod
    def _make_cutout(
        band_data: Band_Data,
        sky_coord: SkyCoord,
        cutout_size: u.Quantity,
        save_path: str,
        meta: Dict[str, Any] = {},
        overwrite: bool = False,
    ) -> NoReturn:
        # make cutout from data at the sky co-ordinate
        if not Path(save_path).is_file() or overwrite:
            im_data, im_header, seg_data, seg_header = band_data.load_data(
                incl_mask=False
            )
            pix_scale = band_data.pix_scale
            data_dict = {
                "SCI": im_data,
                "SEG": seg_data,
                "RMS_ERR": band_data.load_rms_err(),
                "WHT": band_data.load_wht(),
            }
            hdul = [fits.PrimaryHDU(header=fits.Header(meta))]

            cutout_size_pix = (
                (cutout_size / pix_scale).to(u.dimensionless_unscaled).value
            )

            for i, (label_i, data_i) in enumerate(data_dict.items()):
                if i == 0 and label_i == "SCI":
                    sci_shape = data_i.shape
                if data_i is None:
                    galfind_logger.warning(
                        f"No data found for {label_i} in "
                        + f"{band_data.filt_name}!"
                    )
                else:
                    if data_i.shape == sci_shape:
                        cutout = Cutout2D(
                            data_i,
                            sky_coord,
                            size=(cutout_size_pix, cutout_size_pix),
                            wcs=band_data.load_wcs(),
                        )
                        im_header.update(cutout.wcs.to_header())
                        im_header["EXTNAME"] = label_i
                        hdul.append(
                            fits.ImageHDU(
                                cutout.data, header=im_header, name=label_i
                            )
                        )
                        galfind_logger.debug(
                            f"Created cutout for {label_i} in "
                            + f"{band_data.filt_name}"
                        )
                    else:
                        galfind_logger.warning(
                            f"Incorrect data shape. {data_i=} != "
                            + f"{sci_shape=}, skipping extension!"
                        )
            funcs.make_dirs(save_path)
            fits_hdul = fits.HDUList(hdul)
            fits_hdul.writeto(save_path, overwrite=True)
            funcs.change_file_permissions(save_path)
            galfind_logger.info(f"Saved fits cutout to: {save_path}")
        else:
            ID = Band_Cutout_Base._get_ID(meta)
            galfind_logger.debug(
                f"Already made fits cutout for {band_data.survey}"
                f" {band_data.version} {ID} {band_data.filt_name}"
                f" at {save_path=}"
            )

    @staticmethod
    def _update_band_data_base(
        band_data_base: Type[Band_Data_Base],
        cutout_path: str,
    ) -> Band_Data:
        if isinstance(band_data_base, Band_Data):
            filt = band_data_base.filt
        else:
            assert isinstance(
                band_data_base, Stacked_Band_Data
            ), galfind_logger.critical(
                "band_data_base must be Band_Data or "
                + f"Stacked_Band_Data, not {type(band_data_base)=}"
            )
            filt = band_data_base.filterset
        new_band_data = band_data_base.__class__(
            filt,
            band_data_base.survey,
            band_data_base.version,
            cutout_path,
            1,
            cutout_path,
            3,
            cutout_path,
            4,
            pix_scale=band_data_base.pix_scale,
            rms_err_ext_name="RMS_ERR",
            psf=band_data_base.psf,
        )
        new_band_data.is_native = band_data_base.is_native
        new_band_data.seg_path = cutout_path
        new_band_data.seg_args = band_data_base.seg_args
        return new_band_data

    def __add__(
        self: Self, other: Union[Band_Cutout, List[Band_Cutout]]
    ) -> Union[Stacked_Band_Cutout, RGB]:
        # TODO: THIS IS NOT FINISHED
        # make other a list of Cutout objects if not already
        if isinstance(other, Band_Cutout):
            other = [other]
        # stack cutouts that are from the same filter

        # make an RGB if all
        # ensure all cutout filters are the same
        if not all([cutout.filt == self.filt for cutout in other]):
            raise ValueError(
                "All cutouts must have the same filter as "
                + f"{repr(self.filter)=}"
            )


class Stacked_Band_Cutout(Band_Cutout_Base):
    """An inverse-variance-weighted stack of single-band cutouts.

    Represents a cutout formed by stacking multiple `Band_Cutout` images
    (for the same filter, sky position and cutout size) into a single
    ``SCI``/``RMS_ERR``/``WHT`` cutout. Instances are typically constructed
    via `from_cat`, `from_data_skycoords` or `from_cutouts`.

    Parameters
    ----------
    cutout_path : `str`
        Path to the saved stacked cutout ``.fits`` file.
    band_data : `Band_Data`
        Band data object representing the stacked cutout.
    cutout_size : `astropy.units.Quantity`
        Angular size of the cutout.
    origin_paths : `list` of `str`
        Paths to the individual cutout files that were stacked.

    Attributes
    ----------
    origin_paths : `list` of `str`
        Paths to the individual cutout files that were stacked.
    """

    def __init__(
        self,
        cutout_path: str,
        band_data: Band_Data,
        cutout_size: u.Quantity,
        origin_paths: List[str],
    ) -> Self:
        self.origin_paths = origin_paths
        super().__init__(cutout_path, band_data, cutout_size)

    @classmethod
    def from_cat(
        cls,
        cat: Catalogue,
        filt: Union[str, Filter],
        cutout_size: u.Quantity,
        overwrite: bool = False,
    ) -> Self:
        """Construct a stacked cutout for one filter from all
        galaxies in a `Catalogue`.

        Loads SExtractor-derived metadata onto the catalogue, makes an
        individual `Band_Cutout` for every galaxy in `cat` in the given
        filter, and stacks them via `from_cutouts`.

        Parameters
        ----------
        cat : `Catalogue`
            Catalogue whose galaxies are stacked.
        filt : `str` or `Filter`
            Filter (or filter name) to make the stacked cutout for.
        cutout_size : `astropy.units.Quantity`
            Angular size of the cutout.
        overwrite : `bool`, optional
            Whether to overwrite existing individual/stacked cutout files.
            Default is `False`.

        Returns
        -------
        `Stacked_Band_Cutout`
            The stacked cutout.
        """
        # load sextractor parameters for metadata inclusion
        cat.load_sextractor_auto_mags()
        cat.load_sextractor_auto_fluxes()
        cat.load_sextractor_kron_radii()
        cat.load_sextractor_Re()

        if isinstance(filt, Filter):
            filt = filt.filt_name
        # make every individual cutout from the catalogue
        cutouts = [
            Band_Cutout.from_gal_band_data(
                gal, cat.data[filt], cutout_size, overwrite=overwrite
            )
            for gal in cat
        ]
        save_path = cls._get_save_path(
            cat.data[filt],
            cutout_size,
            cat.crop_name,
            cat.filterset.instrument_name,
            "data",
        )
        return cls.from_cutouts(cutouts, save_path, overwrite=overwrite)

    @classmethod
    def from_data_skycoords(
        cls,
        data: Data,
        filt: Union[str, Filter],
        sky_coords: Union[SkyCoord, List[SkyCoord]],
        cutout_size: u.Quantity,
        save_path: str = None,
        overwrite: bool = False,
    ) -> Self:
        """Construct a stacked cutout for one filter at a set of sky positions.

        Makes an individual `Band_Cutout` for every sky position in
        `sky_coords` and stacks them via `from_cutouts`.

        Parameters
        ----------
        data : `Data`
            Data object to extract the cutouts from.
        filt : `str` or `Filter`
            Filter (or filter name) to make the stacked cutout for.
        sky_coords : `astropy.coordinates.SkyCoord` or `list` of `SkyCoord`
            Sky position(s) to make and stack cutouts at.
        cutout_size : `astropy.units.Quantity`
            Angular size of the cutout.
        save_path : `str`, optional
            Path to save the stacked cutout to. Default is `None`.
        overwrite : `bool`, optional
            Whether to overwrite existing individual/stacked cutout files.
            Default is `False`.

        Returns
        -------
        `Stacked_Band_Cutout`
            The stacked cutout.
        """
        # make every individual cutout from the data at the given SkyCoord
        cutouts = [
            Band_Cutout.from_data_skycoord(
                data, filt, sky_coord, cutout_size, overwrite=overwrite
            )
            for sky_coord in sky_coords
        ]
        return cls.from_cutouts(cutouts, save_path, overwrite=overwrite)

    @classmethod
    def from_cutouts(
        cls,
        cutouts: List[Band_Cutout],
        save_path: str,
        overwrite: bool = False,
    ) -> Self:
        """Construct a `Stacked_Band_Cutout` by stacking a list of
        `Band_Cutout` objects.

        Parameters
        ----------
        cutouts : `list` of `Band_Cutout`
            Cutouts to stack. Must all share the same filter and cutout
            size.
        save_path : `str`
            Path to save the stacked cutout ``.fits`` file to.
        overwrite : `bool`, optional
            Whether to overwrite an existing stacked cutout file. Default
            is `False`.

        Returns
        -------
        `Stacked_Band_Cutout`
            The stacked cutout.
        """
        # ensure all cutouts are from the same filter
        assert all(
            [cutout.filt_name == cutouts[0].filt_name for cutout in cutouts]
        )
        assert all(
            [
                cutout.cutout_size == cutouts[0].cutout_size
                for cutout in cutouts
            ]
        )
        # stack cutouts if they have not been already
        cls._stack_cutouts(cutouts, save_path, overwrite=overwrite)
        band_data = cls._update_band_data(
            [cutout.band_data for cutout in cutouts], save_path
        )
        # extract original cutout paths
        origin_paths = [cutout.cutout_path for cutout in cutouts]
        return cls(save_path, band_data, cutouts[0].cutout_size, origin_paths)

    @staticmethod
    def _stack_cutouts(
        cutouts: List[Band_Cutout], save_path: str, overwrite: bool = False
    ) -> NoReturn:
        if not Path(save_path).is_file() or overwrite:
            # ensure all band data images have the same ZP
            assert all(
                cutout.band_data.ZP == cutouts[0].band_data.ZP
                for cutout in cutouts
            ), galfind_logger.critical("All cutout ZPs must be the same!")
            # ensure all band data images have the same pixel scale
            assert all(
                cutout.band_data.pix_scale == cutouts[0].band_data.pix_scale
                for cutout in cutouts
            ), galfind_logger.critical(
                "All image pixel scales must be the same!"
            )
            # stack band data SCI/ERR/WHT images (inverse variance weighted)
            surveys_versions = np.unique(
                [
                    f"{cutout.band_data.survey}," + cutout.band_data.version
                    for cutout in cutouts
                ]
            )
            galfind_logger.info(
                f"Stacking {len(cutouts)} {cutouts[0].filt_name}"
                + f" cutouts for {'+'.join(surveys_versions)}!"
            )
            # load all cutouts
            cutout_data_arr = [cutout.load(None) for cutout in cutouts]
            for i, cutout_data in enumerate(cutout_data_arr):
                sci_hdr = cutout_data["SCI"][0]
                sci_data = cutout_data["SCI"][1]
                rms_err_hdr = cutout_data["RMS_ERR"][0]
                cutout_data["RMS_ERR"][1]
                wht_hdr = cutout_data["WHT"][0]
                wht_data = cutout_data["WHT"][1]
                if i == 0:
                    sum = sci_data * wht_data
                    sum_wht = wht_data
                else:
                    sum += sci_data * wht_data
                    sum_wht += wht_data
            sci = sum / sum_wht
            err = np.sqrt(1.0 / sum_wht)
            wht = sum_wht
            # save stacked cutout
            hdr = {
                "ID": save_path.split("/")[-1].split(".fits")[0],
                "SURVEYS_VERSIONS": "+".join(surveys_versions),
                "N_CUTOUTS": len(cutouts),
                "FILT": cutouts[0].filt_name,
                "ZP": cutouts[0].band_data.ZP,
                "SIZE_AS": cutouts[0].meta["SIZE_AS"],
                "SIZE_PIX": cutouts[0].meta["SIZE_PIX"],
            }
            sci_hdr = deepcopy(hdr)
            sci_hdr["EXTNAME"] = "SCI"
            rms_err_hdr = deepcopy(hdr)
            rms_err_hdr["EXTNAME"] = "RMS_ERR"
            wht_hdr = deepcopy(hdr)
            wht_hdr["EXTNAME"] = "WHT"
            hdr = fits.Header(hdr)
            primary = fits.PrimaryHDU(header=fits.Header(hdr))
            hdu = fits.ImageHDU(sci, header=fits.Header(sci_hdr), name="SCI")
            hdu_err = fits.ImageHDU(
                err, header=fits.Header(rms_err_hdr), name="RMS_ERR"
            )
            hdu_wht = fits.ImageHDU(
                wht, header=fits.Header(wht_hdr), name="WHT"
            )
            hdul = fits.HDUList([primary, hdu, hdu_err, hdu_wht])
            hdul.writeto(save_path, overwrite=True)
            funcs.change_file_permissions(save_path)
            galfind_logger.info(f"Saved stacked cutout to: {save_path}")

    @staticmethod
    def _update_band_data(
        band_data_arr: List[Band_Data],
        cutout_path: str,
    ) -> Band_Data:
        surveys = "+".join(
            np.unique([band_data.survey for band_data in band_data_arr])
        )
        versions = "+".join(
            np.unique([band_data.version for band_data in band_data_arr])
        )
        new_band_data = Band_Data(
            band_data_arr[0].filt,
            surveys,
            versions,
            cutout_path,
            1,
            cutout_path,
            2,
            cutout_path,
            3,
            pix_scale=band_data_arr[0].pix_scale,
            rms_err_ext_name="RMS_ERR",
        )
        new_band_data.seg_path = cutout_path
        new_band_data.seg_args = {
            key: "+".join(
                np.unique(
                    [band_data.seg_args[key] for band_data in band_data_arr]
                )
            )
            for key in band_data_arr[0].seg_args.keys()
        }
        return new_band_data


class RGB_Base(Cutout_Base, ABC):
    """Abstract base class for a three-colour (
        RGB) combination of band cutouts.

    Combines cutouts from up to three colour channels (``"B"``, ``"G"``,
    ``"R"``), each of which may itself contain multiple filters, and
    provides shared RGB-image construction/plotting functionality. Concrete
    subclasses (`RGB`, `Stacked_RGB`) implement construction from
    individual or stacked band cutouts.

    Parameters
    ----------
    cutouts : `dict` of `str` -> `list` of `Band_Cutout_Base`
        Mapping of colour channel (``"B"``, ``"G"``, ``"R"``) to the list of
        cutouts to combine into that channel. All cutouts across all
        channels must be from different filters.

    Attributes
    ----------
    cutouts : `dict` of `str` -> `list` of `Band_Cutout_Base`
        Mapping of colour channel to the cutouts combined into it.
    """

    def __init__(
        self: Type[Self],
        cutouts: Dict[str, List[Type[Band_Cutout_Base]]],
    ) -> Self:
        # ensure cutouts have ['B', 'G', 'R'] keys
        assert all(
            colour in list(cutouts.keys()) for colour in ["B", "G", "R"]
        ), galfind_logger.critical(
            f"['B', 'G', 'R'], not {list(cutouts.keys())=}"
        )
        # ensure all cutouts are from different filters
        cutout_filt_names = [
            cutout.band_data.filt_name
            for colour in ["B", "G", "R"]
            for cutout in cutouts[colour]
        ]
        assert len(np.unique(cutout_filt_names)) == len(cutout_filt_names)
        self.cutouts = cutouts

    def __len__(self) -> int:
        return len(self.cutouts)

    def __iter__(self):
        return iter(self.container)

    def __getitem__(self, i: str) -> List[Type[Band_Cutout_Base]]:
        i = i.upper()
        if i in ["B", "G", "R"]:
            return self.cutouts[i]
        elif i in self.filt_names:
            # get which colour filter
            colour = [
                col
                for col in ["B", "G", "R"]
                if i in [cutout.filt.filt_name for cutout in self[col]]
            ]
            assert len(colour) == 1, galfind_logger.critical(
                f"band={i} in != 1 of ['B', 'G', 'R']"
            )
            return self.cutouts[colour][i]

    def __copy__(self) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, deepcopy(value, memo))
        return result

    # need to determine whether this is a good place for this
    def __add__(self):
        pass

    def __sub__(self):
        pass

    @property
    def ID(self) -> str:
        """`str`: Unique identifier shared by all cutouts making up the
        RGB image."""
        ID_list = [
            cutout.ID
            for cutout in np.array(
                [val for val in self.cutouts.values()]
            ).flatten()
        ]
        assert all([ID == ID_list[0] for ID in ID_list])
        return ID_list[0]

    @property
    def survey(self) -> str:
        """`str`: Survey shared by all cutouts making up the RGB image."""
        survey_list = [
            cutout.survey
            for cutout in np.array(
                [val for val in self.cutouts.values()]
            ).flatten()
        ]
        assert all([survey == survey_list[0] for survey in survey_list])
        return survey_list[0]

    @property
    def version(self) -> str:
        """`str`: Data reduction version shared by all cutouts
        making up the RGB image."""
        version_list = [
            cutout.version
            for cutout in np.array(
                [val for val in self.cutouts.values()]
            ).flatten()
        ]
        assert all([version == version_list[0] for version in version_list])
        return version_list[0]

    @property
    def cutout_size(self) -> u.Quantity:
        """`astropy.units.Quantity`: Angular cutout size shared by
        all cutouts making up the RGB image."""
        cutout_size_list = [
            cutout.cutout_size
            for cutout in np.array(
                [val for val in self.cutouts.values()]
            ).flatten()
        ]
        assert all(
            [
                cutout_size == cutout_size_list[0]
                for cutout_size in cutout_size_list
            ]
        )
        return cutout_size_list[0]

    @property
    def meta(self) -> dict:
        """`dict`: Metadata of the first cutout making up the RGB image.

        Consistency of metadata across cutouts is not currently enforced.
        """
        meta_list = [
            cutout.meta
            for cutout in np.array(
                [val for val in self.cutouts.values()]
            ).flatten()
        ]
        # TODO: ensure the same meta for all cutouts
        # try:
        #     assert all(
        #         meta[key] == val
        #         for meta in meta_list
        #         for key, val in meta_list[0].items()
        #     )
        # except AssertionError:
        #     breakpoint()
        return meta_list[0]

    @property
    def name(self):
        """`str`: Human-readable description of the RGB
        colour-filter mapping (e.g. ``"B=F090W,G=F150W,R=F200W"``)."""
        return ",".join(
            f"{colour}={'+'.join(self.get_colour_filt_names(colour))}"
            for colour in ["B", "G", "R"]
        )

    @property
    def filt_names(self) -> List[str]:
        """`list` of `str`: Names of all filters making up the RGB
        image, across all colour channels."""
        return [
            cutout.band_data.filt_name
            for colour in ["B", "G", "R"]
            for cutout in self[colour]
        ]

    @property
    def filterset(self) -> Dict[Multiple_Filter]:
        """`dict` of `str` -> `Multiple_Filter`: Filters making up
        each colour channel of the RGB image."""
        return {
            colour: Multiple_Filter(
                [deepcopy(cutout.filt) for cutout in cutouts]
            )
            for colour, cutouts in self.items()
        }

    @property
    def instr_name(self) -> Optional[str]:
        """`str` or `None`: Name of the instrument for the RGB
        image's blue-channel cutouts."""
        return self.cutouts["B"][0].instr_name

    def get_colour_filt_names(self: Self, colour: str) -> List[str]:
        """Get the filter names making up a given colour channel.

        Parameters
        ----------
        colour : `str`
            Colour channel to query, one of ``"B"``, ``"G"``, ``"R"``.

        Returns
        -------
        `list` of `str`
            Names of the filters combined into `colour`.
        """
        assert colour in ["B", "G", "R"]
        return [cutout.band_data.filt.filt_name for cutout in self[colour]]

    def _get_save_path(self: Self) -> str:
        if self.instr_name is None:
            instr_name = ""
        else:
            instr_name = f"{self.instr_name}/"
        subdir = self["B"][0].cutout_path.split("/")[
            -3
        ]  # get subdir from first cutout
        save_path = (
            f"{config['Cutouts']['CUTOUT_DIR']}/{self.version}/"
            + f"{self.survey}/{instr_name}"
            + f"{self.cutout_size.to(u.arcsec).value:.2f}as/"
            + f"{self.name}/{subdir}/pdf/{self.ID}.pdf"
        )
        funcs.make_dirs(save_path)
        return save_path

    def load(
        self: Self,
        filt_name: str,
        hdu_name: str = "SCI",
    ) -> Union[
        Dict[str, Tuple[Dict[str, Any], np.ndarray]],
        Tuple[Dict[str, Any], np.ndarray],
    ]:
        """Load cutout data for a specific filter.

        Parameters
        ----------
        filt_name : `str`
            Filter name to load data for.
        hdu_name : `str`, optional
            Name of the HDU to load from the FITS file. Default is ``"SCI"``.

        Returns
        -------
        `dict` or `tuple`
            Cutout data for the specified filter; either a tuple of
            ``(header, data)`` or a dictionary of such tuples.
        """
        assert filt_name in self.filt_names
        return self[filt_name].load(hdu_name)

    def plot(
        self: Self,
        ax: Optional[plt.Axes] = None,
        method: str = "lupton",
        plot_type: str = "SCI",
        unit: Optional[u.Unit] = u.uJy,
        rgb_kwargs: Dict[str, Any] = {},
        plot_regions: List[Dict[str, Any]] = [],
        save: bool = False,
        show: bool = False,
        overwrite: bool = False,
        imshow_kwargs: Dict[str, Any] = {},
        *args,
        **kwargs,
    ) -> Optional[List[plt.Text]]:
        """Plot an RGB color composite cutout image.

        Creates a false-color RGB image from the cutouts using either
        Lupton or Trilogy method, optionally overlaying regions,
        saving and/or displaying the figure.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`, optional
            Axes to plot on. A new figure/axes is created if `None`.
            Default is `None`.
        method : `str`, optional
            Plotting method: either ``"lupton"`` (matplotlib-based RGB
            rendering) or ``"trilogy"`` (Trilogy software). Default is
            ``"lupton"``.
        plot_type : `str`, optional
            HDU name to plot (e.g. ``"SCI"`` or ``"WEIGHT"``). Default
            is ``"SCI"``.
        unit : `astropy.units.Unit`, optional
            Flux unit for the image arrays. Default is `astropy.units.uJy`.
        rgb_kwargs : `dict`, optional
            Keyword arguments passed to `make_lupton_rgb`. Default is
            an empty dict.
        plot_regions : `list` of `dict`, optional
            Region specifications to overlay on the image. Default is
            an empty list.
        save : `bool`, optional
            Whether to save the figure to disk. Default is `False`.
        show : `bool`, optional
            Whether to display the figure. Default is `False`.
        overwrite : `bool`, optional
            Whether to overwrite existing files (Trilogy only). Default
            is `False`.
        imshow_kwargs : `dict`, optional
            Keyword arguments passed to `ax.imshow`. Default is an
            empty dict.
        **kwargs
            Additional keyword arguments (currently unused).

        Returns
        -------
        `list` of `matplotlib.text.Text` or `None`
            For Lupton method, a list of text labels for RGB filter names.
            For Trilogy method, `None`.
        """
        method = method.lower()  # make method lowercase
        # construct out_path
        # save_path = (
        #     f"{config['Cutouts']['CUTOUT_DIR']}/{data.version}/"
        #     f"{data.survey}/{self.name}/{method}/{self.ID}.pdf"
        # )
        # funcs.make_dirs(save_path)
        if method == "trilogy":
            galfind_logger.warning(
                "trilogy plots do not currently support image unit conversion!"
            )
            save_path = self._get_save_path()
            # Write trilogy.in
            in_path = save_path.replace(".pdf", "_trilogy.in")
            if not Path(in_path).is_file() or overwrite:
                with open(in_path, "w") as f:
                    for colour, cutout_list in self.cutouts.items():
                        f.write(f"{colour}\n")
                        for cutout in cutout_list:
                            f.write(f"{cutout.cutout_path}[1]\n")
                        f.write("\n")
                    f.write("indir  /\n")
                    out_name = funcs.split_dir_name(save_path, "name").replace(
                        ".pdf", "_trilogy"
                    )
                    f.write(f"outname  {out_name}\n")
                    f.write(
                        f"outdir  {funcs.split_dir_name(save_path, 'dir')}\n"
                    )
                    stamp_size = int(
                        self.cutout_size.to(u.arcsec)
                        / self["B"][0].band_data.pix_scale
                    )
                    f.write(f"samplesize {stamp_size // 2}\n")
                    f.write(f"stampsize  {stamp_size}\n")
                    f.write("showstamps  0\n")
                    f.write("satpercent  0.001\n")
                    f.write("noiselum    0.10\n")
                    f.write("colorsatfac  1\n")
                    f.write("deletetests  1\n")
                    f.write("testfirst   0\n")
                    f.write("sampledx  0\n")
                    f.write("sampledy  0\n")

                funcs.change_file_permissions(in_path)
                # Run trilogy
                sys.path.insert(1, "/nvme/scratch/software/trilogy")
                from trilogy3 import Trilogy

                galfind_logger.info(
                    f"Making trilogy cutout RGB at {save_path}"
                )
                Trilogy(in_path, images=None).run()

        elif method == "lupton":
            if ax is None:
                fig, ax = plt.subplots()
            data = {
                colour: [
                    funcs.flux_image_to_Jy(
                        cutout.load(plot_type)[1], cutout.band_data.ZP
                    )
                    .to(unit)
                    .value
                    for cutout in self[colour]
                ]
                for colour in ["B", "G", "R"]
            }
            # red_mad_std = mad_std(data["R"][0])
            # scale = 0.3 / (5. * red_mad_std)
            # offset = 0.2
            r = data["R"][0]  # * scale + offset
            g = data["G"][0]  # * scale * 1.3 + offset
            b = data["B"][0]  # * scale * 1.6 + offset
            # from astropy.visualization import PercentileInterval
            # stretch_percentile = PercentileInterval(99.9)
            # r = stretch_percentile(r)
            # g = stretch_percentile(g)
            # b = stretch_percentile(b)
            # r = self.channel_scale(
            #     r, satpercent = rgb_kwargs.pop("satpercent", 0.001)
            # )
            # g = self.channel_scale(
            #     g, satpercent = rgb_kwargs.pop("satpercent", 0.001)
            # )
            # b = self.channel_scale(
            #     b, satpercent = rgb_kwargs.pop("satpercent", 0.001)
            # )
            rgb_img = make_lupton_rgb(r, g, b, **rgb_kwargs)
            # norm = ImageNormalize(
            #     vmin=-scale*red_mad_std, vmax=scale*red_mad_std,
            #     stretch=SqrtStretch()
            # )
            ax.imshow(
                rgb_img, origin="lower", **imshow_kwargs
            )  # , norm = norm)
            # turn off grid
            ax.grid(False, which="both")
            # turn off ticks
            ax.set_xticks([])
            ax.set_yticks([])
            # label RGB filters
            all_texts = []
            for i, (colour, plt_colour) in enumerate(
                zip(["B", "G", "R"], ["blue", "green", "red"])
            ):
                filt_name = "+".join(self.get_colour_filt_names(colour))
                txt = ax.text(
                    0.15 + i * 0.35,
                    0.1,
                    filt_name,
                    color=plt_colour,
                    fontweight="bold",
                    fontsize=8.0,
                    ha="center",
                    va="center",
                    path_effects=[
                        pe.withStroke(linewidth=2.0, foreground="white")
                    ],
                    transform=ax.transAxes,
                    zorder=10_000,
                )
                all_texts.append(txt)
            # plot regions
            self._plot_regions(ax, plot_regions)

            if save:
                save_path = self._get_save_path()
                funcs.make_dirs(save_path)
                plt.savefig(save_path)
                funcs.change_file_permissions(save_path)
                galfind_logger.info(f"Saved png cutout to: {save_path}")

            if show:
                plt.show()

            return all_texts

    @staticmethod
    def channel_scale(arr, satpercent=0.001):
        """Scale an image array to [0, 1] with saturation at high percentiles.

        Parameters
        ----------
        arr : array-like
            Input image array.
        satpercent : `float`, optional
            Saturation percentile; pixels above the ``(100 - satpercent)``th
            percentile are clipped. Default is 0.001.

        Returns
        -------
        array-like
            Scaled array with values in [0, 1].
        """
        vmax = np.nanpercentile(arr, 100 - satpercent)
        vmin = np.nanpercentile(arr, 10)
        return (arr - vmin) / (vmax - vmin)


class RGB(RGB_Base):
    """RGB color image cutout for a single galaxy.

    Combines band cutouts in RGB channels to create a false-color image.
    """

    @classmethod
    def from_gal_data(
        cls: Type[Self],
        gal: Galaxy,
        data: Data,
        rgb_bands: Dict[str, Union[str, List[str]]],
        cutout_size: u.Quantity,
        overwrite: bool = False,
    ) -> Self:
        """Create an RGB cutout from a galaxy and data object.

        Parameters
        ----------
        gal : `Galaxy`
            Galaxy to extract cutout for.
        data : `Data`
            Data object containing the survey and bands. Must have the
            same survey as `gal`.
        rgb_bands : `dict` of `str` to `str` or `list` of `str`
            Mapping of RGB channel names (``"B"``, ``"G"``, ``"R"``) to
            filter name(s).
        cutout_size : `astropy.units.Quantity`
            Size of the cutout region.
        overwrite : `bool`, optional
            Whether to overwrite existing cutout files. Default is `False`.

        Returns
        -------
        `RGB`
            An RGB cutout object.
        """
        rgb_bands = {
            key: [val] if isinstance(val, str) else val
            for key, val in rgb_bands.items()
            if key in ["B", "G", "R"]
        }
        assert gal.survey == data.survey, galfind_logger.critical(
            f"{gal.survey=}!={data.survey=}!"
        )
        # make a cutout for each filter
        cutouts = {
            colour: [
                Band_Cutout.from_gal_band_data(
                    gal, data[band], cutout_size, overwrite=overwrite
                )
            ]
            for colour, bands in rgb_bands.items()
            for band in bands
        }
        return cls(cutouts)

    @classmethod
    def from_data_skycoord(
        cls: Type[Self],
        data: Data,
        sky_coord: SkyCoord,
        rgb_bands: Dict[str, List[str]],
    ) -> Self:
        """Create an RGB cutout from a data object and sky coordinate.

        Parameters
        ----------
        data : `Data`
            Data object containing the survey and bands.
        sky_coord : `astropy.coordinates.SkyCoord`
            Sky position to extract the cutout around.
        rgb_bands : `dict` of `str` to `list` of `str`
            Mapping of RGB channel names (``"B"``, ``"G"``, ``"R"``) to
            filter names.

        Returns
        -------
        `RGB`
            An RGB cutout object.
        """
        # make a cutout for each filter
        cutouts = {
            colour: Band_Cutout.from_data_skycoord(data, filt, sky_coord)
            for colour in ["B", "G", "R"]
            for filt in data.filterset
            if filt.filt_name in rgb_bands[colour]
        }
        return cls(cutouts)

    @property
    def ID(self) -> str:
        """Object ID from the cutout.

        Returns
        -------
        `str`
            ID of the object (same for all cutouts in the RGB).
        """
        ID_list = [
            cutout.ID
            for cutout in np.array(
                [val for val in self.cutouts.values()]
            ).flatten()
        ]
        assert all([ID == ID_list[0] for ID in ID_list])
        return ID_list[0]

    @property
    def survey(self) -> str:
        """Survey name for the cutout.

        Returns
        -------
        `str`
            Survey name (same for all cutouts in the RGB).
        """
        surveys = np.unique(
            [
                cutout.band_data.survey
                for cutout in np.array(
                    [val for val in self.cutouts.values()]
                ).flatten()
            ]
        )
        assert len(surveys) == 1, galfind_logger.critical(
            f"Multiple surveys found in RGB cutout: {surveys}"
        )
        return surveys[0]


class Stacked_RGB(RGB_Base):
    """RGB color composite from multiple stacked-band cutouts.

    Combines stacked band cutouts in RGB channels.
    """

    @classmethod
    def from_cat(
        cls: Type[Self],
        cat: Catalogue,
        rgb_bands: Dict[str, Union[str, List[str]]],
        cutout_size: u.Quantity,
        overwrite: bool = False,
    ) -> Self:
        """Create a stacked RGB cutout from a catalogue.

        Parameters
        ----------
        cat : `Catalogue`
            Catalogue object containing the data and filterset.
        rgb_bands : `dict` of `str` to `str` or `list` of `str`
            Mapping of RGB channel names (``"B"``, ``"G"``, ``"R"``) to
            filter names.
        cutout_size : `astropy.units.Quantity`
            Size of the cutout region.
        overwrite : `bool`, optional
            Whether to overwrite existing cutout files. Default is `False`.

        Returns
        -------
        `Stacked_RGB`
            A stacked RGB cutout object.
        """

        # make a stacked cutout for each filter
        stacked_cutouts = {
            colour: [
                Stacked_Band_Cutout.from_cat(
                    cat, band_data.filt, cutout_size, overwrite=overwrite
                )
                for band_data in cat.data
                if band_data.filt_name in rgb_bands[colour]
            ]
            for colour in ["B", "G", "R"]
        }
        return cls(stacked_cutouts)

    @classmethod
    def from_data_skycoords(
        cls: Type[Self],
        data: Data,
        sky_coords: Union[SkyCoord, List[SkyCoord]],
        rgb_bands: Dict[str, List[str]],
    ) -> Self:
        """Create a stacked RGB cutout from a data object and sky coordinates.

        Parameters
        ----------
        data : `Data`
            Data object containing the survey and bands.
        sky_coords : `astropy.coordinates.SkyCoord` or `list` of
            `astropy.coordinates.SkyCoord`
            Sky position(s) to extract the cutout(s) around. If a list,
            cutouts are stacked.
        rgb_bands : `dict` of `str` to `list` of `str`
            Mapping of RGB channel names (``"B"``, ``"G"``, ``"R"``) to
            filter names.

        Returns
        -------
        `Stacked_RGB`
            A stacked RGB cutout object.
        """
        # make a stacked cutout for each filter
        stacked_cutouts = {
            colour: [
                Stacked_Band_Cutout.from_data_skycoords(data, filt, sky_coords)
                for filt in data.filterset
                if filt in rgb_bands[colour]
            ]
            for colour in ["B", "G", "R"]
        }
        return cls(stacked_cutouts)


class Multiple_Cutout_Base(ABC):
    """Base class for collections of cutouts.

    Provides common interface for managing multiple cutout objects with
    shared attributes and plotting capabilities.
    """

    def __init__(
        self: Self,
        cutouts: List[Type[Cutout_Base]],
        name: Optional[str] = None,
    ) -> Self:
        self.cutouts = cutouts
        self.name = name

    def __len__(self) -> int:
        return len(self.cutouts)

    def __iter__(self):
        return iter(self.cutouts)

    def __next__(self):
        if self.iter > len(self) - 1:
            raise StopIteration
        else:
            cutout = self[self.iter]
            self.iter += 1
            return cutout

    def __getitem__(self, index: int) -> Type[Self]:
        # improve here
        return self.cutouts[index]

    def __copy__(self) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, deepcopy(value, memo))
        return result

    def __add__(self, other):
        pass

    def __sub__(self, other):
        pass

    @property
    def cutout_size(self) -> u.Quantity:
        """Cutout region size.

        Returns
        -------
        `astropy.units.Quantity`
            Size of the cutout (same for all cutouts).
        """
        assert all(
            [cutout.cutout_size == self[0].cutout_size for cutout in self]
        )
        return self[0].cutout_size

    @property
    def survey(self) -> str:
        """Survey name.

        Returns
        -------
        `str`
            Survey name (same for all cutouts).
        """
        assert all([cutout.survey == self[0].survey for cutout in self])
        return self[0].survey

    @property
    def version(self) -> str:
        """Data reduction version.

        Returns
        -------
        `str`
            Version identifier (same for all cutouts).
        """
        assert all([cutout.version == self[0].version for cutout in self])
        return self[0].version

    @property
    def instr_name(self) -> str:
        """Instrument name.

        Returns
        -------
        `str`
            Instrument identifier (same for all cutouts).
        """
        assert all(
            [cutout.instr_name == self[0].instr_name for cutout in self]
        )
        return self[0].instr_name

    # @property
    # def filterset(self) -> Multiple_Filter:
    #     _filterset = []
    #     for cutout in self:
    #         if cutout.band_data.__class__.__name__ == "Band_Data":
    #             _filterset.extend([cutout.band_data.filt])
    #     return _filterset

    @abstractmethod
    def _get_save_path(self) -> str:
        pass

    def plot(
        self: Self,
        fig: Optional[plt.Figure] = None,
        ax_arr: Optional[np.ndarray] = None,
        n_rows: int = 1,
        fig_scaling: float = 1.5,
        split_by_instr: bool = False,
        split_by_instr_cmap: str = "plasma",
        imshow_kwargs: Dict[str, Any] = {},
        norm_kwargs: Dict[str, Any] = {},
        label_kwargs: Dict[str, Any] = {},
        plot_regions: Dict[str, List[Union[Dict[str, Any], Type[Patch]]]] = {},
        scalebars: Optional[Dict] = [],
        mask: Optional[List[bool]] = None,
        incl_title: bool = False,
        overwrite: bool = False,
        show: bool = False,
        save: bool = True,
        save_path: Optional[str] = None,
        close_fig: bool = False,
        gridspec_kwargs: Dict[str, Any] = {},
        *args,
        **kwargs,
    ) -> List[plt.Figure, plt.Axes]:
        """Plot multiple cutouts in a grid.

        Parameters
        ----------
        fig : `matplotlib.figure.Figure`, optional
            Figure to plot on. A new one is created if `None`. Default
            is `None`.
        ax_arr : `numpy.ndarray`, optional
            Array of axes to plot on. Axes are generated from `fig` if
            `None`. Default is `None`.
        n_rows : `int`, optional
            Number of rows in the subplot grid. Default is 1.
        fig_scaling : `float`, optional
            Figure size scaling factor. Default is 1.5.
        split_by_instr : `bool`, optional
            Whether to color-code cutouts by instrument. Default is
            `False`.
        split_by_instr_cmap : `str`, optional
            Colormap used to color-code instruments. Default is
            ``"plasma"``.
        imshow_kwargs : `dict`, optional
            Keyword arguments passed to `ax.imshow`. Default is an
            empty dict.
        norm_kwargs : `dict`, optional
            Keyword arguments for image normalization. Default is an
            empty dict.
        label_kwargs : `dict`, optional
            Keyword arguments for labels. Default is an empty dict.
        plot_regions : `dict`, optional
            Regions to overlay, keyed by filter name. Default is an
            empty dict.
        scalebars : `list`, optional
            Scalebar specifications for each cutout. Default is an
            empty list.
        mask : `list` of `bool`, optional
            Boolean mask to select which cutouts to plot. Default is
            `None` (plot all).
        incl_title : `bool`, optional
            Whether to include a title on the figure. Default is `False`.
        overwrite : `bool`, optional
            Whether to overwrite existing saved figures. Default is
            `False`.
        show : `bool`, optional
            Whether to display the figure. Default is `False`.
        save : `bool`, optional
            Whether to save the figure. Default is `True`.
        save_path : `str`, optional
            Custom path to save the figure. Default is `None` (uses
            default naming).
        close_fig : `bool`, optional
            Whether to close the figure after plotting. Default is
            `False`.
        gridspec_kwargs : `dict`, optional
            Keyword arguments for the grid layout. Default is an empty
            dict.
        **kwargs
            Additional keyword arguments (currently unused).

        Returns
        -------
        `list` of [`matplotlib.figure.Figure`, `numpy.ndarray`]
            The figure and axes used for plotting.
        """
        assert n_rows > 0
        if n_rows > len(self):
            n_y = len(self)
        else:
            n_y = n_rows
        n_x = len(self) // n_y
        if len(self) % n_y != 0:
            n_x += 1

        if fig is None:  # not None:
            #     # Delete everything on the figure
            #     fig.clf()
            # else:
            fig = figs.make_fig(n_x, n_y, scaling=fig_scaling)
        # make appropriate axes from the figure and ax_ratio
        if ax_arr is None:
            ax_arr = figs.make_cutout_ax(fig, n_x, n_y, **gridspec_kwargs)
            # remove blank axes
            n_blank_ax = n_x * n_y - len(self)
            [fig.delaxes(ax_arr[-(i + 1)]) for i in range(n_blank_ax)]

        if split_by_instr:
            instr_names, n_bands = np.unique(
                [cutout.band_data.instr_name for cutout in self],
                return_counts=True,
            )
            n_bands = {name: n for name, n in zip(instr_names, n_bands)}
            # instr_names = [name for name in json.loads( \
            #    config["Other"]["INSTRUMENT_NAMES"]) if name in instr_names]
            # determine appropriate colours from the colour map
            instr_split_cmap = plt.get_cmap(
                split_by_instr_cmap, len(instr_names)
            )
            norm = Normalize(vmin=0, vmax=len(instr_names) - 1)
            colours = {
                name: instr_split_cmap(norm(i))
                for i, name in enumerate(instr_names)
            }
            plot_band_counts = {name: 0 for name in instr_names}
            for ax, cutout in zip(ax_arr, self):
                plot_band_counts[cutout.band_data.instr_name] += 1
                colour = colours[cutout.band_data.instr_name]
                ax.patch.set_edgecolor(colour)
                ax.patch.set_linewidth(12.0)
                if (
                    n_bands[cutout.band_data.instr_name]
                    == plot_band_counts[cutout.band_data.instr_name]
                ):
                    ax.text(
                        1.05,
                        0.0,
                        cutout.band_data.instr_name.replace("_", " "),
                        transform=ax.transAxes,
                        c=colour,
                        path_effects=[
                            pe.withStroke(linewidth=3.0, foreground="white")
                        ],
                        ha="right",
                        va="center",
                    )

        # if mask is not None:
        #     assert len(mask) == len(self)
        #     masked_self = self[mask]
        # else:
        #     masked_self = self

        if scalebars == []:
            scalebars = list(itertools.repeat([], len(self)))
        assert len(scalebars) == len(self)
        # get shared attributes
        attrs = ["survey", "ID", "filt_name"]
        shared_attrs = {
            name: np.unique(
                [
                    getattr(cutout, name, "")
                    for cutout in self
                    if hasattr(cutout, name)
                ]
            )[0]
            for name in attrs
            if len(
                np.unique(
                    [
                        getattr(cutout, name, "")
                        for cutout in self
                        if hasattr(cutout, name)
                    ]
                )
            )
            == 1
        }

        if incl_title:
            # determine title from shared attributes
            title = ""
        else:
            title = None
        for i, (ax, cutout, scalebars_band) in enumerate(
            zip(ax_arr, self, scalebars)
        ):
            if isinstance(cutout, RGB):
                plot_regions_band = []
            else:
                filt_name = cutout.band_data.filt_name
                if filt_name in plot_regions.keys():
                    plot_regions_band = plot_regions[filt_name]
                else:
                    plot_regions_band = []

            if "label" not in label_kwargs.keys():
                label_kwargs["label"] = "\n".join(
                    [
                        str(getattr(cutout, name, ""))
                        for name in attrs
                        if name not in shared_attrs.keys()
                        and hasattr(cutout, name)
                    ]
                )

            cutout.plot(
                ax,
                imshow_kwargs=imshow_kwargs,
                norm_kwargs=norm_kwargs,
                plot_regions=plot_regions_band,
                scalebars=scalebars_band,
                label_kwargs=label_kwargs,
                rgb_kwargs=kwargs.get("rgb_kwargs", {}),
                overwrite=overwrite,
                show=False,
                save=False,
            )

        if title is not None:
            fig.suptitle(title)

        if save:
            if save_path is None:
                save_path = self._get_save_path()
            funcs.make_dirs(save_path)
            plt.savefig(save_path, bbox_inches="tight")
            funcs.change_file_permissions(save_path)
            galfind_logger.info(f"Saved cutout plot to: {save_path}")
        if show:
            plt.show()
        if close_fig:
            plt.close(fig)
        return fig, ax_arr


# Galaxy_Cutouts
class Multiple_Band_Cutout(Multiple_Cutout_Base):
    """Collection of band cutouts for a single galaxy across multiple
    filters."""

    # Each plot is a different Filter
    @classmethod
    def from_cat(
        cls: Type[Self],
        cat: Catalogue,
        cutout_size: u.Quantity,
        overwrite: bool = False,
    ) -> Self:
        """Create cutouts for all galaxies in a catalogue.

        Parameters
        ----------
        cat : `Catalogue`
            Catalogue of galaxies.
        cutout_size : `astropy.units.Quantity`
            Size of the cutout region.
        overwrite : `bool`, optional
            Whether to overwrite existing cutout files. Default is `False`.

        Returns
        -------
        `Multiple_Cutout_Base`
            Collection of cutouts for the catalogue.
        """
        # make a cutout for each filter
        cutouts = [
            Stacked_Band_Cutout.from_cat(cat, filt, cutout_size, overwrite)
            for filt in cat.data.filterset
        ]
        return cls(cutouts)

    @classmethod
    def from_gal_data(
        cls: Type[Self],
        gal: Galaxy,
        data: Data,
        cutout_size: u.Quantity,
        overwrite: bool = False,
    ) -> Self:
        """Create cutouts from galaxy data across all bands.

        Parameters
        ----------
        gal : `Galaxy`
            Galaxy object to create cutouts for.
        data : `Data`
            Data object containing band information.
        cutout_size : `astropy.units.Quantity`
            Size of the cutout region.
        overwrite : `bool`, optional
            Whether to overwrite existing cutout files. Default is `False`.

        Returns
        -------
        `Multiple_Cutout_Base`
            Collection of cutouts for each band.
        """
        # make a cutout for each filter
        cutouts = [
            Band_Cutout.from_gal_band_data(
                gal, band_data, cutout_size, overwrite
            )
            for band_data in data
        ]
        return cls(cutouts)

    @classmethod
    def from_data_skycoord(
        cls: Type[Self],
        data: Data,
        sky_coord: SkyCoord,
        cutout_size: u.Quantity,
        **meta,
    ) -> Self:
        """Create cutouts at a given sky coordinate from data.

        Parameters
        ----------
        data : `Data`
            Data object containing band information.
        sky_coord : `astropy.coordinates.SkyCoord`
            Sky coordinate for the cutout center.
        cutout_size : `astropy.units.Quantity`
            Size of the cutout region.
        **meta
            Additional metadata for the cutouts.

        Returns
        -------
        `Multiple_Cutout_Base`
            Collection of cutouts for each band at the specified coordinate.
        """
        # make a cutout for each filter
        cutouts = [
            Band_Cutout.from_data_skycoord(band_data, sky_coord, cutout_size)
            for band_data in data
        ]
        return cls(cutouts)

    @classmethod
    def from_data_skycoords(
        cls: Type[Self],
        data: Data,
        sky_coords: Union[SkyCoord, List[SkyCoord]],
        cutout_size: u.Quantity,
    ) -> Self:
        """Create stacked cutouts at multiple sky coordinates.

        Parameters
        ----------
        data : `Data`
            Data object containing band information.
        sky_coords : `astropy.coordinates.SkyCoord` or `list` of `SkyCoord`
            Sky coordinates for cutout centers.
        cutout_size : `astropy.units.Quantity`
            Size of the cutout region.

        Returns
        -------
        `Multiple_Cutout_Base`
            Collection of stacked cutouts for each band.
        """
        # make a cutout for each filter
        cutouts = [
            Stacked_Band_Cutout.from_data_skycoords(
                band_data, sky_coords, cutout_size
            )
            for band_data in data
        ]
        return cls(cutouts)

    @property
    def ID(self) -> str:
        """Galaxy or object identifier (same for all cutouts).

        Returns
        -------
        `str`
            ID of the galaxy or object.
        """
        assert all([cutout.ID == self[0].ID for cutout in self])
        return self[0].ID

    def _get_save_path(self: Self) -> str:
        if self.instr_name is None:
            instr_name = ""
        else:
            instr_name = f"{self.instr_name}/"
        subdir = self[0].cutout_path.split("/")[
            -3
        ]  # get subdir from first cutout
        save_path = (
            f"{config['Cutouts']['CUTOUT_DIR']}/{self.version}/"
            + f"{self.survey}/{instr_name}"
            + f"{self.cutout_size.to(u.arcsec).value:.2f}as/"
            + f"multi_band/{subdir}/png/{self.ID}.pdf"
        )
        # '+'.join(filt.filt_name for filt in self.filterset)
        funcs.make_dirs(save_path)
        return save_path

    # part of __getattr__?
    # @property
    # def filterset(self) -> Multiple_Filter:
    #     return Multiple_Filter(
    #         [deepcopy(cutout.filt) for cutout in self.cutouts]
    #     )


class Catalogue_Cutouts(Multiple_Cutout_Base):
    """Collection of band cutouts for multiple galaxies in a single filter.

    Each cutout is for a different galaxy in the same filter.
    """

    def __init__(
        self: Self, cutouts: List[Type[Cutout_Base]], ID: str
    ) -> Self:
        """Initialize a catalogue cutouts collection.

        Parameters
        ----------
        cutouts : `list` of `Cutout_Base`
            Band cutouts for multiple galaxies.
        ID : `str`
            Identifier for the cutout collection.
        """
        # each plot is a different galaxy using the same filter
        self.ID = ID
        super().__init__(cutouts)

    @classmethod
    def from_cat_filt(
        cls: Type[Self],
        cat: Catalogue,
        filt: Union[str, Filter],
        cutout_size: u.Quantity,
        overwrite: bool = False,
    ) -> Self:
        """Create catalogue cutouts for a specific filter.

        Parameters
        ----------
        cat : `Catalogue`
            Catalogue containing the galaxies to extract cutouts for.
        filt : `str` or `Filter`
            Filter name or object to create cutouts for.
        cutout_size : `astropy.units.Quantity`
            Size of the cutout region.
        overwrite : `bool`, optional
            Whether to overwrite existing cutout files. Default is `False`.

        Returns
        -------
        `Catalogue_Cutouts`
            A catalogue cutouts object.
        """
        if isinstance(filt, Filter):
            filt = filt.filt_name
        cutouts = [
            Band_Cutout.from_gal_band_data(
                gal, cat.data[filt], cutout_size, overwrite
            )
            for gal in cat
        ]

        return cls(cutouts, cat.crop_name)

    @property
    def survey(self) -> str:
        """Survey name(s) for the cutouts.

        Returns
        -------
        `str`
            Combined survey names (``"+"`` separated if multiple).
        """
        unique_surveys = np.unique([cutout.survey for cutout in self])
        return "+".join(unique_surveys)

    @property
    def version(self) -> str:
        """Data reduction version(s).

        Returns
        -------
        `str`
            Combined version identifiers (``"+"`` separated if multiple).
        """
        unique_versions = np.unique([cutout.version for cutout in self])
        return "+".join(unique_versions)

    @property
    def instr_name(self) -> str:
        """Instrument name.

        Returns
        -------
        `str`
            Instrument identifier (same for all cutouts).
        """
        assert all(
            [cutout.instr_name == self[0].instr_name for cutout in self]
        )
        return self[0].instr_name

    def _get_save_path(self: Self) -> str:
        if self.instr_name is None:
            instr_name = ""
        else:
            instr_name = f"{self.instr_name}/"
        save_path = (
            f"{config['Cutouts']['CUTOUT_DIR']}/{self.version}/"
            + f"{self.survey}/{instr_name}"
            + f"{self.cutout_size.to(u.arcsec).value:.2f}as/"
            + f"{self[0].band_data.filt_name}/pdf/{self.ID}.pdf"
        )
        # '+'.join(filt.filt_name for filt in self.filterset)
        funcs.make_dirs(save_path)
        return save_path

    def plot(
        self: Self,
        fig: Optional[plt.Figure] = None,
        fig_scaling: float = 1.5,
        imshow_kwargs: Dict[str, Any] = {},
        norm_kwargs: Dict[str, Any] = {},
        plot_regions: List[List[Dict]] = {},
        scalebars: Optional[Dict] = [],
        mask: Optional[List[bool]] = None,
        show: bool = False,
        save: bool = True,
        save_path: Optional[str] = None,
        *args,
        **kwargs,
    ) -> plt.Figure:
        """Plot catalogue cutouts in an automatically-sized grid.

        Parameters
        ----------
        fig : `matplotlib.figure.Figure`, optional
            Figure to plot on. A new one is created if `None`. Default
            is `None`.
        fig_scaling : `float`, optional
            Figure size scaling factor. Default is 1.5.
        imshow_kwargs : `dict`, optional
            Keyword arguments passed to `ax.imshow`. Default is an
            empty dict.
        norm_kwargs : `dict`, optional
            Keyword arguments for image normalization. Default is an
            empty dict.
        plot_regions : `list` of `list` of `dict`, optional
            Regions to overlay. Default is an empty dict.
        scalebars : `list`, optional
            Scalebar specifications for each cutout. Default is an
            empty list.
        mask : `list` of `bool`, optional
            Boolean mask to select which cutouts to plot. Default is
            `None` (plot all).
        show : `bool`, optional
            Whether to display the figure. Default is `False`.
        save : `bool`, optional
            Whether to save the figure. Default is `True`.
        save_path : `str`, optional
            Custom path to save the figure. Default is `None` (uses
            default naming).
        **kwargs
            Additional keyword arguments passed to parent `plot` method.

        Returns
        -------
        `matplotlib.figure.Figure`
            The figure object.
        """
        n_rows = np.sqrt(2 * len(self))
        n_rows = int(n_rows // 1)
        if n_rows % 1 != 0:
            n_rows += 1
        return super().plot(
            fig=fig,
            n_rows=n_rows,
            fig_scaling=fig_scaling,
            split_by_instr=False,
            imshow_kwargs=imshow_kwargs,
            norm_kwargs=norm_kwargs,
            plot_regions=plot_regions,
            scalebars=scalebars,
            mask=mask,
            show=show,
            save=save,
            save_path=save_path,
        )


class Multiple_RGB(Multiple_Cutout_Base):
    """Collection of RGB cutouts for multiple galaxies."""

    # Each plot is a different Galaxy

    @classmethod
    def from_cat(
        cls: Type[Self],
        cat: Catalogue,
        rgb_bands: Dict[str, List[str]],
        cutout_size: u.Quantity,
    ) -> Self:
        """Create multiple RGB cutouts from a catalogue.

        Parameters
        ----------
        cat : `Catalogue`
            Catalogue of galaxies to extract cutouts for.
        rgb_bands : `dict` of `str` to `list` of `str`
            Mapping of RGB channel names (``"B"``, ``"G"``, ``"R"``) to
            filter names.
        cutout_size : `astropy.units.Quantity`
            Size of the cutout region.

        Returns
        -------
        `Multiple_RGB`
            A multiple RGB cutouts object.
        """
        # make a cutout for each filter
        cutouts = [
            RGB.from_gal_data(gal, cat.data, rgb_bands, cutout_size)
            for gal in cat
        ]
        return cls(cutouts, cat.crop_name)

    @classmethod
    def from_data_skycoords(
        cls: Type[Self],
        data: Data,
        sky_coords: Union[SkyCoord, List[SkyCoord]],
        rgb_bands: Dict[str, List[str]],
    ) -> Self:
        """Create multiple RGB cutouts from sky coordinates.

        Parameters
        ----------
        data : `Data`
            Data object containing the survey and bands.
        sky_coords : `list` of `astropy.coordinates.SkyCoord`
            Sky positions to extract cutouts around.
        rgb_bands : `dict` of `str` to `list` of `str`
            Mapping of RGB channel names (``"B"``, ``"G"``, ``"R"``) to
            filter names.

        Returns
        -------
        `Multiple_RGB`
            A multiple RGB cutouts object.
        """
        # make a cutout for each filter
        cutouts = [
            RGB.from_data_skycoord(data, sky_coord, rgb_bands)
            for sky_coord in sky_coords
        ]
        return cls(cutouts)

    @classmethod
    def from_multiple_cat(
        cls: Type[Self],
        cats: Union[List[Catalogue], Multiple_Catalogue],
        rgb_bands: Dict[str, List[str]],
    ) -> Self:
        """Create stacked RGB cutouts from multiple catalogues.

        Parameters
        ----------
        cats : `list` of `Catalogue` or `Multiple_Catalogue`
            Catalogues to extract stacked cutouts from.
        rgb_bands : `dict` of `str` to `list` of `str`
            Mapping of RGB channel names (``"B"``, ``"G"``, ``"R"``) to
            filter names.

        Returns
        -------
        `Multiple_RGB`
            A multiple RGB cutouts object.
        """
        # make a cutout for each filter
        cutouts = [Stacked_RGB.from_cat(cat, rgb_bands) for cat in cats]
        return cls(cutouts)

    @classmethod
    def from_multiple_data_skycoords(
        cls: Type[Self],
        data_arr: Union[List[Data], Multiple_Data],
        sky_coords: Union[List[SkyCoord], List[List[SkyCoord]]],
        rgb_bands: Dict[str, List[str]],
    ) -> Self:
        """Create stacked RGB cutouts from multiple data objects and
        sky coordinates.

        Parameters
        ----------
        data_arr : `list` of `Data` or `Multiple_Data`
            Data objects providing the survey and bands.
        sky_coords : `list` of `astropy.coordinates.SkyCoord` or `list`
            of `list` of `astropy.coordinates.SkyCoord`
            Sky position(s) for each data object. If lists of lists,
            cutouts are stacked.
        rgb_bands : `dict` of `str` to `list` of `str`
            Mapping of RGB channel names (``"B"``, ``"G"``, ``"R"``) to
            filter names.

        Returns
        -------
        `Multiple_RGB`
            A multiple RGB cutouts object.
        """
        # make a cutout for each filter
        cutouts = [
            Stacked_RGB.from_data_skycoords(data, sky_coord, rgb_bands)
            for data, sky_coord in zip(data_arr, sky_coords)
        ]
        return cls(cutouts)

    @property
    def rgb_bands(self: Self) -> Dict[str, List[str]]:
        """RGB filter bands.

        Returns
        -------
        `dict` of `str` to `list` of `str`
            Mapping of RGB channel names to filter names (same for all
            cutouts).
        """
        rgb_bands = self[0].rgb_bands
        assert all(
            [cutout.rgb_bands == rgb_bands for cutout in self]
        ), galfind_logger.critical("All cutout rgb_bands must be the same!")
        return rgb_bands

    def _get_save_path(self: Self) -> str:
        if self.instr_name is None:
            instr_name = ""
        else:
            instr_name = f"{self.instr_name}/"
        subdir = self[0]["B"][0].cutout_path.split("/")[
            -3
        ]  # get subdir from first cutout
        save_path = (
            f"{config['Cutouts']['CUTOUT_DIR']}/{self.version}/"
            + f"{self.survey}/{instr_name}"
            + f"{self.cutout_size.to(u.arcsec).value:.2f}as/"
            + f"{self[0].name}/{subdir}.pdf"
        )  # /png/{self.ID}
        if self.name is not None:
            save_path = save_path.replace(".pdf", f"_{self.name}.pdf")
        funcs.make_dirs(save_path)
        return save_path

    def plot(
        self: Self,
        method: str = "trilogy",
        plot_regions: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        save: bool = True,
        overwrite: bool = False,
        gridspec_kwargs: Dict[str, Any] = {},
        imshow_kwargs: Dict[str, Any] = {"rasterized": True},
        *args,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot multiple RGB cutouts using the Trilogy method.

        Parameters
        ----------
        method : `str`, optional
            Plotting method; currently only ``"trilogy"`` is supported.
            Default is ``"trilogy"``.
        plot_regions : `dict`, optional
            Regions to overlay on each cutout. Default is `None`.
        save : `bool`, optional
            Whether to save the figure to disk. Default is `True`.
        overwrite : `bool`, optional
            Whether to overwrite existing Trilogy outputs. Default is
            `False`.
        gridspec_kwargs : `dict`, optional
            Keyword arguments for the subplot grid layout. Default is
            an empty dict.
        imshow_kwargs : `dict`, optional
            Keyword arguments for `ax.imshow`. Default is
            ``{"rasterized": True}``.
        **kwargs
            Additional keyword arguments (currently unused).

        Returns
        -------
        `tuple` of (`matplotlib.figure.Figure`, `numpy.ndarray`)
            The figure and axes array used for plotting.
        """

        if method == "trilogy":
            # make axes
            fig, axs = figs.make_rectangular_fig(
                len(self),
                xy_ratio=2 / 3,
                **gridspec_kwargs,
            )
            # plot each cutout
            for ax, cutout in zip(axs, self):
                try:
                    failed = False
                    cutout.plot(method="trilogy", overwrite=overwrite)
                except Exception:
                    failed = True
                save_path = cutout._get_save_path()
                trilogy_path = save_path.replace(".pdf", "_trilogy.png")
                if not Path(trilogy_path).is_file():
                    galfind_logger.error(
                        "Trilogy failed to produce output for "
                        + f"cutout ID={cutout.ID} at {trilogy_path}"
                    )
                    failed = True
                if not failed:
                    # Write trilogy.in
                    img = mpimg.imread(trilogy_path)
                    ax.imshow(img, origin="lower", **imshow_kwargs)
                    if plot_regions is not None:
                        cutout._plot_regions(ax, plot_regions)
                    for i, (colour, plt_colour) in enumerate(
                        zip(["B", "G", "R"], ["blue", "green", "red"])
                    ):
                        filt_name = "+".join(
                            cutout.get_colour_filt_names(colour)
                        )
                        ax.text(
                            0.05,
                            0.2 - i * 0.075,
                            filt_name,
                            color=plt_colour,
                            fontweight="bold",
                            fontsize=10.0,
                            ha="left",
                            va="center",
                            path_effects=[
                                pe.withStroke(
                                    linewidth=2.0, foreground="white"
                                )
                            ],
                            transform=ax.transAxes,
                        )
                else:
                    # plot a large red cross to indicate failure
                    ax.plot(
                        [0, 1],
                        [0, 1],
                        transform=ax.transAxes,
                        color="red",
                        linewidth=5.0,
                    )
                ax.text(
                    0.05,
                    0.95,
                    f"{cutout.survey};" + "\n" + f"ID={cutout.ID}",
                    transform=ax.transAxes,
                    color="white",
                    fontweight="bold",
                    fontsize=11.0,
                    ha="left",
                    va="top",
                    path_effects=[
                        pe.withStroke(linewidth=2.0, foreground="black")
                    ],
                )
            # remove unused axes
            for i in range(len(self), len(axs)):
                fig.delaxes(axs[i])

            # save plot
            if save:
                save_path = self._get_save_path().replace(
                    ".pdf", "_trilogy.pdf"
                )
                funcs.make_dirs(save_path)
                plt.savefig(save_path, bbox_inches="tight")
                funcs.change_file_permissions(save_path)
                galfind_logger.info(
                    f"Saved trilogy RGB cutout plot to: {save_path}"
                )
            return fig, axs
        else:
            return super().plot(
                plot_regions=plot_regions,
                save=save,
                overwrite=overwrite,
                *args,
                **kwargs,
            )

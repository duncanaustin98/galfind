from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import astropy.units as u
from copy import deepcopy
from astropy.nddata import Cutout2D
from pathlib import Path
import sys
import itertools
import json
from astropy.stats import mad_std
import matplotlib.patheffects as pe
import matplotlib.patches as patches
from matplotlib.patches import Patch
from matplotlib.artist import Artist, ArtistInspector
from matplotlib.colors import Colormap, Normalize
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
from astropy.io import fits
from astropy.visualization import (
    ImageNormalize,
    LinearStretch,
    LogStretch,
    SqrtStretch,
    ManualInterval,
)
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from typing import (
    Any,
    List,
    Union,
    NoReturn,
    Optional,
    Tuple,
    Dict,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from . import Band_Data
    from . import Data
    from . import Multiple_Data
    from . import Galaxy
    from . import Catalogue
    from . import Multiple_Catalogue
    from . import Filter
    from . import Multiple_Filter
    from . import Morphology_Result
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import Filter, Band_Data
from . import config, galfind_logger, astropy_cosmo, figs
from . import useful_funcs_austind as funcs


class Cutout_Base(ABC):

    @property
    @abstractmethod
    def ID(self) -> str:
        pass

    @property
    @abstractmethod
    def meta(self) -> dict:
        pass

    @abstractmethod
    def load(
        self: Self,
        hdu_name: str = "SCI"
    ) -> Union[Dict[str, Tuple[Dict[str, Any], np.ndarray]], Tuple[Dict[str, Any], np.ndarray]]:
        pass

    @abstractmethod
    def plot(self) -> plt.Axes:
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
            # add circles to show extraction aperture and sextractor FLUX_RADIUS
            xpos = np.mean(ax.get_xlim())
            ypos = np.mean(ax.get_ylim())
            for plot_region in plot_regions:
                skip_region = False
                if isinstance(plot_region, dict):
                    assert "aper_diam" in plot_region.keys()
                    pix_scale = (
                        self.meta["SIZE_AS"]
                        * u.arcsec
                        / self.meta["SIZE_PIX"]
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
                elif isinstance(plot_region, tuple([patches.Ellipse] + patches.Ellipse.__subclasses__())):
                    region = plot_region
                    if region.center == (-99., -99.):
                        region.set_center((xpos, ypos))
                    # update default kwargs with pre-set ones
                    blank_patch = Patch()
                    kwarg_names = ArtistInspector(blank_patch).get_setters()
                    kwarg_names.remove("transform")
                    blank_kwargs = {key: value for key, value in \
                        ArtistInspector(blank_patch).properties().items() \
                        if key in kwarg_names}
                    reg_kwargs = {key: value for key, value in \
                        ArtistInspector(region).properties().items() \
                        if key in kwarg_names}
                    assert len(blank_kwargs) == len(reg_kwargs)
                    added_reg_kwargs = {key: value for key, value in \
                        reg_kwargs.items() if value != blank_kwargs[key]}
                    for key, value in added_reg_kwargs.items():
                        def_plot_region_kwargs[key] = value
                    # set region kwargs
                    region.set(**def_plot_region_kwargs)
                else:
                    skip_region = True
                    galfind_logger.warning(
                        f"{plot_region=} does not contain " + \
                        f"'aper_diam' or {type(plot_region)=} not in " + \
                        tuple([patches.Ellipse] + patches.Ellipse.__subclasses__()) + \
                        ", skipping!"
                    )
                if not skip_region:
                    ax.add_patch(region)


class Band_Cutout_Base(Cutout_Base, ABC):
    def __init__(
        self: Self, 
        cutout_path: str, 
        band_data: Band_Data, 
        cutout_size: u.Quantity
    ) -> Self:
        assert Path(cutout_path).is_file(), \
            galfind_logger.critical(
                f"Cutout path {cutout_path} does not exist!"
            )
        self.cutout_path = cutout_path
        self.band_data = band_data
        self.cutout_size = cutout_size

    def __repr__(self) -> str:
        return f"{self.__class__.__name__.upper()}({self.ID}" + \
            f",{self.filt_name}" + \
            f",{self.cutout_size.to(u.arcsec).value:.2f}as)"
    
    def __str__(self) -> str:
        output_str = funcs.line_sep
        output_str += f"{repr(self)}:\n"
        output_str += funcs.line_sep
        output_str += f"Cutout path: {self.cutout_path}\n"
        if hasattr(self, "morph_fits"):
            if len(self.morph_fits) > 0:
                output_str += f"Morphology fits:\n"
                output_str += f"{repr(self.morph_fits)}\n"
        output_str += f"Meta:\n"
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
        return self._get_ID(self.meta)
        
    @staticmethod
    def _get_ID(meta: Dict[str, Any]) -> str:
        if "ID" in meta:
            return meta["ID"]
        else:
            return f"({meta['RA']:.5f},{meta['DEC']:.5f})"

    @property
    def instr_name(self) -> str:
        return self._get_instr_name(self.meta)

    @staticmethod
    def _get_instr_name(meta: Dict[str, Any]) -> str:
        if "INSTR" in meta.keys():
            return meta["INSTR"]
        else:
            return None

    @property
    def meta(self) -> dict:
        return dict(self.load("PRIMARY")[0])

    # sky_coord, survey, version may need to be stored in Cutout_Base
    @property
    def sky_coord(self) -> SkyCoord:
        return SkyCoord(
            ra=self.meta["RA"] * u.deg, dec=self.meta["DEC"] * u.deg
        )
    
    @property
    def survey(self) -> str:
        return self.band_data.survey
    
    @property
    def version(self) -> str:
        return self.band_data.version
    
    @property
    def filt_name(self) -> Filter:
        return self.band_data.filt_name

    @staticmethod
    def _get_save_path(
        band_data: Band_Data, 
        cutout_size: u.Quantity, 
        ID: str, 
        instr_name: Optional[str],
        data_type: str
    ) -> str:
        assert data_type in ["data", "png"], \
            galfind_logger.critical(f"Invalid {data_type=}")
        if data_type == "data":
            ext = ".fits"
        elif data_type == "png":
            ext = ".png"
        if instr_name is None:
            instr_name = ""
        else:
            instr_name = f"{instr_name}/"
        save_path = f"{config['Cutouts']['CUTOUT_DIR']}/{band_data.version}/" + \
            f"{band_data.survey}/{instr_name}{cutout_size.to(u.arcsec).value:.2f}as/" + \
            f"{band_data.filt_name}/{data_type}/{ID}{ext}"
        funcs.make_dirs(save_path)
        return save_path

    def load(
        self: Self, 
        hdu_name: str = "SCI"
    ) -> Union[Dict[str, Tuple[Dict[str, Any], np.ndarray]], Tuple[Dict[str, Any], np.ndarray]]:
        if hdu_name is None:
            hdul = fits.open(self.cutout_path, ignore_missing_simple = True)
            return {hdu.name: (dict(hdu.header), hdu.data) for hdu in hdul}
        else:
            hdu = fits.open(self.cutout_path, ignore_missing_simple = True)[hdu_name]
            return dict(hdu.header), hdu.data

    def update_morph_fits(
        self: Self,
        morph_results: Union[Morphology_Result, List[Morphology_Result]],
        overwrite: bool = False,
    ) -> NoReturn:
        from . import Morphology_Result
        if overwrite or not hasattr(self, "morph_fits"):
            self.morph_fits = {}
        if isinstance(morph_results, Morphology_Result):
            morph_results = [morph_results]
        self.morph_fits = {**self.morph_fits, **{result.fitter.name: result for result in morph_results}}

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
    ) -> NoReturn:
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
                    cutout_data, norm = self._EPOCHS_cutout_scaling(cutout_data, **norm_kwargs)
                    imshow_kwargs["norm"] = norm
        for key, value in imshow_kwargs.items():
            def_imshow_kwargs[key] = value
        # plot cutout
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
        for key, value in label_kwargs.items():
            def_label_kwargs[key] = value
        label = def_label_kwargs.pop("label", None)
        if label is not None:
            text_unpack_kwargs = deepcopy(def_label_kwargs)
            text_unpack_kwargs.pop("xpos")
            text_unpack_kwargs.pop("ypos")
            # plot text for band label
            ax.text(
                def_label_kwargs["xpos"],
                def_label_kwargs["ypos"],
                label,
                transform = ax.transAxes,
                **text_unpack_kwargs
            )

        # plot any regions wanted
        self._plot_regions(ax, plot_regions)

        # add scalebars
        if len(scalebars) > 0:
            pix_scale = (
                self.meta["SIZE_AS"]
                * u.arcsec
                / self.meta["SIZE_PIX"]
            )
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
                    label = f"{str(scalebar_kwargs['as_length']):.1f}\""
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
            save_path = self._get_save_path(self.band_data, self.cutout_size, self.ID, self.instr_name, "png")
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
        **kwargs
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
    @classmethod
    def from_gal_band_data(
        cls: Type[Self],
        gal: Galaxy,
        band_data: Band_Data,
        cutout_size: u.Quantity,
        overwrite: bool = False,
    ) -> Self:
        # TODO: ensure in some way that the galaxy arises from the data
        # extract the position of the galaxy
        sky_coord = gal.sky_coord
        meta = {"ID": gal.ID, "INSTR": gal.cat_filterset.instrument_name}
        meta_keys = ["Re", "FLUX_AUTO", "MAG_AUTO", "KRON_RADIUS", "A_IMAGE", "B_IMAGE", "THETA_IMAGE", "A_IMAGE_AS", "B_IMAGE_AS"]
        suffixes = ["_AS", "_JY", "", "", "", "", "", "", ""]
        filt_name = band_data.filt.band_name
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
                    **{f"{meta_key.replace('sex_', '').upper()}{suffix.upper()}": attr}
                }
            else:
                galfind_logger.debug(f"No {meta_key} found for {repr(gal)}!")
        return cls.from_data_skycoord(
            band_data, 
            sky_coord, 
            cutout_size, 
            meta = meta, 
            overwrite = overwrite
        )

    @classmethod
    def from_data_skycoord(
        cls: Type[Self],
        band_data: Band_Data,
        sky_coord: SkyCoord,
        cutout_size: u.Quantity,
        meta: dict = {},
        overwrite: bool = False,
    ) -> Self:
        # make cutout from data at the sky co-ordinate and save
        meta = {
            **meta,
            "SURVEY": band_data.survey,
            "VERSION": band_data.version,
            "RA": sky_coord.ra.value,
            "DEC": sky_coord.dec.value,
            "SIZE_AS": cutout_size.to(u.arcsec).value,
            "SIZE_PIX": (cutout_size / band_data.pix_scale) \
                .to(u.dimensionless_unscaled).value,
        }
        ID = cls._get_ID(meta)
        instr_name = cls._get_instr_name(meta)
        save_path = cls._get_save_path(band_data, cutout_size, ID, instr_name, data_type="data")
        cls._make_cutout(band_data, sky_coord, cutout_size, save_path, meta, overwrite)
        band_data = cls._update_band_data(band_data, save_path)
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
    #     self.cutout_path = self._get_save_path(self.band_data, cutout_size, self.ID, "data")
    #     self._make_cutout(
    #         self.band_data, 
    #         sky_coord, 
    #         cutout_size, 
    #         self.cutout_path, 
    #         meta, 
    #         overwrite=overwrite
    #         )
    #     self.band_data = self._update_band_data(self.band_data, self.cutout_path)

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
            im_data, im_header, seg_data, seg_header = \
                band_data.load_data(incl_mask=False)
            pix_scale = band_data.pix_scale
            data_dict = {
                "SCI": im_data,
                "SEG": seg_data,
                "RMS_ERR": band_data.load_rms_err(),
                "WHT": band_data.load_wht(),
            }
            hdul = [fits.PrimaryHDU(header=fits.Header(meta))]

            cutout_size_pix = (cutout_size / pix_scale) \
                .to(u.dimensionless_unscaled).value
            
            for i, (label_i, data_i) in enumerate(data_dict.items()):
                if i == 0 and label_i == "SCI":
                    sci_shape = data_i.shape
                if data_i is None:
                    galfind_logger.warning(
                        f"No data found for {label_i} in {band_data.filt_name}!"
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
                        galfind_logger.info(
                            f"Created cutout for {label_i} in {band_data.filt_name}"
                        )
                    else:
                        galfind_logger.warning(
                            f"Incorrect data shape. {data_i=} != {sci_shape=}, skipping extension!"
                        )
            funcs.make_dirs(save_path)
            fits_hdul = fits.HDUList(hdul)
            fits_hdul.writeto(save_path, overwrite=True)
            funcs.change_file_permissions(save_path)
            galfind_logger.info(f"Saved fits cutout to: {save_path}")
        else:
            ID = Band_Cutout_Base._get_ID(meta)
            galfind_logger.debug(
                f"Already made fits cutout for {band_data.survey}" + \
                f" {band_data.version} {ID} {band_data.filt_name}"
            )

    @staticmethod
    def _update_band_data(
        band_data: Band_Data,
        cutout_path: str,
    ) -> Band_Data:
        new_band_data = Band_Data(
            band_data.filt, 
            band_data.survey, 
            band_data.version, 
            cutout_path, 
            1,
            cutout_path,
            3,
            cutout_path,
            4,
            pix_scale = band_data.pix_scale,
            rms_err_ext_name = "RMS_ERR",
        )
        new_band_data.seg_path = cutout_path
        new_band_data.seg_args = band_data.seg_args
        return new_band_data


    def __add__(
        self: Self, 
        other: Union[Band_Cutout, List[Band_Cutout]]
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
                f"All cutouts must have the same filter as {repr(self.filter)=}"
            )


class Stacked_Band_Cutout(Band_Cutout_Base):
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
        overwrite: bool = False
    ) -> Self:
        
        # load sextractor parameters for metadata inclusion
        cat.load_sextractor_auto_mags()
        cat.load_sextractor_auto_fluxes()
        cat.load_sextractor_kron_radii()
        cat.load_sextractor_Re()

        if isinstance(filt, Filter):
            filt = filt.band_name
        # make every individual cutout from the catalogue
        cutouts = [
            Band_Cutout.from_gal_band_data(gal, cat.data[filt], cutout_size, overwrite = overwrite)
            for gal in cat
        ]
        save_path = cls._get_save_path(cat.data[filt], cutout_size, cat.crop_name, cat.filterset.instrument_name, "data")
        return cls.from_cutouts(cutouts, save_path, overwrite = overwrite)

    @classmethod
    def from_data_skycoords(
        cls,
        data: Data,
        filt: Union[str, Filter],
        sky_coords: Union[SkyCoord, List[SkyCoord]],
        cutout_size: u.Quantity,
        save_path: str = None,
        overwrite: bool = False
    ) -> Self:
        # make every individual cutout from the data at the given SkyCoord
        cutouts = [
            Band_Cutout.from_data_skycoord(data, filt, sky_coord, cutout_size, overwrite = overwrite)
            for sky_coord in sky_coords
        ]
        return cls.from_cutouts(cutouts, save_path, overwrite = overwrite)

    @classmethod
    def from_cutouts(
        cls, 
        cutouts: List[Band_Cutout], 
        save_path: str,
        overwrite: bool = False
    ) -> Self:
        # ensure all cutouts are from the same filter
        assert all([cutout.filt_name == cutouts[0].filt_name for cutout in cutouts])
        assert all([cutout.cutout_size == cutouts[0].cutout_size for cutout in cutouts])
        # stack cutouts if they have not been already
        cls._stack_cutouts(cutouts, save_path, overwrite = overwrite)
        band_data = cls._update_band_data([cutout.band_data for cutout in cutouts], save_path)
        # extract original cutout paths
        origin_paths = [cutout.cutout_path for cutout in cutouts]
        return cls(save_path, band_data, cutouts[0].cutout_size, origin_paths)

    @staticmethod
    def _stack_cutouts(
        cutouts: List[Band_Cutout], 
        save_path: str,
        overwrite: bool = False
    ) -> NoReturn:
        if not Path(save_path).is_file() or overwrite:
            # ensure all band data images have the same ZP
            assert all(
                cutout.band_data.ZP == cutouts[0].band_data.ZP
                for cutout in cutouts
            ), galfind_logger.critical(
                "All cutout ZPs must be the same!"
            )
            # ensure all band data images have the same pixel scale
            assert all(
                cutout.band_data.pix_scale == cutouts[0].band_data.pix_scale
                for cutout in cutouts
            ), galfind_logger.critical(
                "All image pixel scales must be the same!"
            )
            # stack band data SCI/ERR/WHT images (inverse variance weighted)
            surveys_versions = np.unique([f"{cutout.band_data.survey}," + \
                cutout.band_data.version for cutout in cutouts])
            galfind_logger.info(
                f"Stacking {len(cutouts)} {cutouts[0].filt_name}" + \
                f" cutouts for {'+'.join(surveys_versions)}!"
            )
            # load all cutouts
            cutout_data_arr = [cutout.load(None) for cutout in cutouts]
            for i, cutout_data in enumerate(cutout_data_arr):
                sci_hdr = cutout_data["SCI"][0]
                sci_data = cutout_data["SCI"][1]
                rms_err_hdr = cutout_data["RMS_ERR"][0]
                rms_err_data = cutout_data["RMS_ERR"][1]
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
            hdu_err = fits.ImageHDU(err, header=fits.Header(rms_err_hdr), name="RMS_ERR")
            hdu_wht = fits.ImageHDU(wht, header=fits.Header(wht_hdr), name="WHT")
            hdul = fits.HDUList([primary, hdu, hdu_err, hdu_wht])
            hdul.writeto(save_path, overwrite=True)
            funcs.change_file_permissions(save_path)
            galfind_logger.info(f"Saved stacked cutout to: {save_path}")

    @staticmethod
    def _update_band_data(
        band_data_arr: List[Band_Data],
        cutout_path: str,
    ) -> Band_Data:
        surveys = "+".join(np.unique([band_data.survey for band_data in band_data_arr]))
        versions = "+".join(np.unique([band_data.version for band_data in band_data_arr]))
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
            rms_err_ext_name = "RMS_ERR",
            )
        new_band_data.seg_path = cutout_path
        new_band_data.seg_args = {key: "+".join(np.unique([band_data.seg_args[key] \
            for band_data in band_data_arr])) for key in band_data_arr[0].seg_args.keys()}
        return new_band_data


class RGB_Base(Cutout_Base, ABC):
    def __init__(
        self: Type[Self], 
        cutouts: Dict[str, List[Type[Band_Cutout_Base]]]
    ) -> Self:
        # ensure cutouts have ['B', 'G', 'R'] keys
        assert list(cutouts.keys()) == ["B", "G", "R"]
        # ensure all cutouts are from different filters
        cutout_band_names = [
            cutout.band_data.filt_name
            for colour in ["B", "G", "R"]
            for cutout in cutouts[colour]
        ]
        assert len(np.unique(cutout_band_names)) == len(cutout_band_names)
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
                if i in [cutout.filt.band_name for cutout in self[col]]
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
        ID_list = [cutout.ID for cutout in np.array([val for val in self.cutouts.values()]).flatten()]
        assert all([ID == ID_list[0] for ID in ID_list])
        return ID_list[0]

    @property
    def meta(self) -> dict:
        meta_list = [cutout.meta for cutout in np.array([val for val in self.cutouts.values()]).flatten()]
        # TODO: ensure the same meta for all cutouts
        # try:
        #     assert all(meta[key] == val for meta in meta_list for key, val in meta_list[0].items())
        # except AssertionError:
        #     breakpoint()
        return meta_list[0]

    @property
    def name(self):
        return ",".join(
            f"{colour}={'+'.join(self.get_colour_band_names(colour))}"
            for colour in ["B", "G", "R"]
        )

    @property
    def filt_names(self) -> List[str]:
        return [
            cutout.band_data.filt_name
            for colour in ["B", "G", "R"]
            for cutout in self[colour]
        ]

    @property
    def filterset(self) -> Dict[Multiple_Filter]:
        return {
            colour: Multiple_Filter(
                [deepcopy(cutout.filt) for cutout in cutouts]
            )
            for colour, cutouts in self.items()
        }

    def get_colour_band_names(self, colour: str) -> List[str]:
        assert colour in ["B", "G", "R"]
        return [cutout.band_data.filt.band_name for cutout in self[colour]]

    def load(
        self: Self,
        filt_name: str,
        hdu_name: str = "SCI",
    ) -> Union[Dict[str, Tuple[Dict[str, Any], np.ndarray]], Tuple[Dict[str, Any], np.ndarray]]:
        assert filt_name in self.filt_names
        return self[filt_name].load(hdu_name)

    def plot(
        self: Self,
        ax: Optional[plt.Axes] = None, 
        method: str = "lupton",
        plot_type: str = "SCI",
        rgb_kwargs: Dict[str, Any] = {},
        plot_regions: List[Dict[str, Any]] = [],
        show: bool = False,
    ) -> NoReturn:
        method = method.lower()  # make method lowercase
        # construct out_path
        # save_path = f"{config['Cutouts']['CUTOUT_DIR']}/{data.version}/{data.survey}/{self.name}/{method}/{self.ID}.png"
        # funcs.make_dirs(save_path)
        if method == "trilogy":
            # Write trilogy.in
            in_path = save_path.replace(".png", "_trilogy.in")
            with open(in_path, "w") as f:
                for colour, cutout_list in self.items():
                    f.write(f"{colour}\n")
                    for cutout in cutout_list:
                        f.write(f"{cutout.cutout_path}[1]\n")
                    f.write("\n")
                f.write("indir  /\n")
                f.write(
                    f"outname  {funcs.split_dir_name(save_path, 'name').replace('.png', '')}\n"
                )
                f.write(f"outdir  {funcs.split_dir_name(save_path, 'dir')}\n")
                f.write("samplesize 20000\n")
                f.write("stampsize  2000\n")
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
            sys.path.insert(
                1, "/nvme/scratch/software/trilogy"
            )  # Not sure why this path doesn't work: config["Other"]["TRILOGY_DIR"]
            from trilogy3 import Trilogy

            galfind_logger.info(f"Making trilogy cutout RGB at {save_path}")
            Trilogy(in_path, images=None).run()

        elif method == "lupton":
            if ax is None:
                fig, ax = plt.subplots()
            data = {colour: [cutout.load(plot_type)[1] \
                for cutout in self[colour]] for colour in ["B", "G", "R"]}
            # red_mad_std = mad_std(data["R"][0])
            # scale = 0.3 / (5. * red_mad_std)
            # offset = 0.2
            r = data["R"][0] #* scale + offset
            g = data["G"][0] #* scale * 1.3 + offset
            b = data["B"][0] #* scale * 1.6 + offset
            #from astropy.visualization import PercentileInterval
            #stretch_percentile = PercentileInterval(99.9)
            #r = stretch_percentile(r)
            #g = stretch_percentile(g)
            #b = stretch_percentile(b)
            rgb_img = make_lupton_rgb(r, g, b, **rgb_kwargs)
            #norm = ImageNormalize(vmin=-scale*red_mad_std, vmax=scale*red_mad_std, stretch=SqrtStretch())
            ax.imshow(rgb_img, origin = "lower")#, norm = norm)
            # turn off grid
            ax.grid(False, which = "both")
            # turn off ticks
            ax.set_xticks([])
            ax.set_yticks([])
            # label RGB filters
            for i, (colour, plt_colour) in enumerate(zip(["B", "G", "R"], ["blue", "green", "red"])):
                filt_name = "+".join(self.get_colour_band_names(colour))
                ax.text(
                    0.15 + i * 0.35,
                    0.1,
                    filt_name,
                    color = plt_colour,
                    fontweight = "bold",
                    fontsize = 8.0,
                    ha = "center",
                    va = "center",
                    path_effects = [
                        pe.withStroke(linewidth = 2.0, foreground = "white")
                    ],
                    transform = ax.transAxes,
                )
            # plot regions
            self._plot_regions(ax, plot_regions)

            if show:
                plt.show()

class RGB(RGB_Base):
    @classmethod
    def from_gal_data(
        cls: Type[Self],
        gal: Galaxy,
        data: Data,
        rgb_bands: Dict[str, Union[str, List[str]]],
        cutout_size: u.Quantity,
        overwrite: bool = False,
    ) -> Self:
        rgb_bands = {key: [val] if isinstance(val, str) else val \
            for key, val in rgb_bands.items() if key in ["B", "G", "R"]}
        # make a cutout for each filter
        cutouts = {
            colour: [Band_Cutout.from_gal_band_data
            (gal, data[band], cutout_size, overwrite=overwrite)]
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
        # make a cutout for each filter
        cutouts = {
            colour: Band_Cutout.from_data_skycoord(data, filt, sky_coord)
            for filt in data.filterset
            if filt.band_name in rgb_bands[colour]
            for colour in ["B", "G", "R"]
        }
        return cls(cutouts)


class Stacked_RGB(RGB_Base):
    @classmethod
    def from_cat(
        cls: Type[Self], 
        cat: Catalogue,
        rgb_bands: Dict[str, Union[str, List[str]]],
        cutout_size: u.Quantity,
        overwrite: bool = False
    ) -> Self:
        """
        Create an instance of the class from a Catalogue object.

        This class method generates a stacked cutout for each filter in the
        provided Catalogue and returns an instance of the class containing
        these stacked cutouts.

        Args:
            cat (Catalogue): The Catalogue object containing the data and
                filterset information.

        Returns:
            Self: An instance of the class with the generated stacked cutouts.
        """
        
        # make a stacked cutout for each filter
        stacked_cutouts = {
            colour: [
                Stacked_Band_Cutout.from_cat
                (cat, band_data.filt, cutout_size, overwrite = overwrite)
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
        """
        Create a new instance of the class from data and sky coordinates.

        This class method generates a stacked cutout for each filter in the provided data
        and returns a new instance of the class containing these stacked cutouts.

        Args:
            data (Data): The data object containing the necessary information for creating cutouts.
            sky_coords (Union[SkyCoord, List[SkyCoord]]): The sky coordinates for which the cutouts are to be made.
                This can be a single SkyCoord object or a list of SkyCoord objects.

        Returns:
            Self: A new instance of the class containing the stacked cutouts for each filter.
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
    def __init__(self, cutouts: List[Type[Cutout_Base]]) -> Self:
        self.cutouts = cutouts

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
        assert all([cutout.cutout_size == self[0].cutout_size for cutout in self])
        return self[0].cutout_size

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
        n_rows: int = 1,
        fig_scaling: float = 1.5,
        split_by_instr: bool = False,
        imshow_kwargs: Dict[str, Any] = {},
        norm_kwargs: Dict[str, Any] = {},
        plot_regions: Dict[str, List[Union[Dict[str, Any], Type[Patch]]]] = {},
        scalebars: Optional[Dict] = [],
        mask: Optional[List[bool]] = None,
        instr_split_cmap: str = "Spectral_r",
        incl_title: bool = False,
        show: bool = False,
        save: bool = True,
        save_path: Optional[str] = None,
        close_fig: bool = False,
    ) -> List[plt.Figure, plt.Axes]:
        
        assert n_rows > 0
        if n_rows > len(self):
            n_y = len(self)
        else:
            n_y = n_rows
        n_x = len(self) // n_y
        if len(self) % n_y != 0:
            n_x += 1

        if fig is not None:
            # Delete everything on the figure
            fig.clf()
        else:
            fig = figs.make_fig(n_x, n_y, scaling = fig_scaling)

        # make appropriate axes from the figure and ax_ratio
        ax_arr = figs.make_ax(fig, n_x, n_y)
        # remove blank axes
        n_blank_ax = n_x * n_y - len(self)
        [fig.delaxes(ax_arr[-(i + 1)]) for i in range(n_blank_ax)]

        if split_by_instr:
            instr_names, n_bands = np.unique([cutout.band_data.instr_name \
                for cutout in self], return_counts = True)
            n_bands = {name: n for name, n in zip(instr_names, n_bands)}
            #instr_names = [name for name in json.loads( \
            #    config["Other"]["INSTRUMENT_NAMES"]) if name in instr_names]
            # determine appropriate colours from the colour map
            instr_split_cmap = plt.get_cmap(instr_split_cmap, len(instr_names))
            norm = Normalize(vmin=0, vmax=len(instr_names) - 1)
            colours = {name: instr_split_cmap(norm(i)) for i, name in enumerate(instr_names)}
            plot_band_counts = {name: 0 for name in instr_names}
            for ax, cutout in zip(ax_arr, self):
                plot_band_counts[cutout.band_data.instr_name] += 1
                colour = colours[cutout.band_data.instr_name]
                ax.patch.set_edgecolor(colour)
                ax.patch.set_linewidth(12.)
                if n_bands[cutout.band_data.instr_name] == \
                        plot_band_counts[cutout.band_data.instr_name]:
                    ax.text(
                        1.05,
                        0.,
                        cutout.band_data.instr_name.replace("_", " "),
                        transform=ax.transAxes,
                        c=colour,
                        path_effects=[pe.withStroke(linewidth=3., foreground="white")],
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
            name: np.unique([getattr(cutout, name) for cutout in self])[0] for name in 
            attrs if len(np.unique([getattr(cutout, name) for cutout in self])) == 1
        }
        
        if incl_title:
            # determine title from shared attributes
            title = ""
        else:
            title = None
        for i, (ax, cutout, scalebars_band) in enumerate(
            zip(ax_arr, self, scalebars)
        ):
            filt_name = cutout.band_data.filt_name
            if filt_name in plot_regions.keys():
                plot_regions_band = plot_regions[filt_name]
            else:
                plot_regions_band = []
            label_kwargs = {
                "label": "\n".join([
                    getattr(cutout, name) for name in attrs
                    if name not in shared_attrs.keys()
                ]),
            }
            
            cutout.plot(
                ax,
                imshow_kwargs = imshow_kwargs,
                norm_kwargs = norm_kwargs,
                plot_regions = plot_regions_band,
                scalebars = scalebars_band,
                label_kwargs = label_kwargs,
                show = False,
                save = False,
            )

        if title is not None:
            fig.suptitle(title)

        if save:
            if save_path is None:
                save_path = self._get_save_path()
            plt.savefig(save_path, bbox_inches = "tight")
            funcs.change_file_permissions(save_path)
            galfind_logger.info(f"Saved cutout plot to: {save_path}")
        if show:
            plt.show()
        if close_fig:
            plt.close(fig)
        return fig, ax_arr


# Galaxy_Cutouts
class Multiple_Band_Cutout(Multiple_Cutout_Base):
    # Each plot is a different Filter
    @classmethod
    def from_cat(
        cls: Type[Self], 
        cat: Catalogue, 
        cutout_size: u.Quantity,
        overwrite: bool = False
    ) -> Self:
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
        overwrite: bool = False
    ) -> Self:
        # make a cutout for each filter
        cutouts = [
            Band_Cutout.from_gal_band_data(
                gal, 
                band_data, 
                cutout_size, 
                overwrite
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
    ) -> Self:
        # make a cutout for each filter
        cutouts = [
            Band_Cutout.from_data_skycoord(data, filt, sky_coord, cutout_size)
            for filt in data.filterset
        ]
        return cls(cutouts)

    @classmethod
    def from_data_skycoords(
        cls: Type[Self],
        data: Data,
        sky_coords: Union[SkyCoord, List[SkyCoord]],
        cutout_size: u.Quantity,
    ) -> Self:
        # make a cutout for each filter
        cutouts = [
            Stacked_Band_Cutout.from_data_skycoord(
                data, filt, sky_coords, cutout_size
            )
            for filt in data.filterset
        ]
        return cls(cutouts)

    @property
    def survey(self) -> str:
        assert all([cutout.survey == self[0].survey for cutout in self])
        return self[0].survey
    
    @property
    def version(self) -> str:
        assert all([cutout.version == self[0].version for cutout in self])
        return self[0].version
    
    @property
    def ID(self) -> str:
        assert all([cutout.ID == self[0].ID for cutout in self])
        return self[0].ID

    @property
    def instr_name(self) -> str:
        assert all([cutout.instr_name == self[0].instr_name for cutout in self])
        return self[0].instr_name

    def _get_save_path(self: Self) -> str:
        if self.instr_name is None:
            instr_name = ""
        else:
            instr_name = f"{self.instr_name}/"
        save_path = f"{config['Cutouts']['CUTOUT_DIR']}/{self.version}/" + \
            f"{self.survey}/{instr_name}{self.cutout_size.to(u.arcsec).value:.2f}as/" + \
            f"multi_band/png/{self.ID}.png"
        # '+'.join(filt.band_name for filt in self.filterset)
        funcs.make_dirs(save_path)
        return save_path

    # part of __getattr__?
    # @property
    # def filterset(self) -> Multiple_Filter:
    #     return Multiple_Filter([deepcopy(cutout.filt) for cutout in self.cutouts])


class Catalogue_Cutouts(Multiple_Cutout_Base):

    def __init__(
        self: Self, 
        cutouts: List[Type[Cutout_Base]],
        ID: str
    ) -> Self:
        # each plot is a different galaxy using the same filter
        self.ID = ID
        super().__init__(cutouts)

    @classmethod
    def from_cat_filt(
        cls: Type[Self],
        cat: Catalogue,
        filt: Union[str, Filter],
        cutout_size: u.Quantity,
        overwrite: bool = False
    ) -> Self:
        if isinstance(filt, Filter):
            filt = filt.band_name
        cutouts = [Band_Cutout.from_gal_band_data
            (gal, cat.data[filt], cutout_size, overwrite) for gal in cat]
        
        return cls(cutouts, cat.crop_name)
    
    @property
    def survey(self) -> str:
        unique_surveys = np.unique([cutout.survey for cutout in self])
        return "+".join(unique_surveys)
        # NOT GENERAL
        #assert all([cutout.survey == self[0].survey for cutout in self])
        #return self[0].survey
    
    @property
    def version(self) -> str:
        unique_versions = np.unique([cutout.version for cutout in self])
        return "+".join(unique_versions)
        # NOT GENERAL
        #assert all([cutout.version == self[0].version for cutout in self])
        #return self[0].version

    @property
    def instr_name(self) -> str:
        # NOT GENERAL
        assert all([cutout.instr_name == self[0].instr_name for cutout in self])
        return self[0].instr_name
    
    def _get_save_path(self: Self) -> str:
        if self.instr_name is None:
            instr_name = ""
        else:
            instr_name = f"{self.instr_name}/"
        save_path = f"{config['Cutouts']['CUTOUT_DIR']}/{self.version}/" + \
            f"{self.survey}/{instr_name}{self.cutout_size.to(u.arcsec).value:.2f}as/" + \
            f"{self[0].band_data.filt_name}/png/{self.ID}.png"
        # '+'.join(filt.band_name for filt in self.filterset)
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
    ) -> plt.Figure:
        n_rows = np.sqrt(2 * len(self))
        n_rows = int(n_rows // 1)
        if n_rows % 1 != 0:
            n_rows += 1
        return super().plot(
            fig = fig,
            n_rows = n_rows,
            fig_scaling = fig_scaling,
            split_by_instr = False,
            imshow_kwargs = imshow_kwargs,
            norm_kwargs = norm_kwargs,
            plot_regions = plot_regions,
            scalebars = scalebars,
            mask = mask,
            show = show,
            save = save,
            save_path = save_path,
        )


class Multiple_RGB(Multiple_Cutout_Base):
    # Each plot is a different Galaxy

    @classmethod
    def from_cat(
        cls: Type[Self], cat: Catalogue, rgb_bands: Dict[str, List[str]]
    ) -> Self:
        # make a cutout for each filter
        cutouts = [RGB.from_gal(cat.data, rgb_bands) for gal in cat]
        return cls(cutouts)

    @classmethod
    def from_data_skycoords(
        cls: Type[Self],
        data: Data,
        sky_coords: Union[SkyCoord, List[SkyCoord]],
        rgb_bands: Dict[str, List[str]],
    ) -> Self:
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
        # make a cutout for each filter
        cutouts = [
            Stacked_RGB.from_data_skycoords(data, sky_coord, rgb_bands)
            for data, sky_coord in zip(data_arr, sky_coords)
        ]
        return cls(cutouts)

    # part of __getattr__?
    # @property
    # def IDs(self) -> List[str]:
    #     return [cutout.ID for cutout in self.cutouts]

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import astropy.units as u
from copy import deepcopy
from astropy.nddata import Cutout2D
from pathlib import Path
import sys
import matplotlib.patches as patches
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import (
    ImageNormalize,
    LinearStretch,
    LogStretch,
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
    from . import Data
    from . import Multiple_Data
    from . import Galaxy
    from . import Catalogue
    from . import Multiple_Catalogue
    from . import Filter
    from . import Multiple_Filter
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

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
    def load(self) -> Tuple[dict, np.ndarray]:
        pass

    @abstractmethod
    def plot(self) -> plt.Axes:
        pass


class Band_Cutout_Base(Cutout_Base, ABC):
    def __init__(
        self, cutout_path: str, filt: Filter, cutout_size: u.Quantity
    ) -> Self:
        self.cutout_path = cutout_path
        self.filt = filt
        self.cutout_size = cutout_size

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

    @property
    def ID(self) -> str:
        if "ID" in self.meta:
            return self.meta["ID"]
        else:
            return f"({self.meta['RA']:.5f},{self.meta['DEC']:.5f})"

    @property
    def meta(self) -> dict:
        return dict(self.load("PRIMARY")[0])

    def load(self, hdu_name: str = "SCI") -> np.ndarray:
        hdu = fits.open(self.cutout_path)[hdu_name]
        return hdu.header, hdu.data

    def plot(
        self,
        ax: plt.Axes,
        high_dyn_range: bool = False,
        cmap: str = "magma",
        plot_radii: List[dict] = [],
        scalebars: Optional[dict] = None,
        SNR: Optional[float] = None,
    ) -> plt.Axes:
        # load cutout
        cutout_hdr, cutout_data = self.load()
        # scale cutout
        cutout_data, norm = self._scale_cutout(
            cutout_data, cutout_hdr["cutout_size_pix"], high_dyn_range, SNR
        )
        # plot cutout
        ax.imshow(cutout_data, norm=norm, cmap=cmap, origin="lower")
        ax.text(
            0.95,
            0.95,
            self.filt.band_name,
            fontsize="small",
            c="white",
            transform=ax.transAxes,
            ha="right",
            va="top",
            zorder=10,
            fontweight="bold",
        )

        # plot radii
        if len(plot_radii) > 0:
            # add circles to show extraction aperture and sextractor FLUX_RADIUS
            xpos = np.mean(ax.get_xlim())
            ypos = np.mean(ax.get_ylim())
            for plot_radius in plot_radii:
                skip_region = False
                if (
                    "region" in plot_radius.keys()
                    and "aper_diam" not in plot_radius.keys()
                ):
                    region = plot_radius["region"]
                elif (
                    "aper_diam" in plot_radius.keys()
                    and "region" not in plot_radius.keys()
                ):
                    pix_scale = (
                        cutout_hdr["cutout_size_as"]
                        * u.arcsec
                        / cutout_hdr["cutout_size_pix"]
                    )
                    radius = (
                        (plot_radius["aper_diam"] / (2.0 * pix_scale))
                        .to(u.dimensionless_unscaled)
                        .value
                    )
                    # make circular region with given radius
                    region = patches.Circle(
                        (xpos, ypos),
                        plot_radius["radius"],
                        **plot_radius["kwargs"],
                    )
                else:
                    skip_region = True
                    galfind_logger.warning(
                        f"{plot_radius.keys()} does not contain one of either 'region' or 'radius', skipping!"
                    )
                if not skip_region:
                    ax.add_patch(region)

        # add scalebars
        if scalebars is not None:
            pix_scale = (
                cutout_hdr["cutout_size_as"]
                * u.arcsec
                / cutout_hdr["cutout_size_pix"]
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
        return ax

    @staticmethod
    def _scale_cutout(
        cutout_data: np.ndarray,
        cutout_size_pix: Union[int, float],
        high_dyn_range: bool,
        SNR: Optional[float] = None,
    ) -> np.ndarray:
        # Set top value based on central 10x10 pixel region
        top = np.max(cutout_data[:20, 10:20])
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
    def from_gal(
        cls: Type[Self],
        data: Data,
        filt: Filter,
        gal: Galaxy,
        cutout_size: u.Quantity,
    ) -> Self:
        # TODO: ensure in some way that the galaxy arises from the data
        # extract the position of the galaxy
        sky_coord = gal.sky_coord
        return cls.from_data_skycoord(
            data, filt, sky_coord, cutout_size, meta={"ID": gal.ID}
        )

    @classmethod
    def from_data_skycoord(
        cls: Type[Self],
        data: Data,
        filt: Filter,
        sky_coord: SkyCoord,
        cutout_size: u.Quantity,
        meta: dict = {},
    ) -> Self:
        # make cutout from data at the sky co-ordinate and save
        meta = {
            **meta,
            "survey": data.survey,
            "version": data.version,
            "RA": sky_coord.ra.value,
            "DEC": sky_coord.dec.value,
            "cutout_size_as": cutout_size.to(u.arcsec).value,
            "cutout_size_pix": (
                cutout_size / data.im_pixel_scales[filt.band_name]
            )
            .to(u.dimensionless_unscaled)
            .value,
        }
        ID = cls._get_ID(meta)
        save_path = f"{config['Cutouts']['CUTOUT_DIR']}/{data.version}/{data.survey}/{cutout_size.to(u.arcsec).value:.2f}as/{filt}/{ID}.fits"
        cls._make_cutout(data, filt, sky_coord, cutout_size, save_path, meta)
        return cls(save_path, filt, cutout_size)

    @staticmethod
    def _make_cutout(
        data: Data,
        filt: Filter,
        sky_coord: SkyCoord,
        cutout_size: u.Quantity,
        save_path: str,
        meta: dict = {},
    ) -> NoReturn:
        # make cutout from data at the sky co-ordinate
        if (
            config.getboolean("Cutouts", "OVERWRITE_CUTOUTS")
            or not Path(save_path).is_file()
        ):
            im_data, im_header, seg_data, seg_header = data.load_data(
                filt, incl_mask=False
            )
            pix_scale = data.im_pixel_scales[filt.band_name]
            wht_data = data.load_wht(filt.band_name)
            rms_err_data = data.load_rms_err(filt.band_name)
            wcs = data.load_wcs(filt.band_name)
            data_dict = {
                "SCI": im_data,
                "SEG": seg_data,
                "WHT": wht_data,
                "RMS_ERR": rms_err_data,
            }
            hdul = [fits.PrimaryHDU(header=fits.Header(meta))]

            cutout_size_pix = (
                (cutout_size / pix_scale).to(u.dimensionless_unscaled).value
            )
            for i, (label_i, data_i) in enumerate(data_dict.items()):
                if i == 0 and label_i == "SCI":
                    sci_shape = data_i.shape
                if type(data_i) == type(None):
                    galfind_logger.warning(
                        f"No data found for {label_i} in {filt.band_name}!"
                    )
                else:
                    if data_i.shape == sci_shape:
                        cutout = Cutout2D(
                            data_i,
                            sky_coord,
                            size=(cutout_size_pix, cutout_size_pix),
                            wcs=wcs,
                        )
                        im_header.update(cutout.wcs.to_header())
                        hdul.append(
                            fits.ImageHDU(
                                cutout.data, header=im_header, name=label_i
                            )
                        )
                        galfind_logger.info(
                            f"Created cutout for {label_i} in {filt.band_name}"
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
            ID = Cutout_Base._get_ID(meta)
            galfind_logger.info(
                f"Already made fits cutout for {data.survey} {data.version} {ID} {filt.band_name}"
            )

    def __add__(
        self, other: Union[Band_Cutout, List[Band_Cutout]]
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
        filt: Filter,
        cutout_size: u.Quantity,
        origin_paths: List[str],
    ) -> Self:
        self.origin_paths = origin_paths
        super().__init__(cutout_path, filt, cutout_size)

    @classmethod
    def from_cat(
        cls,
        cat: Catalogue,
        filt: Filter,
        cutout_size: u.Quantity,
        save_path: str = None,
    ) -> Self:
        # make every individual cutout from the catalogue
        cutouts = [
            Band_Cutout.from_gal(cat.data, filt, gal, cutout_size)
            for gal in cat
        ]
        return cls.from_cutouts(cutouts, save_path)

    @classmethod
    def from_data_skycoords(
        cls,
        data: Data,
        filt: Filter,
        sky_coords: Union[SkyCoord, List[SkyCoord]],
        cutout_size: u.Quantity,
        save_path: str = None,
    ) -> Self:
        # make every individual cutout from the data at the given SkyCoord
        cutouts = [
            Band_Cutout.from_data_skycoord(data, filt, sky_coord, cutout_size)
            for sky_coord in sky_coords
        ]
        return cls.from_cutouts(cutouts, save_path)

    @classmethod
    def from_cutouts(
        cls, cutouts: List[Band_Cutout], save_path: str = None
    ) -> Self:
        # ensure all cutouts are from the same filter
        assert all([cutout.filter == cutouts[0].filter for cutout in cutouts])
        # stack cutouts if they have not been already
        cls._stack_cutouts(cutouts, save_path)
        # extract original cutout paths
        origin_paths = [cutout.cutout_path for cutout in cutouts]
        return cls(save_path, origin_paths, cutouts[0].filter)

    @staticmethod
    def _stack_cutouts(cutouts: List[Band_Cutout], save_path: str) -> NoReturn:
        """
        Stack cutouts to create a stacked cutout
        """
        pass


class RGB_Base(Cutout_Base, ABC):
    def __init__(
        self: Type[Self], cutouts: Dict[str, List[Type[Band_Cutout_Base]]]
    ) -> Self:
        # ensure cutouts have ['B', 'G', 'R'] keys
        assert list(cutouts.keys()) == ["B", "G", "R"]
        # ensure all cutouts are from different filters
        cutout_band_names = [
            cutout.filt.band_name
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
        ID_list = [cutout.ID for cutout in self.cutouts]
        assert all([ID == ID_list[0] for ID in ID_list])
        return ID_list[0]

    @property
    def meta(self) -> dict:
        meta_list = [cutout.meta for cutout in self.cutouts]
        # ensure the same meta for all cutouts
        assert all(meta[key] == val for meta in meta_list for key, val in meta_list[0].items())
        return meta_list[0]

    @property
    def name(self):
        return ",".join(
            f"{colour}={'+'.join(self.get_colour_band_names[colour])}"
            for colour in ["B", "G", "R"]
        )

    @property
    def filt_names(self) -> List[str]:
        return [
            cutout.filt
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
        return [cutout.filt.band_name for cutout in self[colour]]

    def load(self, filt_name: str) -> np.ndarray:
        assert filt_name in self.filt_names
        return self[filt_name].load()

    def plot(
        self, ax: Optional[plt.Axes], method: str = "trilogy"
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
            raise (NotImplementedError())
            if ax is None:
                fig, ax = plt.subplots()


class RGB(RGB_Base):
    @classmethod
    def from_gal(
        cls: Type[Self],
        data: Data,
        gal: Galaxy,
        rgb_bands: Dict[str, List[str]],
    ) -> Self:
        # make a cutout for each filter
        cutouts = {
            colour: Band_Cutout.from_gal(data, filt, gal)
            for filt in data.filterset
            if filt.band_name in rgb_bands[colour]
            for colour in ["B", "G", "R"]
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
        cls: Type[Self], cat: Catalogue, rgb_bands: Dict[str, List[str]]
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
                Stacked_Band_Cutout.from_cat(cat, filt)
                for filt in cat.data.filterset
                if filt in rgb_bands[colour]
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

    def plot(
        self,
        fig: plt.Figure,
        ax_ratio: float,
        high_dyn_range: bool = False,
        cutout_band_cmap: str = "magma",
        plot_radii: List[List[dict]] = {},
        scalebars: Optional[dict] = None,
        mask: Optional[List[bool]] = None,
    ) -> plt.Figure:
        # Delete everything on the figure
        fig.clf()

        # make appropriate axes from the figure and ax_ratio
        n_x = len(self)
        n_y = 1
        ax_arr = figs.make_ax(fig, n_x, n_y)

        if mask is not None:
            assert len(mask) == len(self)
            masked_self = self[mask]
        else:
            masked_self = self

        for i, (ax, cutout, scalebars_band, plot_radii_band) in enumerate(
            zip(ax_arr, masked_self, scalebars, plot_radii)
        ):
            if isinstance(cutout, tuple(Band_Cutout_Base.__subclasses__)):
                cutout.plot(
                    ax,
                    high_dyn_range=high_dyn_range,
                    cmap=cutout_band_cmap,
                    plot_radii=plot_radii_band,
                    scalebars=scalebars_band,
                )
            else:
                cutout.plot(
                    ax,
                    high_dyn_range=high_dyn_range,
                    plot_radii=plot_radii_band,
                    scalebars=scalebars_band,
                )

        return fig


class Multiple_Band_Cutout(Multiple_Cutout_Base):
    # Each plot is a different Filter

    @classmethod
    def from_gal(
        cls: Type[Self], data: Data, gal: Galaxy, cutout_size: u.Quantity
    ) -> Self:
        # make a cutout for each filter
        cutouts = [
            Band_Cutout.from_gal(data, filt, gal, cutout_size)
            for filt in data.filterset
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
    def from_cat(
        cls: Type[Self], cat: Catalogue, cutout_size: u.Quantity
    ) -> Self:
        # make a cutout for each filter
        cutouts = [
            Stacked_Band_Cutout.from_cat(cat, filt, cutout_size)
            for filt in cat.data.filterset
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

    # part of __getattr__?
    # @property
    # def filterset(self) -> Multiple_Filter:
    #     return Multiple_Filter([deepcopy(cutout.filt) for cutout in self.cutouts])


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

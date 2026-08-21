"""Point spread function (PSF) model classes and utilities.

Provides PSF_Base for storing encircled energy curves and PSF metadata, along
with utilities for PSF generation, manipulation, visualization, and star
detection.
"""

from __future__ import annotations

import logging
import os
from abc import ABC
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import convolve_fft
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.modeling.fitting import (
    FittingWithOutlierRemoval,
    LinearLSQFitter,
)
from astropy.modeling.models import Linear1D
from astropy.nddata import CCDData, Cutout2D, block_reduce
from astropy.stats import mad_std, sigma_clip
from astropy.table import Table, hstack
from astropy.time import Time
from astropy.visualization import ImageNormalize, LinearStretch
from numpy.typing import NDArray
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.centroids import centroid_2dg, centroid_com
from photutils.detection import find_peaks
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.ndimage import binary_dilation, zoom
from tqdm import tqdm

if TYPE_CHECKING:
    from . import (
        Band_Cutout,
        Band_Data,
        Catalogue,
        Filter,
        Multiple_Band_Data_Base,
    )
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from .. import config, figs, galfind_logger
from ..utils import useful_funcs_austind as funcs


class PSF_Base(ABC):
    """Abstract base class for Point Spread Function (PSF) models.

    Stores encircled energy curve (EEC) data and PSF metadata. Subclasses
    implement PSF generation and manipulation methods.

    Parameters
    ----------
    eec_path : `str`
        Path to HDF5 file containing the encircled energy curve data.
    name : `str`
        Descriptive name for this PSF model.
    **kwargs
        Additional attributes to set on the instance.

    Attributes
    ----------
    eec_path : `str`
        Path to encircled energy curve file.
    name : `str`
        PSF model name.
    """

    def __init__(
        self: Self,
        eec_path: str,
        name: str,
        **kwargs: Dict[str, Any],
    ):
        assert Path(eec_path).is_file(), galfind_logger.critical(
            f"{eec_path=} not found!"
        )
        self.eec_path = eec_path
        self.name = name

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            try:
                setattr(result, key, deepcopy(value, memo))
            except Exception:
                galfind_logger.critical(
                    f"deepcopy({self.__class__.__name__}) {key}: {value} FAIL!"
                )
                breakpoint()
        return result

    def __repr__(self: Self) -> str:
        return f"{self.__class__.__name__}(eec_path={self.eec_path})"

    def open_eec(
        self: Self,
        radii: Optional[u.Quantity] = None,
    ) -> Tuple[NDArray[float], NDArray[float]]:
        """Load encircled energy curve from file.

        Parameters
        ----------
        radii : `astropy.units.Quantity`, optional
            Radii at which to interpolate the EEC. Default is None (use
            stored values).

        Returns
        -------
        `tuple` of `numpy.ndarray`
            (radii in arcsec, encircled energy fractions)
        """
        # open the encircled energy curve from file and save in self
        with h5py.File(self.eec_path, "r") as f:
            eec_radii = f["radii"][:]
            eec = f["eec"][:]
        if radii is not None:
            # interpolate eec curve to given radii
            if isinstance(radii, u.Quantity):
                radii = radii.to(u.arcsec).value
            eec = interp1d(eec_radii, eec, fill_value="extrapolate")(radii)
            eec_radii = radii
        return eec_radii, eec

    def plot_eec(
        self: Self,
        ax: plt.Axes,
        radii: Optional[u.Quantity] = None,
        annotate: bool = False,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Plot the encircled energy curve.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes to plot on.
        radii : `astropy.units.Quantity`, optional
            Radii to plot. Default is None (use stored values).
        annotate : `bool`, optional
            Whether to add axis labels and legend. Default is False.
        **kwargs
            Additional kwargs passed to ax.plot().
        """
        radii_as, eec = self.open_eec(radii=radii)
        ax.plot(radii_as, eec, **kwargs)
        if annotate:
            ax.legend(loc="lower right")
            ax.set_xlabel("Radius (arcsec)")
            ax.set_ylabel("Encircled Energy")

    def get_aper_corrs(
        self: Self,
        aper_diam: u.Quantity,
        out_type: str = "mag",
    ) -> Union[u.Quantity, u.Magnitude]:
        """Calculate aperture corrections from encircled energy curve.

        Parameters
        ----------
        aper_diam : `astropy.units.Quantity`
            Aperture diameter.
        out_type : `str`, optional
            Output type ("mag" for magnitudes or "flux" for flux ratio).
            Default is "mag".

        Returns
        -------
        `float` or `astropy.units.Quantity`
            Aperture correction value.
        """
        radii_as, eec = self.open_eec()
        aper_radius = (aper_diam / 2).to(u.arcsec).value
        # interpolate eec curve to get encircled energy at aperture radius
        encircled = np.interp(aper_radius, radii_as, eec)
        if out_type == "flux":
            aper_corr = 1 / encircled
        elif out_type == "mag":
            aper_corr = -2.5 * np.log10(encircled)
        else:
            err_message = (
                f"{out_type=} not in ['flux', 'mag'] for aperture "
                "correction output type!"
            )
            galfind_logger.critical(err_message)
            raise ValueError(err_message)
        return aper_corr

    def make_kernel(
        self: Self,
        match_psf: PSF_Base,
        **kwargs: Dict[str, Any],
    ) -> None:
        """No-op kernel construction for encircled-energy-only PSFs.

        `PSF_Base` stores only an encircled energy curve, not pixel
        data, so a pixel-based convolution kernel cannot be built.
        Callers (e.g. `Band_Data.psf_homogenize`) treat a `None`
        return the same as an already-matching PSF: the original
        data is symlinked rather than convolved.

        Parameters
        ----------
        match_psf : `PSF_Base`
            Target PSF to match to (unused).
        **kwargs
            Accepted for interface compatibility with
            `PSF_Cutout.make_kernel`; unused.

        Returns
        -------
        `None`
        """
        galfind_logger.warning(
            f"Cannot build a pixel-based kernel for {repr(self)}: "
            "no cutout data available (only an encircled energy curve). "
            f"Skipping PSF homogenization to {repr(match_psf)}."
        )
        return None


class PSF_Cutout(PSF_Base):
    """PSF derived from a cutout image of a star or point source.

    Generates an encircled energy curve from a star/PSF cutout image,
    storing it for use in photometric operations.

    Parameters
    ----------
    cutout : `Band_Cutout`
        Cutout image of the PSF reference source (typically a star).
    eec_path : `str`
        Path where the generated EEC should be saved.
    name : `str`
        Name for this PSF model.
    is_kernel : `bool`, optional
        Whether this PSF is meant for use as a convolution kernel.
        Default is `False`.
    overwrite : `bool`, optional
        Regenerate EEC even if it already exists. Default is `False`.
    **kwargs
        Additional attributes to set.

    Attributes
    ----------
    cutout : `Band_Cutout`
        The source cutout image.
    is_kernel : `bool`
        Whether used as a kernel.
    """

    def __init__(
        self: Self,
        cutout: Band_Cutout,
        eec_path: str,
        name: str,
        is_kernel: bool = False,
        overwrite: bool = False,
        **kwargs: Dict[str, Any],
    ):
        self.cutout = cutout
        self.is_kernel = is_kernel
        self._make_eec(cutout, eec_path, overwrite=overwrite)
        super().__init__(eec_path, name=name, **kwargs)

    @classmethod
    def from_stpsf(
        cls: Type[Self],
        band_data: Band_Data,
        size: u.Quantity = 4.0 * u.arcsec,
        pa_v3: Optional[float] = None,
        jitter_kernel: Optional[str] = None,
        jitter_sigma: Optional[u.Quantity] = None,
        overwrite: bool = False,
    ) -> Self:
        """Create PSF from STellar Point Spread Function (stpsf) model.

        Generates a PSF model using the stpsf library based on JWST instrument
        and filter properties from band data.

        Parameters
        ----------
        band_data : `Band_Data`
            Band data specifying instrument and filter.
        size : `astropy.units.Quantity`, optional
            PSF model size. Default is 4 arcsec.
        pa_v3 : `float`, optional
            Position angle (V3 reference). Uses image header if None.
            Default is None.
        jitter_kernel : `str`, optional
            Jitter kernel type ("gaussian"). Default is None.
        jitter_sigma : `astropy.units.Quantity`, optional
            Jitter RMS. Default is None.
        overwrite : `bool`, optional
            Regenerate PSF even if it exists. Default is False.

        Returns
        -------
        `PSF_Cutout`
            PSF model generated by stpsf.
        """
        funcs.make_dirs(f"{config['PSF']['STPSF_CACHE']}/dummy")
        os.environ["STPSF_PATH"] = config["PSF"]["STPSF_CACHE"]
        import stpsf

        stpsf_version = stpsf.__version__
        galfind_logger.debug(
            f"Using stpsf version {stpsf_version} for PSF generation."
        )

        band_data_hdr = band_data.load_im()[1]

        out_name = (
            f"stpsf-v{stpsf_version}_pixscl="
            + f"{band_data.pix_scale.to(u.arcsec).value:.2f}as"
            + f"_size={size.to(u.arcsec).value:.1f}as"
        )
        if pa_v3 is not None:
            out_name += f"_pa{pa_v3:.1f}"
        else:
            pa_v3 = band_data_hdr.get("PA_V3", None)
            assert pa_v3 is not None, galfind_logger.critical(
                "PA_V3 not found in image header for PSF normalization!"
            )
            pa_v3 = float(pa_v3)
        if jitter_sigma is not None and jitter_kernel is not None:
            assert jitter_kernel in ["gaussian"], galfind_logger.critical(
                f"{jitter_kernel=} not recognised for PSF_Cutout "
                "from_stpsf method!"
            )
            out_name += f"_jitter{int(jitter_sigma.to(u.mas).value)}mas"

        out_dir = f"{band_data.filt.instrument.get_psf_dir(band_data)}/model"
        out_path = f"{out_dir}/{out_name}.fits"
        funcs.make_dirs(out_path)

        if not Path(out_path).is_file() or overwrite:
            # make PSF
            assert (
                band_data.filt.instrument.facility.__class__.__name__ == "JWST"
            ), galfind_logger.critical(
                "Currently only JWST PSFs can be made with stpsf!"
            )
            if band_data.filt.instrument.__class__.__name__ == "NIRCam":
                psf_model = stpsf.NIRCam()
            else:  # band_data.filt.instrument.__class__.__name__ == "MIRI":
                # TODO: Test with MIRI!
                psf_model = stpsf.MIRI()
            psf_model.options["parity"] = "odd"
            if jitter_sigma is not None:
                psf_model.options["jitter"] = (
                    jitter_kernel  # jitter model name or None
                )
                psf_model.options["jitter_sigma"] = jitter_sigma.to(
                    u.arcsec
                ).value  # in arcsec per axis, default 0.007
            # make an oversampled webbpsf
            # nc.detector = detector
            psf_model.filter = band_data.filt.filt_name
            psf_model.pixelscale = band_data.pix_scale.to(u.arcsec).value
            date = band_data_hdr.get("MJD-AVG", None)
            if date is not None:
                date = float(date)
                iso_string = Time(date, format="mjd", scale="utc").isot
                psf_model.load_wss_opd_by_date(iso_string, plot=False)
            psfraw = psf_model.calc_psf(
                oversample=4,
                fov_arcsec=10.0,
                normalize="exit_pupil",
            )
            psf = psfraw["DET_SAMP"].data
            # newhdu = fits.PrimaryHDU(psfraw[0].data)
            # newhdu.writeto(out_name + "_orig.fits", overwrite=True)
            # rotate and handle interpolation internally; keep k = 1 to
            # avoid -ve pixels
            rotated = ndimage.rotate(
                psf, -pa_v3, reshape=False, order=1, mode="constant", cval=0.0
            )
            clip = int(
                (10.0 * u.arcsec - size).to(u.arcsec).value
                / 2
                / psf_model.pixelscale
            )
            rotated = rotated[clip:-clip, clip:-clip]

            encircled, norm_fov = band_data.filt.instrument.get_psf_norm(
                band_data, size
            )
            # Normalize to correct for missing flux
            # Has to be done encircled! Ensquared were calibated to zero angle
            w, h = np.shape(rotated)
            Y, X = np.ogrid[:h, :w]
            r = norm_fov / 2.0 / psf_model.pixelscale
            center = [w / 2.0, h / 2.0]
            dist_from_center = np.sqrt(
                (X - center[0]) ** 2 + (Y - center[1]) ** 2
            )
            rotated /= np.sum(rotated[dist_from_center < r])
            rotated *= encircled  # to get the missing flux accounted for
            galfind_logger.debug(f"Final stamp normalization: {rotated.sum()}")
            newhdu = fits.PrimaryHDU(rotated)
            funcs.make_dirs(out_path)
            newhdu.writeto(out_path, overwrite=True)
            funcs.change_file_permissions(out_path)
            galfind_logger.info(f"Saved PSF cutout to {out_path}")

        return cls.from_fits(
            out_path,
            band_data.filt,
            pix_scale=band_data.pix_scale,
            size=size,
            # psf_out_dir = "/".join(out_path.split("/")[:-1]),
            is_kernel=False,
            origin="stpsf",
        )

    @classmethod
    def from_empirical_psf(
        cls: Type[Self],
        band_data: Union[Band_Data, Multiple_Band_Data_Base],
        size: u.Quantity = 4.0 * u.arcsec,
        name: Optional[str] = None,  # Whitaker+19,Weaver+24
        cat: Optional[Catalogue] = None,
        # kernel_dir,
        radii: NDArray[float] = np.array(
            [0.5, 1.0, 2.0, 4.0, 7.5]
        ),  # np.array([0.03, 0.10, 0.16, 0.20, 0.32]) / (2.0 * 0.03),
        oversample=3,
        mag_lims: Tuple[float, float] = (18.0, 25.0),
        r_lims: Tuple[float, float] = (1.2, 5.0),
        exclude_ids: Optional[Union[List[int], NDArray[int]]] = None,
        npeaks: int = 1_000,
        snr_lim: Union[int, float] = 1_000,
        clip_sigma: Union[int, float] = 3.5,
        stack_sigma: Union[int, float] = 2.8,
        overwrite: bool = False,
    ) -> Optional[Self]:
        # Gratefully modified from aperpy
        # (https://github.com/astrowhit/aperpy/tree/main), all credit to
        # Weaver et al for the original code.
        """
        Generate PSF (Point Spread Function) for a given set of images.

        Parameters:
        match_band (str): The reference band for matching the PSFs. None
            if no matching is required.
        oversample (int): The oversampling factor for the PSF. The output
            PSF will be sampled at the original pixel scale but this
            factor is used to derive the kernel.
        mag_lims (Tuple[float, float]): The magnitude limits for the PSF
            generation.
        exclude_ids (list, optional): Defaults to None. A list of star
            IDs to be excluded manually.
        npeaks (int, optional): Defaults to 1000. The number of peaks to
            be selected for PSF generation.
        snr_lim (int, optional): Defaults to 1000. The signal-to-noise
            ratio limit for star selection.
        clip_sigma (float, optional): Defaults to 3.5. The sigma value
            for clipping outliers.
        stack_sigma (float, optional): Defaults to 2.8. The sigma value
            for stacking PSFs.
        Returns:
        None
        """
        from .Data import Band_Data_Base, Multiple_Band_Data_Base

        psf_filepath = cls.get_empirical_psf_path(
            band_data, name=name, size=size
        )

        if isinstance(band_data, Multiple_Band_Data_Base):
            assert all(
                band_data_.filt == band_data.filt for band_data_ in band_data
            ), galfind_logger.critical(
                "All bands in Multiple_Band_Data_Base must have the same "
                "filter for PSF generation!"
            )

        if not Path(psf_filepath).is_file() or overwrite:
            if isinstance(band_data, Multiple_Band_Data_Base):
                [
                    cls.from_empirical_psf(
                        band_data_,
                        size=size,
                        name=name,
                        npeaks=npeaks,
                        exclude_ids=exclude_ids,
                        r_lims=r_lims,
                        mag_lims=mag_lims,
                        radii=radii,
                        stack_sigma=stack_sigma,
                        clip_sigma=clip_sigma,
                        overwrite=overwrite,
                    )
                    for band_data_ in band_data
                ]
                psfmodel = stack_psfs(
                    band_data,
                    name=name,
                    size=size,
                    stack_sigma=stack_sigma,
                )
                if psfmodel is None:
                    return None
                funcs.make_dirs(psf_filepath)
                fits.writeto(
                    psf_filepath,
                    np.array(psfmodel),
                    overwrite=True,
                )
                funcs.change_file_permissions(psf_filepath)
            elif isinstance(band_data, funcs.all_subclasses(Band_Data_Base)):
                stars = find_stars(
                    band_data,
                    npeaks=npeaks,
                    clip_sigma=clip_sigma,
                    radii=radii,
                    r_lims=r_lims,
                    mag_lims=mag_lims,
                )
                ids = psf_from_stars(
                    # cat,
                    band_data,
                    stars,
                    size=size,
                    exclude_ids=exclude_ids,
                    stack_sigma=stack_sigma,
                    name=name,
                    overwrite=overwrite,
                )
                if ids is None:
                    return None

                if cat is not None:
                    from . import ID_Selector

                    assert (
                        band_data.filt_name in cat.filterset.filt_names
                    ), galfind_logger.critical(
                        f"{band_data.filt_name=} not found in "
                        f"{repr(cat)} filterset!"
                    )
                    assert (
                        band_data.survey == cat.survey
                    ), galfind_logger.critical(
                        f"{band_data.survey=}!={cat.survey}!"
                    )
                    assert (
                        band_data.version == cat.version
                    ), galfind_logger.critical(
                        f"{band_data.version=}!={cat.version}!"
                    )
                    # sky cross-match for IDs
                    cat_ids = []
                    raise NotImplementedError()
                    star_selector = ID_Selector(
                        cat_ids, name=f"{band_data.filt_name}_star_{name}"
                    )
                    star_selector(cat)
            else:
                err_message = (
                    f"{repr(band_data)=} not in ['Band_Data', "
                    "'Multiple_Band_Data_Base'] for PSF generation!"
                )
                galfind_logger.critical(err_message)
                raise ValueError(err_message)

        return cls.from_fits(
            psf_filepath,
            band_data.filt,
            pix_scale=band_data.pix_scale,
            size=size,
            # psf_out_dir = "/".join(psf_filepath.split("/")[:-1]),
            is_kernel=False,
            origin="empirical",
            overwrite=overwrite,
        )

    @staticmethod
    def get_empirical_psf_path(
        band_data: Union[Band_Data, Multiple_Band_Data_Base],
        name: Optional[str] = None,  # Whitaker+19,Weaver+24
        size: u.Quantity = 4.0 * u.arcsec,
    ) -> str:
        """Get file path for empirical PSF data."""
        if name is None:
            name_ = ""
        else:
            name_ = f"{name}_"
        psf_dir = band_data.filt.instrument.get_psf_dir(band_data)
        pix_scale_str = f"{band_data.pix_scale.to(u.arcsec).value:.2f}"
        size_str = f"{size.to(u.arcsec).value:.1f}"
        return (
            f"{psf_dir}/empirical/{name_}pixscl={pix_scale_str}as"
            f"_size={size_str}as.fits"
        )

    @classmethod
    def from_fits(
        cls,
        fits_path: str,
        filt: Filter,
        origin: str,
        pix_scale: u.Quantity = 0.03 * u.arcsec,
        size: u.Quantity = 4.0 * u.arcsec,
        psf_name: Optional[str] = None,
        is_kernel: bool = False,
        overwrite: bool = False,
    ) -> PSF_Cutout:
        """Create PSF model from FITS file."""
        if not Path(fits_path).is_file():
            err_message = f"File {fits_path} not found!"
            galfind_logger.critical(err_message)
            raise FileNotFoundError(err_message)
        # if psf_out_dir is None:
        #     psf_out_dir = f"{config['PSF']['PSF_WORK_DIR']}/" + \
        #         f"{pix_scale.to(u.arcsec).value}as/{filt.filt_name}"
        psf_out_dir = "/".join(fits_path.split("/")[:-1])
        if psf_name is None:
            psf_name = fits_path.split("/")[-1].replace(".fits", "")
        psf_out_path = f"{psf_out_dir}/{psf_name}_galfind.fits"
        # fits_path.replace(".fits", "_galfind.fits") #
        if not Path(psf_out_path).is_file() or overwrite:
            funcs.make_dirs(psf_out_path)
            # TODO:resample onto appropriate pixel scale - assume already done
            # load in PSF data and make appropriately sized cutout
            image = CCDData.read(fits_path, unit="adu")
            hdul = fits.open(fits_path)
            hdr = hdul[0].header
            assert all(key in hdr.keys() for key in ["NAXIS1", "NAXIS2"])
            dim_x = hdr["NAXIS1"]
            dim_y = hdr["NAXIS2"]
            psf_cutout = Cutout2D(
                image,
                position=(dim_x / 2 + 1, dim_y / 2 + 1),
                size=(size / pix_scale).to(u.dimensionless_unscaled).value,
            )
            # save cutout to file
            hdr = {
                "EXTNAME": "SCI",
                "ID": psf_name,
                "PIXSCALE": pix_scale.to(u.arcsec).value,
                "BAND": filt.filt_name,
                "ORIGIN": origin,
            }
            cutout_fits = fits.PrimaryHDU(
                psf_cutout.data, header=fits.Header(hdr)
            )
            cutout_fits.writeto(psf_out_path, overwrite=True)
            galfind_logger.info(
                f"Saved PSF cutout with {str(size)=} to {psf_out_path}"
            )

        # make Band_Cutout object
        from ..visualization.Cutout import Band_Cutout
        from .Data import Band_Data

        band_data = Band_Data(
            filt,
            survey=psf_name,
            version=f"{pix_scale.to(u.arcsec)}as",
            im_path=psf_out_path,
            im_ext=0,
            pix_scale=pix_scale,
            im_ext_name="SCI",
        )
        cutout = Band_Cutout(psf_out_path, band_data, size)
        eec_path = psf_out_path.replace(".fits", "_EEC.h5")
        funcs.make_dirs(eec_path)
        return cls(
            cutout,
            eec_path,
            name=f"{filt.filt_name}_{origin}",
            size=size,
            is_kernel=is_kernel,
            overwrite=overwrite,
        )

    # @property
    # def name(self) -> str:
    #     if self.is_kernel:
    #         name = "kernel"
    #     else:
    #         name = self.cutout.band_data.filt_name
    #     id_label = self.cutout.load()[0]['ID'] \
    #         .replace("_pixscl=...", "") \
    #         .replace("_size=...", "") \
    #         .replace(".fits", "")
    #     if id_label == "":
    #         return name
    #     else:
    #         return f"{name}_{id_label}"

    @property
    def fwhm(self: Self) -> u.Quantity:
        """Full width at half maximum of the PSF.

        Returns
        -------
        `astropy.units.Quantity`
            FWHM in arcseconds.
        """
        radii_as, profile = self.calc_radial_profile()
        # breakpoint()
        fwhm_radius = np.interp(0.5, profile, radii_as)
        return fwhm_radius

    @property
    def radial_profile_path(self: Self) -> str:
        """Path to the radial profile data file.

        Returns
        -------
        `str`
            Path to radial profile HDF5 file.
        """
        return self.eec_path.replace("_EEC.h5", "_radial_profile.h5")

    def calc_radial_profile(
        self: Self,
        radii: Optional[u.Quantity] = None,
        rmax: Optional[float] = None,
        dr: float = 0.1,
        order: int = 1,
        overwrite: bool = False,
    ) -> Tuple[NDArray[float], NDArray[float]]:
        """Calculate radial profile from PSF cutout.

        Parameters
        ----------
        radii : `astropy.units.Quantity`, optional
            Radii at which to calculate profile. Default is None.
        rmax : `float`, optional
            Maximum radius in pixels. Default is None.
        dr : `float`, optional
            Radial bin width. Default is 0.1.
        order : `int`, optional
            Interpolation order. Default is 1.
        overwrite : `bool`, optional
            Recalculate if file exists. Default is False.

        Returns
        -------
        `tuple`
            (radii in pixels, profile normalized to peak)
        """

        if not Path(self.radial_profile_path).is_file() or overwrite:
            # open fits data
            data = self.cutout.band_data.load_im()[0]
            ny, nx = data.shape
            y0, x0 = centroid_2dg(data)

            if rmax is None:
                rmax = min(ny, nx) / 2.0

            # radial bin edges and centres
            edges = np.arange(0, rmax + dr, dr)
            r_bins = 0.5 * (edges[:-1] + edges[1:])
            n_bins = len(r_bins)

            radial_profile = np.full(n_bins, np.nan)
            # std     = np.full(n_bins, np.nan)
            # npix    = np.zeros(n_bins, dtype=int)

            angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
            cos_a = np.cos(angles)
            sin_a = np.sin(angles)

            # working array to collect all samples in an annulus
            # process bins in a vectorised loop over angles
            for i, r in enumerate(r_bins):
                # fractional pixel positions of the sampling ring
                col_f = x0 + r * cos_a  # x  → column
                row_f = y0 + r * sin_a  # y  → row

                coords = np.array([row_f, col_f])
                from scipy.ndimage import map_coordinates

                # cval=nan not supported for float map_coordinates
                # directly, so use prefilter + mode='constant' with a
                # sentinel then mask
                vals = map_coordinates(
                    data.astype(float),
                    coords,
                    order=order,
                    mode="nearest",
                    prefilter=True,
                )
                # mask truly out-of-bounds positions
                oob = (row_f < 0) | (row_f >= ny) | (col_f < 0) | (col_f >= nx)
                vals[oob] = np.nan

                # vals = sample_image(data, row_f, col_f, order=interp_order)

                # if mask_nan:
                vals = vals[np.isfinite(vals)]

                if vals.size > 0:
                    radial_profile[i] = np.mean(vals)
                    # std[i]     = np.std(vals)
                    # npix[i]    = vals.size
            # normalize profile
            radial_profile /= np.max(radial_profile)
            # y, x = np.indices(data.shape)
            # r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            # r_int = r.astype(int)
            # tbin = np.bincount(r_int.ravel(), data.ravel())
            # #breakpoint()
            # nr = np.bincount(r_int.ravel())
            # radial_profile = tbin / nr
            # radial_profile /= np.max(radial_profile)
            # radii_pix = np.arange(len(radial_profile))
            # radii_as = (
            #     radii_pix *
            #     self.cutout.band_data.pix_scale.to(u.arcsec).value
            # )
            radii_as = (
                r_bins * self.cutout.band_data.pix_scale.to(u.arcsec).value
            )
            # breakpoint()
            # save to h5 file
            with h5py.File(self.radial_profile_path, "w") as f:
                f.create_dataset("radii", data=radii_as, compression="gzip")
                f.create_dataset(
                    "profile", data=radial_profile, compression="gzip"
                )
                galfind_logger.info(
                    f"Saved radial profile for {repr(self)} to "
                    f"{self.radial_profile_path}"
                )
        else:
            galfind_logger.debug(
                f"Radial profile file {self.radial_profile_path} already "
                "exists, skipping radial profile calculation."
            )
            with h5py.File(self.radial_profile_path, "r") as f:
                radii_as = f["radii"][:]
                radial_profile = f["profile"][:]

        # interpolate to give radii
        if radii is not None:
            # convert to interpolation radii to arcsec
            if isinstance(radii, u.Quantity):
                radii = radii.to(u.arcsec).value
            # else:
            #    radii *= self.cutout.band_data.pix_scale.to(u.arcsec).value
            radial_profile = interp1d(
                radii_as, radial_profile, fill_value="extrapolate"
            )(radii)
            radii_as = radii
        return radii_as, radial_profile

    def plot_radial_profile(
        self: Self,
        ax: plt.Axes,
        radii: Optional[u.Quantity] = None,
        log_y: bool = True,
        annotate: bool = False,
        overwrite: bool = False,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Plot the radial profile of the PSF.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`, optional
            Axes to plot on. Created if None.
        radii : `astropy.units.Quantity`, optional
            Radii to plot. Default is None.
        **kwargs
            Additional kwargs for plot styling.
        """
        radii_as, profile = self.calc_radial_profile(
            radii=radii,
            overwrite=overwrite,
        )
        if log_y:
            profile = np.log10(profile)
        ax.plot(radii_as, profile, **kwargs)
        if annotate:
            ax.legend(loc="lower right")
            ax.set_xlabel("Radius (arcsec)")
            ax.set_ylabel("Radial Profile")

    @staticmethod
    def _make_eec(
        cutout: Band_Cutout,
        eec_path: str,
        overwrite: bool = False,
    ) -> None:
        if not Path(eec_path).is_file() or overwrite:
            # make the encircled energy curve from the PSF cutout and
            # save to file
            radii_as = np.linspace(
                cutout.band_data.pix_scale.to(u.arcsec).value, 2.0, 1_000
            )
            radii_pix = (
                (radii_as * u.arcsec / cutout.band_data.pix_scale)
                .to(u.dimensionless_unscaled)
                .value
            )
            psf = fits.open(cutout.cutout_path)[0].data
            center = centroid_2dg(psf)
            eec = np.zeros_like(radii_pix)
            for i, radius in enumerate(radii_pix):
                aperture = CircularAperture(center, r=radius)
                eec[i] = aperture.do_photometry(psf)[0]
            eec /= np.sum(psf)
            # save to file
            with h5py.File(eec_path, "w") as f:
                f.create_dataset("radii", data=radii_as, compression="gzip")
                f.create_dataset("eec", data=eec, compression="gzip")
                galfind_logger.info(
                    f"Saved EEC data for {repr(cutout)} to {eec_path}"
                )
        else:
            galfind_logger.debug(
                f"EEC file {eec_path} already exists, skipping "
                "EEC calculation."
            )

    def make_kernel(
        self: Self,
        match_psf: PSF_Cutout,
        kernel_dir: Optional[str] = None,
        method: str = "pypher",
        pypher_r: float = 1e-4,
        oversample: Union[int, float] = 3,
    ) -> Optional[PSF_Cutout]:
        """Create a PSF convolution kernel.

        Parameters
        ----------
        target_shape : `tuple`, optional
            Target shape for the kernel. Default is None.
        **kwargs
            Additional kernel configuration options.

        Returns
        -------
        `numpy.ndarray`
            PSF convolution kernel.
        """
        # pypher_r (float): The radius parameter for the PyPHER
        # algorithm. - only used if method is 'pypher'.
        if (
            self.cutout.band_data.filt_name
            == match_psf.cutout.band_data.filt_name
        ):
            galfind_logger.info(
                f"{repr(self)} with {self.cutout.band_data.filt_name=}"
                + f"={match_psf.cutout.band_data.filt_name=}"
            )
            return None

        kernel_path = self._get_kernel_path(match_psf, kernel_dir)
        if not Path(kernel_path).is_file():
            galfind_logger.info(
                f"Generating {repr(self)}->{repr(match_psf)} kernel "
                f"with {method=}"
            )
            # open PSF data
            self_psf_hdr, self_psf_data = self.cutout.load()
            match_psf_hdr, match_psf_data = match_psf.cutout.load()
            assert (
                self_psf_data.shape == match_psf_data.shape
            ), galfind_logger.critical(
                f"{self_psf_data.shape}!={match_psf_data.shape}!"
            )
            if oversample > 1:
                galfind_logger.debug(f"Oversampling PSFs by {oversample}x")
                self_psf_data = zoom(self_psf_data, oversample)
                match_psf_data = zoom(match_psf_data, oversample)

            galfind_logger.debug("Normalizing PSFs")
            self_psf_data /= self_psf_data.sum()
            match_psf_data /= match_psf_data.sum()
            if method == "pypher":
                kernel_dir = "/".join(kernel_path.split("/")[:-1])
                self_psf_name = f"{kernel_dir}/psf_self.fits"
                fits.writeto(
                    self_psf_name,
                    self_psf_data,
                    header=fits.Header(self_psf_hdr),
                    overwrite=True,
                )
                match_psf_name = f"{kernel_dir}/psf_match.fits"
                fits.writeto(
                    match_psf_name,
                    match_psf_data,
                    header=fits.Header(match_psf_hdr),
                    overwrite=True,
                )
                pix_scale = self.cutout.band_data.pix_scale.to(u.arcsec).value
                os.system(f"addpixscl {self_psf_name} {pix_scale}")
                os.system(f"addpixscl {match_psf_name} {pix_scale}")
                cmd = (
                    f"pypher {self_psf_name} {match_psf_name} "
                    f"{kernel_path} -r {pypher_r:.3g}"
                )
                os.system(cmd)
                kernel = fits.getdata(kernel_path)
                os.remove(self_psf_name)
                os.remove(match_psf_name)
                os.remove(kernel_path)
                # os.remove(kernel_path.replace(".fits", ".log"))
            else:
                raise NotImplementedError(
                    f"{method=} not implemented for PSF kernel generation!"
                    + " Currently only 'pypher' method is available."
                )
                # # Does a 2D cut around image - kinda like a tapered tophat
                # # Only used if method is not pypher
                # window = SplitCosineBellWindow(alpha=alpha, beta=beta)
                # kernel = create_matching_kernel(
                #     filt_psf, target_psf, window=window
                # )
            if oversample > 1:
                kernel = block_reduce(
                    kernel, block_size=oversample, func=np.sum
                )
            kernel /= kernel.sum()
            kernel = np.float32(kernel)

            galfind_logger.info(f"Writing kernel to {kernel_path}")
            fits.writeto(
                kernel_path,
                kernel,
                overwrite=True,
            )
        else:
            galfind_logger.debug(
                f"{kernel_path=} already exists, skipping kernel generation!"
            )
        return PSF_Cutout.from_fits(
            kernel_path,
            self.cutout.band_data.filt,
            pix_scale=self.cutout.band_data.pix_scale,
            size=self.cutout.cutout_size,
            is_kernel=True,
            # probably no need for this if origin already stores
            origin=f"kernel_to_{match_psf.name}",
        )

    def _get_kernel_path(
        self: Self,
        match_psf: PSF_Cutout,
        kernel_dir: Optional[str] = None,
    ) -> Optional[str]:
        assert (
            self.cutout.band_data.pix_scale
            == match_psf.cutout.band_data.pix_scale
        ), galfind_logger.critical(
            f"{self.cutout.band_data.pix_scale=}!="
            + f"{match_psf.cutout.band_data.pix_scale=}"
        )
        assert (
            self.cutout.cutout_size == match_psf.cutout.cutout_size
        ), galfind_logger.critical(
            f"{self.cutout.cutout_size=}!="
            + f"{match_psf.cutout.cutout_size=}"
        )
        # determine kernel save path
        if kernel_dir is None:
            kernel_dir = "/".join(self.cutout.cutout_path.split("/")[:-1])
        pix_scale = self.cutout.band_data.pix_scale.to(u.arcsec).value
        kernel_name = (
            f"{self.name}-to-{match_psf.name}_pixscl={pix_scale:.2f}as"
            + f"_size={self.cutout.cutout_size.to(u.arcsec).value:.1f}as"
        )
        kernel_path = f"{kernel_dir}/{kernel_name}.fits"
        funcs.make_dirs(kernel_path)
        return kernel_path

    def convolve(
        self: Self,
        kernel: Optional[PSF_Cutout],
    ) -> Optional[PSF_Cutout]:
        """Convolve an image with the PSF kernel.

        Parameters
        ----------
        image : `numpy.ndarray`
            Input image to convolve.
        **kwargs
            Convolution options.

        Returns
        -------
        `numpy.ndarray`
            Convolved image.
        """
        if kernel is None:
            galfind_logger.info(
                f"No kernel provided for convolution with {repr(self)}, "
                "skipping convolution!"
            )
            return None
        assert (
            kernel.is_kernel and not self.is_kernel
        ), galfind_logger.critical(
            f"{repr(self)} must be a PSF cutout "
            + f"and {repr(kernel)} must be a kernel cutout!"
        )
        assert (
            self.cutout.band_data.pix_scale
            == kernel.cutout.band_data.pix_scale
        ), galfind_logger.critical(
            f"{repr(self)} and {repr(kernel)} must have the same pixel scale!"
        )
        assert (
            self.cutout.cutout_size == kernel.cutout.cutout_size
        ), galfind_logger.critical(
            f"{repr(self)} and {repr(kernel)} must have the same "
            "cutout size!"
        )
        galfind_logger.info(
            f"Convolving {repr(kernel)} with kernel {repr(self)}"
        )
        self_hdr, self_data = self.cutout.load()
        kernel_data = kernel.cutout.load()[1]
        convolved_data = convolve_fft(self_data, kernel_data)
        # save convolved PSF to file
        convolved_dir = "/".join(self.cutout.cutout_path.split("/")[:-1])
        pix_scale = self.cutout.band_data.pix_scale.to(u.arcsec).value
        convolved_name = (
            f"{self.name}-to-{kernel.name}_pixscl={pix_scale:.2f}as"
            + f"_size={self.cutout.cutout_size.to(u.arcsec).value:.1f}as"
        )
        convolved_path = f"{convolved_dir}/{convolved_name}.fits"
        funcs.make_dirs(convolved_path)
        fits.writeto(
            convolved_path,
            convolved_data,
            header=fits.Header(self_hdr),
            overwrite=True,
        )
        return PSF_Cutout.from_fits(
            convolved_path,
            self.cutout.band_data.filt,
            pix_scale=self.cutout.band_data.pix_scale,
            size=self.cutout.cutout_size,
            # psf_out_dir = convolved_dir,
            is_kernel=False,
            origin="convolved",
        )


def find_stars(
    band_data: Band_Data,
    block_size: int = 5,
    npeaks: int = 1_000,
    size=20,
    radii: Optional[NDArray[float]] = np.array(
        [0.5, 1.0, 2.0, 4.0, 7.5]
    ),  # np.array([0.03, 0.10, 0.16, 0.20, 0.32]) / (2.0 * 0.03),
    range=[0, 4],
    mag_lims: Tuple[float, float] = (18.0, 25.0),
    threshold_min=-0.5,
    threshold_mode=[-0.2, 0.2],
    shift_lim=2,
    r_lims: Tuple[float, float] = (1.2, 5.0),
    plot: bool = True,
    mask_type: Optional[str] = None,
    clip_sigma: Union[int, float] = 3.5,
):
    """Find and characterize point sources (stars) in a band image.

    Detects peaks in the image, filters by magnitude and morphology,
    and characterizes their properties for PSF extraction.

    Parameters
    ----------
    band_data : `Band_Data`
        Image data to search for stars.
    block_size : `int`, optional
        Block size for noise estimation. Default is 5.
    npeaks : `int`, optional
        Maximum number of peaks to detect. Default is 1,000.
    size : `float`, optional
        Characteristic size of cutouts around stars. Default is 20.
    radii : `numpy.ndarray` or `None`, optional
        Radii (in pixels) for aperture measurements. Default is
        [0.5, 1.0, 2.0, 4.0, 7.5].
    range : `list`, optional
        Magnitude range for filtering stars. Default is [0, 4].
    mag_lims : `tuple` of (`float`, `float`), optional
        Magnitude limits for candidate stars. Default is (18.0, 25.0).
    threshold_min : `float`, optional
        Minimum detection threshold. Default is -0.5.
    threshold_mode : `list`, optional
        Mode detection threshold range. Default is [-0.2, 0.2].
    shift_lim : `int`, optional
        Maximum shift (pixels) allowed in centroid. Default is 2.
    r_lims : `tuple` of (`float`, `float`), optional
        Radius limits for stars. Default is (1.2, 5.0).
    plot : `bool`, optional
        Generate diagnostic plots. Default is `True`.
    mask_type : `str` or `None`, optional
        Type of mask to apply ("MASK", etc.). Default is `None`.
    clip_sigma : `int` or `float`, optional
        Sigma for clipping outliers. Default is 3.5.

    Returns
    -------
    `astropy.table.Table`
        Table of detected stars with positions, magnitudes, and other metrics.
    """
    im_data, hdr = band_data.load_im()
    wcs = band_data.load_wcs()

    if mask_type is None:
        im_data_mask = ~np.isfinite(im_data)
    else:
        mask = band_data.load_mask()[0]["MASK"].astype(bool)
        im_data_mask = ~mask & ~np.isfinite(im_data)

    im_data[im_data_mask] = 0
    imgb = block_reduce(
        im_data,
        block_size,
        func=np.sum,
    )
    sig = mad_std(imgb[imgb > 0], ignore_nan=True) / block_size
    peaks = find_peaks(im_data, threshold=10 * sig, npeaks=npeaks)
    ra, dec = wcs.all_pix2world(peaks["x_peak"], peaks["y_peak"], 0)

    if radii is None:
        start = 0.5
        stop = size / 2
        power = 0.5
        radii = np.linspace(start**power, stop**power, num=30) ** (1 / power)
    radii = np.array(radii)

    peaks["ra"] = ra
    peaks["dec"] = dec
    peaks["x0"] = 0.0
    peaks["y0"] = 0.0
    peaks["minv"] = 0.0
    for ir in np.arange(len(radii)):
        peaks["r" + str(ir)] = 0.0
    for ir in np.arange(len(radii)):
        peaks["p" + str(ir)] = 0.0

    stars = np.zeros_like(peaks, dtype=object)
    for ip, p in tqdm(
        enumerate(peaks),
        total=len(peaks),
        desc=(
            "Measuring curve of growth for candidate stars in "
            f"{repr(band_data)}"
        ),
        disable=galfind_logger.getEffectiveLevel() > logging.INFO,
    ):
        co = Cutout2D(
            im_data, (p["x_peak"], p["y_peak"]), size, mode="partial"
        )
        # measure offset, feed it to measure cog
        # if offset > 3 pixels -> skip
        position = centroid_com(co.data)
        peaks["x0"][ip] = position[0] - size // 2
        peaks["y0"][ip] = position[1] - size // 2
        peaks["minv"][ip] = np.nanmin(co.data)
        _, cog, profile = measure_curve_of_growth(
            co.data,
            radii=radii,
            position=position,
            rnorm=None,
            rbg=None,
        )
        for ir in np.arange(len(radii)):
            peaks["r" + str(ir)][ip] = cog[ir]
        for ir in np.arange(len(radii)):
            peaks["p" + str(ir)][ip] = profile[ir]
        co.radii = np.array(radii)
        co.cog = cog
        co.profile = profile
        stars[ip] = co

    peaks["mag"] = band_data.ZP - 2.5 * np.log10(peaks["r4"])
    r = peaks["r4"] / peaks["r2"]
    shift_lim_root = np.sqrt(shift_lim)

    ok_mag = peaks["mag"] < mag_lims[1]
    ok_min = peaks["minv"] > threshold_min
    ok_phot = (
        np.isfinite(peaks["r" + str(len(radii) - 1)])
        & np.isfinite(peaks["r2"])
        & np.isfinite(peaks["p1"])
    )
    ok_shift = (
        (np.sqrt(peaks["x0"] ** 2 + peaks["y0"] ** 2) < shift_lim)
        & (np.abs(peaks["x0"]) < shift_lim_root)
        & (np.abs(peaks["y0"]) < shift_lim_root)
    )

    # ratio apertures @@@ hardcoded
    r_mask = (r > r_lims[0]) & (r < r_lims[1]) & ok_mag
    h = np.histogram(
        r[r_mask],
        bins=np.arange(0, range[1], threshold_mode[1] / 2.0),
        range=range,
    )
    ih = np.argmax(h[0])
    rmode = h[1][ih]
    assert rmode > 0, galfind_logger.critical(
        f"Mode of r distribution is non-positive: {rmode=}"
    )
    ok_mode = ((r / rmode - 1) > threshold_mode[0]) & (
        (r / rmode - 1) < threshold_mode[1]
    )
    ok = ok_phot & ok_mode & ok_min & ok_shift & ok_mag

    # sigma clip around linear relation
    try:
        fitter = FittingWithOutlierRemoval(
            LinearLSQFitter(), sigma_clip, sigma=clip_sigma, niter=2
        )
        lfit, outlier = fitter(
            Linear1D(),
            x=band_data.ZP - 2.5 * np.log10(peaks["r4"][ok]),
            y=(peaks["r4"] / peaks["r2"])[ok],
        )
        ioutlier = np.where(ok)[0][outlier]
        ok[ioutlier] = False
    except Exception:
        print("linear fit failed")
        ioutlier = 0
        lfit = None

    mags = band_data.ZP - 2.5 * np.log10(peaks["r4"])

    peaks["id"] = 1
    peaks["id"][ok] = np.arange(1, len(peaks[ok]) + 1)

    if plot:
        fig, axs = plt.subplots(2, 3, figsize=(14, 8))

        mags = peaks["mag"]
        mlim_plot = np.nanpercentile(mags, [5, 95]) + np.array([-2, 1])
        # print(mlim_plot)
        axs[0, 0].scatter(mags, r, 10)
        axs[0, 0].scatter(
            mags[~ok_shift], r[~ok_shift], 10, label="bad shift", c="C2"
        )
        axs[0, 0].scatter(mags[ok], r[ok], 10, label="ok", c="C1")
        axs[0, 0].scatter(
            mags[ioutlier], r[ioutlier], 10, label="outlier", c="darkred"
        )
        if lfit:
            axs[0, 0].plot(
                np.arange(14, 30),
                lfit(np.arange(14, 30)),
                "--",
                c="k",
                alpha=0.3,
                label="slope = {:.3f}".format(lfit.slope.value),
            )
        axs[0, 0].legend()
        axs[0, 0].set_ylim(0, 14)
        axs[0, 0].set_xlim(mlim_plot[0], mlim_plot[1])
        axs[0, 0].set_title(" aper(4) / aper(2) vs mag(aper(4))")

        ratio_median = np.nanmedian(r[ok])
        axs[0, 1].scatter(mags, r, 10)
        axs[0, 1].scatter(
            mags[~ok_shift], r[~ok_shift], 10, label="bad shift", c="C2"
        )
        axs[0, 1].scatter(mags[ok], r[ok], 10, label="ok", c="C1")
        axs[0, 1].scatter(
            mags[ioutlier], r[ioutlier], 10, label="outlier", c="darkred"
        )
        if lfit:
            axs[0, 1].plot(
                np.arange(15, 30),
                lfit(np.arange(15, 30)),
                "--",
                c="k",
                alpha=0.3,
                label="slope = {:.3f}".format(lfit.slope.value),
            )
        axs[0, 1].legend()
        axs[0, 1].set_ylim(ratio_median - 1, ratio_median + 1)
        axs[0, 1].set_xlim(mlim_plot[0], mlim_plot[1])
        axs[0, 1].set_title("aper(2) / aper(4) vs mag(aper(4))")

        _ = axs[0, 2].hist(r, bins=range[1] * 20, range=range)
        _ = axs[0, 2].hist(r[ok], bins=range[1] * 20, range=range)
        axs[0, 2].set_title("aper(2) / aper(4)")

        axs[1, 0].scatter(
            band_data.ZP - 2.5 * np.log10(peaks["r3"][ok]),
            (peaks["peak_value"] / peaks["r3"])[ok],
        )
        axs[1, 0].scatter(
            band_data.ZP - 2.5 * np.log10(peaks["r3"])[ioutlier],
            (peaks["peak_value"] / peaks["r3"])[ioutlier],
            c="darkred",
        )
        axs[1, 0].set_ylim(0, 1)
        axs[1, 0].set_title("peak / aper(3) vs maper(3)")

        axs[1, 1].scatter(peaks["x0"][ok], peaks["y0"][ok], c="C1")
        axs[1, 1].scatter(
            peaks["x0"][ioutlier], peaks["y0"][ioutlier], c="darkred"
        )
        axs[1, 1].set_xlim(-2, 2)
        axs[1, 1].set_ylim(-2, 2)
        axs[1, 1].set_title("offset (pix)")

        axs[1, 2].scatter(peaks["x_peak"][ok], peaks["y_peak"][ok], c="C1")
        axs[1, 2].scatter(
            peaks["x_peak"][ioutlier], peaks["y_peak"][ioutlier], c="darkred"
        )
        axs[1, 2].axis("scaled")
        axs[1, 2].set_title("position (pix)")
        plt.tight_layout()

        save_dir = (
            f"{config['PSF']['PSF_PLOT_DIR']}/{band_data.version}/"
            f"{band_data.survey}"
        )
        save_name = (
            f"{band_data.filt_name}_Whitaker+19,Weaver+24_diagnostic.pdf"
        )
        save_path = f"{save_dir}/{save_name}"
        funcs.make_dirs(save_path)
        plt.savefig(save_path, dpi=300)
        funcs.change_file_permissions(save_path)
        plt.close(fig)
    # #breakpoint()
    # raise Exception()
    # peaks[ok].write(
    #     outdir + "/" + os.path.basename(filename).replace(
    #         suffix, "_star_cat.fits"
    #     ),
    #     overwrite=True,
    # )
    ok_final = ok & (peaks["mag"] > mag_lims[0]) & (peaks["mag"] < mag_lims[1])
    peaks = peaks[ok_final]
    stars = stars[ok_final]
    for i, star in enumerate(stars):
        for key in ["ra", "dec", "id", "mag"]:
            setattr(star, key, peaks[key][i])
    return stars


def psf_from_stars(
    band_data: Band_Data,
    stars: List[Cutout2D],
    size: u.Quantity = 4.0 * u.arcsec,
    snr_lim: Union[int, float] = 1_000,  # new variable!
    stack_sigma: float = 2.8,  # new variable!
    exclude_ids: Optional[
        Union[List[int], NDArray[int]]
    ] = None,  # new variable!
    name: Optional[str] = None,
    overwrite: bool = False,
) -> Optional[str]:
    """Create a PSF model by stacking cutouts of multiple stars.

    Combines star cutouts after quality filtering (SNR threshold, sigma
    clipping)
    to produce an empirical PSF, then saves it as a PSF_Cutout object.

    Parameters
    ----------
    band_data : `Band_Data`
        Band data object providing image metadata and configuration.
    stars : `list` of `Cutout2D`
        Cutout images of individual stars to stack.
    size : `astropy.units.Quantity`, optional
        Size of the PSF model field-of-view. Default is 4.0 arcsec.
    snr_lim : `int` or `float`, optional
        Minimum signal-to-noise ratio for including a star. Default is 1,000.
    stack_sigma : `float`, optional
        Sigma level for clipping during stacking. Default is 2.8.
    exclude_ids : `list` or `array` of `int`, optional
        Star IDs to exclude from stacking. Default is `None`.
    name : `str` or `None`, optional
        Name for the PSF model. Default is `None` (auto-generated).
    overwrite : `bool`, optional
        Regenerate PSF even if cached. Default is `False`.

    Returns
    -------
    `str` or `None`
        Path to the saved PSF FITS file, or `None` if no suitable stars found.
    """

    psf_filepath = PSF_Cutout.get_empirical_psf_path(
        band_data, name=name, size=size
    )

    if not Path(psf_filepath).is_file() or overwrite:
        import cv2

        galfind_logger.info(
            f"Making empirical PSF for {repr(band_data)} with {len(stars)=}"
        )
        ra = np.array([star.ra for star in stars])
        dec = np.array([star.dec for star in stars])
        ids = np.array([star.id for star in stars])
        mags = np.array([star.mag for star in stars])

        im_data, hdr = band_data.load_im()
        wcs = band_data.load_wcs()

        xx, yy = wcs.all_world2pix(ra, dec, 0)

        pixsize = int(
            (size / band_data.pix_scale).to(u.dimensionless_unscaled).value
        )
        c0 = pixsize // 2

        cat = Table(
            [ids, xx, yy, ra, dec, mags],
            names=["id", "x", "y", "ra", "dec", "mag"],
        )
        data = np.array(
            [
                Cutout2D(
                    im_data, (xx[i], yy[i]), (pixsize, pixsize), mode="partial"
                ).data
                for i in np.arange(len(ra))
            ]
        )
        data = np.ma.array(data, mask=~np.isfinite(data) | (data == 0))
        data_orig = data.copy()
        ok = np.ones(len(cat))

        ############

        # psf.center()
        window = 21
        # if "x0" in cat.colnames:
        #     return
        cw = window // 2

        pos = np.zeros((len(data), 5))

        for i in np.arange(len(data)):
            p = data[i, :, :]
            st = Cutout2D(
                p, (c0, c0), window, mode="partial", fill_value=0
            ).data
            st[~np.isfinite(st)] = 0
            x0, y0 = centroid_com(st)

            # affine matrix
            M = np.float32([[1, 0, (cw - x0)], [0, 1, (cw - y0)]])
            # output shape
            wxh = p.shape[::-1]
            p = cv2.warpAffine(p, M, wxh, flags=cv2.INTER_CUBIC)

            # now measure shift on recentered cutout
            # first in small window
            st = Cutout2D(
                p, (c0, c0), window, mode="partial", fill_value=0
            ).data
            x1, y1 = centroid_com(st)

            # now in central half of stamp
            # measure moment shift in positive definite in case there
            # are strong ying yang residuals
            x2, y2 = centroid_com(np.maximum(p, 0))

            p = np.ma.array(p, mask=~np.isfinite(p) | (p == 0))
            data[i, :, :] = p

            # difference in shift between central window and half of stamp
            # is measure of contamination from bright off axis sources
            dsh = np.sqrt(
                ((c0 - x2) - (cw - x1)) ** 2 + ((c0 - y2) - (cw - y1)) ** 2
            )
            pos[i] = cw - x0, cw - y0, cw - x1, cw - y1, dsh

        cat = hstack(
            [
                cat,
                Table(np.array(pos), names=["x0", "y0", "x1", "y1", "dshift"]),
            ]
        )

        ########

        # measure
        norm_radius = 8
        # self.nx // 2
        peaks = np.array([st.max() for st in data])
        peaks[~np.isfinite(peaks) | (peaks == 0)] = 0
        caper = CircularAperture((c0, c0), r=norm_radius)
        cmask = Cutout2D(
            caper.to_mask(),
            (norm_radius, norm_radius),
            pixsize,
            mode="partial",
        ).data
        phot = [
            aperture_photometry(st, caper)["aperture_sum"][0] for st in data
        ]

        cmin = [np.nanmin(st * cmask) for st in data]
        cat["frac_mask"] = 0.0
        for i in np.arange(len(data)):
            data[i].mask |= (data[i] * cmask) < 0.0
        cat["peak"] = peaks
        cat["cmin"] = np.array(cmin)
        cat["phot"] = np.array(phot)
        cat["saturated"] = np.int32(~np.isfinite(np.array(phot)))

        flip_ratio = np.zeros(len(data))
        rms_array = np.zeros(len(data))
        for i, st in enumerate(data):
            block_size_ = 3
            rotate_ = True
            if rotate_:
                p180 = np.flip(st, axis=(0, 1))
                dp = st - p180
            else:
                dp = st.copy()
            s = dp.shape[1]
            buf = 6
            dp[
                s // buf : (buf - 1) * s // buf,
                s // buf : (buf - 1) * s // buf,
            ] = np.nan
            block_reduce(dp, block_size=3)
            rms = mad_std(dp, ignore_nan=True) / block_size_ * np.sqrt(st.size)
            if rotate_:
                rms /= np.sqrt(2)
            # snr = st.sum() / rms
            rms_array[i] = rms
            dp = st - np.flip(st, axis=(0, 1))
            flip_ratio[i] = np.abs(st).sum() / np.abs(dp).sum()

        cat["snr"] = 2 * np.array(phot) / np.array(rms_array)
        cat["flip_ratio"] = np.array(flip_ratio)
        cat["phot_frac_mask"] = 1.0

        ########
        # psf.select()
        dshift_lim_ = 3.5
        mask_lim_ = 0.99
        # nsig_=30
        phot_frac_mask_lim_ = 0.85

        ok = (
            (cat["dshift"] < dshift_lim_)
            & (cat["snr"] > snr_lim)
            & (cat["frac_mask"] < mask_lim_)
            & (cat["phot_frac_mask"] > phot_frac_mask_lim_)
        )

        cat["ok"] = np.int32(ok)
        cat["ok_shift"] = cat["dshift"] < dshift_lim_
        cat["ok_snr"] = cat["snr"] > snr_lim
        cat["ok_frac_mask"] = cat["frac_mask"] < mask_lim_
        cat["ok_phot_frac_mask"] = cat["phot_frac_mask"] > phot_frac_mask_lim_
        for c in cat.colnames:
            if "id" not in c:
                cat[c].format = ".3g"

        ##############

        galfind_logger.debug(
            f"First {repr(band_data)} empirical PSF "
            + f"star selection: {np.sum(ok)} remaining"
        )
        # psf.stack(sigma=sigma)  # Moved from between selects

        maxiters_ = 2
        update_ok_ = True
        iok = np.where(ok)[0]

        norm = cat["phot"][iok]
        data_ = data_orig[iok].copy()
        for i in np.arange(len(data_)):
            data_[i] = data_[i] / norm[i]

        stack, lo, hi, clipped = sigma_clip_3d(
            data_, sigma=stack_sigma, axis=0, maxiters=maxiters_
        )
        # self.clipped[~np.isfinite(self.clipped)] = 0

        for i in np.arange(len(data_)):
            shape = clipped[i].mask.shape
            if update_ok_:
                ok[iok[i]] = (
                    ok[iok[i]]
                    and ~clipped[i].mask[shape[0] // 2, shape[1] // 2]
                )
            # print(~self.clipped[i].mask[shape[0]//2, shape[1]//2],
            #       np.shape(self.clipped[i].mask))
            data[iok[i]].mask = clipped[i].mask
            mask = data[iok[i]].mask
            cat["frac_mask"][iok[i]] = np.size(mask[mask]) / np.size(mask)
        cat["phot_frac_mask"] = [
            aperture_photometry(st, caper)["aperture_sum"][0] for st in data
        ] / cat["phot"]

        ########
        # psf.select()
        dshift_lim_ = 3.5
        mask_lim_ = 0.6
        # nsig_=30
        phot_frac_mask_lim_ = 0.85

        ok = (
            (cat["dshift"] < dshift_lim_)
            & (cat["snr"] > snr_lim)
            & (cat["frac_mask"] < mask_lim_)
            & (cat["phot_frac_mask"] > phot_frac_mask_lim_)
        )
        cat["ok"] = np.int32(ok)
        cat["ok_shift"] = cat["dshift"] < dshift_lim_
        cat["ok_snr"] = cat["snr"] > snr_lim
        cat["ok_frac_mask"] = cat["frac_mask"] < mask_lim_
        cat["ok_phot_frac_mask"] = cat["phot_frac_mask"] > phot_frac_mask_lim_

        if exclude_ids is not None:
            for i in exclude_ids:
                ok[i - 1] = False
                galfind_logger.debug(f"Removing star ID={i}")

        ########

        iok = np.where(ok)[0]

        norm = cat["phot"][iok]
        data_ = data_orig[iok].copy()
        for i in np.arange(len(data_)):
            data_[i] = data_[i] / norm[i]

        psfmodel, lo, hi, clipped = sigma_clip_3d(
            data_, sigma=stack_sigma, axis=0, maxiters=maxiters_
        )

        for i in np.arange(len(data_)):
            shape = clipped[i].mask.shape
            # print(~self.clipped[i].mask[shape[0]//2, shape[1]//2],
            #       np.shape(self.clipped[i].mask))
            data[iok[i]].mask = clipped[i].mask
            mask = data[iok[i]].mask
            cat["frac_mask"][iok[i]] = np.size(mask[mask]) / np.size(mask)
        cat["phot_frac_mask"] = [
            aperture_photometry(st, caper)["aperture_sum"][0] for st in data
        ] / cat["phot"]

        # save data for easier PSF stacking across different fields
        psf_data_filepath = psf_filepath.replace(".fits", "_psf_data.fits")
        funcs.make_dirs(psf_data_filepath)
        fits.writeto(
            psf_data_filepath,
            np.array(data_),
            overwrite=True,
        )
        funcs.change_file_permissions(psf_data_filepath)

        ########
        nsig = 30
        stretch = LinearStretch()  # LogStretch()

        fig, axs = figs.make_rectangular_fig(len(stars), 1.0)
        for i, (star, ax, ok_) in enumerate(zip(stars, axs, ok)):
            if ok_:
                colour = "green"
            else:
                colour = "red"
            star_data = star.data.copy()
            sig = mad_std(star_data[(star_data != 0) & np.isfinite(star_data)])
            if sig == 0:
                sig = 1
            norm = ImageNormalize(
                np.float32(star_data),
                vmin=-nsig * sig,
                vmax=nsig * sig,
                stretch=stretch,
            )
            # axi.imshow(arg, norm=norm, origin='lower', cmap='gray',
            #            interpolation='nearest')
            ax.imshow(
                star_data, norm=norm, origin="lower", interpolation="nearest"
            )
            ax.set_axis_off()
            ax.text(
                0.02,
                0.02,
                f"ID={cat['id'][i]}\n"
                + f"mag={cat['mag'][i]:.2f}\n"
                + f"snr={cat['snr'][i]:.1f}\n"
                + f"dshift={cat['dshift'][i]:.2f}\n"
                + f"frac_mask={cat['frac_mask'][i]:.2f}\n"
                + f"phot_frac_mask={cat['phot_frac_mask'][i]:.2f}",
                transform=ax.transAxes,
                verticalalignment="bottom",
                fontsize=10.0,
                color=colour,  # "black",
                fontweight="bold",
                # path_effects=[pe.Stroke(linewidth=2, foreground=colour)],
            )
            # plot red cross hairs at center of cutout
            ax.plot(
                star_data.shape[0] / 2,
                star_data.shape[1] / 2,
                color=colour,
                marker="+",
                ms=10,
                mew=1,
            )
        # hard coded for now!
        fig.suptitle(
            f"dshift_lim<{dshift_lim_}, phot_frac_mask>"
            f"{phot_frac_mask_lim_}, mask_frac<[0.99,0.6], snr>{snr_lim}"
        )

        save_dir = (
            f"{config['PSF']['PSF_PLOT_DIR']}/{band_data.version}/"
            f"{band_data.survey}"
        )
        save_name = (
            f"{band_data.filt_name}_Whitaker+19,Weaver+24_star_stamps.pdf"
        )
        save_path = f"{save_dir}/{save_name}"
        funcs.make_dirs(save_path)
        plt.savefig(save_path, dpi=300)
        funcs.change_file_permissions(save_path)
        plt.close(fig)

        if np.sum(ok) == 0:
            err_message = (
                f"No stars left after selection for {repr(band_data)} "
                "PSF construction! "
            )
            galfind_logger.warning(err_message)
            return None
        else:
            # for aperture corrections!!!
            funcs.make_dirs(psf_filepath)
            fits.writeto(
                psf_filepath,
                np.array(psfmodel),
                overwrite=True,
            )
            funcs.change_file_permissions(psf_filepath)

            psf_cat_path = psf_filepath.replace(".fits", "_psf_cat.fits")
            funcs.make_dirs(psf_cat_path)
            cat[ok].write(psf_cat_path, overwrite=True)
            funcs.change_file_permissions(psf_cat_path)

            galfind_logger.info(
                f"{np.sum(ok)} stars used for {repr(band_data)} PSF"
            )

        # for convolution!!!

        # encircled, norm_fov = (
        #     band_data.filt.instrument.get_psf_norm(band_data, size)[0]
        # )

        # w, h = np.shape(psfmodel)
        # Y, X = np.ogrid[:h, :w]
        # r = (
        #     (size / 2.0 / band_data.pix_scale)
        #     .to(u.dimensionless_unscaled)
        #     .value
        # )
        # center = [w / 2.0, h / 2.0]
        # dist_from_center = np.sqrt(
        #     (X - center[0]) ** 2 + (Y - center[1]) ** 2
        # )
        # psfmodel /= np.sum(psfmodel[dist_from_center < r])
        # psfmodel *= encircled

        # fits.writeto(
        #     "_".join([outname.replace(".fits", ""), "psf_norm.fits"]),
        #     np.array(psfmodel),
        #     overwrite=True,
        # )
    else:
        galfind_logger.debug(
            f"Empirical PSF file {psf_filepath} already exists, "
            "skipping PSF generation!"
        )
    return psf_filepath


def stack_psfs(
    multi_band_data: Multiple_Band_Data_Base,
    name: Optional[str] = None,
    size: u.Quantity = 4.0 * u.arcsec,
    stack_sigma: float = 2.8,
    maxiters: int = 2,
) -> Optional[NDArray[float]]:
    """Stack PSF models from multiple bands into a single combined PSF.

    Combines empirical PSF data from each band, applying sigma clipping to
    reject outliers and produce a robust average PSF.

    Parameters
    ----------
    multi_band_data : `Multiple_Band_Data_Base`
        Multi-band data object containing PSF models for each band.
    name : `str` or `None`, optional
        Name of the PSF model set. Default is `None`.
    size : `astropy.units.Quantity`, optional
        Size of each PSF model. Default is 4.0 arcsec.
    stack_sigma : `float`, optional
        Sigma threshold for clipping outliers. Default is 2.8.
    maxiters : `int`, optional
        Maximum iterations for sigma clipping. Default is 2.

    Returns
    -------
    `numpy.ndarray` or `None`
        The stacked PSF model, or `None` if no PSF data found.
    """
    data = []
    for band_data in multi_band_data:
        psf_filepath = PSF_Cutout.get_empirical_psf_path(
            band_data, name=name, size=size
        )
        if not Path(psf_filepath).is_file():
            galfind_logger.warning(f"{psf_filepath} does not exist!")
            continue
        psf_data_filepath = psf_filepath.replace(".fits", "_psf_data.fits")
        data.append(fits.getdata(psf_data_filepath))
    if len(data) == 0:
        galfind_logger.warning(
            f"No PSF data files found for {repr(multi_band_data)}! "
            "Cannot stack PSFs."
        )
        return None
    else:
        data = np.concatenate(data, axis=0)
        psfmodel = sigma_clip_3d(
            data, sigma=stack_sigma, axis=0, maxiters=maxiters
        )[0]
        return psfmodel


def sigma_clip_3d(data, maxiters=2, axis=0, **kwargs):
    """Perform iterative sigma-clipping on 3D data along a specified axis.

    Clips outliers from a 3D array along the given axis, returning both the
    clipped data and a mask of clipped pixels.

    Parameters
    ----------
    data : `numpy.ndarray`
        3D array to clip.
    maxiters : `int`, optional
        Maximum number of clipping iterations. Default is 2.
    axis : `int`, optional
        Axis along which to perform clipping. Default is 0.
    **kwargs
        Additional keyword arguments passed to `astropy.stats.sigma_clip()`.

    Returns
    -------
    `tuple` of (`numpy.ndarray`, `numpy.ndarray`)
        Tuple containing:
        - clipped data (same shape as input, with outliers set to median)
        - mask of clipped pixels (True where outliers were found)
    """
    from skimage.morphology import disk

    clipped_data = data.copy()
    for i in range(maxiters):
        clipped_data, lo, hi = sigma_clip(
            clipped_data,
            maxiters=0,
            axis=0,
            masked=True,
            grow=False,
            return_bounds=True,
            **kwargs,
        )
        # grow mask
        for i in range(len(clipped_data.mask)):
            clipped_data.mask[i, :, :] = binary_dilation(
                clipped_data.mask[i, :, :], structure=disk(2), iterations=1
            )  # grow(clipped_data.mask[i, :, :], iterations=1)

    return np.mean(clipped_data, axis=axis), lo, hi, clipped_data


def measure_curve_of_growth(
    im_data: NDArray[float],
    position: Optional[SkyCoord] = None,
    radii: Optional[u.Quantity] = None,
    rnorm: Optional[str] = "auto",
    verbose: bool = False,
    # showme=False,
    rbg: Optional[float] = "auto",
):
    """
    Measure a curve of growth from cumulative circular aperture photometry
    on a list of radii centered on the center of mass of a source in a 2D
    image.

    Parameters
    ----------
    im_data : `~numpy.ndarray`
        2D image array.
    position : `~astropy.coordinates.SkyCoord`
        Position of the source.
    radii : `~astropy.units.Quantity` array
        Array of aperture radii.

    Returns
    -------
    `~astropy.table.Table`
        Table of photometry results, with columns 'aperture_radius' and
        'aperture_flux'.
    """

    # Calculate the centroid of the source in the image
    if position is None:
        # x0, y0 = centroid_2dg(im_data)
        position = centroid_com(im_data)

    if rnorm == "auto":
        rnorm = im_data.shape[1] / 2.0
    if rbg == "auto":
        rbg = im_data.shape[1] / 2.0

    apertures = [CircularAperture(position, r=r) for r in radii]

    if rbg:
        bg_mask = apertures[-1].to_mask().to_image(im_data.shape) == 0
        bg = np.nanmedian(im_data[bg_mask])
        if verbose:
            print("background", bg)
    else:
        bg = 0.0

    # Perform aperture photometry for each aperture
    phot_table = aperture_photometry(im_data - bg, apertures)
    # Calculate cumulative aperture fluxes
    cog = np.array(
        [phot_table["aperture_sum_" + str(i)][0] for i in range(len(radii))]
    )

    if rnorm:
        rnorm_indx = np.searchsorted(radii, rnorm)
        cog /= cog[rnorm_indx]

    area = np.pi * radii**2
    area_cog = np.insert(np.diff(area), 0, area[0])
    profile = np.insert(np.diff(cog), 0, cog[0]) / area_cog
    profile /= profile.max()

    # if plot:
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     ax.plot(radii, cog, marker="o")
    #     ax.plot(radii, profile / profile.max())
    #     ax.set_xlabel("radius pix")
    #     ax.set_ylabel("curve of growth")

    # Create output table of aperture radii and cumulative fluxes
    return radii, cog, profile


# def convolve_images(
#     bands,
#     im_paths,
#     wht_paths,
#     err_paths,
#     outdirs,
#     kernel_dir,
#     match_band,
#     use_fft_conv=True,
#     overwrite=False,
# ):
#     # kernel = fits.getdata(f'{kernel_dir}{match_band}_kernel.fits')

#     if use_fft_conv:
#         convolve_func = convolve_fft
#         convolve_kwargs = {"allow_huge": True}
#     else:
#         convolve_func = convolve
#         convolve_kwargs = {}

#     # for filename in sci_paths:
#     for band in bands:
#         im_filename = im_paths[band]
#         wht_filename = wht_paths[band]
#         err_filename = err_paths[band]
#         if type(im_filename) is list:
#             print("WARNING!, Only doing the first image")
#             im_filename = im_filename[0]
#         if type(wht_filename) is list:
#             print("WARNING!, Only doing the first weight image")
#             wht_filename = wht_filename[0]
#         if type(err_filename) is list:
#             print("WARNING!, Only doing the first error image")
#             err_filename = err_filename[0]

#         same_file = im_filename == wht_filename == err_filename
#         outnames = []
#         outdir = outdirs[band]
#         if type(outdir) is list:
#             print("WARNING!, Only doing the first outdir")
#             outdir = outdir[0]

#         if not os.path.exists(outdir):
#             os.makedirs(outdir)

#         if same_file:
#             print(
#                 "WHT, SCI, and ERR are the same file!. "
#                 "Output will be written to the same file."
#             )
#             outname = (
#                 im_filename.replace(".fits", f"_{match_band}-matched.fits")
#                 .replace(os.path.dirname(im_filename), outdir)
#             )
#             outnames.append(outname)
#             outsciname = outwhtname = outerrname = outname
#         else:
#             outsciname = (
#                 im_filename.replace(
#                     ".fits", f"_sci_{match_band}-matched.fits"
#                 )
#                 .replace(os.path.dirname(im_filename), outdir)
#             )
#             outwhtname = (
#                 wht_filename.replace(
#                     ".fits", f"_wht_{match_band}-matched.fits"
#                 )
#                 .replace(os.path.dirname(wht_filename), outdir)
#             )
#             if err_filename != "":
#                 outerrname = err_filename.replace(
#                     ".fits", f"_err_{match_band}-matched.fits"
#                 ).replace(os.path.dirname(err_filename), outdir)
#                 outnames.append(outerrname)

#             outnames.append(outsciname)
#             outnames.append(outwhtname)

#         skip = False
#         for outname in outnames:
#             if os.path.exists(outname) and not overwrite:
#                 print(outsciname, outwhtname)
#                 print("Convolved images exist, I will not overwrite")
#                 skip = True

#         if skip:
#             continue

#         print("  science image: ", im_filename)
#         print("  weight image: ", wht_filename)
#         print("  error image: ", err_filename)
#         hdul = fits.open(im_filename)

#         hdul_wht = fits.open(wht_filename)
#         if err_filename != "":
#             hdul_err = fits.open(err_filename)

#         if band != match_band:
#             print(f"  PSF-matching sci {band} to {match_band}")
#             tstart = time.time()
#             fn_kernel = os.path.join(kernel_dir, f"{band}_kernel.fits")
#             print("  using kernel ", fn_kernel.split("/")[-1])
#             kernel = fits.getdata(fn_kernel, ignore_missing_simple=True)
#             kernel /= np.sum(kernel)

#             if same_file:
#                 wht_ext = "WHT"
#             else:
#                 wht_ext = 0
#             weight = hdul_wht[wht_ext].data

#             out_hdul = fits.HDUList([])

#             if overwrite or not os.path.exists(outsciname):
#                 print("Running science image convolution...")
#                 if same_file:
#                     sci_ext = "SCI"
#                 else:
#                     sci_ext = 0

#                 sci = hdul[sci_ext].data
#                 data = convolve_func(
#                     sci, kernel, **convolve_kwargs
#                 ).astype(np.float32)
#                 data[weight == 0] = 0.0
#                 print("convolved...")

#                 out_hdu = fits.PrimaryHDU(
#                     data, header=hdul[sci_ext].header
#                 )
#                 out_hdu.name = "SCI"
#                 out_hdu.header["HISTORY"] = (
#                     f"Convolved with {match_band} kernel"
#                 )
#                 out_hdul.append(out_hdu)

#                 if not same_file:
#                     out_hdul.writeto(outsciname, overwrite=True)
#                     print("Wrote file to ", outsciname)
#                     out_hdul = fits.HDUList([])

#             else:
#                 print(outsciname)
#                 print(
#                     f"{band.upper()} convolved science image exists, "
#                     "I will not overwrite"
#                 )

#             hdul.close()

#             if overwrite or not os.path.exists(outwhtname):
#                 print("Running weight image convolution...")
#                 err = np.where(weight == 0, 0, 1 / np.sqrt(weight))
#                 err_conv = convolve_func(
#                     err, kernel, **convolve_kwargs
#                 ).astype(np.float32)
#                 data = np.where(err_conv == 0, 0, 1.0 / (err_conv**2))
#                 data[weight == 0] = 0.0

#                 out_hdu_wht = fits.PrimaryHDU(
#                     data, header=hdul_wht[wht_ext].header)
#                 out_hdu_wht.name = "WHT"
#                 out_hdu_wht.header["HISTORY"] = (
#                     f"Convolved with {match_band} kernel"
#                 )

#                 out_hdul.append(out_hdu_wht)
#                 if not same_file:
#                     out_hdul.writeto(outwhtname, overwrite=True)
#                     print("Wrote file to ", outwhtname)
#                     out_hdul = fits.HDUList([])

#             else:
#                 print(outwhtname)
#                 print(
#                     f"{band.upper()} convolved weight image exists, "
#                     "I will not overwrite"
#                 )

#             hdul_wht.close()

#             if (
#                 err_filename != ""
#                 and outerrname != ""
#                 and (overwrite or not os.path.exists(outerrname))
#             ):
#                 print("Running error image convolution...")
#                 extt = 0 if not same_file else "ERR"
#                 data = (
#                     convolve_func(
#                         hdul_err[extt].data, kernel, **convolve_kwargs
#                     ).astype(np.float32)
#                 )
#                 data[weight == 0] = 0.0

#                 out_hdu_err = fits.PrimaryHDU(
#                     data, header=hdul_err[extt].header)
#                 out_hdu_err.name = "ERR"
#                 out_hdu_err.header["HISTORY"] = (
#                     f"Convolved with {match_band} kernel"
#                 )
#                 out_hdul.append(out_hdu_err)
#                 if not same_file:
#                     out_hdul.writeto(outerrname, overwrite=True)
#                     print("Wrote file to ", outerrname)
#                     out_hdul = fits.HDUList([])

#                 hdul_err.close()

#             print(f"Finished in {time.time()-tstart:2.2f}s")

#             if same_file and len(out_hdul) > 1:
#                 out_hdul.writeto(outname, overwrite=True)
#             else:
#                 print("Not writing empty HDU")
#         else:
#             outsciname = (
#                 im_filename.replace(
#                     ".fits", f"_sci_{match_band}-matched.fits"
#                 ).replace(os.path.dirname(im_filename), outdir)
#             )
#             outwhtname = (
#                 wht_filename.replace(
#                     ".fits", f"_wht_{match_band}-matched.fits"
#                 ).replace(os.path.dirname(wht_filename), outdir)
#             )
#             outerrname = (
#                 err_filename.replace(
#                     ".fits", f"_err_{match_band}-matched.fits"
#                 ).replace(os.path.dirname(err_filename), outdir)
#             )
#             outname = (
#                 im_filename.replace(
#                     ".fits", f"_{match_band}-matched.fits"
#                 ).replace(os.path.dirname(im_filename), outdir)
#             )

#             if same_file:
#                 hdul.writeto(outname, overwrite=True)
#                 print(hdul.info())
#             else:
#                 hdul.writeto(outsciname, overwrite=True)
#                 hdul_wht.writeto(outwhtname, overwrite=True)
#                 if err_filename != "":
#                     hdul_err.writeto(outerrname, overwrite=True)

#             print("Written files to ", outname)

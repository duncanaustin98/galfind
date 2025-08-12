
from __future__ import annotations
import numpy as np
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
import astropy.units as u
import photutils
from astropy.io import fits
import json
from copy import deepcopy
from pathlib import Path
from photutils.aperture import (
    SkyCircularAperture,
    aperture_photometry,
)
from photutils.segmentation import (
    detect_threshold,
    detect_sources,
    SourceCatalog
)
from photutils.background import (
    Background2D,
    MedianBackground
)

from typing import TYPE_CHECKING, NoReturn, Optional, Dict, Any
if TYPE_CHECKING:
    from . import Band_Data_Base
try:
    from typing import Type  # python 3.11+
except ImportError:
    from typing_extensions import Type  # python > 3.7 AND python < 3.11

from . import useful_funcs_austind as funcs
from . import config, galfind_logger

def get_code() -> str:
    return f"photutils-v{photutils.__version__}"

def get_segmentation_path(
    self: Type[Band_Data_Base],
    segment_type: str,
) -> str:
    seg_dir = f"{config['Photutils']['PHOTUTILS_DIR']}/{self.instr_name}/{self.version}/{self.survey}/{segment_type}/segmentation"
    seg_path = f"{seg_dir}/{self.survey}_{self.filt_name}_{self.filt_name}_{self.version}_seg.fits"
    funcs.make_dirs(seg_path)
    return seg_path

def get_forced_phot_path(
    self: Type[Band_Data_Base],
    segment_type: str,
    forced_phot_band: Optional[Type[Band_Data_Base]] = None,
) -> str:
    if forced_phot_band is None:
        select_band_name = self.filt_name
        forced_phot_code = f"photutils-v{photutils.__version__}"
    else:
        select_band_name = forced_phot_band.filt_name
        forced_phot_code = forced_phot_band.forced_phot_args["method"]
    forced_phot_dir = f"{config['Photutils']['PHOTUTILS_DIR']}/{self.instr_name}/" + \
        f"{self.version}/{self.survey}/{segment_type}/forced_phot/" + \
        funcs.aper_diams_to_str(self.aper_diams)
    forced_phot_path = f"{forced_phot_dir}/{self.survey}_{self.filt_name}" + \
        f"_{select_band_name}_{forced_phot_code}_{self.version}.fits"
    funcs.make_dirs(forced_phot_path)
    return forced_phot_path

def segment(
    self: Type[Band_Data_Base],
    segment_type: str = "bkg",
    overwrite: bool = False,
    segment_kwargs: Dict[str, Any] = {"npixels": 30},
    bkg_kwargs: Dict[str, Any] = {"thresh_mult": 10, "box_size": (50, 50), "filter_size": (3, 3)},
    detect_thresh_kwargs: Dict[str, Any] = {"nsigma": 3.0},
) -> str:
    assert segment_type in ["bkg", "rms"], \
        galfind_logger.critical(
            f"{segment_type=} not in ['bkg', 'rms']"
        )
    
    from stwcs.wcsutil import HSTWCS

    seg_path = get_segmentation_path(self, segment_type = segment_type)
    source_cat_path = seg_path.replace("_seg.fits", "_cat.fits")
    if not (Path(seg_path).is_file() and Path(source_cat_path).is_file()) or overwrite:
        galfind_logger.info(f"Segmenting {repr(self)} with photutils using {segment_type=}!")
        data, hdr, hdul = self.load_im(return_hdul = True, mode = "update")
        if segment_type == "bkg":
            req_bkg_kwargs = ["box_size", "thresh_mult"]
            assert all(name in bkg_kwargs.keys() for name in req_bkg_kwargs), \
                galfind_logger.critical(
                    f"One of {req_bkg_kwargs=} not in {bkg_kwargs.keys()=}"
                )
            bkg_kwargs_ = deepcopy(bkg_kwargs)
            box_size = bkg_kwargs_.pop("box_size")
            thresh_mult = bkg_kwargs_.pop("thresh_mult")
            bkg = Background2D(data, box_size, bkg_estimator = MedianBackground(), **bkg_kwargs_)
            # save the background map
            bkg_path = seg_path.replace("_seg.fits", "_bkg.fits")
            primary_hdu = fits.PrimaryHDU(header = hdr)  # no data
            bkg_hdu = fits.ImageHDU(bkg.background, header = hdr, name = "BACKGROUND")
            bkg_rms_hdu = fits.ImageHDU(bkg.background_rms, header = hdr, name = "BACKGROUND_RMS")
            bkg_hdul = fits.HDUList([primary_hdu, bkg_hdu, bkg_rms_hdu])
            bkg_hdul.writeto(bkg_path, overwrite = True)
            funcs.change_file_permissions(bkg_path)
            galfind_logger.info(f"Saved {repr(self)} background map to {bkg_path}")
            threshold = bkg.background_rms * thresh_mult
            thresh_kwargs = bkg_kwargs
        else:
            threshold = detect_threshold(data, error = self.load_rms_err()[0], **detect_thresh_kwargs)
            thresh_kwargs = detect_thresh_kwargs
        segment_map = detect_sources(data, threshold, **segment_kwargs)
        # save the segmentation map
        fits.writeto(seg_path, segment_map.data, header = hdr, overwrite = True)
        funcs.change_file_permissions(seg_path)
        galfind_logger.info(f"Saved {repr(self)} segmentation map to {seg_path}")

        # write source catalogue
        input_cat = SourceCatalog(data, segment_map, wcs = HSTWCS(hdul, self.im_ext))
        input_cat = input_cat.to_table()
        input_cat['RA']  = input_cat['sky_centroid'].ra.degree
        input_cat['DEC'] = input_cat['sky_centroid'].dec.degree
        input_cat.rename_column('xcentroid', 'x')
        input_cat.rename_column('ycentroid', 'y')
        # save source catalogue
        input_cat.write(source_cat_path, format = "fits", overwrite = True)
        funcs.change_file_permissions(source_cat_path)
        galfind_logger.info(f"Saved {repr(self)} source catalogue to {source_cat_path}. Found {len(input_cat)} sources.")

        # save segmentation kwargs
        out_seg_kwargs = {
            "segment_type": segment_type,
            "segment_kwargs": segment_kwargs,
            "thresh_kwargs": thresh_kwargs,
            "method": get_code(),
        }
        seg_kwargs_path = seg_path.replace("_seg.fits", "_seg_kwargs.json")
        with open(seg_kwargs_path, "w") as f:
            json.dump(out_seg_kwargs, f, indent = 2)
    return source_cat_path


def perform_forced_phot(
    self: Type[Band_Data_Base],
    forced_phot_band: Type[Band_Data_Base],
    segment_type: str = "bkg",
    overwrite: bool = False,
) -> NoReturn:

    forced_phot_path = get_forced_phot_path(
        self, segment_type, forced_phot_band
    )
    if not Path(forced_phot_path).is_file() or overwrite:

        assert self.survey == forced_phot_band.survey
        assert self.version == forced_phot_band.version

        # Load/calculate forced photometry positions
        if hasattr(forced_phot_band, "forced_phot_path"):
            cat = Table.read(forced_phot_band.forced_phot_path)
            ra = cat[forced_phot_band.forced_phot_args["ra_label"]]
            dec = cat[forced_phot_band.forced_phot_args["dec_label"]]
        else:
            # calculate aperture photometry positions
            seg_cat_path = segment(segment_type)
            cat = Table.read(seg_cat_path)
            ra = cat["RA"]
            dec = cat["DEC"]
        # make array of aperture locations/sizes
        sky_coords = SkyCoord(ra, dec, unit = u.deg)
        apertures = []
        for aper_diam in self.aper_diams.to(u.arcsec).value:
            apertures_ = SkyCircularAperture(sky_coords, r = aper_diam * u.arcsec / 2.)
            apertures.append(apertures_)

        # Do aperture photometry
        phot_table = aperture_photometry(self.load_im()[0], apertures, wcs=self.load_wcs())
        assert len(phot_table) == len(sky_coords)
        phot_table.rename_column("id", "NUMBER")

        # Add sky coordinates to catalogue
        sky = SkyCoord(phot_table["sky_center"])
        phot_table["ALPHA_J2000"] = sky.ra.to("deg")
        phot_table["DELTA_J2000"] = sky.dec.to("deg")
        phot_table.remove_column("sky_center")

        colnames = [f"aperture_sum_{i}" for i in range(len(self.aper_diams))]
        aper_tab = Column(
            np.array(phot_table[colnames].as_array().tolist()),
            name=f"FLUX_APER_{self.filt_name}",
        )
        phot_table["FLUX_APER"] = aper_tab
        phot_table["FLUXERR_APER"] = phot_table["FLUX_APER"] * -99.0
        phot_table["MAGERR_APER"] = phot_table["FLUX_APER"] * 99.0

        # This converts the fluxes to magnitudes using the correct ZP
        # and puts them in the same format as the sextractor catalogue
        mag_colnames = []
        for pos, col in enumerate(colnames):
            name = f"MAG_APER_{pos}"
            phot_table[name] = (
                -2.5 * np.log10(phot_table[col]) + self.ZP
            )
            phot_table[name][np.isnan(phot_table[name])] = 99.0
            mag_colnames.append(name)
        aper_tab = Column(
            np.array(phot_table[mag_colnames].as_array().tolist()),
            name=f"MAG_APER_{self.filt_name}",
        )
        phot_table["MAG_APER"] = aper_tab
        # Remove old columns
        phot_table.remove_columns(colnames)
        phot_table.remove_columns(mag_colnames)

        phot_table.write(forced_phot_path, format="fits", overwrite=True)
        funcs.change_file_permissions(forced_phot_path)
    
    forced_phot_args = \
        {
            "forced_phot_band": forced_phot_band,
            "segment_type": segment_type,
            "method": get_code(),
            "id_label": "NUMBER",
            "ra_label": "ALPHA_J2000",
            "dec_label": "DELTA_J2000",
        }
    return forced_phot_path, forced_phot_args
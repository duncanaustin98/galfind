# Purpose: Class to hold PSF data

from __future__ import annotations

import astropy.units as u
from pathlib import Path
from astropy.io import fits
import os
from abc import ABC, abstractmethod
from astropy.nddata import CCDData, Cutout2D
from typing import Optional, Any, Union, Dict, NoReturn, TYPE_CHECKING
if TYPE_CHECKING:
    from . import Filter, Band_Cutout
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import galfind_logger, config
from . import useful_funcs_austind as funcs


class PSF_Base(ABC):
    def __init__(
        self: Self,
        properties: Dict[str, Any] = {}, 
        aper_corrs: Dict[u.Quantity, Any] = {},
    ):
        self.properties = properties
        self.aper_corrs = aper_corrs

class PSF_Cutout(PSF_Base):
    def __init__(
        self,
        cutout: Band_Cutout,
        properties: Dict[str, Any] = {},
        aper_corrs: Dict[u.Quantity, Any] = {},
    ):
        self.cutout = cutout
        super().__init__(properties, aper_corrs)

    @classmethod
    def from_fits(
        cls,
        fits_path: str,
        filt: Filter,
        unit: str = "adu",
        pix_scale: u.Quantity = 0.03 * u.arcsec,
        size: u.Quantity = 0.96 * u.arcsec,
    ) -> PSF_Cutout:
        if not Path(fits_path).is_file():
            err_message = f"File {fits_path} not found."
            galfind_logger.critical(err_message)
            raise FileNotFoundError(err_message)
        
        psf_name = fits_path.split("/")[-1].replace(".fits", "")
        psf_out_path = psf_out_path = f"{config['PSF']['PSF_WORK_DIR']}/" + \
            f"{pix_scale.to(u.arcsec).value}as/{filt.band_name}/{psf_name}.fits"
        if not Path(psf_out_path).is_file():
            funcs.make_dirs(psf_out_path)
            # TODO:resample onto appropriate pixel scale - assume already done
            # load in PSF data and make appropriately sized cutout
            image = CCDData.read(fits_path, unit=unit)
            hdul = fits.open(fits_path)
            hdr = hdul[0].header
            assert all(key in hdr.keys() for key in ["NAXIS1", "NAXIS2"])
            dim_x = hdr['NAXIS1']
            dim_y = hdr['NAXIS2']
            psf_cutout = Cutout2D(
                image,
                position = (dim_x / 2 + 1, dim_y / 2 + 1),
                size = (size / pix_scale).to(u.dimensionless_unscaled).value
            )
            # save cutout to file
            hdr = {
                "EXTNAME": psf_name,
                "ID": psf_name,
                "PIXSCALE": pix_scale.to(u.arcsec).value,
                "BAND": filt.band_name
            }
            cutout_fits = fits.PrimaryHDU(psf_cutout.data, header = fits.Header(hdr))
            cutout_fits.writeto(psf_out_path, overwrite = True)
            galfind_logger.info(f"Saved PSF cutout with {size=} to {psf_out_path}")

        # make Band_Cutout object
        from . import Band_Data, Band_Cutout
        band_data = Band_Data(
            filt,
            survey = psf_name,
            version = f"{pix_scale.to(u.arcsec)}as",
            im_path = psf_out_path,
            im_ext = 0,
            pix_scale = pix_scale,
            im_ext_name = psf_name
        )
        cutout = Band_Cutout(psf_out_path, band_data, size)
        return cls(cutout, properties = {"size": size})

    #@abstractmethod
    def _calc_aper_corr(
        self, aper_diam: u.Quantity, out_type: str = "flux"
    ) -> Union[u.Quantity, u.Magnitude]:
        # calculate the aperture correction from the PSF or save previously calculated values
        # save in self
        pass

    #@abstractmethod
    def _calc_aper_corrs(
        self, aper_diams: u.Quantity, out_type: str = "flux"
    ) -> Union[u.Quantity, u.Magnitude]:
        pass
        # aper_corrs = [
        #     self.calc_aper_corr(aper_diam, out_type)
        #     for aper_diam in aper_diams
        # ]
        # assert all(
        #     [aper_corr.unit == aper_corrs[0].unit for aper_corr in aper_corrs]
        # ), galfind_logger.critical(
        #     "All aperture corrections must have the same units."
        # )
        # return [
        #     aper_corr.values.value for aper_corr in aper_corrs
        # ] * aper_corrs[0].unit

    def save_cutout(self) -> NoReturn:
        # save the PSF cutout to a file
        pass

# Could potentially have different types of PSF classes, e.g. GaussianPSF, MoffatPSF, Model_PSF, etc.



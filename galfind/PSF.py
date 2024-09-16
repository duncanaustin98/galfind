# Purpose: Class to hold PSF data

from __future__ import annotations

from typing import Union, NoReturn, TYPE_CHECKING

if TYPE_CHECKING:
    from . import Filter
import astropy.units as u

from . import galfind_logger


class PSF:
    def __init__(self, band: Filter, properties: dict = {}):
        self.band = band
        self.properties = properties
        self.aper_corrs = {}

    def calc_aper_corr(
        self, aper_diam: u.Quantity, out_type: str = "flux"
    ) -> Union[u.Quantity, u.Magnitude]:
        # calculate the aperture correction from the PSF or save previously calculated values
        # save in self
        pass

    def calc_aper_corrs(
        self, aper_diams: u.Quantity, out_type: str = "flux"
    ) -> Union[u.Quantity, u.Magnitude]:
        aper_corrs = [
            self.calc_aper_corr(aper_diam, out_type)
            for aper_diam in aper_diams
        ]
        assert all(
            [aper_corr.unit == aper_corrs[0].unit for aper_corr in aper_corrs]
        ), galfind_logger.critical(
            "All aperture corrections must have the same units."
        )
        return [
            aper_corr.values.value for aper_corr in aper_corrs
        ] * aper_corrs[0].unit

    def save_cutout(self) -> NoReturn:
        # save the PSF cutout to a file
        pass


# Could potentially have different types of PSF classes, e.g. GaussianPSF, MoffatPSF, Model_PSF, etc.

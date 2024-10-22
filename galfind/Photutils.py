import numpy as np
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
import astropy.units as u
import photutils
from pathlib import Path
from photutils.aperture import (
    SkyCircularAperture,
    aperture_photometry,
)
from typing import TYPE_CHECKING, NoReturn, Optional
if TYPE_CHECKING:
    from . import Band_Data_Base
try:
    from typing import Type  # python 3.11+
except ImportError:
    from typing_extensions import Type  # python > 3.7 AND python < 3.11

from . import useful_funcs_austind as funcs
from . import config

def get_code() -> str:
    return f"photutils-v{photutils.__version__}"

def get_segmentation_path(
    self: Type[Band_Data_Base],
    err_map_type: str,
) -> str:
    seg_dir = f"{config['Photutils']['SEG_DIR']}/{self.instr_name}/" + \
        f"{self.version}/{self.survey}/{err_map_type}/segmentation"
    seg_path = f"{seg_dir}/{self.survey}_{self.filt_name}_{self.filt_name}_{self.version}_seg.fits"
    funcs.make_dirs(seg_path)
    return seg_path

def get_forced_phot_path(
    self: Type[Band_Data_Base],
    err_map_type: str,
    forced_phot_band: Optional[Type[Band_Data_Base]] = None,
) -> str:
    if forced_phot_band is None:
        select_band_name = self.filt_name
        forced_phot_code = f"photutils-v{photutils.__version__}"
    else:
        select_band_name = forced_phot_band.filt_name
        forced_phot_code = forced_phot_band.forced_phot_args["method"]
    forced_phot_dir = f"{config['Photutils']['FORCED_PHOT_DIR']}/{self.instr_name}/" + \
        f"{self.version}/{self.survey}/{err_map_type}/forced_phot/" + \
        funcs.aper_diams_to_str(self.aper_diams)
    forced_phot_path = f"{forced_phot_dir}/{self.survey}_{self.filt_name}" + \
        f"_{select_band_name}_{forced_phot_code}_{self.version}.fits"
    funcs.make_dirs(forced_phot_path)
    return forced_phot_path


def perform_forced_phot(
    self: Type[Band_Data_Base],
    forced_phot_band: Type[Band_Data_Base],
    err_type: str = "rms_err",
    overwrite: bool = False,
) -> NoReturn:

    forced_phot_path = get_forced_phot_path(
        self, err_type, forced_phot_band
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
            raise(NotImplementedError)
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
        phot_table["FLUXERR_APER"] = phot_table["FLUX_APER"] * -99
        phot_table["MAGERR_APER"] = phot_table["FLUX_APER"] * 99

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
        "err_type": err_type,
        "method": get_code(),
        "id_label": "NUMBER",
        "ra_label": "ALPHA_J2000",
        "dec_label": "DELTA_J2000",
    }
    return forced_phot_path, forced_phot_args
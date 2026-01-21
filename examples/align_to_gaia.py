from typing import Dict, List, NoReturn, Union
import astropy.units as u

from galfind import Data
from galfind.Data import morgan_version_to_dir

def main(
    survey: str,
    version: str,
    version_to_dir_dict: Dict[str, str] = morgan_version_to_dir,
    alignment_band: str = "F444W",
    pix_scales: Union[u.Quantity, Dict[str, u.Quantity]] = {
        "ACS_SBC": 0.025 * u.arcsec,
        "ACS_WFC": 0.03 * u.arcsec,
        "WFC3_IR": 0.03 * u.arcsec,
        "NIRCam": 0.03 * u.arcsec,
        "MIRI": 0.09 * u.arcsec,
    },
) -> NoReturn:
    # load data object
    data = Data.from_survey_version(
        survey,
        version,
        instrument_names = ["ACS_SBC"],
        #version_to_dir_dict = version_to_dir_dict,
        pix_scales = pix_scales,
    )
    data.sky_align(alignment_band, xoffset = -47.7, yoffset = 16.65)
    data.xy_align(alignment_band)
    breakpoint()

if __name__ == "__main__":
    survey = "nltt3330"
    version = "v1"
    alignment_band = "F115LP"
    pix_scales = {"ACS_SBC": 0.025 * u.arcsec}
    main(
        survey,
        version,
        alignment_band = alignment_band,
        pix_scales = pix_scales
    )
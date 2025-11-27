
import astropy.units as u
import os
from typing import List, Union, Type
from galfind import Data, Catalogue, Band_Data_Base, galfind_logger
from galfind.Data import morgan_version_to_dir

from test_config import test_galfind_data_dir, test_survey, test_version, test_bands

def main(
    survey: str,
    version: str,
    instrument_names: List[str],
    aper_diams: u.Quantity = [0.32] * u.arcsec,
    forced_phot_band: Union[str, List[str], Type[Band_Data_Base]] = ["F277W", "F356W", "F444W"],
    cutout_size: u.Quantity = 15.0 * u.arcsec,
):
    # load in data
    data = Data.pipeline(
        survey,
        version,
        instrument_names,
        version_to_dir_dict = morgan_version_to_dir,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
    )
    # load in catalogue
    cat = Catalogue.from_data(data)
    # choose an appropriate galaxy in the middle of the catalogue
    random_gal = cat[len(cat) // 2]
    cutouts = random_gal.make_cutouts(data, cutout_size)
    # move cutouts to new data directory
    for cutout in cutouts:
        if cutout.band_data.filt.band_name in test_bands:
            output_dir = Data._get_data_dir(
                test_survey,
                test_version,
                pix_scale = 0.03 * u.arcsec,
                instrument = cutout.band_data.filt.instrument,
                data_dir = test_galfind_data_dir,
            )
            output_path = f"{output_dir}/{cutout.band_data.filt.band_name}_{test_survey}.fits"
            # copy cutout to new location
            os.system(f"cp {cutout.cutout_path} {output_path}")
            galfind_logger.info(f"Copied {cutout.cutout_path} to {output_path}")

if __name__ == "__main__":
    survey = "JADES-DR3-GS-East"
    version = "v13"
    instrument_names = ["ACS_WFC", "NIRCam"]
    main(
        survey,
        version,
        instrument_names,
    )

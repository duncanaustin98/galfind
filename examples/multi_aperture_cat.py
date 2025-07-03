
import astropy.units as u
from typing import Optional, Union, List

import os
#os.environ["GALFIND_WORK_DIR"] = "/nvme/scratch/work/Griley/galfind_scripts/galfind_config_Griley.ini"

from galfind.Data import morgan_version_to_dir
from galfind import Catalogue, EAZY, SED_code


def main(
    survey: str,
    version: str,
    instrument_names: List[str],
    aper_diams: u.Quantity,
    forced_phot_band: Optional[Union[str, List[str]]],
    SED_fitter_arr: List[SED_code],
):
    cat = Catalogue.pipeline(
        survey,
        version,
        instrument_names = instrument_names,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        version_to_dir_dict = morgan_version_to_dir,
    )
    #breakpoint()
    # load SED fitting results
    for SED_fitter in SED_fitter_arr:
        for aper_diam in aper_diams:
            SED_fitter(cat, aper_diam, load_PDFs = True, load_SEDs = True, update = True)
            #breakpoint()
    breakpoint()


if __name__ == "__main__":

    survey = "CEERSP1"
    version = "v9"
    instrument_names = ["ACS_WFC", "NIRCam"]
    aper_diams = [0.32, 0.5] * u.arcsec
    forced_phot_band = ["F277W", "F356W", "F444W"]
    SED_fitter_arr = [
        #EAZY({"templates": "fsps_larson", "lowz_zmax": 6.0}),
        EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
    ]

    main(
        survey,
        version,
        instrument_names,
        aper_diams,
        forced_phot_band,
        SED_fitter_arr,
    )
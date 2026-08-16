
from astropy import units as u
from typing import List, Union

from galfind import Data
from galfind.imaging.Data import morgan_version_to_dir


def main(
    survey: str,
    version: str,
    instrument_names: List[str] = ["ACS_WFC", "NIRCam"],
    aper_diams: u.Quantity = [0.32] * u.arcsec,
    forced_phot_band: Union[str, List[str]] = ["F277W", "F356W", "F444W"],
):
    #try:
    data = Data.from_survey_version_psfs(
        survey,
        version,
        instrument_names = instrument_names,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        version_to_dir_dict = morgan_version_to_dir,
    )
    data.mask()
    for band_data in data:
        #if band_data.filt.instrument.__class__.__name__ == "ACS_WFC":
        band_data.load_psf(method = "empirical")
        #breakpoint()
    data.plot_psf_eec()
    # [
    #     band_data.load_psf(method = "empirical")
    #     for band_data in data
    #     if band_data.filt.instrument.__class__.__name__ == "ACS_WFC"
    # ]
    # f444w_data = data["F444W"]
    # f444w_data.load_psf(
    #     method = "empirical",
    # )
    # except Exception as e:
    #     print(f"Error: {e}")

    #return data


if __name__ == "__main__":

    surveys = [f"JADES-DR3-GS-{loc}" for loc in ["North", "South", "East", "West"]] + \
        [f"JADES-DR3-GN-{loc}" for loc in ["Deep", "Medium", "Parallel"]] + \
        [f"PRIMER-{field}" for field in ["UDS", "COSMOS"]] + \
        [f"NEP-{int(i + 1)}" for i in range(4)] + \
        [f"CEERSP{i}" for i in range(1, 11)] + \
        ["NGDEEP"]
    versions = ["v13"] * 7 + ["v12"] * 2 + ["v14"] * 4 + ["v14"] * 10 + ["v14"]

    surveys = ["CEERSP10"]
    versions = ["v14"] * len(surveys)

    assert len(surveys) == len(versions)
    for survey, version in zip(surveys, versions):
        main(survey, version)
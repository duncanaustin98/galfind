
import astropy.units as u
import os

os.environ["GALFIND_CONFIG_DIR"] = "/nvme/scratch/work/austind/GALFIND/testing"
os.environ["GALFIND_CONFIG_NAME"] = "test_galfind_config.ini"

from galfind import Catalogue_Creator, Catalogue, LePhare
from galfind.Data import morgan_version_to_dir

def main(
    survey,
    version,
    instrument_names,
    forced_phot_band,
    aper_diams,
    min_flux_pc_err,
):
    cat = Catalogue.pipeline(
        survey, 
        version, 
        instrument_names = instrument_names, 
        #version_to_dir_dict = morgan_version_to_dir,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        min_flux_pc_err = min_flux_pc_err,
        im_str = ["test"],
    )
    SED_fit_params = {"GAL_TEMPLATES": "BC03_Chabrier2003_Z(m42_m62)"}
    LePhare_fitter = LePhare(SED_fit_params)
    LePhare_fitter.compile(cat.filterset)
    # fit catalogue
    LePhare_fitter(cat, aper_diams[0])


if __name__ == "__main__":

    survey = "test"
    version = "v0"
    instrument_names = ["ACS_WFC", "NIRCam"]
    forced_phot_band = ["F200W", "F444W"]
    aper_diams = [0.32] * u.arcsec
    min_flux_pc_err = 10.0

    # import time
    # print("COSMOS-Web-1A LePhare sleeping for 2hrs")
    # time.sleep((2 * u.hour).to(u.second).value)

    # survey = "COSMOS-Web-1A"
    # version = "v11"
    # instrument_names = ["ACS_WFC", "NIRCam"]
    # forced_phot_band = ["F444W"]
    # aper_diams = [0.32] * u.arcsec
    # min_flux_pc_err = 10.0

    main(survey, version, instrument_names, forced_phot_band, aper_diams, min_flux_pc_err)


import astropy.units as u
import os

# os.environ["GALFIND_CONFIG_DIR"] = "/nvme/scratch/work/austind/GALFIND/testing"
# os.environ["GALFIND_CONFIG_NAME"] = "test_galfind_config.ini"

from galfind import Multiple_Filter, Catalogue, LePhare
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
        version_to_dir_dict = morgan_version_to_dir,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        min_flux_pc_err = min_flux_pc_err,
        #im_str = ["test"],
    )
    SED_fit_params = {"GAL_TEMPLATES": "BC03_Chabrier2003_Zm42m62"}
    acs_wfc_nircam_medwide = Multiple_Filter.from_instruments(["ACS_WFC", "NIRCam"], keep_suffix = ["M", "W", "LP"])
    LePhare_fitter = LePhare(
        SED_fit_params,
        filterset = acs_wfc_nircam_medwide,
        gal_lib_name = "BC03_ACS_WFC+NIRCam_MedWide_HZ_Dusty",
        star_lib_name = "STAR+BD_ACS_WFC+NIRCam_MedWide",
        qso_lib_name = "QSO_MARA_ACS_WFC+NIRCam_MedWide_HZ_Dusty",
    ) # filterset = cat.filterset, 
    # fit catalogue
    LePhare_fitter(
        cat,
        aper_diams[0],
        load_PDFs = True,
        load_SEDs = True,
        update = False,
    )


if __name__ == "__main__":

    # survey = "test"
    # version = "v0"
    # instrument_names = ["ACS_WFC", "NIRCam"]
    # forced_phot_band = ["F200W", "F444W"]
    # aper_diams = [0.32] * u.arcsec
    # min_flux_pc_err = 10.0

    # import time
    # print("COSMOS-Web LePhare sleeping for 4hrs")
    # time.sleep((4 * u.hour).to(u.second).value)

    #survey = "COSMOS-Web-3B"
    version = "v11"
    instrument_names = ["ACS_WFC", "NIRCam"]
    forced_phot_band = ["F444W"]
    aper_diams = [0.32] * u.arcsec
    min_flux_pc_err = 10.0
    #breakpoint()
    for survey in [f"COSMOS-Web-{x}{letter}" for x in range(8) for letter in ["A"]]: # , "B"
        try:
            main(survey, version, instrument_names, forced_phot_band, aper_diams, min_flux_pc_err)
        except Exception as e:
           print(f"Failed to run LePhare on {survey}: {e}")

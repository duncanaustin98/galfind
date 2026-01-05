import astropy.units as u
import os

from galfind import Data, Catalogue, EAZY, Bagpipes, Band_SNR_Selector, MUV_Calculator
from galfind.Data import morgan_version_to_dir
from galfind.Property_calculator import MUV_SED_Property_Calculator

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
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        min_flux_pc_err = min_flux_pc_err,
        version_to_dir_dict = morgan_version_to_dir,
    )
    # load EAZY results into catalogue
    EAZY_SED_fitter = EAZY({"templates": "fsps_larson", "lowz_zmax": None})
    EAZY_SED_fitter(cat, aper_diams[0], update = True)
    # # construct MUV calculator
    # Muv_calculator_eazy = MUV_SED_Property_Calculator(aper_diams[0], EAZY_SED_fitter)
    # Muv_calculator_eazy(cat, update = True)
    # construct MUV calculator
    Muv_calculator_eazy = MUV_SED_Property_Calculator(aper_diams[0], EAZY_SED_fitter, ext_src_corrs = None)
    Muv_calculator_eazy(cat, update = True)

if __name__ == "__main__":

    #survey = "COSMOS-Web-3A"
    version = "v11"
    instrument_names = ["ACS_WFC", "NIRCam"]
    forced_phot_band = ["F444W"]
    aper_diams = [0.32] * u.arcsec
    min_flux_pc_err = 10.0

    for survey in [f"COSMOS-Web-{x}{letter}" for x in range(8) for letter in ["A", "B"]]:
        try:
            main(survey, version, instrument_names, forced_phot_band, aper_diams, min_flux_pc_err)
        except Exception as e:
            print(f"Failed to run {survey}: {e}")
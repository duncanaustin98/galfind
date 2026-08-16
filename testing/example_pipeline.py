
import time
import os
os.environ["GALFIND_CONFIG_DIR"] = os.getcwd()
os.environ["GALFIND_CONFIG_NAME"] = "test_galfind_config.ini"
from galfind import Data, Catalogue, EAZY, Bagpipes, Band_SNR_Selector, MUV_Calculator

from ..conftest import (
    test_survey,
    test_version,
    test_instrument_names,
    test_aper_diams,
    test_forced_phot_band_,
)

def main():
    start = time.time()

    # load data
    data = Data.pipeline(
        survey=test_survey,
        version=test_version,
        instrument_names=test_instrument_names,
        im_str = "test",
        rms_err_ext_name = "RMS_ERR",
        aper_diams = test_aper_diams,
        forced_phot_band = test_forced_phot_band_,
    )
    print(data)
    breakpoint()

    # load catalogue
    cat = Catalogue.from_data(data)
    #print(cat)
    #breakpoint()

    # run SED fitting
    eazy_fitter = EAZY({"templates": "fsps_larson", "lowz_zmax": None})
    #print(eazy_fitter)
    #breakpoint()
    for aper_diam in test_aper_diams:
        eazy_fitter(cat, aper_diam, update = True)
    #print(cat)
    #breakpoint()

    # compute UV magnitudes for these sources
    Muv_calculator = MUV_Calculator(test_aper_diams[0], eazy_fitter.label)
    #print(Muv_calculator)
    #breakpoint()
    Muv_calculator(cat)
    #print(cat)
    breakpoint()

    # select >5sigma in 0.32as,F444W
    five_sigma_f444w = Band_SNR_Selector(test_aper_diams[0], "F444W", "detect", 5.0)
    cropped_cat = five_sigma_f444w(cat)
    print(cropped_cat)
    breakpoint()

    # run bagpipes on cropped catalogue
    pipes_fitter = Bagpipes(
        {
            "fix_z": False, #EAZY({"templates": "fsps_larson", "lowz_zmax": None}).label,
            "z_sigma": 3.0,
            "sfh": "continuity_bursty",
            "z_calculator": Redshift_Extractor(aper_diams[0], EAZY({"templates": "fsps_larson", "lowz_zmax": None})),
            "sps_model": "BPASS",
            "fixed_bin_ages": [3.0, 10.0] * u.Myr,
        }
    )
    print(pipes_fitter)
    # pipes_fitter(cropped_cat, aper_diam = test_aper_diams[0], update = True)
    breakpoint()
    end = time.time()
    print(f"Total runtime: {end - start:.2f} seconds")



if __name__ == "__main__":
    main()
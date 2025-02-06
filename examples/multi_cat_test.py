
import numpy as np
import astropy.units as u

from galfind import Combined_Catalogue, Catalogue
from galfind.Data import morgan_version_to_dir

def main():
    surveys = [f"CEERSP{i}" for i in range(1, 11)]
    versions = list(np.full(len(surveys), "v9"))
    instrument_names = ["ACS_WFC", "NIRCam"] # "ACS_WFC"
    aper_diams = [0.32] * u.arcsec
    forced_phot_band = ["F277W", "F356W", "F444W"]
    min_flux_pc_err = 10.

    assert len(surveys) == len(versions)
    cat_arr = np.zeros(len(surveys), dtype=Catalogue)
    for i, (survey, version) in enumerate(zip(surveys, versions)):
        cat_arr[i] = Catalogue.pipeline(
            survey,
            version,
            instrument_names = instrument_names,
            version_to_dir_dict = morgan_version_to_dir,
            aper_diams = aper_diams,
            forced_phot_band = forced_phot_band,
            min_flux_pc_err = min_flux_pc_err,
            #crops = EPOCHS_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), allow_lowz=False)
        )
    combined_cat = Combined_Catalogue.from_cats(cat_arr, survey = "CEERS", version = "v9", overwrite = False)
    breakpoint()



if __name__ == "__main__":
    main()

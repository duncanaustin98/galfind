import astropy.units as u

from galfind import EAZY, Catalogue, Galfind_Catalogue_Creator, config


def nadams_jaguar_path(
    version,
    survey,
    is_half=True,
    fits_cat_dir=config["DEFAULT"]["GALFIND_DATA"],
):
    return f"{fits_cat_dir}/{version}_SimDepth_{survey}_{'half' if is_half else ''}.fits"


def make_EAZY_SED_fit_params_arr(SED_code_arr, templates_arr, lowz_zmax_arr):
    return [
        {"code": code, "templates": templates, "lowz_zmax": lowz_zmax}
        for code, templates, lowz_zmaxs in zip(
            SED_code_arr, templates_arr, lowz_zmax_arr
        )
        for lowz_zmax in lowz_zmaxs
    ]


def main():
    version = "JAGUAR"
    survey = "CLIO_v9"
    instrument_names = ["NIRCam"]
    aper_diam = 0.32 * u.arcsec
    min_pc_err = 10
    SED_code_arr = [EAZY()]
    templates_arr = ["fsps_larson"]
    lowz_zmax_arr = [[4.0, 6.0, None]]
    fits_cat_path = nadams_jaguar_path(version, survey)

    cat_creator = Galfind_Catalogue_Creator("loc_depth", aper_diam, min_pc_err)
    SED_fit_params_arr = make_EAZY_SED_fit_params_arr(
        SED_code_arr, templates_arr, lowz_zmax_arr
    )
    cat = Catalogue.from_fits_cat(
        fits_cat_path,
        version,
        instrument_names,
        cat_creator,
        survey,
        SED_fit_params_arr,
    )


if __name__ == "__main__":
    main()

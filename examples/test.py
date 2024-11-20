import astropy.units as u
from galfind import Catalogue, Catalogue_Creator, Data, EAZY, LePhare
from galfind import Colour_Selector, Unmasked_Instrument_Selector, EPOCHS_Selector
from galfind.Data import morgan_version_to_dir
# Load in a JOF data object
survey = "JOF"
version = "v11"
instrument_names = ["NIRCam"]
aper_diams = [0.32] * u.arcsec
forced_phot_band = ["F277W", "F356W", "F444W"]
min_flux_pc_err = 10.

def test_selection():
    # imports
    JOF_cat = Catalogue.pipeline(
        survey,
        version,
        instrument_names = instrument_names, 
        version_to_dir_dict = morgan_version_to_dir,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        min_flux_pc_err = min_flux_pc_err
    )
    JOF_cat.load_sextractor_Re()
    SED_fit_params_arr = [{"templates": "fsps_larson", "lowz_zmax": 4.0}, {"templates": "fsps_larson", "lowz_zmax": 6.0}, {"templates": "fsps_larson", "lowz_zmax": None}]
    for SED_fit_params in SED_fit_params_arr:
        EAZY_fitter = EAZY(SED_fit_params)
        EAZY_fitter(JOF_cat, aper_diams[0], load_PDFs = True, load_SEDs = True, update = True)

    from galfind import EPOCHS_Selector
    epochs_selector = EPOCHS_Selector(allow_lowz = False, unmasked_instruments = "NIRCam")
    epochs_selected_cat = epochs_selector(JOF_cat, aper_diams[0], EAZY_fitter, return_copy = True)
    print(epochs_selected_cat)
    breakpoint()
    
    # Kokorev_red1_selector = Kokorev24_LRD_red1()
    # Kokorev_red2_selector = Kokorev24_LRD_red2()
    # Kokorev_LRD_selector = Kokorev24_LRD()
    # #selector = Colour_Selector(colour_bands = ["F115W", "F150W"], bluer_or_redder = "bluer", colour_val = 0.8)
    # Kokorev_red1_selector(JOF_cat, aper_diams[0])
    # Kokorev_red2_selector(JOF_cat, aper_diams[0])
    # Kokorev_LRD_selector(JOF_cat, aper_diams[0])
    #JOF_cat.del_hdu(hdu = "SELECTION")

def main():
    JOF_data = Data.pipeline(
        survey, 
        version, 
        instrument_names = instrument_names, 
        version_to_dir_dict = morgan_version_to_dir,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        min_flux_pc_err = min_flux_pc_err
    )
    cat_path = JOF_data.phot_cat_path
    filterset = JOF_data.filterset
    # [0.32] * u.arcsec hardcoded for now
    cat_creator = Catalogue_Creator(survey, version, cat_path, filterset, aper_diams)
    cat = cat_creator(cropped = False)

    # LePhare_SED_fit_params = {"GAL_TEMPLATES": "BC03_Chabrier2003_Z(m42_m62)"}
    # LePhare_fitter = LePhare(LePhare_SED_fit_params)
    # LePhare_fitter.compile(filterset)

    #SED_fit_params_arr = [{"templates": "fsps_larson", "lowz_zmax": None}]
    SED_fit_params_arr = [{"templates": "fsps_larson", "lowz_zmax": 4.0}, {"templates": "fsps_larson", "lowz_zmax": 6.0}, {"templates": "fsps_larson", "lowz_zmax": None}]
    for SED_fit_params in SED_fit_params_arr:
        EAZY_fitter = EAZY(SED_fit_params)
        EAZY_fitter(cat, aper_diams[0], load_PDFs = True, load_SEDs = True, update = True)
    EPOCHS_Selector()(cat, aper_diams[0], EAZY_fitter)

if __name__ == "__main__":
    #main()
    test_selection()

    # LePhare_SED_fit_params = {"GAL_TEMPLATES": "BC03_Chabrier2003_Z(m42_m62)"}
    # EAZY_SED_fit_params = {"templates": "fsps_larson", "lowz_zmax": None}
    # LePhare_fitter = LePhare(LePhare_SED_fit_params)
    # LePhare_fitter.compile()
    # print(LePhare_fitter.SED_fit_params)
    #EAZY_fitter = EAZY(EAZYSED_fit_params)
    #breakpoint()
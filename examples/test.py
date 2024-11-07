import astropy.units as u
from galfind import Catalogue_Creator, Data, EAZY
from galfind.Data import morgan_version_to_dir
# Load in a JOF data object
survey = "JOF"
version = "v11"
instrument_names = ["NIRCam"]
aper_diams = [0.32] * u.arcsec
forced_phot_band = ["F277W", "F356W", "F444W"]
min_flux_pc_err = 10.

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
cat = cat_creator()

SED_fit_params = [{"templates": "fsps_larson", "lowz_zmax": 4.0}, {"templates": "fsps_larson", "lowz_zmax": 6.0}, {"templates": "fsps_larson", "lowz_zmax": None}]
EAZY_fitter = EAZY(SED_fit_params)
EAZY_SED_results_arr = EAZY_fitter(cat, aper_diams[0], load_PDFs = False, load_SEDs = False)
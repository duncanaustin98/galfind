# make_cutouts.py
import astropy.units as u

from galfind import Catalogue, config #, LePhare, EAZY, 
from galfind.Catalogue_Creator import GALFIND_Catalogue_Creator

def Trussler2024_smouldering_IDs(survey):
    # v9 in JADES+JEMS, v11 in JOF
    ID_dict = {"JADES-Deep-GS+JEMS": [7405, 8222, 12194, 13229, 13993, 15190, 18287, 20716, 26289, 36359], "JOF": [377, 1251, 1667, 12149, 15795]}
    return ID_dict[survey]

if __name__ == "__main__":
    # parameters to change
    surveys = ["JADES-Deep-GS+JEMS"] #[f"CEERSP{i + 1}" for i in range(10)]
    version = "v9"
    instruments = ["NIRCam"] #, "ACS_WFC"]
    excl_bands = []

    # IDs for cutouts
    IDs_arr = []
    #IDs_arr = [[52]] # for i in range(10)]
    cutout_size = 32

    # fixed parameters (for now)
    forced_phot_band = ["F277W", "F356W", "F444W"]
    aper_diams = [0.32] * u.arcsec
    cat_creator = GALFIND_Catalogue_Creator("loc_depth", aper_diams[0], 10)
    for survey, IDs in zip(surveys, IDs_arr):
        cat = Catalogue.from_pipeline(survey = survey, version = version, instruments = instruments, aper_diams = aper_diams, \
            cat_creator = cat_creator, code_names = [], lowz_zmax = [], xy_offset = [0, 0], \
            forced_phot_band = forced_phot_band, excl_bands = excl_bands, loc_depth_min_flux_pc_errs = [10], \
            n_loc_depth_samples = 20, templates_arr = [], fast = False)
        cat.make_cutouts(IDs, cutout_size = cutout_size)
        
            
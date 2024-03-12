# make_cutouts.py
import astropy.units as u

from galfind import Catalogue, config #, LePhare, EAZY, 
from galfind.Catalogue_Creator import GALFIND_Catalogue_Creator

if __name__ == "__main__":
    # parameters to change
    surveys = ["CEERSP8"] #[f"CEERSP{i + 1}" for i in range(10)]
    version = "v9"
    instruments = ["NIRCam", "ACS_WFC"]
    excl_bands = []

    # IDs for cutouts
    IDs_arr = [[52]] # for i in range(10)]
    cutout_size = 32

    # fixed parameters (for now)
    forced_phot_band = ["f277W", "f356W", "f444W"]
    aper_diams = [0.32] * u.arcsec
    cat_creator = GALFIND_Catalogue_Creator("loc_depth", aper_diams[0], 10)
    for survey, IDs in zip(surveys, IDs_arr):
        cat = Catalogue.from_pipeline(survey = survey, version = version, instruments = instruments, aper_diams = aper_diams, \
            cat_creator = cat_creator, code_names = [], lowz_zmax = [], xy_offset = [0, 0], \
            forced_phot_band = forced_phot_band, excl_bands = excl_bands, loc_depth_min_flux_pc_errs = [10], \
            n_loc_depth_samples = 20, templates_arr = [], fast = False)
        cat.make_cutouts(IDs, cutout_size = cutout_size)
        
            
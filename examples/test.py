import astropy.units as u
import numpy as np
from tqdm import tqdm
from galfind import Filter, Catalogue, Catalogue_Creator, Data, EAZY, LePhare, Bagpipes
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
    SED_fit_params_arr = [
        {"templates": "fsps_larson", "lowz_zmax": 4.0},
        {"templates": "fsps_larson", "lowz_zmax": 6.0},
        {"templates": "fsps_larson", "lowz_zmax": None}
    ]

    JOF_cat = Catalogue.pipeline(
        survey,
        version,
        instrument_names = instrument_names, 
        version_to_dir_dict = morgan_version_to_dir,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        min_flux_pc_err = min_flux_pc_err
    )
    # load sextractor half-light radii
    JOF_cat.load_sextractor_Re()

    # load EAZY SED fitting results
    for SED_fit_params in SED_fit_params_arr:
        EAZY_fitter = EAZY(SED_fit_params)
        EAZY_fitter(JOF_cat, aper_diams[0], load_PDFs = True, load_SEDs = True, update = True)

    # perform EPOCHS selection
    epochs_selector = EPOCHS_Selector(aper_diams[0], EAZY_fitter, allow_lowz = False, unmasked_instruments = "NIRCam")
    EPOCHS_JOF_cat = epochs_selector(JOF_cat, return_copy = True)
    breakpoint()
    
    # from galfind import EPOCHS_Selector
    # epochs_selector = EPOCHS_Selector(allow_lowz = False, unmasked_instruments = "NIRCam")
    # epochs_selected_cat = epochs_selector(JOF_cat, aper_diams[0], EAZY_fitter, return_copy = True)
    # epochs_selector_lowz = EPOCHS_Selector(allow_lowz = True, unmasked_instruments = "NIRCam")
    # epochs_selected_cat_lowz = epochs_selector_lowz(JOF_cat, aper_diams[0], EAZY_fitter, return_copy = True)

    SED_fit_label = "EAZY_fsps_larson_zfree"
    from galfind import MUV_Calculator, Xi_Ion_Calculator, M99
    for beta_dust_conv in [None, M99]: #, Reddy18(C00(), 100 * u.Myr), Reddy18(C00(), 300 * u.Myr)]:
        for fesc_conv in ["Chisholm22"]: # None, 0.1, 0.2, 0.5, 
            calculator = Xi_Ion_Calculator(aper_diams[0], SED_fit_label, beta_dust_conv = beta_dust_conv, fesc_conv = fesc_conv)
            calculator(JOF_cat, n_chains = 10_000, output = False, n_jobs = 1)
    MUV_calculator = MUV_Calculator(aper_diams[0], SED_fit_label)
    MUV_calculator(JOF_cat, n_chains = 10_000, output = False, n_jobs = 1)
    breakpoint()

def test_pipes():
    
    JOF_cat = Catalogue.pipeline(
        survey,
        version,
        instrument_names = instrument_names, 
        version_to_dir_dict = morgan_version_to_dir,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        min_flux_pc_err = min_flux_pc_err,
        crops = {"SELECTION": EPOCHS_Selector(allow_lowz=True). \
            _get_selection_name(aper_diams[0], \
            EAZY({"templates": "fsps_larson", "lowz_zmax": None}).label)}
    )

    #JOF_cat.load_sextractor_Re()
    # {"templates": "fsps_larson", "lowz_zmax": 4.0}, {"templates": "fsps_larson", "lowz_zmax": 6.0}, 

    SED_fit_params_arr = [{"templates": "fsps_larson", "lowz_zmax": None}]
    for SED_fit_params in SED_fit_params_arr:
        EAZY_fitter = EAZY(SED_fit_params)
        EAZY_fitter(JOF_cat, aper_diams[0], load_PDFs = False, load_SEDs = False, update = True)

    # EPOCHS_JOF_cat = EPOCHS_Selector()(JOF_cat, aper_diams[0], EAZY_fitter, return_copy = True)

    pipes_SED_fit_params = {"fix_z": EAZY_fitter.label, "fesc": None}
    pipes_fitter = Bagpipes(pipes_SED_fit_params)
    pipes_fitter(JOF_cat, aper_diams[0], save_PDFs = False, load_SEDs = False, load_PDFs = False, overwrite = False)

def test_UVLF():

    JOF_cat = Catalogue.pipeline(
        survey,
        version,
        instrument_names = instrument_names, 
        version_to_dir_dict = morgan_version_to_dir,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        min_flux_pc_err = min_flux_pc_err,
        crops = {"SELECTION": EPOCHS_Selector(allow_lowz=True). \
            _get_selection_name(aper_diams[0], \
            EAZY({"templates": "fsps_larson", "lowz_zmax": None}).label)}
    )
    #JOF_cat.load_sextractor_Re()
    # {"templates": "fsps_larson", "lowz_zmax": 4.0}, {"templates": "fsps_larson", "lowz_zmax": 6.0}, 
    SED_fit_params_arr = [{"templates": "fsps_larson", "lowz_zmax": None}]
    for SED_fit_params in SED_fit_params_arr:
        EAZY_fitter = EAZY(SED_fit_params)
        EAZY_fitter(JOF_cat, aper_diams[0], load_PDFs = True, load_SEDs = True, update = True)

    from galfind import MUV_Calculator
    MUV_calculator = MUV_Calculator(aper_diams[0], EAZY_fitter.label)
    MUV_calculator(JOF_cat, n_chains = 10_000, output = False, n_jobs = 1)

    from galfind import Number_Density_Function
    UV_LF_z9 = Number_Density_Function.from_single_cat(
        JOF_cat,
        MUV_calculator.name,
        np.arange(-21.25, -17.25, 0.5),
        [8.5, 9.5],
        aper_diam = aper_diams[0],
        SED_fit_code = EAZY_fitter,
        x_origin = "phot_rest",
    )

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

def check_multinest():
    import pymultinest as pmn

if __name__ == "__main__":
    #test_load()
    #main()
    test_selection()
    #test_UVLF()
    #test_pipes()
    #check_multinest()

    # LePhare_SED_fit_params = {"GAL_TEMPLATES": "BC03_Chabrier2003_Z(m42_m62)"}
    # EAZY_SED_fit_params = {"templates": "fsps_larson", "lowz_zmax": None}
    # LePhare_fitter = LePhare(LePhare_SED_fit_params)
    # LePhare_fitter.compile()
    # print(LePhare_fitter.SED_fit_params)
    #EAZY_fitter = EAZY(EAZYSED_fit_params)
    #breakpoint()

# def test_load():
#     import numpy as np
#     import time
#     from astropy.table import Table

#     # Create a large random array
#     array = np.random.rand(10000, 1)
#     meta = {"blah": "blah"}

#     # Save as .npy
#     npy_file = "data.npy"
#     npy_meta_file = "data.meta.npy"
#     np.save(npy_file, array)
#     np.save(npy_meta_file, meta)

#     # Save as .npz
#     npz_file = "data.npz"
#     np.savez(npz_file, array=array, meta=meta)

#     save_tab = Table({"x": array})
#     save_tab.meta = meta
#     save_tab.write("data.ecsv", overwrite=True)

#     # Load .npy
#     start = time.time()
#     loaded_npy = np.load(npy_file)
#     loaded_npy_meta = np.load(npy_meta_file, allow_pickle=True).item()
#     end = time.time()
#     print(f"Loading .npy: {end - start:.6f} seconds")

#     # Load .npz
#     start = time.time()
#     loaded_npz = np.load(npz_file, allow_pickle=True)
#     loaded_array_from_npz = loaded_npz["array"] # Extract the array
#     loaded_npz_meta = loaded_npz["meta"]
#     end = time.time()
#     print(f"Loading .npz: {end - start:.6f} seconds")

#     start = time.time()
#     loaded_tab = Table.read("data.ecsv")
#     loaded_array_from_tab = loaded_tab["x"]
#     loaded_ecsv_meta = loaded_tab.meta
#     end = time.time()
#     print(f"Loading .ecsv: {end - start:.6f} seconds")
#     breakpoint()
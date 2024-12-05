import astropy.units as u
from tqdm import tqdm
from galfind import Filter, Catalogue, Catalogue_Creator, Data, EAZY, LePhare
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

    # JOF_cat.load_sextractor_Re()
    # {"templates": "fsps_larson", "lowz_zmax": 4.0}, {"templates": "fsps_larson", "lowz_zmax": 6.0}, 
    SED_fit_params_arr = [{"templates": "fsps_larson", "lowz_zmax": None}]
    for SED_fit_params in SED_fit_params_arr:
        EAZY_fitter = EAZY(SED_fit_params)
        EAZY_fitter(JOF_cat, aper_diams[0], load_PDFs = True, load_SEDs = True, update = True)

    # from galfind import EPOCHS_Selector
    # epochs_selector = EPOCHS_Selector(allow_lowz = False, unmasked_instruments = "NIRCam")
    # epochs_selected_cat = epochs_selector(JOF_cat, aper_diams[0], EAZY_fitter, return_copy = True)
    # epochs_selector_lowz = EPOCHS_Selector(allow_lowz = True, unmasked_instruments = "NIRCam")
    # epochs_selected_cat_lowz = epochs_selector_lowz(JOF_cat, aper_diams[0], EAZY_fitter, return_copy = True)

    SED_fit_label = "EAZY_fsps_larson_zfree"
    from galfind import Xi_Ion_Calculator
    calculator = Xi_Ion_Calculator(aper_diams[0], SED_fit_label)
    calculator(JOF_cat, n_chains = 10_000, output = False, n_jobs = 1)
    breakpoint()

def test_docs():
    import astropy.units as u
    from copy import deepcopy
    from galfind import Catalogue, EAZY
    from galfind.Data import morgan_version_to_dir
    survey = "JOF"
    version = "v11"
    instrument_names = ["NIRCam"]
    aper_diams = [0.32] * u.arcsec
    forced_phot_band = ["F277W", "F356W", "F444W"]
    min_flux_pc_err = 10.

    JOF_cat = Catalogue.pipeline(
        survey,
        version,
        instrument_names = instrument_names, 
        version_to_dir_dict = morgan_version_to_dir,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        min_flux_pc_err = min_flux_pc_err
    )

    SED_fit_params_arr = [{"templates": "fsps_larson", "lowz_zmax": None}]
    for SED_fit_params in SED_fit_params_arr:
        EAZY_fitter = EAZY(SED_fit_params)
        EAZY_fitter(JOF_cat, aper_diams[0], load_PDFs = True, load_SEDs = True, update = True)
    SED_fit_label = EAZY_fitter.label
    from galfind import UV_Beta_Calculator
    beta_calculator = UV_Beta_Calculator(
        aper_diam = aper_diams[0],
        SED_fit_label = SED_fit_label,
        rest_UV_wav_lims = [1_250., 3_000.] * u.AA
    )
    phot_rest_z14 = deepcopy(JOF_cat[717].aper_phot[aper_diams[0]].SED_results[SED_fit_label].phot_rest)
    print(phot_rest_z14)
    print(phot_rest_z14.__dict__)
    beta_calculator(
        phot_rest_z14,
        n_chains = 1, 
        output = False,
        overwrite = False,
        n_jobs = 1
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


if __name__ == "__main__":
    #test_load()
    #main()
    #test_selection()
    test_docs()

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
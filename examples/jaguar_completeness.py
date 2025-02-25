
import astropy.units as u
from typing import List

from galfind import Data, Catalogue_Creator, EAZY
from galfind.selection import Completeness
from galfind.Data import morgan_version_to_dir
from galfind.Catalogue import jaguar_phot_labels, load_jaguar_phot

def main(
    survey: str,
    version: str,
    instrument_names: List[str],
    aper_diams: u.Quantity,
    forced_phot_band: List[str],
    realization: int,
):
    # construct appropriate galfind data object
    data = Data.pipeline(
        survey = survey,
        version = version,
        instrument_names = instrument_names,
        version_to_dir_dict = morgan_version_to_dir,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
    )
    # load the Jaguar catalogue
    jaguar_cat_path = f"/raid/scratch/data/JAGUAR/JADES_SF_mock_r{str(int(realization))}_v1.2.fits"
    jaguar_cat_creator = Catalogue_Creator(
        survey = f"JAGUAR-{survey}-r{str(int(realization))}",
        version = version,
        cat_path = jaguar_cat_path,
        filterset = data.filterset,
        aper_diams = aper_diams, # not relevant in this case, but still required
        ID_label = "ID",
        skycoords_labels = {"RA": "RA", "DEC": "DEC"},
        get_phot_labels = jaguar_phot_labels,
        load_phot_func = load_jaguar_phot,
        load_phot_kwargs = {
            "ZP": u.nJy.to(u.ABmag),
            "min_flux_pc_err": 10.0,
            "incl_errs": False,
        },
        load_mask_func = None,
        load_depth_func = None,
        apply_gal_instr_mask = False,
    )
    # load the Jaguar catalogue
    jaguar_cat = jaguar_cat_creator()
    # attach the relevant data object to the Jaguar catalogue
    jaguar_cat.data = data
    # construct array of required EAZY SED fitter objects
    SED_fitter_arr = [
        EAZY({"templates": "fsps_larson", "lowz_zmax": 4.0}),
        EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
    ]
    # make 2D completeness grid from the Jaguar catalogue
    completeness = Completeness.from_sim_cat(
        jaguar_cat,
        SED_fitter_arr = SED_fitter_arr,
        sampler = None,
        aper_diam = aper_diams[0],
        mode = "n_nearest",
        depth_region = "all",
        x_calculator = None,
        y_calculator = None,
        x_arr = None,
        y_arr = None,
    )


if __name__ == "__main__":
    # input parameters for data objectconda act
    survey = "JOF"
    version = "v11"
    instrument_names = ["ACS_WFC", "NIRCam"]
    aper_diams = [0.32] * u.arcsec
    forced_phot_band = ["F277W", "F356W", "F444W"]
    realization = 1
    import time
    time.sleep(4*60*60) #4 * 60 * 60)
    main(
        survey,
        version,
        instrument_names,
        aper_diams,
        forced_phot_band,
        realization,
    )

    cat_path = "/raid/scratch/work/austind/GALFIND_WORK/"
    make_completeness_grid_from_cat(cat_path)

    # from astropy.table import Table
    # jaguar_cat_path = "/raid/scratch/data/JAGUAR/JADES_SF_mock_r1_v1.2.fits"
    # # open Jaguar catalogue to have a look at its length
    # tab = Table.read(jaguar_cat_path)
    # breakpoint()
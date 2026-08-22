import os
from copy import deepcopy
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
from pytest_lazy_fixtures import lf

# galfind/__init__.py resolves GALFIND_WORK/GALFIND_DATA (and derived
# paths like LOGGING_OUT_DIR) from the config file's *relative* values
# via os.path.abspath()/plain string use at import time -- i.e.
# relative to the process's cwd at that exact moment, not to this repo
# or config file. chdir to the repo root (this file's own directory)
# before galfind is ever imported so those always land under
# <repo_root>/testing/test_work, regardless of which directory tests
# are invoked from (this matches what the rest of the test suite
# already implicitly assumes via f"{os.getcwd()}/testing"-style paths).
_repo_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(_repo_root)

os.environ["GALFIND_CONFIG_DIR"] = os.path.join(_repo_root, "testing")
os.environ["GALFIND_CONFIG_NAME"] = "test_galfind_config.ini"

import galfind
from galfind.catalogues import Catalogue, Catalogue_Creator
from galfind.catalogues.Catalogue import (
    load_galfind_depths,
    load_phot,
    scattered_depth_labels,
    scattered_phot_labels,
)
from galfind.galaxy import Galaxy
from galfind.imaging import Data, Filter, Multiple_Filter, NIRCam
from galfind.imaging.Data import Band_Data, Stacked_Band_Data
from galfind.imaging.Instrument import JWST, Facility, Instrument
from galfind.photometry.Photometry_obs import Photometry_obs
from galfind.photometry.Photometry_rest import Photometry_rest
from galfind.sed_fitting import EAZY, LePhare
from galfind.sed_fitting.SED_result import SED_result
from galfind.selection.Selector import ID_Selector
from galfind.visualization.PDF import Redshift_PDF

test_galfind_data_dir = "test_data"
test_survey = "test"
test_version = "v0"
test_instrument_names = ["ACS_WFC", "NIRCam"]
test_bands_ = ["F814W", "F090W", "F200W", "F444W"]
test_aper_diams = [0.32] * u.arcsec
test_forced_phot_band_ = ["F200W", "F444W"]


def _mock_gaia_launch_job_async(*args, **kwargs):
    # Stands in for a live astroquery.gaia.Gaia.launch_job_async call,
    # returning an empty result table (no Gaia stars in the footprint) so
    # masking doesn't depend on the live Gaia archive, which is flaky/
    # rate-limited and unsuitable for automated test runs.
    class _MockJob:
        @staticmethod
        def get_results():
            return Table(
                {
                    "source_id": np.array([], dtype=np.int64),
                    "ra": np.array([], dtype=np.float64),
                    "dec": np.array([], dtype=np.float64),
                    "phot_g_mean_mag": np.array([], dtype=np.float64),
                    "radius_sersic": np.array([], dtype=np.float64),
                    "classlabel_dsc_joint": np.array([], dtype="U16"),
                    "vari_best_class_name": np.array([], dtype="U16"),
                }
            )

    return _MockJob()


@pytest.fixture(scope="session")
def test_bands():
    return test_bands_


@pytest.fixture(scope="session")
def test_forced_phot_band():
    return test_forced_phot_band_


@pytest.fixture(scope="session")
def survey():
    return test_survey


@pytest.fixture(scope="session")
def version():
    return test_version


@pytest.fixture(scope="session")
def instrument_names():
    return test_instrument_names


@pytest.fixture(scope="session")
def aper_diams():
    return test_aper_diams


@pytest.fixture(scope="session")
def nircam_pix_scale():
    return 0.03 * u.arcsec


def pytest_generate_tests(metafunc):
    slow_marker = metafunc.definition.get_closest_marker("slow")
    if "facility" in metafunc.fixturenames:
        if slow_marker:
            params = [
                pytest.param(param) for param in Facility.__subclasses__()
            ]
        else:
            params = [JWST]
        metafunc.parametrize("facility", params, indirect=True)
    elif "instrument" in metafunc.fixturenames:
        if slow_marker:
            params = [
                pytest.param(param) for param in Instrument.__subclasses__()
            ]
        else:
            params = [NIRCam]
        metafunc.parametrize("instrument", params, indirect=True)
    elif "filter" in metafunc.fixturenames:
        if slow_marker:
            multi_filter = Multiple_Filter.from_facilities(
                [facility for facility in Facility.__subclasses__()]
            )
            params = list(multi_filter)
        else:
            params = [Filter.from_SVO("JWST", "NIRCam", "F444W")]
        metafunc.parametrize("filter", params, indirect=True)


def pytest_collection_modifyitems(items):
    # auto-mark any test whose fixture closure includes `data`, since it
    # requires STPSF's external calibration data to build model PSFs
    for item in items:
        if "data" in item.fixturenames:
            item.add_marker(pytest.mark.requires_data)


@pytest.fixture(scope="session")
def facility(request):
    return request.param


@pytest.fixture(scope="session")
def facility_inst(facility):
    return facility()


@pytest.fixture(scope="session")
def instrument(request):
    return request.param


@pytest.fixture(scope="session")
def instrument_inst(instrument):
    return instrument()


@pytest.fixture(scope="session")
def filter(request):
    return request.param


@pytest.fixture(scope="session")
def f444w():
    return Filter.from_SVO("JWST", "NIRCam", "F444W")


@pytest.fixture(scope="session")
def blank_multi_filt():
    return Multiple_Filter([])


@pytest.fixture(scope="session")
def nircam_multi_filter():
    return Multiple_Filter.from_instrument("NIRCam")


@pytest.fixture(scope="session")
def multi_filter_test_bands(test_bands):
    return Multiple_Filter(
        [Filter.from_filt_name(name) for name in test_bands]
    )


# @pytest.fixture(scope = "session")
# def cat():
#     return

# @pytest.fixture(scope = "session")
# def gal():
#     return


@pytest.fixture(scope="session")
def eazy_fsps_larson_sed_fitter():
    return EAZY({"templates": "fsps_larson", "lowz_zmax": None})


@pytest.fixture(scope="session")
def eazy_sfhz_sed_fitter():
    return EAZY({"templates": "sfhz", "lowz_zmax": None})


@pytest.fixture(scope="session")
def lephare_sed_fitter():
    return LePhare({"GAL_TEMPLATES": "BC03_Chabrier2003_Zm42m62"})


@pytest.fixture(
    scope="session",
    params=[
        lf(eazy_fsps_larson_sed_fitter),
        lf(eazy_sfhz_sed_fitter),
        lf(lephare_sed_fitter),
    ],
)
def sed_fitter(request):
    return request.param


#################################################
# Hand-built z~9.5 Lyman-break "galaxy", constructed entirely from
# made-up flux measurements rather than real FITS/Gaia/EAZY data, so
# selectors' actual _selection_criteria/_failure_criteria logic (not
# just __init__) can be exercised without any real-data dependency.
#################################################

high_z_gal_bands = [
    "F090W",
    "F115W",
    "F150W",
    "F200W",
    "F277W",
    "F356W",
    "F410M",
    "F444W",
]
# fully absorbed by the IGM bluewards of Lyman-alpha at z=9.5
high_z_gal_non_detect_bands = {"F090W", "F115W"}
high_z_gal_z_true = 9.5
high_z_gal_depth = 28.5 * u.ABmag


@pytest.fixture(scope="session")
def high_z_filterset():
    return Multiple_Filter(
        [Filter.from_filt_name(name) for name in high_z_gal_bands]
    )


def _build_synthetic_gal(
    ID,
    filterset,
    sed_fitter,
    non_detect_bands,
    z_true,
    chi_sq,
    z_pdf_std,
    lowz_zmax_properties,
    sex_Re,
    detect_sigma_mult=8.0,
    non_detect_sigma_mult=-0.3,
    depth=high_z_gal_depth,
    seed=0,
):
    """Hand-build a `Galaxy` from made-up flux/SED-fit values -- shared
    by the `high_z_gal` and `garbage_gal` fixtures, which only differ in
    which bands are "detected" and what SED-fit/PDF/morphology values
    are attached."""
    filt_names = filterset.filt_names
    aper_diam = test_aper_diams[0]

    # 5-sigma depth -> 1-sigma flux uncertainty, uniform across bands
    sigma = depth.to(u.nJy) / 5.0
    flux = (
        np.array(
            [
                non_detect_sigma_mult
                if name in non_detect_bands
                else detect_sigma_mult
                for name in filt_names
            ]
        )
        * sigma
    )
    flux_errs = np.full(len(filt_names), 1.0) * sigma
    depths = {name: depth for name in filt_names}

    phot_obs = Photometry_obs(
        filterset, flux, flux_errs, depths, aper_diam, simulated=True
    )

    z_pdf = Redshift_PDF.from_1D_arr(
        np.random.default_rng(seed).normal(z_true, z_pdf_std, 5_000)
        * u.dimensionless_unscaled,
        SED_fit_params={},
    )
    sed_result = SED_result(
        sed_fitter,
        phot_obs,
        properties={
            "z": z_true * u.dimensionless_unscaled,
            "chi_sq": chi_sq,
        },
        property_errs={},
        property_PDFs={"z": z_pdf},
        SED=None,
    )
    sed_result.lowz_zmax_properties = lowz_zmax_properties
    phot_obs.SED_results = {sed_fitter.label: sed_result}

    gal = Galaxy(
        ID=ID,
        sky_coord=SkyCoord(ra=150.0 * u.deg, dec=2.0 * u.deg),
        aper_phot={aper_diam: phot_obs},
        selection_flags={},
        selection_kwargs={},
        cat_filterset=filterset,
        survey="synthetic",
        version="v0",
        simulated=True,
    )
    # SExtractor radii aren't derivable from photometry alone; fake a
    # value so the Sextractor_*_Radius_Selector sub-criteria in
    # EPOCHS/COSMOS-Web can also be exercised without a real
    # SExtractor catalogue.
    gal.sex_Re = {name: sex_Re for name in filt_names}
    return gal


@pytest.fixture(scope="session")
def high_z_gal(high_z_filterset, eazy_fsps_larson_sed_fitter):
    return _build_synthetic_gal(
        ID=1,
        filterset=high_z_filterset,
        sed_fitter=eazy_fsps_larson_sed_fitter,
        non_detect_bands=high_z_gal_non_detect_bands,
        z_true=high_z_gal_z_true,
        chi_sq=15.0,
        z_pdf_std=0.15,
        # a genuine high-z source should be a much worse fit when its
        # redshift is capped at low-z
        lowz_zmax_properties={
            "4.0": {"zbest": 3.5, "chi2_best": 200.0},
            "6.0": {"zbest": 5.5, "chi2_best": 180.0},
        },
        sex_Re=60.0 * u.marcsec,  # resolved, not hot-pixel-sized
    )


@pytest.fixture(scope="session")
def garbage_gal(high_z_filterset, eazy_fsps_larson_sed_fitter):
    """A "garbage" contaminant: undetected in every band (pure noise),
    an ambiguous low-z best fit that's a poor fit overall, a broad/
    unconstrained redshift PDF, a low-z solution that fits about as
    well as the best fit, and a hot-pixel-sized SExtractor radius --
    i.e. everything a real robust high-z selection should reject."""
    return _build_synthetic_gal(
        ID=2,
        filterset=high_z_filterset,
        sed_fitter=eazy_fsps_larson_sed_fitter,
        non_detect_bands=set(high_z_gal_bands),  # undetected everywhere
        detect_sigma_mult=0.5,
        non_detect_sigma_mult=0.5,
        z_true=1.3,
        chi_sq=45.0,
        z_pdf_std=4.0,  # broad, unconstrained
        # low-z-capped fit is about as good as the free fit -> should
        # *not* be selected as a robust high-z dropout
        lowz_zmax_properties={
            "4.0": {"zbest": 1.1, "chi2_best": 44.0},
            "6.0": {"zbest": 1.2, "chi2_best": 46.0},
        },
        sex_Re=5.0 * u.marcsec,  # hot-pixel-sized
        seed=1,
    )


@pytest.fixture(scope="session")
def synthetic_test_cat(high_z_gal, garbage_gal, high_z_filterset):
    """A hand-built 2-source `Catalogue` (one genuine z~9.5 dropout,
    one garbage contaminant) for exercising selectors' per-galaxy
    selection logic without any real FITS/Gaia/EAZY data. Note this
    catalogue has no real FITS file backing it, so it can't be passed
    directly to a `Selector` (which needs `cat.open_cat()` to persist
    results) -- iterate `for gal in synthetic_test_cat` and call
    `selector(gal)` per galaxy instead.
    """
    cat_creator = SimpleNamespace(
        survey="synthetic",
        version="v0",
        filterset=high_z_filterset,
        cat_path=None,
    )
    return Catalogue([high_z_gal, garbage_gal], cat_creator)


@pytest.fixture(scope="session")
def synthetic_eazy_cat_path(high_z_filterset):
    """Hand-author a minimal, real, on-disk FITS catalogue holding the
    same two synthetic sources as `high_z_gal`/`garbage_gal` (one z~9.5
    dropout, one all-noise contaminant), using the "scattered" column
    convention (`scattered_phot_labels`/`scattered_depth_labels`) so it
    can be loaded by a plain `Catalogue_Creator` without needing any
    real FITS imaging or Gaia masking. This is what lets real SED
    fitting (which needs a genuine FITS-backed Catalogue, unlike
    Selectors/property calculators) run on synthetic photometry.
    """
    filt_names = high_z_filterset.filt_names
    sigma_Jy = (high_z_gal_depth.to(u.Jy) / 5.0).value

    row_flux = {
        1: [
            (-0.3 if name in high_z_gal_non_detect_bands else 8.0) * sigma_Jy
            for name in filt_names
        ],
        2: [0.5 * sigma_Jy for _ in filt_names],  # garbage: pure noise
    }

    tab = Table()
    tab["NUMBER"] = np.array([1, 2], dtype=int)
    tab["ALPHA_J2000"] = np.array([150.0, 150.001]) * u.deg
    tab["DELTA_J2000"] = np.array([2.0, 2.001]) * u.deg
    for i, filt in enumerate(high_z_filterset):
        flux_col = f"{filt.instrument_name}.{filt.filt_name}_scattered"
        err_col = f"{filt.instrument_name}.{filt.filt_name}_err"
        depth_col = f"loc_depth_{filt.instrument_name}.{filt.filt_name}"
        tab[flux_col] = np.array(
            [row_flux[1][i], row_flux[2][i]], dtype=np.float64
        )
        tab[err_col] = np.full(2, sigma_Jy, dtype=np.float64)
        tab[depth_col] = np.full(
            2, high_z_gal_depth.value, dtype=np.float64
        )

    cat_path = (
        f"{galfind.config['DEFAULT']['GALFIND_WORK']}/Catalogues/"
        "synthetic/v0/synthetic_v0.fits"
    )
    os.makedirs(os.path.dirname(cat_path), exist_ok=True)
    tab.write(cat_path, overwrite=True)
    return cat_path


@pytest.fixture(scope="session")
def synthetic_eazy_cat(synthetic_eazy_cat_path, high_z_filterset):
    cat_creator = Catalogue_Creator(
        survey="synthetic",
        version="v0",
        cat_path=synthetic_eazy_cat_path,
        filterset=high_z_filterset,
        aper_diams=test_aper_diams,
        load_phot_func=load_phot,
        get_phot_labels=scattered_phot_labels,
        load_phot_kwargs={
            "ZP": u.Jy.to(u.ABmag),
            "incl_errs": True,
            "min_flux_pc_err": 10.0,
        },
        load_mask_func=None,
        load_depth_func=load_galfind_depths,
        get_depth_labels=scattered_depth_labels,
        load_selection_func=None,
        apply_gal_instr_mask=False,
        simulated=True,
    )
    return cat_creator()


# @pytest.fixture(autouse = True, scope = "session")
# def galfind_work_dir(tmp_path_factory):
#     tmp = tmp_path_factory.mktemp("session_workdir")
#     config.set("DEFAULT", "GALFIND_WORK", str(tmp))
#     # monkeypatch.("GALFIND_WORK_DIR", str(tmpdir))


@pytest.fixture(scope="session")
def data_dir_nircam(survey, version, nircam_pix_scale):
    return Data._get_data_dir(
        survey,
        version,
        pix_scale=nircam_pix_scale,
        instrument=NIRCam(),
        data_dir=galfind.config["DEFAULT"]["GALFIND_DATA"],
    )


@pytest.fixture(scope="session")
def forced_phot_stacked_band_data_from_arr(
    survey, version, data_dir_nircam, aper_diams, test_forced_phot_band
):
    band_data_arr = [
        Band_Data(
            filt=Filter.from_SVO("JWST", "NIRCam", band_name),
            survey=survey,
            version=version,
            im_path=f"{data_dir_nircam}/{band_name}_{survey}.fits",
            im_ext=1,
            rms_err_path=f"{data_dir_nircam}/{band_name}_{survey}.fits",
            rms_err_ext=3,
            rms_err_ext_name="RMS_ERR",
            wht_path=f"{data_dir_nircam}/{band_name}_{survey}.fits",
            wht_ext=4,
            aper_diams=aper_diams,
        )
        for band_name in test_forced_phot_band
    ]
    return Stacked_Band_Data.from_band_data_arr(band_data_arr)


@pytest.fixture(scope="session")
def data(
    survey,
    version,
    instrument_names,
    aper_diams,
    forced_phot_stacked_band_data_from_arr,
):
    from astroquery.gaia import Gaia

    mp = pytest.MonkeyPatch()
    mp.setattr(Gaia, "launch_job_async", _mock_gaia_launch_job_async)
    try:
        return Data.pipeline(
            survey,
            version,
            instrument_names,
            aper_diams=aper_diams,
            forced_phot_band=forced_phot_stacked_band_data_from_arr,
            im_str=["test"],
        )
    finally:
        mp.undo()


@pytest.fixture(scope="session")
def cat_creator_id_cropped(data):
    id_selector = ID_Selector([23])
    cat_creator = Catalogue_Creator.from_data(data, crops=id_selector)
    return cat_creator


@pytest.fixture(scope="session")
def cat(data):
    return Catalogue.from_data(data)


@pytest.fixture(scope="session")
def gal(cat):
    return cat[0]


@pytest.fixture(scope="session")
def cat_eazy_loaded(data, eazy_fsps_larson_sed_fitter, aper_diams):
    # load catalogue from data
    cat = Catalogue.from_data(data)
    # load/run SED fitting; overwrite=True forces a fresh fit rather than
    # trusting a cached .h5 fit-result file that may be stale (e.g. written
    # by an older template config with a different tempfilt_data shape) --
    # this fit is cheap (~0s) on the tiny synthetic test catalogue
    eazy_fsps_larson_sed_fitter(
        cat, aper_diams[0], update=True, overwrite=True
    )
    return cat


@pytest.fixture(scope="session")
def gal_eazy_loaded(cat_eazy_loaded):
    return cat_eazy_loaded[0]


@pytest.fixture(scope="session")
def cat_lephare_loaded(data, lephare_sed_fitter, aper_diams):
    # load catalogue from data
    cat = Catalogue.from_data(data)
    # load/run SED fitting; overwrite=True forces a fresh fit rather than
    # trusting a possibly-stale cached fit-result file (see cat_eazy_loaded)
    lephare_sed_fitter(cat, aper_diams[0], update=True, overwrite=True)
    return cat


@pytest.fixture(scope="session")
def gal_lephare_loaded(cat_lephare_loaded):
    return cat_lephare_loaded[0]


@pytest.fixture(scope="session")
def cat_eazy_sex_params_loaded(cat_eazy_loaded):
    cat_eazy_loaded_ = deepcopy(cat_eazy_loaded)
    cat_eazy_loaded_.load_sextractor_params()
    # cat_eazy_loaded_.load_sextractor_ext_src_corrs()
    return cat_eazy_loaded_


@pytest.fixture(scope="session")
def gal_eazy_sex_params_loaded(cat_eazy_sex_params_loaded):
    return cat_eazy_sex_params_loaded[0]


@pytest.fixture(scope="session")
def cat_lephare_eazy_loaded(
    cat_lephare_loaded,
    eazy_fsps_larson_sed_fitter,
    aper_diams,
):
    # load/run EAZY SED fitting; overwrite=True as in cat_eazy_loaded
    eazy_fsps_larson_sed_fitter(
        cat_lephare_loaded, aper_diams[0], update=True, overwrite=True
    )
    return cat_lephare_loaded


@pytest.fixture(scope="session")
def gal_lephare_eazy_loaded(cat_lephare_eazy_loaded):
    return cat_lephare_eazy_loaded[0]


@pytest.fixture(scope="session")
def custom_lephare_sed_fitter():
    # TODO: SED_fit_params here doesn't actually fold in anywhere
    SED_fit_params = {"GAL_TEMPLATES": "BC03_Chabrier2003_Zm42m62"}
    acs_wfc_nircam_medwide = Multiple_Filter.from_instruments(
        ["ACS_WFC", "NIRCam"], keep_suffix=["M", "W", "LP"]
    )
    LePhare_fitter = LePhare(
        SED_fit_params,
        filterset=acs_wfc_nircam_medwide,
        gal_lib_name="BC03_ACS_WFC+NIRCam_MedWide_HZ_Dusty",
        star_lib_name="STAR+BD_ACS_WFC+NIRCam_MedWide",
        qso_lib_name="QSO_MARA_ACS_WFC+NIRCam_MedWide_HZ_Dusty",
    )
    return LePhare_fitter


@pytest.fixture(scope="session")
def cat_custom_lephare_loaded(cat, custom_lephare_sed_fitter, aper_diams):
    # load custom lephare; overwrite=True as in cat_eazy_loaded
    custom_lephare_sed_fitter(
        cat,
        aper_diams[0],
        load_PDFs=True,
        load_SEDs=True,
        update=True,
        overwrite=True,
    )
    return cat


@pytest.fixture(scope="session")
def gal_custom_lephare_loaded(cat_custom_lephare_loaded):
    return cat_custom_lephare_loaded[0]


@pytest.fixture(scope="session")
def cat_custom_lephare_eazy_loaded(
    cat_custom_lephare_loaded,
    eazy_fsps_larson_sed_fitter,
    aper_diams,
):
    # load/run EAZY SED fitting; overwrite=True as in cat_eazy_loaded
    eazy_fsps_larson_sed_fitter(
        cat_custom_lephare_loaded,
        aper_diams[0],
        update=True,
        overwrite=True,
    )
    return cat_custom_lephare_loaded


@pytest.fixture(scope="session")
def gal_custom_lephare_eazy_loaded(cat_custom_lephare_eazy_loaded):
    return cat_custom_lephare_eazy_loaded[0]


@pytest.fixture(scope="session")
def phot_rest(
    gal_eazy_loaded,
    eazy_fsps_larson_sed_fitter,
    aper_diams,
):
    return (
        gal_eazy_loaded.aper_phot[aper_diams[0]]
        .SED_results[eazy_fsps_larson_sed_fitter.label]
        .phot_rest
    )


@pytest.fixture(scope="session")
def phot_rest_sex_params_loaded(
    gal_eazy_sex_params_loaded,
    eazy_fsps_larson_sed_fitter,
    aper_diams,
):
    return (
        gal_eazy_sex_params_loaded.aper_phot[aper_diams[0]]
        .SED_results[eazy_fsps_larson_sed_fitter.label]
        .phot_rest
    )


@pytest.fixture(scope="session")
def blank_phot_rest(blank_multi_filt):
    return Photometry_rest(
        blank_multi_filt,
        flux=[] * u.uJy,
        flux_errs=[] * u.uJy,
        depths=[] * u.ABmag,
        z=10.0,
    )

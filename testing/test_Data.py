import inspect
import os
from copy import copy, deepcopy

import numpy as np
import pytest
from astropy.table import Table

# anchor to this file's own location, not the process cwd, so
# GALFIND_WORK/GALFIND_DATA always land under <repo_root>/testing/
# test_work regardless of the directory tests are invoked from
os.environ["GALFIND_CONFIG_DIR"] = os.path.dirname(os.path.abspath(__file__))
os.environ["GALFIND_CONFIG_NAME"] = "test_galfind_config.ini"

from galfind.imaging import Data, Filter
from galfind.imaging.Data import Band_Data, Stacked_Band_Data
from galfind.photometry import SExtractor
from galfind.utils import Masking


@pytest.fixture(scope="module")
def f444w_band_data(f444w, survey, version, data_dir_nircam, aper_diams):
    fits_path = f"{data_dir_nircam}/{f444w.filt_name}_{survey}.fits"
    return Band_Data(
        filt=f444w,
        survey=survey,
        version=version,
        im_path=fits_path,
        rms_err_path=fits_path,
        wht_path=fits_path,
        im_ext=1,
        rms_err_ext=3,
        wht_ext=4,
        rms_err_ext_name="RMS_ERR",
        wht_ext_name="WHT",
        aper_diams=aper_diams,
    )


def _mock_gaia_launch_job_async(*args, **kwargs):
    # Stands in for a live astroquery.gaia.Gaia.launch_job_async call,
    # returning an empty result table (no Gaia stars in the footprint)
    # so masking tests don't depend on the live Gaia archive.
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


@pytest.fixture(scope="module")
def f444w_band_data_masked(f444w_band_data):
    from astroquery.gaia import Gaia

    band_data = deepcopy(f444w_band_data)
    mp = pytest.MonkeyPatch()
    mp.setattr(Gaia, "launch_job_async", _mock_gaia_launch_job_async)
    try:
        band_data.mask(overwrite=True)
    finally:
        mp.undo()
    return band_data


@pytest.fixture(scope="module")
def f444w_band_data_segmented(f444w_band_data):
    band_data = deepcopy(f444w_band_data)
    band_data.segment(overwrite=True)
    return band_data


@pytest.fixture(scope="module")
def local_forced_phot_stacked_band_data_from_arr(
    survey, version, data_dir_nircam, test_forced_phot_band
):
    band_data_arr = [
        Band_Data(
            filt=Filter.from_SVO("JWST", "NIRCam", filt_name),
            survey=survey,
            version=version,
            im_path=f"{data_dir_nircam}/{filt_name}_{survey}.fits",
            im_ext=1,
        )
        for filt_name in test_forced_phot_band
    ]
    return Stacked_Band_Data.from_band_data_arr(band_data_arr)


# @pytest.fixture(scope="module")
# def stacked_band_data(survey, version):
#     pass


@pytest.fixture(scope="module", params=[False, True])
def output_hdr(request):
    return request.param


class TestBandDataLoad:
    @pytest.fixture(
        scope="class",
        params=[
            [],
            ["rms_err"],
            ["wht"],
            ["rms_err", "wht"],
        ],
    )
    def band_data_to_load(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def f444w_band_data_cls(
        self, f444w, survey, version, data_dir_nircam, band_data_to_load
    ):
        kwargs = {}
        fits_path = f"{data_dir_nircam}/{f444w.filt_name}_{survey}.fits"
        if "rms_err" in band_data_to_load:
            kwargs = {
                **kwargs,
                "rms_err_path": fits_path,
                "rms_err_ext": 3,
                "rms_err_ext_name": "RMS_ERR",
            }
        if "wht" in band_data_to_load:
            kwargs = {
                **kwargs,
                "wht_path": fits_path,
                "wht_ext": 4,
                "wht_ext_name": "WHT",
            }
        return Band_Data(
            filt=f444w,
            survey=survey,
            version=version,
            im_path=fits_path,
            im_ext=1,
            **kwargs,
        )

    @pytest.fixture(scope="class", params=[False, True])
    def return_hdul(self, request):
        return request.param

    def test_f444w_band_data(self, f444w_band_data_cls):
        assert isinstance(f444w_band_data_cls, Band_Data)

    def test_f444w_band_data_load(self, f444w_band_data_cls, return_hdul):
        output = f444w_band_data_cls.load_im(return_hdul)
        if return_hdul:
            im_data, im_header, im_hdul = output
            assert im_hdul is not None
        else:
            im_data, im_header = output
        assert len(im_data.shape) == 2
        assert im_header is not None

    def test_f444w_band_data_load_rms_err(
        self, f444w_band_data_cls, output_hdr, return_hdul
    ):
        if f444w_band_data_cls.rms_err_path is not None:
            output = f444w_band_data_cls.load_rms_err(output_hdr, return_hdul)
            if output_hdr:
                if return_hdul:
                    rms_err_data, rms_err_header, rms_err_hdul = output
                    assert rms_err_hdul is not None
                else:
                    rms_err_data, rms_err_header = output
                assert rms_err_header is not None
            else:
                if return_hdul:
                    rms_err_data, rms_err_hdul = output
                    assert rms_err_hdul is not None
                else:
                    rms_err_data = output
            assert len(rms_err_data.shape) == 2
            assert rms_err_data.shape == f444w_band_data_cls.data_shape

    def test_f444w_band_data_load_wht(
        self, f444w_band_data_cls, output_hdr, return_hdul
    ):
        if f444w_band_data_cls.wht_path is not None:
            output = f444w_band_data_cls.load_wht(output_hdr, return_hdul)
            if output_hdr:
                if return_hdul:
                    wht_data, wht_header, wht_hdul = output
                    assert wht_hdul is not None
                else:
                    wht_data, wht_header = output
                assert wht_header is not None
            else:
                if return_hdul:
                    wht_data, wht_hdul = output
                    assert wht_hdul is not None
                else:
                    wht_data = output
            assert len(wht_data.shape) == 2
            assert wht_data.shape == f444w_band_data_cls.data_shape

    def test_f444w_band_data_aper_diams(
        self, f444w, survey, version, data_dir_nircam, aper_diams
    ):
        band_data_blank = Band_Data(
            filt=f444w,
            survey=survey,
            version=version,
            im_path=f"{data_dir_nircam}/{f444w.filt_name}_{survey}.fits",
            im_ext=1,
            aper_diams=None,
        )
        assert getattr(band_data_blank, "aper_diams", None) is None
        band_data_aper_diams = Band_Data(
            filt=f444w,
            survey=survey,
            version=version,
            im_path=f"{data_dir_nircam}/{f444w.filt_name}_{survey}.fits",
            im_ext=1,
            aper_diams=aper_diams,
        )
        assert band_data_aper_diams.aper_diams == aper_diams
        # update aper_diams
        band_data_blank.set_aper_diams(aper_diams)
        assert band_data_blank.aper_diams == aper_diams
        band_data_aper_diams.set_aper_diams(aper_diams)
        assert band_data_aper_diams.aper_diams == aper_diams

    def test_invalid_im_path(self, f444w_band_data):
        f444w_band_data_ = deepcopy(f444w_band_data)
        f444w_band_data_.im_path = "invalid/path.fits"
        with pytest.raises(Exception):
            f444w_band_data_.load_im()

    def test_invalid_rms_err_path(self, f444w_band_data):
        f444w_band_data_ = deepcopy(f444w_band_data)
        f444w_band_data_.rms_err_path = "invalid/path.fits"
        rms_err, hdr = f444w_band_data_.load_rms_err(
            output_hdr=True, return_hdul=False
        )
        assert all([output is None for output in [rms_err, hdr]])

    def test_invalid_wht_path(self, f444w_band_data):
        f444w_band_data_ = deepcopy(f444w_band_data)
        f444w_band_data_.wht_path = "invalid/path.fits"
        wht, hdr = f444w_band_data_.load_wht(
            output_hdr=True, return_hdul=False
        )
        assert all([output is None for output in [wht, hdr]])

    def test_load_wcs(self, f444w_band_data):
        wcs = f444w_band_data.load_wcs()
        assert wcs is not None
        wcs_new = f444w_band_data.load_wcs()
        assert wcs_new == wcs


class TestBandDataDunder:
    def test_f444w_band_data_str(self, f444w_band_data):
        print(f444w_band_data)

    def test_f444w_band_data_repr(self, f444w_band_data):
        repr(f444w_band_data)

    def test_f444w_band_data_attr(
        self, f444w_band_data, f444w, survey, version, data_dir_nircam
    ):
        assert f444w_band_data.filt == f444w
        assert f444w_band_data.survey == survey
        assert f444w_band_data.version == version
        # im_path is normalized to an absolute path in __init__ (relative
        # paths break under methods decorated with `run_in_dir`, which
        # change the working directory before running)
        assert f444w_band_data.im_path == os.path.abspath(
            f"{data_dir_nircam}/{f444w.filt_name}_{survey}.fits"
        )
        assert f444w_band_data.im_ext == 1
        assert f444w_band_data.instr_name == "NIRCam"
        assert f444w_band_data.filt_name == "F444W"
        assert f444w_band_data.ZP == f444w.instrument.calc_ZP(f444w_band_data)
        # ensure ZP is within 0.001 mag of expected value
        assert abs(f444w_band_data.ZP - 28.0865) < 0.001

    def test_f444w_band_data_copy(self, f444w_band_data):
        copy_band_data = copy(f444w_band_data)
        copy_band_data is not f444w_band_data
        assert copy_band_data == f444w_band_data
        setattr(copy_band_data, "test_attr", 123)
        assert not hasattr(f444w_band_data, "test_attr")

    def test_f444w_band_data_deepcopy(self, f444w_band_data):
        deepcopy_band_data = deepcopy(f444w_band_data)
        deepcopy_band_data is not f444w_band_data
        assert deepcopy_band_data == f444w_band_data
        setattr(deepcopy_band_data, "test_attr", 123)
        assert not hasattr(f444w_band_data, "test_attr")

    def test_f444w_band_data_eq(self, f444w_band_data):
        deepcopy_band_data = deepcopy(f444w_band_data)
        assert deepcopy_band_data == f444w_band_data
        # im_path is one of the attributes compared by __eq__, so changing
        # it (unlike aper_diams, which set_aper_diams refuses to overwrite
        # once loaded, and which __eq__ does not compare) must break equality
        deepcopy_band_data.im_path = "invalid/path.fits"
        assert deepcopy_band_data != f444w_band_data


class TestStackedBandData:
    def test_local_forced_phot_stacked_band_data_from_arr_init(
        self, local_forced_phot_stacked_band_data_from_arr
    ):
        assert isinstance(
            local_forced_phot_stacked_band_data_from_arr, Stacked_Band_Data
        )

    def test_local_forced_phot_stacked_band_data_from_arr_len(
        self,
        local_forced_phot_stacked_band_data_from_arr,
        test_forced_phot_band,
    ):
        assert len(
            local_forced_phot_stacked_band_data_from_arr.band_data_arr
        ) == len(test_forced_phot_band)


class TestBandDataMask:
    def test_f444w_base_mask(self, f444w_band_data_masked):
        assert hasattr(f444w_band_data_masked, "mask_path")
        assert f444w_band_data_masked.mask_path == (
            Masking.get_mask_path(f444w_band_data_masked)
        )
        assert hasattr(f444w_band_data_masked, "mask_args")
        sig = inspect.signature(f444w_band_data_masked.mask)
        for key in f444w_band_data_masked.mask_args.keys():
            if key == "angle":
                # angle=None (the default) is resolved to a concrete value
                # computed from the image WCS/header by auto_mask, so the
                # stored mask_args value need not match the raw signature
                # default
                continue
            assert f444w_band_data_masked.mask_args[key] == (
                sig.parameters[key].default
            )

    def test_f444w_base_print_diff(
        self, f444w_band_data, f444w_band_data_masked
    ):
        band_data_str = str(f444w_band_data)
        band_data_masked_str = str(f444w_band_data_masked)
        assert band_data_str != band_data_masked_str


class TestBandDataSegmentation:
    # @pytest.fixture(
    #     scope="class",
    #     params = [
    #         ({}, True),
    #     ]
    # )
    # def seg_args_case(self, request):
    #     return request.param

    def test_get_sex_code(self):
        code = SExtractor.get_code()
        assert isinstance(code, str)

    def test_f444w_base_segmentation(self, f444w_band_data_segmented):
        assert isinstance(f444w_band_data_segmented, Band_Data)
        sig = inspect.signature(f444w_band_data_segmented.segment)
        method_name = {
            "sextractor": SExtractor.get_segmentation_path,
        }
        assert sig.parameters["method"].default in method_name.keys()
        seg_path_func = method_name[sig.parameters["method"].default]
        # get_segmentation_path takes the *converted* error map type
        # ("MAP_RMS"/"MAP_WEIGHT"), not the raw "rms_err"/"wht" err_type
        _, _, err_map_type = SExtractor.get_err_map(
            f444w_band_data_segmented, sig.parameters["err_type"].default
        )
        assert f444w_band_data_segmented.seg_path == seg_path_func(
            f444w_band_data_segmented, err_map_type
        )
        assert hasattr(f444w_band_data_segmented, "seg_args")
        for key in f444w_band_data_segmented.seg_args.keys():
            assert f444w_band_data_segmented.seg_args[key] == (
                sig.parameters[key].default
            )

    def test_f444w_base_print_diff(
        self, f444w_band_data, f444w_band_data_segmented
    ):
        band_data_str = str(f444w_band_data)
        band_data_segmented_str = str(f444w_band_data_segmented)
        assert band_data_str != band_data_segmented_str

    def test_load_segmap(self, f444w_band_data_segmented, output_hdr):
        output = f444w_band_data_segmented.load_seg(incl_hdr=output_hdr)
        if output_hdr:
            segmap, segmap_header = output
            assert segmap_header is not None
        else:
            segmap = output
        assert len(segmap.shape) == 2
        assert segmap.shape == f444w_band_data_segmented.data_shape

    def test_invalid_segmap_path(self, f444w_band_data_segmented):
        f444w_band_data_segmented_ = deepcopy(f444w_band_data_segmented)
        f444w_band_data_segmented_.seg_path = "invalid/path.fits"
        with pytest.raises(Exception):
            f444w_band_data_segmented_.load_seg(incl_hdr=True)


class TestBandDataForcedPhotometry:
    @pytest.fixture(scope="class")
    def f444w_base_forced_phot(
        self, f444w_band_data, local_forced_phot_stacked_band_data_from_arr
    ):
        f444w_band_data.perform_forced_phot(
            forced_phot_band=local_forced_phot_stacked_band_data_from_arr,
            overwrite=True,
        )
        return f444w_band_data

    def test_f444w_base_forced_phot(self, f444w_base_forced_phot):
        assert isinstance(f444w_base_forced_phot, Band_Data)
        # ensure band_data has an associated forced_photometry_band


class TestBandDataPSFHomogenize:
    def test_f444w_band_data_psf_homogenize(self, f444w_band_data):
        # f444w_band_data has no PSF loaded (self.psf defaults to None),
        # so psf_homogenize should refuse to run rather than raise
        # NotImplementedError (it is fully implemented)
        with pytest.raises(AssertionError):
            f444w_band_data.psf_homogenize("PSF")


def test_data(data):
    assert isinstance(data, Data)


class TestBandDataDepths:
    # `run_depths` (the actual method; `calc_depths` does not exist) requires
    # forced photometry to already have been run, which in turn requires
    # masking + segmentation, so build a dedicated, independent band_data
    # taken through the full pipeline rather than reusing the shared
    # (unmasked/unsegmented) `f444w_band_data` fixture.
    @pytest.fixture(scope="class")
    def f444w_band_data_depths_ready(
        self,
        f444w_band_data,
        local_forced_phot_stacked_band_data_from_arr,
        aper_diams,
    ):
        from astroquery.gaia import Gaia

        band_data = deepcopy(f444w_band_data)
        mp = pytest.MonkeyPatch()
        mp.setattr(Gaia, "launch_job_async", _mock_gaia_launch_job_async)
        try:
            band_data.mask(overwrite=True)
        finally:
            mp.undo()
        band_data.segment(overwrite=True)
        band_data.perform_forced_phot(
            forced_phot_band=local_forced_phot_stacked_band_data_from_arr,
            overwrite=True,
        )
        band_data.run_depths(plot=False, overwrite=True)
        return band_data

    def test_f444w_area_depth_plot(
        self,
        f444w_band_data_depths_ready,
        aper_diams,
    ):
        f444w_band_data_depths_ready.plot_area_depth(
            aper_diam=aper_diams[0],
            show=False,
        )


# @pytest.fixture(scope="session")
# def data_from_survey_version_psfs(
#     survey: str,
#     version: str,
#     instrument_names: List[str],
# ):
#     return Data.from_survey_version_psfs(
#         survey = survey,
#         version = version,
#         instrument_names = instrument_names,
#         im_str = "test",
#         rms_err_ext_name = "RMS_ERR",
#     )

# def test_data_found(
#     data_from_survey_version, survey, version, instrument_names, test_bands
# ):
#     assert len(data_from_survey_version) == len(test_bands)
#     assert data_from_survey_version.survey == survey
#     assert data_from_survey_version.version == version
#     #assert data_from_survey_version.instrument_names == instrument_names

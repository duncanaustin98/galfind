
from typing import List
import pytest
import astropy.units as u
import inspect
from copy import copy, deepcopy
import os

os.environ["GALFIND_CONFIG_DIR"] = f"{os.getcwd()}/testing"
os.environ["GALFIND_CONFIG_NAME"] = "test_galfind_config.ini"

import galfind
from galfind import (
    Band_Data,
    Data,
    NIRCam,
    SExtractor,
)


@pytest.fixture(scope="module")
def f444w_band_data(f444w, survey, version, data_dir_nircam, aper_diams):
    fits_path = f"{data_dir_nircam}/{f444w.band_name}_{survey}.fits"
    return Band_Data(
        filt = f444w,
        survey = survey,
        version = version,
        im_path = fits_path,
        rms_err_path = fits_path,
        wht_path = fits_path,
        im_ext = 1,
        rms_err_ext = 3,
        wht_ext = 4,
        rms_err_ext_name = "RMS_ERR",
        wht_ext_name = "WHT",
        aper_diams = aper_diams,
    )

@pytest.fixture(scope="module")
def f444w_band_data_masked(f444w_band_data):
    f444w_band_data.mask(overwrite = True)
    return f444w_band_data

@pytest.fixture(scope="module")
def f444w_band_data_segmented(f444w_band_data):
    f444w_band_data.segment(overwrite = True)
    return f444w_band_data

@pytest.fixture(scope="module")
def forced_phot_stacked_band_data_from_arr(survey, version, data_dir_nircam, test_forced_phot_band):
    band_data_arr = [
        Band_Data(
            filt = Filter.from_SVO("JWST", "NIRCam", band_name),
            survey = survey,
            version = version,
            im_path = f"{data_dir_nircam}/{band_name}_{survey}.fits",
            im_ext = 1,
        ) for band_name in test_forced_phot_band
    ]
    return Stacked_Band_Data.from_band_data_arr(band_data_arr)

# @pytest.fixture(scope="module")
# def stacked_band_data(survey, version):
#     pass

@pytest.fixture(
    scope="module",
    params = [False, True]
)
def output_hdr(request):
    return request.param


class TestBandDataLoad:

    @pytest.fixture(
        scope="class",
        params = [
            [],
            ["rms_err"],
            ["wht"],
            ["rms_err", "wht"],
        ]
    )
    def band_data_to_load(self, request):
        return request.param

    @pytest.fixture(scope = "class")
    def f444w_band_data_cls(self, f444w, survey, version, data_dir_nircam, band_data_to_load):
        kwargs = {}
        fits_path = f"{data_dir_nircam}/{f444w.band_name}_{survey}.fits"
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
            filt = f444w,
            survey = survey,
            version = version,
            im_path = fits_path,
            im_ext = 1,
            **kwargs,
        )
    
    @pytest.fixture(
        scope="class",
        params = [False, True]
    )
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
    
    def test_f444w_band_data_load_rms_err(self, f444w_band_data_cls, output_hdr, return_hdul):
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
    
    def test_f444w_band_data_load_wht(self, f444w_band_data_cls, output_hdr, return_hdul):
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
    
    def test_f444w_band_data_aper_diams(self, f444w, survey, version, data_dir_nircam, aper_diams):
        band_data_blank = Band_Data(
            filt = f444w,
            survey = survey,
            version = version,
            im_path = f"{data_dir_nircam}/{f444w.band_name}_{survey}.fits",
            im_ext = 1,
            aper_diams = None
        )
        assert getattr(band_data_blank, "aper_diams", None) is None
        band_data_aper_diams = Band_Data(
            filt = f444w,
            survey = survey,
            version = version,
            im_path = f"{data_dir_nircam}/{f444w.band_name}_{survey}.fits",
            im_ext = 1,
            aper_diams = aper_diams
        )
        assert band_data_aper_diams.aper_diams == aper_diams
        # update aper_diams
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
        rms_err, hdr = f444w_band_data_.load_rms_err(output_hdr = True, return_hdul = False)
        assert all([output is None for output in [rms_err, hdr]])
    
    def test_invalid_wht_path(self, f444w_band_data):
        f444w_band_data_ = deepcopy(f444w_band_data)
        f444w_band_data_.wht_path = "invalid/path.fits"
        wht, hdr = f444w_band_data_.load_wht(output_hdr = True, return_hdul = False)
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
    
    def test_f444w_band_data_attr(self, f444w_band_data, f444w, survey, version, data_dir_nircam):
        assert f444w_band_data.filt == f444w
        assert f444w_band_data.survey == survey
        assert f444w_band_data.version == version
        assert f444w_band_data.im_path == f"{data_dir_nircam}/{f444w.band_name}_{survey}.fits"
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
        assert hasattr(f444w_band_data, "test_attr")
        assert getattr(f444w_band_data, "test_attr") == 123

    def test_f444w_band_data_deepcopy(self, f444w_band_data):
        deepcopy_band_data = deepcopy(f444w_band_data)
        deepcopy_band_data is not f444w_band_data
        assert deepcopy_band_data == f444w_band_data
        setattr(deepcopy_band_data, "test_attr", 123)
        assert not hasattr(f444w_band_data, "test_attr")

    def test_f444w_band_data_eq(self, f444w_band_data, aper_diams):
        deepcopy_band_data = deepcopy(f444w_band_data)
        assert deepcopy_band_data == f444w_band_data
        deepcopy_band_data.set_aper_diams(aper_diams)
        assert deepcopy_band_data != f444w_band_data


class TestStackedBandData:

    def test_forced_phot_stacked_band_data_from_arr_init(
        self,
        forced_phot_stacked_band_data_from_arr
    ):
        assert isinstance(forced_phot_stacked_band_data_from_arr, Stacked_Band_Data)
    
    def test_forced_phot_stacked_band_data_from_arr_len(
        self,
        forced_phot_stacked_band_data_from_arr,
        test_forced_phot_band
    ):
        assert len(forced_phot_stacked_band_data_from_arr.band_data_arr) == len(test_forced_phot_band)


class TestBandDataMask:
    
    def test_f444w_base_mask(self, f444w_band_data_masked):
        assert hasattr(f444w_band_data_masked, "mask_path")
        assert f444w_band_data_masked.mask_path == Masking.get_mask_path(f444w_band_data_masked)
        assert hasattr(f444w_band_data_masked, "mask_args")
        sig = inspect.signature(f444w_band_data_masked.mask)
        for key in self.seg_args.keys():
            assert f444w_band_data_masked.mask_args[key] == sig.parameters[key].default
    
    def test_f444w_base_print_diff(self, f444w_band_data, f444w_band_data_masked):
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
        assert f444w_band_data_segmented.seg_path == method_name[sig.parameters["method"].default]\
            (f444w_band_data_segmented, sig.parameters["err_type"].default)
        assert hasattr(f444w_band_data_segmented, "seg_args")
        for key in self.seg_args.keys():
            assert f444w_band_data_segmented.seg_args[key] == sig.parameters[key].default
    
    def test_f444w_base_print_diff(self, f444w_band_data, f444w_band_data_segmented):
        band_data_str = str(f444w_band_data)
        band_data_segmented_str = str(f444w_band_data_segmented)
        assert band_data_str != band_data_segmented_str
    
    def test_load_segmap(self, f444w_band_data_segmented, output_hdr):
        output = f444w_band_data_segmented.load_segmap(incl_hdr = output_hdr)
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
            f444w_band_data_segmented_.load_segmap(incl_hdr = True)


class TestBandDataForcedPhotometry:

    @pytest.fixture(scope = "class")
    def f444w_base_forced_phot(self, f444w_band_data, forced_phot_stacked_band_data_from_arr):
        f444w_band_data.perform_forced_phot(
            forced_phot_band = forced_phot_stacked_band_data_from_arr,
            overwrite = True
        )
    
    def test_f444w_base_forced_phot(self, f444w_base_forced_phot):
        assert isinstance(f444w_base_forced_phot, Band_Data)
        # ensure band_data has an associated forced_photometry_band


class TestBandDataPSFHomogenize:

    def test_f444w_band_data_psf_homogenize(self, f444w_band_data):
        with pytest.raises(NotImplementedError):
            f444w_band_data.psf_homogenize("PSF")

def test_data(data):
    assert isinstance(data, Data)


class TestBandDataDepths:

    def test_f444w_area_depth_plot(
        self,
        f444w_band_data,
        aper_diams,
    ):
        #f444w_band_data.calc_depths(aper_diams, overwrite = True)
        f444w_band_data.plot_area_depth(
            aper_diam = aper_diams[0],
            show = False,
            #save_path = None,
        )

# @pytest.fixture(scope="session")
# def data_from_survey_version(
#     survey: str,
#     version: str,
#     instrument_names: List[str],
# ):
#     return Data.from_survey_version(
#         survey = survey,
#         version = version,
#         instrument_names = instrument_names,
#         im_str = "test",
#         rms_err_ext_name = "RMS_ERR",
#     )

# def test_data_found(data_from_survey_version, survey, version, instrument_names, test_bands):
#     assert len(data_from_survey_version) == len(test_bands)
#     assert data_from_survey_version.survey == survey
#     assert data_from_survey_version.version == version
#     #assert data_from_survey_version.instrument_names == instrument_names


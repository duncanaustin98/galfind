
import pytest
from copy import copy, deepcopy
import numpy as np
import astropy.units as u
from pathlib import Path
from galfind import Multiple_Filter, Filter, Tophat_Filter, Facility, HST, JWST, ACS_WFC, NIRCam
from galfind.Filter import UVJ, U, V, J

@pytest.fixture(scope = "module")
def forced_phot_multi_filter(test_forced_phot_band):
    return Multiple_Filter([Filter.from_SVO("JWST", "NIRCam", band) for band in test_forced_phot_band])

# def test_multiple_filter_getitem_list(multiple_filter_all, multiple_filter_forced_phot_band):
#     multiple_filter_list = multiple_filter_all[test_forced_phot_band]
#     assert multiple_filter == multiple_filter_forced_phot_band

# def test_multiple_filter_getitem_array(multiple_filter_all, multiple_filter_forced_phot_band):
#     multiple_filter_list = multiple_filter_all[np.array(test_forced_phot_band)]
#     assert multiple_filter == multiple_filter_forced_phot_band

# def test_multiple_filter_sub(multiple_filter_nircam, multiple_filter_forced_phot_band):
#     sub_multiple_filter = multiple_filter_nircam - multiple_filter_forced_phot_band
#     for filt in multiple_filter_forced_phot_band:
#         assert filt not in sub_multiple_filter
#     assert len(sub_multiple_filter) == len(multiple_filter_nircam) - len(multiple_filter_forced_phot_band)

# def test_multiple_filter_eq_invalid(multiple_filter_all, multiple_filter_forced_phot_band):
#     assert multiple_filter_all != multiple_filter_forced_phot_band

# def test_multiple_instrument_getitem_filtstr(multiple_filter_nircam, f444w):
#     assert multiple_filter_nircam["F444W"] == f444w

# @pytest.fixture(
#     params = [
#         ("NON_EXISTENT_FILTER", KeyError),
#         (12345, KeyError),
#         (5.5, TypeError),
#         (0.32 * u.arcsec, TypeError),

#     ]
# )
# def invalid_multiple_filter_getitem_case(request):
#     return request.param

# def test_multiple_instrument_getitem_invalid(
#     multiple_filter_nircam,
#     invalid_multiple_filter_getitem_case
# ):
#     key, exception = invalid_multiple_filter_getitem_case
#     with pytest.raises(exception):
#         multiple_filter_nircam[key]

# def test_multiple_instrument_iterable(multiple_filter_nircam):
#     for filter_ in multiple_filter_nircam:
#         assert isinstance(filter_, Filter)

@pytest.fixture(scope = "module")
def multiple_filter_all():
    multiple_filter = Multiple_Filter.from_facilities(
        [facility_ for facility_ in Facility.__subclasses__()]
    )
    return multiple_filter


class TestFilterInstantiation:
    
    @pytest.fixture(
        scope="class",
        params = [
            (["JWST", "NIRCam", "F444W"], True),
            (["HST", "ACS_WFC", "F606W"], True),
            (["HST", "WFC3_IR", "F160W", None, "WFC3"], True),
            (["JWST", "MIRI", "F770W"], True),
            (["Paranal", "VISTA", "Z"], True),
            (["Spitzer", "IRAC", "I1"], True),
            (["JWST", "NON_EXISTENT_INSTRUMENT", "F444W"], Exception),
            (["JWST", "NIRCam", "NON_EXISTENT_FILTER"], Exception),
            (["NON_EXISTENT_FACILITY", "NIRCam", "F444W"], Exception),
        ]
    )
    def from_svo_case(self, request):
        return request.param

    @pytest.fixture(
        scope="class",
        params = [
            ("F444W", True),
            ("JWST/NIRCam/F444W", True),
            ("JWST/NIRCam.F444W", True),
            ("JWST/UNKNOWN_INSTRUMENT/F444W", Exception),
            ("JWST/NIRCam/UNKNOWN_FILTER", Exception),
            ("UNKNOWN_FACILITY/NIRCam/F444W", Exception),
        ]
    )
    def from_filt_name_case(self, request):
        return request.param

    @pytest.fixture(
        scope = "module",
        params = [
            filt for filt in Tophat_Filter.__subclasses__()
        ]
    )
    def tophat_filter(self, request):
        return request.param

    
    def test_from_svo(self, filter):
        assert isinstance(filter, Filter)

    def test_from_svo_case(self, from_svo_case):
        inputs, outcome = from_svo_case
        if outcome != True:
            with pytest.raises(outcome):
                filt = Filter.from_SVO(*inputs)
        else:
            filt = Filter.from_SVO(*inputs)
            assert isinstance(filt, Filter)
    

    def test_from_filt_name_case(self, from_filt_name_case):
        input_, outcome = from_filt_name_case
        if outcome != True:
            with pytest.raises(outcome):
                filt = Filter.from_filt_name(input_)
        else:
            filt = Filter.from_filt_name(input_)
            assert isinstance(filt, Filter)


    def test_tophat_init(self, tophat_filter):
        filt = tophat_filter()
        assert isinstance(filt, Tophat_Filter)

    

class TestFilterDunderMethods:

    @pytest.fixture(
        scope="class",
        params = [
            ("JWST/NIRCam.F356W", False, True),
            ("JWST/NIRCam.F444W", True, True),
            (U, False, True),
            (Tophat_Filter, None, Exception),
            (NIRCam, None, Exception),
            (U(), False, True),
            (12345, None, Exception),
        ]
    )
    def add_to_f444w_case(self, request):
        return request.param


    def test_f444w_name(self, f444w):
        assert f444w.facility_name == "JWST"
        assert f444w.instrument_name == "NIRCam"
        assert f444w.band_name == "F444W"

    def test_str(self, filter):
        print(filter)

    def test_repr(self, filter):
        repr(filter)
    
    def test_copy(self, filter):
        copy_filter = copy(filter)
        assert copy_filter is not filter
        assert copy_filter == filter
        setattr(copy_filter, "test_attr", 123)
        assert hasattr(filter, "test_attr")
        assert filter["test_attr"] == 123
    
    def test_deepcopy(self, filter):
        deepcopy_filter = deepcopy(filter)
        assert deepcopy_filter is not filter
        assert deepcopy_filter == filter
        setattr(deepcopy_filter, "test_attr", 123)
        assert not hasattr(filter, "test_attr")
    
    def test_len(self, filter):
        assert len(filter) == 1

    
    def test_filt_add_to_f444w_case(self, f444w, add_to_f444w_case):
        other, is_same_band, outcome = add_to_f444w_case
        if outcome != True:
            with pytest.raises(outcome):
                multiple_filter = f444w + other
        else:
            multiple_filter = f444w + other
            if is_same_band:
                assert isinstance(multiple_filter, Filter)
                assert len(multiple_filter) == 1
                assert multiple_filter is f444w
            else:
                assert isinstance(multiple_filter, Multiple_Filter)
                assert len(multiple_filter) == 2
                assert f444w in multiple_filter



class TestMultipleFilterInstantiation:

    @pytest.fixture(
        scope="class",
        params = [
            (["NON_EXISTENT_FACILITY"], Exception),
            #([JWST, JWST], Exception),
            #(NIRCam, Exception),
            #(JWST, Exception),
            #([HST, JWST], True),
            #(["HST", JWST], True),
            #([JWST], True),
            #([JWST()], True),
            #(np.array(["JWST", HST()]), True),
        ]
    )
    def facilities_case(self, request):
        return request.param

    @pytest.fixture(
        scope="class",
        params = [
            ("NON_EXISTENT_FACILITY", Exception),
            (0.32 * u.arcsec, Exception),
            (NIRCam, Exception),
        ]
    )
    def facility_case(self, request):
        return request.param

    @pytest.fixture(
        scope="class",
        params = [
            (["NON_EXISTENT_INSTRUMENT"], Exception),
            #([NIRCam, NIRCam], Exception),
            (JWST, Exception),
            #([NIRCam, ACS_WFC], True),
            #([NIRCam()], True),
            #(np.array([NIRCam, ACS_WFC()]), True),
        ]
    )
    def instrument_case(self, request):
        return request.param

    @pytest.fixture(
        scope="class",
        params = [
            ("F444W", True, 1),
            (Filter.from_SVO("JWST", "NIRCam", "F444W"), True, 1),
            (
                Multiple_Filter(
                    [
                        Filter.from_SVO("JWST", "NIRCam", band)
                        for band in ["F277W", "F356W", "F444W"]
                    ]
                ), True, 3
            ),
        ]
    )
    def excl_bands(self, request):
        return request.param

    @pytest.mark.slow
    def test_from_facilities_all(self, multiple_filter_all):
        assert isinstance(multiple_filter_all, Multiple_Filter)
    
    def test_from_facilities(self, facilities_case):
        facilities_input, outcome = facilities_case
        if outcome != True:
            with pytest.raises(outcome):
                Multiple_Filter.from_facilities(facilities_input)
        else:
            multiple_filter = Multiple_Filter.from_facilities(facilities_input)
            assert isinstance(multiple_filter, Multiple_Filter)


    def test_from_facility_cls(self, facility):
        multiple_filter = Multiple_Filter.from_facility(facility)
        assert isinstance(multiple_filter, Multiple_Filter)
    
    def test_from_facility_inst(self, facility_inst):
        multiple_filter = Multiple_Filter.from_facility(facility_inst)
        assert isinstance(multiple_filter, Multiple_Filter)
    
    def test_from_facility(self, facility_case):
        facility_input, outcome = facility_case
        if outcome != True:
            with pytest.raises(outcome):
                Multiple_Filter.from_facility(facility_input)
        else:
            multiple_filter = Multiple_Filter.from_facility(facility_input)
            assert isinstance(multiple_filter, Multiple_Filter)

    
    def test_from_instruments(self, instrument_case):
        instrument, outcome = instrument_case
        if outcome != True:
            with pytest.raises(outcome):
                Multiple_Filter.from_instruments(instrument)
        else:
            multiple_filter = Multiple_Filter.from_instruments(instrument)
            assert isinstance(multiple_filter, Multiple_Filter)


    def test_from_instrument_cls(self, instrument):
        multiple_filter = Multiple_Filter.from_instruments(instrument)
        assert isinstance(multiple_filter, Multiple_Filter)

    def test_from_instrument_inst(self, instrument_inst):
        multiple_filter = Multiple_Filter.from_instruments(instrument_inst)
        assert isinstance(multiple_filter, Multiple_Filter)

    def test_from_instrument_excl_bands(self, nircam_multi_filter, instrument_inst, excl_bands):
        excl_bands_input, outcome, excl_bands_len = excl_bands
        if outcome != True:
            with pytest.raises(outcome):
                Multiple_Filter.from_instrument(
                    instrument_inst,
                    excl_bands = excl_bands_input
                )
        else:
            multiple_filter = Multiple_Filter.from_instrument(
                instrument_inst,
                excl_bands = excl_bands_input
            )
            assert isinstance(multiple_filter, Multiple_Filter)
            assert len(multiple_filter) == len(nircam_multi_filter) - excl_bands_len

    
    def test_uvj_multiple_filter(self):
        multiple_filter_uvj = Multiple_Filter([U(), V(), J()])
        uvj = UVJ()
        assert len(multiple_filter_uvj) == len(uvj) == 3
        assert multiple_filter_uvj == uvj


class TestMultipleFilterDunderMethods:
    
    @pytest.fixture(
        scope="class",
        params = [
            ("JWST/NIRCam.F356W", True, True),
            ("JWST/NIRCam.F444W", True, True),
            (U, False, True),
            (Tophat_Filter, None, Exception),
            (NIRCam, None, Exception),
            (U(), False, True),
            (12345, None, Exception),
        ]
    )
    def add_sub_nircam_case(self, request):
        return request.param

    def test_filt_add_to_nircam_case(self, nircam_multi_filter, add_sub_nircam_case):
        other, in_nircam, outcome = add_sub_nircam_case
        if outcome != True:
            with pytest.raises(outcome):
                multiple_filter = nircam_multi_filter + other
        else:
            multiple_filter = nircam_multi_filter + other
            assert isinstance(multiple_filter, Multiple_Filter)
            if in_nircam:
                assert len(multiple_filter) == len(nircam_multi_filter)
                assert multiple_filter == nircam_multi_filter
                assert other not in multiple_filter
            else:
                assert len(multiple_filter) == len(nircam_multi_filter) + 1
                assert other in multiple_filter
    
    def test_filt_sub_from_nircam_case(self, nircam_multi_filter, add_sub_nircam_case):
        other, in_nircam, outcome = add_sub_nircam_case
        if outcome != True:
            with pytest.raises(outcome):
                multiple_filter = nircam_multi_filter - other
        else:
            multiple_filter = nircam_multi_filter - other
            assert isinstance(multiple_filter, Multiple_Filter)
            if in_nircam:
                assert len(multiple_filter) == len(nircam_multi_filter) - 1
                assert all(
                    filt != other
                    for filt in multiple_filter
                )
            else:
                assert len(multiple_filter) == len(nircam_multi_filter)
                assert multiple_filter == nircam_multi_filter
    
    def test_str(self, nircam_multi_filter):
        print(nircam_multi_filter)
    
    def test_repr(self, nircam_multi_filter):
        repr(nircam_multi_filter)
    
    def test_copy(self, nircam_multi_filter):
        copy_nircam_multi_filter = copy(nircam_multi_filter)
        assert copy_nircam_multi_filter is not nircam_multi_filter
        assert copy_nircam_multi_filter == nircam_multi_filter
        setattr(copy_nircam_multi_filter, "test_attr", 123)
        assert hasattr(nircam_multi_filter, "test_attr")
        assert nircam_multi_filter["test_attr"] == 123

    def test_deepcopy(self, nircam_multi_filter):
        deepcopy_nircam_multi_filter = deepcopy(nircam_multi_filter)
        assert deepcopy_nircam_multi_filter is not nircam_multi_filter
        assert deepcopy_nircam_multi_filter == nircam_multi_filter
        setattr(deepcopy_nircam_multi_filter, "test_attr", 123)
        assert not hasattr(nircam_multi_filter, "test_attr")
    
    def test_len(self, nircam_multi_filter):
        assert len(nircam_multi_filter) == 27
    
    def test_multiple_filter_is_iterable(self, nircam_multi_filter):
        iterator = iter(nircam_multi_filter)
        first = next(iterator)
        assert isinstance(first, Filter)
        assert first is not None
        for filter_ in nircam_multi_filter:
            assert isinstance(filter_, Filter)
        assert len(nircam_multi_filter) == len(list(nircam_multi_filter))


class TestPlotting:

    @pytest.fixture(scope="class")
    def plot_dir(self, tmp_path_factory):
        return tmp_path_factory.mktemp("plots")

    @pytest.fixture(scope="class")
    def ax(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize = (8,6))
        yield ax
        fig.clf()

    @pytest.fixture(
        scope="class",
        params = [
            {"annotate": True, "save": True, "show": True, "fmt": "png"},
        ]
    )
    def plot_args(self, request):
        return request.param

    def test_f444w_plot(self, ax, f444w, plot_dir, plot_args):
        f444w.plot(
            ax,
            save_dir = str(plot_dir),
            save_name = "f444w_plot",
            **plot_args
        )
        path = plot_dir / Path(f"f444w_plot.{plot_args['fmt']}")
        assert path.is_file()
    
    def test_nircam_multi_filter_plot(self, ax, nircam_multi_filter, plot_dir, plot_args):
        nircam_multi_filter.plot(
            ax,
            save_dir = str(plot_dir),
            save_name = "nircam_multi_filter_plot",
            **plot_args
        )
        path = plot_dir / Path(f"nircam_multi_filter_plot.{plot_args['fmt']}")
        assert path.is_file()


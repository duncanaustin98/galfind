
import pytest
import itertools
import json
import numpy as np
import astropy.units as u

from galfind import config
from galfind import Filter, ACS_WFC, WFC3_IR, NIRCam, MIRI

# Expected instrument attributes
expected_instr_facilities = {"ACS_WFC": "HST", "WFC3_IR": "HST", "NIRCam": "JWST", "MIRI": "JWST"}

ACS_WFC_bands = ['FR388N', 'FR423N', 'F435W', 'FR459M', 'FR462N', 'F475W', 'F502N',
       'FR505N', 'F555W', 'FR551N', 'F550M', 'FR601N', 'F606W', 'F625W',
       'FR647M', 'FR656N', 'F658N', 'F660N', 'FR716N', 'POL_UV', 'POL_V',
       'G800L', 'F775W', 'FR782N', 'F814W', 'FR853N', 'F892N', 'FR914M',
       'F850LP', 'FR931N', 'FR1016N']
WFC3_IR_bands = ['F098M', 'G102', 'F105W', 'F110W', 'F125W', 'F126N', 'F127M',
       'F128N', 'F130N', 'F132N', 'F139M', 'F140W', 'G141', 'F153M',
       'F160W', 'F164N', 'F167N']
NIRCam_bands = ['F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F162M', 'F164N',
       'F150W2', 'F182M', 'F187N', 'F200W', 'F210M', 'F212N', 'F250M',
       'F277W', 'F300M', 'F323N', 'F322W2', 'F335M', 'F356W', 'F360M',
       'F405N', 'F410M', 'F430M', 'F444W', 'F460M', 'F466N', 'F470N',
       'F480M']
MIRI_bands = ['F560W', 'F770W', 'F1000W', 'F1065C', 'F1140C', 'F1130W', 'F1280W',
       'F1500W', 'F1550C', 'F1800W', 'F2100W', 'F2300C', 'F2550W']
expected_instr_bands = { \
    "ACS_WFC": ACS_WFC_bands, \
    "WFC3_IR": WFC3_IR_bands, \
    "NIRCam":  NIRCam_bands, \
    "MIRI":    MIRI_bands
    }

instr_names = json.loads(config.get("Other", "INSTRUMENT_NAMES"))

# Ensure that the expected instrument attributes have been copied correctly
def test_expected_instr():
    assert all(key in expected_instr_facilities.keys() for key in expected_instr_bands.keys() if key in instr_names)
    assert all(key in expected_instr_bands.keys() for key in expected_instr_facilities.keys() if key in instr_names)
    assert all(len(values) > 0 for (key, values) in expected_instr_bands.items() if key in instr_names)

all_filters = [(expected_instr_facilities[instr_name], instr_name, band) \
    for instr_name in instr_names for band in expected_instr_bands[instr_name]]
# choose which filters to skip
filters_to_test = [pytest.param(_all_filters, marks = pytest.mark.skip) if _all_filters[2] \
    not in ["F814W", "F090W"] else _all_filters for _all_filters in all_filters]
@pytest.fixture(scope = "module", params = filters_to_test, \
    ids = lambda filt_tuple: f"{filt_tuple[0]}/{filt_tuple[1]}.{filt_tuple[2]}")
def filter(request):
    return Filter.from_SVO(request.param[0], request.param[1], request.param[2])

@pytest.fixture(scope = "module", params = instr_names)
def instrument(request):
    return globals()[request.param]()

def NIRCam_excl_bands_ids(excl_bands):
    instr_name = "NIRCam"
    if excl_bands == []:
        return instr_name
    elif len(excl_bands) == len(expected_instr_bands["NIRCam"]):
        return f"{instr_name},excl_bands=ALL"
    else:
        return f"{instr_name},excl_bands={'+'.join(excl_bands)}"

@pytest.fixture(scope = "module", params = [[], ["F090W"], \
    expected_instr_bands["NIRCam"]], ids = NIRCam_excl_bands_ids)
def NIRCam_instrument(request):
    return NIRCam(excl_bands = request.param)

# Test Instrument.__len__
def test_instr_len(instrument):
    assert len(instrument) == len(expected_instr_bands[instrument.name])

# Test whether the loaded instrument has all expected bands and facility
def test_expected_SVO_instr(instrument):
    # ensure facility is correct
    assert instrument.facility == expected_instr_facilities[instrument.name]
    # ensure the correct bands are loaded
    assert all(instr_band == expected_band for instr_band, expected_band \
        in zip(instrument.band_names, expected_instr_bands[instrument.name]))

# Test whether bands in instrument loaded from SVO is the same as loading individually
def test_instr_filters_match(instrument):
    assert all(Filter.from_SVO(instrument.facility, instrument.name, \
        band.band_name) == band for band in instrument)

# test that excl_bands works when instantiating objects

# Tests which include Combined_Instrument instruments ------

combined_instr_names = [list(chain) for chain in itertools.chain.from_iterable \
    (itertools.permutations(instr_names, i + 1) for i in range(len(instr_names)))]
@pytest.fixture(scope = "module", params = combined_instr_names, ids = lambda names: "+".join(names))
def combined_instrument_arr(request):
    return [globals()[instr_name]() for instr_name in request.param]

@pytest.fixture(scope = "module")
def combined_instrument(combined_instrument_arr):
    for i, _instrument in enumerate(combined_instrument_arr):
        if i == 0:
            combined_instrument = _instrument
        else:
            combined_instrument += _instrument
    return combined_instrument

# Test Instrument.__add__ where other is a filter
def test_instr_filt_add(instrument, filter):
    if filter.band_name in instrument.band_names:
        with pytest.warns(UserWarning):
            new_instrument = instrument + filter
        assert new_instrument == instrument
    else:
        new_instrument = instrument + filter
        assert len(new_instrument) == len(instrument) + 1
        assert len(new_instrument.band_names) == len(instrument.band_names) + 1
        # ensure the correct filters are present, sorted blue -> red
        new_band_names = [band.band_name for band in sorted(np.concatenate \
            ([np.array([band for band in instrument]), np.array([filter])]), \
            key = lambda band: band.WavelengthCen.to(u.AA).value)]
        assert all(band_name == expected_band_name for band_name, expected_band_name in \
            zip(new_instrument.band_names, new_band_names))

# Test Instrument.__add__ where other is an instrument
def test_instr_instr_add(combined_instrument, NIRCam_instrument):
    # determine which bands require adding
    bands_to_add = np.array([band for band in NIRCam_instrument \
        if band.band_name not in combined_instrument.band_names])
    if len(bands_to_add) == 0:
        # expect warning as you cannot add these together
        with pytest.warns(UserWarning):
            new_instrument = combined_instrument + NIRCam_instrument
        assert new_instrument == combined_instrument
    else: # there is at least one band that needs to be added
        if len(bands_to_add) > 0 and len(bands_to_add) < len(NIRCam_instrument):
            # expect warning
            with pytest.warns(UserWarning):
                new_instrument = combined_instrument + NIRCam_instrument
        else:
            new_instrument = combined_instrument + NIRCam_instrument
        # ensure Instrument.__add__ is commutative when there are bands to add
        assert new_instrument == NIRCam_instrument + combined_instrument
        # ensure new instrument length is appropriate
        assert len(new_instrument) == len(bands_to_add) + len(combined_instrument)
        assert len(new_instrument.band_names) == len(bands_to_add) + len(combined_instrument.band_names)
        # ensure the correct filters are present, sorted blue -> red
        new_band_names = [band.band_name for band in sorted(np.concatenate \
            ([np.array([band for band in combined_instrument]), bands_to_add]), \
            key = lambda band: band.WavelengthCen.to(u.AA).value)]
        assert all(band_name == expected_band_name for band_name, expected_band_name in \
            zip(new_instrument.band_names, new_band_names))

# test removing included band(s) from instrument
# test removing band(s) not included in instrument
# test removing full instrument from combined instrument
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 11:28:44 2023

@author: u92876da
"""

# Instrument_test.py

import unittest
from astropy.table import Table

from galfind import Instrument, NIRCam, ACS_WFC, WFC3_IR, Combined_Instrument

class CombinedInstrumentTest(unittest.TestCase):
    
    def test__add__(self):
        combined_instrument = NIRCam() + ACS_WFC() + WFC3_IR()
        self.assertEqual(combined_instrument.name, "ACS_WFC+WFC3_IR+NIRCam", "Adding instrument names not performed appropriately")
        self.assertEqual(combined_instrument.telescope, "HST+JWST", "Adding telescopes not performed appropriately")

    def test_from_name(self):
        combined_instrument = Combined_Instrument.combined_instrument_from_name("NIRCam+ACS_WFC+WFC3_IR")
        self.assertEqual(combined_instrument.name, "ACS_WFC+WFC3_IR+NIRCam", "Creating instrument names not performed appropriately")
        self.assertEqual(combined_instrument.telescope, "HST+JWST", "Creating telescopes not performed appropriately")

    def test_load_instrument_filter_profiles_from_SVO(self):
        # load in combined instrument
        combined_instrument = NIRCam() + ACS_WFC() + WFC3_IR()
        combined_instrument.load_instrument_filter_profiles(from_SVO = True)
        # ensure that the filter_profile dict is filled with all appropriate bands
        self.assertCountEqual(list(combined_instrument.filter_profiles.keys()), combined_instrument.band_names, "Not loaded instrument filter profiles for each band")
        # ensure that each Table in filter_profile dict has appropriate column names and lengths
        for band in combined_instrument.band_names:
            self.assertEqual(len(combined_instrument.filter_profiles[band]["Wavelength"]), len(combined_instrument.filter_profiles[band]["Transmission"]))
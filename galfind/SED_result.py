#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:50:27 2023

@author: austind
"""

# SED_result.py
import warnings
import numpy as np
import astropy.units as u
from astropy.table import Table
from glob import glob
import tqdm
import h5py

from .Photometry import Photometry, Multiple_Photometry
from .Photometry_rest import Photometry_rest
from .PDF import PDF, Redshift_PDF
from . import useful_funcs_austind as funcs
from . import config, galfind_logger

class SED_result:
    
    def __init__(self, SED_fit_params, phot, properties, property_errs, property_PDFs, SED, rest_UV_wav_lims = [1268., 2580.] * u.Angstrom):
        self.SED_fit_params = SED_fit_params
        #[setattr(key, value) for key, value in SED_fit_params.items()]
        [setattr(key, value) for key, value in properties.items()]
        [setattr(f"{key}_l1", value[0]) for key, value in property_errs.items()]
        [setattr(f"{key}_u1", value[1]) for key, value in property_errs.items()]
        [setattr(f"{key}_PDF", value) for key, value in property_PDFs.items()]
        self.phot_rest = Photometry_rest.from_phot(phot, self.z, rest_UV_wav_lims = rest_UV_wav_lims)
        self.SED = SED

    def __str__(self, print_rest_phot = False):
        line_sep = "*" * 40 + "\n"
        band_sep = "-" * 10 + "\n"
        output_str = line_sep
        output_str += "SED FITTING RESULT:\n"
        output_str += band_sep
        output_str += f"CODE: {self.code.code_name}\n"
        output_str += f"TEMPLATES: {self.templates}\n"
        if self.lowz_zmax == None:
            output_str += "LOW Z RUN: False\n"
        else:
            output_str += "LOW Z RUN: True\n"
            output_str += f"Z MAX: {self.lowz_zmax}\n"
        output_str += band_sep
        output_str += f"PHOTO-Z = {self.z}\n"
        output_str += f"CHI-SQ = {self.chi_sq}\n"
        for key, value in self.properties:
            output_str += f"{key.upper()} = {value}\n"
        for phot_rest in self.phot_rest.values():
            output_str += str(phot_rest)
        output_str += line_sep
        return output_str
    
    @classmethod
    def from_gal(cls, gal, SED_fit_params, rest_UV_wav_lims = [1268., 2580.] * u.Angstrom):
        assert("code" in SED_fit_params.keys())
        fits_cat_row = gal.open_cat_row() # row of fits catalogue
        # extract best fitting properties from galaxy given the SED_fit_params
        properties = {gal_property: float(SED_fit_params["code"].\
            get_gal_property(fits_cat_row, gal_property, SED_fit_params)) \
            for gal_property in SED_fit_params["code"].galaxy_property_dict.keys()}
        # extract best fitting errors from galaxy given the SED_fit_params
        property_errs = {gal_property: float(SED_fit_params["code"].\
            get_gal_property_errs(fits_cat_row, gal_property, SED_fit_params)) \
            for gal_property in SED_fit_params["code"].galaxy_property_dict.keys()}
        # create property PDFs from galaxy given the SED_fit_params
        property_PDFs = {}
        # load SED from galaxy given the SED_fit_params
        SED = None
        return cls(SED_fit_params, gal.phot, properties, property_errs, property_PDFs, SED)
    

class Galaxy_SED_results:
    
    def __init__(self, SED_fit_params_arr, SED_result_arr):
        self.SED_results = {SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params): \
            SED_result for SED_fit_params, SED_result in zip(SED_fit_params_arr, SED_result_arr)}

    def __len__(self):
        return len([True for (code_names, templates_lowz_zmax_results) in self.SED_results.items() \
            for (templates, lowz_zmax_results) in templates_lowz_zmax_results.items() \
            for (lowz_zmax, results) in lowz_zmax_results.items()])
    
    @classmethod
    def from_gal(cls, gal, SED_fit_params_arr, rest_UV_wav_lims = [1268., 2580.] * u.Angstrom):
        return cls(SED_fit_params_arr, [SED_result.from_gal(gal, SED_fit_params, \
            rest_UV_wav_lims = rest_UV_wav_lims) for SED_fit_params in SED_fit_params_arr])

    @classmethod
    def from_SED_result_inputs(cls, SED_fit_params_arr, phot, property_arr, \
            property_errs_arr, property_PDFs_arr, SED_arr, rest_UV_wav_lims = [1268., 2580.] * u.Angstrom):
        SED_result_arr = [SED_result(SED_fit_params, phot, properties, property_errs, property_PDFs, SED, \
            rest_UV_wav_lims = rest_UV_wav_lims) for SED_fit_params, properties, property_errs, property_PDFs, \
            SED in zip(SED_fit_params_arr, property_arr, property_errs_arr, property_PDFs_arr, SED_arr)]
        return cls(SED_fit_params_arr, SED_result_arr)

    
class Catalogue_SED_results:

    def __init__(self, SED_fit_params_arr, cat_SED_results):
        self.SED_results = [Galaxy_SED_results(SED_fit_params_arr, SED_result_arr).SED_results for SED_result_arr in cat_SED_results]
    
    def __len__(self):
        return len(self.SED_results)
    
    @classmethod
    def from_cat(cls, cat, SED_fit_params_arr):
        cat_PDF_paths = [cat.PDF_paths[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)] for SED_fit_params in SED_fit_params_arr]
        cat_SED_paths = [cat.SED_paths[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)] for SED_fit_params in SED_fit_params_arr]
        return cls.from_fits_cat(cat.open_cat(), cat.cat_creator, SED_fit_params_arr, cat_PDF_paths, cat_SED_paths, phot_arr = [gal.phot for gal in cat])

    @classmethod
    def from_fits_cat(cls, fits_cat, cat_creator, SED_fit_params_arr, cat_PDF_paths = None, SED_paths = None, \
            phot_arr = None, instrument = None, rest_UV_wav_lims = [1268., 2580.] * u.Angstrom):
        assert(all(True if "code" in SED_fit_params else False for SED_fit_params in SED_fit_params_arr))
        # calculate array of galaxy photometries if required
        if type(phot_arr) == type(None) and type(instrument) != type(None):
            phot_arr = Multiple_Photometry.from_fits_cat(fits_cat, instrument, cat_creator).phot_arr
        elif type(phot_arr) != type(None) and type(instrument) == type(None):
            pass
        else:
            galfind_logger.critical("Must specify either phot or instrument in Galaxy_SED_results!")

        # IDs = list(np.full(len(SED_fit_params_arr), np.array(fits_cat[cat_creator.ID_label]).astype(int)))
        # convert cat_properties from array of len(SED_fit_params_arr), with each element a dict of galaxy properties for the entire catalogue with values of arrays of len(fits_cat)
        # to array of len(fits_cat), with each element an array of len(SED_fit_params_arr) containing a dict of properties for a single galaxy
        labels_arr = [{key: SED_fit_params["code"].galaxy_property_labels(value) for key, value in SED_fit_params["code"].galaxy_property_dict.items()} for SED_fit_params in SED_fit_params_arr]
        cat_properties = [{gal_property: list(fits_cat[label]) for gal_property, label in labels.items()} for labels in labels_arr]
        cat_properties = [[{key: value[i] for key, value in SED_fitting_properties.items()} for SED_fitting_properties in cat_properties] for i in range(len(fits_cat))]
        # convert cat_property_errs from array of len(SED_fit_params_arr), with each element a dict of galaxy properties for the entire catalogue with values of arrays of len(fits_cat)
        # to array of len(fits_cat), with each element an array of len(SED_fit_params_arr) containing a dict of properties for a single galaxy
        err_labels_arr = [{key: SED_fit_params["code"].galaxy_property_labels(value) for key, value in SED_fit_params["code"].galaxy_property_errs_dict.items()} for SED_fit_params in SED_fit_params_arr]
        cat_property_errs = [{gal_property: list(fits_cat[label]) for gal_property, label in labels.items()} for labels in labels_arr]
        cat_property_errs = [[{key: value[i] for key, value in SED_fitting_properties.items()} for SED_fitting_properties in cat_properties] for i in range(len(fits_cat))]

        if type(cat_PDF_paths) == type(None):
            # make array of the correct shape for appropriate parsing
            pass
        else:
            assert(len(SED_fit_params_arr) == len(cat_PDF_paths))
            cat_property_PDFs = []
            # loop through SED fit params and their corresponding PDF directories
            for SED_fit_params, PDF_paths in tqdm(zip(SED_fit_params_arr, cat_PDF_paths), total = len(SED_fit_params_arr), desc = "Constructing galaxy property PDFs"):
                # dict of paths to PDFs for each galaxy property, type = dict(key: array of len(cat))
                cat_property_PDF_paths = {gal_property: PDF_paths[gal_property] for gal_property in SED_fit_params["code"].galaxy_property_dict.keys()}
                # check that these paths correspond to the correct galaxies
                assert(len(PDF_paths[gal_property]) == len(fits_cat) for gal_property in SED_fit_params["code"].galaxy_property_dict.keys())
                # construct PDF objects, type = array of len(fits_cat), each element a dict of {gal_property: PDF object}
                for gal_properties, paths in cat_property_PDF_paths.items():
                    if all(path == paths[0] for path in paths) and ".h5" in paths[0]:
                        # open .h5 file
                        hf = h5py.File(paths[0], "r")
                        gal_property_arrs = {}
                cat_property_PDFs_ = [{gal_property: getattr(globals()[f"{gal_property}_PDF"], f"{gal_property}_PDF")()} \
                    if f"{gal_property}_PDF" in globals() else {gal_property: PDF()} for PDF_path in PDF_paths \
                    for gal_property, PDF_paths in cat_property_PDF_paths.items()]

        # SEDs - need path to saved SEDs
        cat_SEDs = []
        
        return cls.from_SED_result_inputs(SED_fit_params_arr, phot_arr, cat_properties, \
            cat_property_errs, cat_property_PDFs, cat_SEDs, rest_UV_wav_lims = rest_UV_wav_lims)
    
    @classmethod
    def from_SED_result_inputs(cls, SED_fit_params_arr, phot_arr, cat_properties, \
            cat_property_errs, cat_property_PDFs, cat_SEDs, rest_UV_wav_lims = [1268., 2580.] * u.Angstrom):
        cat_SED_results = [[SED_result.from_SED_result_inputs(SED_fit_params, phot, properties, \
            property_errs, property_PDFs, SED, rest_UV_wav_lims = rest_UV_wav_lims) \
            for SED_fit_params, properties, property_errs, property_PDFs, SED in \
            zip(SED_fit_params_arr, property_arr, property_errs_arr, property_PDF_arr, SED_arr)] \
            for phot, property_arr, property_errs_arr, property_PDF_arr, SED_arr \
            in zip(phot_arr, cat_properties, cat_property_errs, cat_property_PDFs, cat_SEDs)]
        return cls(SED_fit_params_arr, cat_SED_results)
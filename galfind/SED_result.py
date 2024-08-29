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
from tqdm import tqdm
import h5py
import time
from typing import Union
import itertools
from copy import deepcopy

from .Photometry import Photometry, Multiple_Photometry
from .Photometry_rest import Photometry_rest
from .PDF import PDF, Redshift_PDF
from . import useful_funcs_austind as funcs
from . import config, galfind_logger

class SED_result:
    
    def __init__(self, SED_fit_params, phot, properties, property_errs, property_PDFs, SED):
        self.SED_fit_params = SED_fit_params
        #[setattr(self, key, value) for key, value in SED_fit_params.items()]
        self.properties = properties
        [setattr(self, key, value) for key, value in properties.items()]
        self.property_errs = property_errs
        #[setattr(self, f"{key}_l1", value[0]) for key, value in property_errs.items()]
        #[setattr(self, f"{key}_u1", value[1]) for key, value in property_errs.items()]
        # load in peaks
        if type(property_PDFs) == type(None):
            self.property_PDFs = None
        else:
            self.property_PDFs = {property: property_PDF.load_peaks_from_best_fit(properties[property], \
                properties["chi_sq"]) for property, property_PDF in property_PDFs.items()}
        #[setattr(self, f"{key}_PDF", value) for key, value in property_PDFs.items()]
        self.SED = SED
        self.phot_rest = Photometry_rest.from_phot(phot, self.z)

    def __str__(self, print_phot_rest = True, print_PDFs = True, print_SED = True):
        line_sep = "*" * 40 + "\n"
        band_sep = "-" * 10 + "\n"
        output_str = line_sep
        output_str += "SED FITTING RESULT:\n"
        output_str += band_sep
        for key, value in self.SED_fit_params.items():
            if key == "code":
                output_str += f"{key.upper()}: {value.__class__.__name__}\n"
            else:
                output_str += f"{key.upper()}: {value}\n"
        output_str += band_sep
        # print property errors and units here too
        for key, value in self.properties.items():
            output_str += f"{key} = {str(value)}\n"
        if print_PDFs:
            for PDF_obj in self.property_PDFs.values():
                output_str += str(PDF_obj)
        else:
            output_str += f"PDFs LOADED: {', '.join([key for key in self.property_PDFs.keys()])}\n"
        if print_SED:
            output_str += str(self.SED)
        # phot rest should really be contained in self.SED
        if print_phot_rest:
            output_str += self.phot_rest.__str__(print_PDFs)
        output_str += line_sep
        return output_str
    
    def __getattr__(self, property_name: str, origin: str = "SED_result", property_type: str = "val") -> Union[None, u.Quantity, u.Magnitude, u.Dex]:
        assert origin in ["SED_result", "phot_rest", "SED"], galfind_logger.critical(f"SED_result.__getattr__ {origin=} not in ['SED_result', 'phot_rest']!")
        property_type = property_type.lower()
        # extract relevant SED result properties
        if origin == "SED_result":
            assert property_type in ["val", "errs", "l1", "u1", "pdf"], galfind_logger.critical(f"{property_type=} not in ['val', 'errs', 'l1', 'u1', 'pdf']!")
            if property_type == "val":
                access_dict = self.properties
            elif property_type in ["errs", "l1", "u1"]:
                access_dict = self.property_errs
            else: #property_type == "pdf"
                access_dict = self.property_PDFs
            if property_name not in access_dict.keys():
                err_message = f"{property_name} {property_type} not available in SED_result object!"
                galfind_logger.warning(err_message)
                raise AttributeError(err_message)
            else:
                if property_type == "l1":
                    return access_dict[property_name][0]
                elif property_type == "u1":
                    return access_dict[property_name][1]
                else:
                    return access_dict[property_name]
        # extract relevant photometry rest properties
        elif "phot_rest" in origin:
            return self.phot_rest.__getattr__(property_name, origin.replace("phot_rest_", ""), property_type = property_type)
        else: # origin == "SED"
            return self.SED.__getattr__(property_name, origin)
        
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, deepcopy(value, memo))
        return result

    @classmethod
    def from_gal(cls, gal, SED_fit_params):
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
        return NotImplementedError # need to load in PDFs and SED
        #return cls(SED_fit_params, gal.phot, properties, property_errs, property_PDFs, SED)
    

class Galaxy_SED_results:
    
    def __init__(self, SED_fit_params_arr, SED_result_arr):
        self.SED_results = {SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params): \
            SED_result for SED_fit_params, SED_result in zip(SED_fit_params_arr, SED_result_arr)}

    def __len__(self):
        return len([True for (code_names, templates_lowz_zmax_results) in self.SED_results.items() \
            for (templates, lowz_zmax_results) in templates_lowz_zmax_results.items() \
            for (lowz_zmax, results) in lowz_zmax_results.items()])
    
    @classmethod
    def from_gal(cls, gal, SED_fit_params_arr):
        return cls(SED_fit_params_arr, [SED_result.from_gal(gal, SED_fit_params) for SED_fit_params in SED_fit_params_arr])

    @classmethod
    def from_SED_result_inputs(cls, SED_fit_params_arr, phot, property_arr, \
            property_errs_arr, property_PDFs_arr, SED_arr):
        SED_result_arr = [SED_result(SED_fit_params, phot, properties, property_errs, property_PDFs, SED) for SED_fit_params, properties, property_errs, property_PDFs, \
            SED in zip(SED_fit_params_arr, property_arr, property_errs_arr, property_PDFs_arr, SED_arr)]
        return cls(SED_fit_params_arr, SED_result_arr)

    
class Catalogue_SED_results:

    def __init__(self, SED_fit_params_arr, cat_SED_results):
        self.SED_results = [Galaxy_SED_results(SED_fit_params_arr, SED_result_arr).SED_results for SED_result_arr in cat_SED_results]
    
    def __len__(self):
        return len(self.SED_results)
    
    @classmethod
    def from_cat(cls, cat, SED_fit_params_arr: Union[list, np.array], \
            load_PDFs: bool = True, load_SEDs: bool = True, timed: bool = True):
        if load_PDFs:
            if timed:
                cat_PDF_paths = [cat.phot_PDF_paths[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)] \
                    for SED_fit_params in tqdm(SED_fit_params_arr, desc = "Collecting cat_PDF_paths", total = len(SED_fit_params_arr))]
            else:
                cat_PDF_paths = [cat.phot_PDF_paths[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)] for SED_fit_params in SED_fit_params_arr]
        else:
            cat_PDF_paths = None
        if load_SEDs:
            if timed:
                cat_SED_paths = [cat.phot_SED_paths[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)] \
                    for SED_fit_params in tqdm(SED_fit_params_arr, desc = "Collecting cat_SED_paths", total = len(SED_fit_params_arr))]
            else:
                cat_SED_paths = [cat.phot_SED_paths[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)] for SED_fit_params in SED_fit_params_arr]
        else:
            cat_SED_paths = None
        # these arrays are have length == len(cat), and may in principle included None's where SED fitting has now been performed
        
        SED_fit_cats = [cat.open_cat(hdu = SED_fit_params["code"].hdu_from_SED_fit_params(SED_fit_params), cropped = True) for SED_fit_params in SED_fit_params_arr]
        return cls.from_fits_cat(SED_fit_cats, SED_fit_params_arr, cat_PDF_paths, cat_SED_paths, phot_arr = [gal.phot for gal in cat], timed = timed)


    @classmethod
    def from_fits_cat(cls, SED_fit_cats: Union[Table, list, np.array], SED_fit_params_arr: Union[list, np.array], \
            cat_PDF_paths: Union[list, np.array, None] = None, cat_SED_paths: Union[list, np.array, None] = None, \
            phot_arr = None, instrument = None, phot_cat: Union[Table, None] = None, cat_creator = None, timed: bool = True):
        # input assertions
        assert all(True if "code" in SED_fit_params else False for SED_fit_params in SED_fit_params_arr)
        assert len(SED_fit_cats) == len(SED_fit_params_arr)

        # calculate array of galaxy photometries if required
        if type(phot_arr) == type(None) and type(instrument) != type(None) and type(phot_cat) != type(None) and type(cat_creator) != type(None):
            phot_arr = Multiple_Photometry.from_fits_cat(phot_cat, instrument, cat_creator, timed = timed).phot_arr
        elif type(phot_arr) != type(None) and type(instrument) == type(None) and type(phot_cat) == type(None) and type(cat_creator) == type(None):
            pass
        else:
            galfind_logger.critical("Must specify either phot or instrument AND phot_cat AND cat_creator in Galaxy_SED_results!")

        if timed:
            start = time.time()

        # assume properties all come from the same table if only a single catalogue is given
        if type(SED_fit_cats) in [Table]:
            SED_fit_cats = [SED_fit_cats for i in range(len(SED_fit_params_arr))]

        # determine IDs
        ID_labels = [SED_fit_params["code"].ID_label for SED_fit_params in SED_fit_params_arr]
        IDs_arr = [np.array(fits_cat[ID_label]).astype(int) for fits_cat, ID_label in zip(SED_fit_cats, ID_labels)]
        assert all(len(IDs) == len(phot_arr) for IDs in IDs_arr), galfind_logger.critical(f"Not all catalogues in {SED_fit_params_arr=} have the same length as phot_cat. {[len(IDs) for IDs in IDs_arr]=} != {len(phot_arr)=}")
        assert all(all(IDs[i] == IDs_arr[0][i] for i in range(len(IDs))) for IDs in IDs_arr), galfind_logger.critical(f"Not all catalogues in {SED_fit_params_arr=} have the same IDs!")
        IDs = IDs_arr[0]

        # convert cat_properties from array of len(SED_fit_params_arr), with each element a dict of galaxy properties for the entire catalogue with values of arrays of len(fits_cat)
        # to array of len(fits_cat), with each element an array of len(SED_fit_params_arr) containing a dict of properties for a single galaxy
        labels_arr = [{key: SED_fit_params["code"].galaxy_property_labels(key, SED_fit_params) \
            for key, val in SED_fit_params["code"].galaxy_property_dict.items()} for SED_fit_params in SED_fit_params_arr]
        cat_properties = [{gal_property: list(fits_cat[label]) \
            for gal_property, label in labels.items()} for labels, fits_cat in zip(labels_arr, SED_fit_cats)]
        cat_properties = [[{key: value[i] * u.Unit(SED_fit_params["code"].gal_property_unit_dict[key]) \
            if key in SED_fit_params["code"].gal_property_unit_dict.keys() else value[i] * u.dimensionless_unscaled \
            for key, value in SED_fitting_properties.items()} for SED_fit_params, SED_fitting_properties \
            in zip(SED_fit_params_arr, cat_properties)] for i in range(len(IDs))] # may include invalid values where SED fitting not performed
        
        # convert cat_property_errs from array of len(SED_fit_params_arr), with each element a dict of galaxy properties for the entire catalogue with values of arrays of len(fits_cat)
        # to array of len(fits_cat), with each element an array of len(SED_fit_params_arr) containing a dict of properties for a single galaxy
        err_labels_arr = [{key: SED_fit_params["code"].galaxy_property_labels(key, SED_fit_params, is_err = True) \
            for key, val in SED_fit_params["code"].galaxy_property_errs_dict.items()} for SED_fit_params in SED_fit_params_arr]
        # ensure that all errors have an associated property
        assert all(err_key in labels.keys() for labels, SED_fit_params in zip(labels_arr, SED_fit_params_arr) for err_key in SED_fit_params["code"].galaxy_property_errs_dict.keys())
        adjust_errs_arr = [SED_fit_params["code"].are_errs_percentiles for SED_fit_params in SED_fit_params_arr]
        # adjust errors if required (i.e. if 16th and 84th percentiles rather than errors)
        cat_property_errs = [{gal_property: list(funcs.adjust_errs(np.array(fits_cat[labels[gal_property]]), \
            np.array([np.array(fits_cat[err_label[0]]), np.array(fits_cat[err_label[1]])]))[1]) if adjust_errs else \
            [list(fits_cat[err_label[0]]), list(fits_cat[err_label[1]])] for gal_property, err_label in err_labels.items()} \
            for adjust_errs, labels, err_labels, fits_cat in zip(adjust_errs_arr, labels_arr, err_labels_arr, SED_fit_cats)]
        cat_property_errs = [[{key: [value[0][i], value[1][i]] for key, value in SED_fitting_property_errs.items()} \
            for SED_fitting_property_errs in cat_property_errs] for i in range(len(IDs))] # may include invalid values where SED fitting not performed
        
        if timed:
            mid = time.time()
            print(f"Loading properties and associated errors took {(mid - start):.1f}s")
        
        # load in PDFs
        # make array of the correct shape for appropriate parsing
        if timed:
            start_1 = time.time()

        if type(cat_PDF_paths) != type(None):
            assert(len(SED_fit_params_arr) == len(cat_PDF_paths))
            cat_property_PDFs = np.full((len(IDs), len(SED_fit_params_arr)), None)
            # loop through SED_fit_params_arr and corresponding cat_PDF_paths
            for i, (SED_fit_params, PDF_paths) in enumerate(zip(SED_fit_params_arr, cat_PDF_paths)): # tqdm(, \
                # total = len(SED_fit_params_arr), desc = "Constructing galaxy property PDFs"):
                # check that these paths correspond to the correct galaxies
                assert(len(PDF_paths[gal_property]) == len(IDs) for gal_property in PDF_paths.keys())
                # construct PDF objects, type = array of len(fits_cat), each element a dict of {gal_property: PDF object} excluding None PDFs
                cat_property_PDFs_ = {gal_property: SED_fit_params["code"].extract_PDFs(gal_property, IDs, \
                    PDF_paths[gal_property], SED_fit_params) for gal_property in PDF_paths.keys()}
                cat_property_PDFs_ = [{gal_property: PDF_arr[j] for gal_property, PDF_arr \
                    in cat_property_PDFs_.items() if type(PDF_arr[j]) != type(None)} for j in range(len(IDs))]
                cat_property_PDFs[:, i] = [None if len(cat_property_PDF_) == 0 else cat_property_PDF_ for cat_property_PDF_ in cat_property_PDFs_]
        else:
            galfind_logger.info("Not loading catalogue property PDFs")
            cat_property_PDFs = None

        # load in SEDs
        # make array of the correct shape for appropriate parsing
        if timed:
            start = time.time()

        if type(cat_SED_paths) != type(None):
            assert(len(SED_fit_params_arr) == len(cat_SED_paths))
            cat_SEDs = np.full((len(IDs), len(SED_fit_params_arr)), None)
            # loop through SED_fit_params_arr and corresponding cat_SED_paths
            for i, (SED_fit_params, SED_paths) in enumerate(zip(SED_fit_params_arr, cat_SED_paths)): # tqdm(, \
                   # total = len(SED_fit_params_arr), desc = "Constructing galaxy SEDs"):
                # check that these paths correspond to the correct galaxies
                assert(len(SED_paths) == len(IDs) for gal_property in SED_fit_params["code"].galaxy_property_dict.keys())
                # construct SED objects, type = array of len(fits_cat), each element containing an SED object
                cat_SEDs[:, i] = SED_fit_params["code"].extract_SEDs(IDs, SED_paths)
        else:
            galfind_logger.info("Not loading catalogue SEDs")
            cat_SEDs = None

        if timed:
            mid = time.time()
        cls_obj = cls.from_SED_result_inputs(SED_fit_params_arr, phot_arr, cat_properties, \
            cat_property_errs, cat_property_PDFs, cat_SEDs)
        if timed:
            end = time.time()
            print("Loading PDFs, SEDs: ", start - start_1, mid - start, end - mid)
        return cls_obj
    
    @classmethod
    def from_SED_result_inputs(cls, SED_fit_params_arr: Union[list, np.array], \
            phot_arr: Union[list, np.array], cat_properties: Union[list, np.array], \
            cat_property_errs: Union[list, np.array], cat_property_PDFs: Union[np.array, None], \
            cat_SEDs: Union[np.array, None]):
        
        # if not loaded, construct appropriately shaped None arrays
        if type(cat_property_PDFs) == type(None):
            out_shape = np.array(cat_properties).shape
            cat_property_PDFs_ = np.array(list(itertools.repeat(list(itertools.repeat(None, out_shape[1])), out_shape[0])))
            assert out_shape == cat_property_PDFs_.shape
        else:
            cat_property_PDFs_ = cat_property_PDFs
        if type(cat_SEDs) == type(None):
            out_shape = np.array(cat_properties).shape
            cat_SEDs_ = np.array(list(itertools.repeat(list(itertools.repeat(None, out_shape[1])), out_shape[0])))
            assert out_shape == cat_SEDs_.shape
        else:
            cat_SEDs_ = cat_SEDs

        cat_SED_results = [[SED_result(SED_fit_params, phot, properties, property_errs, property_PDFs, SED) \
            for SED_fit_params, properties, property_errs, property_PDFs, SED in \
            zip(SED_fit_params_arr, property_arr, property_errs_arr, property_PDF_arr, SED_arr)] \
            for phot, property_arr, property_errs_arr, property_PDF_arr, SED_arr \
            in zip(phot_arr, cat_properties, cat_property_errs, cat_property_PDFs_, cat_SEDs_)]
        
        return cls(SED_fit_params_arr, cat_SED_results)
    

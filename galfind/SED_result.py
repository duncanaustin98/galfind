#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:50:27 2023

@author: austind
"""

# SED_result.py
import warnings
import numpy as np

from .Photometry import Photometry, Multiple_Photometry
from .Photometry_rest import Photometry_rest
from .SED_codes import SED_code
from . import useful_funcs_austind as funcs

class SED_result:
    
    def __init__(self, phot, z, chi_sq, z_PDF_path, SED_path, code_name, lowz_zmax, templates):
        self.phot_rest = Photometry_rest.from_phot(phot, z)
        self.z = z
        self.chi_sq = chi_sq
        self.z_PDF_path = z_PDF_path
        self.SED_path = SED_path
        # identifies SED fitting origin
        self.code_name = code_name
        self.lowz_zmax = lowz_zmax
        self.templates = templates

class Galaxy_SED_results:
    
    def __init__(self, phot, redshifts, chi_sqs, z_PDF_paths, SED_paths, code_names, lowz_zmaxs, templates_arr):
        self.SED_results = [SED_result(phot, z, chi_sq, z_PDF_path, SED_path, code_name, lowz_zmax, templates) for z, chi_sq, z_PDF_path, SED_path, code_name, lowz_zmax, templates \
            in zip(redshifts, chi_sqs, z_PDF_paths, SED_paths, code_names, lowz_zmaxs, templates_arr)]
        
    @classmethod
    def from_fits_cat(cls, fits_cat_row, cat_creator, code_names, lowz_zmaxs, templates_arr, phot = None, instrument = None, fits_cat_path = None):
        
        # calculate photometry if required
        if phot == None and instrument != None:
            phot = Photometry.from_fits_cat(fits_cat_row, instrument, cat_creator)
        elif phot != None and instrument == None:
            pass
        else:
            raise(Exception("Must specify either phot or instrument in Galaxy_SED_results!"))
        
        try:
            fits_cat_path = fits_cat_row.meta["cat_path"]
        except:
            warnings.warn("fits_cat_path not loaded from catalogue meta, instead using SED_result.from_fits_cat(fits_cat_path) = {fits_cat_path}!")
        
        # efficiency of this can probably be improved using column_labels
        redshifts = []
        chi_sqs = []
        z_PDF_paths = []
        SED_paths = []
        for code_name, lowz_zmax, templates in zip(code_names, lowz_zmaxs, templates_arr):
            try:
                z = float(fits_cat_row[SED_code.code_from_name(code_name).galaxy_property_labels("z_phot", templates, lowz_zmax)])
                chi_sq = float(fits_cat_row[SED_code.code_from_name(code_name).galaxy_property_labels("chi_sq", templates, lowz_zmax)])
            except:
                raise(Exception(f"SED run not performed for {code_name}, lowz_zmax = {lowz_zmax}"))
            redshifts.append(z)
            chi_sqs.append(chi_sq)
            ID = int(fits_cat_row[cat_creator.ID_label])
            lowz_label = funcs.lowz_label(lowz_zmax)
            z_PDF_paths.append(SED_code.code_from_name(code_name).z_PDF_path_from_cat_path(fits_cat_path, ID, templates, lowz_label))
            SED_paths.append(SED_code.code_from_name(code_name).SED_path_from_cat_path(fits_cat_path, ID, templates, lowz_label))
        return cls(phot, redshifts, chi_sqs, z_PDF_paths, SED_paths, code_names, lowz_zmaxs, templates_arr)
    
class Catalogue_SED_results:
    
    def __init__(self, phot_arr, cat_redshifts, cat_chi_sqs, cat_z_PDF_paths, cat_SED_paths, code_names, lowz_zmaxs, templates_arr):
        self.SED_results = [Galaxy_SED_results(gal_phot, gal_redshifts, gal_chi_sqs, gal_z_PDF_paths, gal_SED_paths, code_names, lowz_zmaxs, templates_arr) \
                            for gal_phot, gal_redshifts, gal_chi_sqs, gal_z_PDF_paths, gal_SED_paths in zip(phot_arr, cat_redshifts, cat_chi_sqs, cat_z_PDF_paths, cat_SED_paths)]
    
    @classmethod
    def from_fits_cat(cls, fits_cat, cat_creator, code_names, lowz_zmaxs, templates_arr, phot_arr = None, instrument = None, fits_cat_path = None, gal_properties = ["z_phot", "chi_sq"]):
        
        # calculate array of galaxy photometries if required
        if phot_arr == None and instrument != None:
            phot_arr = Multiple_Photometry.from_fits_cat(fits_cat, instrument, cat_creator).phot_arr
        elif phot_arr != None and instrument == None:
            pass
        else:
            raise(Exception("Must specify either phot or instrument in Galaxy_SED_results!"))
            
        try:
            fits_cat_path = fits_cat.meta["cat_path"]
        except:
            warnings.warn("fits_cat_path not loaded from catalogue meta, instead using SED_result.from_fits_cat(fits_cat_path) = {fits_cat_path}!")
        
        labels_dict = {gal_property: funcs.GALFIND_SED_column_labels(code_names, lowz_zmaxs, templates_arr, gal_property) for gal_property in gal_properties}
        cat_redshifts = [fits_cat[labels] for labels in labels_dict["z_phot"]]
        cat_chi_sqs = [fits_cat[labels] for labels in labels_dict["chi_sq"]]
        
        IDs = np.array(fits_cat[cat_creator.ID_label]).astype(int)
        cat_z_PDF_paths = []
        cat_SED_paths = []
        raise(Exception("This implementation still requires fixing!"))
        # should be a faster way of reading in this data
        for code_name, lowz_zmax, templates in zip(code_names, lowz_zmaxs, templates_arr):
            lowz_label = funcs.lowz_label(lowz_zmax)
            cat_z_PDF_paths.append([code.z_PDF_paths_from_cat_path(fits_cat_path, ID, templates, lowz_label) for ID in IDs])
            cat_SED_paths.append([code.SED_paths_from_cat_path(fits_cat_path, ID, templates, lowz_label) for ID in IDs])
        return cls(phot_arr, cat_redshifts, cat_chi_sqs, cat_z_PDF_paths, cat_SED_paths, code_names, lowz_zmaxs, templates_arr)
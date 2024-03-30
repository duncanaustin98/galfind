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

from .Photometry import Photometry, Multiple_Photometry
from .Photometry_rest import Photometry_rest

from . import useful_funcs_austind as funcs

class SED_result:
    
    def __init__(self, phot, z, chi_sq, z_PDF_path, SED_path, code_name, lowz_zmax, templates, rest_UV_wavs_arr = [[1268., 2580.] * u.Angstrom, [1250., 3000.] * u.Angstrom], properties = {}):
        self.phot_rest = {Photometry_rest.rest_UV_wavs_name(rest_UV_wavs): Photometry_rest.from_phot(phot, z, rest_UV_wavs) for rest_UV_wavs in rest_UV_wavs_arr}
        self.z = z
        self.chi_sq = chi_sq
        self.z_PDF_path = z_PDF_path
        self.SED_path = SED_path
        # identifies SED fitting origin
        self.code_name = code_name
        self.lowz_zmax = lowz_zmax
        self.templates = templates
        self.properties = properties

    def __str__(self):
        line_sep = "*" * 40 + "\n"
        band_sep = "-" * 10 + "\n"
        output_str = line_sep
        output_str += "SED FITTING RESULT:\n"
        output_str += band_sep
        output_str += f"CODE: {self.code_name}\n"
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
        # could also print rest frame photometry here
        output_str += line_sep
        return output_str

class Galaxy_SED_results:
    
    def __init__(self, phot, redshifts, chi_sqs, z_PDF_paths, SED_paths, code_names, lowz_zmaxs, templates_arr):
        self.SED_results = {code_name: {templates: {funcs.lowz_label(lowz_zmax): SED_result(phot, z, chi_sq, z_PDF_path, SED_path, code_name, lowz_zmax, templates) for z, chi_sq, z_PDF_path, SED_path, lowz_zmax \
            in zip(redshifts, chi_sqs, z_PDF_paths, SED_paths, lowz_zmaxs)}} for code_name, templates in zip(code_names, templates_arr)}

    def __len__(self):
        return len([True for (code_names, templates_lowz_zmax_results) in self.SED_results.items() \
            for (templates, lowz_zmax_results) in templates_lowz_zmax_results.items() \
            for (lowz_zmax, results) in lowz_zmax_results.items()])

    @classmethod
    def from_fits_cat(cls, fits_cat_row, cat_creator, codes, lowz_zmaxs, templates_arr, phot = None, instrument = None, fits_cat_path = None):
        
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
        for code, lowz_zmax, templates in zip(codes, lowz_zmaxs, templates_arr):
            try:
                z = float(fits_cat_row[code.galaxy_property_labels("z_phot", templates, lowz_zmax)])
                chi_sq = float(fits_cat_row[code.galaxy_property_labels("chi_sq", templates, lowz_zmax)])
            except:
                raise(Exception(f"SED run not performed for {code.code_name}, lowz_zmax = {lowz_zmax}"))
            redshifts.append(z)
            chi_sqs.append(chi_sq)
            ID = int(fits_cat_row[cat_creator.ID_label])
            lowz_label = funcs.lowz_label(lowz_zmax)
            z_PDF_paths.append(code.z_PDF_path_from_cat_path(fits_cat_path, ID, templates, lowz_label))
            SED_paths.append(code.SED_path_from_cat_path(fits_cat_path, ID, templates, lowz_label))
        return cls(phot, redshifts, chi_sqs, z_PDF_paths, SED_paths, [code.code_name for code in codes], lowz_zmaxs, templates_arr)
    
class Catalogue_SED_results:
    
    def __init__(self, phot_arr, cat_redshifts, cat_chi_sqs, cat_z_PDF_paths, cat_SED_paths, code_names, lowz_zmaxs, templates_arr):
        # an array (each element is a galaxy) of dictionaries (each element is a single SED fitting code) containing a dictionary (each containing SED results from a specific template set)
        self.SED_results = [Galaxy_SED_results(gal_phot, gal_redshifts, gal_chi_sqs, gal_z_PDF_paths, gal_SED_paths, code_names, lowz_zmaxs, templates_arr).SED_results \
                            for gal_phot, gal_redshifts, gal_chi_sqs, gal_z_PDF_paths, gal_SED_paths in zip(phot_arr, cat_redshifts, cat_chi_sqs, cat_z_PDF_paths, cat_SED_paths)]
    
    def __len__(self):
        return len(self.SED_results)
    
    @classmethod
    def from_fits_cat(cls, fits_cat, cat_creator, codes, lowz_zmaxs, templates_arr, phot_arr = None, instrument = None, fits_cat_path = None, gal_properties = ["z_phot", "chi_sq"]):
        
        # calculate array of galaxy photometries if required
        if phot_arr == None and instrument != None:
            phot_arr = Multiple_Photometry.from_fits_cat(fits_cat, instrument, cat_creator).phot_arr
        elif phot_arr != None and instrument == None:
            pass
        else:
            raise(Exception("Must specify either phot or instrument in Galaxy_SED_results!"))
        if fits_cat_path != None:
            pass
        elif "cat_path" in fits_cat.meta.keys():
            fits_cat_path = fits_cat.meta["cat_path"]
        else:
            warnings.warn(f"fits_cat_path not loaded from catalogue meta, instead using SED_result.from_fits_cat(fits_cat_path) = {fits_cat_path}!")
            raise(Exception())
        
        labels_dict = {gal_property: funcs.GALFIND_SED_column_labels(codes, lowz_zmaxs, templates_arr, gal_property) for gal_property in gal_properties}
        print(labels_dict)
        #print(labels_dict)
        #print(codes, lowz_zmaxs, templates_arr, gal_properties)
        #print("Need to sort out cat_redshifts and cat_chi_sqs in Catalogue_SED_results.from_fits_cat")
        #galfind_logger.error("ERROR!")
        #z = float(fits_cat_row[code.galaxy_property_labels("z_phot", templates, lowz_zmax)])
        #print(fits_cat.columns)
        cat_redshifts = np.array([fits_cat[labels] for labels in labels_dict["z_phot"]])
        cat_chi_sqs = np.array([fits_cat[labels] for labels in labels_dict["chi_sq"]])
        #print(cat_redshifts, cat_chi_sqs, fits_cat, labels_dict, labels_dict["z_phot"])
        IDs = np.array(fits_cat[cat_creator.ID_label]).astype(int)
        cat_z_PDF_paths = []
        cat_SED_paths = []
        #raise(Exception("This implementation still requires fixing!"))
        # should be a faster way of reading in this data
        for code, templates in zip(codes, templates_arr):
            for lowz_zmax in lowz_zmaxs:
                lowz_label = funcs.lowz_label(lowz_zmax)
                cat_z_PDF_paths.append([code.z_PDF_paths_from_cat_path(fits_cat_path, ID, templates, lowz_label) for ID in IDs])
                cat_SED_paths.append([code.SED_paths_from_cat_path(fits_cat_path, ID, templates, lowz_label) for ID in IDs])
        return cls(phot_arr, cat_redshifts.T, cat_chi_sqs.T, np.array(cat_z_PDF_paths).T, np.array(cat_SED_paths).T, [code.__class__.__name__ for code in codes], lowz_zmaxs, templates_arr)

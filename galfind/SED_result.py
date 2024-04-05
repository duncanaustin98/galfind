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
from .PDF import Redshift_PDF
from . import useful_funcs_austind as funcs

class SED_result:
    
    def __init__(self, phot, z, chi_sq, z_PDF_path, SED_path, ID, SED_code, rest_UV_wavs_arr = [[1268., 2580.] * u.Angstrom, [1250., 3000.] * u.Angstrom], properties = {}):
        self.phot_rest = Photometry_rest.from_phot(phot, z) #{Photometry_rest.rest_UV_wavs_name(rest_UV_wavs): Photometry_rest.from_phot(phot, z, rest_UV_wavs) for rest_UV_wavs in rest_UV_wavs_arr}
        self.z = z
        self.chi_sq = chi_sq
        self.ID = ID
        self.SED_code = SED_code
        self.z_PDF_path = z_PDF_path
        self.SED_path = SED_path
        self.properties = properties

    def __str__(self, print_rest_phot = False, print_pdf_sed_paths = True):
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
        if print_pdf_sed_paths:
            output_str += f"Z-PDF PATH = {self.z_PDF_path}\n"
            output_str += f"SED PATH = {self.SED_path}\n"
        if print_rest_phot:
            for phot_rest in self.phot_rest.values():
                output_str += str(phot_rest)
        output_str += line_sep
        return output_str
    
    # property so that it is not loaded at instantiation of object
    @property
    def z_PDF(self):
        return Redshift_PDF.from_SED_code_output(self.z_PDF_path, self.ID, self.SED_code)

class Galaxy_SED_results:
    
    def __init__(self, phot, redshifts, chi_sqs, z_PDF_paths, SED_paths, IDs, SED_codes, templates_arr, lowz_zmaxs):
        self.SED_results = {SED_code.__class__.__name__: {templates: {funcs.lowz_label(lowz_zmax): \
            SED_result(phot, z, chi_sq, z_PDF_path, SED_path, SED_code.__class__.__name__, ID) for z, chi_sq, z_PDF_path, SED_path, ID, lowz_zmax \
            in zip(redshifts, chi_sqs, z_PDF_paths, SED_paths, IDs, lowz_zmaxs)}} for SED_code, templates in zip(SED_codes, templates_arr)}

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
        IDs = []
        redshifts = []
        chi_sqs = []
        z_PDF_paths = []
        SED_paths = []
        for code, lowz_zmax, templates in zip(codes, lowz_zmaxs, templates_arr):
            try:
                ID = int(fits_cat_row[code.ID_label])
                z = float(fits_cat_row[code.galaxy_property_labels("z_phot", templates, lowz_zmax)])
                chi_sq = float(fits_cat_row[code.galaxy_property_labels("chi_sq", templates, lowz_zmax)])
            except:
                raise(Exception(f"SED run not performed for {code.code_name}, lowz_zmax = {lowz_zmax}"))
            IDs.append(ID)
            redshifts.append(z)
            chi_sqs.append(chi_sq)
            lowz_label = funcs.lowz_label(lowz_zmax)
            z_PDF_paths.append(code.z_PDF_path(fits_cat_path, ID, templates, lowz_label))
            SED_paths.append(code.SED_path(fits_cat_path, ID, templates, lowz_label))
        return cls(phot, redshifts, chi_sqs, z_PDF_paths, SED_paths, IDs, codes, templates_arr, lowz_zmaxs)
    
class Catalogue_SED_results:
    
    def __init__(self, phot_arr, cat_redshifts, cat_chi_sqs, cat_z_PDF_paths, cat_SED_paths, IDs, SED_codes, templates_arr, lowz_zmaxs):
        print([len(arr) for arr in [phot_arr, cat_redshifts, cat_chi_sqs, cat_z_PDF_paths, cat_SED_paths, IDs, SED_codes, templates_arr, lowz_zmaxs]])
        # an array (each element is a galaxy) of dictionaries (each element is a single SED fitting code) containing a dictionary (each containing SED results from a specific template set)
        self.SED_results = [Galaxy_SED_results(gal_phot, gal_redshifts, gal_chi_sqs, gal_z_PDF_paths, gal_SED_paths, ID, SED_codes, templates_arr, lowz_zmaxs).SED_results \
            for gal_phot, gal_redshifts, gal_chi_sqs, gal_z_PDF_paths, gal_SED_paths, ID in zip(phot_arr, cat_redshifts, cat_chi_sqs, cat_z_PDF_paths, cat_SED_paths, IDs)]
    
    def __len__(self):
        return len(self.SED_results)
    
    @classmethod
    def from_fits_cat(cls, fits_cat, cat_creator, SED_codes, templates_arr, lowz_zmaxs, \
            phot_arr = None, instrument = None, fits_cat_path = None, gal_properties = ["z_phot", "chi_sq"]):
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
        labels_dict = {gal_property: funcs.GALFIND_SED_column_labels(SED_codes, lowz_zmaxs, templates_arr, gal_property) for gal_property in gal_properties}
        ID_labels = [cat_creator.ID_label for SED_code, templates in zip(SED_codes, templates_arr) for lowz_zmax in lowz_zmaxs]
        
        IDs = np.array([fits_cat[labels] for labels in ID_labels]).astype(int)
        cat_redshifts = np.array([fits_cat[labels] for labels in labels_dict["z_phot"]])
        cat_chi_sqs = np.array([fits_cat[labels] for labels in labels_dict["chi_sq"]])
        
        cat_z_PDF_paths = []
        cat_SED_paths = []
        # should be a faster way of reading in this data
        for SED_code, templates in zip(SED_codes, templates_arr):
            for lowz_zmax in lowz_zmaxs:
                lowz_label = funcs.lowz_label(lowz_zmax)
                cat_z_PDF_paths.append([SED_code.get_z_PDF_path(ID) for ID in IDs])
                cat_SED_paths.append([SED_code.get_SED_path(ID) for ID in IDs])
        
        return cls(phot_arr, cat_redshifts.T, cat_chi_sqs.T, np.array(cat_z_PDF_paths).T, \
            np.array(cat_SED_paths).T, IDs.T, SED_codes, templates_arr, lowz_zmaxs)

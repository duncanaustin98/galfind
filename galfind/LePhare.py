#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:44:23 2023

@author: austind
"""

# LePhare.py
import numpy as np
import astropy.units as u
from pathlib import Path
import itertools
from astropy.table import Table, join
import subprocess
from astropy.io import fits
import json

from . import SED_code
from . import useful_funcs_austind as funcs
from . import config

# %% LePhare SED fitting code

class LePhare(SED_code):
    
    def __init__(self, low_z_run = False):
        code_name = "LePhare"
        #ID_label = "IDENT"
        galaxy_property_labels = {"z_phot": "Z_BEST", "mass": "MASS_BEST"}
        chi_sq_labels = {}
        super().__init__(code_name, galaxy_property_labels, chi_sq_labels, low_z_run)
    
    def from_name(self):
        return LePhare()
    
    def make_in(self, cat, units = u.ABmag, fix_z = False, *args, **kwargs): # from FITS_organiser.py
        lephare_in_path = f"{self.code_dir}/input/{cat.data.instrument.name}/{cat.data.version}/{cat.data.survey}/{cat.cat_name.replace('.fits', '')}_{cat.cat_creator.min_flux_pc_err}pc.in"
        if not Path(lephare_in_path).is_file():
        # 1) obtain input data
            IDs = np.array([gal.ID for gal in cat.gals]) # load IDs
            # load redshifts
            if not fix_z:
                redshifts = np.array([-99. for gal in cat.gals])
            else:
                raise(Exception("The 'fix_z' functionality still requires some work in 'LePhare.convert_fits_to_in'"))
            # load photometry (STILL SHOULD BE MORE GENERAL!!!)
            print("FIX LePhare SED_input_bands!")
            SED_input_bands = cat.data.instrument.new_instrument().bands
            phot, phot_err = self.load_photometry(cat, SED_input_bands, units, -99., {"threshold": 2., "value": 3.})
            # calculate context
            contexts = self.calc_context(cat, SED_input_bands)
        
        # 2) make and save LePhare .in catalogue
            in_data = np.array([np.concatenate(([IDs[i]], list(itertools.chain(*zip(phot[i], phot_err[i]))), [contexts[i]], [redshifts[i]]), axis = None) for i in range(len(IDs))])
            in_names = ["ID"] + list(itertools.chain(*zip(SED_input_bands, [band + "_err" for band in SED_input_bands]))) + ["context", "z"]
            in_types = [int] + list(np.full(len(SED_input_bands) * 2, float)) + [int, float]
            in_tab = Table(in_data, dtype = in_types, names = in_names)
            funcs.make_dirs(lephare_in_path)
            in_tab.write(lephare_in_path, format = "ascii.no_header", delimiter = " ", overwrite = True)
            #print(in_tab)
        return lephare_in_path
    
    def calc_context(self, cat, SED_input_bands):
        print("May need to update 'calc_context' in 'LePhare' for the case where some galaxies have no data for a specific band!")
        contexts = []
        for i, gal in enumerate(cat):
            gal_context = 2 * (2 ** (len(SED_input_bands) - 1)) - 1
            for j, band in enumerate(SED_input_bands):
                if band not in gal.phot_obs.instrument.bands:
                    band_context = 2 ** j
                    gal_context = gal_context - band_context
            contexts.append(gal_context)
        return np.array(contexts).astype(int)
    
    # Currently black box fitting from the lephare config path. Need to make this function more general
    def run_fit(self, in_path, out_path, SED_folder, instrument):
        template_name = f"{instrument.name}_MedWide"
        lephare_config_path = f"{self.code_dir}/Photo_z.para"
        # LePhare bash script python wrapper
        process = subprocess.Popen([f"{config['DEFAULT']['GALFIND_DIR']}/run_lephare.sh", lephare_config_path, in_path, out_path, config['DEFAULT']['GALFIND_DIR'], SED_folder, template_name])
        process.wait()
    
    def make_fits_from_out(self, out_path, *args, **kwargs): # from TXT_to_FITS_converter.py
        fits_out_path = self.out_fits_name(out_path, *args, **kwargs)
        
        # read in the data from the .out table
        txt_in = np.genfromtxt(out_path, comments = "#")
        
        # store the column labels
        column_labels = []
        reached_output_format = False
        with open(out_path) as open_file:
            while True:
                line = open_file.readline()
                # break while statement if it is not a comment line
                # i.e. does not start with #
                if not line.startswith('#'):
                    break
                
                if line.startswith("#  IDENT"):
                    reached_output_format = True
                
                if reached_output_format:
                    if line.startswith("#########################"):
                        break
                    params_numbers = line.split(", ")
                    params_numbers[0] = params_numbers[0].replace("#  ","")
                    params_numbers.remove(params_numbers[-1])
                    #print(params_numbers)
                    for param_number in params_numbers:
                        output_param = param_number.split("  ")
                        #print(output_param)
                        column_labels.append(output_param[0])   
            open_file.close()

        # write data to a .fits file
        fits_columns = []
        for i in range(len(column_labels)):
            loc_col = fits.Column(name  = column_labels[i], array = np.array((txt_in.T)[i]), format = "D")
            fits_columns.append(loc_col)
        fits_table = fits.BinTableHDU.from_columns(fits_columns)
        fits_table.writeto(fits_out_path, overwrite = True)
        return fits_out_path
    
    def out_fits_name(self, out_path):
        return out_path.replace(".out", "_LePhare.fits")
    
    def extract_SEDs(self, cat_path, ID, units = u.ABmag, low_z_run = False):
        pass
    
    def extract_z_PDF(self, cat_path, ID, low_z_run = False):
        str_ID = "0" * (9 - len(str(ID))) + str(ID)
        print("ID = " + str_ID)

        z = []
        PDF = []
        reached_output_format = False
        with open(self.z_PDF_path_from_cat_path(cat_path, ID, low_z_run)) as open_file:
            while True:
                line = open_file.readline()
                # start at z = 0
                if line.startswith("  0.00000"):
                    reached_output_format = True
                if reached_output_format:
                    line = line.replace("  "," ")
                    line = line.replace("\n", "")
                    z_PDF = line.split(" ")
                    z_PDF.remove(z_PDF[0])
                    z.append(float(z_PDF[0]))
                    PDF.append(float(z_PDF[1]))
                
                # end at z = 15
                if line.startswith(" 25.00000"):
                    break
            open_file.close()
        return z, PDF
    
    def z_PDF_path_from_cat_path(self, cat_path, ID, low_z_run = False):
        pass
    
    def SED_path_from_cat_path(self, cat_path, ID, low_z_run = False):
        pass

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:52:36 2023

@author: austind
"""

# EAZY.py
import numpy as np
import astropy.units as u
import itertools
from astropy.table import Table
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import eazy
import os
import warnings
from astropy.utils.exceptions import AstropyWarning
from tqdm import tqdm
import h5py
from eazy import hdf5, visualization
from astropy.io import fits

from . import SED_code, Instrument
from . import useful_funcs_austind as funcs
from . import config, galfind_logger
from .decorators import run_in_dir, hour_timer, email_update

# %% EAZY SED fitting code

EAZY_FILTER_CODES = {'NIRCam': {'F070W':36, 'F090W':1, 'F115W':2,'F140M':37, 'F150W':3,'F162M':38, 'F182M':39, 'F200W':4, 'F210M':40, 
                                'F250M':41, 'F277W':5, 'F300M':42, 'F335M':43, 'F356W':6,'F360M':44, 'F410M':7, 'F430M':45, 'F444W':8, 'F460M':46, 'F480M':47},
                    'ACS_WFC':{'F435W':22, 'F606W':23, 'F814W':24,'F105W':25,'F125W':26, 'F140W':27,'F150W':28}, 
                    'MIRI': {'F560W':13, 'F770W':14, 'F1000W':15, 'F1130W':16, 'F1280W':17, 'F1500W':18,'F1800W':19, 'F2100W':20, 'F2550W':21}}

class EAZY(SED_code):
    
    def __init__(self):
        #ID_label = "IDENT"
        # now includes UBVJ flux/errs
        galaxy_property_dict = {**{"z_phot": "zbest", "chi_sq": "chi2_best"}, \
            **{f"{ubvj_filt}_{flux_or_err}": f"{ubvj_filt}_rf_{flux_or_err}" \
            for ubvj_filt in ["U", "B", "V", "J"] for flux_or_err in ["flux", "flux_err"]}}
        available_templates = ["fsps", "fsps_larson", "fsps_jades"]
        super().__init__(galaxy_property_dict, available_templates)
    
    def make_in(self, cat, fix_z = False): #, *args, **kwargs):
        eazy_in_dir = f"{config['EAZY']['EAZY_DIR']}/input/{cat.instrument.name}/{cat.version}/{cat.survey}"
        eazy_in_path = f"{eazy_in_dir}/{cat.cat_name.replace('.fits', '.in')}"
        if not Path(eazy_in_path).is_file():
            # 1) obtain input data
            IDs = np.array([gal.ID for gal in cat.gals]) # load IDs
            
            # load redshifts
            if not fix_z:
                redshifts = np.array([-99. for gal in cat.gals])
            else:
                redshifts = None
            # Define SED input bands on the fly
            SED_input_bands = cat.instrument.band_names
            # load photometry 
            phot, phot_err = self.load_photometry(cat, SED_input_bands, u.uJy, -99., None)
            # Get filter codes (referenced to GALFIND/EAZY/jwst_nircam_FILTER.RES.info) for the given instrument and bands
            filt_codes = [EAZY_FILTER_CODES[cat.instrument.instrument_from_band(band).name][band] for band in SED_input_bands]
            
            # Make input file
            #print(IDs, phot, phot_err, redshifts, len(IDs), len(phot), len(phot_err), len(redshifts))
            in_data = np.array([np.concatenate(([IDs[i]], list(itertools.chain(*zip(phot[i], phot_err[i]))), [redshifts[i]]), axis = None) for i in range(len(IDs))])
            in_names = ["ID"] + list(itertools.chain(*zip([f'F{filt_code}' for filt_code in filt_codes], [f'E{filt_code}' for filt_code in filt_codes]))) + ["z_spec"]
            #print(in_names)
            in_types = [int] + list(np.full(len(SED_input_bands) * 2, float)) + [float]
            in_tab = Table(in_data, dtype = in_types, names = in_names)
            funcs.make_dirs(eazy_in_path)
            in_tab.write(eazy_in_path, format = "ascii.commented_header", delimiter = " ", overwrite = True)
            #print(in_tab)
        return eazy_in_path
    
    @run_in_dir(path = config['EAZY']['EAZY_DIR'])
    def run_fit(self, in_path, fits_out_path, instrument, default_templates = 'fsps_larson', fix_z = False, n_proc = 1, z_step = 0.01, z_min = 0, z_max = 25,
                save_best_seds = config.getboolean('EAZY', 'SAVE_SEDS'), save_PDFs = config.getboolean('EAZY', 'SAVE_SEDS'), write_hdf = True, save_plots = False, plot_ids = None, plot_all = False, save_ubvj = True, run_lowz = True, \
                    lowz_zmax = None, wav_unit = u.AA, flux_unit = u.nJy, overwrite = False, *args, **kwargs):
        '''
        in_path - input EAZY catalogue path
        fits_out_path - output EAZY catalogue path
        template - which EAZY template to use - see below for list
        fix_z - whether to fix photo-z or not 
        z_step  - redshift step size - default 0.01
        z_min - minimum redshift to fit - default 0 
        z_max - maximum redshift to fit - default 25.
        save_best_seds - whether to write out best-fitting SEDs. Default True.
        save_PDFs - Whether to write out redshift PDF. Default True.
        write_hdf - whether to backup output to hdf5 - default True
        save_plots - whether to save SED plots - default False. Use in conjunction with plot_ids to plot SEDS of specific ids.
        plot_ids - list of ids to plot if save_plots is True.
        plot_all - whether to plot all SEDs. Default False.
        save_ubvj - whether to save restframe UBVJ fluxes -default True.
        run_lowz - whether to run low-z fit. Default True.
        lowz_zmax_arr - maximum redshifts to fit in low-z fits. Default [4., 6., None]
        **kwargs - additional arguments to pass to EAZY to overide defaults
        '''
        # Change this to config file path
        # This if/else tree chooses which template file to use based on 'templates' argument
        # FSPS - default EAZY templates, good allrounders
        # fsps_larson - default here, optimized for high redshift (see Larson et al. 2023)
        # HOT_45K - modified IMF high-z templates for use between 8 < z < 12
        # HOT_60K - modified IMF high-z templates for use at z > 12
        # Nakajima - unobscured AGN templates

        # update templates from within kwargs
        try:
            templates = kwargs.get("templates")
        except:
            templates = default_templates

        os.makedirs("/".join(fits_out_path.split("/")[:-1]), exist_ok = True)
        h5_path = fits_out_path.replace('.fits', '.h5')
        zPDF_path = h5_path.replace(".h5", "_zPDFs.h5")
        SED_path = h5_path.replace(".h5", "_SEDs.h5")
        lowz_label = funcs.lowz_label(lowz_zmax)

        eazy_templates_path = config['EAZY']['EAZY_TEMPLATE_DIR']
        default_param_path = f"{config['DEFAULT']['GALFIND_DIR']}/configs/zphot.param.default"
        translate_file = f"{config['DEFAULT']['GALFIND_DIR']}/configs/zphot_jwst.translate"
        params = {}
        if templates == 'fsps_larson':
            params['TEMPLATES_FILE'] = f'{eazy_templates_path}/LarsonTemplates/tweak_fsps_QSF_12_v3_newtemplates.param'
        elif templates == 'BC03':
            # This path is broken
            params['TEMPLATES_FILE'] = f"{eazy_templates_path}/bc03_chabrier_2003.param"
        elif templates == 'fsps':
            params['TEMPLATES_FILE'] = f"{eazy_templates_path}/fsps_full/tweak_fsps_QSF_12_v3.param"
        elif templates == 'nakajima_full':
            params['TEMPLATES_FILE'] = f"{eazy_templates_path}/Nakajima2022/tweak_fsps_QSF_12_v3_larson_nakajima_all.param"
        elif templates == 'nakajima_subset':
            params['TEMPLATES_FILE'] = f"{eazy_templates_path}/Nakajima2022/tweak_fsps_QSF_12_v3_larson_nakajima_subset.param"
        elif templates == 'fsps_jades':
            params['TEMPLATES_FILE'] = f"{eazy_templates_path}/jades/jades.param"
        elif templates == 'HOT_45K':
            params['TEMPLATES_FILE'] = f"{eazy_templates_path}/fsps-hot/45k/fsps_45k.param"
            z_min = 8
            z_max = 12
            #print(f'Running HOT 45K with fixed redshift = {fix_z}')
            if not fix_z:
                #print('Fixing 8<z<12')
                pass
        elif templates=='HOT_60K':
            params['TEMPLATES_FILE'] = f"{eazy_templates_path}/fsps-hot/60k/fsps_60k.param"
            z_min = 12
            z_max = 25
            #print(f'Running HOT 45K with fixed redshift = {fix_z}')
            if not fix_z:
                #print('Fixing 12<z<25')
                pass
        
        # Next section deals with passing config parameters into EAZY config dictionary
        # JWST filter_file
        params["FILTERS_RES"] = f"{config['DEFAULT']['GALFIND_DIR']}/configs/jwst_nircam_FILTER.RES"

        # Redshift limits
        params['Z_STEP'] = z_step # Setting photo-z step
        params['Z_MIN'] = z_min # Setting minimum Z
        if lowz_zmax == None:
            params['Z_MAX'] = z_max # Setting maximum Z
        else:
            params['Z_MAX'] = lowz_zmax # Setting maximum Z

        # Errors
        params['WAVELENGTH_FILE'] = f"{eazy_templates_path}/lambda.def"  # Wavelength grid definition file
        params['TEMP_ERR_FILE'] = f"{eazy_templates_path}/TEMPLATE_ERROR.eazy_v1.0" # Template error definition file
        
        # Priors
        params['FIX_ZSPEC'] = fix_z # Fix redshift to catalog zspec
    
        # Input files
        #-------------------------------------------------------------------------------------------------------------
        
        # Defining in/out files
        params['CATALOG_FILE'] = in_path
        params['MAIN_OUTPUT_FILE'] = fits_out_path
        params['OUTPUT_DIRECTORY'] = '/'.join(fits_out_path.split('/')[:-1])
        
        # Pass in optional arguments
        params.update(kwargs)
            
        if not Path(h5_path).is_file() or overwrite:
            # Initialize photo-z object with above parameters
            galfind_logger.info(f'Running {self.__class__.__name__} {templates} {lowz_label}')
            fit = eazy.photoz.PhotoZ(param_file = default_param_path, zeropoint_file = None,
                params = params, load_prior = False, load_products = False, translate_file = translate_file, n_proc = n_proc)
            fit.fit_catalog(n_proc = n_proc, get_best_fit = True)
            # Save backup of fit in hdf5 file
            hdf5.write_hdf5(fit, h5file = h5_path, include_fit_coeffs = False, include_templates = True, verbose = False)
            galfind_logger.info(f'Finished running {self.__class__.__name__} {templates} {lowz_label}')
        elif not Path(fits_out_path).is_file() or not Path(zPDF_path).is_file() or not Path(SED_path).is_file():
            # load in .h5 file
            fit = hdf5.initialize_from_hdf5(h5file = h5_path, verbose = True)
        else:
            fit = None

        if not Path(fits_out_path).is_file() and fit != None:
            # If not using Fsps larson, use standard saving output. Otherwise generate own fits file.
            if templates == 'HOT_45K' or templates == 'HOT_60K':
                fit.standard_output(UBVJ=(9, 10, 11, 12), absmag_filters=[9, 10, 11, 12], extra_rf_filters=[9, 10, 11, 12] ,n_proc=n_proc, save_fits=1, get_err=True, simple=False)
            else:
                colnames = ['IDENT', 'zbest', 'zbest_16', 'zbest_84', 'chi2_best']
                data = [fit.OBJID, fit.zbest, fit.pz_percentiles([16]), fit.pz_percentiles([84]), fit.chi2_best]

                table = Table(data = data, names = colnames)
            
                # Get rest frame colors
                if save_ubvj:
                    # This is all duplicated from base code.
                    rf_tempfilt, lc_rest, ubvj = fit.rest_frame_fluxes(f_numbers = [9, 10, 11, 12], simple = False, n_proc = n_proc)
                    for i, ubvj_filt in enumerate(["U", "B", "V", "J"]):
                        table[f"{ubvj_filt}_rf_flux"] = ubvj[:, i, 2]
                        # symmetric errors
                        table[f"{ubvj_filt}_rf_flux_err"] = (ubvj[:, i, 3] - ubvj[:, i, 1]) / 2.
                    galfind_logger.info(f'Finished calculating UBVJ fluxes for {self.__class__.__name__} {templates} {lowz_label}')
                    
                # add the template name to the column labels except for IDENT
                for col_name in table.colnames:
                    if col_name != "IDENT":
                        table.rename_column(col_name, f"{col_name}_{templates}_{lowz_label}")
                # Write fits file
                table.write(fits_out_path, overwrite = True)
                galfind_logger.info(f'Written {self.__class__.__name__} {templates} {lowz_label} fits out file to: {fits_out_path}')

        # save PDFs in h5 file
        if save_PDFs and not Path(zPDF_path).is_file():
            pz = 10 ** (fit.lnp)
            hf = h5py.File(zPDF_path, "w")
            hf.create_dataset("z", data = np.array(fit.zgrid))
            print(fit.zgrid, pz[0], len(fit.zgrid), len(pz[0]))
            [self.save_zPDF(pos_obj, ID, hf, fit.zgrid, pz) for pos_obj, ID in \
                tqdm(enumerate(fit.OBJID), total = len(fit.OBJID), \
                desc = f"Saving z-PDFs for {self.__class__.__name__} {templates} {lowz_label}")]
            hf.close()
            galfind_logger.info(f'Finished saving z-PDFs for {self.__class__.__name__} {templates} {lowz_label}')
        
        # Save best-fitting SEDs
        if save_best_seds and not Path(SED_path).is_file():
            hf = h5py.File(SED_path, "w")
            print(wav_unit, flux_unit)
            hf.create_dataset("wav_unit", data = str(wav_unit))
            hf.create_dataset("flux_unit", data = str(flux_unit))
            [self.save_SED(ID, z, hf, fit, wav_unit = wav_unit, flux_unit = flux_unit) \
                for ID, z in tqdm(zip(fit.OBJID, np.array(table[f"zbest_{templates}_{lowz_label}"]).astype(float)), total = len(fit.OBJID), \
                desc = f"Saving best-fit template SEDs for {self.__class__.__name__} {templates} {lowz_label}")]
            hf.close()
            galfind_logger.info(f'Finished saving SEDss for {self.__class__.__name__} {templates} {lowz_label}')

        # Write used parameters
        if fit != None:
            fit.param.write(fits_out_path.replace(".fits", "_params.csv"))
            galfind_logger.info(f'Written output pararmeters for {self.__class__.__name__} {templates} {lowz_label}')

    @staticmethod
    def save_zPDF(pos_obj, ID, hf, fit_zgrid, fit_pz):
        gal_zPDF = hf.create_group(f"ID={int(ID)}")
        gal_zPDF.create_dataset("p(z)", data = np.array([fit_pz[pos_obj][pos] for pos, z in enumerate(fit_zgrid)]))

    @staticmethod
    def save_SED(ID, z, hf, fit, wav_unit = u.AA, flux_unit = u.nJy):
        # Load best-fitting SED
        fit_data = fit.show_fit(ID, id_is_idx = False, show_components = False, \
            show_prior = False, logpz = False, get_spec = True, show_fnu = 1)
        wav = (np.array(fit_data['templz']) * fit_data['wave_unit']).to(wav_unit)
        flux = (np.array(fit_data['templf']) * fit_data['flux_unit']).to(flux_unit)
        gal_SED = hf.create_group(f"ID={int(fit_data['id'])}")
        gal_SED.create_dataset("z", data = z)
        gal_SED.create_dataset("wav", data = wav)
        gal_SED.create_dataset("flux", data = flux)

    def make_fits_from_out(self, out_path, templates, lowz_zmax): #*args, **kwargs):
        pass
    
    def out_fits_name(self, out_path, templates, lowz_zmax): #*args, **kwargs):
        fits_out_path = f"{out_path.replace('.out', '')}_EAZY_{templates}_{funcs.lowz_label(lowz_zmax)}.fits"
        return fits_out_path
    
    def extract_SEDs(self, fits_cat, ID, low_z_run = False, units = u.ABmag, just_header = False):
        SED_path = self.SED_path_from_cat_path(fits_cat.meta[f"{self.__class__.__name__}_path"], ID, low_z_run)
       
        if not Path(SED_path).is_file():
            print(f'Not found EAZY SED at {SED_path}')
        if not just_header: 
            SED = Table.read(SED_path, format = 'ascii.no_header', delimiter = '\s', names = ['wav', 'mag'], data_start = 0)
            SED['mag'][np.isinf(SED['mag'])] = 99.
        return {"best_gal": SED}
    
    def extract_z_PDF(self, fits_cat, ID, low_z_run = False):
        PDF_path = self.z_PDF_path_from_cat_path(fits_cat.meta[f"{self.__class__.__name__}_path"], ID, low_z_run)
        try:
            z, PDF = np.loadtxt(PDF_path, delimiter = ',').T  
        except FileNotFoundError:
            print(f'{PDF_path} not found.')
            return None, None
        return z, PDF
        
    def get_z_PDF_path(self, cat, ID, templates, lowz_zmax):
        PDF_dir = f"{config['EAZY']['EAZY_DIR']}/output/{cat.instrument.name}/{cat.version}/{cat.survey}"
        PDF_name = f"{cat.cat_name.replace('.fits', f'_EAZY_{templates}_{funcs.lowz_label(lowz_zmax)}_zPDFs.h5')}"
        return f"{PDF_dir}/{PDF_name}"
    
    def get_SED_path(self, cat, ID, templates, lowz_zmax):
        # should still include aper_diam here
        min_flux_pc_err = str(cat.cat_path.replace(f"_{templates}", "").split("_")[-2].replace("pc", ""))
        SED_dir = f"{funcs.split_dir_name(cat.cat_path, 'dir')}SEDs/{str(min_flux_pc_err)}pc/{templates}"
        if lowz_label == "zfree":
            lowz_label = ""
        SED_name = f"{str(ID)}{lowz_label}.spec"
        return f"{SED_dir}/{SED_name}"

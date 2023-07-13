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
from eazy import hdf5, visualization
from astropy.io import fits

from . import SED_code, Instrument
from . import useful_funcs_austind as funcs
from . import config
from .decorators import run_in_dir, hour_timer, email_update

# %% EAZY SED fitting code

EAZY_FILTER_CODES = {'NIRCam': {'f070W':36, 'f090W':1, 'f115W':2,'f140M':37, 'f150W':3,'f162M':38, 'f182M':39, 'f200W':4, 'f210M':40, 
                                'f250M':41, 'f277W':5, 'f300M':42, 'f335M':43, 'f356W':6,'f360M':44, 'f410M':7, 'f430M':45, 'f444W':8, 'f460M':46, 'f480M':47},
                    'ACS_WFC':{'f435W':22, 'f606W':23, 'f814W':24,'f105W':25,'f125W':26, 'f140W':27,'f150W':28}, 
                    'MIRI': {'f0560W':13, 'f0770W':14, 'f1000W':15, 'f1130W':16, 'f1280W':17, 'f1500W':18,'f1800W':19, 'f2100W':20, 'f2550W':21}}
# NOT SO SURE THAT f150W ACS WFC EXISTS!

class EAZY(SED_code):
    
    def __init__(self, templates = "fsps_larson", low_z_run = False):
        code_name = "EAZY"
        ID_label = "IDENT"
        galaxy_property_labels = {"z_phot": "zbest"}
        chi_sq_labels = {}
        self.templates = templates
        super().__init__(code_name, ID_label, galaxy_property_labels, chi_sq_labels, low_z_run)
    
    def from_name(self):
        return EAZY()
    
    def make_in(self, cat, fix_z = False, *args, **kwargs):
        print("MAKE_IN_EAZY_CAT.DATA = ", cat.data)
        eazy_in_path = f"{self.code_dir}/input/{cat.data.instrument.name}/{cat.data.version}/{cat.data.survey}/{cat.cat_name.replace('.fits', '')}_{cat.cat_creator.min_flux_pc_err}pc.in"
        if not Path(eazy_in_path).is_file():
            # 1) obtain input data
            IDs = np.array([gal.ID for gal in cat.gals]) # load IDs
            # load redshifts
            if not fix_z:
                redshifts = np.array([-99. for gal in cat.gals])
            else:
                redshifts = None
            # Define SED input bands on the fly
            SED_input_bands = cat.data.instrument.bands
            # load photometry 
            phot, phot_err = self.load_photometry(cat, SED_input_bands, u.uJy, -99., None)
            # Get filter codes (referenced to GALFIND/EAZY/jwst_nircam_FILTER.RES.info) for the given instrument and bands
            filt_codes = [EAZY_FILTER_CODES[cat.data.instrument.instrument_from_band(band)][band] for band in SED_input_bands]
            
            # Make input file
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
    def run_fit(self, in_path, out_path, sed_folder, instrument, default_templates = 'fsps_larson', fix_z = False, n_proc=6, z_step = 0.01, z_min=0, z_max =25,
                save_best_seds = True, save_pz = True, write_hdf = True, save_plots = False, plot_ids = None, plot_all = False, save_ubvj = True, run_lowz = True, \
                    z_max_lowz=7, *args, **kwargs):
        '''
        in_path - input EAZY catalogue path
        out_path - output EAZY catalogue path - currently modified by code, needs updating
        sed_folder - folder for SEDs
        template - which EAZY template to use - see below for list
        fix_z - whether to fix photo-z or not 
        z_step  - redshift step size - default 0.01
        z_min - minimum redshift to fit - default 0 
        z_max - maximum redshift to fit - default 25.
        save_best_seds - whether to write out best-fitting SEDs. Default True.
        save_pz - Whether to write out redshift PDF. Default True.
        write_hdf - whether to backup output to hdf5 - default True
        save_plots - whether to save SED plots - default False. Use in conjunction with plot_ids to plot SEDS of specific ids.
        plot_ids - list of ids to plot if save_plots is True.
        plot_all - whether to plot all SEDs. Default False.
        save_ubvj - whether to save restframe UBVJ fluxes -default True.
        run_lowz - whether to run low-z fit. Default True.
        z_max_lowz - maximum redshift to fit in low-z fit. Default 7.
        **kwargs - additional arguments to pass to EAZY to overide defaults
        '''
        # Change this to config file path
        # This if/else tree chooses which template file to use based on 'templates' argument
        # FSPS - default EAZY templates, good allrounders
        # fsps_larson - default here, optimized for high redshift (see Larson et al. 2022)
        # HOT_45K - modified IMF high-z templates for use between 8 < z < 12
        # HOT_60K - modified IMF high-z templates for use at z > 12
        # Nakajima - unobscured AGN templates
        
        # update templates from within kwargs
        try:
            templates = kwargs.get("templates")
        except:
            print(f"Using default EAZY templates = {default_templates}!")
            templates = default_templates
        
        path = config['EAZY']['EAZY_DIR']
        eazy_templates_path =  config['EAZY']['EAZY_TEMPLATE_DIR']
        default_param_path = f"{config['DEFAULT']['GALFIND_DIR']}/configs/zphot.param.default"
        translate_file = f"{config['DEFAULT']['GALFIND_DIR']}/configs/zphot_jwst.translate"
        #param_file = eazy.param.read_param_file(default_param_path)
        params = {}
        if templates == 'fsps_larson':
            params['TEMPLATES_FILE'] =f'{eazy_templates_path}/LarsonTemplates/tweak_fsps_QSF_12_v3_newtemplates.param'
        elif templates == 'BC03':
            # This path is broken
            params['TEMPLATES_FILE'] =  f"{eazy_templates_path}/bc03_chabrier_2003.param"
        elif templates == 'fsps':
            params['TEMPLATES_FILE'] =  f"{eazy_templates_path}/fsps_full/tweak_fsps_QSF_12_v3.param"
        elif templates == 'nakajima_full':
            params['TEMPLATES_FILE'] =  f"{eazy_templates_path}/Nakajima2022/tweak_fsps_QSF_12_v3_larson_nakajima_all.param"
        elif templates == 'nakajima_subset':
            params['TEMPLATES_FILE'] =  f"{eazy_templates_path}/Nakajima2022/tweak_fsps_QSF_12_v3_larson_nakajima_subset.param"
        elif templates == 'jades':
            params['TEMPLATES_FILE'] = f"{eazy_templates_path}/inputs/templates/jades/jades.param"
        elif templates == 'HOT_45K':
            params['TEMPLATES_FILE'] = f"{eazy_templates_path}/fsps-hot/45k/fsps_45k.param"
            z_min = 8
            z_max = 12
            print(f'Running HOT 45K with fixed redshift = {fix_z}')
            if not fix_z:
                print('Fixing 8<z<12')
        elif templates=='HOT_60K':
            params['TEMPLATES_FILE'] =  f"{eazy_templates_path}/inputs/templates/fsps-hot/60k/fsps_60k.param"
            z_min = 12
            z_max = 25
            print(f'Running HOT 45K with fixed redshift = {fix_z}')
            if not fix_z:
                print('Fixing 12<z<25')
        
        # Next section deals with passing config parameters into EAZY config dictionary
        # JWST filter_file
        params["FILTERS_RES"] = f"{config['DEFAULT']['GALFIND_DIR']}/configs/jwst_nircam_FILTER.RES"

        # Redshift stuff
        params['Z_STEP'] = z_step # Setting photo-z step
        params['Z_MIN'] = z_min # Setting minimum Z
        params['Z_MAX'] = z_max # Setting maximum Z

        # Errors
        params['WAVELENGTH_FILE'] = f"{eazy_templates_path}/lambda.def"  # Wavelength grid definition file
        params['TEMP_ERR_FILE'] = f"{eazy_templates_path}/TEMPLATE_ERROR.eazy_v1.0" # Template error definition file
        
        # Priors
        params['FIX_ZSPEC'] = fix_z # Fix redshift to catalog zspec
       
        # Input files
        #-------------------------------------------------------------------------------------------------------------
        
        params['CATALOG_FILE'] = in_path
        # Defining outfiles - top fixes path of out file, second, adds template name to filename
        fits_out_path = self.out_fits_name(out_path, *args, **kwargs)
        # Setting output directory
        out_directory = '/'.join(fits_out_path.split('/')[:-1])
        params['OUTPUT_DIRECTORY'] = out_directory
        params['MAIN_OUTPUT_FILE'] = fits_out_path
    
        h5path = fits_out_path.replace('.fits', '.h5')
        
        # Pass in optional arguments
        params.update(kwargs)
        
        # Catch custom arguments?
        # Initialize photo-z object with above parameters
        fit = eazy.photoz.PhotoZ(param_file = default_param_path, zeropoint_file = None,
                                params = params, load_prior = False, load_products = False, translate_file = translate_file)
        # Fit templates to catalog                          
        fit.fit_catalog(n_proc = n_proc, get_best_fit = True)
        
        params['Z_MAX'] = z_max_lowz # Setting maximum Z

        if run_lowz:
            lowz_fit = eazy.photoz.PhotoZ(param_file = default_param_path,  zeropoint_file = None,
                                    params = params, load_prior = False, load_products = False, translate_file = translate_file)
            lowz_fit.fit_catalog(n_proc = n_proc, get_best_fit = True)

        if plot_all:
            save_plots = True
            ids_to_plot = fit.OBJID
        if save_plots:
            # Make output directory if it doesn't exist
            out_path_plots = out_directory + '/plots/'
            if not os.path.exists(out_path_plots):
                os.makedirs(out_path_plots)
            # Make plot for each object, save fit and close
            for i in ids_to_plot:
                fit.show_fit(i, show_fnu=1)
                plt.savefig(f"{out_path_plots}/{i}_{templates}.png",)
                plt.close()   

        # Save backup of fit in hdf5 file
        if write_hdf:
            hdf5.write_hdf5(fit, h5file=h5path, include_fit_coeffs=False, include_templates=True, verbose=False)
        # If not using Fsps larson, use standard saving output. Otherwise generate own fits file.
        if templates == 'fsps' or templates == 'HOT_45K' or templates == 'HOT_60K':
            fit.standard_output(UBVJ=(9, 10, 11, 12), absmag_filters=[9, 10, 11, 12], extra_rf_filters=[9, 10, 11, 12] ,n_proc=n_proc, save_fits=1, get_err=True, simple=False)
            lowz_fit.standard_output(UBVJ=(9, 10, 11, 12), absmag_filters=[9, 10, 11, 12], extra_rf_filters=[9, 10, 11, 12] ,n_proc=n_proc, save_fits=1, get_err=True, simple=False)
        else:
            colnames = ['IDENT', 'zbest', 'zbest_16', 'zbest_84', 'chi2_best']
            data = [fit.OBJID, fit.zbest,fit.pz_percentiles([16]), fit.pz_percentiles([84]), fit.chi2_best ]
            if run_lowz:
                data += [lowz_fit.zbest,lowz_fit.pz_percentiles([16]), lowz_fit.pz_percentiles([84]), lowz_fit.chi2_best ]
                colnames += ['zbest_lowz', 'zbest_16_lowz', 'zbest_84_lowz', 'chi2_best_lowz']

            table = Table(data = data, names = colnames)
           
            # Get rest frame colors
            if save_ubvj:
                # This is all duplicated from base code.
                rf_tempfilt, lc_rest, ubvj = fit.rest_frame_fluxes(f_numbers=[9, 10, 11, 12], simple=False, n_proc=n_proc)

                table['U_rf_flux'] = ubvj[:,0,2]
                table['U_rf_flux_err'] = (ubvj[:,0,3] - ubvj[:,0,1])/2.

                table['B_rf_flux'] = ubvj[:,1,2]
                table['B_rf_flux_err'] = (ubvj[:,1,3] - ubvj[:,1,1])/2.
               
                table['V_rf_flux'] = ubvj[:,2,2]
                table['V_rf_flux_err'] = (ubvj[:,2,3] - ubvj[:,2,1])/2.
               
                table['J_rf_flux'] = ubvj[:,3,2]
                table['J_rf_flux_err'] = (ubvj[:,3,3] - ubvj[:,3,1])/2.
               
        if save_pz:
            # Make folders if they don't exist
            out_path_pdf = sed_folder.replace("SEDs", "PDFs")
            if not os.path.exists(out_path_pdf):
                os.makedirs(out_path_pdf)
            out_path_pdf_template = f'{out_path_pdf}/{templates}'
            if not os.path.exists(out_path_pdf_template):
                os.makedirs(out_path_pdf_template)
            # Generate PDF
            pz = 10 ** (fit.lnp)
            lowz_pz = 10 ** (lowz_fit.lnp)
            # Save PDFs in loop
            for pos_obj, i in enumerate(fit.OBJID):
                with open(f'{out_path_pdf_template}/{i}.pz', "w") as pz_save:
                    for pos, z in enumerate(fit.zgrid):
                        pz_save.write(f"{z}, {pz[pos_obj][pos]}\n")
                if run_lowz:
                    with open(f'{out_path_pdf_template}/{i}_lowz.pz', "w") as pz_save:
                        for pos, z in enumerate(lowz_fit.zgrid):
                            pz_save.write(f"{z}, {lowz_pz[pos_obj][pos]}\n")
        # Save best-fitting SEDs
        if save_best_seds:
            percentiles = fit.pz_percentiles([16, 84])
            if run_lowz:
                percentiles_lowz = lowz_fit.pz_percentiles([16, 84])
            else:
                percentiles_lowz = False
            [self.save_sed(id, fit, lowz_fit, percentiles, percentiles_lowz, templates, sed_folder) for id in tqdm(fit.OBJID, total = len(fit.OBJID), desc = "Saving best template SEDs")]
            print('Saved best SEDs')

        # Write used parameters
        fit.param.write(fits_out_path.replace(".fits", "_params.csv"))
        print(f'Finished running EAZY!')
        
        # Write fits file
        table.write(fits_out_path, overwrite=True)
        print(f'Written out file to: {fits_out_path}')


    def save_sed(self, id, fit, lowz_fit, percentiles, percentiles_lowz, templates, out_path):
        # Find location of matching Id
        pos = [fit.OBJID == id]
        # Find percentiles
        percentiles_run = percentiles[pos]
        percentiles_run = (percentiles_run[0][0], percentiles_run[0][1])

        self.save_fit(id, fit, out_path=out_path, percentiles_run=percentiles_run, out_flux_unit='mag',  template=templates)
        
        if type(percentiles_lowz) != bool:
            percentiles_run_lowz = percentiles_lowz[pos]
            percentiles_run_lowz = (percentiles_run_lowz[0][0], percentiles_run_lowz[0][1])
            self.save_fit(id, lowz_fit, out_path=out_path, percentiles_run=percentiles_run_lowz, out_flux_unit='mag',  template=templates, lowz=True)

    def save_fit(self, id, photz_obj, percentiles_run=[], out_flux_unit='mag', id_is_idx=False,template='BC03', out_path='', lowz=False):
        # Generate best-fitting SED
        data = photz_obj.show_fit(id, id_is_idx=id_is_idx, show_components=False, show_prior=False, logpz=False,  get_spec=True, show_fnu = 1)
        # Get info from data object
        id_phot = data['id']
        z_best = data['z']
        chi2 = data['chi2']
        flux_unit = data['flux_unit']
        wav_unit = data['wave_unit']
        model_lam = data['templz'] * wav_unit
        model_flux = data['templf'] * flux_unit
        # Convert units of ouput
        if out_flux_unit == 'mag':
            model_flux_converted = -2.5 * np.log10(model_flux.to("Jy").value) + 8.90
        model_flux_converted[np.isinf(model_flux_converted)] = 99 
        # Construct output
        data_out = np.transpose(np.vstack((model_lam.value, model_flux_converted)))
        out_path = f'{out_path}/{template}/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if lowz:
            extra = '_lowz'
        else:
            extra = ''

        np.savetxt(f"{out_path}/{id_phot}{extra}.spec", data_out, delimiter="  ", header=f'ID  ZBEST  PERC_16  PERC_84  CHIBEST  WAV_UNIT  FLUX_UNIT\n{id_phot}  {z_best:.3f}  {float(percentiles_run[0]):.3f}  {float(percentiles_run[1]):.3f}  {chi2:.3f}  {wav_unit}  {out_flux_unit}')

    def make_fits_from_out(self, out_path, *args, **kwargs):
        # not required for EAZY
        pass
    
    def out_fits_name(self, out_path, *args, **kwargs):
        fits_out_path = out_path.replace('.out', '.fits')
        templates = kwargs.get('templates')
        fits_out_path = fits_out_path[:-5] + f"_eazy_{templates}" + fits_out_path[-5:] 
        return fits_out_path
    
    def extract_SEDs(self, cat_path, ID, low_z_run = False, units = u.ABmag, just_header = False):
        SED_path = self.SED_path_from_cat_path(cat_path, ID, low_z_run)
        if not Path(SED_path).is_file():
            print(f'Not found EAZY SED at {SED_path}')
        if not just_header: 
            SED = Table.read(SED_path, format = 'ascii.no_header', delimiter = '\s', names = ['wav', 'mag'], data_start = 0)
            SED['mag'][np.isinf(SED['mag'])] = 99.
        return {"best_gal": SED}
    
    def extract_z_PDF(self, cat_path, ID, low_z_run = False):
        PDF_path = self.PDF_path_from_cat_path(cat_path, ID, low_z_run)
        try:
            z, PDF = np.loadtxt(PDF_path, delimiter = ',').T  
        except FileNotFoundError:
            print(f'{PDF_path} not found.')
            return None, None
        return z, PDF
        
    def z_PDF_path_from_cat_path(self, cat_path, ID, low_z_run = False):
        # should still include aper_diam here
        min_flux_pc_err = str(cat_path.replace(f"_{self.templates}", "").split("_")[-2].replace("pc", ""))
        if low_z_run:
            low_z_name = "_lowz"
        else:
            low_z_name = ""
        PDF_dir = f"{funcs.split_dir_name(cat_path, 'dir')}/PDFs/{str(min_flux_pc_err)}pc/{self.templates}"
        PDF_name = f"{str(ID)}{low_z_name}.pz"
        return f"{PDF_dir}/{PDF_name}"
    
    def SED_path_from_cat_path(self, cat_path, ID, low_z_run = False):
        # should still include aper_diam here
        min_flux_pc_err = str(cat_path.replace(f"_{self.templates}", "").split("_")[-2].replace("pc", ""))
        if low_z_run:
            low_z_name = "_lowz"
        else:
            low_z_name = ""
        SED_dir = f"{funcs.split_dir_name(cat_path, 'dir')}/SEDs/{str(min_flux_pc_err)}pc/{self.templates}"
        SED_name = f"{str(ID)}{low_z_name}.pz"
        return f"{SED_dir}/{SED_name}"

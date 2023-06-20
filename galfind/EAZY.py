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

from . import SED_code
from . import useful_funcs_austind as funcs
from . import config

# %% EAZY SED fitting code

EAZY_FILTER_CODES = {'NIRCam': {'f090W':1, 'f115W':2, 'f150W':3, 'f200W':4, 'f277W':5, 'f356W':6, 'f410M':7,'f444W':8},
 'HST-ACS':{'f435W':22, 'f606W':23, 'f814W':24,'f105W':25,'f125W':26, 'f140W':27,'f150W':28}, 
 'MIRI': {'f0560W':13, 'f0770W':14, 'f1000W':15, 'f1130W':16, 'f1280W':17, 'f1500W':18,'f1800W':19, 'f2100W':20, 'f2550W':21}}

class EAZY(SED_code):
    
    def __init__(self):
        code_name = "EAZY"
        galaxy_property_labels = {"z": "zbest"}
        super().__init__(code_name, galaxy_property_labels)

    def make_in(self, cat, fix_z = False):
        eazy_in_path = f"{self.code_dir}/input/{cat.data.instrument.name}/{cat.data.version}/{cat.data.survey}/{cat.cat_name[:-5]}.in"
        if not Path(eazy_in_path).is_file():
            # 1) obtain input data
            IDs = np.array([gal.ID for gal in cat.gals]) # load IDs
            # load redshifts
            if not fix_z:
                redshifts = np.array([-99. for gal in cat.gals])
            else:
                redshifts = None
            # Define instrument
            SED_input_bands = cat.data.instrument.bands
            instrument_name = cat.data.instrument.name
            # load photometry 
            phot, phot_err = self.load_photometry(cat, SED_input_bands, u.uJy, -99., None)
            # Get filter codes (referenced to GALFIND/EAZY/jwst_nircam_FILTER.RES.info) for the given instrument and bands
            codes = [EAZY_FILTER_CODES[instrument_name][band] for band in SED_input_bands]
            
            # Make input file
            in_data = np.array([np.concatenate(([IDs[i]], list(itertools.chain(*zip(phot[i], phot_err[i]))), [redshifts[i]]), axis = None) for i in range(len(IDs))])
            in_names = ["ID"] + list(itertools.chain(*zip([f'F{code}' for code in codes], [f'E{code}' for code in codes]))) + ["z_spec"]
            print(in_names)
            in_types = [int] + list(np.full(len(SED_input_bands) * 2, float)) + [float]
            in_tab = Table(in_data, dtype = in_types, names = in_names)
            funcs.make_dirs(eazy_in_path)
            in_tab.write(eazy_in_path, format = "ascii.commented_header", delimiter = " ", overwrite = True)
            #print(in_tab)
        return eazy_in_path
    
    def run_fit(self, in_path, out_path, sed_folder, templates='fsps_larson', fix_z = False, n_proc=6, z_step = 0.01, z_min=0, z_max =25,
    save_best_seds = True, save_pz = True,write_hdf=True, save_plots=False, plot_ids=None, plot_all=False, save_ubvj = True):
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
        '''
        # Change this to config file path
        path = '/nvme/scratch/work/austind/GALFIND/EAZY/'
        
        # This if/else tree chooses which template file to use based on 'templates' argument
        # FSPS - default EAZY templates, good allrounders
        # fsps_larson - default here, optimized for high redshift (see Larson et al. 2022)
        # HOT_45K - modified IMF high-z templates for use between 8 < z < 12
        # HOT_60K - modified IMF high-z templates for use at z > 12
        # Nakajima - unobscured AGN templates
        params = {}
        if templates=='fsps_larson':
            params['TEMPLATES_FILE'] = os.path.join(path,"templates/LarsonTemplates/tweak_fsps_QSF_12_v3_newtemplates.param")
        elif templates=='BC03':
            params['TEMPLATES_FILE'] =  os.path.join(path,"templates/bc03_chabrier_2003.param")
        elif templates=='HOT_45K':
            params['TEMPLATES_FILE'] = os.path.join(path, f"templates/fsps-hot/45k/fsps_45k.param")
            z_min = 8
            z_max = 12
            print(f'Running HOT 45K with fixed redshift = {fix_z}')
            if not fix_z:
                print('Fixing 8<z<12')
        elif templates=='HOT_60K':
            params['TEMPLATES_FILE'] = os.path.join(path, f"inputs/templates/fsps-hot/60k/fsps_60k.param")
            z_min = 12
            z_max = 25
            print(f'Running HOT 45K with fixed redshift = {fix_z}')
            if not fix_z:
                print('Fixing 12<z<25')
        elif templates=='fsps':
            params['TEMPLATES_FILE'] = os.path.join(path,"templates/fsps_full/tweak_fsps_QSF_12_v3.param")
        elif templates=='nakajima_full':
            params['TEMPLATES_FILE'] = os.path.join(path,"templates/Nakajima2022/tweak_fsps_QSF_12_v3_larson_nakajima_all.param")
        elif templates=='nakajima_subset':
            params['TEMPLATES_FILE'] = os.path.join(path,"templates/Nakajima2022/tweak_fsps_QSF_12_v3_larson_nakajima_subset.param")
        
        # Next section deals with passing config parameters into EAZY config dictionary
        # JWST filter_file
        params["FILTERS_RES"] = os.path.join(path, 'jwst_nircam_FILTER.RES')

        # Galactic extinction
        params['MW_EBV'] = 0 # Setting MW E(B-V) extinction
        params['CAT_HAS_EXTCORR'] = False #Catalog already corrected for reddening?

        # Redshift stuff
        params['Z_STEP'] = z_step # Setting photo-z step
        params['Z_MIN'] = z_min # Setting minimum Z
        params['Z_MAX'] =z_max # Setting maxium Z

        # Errors
        params['WAVELENGTH_FILE'] = os.path.join(path, 'templates/lambda.def')  # Wavelength grid definition file
        params['TEMP_ERR_FILE'] = os.path.join(path, 'templates/TEMPLATE_ERROR.eazy_v1.0') # Template error definition file
        params['TEMP_ERR_A2']= 0 # Template error amplitude
        params['SYS_ERR'] = 0 

        # Priors
        params['APPLY_PRIOR'] = "n" # Apply priors?
        params['PRIOR_ABZP'] = 23.91 #25 # AB zeropoint of fluxes in catalog.  Needed for calculating apparent mags! This is for uJy
        params['PRIOR_FILTER'] = 28 # K #  # Filter from FILTER_RES corresponding to the columns in PRIOR_FILE
        params['PRIOR_FILE'] = '' # No prior used

        params['FIX_ZSPEC'] = fix_z # Fix redshift to catalog zspec
        params['IGM_SCALE_TAU'] = 1.0 # Scale factor times Inoue14 IGM tau
        # Min number of filters
        params['N_MIN_COLORS'] = 2 # Default is 5
        # Input files
        #-------------------------------------------------------------------------------------------------------------
        
        params['CATALOG_FILE'] = in_path
        # Defining outfiles - top fixes path of out file, second, adds template name to filename
        out_path = out_path.replace('.out', '.fits')
        out_path = out_path[:-5] + f'_eazy_{templates}' + out_path[-5:] 
        # Setting output directory
        out_directory = '/'.join(out_path.split('/')[:-1])
        params['OUTPUT_DIRECTORY'] = out_directory
        params['MAIN_OUTPUT_FILE'] = out_path
    
        h5path  = out_path.replace('.fits', '.h5')
        
        # Catch custom arguments?
        #params.update(custom_params)
        # Initialize photo-z object with above parameters
        fit = eazy.photoz.PhotoZ(param_file=None,  zeropoint_file=None, translate_file= params["FILTERS_RES"],
                                params=params, load_prior=False, load_products=False)
        # Fit templates to catalog                          
        fit.fit_catalog(n_proc=n_proc, get_best_fit=True)
        
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
                plt.savefig(out_path_plots+f"/{i}_{templates}.png",)
                plt.close()   

        # Save backup of fit in hdf5 file
        if write_hdf:
            hdf5.write_hdf5(fit, h5file=h5path, include_fit_coeffs=False, include_templates=True, verbose=False)
        # If not using Fsps larson, use standard saving output. Otherwise generate own fits file.
        if templates == 'fsps' or templates == 'HOT_45K' or templates == 'HOT_60K':
            fit.standard_output(UBVJ=(9, 10, 11, 12), absmag_filters=[9, 10, 11, 12], extra_rf_filters=[9, 10, 11, 12] ,n_proc=n_proc, save_fits=1, get_err=True, simple=False)
        else:
            colnames = ['IDENT', 'zbest', 'zbest_16', 'zbest_84', 'chi2_best']
            data = [fit.OBJID, fit.zbest,fit.pz_percentiles([16]), fit.pz_percentiles([84]), fit.chi2_best ]
            table = Table(data=data, names=colnames)
           
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
               
            # Write fits file
            
            table.write(out_path, overwrite=True)
            print(f'Written out file to: {out_path}')
        if save_pz:
            # Make folders if they don't exist
            out_path_pdf = f'{out_directory}/PDFs/'
            if not os.path.exists(out_path_pdf):
                os.makedirs(out_path_pdf)
            out_path_pdf_template = f'{out_path_pdf}/{templates}'
            if not os.path.exists(out_path_pdf_template):
                os.makedirs(out_path_pdf_template)
            # Generate PDF
            pz=10**(fit.lnp)
            # Save PDFs in loop
            for pos_obj, i in enumerate(fit.OBJID):
                with open(f'{out_path_pdf_template}/{i}.pz', "w") as pz_save:
                    for pos, z in enumerate(fit.zgrid):
                        pz_save.write(f"{z}, {pz[pos_obj][pos]} \n")
        # Save best-fitting SEDs
        if save_best_seds:
            out_path = f'{out_directory}/SEDs/{templates}/'
            print("Saving best template SEDs")
            percentiles = fit.pz_percentiles([16, 84])
            ids =  fit.OBJID
            for id in ids:
                self.save_sed(id, fit, percentiles, templates, sed_folder)
            
            print('Saved best SEDs')

        # Write used parameters
        fit.param.write(out_directory+f'param_used_just_{templates}.csv')

        print(f'Finished running EAZY.')

        return out_path

    def save_sed(self, id, fit, percentiles, templates, out_path):
        # Find location of matching Id
        pos = [fit.OBJID == id]
        # Find percentiles
        percentiles_run = percentiles[pos]
        percentiles_run = (percentiles_run[0][0], percentiles_run[0][1])
        self.save_fit(id, fit, out_path=out_path, percentiles_run=percentiles_run, out_flux_unit='mag',  template=templates)

    def save_fit(self, id,photz_obj, percentiles_run=[], out_flux_unit='mag', id_is_idx=False,template='BC03', out_path=''):
        # Generate best-fitting SED
        data = photz_obj.show_fit(id, id_is_idx=id_is_idx, show_components=False, show_prior=False, logpz=False,  get_spec=True, show_fnu = 1)
        # Get info from data object
        id_phot=data['id']
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
        np.savetxt(f"{out_path}/{id_phot}.spec", data_out, delimiter="  ", header=f'ID  ZBEST  PERC_16  PERC_84  CHIBEST  WAV_UNIT  FLUX_UNIT\n{id_phot}  {z_best:.3f}  {float(percentiles_run[0]):.3f}  {float(percentiles_run[1]):.3f}  {chi2:.3f}  {wav_unit}  {out_flux_unit}')

    def make_fits_from_out(self, out_path):
        return out_path.replace('.out', '.fits')
    
    def extract_SED(self, cat, ID, units = u.ABmag, templates = 'fsps_larson'):
        pass
    
    def extract_z_PDF(self, cat, ID, templates = 'fsps_larson'):
        path = '' #Make path
        try:
            z, pz  = np.loadtxt(path, delimiter=',').T  
        except FileNotFoundError:
            print('PDF not found.')
        
        return z, pz
        
# %%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:55:57 2023

@author: austind
"""

# Bagpipes.py
import astropy.units as u
import os
import numpy as np
import time
from astropy.table import Table
from typing import Union
from copy import deepcopy

from . import useful_funcs_austind as funcs
from . import SED_code, galfind_logger

# %% Bagpipes SED fitting code

class Bagpipes(SED_code):

    galaxy_property_dict = {}
    galaxy_property_errs_dict = {}
    available_templates = ["BC03", "BPASS"]
    
    def __init__(self):
        super().__init__(self.galaxy_property_dict, self.galaxy_property_errs_dict, self.available_templates)
    
    def make_in(self, cat):
        # no need for bagpipes input catalogue
        # return log file path
        return ""
    
    def run_fit(self, in_path, fits_out_path, instrument, SED_fit_params, ):
        pass
    
    def make_fits_from_out(self, out_path):
        pass
    
    def out_fits_name(self, out_path):
        pass
    
    def extract_SEDs(self, path):
        pass
    
    def extract_z_PDF(self, path):
        pass
    
    def z_PDF_path_from_cat_path(self, cat_path, ID, low_z_run = False):
        pass
    
    def SED_path_from_cat_path(self, cat_path, ID, low_z_run = False):
        pass
    
    @staticmethod
    def load_phot(ID: int, cat = None, verbose: bool = True):
        assert type(cat) != type(None) # default does not work in this case
        assert cat.__class__.__name__ in "Catalogue"

        #start = time.time()
        
        # get appropriate galaxy photometry from catalogue
        phot_obj = cat.crop(ID, "ID")[0].phot
        # extract bands to be used
        bands = phot_obj.instrument.band_names
        # extract fluxes and errors in uJy
        band_wavs = np.array([band.WavelengthCen for band in phot_obj.instrument])
        flux = funcs.convert_mag_units(band_wavs, phot_obj.flux_Jy, u.uJy)
        flux_errs = funcs.convert_mag_err_units(band_wavs, phot_obj.flux_Jy, phot_obj.flux_Jy_errs, u.uJy)
        assert len(flux_errs) == len(phot_obj)
        # if flux < 1e19 and flux != -99 and flux != 0:
        #     pass
        pipes_input = np.vstack((np.array(flux), np.array(flux_errs))).T
        if verbose:
            galfind_logger.info(f"{cat.survey} {ID}: {pipes_input}, \n bands = {', '.join(phot_obj.instrument.band_names)}")
            #end = time.time()
            #print(f'{end-start:.2f}s to load this galaxy')
        # append to bagpipes log file for survey/version/instrument - Not Implemented Yet!
        return pipes_input, bands

    @staticmethod
    def make_fit_instructions_dict(
            age_prior: str = "log_10", 
            metallicity_prior: str = "log_10",
            dust_prior: str = "log_10",
            dust_type: str = "Calzetti",
            dust_Av: Union[tuple, float, None] = None, # if not None, force overwrite default for given prior / dust type
            sfh: str = "continuity_bursty",
            max_birth_cloud_age: u.Quantity = 10 * u.Myr, 
            fix_z_SED_fit_params: dict = {}, # leave redshifts free by default
            logU: Union[tuple, float, None] = (-3., -1.),
            logU_prior: str = "uniform",
            fesc: Union[tuple, float, None] = (1e-4, 1.), 
            fesc_prior: str = "log_10"
            ):
        
        fit_instructions = {}
        fit_instructions["t_bc"] = max_birth_cloud_age.to(u.Gyr).value # Max age of birth clouds: Gyr
        exp = {}      
        const = {}
        delayed = {}
        burst = {}
        lognorm = {}

        # exponential SF history                            
        exp["age"] = (0.001, 15.) # Automatically adjusts for age of the universe
        exp['age_prior'] = age_prior
        exp["tau"] = (0.01, 15.)
        exp["massformed"] = (5., 12.) #Change this?
        
        exp['metallicity_prior'] = metallicity_prior
        if metallicity_prior == 'log_10':
            exp["metallicity"] = (1e-03, 3)
        elif metallicity_prior == 'const':
            exp['metallicity'] = (0, 3)

        #1e-4 1e1
        const["age_max"] = (0.01, 15) # Gyr
        const["age_min"] = 0.001 # Gyr
        const['age_prior'] = age_prior
        const["massformed"] = (5., 12.)  # Log_10 total stellar mass formed: M_Solar
        
        const['metallicity_prior'] = metallicity_prior
        if metallicity_prior == 'log_10':
            const["metallicity"] = (1e-03, 3)
        elif metallicity_prior == 'uniform':
            const['metallicity'] = (0, 3)

        delayed["tau"] = (0.01, 15) # `Gyr`
        delayed["massformed"] = (5., 12.)   # Log_10 total stellar mass formed: M_Solar
        
        delayed["age"] = (0.001, 15) # Gyr
        delayed['age_prior'] = age_prior
        delayed['metallicity_prior'] = metallicity_prior
        if metallicity_prior == 'log_10':
            delayed["metallicity"] = (1e-03, 3)
        elif metallicity_prior == 'uniform':
            delayed['metallicity'] = (0, 3)

        burst["age"] = (0.01, 15) # Gyr time since burst
        burst['age_prior'] = age_prior
        burst["massformed"] = (0., 12.)   # Log_10 total stellar mass formed: M_Solar
        
        burst['metallicity_prior'] = metallicity_prior
        if metallicity_prior == 'log_10':
            burst["metallicity"] = (1e-03, 3)
        elif metallicity_prior == 'uniform':
            burst['metallicity'] = (0, 3)

        #1e-4 1e1
        #lognorm["tstart"] = (0.001, 15) # Gyr THIS NEVER DID ANYTHING!
        #lognorm["tstart_prior"] = age_prior
        lognorm["tmax"] = (0.01, 15) # these will default to a flat prior probably
        lognorm['fwhm'] = (0.01, 15) 
        lognorm["massformed"] = (5., 12.)  # Log_10 total stellar mass formed: M_Solar
        
        lognorm['metallicity_prior'] = metallicity_prior

        if metallicity_prior == 'log_10':
            lognorm["metallicity"] = (1e-03, 3.)
        elif metallicity_prior == 'uniform':
            lognorm['metallicity'] = (0., 3.)

        # DPL
                    
        dblplaw = {} # double-power-law
        dblplaw["tau"] = (0., 15.)              # Vary the time of peak star-formation between
                                                # the Big Bang at 0 Gyr and 15 Gyr later. In 
                                                # practice the code automatically stops this
                                                # exceeding the age of the universe at the 
                                                # observed redshift.

        dblplaw["tau_prior"] = age_prior         # Impose a prior which is uniform in log_10 of the
                    
        dblplaw["alpha"] = (0.01, 1000.)          # Vary the falling power law slope from 0.01 to 1000.
        dblplaw["beta"] = (0.01, 1000.)           # Vary the rising power law slope from 0.01 to 1000.
        dblplaw["alpha_prior"] = "log_10"         # Impose a prior which is uniform in log_10 of the 
        dblplaw["beta_prior"] = "log_10"          # parameter between the limits which have been set 
                                                # above as in Carnall et al. (2017).
        dblplaw["massformed"] = (5., 12.)
        #dblplaw["metallicity"] = (0., 2.5)
        if metallicity_prior == 'log_10':
            dblplaw["metallicity"] = (1e-03, 3.)
        elif metallicity_prior == 'uniform':
            dblplaw['metallicity'] = (0., 3.)

        # Leja et al. 2019 continuity SFH
        continuity = {}
        #continuity["age"] = (0.01, 15) # Gyr
        #continuity['age_prior'] = age_prior
        continuity["massformed"] = (5., 12.)  # Log_10 total stellar mass formed: M_Solar
        continuity['metallicity_prior'] = metallicity_prior
        if metallicity_prior == 'log_10':
            continuity["metallicity"] = (1e-03, 3)
        elif metallicity_prior == 'uniform':
            continuity['metallicity'] = (0, 3)
        
            
        if len(fix_z_SED_fit_params) != 0:
            if sfh == 'continuity':
                raise Exception('Continuity model not compatible with varying redshift range.')
                continuity['redshift'] = redshift_range

        # This is a filler - real one is generated below when catalogue is loaded in
        cont_nbins = num_bins
        continuity['bin_edges'] = list(calculate_bins(redshift = 8, num_bins=cont_nbins, first_bin=first_bin, second_bin=second_bin, return_flat=True, output_unit='Myr', log_time=False))
        scale = 0
        if sfh == 'continuity':
            scale = 0.3
        if sfh == 'continuity_bursty':
            scale = 1.

        for i in range(1, len(continuity["bin_edges"])-1):
            continuity["dsfr" + str(i)] = (-10., 10.)
            continuity["dsfr" + str(i) + "_prior"] = "student_t"
            continuity["dsfr" + str(i) + "_prior_scale"] = scale  # Defaults to this value as in Leja19, but can be set
            continuity["dsfr" + str(i) + "_prior_df"] = 2       # Defaults to this value as in Leja19, but can be set

        # Iyer et al. (2019) Non-parametric SFH
        nbins = 6
        iyer = {}                            # The model of Iyer et al. (2019)
        iyer["sfr"] = (1e-3, 1e3)  
        iyer["sfr_prior"] = 'uniform'            # Solar masses per year
        iyer["bins"] = nbins             # Integer
        iyer["bins_prior"] = "dirichlet"     # This prior distribution must be used
        iyer["alpha"] = 5.0 # The Dirichlet prior has a single tunable parameter α that specifies how correlated the values are. In our case, values of this parameter α<1 result in values that can be arbitrarily close, leading to extremely spiky SFHs because galaxies have to assemble a significant fraction of their mass in a very short period of time, while α>1 leads to smoother SFHs with more evenly spaced values that never- theless have considerable diversity. In practice, we use a value of α=5, which leads to a distribution of parameters that is similar to what we find in SAM and MUFASA.    

        iyer["massformed"] = (5., 12.)  # Log_10 total stellar mass formed: M_Solar
        
        iyer['metallicity_prior'] = metallicity_prior

        if metallicity_prior == 'log_10':
            iyer["metallicity"] = (1e-03, 3)
        elif metallicity_prior == 'uniform':
            iyer['metallicity'] = (0, 3)
        
        # Put prior dictionaries in main fit_instructions dictionary
        age_prior_set = False
        if sfh == "exp":
            fit_instructions["exponential"] = exp   
            age_prior_set = True
        elif sfh == "const":
            fit_instructions["constant"] = const
            age_prior_set = True
        elif sfh == "burst":
            fit_instructions["burst"] = burst
            age_prior_set = True
        elif sfh == 'delayed':
            fit_instructions["delayed"] = delayed
            age_prior_set = True
        elif sfh == 'delayed+burst':
            fit_instructions['delayed'] = delayed
            fit_instructions["burst"] = burst 
            age_prior_set = True
        elif sfh=="exp+burst":
            fit_instructions["exponential"] = exp  
            fit_instructions["burst"] = burst 
            age_prior_set = True
        elif sfh=="const+burst":
            fit_instructions["const"] = const  
            fit_instructions["burst"] = burst 
            age_prior_set = True
        elif sfh == 'rising':
            delayed['tau'] = (0.5, 15)
            fit_instructions['delayed'] = delayed
            age_prior_set = True
        elif sfh == 'lognorm':
            fit_instructions['lognormal'] = lognorm
        elif sfh == 'iyer':
            fit_instructions['iyer'] = iyer
        elif sfh == 'continuity':
            fit_instructions['continuity'] = continuity
        elif sfh == 'continuity_bursty':
            fit_instructions['continuity'] = continuity
        elif sfh == 'dblplaw':
            fit_instructions['dblplaw'] = dblplaw
        else:
            print(f'SFH {sfh} not found.')
            return False

        # nebular emission (lines/continuum)
        if all([type(name) != type(None) for name in [logU, fesc]]):
            fit_instructions["nebular"] = {"logU": logU, "logU_prior": logU_prior, "fesc": fesc, "fesc_prior": fesc_prior}
        # dust
        if all(type(name) != type(None) for name in [dust_prior, dust_type]):
            dust = {}
            dust["eta"] = 1.  # Multiplicative factor on Av for stars in birth clouds
            if dust_type.lower() == 'salim': 
                dust["type"] = "Salim" # Salim  
                dust["delta"] = (-0.3, 0.3)   # Deviation from Calzetti slope ("Salim" type only)
                dust["delta_prior"] = 'Gaussian'
                dust["delta_prior_mu"] = 0. # This is Calzetti (approx)
                dust["delta_prior_sigma"] = 0.1 
                dust["B"] = (0. ,5.)
                dust["B_prior"] = 'uniform'
            elif dust_type.lower() == 'calzetti':
                dust["type"] = "Calzetti"
            elif dust_type.lower() == 'cf00': # Below taken from Tacchella+2022
                # This is taken from Example 5 in the bagpipes documentation		
                dust["type"] = "CF00"
                #dust["eta"] = 2.
                #dust["Av"] = (0., 2.0)
                dust["n"] = (0.3, 2.5)
                dust["n_prior"] = "Gaussian"
                dust["n_prior_mu"] = 0.7 # This is Calzetti (approx)
                dust["n_prior_sigma"] = 0.3

                #dust['n'] = (-1.0, 0.4) # 0.7088 is slope of calzetti - so deviation is - 0.3 < n < 1.1 
                # I think as it is offset from -1 (see Tachella, and is not given as negative, we want (0, 1.4), to represent (-1, 04))
                dust['n_prior'] = 'uniform'
                dust['eta'] = (1, 3.) # eta - 1 is done in code. Make eta be (1, 3) to represent (0, 2)
                dust['eta_prior'] = 'Gaussian' 
                dust['eta_prior_mu'] = 2.
                dust['eta_prior_sigma'] = 0.3

            dust['Av_prior'] = dust_prior
            if dust_prior == 'log_10':
                dust["Av"] = (0.0001, 10.)
            elif dust_prior == 'uniform':
                dust["Av"] = (0., 6.)
            
            fit_instructions["dust"] = dust
            
        return fit_instructions


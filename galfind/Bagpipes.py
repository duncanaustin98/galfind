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
import glob
from pathlib import Path
from tqdm import tqdm
import itertools

from . import useful_funcs_austind as funcs
from . import galfind_logger, config, SED_code, SED_fit_PDF, Redshift_PDF
from .SED import SED_obs

# %% Bagpipes SED fitting code

class Bagpipes(SED_code):

    # these should be made on runtime!
    galaxy_properties = ["redshift", "stellar_mass", "formed_mass", \
        "dust:Av", "beta_C94", "m_UV", "M_UV", \
        "sfr", "sfr_10myr", "ssfr", "ssfr_10myr"] # "Halpha_EWrest", "xi_ion_caseB",
    gal_property_fmt_dict = {
        "z": "Redshift, z",
        "stellar_mass": r"$M_{\star}$",
        "formed_mass": r"$M_{\star, \mathrm{formed}}$",
        "dust:Av": r"$A_V$",
        "beta_C94": r"$\beta_{\mathrm{C94}}$",
        "m_UV": r"$m_{\mathrm{UV}}$",
        "M_UV": r"$M_{\mathrm{UV}}$",
        "Halpha_EWrest": r"EW$_{\mathrm{rest}}$(H$\alpha$)",
        "xi_ion_caseB": r"$\xi_{\mathrm{ion}}$",
        "sfr": r"SFR$_{\mathrm{100Myr}}$",
        "sfr_10myr": r"SFR$_{\mathrm{10Myr}}$",
        "ssfr": r"sSFR$_{\mathrm{100Myr}}$",
        "ssfr_10myr": r"sSFR$_{\mathrm{10Myr}}$"
    }
    gal_property_unit_dict = {
        "z": u.dimensionless_unscaled,
        "stellar_mass": "dex(solMass)", 
        "formed_mass": "dex(solMass)", 
        "dust:Av": u.ABmag,
        "beta_C94": u.dimensionless_unscaled,
        "m_UV": u.ABmag, 
        "M_UV": u.ABmag, 
        "Halpha_EWrest": u.AA,
        "xi_ion_caseB": u.Hz / u.erg, 
        "sfr": u.solMass / u.yr, 
        "sfr_10myr": u.solMass / u.yr,
        "ssfr": u.yr ** -1, 
        "ssfr_10myr": u.yr ** -1
    }
    galaxy_property_dict = {**{gal_property if "redshift" not in gal_property else "z": f"{gal_property}_50" for gal_property in galaxy_properties}, **{"chi_sq": "chisq_phot"}}
    galaxy_property_errs_dict = {gal_property if "redshift" not in gal_property else "z": [f"{gal_property}_16", f"{gal_property}_84"] for gal_property in galaxy_properties}
    available_templates = ["BC03", "BPASS"]
    ext_src_corr_properties = ["stellar_mass", "formed_mass", "m_UV", "M_UV", "sfr", "sfr_10myr"]
    ID_label = "ID"
    are_errs_percentiles = True
    
    def __init__(self, SED_fit_params = None):
        # Bagpipes specific SED fit params assertions here
        super().__init__(SED_fit_params, self.galaxy_property_dict, self.galaxy_property_errs_dict, \
            self.available_templates, self.ID_label, self.are_errs_percentiles)
    
    @staticmethod
    def label_from_SED_fit_params(SED_fit_params, short = False):
        # should be generalized more here including e.g. SED_fit_params assertions
        if not short:
            # sort redshift label
            if not "fix_z" in SED_fit_params.keys():
                SED_fit_params["fix_z"] = False
            if SED_fit_params["fix_z"]:
                redshift_label = "zfix"
            else:
                if "z_range" in SED_fit_params.keys():
                    assert len(SED_fit_params["z_range"]) == 2
                    redshift_label = f"{int(SED_fit_params['z_range'][0])}_z_{int(SED_fit_params['z_range'][1])}"
                else:
                    galfind_logger.critical(f"Bagpipes {SED_fit_params=} must include either 'z_range' if 'fix_z' == False or not included!")
                    breakpoint()
            # sort SPS label
            assert "sps_model" in SED_fit_params.keys()
            if SED_fit_params["sps_model"].upper() == "BC03":
                sps_label = "" # should change this probably to read BC03
            elif SED_fit_params["sps_model"].upper() == "BPASS":
                sps_label = f"_{SED_fit_params['sps_model'].lower()}"
            else:
                galfind_logger.critical(f"Bagpipes {SED_fit_params=} must include 'sps_model' with .upper() in ['BC03', 'BPASS']")
                breakpoint()
            return f"Bagpipes_sfh_{SED_fit_params['sfh']}_dust_{SED_fit_params['dust']}_" + \
                f"{SED_fit_params['dust_prior']}_Z_{SED_fit_params['metallicity_prior']}" + \
                f"{sps_label}_{redshift_label}"
        else:
            return "Bagpipes"
        
    @staticmethod
    def hdu_from_SED_fit_params(SED_fit_params):
        return Bagpipes.label_from_SED_fit_params(SED_fit_params)

    def SED_fit_params_from_label(self, label):
        SED_fit_params = {"code": Bagpipes()}
        SED_fit_params["sfh"] = label.split("_sfh_")[1].split("_dust_")[0]
        dust_label = label.split("_dust_")[1].split("_Z_")[0]
        if "log_10" in dust_label:
            assert dust_label[-6:] == "log_10"
            SED_fit_params["dust_prior"] = "log_10"
            SED_fit_params["dust"] = dust_label[:-7]
        elif "uniform" in dust_label:
            assert dust_label[-7:] == "uniform"
            SED_fit_params["dust_prior"] = "uniform"
            SED_fit_params["dust"] = dust_label[:-8]
        else:
            galfind_logger.critical(f"Invalid dust prior from {dust_label=}! Must be in ['log_10', 'uniform']")
            breakpoint()
        split_metallicity_label = label.split("_Z_")[1].split("_")
        if split_metallicity_label[0] == "log" and split_metallicity_label[1] == "10":
            SED_fit_params["metallicity_prior"] = "log_10"
        elif split_metallicity_label[0] == "uniform":
            SED_fit_params["metallicity_prior"] = "uniform"
        else:
            galfind_logger.critical(f"Invalid metallicity prior from {split_metallicity_label=}! Must be in ['log_10', 'uniform']")
            breakpoint()
        # easier if BC03 read properly
        if "BPASS" in label:
            SED_fit_params["sps_model"] = "BPASS"
            redshift_label = label.split(SED_fit_params["sps_model"])[1][1:]
        else:
            SED_fit_params["sps_model"] = "BC03"
            redshift_label = label.split(SED_fit_params["metallicity_prior"])[-1][1:]
        if redshift_label == "zfix":
            SED_fit_params["fix_z"] = True
        else:
            split_zlabel = redshift_label.split("_z_")
            SED_fit_params["z_range"] = (float(split_zlabel[0]), float(split_zlabel[1]))
            SED_fit_params["fix_z"] = False
        return SED_fit_params

    def galaxy_property_labels(self, gal_property, SED_fit_params, is_err = False, **kwargs):
        suffix = self.label_from_SED_fit_params(SED_fit_params, short = True)
        if gal_property in self.galaxy_property_dict.keys() and not is_err:
            if gal_property == "z" and SED_fit_params["fix_z"]:
                return f"input_redshift_{suffix}"
            else:
                return f"{self.galaxy_property_dict[gal_property]}_{suffix}"
        elif gal_property in self.galaxy_property_errs_dict.keys() and is_err:
            if gal_property == "z" and SED_fit_params["fix_z"]:
                return list(itertools.repeat(None, 2)) # array of None's
            else:
                return [f"{self.galaxy_property_errs_dict[gal_property][0]}_{suffix}", \
                    f"{self.galaxy_property_errs_dict[gal_property][1]}_{suffix}"]
        else:
            return f"{gal_property}_{suffix}"

    def make_in(self, cat):
        # no need for bagpipes input catalogue
        pass
    
    def run_fit(self, in_path: Union[str, None], fits_out_path: str, instrument, SED_fit_params: dict, overwrite: bool = False, **kwargs):
        pass

    def make_fits_from_out(self, out_path, SED_fit_params, overwrite: bool = True):
        fits_out_path = self.get_galfind_fits_path(out_path)
        if not Path(fits_out_path).is_file() or overwrite:
            tab = Table.read(out_path)
            tab[self.ID_label] = np.array([id.split("_")[0] for id in tab["#ID"]]).astype(int)
            tab.remove_column("#ID")
            if "input_redshift" in tab.colnames:
                if all(z == 0. for z in tab["input_redshift"]):
                    tab.remove_column("input_redshift")
            for name in tab.colnames:
                if name != self.ID_label:
                    tab.rename_column(name, self.galaxy_property_labels(name, SED_fit_params))
            tab.write(fits_out_path, overwrite = True)

    @staticmethod
    def get_galfind_fits_path(path):
        return path.replace(".fits", "_galfind.fits")
    
    @staticmethod
    def extract_SEDs(IDs, SED_paths):
        # ensure this works if only extracting 1 galaxy
        if type(IDs) in [str, int, float]:
            IDs = np.array([int(IDs)])
        if type(SED_paths) == str:
           SED_paths = [SED_paths]
        assert len(IDs) == len(SED_paths), galfind_logger.critical(f"len(IDs) = {len(IDs)} != len(data_paths) = {len(SED_paths)}!")
        z_arr = np.zeros(len(IDs))
        for i, path in enumerate(SED_paths):
            if type(path) != type(None):
                f = open(path)
                header = f.readline()
                z_arr[i] = float(header.replace("\n", "").split("z=")[-1])
                f.close()
        data_arr = [np.loadtxt(path) if type(path) != type(None) else None for path in SED_paths]
        wavs = [data[:, 0] if type(data) != type(None) else None for data in data_arr]
        fluxes = [data[:, 2] if type(data) != type(None) else None for data in data_arr]
        SED_obs_arr = [SED_obs(z, wav, flux, u.um, u.uJy) if all(type(i) != type(None) \
            for i in [z, wav, flux]) else None for z, wav, flux in \
            tqdm(zip(z_arr, wavs, fluxes), desc = "Constructing pipes SEDs", total = len(wavs))]
        return SED_obs_arr

    # should transition away from staticmethod
    @staticmethod
    def extract_PDFs(gal_property, IDs, PDF_paths, SED_fit_params, timed: bool = True):
        # ensure this works if only extracting 1 galaxy
        if type(IDs) in [str, int, float]:
            IDs = np.array([int(IDs)])
        if type(PDF_paths) == str:
           PDF_paths = [PDF_paths]
        # # return list of None's if gal_property not in the PDF_paths, else load the PDFs
        # if gal_property not in PDF_paths.keys():
        #     return list(np.full(len(IDs), None))
        # else:
        if not gal_property in Bagpipes.gal_property_unit_dict.keys():
            Bagpipes.gal_property_unit_dict[gal_property] = u.dimensionless_unscaled
        pdf_arrs = [np.array(Table.read(path, format = "ascii.fast_no_header")["col1"]) \
            if type(path) != type(None) else None for path in \
            tqdm(PDF_paths, desc = f"Loading {gal_property} PDFs", total = len(PDF_paths))]
        if gal_property == "z":
            pdfs = [Redshift_PDF.from_1D_arr(pdf * u.Unit(Bagpipes.gal_property_unit_dict[gal_property]), \
                SED_fit_params, timed = timed) if type(pdf) != type(None) else None \
                for pdf in tqdm(pdf_arrs, desc = f"Constructing {gal_property} PDFs", total = len(pdf_arrs))]
        else:
            pdfs = [SED_fit_PDF.from_1D_arr(gal_property, pdf * u.Unit(Bagpipes.gal_property_unit_dict[gal_property]), \
                SED_fit_params, timed = timed) if type(pdf) != type(None) else None \
                for pdf in tqdm(pdf_arrs, desc = f"Constructing {gal_property} PDFs", total = len(pdf_arrs))]
        # add save path to PDF
        pdfs = [pdf.add_save_path(path) if type(pdf) != type(None) else None for path, pdf in zip(PDF_paths, pdfs)]
        return pdfs
    
    def load_pipes_fit_obj(self):
        pass

    @staticmethod
    def get_out_paths(cat, SED_fit_params, IDs, load_properties = \
            ["stellar_mass", "formed_mass", "dust:Av", \
            "beta_C94", "m_UV", "M_UV", "sfr", "sfr_10myr"]): # , "Halpha_EWrest", "xi_ion_caseB"
        pipes_name = Bagpipes.label_from_SED_fit_params(SED_fit_params)
        in_path = None
        out_path = f"{config['Bagpipes']['BAGPIPES_DIR']}/cats/{cat.survey}/{pipes_name.replace('Bagpipes_', '')}.fits"
        fits_out_path = Bagpipes.get_galfind_fits_path(out_path)
        PDF_dir = out_path.replace(".fits", "").replace("cats", "pdfs")
        SED_dir = out_path.replace(".fits", "").replace("cats", "seds")
        # else:
        if not SED_fit_params["fix_z"]:
            load_properties += ["redshift"]
        PDF_paths = {gal_property if "redshift" not in gal_property else "z": \
            [f"{PDF_dir}/{gal_property}/{str(int(ID))}_{cat.survey}.txt" \
            if Path(f"{PDF_dir}/{gal_property}/{str(int(ID))}_{cat.survey}.txt").is_file() \
            else None for ID in IDs] for gal_property in load_properties}
        # determine SED paths
        SED_paths = [f"{SED_dir}/{str(int(ID))}_{cat.survey}.dat" \
            if Path(f"{SED_dir}/{str(int(ID))}_{cat.survey}.dat").is_file() \
            else None for ID in IDs]
        return in_path, out_path, fits_out_path, PDF_paths, SED_paths

    def make_templates(self):
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
                dust["Av"] = (1e-4, 10.)
            elif dust_prior == 'uniform':
                dust["Av"] = (0., 6.)
            
            fit_instructions["dust"] = dust
            
        return fit_instructions

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:14:30 2023

@author: austind
"""

# Photometry_obs.py
import numpy as np
import astropy.constants as const
import astropy.units as u
from copy import copy, deepcopy
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from . import useful_funcs_austind as funcs
from . import astropy_cosmo
from . import config

# each "Photometry_obs" should have an "Instrument" object inside it (e.g. NIRCam/MIRI/HST_ACS/HST_WFC3IR)
class Photometry:
    
    def __init__(self, instrument, flux_Jy, flux_Jy_errs, loc_depths):
        self.instrument = instrument
        self.flux_Jy = flux_Jy
        self.flux_Jy_errs = flux_Jy_errs
        self.loc_depths = loc_depths
    
    @property
    def flux_lambda(self): # wav and flux_nu must have units here!
        return (self.flux_Jy * const.c / ((np.array([value.value for value in self.instrument.band_wavelengths.values()]) * u.Angstrom) ** 2)).to(u.erg / (u.s * (u.cm ** 2) * u.Angstrom)) # both flux_nu and wav must be in the same rest or observed frame
    
    @property
    def flux_lambda_errs(self):
        return (self.flux_Jy_errs * const.c / ((np.array([value.value for value in self.instrument.band_wavelengths.values()]) * u.Angstrom) ** 2)).to(u.erg / (u.s * (u.cm ** 2) * u.Angstrom))

    def crop_phot(self, indices):
        indices = np.array(indices).astype(int)
        for index in reversed(indices):
            self.instrument.remove_band(self.instrument.bands[index])
        self.flux_Jy = np.delete(self.flux_Jy, indices)
        self.flux_Jy_errs = np.delete(self.flux_Jy_errs, indices)

class Photometry_obs(Photometry):

    def __init__(self, instrument, flux_Jy, flux_Jy_errs, aper_diam, min_flux_pc_err, loc_depths, SED_results = []):
        self.aper_diam = aper_diam
        self.min_flux_pc_err = min_flux_pc_err
        self.SED_results = SED_results # array of SED_result objects with different SED fitting runs
        super().__init__(instrument, flux_Jy, flux_Jy_errs, loc_depths)

    @classmethod # not a gal object here, more like a catalogue row
    def get_phot_from_sex(cls, sex_cat_row, instrument, cat_creator): # single unit of sextractor catalogue
        # below is contained within cat_creator
        #aper_diam_index = int(json.loads(config.get("SExtractor", "APERTURE_DIAMS")).index(cat_creator.aper_diam.value))
        fluxes = []
        flux_errs = []
        #print("instrument = ", instrument)
        # Copy constructor problem here!
        instrument_copy = instrument.new_instrument(excl_bands = [band for band in instrument.new_instrument().bands if band not in instrument.bands]) # Problem loading the photometry properly here!!!
        #print("instrument_copy = ", instrument_copy)
        for band in instrument_copy.bands:
            try:
                flux, err = cat_creator.load_photometry(sex_cat_row, band)
                fluxes.append(flux.value)
                flux_errs.append(err.value)
            except:
                # no data for the relevant band within the catalogue
                instrument.remove_band(band)
                print(f"{band} flux not loaded")
        phot_obs = cls(instrument, fluxes * u.Jy, flux_errs * u.Jy, cat_creator.aper_diam)
        phot_obs.load_local_depths(sex_cat_row, instrument, cat_creator.aper_diam_index)
        return phot_obs
        
    @classmethod
    def get_phot_from_sim(cls, gal, instrument, sim, min_flux_err_pc = 5):
        fluxes = []
        flux_errs = []
        instrument_copy = instrument.copy()
        for band in instrument_copy.bands:
            try:
                flux = np.array(gal[sim.flux_col_name(band)])
                err = np.array(gal[sim.flux_err_name(band)])
                # encorporate minimum flux error
                err = np.array([err_band if err_band / flux_band >= min_flux_err_pc / 100 else \
                                min_flux_err_pc * flux_band / 100 for flux_band, err_band in zip(flux, err)])
                flux_Jy = funcs.flux_image_to_Jy(flux, sim.zero_point)
                err_Jy = funcs.flux_image_to_Jy(err, sim.zero_point)
                fluxes = np.append(fluxes, flux_Jy.value)
                flux_errs = np.append(flux_errs, err_Jy.value)
            except:
                instrument.remove_band(band)
                print(f"{band} flux not loaded")
    
    def load_local_depths(self, sex_cat_row, instrument, aper_diam_index):
        self.loc_depths = np.array([sex_cat_row[f"loc_depth_{band}"].T[aper_diam_index] for band in instrument.bands])

class Photometry_rest(Photometry):
    
    def __init__(self, instrument, flux_Jy, flux_Jy_errs, loc_depths, z, code_name, rest_UV_wav_lims = [1250., 3000.] * u.Angstrom):
        self.z = z
        self.rest_UV_wav_lims = rest_UV_wav_lims
        super().__init__(instrument, flux_Jy, flux_Jy_errs, loc_depths)
    
    # STILL NEED TO LOOK FURTHER INTO THIS
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, deepcopy(value, memo))
        return result
    
    @property
    def wav(self):
        return funcs.wav_obs_to_rest(np.array([value.value for value in self.phot_obs.instrument.band_wavelengths.values()]) * u.Angstrom, self.z)
    
    @property
    def wav_errs(self):
        return funcs.wav_obs_to_rest(np.array([value.value / 2 for value in self.phot_obs.instrument.band_FWHMs.values()]) * u.Angstrom, self.z)
    
    @property
    def flux_lambda(self):
        return funcs.flux_lambda_obs_to_rest(self.phot_obs.flux_lambda, self.z)
    
    @property
    def flux_lambda_errs(self):
        return funcs.flux_lambda_obs_to_rest(self.phot_obs.flux_lambda_errs, self.z)
    
    @property
    def log_flux_lambda(self):
        return np.log10(self.flux_lambda.value)
    
    @property
    def log_flux_lambda_errs(self): # now asymmetric and unitless
        return funcs.errs_to_log(self.flux_lambda.value, [self.flux_lambda_errs.value, self.flux_lambda_errs.value])[1]
    
    @property
    def lum_distance(self, cosmo = astropy_cosmo):
        return cosmo.luminosity_distance(self.z).to(u.pc)
    
    @property
    def rest_UV_band_index(self):
        return np.abs(self.wav - 1500 * u.Angstrom).argmin()
    
    @property
    def rest_UV_band(self):
        return self.phot_obs.instrument.bands[self.rest_UV_band_index]
    
    @property
    def rest_UV_band_flux_Jy(self):
        return self.phot_obs.flux_Jy[self.rest_UV_band_index]
    
    def make_rest_UV_phot(self):
        phot_rest_copy = deepcopy(self)
        phot_rest_copy.rest_UV_phot_only()
        #print(f"rest frame UV bands = {phot_rest_copy.phot_obs.instrument.bands}")
        self.rest_UV_phot = phot_rest_copy
    
    @classmethod # this is actually a row in a catalogue, not a 'Galaxy' object
    def get_phot_from_sex(cls, gal, instrument, code):
        phot_obs = Photometry_obs.get_phot_from_sex(gal, instrument)
        return cls(phot_obs, np.float(gal[code.galaxy_properties["z"]]), code.code_name)
    
    def rest_UV_phot_only(self):
        crop_indices = []
        for i, (wav, wav_err) in enumerate(zip(self.wav.value, self.wav_errs.value)):
            wav *= u.Angstrom
            wav_err *= u.Angstrom
            #print(wav, wav_err, self.rest_UV_wav_lims)
            if wav - wav_err < self.rest_UV_wav_lims[0] or wav + wav_err > self.rest_UV_wav_lims[1]:
                crop_indices = np.append(crop_indices, i)
        #print(crop_indices)
        self.phot_obs.crop_phot(crop_indices)
     
    def beta_slope_power_law_func(wav_rest, A, beta):
        return (10 ** A) * (wav_rest ** beta)
    
    def set_ext_source_UV_corr(self, ext_source_UV_corr):
        #print(f"Adding extended source UV correction to {self.ID}")
        self.UV_ext_src_corr = ext_source_UV_corr
        
    def plot(self, save_dir, ID, plot_fit = True, iters = 1_000, save = True, show = False, n_interp = 100):
        self.make_rest_UV_phot()
    
        #if not all(beta == -99. for beta in self.beta_PDF):
        sns.set(style="whitegrid")
        warnings.filterwarnings("ignore")

        # Create figure and axes
        fig, ax = plt.subplots()
        
        # Plotting code
        ax.errorbar(np.log10(self.rest_UV_phot.wav.value), self.rest_UV_phot.log_flux_lambda, yerr = self.rest_UV_phot.log_flux_lambda_errs,
                     ls="none", c="black", zorder=10, marker="o", markersize=5, capsize=3)
        
        if plot_fit:
            fit_lines = []
            fit_lines_interped = []
            wav_interp = np.linspace(np.log10(self.rest_UV_phot.wav.value)[0], np.log10(self.rest_UV_phot.wav.value)[-1], n_interp)
            for i in range(iters):
                #percentiles = np.percentile([16, 50, 84], axis=0)
                f_interp = interp1d(np.log10(self.rest_UV_phot.wav.value), np.log10(Photometry_rest.beta_slope_power_law_func(self.rest_UV_phot.wav.value, self.amplitude_PDF[i],
                        self.beta_PDF[i])), kind = 'linear')
                y_new = f_interp(wav_interp)
                fit_lines.append(np.log10(Photometry_rest.beta_slope_power_law_func(self.rest_UV_phot.wav.value, self.amplitude_PDF[i],
                        self.beta_PDF[i])))
                fit_lines_interped.append(y_new)
            fit_lines_interped = np.array(fit_lines_interped)
            fit_lines = np.array(fit_lines)
            fit_lines.reshape(iters, len(self.rest_UV_phot.wav.value))
            fit_lines_interped.reshape(iters, len(wav_interp))
            
            l1_chains = np.array([np.percentile(x, 16) for x in fit_lines_interped.T])
            med_chains = np.array([np.percentile(x, 50) for x in fit_lines.T])
            u1_chains = np.array([np.percentile(x, 84) for x in fit_lines_interped.T])
            
            ax.plot(np.log10(self.rest_UV_phot.wav.value), med_chains, color = "red", zorder = 2)
            ax.fill_between(wav_interp, l1_chains, u1_chains, color="grey", alpha=0.2, zorder=1)
        
        ax.set_xlabel(r"$\log_{10}(\lambda_{\mathrm{rest}} / \mathrm{\AA})$")
        ax.set_ylabel(r"$\log_{10}(\mathrm{f}_{\lambda_{\mathrm{rest}}} / \mathrm{erg} \, \mathrm{s}^{-1} \, \mathrm{cm}^{-2} \, \mathrm{\AA}^{-1})$")
        
        # Add the Galaxy ID label
        ax.text(0.05, 0.05, f"Galaxy ID = {str(ID)}", transform=ax.transAxes, ha="left", va="bottom", fontsize=12)
        # Add the Beta label
        ax.text(0.95, 0.95, r"$\beta$" + " = {:.2f} $^{{+{:.2f}}}_{{-{:.2f}}}$".format(np.percentile(self.beta_PDF, 50),
                                                                                               np.percentile(self.beta_PDF, 84) - np.percentile(self.beta_PDF, 50),
                                                                                               np.percentile(self.beta_PDF, 50) - np.percentile(self.beta_PDF, 16)),
                transform=ax.transAxes, ha="right", va="top", fontsize=12)

        
        ax.set_xlim(*np.log10(self.rest_UV_wav_lims.value))
        
        if save:
            path = f"{save_dir}/plots/{ID}.png"
            funcs.make_dirs(path)
            fig.savefig(path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.tight_layout()
            plt.show()
        
        plt.clf()

    def basic_beta_calc(self):
        self.make_rest_UV_phot()
        #print(self.rest_UV_phot.wav, self.rest_UV_phot.flux_lambda)
        try:
            popt, pcov = curve_fit(Photometry_rest.beta_slope_power_law_func, self.rest_UV_phot.wav, self.rest_UV_phot.flux_lambda, sigma = self.rest_UV_phot.flux_lambda_errs, maxfev = 1_000)
            beta = popt[1]
            return beta
        except:
            return None
        
    def fit_UV_slope(self, save_dir, ID, z_PDF = None, iters = 10_000, plot = True): # 1D redshift PDF
        #print(f"Fitting UV slope for {ID}")
        self.make_rest_UV_phot()
        fluxes = np.array([np.random.normal(mu.value, sigma.value, iters) for mu, sigma in zip(self.rest_UV_phot.flux_lambda, self.rest_UV_phot.flux_lambda_errs)]).T
        
        if z_PDF != None:
            # vary within redshift errors
            pass
        popt_arr = np.array([curve_fit(Photometry_rest.beta_slope_power_law_func, self.rest_UV_phot.wav, flux, maxfev = 1_000)[0] for flux in fluxes])
        if len(popt_arr) > 1:
            amplitude_PDF = popt_arr.T[0]
            beta_PDF = popt_arr.T[1]
        else:
            amplitude_PDF = [-99.]
            beta_PDF = [-99.]
        self.amplitude_PDF = amplitude_PDF # unitless
        self.beta_PDF = beta_PDF # unitless
                
        for name in ["Amplitude", "Beta"]:
            self.save_UV_fit_PDF(save_dir, name, ID)
            
        return amplitude_PDF, beta_PDF
    
    def calc_flux_lambda_1500_PDF(self, save_dir, ID, UV_ext_src_corr):
        self.open_UV_fit_PDF(save_dir, "Amplitude", ID)
        self.open_UV_fit_PDF(save_dir, "Beta", ID)

        if all(value == -99 for value in self.amplitude_PDF):
            self.flux_lambda_1500_PDF = [-99.]
        else:
            self.flux_lambda_1500_PDF = (Photometry_rest.beta_slope_power_law_func(1500., self.amplitude_PDF, self.beta_PDF) \
                                * UV_ext_src_corr) * u.erg / (u.s * (u.cm ** 2) * u.Angstrom)
        self.save_UV_fit_PDF(save_dir, "flux_lambda_1500", ID)
        return self.flux_lambda_1500_PDF
    
    def calc_flux_Jy_1500_PDF(self, save_dir, ID):
        self.open_UV_fit_PDF(save_dir, "flux_lambda_1500", ID)
        if all(value == -99 for value in self.amplitude_PDF):
            self.flux_Jy_1500_PDF = [-99.]
        else:
            self.flux_Jy_1500_PDF = (self.flux_lambda_1500_PDF * ((1500. * u.Angstrom) ** 2) / const.c).to(u.Jy)
        self.save_UV_fit_PDF(save_dir, "flux_Jy_1500", ID)
        return self.flux_Jy_1500_PDF
    
    def calc_M_UV_PDF(self, save_dir, ID):
        self.open_UV_fit_PDF(save_dir, "flux_Jy_1500", ID)
        if all(value == -99 for value in self.amplitude_PDF):
            self.m_1500_PDF = [-99.]
            self.M_UV_PDF = [-99.]
        else:
            self.m_1500_PDF = funcs.flux_to_mag(self.flux_Jy_1500_PDF, 8.9)
            self.M_UV_PDF = self.m_1500_PDF - 5 * np.log10(self.lum_distance.value / 10) + 2.5 * np.log10(1 + self.z)
        self.save_UV_fit_PDF(save_dir, "M_UV", ID)
        return self.M_UV_PDF
    
    def calc_A_UV_PDF(self, save_dir, ID, scatter_dex = 0.5):
        self.open_UV_fit_PDF(save_dir, "Beta", ID)
        if all(value == -99 for value in self.amplitude_PDF):
            self.A_UV_PDF = [-99.]
        else:
            self.A_UV_PDF = 4.43 + (1.99 * self.beta_PDF) + np.random.uniform(-scatter_dex, scatter_dex)
        self.save_UV_fit_PDF(save_dir, "A_UV", ID)
        return self.A_UV_PDF
    
    def calc_L_obs_PDF(self, save_dir, ID, alpha = 0.): # α=0 in Donnan 2022
        self.open_UV_fit_PDF(save_dir, "flux_Jy_1500", ID)
        if all(value == -99 for value in self.amplitude_PDF):
            self.L_obs_PDF = [-99.]
        else:
            self.L_obs_PDF = ((4 * np.pi * self.flux_Jy_1500_PDF * self.lum_distance ** 2) / ((1 + self.z) ** (1 + alpha))).to(u.erg / (u.s * u.Hz))
        self.save_UV_fit_PDF(save_dir, "L_obs", ID)
        return self.L_obs_PDF
    
    def calc_L_int_PDF(self, save_dir, ID):
        self.open_UV_fit_PDF(save_dir, "L_obs", ID)
        self.open_UV_fit_PDF(save_dir, "A_UV", ID)
        if all(value == -99 for value in self.amplitude_PDF):
            self.L_int_PDF = [-99.]
        else:
            self.L_int_PDF = np.array([L_obs * 10 ** (A_UV / 2.5) if A_UV > 0 else L_obs for L_obs, A_UV in zip(self.L_obs_PDF.value, self.A_UV_PDF)]) * (u.erg / (u.s * u.Hz))
        self.save_UV_fit_PDF(save_dir, "L_int", ID)
        return self.L_int_PDF
    
    def calc_SFR_PDF(self, save_dir, ID):
        self.open_UV_fit_PDF(save_dir, "L_int", ID)
        if all(value == -99 for value in self.amplitude_PDF):
            self.SFR_PDF = [-99.]
        else:
            self.SFR_PDF = 1.15e-28 * self.L_int_PDF.value * u.solMass / u.yr # Madau, Dickinson 2014, FSPS, Salpeter IMF templates
        self.save_UV_fit_PDF(save_dir, "SFR", ID)
        return self.SFR_PDF
    
    def save_UV_fit_PDF(self, save_dir, obs_name, ID, UV_ext_src_corr = None, plot_PDF = config.getboolean("RestUVProperties", "PLOT_PDFS")):
        PDF = self.open_UV_fit_PDF(save_dir, obs_name, ID, UV_ext_src_corr)
        try:
            unit = PDF.unit # keep track of PDF units
            PDF = np.array([val.value for val in PDF]) # make PDF unitless
        except:
            unit = "dimensionless"
        
        if plot_PDF:
            funcs.PDF_hist(PDF, save_dir, obs_name, ID, show = True, save = True)
        funcs.save_PDF(PDF, f"{obs_name}, units = {unit}, iters = {len(PDF)}", funcs.PDF_path(save_dir, obs_name, ID))
    
    def open_UV_fit_PDF(self, save_dir, obs_name, ID, UV_ext_src_corr = None, plot = True):
        try:
            # attempt to open PDF from object
            PDF = self.obs_name_to_PDF(obs_name)
        except: # if PDF not in object already
            try:
                # attempt to load the previously saved PDF from directory
                PDF = self.load_UV_fit_PDF(save_dir, obs_name, ID)
                #print(f"Loaded {obs_name} UV fit successfully for {ID}!")
            except: # if PDF not in object and not already saved
                # calculate the PDF (can take on the order of minutes depending on PDF iters)
                if obs_name == "Amplitude" or obs_name == "Beta":
                    self.fit_UV_slope(save_dir, ID)
                elif obs_name == "flux_lambda_1500":
                    self.calc_flux_lambda_1500_PDF(save_dir, ID, UV_ext_src_corr)
                elif obs_name == "flux_Jy_1500":
                    self.calc_flux_Jy_1500_PDF(save_dir, ID)
                elif obs_name == "M_UV":
                    self.calc_M_UV_PDF(save_dir, ID)
                elif obs_name == "A_UV":
                    self.calc_A_UV_PDF(save_dir, ID)
                elif obs_name == "L_obs":
                    self.calc_L_obs_PDF(save_dir, ID)
                elif obs_name == "L_int":
                    self.calc_L_int_PDF(save_dir, ID)
                elif obs_name == "SFR":
                    self.calc_SFR_PDF(save_dir, ID)
                else:
                    raise(Exception(f"{obs_name} not valid for calculating UV fit to photometry for {ID}!"))
                PDF = self.obs_name_to_PDF(obs_name)
                
        if obs_name == "Beta" and plot: # and not all(beta == -99. for beta in self.beta_PDF):
            self.plot(save_dir, ID)
        return PDF
    
    # this function and the one below could be included in the open_UV_fit_PDF
    def load_UV_fit_PDF(self, save_dir, obs_name, ID):
        # load PDF from directory
        PDF = np.array(np.loadtxt(f"{funcs.PDF_path(save_dir, obs_name, ID)}.txt"))
        if obs_name == "Amplitude":
            self.amplitude_PDF = PDF
        elif obs_name == "Beta":
            self.beta_PDF = PDF
        elif obs_name == "flux_lambda_1500":
            self.flux_lambda_1500_PDF = PDF * u.erg / (u.s * (u.cm ** 2) * u.Angstrom)
        elif obs_name == "flux_Jy_1500":
            self.flux_Jy_1500_PDF = PDF * u.Jy
        elif obs_name == "M_UV":
            self.M_UV_PDF = PDF
        elif obs_name == "A_UV":
            self.A_UV_PDF = PDF
        elif obs_name == "L_obs":
            self.L_obs_PDF = PDF * u.erg / (u.s * u.Hz)
        elif obs_name == "L_int":
            self.L_int_PDF = PDF * u.erg / (u.s * u.Hz)
        elif obs_name == "SFR":
            self.SFR_PDF = PDF * u.solMass / u.yr
        else:
            raise(Exception(f"{obs_name} not valid for loading UV fit to photometry for {ID}!"))
        return PDF
    
    # this function and the one above could be included in the open_UV_fit_PDF
    def obs_name_to_PDF(self, obs_name):
        if obs_name == "Amplitude":
            return self.amplitude_PDF
        elif obs_name == "Beta":
            return self.beta_PDF
        elif obs_name == "flux_lambda_1500":
            return self.flux_lambda_1500_PDF
        elif obs_name == "flux_Jy_1500":
            return self.flux_Jy_1500_PDF
        elif obs_name == "M_UV":
            return self.M_UV_PDF
        elif obs_name == "A_UV":
            return self.A_UV_PDF
        elif obs_name == "L_obs":
            return self.L_obs_PDF
        elif obs_name == "L_int":
            return self.L_int_PDF
        elif obs_name == "SFR":
            return self.SFR_PDF
        else:
            raise(Exception(f"No {obs_name} PDF found in object!")) 
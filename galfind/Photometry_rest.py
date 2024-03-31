#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:04:24 2023

@author: austind
"""

# Photometry_rest.py
import numpy as np
import astropy.constants as const
import astropy.units as u
from copy import copy, deepcopy
import traceback
import seaborn as sns
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from . import config, galfind_logger, astropy_cosmo, line_diagnostics, Photometry
from . import useful_funcs_austind as funcs

class beta_fit:
    def __init__(self, z, instrument):
        self.z = z
        self.instrument = instrument
        self.instrument.load_instrument_filter_profiles()
        self.wavelength = {}
        self.transmission = {}
        self.norm = {}
        for band in instrument:
            self.wavelength[band] = np.array(self.instrument.filter_profiles[band]["Wavelength"].value / (1 + self.z))
            self.transmission[band] = np.array(self.instrument.filter_profiles[band]["Transmission"].value)
            self.norm[band] = np.trapz(self.transmission[band], x = self.wavelength[band])
    def beta_slope_power_law_func_conv_filt(self, _, A, beta):
        return np.array([np.trapz((10 ** A) * (self.wavelength[band] ** beta) * self.transmission[band], x = self.wavelength[band]) / self.norm[band] for band in self.instrument])

class Photometry_rest(Photometry):
    
    def __init__(self, instrument, flux_Jy, flux_Jy_errs, depths, z, rest_UV_wav_lims = [1250., 3000.] * u.Angstrom):
        self.z = z
        self.rest_UV_wav_lims = rest_UV_wav_lims
        super().__init__(instrument, flux_Jy, flux_Jy_errs, depths)
    
    @classmethod
    def from_fits_cat(cls, fits_cat_row, instrument, cat_creator, code):
        phot = Photometry.from_fits_cat(fits_cat_row, instrument, cat_creator)
        return cls(phot, np.float(fits_cat_row[code.galaxy_properties["z_phot"]]))
    
    @classmethod
    def from_phot(cls, phot, z, rest_UV_wav_lims = [1250., 3000.] * u.Angstrom):
        return cls(phot.instrument, phot.flux_Jy, phot.flux_Jy_errs, phot.loc_depths, z, rest_UV_wav_lims)
    
    @classmethod
    def from_phot_obs(cls, phot, rest_UV_wav_lims = [1250., 3000.] * u.Angstrom):
        return cls(phot.instrument, phot.flux_Jy, phot.flux_Jy_errs, phot.loc_depths, phot.z, rest_UV_wav_lims)
    
    # STILL NEED TO LOOK FURTHER INTO THIS
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, deepcopy(value, memo))
        return result
    
    @property
    def first_Lya_detect_band(self):
        first_detect_band = ""
        for band, wav in zip(self.instrument, self.wav):
            line_diagnostics["Lya"][""]
        pass

    @property
    def first_Lya_non_detect_band(self):
        first_non_detect_band = ""
        for band, wav in zip(self.instrument, self.wav):
            pass

    # Old properties! This class should probably be overwritten
    
    @property
    def wav(self):
        return funcs.wav_obs_to_rest(np.array([self.instrument.band_wavelengths[band].value for band in self.instrument.bands]) * u.Angstrom, self.z)
    
    @property
    def wav_errs(self):
        return funcs.wav_obs_to_rest(np.array([self.instrument.band_FWHMs[band].value / 2 for band in self.instrument.bands]) * u.Angstrom, self.z)
    
    @property
    def flux_lambda(self):
        flux_lambda_obs = (self.flux_Jy * const.c / ((np.array([self.instrument.band_wavelengths[band].value for band in self.instrument.bands]) * u.Angstrom) ** 2)).to(u.erg / (u.s * (u.cm ** 2) * u.Angstrom))
        return funcs.flux_lambda_obs_to_rest(flux_lambda_obs, self.z)
    
    @property
    def flux_lambda_errs(self):
        flux_lambda_obs_errs = (self.flux_Jy_errs * const.c / ((np.array([self.instrument.band_wavelengths[band].value for band in self.instrument.bands]) * u.Angstrom) ** 2)).to(u.erg / (u.s * (u.cm ** 2) * u.Angstrom))
        return funcs.flux_lambda_obs_to_rest(flux_lambda_obs_errs, self.z)
    
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
        return self.instrument.bands[self.rest_UV_band_index]
    
    @property
    def rest_UV_band_flux_Jy(self):
        return self.flux_Jy[self.rest_UV_band_index]
    
    @property
    def N_bands(self):
        return len(self.flux_Jy)
    
    @staticmethod
    def rest_UV_wavs_name(rest_UV_wavs):
        try:
            rest_UV_wavs.unit
            rest_UV_wavs = rest_UV_wavs.to(u.Angstrom).value
            #print("Converted to Angstrom")
        except:
            #print("Assumed the unitless value is inserted in Angstrom!")
            pass
        return f"{str(int(rest_UV_wavs[0]))}-{str(int(rest_UV_wavs[1]))}Angstrom"
    
    def make_rest_UV_phot(self):
        phot_rest_copy = deepcopy(self)
        phot_rest_copy.rest_UV_phot_only()
        #print(f"rest frame UV bands = {phot_rest_copy.phot_obs.instrument.bands}")
        self.rest_UV_phot = phot_rest_copy
    
    def rest_UV_phot_only(self):
        crop_indices = []
        for i, (wav, wav_err) in enumerate(zip(self.wav.value, self.wav_errs.value)):
            wav *= u.Angstrom
            wav_err *= u.Angstrom
            if wav - wav_err < self.rest_UV_wav_lims[0] or wav + wav_err > self.rest_UV_wav_lims[1]:
                crop_indices = np.append(crop_indices, i)
        self.crop_phot(crop_indices)
     
    def beta_slope_power_law_func(wav_rest, A, beta):
        return (10 ** A) * (wav_rest ** beta)
    
    def set_ext_source_UV_corr(self, UV_ext_source_corr):
        self.UV_ext_src_corr = UV_ext_source_corr
        
    def plot(self, save_dir, ID, plot_fit = True, iters = 10_000, save = True, show = False, n_interp = 100, conv_filt = False):
        self.make_rest_UV_phot()
        assert(conv_filt == False)
        #if not all(beta == -99. for beta in self.beta_PDF):
        sns.set(style="whitegrid")
        warnings.filterwarnings("ignore")

        # Create figure and axes
        fig, ax = plt.subplots()
        
        # Plotting code
        ax.errorbar(np.log10(self.rest_UV_phot.wav.value), self.rest_UV_phot.log_flux_lambda, yerr = self.rest_UV_phot.log_flux_lambda_errs,
                     ls = "none", c = "black", zorder = 10, marker = "o", markersize = 5, capsize = 3)
        
        if plot_fit:
            fit_lines = []
            fit_lines_interped = []
            wav_interp = np.linspace(np.log10(self.rest_UV_phot.wav.value)[0], np.log10(self.rest_UV_phot.wav.value)[-1], n_interp)
            if iters > len(self.amplitude_PDF[conv_filt]):
                iters = len(self.amplitude_PDF[conv_filt])
            for i in range(iters):
                #percentiles = np.percentile([16, 50, 84], axis=0)
                f_interp = interp1d(np.log10(self.rest_UV_phot.wav.value), np.log10(Photometry_rest.beta_slope_power_law_func(self.rest_UV_phot.wav.value, \
                    self.amplitude_PDF[conv_filt][i], self.beta_PDF[conv_filt][i])), kind = 'linear')
                y_new = f_interp(wav_interp)
                fit_lines.append(np.log10(Photometry_rest.beta_slope_power_law_func(self.rest_UV_phot.wav.value, self.amplitude_PDF[conv_filt][i], self.beta_PDF[conv_filt][i])))
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
        ax.text(0.95, 0.95, r"$\beta$" + " = {:.2f} $^{{+{:.2f}}}_{{-{:.2f}}}$".format(np.percentile(self.beta_PDF[conv_filt], 50), \
            np.percentile(self.beta_PDF[conv_filt], 84) - np.percentile(self.beta_PDF[conv_filt], 50), np.percentile(self.beta_PDF[conv_filt], 50) - \
            np.percentile(self.beta_PDF[conv_filt], 16)), transform = ax.transAxes, ha = "right", va = "top", fontsize = 12)
            
        ax.set_xlim(*np.log10(self.rest_UV_wav_lims.value))
        
        if save:
            path = f"{save_dir}/plots/{ID}.png"
            funcs.make_dirs(path)
            fig.savefig(path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.tight_layout()
            plt.show()
        
        plt.clf()

    def basic_beta_calc(self, conv_filt = True, incl_errs = True, output_errs = False):
        self.make_rest_UV_phot()
        #print(self.rest_UV_phot.wav, self.rest_UV_phot.flux_lambda)
        try:
            if incl_errs:
                popt, pcov = curve_fit(Photometry_rest.beta_slope_power_law_func, self.rest_UV_phot.wav, self.rest_UV_phot.flux_lambda, sigma = self.rest_UV_phot.flux_lambda_errs, maxfev = 1_000)
            else:
                popt, pcov = curve_fit(Photometry_rest.beta_slope_power_law_func, self.rest_UV_phot.wav, self.rest_UV_phot.flux_lambda, maxfev = 1_000)
            beta = popt[1]
            if output_errs:
                beta_err = np.sqrt(pcov[1][1])
                return beta, beta_err
            else:
                return beta
        except:
            return None
        
    def basic_UV_properties_calc(self, UV_ext_src_corr, conv_filt = True, incl_errs = True, output_errs = False, maxfev = 100_000):
        self.make_rest_UV_phot()
        #print(self.z, len(self.rest_UV_phot.instrument))
        if len(self.rest_UV_phot.instrument) >= 2:
            if conv_filt:
                if incl_errs:
                    popt, pcov = curve_fit(beta_fit(self.z, self.rest_UV_phot.instrument).beta_slope_power_law_func_conv_filt, None, self.rest_UV_phot.flux_lambda, sigma = self.rest_UV_phot.flux_lambda_errs, maxfev = 1_000)
                else:
                    popt, pcov = curve_fit(beta_fit(self.z, self.rest_UV_phot.instrument).beta_slope_power_law_func_conv_filt, None, self.rest_UV_phot.flux_lambda, maxfev = maxfev)
            else:
                if incl_errs:
                    popt, pcov = curve_fit(Photometry_rest.beta_slope_power_law_func, self.rest_UV_phot.wav, self.rest_UV_phot.flux_lambda, sigma = self.rest_UV_phot.flux_lambda_errs, maxfev = maxfev)
                else:
                    popt, pcov = curve_fit(Photometry_rest.beta_slope_power_law_func, self.rest_UV_phot.wav, self.rest_UV_phot.flux_lambda, maxfev = maxfev)
            amplitude = popt[0]
            beta = popt[1]
            flux_lambda_1500 = Photometry_rest.beta_slope_power_law_func(1500., amplitude, beta) * UV_ext_src_corr * u.erg / (u.s * (u.cm ** 2) * u.Angstrom)
            m_UV = funcs.flux_to_mag((flux_lambda_1500 * ((1500. * u.Angstrom) ** 2) / const.c).to(u.Jy), 8.9)
            M_UV = m_UV - 5 * np.log10(self.lum_distance.value / 10) + 2.5 * np.log10(1 + self.z)
            return beta, m_UV, M_UV
        else:
            return -99., -99., -99.
    
    def basic_M_UV_calc(self, UV_ext_src_corr, incl_errs = True):
        self.make_rest_UV_phot()
        try:
            if incl_errs:
                popt, pcov = curve_fit(Photometry_rest.beta_slope_power_law_func, self.rest_UV_phot.wav, self.rest_UV_phot.flux_lambda, sigma = self.rest_UV_phot.flux_lambda_errs, maxfev = 1_000)
            else:
                popt, pcov = curve_fit(Photometry_rest.beta_slope_power_law_func, self.rest_UV_phot.wav, self.rest_UV_phot.flux_lambda, maxfev = 1_000)
            amplitude = popt[0]
            beta = popt[1]
            flux_lambda_1500 = Photometry_rest.beta_slope_power_law_func(1500., amplitude, beta) * UV_ext_src_corr * u.erg / (u.s * (u.cm ** 2) * u.Angstrom)
            M_UV = funcs.flux_to_mag((flux_lambda_1500 * ((1500. * u.Angstrom) ** 2) / const.c).to(u.Jy), 8.9) - 5 * np.log10(self.lum_distance.value / 10) + 2.5 * np.log10(1 + self.z)
            return M_UV
        except:
            return None
        
    def fit_UV_slope(self, save_dir, ID, z_PDF = None, iters = 1_000, plot = True, conv_filt = False, maxfev = 100_000): # 1D redshift PDF
        #print(f"Fitting UV slope for {ID}")
        if not hasattr(self, "amplitude_PDF"):
            self.amplitude_PDF = {}
            self.beta_PDF = {}
        # avoid fitting twice
        if conv_filt not in self.amplitude_PDF.keys():
            self.make_rest_UV_phot()
            fluxes = np.array([np.random.normal(mu.value, sigma.value, iters) for mu, sigma in zip(self.rest_UV_phot.flux_lambda, self.rest_UV_phot.flux_lambda_errs)]).T
            
            if z_PDF != None:
                # vary within redshift errors
                pass
            try:
                if conv_filt:
                    popt_arr = np.array([curve_fit(beta_fit(self.z, self.rest_UV_phot.instrument).beta_slope_power_law_func_conv_filt, None, flux, maxfev = maxfev)[0] for flux in tqdm(fluxes, desc = f"Fitting conv_filt UV properties for {str(ID)}")])
                else:
                    popt_arr = np.array([curve_fit(Photometry_rest.beta_slope_power_law_func, self.rest_UV_phot.wav, flux, maxfev = maxfev)[0] for flux in tqdm(fluxes, desc = f"Fitting pure PL UV properties for {str(ID)}")])
            except Exception as e:
                #print(traceback.format_exc())
                popt_arr = []
                
            if len(popt_arr) > 1:
                amplitude_PDF = popt_arr.T[0]
                beta_PDF = popt_arr.T[1]
            else:
                amplitude_PDF = [-99.]
                beta_PDF = [-99.]
            
            self.amplitude_PDF = {**self.amplitude_PDF, **{conv_filt: amplitude_PDF}} # unitless
            self.beta_PDF = {**self.beta_PDF, **{conv_filt: beta_PDF}} # unitless
            
            for name in ["Amplitude", "Beta"]:
                self.save_UV_fit_PDF(save_dir, name, ID, conv_filt = conv_filt)

        return amplitude_PDF, beta_PDF
    
    def calc_flux_lambda_1500_PDF(self, save_dir, ID, UV_ext_src_corr, conv_filt = False):
        self.open_UV_fit_PDF(save_dir, "Amplitude", ID, conv_filt = conv_filt)
        self.open_UV_fit_PDF(save_dir, "Beta", ID, conv_filt = conv_filt)
        if not hasattr(self, "flux_lambda_1500_PDF"):
            self.flux_lambda_1500_PDF = {}
        if all(value == -99 for value in self.amplitude_PDF[conv_filt]):
            flux_lambda_1500_PDF = [-99.]
        else:
            flux_lambda_1500_PDF = (Photometry_rest.beta_slope_power_law_func(1500., self.amplitude_PDF[conv_filt], self.beta_PDF[conv_filt]) \
                                * UV_ext_src_corr) * u.erg / (u.s * (u.cm ** 2) * u.Angstrom)
        self.flux_lambda_1500_PDF = {**self.flux_lambda_1500_PDF, **{conv_filt: flux_lambda_1500_PDF}}
        self.save_UV_fit_PDF(save_dir, "flux_lambda_1500", ID, conv_filt = conv_filt)
        return flux_lambda_1500_PDF
    
    def calc_flux_Jy_1500_PDF(self, save_dir, ID, UV_ext_src_corr = None, conv_filt = False):
        self.open_UV_fit_PDF(save_dir, "flux_lambda_1500", ID, UV_ext_src_corr = UV_ext_src_corr, conv_filt = conv_filt)
        if not hasattr(self, "flux_Jy_1500_PDF"):
            self.flux_Jy_1500_PDF = {}
        if all(value == -99 for value in self.amplitude_PDF[conv_filt]):
            flux_Jy_1500_PDF = [-99.]
        else:
            flux_Jy_1500_PDF = (self.flux_lambda_1500_PDF[conv_filt] * ((1500. * u.Angstrom) ** 2) / const.c).to(u.Jy)
        self.flux_Jy_1500_PDF = {**self.flux_Jy_1500_PDF, **{conv_filt: flux_Jy_1500_PDF}}
        self.save_UV_fit_PDF(save_dir, "flux_Jy_1500", ID, conv_filt = conv_filt)
        return flux_Jy_1500_PDF
    
    def calc_M_UV_PDF(self, save_dir, ID, UV_ext_src_corr = None, conv_filt = False):
        self.open_UV_fit_PDF(save_dir, "flux_Jy_1500", ID, UV_ext_src_corr = UV_ext_src_corr, conv_filt = conv_filt)
        if not hasattr(self, "m_UV_PDF"):
            self.m_UV_PDF = {}
        if not hasattr(self, "M_UV_PDF"):
            self.M_UV_PDF = {}
        if all(value == -99 for value in self.amplitude_PDF[conv_filt]):
            m_UV_PDF = [-99.]
            M_UV_PDF = [-99.]
        else:
            m_UV_PDF = funcs.flux_to_mag(self.flux_Jy_1500_PDF[conv_filt], 8.9)
            M_UV_PDF = m_UV_PDF - 5 * np.log10(self.lum_distance.value / 10) + 2.5 * np.log10(1 + self.z)
        self.m_UV_PDF = {**self.m_UV_PDF, **{conv_filt: m_UV_PDF}}
        self.M_UV_PDF = {**self.M_UV_PDF, **{conv_filt: M_UV_PDF}}
        self.save_UV_fit_PDF(save_dir, "M_UV", ID, conv_filt = conv_filt)
        return M_UV_PDF
    
    def calc_A_UV_PDF(self, save_dir, ID, conv_filt = False, scatter_dex = 0.5):
        self.open_UV_fit_PDF(save_dir, "Beta", ID, conv_filt = conv_filt)
        if not hasattr(self, "A_UV_PDF"):
            self.A_UV_PDF = {}
        if all(value == -99 for value in self.amplitude_PDF[conv_filt]):
            A_UV_PDF = [-99.]
        else:
            A_UV_PDF = 4.43 + (1.99 * self.beta_PDF[conv_filt]) + np.random.uniform(-scatter_dex, scatter_dex)
        self.A_UV_PDF = {**self.A_UV_PDF, **{conv_filt: A_UV_PDF}}
        self.save_UV_fit_PDF(save_dir, "A_UV", ID, conv_filt = conv_filt)
        return A_UV_PDF
    
    def calc_L_obs_PDF(self, save_dir, ID, UV_ext_src_corr = None, conv_filt = False, alpha = 0.): # α=0 in Donnan 2022
        self.open_UV_fit_PDF(save_dir, "flux_Jy_1500", ID, UV_ext_src_corr = UV_ext_src_corr, conv_filt = conv_filt)
        if not hasattr(self, "L_obs_PDF"):
            self.L_obs_PDF = {}
        if all(value == -99 for value in self.amplitude_PDF[conv_filt]):
            L_obs_PDF = [-99.]
        else:
            L_obs_PDF = ((4 * np.pi * self.flux_Jy_1500_PDF[conv_filt] * self.lum_distance ** 2) / ((1 + self.z) ** (1 + alpha))).to(u.erg / (u.s * u.Hz))
        self.L_obs_PDF = {**self.L_obs_PDF, **{conv_filt: L_obs_PDF}}
        self.save_UV_fit_PDF(save_dir, "L_obs", ID, conv_filt = conv_filt)
        return L_obs_PDF
    
    def calc_L_int_PDF(self, save_dir, ID, UV_ext_src_corr = None, conv_filt = False):
        self.open_UV_fit_PDF(save_dir, "L_obs", ID, UV_ext_src_corr = UV_ext_src_corr, conv_filt = conv_filt)
        self.open_UV_fit_PDF(save_dir, "A_UV", ID, conv_filt = conv_filt)
        if not hasattr(self, "L_int_PDF"):
            self.L_int_PDF = {}
        if all(value == -99 for value in self.amplitude_PDF[conv_filt]):
            L_int_PDF = [-99.]
        else:
            L_int_PDF = np.array([L_obs * 10 ** (A_UV / 2.5) if A_UV > 0 else L_obs for L_obs, A_UV in zip(self.L_obs_PDF[conv_filt].value, self.A_UV_PDF[conv_filt])]) * (u.erg / (u.s * u.Hz))
        self.L_int_PDF = {**self.L_int_PDF, **{conv_filt: L_int_PDF}}
        self.save_UV_fit_PDF(save_dir, "L_int", ID, conv_filt = conv_filt)
        return L_int_PDF
    
    def calc_SFR_PDF(self, save_dir, ID, UV_ext_src_corr = None, conv_filt = False):
        self.open_UV_fit_PDF(save_dir, "L_int", ID, UV_ext_src_corr = UV_ext_src_corr, conv_filt = conv_filt)
        if not hasattr(self, "SFR_PDF"):
            self.SFR_PDF = {}
        if all(value == -99 for value in self.amplitude_PDF[conv_filt]):
            SFR_PDF = [-99.]
        else:
            SFR_PDF = 1.15e-28 * self.L_int_PDF[conv_filt].value * u.solMass / u.yr # Madau, Dickinson 2014, FSPS, Salpeter IMF templates
        self.SFR_PDF = {**self.SFR_PDF, **{conv_filt: SFR_PDF}}
        self.save_UV_fit_PDF(save_dir, "SFR", ID, conv_filt = conv_filt)
        return SFR_PDF
    
    def save_UV_fit_PDF(self, save_dir, obs_name, ID, UV_ext_src_corr = None, conv_filt = False, plot_PDF = config.getboolean("RestUVProperties", "PLOT_PDFS")):
        PDF = self.open_UV_fit_PDF(save_dir, obs_name, ID, UV_ext_src_corr, conv_filt = conv_filt)
        try:
            unit = PDF.unit # keep track of PDF units
            PDF = np.array([val.value for val in PDF]) # make PDF unitless
        except:
            unit = "dimensionless"
        
        if plot_PDF:
            funcs.PDF_hist(PDF, save_dir, obs_name, ID, show = True, save = True)
        funcs.save_PDF(PDF, f"{obs_name}, units = {unit}, iters = {len(PDF)}", funcs.PDF_path(save_dir, obs_name, ID, self.rest_UV_wav_lims.value, conv_filt = conv_filt))
    
    def open_UV_fit_PDF(self, save_dir, obs_name, ID, UV_ext_src_corr = None, conv_filt = False, plot = True):
        # if obs_name == "flux_lambda_1500":
        #     print(f"UV_ext_src_corr = {UV_ext_src_corr}")
        try:
            # attempt to open PDF from object
            PDF = self.obs_name_to_PDF(obs_name, conv_filt = conv_filt)
        except: # if PDF not in object already
            try:
                # attempt to load the previously saved PDF from directory
                PDF = self.load_UV_fit_PDF(save_dir, obs_name, ID, conv_filt = conv_filt)
                #print(f"Loaded {obs_name} UV fit successfully for {ID}!")
            except: # if PDF not in object and not already saved
                # calculate the PDF (can take on the order of minutes depending on PDF iters)
                if obs_name == "Amplitude" or obs_name == "Beta":
                    self.fit_UV_slope(save_dir, ID, conv_filt = conv_filt)
                elif obs_name == "flux_lambda_1500":
                    self.calc_flux_lambda_1500_PDF(save_dir, ID, UV_ext_src_corr = UV_ext_src_corr, conv_filt = conv_filt)
                elif obs_name == "flux_Jy_1500":
                    self.calc_flux_Jy_1500_PDF(save_dir, ID, UV_ext_src_corr = UV_ext_src_corr, conv_filt = conv_filt)
                elif obs_name == "M_UV":
                    self.calc_M_UV_PDF(save_dir, ID, UV_ext_src_corr = UV_ext_src_corr, conv_filt = conv_filt)
                elif obs_name == "A_UV":
                    self.calc_A_UV_PDF(save_dir, ID, conv_filt = conv_filt)
                elif obs_name == "L_obs":
                    self.calc_L_obs_PDF(save_dir, ID, UV_ext_src_corr = UV_ext_src_corr, conv_filt = conv_filt)
                elif obs_name == "L_int":
                    self.calc_L_int_PDF(save_dir, ID, UV_ext_src_corr = UV_ext_src_corr, conv_filt = conv_filt)
                elif obs_name == "SFR":
                    self.calc_SFR_PDF(save_dir, ID, UV_ext_src_corr = UV_ext_src_corr, conv_filt = conv_filt)
                else:
                    raise(Exception(f"{obs_name} not valid for calculating UV fit to photometry for {ID}!"))
                PDF = self.obs_name_to_PDF(obs_name, conv_filt = conv_filt)
                
        if obs_name == "Beta" and plot and not conv_filt: # and not all(beta == -99. for beta in self.beta_PDF):
            self.plot(save_dir, ID)
        return PDF
    
    # this function and the one below could be included in the open_UV_fit_PDF
    def load_UV_fit_PDF(self, save_dir, obs_name, ID, conv_filt = False):
        # load PDF from directory
        PDF = np.array(np.loadtxt(f"{funcs.PDF_path(save_dir, obs_name, ID, self.rest_UV_wav_lims.value, conv_filt = conv_filt)}.txt"))
        if obs_name == "Amplitude":
            self.amplitude_PDF[conv_filt] = PDF
        elif obs_name == "Beta":
            self.beta_PDF[conv_filt] = PDF
        elif obs_name == "flux_lambda_1500":
            self.flux_lambda_1500_PDF[conv_filt] = PDF * u.erg / (u.s * (u.cm ** 2) * u.Angstrom)
        elif obs_name == "flux_Jy_1500":
            self.flux_Jy_1500_PDF[conv_filt] = PDF * u.Jy
        elif obs_name == "M_UV":
            self.M_UV_PDF[conv_filt] = PDF
        elif obs_name == "A_UV":
            self.A_UV_PDF[conv_filt] = PDF
        elif obs_name == "L_obs":
            self.L_obs_PDF[conv_filt] = PDF * u.erg / (u.s * u.Hz)
        elif obs_name == "L_int":
            self.L_int_PDF[conv_filt] = PDF * u.erg / (u.s * u.Hz)
        elif obs_name == "SFR":
            self.SFR_PDF[conv_filt] = PDF * u.solMass / u.yr
        else:
            raise(Exception(f"{obs_name} not valid for loading UV fit to photometry for {ID}!"))
        return PDF
    
    # this function and the one above could be included in the open_UV_fit_PDF
    def obs_name_to_PDF(self, obs_name, conv_filt = False):
        if obs_name == "Amplitude":
            return self.amplitude_PDF[conv_filt]
        elif obs_name == "Beta":
            return self.beta_PDF[conv_filt]
        elif obs_name == "flux_lambda_1500":
            return self.flux_lambda_1500_PDF[conv_filt]
        elif obs_name == "flux_Jy_1500":
            return self.flux_Jy_1500_PDF[conv_filt]
        elif obs_name == "M_UV":
            return self.M_UV_PDF[conv_filt]
        elif obs_name == "A_UV":
            return self.A_UV_PDF[conv_filt]
        elif obs_name == "L_obs":
            return self.L_obs_PDF[conv_filt]
        elif obs_name == "L_int":
            return self.L_int_PDF[conv_filt]
        elif obs_name == "SFR":
            return self.SFR_PDF[conv_filt]
        else:
            raise(Exception(f"No {obs_name} PDF found in object with conv_filt = {conv_filt}!"))
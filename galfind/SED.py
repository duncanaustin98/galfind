#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 13:30:14 2023

@author: u92876da
"""

# SED.py

import astropy.units as u
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import curve_fit

from . import config, astropy_cosmo, Photometry, Photometry_obs, Mock_Photometry, IGM_attenuation, wav_lyman_lim, wav_lyman_alpha
from . import useful_funcs_austind as funcs

class SED:
    
    def __init__(self, wavs, mags, wav_units, mag_units):
        self.wavs = wavs * wav_units
        self.mags = mags * mag_units
        #self.mag_units = mag_units
    
    def convert_wav_units(self, units, update = True):
        wavs = self.wavs.to(units)
        if update:
            self.wavs = wavs
        return wavs
    
    def convert_mag_units(self, units, update = True):
        if units == self.mags.unit:
            mags = self.mags
        elif units == u.ABmag:
            if u.get_physical_type(self.mags.unit) == "ABmag/spectral flux density": # f_ν -> derivative of u.Jy
                mags = self.mags.to(u.ABmag)
            elif u.get_physical_type(self.mags.unit) == "power density/spectral flux density wav": # f_λ -> derivative of u.erg / (u.s * (u.cm ** 2) * u.AA)
                mags = self.mags.to(u.ABmag, equivalencies = u.spectral_density(self.wavs))
        elif u.get_physical_type(units) == "ABmag/spectral flux density": # f_ν -> derivative of u.Jy
            if self.mags.unit == u.ABmag:
                mags = self.mags.to(units)
            elif u.get_physical_type(self.mags.unit) == "power density/spectral flux density wav" or u.get_physical_type(self.mags.unit) == "ABmag/spectral flux density":
                mags = self.mags.to(units, equivalencies = u.spectral_density(self.wavs))
        
        elif u.get_physical_type(units) == "power density/spectral flux density wav": # f_λ -> derivative of u.erg / (u.s * (u.cm ** 2) * u.AA):
            #if self.mags.unit == u.ABmag:
            mags = self.mags.to(units, equivalencies = u.spectral_density(self.wavs))
            #elif u.get_physical_type(self.mags.unit) == "spectral flux density": # f_ν -> derivative of u.Jy
            #mags = self.mags.to(units, equivalencies = u.spectral_density(self.wavs))
        else:
            raise(Exception("Units must be either ABmag or have physical units of 'spectral flux density' or 'power density/spectral flux density wav'!"))
        if update:
            self.mags = mags
        return mags
        
    def plot_SED(self, ax, wav_units = u.AA, mag_units = u.ABmag, label = None, annotate = True, plot_kwargs = {}, legend_kwargs = {}):
        wavs = self.convert_wav_units(wav_units, update = False)
        mags = self.convert_mag_units(mag_units, update = False)
        plot = ax.plot(wavs, mags, label = label, **plot_kwargs)
        if annotate:
            if wav_units == u.AA:
                ax.set_xlabel(r"$\lambda / \mathrm{\AA}$")
            elif wav_units == u.um:
                ax.set_xlabel(r"$\lambda / \mu\mathrm{m}$")
            else:
                ax.set_xlabel(r"$\lambda / \mathrm{%s}$" % str(wav_units))
            if mag_units == u.ABmag:
                y_label = r"$m_{\mathrm{AB}}$"
                plt.gca().invert_yaxis()
            elif u.get_physical_type(mag_units) == "spectral flux density":
                y_label = r"$f_{\nu} / \mathrm{%s}$" % str(mag_units)
            elif u.get_physical_type(mag_units) == "power density/spectral flux density wav":
                y_label = r"$f_{\lambda} / \mathrm{%s}$" % str(mag_units)
            ax.set_ylabel(y_label)
            ax.legend(**legend_kwargs)
        return plot


class SED_rest(SED):
    
    def __init__(self, wavs, mags, wav_units, mag_units, wav_range = [900, 100_000] * u.AA):
        mags = mags[(wavs.value > wav_range.to(wav_units).value[0]) & (wavs.value < wav_range.to(wav_units).value[1])]
        wavs = wavs[(wavs.value > wav_range.to(wav_units).value[0]) & (wavs.value < wav_range.to(wav_units).value[1])]
        super().__init__(wavs, mags, wav_units, mag_units)
        
    def crop_to_Calzetti94_filters(self, update = False):
        wavs = self.wavs.to(u.AA)
        Calzetti94_filter_indices = np.logical_or.reduce([(wavs.value > low_lim) & (wavs.value < up_lim) \
                        for low_lim, up_lim in zip(funcs.lower_Calzetti_filt, funcs.upper_Calzetti_filt)])
        wavs = self.wavs[Calzetti94_filter_indices]
        mags = self.mags[Calzetti94_filter_indices]
        if update:
            self.wavs = wavs
            self.wav_units = u.AA # should improve this functionality
            self.mags = mags
        return wavs, mags

class SED_obs(SED):
    
    def __init__(self, z, wavs, mags, wav_units, mag_units):
        self.z = z
        super().__init__(wavs, mags, wav_units, mag_units)
    
    @classmethod
    def from_SED_rest(cls, z_int, SED_rest):
        wav_obs = funcs.wav_rest_to_obs(SED_rest.wavs, z_int)
        if SED_rest.mags.unit == u.ABmag:
            mag_obs = SED_rest.mags
        else:
            raise(Exception("Not yet implemented!"))
        return cls(z_int, wav_obs, mag_obs, SED_rest.wavs.unit, SED_rest.mags.unit)
        
    def create_mock_photometry(self, instrument, depth_dict = {}, min_pc_err = 0.):
        # calculate band pass averaged fluxes from each band within the instrument
        instrument.load_instrument_filter_profiles()
        bp_averaged_fluxes = []
        depths = []
        given_depth_dir = True
        for i, band in enumerate(instrument):
            # interpolate filter to be on the same wavelength grid as the SED template
            band_transmission = interp1d(np.array(instrument.filter_profiles[band]["Wavelength"]), \
                    np.array(instrument.filter_profiles[band]["Transmission"]), fill_value = "extrapolate")
            band_interp = band_transmission(self.wavs.value)
            # convert self.mags to f_ν
            mags = self.convert_mag_units(u.Jy, update = False)
            # calculate integral(f(ν) * T(ν)dν)
            #numerator = np.trapz(mags.value * band_interp, x = self.wavs.value)
            numerator = quad(interp1d(self.wavs.value, mags.value * band_interp), np.min(np.array(instrument.filter_profiles[band]["Wavelength"])), \
                             np.max(np.array(instrument.filter_profiles[band]["Wavelength"])))[0]
            # calculate integral(T(ν)dν)
            #denominator = np.trapz(band_interp, x = self.wavs.value)
            denominator = quad(interp1d(self.wavs.value, band_interp), np.min(np.array(instrument.filter_profiles[band]["Wavelength"])), \
                             np.max(np.array(instrument.filter_profiles[band]["Wavelength"])))[0]
            # calculate bandpass-averaged flux in Jy
            bp_averaged_fluxes.append(numerator / denominator)
            if given_depth_dir:
                try:
                    depths.append(depth_dict[band])
                except:
                    print(f"Not given depth value for depth_dict['{band}']")
        self.mock_photometry = Mock_Photometry(instrument, np.array(bp_averaged_fluxes) * u.Jy, np.array(depths), min_pc_err)

# class Mock_SED(SED):
    
#     def __init__(self, wavs, mags, wav_units, mag_units, template_name):
#         self.template_name = template_name
#         super().__init__(wavs, mags, wav_units, mag_units)
#         pass
    
    
class Mock_SED_rest(SED_rest): #, Mock_SED):
    
    def __init__(self, wavs, mags, wav_units, mag_units, template_name):
        self.template_name = template_name
        super().__init__(wavs, mags, wav_units, mag_units)
        
    @classmethod
    def from_Mock_SED_obs(cls, mock_SED_obs, out_wav_units = u.AA, out_mag_units = u.ABmag, IGM_out = None):
        wavs = funcs.convert_wav_units(mock_SED_obs.wavs / (1 + mock_SED_obs.z), out_wav_units)
        mags = funcs.convert_mag_units(wavs, mock_SED_obs.mags, out_mag_units)
        # ensure IGM output is of the correct type
        mock_sed_rest_obj = cls(wavs.value, mags.value, wavs.unit, mags.unit, mock_SED_obs.template_name)
        if IGM_out == None:
            mock_sed_rest_obj.un_attenuate_IGM(mock_SED_obs.IGM)
        elif isinstance(IGM_out, IGM_attenuation.IGM):
            if IGM_out.prescription != mock_SED_obs.IGM.prescription:
                raise(Exception("Not currently included the functionality to swap IGM attenuation whilst creating new Mock_SED_rest object from Mock_SED_obs object yet"))
        else:
            raise(Exception(f"'IGM_out' = {IGM_out} must be either 'None' or 'IGM' class"))
        return mock_sed_rest_obj
        
    @classmethod
    def power_law_from_beta_m_UV(cls, beta, m_UV, wav_range = [912., 5_000.], wav_res = 1):
        wavs = np.linspace(wav_range[0], wav_range[1], int((wav_range[1] - wav_range[0]) / wav_res))
        print("Still need to include Inoue+2014 Lyman alpha forest attenuation here, relevant at λ<1216 Angstrom!")
        mags = funcs.flux_to_mag(funcs.flux_lambda_to_Jy(((wavs * u.Angstrom) ** beta) \
                .to(u.erg / (u.s * (u.cm ** 2) * u.Angstrom)), 1_500. * u.Angstrom), 8.9)
        mock_sed = cls(wavs, mags, u.AA, u.ABmag)
        mock_sed.normalize_to_m_UV(m_UV)
        return mock_sed
    
    @classmethod
    def load_SED_in_template(cls, code_name, m_UV, template_set, template_number):
        if code_name == "EAZY":
            return cls.load_EAZY_in_template(m_UV, template_set, template_number)
        else:
            raise(Exception(f"Rest frame template load in currently unavailable for code_name = {code_name}"))
    
    @classmethod
    def load_EAZY_in_template(cls, m_UV, template_set, template_number):
        EAZY_template_path = "/Users/user/Documents/PGR/SED_templates/EAZY"
        EAZY_template_units = {"fsps_larson": {"wavs": u.AA, "mags": u.erg / (u.s * (u.cm ** 2) * u.AA)}}
        # read in .txt file if it exists
        template_labels = open(f"{EAZY_template_path}/{template_set}.txt", "r")
        template_filename = template_labels.readlines()[template_number - 1].replace("\n", "")
        template_labels.close()
        template = Table.read(f"{EAZY_template_path}/templates/{template_filename}", names = ["Wavelength", "SED"], format = "ascii")
        # restrict template to appropriate wavelength range
        template_obj = cls(template["Wavelength"], template["SED"], EAZY_template_units[template_set]["wavs"], \
                           EAZY_template_units[template_set]["mags"], template_filename.split("/")[1])
        template_obj.convert_mag_units(u.Jy, update = True)
        template_obj.convert_wav_units(u.AA, update = True)
        template_obj.normalize_to_m_UV(m_UV)
        return template_obj
    
    def normalize_to_m_UV(self, m_UV):
        if not m_UV == None:
            norm = (m_UV * u.ABmag).to(u.Jy).value / self.mags.to(u.Jy).value[np.abs(self.wavs.to(u.AA).value - 1500.).argmin()]
            self.mags = np.array([norm * mag for mag in self.mags.value]) * self.mags.unit
            
    def renorm_at_wav(self, mag): # this mag can also be a flux, but must have astropy units
        pass
    
    def un_attenuate_IGM(self, IGM):
        if isinstance(IGM, IGM_attenuation.IGM):
            # un_attenuate SED for IGM absorption
            # calculate rest wavelengths from self in the appropriate wavelength range between wav_lyman_lim and wav_lyman_alpha
            attenuated_indices = ((self.wavs.value > wav_lyman_lim) & (self.wavs.value < wav_lyman_alpha))
            IGM_transmission = IGM.interp_transmission(self.z, self.wavs[attenuated_indices])
            if self.mags.unit == u.ABmag:
                self.mags[attenuated_indices] = (self.mags[attenuated_indices].value - -2.5 * np.log10(IGM_transmission)) * u.ABmag
            else:
                self.mags[attenuated_indices] /= IGM_transmission
        elif IGM == None: # nothing to un-attenuate
            pass
        else:
            raise(Exception(f"Could not attenuate by a non IGM object = {IGM}"))
    
    def calc_UV_slope(self, output_errs = False, method = "Calzetti+94"):
        if method == "Calzetti+94":
            # crop to Calzetti+94 filters
            wavs, mags = self.crop_to_Calzetti94_filters()
            # convert mag units to f_λ
            mags = funcs.convert_mag_units(wavs, mags, u.erg / (u.s * (u.cm ** 2) * u.AA))
        popt, pcov = curve_fit(funcs.beta_slope_power_law_func, wavs.value, mags.value, maxfev = 1_000)
        A, beta = popt[0], popt[1]
        if output_errs:
            A_err = np.sqrt(pcov[0][0])
            beta_err = np.sqrt(pcov[1][1])
            return A, beta, A_err, beta_err
        else:
            return A, beta

class Mock_SED_obs(SED_obs):
    
    def __init__(self, z, wavs, mags, wav_units, mag_units, template_name, IGM = IGM_attenuation.IGM()):
        self.template_name = template_name
        super().__init__(z, wavs, mags, wav_units, mag_units)
        if IGM != None:
            self.attenuate_IGM(IGM)
    
    @classmethod
    def from_Mock_SED_rest(cls, mock_SED_rest, z, out_wav_units = u.AA, out_mag_units = u.ABmag, IGM = IGM_attenuation.IGM()):
        mags = mock_SED_rest.convert_mag_units(out_mag_units)
        wavs = (mock_SED_rest.wavs * (1 + z)).to(out_wav_units)
        mock_SED_obs_obj = cls(z, wavs.value, mags.value, out_wav_units, out_mag_units, mock_SED_rest.template_name, IGM)
        return mock_SED_obs_obj
    
    @classmethod
    def power_law_from_beta_M_UV(cls, z, beta, M_UV):
        lum_distance = astropy_cosmo.luminosity_distance(z).to(u.pc)
        m_UV = M_UV - 2.5 * np.log10(1 + z) + 5 * np.log10(lum_distance.value / 10)
        mock_SED_rest = Mock_SED_rest.from_beta_m_UV(beta, m_UV)
        obs_SED = cls.from_SED_rest(z, mock_SED_rest)
        obs_SED.attenuate_IGM()
        return obs_SED
    
    # def create_phot_obs(self, instrument, loc_depths, min_pc_err = 10):
    #     # calculate band pass averaged fluxes
    #     fluxes = []
    #     # calculate flux errors from loc depths
    #     flux_errs = [funcs.loc_depth_to_flux_err(depth, 8.9) if funcs.loc_depth_to_flux_err(depth, 8.9) * 100 / flux > min_pc_err \
    #                  else min_pc_err * flux / 100 for flux, depth in zip(fluxes, loc_depths.items)]
    #     return Photometry(instrument, fluxes, flux_errs, loc_depths)
    
    def attenuate_IGM(self, IGM = IGM_attenuation.IGM()):
        if not hasattr(self, IGM):
            self.IGM = None
        if isinstance(IGM, IGM_attenuation.IGM):
            if self.IGM == None: # not already been attenuated
                # attenuate SED for IGM absorption
                # calculate rest wavelengths from self in the appropriate wavelength range between wav_lyman_lim and wav_lyman_alpha
                wav_rest_arr = self.wavs / (1 + self.z)
                attenuated_indices = ((wav_rest_arr.value > wav_lyman_lim) & (wav_rest_arr.value < wav_lyman_alpha))
                IGM_transmission = IGM.interp_transmission(self.z, wav_rest_arr[attenuated_indices])
                if self.mags.unit == u.ABmag:
                    self.mags[attenuated_indices] = (self.mags[attenuated_indices].value -2.5 * np.log10(IGM_transmission)) * u.ABmag
                else:
                    self.mags[attenuated_indices] *= IGM_transmission
                # save IGM object after attenuating
                self.IGM = IGM
            else:
                print("SED has already been attenuated! Ignoring")
        else:
            raise(Exception(f"Could not attenuate by a non IGM object = {IGM}"))

    def scatter_mock_photometry(self):
        if not hasattr(self, "mock_photometry"):
            raise(Exception("Must have previously created mock photometry from the observed frame template"))
            
    def calc_UV_slope(self, output_errs = False, method = "Calzetti+94"):
        # create rest frame mock SED object
        mock_sed_rest = Mock_SED_rest.from_Mock_SED_obs(self)
        # calculate amplitude and beta of power law fit
        A, beta = mock_sed_rest.calc_UV_slope(output_errs = output_errs, method = method)
        return A, beta

class Mock_SED_template_set:
    
    def __init__(self, mock_SED_arr):
        self.SED_arr = mock_SED_arr
    
class Mock_SED_rest_template_set(Mock_SED_template_set):
    
    def __init__(self, mock_SED_rest_arr):
        super().__init__(mock_SED_rest_arr)
        
    @classmethod
    def load_SED_in_template(cls):
        pass

class Mock_SED_obs_template_set(Mock_SED_template_set):
    
    def __init__(self, mock_SED_obs_arr):
        super().__init__(mock_SED_obs_arr)
    

    
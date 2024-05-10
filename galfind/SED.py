#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 13:30:14 2023

@author: u92876da
"""

# SED.py

import astropy.units as u
import astropy.constants as const
import numpy as np
import astropy.io.ascii as ascii
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import curve_fit
import os
from copy import deepcopy
import glob

from . import config, astropy_cosmo, Photometry, Photometry_obs, Mock_Photometry, IGM_attenuation, wav_lyman_lim, Emission_lines
from . import useful_funcs_austind as funcs
from . import galfind_logger
from .Emission_lines import wav_lyman_alpha, line_diagnostics

class SED:
    # should include mag errors here
    def __init__(self, wavs, mags, wav_units, mag_units):
        self.wavs = wavs * wav_units
        self.mags = mags * mag_units
        #self.mag_units = mag_units

    def __str__(self):
        return "LOADED SED\n"
    
    def convert_wav_units(self, units, update = True):
        wavs = self.wavs.to(units)
        if update:
            self.wavs = wavs
        return wavs
    
    def convert_mag_units(self, units, update = True):
        if units == self.mags.unit:
            mags = self.mags
        elif units == u.ABmag:
            if u.get_physical_type(self.mags.unit) in ["ABmag/spectral flux density", "spectral flux density"]: # f_ν -> derivative of u.Jy
                mags = self.mags.to(u.ABmag)
            elif u.get_physical_type(self.mags.unit) == "power density/spectral flux density wav": # f_λ -> derivative of u.erg / (u.s * (u.cm ** 2) * u.AA)
                mags = self.mags.to(u.ABmag, equivalencies = u.spectral_density(self.wavs))
        elif u.get_physical_type(units) in ["ABmag/spectral flux density", "spectral flux density"]: # f_ν -> derivative of u.Jy
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
        wavs = funcs.convert_wav_units(self.wavs, wav_units)
        mags = funcs.convert_mag_units(self.wavs, self.mags, mag_units)

        if mag_units != u.ABmag:
            mags = funcs.log_scale_fluxes(mags)
        else:
            mags = mags.value

        plot = ax.plot(wavs.value, mags, label = label, **plot_kwargs)
        if annotate:
            ax.set_xlabel(funcs.label_wavelengths(wav_units, False, \
                "" if self.__class__.__name__.split("_")[-1] == "Photometry" else self.__class__.__name__.split("_")[-1]))
            ax.set_ylabel(funcs.label_fluxes(mag_units, False if mag_units == u.ABmag else True))
            ax.legend(**legend_kwargs)
        return plot
    
    def calc_bandpass_averaged_flux(self, filter_wavs, filter_trans):
        wavs = funcs.convert_wav_units(self.wavs, u.AA).value
        mags = funcs.convert_mag_units(self.wavs, self.mags, u.erg / (u.s * u.AA * u.cm ** 2)).value
        # update filter wavelengths to correct units
        filter_wavs = funcs.convert_wav_units(filter_wavs, u.AA).value
        # interpolate SED to be on same grid as filter_profile
        sed_interp = interp1d(wavs, mags, fill_value = "extrapolate")(filter_wavs) # in f_lambda
        # calculate integral(f(λ) * T(λ)dλ)
        numerator = np.trapz(sed_interp * filter_trans, x = filter_wavs)
        # calculate integral(T(λ)dλ)
        denominator = np.trapz(filter_trans, x = filter_wavs)
        # calculate bandpass-averaged flux in Jy
        return numerator / denominator
    
    def calc_line_EW(self, line_name, plot = False): # ONLY WORKS FOR EMISSION LINES, NOT ABSORPTION AT THIS POINT
        wavs_AA = self.convert_wav_units(u.AA, update = True)
        flux_lambda = self.convert_mag_units(u.erg / (u.s * u.cm ** 2 * u.AA), update = False)
        # calculate line + continuum flux
        if "rest" in type(self).__name__:
            line_lims = line_diagnostics[line_name]["feature_wavs"]
            cont_lims = line_diagnostics[line_name]["cont_wavs"]
        elif "obs" in type(self).__name__:
            line_lims = line_diagnostics[line_name]["feature_wavs"] * (1 + self.z)
            cont_lims = line_diagnostics[line_name]["cont_wavs"] * (1 + self.z)
        else:
            raise(Exception("Attempted EW calculation in a class not containing 'rest' or 'obs' in the class name!"))
        # mask everything but the line of interest
        feature_mask = (wavs_AA > line_lims[0].to(u.AA)) & (wavs_AA < line_lims[1].to(u.AA))
        if plot:
            plt.plot(wavs_AA[feature_mask], flux_lambda[feature_mask])
            plt.show()
        line_plus_cont_flux = np.trapz(flux_lambda[feature_mask], x = wavs_AA[feature_mask])
        # calculate continuum flux and mean continuum level
        cont_mask = np.logical_or.reduce(np.array([((wavs_AA > lims[0].to(u.AA)) & \
                        (wavs_AA < lims[1].to(u.AA))) for lims in cont_lims]))
        # mask everything but the line of interest
        cont_flux = interp1d(wavs_AA[cont_mask], flux_lambda[cont_mask], fill_value = "extrapolate")(wavs_AA[feature_mask]) * u.erg / (u.s * u.cm ** 2 * u.AA)
        mean_cont = np.trapz(cont_flux, x = wavs_AA[feature_mask]) # not 100% correct here in the case of cont > line flux
        # calculate line flux
        # if line flux goes negative at any point, set to zero (only works for emission and not absorption)
        line_flux_integrand = np.array([(flux_lambda[feature_mask][i] - cont_flux[i]).value if (flux_lambda[feature_mask][i] - cont_flux[i]).value > 0. else 0. for i, x in enumerate(wavs_AA[feature_mask])]) * u.erg / (u.s * u.cm ** 2 * u.AA)
        line_flux = np.trapz(line_flux_integrand, x = wavs_AA[feature_mask]) #(line_plus_cont_flux - cont_flux) * feature_width # emission == positive EW
        # calculate line EW
        line_EW = np.trapz(line_flux_integrand / cont_flux, x = wavs_AA[feature_mask])
        # save result in self
        if not hasattr(self, "line_EWs"):
            self.line_EWs = {line_name: line_EW}
            self.line_fluxes = {line_name: line_flux}
            self.line_cont = {line_name: mean_cont}
        else:
            self.line_EWs[line_name] = line_EW
            self.line_fluxes[line_name] = line_flux
            self.line_cont[line_name] = mean_cont
        return line_EW
    
    def calc_UVJ_colours(self, resolution = 1. * u.AA):
        UVJ_filters = {
        "U": {"lam_eff": 3_650. * u.AA, "lam_fwhm": 660. * u.AA},
        "V": {"lam_eff": 5_510. * u.AA, "lam_fwhm": 880. * u.AA},
        "J": {"lam_eff": 12_200. * u.AA, "lam_fwhm": 2130. * u.AA}
        } # these are rest frame wavelengths
    
        bp_averaged_fluxes = np.zeros(3)
        for i, band in enumerate(["U", "V", "J"]):
            galfind_logger.info(f"Calculating {band} fluxes")
            band_wavs = np.linspace((UVJ_filters[band]["lam_eff"] - UVJ_filters[band]["lam_fwhm"] / 2.), \
                (UVJ_filters[band]["lam_eff"] + UVJ_filters[band]["lam_fwhm"] / 2.), \
                int(np.round((UVJ_filters[band]["lam_fwhm"] / resolution).to(u.dimensionless_unscaled).value))) * u.AA
            filter_profile = {"Wavelength": band_wavs, "Transmission": np.ones(len(band_wavs))}
            bp_averaged_fluxes[i] = self.calc_bandpass_averaged_flux(filter_profile)
        # convert bp_averaged_fluxes to Jy
        bp_averaged_fluxes_Jy = funcs.convert_mag_units([UVJ_filters[band]["lam_eff"] \
            for band in ["U", "V", "J"]], bp_averaged_fluxes * u.erg / (u.s * (u.cm ** 2) * u.AA), u.Jy)
        
        self.UVJ_fluxes = {band: flux_Jy for band, flux_Jy in zip(["U", "V", "J"], bp_averaged_fluxes_Jy)}
        self.UVJ_colours = {"U-V": -2.5 * np.log10(self.UVJ_fluxes["U"] / self.UVJ_fluxes["V"]), \
                            "V-J": -2.5 * np.log10(self.UVJ_fluxes["V"] / self.UVJ_fluxes["J"])}

class SED_rest(SED):
    # should include mag errors here
    def __init__(self, wavs, mags, wav_units, mag_units, wav_range = [0, 10_000] * u.AA):
        try:
            wavs = wavs.value # if wavs is in Angstrom
        except:
            pass
        mags = mags[(wavs > wav_range.to(wav_units).value[0]) & (wavs < wav_range.to(wav_units).value[1])]
        wavs = wavs[(wavs > wav_range.to(wav_units).value[0]) & (wavs < wav_range.to(wav_units).value[1])]
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
    # should include mag errors here
    def __init__(self, z, wavs, mags, wav_units, mag_units):
        self.z = z
        super().__init__(wavs, mags, wav_units, mag_units)
    
    @classmethod
    def from_SED_rest(cls, z_int, SED_rest):
        wav_obs = funcs.wav_rest_to_obs(SED_rest.wavs, z_int)
        mag_obs = funcs.convert_mag_units(SED_rest.wavs, SED_rest.mags, u.ABmag)
        return cls(z_int, wav_obs.value, mag_obs.value, SED_rest.wavs.unit, u.ABmag)
    
    def create_mock_phot(self, instrument, depths = [], min_pc_err = 10.):
        if type(depths) == dict:
           depths = [depth for (band, depth) in depths.items()]
        if depths == []: # if depths not given, expect the galaxy to be very well detected
            depths = [99. for band in instrument.band_names]
        bp_averaged_fluxes = [self.calc_bandpass_averaged_flux(band.wav, band.trans) for band in instrument] * u.erg / (u.s * (u.cm ** 2) * u.AA)
        # convert bp_averaged_fluxes to Jy
        band_wavs = np.array([band.WavelengthCen.value for band in instrument]) * u.AA
        bp_averaged_fluxes_Jy = funcs.convert_mag_units(band_wavs, bp_averaged_fluxes, u.Jy)
        self.mock_phot = Mock_Photometry(instrument, bp_averaged_fluxes_Jy, depths, min_pc_err)
        return self.mock_phot
    

class Mock_SED_rest(SED_rest): #, Mock_SED):
    
    def __init__(self, wavs, mags, wav_units, mag_units, template_name = None, meta = None):
        self.template_name = template_name
        self.meta = meta
        super().__init__(wavs, mags, wav_units, mag_units)
        
    @classmethod
    def from_Mock_SED_obs(cls, mock_SED_obs, out_wav_units = u.AA, out_mag_units = u.ABmag, IGM = None):
        wavs = funcs.convert_wav_units(mock_SED_obs.wavs / (1 + mock_SED_obs.z), out_wav_units)
        mags = funcs.convert_mag_units(wavs, mock_SED_obs.mags, out_mag_units)
        # ensure IGM output is of the correct type
        mock_sed_rest_obj = cls(wavs.value, mags.value, wavs.unit, mags.unit, mock_SED_obs.template_name)
        # if IGM_out == None:
        #     mock_sed_rest_obj.un_attenuate_IGM(mock_SED_obs.z, mock_SED_obs.IGM)
        # elif isinstance(IGM_out, IGM_attenuation.IGM):
        #     if IGM_out.prescription != mock_SED_obs.IGM.prescription:
        #         raise(Exception("Not currently included the functionality to swap IGM attenuation whilst creating new Mock_SED_rest object from Mock_SED_obs object yet"))
        # else:
        #     raise(Exception(f"'IGM_out' = {IGM_out} must be either 'None' or 'IGM' class"))
        return mock_sed_rest_obj
        
    @classmethod
    def power_law_from_beta_m_UV(cls, beta, m_UV, wav_range = [912., 10_000.] * u.AA, wav_res = 1. * u.AA, template_name = None):
        wavs = np.linspace(wav_range[0].value, wav_range[1].value, \
            int(((wav_range[1] - wav_range[0]) / wav_res).to(u.dimensionless_unscaled).value)) * wav_range.unit
        mags = funcs.convert_wav_units(wavs, u.AA).value ** beta
        mock_sed = cls(wavs.value, mags, u.AA, u.erg / (u.s * u.AA * u.cm ** 2), template_name = template_name)
        mock_sed.normalize_to_m_UV(m_UV)
        return mock_sed
    
    @classmethod
    def load_SED_in_template(cls, code_name, m_UV, template_set, template_number):
        if code_name == "EAZY":
            return cls.load_EAZY_in_template(m_UV, template_set, template_number)
        elif code_name == "Bagpipes":
            return cls.load_pipes_in_template()
        else:
            raise(Exception(f"Rest frame template load in currently unavailable for code_name = {code_name}"))
    
    @classmethod
    def load_EAZY_in_template(cls, m_UV, template_set, template_filename):
        EAZY_template_units = {"fsps_larson": {"wavs": u.AA, "mags": u.erg / (u.s * (u.cm ** 2) * u.AA)}}
        if isinstance(template_filename, int):
            template_labels = open(f"{config['EAZY']['EAZY_TEMPLATE_DIR']}/{template_set}.txt", "r")
            template_filename = template_labels.readlines()[template_filename].replace("\n", "")
            template_labels.close()
        template = Table.read(f"{config['EAZY']['EAZY_TEMPLATE_DIR']}/{template_filename}", names = ["Wavelength", "SED"], format = "ascii")
        # restrict template to appropriate wavelength range
        template_obj = cls(template["Wavelength"], template["SED"], EAZY_template_units[template_set]["wavs"], \
                           EAZY_template_units[template_set]["mags"], template_filename.split("/")[1])
        template_obj.convert_mag_units(u.Jy, update = True)
        template_obj.convert_wav_units(u.AA, update = True)
        template_obj.normalize_to_m_UV(m_UV)
        return template_obj
    
    @classmethod
    def load_pipes_in_template(cls, m_UV, template_set, template_filename):
        pipes_template_units = {"wavs": u.AA, "mags": u.erg / (u.s * (u.cm ** 2))} # rest frame f_lambda
        if isinstance(template_filename, int):
            template_name = glob.glob(f"{config['Bagpipes']['BAGPIPES_TEMPLATE_DIR']}/{template_set}/*_{str(template_filename)}.ecsv")[0]
        template = Table.read(template_name, names = ["Wavelength", "SED"], format = "ascii.ecsv")
        # restrict template to appropriate wavelength range
        template_obj = cls(template["Wavelength"].value, template["SED"].value, pipes_template_units["wavs"], \
                           pipes_template_units["mags"] / u.AA, template_name.split("/")[-1].replace(".ecsv", ""), meta = dict(template.meta))
        template_obj.convert_mag_units(u.Jy, update = True)
        template_obj.convert_wav_units(u.AA, update = True)
        template_obj.normalize_to_m_UV(m_UV)
        return template_obj
    
    @classmethod
    def load_Yggdrasil_popIII_in_template(cls, imf, fcov, sfh, template_filename):
        #print("Incorrect normalization for yggdrasil input templates!")
        yggdrasil_dir = f"/Users/user/Documents/PGR/yggdrasil_grids/{imf}_fcov_{str(fcov)}_SFR_{sfh}_Spectra"
        # if isinstance(template_filename, int):
        #     SED_arr = glob.glob(yggdrasil_dir)
        #     print(SED_arr)
        template = Table.read(f"{yggdrasil_dir}/{template_filename}", format = "ascii")
        # convert template fluxes to appropriate units
        #  * (astropy_cosmo.luminosity_distance(z) ** 2 / (1 + z))
        fluxes = ((template["flux"] * (u.erg / (u.s * u.AA)) / (4 * np.pi * u.pc ** 2)).to(u.erg / (u.s * (u.cm ** 2) * u.AA))).value
        template_obj = cls(template["wav"], fluxes, u.AA, u.erg / (u.s * (u.cm ** 2) * u.AA), template_filename)
        template_obj.convert_mag_units(u.Jy, update = True)
        template_obj.convert_wav_units(u.AA, update = True)
        template_obj.age = template.meta["age"] * u.Myr
        return template_obj
    
    def normalize_to_m_UV(self, m_UV):
        if not type(m_UV) == type(None):
            assert(type(m_UV) in [u.Quantity, u.Magnitude])
            norm = funcs.convert_mag_units(1_500. * u.AA, m_UV, u.Jy).value / \
                funcs.convert_mag_units(self.wavs, self.mags, u.Jy).value[np.abs(self.wavs.to(u.AA).value - 1_500.).argmin()]
            self.mags = np.array([norm * mag for mag in self.mags.value]) * self.mags.unit
            
    def renorm_at_wav(self, mag): # this mag can also be a flux, but must have astropy units
        pass
    
    def calc_UV_slope(self, output_errs = False, method = "Calzetti+94"):
        if method == "Calzetti+94":
            # crop to Calzetti+94 filters
            wavs, mags = self.crop_to_Calzetti94_filters()
            # convert self.mags to f_λ if needed
            if mags.unit == u.ABmag:
                mags = funcs.convert_mag_units(wavs, mags, u.erg / (u.s * (u.cm ** 2) * u.AA))
            elif u.get_physical_type(mags.unit) != "power density/spectral flux density wav":
                mags = funcs.convert_mag_units(wavs, mags, u.erg / (u.s * (u.cm ** 2) * u.AA))
        popt, pcov = curve_fit(funcs.beta_slope_power_law_func, wavs.value, mags.value, maxfev = 1_000)
        A, beta = popt[0], popt[1]
        if output_errs:
            A_err = np.sqrt(pcov[0][0])
            beta_err = np.sqrt(pcov[1][1])
            return A, beta, A_err, beta_err
        else:
            return A, beta
        
    def add_emission_lines(self, emission_lines):
        assert(type(emission_lines) in [list, np.array])
        for emission_line in emission_lines:
            # update units
            orig_wav_unit = self.wavs.unit
            orig_mag_unit = self.mags.unit
            self.convert_wav_units(emission_line.line_profile["wavs"].unit, update = True)
            self.convert_mag_units(emission_line.line_profile["flux"].unit, update = True)
            # interpolate emission line to be on the same wavelength grid as the spectrum
            interp_line_profile = interp1d(emission_line.line_profile["wavs"], \
                emission_line.line_profile["flux"].value, bounds_error = False, fill_value = 0.)(self.wavs)
            # correct for line flux difference between interped and non-interped line profile
            # self_copy = deepcopy(self)
            # self_copy.mags += interp_line_profile * self_copy.mags.unit
            # self_copy.calc_line_EW(emission_line.line_name)
            # line_flux = (self_copy.line_fluxes[emission_line.line_name] / u.AA).to(u.erg / (u.s * u.AA * u.cm ** 2), \
            #                 equivalencies = u.spectral_density(line_diagnostics[emission_line.line_name]["line_wav"])) * u.AA
            # add normalized line profile to spectrum
            self.mags += interp_line_profile * self.mags.unit #* emission_line.line_flux / line_flux

class Mock_SED_obs(SED_obs):
    
    def __init__(self, z, wavs, mags, wav_units, mag_units, template_name = None, IGM = IGM_attenuation.IGM(), meta = None):
        self.template_name = template_name
        self.meta = meta
        super().__init__(z, wavs, mags, wav_units, mag_units)
        if IGM != None:
            self.attenuate_IGM(IGM)
    
    @classmethod
    def from_Mock_SED_rest(cls, mock_SED_rest, z, out_wav_units = u.AA, out_mag_units = u.ABmag, IGM = IGM_attenuation.IGM()):
        mags = mock_SED_rest.convert_mag_units(out_mag_units)
        wavs = (mock_SED_rest.wavs * (1 + z)).to(out_wav_units)
        mock_SED_obs_obj = cls(z, wavs.value, mags.value, out_wav_units, out_mag_units, mock_SED_rest.template_name, IGM, meta = mock_SED_rest.meta)
        return mock_SED_obs_obj
    
    @classmethod
    def power_law_from_beta_M_UV(cls, z, beta, M_UV, template_name = None, IGM = IGM_attenuation.IGM()):
        lum_distance = astropy_cosmo.luminosity_distance(z).to(u.pc)
        m_UV = M_UV - 2.5 * np.log10(1 + z) + 5 * np.log10(lum_distance.value / 10)
        mock_SED_rest = Mock_SED_rest.power_law_from_beta_m_UV(beta, m_UV, template_name = template_name)
        obs_SED = cls.from_SED_rest(z, mock_SED_rest)
        obs_SED.attenuate_IGM(IGM)
        return obs_SED
    
    def attenuate_IGM(self, IGM = IGM_attenuation.IGM()):
        if not hasattr(self, "IGM"):
            self.IGM = None
        if isinstance(IGM, IGM_attenuation.IGM):
            if self.IGM == None: # not already been attenuated
                # attenuate SED for IGM absorption
                IGM_transmission = IGM.interp_transmission(self.z, self.wavs / (1 + self.z))
                if self.mags.unit == u.ABmag:
                    self.mags = (self.mags.value -2.5 * np.log10(IGM_transmission)) * u.ABmag
                else:
                    self.mags *= IGM_transmission
                # save IGM object after attenuating
                self.IGM = IGM
            else:
                #print("SED has already been attenuated! Ignoring")
                pass
        else:
            raise(Exception(f"Could not attenuate by a non IGM object = {IGM}"))
            
    def calc_UV_slope(self, output_errs = False, method = "Calzetti+94"):
        # create rest frame mock SED object
        mock_sed_rest = Mock_SED_rest.from_Mock_SED_obs(self)
        # calculate amplitude and beta of power law fit
        A, beta = mock_sed_rest.calc_UV_slope(output_errs = output_errs, method = method)
        return A, beta
    
    def get_colour(self, colour_name):
        if "mock_photometry" in self.__dict__:
            bands = colour_name.split("-")
            assert(len(bands) == 2)
            # requires colour to exist in the mock photometry
            for band in bands:
                if band not in self.mock_photometry.instrument:
                    galfind_logger.critical(f"self.mock_photometry includes the bands = {self.mock_photometry.instrument.band_names}, and {band} is not included!")
                assert(self.mock_photometry[band].unit == u.Jy)
            # calculate colour in mags
            colour = -2.5 * np.log10(self.mock_photometry[bands[0]] / self.mock_photometry[bands[1]])
            # save colour in Mock_SED object
            if "colours" in self.__dict__:
                self.colours = {**self.colours, **{colour_name: colour.value}}
            else:
                self.colours = {colour_name: colour.value}
            return colour.value
        else:
            galfind_logger.critical("self.mock_photometry does not exist! Please first create photometry from SED template!")
    
    def add_DLA(self, DLA_obj):
        self.colours = {} # reset colours
        self.DLA = DLA_obj
        self.wavs = funcs.convert_wav_units(self.wavs, u.AA)
        self.mags = funcs.convert_mag_units(self.wavs, self.mags, u.Jy)
        self.mags *= DLA_obj.transmission(self.wavs / (1 + self.z))

    def add_dust_attenuation(self, dust_attenuation, E_BminusV):
        self.colours = {} # reset colours
        self.dust_attenuation = dust_attenuation
        self.E_BminusV = E_BminusV
        self.wavs = funcs.convert_wav_units(self.wavs, u.AA)
        self.mags = funcs.convert_mag_units(self.wavs, self.mags, u.Jy)
        # attenuate flux in Jy (named self.mags)
        print(dust_attenuation)
        self.mags *= 10 ** (-0.4 * dust_attenuation.attenuate(self.wavs, E_BminusV))
    
    def add_emission_lines(self, line_diagnostics):
        pass


class Mock_SED_template_set(ABC):
    
    def __init__(self, mock_SED_arr):
        self.SED_arr = mock_SED_arr
    
    def __iter__(self):
        self.iter = 0
        return self
    
    def __next__(self):
        if self.iter > len(self) - 1:
            raise StopIteration
        else:
            template = self[self.iter]
            self.iter += 1
            return template
    
    def __getitem__(self, index):
        return self.SED_arr[index]
    
    def __len__(self):
        return len(self.SED_arr)
    
    def create_mock_phot(self, instrument):
        [sed.create_mock_phot(instrument) for sed in self.SED_arr]
        
    # @abstractmethod
    # def calc_UV_slope():
    #     pass
    
class Mock_SED_rest_template_set(Mock_SED_template_set):
    
    def __init__(self, mock_SED_rest_arr):
        super().__init__(mock_SED_rest_arr)
    
    @classmethod
    def load_SED_in_templates():
        pass
    
    @classmethod
    def load_EAZY_in_templates(cls, m_UV, template_set):
        mock_SED_rest_arr = []
        # read in .txt file if it exists
        template_labels = open(f"{config['EAZY']['EAZY_DIR']}/{template_set}.txt", "r")
        for name in template_labels.readlines():
            mock_SED_rest_arr.append(Mock_SED_rest.load_EAZY_in_template(m_UV, template_set, name.replace("\n", "")))
        template_labels.close()
        return cls(mock_SED_rest_arr)
    
    @classmethod
    def load_Yggdrasil_popIII_in_templates(cls, imf, fcov, sfh):
        mock_SED_rest_arr = []
        yggdrasil_dir = f"/Users/user/Documents/PGR/yggdrasil_grids/{imf}_fcov_{str(fcov)}_SFR_{sfh}_Spectra"
        SED_filenames = glob.glob(f"{yggdrasil_dir}/*.ecsv")
        for name in SED_filenames:
            mock_SED_rest_arr.append(Mock_SED_rest.load_Yggdrasil_popIII_in_template(imf, fcov, sfh, name.split("/")[-1]))
        return cls(mock_SED_rest_arr)
    
    @classmethod
    def load_bpass_in_templates(cls, metallicity = "z020", imf = "imf135_300", model_type = "bin", alpha_enhancement = "a+06", log_ages = np.linspace(6., 9., int(np.round(3 / 0.1)) + 1), bpass_version = 2.3, m_UV_norm = 26.):
        meta = {"metallicity": metallicity, "imf": imf, "model_type": model_type, "alpha_enhancement": alpha_enhancement, "bpass_version": bpass_version}
        bpass_Lsun = 3.848e26 * u.J / u.s
        bpass_dir = f"/raid/scratch/data/BPASS/BPASS_v{str(bpass_version)}_release/bpass_v{str(bpass_version)}.{alpha_enhancement}"
        bpass_name = f"spectra-{model_type}-{imf}.{alpha_enhancement}.{metallicity}.dat"
        SED_file = f"{bpass_dir}/{bpass_name}"
        mock_SED_rest_arr = []
        spectra = ascii.read(SED_file)
        rest_wavs = np.array(spectra["col1"]) * u.AA
        for i in range(2, 53):
            age = 10 ** (0.1 * (i - 2)) * u.Myr
            print(log_ages)
            load_spectrum = False
            if log_ages == "all":
                load_spectrum = True
            elif np.isclose(log_ages, np.log10(age.value) + 6.).any():
                load_spectrum = True
            else:
                load_spectrum = False
            if load_spectrum:
                spectrum = (np.array(spectra[f"col{i}"])) * bpass_Lsun / u.AA
                spectrum_Jy = funcs.luminosity_to_flux(spectrum, rest_wavs, out_units = u.Jy)
                mock_sed_rest = Mock_SED_rest(rest_wavs.value, spectrum_Jy.value, u.AA, u.Jy, \
                    template_name = bpass_name.replace(".dat", f"{age.value:.1f}Myr") , meta = {**meta, **{"age": age, "log_age_yr": np.log10(age.value) + 6.}})
                mock_sed_rest.normalize_to_m_UV(26.)
                mock_SED_rest_arr.append(mock_sed_rest)
        return cls(mock_SED_rest_arr)

    def calc_mock_beta_phot(self, m_UV, template_set, instrument, depths):
        pass

class Mock_SED_obs_template_set(Mock_SED_template_set):
    
    def __init__(self, mock_SED_obs_arr):
        super().__init__(mock_SED_obs_arr)

    @classmethod
    def from_Mock_SED_rest_template_set(cls, Mock_SED_rest_template_set, z_arr):
        if type(z_arr) in [float, int]:
            z_arr = np.full(len(Mock_SED_rest_template_set), z_arr)
        return cls([Mock_SED_obs.from_Mock_SED_rest(mock_sed_rest, z) for mock_sed_rest, z in zip(Mock_SED_rest_template_set, z_arr)])
    
    def get_colours(self, colour_names):
        return [sed.get_colour(colour) for sed in self.SED_arr for colour in colour_names]
    
    def plot_colour_colour_tracks(self, ax, colour_x_name, colour_y_name, shown_log_ages = [7., 8., 9.], \
            line_kwargs = {}, save = False, show = False, save_dir = "/nvme/scratch/work/austind/EPOCHS_I_plots"):
        # assumes the SEDs are already sorted by age, starting with the youngest
        colour_x = np.array([sed.colours[colour_x_name] for sed in self.SED_arr])
        colour_y = np.array([sed.colours[colour_y_name] for sed in self.SED_arr])
        ax.plot(colour_x, colour_y, **line_kwargs)

        log_ages = [sed.meta["log_age_yr"] for sed in self.SED_arr]
        plot_indices = np.array([True if np.isclose(shown_log_ages, log_age).any() and i != 0 else False for i, log_age in enumerate(log_ages)])
        # plot youngest age SED as star
        ax.scatter(colour_x[0], colour_y[0], color = line_kwargs["c"], marker = "*", zorder = 999., s = 50., path_effects = line_kwargs["path_effects"])
        ax.scatter(colour_x[plot_indices], colour_y[plot_indices], color = line_kwargs["c"], marker = "s", s = 10., zorder = 999., path_effects = line_kwargs["path_effects"])
        if save:
            os.makedirs(save_dir, exist_ok = True)
            plt.savefig(f"{save_dir}/{colour_x_name}_vs_{colour_y_name}.png", dvi = 400)
        if show:
            plt.show()

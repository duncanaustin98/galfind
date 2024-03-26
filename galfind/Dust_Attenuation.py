# Dust_Attenuation.py

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.optimize import curve_fit
from abc import ABC, abstractmethod

class Dust_Attenuation(ABC):

    def __init__(self, R_V, description = ""):
        """ Abstract class to contain the dust attenuation curve and related properties

        Args:
            R_V (float): Fundamental property of the attenuation curve given by R_V = A_V / E(B-V)
            description (str, optional): Description of the class, usually including the author and year initially published. Defaults to "".
        """
        self.R_V = R_V
        self.description = description

    def __str__(self):
        line_sep = "*" * 40 + "\n"
        band_sep = "-" * 10 + "\n"
        output_str = line_sep
        output_str += f"DUST ATTENUATION LAW: {self.__class__.__name__}\n"
        output_str += band_sep
        output_str += f"Description: {self.description}\n"
        output_str += f"R_V: {self.R_V}\n"
        output_str += f"UV-optical slope: {self.UV_optical_slope:.3f}\n"
        output_str += f"UV slope: {self.UV_slope:.3f}\n"
        output_str += f"Optical slope: {self.optical_slope:.3f}\n"
        output_str += f"Near-IR slope: {self.near_IR_slope:.3f}\n"
        output_str += f"UV-bump strength: {self.UV_bump_strength:.3f}\n"
        output_str += line_sep
        return output_str

    @abstractmethod
    def k_lambda(self, wavs):
        """ Dust attenuation curve as a function of wavelength, λ 

        Args:
            wavs ([int, float, list, np.array]): Wavelength to calculate attenuation curve at (should have astropy units with dimension L)
        """
        pass

    def attenuation(self, wavs, E_BminusV):
        """ Function to calculate the amount of dust attenuation at wavelength(s) λ, A(λ) = m(λ) - m(λ,0)

        Args:
            wavs ([int, float, list, np.array]): Wavelength(s) to calculate attenuation at (should have astropy units with dimension L)    
            E_BminusV (int, float): Amount of dust attenuation A(B) - A(V) = m(B) - m(B,0) - m(V) + m(V,0)

        Returns:
            A_lambda ([int, float, list, np.array]): Dust attenuation at wavelength(s) λ, A(λ)
        """
        A_lambda = self.k_lambda(wavs) * E_BminusV
        return A_lambda

    @property
    def UV_optical_slope(self):
        """ Property giving the UV optical slope of the dust attenuation curve, defined as S = A(1500 Angstrom) - A(V=5500 Angstrom)

        Returns:
            slope (float): UV optical slope, S, of the dust attenuation curve
        """
        slope = self.k_lambda(1_500 * u.AA) / self.k_lambda(5_500 * u.AA)
        return slope[0]

    @property
    def UV_slope(self):
        """ Property giving the UV slope (different to β) of the dust attenuation curve, defined as A(1500 Angstrom) - A(3000 Angstrom)

        Returns:
            slope (float): UV slope of the dust attenuation curve
        """
        slope = self.k_lambda(1_500 * u.AA) / self.k_lambda(3_000 * u.AA)
        return slope[0]
    
    @property
    def optical_slope(self):
        """ Property giving the optical slope of the dust attenuation curve, defined as A(B=4400 Angstrom) - A(V=5500 Angstrom)

        Returns:
            slope (float): Optical slope of the dust attenuation curve
        """
        slope = self.k_lambda(4_400 * u.AA) / self.k_lambda(5_500 * u.AA)
        return slope[0]

    @property
    def near_IR_slope(self, NIR_range = [0.9, 5] * u.um, resolution = 0.1 * u.um):
        """ Property giving the near-IR slope of the dust attenuation curve, β_NIR
        The equation concerning this slope is as follows:
        A(λ)/A(V) = (λ / 5500 Angstrom)^-β_NIR, or equivalently
        k(λ) = R(V) * (5500 Angstrom / λ)^β_NIR

        Args:
            NIR_range ([np.array, list], optional): Near-infrared wavelength range used to calculate near-IR slope. Defaults to [0.9, 5] * u.um. Astropy unit dimensions L
            resolution (float): Wavelength resolution used to create the array of wavelengths. Defaults to 0.1 * u.um. Astropy unit dimensions L

        Returns:
            slope (float): Near-IR slope of the dust attenuation curve
        """
        wavs = np.linspace(NIR_range[0], NIR_range[1], int(np.round(((NIR_range[1] - NIR_range[0]) / resolution).to(u.dimensionless_unscaled).value)))
        popt, pcov = curve_fit(lambda wav, beta: self.R_V * (5_500 / wav) ** beta, wavs.to(u.AA), self.k_lambda(wavs))
        slope = popt[0]
        return slope

    @property
    def UV_bump_strength(self):
        """ Function to calculate the UV bump strength at 2175 Angstrom, B.
        Defined as B = A_bump / A_2175_tot, where A_bump is the additional attenuation due to the dust bump and A_2175_tot is the total attenuation at 2175 Angstrom.
        A_2175_tot = A_bump + A_2175_0, where A_2175_0 is the baseline attenuation
        A_2175_0 = 0.33 * A_1500 + 0.67 * A_3000, defined empirically from simulated curves in Narayanan+18a and given in Salim+20

        Returns:
            bump (float): UV bump strength from carbonaceous dust at 2175 Angstrom
        """
        bump = 1. - (0.33 * self.k_lambda(1_500 * u.AA) + 0.67 * self.k_lambda(3_000 * u.AA)) / self.k_lambda(2_175 * u.AA)
        return bump[0]
    
    def plot_attenuation_law(self, ax, wavs):
        ax.plot(wavs, self.k_lambda(wavs))
        ax.set_xlabel(r"$\lambda_{\mathrm{rest}} / \AA $")
        ax.set_ylabel(r"$k(\lambda)$")

class Calzetti00(Dust_Attenuation):

    def __init__(self):
        R_V = 4.05
        description = "Calzetti+00 attenuation curve with fixed R_V = 4.05"
        super().__init__(R_V, description)

    def k_lambda(self, wavs):
        """ Calzetti+00 attenuation curve

        Args:
            wavs ([int, float, list, np.array]): Wavelength to calculate attenuation curve at (should have astropy units with dimension L)

        Returns:
            k ([int, float, list, np.array]): Un-normalized Calzetti+00 attenuation at specified input wavelengths
        """
        # convert wavelength(s) to microns
        if type(wavs.value) in [int, float, np.float64]:
            k = np.array([0.])
            wavs = np.array([wavs.to(u.um).value])
        else:
            k = np.zeros(len(wavs.value))
            wavs = np.array([wav.to(u.um).value for wav in wavs])
        
        # extrapolate bluewards of 0.12 microns
        # mask_1 = 
        # k[mask_1] = 
        mask_2 = ((wavs > 0.12) & (wavs <= 0.63))
        k[mask_2] = 2.659 * (-2.156 + (1.509 / (wavs[mask_2])) - (0.198 / (wavs[mask_2] ** 2)) + (0.011 / (wavs[mask_2] ** 3))) + self.R_V
        mask_3 = ((wavs > 0.63) & (wavs <= 2.2))
        k[mask_3] = 2.659 * (-1.857 + 1.040 / wavs[mask_3]) + self.R_V
        # extrapolate redwards of 2.2 microns
        # mask_4 = 
        # k[mask_4] = 
        return k



    
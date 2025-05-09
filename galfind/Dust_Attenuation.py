from __future__ import annotations

from abc import ABC, abstractmethod
import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d
import h5py
from scipy.optimize import curve_fit
from typing import TYPE_CHECKING, Dict, Any, List, Union, Optional, NoReturn
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import useful_funcs_austind as funcs
from . import config


class Dust_Law(ABC):
    def __init__(
        self: Self,
        label: Optional[str] = None
    ) -> NoReturn:
        """Abstract class to contain the dust attenuation curve and related properties

        Args:
            R_V (float): The ratio of total to selective extinction, R(V) = A(V) / E(B-V)
            label (str, optional): Class label, usually including the author and year initially published.
        """
        self.label = label

    def __str__(self):
        output_str = funcs.line_sep
        output_str += f"DUST ATTENUATION LAW: {self.__class__.__name__}\n"
        output_str += funcs.band_sep
        output_str += f"Label: {self.label}\n"
        output_str += f"R_V: {self.R_V}\n"
        output_str += f"UV-optical slope: {self.UV_optical_slope:.3f}\n"
        output_str += f"UV slope: {self.UV_slope:.3f}\n"
        output_str += f"Optical slope: {self.optical_slope:.3f}\n"
        output_str += f"Near-IR slope: {self.near_IR_slope:.3f}\n"
        output_str += f"UV-bump strength: {self.UV_bump_strength:.3f}\n"
        output_str += funcs.line_sep
        return output_str

    @abstractmethod
    def k_lambda(self, wavs):
        """Dust attenuation curve as a function of wavelength, λ

        Args:
            wavs ([int, float, list, np.array]): Wavelength to calculate attenuation curve at (should have astropy units with dimension L)
        """
        pass

    def attenuate(self, wavs, E_BminusV):
        """Function to calculate the amount of dust attenuation at wavelength(s) λ, A(λ) = m(λ) - m(λ,0)

        Args:
            wavs ([int, float, list, np.array]): Wavelength(s) to calculate attenuation at (should have astropy units with dimension L)
            E_BminusV (int, float): Amount of dust attenuation A(B) - A(V) = m(B) - m(B,0) - m(V) + m(V,0)

        Returns:
            A_lambda ([int, float, list, np.array]): Dust attenuation at wavelength(s) λ, A(λ)
        """
        A_lambda = self.k_lambda(wavs) * E_BminusV
        return A_lambda

    @property
    def R_V(self):
        """Property giving the ratio of total to selective extinction, R(V) = A(V) / E(B-V)

        Returns:
            R_V (float): Ratio of total to selective extinction
        """
        return self.k_lambda(5_500 * u.AA)

    @property
    def UV_optical_slope(self):
        """Property giving the UV optical slope of the dust attenuation curve, defined as S = A(1500 Angstrom) - A(V=5500 Angstrom)

        Returns:
            slope (float): UV optical slope, S, of the dust attenuation curve
        """
        slope = self.k_lambda(1_500 * u.AA) / self.k_lambda(5_500 * u.AA)
        return slope[0]

    @property
    def UV_slope(self):
        """Property giving the UV slope (different to β) of the dust attenuation curve, defined as A(1500 Angstrom) - A(3000 Angstrom)

        Returns:
            slope (float): UV slope of the dust attenuation curve
        """
        slope = self.k_lambda(1_500 * u.AA) / self.k_lambda(3_000 * u.AA)
        return slope[0]

    @property
    def optical_slope(self):
        """Property giving the optical slope of the dust attenuation curve, defined as A(B=4400 Angstrom) - A(V=5500 Angstrom)

        Returns:
            slope (float): Optical slope of the dust attenuation curve
        """
        slope = self.k_lambda(4_400 * u.AA) / self.k_lambda(5_500 * u.AA)
        return slope[0]

    @property
    def near_IR_slope(
        self, NIR_range=[0.9, 5.0] * u.um, resolution=0.1 * u.um
    ):
        """Property giving the near-IR slope of the dust attenuation curve, β_NIR
        The equation concerning this slope is as follows:
        A(λ)/A(V) = (λ / 5500 Angstrom)^-β_NIR, or equivalently
        k(λ) = R(V) * (5500 Angstrom / λ)^β_NIR

        Args:
            NIR_range ([np.array, list], optional): Near-infrared wavelength range used to calculate near-IR slope. Defaults to [0.9, 5] * u.um. Astropy unit dimensions L
            resolution (float): Wavelength resolution used to create the array of wavelengths. Defaults to 0.1 * u.um. Astropy unit dimensions L

        Returns:
            slope (float): Near-IR slope of the dust attenuation curve
        """
        wavs = np.linspace(
            NIR_range[0],
            NIR_range[1],
            int(
                np.round(
                    ((NIR_range[1] - NIR_range[0]) / resolution)
                    .to(u.dimensionless_unscaled)
                    .value
                )
            ),
        )
        popt, pcov = curve_fit(
            lambda wav, beta: self.R_V * (5_500 / wav) ** beta,
            wavs.to(u.AA),
            self.k_lambda(wavs),
        )
        slope = popt[0]
        return slope

    @property
    def UV_bump_strength(self):
        """Function to calculate the UV bump strength at 2175 Angstrom, B.
        Defined as B = A_bump / A_2175_tot, where A_bump is the additional attenuation due to the dust bump and A_2175_tot is the total attenuation at 2175 Angstrom.
        A_2175_tot = A_bump + A_2175_0, where A_2175_0 is the baseline attenuation
        A_2175_0 = 0.33 * A_1500 + 0.67 * A_3000, defined empirically from simulated curves in Narayanan+18a and given in Salim+20

        Returns:
            bump (float): UV bump strength from carbonaceous dust at 2175 Angstrom
        """
        bump = 1.0 - (
            0.33 * self.k_lambda(1_500 * u.AA)
            + 0.67 * self.k_lambda(3_000 * u.AA)
        ) / self.k_lambda(2_175 * u.AA)
        return bump[0]
    
    @property
    def n(self):
        """Property giving the power law index of the dust attenuation curve, n.
        Defined as n = -d(log(k(λ))) / d(log(λ)), where k(λ) is the dust attenuation curve

        Returns:
            n (float): Power law index of the dust attenuation curve
        """
        if not hasattr(self, "_n"):
            wavs = np.linspace(1_500., 6_500., 1000) * u.AA
            k_lambda = self.k_lambda(wavs)
            #breakpoint()
            n = -curve_fit(funcs.power_law_func, wavs.value / 5_500., k_lambda)[0][1]
        else:
            n = self._n
        return n

    def plot(self, ax, wavs, label = True, **kwargs):
        ax.plot(wavs, self.k_lambda(wavs), label = self.label if label else None, **kwargs)
        ax.set_xlabel(r"$\lambda_{\mathrm{rest}} / \AA $")
        ax.set_ylabel(r"$k(\lambda)$")


class Calzetti00(Dust_Law):
    def __init__(self):
        #self._n = 0.7
        label = "Calzetti+00"
        super().__init__(label)

    def k_lambda(self, wavs):
        """Calzetti+00 attenuation curve

        Args:
            wavs ([int, float, list, np.array]): Wavelength to calculate attenuation curve at (should have astropy units with dimension L)

        Returns:
            k ([int, float, list, np.array]): Un-normalized Calzetti+00 attenuation at specified input wavelengths
        """
        # convert wavelength(s) to microns
        if type(wavs.value) in [int, float, np.float64]:
            k = np.array([0.0])
            wavs = np.array([wavs.to(u.um).value])
        else:
            k = np.zeros(len(wavs.value))
            wavs = np.array([wav.to(u.um).value for wav in wavs])

        mask_1 = wavs <= 0.12
        mask_2 = (wavs > 0.12) & (wavs <= 0.63)
        mask_3 = (wavs > 0.63) & (wavs <= 2.2)
        mask_4 = wavs > 2.2

        k[mask_2] = (
            2.659
            * (
                -2.156
                + (1.509 / (wavs[mask_2]))
                - (0.198 / (wavs[mask_2] ** 2))
                + (0.011 / (wavs[mask_2] ** 3))
            )
            + self.R_V
        )
        k[mask_3] = 2.659 * (-1.857 + 1.040 / wavs[mask_3]) + self.R_V
        if len(wavs[mask_2]) > 1 and len(wavs[mask_1]) > 0:
            # extrapolate bluewards of 0.12 microns
            k[mask_1] = interp1d(
                wavs[mask_2], k[mask_2], fill_value="extrapolate"
            )(wavs[mask_1])
        if len(wavs[mask_3]) > 1 and len(wavs[mask_4]) > 0:
            # extrapolate redwards of 2.2 microns
            k[mask_4] = interp1d(
                wavs[mask_3], k[mask_3], fill_value="extrapolate"
            )(wavs[mask_4])
        return k
    
    @property
    def R_V(self):
        return 4.05

class Power_Law_Dust(Dust_Law):

    def __init__(self, R_V: float, n: float = 1.0):
        """Power law dust attenuation curve

        Args:
            n (float, optional): Power law index. Defaults to 1.0.
        """
        self._R_V = R_V
        self._n = n
        label = f"PL (R_V={R_V},n={self._n})"
        super().__init__(label)

    def k_lambda(self, wavs):
        """Power law dust attenuation curve

        Args:
            wavs ([int, float, list, np.array]): Wavelength to calculate attenuation curve at (should have astropy units with dimension L)

        Returns:
            k ([int, float, list, np.array]): Power law dust attenuation at specified input wavelengths
        """
        return self.R_V * (wavs / (5_500 * u.AA)).to(u.dimensionless_unscaled) ** -self.n

    @property
    def R_V(self):
        return self._R_V

class Modified_Calzetti00(Dust_Law):

    def __init__(self, delta: float):
        """Modified Calzetti+00 attenuation curve from Salim+18

        Args:
            delta (float): The amount of modification to the slope of the Calzetti+00 attenuation curve. A value of 0.0 is the original Calzetti+00 curve
        """
        self.delta = delta
        self.calzetti = Calzetti00()
        label = f"Modifed Calzetti+00 (delta={self.delta})"
        super().__init__(label)

    @property
    def R_V(self):
        """Property giving the ratio of total to selective extinction, R(V) = A(V) / E(B-V)

        Returns:
            R_V (float): Ratio of total to selective extinction
        """
        return self.calzetti.R_V / ((self.calzetti.R_V + 1.0) * (4_400.0 / 5_500.0) ** self.delta - self.calzetti.R_V)

    def k_lambda(self, wavs):
        """Modified Calzetti+00 attenuation curve

        Args:
            wavs ([int, float, list, np.array]): Wavelength to calculate attenuation curve at (should have astropy units with dimension L)

        Returns:
            k ([int, float, list, np.array]): Modified Calzetti+00 attenuation at specified input wavelengths
        """
        return self.calzetti.k_lambda(wavs) * self.R_V_mod / self.calzetti.R_V * (wavs / (5_500 * u.AA)).to(u.dimensionless_unscaled) ** self.delta


class Reddy15(Dust_Law):

    def __init__(self):
        """
        Reddy+15 attenuation curve with R_V = 2.505
        """
        label = "Reddy+15"
        super().__init__(label)

    @property
    def R_V(self):
        return 2.505

    def k_lambda(self, wavs):
        """Reddy+15 attenuation curve

        Args:
            wavs ([int, float, list, np.array]): Wavelength to calculate attenuation curve at (should have astropy units with dimension L)

        Returns:
            k ([int, float, list, np.array]): Reddy+15 attenuation at specified input wavelengths
        """

        # convert wavelength(s) to microns
        if type(wavs.value) in [int, float, np.float64]:
            k = np.array([0.0])
            wavs = np.array([wavs.to(u.um).value])
        else:
            k = np.zeros(len(wavs.value))
            wavs = np.array([wav.to(u.um).value for wav in wavs])

        mask_1 = wavs <= 0.15
        mask_2 = (wavs > 0.15) & (wavs <= 0.6)
        mask_3 = (wavs > 0.6) & (wavs <= 2.2)
        mask_4 = wavs > 2.2

        k[mask_2] = (
            - 5.726
            + 4.004 / (wavs[mask_2])
            - 0.525 / (wavs[mask_2] ** 2)
            + 0.029 / (wavs[mask_2] ** 3)
            + self.R_V
        )
        k[mask_3] = (
            - 2.672
            - 0.010 / (wavs[mask_3])
            + 1.532 / (wavs[mask_3] ** 2)
            - 0.412 / (wavs[mask_3] ** 3)
            + self.R_V
        )
        if len(wavs[mask_2]) > 1 and len(wavs[mask_1]) > 0:
            # extrapolate bluewards of 0.12 microns
            k[mask_1] = interp1d(
                wavs[mask_2], k[mask_2], fill_value="extrapolate"
            )(wavs[mask_1])
        if len(wavs[mask_3]) > 1 and len(wavs[mask_4]) > 0:
            # extrapolate redwards of 2.2 microns
            k[mask_4] = interp1d(
                wavs[mask_3], k[mask_3], fill_value="extrapolate"
            )(wavs[mask_4])
        return k
    

class Salim18(Dust_Law):

    def __init__(self):
        """
            Salim+18 attenuation curve with n = 1.15, R_V = 3.15
        """
        self._n = 1.15
        label = "Salim+18"
        super().__init__(label)

    @property
    def R_V(self):
        return 3.15
    
    def k_lambda(self, wavs):
        """Salim+18 attenuation curve

            Args:
                wavs ([int, float, list, np.array]): Wavelength to calculate attenuation curve at (should have astropy units with dimension L)
            
            Returns:
                k ([int, float, list, np.array]): Salim+18 attenuation at specified input wavelengths
        """
        
                # convert wavelength(s) to microns
        if type(wavs.value) in [int, float, np.float64]:
            k = np.array([0.0])
            wavs = np.array([wavs.to(u.um).value])
        else:
            k = np.zeros(len(wavs.value))
            wavs = np.array([wav.to(u.um).value for wav in wavs])

        mask_1 = wavs <= 0.09
        mask_2 = (wavs > 0.09) & (wavs <= 2.2)
        mask_3 = (wavs > 2.2) & (wavs <= 2.28)
        mask_4 = wavs > 2.28

        k[mask_2] = (
            - 4.30
            + 2.71 / (wavs[mask_2])
            - 0.191 / (wavs[mask_2] ** 2)
            + 0.0121 / (wavs[mask_2] ** 3)
            + self.drude(wavs[mask_2], 1.57)
            + self.R_V
        )
        if len(wavs[mask_2]) > 1 and len(wavs[mask_1]) > 0:
            # extrapolate bluewards of 0.12 microns
            k[mask_1] = interp1d(
                wavs[mask_2], k[mask_2], fill_value="extrapolate"
            )(wavs[mask_1])
        if len(wavs[mask_2]) > 1 and len(wavs[mask_3]) > 0:
            # extrapolate redwards of 2.2 microns
            k[mask_3] = interp1d(
                wavs[mask_2], k[mask_2], fill_value="extrapolate"
            )(wavs[mask_3])

        k[mask_4] = 0.0

        return k
    
    def drude(self, wavs, bump_strength):
        """
        Function to calculate the Drude profile for the UV bump at 2175 Angstrom
        Args:
            wavs ([int, float, list, np.array]): Wavelength to calculate attenuation curve at (should have astropy units with dimension L)
        """
        # wavelength in microns
        numerator = bump_strength * (wavs * 0.035) ** 2
        denominator = ((wavs ** 2) - (0.2175 ** 2)) ** 2 + (wavs * 0.035) ** 2
        return numerator / denominator


class SMC(Dust_Law):
    # Gordon+03
    
    def __init__(self):
        """
            SMC attenuation curve from Gordon+03 - specifically for the SMC bar
        """
        label = "SMC (Gordon+03)"
        super().__init__(label)

    @property
    def R_V(self):
        return 2.74
    
    def k_lambda(self, wavs):
        """
        SMC attenuation curve
        Args:
            wavs ([int, float, list, np.array]): Wavelength to calculate attenuation curve at (should have astropy units with dimension L)
        Returns:
            k ([int, float, list, np.array]): SMC attenuation at specified input wavelengths
        """
        # load data from the .h5 file
        with h5py.File(f"{config['DEFAULT']['GALFIND_DIR']}/../data/SMC_Gordon+03.h5", "r") as f:
            lam = f["lam"][:]
            Alam_AV = f["Alam_AV"][:]
            #Alam_AV_err = f["Alam_AV_err"][:]
            f.close()
        k_func = interp1d(
            lam,
            Alam_AV * self.R_V,
            kind = "cubic",
            fill_value = "extrapolate",
        )
        return k_func(wavs.to(u.um).value) * u.dimensionless_unscaled


class AUV_from_beta(ABC):
    def __init__(self, beta_int, slope, dust_law, ref_wav):
        self.beta_int = beta_int
        self.slope = slope
        self.dust_law = dust_law
        self.ref_wav = ref_wav

    def __call__(self, beta):
        # beta = beta_int + slope * A_UV
        return ((beta - self.beta_int) / self.slope).value * u.ABmag

    def change_ref_wav(self, ref_wav):
        if not ref_wav == self.ref_wav:
            pass
        raise NotImplementedError


class M99(AUV_from_beta):
    def __init__(self):
        super().__init__(-4.43 / 1.99, 1.0 / 1.99, Calzetti00(), 1_600.0 * u.AA)


class Reddy15_conv(AUV_from_beta):
    def __init__(self):
        super().__init__(
            -4.48 / 1.84, 1.0 / 1.84, Reddy15(), 1_600.0 * u.AA
        )


class Reddy18(AUV_from_beta):
    def __init__(
        self: Self, 
        dust_law: Type[Dust_Law] = Reddy15(), 
        BPASS_age: u.Quantity = 100 * u.Myr
    ) -> NoReturn:
        assert dust_law.__class__.__name__ in ["SMC", "Calzetti00", "Reddy15"]
        assert BPASS_age in [100 * u.Myr, 300 * u.Myr]
        beta_int = {100 * u.Myr: -2.520, 300 * u.Myr: -2.616}
        slope = {"Reddy15": 0.55}
        super().__init__(
            beta_int[BPASS_age],
            slope[dust_law.__class__.__name__],
            dust_law,
            1_600.0 * u.AA,
        )

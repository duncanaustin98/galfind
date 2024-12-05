from __future__ import annotations

from abc import ABC, abstractmethod
import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from typing import TYPE_CHECKING, Dict, Any, List, Union, Optional, NoReturn
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import useful_funcs_austind as funcs


class Dust_Law(ABC):
    def __init__(
        self: Self, 
        R_V: float, 
        description: Optional[str] = None
    ) -> NoReturn:
        """Abstract class to contain the dust attenuation curve and related properties

        Args:
            R_V (float): Fundamental property of the attenuation curve given by R_V = A_V / E(B-V)
            description (str, optional): Description of the class, usually including the author and year initially published. Defaults to "".
        """
        self.R_V = R_V
        self.description = description

    def __str__(self):
        output_str = funcs.line_sep
        output_str += f"DUST ATTENUATION LAW: {self.__class__.__name__}\n"
        output_str += funcs.band_sep
        output_str += f"Description: {self.description}\n"
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

    def plot_attenuation_law(self, ax, wavs):
        ax.plot(wavs, self.k_lambda(wavs))
        ax.set_xlabel(r"$\lambda_{\mathrm{rest}} / \AA $")
        ax.set_ylabel(r"$k(\lambda)$")


class C00(Dust_Law):
    def __init__(self):
        R_V = 4.05
        description = "Calzetti+00 attenuation curve with fixed R_V = 4.05"
        super().__init__(R_V, description)

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


class Reddy15_dust_law(Dust_Law):
    def __init__(self):
        pass

    def k_lambda(self):
        raise NotImplementedError


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
        super().__init__(-4.43 / 1.99, 1.0 / 1.99, C00(), 1_600.0 * u.AA)


class Reddy15(AUV_from_beta):
    def __init__(self):
        super().__init__(
            -4.48 / 1.84, 1.0 / 1.84, Reddy15_dust_law(), 1_600.0 * u.AA
        )


class Reddy18(AUV_from_beta):
    def __init__(
        self: Self, 
        dust_law: Type[Dust_Law] = Reddy15_dust_law(), 
        BPASS_age: u.Quantity = 100 * u.Myr
    ) -> NoReturn:
        assert dust_law.__class__.__name__ in ["SMC", "C00", "Reddy15"]
        assert BPASS_age in [100 * u.Myr, 300 * u.Myr]
        beta_int = {100 * u.Myr: -2.520, 300 * u.Myr: -2.616}
        slope = {"Reddy15": 0.55}
        super().__init__(
            beta_int[BPASS_age],
            slope[dust_law.__class__.__name__],
            dust_law,
            1_600.0 * u.AA,
        )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:04:24 2023

@author: austind
"""

from __future__ import annotations

# Photometry_rest.py
import inspect
from copy import deepcopy
import time
import astropy.units as u
import numpy as np
from scipy.optimize import curve_fit
from astropy.utils.masked import Masked
from tqdm import tqdm
from typing import TYPE_CHECKING, List, Union, Dict, Optional, Tuple
if TYPE_CHECKING:
    from . import Multiple_Filter, PDF, Filter
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import PDF, PDF_nD, Photometry, galfind_logger
from . import useful_funcs_austind as funcs
from .decorators import ignore_warnings
from .Dust_Attenuation import AUV_from_beta
from .Emission_lines import line_diagnostics, strong_optical_lines


class Photometry_rest(Photometry):
    def __init__(
        self: Self,
        filterset: Multiple_Filter,
        flux: u.Quantity,
        flux_errs: u.Quantity,
        depths: u.Quantity,
        z: float,
        properties: Optional[Dict[str, Union[u.Mangitude, u.Quantity, u.Dex]]] = None,
        property_errs: Optional[Dict[str, Tuple[Union[u.Mangitude, u.Quantity, u.Dex], Union[u.Mangitude, u.Quantity, u.Dex]]]] = None,
        property_PDFs: Optional[Dict[str, Type[PDF]]] = None,
        property_kwargs: Optional[Dict[str, Dict[str, Union[str, int, float]]]] = None
    ):
        self.z = z
        if properties is None:
            properties = {}
        if property_errs is None:
            property_errs = {}
        if property_PDFs is None:
            property_PDFs = {}
        if property_kwargs is None:
            property_kwargs = {}
        self.properties = properties
        self.property_errs = property_errs
        self.property_PDFs = property_PDFs
        self.property_kwargs = property_kwargs
        # unmask if given as mask
        if isinstance(flux, Masked):
            flux = flux.unmasked
        if isinstance(flux_errs, Masked):
            flux_errs = flux_errs.unmasked
        super().__init__(filterset, flux, flux_errs, depths)

    # these class methods need updating!
    @classmethod
    def from_fits_cat(cls, fits_cat_row, filterset, cat_creator, code):
        phot = Photometry.from_fits_cat(fits_cat_row, filterset, cat_creator)
        # TODO: mask the photometry object
        return cls.from_phot(
            phot, np.float(fits_cat_row[code.galaxy_properties["z"]])
        )

    @classmethod
    def from_phot(cls, phot, z):
        return cls(phot.filterset, phot.flux, phot.flux_errs, phot.depths, z)

    @classmethod
    def from_phot_obs(cls, phot):
        return cls(
            phot.filterset,
            phot.flux,
            phot.flux_errs,
            phot.depths,
            phot.z,
        )

    def __str__(self: Self) -> str:
        output_str = funcs.line_sep
        output_str += f"PHOTOMETRY_REST: z = {self.z}\n"
        output_str += funcs.band_sep
        # don't print the photometry here, only the derived properties
        #if print_PDFs:
        #for PDF_obj in self.property_PDFs.values():
        #    output_str += str(PDF_obj)
        output_str += funcs.line_sep
        return output_str

    def __len__(self):
        return len(self.flux)

    # def __getattr__(
    #     self,
    #     property_name: str,
    #     origin: str = "phot_rest",
    #     property_type: Union[None, str] = None,
    # ) -> Union[None, bool, u.Quantity, u.Magnitude, u.Dex]:
    #     if origin == "phot_rest":
    #         if type(property_type) == type(None):
    #             return super().__getattr__(property_name, "phot")
    #         assert property_type in [
    #             "val",
    #             "errs",
    #             "l1",
    #             "u1",
    #             "pdf",
    #             "recently_updated",
    #         ], galfind_logger.critical(
    #             f"{property_type=} not in ['val', 'errs', 'l1', 'u1', 'pdf', 'recently_updated']!"
    #         )
    #         # boolean output to say whether property has been recently updated
    #         if property_type == "recently_updated":
    #             return (
    #                 True if property_name in self.recently_updated else False
    #             )
    #         else:
    #             # extract relevant property if name in dict.keys()
    #             if property_type == "val":
    #                 access_dict = self.properties
    #             elif property_type in ["errs", "l1", "u1"]:
    #                 access_dict = self.property_errs
    #             else:
    #                 access_dict = self.property_PDFs
    #             # return None if relevant property is not available
    #             if property_name not in access_dict.keys():
    #                 err_message = f"{property_name} {property_type} not available in Photometry_rest object!"
    #                 galfind_logger.warning(err_message)
    #                 raise AttributeError(err_message)  # may be required here
    #             else:
    #                 if property_type == "l1":
    #                     return access_dict[property_name][0]
    #                 elif property_type == "u1":
    #                     return access_dict[property_name][1]
    #                 else:
    #                     return access_dict[property_name]
    #     else:
    #         galfind_logger.critical(
    #             f"Photometry_rest.__getattr__ currently has no implementation of {origin=} != 'phot_rest'"
    #         )
    #         raise NotImplementedError

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, deepcopy(value, memo))
        return result

    # @property
    # def first_Lya_detect_band(
    #     self: Self,
    #     Lya_wav: u.Quantity = line_diagnostics["Lya"]["line_wav"]
    # ):
    #     try:
    #         return self._first_Lya_detect_band
    #     except AttributeError:
    #         first_band = None
    #         for band in self.filterset:
    #             lower_wav = band.WavelengthLower50
    #             if lower_wav > Lya_wav * (1 + self.z):
    #                 first_band = band.band_name
    #                 break
    #         self._first_Lya_detect_band = first_band
    #         return self._first_Lya_detect_band

    # @property
    # def first_Lya_non_detect_band(
    #     self, Lya_wav=line_diagnostics["Lya"]["line_wav"]
    # ):
    #     try:
    #         return self._first_Lya_non_detect_band
    #     except AttributeError:
            
    #         first_band = None
    #         # bands already ordered from blue -> red
    #         for band in self.filterset:
    #             upper_wav = band.WavelengthUpper50
    #             if upper_wav < Lya_wav * (1 + self.z):
    #                 first_band = band.band_name
    #                 break
    #         self._first_Lya_non_detect_band = first_band
    #     return self._first_Lya_non_detect_band

    def get_first_redwards_band(
        self: Self,
        ref_wav: u.Quantity,
        ignore_bands: Optional[Union[str, List[str]]] = None,
    ) -> Optional[str]:
        return funcs.get_first_redwards_band(self.z, self.filterset, ref_wav, ignore_bands)
    
    def get_first_bluewards_band(
        self: Self,
        ref_wav: u.Quantity,
        ignore_bands: Optional[Union[str, List[str]]] = None,
    ) -> Optional[str]:
        return funcs.get_first_bluewards_band(self.z, self.filterset, ref_wav, ignore_bands)

    def _make_phot_from_scattered_fluxes(
        self: Self,
        scattered_fluxes: np.ndarray, 
        n_scatter: int
    ) -> List[Photometry_rest]:
        return [
            Photometry_rest(
                self.filterset,
                scattered_fluxes[:, i],
                self.flux_errs,
                self.depths,
                self.z,
            )
            for i in range(n_scatter)
        ]

    # @property
    # def lum_distance(self):
    #     try:
    #         return self._lum_distance
    #     except AttributeError:
    #         self._lum_distance = astropy_cosmo.luminosity_distance(self.z).to(u.pc)
    #         return self._lum_distance

    # @property
    # def rest_UV_band_index(self):
    #     return np.abs(self.wav - 1500 * u.Angstrom).argmin()

    # @property
    # def rest_UV_band(self):
    #     return self.instrument[self.rest_UV_band_index]

    # @property
    # def rest_UV_band_flux(self):
    #     return self.flux[self.rest_UV_band_index]

    # Should already exist in Photometry object
    # def scatter(self, n_scatter=1):
    #     assert self.flux.unit != u.ABmag, galfind_logger.critical(
    #         f"{self.flux.unit=} == 'ABmag'"
    #     )
    #     phot_matrix = np.array(
    #         [
    #             np.random.normal(flux, err, n_scatter)
    #             for flux, err in zip(
    #                 self.flux.value, self.flux_errs.value
    #             )
    #         ]
    #     )
    #     return [
    #         self.__class__(
    #             self.instrument,
    #             phot_matrix[:, i] * self.flux.unit,
    #             self.flux_errs,
    #             self.depths,
    #             self.z,
    #         )
    #         for i in range(n_scatter)
    #     ]

    def is_correctly_UV_cropped(self, rest_UV_wav_lims):
        if hasattr(self, "UV_wav_range"):
            if self.UV_wav_range == self.rest_UV_wavs_name(rest_UV_wav_lims):
                assert (
                    u.get_physical_type(self.flux.unit)
                    == "power density/spectral flux density wav"
                )
                return True
        return False

    def PL_amplitude_name(self, rest_UV_wav_lims):
        return f"A_PL_{self.rest_UV_wavs_name(rest_UV_wav_lims)}"

    def _calc_property(
        self,
        SED_rest_property_function,
        iters,
        save_path: Optional[str] = None,
        **kwargs,
    ):
        assert type(iters) in [int]
        # Add iters to kwargs
        kwargs["iters"] = iters
        if not inspect.ismethod(SED_rest_property_function):
            kwargs["self"] = self

        property_names = SED_rest_property_function(
            extract_property_name=True, **kwargs
        )
        # calculate number of new iterations that should be run
        property_iters_arr = np.zeros(len(property_names))
        for i, property_name in enumerate(property_names):
            property_iters = iters
            if property_name in self.property_PDFs.keys():
                PDF_obj = self.property_PDFs[property_name]
                if type(PDF_obj) == type(None):
                    pass
                elif len(PDF_obj) < iters:
                    property_iters = iters - len(PDF_obj)
                else:
                    property_iters = 0
            property_iters_arr[i] = property_iters
        assert all(
            property_iters == property_iters_arr[0]
            for property_iters in property_iters_arr
        ), f"All {property_names=} must have the same number of iterations to run!, {[(property_iters, property_iters_arr[0]) for property_iters in property_iters_arr]}"
        # do nothing if PDFs of the required length have already been loaded
        if property_iters_arr[0] == 0:
            return self, property_names
        # compute the properties
        kwargs["iters"] = property_iters
        # if 'self' not in kwargs.keys():
        #    kwargs["self"] = self
        # kwargs["self"] = self
        kwargs["save_path"] = save_path

        properties = SED_rest_property_function(**kwargs)[0]
        # breakpoint()
        # print(properties, property_name)
        for property, property_name in zip(properties, property_names):
            # update property PDFs
            if type(property) == type(None):
                self.property_PDFs[property_name] = None
            else:
                if type(property) not in [dict]:
                    galfind_logger.critical(
                        f"{type(property)=} not in [dict]!"
                    )
                    breakpoint()
                # breakpoint()
                # construct PDF from property output if required
                if "PDF" in property.keys():
                    new_PDF = property["PDF"]
                elif (
                    "vals" in property.keys()
                    and "PDF_kwargs" in property.keys()
                ):
                    new_PDF = PDF.from_1D_arr(
                        property_name, property["vals"], property["PDF_kwargs"]
                    )
                else:
                    galfind_logger.critical(
                        f"{property.keys()=} for {property_name} does not include either ['vals', 'PDF_kwargs'] or 'PDF'!"
                    )
                if property_name in self.property_PDFs.keys():
                    old_PDF = self.property_PDFs[property_name]
                    PDF_obj = old_PDF + new_PDF
                else:
                    PDF_obj = new_PDF
                self.recently_updated.append(property_name)
                self.property_PDFs[property_name] = PDF_obj
                self._update_properties_from_PDF(property_name)
                if type(save_path) != type(None):
                    self._save_SED_rest_PDF(
                        property_name,
                        save_path.replace(
                            "/property_name/", f"/{property_name}/"
                        ),
                    )
        return self, property_names

    def _save_SED_rest_PDF(self, property_name, save_path):
        funcs.make_dirs(save_path)
        if self.property_PDFs[property_name]is not None:
            self.property_PDFs[property_name].save(save_path)

    def _update_properties_from_PDF(self, property_name):
        if self.property_PDFs[property_name] is None:
            self.properties[property_name] = np.nan
            self.property_errs[property_name] = [np.nan, np.nan]
        else:
            self.properties[property_name] = self.property_PDFs[
                property_name
            ].median
            self.property_errs[property_name] = self.property_PDFs[
                property_name
            ].errs

    # def plot(self, save_dir, ID, plot_fit = True, iters = 10_000, save = True, show = False, n_interp = 100, conv_filt = False):
    #     self.make_rest_UV_phot()
    #     assert(conv_filt == False)
    #     #if not all(beta == -99. for beta in self.beta_PDF):
    #     sns.set(style="whitegrid")
    #     warnings.filterwarnings("ignore")

    #     # Create figure and axes
    #     fig, ax = plt.subplots()

    #     # Plotting code
    #     ax.errorbar(np.log10(self.rest_UV_phot.wav.value), self.rest_UV_phot.log_flux_lambda, yerr = self.rest_UV_phot.log_flux_lambda_errs,
    #                  ls = "none", c = "black", zorder = 10, marker = "o", markersize = 5, capsize = 3)

    #     if plot_fit:
    #         fit_lines = []
    #         fit_lines_interped = []
    #         wav_interp = np.linspace(np.log10(self.rest_UV_phot.wav.value)[0], np.log10(self.rest_UV_phot.wav.value)[-1], n_interp)
    #         if iters > len(self.amplitude_PDF[conv_filt]):
    #             iters = len(self.amplitude_PDF[conv_filt])
    #         for i in range(iters):
    #             #percentiles = np.percentile([16, 50, 84], axis=0)
    #             f_interp = interp1d(np.log10(self.rest_UV_phot.wav.value), np.log10(Photometry_rest.beta_slope_power_law_func(self.rest_UV_phot.wav.value, \
    #                 self.amplitude_PDF[conv_filt][i], self.beta_PDF[conv_filt][i])), kind = 'linear')
    #             y_new = f_interp(wav_interp)
    #             fit_lines.append(np.log10(Photometry_rest.beta_slope_power_law_func(self.rest_UV_phot.wav.value, self.amplitude_PDF[conv_filt][i], self.beta_PDF[conv_filt][i])))
    #             fit_lines_interped.append(y_new)
    #         fit_lines_interped = np.array(fit_lines_interped)
    #         fit_lines = np.array(fit_lines)
    #         fit_lines.reshape(iters, len(self.rest_UV_phot.wav.value))
    #         fit_lines_interped.reshape(iters, len(wav_interp))

    #         l1_chains = np.array([np.percentile(x, 16) for x in fit_lines_interped.T])
    #         med_chains = np.array([np.percentile(x, 50) for x in fit_lines.T])
    #         u1_chains = np.array([np.percentile(x, 84) for x in fit_lines_interped.T])

    #         ax.plot(np.log10(self.rest_UV_phot.wav.value), med_chains, color = "red", zorder = 2)
    #         ax.fill_between(wav_interp, l1_chains, u1_chains, color="grey", alpha=0.2, zorder=1)

    #     ax.set_xlabel(r"$\log_{10}(\lambda_{\mathrm{rest}} / \mathrm{\AA})$")
    #     ax.set_ylabel(r"$\log_{10}(\mathrm{f}_{\lambda_{\mathrm{rest}}} / \mathrm{erg} \, \mathrm{s}^{-1} \, \mathrm{cm}^{-2} \, \mathrm{\AA}^{-1})$")

    #     # Add the Galaxy ID label
    #     ax.text(0.05, 0.05, f"Galaxy ID = {str(ID)}", transform=ax.transAxes, ha="left", va="bottom", fontsize=12)
    #     # Add the Beta label
    #     ax.text(0.95, 0.95, r"$\beta$" + " = {:.2f} $^{{+{:.2f}}}_{{-{:.2f}}}$".format(np.percentile(self.beta_PDF[conv_filt], 50), \
    #         np.percentile(self.beta_PDF[conv_filt], 84) - np.percentile(self.beta_PDF[conv_filt], 50), np.percentile(self.beta_PDF[conv_filt], 50) - \
    #         np.percentile(self.beta_PDF[conv_filt], 16)), transform = ax.transAxes, ha = "right", va = "top", fontsize = 12)

    #     ax.set_xlim(*np.log10(self.rest_UV_wav_lims.value))

    #     if save:
    #         path = f"{save_dir}/plots/{ID}.png"
    #         funcs.make_dirs(path)
    #         fig.savefig(path, dpi=300, bbox_inches='tight')

    #     if show:
    #         plt.tight_layout()
    #         plt.show()

    #     plt.clf()

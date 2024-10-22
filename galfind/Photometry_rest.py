#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:04:24 2023

@author: austind
"""

# Photometry_rest.py
import inspect
from copy import deepcopy
from typing import Union, Optional

import astropy.units as u
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

from . import PDF, PDF_nD, Photometry, galfind_logger
from . import useful_funcs_austind as funcs
from .decorators import ignore_warnings
from .Dust_Attenuation import AUV_from_beta, Dust_Attenuation
from .Emission_lines import line_diagnostics, strong_optical_lines


class beta_fit:
    def __init__(self, z, bands):
        self.band_names = [band.band_name for band in bands]
        self.wavelength_rest = {}
        self.transmission = {}
        self.norm = {}
        for band in bands:
            self.wavelength_rest[band.band_name] = np.array(
                funcs.convert_wav_units(band.wav, u.AA).value / (1.0 + z)
            )
            self.transmission[band.band_name] = np.array(band.trans)
            self.norm[band.band_name] = np.trapz(
                self.transmission[band.band_name],
                x=self.wavelength_rest[band.band_name],
            )

    def beta_slope_power_law_func_conv_filt(self, _, A, beta):
        return np.array(
            [
                np.trapz(
                    (10**A)
                    * (self.wavelength_rest[band_name] ** beta)
                    * self.transmission[band_name],
                    x=self.wavelength_rest[band_name],
                )
                / self.norm[band_name]
                for band_name in self.band_names
            ]
        )


SFR_conversions = {
    "MD14": 1.15e-28 * (u.solMass / u.yr) / (u.erg / (u.s * u.Hz))
}

fesc_from_beta_conversions = {
    "Chisholm22": lambda beta: np.random.normal(1.3, 0.6, len(beta))
    * 10 ** (-4.0 - np.random.normal(1.22, 0.1, len(beta)) * beta)
}


class Photometry_rest(Photometry):
    def __init__(
        self,
        instrument,
        flux,
        flux_errs,
        depths,
        z,
        properties={},
        property_errs={},
        property_PDFs={},
    ):
        self.z = z
        self.properties = properties
        self.property_errs = property_errs
        self.property_PDFs = property_PDFs
        self.recently_updated = []
        super().__init__(instrument, flux, flux_errs, depths)

    # these class methods need updating!
    @classmethod
    def from_fits_cat(cls, fits_cat_row, instrument, cat_creator, code):
        phot = Photometry.from_fits_cat(fits_cat_row, instrument, cat_creator)
        # TODO: mask the photometry object
        return cls.from_phot(
            phot, np.float(fits_cat_row[code.galaxy_properties["z"]])
        )

    @classmethod
    def from_phot(cls, phot, z):
        return cls(phot.instrument, phot.flux, phot.flux_errs, phot.depths, z)

    @classmethod
    def from_phot_obs(cls, phot):
        return cls(
            phot.instrument,
            phot.flux,
            phot.flux_errs,
            phot.depths,
            phot.z,
        )

    def __str__(self, print_PDFs=True):
        line_sep = "*" * 40 + "\n"
        band_sep = "-" * 10 + "\n"
        output_str = line_sep
        output_str += f"PHOTOMETRY_REST: z = {self.z}\n"
        output_str += band_sep
        # don't print the photometry here, only the derived properties
        if print_PDFs:
            for PDF_obj in self.property_PDFs.values():
                output_str += str(PDF_obj)
        output_str += line_sep
        return output_str

    def __len__(self):
        return len(self.flux)

    def __getattr__(
        self,
        property_name: str,
        origin: str = "phot_rest",
        property_type: Union[None, str] = None,
    ) -> Union[None, bool, u.Quantity, u.Magnitude, u.Dex]:
        if origin == "phot_rest":
            if type(property_type) == type(None):
                return super().__getattr__(property_name, "phot")
            assert property_type in [
                "val",
                "errs",
                "l1",
                "u1",
                "pdf",
                "recently_updated",
            ], galfind_logger.critical(
                f"{property_type=} not in ['val', 'errs', 'l1', 'u1', 'pdf', 'recently_updated']!"
            )
            # boolean output to say whether property has been recently updated
            if property_type == "recently_updated":
                return (
                    True if property_name in self.recently_updated else False
                )
            else:
                # extract relevant property if name in dict.keys()
                if property_type == "val":
                    access_dict = self.properties
                elif property_type in ["errs", "l1", "u1"]:
                    access_dict = self.property_errs
                else:
                    access_dict = self.property_PDFs
                # return None if relevant property is not available
                if property_name not in access_dict.keys():
                    err_message = f"{property_name} {property_type} not available in Photometry_rest object!"
                    galfind_logger.warning(err_message)
                    raise AttributeError(err_message)  # may be required here
                else:
                    if property_type == "l1":
                        return access_dict[property_name][0]
                    elif property_type == "u1":
                        return access_dict[property_name][1]
                    else:
                        return access_dict[property_name]
        else:
            galfind_logger.critical(
                f"Photometry_rest.__getattr__ currently has no implementation of {origin=} != 'phot_rest'"
            )
            raise NotImplementedError

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, deepcopy(value, memo))
        return result

    @property
    def first_Lya_detect_band(
        self, Lya_wav=line_diagnostics["Lya"]["line_wav"]
    ):
        try:
            return self._first_Lya_detect_band
        except AttributeError:
            first_band = None
            for band, lower_wav in zip(
                self.instrument.band_names, self.instrument.band_lower_wav_lims
            ):
                if lower_wav > Lya_wav * (1 + self.z):
                    first_band = band
                    break
            self._first_Lya_detect_band = first_band
            return self._first_Lya_detect_band

    @property
    def first_Lya_non_detect_band(
        self, Lya_wav=line_diagnostics["Lya"]["line_wav"]
    ):
        try:
            return self._first_Lya_non_detect_band
        except AttributeError:
            first_band = None
            # bands already ordered from blue -> red
            for band, upper_wav in zip(
                self.instrument.band_names, self.instrument.band_upper_wav_lims
            ):
                if upper_wav < Lya_wav * (1 + self.z):
                    first_band = band
                    break
            self._first_Lya_non_detect_band = first_band
        return self._first_Lya_non_detect_band

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

    @staticmethod
    def rest_UV_wavs_name(rest_UV_wav_lims):
        assert u.get_physical_type(
            rest_UV_wav_lims == "length"
        ), galfind_logger.critical(
            f"{u.get_physical_type(rest_UV_wav_lims)=} != 'length'"
        )
        rest_UV_wav_lims = [
            int(
                funcs.convert_wav_units(
                    rest_UV_wav_lim * rest_UV_wav_lims.unit, u.AA
                ).value
            )
            for rest_UV_wav_lim in rest_UV_wav_lims.value
        ]
        return f"{str(rest_UV_wav_lims).replace(' ', '')}AA"

    def get_rest_UV_phot(self, rest_UV_wav_lims):
        phot_rest_copy = deepcopy(self)
        rest_UV_wav_lims = funcs.convert_wav_units(rest_UV_wav_lims, u.AA)
        crop_indices = [
            int(i)
            for i, band in enumerate(self.instrument)
            if funcs.convert_wav_units(band.WavelengthLower50, u.AA).value
            < rest_UV_wav_lims.value[0] * (1.0 + self.z)
            or funcs.convert_wav_units(band.WavelengthUpper50, u.AA).value
            > rest_UV_wav_lims.value[1] * (1.0 + self.z)
        ]
        phot_rest_copy.crop_phot(crop_indices)
        phot_rest_copy.UV_wav_range = phot_rest_copy.rest_UV_wavs_name(
            rest_UV_wav_lims
        )
        return phot_rest_copy

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

    # Rest-frame UV property calculations

    @ignore_warnings
    def calc_beta_phot(
        self,
        rest_UV_wav_lims,
        iters=10,
        maxfev=100_000,
        beta_fit_func=None,
        extract_property_name=False,
        incl_errs=True,
        save_path=None,
        single_iter: bool = False,
        output_kwargs: bool = True,
    ):
        assert type(single_iter) == bool
        assert iters >= 0, galfind_logger.critical(
            f"{iters=} < 0 in Photometry_rest.calc_beta_phot !!!"
        )
        assert type(iters) == int, galfind_logger.critical(
            f"{type(iters)=} != 'int' in Photometry_rest.calc_beta_phot !!!"
        )
        # if iters == 1:
        #     # iters = 1 -> fit without errors, iters >> 1 -> fit with errors
        #     galfind_logger.warning("Cannot properly load from catalogue if iters == 1")
        property_name = f"beta_PL_{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
        PL_amplitude_name = self.PL_amplitude_name(rest_UV_wav_lims)
        if extract_property_name:
            return [PL_amplitude_name, property_name]

        property_stored = (
            property_name in self.properties.keys()
            and property_name in self.property_errs.keys()
            and property_name in self.property_PDFs.keys()
            and PL_amplitude_name in self.properties.keys()
            and PL_amplitude_name in self.property_errs.keys()
            and PL_amplitude_name in self.property_PDFs.keys()
        )
        if property_stored:
            if type(self.property_PDFs[property_name]) == type(None) or type(
                self.property_PDFs[PL_amplitude_name]
            ) == type(None):
                # run already and returned None for whatever reason (usually no rest-frame UV bands)
                return [None, None], [PL_amplitude_name, property_name]
            elif iters == 0:
                # run already to the required length
                return [
                    {
                        "vals": self.property_PDFs[
                            PL_amplitude_name
                        ].input_arr,
                        "PDF_kwargs": self.property_PDFs[
                            PL_amplitude_name
                        ].kwargs,
                    },
                    {
                        "vals": self.property_PDFs[property_name].input_arr,
                        "PDF_kwargs": self.property_PDFs[property_name].kwargs,
                    },
                ], [PL_amplitude_name, property_name]
        assert (
            iters != 0
        )  # iterations only zero when the property is stored already, no point in running this function otherwise
        # if either not already stored or there are still iterations to run, run iterations
        if self.is_correctly_UV_cropped(rest_UV_wav_lims):
            rest_UV_phot = self
        else:
            rest_UV_phot = self.get_rest_UV_phot(rest_UV_wav_lims)
        if single_iter:
            if len(rest_UV_phot) == 0:
                return [np.nan, np.nan], {}
            if output_kwargs:
                kwargs = {
                    "rest_UV_band_names": "+".join(
                        rest_UV_phot.instrument.band_names
                    ),
                    "n_UV_bands": len(rest_UV_phot.instrument),
                }
            else:
                kwargs = {}
            # ideally this is pre-computed
            if type(beta_fit_func) == type(None):
                beta_fit_func = beta_fit(
                    rest_UV_phot.z, rest_UV_phot.instrument.bands
                ).beta_slope_power_law_func_conv_filt
            f_lambda = funcs.convert_mag_units(
                [
                    funcs.convert_wav_units(band.WavelengthCen, u.AA).value
                    for band in rest_UV_phot.instrument
                ]
                * u.AA,
                rest_UV_phot.flux,
                u.erg / (u.s * u.AA * u.cm**2),
            )
            if not incl_errs:
                return curve_fit(beta_fit_func, None, f_lambda, maxfev=maxfev)[
                    0
                ], kwargs  # , [PL_amplitude_name, property_name]
            else:
                f_lambda_errs = funcs.convert_mag_err_units(
                    [
                        funcs.convert_wav_units(band.WavelengthCen, u.AA).value
                        for band in rest_UV_phot.instrument
                    ]
                    * u.AA,
                    rest_UV_phot.flux,
                    [
                        rest_UV_phot.flux_errs.value,
                        rest_UV_phot.flux_errs.value,
                    ]
                    * rest_UV_phot.flux_errs.unit,
                    u.erg / (u.s * u.AA * u.cm**2),
                )
                return curve_fit(
                    beta_fit_func,
                    None,
                    f_lambda,
                    sigma=f_lambda_errs[0],
                    maxfev=maxfev,
                )[0], kwargs  # , [PL_amplitude_name, property_name]
        else:
            if len(rest_UV_phot) == 0:
                return [None, None], [PL_amplitude_name, property_name]
            else:
                assert type(beta_fit_func) == type(None)
                scattered_rest_UV_phot_arr = rest_UV_phot.scatter(iters)
                beta_fit_func = beta_fit(
                    rest_UV_phot.z, rest_UV_phot.instrument.bands
                ).beta_slope_power_law_func_conv_filt
                popt_arr = np.array(
                    [
                        scattered_rest_UV_phot.calc_beta_phot(
                            rest_UV_wav_lims,
                            single_iter=True,
                            output_kwargs=False,
                            beta_fit_func=beta_fit_func,
                        )[0]
                        for scattered_rest_UV_phot in tqdm(
                            scattered_rest_UV_phot_arr,
                            total=iters,
                            desc="Calculating beta_PL",
                        )
                    ]
                )
                A_arr = (10 ** popt_arr[:, 0]) * u.erg / (u.s * u.AA * u.cm**2)
                beta_arr = popt_arr[:, 1] * u.dimensionless_unscaled
                PDF_kwargs = {
                    "rest_UV_band_names": "+".join(
                        rest_UV_phot.instrument.band_names
                    ),
                    "n_UV_bands": len(rest_UV_phot.instrument),
                }
                return [
                    {"vals": A_arr, "PDF_kwargs": PDF_kwargs},
                    {"vals": beta_arr, "PDF_kwargs": PDF_kwargs},
                ], [PL_amplitude_name, property_name]

    def calc_fesc_from_beta_phot(
        self,
        rest_UV_wav_lims,
        conv_author_year,
        iters=10,
        extract_property_name=False,
        save_path=None,
        single_iter: bool = False,
    ):
        assert type(single_iter) == bool
        assert conv_author_year in fesc_from_beta_conversions.keys()
        property_name = f"fesc_{conv_author_year}"  # _{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
        if extract_property_name:
            return [property_name]
        if single_iter:
            popt, kwargs = self.calc_beta_phot(
                rest_UV_wav_lims, output_kwargs=True, single_iter=True
            )
            return fesc_from_beta_conversions[conv_author_year](
                popt[1]
            ), kwargs
        else:
            beta_property_name = self._calc_property(
                Photometry_rest.calc_beta_phot,
                iters=iters,
                rest_UV_wav_lims=rest_UV_wav_lims,
                save_path=save_path,
            )[1][1]
            # update PDF
            if type(self.property_PDFs[beta_property_name]) == type(None):
                return [None], [property_name]
            else:
                return [
                    {
                        "PDF": self.property_PDFs[
                            beta_property_name
                        ].manipulate_PDF(
                            property_name,
                            fesc_from_beta_conversions[conv_author_year],
                            size=iters,
                        )
                    }
                ], [property_name]

    def calc_AUV_from_beta_phot(
        self,
        rest_UV_wav_lims,
        ref_wav,
        dust_author_year,
        iters=10,
        extract_property_name=False,
        save_path=None,
        single_iter: bool = False,
    ):
        assert type(single_iter) == bool
        dust_author_year_cls = globals()[dust_author_year]
        assert issubclass(dust_author_year_cls, AUV_from_beta)
        UV_dust_label = self._get_UV_dust_label(dust_author_year)
        property_name = f"A{ref_wav.to(u.AA).value:.0f}{UV_dust_label}"  # _{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
        if extract_property_name:
            return [property_name]
        if single_iter:
            popt, kwargs = self.calc_beta_phot(
                rest_UV_wav_lims=rest_UV_wav_lims, single_iter=True
            )
            return dust_author_year_cls()(
                popt[1] * u.dimensionless_unscaled
            ), kwargs
        else:
            beta_property_name = self._calc_property(
                Photometry_rest.calc_beta_phot,
                iters=iters,
                rest_UV_wav_lims=rest_UV_wav_lims,
                save_path=save_path,
            )[1][1]
            if type(self.property_PDFs[beta_property_name]) == type(None):
                return [None], [property_name]
            else:
                return [
                    {
                        "PDF": self.property_PDFs[
                            beta_property_name
                        ].manipulate_PDF(
                            property_name, dust_author_year_cls(), size=iters
                        )
                    }
                ], [property_name]

    def calc_mUV_phot(
        self,
        rest_UV_wav_lims,
        ref_wav,
        top_hat_width=100.0 * u.AA,
        resolution=1.0 * u.AA,
        iters=10,
        extract_property_name=False,
        save_path=None,
        single_iter: bool = True,
    ):
        assert type(single_iter) == bool
        property_name = f"m{ref_wav.to(u.AA).value:.0f}"  # _{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
        if extract_property_name:
            return [property_name]
        if single_iter:
            popt, kwargs = self.calc_beta_phot(
                rest_UV_wav_lims, single_iter=True
            )
        else:
            name_arr = self._calc_property(
                Photometry_rest.calc_beta_phot,
                iters=iters,
                rest_UV_wav_lims=rest_UV_wav_lims,
                save_path=save_path,
            )[1]
            ampl_name, beta_name = name_arr[0], name_arr[1]
            if type(self.property_PDFs[ampl_name]) == type(None) or type(
                self.property_PDFs[beta_name]
            ) == type(None):
                return [None], [property_name]
            assert len(self.property_PDFs[ampl_name]) == len(
                self.property_PDFs[beta_name]
            )
            ampl_beta_joint_PDF = PDF_nD(
                [self.property_PDFs[ampl_name], self.property_PDFs[beta_name]]
            )
        rest_wavelengths = funcs.convert_wav_units(
            np.linspace(
                ref_wav - top_hat_width / 2,
                ref_wav + top_hat_width / 2,
                int(
                    np.round(
                        (top_hat_width / resolution)
                        .to(u.dimensionless_unscaled)
                        .value,
                        0,
                    )
                ),
            ),
            u.AA,
        )
        if single_iter:
            chain = funcs.power_law_beta_func(
                rest_wavelengths.value, popt[0], popt[1]
            )
            mUV = (
                np.array(
                    np.median(
                        funcs.convert_mag_units(
                            rest_wavelengths * (1.0 + self.z),
                            chain * u.erg / (u.s * u.AA * u.cm**2),
                            u.ABmag,
                        ).value
                    )
                )
                * u.ABmag
            )
            return mUV, kwargs
        else:
            power_law_chains = ampl_beta_joint_PDF(
                funcs.power_law_beta_func, rest_wavelengths.value, size=iters
            )
            # take the median of each chain to form a new chain
            mUV_chain = (
                np.array(
                    [
                        np.median(
                            funcs.convert_mag_units(
                                rest_wavelengths * (1.0 + self.z),
                                chain * u.erg / (u.s * u.AA * u.cm**2),
                                u.ABmag,
                            ).value
                        )
                        for chain in power_law_chains
                    ]
                )
                * u.ABmag
            )
            return [
                {
                    "vals": mUV_chain,
                    "PDF_kwargs": self.property_PDFs[beta_name].kwargs,
                }
            ], [property_name]

    def calc_MUV_phot(
        self,
        rest_UV_wav_lims,
        ref_wav,
        iters=10,
        extract_property_name=False,
        save_path: Union[str, None] = None,
    ):
        property_name = f"M{ref_wav.to(u.AA).value:.0f}"  # _{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
        if extract_property_name:
            return [property_name]
        mUV_property_name = self._calc_property(
            Photometry_rest.calc_mUV_phot,
            iters=iters,
            rest_UV_wav_lims=rest_UV_wav_lims,
            ref_wav=ref_wav,
            save_path=save_path,
        )[1][0]
        if type(self.property_PDFs[mUV_property_name]) == type(None):
            return [None], [property_name]
        else:
            return [
                {
                    "PDF": self.property_PDFs[
                        mUV_property_name
                    ].manipulate_PDF(
                        property_name,
                        lambda mUV: mUV.unit
                        * (
                            mUV.value
                            - 5.0
                            * np.log10(
                                funcs.calc_lum_distance(self.z).to(u.pc).value
                                / 10.0
                            )
                            + 2.5 * np.log10(1.0 + self.z)
                        ),
                        size=iters,
                    )
                }
            ], [property_name]

    def calc_LUV_phot(
        self,
        frame: str,
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        dust_author_year: Union[str, None] = "M99",
        iters=10,
        extract_property_name: bool = False,
        save_path: Union[str, None] = None,
        single_iter: bool = False,
    ):
        assert type(single_iter) == bool
        assert frame in ["rest", "obs"]
        if type(dust_author_year) == type(None):
            UV_dust_label = ""
        else:
            UV_dust_label = self._get_UV_dust_label(dust_author_year)
        property_name = f"L{frame}_{ref_wav.to(u.AA).value:.0f}{UV_dust_label}"  # _{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
        if extract_property_name:
            return [property_name]
        if single_iter:
            mUV, mUV_kwargs = self.calc_mUV_phot(
                rest_UV_wav_lims, ref_wav=ref_wav, single_iter=True
            )
            if type(dust_author_year) != type(None):
                AUV, AUV_kwargs = self.calc_AUV_from_beta_phot(
                    rest_UV_wav_lims,
                    conv_author_year=dust_author_year,
                    ref_wav=ref_wav,
                    single_iter=True,
                )
            wav_len = 1
        else:
            mUV_property_name = self._calc_property(
                Photometry_rest.calc_mUV_phot,
                iters=iters,
                rest_UV_wav_lims=rest_UV_wav_lims,
                ref_wav=ref_wav,
                save_path=save_path,
            )[1][0]
            if type(dust_author_year) == type(None):
                AUV_property_name = None
            else:
                AUV_property_name = self._calc_property(
                    Photometry_rest.calc_AUV_from_beta_phot,
                    iters=iters,
                    rest_UV_wav_lims=rest_UV_wav_lims,
                    ref_wav=ref_wav,
                    conv_author_year=dust_author_year,
                )[1][0]
            if type(self.property_PDFs[mUV_property_name]) == type(None):
                return [None], [property_name]
            if AUV_property_name in self.property_PDFs.keys():
                if type(self.property_PDFs[AUV_property_name]) == type(None):
                    return [None], [property_name]
            wav_len = len(self.property_PDFs[mUV_property_name])

        if frame == "rest":
            z = 0.0
            wavs = np.full(wav_len, ref_wav)
        else:
            z = self.z
            wavs = np.full(wav_len, ref_wav * (1.0 + self.z))
        if single_iter:
            UV_lum = funcs.flux_to_luminosity(
                mUV, wavs=wavs, z=z, out_units=u.erg / (u.s * u.Hz)
            )
            if type(dust_author_year) == type(None):  # do not dust correct
                return UV_lum, mUV_kwargs
            else:
                breakpoint()
                return funcs.dust_correct(
                    np.array(UV_lum), dust_mag=np.array(AUV)
                ), {**mUV_kwargs, **AUV_kwargs}
        else:
            UV_lum = self.property_PDFs[mUV_property_name].manipulate_PDF(
                property_name,
                funcs.flux_to_luminosity,
                wavs=wavs,
                z=z,
                out_units=u.erg / (u.s * u.Hz),
                size=iters,
            )
            if type(dust_author_year) == type(None):  # do not dust correct
                PDF = UV_lum
            else:  # dust correct
                PDF = UV_lum.manipulate_PDF(
                    property_name,
                    funcs.dust_correct,
                    dust_mag=self.property_PDFs[AUV_property_name].input_arr[
                        -iters:
                    ],
                    size=iters,
                )
            return [{"PDF": PDF}], [property_name]

    def calc_SFR_UV_phot(
        self,
        frame: str = "obs",
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        dust_author_year: Optional[str] = "M99",
        kappa_UV_conv_author_year: str = "MD14",
        iters=10,
        extract_property_name=False,
        save_path: Union[str, None] = None,
    ):
        assert kappa_UV_conv_author_year in SFR_conversions.keys()
        UV_dust_label = self._get_UV_dust_label(dust_author_year)
        property_name = f"SFR{frame}_{ref_wav.to(u.AA).value:.0f}{UV_dust_label}_{kappa_UV_conv_author_year}"  # _{self.rest_UV_wavs_name(rest_UV_wav_lims)}"
        if extract_property_name:
            return [property_name]
        LUV_property_name = self._calc_property(
            Photometry_rest.calc_LUV_phot,
            iters=iters,
            frame=frame,
            rest_UV_wav_lims=rest_UV_wav_lims,
            ref_wav=ref_wav,
            dust_author_year=dust_author_year,
            save_path=save_path,
        )[1][0]
        if type(self.property_PDFs[LUV_property_name]) == type(None):
            return [None], [property_name]
        else:
            return [
                {
                    "PDF": self.property_PDFs[
                        LUV_property_name
                    ].manipulate_PDF(
                        property_name,
                        lambda LUV: LUV
                        * SFR_conversions[kappa_UV_conv_author_year],
                        size=iters,
                    )
                }
            ], [property_name]

    def calc_cont_rest_optical(
        self,
        strong_line_names: Union[str, list],
        rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
        iters: int = 10,
        extract_property_name: bool = False,
        save_path: Union[str, None] = None,
        single_iter: bool = False,
    ):
        assert type(single_iter) == bool
        if type(strong_line_names) in [str]:
            strong_line_names = [strong_line_names]
        assert all(
            line_name in strong_optical_lines
            for line_name in strong_line_names
        )
        property_name = f"cont_{'+'.join(strong_line_names)}"
        if extract_property_name:
            return [property_name]
        # calculate wavelength tolerance given typical photo-z precision, dz
        # dlambda = {line_name: self._get_wav_line_precision(line_name, dz) for line_name in strong_line_names}
        # determine the band which contains the first line
        emission_band = self.instrument.nearest_band_to_wavelength(
            line_diagnostics[strong_line_names[0]]["line_wav"]
            * (1.0 + self.z),
            medium_bands_only=False,
            check_wavelength_in_band=False,
        )
        # ensure all lines lie within this band (defined by 50% throughput boundaries plus/minus photo-z errors) and that there are no other strong optical lines in this band
        if not all(
            (line_diagnostics[line_name]["line_wav"]) * (1.0 + self.z)
            < emission_band.WavelengthUpper50
            and (line_diagnostics[line_name]["line_wav"]) * (1.0 + self.z)
            > emission_band.WavelengthLower50
            for line_name in strong_line_names
        ) or any(
            line_diagnostics[line_name]["line_wav"] * (1.0 + self.z)
            < emission_band.WavelengthUpper50
            and line_diagnostics[line_name]["line_wav"] * (1.0 + self.z)
            > emission_band.WavelengthLower50
            for line_name in strong_optical_lines
            if line_name not in strong_line_names
        ):
            if single_iter:
                return np.nan, {}
            else:
                return [None], [property_name]
        # determine photometric bands that trace the continuum both bluewards and redwards of the line(s) - must avoid other strong optical lines and lie within the rest optical wavelength range
        emission_band_index = self.instrument.index_from_band_name(
            emission_band.band_name
        )
        # redwards_i = emission_band_index
        # bluewards_i = emission_band_index
        cont_bands = []
        for band in self.instrument:
            if (
                band.WavelengthUpper50 < rest_optical_wavs[1] * (1.0 + self.z)
                and band.WavelengthLower50
                > rest_optical_wavs[0] * (1.0 + self.z)
                and not any(
                    line_diagnostics[line_name]["line_wav"] * (1.0 + self.z)
                    < band.WavelengthUpper50
                    and line_diagnostics[line_name]["line_wav"]
                    * (1.0 + self.z)
                    > band.WavelengthLower50
                    for line_name in strong_optical_lines
                )
            ):
                cont_bands.append(band)
        # if there are no available continuum bands
        if len(cont_bands) == 0:
            if single_iter:
                return np.nan, {}
            else:
                return [None], [property_name]
        # compute nJy flux chains for each continuum band
        cont_flux_chains = {}
        for band in cont_bands:
            band_index = self.instrument.index_from_band_name(band.band_name)
            cont_flux_nJy = funcs.convert_mag_units(
                self.instrument[band_index].WavelengthCen,
                self.flux[band_index],
                u.nJy,
            ).value
            if single_iter:
                cont_flux_chains[band.band_name] = (
                    np.array(cont_flux_nJy) * u.nJy
                )
            else:
                # errors in Jy are symmetric, therefore take the mean
                cont_flux_nJy_errs = np.mean(
                    [
                        flux_err.value
                        for flux_err in funcs.convert_mag_err_units(
                            self.instrument[band_index].WavelengthCen,
                            self.flux[band_index],
                            [
                                self.flux_errs[band_index],
                                self.flux_errs[band_index],
                            ],
                            u.nJy,
                        )
                    ]
                )
                cont_flux_chains[band.band_name] = (
                    np.random.normal(cont_flux_nJy, cont_flux_nJy_errs, iters)
                    * u.nJy
                )
        # calculate potential contaminant lines
        lims = [
            np.min(
                [
                    (emission_band.WavelengthLower50 / (1.0 + self.z))
                    .to(u.AA)
                    .value
                    for name in strong_line_names
                ]
            ),
            np.max(
                [
                    (emission_band.WavelengthUpper50 / (1.0 + self.z))
                    .to(u.AA)
                    .value
                    for name in strong_line_names
                ]
            ),
        ]
        contam_lines = [
            name
            for name, line in line_diagnostics.items()
            if name not in strong_optical_lines
            and line["line_wav"].to(u.AA).value < lims[1]
            and line["line_wav"].to(u.AA).value > lims[0]
        ]
        # update kwargs
        kwargs = {
            f"{'+'.join(strong_line_names)}_cont_bands": f"{'+'.join([band.band_name for band in cont_bands])}",
            f"{'+'.join(strong_line_names)}_emission_band": emission_band.band_name,
            "contaminant_lines": f"{'+'.join(contam_lines)}",
        }
        # calculate continuum from either the continuum band flux measurement if only one continuum band
        if len(cont_bands) == 1:
            if single_iter:
                return cont_flux_chains[cont_bands[0].band_name], kwargs
            else:
                return [
                    {
                        "vals": cont_flux_chains[cont_bands[0].band_name],
                        "PDF_kwargs": kwargs,
                    }
                ], [property_name]
        # calculate continuum from interpolation to middle of the emission band if two continuum bands
        elif len(cont_bands) >= 2:
            cont_wavs = [
                (band.WavelengthCen.to(u.AA) / (1.0 + self.z)).value
                for band in cont_bands
            ]
            # blue_wav = (cont_bands[0].WavelengthCen.to(u.AA) / (1. + self.z)).value
            # red_wav = (cont_bands[1].WavelengthCen.to(u.AA) / (1. + self.z)).value
            em_wav = (
                emission_band.WavelengthCen.to(u.AA) / (1.0 + self.z)
            ).value
            if single_iter:
                cont_fluxes = [
                    cont_flux_chains[band.band_name].value
                    for band in cont_bands
                ]
                popt, pcov = curve_fit(
                    funcs.simple_power_law_func, cont_wavs, cont_fluxes
                )
                return np.array(
                    funcs.simple_power_law_func(em_wav, *popt)
                ) * u.nJy, kwargs
                # return np.interp(em_wav, cont_wavs, [cont_flux_chains[cont_bands[0].band_name].value, \
                #     cont_flux_chains[cont_bands[1].band_name].value]) * u.nJy, kwargs
            else:
                cont_fluxes = [
                    [
                        cont_flux_chains[band.band_name][i].value
                        for band in cont_bands
                    ]
                    for i in range(
                        len(cont_flux_chains[cont_bands[0].band_name])
                    )
                ]
                cont_chains = (
                    np.array(
                        [
                            funcs.simple_power_law(
                                em_wav,
                                *curve_fit(
                                    funcs.simple_power_law_func,
                                    cont_wavs,
                                    cont_fluxes_,
                                )[0],
                            )
                            for cont_fluxes_ in cont_fluxes
                        ]
                    )
                    * u.nJy
                )
                return [{"vals": cont_chains, "PDF_kwargs": kwargs}], [
                    property_name
                ]
        else:
            breakpoint()

    def calc_EW_rest_optical(
        self,
        strong_line_names: Union[str, list],
        frame: str = "rest",
        rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
        iters: int = 10,
        extract_property_name: bool = False,
        save_path: Union[str, None] = None,
        single_iter: bool = False,
    ):
        assert type(single_iter) == bool
        if type(strong_line_names) in [str]:
            strong_line_names = [strong_line_names]
        assert frame in ["rest", "obs"]
        assert all(
            line_name in strong_optical_lines
            for line_name in strong_line_names
        )

        property_name = f"EW{frame}_{'+'.join(strong_line_names)}"
        if extract_property_name:
            return [property_name]

        # calculate continuum flux PDF in nJy
        if single_iter:
            cont_val, cont_kwargs = self.calc_cont_rest_optical(
                strong_line_names, rest_optical_wavs, single_iter=True
            )
            if np.isnan(cont_val):
                return np.nan, {}
        else:
            cont_property_name = self._calc_property(
                Photometry_rest.calc_cont_rest_optical,
                iters=iters,
                strong_line_names=strong_line_names,
                rest_optical_wavs=rest_optical_wavs,
                save_path=save_path,
            )[1][0]
            if type(self.property_PDFs[cont_property_name]) == type(None):
                return [None], [property_name]
        # determine the band which contains the first line
        emission_band = self.instrument.nearest_band_to_wavelength(
            line_diagnostics[strong_line_names[0]]["line_wav"]
            * (1.0 + self.z),
            medium_bands_only=False,
            check_wavelength_in_band=False,
        )
        emission_band_index = self.instrument.index_from_band_name(
            emission_band.band_name
        )
        line_flux = funcs.convert_mag_units(
            emission_band.WavelengthCen,
            self.flux[emission_band_index],
            u.Jy,
        ).value
        if single_iter:
            line_flux = np.array(line_flux) * u.Jy
        else:
            # errors in Jy are symmetric, therefore take the mean
            line_flux_errs = np.mean(
                [
                    flux_err.value
                    for flux_err in funcs.convert_mag_err_units(
                        emission_band.WavelengthCen,
                        self.flux[emission_band_index],
                        [
                            self.flux_errs[emission_band_index],
                            self.flux_errs[emission_band_index],
                        ],
                        u.Jy,
                    )
                ]
            )
            line_flux_chain = (
                np.random.normal(line_flux, line_flux_errs, iters) * u.Jy
            )
        bandwidth = (
            emission_band.WavelengthUpper50 - emission_band.WavelengthLower50
        )
        if single_iter:
            if frame == "rest":
                return (
                    ((line_flux / cont_val).to(u.dimensionless_unscaled) - 1.0)
                    * bandwidth
                    / (1.0 + self.z)
                ).to(u.AA), cont_kwargs
            else:  # frame == "obs"
                return (
                    ((line_flux / cont_val).to(u.dimensionless_unscaled) - 1.0)
                    * bandwidth
                ).to(u.AA), cont_kwargs
        else:
            if frame == "rest":
                calc_EW_func = lambda cont_flux: (
                    (
                        (line_flux_chain / cont_flux).to(
                            u.dimensionless_unscaled
                        )
                        - 1.0
                    )
                    * bandwidth
                    / (1.0 + self.z)
                ).to(u.AA)
            else:  # frame == "obs"
                calc_EW_func = lambda cont_flux: (
                    (
                        (line_flux_chain / cont_flux).to(
                            u.dimensionless_unscaled
                        )
                        - 1.0
                    )
                    * bandwidth
                ).to(u.AA)
            return [
                {
                    "PDF": self.property_PDFs[
                        cont_property_name
                    ].manipulate_PDF(property_name, calc_EW_func, size=iters)
                }
            ], [property_name]

    def calc_dust_atten(
        self,
        calc_wav: u.Quantity,
        dust_author_year: Union[str, None] = "M99",
        dust_law: str = "C00",
        dust_origin: str = "UV",
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        iters: int = 10,
        extract_property_name: bool = False,
        save_path: Union[str, None] = None,
        single_iter: bool = False,
    ):
        assert type(single_iter) == bool
        assert u.get_physical_type(calc_wav.unit) == "length"
        if dust_origin != "UV":
            raise NotImplementedError
        if type(dust_law) != type(None):
            dust_law_cls = globals()[dust_law]  # un-initialized
            assert issubclass(dust_law_cls, Dust_Attenuation)

        property_name = f"A{calc_wav.to(u.AA).value:.0f}{self._get_dust_corr_label(dust_author_year, dust_law, dust_origin)}"
        if extract_property_name:
            return [property_name]

        if any(
            type(name) == type(None)
            for name in [dust_author_year, dust_law, dust_origin]
        ):
            return [None], [property_name]
        if single_iter:
            A_ref_wav = self.calc_AUV_from_beta_phot(
                rest_UV_wav_lims, ref_wav, dust_author_year, single_iter=True
            )
            if np.isnan(A_ref_wav):
                return np.nan, {}
            else:
                return (
                    np.array(
                        A_ref_wav.to(u.ABmag).value
                        * dust_law_cls.k_lambda(calc_wav.to(u.AA))
                        / dust_law_cls.k_lambda(ref_wav)
                    )
                    * u.ABmag
                )
        else:
            A_ref_wav_name = self._calc_property(
                Photometry_rest.calc_AUV_from_beta_phot,
                iters=iters,
                rest_UV_wav_lims=rest_UV_wav_lims,
                ref_wav=ref_wav,
                dust_author_year=dust_author_year,
                save_path=save_path,
            )[1][0]
            if type(self.property_PDFs[A_ref_wav_name]) == type(None):
                return [None], [property_name]
            dust_law_cls = dust_law_cls()  # initialize dust_law_cls
            dust_PDF = self.property_PDFs[A_ref_wav_name].manipulate_PDF(
                property_name,
                lambda A_ref_wav: (
                    A_ref_wav.to(u.ABmag).value
                    * dust_law_cls.k_lambda(calc_wav.to(u.AA))
                    / dust_law_cls.k_lambda(ref_wav)
                )
                * u.ABmag,
                size=iters,
            )
            return [{"PDF": dust_PDF}], [property_name]

    # def calc_dust_atten_line(self, line_name: str, dust_author_year: str = "M99", dust_law: str = "C00", \
    #         dust_origin: str = "UV", iters: int = 10, ref_wav: u.Quantity = 1_500. * u.AA, extract_property_name: bool = False):
    #     assert(line_name in line_diagnostics.keys())
    #     #return self.calc_dust_atten(line_diagnostics[line_name]["line_wav"].to(u.AA), dust_author_year, dust_law, )
    #     # include option to include additional dust attenuation from birth clouds here!

    def calc_line_flux_rest_optical(
        self,
        strong_line_names: Union[str, list],
        frame: str = "obs",
        dust_author_year: Union[str, None] = "M99",
        dust_law: str = "C00",
        dust_origin: str = "UV",
        rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        iters: int = 10,
        extract_property_name: bool = False,
        save_path: Union[str, None] = None,
        single_iter: bool = False,
    ):
        assert type(single_iter) == bool
        if type(strong_line_names) in [str]:
            strong_line_names = [strong_line_names]
        assert frame in ["rest", "obs"]
        assert all(
            line_name in line_diagnostics.keys()
            for line_name in strong_line_names
        )

        dust_label = self._get_dust_corr_label(
            dust_author_year=dust_author_year,
            dust_law=dust_law,
            dust_origin=dust_origin,
        )
        property_name = (
            f"flux_{'+'.join(strong_line_names)}_{frame}{dust_label}"
        )
        if extract_property_name:
            return [property_name]
        if single_iter:
            cont = self.calc_cont_rest_optical(
                strong_line_names, rest_optical_wavs, single_iter=True
            )[0]
            EW, EW_kwargs = self.calc_EW_rest_optical(
                strong_line_names, frame, rest_optical_wavs, single_iter=True
            )
            if any(np.isnan(name) for name in [cont, EW]):
                return np.nan, {}
        else:
            cont_name = self._calc_property(
                Photometry_rest.calc_cont_rest_optical,
                iters=iters,
                strong_line_names=strong_line_names,
                rest_optical_wavs=rest_optical_wavs,
                save_path=save_path,
            )[1][0]
            EW_name = self._calc_property(
                Photometry_rest.calc_EW_rest_optical,
                iters=iters,
                strong_line_names=strong_line_names,
                frame=frame,
                rest_optical_wavs=rest_optical_wavs,
                save_path=save_path,
            )[1][0]
            A_line_name = self._calc_property(
                Photometry_rest.calc_dust_atten,
                iters=iters,
                calc_wav=line_diagnostics[strong_line_names[0]]["line_wav"],
                dust_author_year=dust_author_year,
                dust_law=dust_law,
                dust_origin=dust_origin,
                rest_UV_wav_lims=rest_UV_wav_lims,
                ref_wav=ref_wav,
                save_path=save_path,
            )[1][0]
            if any(
                type(self.property_PDFs[name]) == type(None)
                for name in [A_line_name, EW_name, cont_name]
            ):
                return [None], [property_name]
            EW_kwargs = self.property_PDFs[EW_name].kwargs
        band_wav = self.instrument[
            self.instrument.index_from_band_name(
                EW_kwargs[f"{'+'.join(strong_line_names)}_emission_band"]
            )
        ].WavelengthCen

        # convert EW to line flux in appropriate frame
        if frame == "rest":
            band_wav /= 1.0 + self.z
        if single_iter:
            line_flux = EW.to(u.AA) * funcs.convert_mag_units(
                band_wav, cont, u.erg / (u.s * u.AA * u.cm**2)
            )
            if dust_label != "":
                A_line, dust_kwargs = self.calc_dust_atten(
                    line_diagnostics[strong_line_names[0]]["line_wav"],
                    dust_author_year,
                    dust_law,
                    dust_origin,
                    rest_UV_wav_lims,
                    ref_wav,
                    single_iter=True,
                )
                line_flux = funcs.dust_correct(line_flux, A_line)
                out_kwargs = {**EW_kwargs, **dust_kwargs}
            else:
                out_kwargs = EW_kwargs
            return line_flux, out_kwargs
        else:
            if dust_label == "":
                PDF_kwargs = {
                    **self.property_PDFs[cont_name].kwargs,
                    **self.property_PDFs[A_line_name].kwargs,
                }
            else:
                PDF_kwargs = self.property_PDFs[cont_name].kwargs
            line_flux_PDF = self.property_PDFs[EW_name].manipulate_PDF(
                "line_flux",
                lambda EW: EW.to(u.AA)
                * funcs.convert_mag_units(
                    band_wav,
                    self.property_PDFs[cont_name].input_arr,
                    u.erg / (u.s * u.AA * u.cm**2),
                ),
                PDF_kwargs,
                size=iters,
            )
            if dust_label != "":
                line_flux_PDF = line_flux_PDF.manipulate_PDF(
                    property_name,
                    funcs.dust_correct,
                    dust_mag=self.property_PDFs[A_line_name].input_arr,
                    size=iters,
                )
            return [{"PDF": line_flux_PDF}], [property_name]

    def calc_line_lum_rest_optical(
        self,
        strong_line_names: Union[str, list],
        frame: str = "obs",
        dust_author_year: Union[str, None] = "M99",
        dust_law: str = "C00",
        dust_origin: str = "UV",
        rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        iters: int = 10,
        extract_property_name=False,
        save_path: Union[str, None] = None,
        single_iter: bool = False,
    ):
        assert type(single_iter) == bool
        if type(strong_line_names) in [str]:
            strong_line_names = [strong_line_names]
        assert frame in ["rest", "obs"]

        dust_label = self._get_dust_corr_label(
            dust_author_year=dust_author_year,
            dust_law=dust_law,
            dust_origin=dust_origin,
        )
        property_name = (
            f"lum_{'+'.join(strong_line_names)}_{frame}{dust_label}"
        )
        if extract_property_name:
            return [property_name]

        if single_iter:
            line_flux, line_flux_kwargs = self.calc_line_flux_rest_optical(
                strong_line_names,
                frame,
                dust_author_year,
                dust_law,
                dust_origin,
                rest_optical_wavs,
                rest_UV_wav_lims,
                ref_wav,
                single_iter=True,
            )
            if np.isnan(line_flux):
                return np.nan, {}
        else:
            line_flux_property_name = self._calc_property(
                Photometry_rest.calc_line_flux_rest_optical,
                iters=iters,
                strong_line_names=strong_line_names,
                frame=frame,
                dust_author_year=dust_author_year,
                dust_law=dust_law,
                dust_origin=dust_origin,
                rest_optical_wavs=rest_optical_wavs,
                rest_UV_wav_lims=rest_UV_wav_lims,
                save_path=save_path,
            )[1][0]
            if type(self.property_PDFs[line_flux_property_name]) == type(None):
                return [None], [property_name]
        if frame == "rest":
            z = 0.0
            lum_distance = funcs.calc_lum_distance(z=z)
        else:  # frame == "obs"
            z = self.z
            lum_distance = funcs.calc_lum_distance(z=z)
        if single_iter:
            return (4 * np.pi * line_flux * lum_distance**2).to(
                u.erg / u.s
            ), line_flux_kwargs
        else:
            out_PDF = self.property_PDFs[
                line_flux_property_name
            ].manipulate_PDF(
                property_name,
                lambda line_flux: (4 * np.pi * line_flux * lum_distance**2).to(
                    u.erg / u.s
                ),
                size=iters,
            )
            return [{"PDF": out_PDF}], [property_name]

    def calc_xi_ion(
        self,
        frame: str = "rest",
        strong_line_names: list = ["Halpha"],
        fesc_author_year: str = "fesc=0.0",
        dust_author_year: Union[str, None] = "M99",
        dust_law: str = "C00",
        dust_origin: str = "UV",
        rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        iters: int = 10,
        extract_property_name: bool = False,
        save_path: Union[str, None] = None,
        single_iter: bool = False,
    ):
        assert type(single_iter) == bool
        if type(strong_line_names) in [str]:
            strong_line_names = [strong_line_names]
        assert frame in ["rest", "obs"]
        assert "Halpha" in strong_line_names

        dust_label = self._get_dust_corr_label(
            dust_author_year=dust_author_year,
            dust_law=dust_law,
            dust_origin=dust_origin,
        )
        property_name = f"xi_ion_{frame}_{'+'.join(strong_line_names)}{dust_label}_{fesc_author_year.replace('=', '')}"
        if extract_property_name:
            return [property_name]

        if fesc_author_year in fesc_from_beta_conversions.keys():
            if single_iter:
                fesc, fesc_kwargs = self.calc_fesc_from_beta_phot(
                    rest_UV_wav_lims, fesc_author_year, single_iter=True
                )
                if np.isnan(fesc):
                    return np.nan, {}
            else:
                fesc_property_name = self._calc_property(
                    Photometry_rest.calc_fesc_from_beta_phot,
                    iters=iters,
                    rest_UV_wav_lims=rest_UV_wav_lims,
                    fesc_author_year=fesc_author_year,
                    save_path=save_path,
                )[1][0]
                fesc_chain = self.property_PDFs[fesc_property_name]
                if type(fesc_chain) == type(None):
                    return [None], [property_name]
                else:
                    fesc_chain = fesc_chain.input_arr
        elif "fesc=" in fesc_author_year:
            if single_iter:
                fesc = np.array(float(fesc_author_year.split("=")[-1]))
            else:
                fesc_chain = np.full(
                    iters, float(fesc_author_year.split("=")[-1])
                )
        else:
            raise NotImplementedError
        if single_iter:
            line_lum, line_kwargs = self.calc_line_lum_rest_optical(
                strong_line_names,
                frame,
                dust_author_year,
                dust_law,
                dust_origin,
                rest_optical_wavs,
                rest_UV_wav_lims,
                ref_wav,
                single_iter=True,
            )
            L_UV, L_UV_kwargs = self.calc_LUV_phot(
                frame,
                rest_UV_wav_lims,
                ref_wav,
                dust_author_year,
                single_iter=True,
            )
            if any(np.isnan(name) for name in [line_lum, L_UV]):
                return np.nan, {}
            else:
                return (
                    line_lum / (1.36e-12 * u.erg * (1.0 - fesc) * L_UV)
                ).to(u.Hz / u.erg), {**line_kwargs, **L_UV_kwargs}
        else:
            line_lum_property_name = self._calc_property(
                SED_rest_property_function=Photometry_rest.calc_line_lum_rest_optical,
                iters=iters,
                strong_line_names=strong_line_names,
                frame=frame,
                dust_author_year=dust_author_year,
                dust_law=dust_law,
                dust_origin=dust_origin,
                rest_optical_wavs=rest_optical_wavs,
                rest_UV_wav_lims=rest_UV_wav_lims,
                ref_wav=ref_wav,
                save_path=save_path,
            )[1][0]
            L_UV_property_name = self._calc_property(
                Photometry_rest.calc_LUV_phot,
                iters=iters,
                frame=frame,
                rest_UV_wav_lims=rest_UV_wav_lims,
                ref_wav=ref_wav,
                dust_author_year=dust_author_year,
                save_path=save_path,
            )[1][0]
            if any(
                type(self.property_PDFs[name]) == type(None)
                for name in [line_lum_property_name, L_UV_property_name]
            ):
                return [None], [property_name]
            # calculate xi_ion from Halpha luminosity assuming some case (A/B) for ISM recombination
            out_PDF = self.property_PDFs[
                line_lum_property_name
            ].manipulate_PDF(
                property_name,
                lambda int_line_lum: (
                    int_line_lum
                    / (
                        1.36e-12
                        * u.erg
                        * (1.0 - fesc_chain)
                        * self.property_PDFs[L_UV_property_name].input_arr
                    )
                ).to(u.Hz / u.erg),
                self.property_PDFs[L_UV_property_name].kwargs,
                size=iters,
            )
            return [{"PDF": out_PDF}], [property_name]

    # Rest optical line property naming functions

    def _get_UV_dust_label(self, dust_author_year: Union[str, None]):
        if type(dust_author_year) == type(None):
            return ""
        else:
            return f"_{dust_author_year}"

    def _get_dust_corr_label(
        self,
        dust_author_year: Union[str, None],
        dust_law: Union[str, None],
        dust_origin: str,
    ):
        if dust_origin != "UV":
            raise NotImplementedError
        if any(
            type(dust_name) == type(None)
            for dust_name in [dust_author_year, dust_law]
        ):
            return ""
        else:
            return f"{self._get_UV_dust_label(dust_author_year)}_{dust_law}"

    def _get_rest_optical_flux_contam_label(
        self, line_names: list, flux_contamination_params: dict
    ):
        assert all(
            line_name in line_diagnostics.keys() for line_name in line_names
        )
        assert type(flux_contamination_params) == dict
        flux_cont_keys = flux_contamination_params.keys()
        if "mu" in flux_cont_keys and "sigma" in flux_cont_keys:
            return f"{line_names[0]}_cont_G({flux_contamination_params['mu']:.1f},{flux_contamination_params['sigma']:.1f})"  # _{'+'.join(line_names[1:])}"
        elif "mu" in flux_cont_keys and "sigma" not in flux_cont_keys:
            return f"{line_names[0]}_cont_{flux_contamination_params['mu']:.1f}"  # _{'+'.join(line_names[1:])}"
        elif len(flux_contamination_params) == 0:
            return "+".join(line_names)
        else:
            raise NotImplementedError

    def _get_rest_optical_flux_contam_scaling(
        self, flux_contamination_params: dict, iters: int
    ):
        assert type(flux_contamination_params) == dict
        flux_cont_keys = flux_contamination_params.keys()
        if "mu" in flux_cont_keys and "sigma" in flux_cont_keys:
            return np.random.normal(
                1.0 - flux_contamination_params["mu"],
                flux_contamination_params["sigma"],
                iters,
            )
        elif "mu" in flux_cont_keys and "sigma" not in flux_cont_keys:
            return 1.0 - flux_contamination_params["mu"]
        elif len(flux_contamination_params) == 0:
            return 1.0
        else:
            raise NotImplementedError

    def _get_wav_line_precision(self, line_name: str, dz: float):
        assert line_name in line_diagnostics.keys()
        wav_rest = line_diagnostics[line_name]["line_wav"]
        dlambda = dz * wav_rest / (1.0 + self.z)
        return dlambda

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
        if type(self.property_PDFs[property_name]) != type(None):
            self.property_PDFs[property_name].save_PDF(save_path)

    def _update_properties_from_PDF(self, property_name):
        if type(self.property_PDFs[property_name]) == type(None):
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

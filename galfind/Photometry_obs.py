#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:03:20 2023

@author: austind
"""

# Photometry_obs.py
import time

import astropy.units as u
import matplotlib.patheffects as pe
import numpy as np
from tqdm import tqdm

from .Photometry import Photometry
from .SED_result import Catalogue_SED_results, Galaxy_SED_results


class Photometry_obs(Photometry):
    def __init__(
        self,
        instrument,
        flux_Jy,
        flux_Jy_errs,
        aper_diam,
        min_flux_pc_err,
        loc_depths,
        SED_results={},
        timed=False,
    ):
        if timed:
            start = time.time()
        self.aper_diam = aper_diam
        self.min_flux_pc_err = min_flux_pc_err
        self.SED_results = SED_results  # array of SED_result objects with different SED fitting runs
        if timed:
            mid = time.time()
        self.aper_corrs = instrument.get_aper_corrs(self.aper_diam)
        if timed:
            mid_end = time.time()
        super().__init__(instrument, flux_Jy, flux_Jy_errs, loc_depths)
        if timed:
            end = time.time()
            print(mid - start, mid_end - mid, end - mid_end)

    def __str__(self):
        line_sep = "*" * 40 + "\n"
        band_sep = "-" * 10 + "\n"
        output_str = line_sep
        output_str += "PHOTOMETRY OBS:\n"
        output_str += band_sep
        output_str += f"APERTURE DIAMETER: {self.aper_diam}\n"
        output_str += f"MIN FLUX PC ERR: {self.min_flux_pc_err}%\n"
        output_str += super().__str__(print_cls_name=False)
        for result in self.SED_results.values():
            output_str += str(result)
        output_str += f"SNR: {[np.round(snr, 2) for snr in self.SNR]}\n"
        output_str += line_sep
        return output_str

    @property
    def SNR(self):
        return [
            (flux_Jy * 10 ** (aper_corr / -2.5)) * 5 / depth
            if flux_Jy > 0.0
            else flux_Jy * 5 / depth
            for aper_corr, flux_Jy, depth in zip(
                self.aper_corrs,
                self.flux_Jy.filled(fill_value=np.nan).to(u.Jy).value,
                self.depths.to(u.Jy).value,
            )
        ]

    @classmethod  # not a gal object here, more like a catalogue row
    def from_fits_cat(
        cls,
        fits_cat_row,
        instrument,
        cat_creator,
        aper_diam,
        min_flux_pc_err,
        codes,
        lowz_zmaxs,
        templates,
    ):
        phot = Photometry.from_fits_cat(fits_cat_row, instrument, cat_creator)
        SED_results = Galaxy_SED_results.from_fits_cat(
            fits_cat_row,
            cat_creator,
            codes,
            lowz_zmaxs,
            templates,
            instrument=instrument,
        )
        return cls.from_phot(phot, aper_diam, min_flux_pc_err, SED_results)

    @classmethod
    def from_phot(cls, phot, aper_diam, min_flux_pc_err, SED_results={}):
        return cls(
            phot.instrument,
            phot.flux_Jy,
            phot.flux_Jy_errs,
            aper_diam,
            min_flux_pc_err,
            phot.loc_depths,
            SED_results,
        )

    def update(self, gal_SED_results):
        if hasattr(self, "SED_results"):
            self.SED_results = {**self.SED_results, **gal_SED_results}
        else:
            self.SED_results = gal_SED_results

    def update_mask(self, mask, update_phot_rest=False):
        assert len(self.flux_Jy) == len(mask)
        assert len(self.flux_Jy_errs) == len(mask)
        self.flux_Jy.mask = mask
        self.flux_Jy_errs.mask = mask
        return self

    def get_SED_fit_params_arr(self, code):
        return [
            code.SED_fit_params_from_label(label)
            for label in self.SED_results.keys()
        ]

    def plot_phot(
        self,
        ax,
        wav_units=u.AA,
        mag_units=u.Jy,
        plot_errs={"x": True, "y": True},
        annotate=True,
        uplim_sigma=2.0,
        auto_scale=True,
        label_SNRs=True,
        errorbar_kwargs={
            "ls": "",
            "marker": "o",
            "ms": 4.0,
            "zorder": 100.0,
            "path_effects": [pe.withStroke(linewidth=2.0, foreground="white")],
        },
        filled=True,
        colour="black",
        label="Photometry",
    ):
        plot, wavs_to_plot, mags_to_plot, yerr, uplims = super().plot_phot(
            ax,
            wav_units,
            mag_units,
            plot_errs,
            annotate,
            uplim_sigma,
            auto_scale,
            errorbar_kwargs,
            filled,
            colour,
            label,
            return_extra=True,
        )

        if label_SNRs:
            label_kwargs = {
                "ha": "center",
                "fontsize": "medium",
                "path_effects": [
                    pe.withStroke(linewidth=2.0, foreground="white")
                ],
                "zorder": 1_000.0,
            }
            label_func = (
                lambda SNR: f"{SNR:.1f}" + r"$\sigma$"
                if SNR < 100
                else f"{SNR:.0f}" + r"$\sigma$"
            )
            if mag_units == u.ABmag:
                offset = 0.15
                [
                    ax.annotate(
                        label_func(SNR),
                        (
                            wav,
                            mag - offset
                            if is_uplim
                            else mag + mag_u1 + offset,
                        ),
                        **label_kwargs,
                    )
                    for i, (
                        SNR,
                        wav,
                        mag,
                        mag_l1,
                        mag_u1,
                        is_uplim,
                    ) in enumerate(
                        zip(
                            self.SNR,
                            wavs_to_plot,
                            mags_to_plot,
                            yerr[0],
                            yerr[1],
                            uplims,
                        )
                    )
                ]
            else:
                offset = {
                    "power density/spectral flux density wav": 0.1,
                    "ABmag/spectral flux density": 0.1,
                    "spectral flux density": 0.1,
                }[str(u.get_physical_type(mag_units))]
                [
                    ax.annotate(
                        label_func(SNR),
                        (
                            wav,
                            mag + offset
                            if is_uplim
                            else mag - mag_l1 - offset,
                        ),
                        **label_kwargs,
                    )
                    for i, (
                        SNR,
                        wav,
                        mag,
                        mag_l1,
                        mag_u1,
                        is_uplim,
                    ) in enumerate(
                        zip(
                            self.SNR,
                            wavs_to_plot,
                            mags_to_plot,
                            yerr[0],
                            yerr[1],
                            uplims,
                        )
                    )
                ]

        if annotate:
            # x/y labels etc here
            ax.legend()

        return plot

    # def load_local_depths(self, sex_cat_row, instrument, aper_diam_index):
    #    self.loc_depths = np.array([sex_cat_row[f"loc_depth_{band}"].T[aper_diam_index] for band in instrument.band_names])

    # def SNR_crop(self, band, sigma_detect_thresh):
    #     index = self.instrument.band_from_index(band)
    #     # local depth in units of Jy
    #     loc_depth_Jy = self.loc_depths[index].to(u.Jy) / 5
    #     detection_Jy = self.flux_Jy[index].to(u.Jy)
    #     sigma_detection = (detection_Jy / loc_depth_Jy).value
    #     if sigma_detection >= sigma_detect_thresh:
    #         return True
    #     else:
    #         return False


# %%


class Multiple_Photometry_obs:
    def __init__(
        self,
        instrument_arr,
        flux_Jy_arr,
        flux_Jy_errs_arr,
        aper_diam,
        min_flux_pc_err,
        loc_depths_arr,
        SED_results_arr=[],
        timed=True,
    ):
        # force SED_results_arr to have the same len as the number of input fluxes
        if SED_results_arr == []:
            SED_results_arr = np.full(len(flux_Jy_arr), {})
        if timed:
            self.phot_obs_arr = [
                Photometry_obs(
                    instrument,
                    flux_Jy,
                    flux_Jy_errs,
                    aper_diam,
                    min_flux_pc_err,
                    loc_depths,
                    SED_results,
                )
                for instrument, flux_Jy, flux_Jy_errs, loc_depths, SED_results in tqdm(
                    zip(
                        instrument_arr,
                        flux_Jy_arr,
                        flux_Jy_errs_arr,
                        loc_depths_arr,
                        SED_results_arr,
                    ),
                    desc="Initializing Multiple_Photometry_obs",
                    total=len(instrument_arr),
                )
            ]
        else:
            self.phot_obs_arr = [
                Photometry_obs(
                    instrument,
                    flux_Jy,
                    flux_Jy_errs,
                    aper_diam,
                    min_flux_pc_err,
                    loc_depths,
                    SED_results,
                )
                for instrument, flux_Jy, flux_Jy_errs, loc_depths, SED_results in zip(
                    instrument_arr,
                    flux_Jy_arr,
                    flux_Jy_errs_arr,
                    loc_depths_arr,
                    SED_results_arr,
                )
            ]

    def __str__(self):
        # string representation of what is stored in this class
        return str(self.__dict__)

    def __len__(self):
        return len(self.phot_obs_arr)

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter > len(self) - 1:
            raise StopIteration
        else:
            phot = self[self.iter]
            self.iter += 1
            return phot

    def __getitem__(self, index):
        return self.phot_obs_arr[index]

    @classmethod
    def from_fits_cat(
        cls, fits_cat, instrument, cat_creator, SED_fit_params_arr, timed=False
    ):
        flux_Jy_arr, flux_Jy_errs_arr, gal_band_mask = (
            cat_creator.load_photometry(
                fits_cat, instrument.band_names, timed=timed
            )
        )
        depths_arr = cat_creator.load_depths(
            fits_cat, instrument.band_names, gal_band_mask, timed=timed
        )
        instrument_arr = cat_creator.load_instruments(
            instrument, gal_band_mask
        )
        if SED_fit_params_arr != [{}]:
            SED_results_arr = Catalogue_SED_results.from_fits_cat(
                fits_cat,
                cat_creator,
                SED_fit_params_arr,
                instrument=instrument,
            ).SED_results
        else:
            SED_results_arr = np.full(len(flux_Jy_arr), {})
        return cls(
            instrument_arr,
            flux_Jy_arr,
            flux_Jy_errs_arr,
            cat_creator.aper_diam,
            cat_creator.min_flux_pc_err,
            depths_arr,
            SED_results_arr,
            timed=timed,
        )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:11:23 2023

@author: austind
"""

from __future__ import annotations

import os
import sys
import time
from copy import deepcopy
from pathlib import Path
import astropy.units as u
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.utils.masked import Masked
from astropy.visualization import (
    ImageNormalize,
    LinearStretch,
    LogStretch,
    ManualInterval,
)
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from tqdm import tqdm
from typing import  Union, Callable, Tuple, List, NoReturn, Optional, Dict, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from . import Filter, SED_code, Selector, Band_Data, Band_Cutout, Multiple_Filter, SED_code, Mask_Selector
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import (
    PDF,
    Data,
    Instrument,
    Multiple_Photometry_obs,
    Photometry_obs,
    astropy_cosmo,
    config,
    galfind_logger,
    instr_to_name_dict,
)
from . import useful_funcs_austind as funcs
from .Cutout import RGB, Multiple_Band_Cutout
from .EAZY import EAZY
from .Emission_lines import line_diagnostics
from .SED import Mock_SED_obs, Mock_SED_rest, SED_obs
from .SED_result import SED_result

# should be exhaustive
select_func_to_type = {
    "select_min_bands": "data",
    "select_min_unmasked_bands": "data",
    "select_unmasked_instrument": "data",
    "select_phot_galaxy_property_bin": "SED",
    "select_phot_galaxy_property": "SED",
    "phot_bluewards_Lya_non_detect": "phot_rest",
    "phot_redwards_Lya_detect": "phot_rest",
    "phot_Lya_band": "phot_rest",
    "phot_SNR_crop": "phot_obs",
    "select_rest_UV_line_emitters_dmag": "phot_rest",
    "select_rest_UV_line_emitters_sigma": "phot_rest",
    "select_colour": "phot_obs",
    "select_colour_colour": "phot_obs",
    "select_UVJ": "SED",
    "select_Kokorev24_LRDs": "phot_obs",
    "select_depth_region": "data",
    "select_chi_sq_lim": "SED",
    "select_chi_sq_diff": "SED",
    "select_robust_zPDF": "SED",
    "select_band_flux_radius": "morphology",
    "select_EPOCHS": "combined",
    "select_combined": "combined",
}

class Galaxy:
    def __init__(
        self: Self, 
        ID: int, 
        sky_coord: SkyCoord, 
        aper_phot: Dict[u.Quantity, Photometry_obs],
        selection_flags: Optional[Dict[u.Quantity, Dict[str, bool]]] = None,
        cat_filterset: Optional[Multiple_Filter] = None,
        survey: Optional[str] = None,
        simulated: bool = False,
    ):
        self.ID = int(ID)
        self.sky_coord = sky_coord
        self.aper_phot = aper_phot
        if selection_flags is None:
            selection_flags = {}
        self.selection_flags = selection_flags
        self.cat_filterset = cat_filterset
        self.survey = survey
        self.simulated = simulated

    def __repr__(self):
        return f"{self.__class__.__name__}({self.ID}, " + \
            f"[{self.RA.to(u.deg).value:.5f}," + \
            f"{self.DEC.to(u.deg).value:.5f}]deg)"

    def __str__(self):
        output_str = funcs.line_sep
        output_str += f"{repr(self)}\n"
        output_str += funcs.line_sep
        output_str += "PHOTOMETRY:\n"
        output_str += funcs.band_sep
        for phot_obs in self.aper_phot.values():
            output_str += f"{repr(phot_obs)}\n"
        output_str += funcs.band_sep
        output_str += f"SELECTION FLAGS:\n"
        output_str += funcs.band_sep
        for i, (select_name, is_selected) in enumerate(self.selection_flags.items()):
            output_str += f"{select_name}: {is_selected}\n"
            if i == len(self.selection_flags) - 1:
                output_str += funcs.band_sep
        output_str += funcs.line_sep
        return output_str

    # def __setattr__(self, name, value, obj = "gal"):
    #     if obj == "gal":
    #         if type(name) != list and type(name) != np.array:
    #             super().__setattr__(name, value)
    #         else:
    #             # use setattr to set values within Galaxy dicts (e.g. properties)
    #             self.globals()[name[0]][name[1]] = value
    #     else:
    #         raise(Exception(f"obj = {obj} must be 'gal'!"))

    def __getattr__(self, property_name: str) -> Any:
        #if property_name in self.__dict__.keys():
        #    return self.__getattribute__(property_name)

        # Avoid recursion for pickling-related attributes
        if property_name in {"__getstate__", "__setstate__"}:
            raise AttributeError(property_name)

        if property_name.upper() == "RA":
           return self.sky_coord.ra.degree * u.deg
        elif property_name.upper() == "DEC":
           return self.sky_coord.dec.degree * u.deg
        if property_name in self.selection_flags:
            return self.selection_flags[property_name]
        else:
            # if property_name not in [
            #     "__array_struct__",
            #     "__array_interface__",
            #     "__array__",
            # ]:
            #     pass
                # galfind_logger.critical(
                #     f"Galaxy {self.ID=} has no {property_name=}!"
                # )
            raise AttributeError(property_name)
        #else:
        #    return self.phot.__getattr__(property_name, origin)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            try:
                setattr(result, key, deepcopy(value, memo))
            except:
                galfind_logger.critical(
                    f"deepcopy({self.__class__.__name__}) {key}: {value} FAIL!"
                )
                breakpoint()
        return result

    def update_SED_results(
        self: Self,
        gal_SED_results: Union[SED_result, List[SED_result]]
    ) -> NoReturn:
        if not isinstance(gal_SED_results, list):
            gal_SED_results = [gal_SED_results]
        missing_aper_diams = [gal_SED_result.aper_diam \
            for gal_SED_result in gal_SED_results \
            if gal_SED_result.aper_diam not in self.aper_phot.keys()]
        assert len(missing_aper_diams) == 0, \
            galfind_logger.critical(
                f"Galaxy {self.ID=} missing " + \
                f"{missing_aper_diams} aperture photometry."
            )
        [self.aper_phot[gal_SED_result.aper_diam]. \
            update_SED_result(gal_SED_result) \
            for gal_SED_result in gal_SED_results]
        
    def load_fixz_SED_result(
        self: Self,
        aper_diam: u.Quantity,
        z_value: float,
        z_label: str = "z",
    ) -> NoReturn:
        self.aper_phot[aper_diam].load_fixz_SED_result(z_value, z_label)
        

    def load_property(
        self, gal_property: Union[dict, u.Quantity], save_name: str
    ) -> None:
        setattr(self, save_name, gal_property)

    def make_RGB(
        self: Type[Self],
        data: Data,
        rgb_bands: Dict[str, List[str]] = {
            "B": ["F090W"],
            "G": ["F200W"],
            "R": ["F444W"],
        },
        cutout_size: u.Quantity = 0.96 * u.arcsec,
    ) -> RGB:
        if not hasattr(self, "RGBs"):
            self.RGBs = {}
        cutout_size_str = f"{cutout_size.to(u.arcsec).value:.2f}as"
        if cutout_size_str not in self.RGBs.keys():
            self.RGBs[cutout_size_str] = {}
        rgb_key = ",".join(
            f"{colour}={'+'.join(self.get_colour_band_names[colour])}"
            for colour in ["B", "G", "R"]
        )
        if (
            rgb_key
            not in self.RGBs[f"{cutout_size.to(u.arcsec).value:.2f}as"].keys()
        ):
            RGB_obj = RGB.from_gal(data, self, rgb_bands)
            assert RGB_obj.name == rgb_key
            self.RGBs[cutout_size_str][rgb_key] = RGB_obj
        return self.RGBs[cutout_size_str][rgb_key]

    def plot_RGB(
        self,
        ax: Optional[plt.Axes],
        rgb_bands: Dict[str, List[str]],
        cutout_size: u.Quantity = 0.96 * u.arcsec,
        method: str = "trilogy",
    ) -> NoReturn:
        cutout_size_str = f"{cutout_size.to(u.arcsec).value:.2f}as"
        rgb_key = ",".join(
            f"{colour}={'+'.join(self.get_colour_band_names[colour])}"
            for colour in ["B", "G", "R"]
        )
        RGB_obj = self.RGBs[cutout_size_str][rgb_key]
        RGB_obj.plot(ax, method)

    def make_cutouts(
        self: Type[Self], 
        data: Data, 
        cutout_size: u.Quantity = 0.96 * u.arcsec,
        overwrite: bool = False
    ) -> Multiple_Band_Cutout:
        if not hasattr(self, "multi_band_cutout"):
            self.multi_band_cutout = Multiple_Band_Cutout. \
                from_gal_data(
                    self,
                    data,
                    cutout_size,
                    overwrite = overwrite
                )
        else:
            galfind_logger.debug(
                f"{self.__class__.__name__} {self.ID=} already has cutouts!"
            )
        return self.multi_band_cutout

    def make_band_cutout(
        self: Type[Self],
        band_data: Band_Data,
        cutout_size: u.Quantity = 0.96 * u.arcsec,
    ) -> Band_Cutout:
        if not hasattr(self, "cutouts"):
            self.cutouts = {}
        cutout_ = None
        cutout_size_str = f"{cutout_size.to(u.arcsec).value:.2f}as"
        if hasattr(self, "multi_band_cutout"):
            # point to relevant cutout that has already been made
            if cutout_size_str in self.multi_band_cutout.keys() and \
                    any(band_data.filt_name == cutout.filt_name for cutout \
                    in self.multi_band_cutout[cutout_size_str]):
                cutout_ = self.multi_band_cutout[cutout_size_str][band_data.filt_name]
        if cutout_ is None:
            from . import Band_Cutout 
            # make the cutout from the relevant band_data
            cutout_ = Band_Cutout.from_gal_band_data(self, band_data, cutout_size)
        self.cutouts[f"{band_data.filt_name}_{cutout_size_str}"] = cutout_
        return cutout_

    def plot_cutouts(
        self: Type[Self],
        fig: plt.Figure,
        data: Data,
        SED_code: SED_code,
        #hide_masked_cutouts: bool = True,
        cutout_size: u.Quantity = 0.96 * u.arcsec,
        aper_diam: u.Quantity = 0.32 * u.arcsec,
        imshow_kwargs: Dict[str, Any] = {},
        norm_kwargs: Dict[str, Any] = {},
        aper_kwargs: Dict[str, Any] = {},
        kron_kwargs: Dict[str, Any] = {},
        n_rows: int = 2,
    ):
        multi_band_cutout = self.make_cutouts(data, cutout_size)

        # # make intructions for radii to plot
        # sex_radius = None
        # aper_kwargs = {
        #     "fill": False,
        #     "linestyle": "--",
        #     "lw": 1,
        #     "color": "white",
        #     "zorder": 20,
        # }
        # sex_rad_kwargs = {
        #     "fill": False,
        #     "linestyle": "--",
        #     "lw": 1,
        #     "color": "blue",
        #     "zorder": 20,
        # }
        # breakpoint()
        # if sex_radius is None:
        #     plot_radii = [
        #         [{"radius": aper_diam, "kwargs": aper_kwargs}]
        #         for cutout in multi_band_cutout
        #     ]
        # else:
        #     plot_radii = [
        #         [
        #             {"radius": aper_diam, "kwargs": aper_kwargs},
        #             {
        #                 "radius": self.sex_radius[cutout.filt.band_name],
        #                 "kwargs": sex_rad_kwargs,
        #             },
        #         ]
        #         for cutout in multi_band_cutout
        #     ]

        # # make instructions for scalebars to plot
        # physical_scalebar_kwargs = {
        #     "loc": "upper left",
        #     "pad": 0.3,
        #     "color": "white",
        #     "frameon": False,
        #     "size_vertical": 1.5,
        # }
        # angular_scalebar_kwargs = {
        #     "loc": "lower right",
        #     "pad": 0.3,
        #     "color": "white",
        #     "frameon": False,
        #     "size_vertical": 2,
        # }
        # z = self.aper_phot[aper_diam].SED_results[SED_code.label].z
        # scalebars = [
        #     {
        #         "physical": {
        #             **physical_scalebar_kwargs,
        #             "z": z,
        #             "pix_length": 10,
        #         }
        #     }
        #     if i == 0
        #     else {"angular": {**angular_scalebar_kwargs, "as_length": 0.3}}
        #     if i == len(multi_band_cutout) - 1
        #     else {}
        #     for i, cutout in enumerate(multi_band_cutout)
        # ]

        # make fixed aperture regions to plot
        aper_kwargs = {"ls": "-", "color": "green", "path_effects": [pe.withStroke(linewidth = 3., foreground = "white")]}
        fixed_apertures = np.full(len(self.aper_phot[aper_diam]), {"aper_diam": aper_diam, **aper_kwargs})
        # make kron aperture regions to plot
        if all(hasattr(self, property_name) for property_name in \
                ["sex_KRON_RADIUS", "sex_A_IMAGE", "sex_B_IMAGE", "sex_THETA_IMAGE"]):
            kron_kwargs = {"ls": "--", "color": "blue", "path_effects": [pe.withStroke(linewidth = 3., foreground = "white")]}
            kron_apertures = [
                Ellipse(
                    (-99., -99.),
                    width = (self.sex_KRON_RADIUS[filt_name] * self.sex_A_IMAGE).to(u.dimensionless_unscaled).value,
                    height = (self.sex_KRON_RADIUS[filt_name] * self.sex_B_IMAGE).to(u.dimensionless_unscaled).value,
                    angle = self.sex_THETA_IMAGE.to(u.deg).value,
                    **kron_kwargs
                )
                for filt_name in self.aper_phot[aper_diam].filterset.band_names
            ]
            plot_regions = {filt_name: [aper, kron] for filt_name, aper, kron in \
                zip(self.aper_phot[aper_diam].filterset.band_names, fixed_apertures, kron_apertures)}
        else:
            plot_regions = {filt_name: [aper] for filt_name, aper in \
                zip(self.aper_phot[aper_diam].filterset.band_names, fixed_apertures)}
        ax_arr = multi_band_cutout.plot(
            fig = fig,
            n_rows = n_rows,
            # fig_scaling: float = 1.5,
            split_by_instr = True,
            imshow_kwargs = imshow_kwargs,
            norm_kwargs = norm_kwargs,
            plot_regions = plot_regions,
            # scalebars: Optional[Dict] = [],
            # mask: Optional[List[bool]] = None,
            show = False,
            save = False,
            close_fig = False,
        )
        return ax_arr
    
    def _extract_lowz_codes(
        self: Type[Self],
        aper_diam: u.Quantity,
        SED_arr: List[SED_code],
        lowz_dz: float = 0.5
    ) -> List[SED_code]:
        lowz_codes = []
        none_zmax_codes = [code for code in SED_arr \
            if code.SED_fit_params.get("lowz_zmax", False) is None]
        for none_zmax_code in none_zmax_codes:
            none_zmax_code_SED_fit_params = deepcopy(none_zmax_code.SED_fit_params)
            lowz_zmax_codes = []
            for SED_result in self.aper_phot[aper_diam].SED_results.values():
                code = SED_result.SED_code
                if code.SED_fit_params.get("lowz_zmax", False) is None:
                    continue
                code_SED_fit_params = deepcopy(code.SED_fit_params)
                remove_keys = [key for key in code_SED_fit_params.keys() if \
                    any(substr in key.lower() for substr in ["z_max", "zmax"])]
                for key in remove_keys:
                    del code_SED_fit_params[key]
                    if key in none_zmax_code_SED_fit_params.keys():
                        del none_zmax_code_SED_fit_params[key]
                if code_SED_fit_params == none_zmax_code_SED_fit_params:
                    lowz_zmax_codes.append(code)
            z_gal = self.aper_phot[aper_diam].SED_results[none_zmax_code.label].z
            lowz_zmax = np.array(sorted([lowz_zmax_code.SED_fit_params["lowz_zmax"] \
                for lowz_zmax_code in lowz_zmax_codes]))
            if any(lowz_zmax + lowz_dz < z_gal):
                lowz_codes.extend([lowz_zmax_codes[np.where(lowz_zmax + lowz_dz < z_gal)[0][-1]]])
        return lowz_codes

    def plot_phot_diagnostic(
        self: Type[Self],
        ax: Tuple[plt.Figure, List[plt.Axes], List[plt.Axes]],
        data: Data,
        SED_arr: List[Union[str, SED_code]],
        zPDF_arr: List[Union[str, SED_code]],
        plot_lowz: bool = True,
        lowz_dz: float = 0.5,
        n_cutout_rows: int = 1,
        wav_unit = u.um,
        flux_unit = u.ABmag,
        log_fluxes: bool = False,
        #hide_masked_cutouts=True,
        cutout_size: u.Quantity = 0.96 * u.arcsec,
        #high_dyn_rng=False,
        annotate_PDFs=True,
        #plot_rejected_reasons=False,
        aper_diam: u.Quantity = 0.32 * u.arcsec,
        imshow_kwargs: Dict[str, Any] = {},
        norm_kwargs: Dict[str, Any] = {},
        aper_kwargs: Dict[str, Any] = {},
        kron_kwargs: Dict[str, Any] = {},
        overwrite: bool = False,
        save: bool = True,
        show: bool = True,
    ):
        
        out_path = f"{config['Other']['PLOT_DIR']}/{data.version}/" + \
            f"{data.filterset.instrument_name}/{data.survey}/SED_plots/" + \
            f"{aper_diam.to(u.arcsec).value:.2f}as/{self.ID}.png"
        
        # unpack tuple
        cutout_fig, phot_ax, PDF_ax = ax

        if not isinstance(SED_arr, list):
            SED_arr = [SED_arr]
        if not isinstance(zPDF_arr, list):
            zPDF_arr = [zPDF_arr]
        # extract lowz_zmax from SED_arr if required
        if plot_lowz:
            SED_arr.extend(self._extract_lowz_codes(aper_diam, SED_arr, lowz_dz))
            zPDF_arr.extend(self._extract_lowz_codes(aper_diam, zPDF_arr, lowz_dz))

        zPDF_labels = [code.label for code in zPDF_arr]
        # reset parameters
        for ax_, label in zip(PDF_ax, zPDF_labels):
            ax_.set_yticks([])
            ax_.set_xlabel("Redshift, z")
            ax_.set_title(label, fontsize="medium")

        if not Path(out_path).is_file() or overwrite:
            # plot cutouts (assuming reference SED_fit_params is at 0th index)
            self.plot_cutouts(
                cutout_fig,
                data,
                SED_arr[0],
                #hide_masked_cutouts=hide_masked_cutouts,
                cutout_size=cutout_size,
                #high_dyn_rng=high_dyn_rng,
                aper_diam = aper_diam,
                imshow_kwargs = imshow_kwargs,
                norm_kwargs = norm_kwargs,
                aper_kwargs = aper_kwargs,
                kron_kwargs = kron_kwargs,
                n_rows = n_cutout_rows,
            )

            # plot specified SEDs and save colours
            SED_colours = {}
            errorbar_kwargs = {
                "ls": "",
                "marker": "o",
                "ms": 8.0,
                "zorder": 100.0,
                "path_effects": [
                    pe.withStroke(linewidth=2.0, foreground="white")
                ],
            }
            for code in reversed(SED_arr):
                if self.aper_phot[aper_diam].SED_results[code.label].SED is not None:
                    SED_plot = self.aper_phot[aper_diam].SED_results[code.label].SED.plot(
                        phot_ax,
                        wav_unit,
                        flux_unit,
                        log_fluxes = log_fluxes,
                        label = code.label,
                    )
                    SED_colours[code.label] = SED_plot[0].get_color()
                    # plot the mock photometry
                    self.aper_phot[aper_diam].SED_results[code.label].SED.create_mock_photometry(
                        self.aper_phot[aper_diam].filterset,
                        depths = self.aper_phot[aper_diam].depths,
                        # min flux pc err = 10.0
                    )
                    self.aper_phot[aper_diam].SED_results[code.label].SED.mock_photometry.plot(
                        phot_ax,
                        wav_unit,
                        flux_unit,
                        uplim_sigma=None,
                        auto_scale=False,
                        plot_errs={"x": False, "y": False},
                        errorbar_kwargs=errorbar_kwargs,
                        label=None,
                        filled=False,
                        colour=SED_colours[code.label],
                        log_scale = log_fluxes,
                    )
                    # ax_photo.scatter(band_wavs_lowz, band_mags_lowz, edgecolors=eazy_color_lowz, marker='o', facecolor='none', s=80, zorder=4.5)
            
            self.aper_phot[aper_diam].plot(
                phot_ax,
                wav_unit,
                flux_unit,
                annotate = False,
                # uplim_sigma = 2.0,
                # auto_scale = True,
                # label_SNRs = True,
                # errorbar_kwargs = {
                #     "ls": "",
                #     "marker": "o",
                #     "ms": 4.0,
                #     "zorder": 100.0,
                #     "path_effects": [pe.withStroke(linewidth=2.0, foreground="white")],
                # },
                # filled = True,
                # colour = "black",
                # label = "Photometry",
                SNR_labelsize = 7.5,
                log_scale = log_fluxes,
            )
            # photometry axis title
            phot_ax.set_title(f"{data.survey} {self.ID} ({data.version})")
            # plot rejected reasons somewhere
            # if plot_rejected_reasons:
            #     rejected = str(row[f'rejected_reasons{col_ext}'][0])
            #     if rejected != '':
            #         phot_ax.annotate(rejected, (0.9, 0.95), ha='center', fontsize='small', xycoords = 'axes fraction', zorder=5)
            # photometry axis legend
            phot_ax.legend(loc="best", fontsize=8.0, frameon=True)
            for text in phot_ax.get_legend().get_texts():
                text.set_path_effects(
                    [pe.withStroke(linewidth=3, foreground="white")]
                )
                text.set_zorder(12)

            # plot PDF on relevant axis
            # again, this is not totally generalized and should be == 2 for now
            #assert (len(zPDF_arr) == len(PDF_ax))
            # could extend to plotting multiple PDFs on the same axis
            for ax, code in zip(PDF_ax, zPDF_arr):
                if code.label in SED_colours.keys():
                    colour = SED_colours[code.label]
                else:
                    colour = "black"
                # load peak value/chi sq from SED_result into redshift PDF 
                # TODO: Load this into the PDF when instantiated from SED_result
                self.aper_phot[aper_diam].SED_results[code.label].property_PDFs["z"]. \
                    load_peaks_from_SED_result(self.aper_phot[aper_diam].SED_results[code.label])
                # plot the PDF
                self.aper_phot[aper_diam].SED_results[code.label].property_PDFs["z"].plot(
                    ax, annotate=annotate_PDFs, colour=colour
                )

            if save:
                funcs.make_dirs(out_path)
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                funcs.change_file_permissions(out_path)
            if show:
                plt.show()
            else:
                for ax in [phot_ax] + PDF_ax:
                    ax.clear()
        return out_path

    # Spectroscopy

    def load_spectra(self, spectra):
        self.spectra = spectra
        return self

    def plot_spec_diagnostic(
        self, ax, grating_filter="PRISM/CLEAR", overwrite=True
    ):
        # bare in mind that not all galaxies have spectroscopic data
        if hasattr(self, "spectra"):
            # plot spectral diagnostic
            pass
        else:
            pass

    # %% Selection methods

    # def is_selected(
    #     self,
    #     selectors: List[Type[Selector]],
    #     aper_diam: u.Quantity,
    #     SED_fit_label: str,
    #     timed: bool = False,
    # ) -> bool:
    #     # # input assertions
    #     # assert type(crop_names) in [str, np.array, list, dict]
    #     # if type(crop_names) in [str]:
    #     #     crop_names = [crop_names]
    #     # if type(incl_selection_types) in [str]:
    #     #     incl_selection_types = [incl_selection_types]
    #     # if incl_selection_types != ["All"]:
    #     #     assert all(
    #     #         fit_type in select_func_to_type.values()
    #     #         for fit_type in incl_selection_types
    #     #     )
    #     # # perform selections if required
    #     # selection_names = []
    #     # if timed:
    #     #     start = time.time()
    #     # for i, crop_name in enumerate(crop_names):
    #     #     func, kwargs, func_type = (
    #     #         Galaxy._get_selection_func_from_output_name(
    #     #             crop_name, SED_fit_params
    #     #         )
    #     #     )
    #     #     if func_type in incl_selection_types or incl_selection_types == [
    #     #         "All"
    #     #     ]:
    #     #         selection_name = func(self, **kwargs)[1]
    #     #         if selection_name != crop_name:
    #     #             breakpoint()  # see what's wrong
    #     #             assert selection_name == crop_name  # break code
    #     #         selection_names.append(selection_name)
    #     #         run = True
    #     #     else:
    #     #         run = False
    #     #     if timed:
    #     #         print(
    #     #             f"{crop_name=} was {'run' if run else 'skipped'} and took:"
    #     #         )
    #     #         if i == 0:
    #     #             mid = time.time()
    #     #             print(f"{mid - start}s")
    #     #         else:
    #     #             print(f"{time.time() - mid}s")
    #     #             mid = time.time()
    #     # breakpoint()
    #     breakpoint()
    #     [selector(self, aper_diam, SED_fit_label, return_copy = False) for selector in selectors]
    #     # determine whether galaxy is selected
    #     if all(self.selection_flags[name] for name in selectors):
    #         selected = True
    #     else:
    #         selected = False
    #     # clear selection flags
    #     # self.selection_flags = {}
    #     return selected

    # @staticmethod
    # def _get_selection_func_from_output_name(
    #     name: str, SED_fit_params: Union[str, dict]
    # ):
    #     # only currently works for standard EPOCHS selection!
    #     if type(SED_fit_params) in [str]:
    #         SED_fit_params_key = SED_fit_params
    #         SED_fit_params = globals()[
    #             SED_fit_params_key.split("_")[0]
    #         ]().SED_fit_params_from_label(SED_fit_params_key)
    #         galfind_logger.warning(
    #             f"Galaxy._get_selection_func_from_output_name faster with {type(SED_fit_params)=} == 'dict'"
    #         )
    #     else:
    #         SED_fit_params_key = SED_fit_params[
    #             "code"
    #         ].label_from_SED_fit_params(SED_fit_params)
    #     # simplest case
    #     if hasattr(Galaxy, f"select_{name}"):
    #         func = getattr(Galaxy, f"select_{name}")
    #         # default arguments
    #         kwargs = {}
    #     # phot_SNR_crop
    #     elif "bluest_band_SNR" in name or "reddest_band_SNR" in name:
    #         # not yet exhaustive! - have not done band SNR cropping
    #         func = Galaxy.phot_SNR_crop
    #         split_name = name.split("_band_SNR")
    #         sign = split_name[1][0]
    #         assert sign in ["<", ">"]
    #         detect_or_non_detect = "detect" if sign == ">" else "non_detect"
    #         SNR_lim = float(name.split(sign)[1])
    #         if split_name[0] == "bluest":
    #             band_name_or_index = 0
    #         elif split_name[0] == "reddest":
    #             band_name_or_index = -1
    #         else:
    #             band_name_or_index = int(split_name[0].split("_")[0][:-2])
    #             if "bluest" in split_name[0]:
    #                 band_name_or_index -= 1
    #             else:  # reddest in split_name[0]
    #                 band_name_or_index *= -1
    #         kwargs = {
    #             "band_name_or_index": band_name_or_index,
    #             "SNR_lim": SNR_lim,
    #             "detect_or_non_detect": detect_or_non_detect,
    #         }
    #     # phot_bluewards_Lya_non_detect
    #     elif "bluewards_Lya_SNR<" in name:
    #         func = Galaxy.phot_bluewards_Lya_non_detect
    #         kwargs = {"SNR_lim": float(name.split("bluewards_Lya_SNR<")[1])}
    #     # phot_redwards_Lya_detect
    #     elif "redwards_Lya_SNR>" in name:
    #         func = Galaxy.phot_redwards_Lya_detect
    #         SNR_str = name.split(">")[1].split("_")
    #         if name.split("_")[0] == "ALL":
    #             SNR_lims = float(SNR_str[0])
    #         else:  # name.split("_") == "redwards"
    #             SNR_lims = [float(SNR) for SNR in SNR_str[0].split(",")]
    #         if len(SNR_str) == 1:
    #             widebands_only = False
    #         else:
    #             assert len(SNR_str) == 2
    #             assert SNR_str[1] == "widebands"
    #             widebands_only = True
    #         kwargs = {"SNR_lims": SNR_lims, "widebands_only": widebands_only}
    #     # select_chi_sq_lim
    #     elif "red_chi_sq<" in name:
    #         func = Galaxy.select_chi_sq_lim
    #         kwargs = {"chi_sq_lim": float(name.split("red_chi_sq<")[1])}
    #     # select_chi_sq_diff
    #     elif "chi_sq_diff" in name and ",dz>" in name:
    #         func = Galaxy.select_chi_sq_diff
    #         split_name = name.split(",dz>")
    #         kwargs = {
    #             "chi_sq_diff": float(split_name[0].split(">")[1]),
    #             "delta_z_low_z": float(split_name[1]),
    #         }
    #     # select_robust_zPDF
    #     elif "zPDF>" in name and "%,|dz|/z<" in name:
    #         func = Galaxy.select_robust_zPDF
    #         split_name = name.split("%,|dz|/z<")
    #         kwargs = {
    #             "integral_lim": int(split_name[0].split(">")[1]) / 100,
    #             "delta_z_over_z": float(split_name[1]),
    #         }
    #     # select_min_unmasked_bands
    #     elif "unmasked_bands>" in name:
    #         func = Galaxy.select_min_unmasked_bands
    #         kwargs = {"min_bands": int(name.split(">")[1]) + 1}
    #     # select_unmasked_instrument
    #     elif "unmasked_" in name:
    #         func = Galaxy.select_unmasked_instrument
    #         instrument_name = name.split("_")[1]
    #         assert instrument_name in [
    #             subcls.__name__
    #             for subcls in Instrument.__subclasses__()
    #             if subcls.__name__ != "Combined_Instrument"
    #         ]
    #         kwargs = {"instrument": instr_to_name_dict[instrument_name]}
    #     # select_band_flux_radius
    #     elif "Re_" in name:
    #         func = Galaxy.select_band_flux_radius
    #         split_name = name.split(">")
    #         if len(split_name) == 1:
    #             # no ">" exists
    #             split_name = name.split("<")
    #             gtr_or_less = "less"
    #         else:  # ">" exists
    #             gtr_or_less = "gtr"
    #         band = split_name[0].split("_")[1]
    #         lim_str = split_name[1]
    #         if "pix" in lim_str:
    #             lim = float(lim_str.split("pix")[0]) * u.dimensionless_unscaled
    #         else:  # lim in as
    #             lim = float(lim_str.split("as")[0]) * u.arcsec
    #         kwargs = {"band": band, "gtr_or_less": gtr_or_less, "lim": lim}
    #     else:
    #         galfind_logger.critical(
    #             f"Galaxy._get_selection_func_from_output_name could not determine {name=}!"
    #         )
    #         raise NotImplementedError
    #     assert func.__name__ in select_func_to_type.keys()
    #     func_type = select_func_to_type[func.__name__]
    #     if func_type in ["SED", "phot_rest", "combined"]:
    #         kwargs["SED_fit_params"] = SED_fit_params
    #     return func, kwargs, func_type

    # Rest-frame SED photometric properties

    # def _calc_SED_rest_property(
    #     self,
    #     SED_rest_property_function,
    #     SED_fit_params_label,
    #     save_dir,
    #     iters,
    #     **kwargs,
    # ):
    #     phot_rest_obj = self.phot.SED_results[SED_fit_params_label].phot_rest
    #     if type(save_dir) == type(None):
    #         save_path = None
    #     else:
    #         save_path = f"{save_dir}/{SED_fit_params_label}/property_name/{self.ID}.ecsv"
    #     phot_rest_obj._calc_property(
    #         SED_rest_property_function, iters, save_path=save_path, **kwargs
    #     )[1]
    #     return self

    # def _load_SED_rest_properties(
    #     self,
    #     PDF_dir,
    #     property_names,
    #     SED_fit_params_label=EAZY({"templates": "fsps_larson", "lowz_zmax": None}).label,
    # ):
    #     # determine which properties have already been calculated
    #     property_names_to_load = [
    #         property_name
    #         for property_name in property_names
    #         if Path(f"{PDF_dir}/{property_name}/{self.ID}.ecsv").is_file()
    #     ]
    #     PDF_paths = [
    #         f"{PDF_dir}/{property_name}/{self.ID}.ecsv"
    #         for property_name in property_names_to_load
    #     ]
    #     for PDF_path, property_name in zip(PDF_paths, property_names_to_load):
    #         self.phot.SED_results[
    #             SED_fit_params_label
    #         ].phot_rest.property_PDFs[property_name] = PDF.from_ecsv(PDF_path)
    #         self.phot.SED_results[
    #             SED_fit_params_label
    #         ].phot_rest._update_properties_from_PDF(property_name)
    #     return self

    # def _del_SED_rest_properties(
    #     self,
    #     property_names,
    #     SED_fit_params_label=EAZY({"templates": "fsps_larson", "lowz_zmax": None}).label,
    # ):
    #     for property_name in property_names:
    #         self.phot.SED_results[
    #             SED_fit_params_label
    #         ].phot_rest.property_PDFs.pop(property_name)
    #         self.phot.SED_results[
    #             SED_fit_params_label
    #         ].phot_rest.properties.pop(property_name)
    #         self.phot.SED_results[
    #             SED_fit_params_label
    #         ].phot_rest.property_errs.pop(property_name)
    #     return self

    # def _get_SED_rest_property_names(self, PDF_dir):
    #     PDF_paths = glob.glob(f"{PDF_dir}/*/{self.ID}.ecsv")
    #     return [path.split("/")[-2] for path in PDF_paths]

    def load_sextractor_ext_src_corrs(
        self: Self, 
        aper_corrs: Optional[Dict[str, Dict[u.Quantity, float]]] = None
    ) -> NoReturn:
        # FLUX_AUTO must already be loaded
        if not hasattr(self, "sex_FLUX_AUTO"):
            galfind_logger.critical(
                f"Galaxy {self.ID=} has no {self.FLUX_AUTO=}!"
            )
            raise AttributeError("sex_FLUX_AUTO")
        # load ext_src_corrs into aper_phot
        for aper_diam in self.aper_phot.keys():
            aper_diam_aper_corrs = {key: val[aper_diam] for key, val in aper_corrs.items()}
            self.aper_phot[aper_diam].load_sextractor_ext_src_corrs(aper_diam_aper_corrs)

    def _make_Vmax_storage(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_code: SED_code,
        crop_name: str,
    ) -> NoReturn:
        SED_result_obj = self.aper_phot[aper_diam].SED_results[SED_fit_code.label]
        # name appropriate empty output dicts if not already made
        if not hasattr(SED_result_obj, "obs_zrange"):
            SED_result_obj.obs_zrange = {}
        if crop_name not in SED_result_obj.obs_zrange.keys():
            SED_result_obj.obs_zrange[crop_name] = {}
        # if not hasattr(self, "V_max_simple"):
        #    self.V_max_simple = {}
        if not hasattr(SED_result_obj, "V_max"):
            SED_result_obj.V_max = { }
        if crop_name not in SED_result_obj.V_max.keys():
            SED_result_obj.V_max[crop_name] = {}

    # Vmax calculation in a single field
    def calc_Vmax(
        self: Self,
        data: Data,
        z_bin: List[float],
        aper_diam: u.Quantity,
        SED_fit_code: SED_code,
        crops: List[Type[Selector]],
        z_step: float = 0.01,
        depth_mode: str = "n_nearest",
        unmasked_area: Union[str, List[str], u.Quantity, Type[Mask_Selector]] = "selection",
    ) -> float:
        # TODO: remove dependence on full_survey_name input
        # input assertions
        assert len(z_bin) == 2
        assert z_bin[0] < z_bin[1]
        assert crops != []
        from . import (
            Data_Selector, 
            SED_fit_Selector, 
            Multiple_Selector, 
            Rest_Frame_Property_Limit_Selector, 
        )
        breakpoint()
        SED_result_obj = self.aper_phot[aper_diam].SED_results[SED_fit_code.label]
        crop_name = funcs.get_crop_name(crops).split("/")[-1]
        z_obs = SED_result_obj.z

        # return V_max if already calculated
        self._make_Vmax_storage(aper_diam, SED_fit_code, crop_name)
        if data.full_name in SED_result_obj.V_max[crop_name].keys():
            return SED_result_obj.V_max[crop_name][data.full_name]

        # flatten multiple selectors and remove 
        # SED_fit_selectors that require SED fitting from crops
        multiple_selectors = tuple(Multiple_Selector.__subclasses__())
        while any(isinstance(crop, multiple_selectors) for crop in crops):
            crops_ = []
            for crop in crops:
                if isinstance(crop, multiple_selectors):
                    crops_.extend([i for i in crop])
                else:
                    crops_.extend([crop])
            crops = crops_
        # TODO: generalize this! i.e. re-perform new morphological fits 
        # and SED fitting with redshifted photometry
        Vmax_crops = []
        for crop in crops:
            # skip all selection based on data and morphology
            if isinstance(crop, tuple(Data_Selector.__subclasses__())):
                continue
            # skip all selection requiring SED fitting
            elif isinstance(crop, tuple(SED_fit_Selector.__subclasses__())):
                if crop.requires_SED_fit or isinstance(crop, Rest_Frame_Property_Limit_Selector):
                    continue
            Vmax_crops.extend([crop])
        assert Vmax_crops != []

        # calculate V_max
        V_max_dict = {}
        if z_obs > z_bin[1] or z_obs < z_bin[0]:
            # V_max_simple = -1.
            region = "all"
            V_max = -1.0
            z_min_used = -1.0
            z_max_used = -1.0
            V_max_dict[region] = V_max
        else:
            distance_detect = astropy_cosmo.luminosity_distance(z_obs)
            sed_obs = SED_result_obj.SED
            # load appropriate depths for each data object in data_arr
            galfind_logger.debug(
                "Should use local depth if the data.full_name " + \
                "is the same as the catalogue the galaxy is measured in!"
            )
            if hasattr(data, "regions"):
                galfind_logger.debug(
                    f"Calculating split Vmax for {repr(self)} in {data.regions=}"
                )
                regions = data.regions
            else:
                galfind_logger.debug(
                    f"Calculating Vmax for {repr(self)} in {data.full_name=}"
                )
                regions = ["all"]
            for region in regions:
                assert all(hasattr(band_data, "med_depth") for band_data in data), \
                    galfind_logger.critical(
                        f"Not all bands in {repr(data)=} have med_depth attributes!"
                    )
                assert all(region in band_data.med_depth[aper_diam].keys() for band_data in data), \
                    galfind_logger.critical(
                        f"Not all bands in {repr(data)=} have med_depths for {region=}"
                    )
                # TODO: make split regions available on a data level without having to pre-load results
                #data._load_depths(aper_diam, depth_mode, region)
                data_depths = [band_data.med_depth[aper_diam][region] for band_data in data]
                # calculate z_range
                # z_test for other fields should be lower than starting z
                z_detect = []
                for z in np.arange(z_bin[0], z_bin[1] + z_step, z_step): #, tqdm(
                #     np.arange(z_bin[0], z_bin[1] + z_step, z_step),
                #     desc=f"Calculating z_max and z_min for ID={self.ID}",
                # ):
                    galfind_logger.debug(
                        "ΙGM attenuation is ignored when redshifting the best-fit galaxy SED!"
                    )
                    wav_z = sed_obs.wavs * ((1.0 + z) / (1.0 + sed_obs.z))
                    distance_test = astropy_cosmo.luminosity_distance(z)
                    mag_z = (
                        funcs.convert_mag_units(
                            sed_obs.wavs, sed_obs.mags, u.ABmag
                        ).value
                        - (
                            5
                            * np.log10(
                                (distance_detect / distance_test)
                                .to(u.dimensionless_unscaled)
                                .value
                            )
                        )
                        + (2.5 * np.log10((1.0 + sed_obs.z) / (1.0 + z)))
                    )
                    # construct galaxy at new redshift with average depths of new field
                    test_sed_obs = SED_obs(
                        z, wav_z.value, mag_z, wav_z.unit, u.ABmag
                    )
                    galfind_logger.debug("Not propagating min_flux_pc_err!")
                    test_mock_phot = test_sed_obs.create_mock_photometry(
                        data.filterset,
                        depths=data_depths,
                        min_flux_pc_err=10.0,
                    )
                    test_phot_obs = Photometry_obs(
                        test_mock_phot.filterset,
                        Masked(
                            test_mock_phot.flux,
                            mask=np.full(len(data.filterset), False),
                        ),
                        Masked(
                            test_mock_phot.flux_errs,
                            mask=np.full(len(data.filterset), False),
                        ),
                        test_mock_phot.depths,
                        aper_diam,
                    )
                    sed_result = SED_result(
                        SED_fit_code,
                        test_phot_obs,
                        {"z": z},
                        property_errs={},
                        property_PDFs={},
                        SED=None,
                    )
                    test_phot_obs.SED_results = {
                        SED_fit_code.label: sed_result
                    }
                    test_gal = Galaxy(
                        self.ID,
                        self.sky_coord,
                        {aper_diam: test_phot_obs},
                        selection_flags={},
                    )
                    # run selection methods on new galaxy
                    [selector(test_gal, return_copy = False) for selector in Vmax_crops]
                    goodz = all(
                        test_gal.selection_flags[selector.name]
                        for selector in Vmax_crops
                    )
                    if goodz:
                        z_detect.append(z)

                if len(z_detect) < 2:
                    z_max = -1.0
                    z_min = -1.0
                    # V_max_simple = -1.
                    V_max = -1.0
                else:
                    z_max = np.round(z_detect[-1], 3)
                    z_min = np.round(z_detect[0], 3)

                # don't worry about completeness and contamination for now
                # completeness_field = detect_completeness[field] * jag_completeness[field]
                # contamination_field = jag_contamination[field]
                # Don't filter objects based on Jaguar sims with no objects in bin
                # if contamination_field > vmax_contam_limit and contamination_field < 1:
                #    continue
                # if completeness_field < comp_limit and completeness_field > 0:
                #    #print(f'Skipping {field} for {id} because completeness {completeness_field} < 0.5')
                #    continue
                # if z_max_field <= z_min_field:
                # print(f'Skipping {field} for {id}')
                # if (field == detect_field) and (detect_field != 'NGDEEP'): # Removes galaxies which aren't detected in their own field. Excludes NGDEEP because some are detected in NGDEEP-HST
                #     print(f'{id} detected in {field} is not found to be detectable.')
                #     V_max = 0
                #     V_max_new = 0
                #     fields_used = -1
                #     fields = []
                #     break
                # continue

                # calculate/load unmasked area of forced photometry band
                if hasattr(data, "region_selector"):
                    region_selector = data.region_selector
                    invert_region = region != data.region_selector.name
                else:
                    region_selector = None
                    invert_region = False
                if unmasked_area == "selection":
                    unmasked_area_ = data.calc_unmasked_area(
                        mask_selector = data.forced_phot_band.filt_name,
                        region_selector = region_selector,
                        invert_region = invert_region,
                    )
                elif isinstance(unmasked_area, u.Quantity):
                    assert isinstance(unmasked_area, u.Quantity)
                    # TODO: ensure this has units of area
                    unmasked_area_ = unmasked_area
                else:
                    unmasked_area_ = data.calc_unmasked_area(
                        mask_selector = unmasked_area,
                        region_selector = region_selector,
                        invert_region = invert_region,
                    )
                z_min_used = np.max([z_min, z_bin[0]])
                z_max_used = np.min([z_max, z_bin[1]])
                if any(_z == -1.0 for _z in [z_min_used, z_max_used]):
                    V_max = -1.0
                else:
                    # V_max_simple = funcs.calc_Vmax(unmasked_area, z_bin[0], z_max_used)
                    V_max = funcs.calc_Vmax(
                        unmasked_area_, 
                        z_min_used, 
                        z_max_used
                    ).to(u.Mpc**3).value
                V_max_dict[region] = V_max

        SED_result_obj.obs_zrange[crop_name][data.full_name] = [z_min_used, z_max_used]
        # self.V_max_simple[z_bin_name][data.full_name] = V_max_simple
        V_max = np.sum(list(V_max_dict.values()))
        if V_max < -1.0:
            V_max = -1.0
        SED_result_obj.V_max[crop_name][data.full_name] = V_max * u.Mpc ** 3

        return V_max
    
        # breakpoint()
        # z_bin_name = f"{z_bin[0]:.1f}<z<{z_bin[1]:.1f}"
        # if len(data_arr) > 1:
        #     if not hasattr(self, "V_max_fields_used"):
        #         self.V_max_fields_used = {}
        #     if not z_bin_name in self.V_max_fields_used.keys():
        #         self.V_max_fields_used[z_bin_name] = {}
        #         V_max_combined = 0.
        #         #V_max_simple_combined = 0.
        #         fields_used = []
        #         for data in data_arr:
        #             if self.V_max[z_bin_name][data.full_name] >= 0.:
        #                 V_max_combined += self.V_max[z_bin_name][data.full_name]
        #                 fields_used.append(data.full_name)
        #             #V_max_simple += self.V_max_simple[z_bin_name][data.full_name]
        #         self.V_max_fields_used[z_bin_name][full_survey_name] = fields_used
        #         #self.V_max_simple[z_bin_name][joint_survey_name] = V_max_simple
        #         self.V_max[z_bin_name][full_survey_name] = V_max_combined
        # breakpoint()

    # def calc_Vmax_multifield(self, detect_cat_name: str, data_arr: Union[list, np.array], z_bin: Union[list, np.array], \
    #         SED_fit_params_key: str = "EAZY_fsps_larson_zfree", z_step: float = 0.01) -> None:
    #     z_bin_name = f"{SED_fit_params_key}_{z_bin[0]:.1f}<z<{z_bin[1]:.1f}"
    #     joint_survey_name = "+".join([data.full_name for data in data_arr])
    #     V_max = 0.
    #     V_max_simple = 0.
    #     fields_used = []
    #     for data in data_arr:
    #         # calculate Vmax in each field
    #         self.calc_Vmax(detect_cat_name, data, z_bin, SED_fit_params_key, z_step)
    #         if self.V_max[z_bin_name][data.full_name] >= 0.:
    #             V_max += self.V_max[z_bin_name][data.full_name]
    #             fields_used.append(data.full_name)
    #         #V_max_simple += self.V_max_simple[z_bin_name][data.full_name]

    #     self.V_max_fields_used[z_bin_name][joint_survey_name] = fields_used
    #     #self.V_max_simple[z_bin_name][joint_survey_name] = V_max_simple
    #     self.V_max[z_bin_name][joint_survey_name] = V_max
    #     return self

    # def save_Vmax(
    #     self: Self,
    #     Vmax: float,
    #     z_bin_name: str,
    #     full_survey_name: str,
    #     is_simple_Vmax: bool = False,
    # ) -> NoReturn:
    #     # if is_simple_Vmax:
    #     #     if not hasattr(self, "V_max_simple"):
    #     #         self.V_max_simple = {}
    #     #     if not z_bin_name in self.V_max_simple.keys():
    #     #         self.V_max_simple[z_bin_name] = {}
    #     #     self.V_max_simple[z_bin_name][full_survey_name] = Vmax
    #     # else:
    #     if not hasattr(self, "V_max"):
    #         self.V_max = {}
    #     if z_bin_name not in self.V_max.keys():
    #         self.V_max[z_bin_name] = {}
    #     self.V_max[z_bin_name][full_survey_name] = Vmax


# class Multiple_Galaxy:
#     def __init__(
#         self,
#         sky_coords,
#         IDs,
#         phots,
#         mask_flags_arr,
#         selection_flags_arr,
#         timed=True,
#     ):
#         if timed:
#             self.gals = [
#                 Galaxy(sky_coord, ID, phot, mask_flags, selection_flags)
#                 for sky_coord, ID, phot, mask_flags, selection_flags in tqdm(
#                     zip(
#                         sky_coords,
#                         IDs,
#                         phots,
#                         mask_flags_arr,
#                         selection_flags_arr,
#                     ),
#                     desc="Initializing galaxy objects",
#                     total=len(sky_coords),
#                 )
#             ]
#         else:
#             self.gals = [
#                 Galaxy(sky_coord, ID, phot, mask_flags, selection_flags)
#                 for sky_coord, ID, phot, mask_flags, selection_flags in zip(
#                     sky_coords, IDs, phots, mask_flags_arr, selection_flags_arr
#                 )
#             ]

#     def __len__(self):
#         return len(self.gals)

#     def __iter__(self):
#         self.iter = 0
#         return self

#     def __next__(self):
#         if self.iter > len(self) - 1:
#             raise StopIteration
#         else:
#             gal = self[self.iter]
#             self.iter += 1
#             return gal

#     def __getitem__(self, index):
#         return self.gals[index]

#     @classmethod
#     def from_fits_cat(
#         cls,
#         fits_cat: Union[Table, list, np.array],
#         instrument,
#         cat_creator,
#         SED_fit_params_arr,
#         timed=True,
#     ):
#         # load photometries from catalogue
#         phots = Multiple_Photometry_obs.from_fits_cat(
#             fits_cat, instrument, cat_creator, SED_fit_params_arr, timed=timed
#         ).phot_obs_arr
#         # load the ID and Sky Coordinate from the source catalogue
#         IDs = np.array(fits_cat[cat_creator.ID_label]).astype(int)
#         # load sky co-ordinates
#         RAs = (
#             np.array(fits_cat[cat_creator.ra_dec_labels["RA"]])
#             * cat_creator.ra_dec_units["RA"]
#         )
#         Decs = (
#             np.array(fits_cat[cat_creator.ra_dec_labels["DEC"]])
#             * cat_creator.ra_dec_units["DEC"]
#         )
#         sky_coords = SkyCoord(RAs, Decs, frame="icrs")
#         # mask flags should come from cat_creator
#         # mask_flags_arr = [{f"unmasked_{band}": cat_creator.load_flag(fits_cat_row, f"unmasked_{band}") for band in instrument.band_names} for fits_cat_row in fits_cat]
#         mask_flags_arr = [
#             {} for fits_cat_row in fits_cat
#         ]  # f"unmasked_{band}": None for band in instrument.band_names
#         selection_flags_arr = [
#             {
#                 selection_flag: bool(fits_cat_row[selection_flag])
#                 for selection_flag in cat_creator.selection_labels(fits_cat)
#             }
#             for fits_cat_row in fits_cat
#         ]
#         return cls(sky_coords, IDs, phots, mask_flags_arr, selection_flags_arr)

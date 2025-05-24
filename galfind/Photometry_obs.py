from __future__ import annotations

import inspect
import time
import astropy.units as u
from astropy.utils.masked import Masked
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from photutils.aperture import (
    CircularAperture,
    aperture_photometry,
)
from typing import TYPE_CHECKING, Union, List, Dict, NoReturn, Optional
if TYPE_CHECKING:
    from . import Multiple_Filter, SED_result, Multiple_Band_Cutout
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from .Photometry import Photometry
from .SED_result import Catalogue_SED_results, Galaxy_SED_results
from . import useful_funcs_austind as funcs
from . import galfind_logger

class Photometry_obs(Photometry):
    def __init__(
        self: Self,
        filterset: Multiple_Filter,
        flux: Masked[u.Quantity],
        flux_errs: Masked[u.Quantity],
        depths: Union[Dict[str, float], List[float]],
        aper_diam: u.Quantity,
        SED_results: Dict[str, SED_result] = {},
        simulated: bool = False,
        timed: bool = False,
    ):
        if timed:
            start = time.time()
        self.aper_diam = aper_diam
        self.SED_results = SED_results
        self.simulated = simulated
        super().__init__(filterset, flux, flux_errs, depths)
        if timed:
            end = time.time()
            print(end - start)

    def __repr__(self):
        sed_result_str = ','.join(list(self.SED_results.keys()))
        if sed_result_str != "":
            sed_result_str = f", {sed_result_str}"
        return f"{self.__class__.__name__}({self.filterset.instrument_name}," + \
            f" {self.aper_diam}{sed_result_str})"

    def __str__(self):
        output_str = funcs.line_sep
        output_str += "PHOTOMETRY OBS:\n"
        output_str += funcs.band_sep
        output_str += f"APERTURE DIAMETER: {self.aper_diam}\n"
        output_str += super().__str__(print_cls_name=False)
        for result in self.SED_results.values():
            output_str += str(result)
        output_str += f"SNR: {[np.round(snr, 2) for snr in self.SNR]}\n"
        output_str += funcs.line_sep
        return output_str

    # def __getattr__(
    #     self, property_name: str, origin: Union[str, dict] = "phot_obs"
    # ) -> Union[None, u.Quantity, u.Magnitude, u.Dex]:
    #     assert type(origin) in [str, dict], galfind_logger.critical(
    #         f"{origin=} with {type(origin)=} not in [str, dict]!"
    #     )
    #     if (
    #         "phot" in origin
    #     ):  # should have origin strings associated with Photometry.__getattr__ in list here!
    #         if property_name in self.__dict__.keys():
    #             return self.__getattribute__(property_name)
    #         # elif (
    #         #     "aper_corr" in property_name
    #         #     and property_name.split("_")[-1] in self.instrument.band_names
    #         # ):
    #         #     # return band aperture corrections
    #         #     return self.aper_corrs[property_name.split("_")[-1]]
    #         else:
    #             return super().__getattr__(
    #                 property_name,
    #                 "phot"
    #                 if origin == "phot_obs"
    #                 else origin.replace("phot_", ""),
    #             )
    #     else:
    #         # determine property type from name
    #         property_type = property_name.split("_")[-1]
    #         if any(
    #             string == property_type.lower()
    #             for string in ["val", "errs", "l1", "u1", "pdf"]
    #         ):
    #             property_name = property_name.replace(f"_{property_type}", "")
    #             property_type = property_type.lower()
    #         else:
    #             property_type = "_".join(property_name.split("_")[-2:])
    #             if property_type.lower() != "recently_updated":
    #                 # no property type, defaulting to value
    #                 galfind_logger.warning(
    #                     f"No property_type given in suffix of {property_name=} for Photometry_rest.__getattr__. Defaulting to value"
    #                 )
    #                 property_name = property_name.replace(
    #                     f"_{property_type}", ""
    #                 )
    #                 property_type = "val"
    #             else:
    #                 property_name = property_name.replace(
    #                     f"_{property_type}", ""
    #                 )
    #                 property_type = "recently_updated"
    #         # determine relevant SED_result to use from origin keyword
    #         if type(origin) in [str]:
    #             if origin.endswith("_REST_PROPERTY"):
    #                 SED_results_key = origin[:-14]
    #                 origin = "phot_rest"
    #             elif origin.endswith("_SED"):
    #                 SED_results_key = origin[:-4]
    #                 origin = "SED"
    #             else:
    #                 SED_results_key = origin
    #                 origin = "SED_result"
    #         else:  # type(origin) in [dict]:
    #             SED_results_key = origin["code"].label_from_SED_fit_params(
    #                 origin
    #             )
    #             origin = "SED_result"
    #         assert (
    #             SED_results_key in self.SED_results.keys()
    #         ), galfind_logger.critical(
    #             f"{SED_results_key=} not in {self.SED_results.keys()=}!"
    #         )
    #         return self.SED_results[SED_results_key].__getattr__(
    #             property_name, origin, property_type
    #         )

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        for key, value in self.__dict__.items():
            try:
                setattr(result, key, value)
            except:
                galfind_logger.critical(
                    f"copy({self.__class__.__name__}) {key}: {value} FAIL!"
                )
                breakpoint()
        return result

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

    @property
    def SNR(self):
        if isinstance(self.flux, u.Quantity):
            fluxes = self.flux
        else:
            fluxes = self.flux.filled(fill_value=np.nan)
        if self.simulated:
            return [
                flux * 5 / depth
                for flux, depth in zip(
                    fluxes.to(u.Jy).value,
                    self.depths.to(u.Jy).value,
                )
            ]
        else:
            return [
                (flux * 10 ** (aper_corr / -2.5)) * 5 / depth
                if flux > 0.0
                else flux * 5 / depth
                for aper_corr, flux, depth in zip(
                    self.aper_corrs,
                    fluxes.to(u.Jy).value,
                    self.depths.to(u.Jy).value,
                )
            ]

    @property
    def aper_corrs(self):
        if self.simulated:
            return [np.nan for filt in self.filterset]
        else:
            return [filt.instrument.aper_corrs[filt.band_name] \
                [self.aper_diam] for filt in self.filterset]

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
        galfind_logger.warning(
            "SED_fit_params should be included in this function"
        )
        galfind_logger.warning(
            "Problems with Photometry_obs.from_fits_cat when photometry and SED fitting properties are in different catalogue extensions"
        )
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
    def from_phot(
        cls,
        phot,
        aper_diam: u.Quantity,
        min_flux_pc_err: Union[int, float],
        SED_results: dict = {},
    ):
        return cls(
            phot.instrument,
            phot.flux,
            phot.flux_errs,
            aper_diam,
            min_flux_pc_err,
            phot.depths,
            SED_results,
        )
    
    @classmethod
    def from_multiple_band_cutout(
        cls: Type[Self],
        multi_band_cutout: Multiple_Band_Cutout,
        aper_diam: u.Quantity,
    ) -> Self:
        # run photutils on every band_data in stacked_band_cutout
        xpos = multi_band_cutout[0].meta["SIZE_PIX"] / 2
        ypos = multi_band_cutout[0].meta["SIZE_PIX"] / 2
        pix_scale = multi_band_cutout[0].meta["SIZE_AS"] / multi_band_cutout[0].meta["SIZE_PIX"]
        aperture = CircularAperture((xpos, ypos), r = aper_diam / (2. * pix_scale))
        for cutout in multi_band_cutout:
            data = cutout.band_data.load_im()[0]
            rms_err = cutout.band_data.load_rms_err(output_hdr = False)
            aper_phot_filt = aperture_photometry(data, aperture, error = rms_err)
            breakpoint()

    def update_SED_result(
        self: Self, 
        gal_SED_result: SED_result
    ) -> NoReturn:
        gal_SED_result_dict = {gal_SED_result.SED_code.label: gal_SED_result}
        if hasattr(self, "SED_results"):
            self.SED_results = {**self.SED_results, **gal_SED_result_dict}
        else:
            self.SED_results = gal_SED_result_dict

    def get_SED_fit_params_arr(self, code) -> list:
        return [
            code.SED_fit_params_from_label(label)
            for label in self.SED_results.keys()
        ]

    def load_property(
        self, gal_property: Union[dict, u.Quantity], save_name: str
    ) -> None:
        setattr(self, save_name, gal_property)

    # def make_ext_src_corrs(
    #     self,
    #     gal_property: str,
    #     origin: Union[str, dict],
    #     ext_src_band: Union[str, list, np.array] = "F444W",
    # ) -> None:
    #     # determine correct SED fitting results to use
    #     if type(origin) in [dict]:
    #         origin_key = origin["code"].label_from_SED_fit_params(origin)
    #         rest_property = False
    #     elif type(origin) in [str]:
    #         if "_REST_PROPERTY" in origin:
    #             rest_property = True
    #         else:
    #             rest_property = False
    #         origin_key = origin.replace("_REST_PROPERTY", "")
    #     else:
    #         galfind_logger.critical(f"{type(origin)=} not in [str, dict]!")
    #     # skip if key not available
    #     if origin_key not in self.SED_results.keys():
    #         galfind_logger.warning(
    #             f"Could not compute ext_src_corrs for {gal_property=} as {origin_key=} not in {self.SED_results.keys()=}!"
    #         )
    #     else:
    #         if rest_property:
    #             data_obj = self.SED_results[origin_key].phot_rest
    #         else:
    #             data_obj = self.SED_results[origin_key]
    #         properties = data_obj.properties
    #         property_errs = data_obj.property_errs
    #         property_PDFs = data_obj.property_PDFs
    #         # skip if galaxy property not in properties + property_errs + property_PDFs dicts
    #         if any(
    #             gal_property not in property_dict.keys()
    #             for property_dict in [properties, property_errs, property_PDFs]
    #         ):
    #             galfind_logger.warning(
    #                 f"{gal_property=},{origin_key=},{rest_property=} not in all of [{properties.keys()=},{property_errs.keys()=},{property_PDFs.keys()=}]!"
    #             )
    #         else:
    #             orig_property = properties[gal_property]
    #             orig_property_PDF = property_PDFs[gal_property]
    #             assert orig_property_PDF.x.unit == orig_property.unit
    #             # errors may not necessarily have the same unit
    #             ext_src_corr = self.ext_src_corrs[ext_src_band]
    #             PDF_add_kwargs = {"ext_src_band": ext_src_band}
    #             if type(orig_property) in [u.Magnitude]:
    #                 correction = (
    #                     funcs.flux_to_mag_ratio(ext_src_corr.value) * u.mag
    #                 )  # units are incorrect
    #                 updated_property = orig_property + correction
    #                 PDF_add_kwargs = {
    #                     **PDF_add_kwargs,
    #                     **{"ext_src_corr": correction},
    #                 }
    #                 updated_property_PDF = orig_property_PDF.__add__(
    #                     correction,
    #                     name_ext=funcs.ext_src_label,
    #                     add_kwargs=PDF_add_kwargs,
    #                     save=True,
    #                 )
    #             elif type(orig_property) in [u.Dex]:
    #                 correction = u.Dex(np.log10(ext_src_corr.value))
    #                 updated_property = orig_property + correction
    #                 PDF_add_kwargs = {
    #                     **PDF_add_kwargs,
    #                     **{"ext_src_corr": correction},
    #                 }
    #                 updated_property_PDF = orig_property_PDF.__add__(
    #                     correction,
    #                     name_ext=funcs.ext_src_label,
    #                     add_kwargs=PDF_add_kwargs,
    #                     save=True,
    #                 )
    #             elif type(orig_property) in [u.Quantity]:
    #                 updated_property = orig_property * ext_src_corr
    #                 PDF_add_kwargs = {
    #                     **PDF_add_kwargs,
    #                     **{"ext_src_corr": ext_src_corr},
    #                 }
    #                 updated_property_PDF = orig_property_PDF.__mul__(
    #                     ext_src_corr,
    #                     name_ext=funcs.ext_src_label,
    #                     add_kwargs=PDF_add_kwargs,
    #                     save=True,
    #                 )
    #             else:
    #                 galfind_logger.warning(
    #                     f"{gal_property}={orig_property} from {origin=} with {type(orig_property)=} not in [Magnitude, Dex, Quantity]"
    #                 )
    #             # update properties and property_PDFs (assume property_errs remain unaffected)
    #             data_obj.properties[updated_property_PDF.property_name] = (
    #                 updated_property
    #             )
    #             data_obj.property_PDFs[updated_property_PDF.property_name] = (
    #                 updated_property_PDF
    #             )
    #             # save non rest_property attributes outside of dict as well
    #             if not rest_property:
    #                 setattr(
    #                     data_obj,
    #                     updated_property_PDF.property_name,
    #                     updated_property,
    #                 )

    # def make_all_ext_src_corrs(
    #     self, ext_src_band: Union[str, list, np.array] = "F444W"
    # ) -> dict:
    #     # extract previously calculated galaxy properties and their origins
    #     code_ext_src_property_dict = {
    #         key: [
    #             gal_property
    #             for gal_property in self.SED_results[key]
    #             .SED_fit_params["code"]
    #             .ext_src_corr_properties
    #             if gal_property in self.SED_results[key].properties.keys()
    #             and gal_property in self.SED_results[key].property_PDFs.keys()
    #         ]
    #         for key in self.SED_results.keys()
    #     }
    #     sed_rest_ext_src_property_dict = {
    #         f"{key}_REST_PROPERTY": [
    #             gal_property
    #             for gal_property in self.SED_results[
    #                 key
    #             ].phot_rest.properties.keys()
    #             if gal_property.split("_")[0] in funcs.ext_src_properties
    #             and gal_property
    #             in self.SED_results[key].phot_rest.property_PDFs.keys()
    #         ]
    #         for key in self.SED_results.keys()
    #     }
    #     ext_src_property_dict = {
    #         **code_ext_src_property_dict,
    #         **sed_rest_ext_src_property_dict,
    #     }
    #     # make the extended source corrections
    #     [
    #         self.make_ext_src_corrs(gal_property, origin)
    #         for origin, gal_properties in ext_src_property_dict.items()
    #         for gal_property in gal_properties
    #     ]
    #     # return dict of {origin: [property for property in gal[origin]]}
    #     return ext_src_property_dict

    def plot(
        self: Self,
        ax: Optional[plt.Axes] = None,
        wav_units: u.Unit = u.AA,
        mag_units: u.Unit = u.Jy,
        plot_errs: dict = {"x": False, "y": True},
        annotate: bool = True,
        uplim_sigma: float = 2.0,
        auto_scale: bool = True,
        SNR_labelsize: Optional[float] = 7.5,
        errorbar_kwargs: dict = {
            "ls": "",
            "marker": "o",
            "ms": 4.0,
            "zorder": 100.0,
            "path_effects": [pe.withStroke(linewidth=2.0, foreground="white")],
        },
        filled: bool = True,
        colour: str = "black",
        label: str = "Photometry",
        log_scale: bool = False,
    ):
        plot, wavs_to_plot, mags_to_plot, yerr, uplims = super().plot(
            ax,
            wav_units,
            mag_units,
            plot_errs,
            # annotate,
            uplim_sigma,
            auto_scale,
            errorbar_kwargs,
            filled,
            colour,
            label,
            return_extra=True,
            log_scale=log_scale,
        )

        if SNR_labelsize is not None:
            label_kwargs = {
                "ha": "center",
                "fontsize": SNR_labelsize,
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
    
    def load_sextractor_ext_src_corrs(
        self: Self, 
        aper_corrs: Optional[Dict[str, float]] = None
    ) -> NoReturn:
        if aper_corrs is None:
            [filt.instrument._load_aper_corrs() for filt in self.filterset]
            aper_corrs = {filt.band_name: filt.instrument. \
                aper_corrs[filt.band_name][self.aper_diam] for filt in self.filterset}
        assert all(filt_name in aper_corrs.keys() \
            for filt_name in self.filterset.band_names)

        ext_src_corrs = {filt_name: (self.sex_FLUX_AUTO[filt_name] \
            / (self.flux[i] * funcs.mag_to_flux_ratio(-aper_corrs[filt_name]))) \
            .to(u.dimensionless_unscaled).unmasked for i, filt_name \
            in enumerate(self.filterset.band_names)}
        self.ext_src_corrs = {
            filt_name: ext_src_corr.value
            if ext_src_corr.value > 1.0 else 1.0
            for filt_name, ext_src_corr in ext_src_corrs.items()
        }
        # propagate ext_src_corrs to SED_results[key].phot_rest
        for key in self.SED_results.keys():
            self.SED_results[key].phot_rest. \
                ext_src_corrs = self.ext_src_corrs

    # def load_local_depths(self, sex_cat_row, instrument, aper_diam_index):
    #    self.depths = np.array([sex_cat_row[f"loc_depth_{band}"].T[aper_diam_index] for band in instrument.band_names])

    # def SNR_crop(self, band, sigma_detect_thresh):
    #     index = self.instrument.band_from_index(band)
    #     # local depth in units of Jy
    #     loc_depth_Jy = self.depths[index].to(u.Jy) / 5
    #     detection_Jy = self.flux[index].to(u.Jy)
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
        flux_arr,
        flux_errs_arr,
        aper_diam,
        min_flux_pc_err,
        loc_depths_arr,
        SED_results_arr=[],
        timed=True,
    ):
        # force SED_results_arr to have the same len as the number of input fluxes
        if SED_results_arr == []:
            SED_results_arr = np.full(len(flux_arr), {})
        if timed:
            self.phot_obs_arr = [
                Photometry_obs(
                    instrument,
                    flux,
                    flux_errs,
                    aper_diam,
                    min_flux_pc_err,
                    loc_depths,
                    SED_results,
                )
                for instrument, flux, flux_errs, loc_depths, SED_results in tqdm(
                    zip(
                        instrument_arr,
                        flux_arr,
                        flux_errs_arr,
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
                    flux,
                    flux_errs,
                    aper_diam,
                    min_flux_pc_err,
                    loc_depths,
                    SED_results,
                )
                for instrument, flux, flux_errs, loc_depths, SED_results in zip(
                    instrument_arr,
                    flux_arr,
                    flux_errs_arr,
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
        flux_arr, flux_errs_arr, gal_band_mask = cat_creator.load_photometry(
            fits_cat, instrument.band_names, timed=timed
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
            SED_results_arr = np.full(len(flux_arr), {})
        return cls(
            instrument_arr,
            flux_arr,
            flux_errs_arr,
            cat_creator.aper_diam,
            cat_creator.min_flux_pc_err,
            depths_arr,
            SED_results_arr,
            timed=timed,
        )

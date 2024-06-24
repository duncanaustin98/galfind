#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:11:23 2023

@author: austind
"""

# Galaxy.py
import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import os
import sys
import json
import time
import glob
from pathlib import Path
from astropy.nddata import Cutout2D
from tqdm import tqdm
import matplotlib.patheffects as pe
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.visualization import LogStretch, LinearStretch, ImageNormalize, ManualInterval
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from typing_extensions import Self
from typing import Union

from . import useful_funcs_austind as funcs
from . import config, galfind_logger, astropy_cosmo
from . import Photometry_rest, Photometry_obs, Multiple_Photometry_obs, Data, Instrument, NIRCam, ACS_WFC, WFC3_IR, PDF
from .SED import SED, Mock_SED_rest, Mock_SED_obs
from .EAZY import EAZY
from .Emission_lines import line_diagnostics

class Galaxy:
    
    def __init__(self, sky_coord, ID, phot, mask_flags = {}, selection_flags = {}): # cat_path,
        self.sky_coord = sky_coord
        self.ID = int(ID) 
        self.phot = phot
        #self.cat_path = cat_path
        self.mask_flags = mask_flags
        self.selection_flags = selection_flags
        self.cutout_paths = {}
        
    @classmethod
    def from_pipeline(cls, ):
        pass

    @classmethod
    def from_fits_cat(cls, fits_cat_row, instrument, cat_creator, codes, lowz_zmax, templates_arr):
        # load multiple photometries from the fits catalogue
        phot = Photometry_obs.from_fits_cat(fits_cat_row, instrument, cat_creator, cat_creator.aper_diam, cat_creator.min_flux_pc_err, codes, lowz_zmax, templates_arr) # \
                # for min_flux_pc_err in cat_creator.min_flux_pc_err for aper_diam in cat_creator.aper_diam]
        # load the ID and Sky Coordinate from the source catalogue
        ID = int(fits_cat_row[cat_creator.ID_label])
        sky_coord = SkyCoord(fits_cat_row[cat_creator.ra_dec_labels["RA"]] * u.deg, fits_cat_row[cat_creator.ra_dec_labels["DEC"]] * u.deg, frame = "icrs")
        # mask flags should come from cat_creator
        mask_flags = {} #{f"unmasked_{band}": cat_creator.load_flag(fits_cat_row, f"unmasked_{band}") for band in instrument.band_names}
        return cls(sky_coord, ID, phot, mask_flags)
    
    def __str__(self):
        line_sep = "*" * 40 + "\n"
        band_sep = "-" * 10 + "\n"
        output_str = line_sep
        output_str += f"GALAXY {self.ID}: (RA, DEC) = ({np.round(self.sky_coord.ra, 5)}, {np.round(self.sky_coord.dec, 5)})\n"
        output_str += band_sep
        output_str += f"MASK FLAGS: {self.mask_flags}\n"
        output_str += f"SELECTION FLAGS: {self.selection_flags}\n"
        output_str += str(self.phot)
        output_str += line_sep
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
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, deepcopy(value, memo))
        return result
    
    def update(self, gal_SED_results, index = 0): # for now just update the single photometry
        self.phot.update(gal_SED_results)
    
    def update_mask(self, mask, update_phot_rest = False):
        self.phot.update_mask(mask, update_phot_rest = update_phot_rest)
        return self
        
    # def update_mask_band(self, band, bool_value):
    #     self.mask_flags[band] = bool_value

    def load_property(self, gal_property: Union[dict, u.Quantity], save_name: str):
        setattr(self, save_name, gal_property)

    def make_cutout(self, band, data, wcs = None, im_header = None, survey = None, version = None, cutout_size = 32):
        
        if type(data) == Data:
            survey = data.survey
            version = data.version
        if survey == None or version == None:
            raise(Exception("'survey' and 'version' must both be given to construct save paths"))
        
        out_path = f"{config['Cutouts']['CUTOUT_DIR']}/{version}/{survey}/{band}/{self.ID}.fits"
        rerun = False
        if Path(out_path).is_file():
            
            size = fits.open(out_path)[0].header["size"]
            if size != cutout_size:
                galfind_logger.info("Cutout size does not match requested size, overwriting...")
                print('Cutout size does not match requested size, overwriting...')
                rerun = True
        if config.getboolean("Cutouts", "OVERWRITE_CUTOUTS") or rerun or not Path(out_path).is_file():
            if type(data) == Data:
                im_data, im_header, seg_data, seg_header = data.load_data(band, incl_mask = False)
                wht_data = data.load_wht(band)
                rms_err_data = data.load_rms_err(band)
                wcs = data.load_wcs(band)
                data = {"SCI": im_data, "SEG": seg_data, "WHT": wht_data, "RMS_ERR": rms_err_data}
            elif type(data) == dict and type(wcs) != type(None) and type(im_header) != type(None):
                pass
            else:
                raise(Exception(""))
            hdul = [fits.PrimaryHDU(header = fits.Header({"ID": self.ID, "survey": survey, "version": version, \
                        "RA": self.sky_coord.ra.value, "DEC": self.sky_coord.dec.value, "size": cutout_size}))]
            for i, (label_i, data_i) in enumerate(data.items()):
                if type(data_i) == type(None):
                    galfind_logger.warning(f"No data found for {label_i} in {band}!")
                    continue
                cutout = Cutout2D(data_i, self.sky_coord, size = (cutout_size, cutout_size), wcs = wcs)
                im_header.update(cutout.wcs.to_header())
                hdul.append(fits.ImageHDU(cutout.data, header = im_header, name = label_i))
            #print(hdul)
            os.makedirs("/".join(out_path.split("/")[:-1]), exist_ok = True)
            fits_hdul = fits.HDUList(hdul)
            fits_hdul.writeto(out_path, overwrite = True)
            funcs.change_file_permissions(out_path)
            galfind_logger.info(f"Saved fits cutout to: {out_path}")
        else:
            galfind_logger.info(f"Already made fits cutout for {survey} {version} {self.ID} {band}")
            fits_hdul = fits.open(out_path)
        self.cutout_paths[band] = out_path
        return fits_hdul

    def make_RGB(self, data, blue_bands = ["F090W"], green_bands = ["F200W"], red_bands = ["F444W"], version = None, survey = None, method = "trilogy", cutout_size = 32):
        method = method.lower() # make method lowercase
        # ensure all blue, green and red bands are contained in the data object
        assert all(band in data.instrument.band_names for band in blue_bands + green_bands + red_bands), \
            galfind_logger.warning(f"Cannot make galaxy RGB as not all {blue_bands + green_bands + red_bands} are in {data.instrument.band_names}")
        # extract survey and version from data
        if type(data) == Data:
            survey = data.survey
            version = data.version
        if survey == None or version == None:
            raise(Exception("'survey' and 'version' must both be given to construct save paths"))
        # construct out_path
        out_path = f"{config['Cutouts']['CUTOUT_DIR']}/{version}/{survey}/B={'+'.join(blue_bands)},G={'+'.join(green_bands)},R={'+'.join(red_bands)}/{method}/{self.ID}.png"
        funcs.make_dirs(out_path)
        if not os.path.exists(out_path):
            # make cutouts for the required bands if they don't already exist, and load cutout paths
            RGB_cutout_paths = {}
            for colour, bands in zip(["B", "G", "R"], [blue_bands, green_bands, red_bands]):
                [self.make_cutout(band, data, cutout_size = cutout_size) for band in bands]
                RGB_cutout_paths[colour] = [self.cutout_paths[band] for band in bands]
            if method == "trilogy":
                # Write trilogy.in
                in_path = out_path.replace(".png", "_trilogy.in")
                with open(in_path, "w") as f:
                    for colour, cutout_paths in RGB_cutout_paths.items():
                        f.write(f"{colour}\n")
                        for path in cutout_paths:
                            f.write(f"{path}[1]\n")
                        f.write("\n")
                    f.write("indir  /\n")
                    f.write(f"outname  {funcs.split_dir_name(out_path, 'name').replace('.png', '')}\n")
                    f.write(f"outdir  {funcs.split_dir_name(out_path, 'dir')}\n")
                    f.write("samplesize 20000\n")
                    f.write("stampsize  2000\n")
                    f.write("showstamps  0\n")
                    f.write("satpercent  0.001\n")
                    f.write("noiselum    0.10\n")
                    f.write("colorsatfac  1\n")
                    f.write("deletetests  1\n")
                    f.write("testfirst   0\n")
                    f.write("sampledx  0\n")
                    f.write("sampledy  0\n")
                
                funcs.change_file_permissions(in_path)
                # Run trilogy
                sys.path.insert(1, "/nvme/scratch/software/trilogy") # Not sure why this path doesn't work: config["Other"]["TRILOGY_DIR"]
                from trilogy3 import Trilogy
                galfind_logger.info(f"Making trilogy cutout RGB at {out_path}")
                Trilogy(in_path, images = None).run()
            elif method == "lupton":
                raise(NotImplementedError())
    
    def plot_cutouts(self, ax_arr, data, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, \
            hide_masked_cutouts = True, cutout_size = 32, high_dyn_rng = False):

        for i, band in enumerate(self.phot.instrument.band_names):
                
            # need to load sextractor flux_radius as a general function somewhere!
            radius = 0.16 * u.arcsec # need access to galfind cat_creator for this
            radius_pix = (radius / data.im_pixel_scales[band]).to(u.dimensionless_unscaled).value
            #flux_radius = None
            #radius_sextractor = flux_radius
            if self.phot.flux_Jy.mask[i] and hide_masked_cutouts:
                data_cutout = None
            else:
                # load cutout if already made, else produce one
                cutout_hdul = self.make_cutout(band, data, cutout_size = cutout_size)
                data_cutout = cutout_hdul[1].data # should handle None in the case of NoOverlapError        

            if type(data_cutout) != type(None):
                # Set top value based on central 10x10 pixel region
                top = np.max(data_cutout[:20, 10:20])
                top = np.max(data_cutout[int(cutout_size // 2 - 0.3 * cutout_size) : int(cutout_size // 2 + 0.3 * cutout_size), \
                    int(cutout_size // 2 - 0.3 * cutout_size) : int(cutout_size // 2 + 0.3 * cutout_size)])
                bottom_val = top / 10 ** 5
                
                if high_dyn_rng:
                    a = 300
                else:
                    a = 0.1
                stretch = LogStretch(a = a)

                n_sig_detect = self.phot.SNR[i]
                if n_sig_detect < 100:
                    bottom_val = top / 10 ** 3
                    a = 100
                if n_sig_detect <= 15:
                    bottom_val = top/10**2
                    a = 0.1
                if n_sig_detect < 8:
                    bottom_val = top / 100000
                    stretch = LinearStretch()
                    
                data_cutout = np.clip(data_cutout * 0.9999, bottom_val * 1.000001, top) # why?
                norm = ImageNormalize(data_cutout, interval = ManualInterval(bottom_val, top), clip = True, stretch = stretch)

                #ax_arr[i].cla()
                ax_arr[i].set_visible(True)
                ax_arr[i].set_aspect('equal', adjustable = 'box', anchor = 'N')
                ax_arr[i].set_xticks([])
                ax_arr[i].set_yticks([])

                ax_arr[i].imshow(data_cutout, norm = norm, cmap='magma', origin = "lower")
                ax_arr[i].text(0.95, 0.95, band, fontsize = 'small', c = 'white', \
                    transform = ax_arr[i].transAxes, ha = 'right', va = 'top', zorder = 10, fontweight = 'bold')         

                # add circles to show extraction aperture and sextractor FLUX_RADIUS
                xpos = np.mean(ax_arr[i].get_xlim())
                ypos = np.mean(ax_arr[i].get_ylim())
                region = patches.Circle((xpos, ypos), radius_pix, fill = False, \
                    linestyle = '--', lw = 1, color = 'white', zorder = 20)
                ax_arr[i].add_patch(region)
                galfind_logger.warning("Need to load in SExtractor FLUX_RADIUS")
                # if radius_sextractor != 0:
                #     region_sextractor = patches.Circle((xpos, ypos), radius_sextractor, \
                #         fill = False, linestyle = '--', lw = 1, color = 'blue', zorder = 20)
                #     ax_arr[i].add_patch(region_sextractor)
                
            else: # if the band is masked and this should not be shown
                #ax_arr[i].cla()
                ax_arr[i].set_visible(False)
            
            # add scalebars to the last cutout
            if len(data.instrument) > 0:
                # re in pixels
                re = 10 # pixels
                d_A = astropy_cosmo.angular_diameter_distance( \
                    self.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].z)
                pix_scal = u.pixel_scale(0.03*u.arcsec/u.pixel)
                re_as = (re * u.pixel).to(u.arcsec, pix_scal)
                re_kpc = (re_as * d_A).to(u.kpc, u.dimensionless_angles())
                
                # First scalebar
                scalebar = AnchoredSizeBar(ax_arr[i].transData, 0.3 / data.im_pixel_scales[band].value, \
                    "0.3\"", 'lower right', pad = 0.3, color='white', frameon=False, size_vertical=2)
                ax_arr[-1].add_artist(scalebar)
                # Plot scalebar with physical size
                scalebar = AnchoredSizeBar(ax_arr[-1].transData, re, f"{re_kpc:.1f}", \
                    'upper left', pad=0.3, color='white', frameon=False, size_vertical=1.5)
                ax_arr[-1].add_artist(scalebar)
    
    def plot_phot_diagnostic(self, ax, data, SED_fit_params_arr, zPDF_plot_SED_fit_params_arr, wav_unit = u.um, flux_unit = u.ABmag, \
            hide_masked_cutouts = True, cutout_size = 32, high_dyn_rng = False, annotate_PDFs = True, plot_rejected_reasons = False, overwrite = True):

        cutout_ax, phot_ax, PDF_ax = ax
        # update SED_fit_params with appropriate lowz_zmax
        SED_fit_params_arr = [SED_fit_params["code"].update_lowz_zmax(SED_fit_params, self.phot.SED_results) for SED_fit_params in SED_fit_params_arr]
        zPDF_plot_SED_fit_params_arr = [SED_fit_params["code"].update_lowz_zmax(SED_fit_params, self.phot.SED_results) for SED_fit_params in zPDF_plot_SED_fit_params_arr] 

        zPDF_labels = [f"{SED_fit_params['code'].label_from_SED_fit_params(SED_fit_params)} PDF" for SED_fit_params in zPDF_plot_SED_fit_params_arr]
        # reset parameters
        for ax_, label in zip(PDF_ax, zPDF_labels):
            ax_.set_yticks([])
            ax_.set_xlabel("Redshift, z")
            ax_.set_title(label, fontsize = "medium")
        
        out_path = f"{config['Selection']['SELECTION_DIR']}/SED_plots/{data.version}/{data.instrument.name}/{data.survey}/{self.ID}.png"
        funcs.make_dirs(out_path)

        if not Path(out_path).is_file() or overwrite:
            # plot cutouts (assuming reference SED_fit_params is at 0th index)
            self.plot_cutouts(cutout_ax, data, SED_fit_params_arr[0], \
                hide_masked_cutouts = hide_masked_cutouts, cutout_size = cutout_size, high_dyn_rng = high_dyn_rng)
                    
            # plot specified SEDs andd save colours
            SED_colours = {}
            errorbar_kwargs = {"ls": "", "marker": "o", "ms": 8., "zorder": 100., "path_effects": [pe.withStroke(linewidth = 2., foreground = "white")]}
            for SED_fit_params in reversed(SED_fit_params_arr):
                key = SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)
                SED_plot = self.phot.SED_results[key].SED.plot_SED(phot_ax, wav_unit, flux_unit, label = key)
                SED_colours[key] = SED_plot[0].get_color()
                # plot the mock photometry
                self.phot.SED_results[key].SED.create_mock_phot(self.phot.instrument, depths = self.phot.depths)
                self.phot.SED_results[key].SED.mock_phot.plot_phot(phot_ax, wav_unit, flux_unit, uplim_sigma = None, \
                    auto_scale = False, plot_errs = {"x": False, "y": False}, errorbar_kwargs = errorbar_kwargs, \
                    label = None, filled = False, colour = SED_colours[key])
                #ax_photo.scatter(band_wavs_lowz, band_mags_lowz, edgecolors=eazy_color_lowz, marker='o', facecolor='none', s=80, zorder=4.5)               
            self.phot.plot_phot(phot_ax, wav_unit, flux_unit, annotate = False, auto_scale = True, label_SNRs = True)
            # photometry axis title
            phot_ax.set_title(f"{data.survey} {self.ID} ({data.version})")
            # plot rejected reasons somewhere
            # if plot_rejected_reasons:
            #     rejected = str(row[f'rejected_reasons{col_ext}'][0])
            #     if rejected != '':
            #         phot_ax.annotate(rejected, (0.9, 0.95), ha='center', fontsize='small', xycoords = 'axes fraction', zorder=5)
            # photometry axis legend
            phot_ax.legend(loc = "upper right", fontsize = "small", frameon = False)
            for text in phot_ax.get_legend().get_texts():
                text.set_path_effects([pe.withStroke(linewidth = 3, foreground = 'white')])
                text.set_zorder(12)

            # plot PDF on relevant axis
            assert(len(zPDF_plot_SED_fit_params_arr) == len(PDF_ax)) # again, this is not totally generalized and should be == 2 for now
            # could extend to plotting multiple PDFs on the same axis
            for ax, SED_fit_params in zip(PDF_ax, zPDF_plot_SED_fit_params_arr):
                key = SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)
                if key in SED_colours.keys():
                    colour = SED_colours[key]
                else:
                    colour = "black"
                self.phot.SED_results[key].property_PDFs["z"].plot(ax, annotate = annotate_PDFs, colour = colour)

            # Save and clear axes
            plt.savefig(out_path, dpi = 300, bbox_inches = 'tight')
            funcs.change_file_permissions(out_path)
            for ax in [phot_ax] + PDF_ax + cutout_ax:
                ax.cla()
            
        return out_path
    
    # Spectroscopy

    def load_spectra(self, spectra):
        self.spectra = spectra
        return self

    def plot_spec_diagnostic(self, ax, grating_filter = "PRISM/CLEAR", overwrite = True):
        # bare in mind that not all galaxies have spectroscopic data
        if hasattr(self, "spectra"):
            # plot spectral diagnostic
            pass
        else:
            pass

    # %% Selection methods
        
    def select_min_bands(self, min_bands, update = True):
        
        if type(min_bands) != int:
            min_bands = int(min_bands)

        selection_name = f"bands>{min_bands - 1}"
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            if len(self.phot) >= min_bands:
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if update:
                    self.selection_flags[selection_name] = False
        return self, selection_name

    def select_min_unmasked_bands(self, min_bands, update = True):

        if type(min_bands) != int:
            min_bands = int(min_bands)

        selection_name = f"unmasked_bands>{min_bands - 1}"
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            if len(self.phot) == 0: # no data at all (not sure why sextractor does this)
                if update:
                    self.selection_flags[selection_name] = False
                return self, selection_name
            # extract mask
            mask = self.phot.flux_Jy.mask
            n_unmasked_bands = len([val for val in mask if val == False])
            if n_unmasked_bands >= min_bands:
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if update:
                    self.selection_flags[selection_name] = False
        return self, selection_name
    
    #  already made these boolean columns in the catalogue
    # def select_unmasked_bands(self, band_names, update = True):
    #     # ensure band_names input is of the required type, convert if not, and raise error if not convertable
    #     if type(band_names) in [list, np.array]:
    #         pass # band_names already of a valid type
    #     elif type(band_names) == str:
    #         # convert to a list, assuming the bands are separated by a "+"
    #         band_names = band_names.split("+")
    #     else:
    #         galfind_logger.critical(f"band_names = {band_names} with type = {type(band_names)} is not in [list, np.array, str]!")
    #     # ensure that each band is a valid band name in galfind
    #     assert(all(band_name in json.loads(config.get("Other", "ALL_BANDS")) for band_name in band_names), \
    #         galfind_logger.critical(f"band_names = {band_names} has at least one invalid band!"))
        
    #     selection_name = f"unmasked_{'+'.join(band_names)}"
    #     if selection_name in self.selection_flags.keys():
    #         galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
    #     else:
    #         # extract band IDs belonging to the input instrument name
    #         band_indices = np.array([i for i, band_name in enumerate(self.phot.instrument.band_names) if band_name in band_names])
    #         mask = self.phot.flux_Jy.mask[band_indices]
    #         if all(mask_band == False for mask_band in mask):
    #             if update:
    #                 self.selection_flags[selection_name] = True
    #         else:
    #             if update:
    #                 self.selection_flags[selection_name] = False
    #     return self, selection_name

    # Masking Selection
    
    def select_unmasked_instrument(self, instrument, update = True):
        assert(issubclass(instrument.__class__, Instrument))
        assert(instrument.__class__.__name__ in self.phot.instrument.name.split("+"))

        selection_name = f"unmasked_{instrument.__class__.__name__}"
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            if len(self.phot) == 0: # no data at all (not sure why sextractor does this)
                if update:
                    self.selection_flags[selection_name] = False
                return self, selection_name
            # extract band IDs belonging to the input instrument name
            band_indices = np.array([i for i, band in enumerate(self.phot.instrument.band_names) if band in instrument.band_names])
            mask = self.phot.flux_Jy.mask[band_indices]
            if all(mask_band == False for mask_band in mask):
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if update:
                    self.selection_flags[selection_name] = False
        return self, selection_name
    
    # Galaxy photometry property selection

    def select_phot_galaxy_property(self, property_name, gtr_or_less, property_lim, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, update = True):
        key = SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)
        assert(property_name in self.phot.SED_results[key].properties.keys())
        galfind_logger.warning("Ideally need to include appropriate units for photometric galaxy property selection")
        assert(type(property_lim) in [int, float])
        assert(gtr_or_less in ["gtr", "less", ">", "<"])
        if gtr_or_less in ["gtr", ">"]:
            selection_name = f"{property_name}>{property_lim}"
        else:
            selection_name = f"{property_name}<{property_lim}"
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            if len(self.phot) == 0: # no data at all (not sure why sextractor does this)
                if update:
                    self.selection_flags[selection_name] = False
                return self, selection_name
            property_val = self.phot.SED_results[key].properties[property_name]
            if ((gtr_or_less in ["gtr", ">"] and property_val > property_lim) or (gtr_or_less in ["less", "<"] and property_val < property_lim)):
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if update:
                    self.selection_flags[selection_name] = False
        return self, selection_name

    def select_phot_galaxy_property_bin(self, property_name, property_lims, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, update = True):
        key = SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)
        assert(property_name in self.phot.SED_results[key].properties.keys())
        galfind_logger.warning("Ideally need to include appropriate units for photometric galaxy property selection")
        assert(type(property_lims) in [np.ndarray, list])
        assert(len(property_lims) == 2)
        assert(property_lims[1] > property_lims[0])
        selection_name = f"{property_lims[0]}<{property_name}<{property_lims[1]}"
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            if len(self.phot) == 0: # no data at all (not sure why sextractor does this)
                if update:
                    self.selection_flags[selection_name] = False
                return self, selection_name
            property_val = self.phot.SED_results[key].properties[property_name]
            if property_val > property_lims[0] and property_val < property_lims[1]:
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if update:
                    self.selection_flags[selection_name] = False
        return self, selection_name
    
    # Rest-frame photometry property selection



    # Photometric SNR selection
        
    def phot_bluewards_Lya_non_detect(self, SNR_lim, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, update = True):
        assert(type(SNR_lim) in [int, float])
        selection_name = f"bluewards_Lya_SNR<{SNR_lim:.1f}"
        # only compute this if not already done so
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            if len(self.phot) == 0: # no data at all (not sure why sextractor does this)
                if update:
                    self.selection_flags[selection_name] = False
                return self, selection_name
            # extract bands, SNRs, mask and first Lya non-detect band
            bands = self.phot.instrument.band_names
            SNRs = self.phot.SNR
            mask = self.phot.flux_Jy.mask
            assert(len(bands) == len(SNRs) == len(mask))
            first_Lya_non_detect_band = self.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].phot_rest.first_Lya_non_detect_band
            if first_Lya_non_detect_band == None:
                if update:
                    self.selection_flags[selection_name] = True
                return self, selection_name
            # find index of first Lya non-detect band
            first_Lya_non_detect_index = np.where(bands == first_Lya_non_detect_band)[0][0]
            SNR_non_detect = SNRs[:first_Lya_non_detect_index + 1]
            mask_non_detect = mask[:first_Lya_non_detect_index + 1]
            # require the first Lya non detect band and all bluewards bands to be non-detected at < SNR_lim if not masked
            if all(SNR < SNR_lim or mask for mask, SNR in zip(mask_non_detect, SNR_non_detect)):
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if update:
                    self.selection_flags[selection_name] = False
        return self, selection_name
    
    def phot_redwards_Lya_detect(self, SNR_lims, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, \
            widebands_only = True, update = True):
        # work out selection name based on SNR_lims input type
        if type(SNR_lims) in [int, float]:
            # require all redwards bands to be detected at >SNR_lims
            selection_name = f"ALL_redwards_Lya_SNR>{SNR_lims:.1f}"
            SNR_lims = np.full(len(self.phot.instrument.band_names), SNR_lims)
        elif type(SNR_lims) in [list, np.array]:
            # require the n^th band after the first band redwards of Lya to be detected at >SNR_lims[n]
            assert(np.all([type(SNR) in [int, float] for SNR in SNR_lims]))
            selection_name = f"redwards_Lya_SNR>{','.join([str(np.round(SNR, 1)) for SNR in SNR_lims])}"
        else:
            galfind_logger.critical(f"SNR_lims = {SNR_lims} has type = {type(SNR_lims)} which is not in [int, float, list, np.array]")

        # only compute this if not already done so
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"Already performed {selection_name} for galaxy ID = {self.ID}, skipping!")
        else:
            if len(self.phot) == 0: # no data at all (not sure why sextractor does this)
                if update:
                    self.selection_flags[selection_name] = False
                return self, selection_name
            galfind_logger.debug(f"Performing {selection_name} for galaxy ID = {self.ID}!")
            # extract bands, SNRs, mask and first Lya non-detect band
            bands = self.phot.instrument.band_names
            first_Lya_detect_band = self.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].phot_rest.first_Lya_detect_band
            if first_Lya_detect_band == None:
                if update:
                    self.selection_flags[selection_name] = False
                return self, selection_name
            # find index of first Lya non-detect band
            first_Lya_detect_index = np.where(bands == first_Lya_detect_band)[0][0]
            bands_detect = np.array(bands[first_Lya_detect_index:])
            SNR_detect = np.array(self.phot.SNR[first_Lya_detect_index:])
            mask_detect = np.array(self.phot.flux_Jy.mask[first_Lya_detect_index:])
            # option as to whether to exclude potentially shallower medium/narrow bands in this calculation
            if widebands_only:
                wide_band_detect_indices = [True if "W" in band.upper() or "LP" in band.upper() else False for band in bands_detect]
                SNR_detect = SNR_detect[wide_band_detect_indices]
                mask_detect = mask_detect[wide_band_detect_indices]
                selection_name += "_widebands"
            # selection criteria
            if all(SNR > SNR_lim or mask for mask, SNR, SNR_lim in zip(mask_detect, SNR_detect, SNR_lims)):
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if update:
                    self.selection_flags[selection_name] = False
        return self, selection_name
    
    def phot_Lya_band(self, SNR_lim, detect_or_non_detect = "detect", SED_fit_params = \
            {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, widebands_only = True, update = True):
        assert(type(SNR_lim) in [int, float])
        assert(detect_or_non_detect.lower() in ["detect", "non_detect"], \
            galfind_logger.critical(f"detect_or_non_detect = {detect_or_non_detect} must be either 'detect' or 'non_detect'!"))
        selection_name = f"Lya_band_SNR{'>' if detect_or_non_detect == 'detect' else '<'}{SNR_lim:.1f}"
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            if len(self.phot) == 0: # no data at all (not sure why sextractor does this)
                if update:
                    self.selection_flags[selection_name] = False
                return self, selection_name
            # load bands
            bands = self.phot.instrument.band_names
            # determine Lya band(s) - usually a single band, but could be two in the case of medium bands
            first_Lya_detect_band = self.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].phot_rest.first_Lya_detect_band
            first_Lya_detect_index = np.where(bands == first_Lya_detect_band)[0][0]
            first_Lya_non_detect_band = self.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].phot_rest.first_Lya_non_detect_band
            first_Lya_non_detect_index = np.where(bands == first_Lya_non_detect_band)[0][0]
            # load SNRs, cropping by the relevant bands
            bands_detect = bands[first_Lya_detect_band : first_Lya_non_detect_index + 1]
            if widebands_only:
                wide_band_detect_indices = [True if "W" in band.upper() or "LP" in band.upper() else False for band in bands_detect]
                SNRs = self.phot.SNR[first_Lya_detect_band : first_Lya_non_detect_index + 1][wide_band_detect_indices]
                selection_name += "_widebands"
            else:
                SNRs = self.phot.SNR[first_Lya_detect_band : first_Lya_non_detect_index + 1]
                mask_bands = self.phot.flux_Jy.mask[first_Lya_detect_band : first_Lya_non_detect_index + 1]
            if len(SNRs) == 0:
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if (detect_or_non_detect == "detect" and all(SNR > SNR_lim or mask for SNR, mask in zip(SNRs, mask_bands))) or \
                    (detect_or_non_detect == "non_detect" and all(SNR < SNR_lim or mask for SNR, mask in zip(SNRs, mask_bands))):
                    if update:
                        self.selection_flags[selection_name] = True
                else:
                    if update:
                        self.selection_flags[selection_name] = False
        return self, selection_name

    def phot_SNR_crop(self, band_name_or_index, SNR_lim, detect_or_non_detect = "detect", update = True):
        assert(type(SNR_lim) in [int, float])
        assert(detect_or_non_detect in ["detect", "non_detect"])
        if detect_or_non_detect == "detect":
            sign = ">"
        else: # "non_detect""
            sign = "<"
        if type(band_name_or_index) == str: # band name given
            band_name = band_name_or_index
            # given str must be a valid band in the instrument, even if the galaxy does not have this data
            assert(band_name in self.phot.instrument.new_instrument().band_names)
            # get the index of the band in question
            band_index = np.where(self.phot.instrument.band_names == band_name)[0][0]
            selection_name = f"{band_name}_SNR{sign}{SNR_lim:.1f}"
        elif type(band_name_or_index) == int: # band index of galaxy specific data
            band_index = band_name_or_index
            galfind_logger.debug("Indexing e.g. 2 and -4 when there are 6 bands results in differing behaviour even though the same band is referenced!")
            if band_index == 0:
                selection_name = f"bluest_band_SNR{sign}{SNR_lim:.1f}"
            elif band_index == -1:
                selection_name = f"reddest_band_SNR{sign}{SNR_lim:.1f}"
            elif band_index > 0:
                selection_name = f"{funcs.ordinal(band_index + 1)}_bluest_band_SNR{sign}{SNR_lim:.1f}"
            elif band_index < -1:
                selection_name = f"{funcs.ordinal(abs(band_index))}_reddest_band_SNR{sign}{SNR_lim:.1f}"
        else:
            galfind_logger.critical(f"band_name_or_index = {band_name_or_index} has type = {type(band_name_or_index)} which must be in [str, int]")
        
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            if len(self.phot) == 0: # no data at all (not sure why sextractor does this)
                if update:
                    self.selection_flags[selection_name] = False
                return self, selection_name
            SNR = self.phot.SNR[band_index]
            mask = self.phot.flux_Jy.mask[band_index]
            # passes if masked
            if mask or (detect_or_non_detect == "detect" and SNR > SNR_lim) or \
                (detect_or_non_detect == "non_detect" and SNR < SNR_lim):
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if update:
                    self.selection_flags[selection_name] = False
        return self, selection_name
    
    # Emission line selection functions

    def select_rest_UV_line_emitters_dmag(self, emission_line_name, delta_m, rest_UV_wav_lims = [1_250., 3_000.] * u.AA, \
            medium_bands_only = True, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, update = True):
        assert(line_diagnostics[emission_line_name]["line_wav"] > rest_UV_wav_lims[0] * rest_UV_wav_lims.unit and \
            line_diagnostics[emission_line_name]["line_wav"] < rest_UV_wav_lims[1] * rest_UV_wav_lims.unit  )
        assert(type(delta_m) in [int, np.int64, float, np.float64])
        assert(u.has_physical_type(rest_UV_wav_lims) == "length")
        assert(type(medium_bands_only) in [bool, np.bool_])
        selection_name = f"{emission_line_name},dm{'_med' if medium_bands_only else ''}>{delta_m:.1f},UV_{str(list(np.array(rest_UV_wav_lims.value).astype(int))).replace(' ', '')}AA"
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            if len(self.phot) == 0: # no data at all (not sure why sextractor does this)
                if update:
                    self.selection_flags[selection_name] = False
                return self, selection_name
            phot_rest = deepcopy(self.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].phot_rest)
            # find bands that the emission line lies within
            obs_frame_emission_line_wav = line_diagnostics[emission_line_name]["line_wav"] * (1. + phot_rest.z)
            included_bands = self.phot.instrument.bands_from_wavelength(obs_frame_emission_line_wav)
            # determine index of the closest band to the emission line
            closest_band_index = self.phot.instrument.nearest_band_index_to_wavelength(obs_frame_emission_line_wav, medium_bands_only)
            central_wav = self.phot.instrument[closest_band_index].WavelengthCen
            # if there are no included bands or the closest band is masked
            if len(included_bands) == 0 or self.phot.flux_Jy.mask[closest_band_index]:
                if update:
                    self.selection_flags[selection_name] = False
                return self, selection_name
            # calculate beta excluding the bands that the emission line contaminates
            phot_rest.crop_phot([self.phot.instrument.index_from_band_name(band.band_name) for band in included_bands])
            A, beta = phot_rest.calc_beta_phot(rest_UV_wav_lims, iters = 1)
            # make mock SED to calculate bandpass averaged flux from
            mock_SED_obs = Mock_SED_obs.from_Mock_SED_rest(Mock_SED_rest.power_law_from_beta_m_UV(beta, \
                funcs.power_law_beta_func(1_500., 10 ** A, beta), mag_units = u.Jy, wav_lims = \
                [self.phot.instrument[closest_band_index].WavelengthLower50, \
                self.phot.instrument[closest_band_index].WavelengthUpper50]), self.z, IGM = None)
            mag_cont = funcs.convert_mag_units(central_wav, mock_SED_obs.calc_bandpass_averaged_flux(self.phot.instrument[closest_band_index].wav, \
                self.phot.instrument[closest_band_index].trans) * u.erg / (u.s * (u.cm ** 2) * u.AA), u.ABmag)
            # determine observed magnitude
            mag_obs = funcs.convert_mag_units(central_wav, self.phot[closest_band_index], u.ABmag)
            if (mag_cont - mag_obs).value > delta_m:
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if update:
                    self.selection_flags[selection_name] = False
        return self, selection_name
    
    def select_rest_UV_line_emitters_sigma(self, emission_line_name, sigma, rest_UV_wav_lims = [1_250., 3_000.] * u.AA, \
            medium_bands_only = True, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, update = True) -> tuple[Self, str]:
        assert(line_diagnostics[emission_line_name]["line_wav"] > rest_UV_wav_lims[0] and \
            line_diagnostics[emission_line_name]["line_wav"] < rest_UV_wav_lims[1])
        assert(type(sigma) in [int, np.int64, float, np.float64])
        assert(u.get_physical_type(rest_UV_wav_lims) == "length")
        assert(type(medium_bands_only) in [bool, np.bool_])
        selection_name = f"{emission_line_name},sigma{'_med' if medium_bands_only else ''}>{sigma:.1f},UV_{str(list(np.array(rest_UV_wav_lims.value).astype(int))).replace(' ', '')}AA"
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            if len(self.phot) == 0: # no data at all (not sure why sextractor does this)
                if update:
                    self.selection_flags[selection_name] = False
                return self, selection_name
            SED_results_key = SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)
            phot_rest = deepcopy(self.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].phot_rest)
            # find bands that the emission line lies within
            obs_frame_emission_line_wav = line_diagnostics[emission_line_name]["line_wav"] * (1. + phot_rest.z)
            included_bands = self.phot.instrument.bands_from_wavelength(obs_frame_emission_line_wav)
            # determine index of the closest band to the emission line
            closest_band = self.phot.instrument.nearest_band_to_wavelength(obs_frame_emission_line_wav, medium_bands_only)
            if type(closest_band) == type(None):
                if update:
                    self.selection_flags[selection_name] = False
                return self, selection_name
            closest_band_index = self.phot.instrument.index_from_band_name(closest_band.band_name)
            central_wav = self.phot.instrument[closest_band_index].WavelengthCen
            # if there are no included bands or the closest band is masked
            if len(included_bands) == 0 or self.phot.flux_Jy.mask[closest_band_index]:
                if update:
                    self.selection_flags[selection_name] = False
                return self, selection_name
            # calculate beta excluding the bands that the emission line contaminates
            phot_rest.crop_phot([self.phot.instrument.index_from_band_name(band.band_name) for band in included_bands])
            A, beta = phot_rest.calc_beta_phot(rest_UV_wav_lims, iters = 1, incl_errs = True)
            m_UV = funcs.convert_mag_units(1_500. * (1. + self.phot.SED_results[SED_results_key].z) * u.AA, \
                funcs.power_law_beta_func(1_500., 10 ** A, beta) * u.erg / (u.s * (u.cm ** 2) * u.AA), u.ABmag)
            # make mock SED to calculate bandpass averaged flux from
            mock_SED_rest = Mock_SED_rest.power_law_from_beta_m_UV(beta, m_UV)#, wav_range = \
                #[funcs.convert_wav_units(self.phot.instrument[closest_band_index].WavelengthLower50, u.AA).value / (1. + self.phot.SED_results[SED_results_key].z), \
                #funcs.convert_wav_units(self.phot.instrument[closest_band_index].WavelengthUpper50, u.AA).value / (1. + self.phot.SED_results[SED_results_key].z)] * u.AA)
            mock_SED_obs = Mock_SED_obs.from_Mock_SED_rest(mock_SED_rest, self.phot.SED_results[SED_results_key].z, IGM = None)
            flux_cont = funcs.convert_mag_units(central_wav, mock_SED_obs.calc_bandpass_averaged_flux(self.phot.instrument[closest_band_index].wav, \
                self.phot.instrument[closest_band_index].trans) * u.erg / (u.s * (u.cm ** 2) * u.AA), u.Jy)
            # determine observed magnitude
            flux_obs_err = funcs.convert_mag_err_units(central_wav, self.phot.flux_Jy[closest_band_index], \
                [self.phot.flux_Jy_errs[closest_band_index].value, self.phot.flux_Jy_errs[closest_band_index].value] * self.phot.flux_Jy_errs.unit, u.Jy)
            flux_obs = funcs.convert_mag_units(central_wav, self.phot.flux_Jy[closest_band_index], u.Jy)
            snr_band = abs((flux_obs - flux_cont).value) / np.mean(flux_obs_err.value)
            mag_cont = funcs.convert_mag_units(central_wav, mock_SED_obs.calc_bandpass_averaged_flux(self.phot.instrument[closest_band_index].wav, \
                self.phot.instrument[closest_band_index].trans) * u.erg / (u.s * (u.cm ** 2) * u.AA), u.ABmag)
            print(self.ID, snr_band, beta, mag_cont, self.phot.SED_results[SED_results_key].z, closest_band.band_name)
            if snr_band > sigma:
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if update:
                    self.selection_flags[selection_name] = False
        return self, selection_name
    
    # Colour selection functions

    def select_colour(self, colour_bands, colour_val, bluer_or_redder, update = True):
        assert(bluer_or_redder in ["bluer", "redder"])
        assert(type(colour_bands) in [str, np.str_, list, np.ndarray])
        if type(colour_bands) in [str, np.str_]:
            colour_bands = colour_bands.split("-")
        assert(len(colour_bands) == 2)
        assert(all(colour in self.phot.instrument.band_names for colour in colour_bands))
        # ensure bands are ordered blue -> red
        assert(self.phot.instrument.index_from_band_name(colour_bands[0]) < self.phot.instrument.index_from_band_name(colour_bands[1]))
        selection_name = f"{'-'.join(colour_bands)}{'<' if bluer_or_redder == 'bluer' else '>'}{colour_val:.2f}"
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            if len(self.phot) == 0: # no data at all (not sure why sextractor does this)
                if update:
                    self.selection_flags[selection_name] = False
                return self, selection_name
            # calculate colour
            band_indices = [self.phot.instrument.index_from_band_name(band_name) for band_name in colour_bands]
            colour = (funcs.convert_mag_errs(self.phot.instrument[band_indices[0]].WavelengthCen, self.phot.flux_Jy[band_indices[0]], u.ABmag) \
                - funcs.convert_mag_errs(self.phot.instrument[band_indices[1]].WavelengthCen, self.phot.flux_Jy[band_indices[1]], u.ABmag)).value
            if (colour < colour_val and bluer_or_redder == "bluer") or \
                (colour > colour_val and bluer_or_redder == "redder"):
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if update:
                    self.selection_flags[selection_name] = False
        return self, selection_name
     
    def select_colour_colour(self, colour_bands_arr, colour_select_func):
        pass
    
    def select_UVJ(self, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, \
            quiescent_or_star_forming = "quiescent", update = True):
        assert(quiescent_or_star_forming in ["quiescent", "star_forming"])
        selection_name = f"UVJ_{quiescent_or_star_forming}"
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            if len(self.phot) == 0: # no data at all (not sure why sextractor does this)
                if update:
                    self.selection_flags[selection_name] = False
                return self, selection_name
            # extract UVJ colours -> still need to sort out the units here
            U_minus_V = -2.5 * np.log10((self.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].properties["U_flux"] \
                / self.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].properties["V_flux"]).to(u.dimensionless_unscaled).value)
            V_minus_J = -2.5 * np.log10((self.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].properties["V_flux"] \
                - self.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].properties["J_flux"]).to(u.dimensionless_unscaled).value)
            # selection from Antwi-Danso2022
            is_quiescent = U_minus_V > 1.23 and V_minus_J < 1.67 and U_minus_V > V_minus_J * 0.98 + 0.38
            if (quiescent_or_star_forming == "quiescent" and is_quiescent) or \
                    (quiescent_or_star_forming == "star_forming" and not is_quiescent):
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if update:
                    self.selection_flags[selection_name] = False
        return self, selection_name

    def select_Kokorev24_LRDs(self, update = True):
        selection_name = "Kokorev+24_LRDs"
        if len(self.phot) == 0: # no data at all (not sure why sextractor does this)
                if update:
                    self.selection_flags[selection_name] = False
                return self, selection_name
        red1_selection = \
        [
            self.select_colour(["F115W", "F150W"], 0.8, "bluer")[1], \
            self.select_colour(["F200W", "F277W"], 0.7, "redder")[1], \
            self.select_colour(["F200W", "F356W"], 1.0, "redder")[1]
        ]
        red2_selection = \
        [
            self.select_colour(["F150W", "F200W"], 0.8, "bluer")[1], \
            self.select_colour(["F277W", "F356W"], 0.6, "redder")[1], \
            self.select_colour(["F277W", "F444W"], 0.7, "redder")[1]
        ]
        
        # if the galaxy passes either red1 or red2 colour selection criteria
        if all(self.selection_flags[name] for name in red1_selection) or \
                all(self.selection_flags[name] for name in red2_selection):
            if update:
                self.selection_flags[selection_name] = True
        else:
            if update:
                self.selection_flags[selection_name] = False
        return self, selection_name

    # Depth selection functions

    def select_depth_region(self, band, region_ID, update = True):
        return NotImplementedError

    # chi squared selection functions

    def select_chi_sq_lim(self, chi_sq_lim, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, reduced = True, update = True):
        assert(type(chi_sq_lim) in [int, float])
        assert(type(reduced) == bool)
        if reduced:
            selection_name = f"red_chi_sq<{chi_sq_lim:.1f}"
            n_bands = len([mask_band for mask_band in self.phot.flux_Jy.mask if not mask_band]) # number of unmasked bands for galaxy
            chi_sq_lim *= (n_bands - 1)
        else:
            raise NotImplementedError
            
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            if len(self.phot) == 0: # no data at all (not sure why sextractor does this)
                if update:
                    self.selection_flags[selection_name] = False
                return self, selection_name
            # extract chi_sq
            chi_sq = self.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].chi_sq
            if chi_sq < chi_sq_lim:
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if update:
                    self.selection_flags[selection_name] = False
        return self, selection_name

    def select_chi_sq_diff(self, chi_sq_diff, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, delta_z_lowz = 0.5, update = True):
        assert(type(chi_sq_diff) in [int, float])
        assert(type(delta_z_lowz) in [int, float])
        assert("lowz_zmax" in SED_fit_params.keys())
        assert(SED_fit_params["lowz_zmax"] == None)
        selection_name = f"chi_sq_diff>{chi_sq_diff:.1f},dz>{delta_z_lowz:.1f}"
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            if len(self.phot) == 0: # no data at all (not sure why sextractor does this)
                if update:
                    self.selection_flags[selection_name] = False
                return self, selection_name
            lowz_zmax_arr = [SED_fit_params["lowz_zmax"] for SED_fit_params in self.phot.get_SED_fit_params_arr(SED_fit_params["code"])]
            assert(lowz_zmax_arr[-1] == None and len(lowz_zmax_arr) > 1)
            lowz_zmax_arr = sorted(lowz_zmax_arr[:-1])
            # extract redshift + chi_sq of zfree run
            zfree = self.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].z
            chi_sq_zfree = self.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].chi_sq
            # extract redshift and chi_sq of lowz runs
            z_lowz_arr = []
            chi_sq_lowz_arr = []
            for lowz_zmax in lowz_zmax_arr:
                SED_fit_params_ = deepcopy(SED_fit_params)
                SED_fit_params_["lowz_zmax"] = lowz_zmax
                z_lowz_arr.append(self.phot.SED_results[SED_fit_params_["code"].label_from_SED_fit_params(SED_fit_params_)].z)
                chi_sq_lowz_arr.append(self.phot.SED_results[SED_fit_params_["code"].label_from_SED_fit_params(SED_fit_params_)].chi_sq)
            # determine which lowz run to use for this galaxy
            z_lowz = [z for z, lowz_zmax in zip(z_lowz_arr, lowz_zmax_arr) if zfree > lowz_zmax + delta_z_lowz]
            chi_sq_lowz = [chi_sq for chi_sq, lowz_zmax in zip(chi_sq_lowz_arr, lowz_zmax_arr) if zfree > lowz_zmax + delta_z_lowz]
            if len(chi_sq_lowz) == 0:
                if update:
                    self.selection_flags[selection_name] = True
            elif (chi_sq_lowz[-1] - chi_sq_zfree > chi_sq_diff) or (chi_sq_lowz[-1] == -1.) or (z_lowz[-1] < 0.):
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if update:
                    self.selection_flags[selection_name] = False
        return self, selection_name
    
    # z-PDF selection functions

    def select_robust_zPDF(self, integral_lim, delta_z_over_z, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, update = True):
        assert(type(integral_lim) == float and (integral_lim * 100).is_integer())
        assert(type(delta_z_over_z) in [int, float])
        if "lowz_zmax" in SED_fit_params.keys():
            assert(SED_fit_params["lowz_zmax"] == None)
        selection_name = f"zPDF>{int(integral_lim * 100)}%,|dz|/z<{delta_z_over_z}"
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            if len(self.phot) == 0: # no data at all (not sure why sextractor does this)
                if update:
                    self.selection_flags[selection_name] = False
                return self, selection_name
            # extract best fitting redshift - peak of the redshift PDF
            zbest = self.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].z
            if zbest < 0.:
                if update:
                    self.selection_flags[selection_name] = False
            else:
                integral = self.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].property_PDFs["z"].integrate_between_lims(float(zbest * delta_z_over_z), float(zbest))
                if integral > integral_lim:
                    if update:
                        self.selection_flags[selection_name] = True
                else:
                    if update:
                        self.selection_flags[selection_name] = False
        return self, selection_name
    
    # Morphology selection functions
    
    def select_band_flux_radius(self, band: str, gtr_or_less: str, lim: Union[int, float, u.Quantity], update: bool = True):
        assert type(band) == str
        assert gtr_or_less in ["gtr", "less"]
        if type(lim) != u.Quantity:
            lim_str = f"{lim:.1f}pix"
        elif lim.unit in u.dimensionless_unscaled:
            lim_str = f"{lim.value:.1f}pix"
        else:
            lim_str = f"{lim.to(u.arcsec).value:.1f}as"
        selection_name = f"Re_{band}{'>' if gtr_or_less == 'gtr' else '<'}{lim_str}"
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
            if len(self.phot) == 0 or band not in self.phot.instrument.band_names: # no data
                if update:
                    self.selection_flags[selection_name] = False
                return self, selection_name
            if self.sex_Re[band] > lim:
                if update:
                    self.selection_flags[selection_name] = True
            else:
                if update:
                    self.selection_flags[selection_name] = False
        return self, selection_name
    
    def select_EPOCHS(self, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, allow_lowz = False, \
            hot_pixel_bands = ["F277W", "F356W", "F444W"], mask_instrument = NIRCam(), update = True):
        
        selection_name = f"EPOCHS{'_lowz' if allow_lowz else ''}"
        if len(self.phot) == 0: # no data at all (not sure why sextractor does this)
                if update:
                    self.selection_flags[selection_name] = False
                return self, selection_name
        if not "NIRCam" in self.phot.instrument.name:
            galfind_logger.critical(f"NIRCam data for galaxy ID = {self.ID} must be included for EPOCHS selection!")
            if update:
                self.selection_flags[selection_name] = False
            return self, selection_name
        
        selection_names = [
            self.select_unmasked_instrument(mask_instrument)[1], # unmasked in all NIRCam bands
            self.phot_bluewards_Lya_non_detect(2., SED_fit_params)[1], # 2σ non-detected in all bands bluewards of Lyα
            self.phot_redwards_Lya_detect([5., 3.], SED_fit_params, widebands_only = True)[1], # 5σ/3σ detected in first/second band redwards of Lyα
            self.select_chi_sq_lim(3., SED_fit_params, reduced = True)[1], # χ^2_red < 3
            self.select_chi_sq_diff(9., SED_fit_params, delta_z_lowz = 0.5)[1], # Δχ^2 > 9 between redshift free and low redshift SED fits, with Δz=0.5 tolerance 
            self.select_robust_zPDF(0.6, 0.1, SED_fit_params)[1] # 60% of redshift PDF must lie within z ± z * 0.1
        ]

        # hot pixel checks
        for band_name in hot_pixel_bands:
            if band_name in self.phot.instrument.band_names:
                selection_names.append(self.select_band_flux_radius(band_name, "gtr", 1.5)[1]) # LW NIRCam wideband Re>1.5 pix

        if not allow_lowz:
            selection_names.append(self.phot_SNR_crop(0, 2., "non_detect")[1]) # 2σ non-detected in first band

        try:
            self.selection_flags["Re_F277W>1.5pix"]
        except:
            breakpoint()
            
        # if the galaxy passes all criteria
        if all(self.selection_flags[name] for name in selection_names):
            if update:
                self.selection_flags[selection_name] = True
        else:
            if update:
                self.selection_flags[selection_name] = False
        return self, selection_name
    
    def select_combined(self):
        pass
    
    # Rest-frame SED photometric properties

    def _calc_SED_rest_property(self, SED_rest_property_function, SED_fit_params_label, *args):
        # calculate and save parameter
        SED_rest_property_function(self.phot.SED_results[SED_fit_params_label].phot_rest, *args)
        return self
    
    def _save_SED_rest_PDFs(self, property_name, save_dir, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}):
        key = SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)
        save_path = f"{save_dir}/{key}/{property_name}/{self.ID}.ecsv"
        funcs.make_dirs(save_path)
        if type(self.phot.SED_results[key].phot_rest.property_PDFs[property_name]) != type(None):
            self.phot.SED_results[key].phot_rest.property_PDFs[property_name].save_PDF(save_path)
        
    def _load_SED_rest_properties(self, PDF_dir, property_names, SED_fit_params_label = EAZY().label_from_SED_fit_params({"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None})):
        # determine which properties have already been calculated
        PDF_paths = [f"{PDF_dir}/{property_name}/{self.ID}.ecsv" for property_name in property_names]
        for PDF_path, property_name in zip(PDF_paths, property_names):
            self.phot.SED_results[SED_fit_params_label].phot_rest.property_PDFs[property_name] = PDF.from_ecsv(PDF_path)
            self.phot.SED_results[SED_fit_params_label].phot_rest._update_properties_from_PDF(property_name)
        return self
    
    def _del_SED_rest_properties(self, property_names, SED_fit_params_label = EAZY().label_from_SED_fit_params({"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None})):
        for property_name in property_names:
            self.phot.SED_results[SED_fit_params_label].phot_rest.property_PDFs.pop(property_name)
            self.phot.SED_results[SED_fit_params_label].phot_rest.properties.pop(property_name)
            self.phot.SED_results[SED_fit_params_label].phot_rest.property_errs.pop(property_name)
        return self
    
    def _get_SED_rest_property_names(self, PDF_dir):
        PDF_paths = glob.glob(f"{PDF_dir}/*/{self.ID}.ecsv")
        return [path.split("/")[-2] for path in PDF_paths]

class Multiple_Galaxy:
    
    def __init__(self, sky_coords, IDs, phots, mask_flags_arr, selection_flags_arr, timed = True):
        if timed:
            self.gals = [Galaxy(sky_coord, ID, phot, mask_flags, selection_flags) for sky_coord, ID, phot, mask_flags, selection_flags \
                in tqdm(zip(sky_coords, IDs, phots, mask_flags_arr, selection_flags_arr), desc = "Initializing galaxy objects", total = len(sky_coords))]
        else:
            self.gals = [Galaxy(sky_coord, ID, phot, mask_flags, selection_flags) for \
                sky_coord, ID, phot, mask_flags, selection_flags in zip(sky_coords, IDs, phots, mask_flags_arr, selection_flags_arr)]
        
    def __repr__(self):
        # string representation of what is stored in this class
        return str(self.__dict__)
    
    def __len__(self):
        return len(self.gals)
    
    def __iter__(self):
        self.iter = 0
        return self
    
    def __next__(self):
        if self.iter > len(self) - 1:
            raise StopIteration
        else:
            gal = self[self.iter]
            self.iter += 1
            return gal
    
    def __getitem__(self, index):
        return self.gals[index]
    
    @classmethod
    def from_fits_cat(cls, fits_cat, instrument, cat_creator, SED_fit_params_arr, timed = True):
        # load photometries from catalogue
        phots = Multiple_Photometry_obs.from_fits_cat(fits_cat, instrument, cat_creator, SED_fit_params_arr, timed = timed).phot_obs_arr
        # load the ID and Sky Coordinate from the source catalogue
        IDs = np.array(fits_cat[cat_creator.ID_label]).astype(int)
        # load sky co-ordinates
        sky_coords = SkyCoord(fits_cat[cat_creator.ra_dec_labels["RA"]], fits_cat[cat_creator.ra_dec_labels["DEC"]], frame = "icrs")
        # mask flags should come from cat_creator
        #mask_flags_arr = [{f"unmasked_{band}": cat_creator.load_flag(fits_cat_row, f"unmasked_{band}") for band in instrument.band_names} for fits_cat_row in fits_cat]
        mask_flags_arr = [{} for fits_cat_row in fits_cat] #f"unmasked_{band}": None for band in instrument.band_names
        selection_flags_arr = [{selection_flag: bool(fits_cat_row[selection_flag]) for selection_flag in cat_creator.selection_labels(fits_cat)} for fits_cat_row in fits_cat]
        return cls(sky_coords, IDs, phots, mask_flags_arr, selection_flags_arr)
    
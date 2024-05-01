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
from pathlib import Path
from astropy.nddata import Cutout2D
from tqdm import tqdm
import matplotlib.patheffects as pe
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.visualization import LogStretch, LinearStretch, ImageNormalize, ManualInterval
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from . import useful_funcs_austind as funcs
from . import config, galfind_logger, astropy_cosmo
from . import Photometry_rest, Photometry_obs, Multiple_Photometry_obs, Data, Instrument, NIRCam, ACS_WFC, WFC3_IR
from .EAZY import EAZY

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

    def update_mask(self, cat, catalogue_creator, update_phot_rest = False):
        self.phot.update_mask(cat, catalogue_creator, self.ID, update_phot_rest = update_phot_rest)
        return self
        
    # def update_mask_band(self, band, bool_value):
    #     self.mask_flags[band] = bool_value

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
                rerun = True
        if not Path(out_path).is_file() or config.getboolean("Cutouts", "OVERWRITE_CUTOUTS") or rerun:
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
                cutout = Cutout2D(data_i, self.sky_coord, size = (cutout_size, cutout_size), wcs = wcs)
                im_header.update(cutout.wcs.to_header())
                hdul.append(fits.ImageHDU(cutout.data, header = im_header, name = label_i))
            #print(hdul)
            os.makedirs("/".join(out_path.split("/")[:-1]), exist_ok = True)
            fits_hdul = fits.HDUList(hdul)
            fits_hdul.writeto(out_path, overwrite = True)
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
                # Run trilogy
                sys.path.insert(1, "/nvme/scratch/software/trilogy") # Not sure why this path doesn't work: config["Other"]["TRILOGY_DIR"]
                from trilogy3 import Trilogy
                galfind_logger.info(f"Making trilogy cutout RGB at {out_path}")
                Trilogy(in_path, images = None).run()
            elif method == "lupton":
                raise(NotImplementedError())
    
    def plot_cutouts(self, ax_arr, data, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, \
            hide_masked_cutouts = True, cutout_size = 32, high_dyn_rng = False):

        for i, band in enumerate(data.instrument.band_names):
                
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
                scalebar = AnchoredSizeBar(ax_arr[i].transData,
                    0.3 / data.im_pixel_scales[band].value, "0.3\"", 'lower right', 
                    pad = 0.3, color='white', frameon=False, size_vertical=2)
                ax_arr[-1].add_artist(scalebar)
                # Plot scalebar with physical size
                scalebar = AnchoredSizeBar(ax_arr[-1].transData,
                    re, f"{re_kpc:.1f}", 'upper left', pad=0.3, color='white',
                    frameon=False, size_vertical=1.5)
                ax_arr[-1].add_artist(scalebar)
    
    def plot_phot_diagnostic(self, ax, data, SED_fit_params_arr, zPDF_plot_SED_fit_params_arr, wav_unit = u.AA, flux_unit = u.ABmag, \
            scaling = {}, hide_masked_cutouts = True, cutout_size = 32, high_dyn_rng = False, overwrite = True):
        
        cutout_ax, phot_ax, PDF_ax = ax

        # update SED_fit_params with appropriate lowz_zmax
        SED_fit_params_arr = [SED_fit_params["code"].update_lowz_zmax(SED_fit_params, self.phot.SED_results) for SED_fit_params in SED_fit_params_arr]
        zPDF_plot_SED_fit_params_arr = [SED_fit_params["code"].update_lowz_zmax(SED_fit_params, self.phot.SED_results) for SED_fit_params in zPDF_plot_SED_fit_params_arr] 
        
        zPDF_labels = [f"{SED_fit_params['code'].label_from_SED_fit_params(SED_fit_params)} PDF" for SED_fit_params in zPDF_plot_SED_fit_params_arr]
        # reset parameters
        if "x" in scaling.keys():
            phot_ax.set_xscale(scaling["x"])
        #phot_ax.set_xlabel(x_label)
        #phot_ax.set_ylabel(y_label)
        PDF_ax[1].set_yticks([])
        PDF_ax[1].set_visible(True)
        PDF_ax[0].set_yticks([])
        PDF_ax[0].set_title(zPDF_labels[0], fontsize = "medium") # lowz_zmax = None eazy run is first
        PDF_ax[0].set_xlabel("Redshift, z") # this label should again be pulled from somewhere else
        
        out_path = f"{config['Selection']['SELECTION_DIR']}/SED_plots/{data.version}/{data.instrument.name}/{data.survey}/{self.ID}.png"
        funcs.make_dirs(out_path)

        if not Path(out_path).is_file() or overwrite:
            # plot cutouts (assuming reference SED_fit_params is at 0th index)
            self.plot_cutouts(cutout_ax, data, SED_fit_params_arr[0], \
                hide_masked_cutouts = hide_masked_cutouts, cutout_size = cutout_size, high_dyn_rng = high_dyn_rng)
                    
            # auto-scale based on available bands (wavelength) and flux/mag values
            if "x" in scaling.keys():
                if scaling["x"] == "log":
                    phot_ax.set_xlim((0.8 * u.um).to(wav_unit).value, (5 * u.um).to(wav_unit).value)
                else:
                    raise NotImplementedError # also not asserted anywhere
            else:
                if 'MIRI' in data.instrument.name:
                    #print('Adjusting xlims for MIRI')
                    phot_ax.set_xlim((0.3 * u.um).to(wav_unit).value, (8.5 * u.um).to(wav_unit).value)
                else:
                    phot_ax.set_xlim((0.3 * u.um).to(wav_unit).value, (5 * u.um).to(wav_unit).value)
            # this should not be hard-coded
            phot_ax.set_ylim((30.6 * u.ABmag).to(flux_unit).value, (25 * u.ABmag).to(flux_unit).value)
                    
            self.phot.plot_phot(phot_ax, wav_units = wav_unit, mag_units = flux_unit, annotate = False, upper_limit_sigma = 2., label_SNRs = True)
            
            # save rejected reasons somewhere
            # if show_rejected_reason:
            #     rejected = str(row[f'rejected_reasons{col_ext}'][0])
            #     if rejected != '':
            #         phot_ax.annotate(rejected, (0.9, 0.95), ha='center', fontsize='small', xycoords = 'axes fraction', zorder=5)
                    
            # plot specified SEDs
            for SED_fit_params in SED_fit_params_arr:
                self.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].SED.plot_SED(phot_ax, wav_unit, flux_unit)
                # could also plot the expected photometry here as well
                #ax_photo.scatter(band_wavs_lowz, band_mags_lowz, edgecolors=eazy_color_lowz, marker='o', facecolor='none', s=80, zorder=4.5)

            # photometry axis legend
            phot_ax.legend(loc='upper left', fontsize='small', frameon=False)
            for text in phot_ax.get_legend().get_texts():
                text.set_path_effects([pe.withStroke(linewidth = 3, foreground = 'white')])
                text.set_zorder(12)

            # plot PDF on relevant axis
            assert(len(zPDF_plot_SED_fit_params_arr) == len(PDF_ax)) # again, this is not totally generalized and should be == 2 for now
            # could extend to plotting multiple PDFs on the same axis
            for ax, SED_fit_params in zip(PDF_ax, zPDF_plot_SED_fit_params_arr):
                self.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].property_PDFs["z"].plot(ax)

            # Save and clear axes
            plt.savefig(out_path, dpi = 300, bbox_inches = 'tight')
            for ax in [phot_ax] + PDF_ax + cutout_ax:
                ax.cla()

    def plot_spec_diagnostic(self, overwrite = True):
        # bare in mind that not all galaxies have spectroscopic data
        pass

    # %% Selection methods

    def select_min_unmasked_bands(self, min_bands, update = True):

        if type(min_bands) != int:
            min_bands = int(min_bands)

        selection_name = f"unmasked_bands>={min_bands}"
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
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
    
    def select_unmasked_instrument(self, instrument, update = True):
        assert(issubclass(instrument.__class__, Instrument))
        assert(instrument.__class__.__name__ in self.phot.instrument.name.split("+"))

        selection_name = f"unmasked_{instrument.__class__.__name__}"
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
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
        
    def phot_bluewards_Lya_non_detect(self, SNR_lim, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, update = True):
        assert(type(SNR_lim) in [int, float])
        selection_name = f"bluewards_Lya_SNR<{SNR_lim:.1f}"
        # only compute this if not already done so
        if selection_name in self.selection_flags.keys():
            galfind_logger.debug(f"{selection_name} already performed for galaxy ID = {self.ID}!")
        else:
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
            lowz_zmax_arr = [SED_fit_params["lowz_zmax"] for SED_fit_params in self.phot.get_SED_fit_params_arr(SED_fit_params["code"])]
            assert(lowz_zmax_arr[-1] == None and len(lowz_zmax_arr) > 1)
            lowz_zmax_arr = sorted(lowz_zmax_arr[:-1])
            # extract redshift + chi_sq of zfree run
            zfree = self.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].z
            chi_sq_zfree = self.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].chi_sq
            # extract redshift and chi_sq of lowz runs
            z_lowz_arr = []
            chi_sq_lowz_arr = []
            for zmax in lowz_zmax_arr:
                SED_fit_params_ = deepcopy(SED_fit_params)
                SED_fit_params_["lowz_zmax"] = zmax
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
    
    def select_EPOCHS(self, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, mask_instrument = NIRCam(), update = True):
        
        selection_name = "EPOCHS"
        if not "NIRCam" in self.phot.instrument.name:
            galfind_logger.critical(f"NIRCam data for galaxy ID = {self.ID} must be included for EPOCHS selection!")
            if update:
                self.selection_flags[selection_name] = False
            return self, selection_name
        
        selection_names = [
            self.select_unmasked_instrument(mask_instrument)[1], # unmasked in NIRCam
            self.phot_SNR_crop(0, 2., "non_detect")[1], # 2σ non-detected in first band
            self.phot_bluewards_Lya_non_detect(2., SED_fit_params)[1], # 2σ non-detected in all bands bluewards of Lyα
            self.phot_redwards_Lya_detect([5., 3.], SED_fit_params, widebands_only = True)[1], # 5σ/3σ detected in first/second band redwards of Lyα
            self.select_chi_sq_lim(3., SED_fit_params, reduced = True)[1], # χ^2_red < 3
            self.select_chi_sq_diff(9., SED_fit_params, delta_z_lowz = 0.5)[1], # Δχ^2 < 9 between redshift free and low redshift SED fits, with Δz=0.5 tolerance 
            self.select_robust_zPDF(0.6, 0.1, SED_fit_params)[1] # 60% of redshift PDF must lie within z ± z * 0.1
        ]
        # masking criteria
        # unmasked in first band
        # SNR criteria
        # if galaxy is detected only in the LW filters
        # first_Lya_detect_band = self.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].phot_rest.first_Lya_detect_band
        #band_names = self.phot.instrument.band_names
        #first_band = self.phot.instrument[np.where()]
        # 7σ/5σ detected in 1st/2nd bands redwards of Lya
        # else
        # 5σ/3σ detected in 1st/2nd bands redwards of Lya
        
        # if the galaxy passes all criteria
        if all(self.selection_flags[name] for name in selection_names):
            if update:
                self.selection_flags[selection_name] = True
        else:
            if update:
                self.selection_flags[selection_name] = False
        return self, selection_name

class Multiple_Galaxy:
    
    def __init__(self, sky_coords, IDs, phots, mask_flags_arr, selection_flags_arr):
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
    def from_fits_cat(cls, fits_cat, instrument, cat_creator, SED_fit_params_arr):
        # load photometries from catalogue
        phots = Multiple_Photometry_obs.from_fits_cat(fits_cat, instrument, cat_creator, SED_fit_params_arr).phot_obs_arr
        # load the ID and Sky Coordinate from the source catalogue
        IDs = np.array(fits_cat[cat_creator.ID_label]).astype(int)
        # load sky co-ordinate one at a time (can improve efficiency here)
        sky_coords = [SkyCoord(ra * u.deg, dec * u.deg, frame = "icrs") \
            for ra, dec in zip(fits_cat[cat_creator.ra_dec_labels["RA"]], fits_cat[cat_creator.ra_dec_labels["DEC"]])]
        # mask flags should come from cat_creator
        #mask_flags_arr = [{f"unmasked_{band}": cat_creator.load_flag(fits_cat_row, f"unmasked_{band}") for band in instrument.band_names} for fits_cat_row in fits_cat]
        mask_flags_arr = [{} for fits_cat_row in fits_cat] #f"unmasked_{band}": None for band in instrument.band_names
        selection_flags_arr = [{selection_flag: bool(fits_cat_row[selection_flag]) for selection_flag in cat_creator.selection_labels(fits_cat)} for fits_cat_row in fits_cat]
        return cls(sky_coords, IDs, phots, mask_flags_arr, selection_flags_arr)
    
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
from astropy.wcs import WCS

from . import useful_funcs_austind as funcs
from . import config, galfind_logger
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
                print("Cutout size does not match requested size, overwriting...")
                rerun = True
        if not Path(out_path).is_file() or config.getboolean("Cutouts", "OVERWRITE_CUTOUTS") or rerun:
            if type(data) == Data:
                im_data, im_header, seg_data, seg_header = data.load_data(band, incl_mask = False)
                wht_data = data.load_wht(band)
                rms_err_data = data.load_rms_err(band)
                data = {"SCI": im_data, "SEG": seg_data, "WHT": wht_data, "RMS_ERR": rms_err_data}
                wcs = WCS(im_header)
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
            galfind_logger.info(f"Already made fits cutout for {survey} {version} {self.ID}")
            # load cutout - NotImplementedError
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
    
    def plot_phot_diagnostic(self, data, SED_fit_params_arr, wav_unit = u.AA, flux_unit = u.ABmag, \
            scaling = {}, hide_masked_cutouts = True, overwrite = True):
        # Tom's SED plotting code here - currently trimming this down

        assert(len(SED_fit_params_arr) == 2) # again, this is not totally generalized - in the galaxy function though

        # Should go in catalogue function until the next relevant statement saying otherwise!!!
        # figure size may well depend on how many bands there are
        overall_fig = plt.figure(figsize = (8, 7), constrained_layout=True)
        bands = self.phot.instrument.band_names # should be updated
        nbands = len(bands) # should be updated
        subfigures = overall_fig.subfigures(2, 1, hspace = -2, height_ratios = [2, 1] if len(bands) <= 8 else [1.8, 1])
        fig = subfigures[0]
        cutout_fig = subfigures[1]
    
        gs = fig.add_gridspec(2, 4)
        phot_ax = fig.add_subplot(gs[:, 0:3])
        if flux_unit != u.Jy:
            galfind_logger.critical("Convert unit here!!!")
        galfind_logger.critical("Still need to appropriately label x/y axis")
        x_label = r"$\lambda_{\mathrm{obs}}~/~\mathrm{\AA}$"
        y_label = r"$\f_{\nu}~/~\mathrm{Jy}$" #funcs.unit_to_label(self.phot.flux_Jy.unit)
        #y_label = f'Flux Density ({flux_unit:latex})'
        #x_label = f'Wavelength ({wav_unit:latex})'

        ax_eazy_lowz_pdf = fig.add_subplot(gs[0, 3:])
        #ax_eazy_lowz_pdf.set_title(f'{SED_code.split("+")[0]} z$\leq$7 PDF', fontsize='medium')
        #ax_eazy_lowz_pdf.set_yticklabels([])
        ax_eazy_pdf = fig.add_subplot(gs[1, 3:])
        
        if nbands <= 8:
            gridspec_cutout = cutout_fig.add_gridspec(1, nbands)
        else:
            gridspec_cutout = cutout_fig.add_gridspec(2, int(np.ceil(nbands / 2)))
        # Get list of gridspec positons
        
        cutout_ax_list = []
        for pos, band in enumerate(bands):
            cutout_ax = cutout_fig.add_subplot(gridspec_cutout[pos])
            #cutout_ax.set_xlabel(band, fontsize='small')
            cutout_ax.set_aspect('equal', adjustable='box', anchor='N')
            cutout_ax.set_xticks([])
            cutout_ax.set_yticks([])
            cutout_ax_list.append(cutout_ax)
        
        # GALAXY FUNCTION!!! (here down)
        
        PDF_labels = [f"{SED_fit_params['code'].label_from_SED_fit_params(SED_fit_params)} PDF" for SED_fit_params in SED_fit_params_arr]
        # reset parameters
        if "x" in scaling.keys():
            phot_ax.set_xscale(scaling["x"])
        phot_ax.set_xlabel(x_label)
        phot_ax.set_ylabel(y_label)
        ax_eazy_lowz_pdf.set_yticks([])
        ax_eazy_lowz_pdf.set_visible(True)
        ax_eazy_pdf.set_yticks([])
        ax_eazy_pdf.set_title(PDF_labels[0], fontsize = "medium") # lowz_zmax = None eazy run is first
        ax_eazy_pdf.set_xlabel("Redshift, z") # this label should again be pulled from somewhere else
        
        out_path = f"{config['Selection']['SELECTION_DIR']}/SED_plots/{data.version}/{data.instrument.name}/{data.survey}/{self.ID}.png"
        funcs.make_dirs(out_path)

        if not Path(out_path).is_file() or overwrite:
            
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
                    phot_ax.set_xlim((0.3*u.um).to(wav_unit).value, (5 * u.um).to(wav_unit).value)
            phot_ax.set_ylim(30.6 * u.ABmag.to(flux_unit), 25 * u.ABmag.to(flux_unit))
            
            redshift_set = False
            # add spec-z information if available
            if hasattr(self, "spec"):
                # print this somewhere else rather than the title
                name +=  f'z$_{{spec}}$:{self.spec.z:.2f}' # self.spec may also be an array depending on how this is set up
            
            # should print z_p either way - probably not in the title though

            # should print specified selection booleans somewhere

            for pos, band in enumerate(bands):
                
                wcs = wcs_im[band] # I think this is a function in Data somethwhere
                # this should probably be a function somewhere!
                pixel_coords = skycoord_to_pixel(self.sky_coord, wcs)
                
                x, y = pixel_coords[0][0], pixel_coords[1][0]
                # load cutout if already made, else produce one
                data = np.array(fits_images[band].section[int(y-cutout_size/2):int(y+cutout_size/2), int(x-cutout_size/2):int(x+cutout_size/2)])
                    
                # need to load sextractor flux_radius as a general function somewhere!
                flux_radius = None
                radius = 0.16
                pixel_scale = 0.03
                radius_pix = radius/pixel_scale
                radius_sextractor = flux_radius
            
                try:
                    data_cutout = data
                    #data_cutout = Cutout2D(data, sky_pos, wcs=wcs, size = size).data

                    skip = False
                except (NoOverlapError, ValueError):
                    data_cutout = None
                    bands = [i for i in bands if i  != band]
                    skip = True

                # Skip if in mask
                if not row[f'unmasked_{band.upper()}']:
                    skip = True

                if overrule_unmasked:
                    skip = False

                if not skip:
                    # Set top value based on central 10x10 pixel region
                    top = np.max(data_cutout[:20, 10:20])
                    top = np.max(data_cutout[int(cutout_size//2-0.3*cutout_size):int(cutout_size//2+0.3*cutout_size),int(cutout_size//2-0.3*cutout_size):int(cutout_size//2+0.3*cutout_size)])
                    #bottom_val = 0.0001
                    bottom_val = top/10**5
                    
                    if high_dyn_rng:
                        a = 300
                    else:
                        a = 0.1
                    stretch = LogStretch(a=a)

                    if n_sig_detect < 100:
                        bottom_val = top / 10 ** 3
                        a = 100
                    if n_sig_detect <= 15:
                        bottom_val = top/10**2
                        a = 0.1
                    if n_sig_detect < 8:
                        bottom_val = top/100000
                        stretch = LinearStretch()
                    
                        
                    #data_cutout = np.clip(data_cutout*0.9999, bottom_val*1.000001, top)
                    #one_sig = useful_funcs_updated_new_galfind.mag_to_flux_mjy_sr(useful_funcs_updated_new_galfind.five_sig_depth_to_n_sig_depth( depth, 1))/area_pix
                    #print(one_sig)
                    norm = ImageNormalize(data_cutout, interval=ManualInterval(bottom_val, top), clip=True, stretch=stretch)
                    #print(top)
                    #print(np.min(data_cutout))
                    use_orig = False
                    origin='lower' 

                    cutout_ax_list[pos].cla()
                    #cutout_ax_list[pos].set_xlabel(band, fontsize='small')
                    cutout_ax_list[pos].set_visible(True)
                    cutout_ax_list[pos].set_aspect('equal', adjustable='box', anchor='N')
                    cutout_ax_list[pos].set_xticks([])
                    cutout_ax_list[pos].set_yticks([])

                    cutout_ax_list[pos].imshow(data_cutout, norm=norm if not use_orig else None, cmap='magma',origin=origin )
                    cutout_ax_list[pos].text(0.95, 0.95, band.upper(), fontsize='small', c='white', transform=cutout_ax_list[pos].transAxes, ha='right', va='top', zorder=10, fontweight='bold')         
                    #cutout_ax_list[pos].plot(y_coord, x_coord)
                    xpos = np.mean(cutout_ax_list[pos].get_xlim())
                    ypos = np.mean(cutout_ax_list[pos].get_ylim())
                    #print(f"radius_pix = {radius_pix} for ID = {id}")
                    region = patches.Circle((xpos, ypos), radius_pix, fill=False, linestyle='--', lw=1, color='white', zorder=20)
                    
                    if pos == len(bands)-1:
                    
                        scalebar = AnchoredSizeBar(cutout_ax_list[pos].transData,
                                                    0.3/pixel_scale, "0.3\"", 'lower right', 
                                                    pad=0.3,
                                                    color='white',
                                                    frameon=False,
                                                    size_vertical=2,
                                                    )
                        cutout_ax_list[pos].add_artist(scalebar)

                    region_sextractor = patches.Circle((xpos, ypos), radius_sextractor, fill=False, linestyle='--', lw=1, color='blue', zorder=20)
                    cutout_ax_list[pos].add_patch(region)
                    if radius_sextractor != 0:
                        cutout_ax_list[pos].add_patch(region_sextractor)
                else:
                    cutout_ax_list[pos].cla()
                    cutout_ax_list[pos].set_visible(False)
            all_mags = []
            for pos, band in enumerate(bands):
                skip = False
                try:
                    # Don't plot photometry if galaxy masked in a band (or photometry is measured as NaN or perfect 0s)
                    
                    if row[f'FLUX_APER_{band.upper()}'][0][0] in [np.NaN, 0, 0.0, '-']:
                        skip = True
                    
                    if row[f'unmasked_{band.upper()}'] == False:
                        skip = True
                    
                    if overrule_unmasked:
                        skip = False

                    mag_list = row[f'MAG_APER_{band.upper()}_aper_corr'][0]
                    
                except KeyError:
                    print(f'No aperture corrected {band} mags found, falling back to unaperture corrected')
                    mag_list = row[f'MAG_APER_{band.upper()}'][0]
                if not skip:
                    if type(mag_list) in [list, np.ndarray, np.ma.core.MaskedArray]:
                        mag = mag_list[0] * u.ABmag  
                        
                    else:
                        print('Doing else')
                        mag = mag_list * u.ABmag
                    
                    try:
                        mag_err_list = row[f'MAGERR_APER_{band.upper()}_l1_loc_depth'][0], row[f'MAGERR_APER_{band.upper()}_u1_loc_depth'][0]
                        asym = True
                    except KeyError:
                        try:
                            mag_err_list = row[f'MAGERR_APER_{band.upper()}_loc_depth'][0]
                            asym = False
                        except KeyError:
                            try:
                                mag_err_list = row[f'MAGERR_APER_{band.upper()}'][0]
                                asym=False
                            except KeyError:
                                mag_err_list = 0
                                asym=False
                    if type(mag_err_list) in [list, np.ndarray,tuple, np.ma.core.MaskedArray]:
                        if asym:
                            mag_err = np.array(mag_err_list)[:, 0].reshape(2,1) * u.ABmag
                        else:
                            mag_err = mag_err_list[0] * u.ABmag
                    else:
                        mag_err = mag_err_list * u.ABmag
                    
                    wav = useful_funcs_updated_new_galfind.band_wavelengths[band] 
                
                    #print('list',wav.to(wav_unit).value, log.five_to_n_sigma_mag_err(row[f'loc_depth_{band}'][0],2), wav.to(wav_unit).value, ax_photo.get_xlim()[0])
                    try:
                        loc_depth = float(row[f'loc_depth_{band.upper()}'][0][0])
                    except KeyError:
                        print('Using average depths.')
                        loc_depth = float(avg_depth[field_name.split("_")[-1]][band])
                    except IndexError:
                        loc_depth = float(row[f'loc_depth_{band.upper()}'][0])
    
                    two_sig_depth = useful_funcs_updated_new_galfind.five_sig_depth_to_n_sig_depth(loc_depth, 2)*u.ABmag
                
                    if mag > two_sig_depth:
                        three_sig_depth = useful_funcs_updated_new_galfind.five_sig_depth_to_n_sig_depth(loc_depth, 3)*u.ABmag
                        p1 = patches.FancyArrowPatch((wav.to(wav_unit).value, three_sig_depth.to(flux_unit).value), (wav.to(wav_unit).value, three_sig_depth.to(flux_unit).value+0.5), arrowstyle='-|>', mutation_scale=10, alpha=1, color='black', zorder=5.6)
                        
                        ax_photo.add_patch(p1)
                    else:
                        xerr = useful_funcs_updated_new_galfind.band_wavelength_errs[band]
                        ax_photo.errorbar(wav.to(wav_unit).value, mag.to(flux_unit).value, yerr=mag_err.to(flux_unit).value, xerr=xerr.to(wav_unit).value, color='black', markersize=5, marker='o', zorder=5.6)
                    
                    #print(mag.value, ax_photo.get_ylim()[1])
                    all_mags.append(mag.value)
                    if mag.value < ax_photo.get_ylim()[1] + 1 and mag.value > 15: 
                        new_lim = mag.value - 1
                        ax_photo.set_ylim(ax_photo.get_ylim()[0], new_lim)
                    #print(mag.value, ax_photo.get_ylim()[1])
                    
                    if mag.value > ax_photo.get_ylim()[0] and mag.value < 32 and mag.value > 15:
                        new_lim = two_sig_depth.value + 0.5
                        ax_photo.set_ylim(new_lim, ax_photo.get_ylim()[1])
                        
                    #one_sig_depth = useful_funcs_updated_new_galfind.mag_to_flux_mjy_sr(useful_funcs_updated_new_galfind.five_sig_depth_to_n_sig_depth(loc_depth, 1))
                    # Check this agrees with flux/flux_err
                    try:
                        #print(row[f'sigma_{band}'])
                        n_sig_detect = float(row[f'sigma_{band.upper()}'][0][0])
                    except KeyError:
                        n_sig_detect = float(row[f'sigma_{band.upper()}'])
                    except IndexError:
                        n_sig_detect = float(row[f'sigma_{band.upper()}'][0])
                    
                    if plot_mag_auto:
                        try:
                            mag_list = row[f'MAG_AUTO_{band.upper()}'][0] * u.ABmag
                        except KeyError:
                            mag_list = 99 * u.ABmag
                    
                        ax_photo.plot(wav.to(wav_unit).value, mag_list.to(flux_unit).value, color='black', markersize=5, marker='s', zorder=10)

                    #print(n_sig_detect)
                    ax_photo.annotate(f"{n_sig_detect:.1f}$\sigma$" if n_sig_detect < 100 else f"{n_sig_detect:.0f}$\sigma$", (wav.to(wav_unit).value, ax_photo.get_ylim()[0]-0.2 if pos % 2 == 0 else ax_photo.get_ylim()[0]-0.6), ha='center', fontsize='medium', path_effects=[pe.withStroke(linewidth=3, foreground='white')], zorder=5)

                    if catalog_version == 'simulated':
                        try:
                            flux = row[f'NRC_{band.upper()}_fnu']
                        except KeyError:
                            try:
                                flux = row[f'HST_{band.upper()}_fnu']
                            except KeyError:
                                try:
                                    flux = row[f'MIRI_{band.upper()}_fnu']
                                except KeyError:
                                    flux = -99
                        if flux != -99:
                            mag = useful_funcs_updated_new_galfind.flux_to_mag(flux, u.nJy)

                            band_wavs = useful_funcs_updated_new_galfind.band_wavelengths[band].to(wav_unit).value
        
                            ax_photo.plot(band_wavs,mag, label='Unscattered Photometry' if pos == 0 else '', color='purple', markersize = 5, marker='s', zorder=1, linestyle='none')
            
            ax_photo.set_ylim(ax_photo.get_ylim()[0], np.min(all_mags)-1)
            
            try:
                if catalog_version == 'simulated':
                    z_spec  = row['redshift'].data[0]

                    realization = row['REALIZATION'].data[0]
                    if 4 < z_spec < 5:
                        spec_path = f'/raid/scratch/data/JAGUAR/Full_Spectra/JADES_SF_mock_r{realization}_v1.2_spec_5A_30um_z_4_5.fits'
                    if z_spec > 5:
                        spec_path = f'/raid/scratch/data/JAGUAR/Full_Spectra/JADES_SF_mock_r{realization}_v1.2_spec_5A_30um_z_5_15.fits'
                    scidata = fits.open(spec_path)
                    spectra = scidata[1].data
                    wavelength = scidata[2].data
                    z = scidata[3].data
                    jag_ids = []
                    for zrow in z:
                        jag_ids.append(zrow[0])
                
                    pos = np.argwhere(jag_ids == row['ID2'])[0][0]

                    #ID =jag_ids[pos]
                    #print(row['ID2'], ID)
                    specuse = spectra[pos] / (1+z_spec)
                    wluse = wavelength * (1+z_spec) 
                    wluse = wluse * u.Angstrom
                    wluse = wluse.to(wav_unit)
                    specuse = specuse * u.erg/(u.cm**2 * u.s * u.AA)
                    nu_use = specuse.to(flux_unit, equivalencies=u.spectral_density(wluse)).value
                    ax_photo.plot(wluse.value, nu_use, label='Unscattered Spectrum', color='purple', alpha=0.8)
                
            except FileNotFoundError as e:
                print('error')
                print(e)   
            try:
                if show_rejected_reason:
                    rejected = str(row[f'rejected_reasons{col_ext}'][0])
                    if rejected != '':
                        ax_photo.annotate(rejected, (0.9, 0.95), ha='center', fontsize='small', xycoords = 'axes fraction', zorder=5)
            except:
                print('Failed to plot rejected reasons')
                pass
                    
            # open EAZY SED
            try:
                if SED_code != 'LePhare':
                    eazy_best_sed, header_info = useful_funcs_updated_new_galfind.open_eazy_spec(gal_id, field_name, template=template, custom=custom_sex, min_percentage_error=min_percentage_err, custom_path=eazy_sed_path)

                    wav = eazy_best_sed['wav'].to(u.Angstrom)
                    mag = eazy_best_sed['flux'].to(u.ABmag)
                    zbest = row[f'zbest{col_ext}_zfree'].data
                    
                
                    custom_lowz = ''
                    if plot_lowz:
                        if zbest > 6.5:
                            custom_lowz = '_zmax=6.0'
                        elif zbest > 4.5:
                            custom_lowz = '_zmax=4.0'
                    

                    if custom_lowz != '':
                        eazy_best_sed_lowz, header_info_lowz = useful_funcs_updated_new_galfind.open_eazy_spec(gal_id, field_name, template=template, min_percentage_error=min_percentage_err, custom_path=eazy_sed_path.replace('_zfree', custom_lowz), custom=custom_lowz)
                        wav_lowz = eazy_best_sed_lowz['wav'].to(u.Angstrom)
                        mag_lowz = eazy_best_sed_lowz['flux'].to(u.ABmag)
                        line_lowz = ax_photo.plot(wav_lowz.to(wav_unit).value, mag_lowz.to(flux_unit), label=f'EAZY {template} low-z ($z_{{max}}$={custom_lowz.split("=")[-1]})', alpha=0.8, lw=2, zorder=4)
                        ax_eazy_lowz_pdf.set_title(f'EAZY PDF ($z_{{max}}$={custom_lowz.split("=")[-1]})', fontsize='medium')

                        band_mags_lowz = useful_funcs_updated_new_galfind.convolve_sed_v2(mag_lowz, wav_lowz.to('um'), bands)
                        print(band_mags_lowz)
                        if band_mags_lowz == 0:
                            print('Setting invisible')
                            ax_eazy_lowz_pdf.set_visible(False)
                        else:
                            band_mags_lowz =  [i.value for i in band_mags_lowz]
                            eazy_color_lowz = line_lowz[0].get_color()

                            band_wavs_lowz = [useful_funcs_updated_new_galfind.band_wavelengths[i].to(wav_unit).value for i in bands]
                            ax_photo.scatter(band_wavs_lowz, band_mags_lowz, edgecolors=eazy_color_lowz, marker='o', facecolor='none', s=80, zorder=4.5)
                    
                    else:
                        #print('Setting invisible')
                        #print(zbest)
                        ax_eazy_lowz_pdf.set_visible(False)

                    band_mags = useful_funcs_updated_new_galfind.convolve_sed_v2(mag, wav.to('um'), bands)
                    band_mags =  [i.value for i in band_mags]
                    band_wavs = [useful_funcs_updated_new_galfind.band_wavelengths[i].to(wav_unit).value for i in bands]
                    
                    line = ax_photo.plot(wav.to(wav_unit).value, mag.to(flux_unit), label=f'EAZY {template}', alpha=0.8, lw=2, zorder=4.3)

                    eazy_color = line[0].get_color()
    
                    ax_photo.scatter(band_wavs, band_mags, edgecolors=eazy_color, marker='o', facecolor='none', s=80, zorder=4.6)
                    
                
            except FileNotFoundError as e:
                print(e)
                eazy_color = 'red'
                print('EAZY best-fitting SED not found')
            
            try:
                for model_version in ['bobcat', 'cholla']:
                    sonora_path='/nvme/scratch/work/tharvey/brown_dwarfs/'
                    if row[f'constant_best_sonora_{model_version}'] > 0:
                        name_bd = row[f'best_template_sonora_{model_version}'][0]
                        chi2_bd = row[f'chi2_best_sonora_{model_version}'][0] 
                        const_bd = row[f'constant_best_sonora_{model_version}'][0]
                        table_bd = Table.read(f'{sonora_path}/sonora_model/{name_bd}', format='ascii.ecsv')
                        
                        table_bd['flux_njy'] = table_bd['flux_nu'].to(u.nJy)
                    
                        mag_bd = (table_bd['flux_njy'] * const_bd).to(flux_unit)
                        line_bd = ax_photo.plot(table_bd['wav'].to(wav_unit), mag_bd, label=f'BD {name_bd}, $\chi^2$: {chi2_bd:.2f}')
                        band_mags_bd = [i.value for i in useful_funcs_updated_new_galfind.convolve_sed_v2(mag_bd, table_bd['wav'].to(wav_unit), bands)]
                        #print(band_mags_bd)
                        band_wavs_bd = [useful_funcs_updated_new_galfind.band_wavelengths[i].to(wav_unit).value for i in bands]    
                        ax_photo.scatter(band_wavs_bd, band_mags_bd, edgecolors=line_bd[0].get_color(), marker='o', facecolor='none', s=80, zorder=4.5)
                        #print('Plotted BD template')
            except KeyError as e:
                pass  

            try:
                if SED_code != 'LePhare':
                    
                    zbest = row[f'zbest{col_ext}_zfree'].data
                    custom_lowz = ''
                    if plot_lowz:
                        if zbest > 6.5:
                            custom_lowz = '_zmax=6.0'
                        elif zbest > 4.5:
                            custom_lowz = '_zmax=4.0'
                    
                    eazy_z, eazy_pdf_z = useful_funcs_updated_new_galfind.load_eazy_pdf(gal_id, field_name, template=template, custom=custom_sex, min_percentage_error=min_percentage_err, use_galfind=use_galfind, custom_path=eazy_pdf_path)
                    ax_eazy_pdf.plot(eazy_z, eazy_pdf_z/np.max(eazy_pdf_z), color=eazy_color)
                    ax_eazy_pdf.set_ylim(0, 1.20)
                    # Set xlim to redshifts of 1% and 99% of PDF cumulative distribution
                    norm = np.cumsum(eazy_pdf_z)
                    norm = norm/np.max(norm)
                    lowz = eazy_z[np.argmin(np.abs(norm-0.02))] - 0.3
                    highz = eazy_z[np.argmin(np.abs(norm-0.98))] + 0.3
                    ax_eazy_pdf.set_xlim(lowz, highz)
    
                    if custom_lowz != '':       
                        eazy_z_lowz, eazy_pdf_z_lowz = useful_funcs_updated_new_galfind.load_eazy_pdf(gal_id, field_name, template=template, custom=custom_lowz, min_percentage_error=min_percentage_err, use_galfind=use_galfind, custom_path=eazy_pdf_path.replace('_zfree', custom_lowz))
                        ax_eazy_lowz_pdf.plot(eazy_z_lowz, eazy_pdf_z_lowz/np.max(eazy_pdf_z_lowz), color=eazy_color_lowz)
                        # low-z 

                        norm = np.cumsum(eazy_pdf_z_lowz)
                        norm = norm/np.max(norm)
                        lowz = eazy_z_lowz[np.argmin(np.abs(norm-0.005))] - 0.3
                        highz = eazy_z_lowz[np.argmin(np.abs(norm-0.995))] + 0.3
                        ax_eazy_lowz_pdf.set_xlim(lowz, highz)
                        
                        # Find pdf percentiles
                        '''zbest_lowz = float(header_info_lowz[1][1])
                        lower_lim_lowz = float(zbest_lowz - float(header_info_lowz[1][2]))
                        upper_lim_lowz = float(float(header_info_lowz[1][3]) - zbest_lowz)
                        chi2_lowz = float(header_info_lowz[1][4])'''
                        zbest_lowz = float(row[f'zbest{col_ext}{custom_lowz}'])
                        lower_lim_lowz = float(row[f'zbest{col_ext}{custom_lowz}']-row[f'zbest_16{col_ext}{custom_lowz}'])
                        upper_lim_lowz = float(row[f'zbest_84{col_ext}{custom_lowz}']-row[f'zbest{col_ext}{custom_lowz}'])
                        chi2_lowz = float(row[f'chi2_best{col_ext}{custom_lowz}'])
            
                    x_lim = ax_eazy_pdf.get_xlim()
                    y_lim = ax_eazy_pdf.get_ylim()
                    
                    zbest = float(row[f'zbest{col_ext}_zfree'])
                    lower_lim = float(row[f'zbest{col_ext}_zfree']-row[f'zbest_16{col_ext}_zfree'])
                    upper_lim = float(row[f'zbest_84{col_ext}_zfree']-row[f'zbest{col_ext}_zfree'])
                    chi2 = float(row[f'chi2_best{col_ext}_zfree'])
    
                    #one_kpc = cosmo.kpc_comoving_per_arcmin(zbest).to(u.kpc/u.arcsec) * pixel_scale * u.arcsec
                    # re in pixels
                    re = 10 # pixels
                    d_A =  cosmo.angular_diameter_distance(zbest)
                    pix_scal = u.pixel_scale(0.03*u.arcsec/u.pixel)
                    re_as = (re * u.pixel).to(u.arcsec, pix_scal)
                    re_kpc = (re_as * d_A).to(u.kpc, u.dimensionless_angles())
                    
                    # Plot scalebar with physical size
                    scalebar = AnchoredSizeBar(cutout_ax_list[-1].transData,
                                                re, f"{re_kpc:.1f}", 'upper left', 
                                                pad=0.3,
                                                color='white',
                                                frameon=False,
                                                size_vertical=1.5,
                                                )
                    cutout_ax_list[-1].add_artist(scalebar)
    
                
                    ax_eazy_pdf.grid(False)
                    # Draw vertical line at zbest
                    ax_eazy_pdf.axvline(zbest, color=eazy_color, linestyle='--', alpha=0.5, lw=2)
                    ax_eazy_pdf.axvline(zbest+upper_lim, color=eazy_color, linestyle=':', alpha=0.5, lw=2)
                    ax_eazy_pdf.axvline(zbest-lower_lim, color=eazy_color, linestyle=':', alpha=0.5, lw=2)
                    ax_eazy_pdf.annotate('-1$\sigma$', (zbest-lower_lim, 0.1), fontsize='small', ha='center', transform=ax_eazy_pdf.get_yaxis_transform(), va='bottom',  color=eazy_color, path_effects=[pe.withStroke(linewidth=3, foreground='white')])
                    # Shade region between zbest-lower_lim and zbest+upper_lim below PDF)
                    ax_eazy_pdf.annotate('+1$\sigma$', (zbest+upper_lim, 0.1), fontsize='small', ha='center', transform=ax_eazy_pdf.get_yaxis_transform(), va='bottom', color=eazy_color, path_effects=[pe.withStroke(linewidth=3, foreground='white')])
                    # Shade region between zbest-lower_lim and zbest+upper_lim below PDF
                    ax_eazy_pdf.annotate(r'$z_{\rm phot}=$'+f'{zbest:.1f}'+f'$^{{+{upper_lim:.1f}}}_{{-{lower_lim:.1f}}}$', (zbest, 1.17), fontsize='medium', va='top', ha='center', color=eazy_color, path_effects=[pe.withStroke(linewidth=3, foreground='white')])
                    # Horizontal arrow at PDF peak going left or right depending on which side PDF is on, labelled with chi2
                    # Check if zbest is closer to xlim[0] or xlim[1]
                    
                    amount = 0.3 * (x_lim[1] - x_lim[0])
                    if zbest - x_lim[0] < x_lim[1] - zbest:
                        direction = 1
                    else:
                        direction = -1
                    ax_eazy_pdf.annotate(r'$\chi^2=$'+f'{chi2:.2f}', (zbest, 1.0), xytext = (zbest + direction*amount, 0.90),  fontsize='small', va='top', ha='center', color=eazy_color, path_effects=[pe.withStroke(linewidth=3, foreground='white')], arrowprops=dict(facecolor=eazy_color, edgecolor=eazy_color, arrowstyle='-|>', lw=1.5, path_effects=[pe.withStroke(linewidth=1, foreground='white')]))
                            
                    if custom_lowz != '':
                        # Do the same as above for the low-z PDF
                        ax_eazy_lowz_pdf.grid(False)
                        x_lim_lowz = ax_eazy_lowz_pdf.get_xlim()
                        y_lim_lowz = ax_eazy_lowz_pdf.get_ylim()
                        ax_eazy_lowz_pdf.set_ylim(0, 1.20)
                        ax_eazy_lowz_pdf.axvline(zbest_lowz, color=eazy_color_lowz, linestyle='--', alpha=0.5, lw=2)
                        ax_eazy_lowz_pdf.axvline(zbest_lowz+upper_lim_lowz, color=eazy_color_lowz, linestyle=':', alpha=0.5, lw=2)
                        ax_eazy_lowz_pdf.axvline(zbest_lowz-lower_lim_lowz, color=eazy_color_lowz, linestyle=':', alpha=0.5, lw=2)
                        ax_eazy_lowz_pdf.annotate('-1$\sigma$', (zbest_lowz-lower_lim_lowz, 0.1), fontsize='small', ha='center', transform=ax_eazy_lowz_pdf.get_yaxis_transform(), va='bottom',  color=eazy_color_lowz, path_effects=[pe.withStroke(linewidth=3, foreground='white')])
                        ax_eazy_lowz_pdf.annotate('+1$\sigma$', (zbest_lowz+upper_lim_lowz, 0.1), fontsize='small', ha='center', transform=ax_eazy_lowz_pdf.get_yaxis_transform(), va='bottom', color=eazy_color_lowz, path_effects=[pe.withStroke(linewidth=3, foreground='white')])
                        ax_eazy_lowz_pdf.annotate(r'$z_{\rm phot}=$'+f'{zbest_lowz:.1f}'+f'$^{{+{upper_lim_lowz:.1f}}}_{{-{lower_lim_lowz:.1f}}}$', (zbest_lowz, 1.17), fontsize='medium', va='top', ha='center', color=eazy_color_lowz, path_effects=[pe.withStroke(linewidth=3, foreground='white')])
                        amount = 0.3 * (x_lim_lowz[1] - x_lim_lowz[0])
                        if zbest_lowz - x_lim_lowz[0] < x_lim_lowz[1] - zbest_lowz:
                            direction = 1
                        else:
                            direction = -1
                        ax_eazy_lowz_pdf.annotate(r'$\chi^2=$'+f'{chi2_lowz:.2f}', (zbest_lowz, 1.0), xytext = (zbest_lowz + direction*amount, 0.90),  fontsize='small', va='top', ha='center', color=eazy_color_lowz, path_effects=[pe.withStroke(linewidth=3, foreground='white')], arrowprops=dict(facecolor=eazy_color_lowz, edgecolor=eazy_color_lowz, arrowstyle='-|>', lw=1.5, path_effects=[pe.withStroke(linewidth=1, foreground='white')]))

                    try:
                        
                        if recalc_pdf_stats:
                            pz_column, integral, peak_z, peak_loc, peak_second_loc, secondary_peak, ratio = useful_funcs_updated_new_galfind.robust_pdf([gal_id], [zbest], SED_code, field_name, rel_limits=True, z_fact=int_limit, use_custom_lephare_seds=custom_lephare, template=template, plot=False, version=catalog_version, custom_sex=custom_sex, min_percentage_err=min_percentage_err, custom_path=eazy_pdf_path, use_galfind=True)
                            print(integral, 'integral', peak_z, 'peak_z', peak_loc, 'peak_loc', peak_second_loc, 'peak_second_loc', secondary_peak, 'secondary_peak', ratio, 'ratio')
                        else:
                            try:
                                integral = row[f'PDF_integral_eazy{col_ext}'].data
                            except KeyError:
                                integral = -999
                            if annotate_eazy_pdf:
                                peak_z = row[f'peak_probheight_eazy{col_ext}'].data 
                                peak_loc = row[f'peak_loc_eazy{col_ext}'].data
                                secondary_peak = row[f'sec_probheight_eazy{col_ext}'].data
                                secondary_loc = row[f'sec_loc_eazy{col_ext}'].data
                                ratio = float(row[f'pri_sec_ratio_eazy{col_ext}'].data)
                                
                                ax_eazy_pdf.scatter(peak_loc, peak_z, color=eazy_color, edgecolors=eazy_color, marker='o', facecolor='none')
                                
                                if secondary_peak > 0: 
                                    ax_eazy_pdf.scatter(secondary_loc, secondary_peak, edgecolor='orange', marker='o', facecolor='none')
                                    ax_eazy_pdf.annotate(f'P(S)/P(P): {ratio:.2f}', loc_ratio, fontsize='x-small')
                    except KeyError as e:
                        print(e)
                        print('Secondary solution columns not found.')
                    if integral != -999:
                        
                        eazy_z_lim = np.linspace(0.93*float(zbest), 1.07*float(zbest), 100)
                        eazy_pdf_lim = np.interp(eazy_z_lim, eazy_z, eazy_pdf_z/np.max(eazy_pdf_z))
                        ax_eazy_pdf.fill_between(eazy_z_lim, eazy_pdf_lim, color=eazy_color, alpha=0.2, hatch='//')
                        # Get zbest in axes coords
                        
                        ax_eazy_pdf.annotate(f'$\\sum = {float(integral):.2f}$', (zbest, 0.45), fontsize='small', transform=ax_eazy_pdf.get_yaxis_transform(), va='bottom', ha='center', fontweight='bold', color=eazy_color, path_effects=[pe.withStroke(linewidth=3, foreground='white')])
                        #ax_eazy_pdf.annotate(f'$\int^{{{:.1f}}}_{{{0.93*float(zbest):.1f}}} P(z): $'+f'{float(integral):.2f}', (0.1, 0.7), fontsize='small', transform=ax_eazy_pdf.transAxes, va='center', ha='left')
                    
            except FileNotFoundError:
                print('EAZY PDF not found')

            for other_SED_code in additional_SED_codes:
                if type(other_SED_code) == list:
                    if other_SED_code[0].lower() == 'bagpipes':
                        for bagpipes_fit in other_SED_code[1:]:
                            wav, mag = useful_funcs_updated_new_galfind.open_bagpipes_sed(gal_id, field_name, bagpipes_fit, load_duncans=False)
                            band_mags = useful_funcs_updated_new_galfind.convolve_sed_v2(mag, wav.to('um'), bands)
                            band_mags =  [i.value for i in band_mags]
                            band_wavs = [useful_funcs_updated_new_galfind.band_wavelengths[i].to(wav_unit).value for i in bands]
                            
                            line = ax_photo.plot(wav.to(wav_unit).value, mag.to(flux_unit), label=f'{bagpipes_fit.split(".")[0]}', alpha = 0.8)
                            bagpipes_color = line[0].get_color()
    
                            ax_photo.scatter(band_wavs, band_mags, edgecolors=bagpipes_color, marker='o', facecolor='none', s=80, zorder=100)

                print(f'Doing additional SED_code: {other_SED_code}')
                if True: #try:
                    if other_SED_code == 'lephare_hot':
                        lephare_fit_statistics, list_of_fits = useful_funcs_updated_new_galfind.open_lephare_spec(gal_id, survey='all_robust_good', SED_code='LePhare-PP', cat_version=catalog_version, min_percentage_err=min_percentage_err)
                        lpp_chi2 = lephare_fit_statistics['Chi2'][0]
                        wav = list_of_fits[0][0] * u.Angstrom
                        
                        mag = list_of_fits[0][1] * u.ABmag
                        line = ax_photo.plot(wav.to(wav_unit).value, mag.to(flux_unit), label=f'LP++ HOT $\chi^2$ {lpp_chi2:.2f}', alpha = 0.8, linestyle='dashed')
                        
                        band_mags = useful_funcs_updated_new_galfind.convolve_sed_v2(mag, wav.to('um'), bands)
                        band_mags =  [i.value for i in band_mags]
                        band_wavs = [useful_funcs_updated_new_galfind.band_wavelengths[i].to(wav_unit).value for i in bands]
                        
                        lephare_color = line[0].get_color()
                        ax_photo.scatter(band_wavs, band_mags, edgecolors=lephare_color, marker='o', facecolor='none', s=80, zorder=100)
                            
                    elif other_SED_code == 'eazy_fsps' or other_SED_code == 'eazy_hot' or other_SED_code == 'eazy_nakajima_full' or other_SED_code == 'eazy_nakajima_subset':
                        
                        if other_SED_code == 'eazy_fsps':
                            survey_new = 'all_robust_good'
                            template_new = 'fsps'
                        elif other_SED_code == 'eazy_hot':
                            survey_new = 'all_robust_good_hot'
                            if row['Z_BEST'].data > 12:
                                template_new = 'HOT_60K'
                            else:
                                template_new = 'HOT_45K'
                        elif other_SED_code == 'eazy_nakajima_full':
                            survey_new = 'all_robust_good'
                            template_new = 'nakajima_full'
                        elif other_SED_code == 'eazy_nakajima_subset':
                            survey_new = field_name
                            template_new = 'nakajima_subset'
                        eazy_sed_path_other = f'/nvme/scratch/work/tharvey/EAZY/outputs/{survey_new}/{min_percentage_err}pc/seds_{template_new}/'
                        galid = f'{gal_id}_{field}'
                        galid = gal_id
                        eazy_best_sed, header_info = useful_funcs_updated_new_galfind.open_eazy_spec(galid, survey_new, template=template_new, min_percentage_error=min_percentage_err, custom_path=eazy_sed_path_other)
                        wav = eazy_best_sed['wav'].to(u.Angstrom)
                        mag = eazy_best_sed['flux'].to(u.ABmag)
                
                        line = ax_photo.plot(wav.to(wav_unit).value, mag.to(flux_unit), label=f'EAZY {template_new} $\chi^2$: {float(header_info["CHIBEST"]):.2f}', alpha=0.8, linestyle='dashed' if other_SED_code == 'eazy_hot' else "solid")
                        band_mags = useful_funcs_updated_new_galfind.convolve_sed_v2(mag, wav.to('um'), bands)
                        band_mags =  [i.value for i in band_mags]
                        band_wavs = [useful_funcs_updated_new_galfind.band_wavelengths[i].to(wav_unit).value for i in bands]
                        
                        eazy_color = line[0].get_color()
                        ax_photo.scatter(band_wavs, band_mags, edgecolors=eazy_color, marker='o', facecolor='none', s=80, zorder=100)
                        
                    elif other_SED_code == 'bagpipes':
                        raise NotImplementedError
    
                    elif other_SED_code == 'prospector':
                        raise NotImplementedError
                    else:
                        print('NOT RECOGNISED OTHER SED CODE')
                        print(other_SED_code)
    
            ax_photo.legend(loc='upper left', fontsize='small', frameon=False)
            # path effects on legend, and zorder
            for text in ax_photo.get_legend().get_texts():
                text.set_path_effects([pe.withStroke(linewidth=3, foreground='white')])
                text.set_zorder(12)

            
            #print('Saving image to:',out_image_path)
            if not os.path.exists(out_image_path):
                print("Creating above directory")
                os.makedirs(out_image_path)

            plt.savefig(out_image_path + f'{gal_id}.png', dpi=300, bbox_inches='tight')
            ax_photo.cla()
            ax_eazy_pdf.cla()
            ax_eazy_lowz_pdf.cla()

    def plot_spec_diagnostic(self, overwrite = True):
        # bare in mind that not all galaxies have spectroscopic data
        pass

    # Selection methods

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
    
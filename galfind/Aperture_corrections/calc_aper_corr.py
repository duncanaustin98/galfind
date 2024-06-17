#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 12:47:06 2022

@author: u92876da
"""

# calc_aper_corr.py

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy import units as u
from matplotlib.patches import Circle
import warnings
from scipy import optimize
from astropy.modeling import models, fitting
import sep
import json
import photutils
import os
import sys

from . import config

def log_transform(im): # function to transform fits image to log scaling
    '''returns log(image) scaled to the interval [0,1]'''
    try:
        (min, max) = (im[im > 0].min(), im.max())
        if (max > min) and (max > 0):
            return (np.log(im.clip(min, max)) - np.log(min)) / (np.log(max) - np.log(min))
    except:
        pass
    return im

def open_PSF_model(band, PSF_loc, PSF_name):
    # load PSF .fits image
    band = band.replace("f", "F")
    PSF_path = PSF_loc + PSF_name + band
    hdul = fits.open(PSF_path + ".fits") #directory of images and image name structure for segmentation map
    PSFdata = hdul[0].data
    PSFheader = hdul[0].header
    #print(PSFheader)
    PSFdata = PSFdata.byteswap().newbyteorder() #convert image to format SExtractor uses
    #pixel_scale = PSFheader["PIXELSCL"] * u.arcsec
    try:
        pixel_scale = PSFheader["PIXELSCL"] * u.arcsec
    except:
        print("No PIXELSCL in header")
        pixel_scale = log.pix_to_as
    #print("pixel scale =", pixel_scale)
    return PSFdata, pixel_scale

def calc_aper_corr(PSFdata, x_cen, y_cen, band, aper_diam, extract_code = "sep", plot_PSF = True, \
                   PSF_loc = "/Users/user/Documents/PGR/JWST_PSFs_003as/", \
                       PSF_name = "PSF_Resample_03_", print_output = True, tot_aper_size = None):
    
    # calculate flux in the aperture
    if extract_code == "sep":
        flux_aper, fluxerr_aper, flag_aper = sep.sum_circle(PSFdata, [x_cen], [y_cen], [aper_diam / 2])
        flux_aper = flux_aper[0]
    elif extract_code == "photutils":
        aper = photutils.CircularAperture([x_cen, y_cen], r = aper_diam.value / 2)
        out_tab = photutils.aperture.aperture_photometry(PSFdata, aper, method = "exact")
        flux_aper = out_tab["aperture_sum"][0]
    tot_flux = sep.sum_circle(PSFdata, [x_cen], [y_cen], tot_aper_size / 2)[0][0]
    #print(tot_aper_size, len(PSFdata))
    #print("tot_flux =", tot_flux)
    flux_pc = flux_aper / tot_flux
    aper_corr = -2.5 * np.log10(flux_pc)
    print(aper_corr)
    aper_corr = np.round(aper_corr, 4)
    if print_output:
        print("flux pc =", np.round(flux_pc, 4))
        print("aper_corr =", aper_corr)
    
    # plot results
    if plot_PSF:
        fig, ax = plt.subplots()
        im = ax.imshow(log_transform(PSFdata), origin = "lower")
        aper = Circle(xy = (x_cen, y_cen), radius = aper_diam / 2)
        aper_tot = Circle(xy = (x_cen, y_cen), radius = tot_aper_size / 2)
        aper.set_facecolor('none')
        aper.set_edgecolor('red')
        aper_tot.set_facecolor('none')
        aper_tot.set_edgecolor('red')
        ax.add_artist(aper)
        ax.add_artist(aper_tot)
        ax.set_title(band)
        plt.show()
    return flux_pc, aper_corr, x_cen, y_cen

def plot_flux_curve(PSFdata, pixel_scale, x_cen, y_cen, band, flux_pcs, aper_corrs, PSF_loc = "/Users/user/Documents/PGR/JWST_PSFs_003as/", PSF_name = "PSF_Resample_03_", \
                    save_loc = "", tot_aper_size = None, aper_diams = []):
    
    rlist = np.arange(0, tot_aper_size * pixel_scale.value / 2, 0.01) / pixel_scale
    print("pix_centre:", (x_cen, y_cen))
    flux, fluxerr, flag = sep.sum_circle(PSFdata, [x_cen], [y_cen], rlist)
    tot_flux = sep.sum_circle(PSFdata, [len(PSFdata[0]) / 2 - 0.5], [len(PSFdata[1]) / 2 - 0.5], len(PSFdata[0]) / 2)[0][0]
    tot_flux_smaller_aper = sep.sum_circle(PSFdata, [x_cen], [y_cen], tot_aper_size / 2)[0][0]
    print(tot_flux_smaller_aper / tot_flux)
    print(tot_flux / np.sum(PSFdata)) #np.sum(PSFdata)
    rlist = rlist * pixel_scale
    plt.plot(rlist, flux / tot_flux, c = "red", label = PSF_name + band.replace("f", "F"))
    plt.axvline(pixel_scale.value, 0, 1, c = "black", ls = "--", label = "pixel scale = " + str(pixel_scale))
    y_0 = 0.7
    x_0 = 1.0
    plt.text(x_0 + 0.385, y_0, "| flux % | aper_corr")
    plt.text(x_0 + 0., y_0 - 0.02, "______________________________")
    for i in range(len(aper_diams)):
        text = "%.2f arcsec | %.3f | %.4f" % (aper_diams[i].value, flux_pcs[i], aper_corrs[i])
        plt.text(x_0, y_0 - 0.08 * (i + 1), text)
        
    plt.xlabel('radius (arcsec)')
    plt.ylabel('Fraction of total flux')
    plt.ylim(0, 1)
    plt.legend(loc = "lower right")
    plt.title(band.replace("f", "F"))
    print("Saved to: " + save_loc + PSF_name + band.replace("f", "F") + "_flux_curve.png")
    plt.savefig(save_loc + PSF_name + band.replace("f", "F") + "_flux_curve.png", dpi = 800)
    plt.show()
        
#def compare_aper_flux_to_full_radius():
#    tot_flux = sep.sum_circle(PSFdata, [len(PSFdata[0]) / 2 - 0.5], [len(PSFdata[1]) / 2 - 0.5], len(PSFdata[0]) / 2)[0][0]
    
'''
def plot_additional_flux_curve(band):
    df = pd.read_csv('/Users/user/Documents/PGR/JWST_PSFs_003as/Encircled_Energy_LW_ETCv2.txt', header = 1)
    #print(df_init[0][0])
    #print(df_init[0][0])
    #df.columns = df_init[0]
    #print(df)
    #print("data[0] = ", data[0], data[2])
    #[Cov], columns = ["Sequence", "Start", "End", "Coverage"]
    #plt.plot(data[0], data[2], c = "green")
'''    
def fit_2d_moffatt(PSFdata, maxfev = 10000):
    def moffatcurve(xdata_tuple, A, a, b, xcen, ycen):
        (x, y) = xdata_tuple
        d = -b
        g = A * (1 + (((x - xcen) ** 2 + (y - ycen) ** 2) / a ** 2)) ** d
        #print(g)
        return g.ravel()

    initial_guess = (np.max(PSFdata), 1, 1, len(PSFdata[0]) / 2, len(PSFdata[1]) / 2)
    x = np.linspace(0, len(PSFdata[0]) - 1, len(PSFdata[0]), endpoint = True)
    y = np.linspace(0, len(PSFdata[1]) - 1, len(PSFdata[1]), endpoint = True)
    x, y = np.meshgrid(x, y)
    popt, pcov = optimize.curve_fit(moffatcurve, (x, y), PSFdata.ravel(), p0 = initial_guess, maxfev = maxfev)
    return [popt[3], popt[4]] # return (x, y) central position

def main(in_bands, extract_code, save_loc, PSF_loc, PSF_name, plot_PSF, aper_diams = json.loads(config.get("SExtractor", "APERTURE_DIAMS")) * u.arcsec):
    print("extract code =", extract_code)
    print_line = [["# aper_diam / arcsec"] + [str(aper_diam.value) for aper_diam in aper_diams]]
    for band in in_bands:
        print(band)
        name = PSF_name[0] + band + PSF_name[1]
        PSFdata, pixel_scale = open_PSF_model(band, PSF_loc, name)
        print(pixel_scale)
        x_cen, y_cen = fit_2d_moffatt(PSFdata)
        flux_pcs = []
        aper_corr = []
        for aper_diam in aper_diams:
            print(aper_diam)
            aper_diam_pix = aper_diam / pixel_scale   
            tot_aper_size = 9. * u.arcsec / pixel_scale
            flux_pc_loc, aper_corr_loc, x_cen, y_cen = calc_aper_corr(PSFdata, x_cen, y_cen, band, aper_diam_pix, plot_PSF = plot_PSF, \
                            extract_code = extract_code, PSF_loc = PSF_loc, PSF_name = name, tot_aper_size = tot_aper_size)
            flux_pcs.append(flux_pc_loc)
            aper_corr.append(aper_corr_loc)
        plot_flux_curve(PSFdata, pixel_scale, x_cen, y_cen, band, flux_pcs, aper_corr, \
                        save_loc = save_loc, PSF_loc = PSF_loc, PSF_name = PSF_name, tot_aper_size = tot_aper_size, aper_diams = aper_diams)
        aper_corr.insert(0, band)
        print_line.append(aper_corr)
        # print(print_line)
    np.savetxt("aper_corr.txt", print_line, fmt = "%s" + len(aper_diams) * " %.6s")

if __name__ == "__main__":
    extract_code = "sep"
    save_loc = f"{config['DEFAULT']['GALFIND_WORK']}/Aperture_corrections"
    PSF_loc = config["DEFAULT"]["PSF_DIR"]
    #PSF_name = "PSF_Resample_03_"
    PSF_name = ["PSF_", "cen_G5V_fov299px_ISIM41"]
    plot_PSF = True
    
    main(extract_code, save_loc, PSF_loc, PSF_name, plot_PSF)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NIRCam aperture correction calculation from PSF models.

Computes aperture corrections from PSF models and measured flux profiles
for photometric calibration and aperture-dependent corrections.
"""

# calc_aper_corr.py

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import photutils
import sep
from astropy import units as u
from astropy.io import fits
from matplotlib.patches import Circle
from scipy import optimize

from .. import config
from ..utils import useful_funcs_austind as funcs


def log_transform(im):
    """Apply logarithmic scaling to an image for improved visualization.

    Scales image values to the interval ``[0, 1]`` using the logarithm,
    with fallback to the original image if the transformation fails.

    Parameters
    ----------
    im : `numpy.ndarray`
        Input image array.

    Returns
    -------
    `numpy.ndarray`
        Image scaled to ``[0, 1]`` in log space, or the original image if
        transformation fails or values are invalid.
    """
    try:
        (min, max) = (im[im > 0].min(), im.max())
        if (max > min) and (max > 0):
            return (np.log(im.clip(min, max)) - np.log(min)) / (
                np.log(max) - np.log(min)
            )
    except Exception:
        pass
    return im


def open_PSF_model(
    band,
    PSF_loc=config["DEFAULT"]["PSF_DIR"],
    PSF_name=["PSF_", "cen_G5V_fov299px_ISIM41"],
):
    """Load a PSF model FITS image and its pixel scale.

    Parameters
    ----------
    band : `str`
        Band name, used to construct the PSF filename (lowercase ``"f"``
        is converted to uppercase ``"F"``).
    PSF_loc : `str`, optional
        Directory containing the PSF FITS file. Default is
        ``config["DEFAULT"]["PSF_DIR"]``.
    PSF_name : `str`, optional
        Name of the PSF FITS file (without the ``.fits`` extension),
        appended to `PSF_loc`. Default is
        ``["PSF_", "cen_G5V_fov299px_ISIM41"]``.

    Returns
    -------
    `tuple`
        ``(PSFdata, pixel_scale)``: the PSF image data (byte-swapped to
        native order for use with SExtractor/`sep`) and the pixel scale
        (`astropy.units.Quantity` in arcsec) read from the ``PIXELSCL``
        FITS header keyword.

    Raises
    ------
    Exception
        If the FITS header does not contain a ``PIXELSCL`` keyword.
    """
    # load PSF .fits image
    band = band.replace("f", "F")
    PSF_path = PSF_loc + "/" + PSF_name
    hdul = fits.open(
        PSF_path + ".fits"
    )  # directory of images and image name structure for segmentation map
    PSFdata = hdul[0].data
    PSFheader = hdul[0].header
    # print(PSFheader)
    PSFdata = (
        PSFdata.byteswap().newbyteorder()
    )  # convert image to format SExtractor uses
    # pixel_scale = PSFheader["PIXELSCL"] * u.arcsec
    try:
        pixel_scale = PSFheader["PIXELSCL"] * u.arcsec
    except Exception:
        raise (Exception("No PIXELSCL in header"))
        # pixel_scale = log.pix_to_as
    # print("pixel scale =", pixel_scale)
    return PSFdata, pixel_scale


def calc_aper_corr(
    PSFdata,
    x_cen,
    y_cen,
    band,
    aper_diam,
    extract_code="sep",
    plot_PSF=True,
    PSF_loc=config["DEFAULT"]["PSF_DIR"],
    PSF_name=["PSF_", "cen_G5V_fov299px_ISIM41"],
    print_output=True,
    tot_aper_size=None,
):
    """Compute the aperture correction for a PSF model at a given
    aperture diameter.

    Measures the flux within a circular aperture of diameter `aper_diam`
    centred at ``(x_cen, y_cen)``, divides it by the flux within a larger
    circular aperture of diameter `tot_aper_size` (assumed to approximate
    the total flux), and converts the resulting flux fraction into a
    magnitude-space aperture correction.

    Parameters
    ----------
    PSFdata : `NDArray`
        2D PSF model image data (in pixels).
    x_cen : `float`
        x pixel coordinate of the PSF centre.
    y_cen : `float`
        y pixel coordinate of the PSF centre.
    band : `str`
        Band name, used for plot labelling only.
    aper_diam : `float`
        Aperture diameter, in pixels.
    extract_code : `str`, optional
        Which photometry code to use to measure the aperture flux, either
        ``"sep"`` or ``"photutils"``. Default is ``"sep"``.
    plot_PSF : `bool`, optional
        Whether to plot the PSF image with the aperture and total-flux
        aperture overlaid. Default is `True`.
    PSF_loc : `str`, optional
        Unused directly here; present for interface compatibility.
        Default is ``config["DEFAULT"]["PSF_DIR"]``.
    PSF_name : `str`, optional
        Unused directly here; present for interface compatibility.
        Default is ``["PSF_", "cen_G5V_fov299px_ISIM41"]``.
    print_output : `bool`, optional
        Whether to print the computed flux percentage and aperture
        correction. Default is `True`.
    tot_aper_size : `float`, optional
        Diameter, in pixels, of the aperture used to approximate the total
        PSF flux. Default is `None`.

    Returns
    -------
    `tuple`
        ``(flux_pc, aper_corr, x_cen, y_cen)``: the fraction of total flux
        contained within `aper_diam`, the corresponding aperture
        correction in magnitudes (``-2.5 * log10(flux_pc)``), and the
        (unmodified) input centre coordinates.
    """
    # calculate flux in the aperture
    if extract_code == "sep":
        flux_aper, fluxerr_aper, flag_aper = sep.sum_circle(
            PSFdata, [x_cen], [y_cen], [aper_diam / 2]
        )
        flux_aper = flux_aper[0]
    elif extract_code == "photutils":
        aper = photutils.CircularAperture(
            [x_cen, y_cen], r=aper_diam.value / 2
        )
        out_tab = photutils.aperture.aperture_photometry(
            PSFdata, aper, method="exact"
        )
        flux_aper = out_tab["aperture_sum"][0]
    tot_flux = sep.sum_circle(PSFdata, [x_cen], [y_cen], tot_aper_size / 2)[0][
        0
    ]
    # print(tot_aper_size, len(PSFdata))
    # print("tot_flux =", tot_flux)
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
        ax.imshow(log_transform(PSFdata), origin="lower")
        aper = Circle(xy=(x_cen, y_cen), radius=aper_diam / 2)
        aper_tot = Circle(xy=(x_cen, y_cen), radius=tot_aper_size / 2)
        aper.set_facecolor("none")
        aper.set_edgecolor("red")
        aper_tot.set_facecolor("none")
        aper_tot.set_edgecolor("red")
        ax.add_artist(aper)
        ax.add_artist(aper_tot)
        ax.set_title(band)
        plt.show()
    return flux_pc, aper_corr, x_cen, y_cen


def plot_flux_curve(
    PSFdata,
    pixel_scale,
    x_cen,
    y_cen,
    band,
    flux_pcs,
    aper_corrs,
    PSF_loc=config["DEFAULT"]["PSF_DIR"],
    PSF_name=["PSF_", "cen_G5V_fov299px_ISIM41"],
    save_loc=f"{config['DEFAULT']['GALFIND_WORK']}/Aperture_corrections",
    tot_aper_size=None,
    aper_diams=[],
):
    """Plot encircled energy curve with NIRCam aperture corrections.

    Similar to plot_flux_curve in calc_aper_corr.py but configured for
    NIRCam PSF models and save locations.

    Parameters
    ----------
    PSFdata : `numpy.ndarray`
        2D PSF image.
    pixel_scale : `astropy.units.Quantity`
        Pixel scale (arcsec/pixel).
    x_cen, y_cen : `float`
        PSF center (pixels).
    band : `str`
        Filter name.
    flux_pcs : `array-like`
        Flux fractions in apertures.
    aper_corrs : `array-like`
        Aperture corrections (mag).
    PSF_loc : `str`, optional
        PSF directory.
    PSF_name : `list`, optional
        PSF name components.
    save_loc : `str`, optional
        Output directory.
    tot_aper_size : `float` or `None`, optional
        Total aperture size (pixels).
    aper_diams : `list`, optional
        Aperture diameters to mark.
    """
    mpl.rcParams.update(mpl.rcParamsDefault)
    rlist = (
        np.arange(0, tot_aper_size * pixel_scale.value / 2, 0.01) / pixel_scale
    )
    print("pix_centre:", (x_cen, y_cen))
    flux, fluxerr, flag = sep.sum_circle(PSFdata, [x_cen], [y_cen], rlist)
    tot_flux = sep.sum_circle(
        PSFdata,
        [len(PSFdata[0]) / 2 - 0.5],
        [len(PSFdata[1]) / 2 - 0.5],
        len(PSFdata[0]) / 2,
    )[0][0]
    tot_flux_smaller_aper = sep.sum_circle(
        PSFdata, [x_cen], [y_cen], tot_aper_size / 2
    )[0][0]
    print(tot_flux_smaller_aper / tot_flux)
    print(tot_flux / np.sum(PSFdata))  # np.sum(PSFdata)
    rlist = rlist * pixel_scale
    plt.plot(rlist, flux / tot_flux, c="red", label=PSF_name)
    plt.axvline(
        pixel_scale.value,
        0,
        1,
        c="black",
        ls="--",
        label="pixel scale = " + str(pixel_scale),
    )
    y_0 = 0.7
    x_0 = 1.0
    plt.text(x_0 + 0.385, y_0, "| flux % | aper_corr")
    plt.text(x_0 + 0.0, y_0 - 0.02, "______________________________")
    for i in range(len(aper_diams)):
        text = "%.2f arcsec | %.3f | %.4f" % (
            aper_diams[i].value,
            flux_pcs[i],
            aper_corrs[i],
        )
        plt.text(x_0, y_0 - 0.08 * (i + 1), text)

    plt.xlabel("radius (arcsec)")
    plt.ylabel("Fraction of total flux")
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.title(band.replace("f", "F"))
    plt.savefig(f"{save_loc}/{PSF_name}_flux_curve.png", dpi=800)
    funcs.change_file_permissions(f"{save_loc}/{PSF_name}_flux_curve.png")
    print(f"Saved to: {save_loc}/{PSF_name}_flux_curve.png")
    plt.show()


# def compare_aper_flux_to_full_radius():
#    tot_flux = sep.sum_circle(
#        PSFdata,
#        [len(PSFdata[0]) / 2 - 0.5],
#        [len(PSFdata[1]) / 2 - 0.5],
#        len(PSFdata[0]) / 2
#    )[0][0]

"""
def plot_additional_flux_curve(band):
    df = pd.read_csv(
        '/Users/user/Documents/PGR/JWST_PSFs_003as/'
        'Encircled_Energy_LW_ETCv2.txt',
        header=1
    )
    #print(df_init[0][0])
    #print(df_init[0][0])
    #df.columns = df_init[0]
    #print(df)
    #print("data[0] = ", data[0], data[2])
    #[Cov], columns = ["Sequence", "Start", "End", "Coverage"]
    #plt.plot(data[0], data[2], c = "green")
"""


def fit_2d_moffatt(PSFdata, maxfev=10000):
    """Fit 2D Moffat profile to NIRCam PSF data.

    Fits an analytic 2D Moffat function to a NIRCam PSF image.

    Parameters
    ----------
    PSFdata : `numpy.ndarray`
        2D PSF image to fit.
    maxfev : `int`, optional
        Maximum function evaluations. Default is 10,000.

    Returns
    -------
    `tuple`
        Fitted parameters (A, a, b, xcen, ycen) and covariance.
    """

    def moffatcurve(xdata_tuple, A, a, b, xcen, ycen):
        """2D Moffat function for curve fitting.

        Parameters
        ----------
        xdata_tuple : `tuple` of `numpy.ndarray`
            (x, y) coordinate arrays.
        A : `float`
            Amplitude parameter.
        a : `float`
            Scale parameter.
        b : `float`
            Power-law index parameter.
        xcen : `float`
            Center x coordinate.
        ycen : `float`
            Center y coordinate.

        Returns
        -------
        `numpy.ndarray`
            Flattened Moffat profile.
        """
        (x, y) = xdata_tuple
        d = -b
        g = A * (1 + (((x - xcen) ** 2 + (y - ycen) ** 2) / a**2)) ** d
        # print(g)
        return g.ravel()

    initial_guess = (
        np.max(PSFdata),
        1,
        1,
        len(PSFdata[0]) / 2,
        len(PSFdata[1]) / 2,
    )
    x = np.linspace(0, len(PSFdata[0]) - 1, len(PSFdata[0]), endpoint=True)
    y = np.linspace(0, len(PSFdata[1]) - 1, len(PSFdata[1]), endpoint=True)
    x, y = np.meshgrid(x, y)
    popt, pcov = optimize.curve_fit(
        moffatcurve, (x, y), PSFdata.ravel(), p0=initial_guess, maxfev=maxfev
    )
    return [popt[3], popt[4]]  # return (x, y) central position


def main(
    in_bands,
    extract_code="sep",
    save_loc=f"{config['DEFAULT']['GALFIND_WORK']}/Aperture_corrections",
    PSF_loc=config["DEFAULT"]["PSF_DIR"],
    PSF_name="PSF_MIRI_in_flight_opd_filter_",
    plot_PSF=True,
    aper_diams=json.loads(config.get("SExtractor", "APERTURE_DIAMS"))
    * u.arcsec,
    instrument_name="NIRCam",
):
    """Compute and plot NIRCam aperture corrections from PSF models.

    Loads PSF models for each band, fits a 2D Moffat profile to find the
    center, measures the flux within specified apertures, computes aperture
    corrections, and generates an encircled energy plot. Results are saved
    to disk.

    Parameters
    ----------
    in_bands : `list`
        Filter band names to process.
    extract_code : `str`, optional
        Photometry code to use, either ``"sep"`` or ``"photutils"``.
        Default is ``"sep"``.
    save_loc : `str`, optional
        Output directory for results. Default is
        ``config['DEFAULT']['GALFIND_WORK']/Aperture_corrections``.
    PSF_loc : `str`, optional
        Directory containing PSF FITS files. Default is
        ``config["DEFAULT"]["PSF_DIR"]``.
    PSF_name : `str`, optional
        PSF filename prefix. Default is ``"PSF_MIRI_in_flight_opd_filter_"``.
    plot_PSF : `bool`, optional
        Whether to display PSF plots during processing. Default is `True`.
    aper_diams : `astropy.units.Quantity`, optional
        Aperture diameters to compute corrections for. Default is loaded from
        config ``SExtractor.APERTURE_DIAMS``.
    instrument_name : `str`, optional
        Instrument name used in output filename. Default is ``"NIRCam"``.
    """
    print("extract code =", extract_code)
    print_line = [
        ["# aper_diam / arcsec"]
        + [str(aper_diam.value) for aper_diam in aper_diams]
    ]
    for band in in_bands:
        print(band)
        # name = PSF_name[0] + band.replace("f", "F") + PSF_name[1]
        name = PSF_name + band
        PSFdata, pixel_scale = open_PSF_model(band, PSF_loc, name)
        print(pixel_scale)
        x_cen, y_cen = fit_2d_moffatt(PSFdata)
        flux_pcs = []
        aper_corr = []
        for aper_diam in aper_diams:
            print(aper_diam)
            aper_diam_pix = aper_diam / pixel_scale
            tot_aper_size = 9.0 * u.arcsec / pixel_scale
            flux_pc_loc, aper_corr_loc, x_cen, y_cen = calc_aper_corr(
                PSFdata,
                x_cen,
                y_cen,
                band,
                aper_diam_pix,
                plot_PSF=plot_PSF,
                extract_code=extract_code,
                PSF_loc=PSF_loc,
                PSF_name=name,
                tot_aper_size=tot_aper_size,
            )
            flux_pcs.append(flux_pc_loc)
            aper_corr.append(aper_corr_loc)
        plot_flux_curve(
            PSFdata,
            pixel_scale,
            x_cen,
            y_cen,
            band,
            flux_pcs,
            aper_corr,
            save_loc=save_loc,
            PSF_loc=PSF_loc,
            PSF_name=name,
            tot_aper_size=tot_aper_size,
            aper_diams=aper_diams,
        )
        aper_corr.insert(0, band)
        print_line.append(aper_corr)
        # print(print_line)
    np.savetxt(
        f"{save_loc}/{instrument_name}_aper_corr.txt",
        print_line,
        fmt="%s" + len(aper_diams) * " %.6s",
    )


if __name__ == "__main__":
    pass

# xi_ion_bias.py

import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from copy import deepcopy
from tqdm import tqdm

from galfind import config, ACS_WFC, NIRCam, Photometry_rest
from galfind import Mock_SED_rest, Mock_SED_obs, Mock_SED_rest_template_set
from galfind.Emission_lines import line_diagnostics

plt.style.use(f"{config['DEFAULT']['GALFIND_DIR']}/galfind_style.mplstyle")

def calc_xi_ion_bias(z_arr, m_UV = 24. * u.ABmag, template_set = "fsps_larson", renorm = True, plot = True):
    sed_rest_template_set = Mock_SED_rest_template_set.load_EAZY_in_templates(m_UV, template_set)
    #sed_obs_template_set = 
    fig, ax = plt.subplots()
    #for i, sed_rest in enumerate([sed_rest_template_set[11]]):
    sed_rest = sed_rest_template_set[11]
    if renorm:
        sed_rest.renorm_at_wav(6_560. * u.AA, 26. * u.ABmag)
    if plot:
        plot = sed_rest.plot_SED(ax, annotate = True)
        #ax.set_xlim(500., 8_000.)
        ax.set_xlim(line_diagnostics["Halpha"]["feature_wavs"].to(u.AA).value[0] - 50., line_diagnostics["Halpha"]["feature_wavs"].to(u.AA).value[1] + 50.)
        #breakpoint()
        for wav in list(line_diagnostics["Halpha"]["feature_wavs"].to(u.AA).value) \
                + list(line_diagnostics["Halpha"]["cont_wavs"].to(u.AA).value[0]) \
                + list(line_diagnostics["Halpha"]["cont_wavs"].to(u.AA).value[1]):
            ax.axvline(wav, 0., 1.)
        ax.set_ylim(26.5, 20.)
        plt.savefig("SED_rest_test.png")
        plt.clf()
    sed_obs_arr = np.array([Mock_SED_obs.from_Mock_SED_rest(mock_sed_rest, z) \
        for z, mock_sed_rest in zip(z_arr, np.full(len(z_arr), sed_rest))])
    Halpha_EWrest = sed_rest.calc_line_EW("Halpha")
    ACS_WFC_bands = ["F435W", "F606W", "F775W", "F814W", "F850LP"]
    NIRCam_bands = ["F090W", "F115W", "F150W", "F162M", "F182M", "F200W", \
        "F210M", "F250M", "F277W", "F300M", "F335M", "F356W", "F410M", "F444W"]
    instrument = ACS_WFC(excl_bands = [band.band_name for band in ACS_WFC() if band.band_name not in ACS_WFC_bands]) + \
        NIRCam(excl_bands = [band.band_name for band in NIRCam() if band.band_name not in NIRCam_bands])
    #[sed_obs.create_mock_phot(instrument) for sed_obs in sed_obs_arr]
    phot_rest_arr = [Photometry_rest.from_phot(sed_obs.create_mock_phot(instrument), z) for sed_obs, z in zip(sed_obs_arr, z_arr)]
    # plot continuum bands wavelength coverage
    plot_band_z_coverage(phot_rest_arr)
    for i in range(len(sed_obs_arr)):
        fig, ax = plt.subplots()
        sed_obs_arr[i].plot_SED(ax)
        sed_obs_arr[i].mock_phot.plot_phot(ax, mag_units = u.ABmag, \
            uplim_sigma = None, plot_errs = {"x": False, "y": False}, auto_scale = False)
        ax.set_ylim(34., 22.)
        plt.savefig("Test_plot.png")
    Halpha_EWrest_arr_obs = [phot_rest.calc_EW_rest_optical(["Halpha"], frame = "rest", single_iter = True)[0] \
        for phot_rest in tqdm(phot_rest_arr, desc = "Calculating Halpha EWs", total = len(phot_rest_arr))]
    Halpha_EWrest_template = sed_rest.calc_line_EW("Halpha")
    fig, ax = plt.subplots()
    dHalpha = [np.nan if np.isnan(obs) else (obs - Halpha_EWrest_template).to(u.AA).value for obs in Halpha_EWrest_arr_obs]
    ax.plot(z_arr, dHalpha)
    ax.set_xlabel("Redshift, z")
    ax.set_ylabel(r"$\Delta$EW$_{\mathrm{H}\alpha}$")
    plt.savefig("Halpha_bias.png")

def make_pipes_templates():
    pass

def plot_band_z_coverage(phot_rest_arr):
    fig, ax = plt.subplots()
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    kwargs_arr = [phot_rest.calc_EW_rest_optical(["Halpha"], single_iter = True)[1] for phot_rest in phot_rest_arr]
    cont_bands_arr = [kwargs["Halpha_cont_bands"] if "Halpha_cont_bands" in kwargs.keys() else "" for kwargs in kwargs_arr]
    em_bands_arr = [kwargs["Halpha_emission_band"] if "Halpha_emission_band" in kwargs.keys() else "" for kwargs in kwargs_arr]
    z_arr = [phot_rest.z for phot_rest in phot_rest_arr]
    plotted_cont = False
    plotted_em = False
    plot_bands = []
    j = 0
    for i, band in enumerate(phot_rest_arr[0].instrument.band_names):
        _cont_band_z_coverage_dict = np.array([z for i, z in enumerate(z_arr) if band in cont_bands_arr[i]])
        if len(_cont_band_z_coverage_dict) > 1:
            z_coverage = []
            ax.broken_barh([(np.min(_cont_band_z_coverage_dict), np.max(_cont_band_z_coverage_dict) - np.min(_cont_band_z_coverage_dict))], (10 * (j + 1), 9), \
                        facecolors = colours[0], label = "Continuum" if not plotted_cont else None)
            plotted_cont = True
        _em_band_z_coverage_dict = np.array([z for i, z in enumerate(z_arr) if band in em_bands_arr[i]])
        if len(_em_band_z_coverage_dict) > 1:
            ax.broken_barh([(np.min(_em_band_z_coverage_dict), np.max(_em_band_z_coverage_dict) - np.min(_em_band_z_coverage_dict))], (10 * (j + 1), 9), \
                        facecolors = colours[1], label = "Emission" if not plotted_em else None)
            plotted_em = True
        if plotted_cont or plotted_em:
            j += 1
            plot_bands.append(band)
    ax.set_xlabel('Redshift, z')
    ax.set_yticks([10 * (i + 1) + 5 for i in range(len(plot_bands))], labels = plot_bands)
    plt.legend(loc = 'lower right')
    plt.savefig("band_coverage_plot.png")

def calc_z_coverage(z, orig_z_arr):
    # find all instances of orig_z_arr that are not in z
    np.searchsorted(z, orig_z_arr)
    z_coverage = [(np.min(z), )]

def main():
    z_arr = np.linspace(2., 9., 100)
    calc_xi_ion_bias(z_arr)
    pass

if __name__ == "__main__":
    main()
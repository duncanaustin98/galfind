#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 16:46:40 2024

@author: austind
"""

# completeness_contamination.py
import numpy as np
import matplotlib.pyplot as plt
from galfind import config
from astropy.table import Table, vstack, join
import os
import h5py
from scipy.ndimage import gaussian_filter
from matplotlib.collections import LineCollection
from scipy.stats import binned_statistic, binned_statistic_2d
from copy import deepcopy
global bin_edges_dict
global frame_names
global var_labels
global mass_corr_dict

bin_dict = {"int": {"mass": np.linspace(6., 10.5, 10), "beta": np.linspace(-3.5, -0.5, 10), "M_UV": np.linspace(-22.5, -16., 10)}, \
    "obs": {"mass": np.linspace(6.5, 10.5, 8), "beta": np.linspace(-7., 0., 8), "M_UV": np.linspace(-22.5, -16.5, 8)}}
frame_names = {"int": {"z": "redshift", "mass": "mStar", "beta": "Beta_true", "M_UV": "M_UV_true"}, "obs": {"z": "zbest_fsps_larson", "mass": "stellar_mass_50", "beta": "Beta_EAZY_fsps_larson", "M_UV": "M_UV_EAZY_fsps_larson"}}
var_labels = {"z": "Redshift, z", "mass": r"$\log_{10}(M_{\star}/\mathrm{M}_{\odot})$", "beta": r"$\beta$", "M_UV": r"$M_{\mathrm{UV}}$"}
mass_corr_dict = {"lognorm_zfix": {"high-z": {"mu": 0.2436, "sigma": 0.}, "interlopers": {"mu": 0.5271, "sigma": 0.}}}
# {"high-z": {"mu": 0.257, "sigma": 0.255}, "interlopers": {"mu": 0.645, "sigma": 0.368}}

# based on Madau, Dickinson 2014 factors using FSPS models (Conroy 2009)
def MD14_mass_conversion(in_IMF, out_IMF):
    if in_IMF == "Salpeter":
        conv_factor = {"Salpeter": 1., "Kroupa": 0.66, "Chabrier": 0.61}
    elif in_IMF == "Chabrier":
        conv_factor = {"Salpeter": 1 / 0.61, "Kroupa": 0.66 / 0.61, "Chabrier": 1.}
    elif in_IMF == "Kroupa":
        conv_factor = {"Salpeter": 1 / 0.66, "Kroupa": 1., "Chabrier": 0.61 / 0.66}
    return conv_factor[out_IMF]

def bin_tab(tab, x_name, y_name, x_bins, y_bins, N_thresh = 0., smooth_sigma = None):
    x = tab[x_name]
    y = tab[y_name]
    stat, x_bin_edges, y_bin_edges, bin_no = binned_statistic_2d(x, y, None, "count", bins = [x_bins, y_bins], expand_binnumbers = True)
    stat_shape = stat.shape
    #stat = np.array([stat[i][j] if stat[i][j] > N_thresh else np.nan for i in range(stat_shape[0]) for j in range(stat_shape[1])]).reshape(stat_shape)
    if smooth_sigma == None:
        return stat.T
    else:
        return gaussian_filter(stat.T, smooth_sigma)

def calc_contamination(fig, ax, survey, instrument, x_name, y_name, z_label, crop_key = "final_sample_highz_fsps_larson", epsilon_lim = 0.15, \
                       z_bin = [6.5, 13.], frame = "obs", sim = "jaguar", version = "v9", cmap_name = "viridis", sfh_zprior = "lognorm_zfix", \
                           pipes_IMF = "Kroupa", cont_IMF = "Chabrier", plot = True, save = False):
    if survey != "CEERS":
        cat_path = f"{config['DEFAULT']['GALFIND_WORK']}/Catalogues/simulated/{instrument}/{survey}-{sim.capitalize()}/{sim.upper()}_SimDepth_{survey}_{version}_half_10pc_EAZY_matched_selection_UV.fits"
    else:
        cat_path = f"{config['DEFAULT']['GALFIND_WORK']}/Catalogues/simulated/{instrument}/{survey}-{sim.capitalize()}/{survey}-{sim.upper()}_matched_bagpipes.fits"
    z_bin_ = deepcopy(z_bin)
    if survey in ["El-Gordo", "MACS-0416", "GLASS"] and z_bin[0] < 7.5:
        z_bin_[0] = 7.5
    x_label = frame_names[frame][x_name]
    y_label = frame_names[frame][y_name]
    #print(cat_path)
    tab = Table.read(cat_path)
    tab = tab[((tab[z_label] > z_bin_[0]) & (tab[z_label] < z_bin_[1]) & (tab[crop_key]))]
    if "epsilon" not in tab.columns:
        tab["epsilon"] = ((1. + tab["zbest_fsps_larson"]) / (1. + tab["redshift"])) - 1.
    if "stellar_mass_50" not in tab.columns and (x_name == "mass" or y_name == "mass") and frame == "obs":
        stellar_mass_50_col = [mass_int + np.random.normal(mass_corr_dict[sfh_zprior]["interlopers"]["mu"], mass_corr_dict[sfh_zprior]["interlopers"]["sigma"]) \
                if epsilon > epsilon_lim or epsilon < -epsilon_lim else mass_int + np.random.normal(mass_corr_dict[sfh_zprior]["high-z"]["mu"], mass_corr_dict[sfh_zprior]["high-z"]["sigma"]) \
                    for i, (epsilon, mass_int) in enumerate(zip(tab["epsilon"], tab[frame_names["int"]["mass"]]))]
        tab["stellar_mass_50"] = stellar_mass_50_col
    
    # correct to Chabrier IMF
    #tab["stellar_mass_50"] += np.full(len(tab), np.log10(MD14_mass_conversion(pipes_IMF, cont_IMF)))
    interloper_tab = tab[((tab["epsilon"] > epsilon_lim) | (tab["epsilon"] < -epsilon_lim))]
    #high_z_tab = tab[((tab["epsilon"] < epsilon_lim) & (tab["epsilon"] > -epsilon_lim))]
    #print([np.median(tab_["stellar_mass_50"] - tab_["mStar"]) for tab_ in [high_z_tab, interloper_tab]])
    x_bins = bin_dict[frame][x_name]
    y_bins = bin_dict[frame][y_name]
    tot_binned = bin_tab(tab, x_label, y_label, x_bins, y_bins)
    interloper_binned = bin_tab(interloper_tab, x_label, y_label, x_bins, y_bins)
    contam_binned = interloper_binned / tot_binned
    
    if plot:
        im = ax.imshow(contam_binned, extent = (min(x_bins), max(x_bins), min(y_bins), max(y_bins)), \
                       cmap = cmap_name, vmin = 0., vmax = 1., aspect = "auto")
        ax.set_xlabel(var_labels[x_name])
        ax.set_ylabel(var_labels[y_name])
        if save:
            ax.yaxis.set_label_position("right")
            ax.tick_params(labelright = True)
            plt.subplots_adjust(wspace = 0.05)
            zlabel = f"{z_bin_[0]}<z<{z_bin_[1]}"
            fig.suptitle(f"{survey}, N={len(interloper_tab)}/{len(tab)}, {zlabel}", y = 0.95)
            save_path = f"{config['DEFAULT']['GALFIND_WORK']}/Bagpipes/EPOCHS_III/plots/contamination/{survey}_{zlabel}_contamination_{frame}.png"
            os.makedirs("/".join(save_path.split("/")[:-1]), exist_ok = True)
            plt.savefig(save_path)
            plt.show()
        return contam_binned, x_bins, y_bins, im
    else:
        return contam_binned, x_bins, y_bins

def calc_contamination_beta_binned(ax, galfind_tab, pipes_tab = Table(), x_name = "M_UV", crop_key = "EPOCHS_III_certain", pipes_IMF = "Kroupa", cont_IMF = "Chabrier", plot_survey = None, plot = False):
    #assert(len(galfind_tab) == len(pipes_tab))
    if x_name == "mass" and len(pipes_tab) == 0:
        raise(Exception("x_name == 'mass' and pipes_tab == 'None'. Could not bin in stellar mass if stellar mass is not given!"))
    # open contamination grid
    cont_data_path = "/raid/scratch/work/austind/GALFIND_WORK/Bagpipes/EPOCHS_III/EPOCHS_III_contamination_data.h5"
    hf = h5py.File(cont_data_path, "r")
    name_to_label_dict = {"M_UV": "M_UV_1250-3000Angstrom_conv_filt_PL", "mass": "stellar_mass_50", "beta": "Beta_1250-3000Angstrom_conv_filt_PL"}
    #print(len(np.unique(galfind_tab["FIELDNAME"])), len(np.unique(pipes_tab["pipes_FIELDNAME"])))
    
    for i, survey in enumerate(np.unique(galfind_tab["FIELDNAME"])):
        galfind_tab_survey = galfind_tab[((galfind_tab["FIELDNAME"] == survey) & (galfind_tab[crop_key]))]
        if len(pipes_tab) > 0:
            pipes_tab_survey = pipes_tab[((pipes_tab["pipes_FIELDNAME"] == survey) & (pipes_tab[crop_key]))]
            #print(len(galfind_tab_survey), len(pipes_tab_survey))
            assert(len(galfind_tab_survey) == len(pipes_tab_survey))
        IDs = np.array(galfind_tab_survey["NUMBER"])
        fieldnames = np.array(galfind_tab_survey["FIELDNAME"])
        if survey != "CLIO" and survey != "SMACS-0723":
            if x_name == "M_UV":
                corr_factor_x = galfind_tab_survey["auto_corr_factor_UV_EAZY_fsps_larson"]
                # correct this factor
                #bands = [name.split("auto_corr_factor_")[-1] for name in phot_tab.columns if "auto_corr_factor_" in name and not "UV" in name and not "mass" in name]
                corr_factor_x = -2.5 * np.log10([corr_factor if corr_factor < 10. else galfind_tab_survey["auto_corr_factor_mass"][i] for i, corr_factor in enumerate(corr_factor_x)])
                x = np.array(galfind_tab_survey[name_to_label_dict[x_name]] + corr_factor_x)
            elif x_name == "mass":
                corr_factor_x = np.log10(galfind_tab_survey["auto_corr_factor_mass"])
                x = np.array(pipes_tab_survey[name_to_label_dict[x_name]] + corr_factor_x + np.log10(MD14_mass_conversion(pipes_IMF, cont_IMF)))
            else:
                raise(Exception(""))
            y = np.array(galfind_tab_survey[name_to_label_dict["beta"]])
            #print(survey)
            if "CEERS" in survey:
                survey_ = "CEERS"
            elif "JADES" in survey:
                survey_ = "JADES"
            elif "NEP" in survey:
                survey_ = "NEP"
            else:
                survey_ = survey
            x_bins = np.array(hf.get(f"{survey_}/beta-{x_name}/x_bins"))
            y_bins = np.array(hf.get(f"{survey_}/beta-{x_name}/y_bins"))
            contam_binned = np.array(hf.get(f"{survey_}/beta-{x_name}/contamination_data"))
            #print(survey, x_bins, y_bins, contam_binned)
            x_bin_no = binned_statistic(x, None, "count", bins = x_bins)[2]
            y_bin_no = binned_statistic(y, None, "count", bins = y_bins)[2]
            #print(x, x_bin_no, y, y_bin_no)
            print(survey)
            print(np.array(x[np.argwhere(x_bin_no >= len(x_bins))]), np.array(y[np.argwhere(x_bin_no >= len(x_bins))]), np.array(IDs[np.argwhere(x_bin_no >= len(x_bins))]))
            print(np.array(x[np.argwhere(y_bin_no >= len(y_bins))]), np.array(y[np.argwhere(y_bin_no >= len(y_bins))]), np.array(IDs[np.argwhere(y_bin_no >= len(y_bins))]))
            EPOCHS_III_special_cases = {"M_UV": {"NGDEEP": {15291: 0., 15409: 0.2, 17230: 0.1}, "NEP-2": {8980: 0.}, "NEP-3": {16814: 0.25}, "NEP-4": {10695: 0.25, 12803: 0.}, "GLASS": {4546: 0.1}, \
                                        "JADES-Deep-GS": {1577: 0., 4054: 0.3, 12044: 0., 23744: 0., 29617: 0., 30485: 0., 36222: 0.}}, "mass": {"CEERSP1": {7463: 0.}, "CEERSP2": {7919: 0.}, "CEERSP5": {7520: 0., 7857: 0.}, "CEERSP7": {8028: 0., 9601: 0.}, \
                                        "CEERSP8": {2503: 0., 2777: 0.}, "JADES-Deep-GS": {29617: 0., 36222: 0., 36673: 1.}, "NEP-2": {8980: 0.}, "NEP-4": {12803: 0.}, "NGDEEP": {2330: 0., 15291: 0., 15409: 0.}}}
            cont = [contam_binned[y_bin - 1][x_bin - 1] if x_bin < len(x_bins) and y_bin < len(y_bins) else EPOCHS_III_special_cases[x_name][survey][ID] for ID, x_bin, y_bin in zip(IDs, x_bin_no, y_bin_no)]
            print(IDs[np.where(np.isnan(cont))], x[np.where(np.isnan(cont))], y[np.where(np.isnan(cont))])
            for k in np.where(np.isnan(cont))[0]:
                cont[k] = EPOCHS_III_special_cases[x_name][survey][IDs[k]]
            
            if survey_ == plot_survey:
                print(x_name, survey_)
                colour = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
                scat = ax.scatter(x, y, c = colour, s = 300, marker = "*") #, cmap = "viridis", vmin = 0., vmax = 1.)
            #print(cont)
        else: # set column to -99 for all objects
            cont = np.full(len(galfind_tab_survey), -99.)
        new_tab = Table({"contam_ID": IDs, "contam_SURVEY": fieldnames, f"cont_beta-{x_name}": cont})
        if i == 0:
            out_tab = new_tab
        else:
            out_tab = vstack([out_tab, new_tab])

    if plot:
        if x_name == "M_UV":
            loc = "lower left"
        elif x_name == "mass":
            loc = "lower right"
        ax.legend(["EPOCHS sample"], loc = loc)
    #print(out_tab)
    if f"cont_beta-{x_name}" in galfind_tab.columns:
        galfind_tab.remove_columns([f"cont_beta-{x_name}"])
    # combinegalfind and new contamination tab
    if len(pipes_tab) == 0:
        pre_combine_tab = galfind_tab
        keys_left = ["NUMBER", "FIELDNAME"]
    else:
        pre_combine_tab = pipes_tab
        keys_left = ["pipes_ID", "pipes_FIELDNAME"]
    combined_tab = join(pre_combine_tab, out_tab, keys_left = keys_left, keys_right = ["contam_ID", "contam_SURVEY"], join_type = "outer")
    for i, survey in enumerate(combined_tab["contam_SURVEY"]):
        if type(survey) == np.ma.core.MaskedConstant:
            print(i, survey)
            combined_tab[f"cont_beta-{x_name}"][i] = -99.
    combined_tab.remove_columns(["contam_ID", "contam_SURVEY"]) 
    return combined_tab

def contam_main():
    surveys = ["JADES", "CEERS", "El-Gordo", "GLASS", "MACS-0416", "NEP", "NGDEEP"] # ["CLIO", "SMACS-0723"]
    instruments = ["ACS_WFC+NIRCam" for i in range(len(surveys))]
    frame = "obs"
    x_name_arr = ["M_UV", "mass"]
    y_name = "beta"
    plot = True
    save_data = False
    EPOCHS_III_tab_path = "/raid/scratch/work/austind/GALFIND_WORK/Catalogues/v9/ACS_WFC+NIRCam/Combined/EPOCHS_MASTER_Sel-f277W+f356W+f444W_v9_loc_depth_masked_10pc_EAZY_matched_selection_ext_src_UV_pipes_v2_final_EPOCHS_III.fits"
    pipes_name = "lognorm_fesc_zfix"
    if plot:
        galfind_tab = Table.read(EPOCHS_III_tab_path)
        pipes_tab = Table.read(EPOCHS_III_tab_path, hdu = f"pipes_{pipes_name}".upper())
        for survey, instrument in zip(surveys, instruments):
            fig, ax = plt.subplots(ncols = len(x_name_arr), figsize = (12, 6), sharey = True)
            for j, (ax_, x_name) in enumerate(zip(ax, x_name_arr)):
                calc_contamination_beta_binned(ax_, galfind_tab, pipes_tab, x_name, plot_survey = survey)
                im = calc_contamination(fig, ax_, survey, instrument, x_name, y_name, frame_names[frame]["z"], \
                                   frame = frame, plot = True, save = True if j == len(x_name_arr) - 1 else False)[3]
                if j == 0:
                    fig.colorbar(im, ax = ax.ravel(), label = "Contamination", location = "bottom", \
                                 orientation = "horizontal", shrink = 0.9, aspect = 40., anchor = (0.5, -2.))
    if save_data:
        save_path = f"{config['DEFAULT']['GALFIND_WORK']}/Bagpipes/EPOCHS_III/EPOCHS_III_contamination_data.h5"
        os.makedirs("/".join(save_path.split("/")[:-1]), exist_ok = True)
        hf = h5py.File(save_path, "w")
        for i, (survey, instrument) in enumerate(zip(surveys, instruments)):
            survey_contam = hf.create_group(survey)
            for j, x_name in enumerate(x_name_arr):
                contam_binned, x_bins, y_bins = calc_contamination(None, None, survey, instrument, x_name, y_name, frame_names[frame]["z"], frame = frame, plot = False)
                survey_contam_mass_lum = survey_contam.create_group(f"{y_name}-{x_name}")
                survey_contam_mass_lum.create_dataset("x_bins", data = np.array(x_bins))
                survey_contam_mass_lum.create_dataset("y_bins", data = np.array(y_bins))
                survey_contam_mass_lum.create_dataset("contamination_data", data = np.array(contam_binned))
        hf.close()
        
def add_nan_line(ax, im, x_bins, y_bins, value = 0., color = "red", ls = "-."):
    v = np.diff(im > value, axis=1)
    h = np.diff(im > value, axis=0)

    l = np.argwhere(v.T)
    vlines = np.array(list(zip(np.stack((x_bins[l[:, 0] + 1], y_bins[l[:, 1]])).T,
                               np.stack((x_bins[l[:, 0] + 1], y_bins[l[:, 1] + 1])).T)))
    l = np.argwhere(h.T)
    hlines = np.array(list(zip(np.stack((x_bins[l[:, 0]], y_bins[l[:, 1] + 1])).T,
                               np.stack((x_bins[l[:, 0] + 1], y_bins[l[:, 1] + 1])).T)))
    lines = LineCollection(np.vstack((vlines, hlines)), lw=1, colors = color, ls = ls)
    ax.add_collection(lines)
    return lines

def completeness_main():
    surveys = ["NGDEEP"]
    plot = True
    frame = "int"
    x_name_arr = ["M_UV", "mass"]
    y_name = "beta"
    N_thresh = 0
    z_range_arr = [[6.5, 8.5], [8.5, 11.], [11., 13.]]
    
    y_label = frame_names[frame][y_name]
    y_bins = bin_dict[frame][y_name]

    comp_data_path = "/raid/scratch/work/austind/GALFIND_WORK/Bagpipes/EPOCHS_III/EPOCHS_III_completeness_data.h5"
    hf = h5py.File(comp_data_path, "w")

    for survey in surveys:
        #hf_survey = hf.create_group(survey)
        cat_path = f"/raid/scratch/work/austind/GALFIND_WORK/Catalogues/simulated/ACS_WFC+NIRCam/{survey}-Jaguar/JAGUAR_SimDepth_{survey}_v9_half_10pc_EAZY_matched_selection_UV.fits"
        tab = Table.read(cat_path)
        for x_name in x_name_arr:
            x_label = frame_names[frame][x_name]
            x_bins = bin_dict[frame][x_name]
            #hf_survey_xname = hf_survey.create_group(x_name)
            fig, ax = plt.subplots(nrows = len(z_range_arr), figsize = (4, 4 * len(z_range_arr)), sharex = True)
            for i, z_range in enumerate(z_range_arr):
                hf_survey_xname_z = hf.create_group(f"{survey}/{x_name}/{z_range}")
                z_label = f"{z_range[0]}<z<{z_range[1]}"
                tab[z_label] = ((tab["redshift"] < z_range[1]) & (tab["redshift"] > z_range[0]))
                tab[f"{z_label}_selected"] = ((tab[z_label]) & (tab["final_sample_highz_fsps_larson"]))
                tab_z_range = tab[tab[z_label]]
                tab_z_range_selected = tab[tab[f"{z_label}_selected"]]

                tot_binned = bin_tab(tab_z_range, x_label, y_label, x_bins, y_bins) #, smooth_sigma = 0.5)
                selected_binned = bin_tab(tab_z_range_selected, x_label, y_label, x_bins, y_bins) #, smooth_sigma = 0.5)
                completeness_binned = selected_binned / tot_binned

                hf_survey_xname_z.create_dataset("counts_binned", data = tot_binned)
                hf_survey_xname_z.create_dataset("completeness_binned", data = completeness_binned)
                hf_survey_xname_z.create_dataset("x_bins", data = x_bins) 
                hf_survey_xname_z.create_dataset("y_bins", data = y_bins) 

                if plot:
                    #im = ax[i].imshow(tot_binned, extent = (min(x_bins), max(x_bins), min(y_bins), max(y_bins)), \
                    #                cmap = "viridis", origin = "lower")
                    im = ax[i].imshow(completeness_binned, extent = (min(x_bins), max(x_bins), min(y_bins), max(y_bins)), \
                                    vmin = 0., vmax = 1., cmap = "viridis", origin = "lower")
                    ax[i].contour(tot_binned, levels = [1.], colors = "black", ls = "-.", linewidths = 1., extent = (min(x_bins), max(x_bins), min(y_bins), max(y_bins)), origin = "lower")
                    contours_20pc = ax[i].contour(completeness_binned, levels = [0.2], colors = "red", extent = (min(x_bins), max(x_bins), min(y_bins), max(y_bins)), origin = "lower")
                    sim_edge = add_nan_line(ax[i], completeness_binned, x_bins, y_bins)
                    ax[i].text(0.95, 0.05, z_label, ha = "right", va = "bottom", transform = ax[i].transAxes)
                    ax[i].set_ylabel(r"$\beta$")

            if plot:
                ax[-1].set_xlabel(x_name)
                fig.colorbar(im, ax = ax.ravel())
                plt.subplots_adjust(hspace = 0.)

                #plt.legend()
                plt.savefig(f"completeness_{y_name}-{x_name}.png")
                plt.show()
    hf.close()

if __name__ == "__main__":
    contam_main()
    #completeness_main()
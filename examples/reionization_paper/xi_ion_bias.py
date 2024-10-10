# xi_ion_bias.py

from copy import deepcopy

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe
from tqdm import tqdm

from galfind import (
    ACS_WFC,
    Mock_SED_obs,
    Mock_SED_rest_template_set,
    NIRCam,
    Photometry_rest,
    config,
)
from galfind.Emission_lines import line_diagnostics, strong_optical_lines

plt.style.use(f"{config['DEFAULT']['GALFIND_DIR']}/galfind_style.mplstyle")


def calc_xi_ion_bias(
    z_arr,
    m_UV=24.0 * u.ABmag,
    template_set="fsps_larson",
    renorm=True,
    plot=True,
):
    sed_rest_template_set = Mock_SED_rest_template_set.load_EAZY_in_templates(
        m_UV, template_set
    )

    for i, sed_rest in enumerate([sed_rest_template_set[0]]):
        fig, ax = plt.subplots()
        # sed_rest = sed_rest_template_set[2]
        if renorm:
            sed_rest.renorm_at_wav(6_560.0 * u.AA, 26.0 * u.ABmag)
        if plot:
            plot = sed_rest.plot_SED(ax, annotate=True)
            # ax.set_xlim(500., 8_000.)
            ax.set_xlim(
                line_diagnostics["Halpha"]["feature_wavs"].to(u.AA).value[0]
                - 50.0,
                line_diagnostics["Halpha"]["feature_wavs"].to(u.AA).value[1]
                + 50.0,
            )
            # breakpoint()
            for wav in (
                list(line_diagnostics["Halpha"]["feature_wavs"].to(u.AA).value)
                + list(
                    line_diagnostics["Halpha"]["cont_wavs"].to(u.AA).value[0]
                )
                + list(
                    line_diagnostics["Halpha"]["cont_wavs"].to(u.AA).value[1]
                )
            ):
                ax.axvline(wav, 0.0, 1.0)
            ax.set_ylim(26.5, 20.0)
            plt.savefig(
                f"test_plots/SED_rest_test_{sed_rest.template_name}.png"
            )
            plt.clf()
        sed_obs_arr = np.array(
            [
                Mock_SED_obs.from_Mock_SED_rest(mock_sed_rest, z)
                for z, mock_sed_rest in zip(
                    z_arr, np.full(len(z_arr), sed_rest)
                )
            ]
        )
        Halpha_EWrest = sed_rest.calc_line_EW("Halpha")
        ACS_WFC_bands = ["F435W", "F606W", "F775W", "F814W", "F850LP"]
        NIRCam_bands = [
            "F090W",
            "F115W",
            "F150W",
            "F162M",
            "F182M",
            "F200W",
            "F210M",
            "F250M",
            "F277W",
            "F300M",
            "F335M",
            "F356W",
            "F410M",
            "F444W",
        ]
        instrument = ACS_WFC(
            excl_bands=[
                band.band_name
                for band in ACS_WFC()
                if band.band_name not in ACS_WFC_bands
            ]
        ) + NIRCam(
            excl_bands=[
                band.band_name
                for band in NIRCam()
                if band.band_name not in NIRCam_bands
            ]
        )
        # [sed_obs.create_mock_phot(instrument) for sed_obs in sed_obs_arr]
        phot_rest_arr = [
            Photometry_rest.from_phot(sed_obs.create_mock_phot(instrument), z)
            for sed_obs, z in zip(sed_obs_arr, z_arr)
        ]
        # plot continuum bands wavelength coverage
        plot_band_z_coverage(phot_rest_arr)
        # for i in range(len(sed_obs_arr)):
        #     fig, ax = plt.subplots()
        #     sed_obs_arr[i].plot_SED(ax)
        #     sed_obs_arr[i].mock_phot.plot(ax, mag_units = u.ABmag, \
        #         uplim_sigma = None, plot_errs = {"x": False, "y": False}, auto_scale = False)
        #     ax.set_ylim(34., 22.)
        #     plt.savefig(f"test_plots/Test_plot_{sed_rest.template_name}.png")
        Halpha_EWrest_arr_obs = [
            deepcopy(phot_rest).calc_EW_rest_optical(
                ["Halpha"], frame="rest", single_iter=True
            )[0]
            for phot_rest in tqdm(
                phot_rest_arr,
                desc="Calculating Halpha EWs",
                total=len(phot_rest_arr),
            )
        ]
        Halpha_EWrest_template = sed_rest.calc_line_EW("Halpha")
        fig, ax = plt.subplots()
        dHalpha = [
            np.nan
            if np.isnan(obs)
            else (obs - Halpha_EWrest_template).to(u.AA).value
            for obs in Halpha_EWrest_arr_obs
        ]
        ax.plot(z_arr, dHalpha)
        ax.set_xlabel("Redshift, z")
        ax.set_ylabel(r"$\Delta$EW$_{\mathrm{H}\alpha}$")
        plt.savefig(f"test_plots/Halpha_bias_{sed_rest.template_name}.png")


def make_pipes_templates():
    pass


def plot_band_z_coverage(phot_rest_arr):
    fig, ax = plt.subplots()
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    kwargs_arr = [
        deepcopy(phot_rest).calc_EW_rest_optical(["Halpha"], single_iter=True)[
            1
        ]
        for phot_rest in phot_rest_arr
    ]
    cont_bands_arr = [
        kwargs["Halpha_cont_bands"]
        if "Halpha_cont_bands" in kwargs.keys()
        else ""
        for kwargs in kwargs_arr
    ]
    em_bands_arr = [
        kwargs["Halpha_emission_band"]
        if "Halpha_emission_band" in kwargs.keys()
        else ""
        for kwargs in kwargs_arr
    ]
    z_arr = [phot_rest.z for phot_rest in phot_rest_arr]
    # print([(z_, cont_bands_, em_band_) for z_, cont_bands_, em_band_ in zip(z_arr, cont_bands_arr, em_bands_arr)])
    # breakpoint()
    plotted_cont = False
    plotted_em = False
    plot_bands = []
    j = 0
    cmap = plt.get_cmap("Spectral_r", len(phot_rest_arr[0].instrument))
    already_labelled = []
    for i, band in enumerate(phot_rest_arr[0].instrument):
        band_name = band.band_name
        # colour = cmap[i]
        _cont_band_z_coverage = np.array(
            [z for i, z in enumerate(z_arr) if band_name in cont_bands_arr[i]]
        )
        if len(_cont_band_z_coverage) > 1:
            z_coverage = calc_z_coverage(
                _cont_band_z_coverage, z_arr, band_name
            )
            plot_band_coverage(
                ax,
                z_coverage,
                band,
                coverage_type="continuum",
                label=True if "continuum" not in already_labelled else False,
            )
            if "continuum" not in already_labelled:
                already_labelled.append("continuum")
            # ax.broken_barh(z_coverage, (10 * (j + 1), 9), \
            #            facecolors = colours[0], label = "Continuum" if not plotted_cont else None)
            plotted_cont = True
        _em_band_z_coverage = np.array(
            [z for i, z in enumerate(z_arr) if band_name in em_bands_arr[i]]
        )
        if len(_em_band_z_coverage) > 1:
            z_coverage = calc_z_coverage(_em_band_z_coverage, z_arr, band_name)
            plot_band_coverage(
                ax,
                z_coverage,
                band,
                coverage_type="emission",
                label=True if "emission" not in already_labelled else False,
            )
            if "emission" not in already_labelled:
                already_labelled.append("emission")
            # ax.broken_barh(z_coverage, (10 * (j + 1), 9), \
            #            facecolors = colours[1], label = "Emission" if not plotted_em else None)
            plotted_em = True
        if plotted_cont or plotted_em:
            j += 1
            plot_bands.append(band_name)
    for line in strong_optical_lines:
        ax.axhline(
            line_diagnostics[line]["line_wav"].to(u.AA).value,
            c="darkgreen",
            lw=2.0,
            ls="--",
            zorder=120.0,
        )
        if line in ["Halpha", "[OIII]-4959"]:
            line_name = {
                "Halpha": r"H$\alpha$",
                "[OIII]-4959": r"[OIII]+H$\beta$",
            }
            ax.text(
                ax.get_xlim()[1],
                line_diagnostics[line]["line_wav"].to(u.AA).value,
                line_name[line],
                size=12.0,
                ha="right",
                va="center",
                zorder=150.0,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            )
    ax.set_xlabel("Redshift, z")
    ax.set_ylabel(r"$\lambda_{\mathrm{rest}}~/~\mathrm{\AA}$")
    ax.grid(False)
    plt.legend(loc="upper right")
    plt.savefig("band_coverage_plot.png")


def calc_z_coverage(z, orig_z_arr, band_name=None):
    # find all instances of orig_z_arr that are not in z
    matching_indices = np.searchsorted(orig_z_arr, z)
    start_indices = [
        i
        for i in matching_indices
        if i - 1 not in matching_indices and i + 1 in matching_indices
    ]
    end_indices = [
        i
        for i in matching_indices
        if i + 1 not in matching_indices and i - 1 in matching_indices
    ]
    assert len(start_indices) == len(end_indices)
    # split_arr = np.array_split(matching_indices, end_indices)
    z_coverage = [
        (orig_z_arr[start_index], orig_z_arr[end_index])
        for start_index, end_index in zip(start_indices, end_indices)
    ]
    return z_coverage


def plot_band_coverage(
    ax, z_coverage, band, delta_z=0.01, coverage_type="emission", label=False
):
    if coverage_type == "emission":
        kwargs = {
            "color": "red",
            "ec": "black",
            "lw": 2.5,
            "hatch": "X",
            "alpha": 0.5,
            "zorder": 100.0,
        }
        label_colour = "pink"
    elif coverage_type == "continuum":
        kwargs = {"color": "blue", "ec": "black", "alpha": 0.8, "lw": 2.5}
        label_colour = "lightblue"
    for z_coverage_ in z_coverage:
        if z_coverage_ == z_coverage[-1] and label:
            kwargs["label"] = coverage_type.capitalize()
        z_range = np.arange(z_coverage_[0], z_coverage_[1] + delta_z, delta_z)
        ax.fill_between(
            z_range,
            band.WavelengthLower50.to(u.AA).value / (1.0 + z_range),
            band.WavelengthUpper50.to(u.AA).value / (1.0 + z_range),
            **kwargs,
        )
        z_mean = np.mean(z_range)
        ax.text(
            z_mean,
            (
                (
                    band.WavelengthLower50.to(u.AA).value
                    + band.WavelengthUpper50.to(u.AA).value
                )
                / 2.0
            )
            / (1.0 + z_mean),
            band.band_name,
            c=label_colour,
            size=10.0,
            va="center",
            ha="center",
            weight="bold",
            zorder=150.0,
            path_effects=[pe.withStroke(linewidth=3, foreground="black")],
        )


def main():
    z_arr = np.linspace(4.5, 7.0, 5001)
    calc_xi_ion_bias(z_arr)
    pass


if __name__ == "__main__":
    main()


from astropy import units as u
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

from galfind import Catalogue, EPOCHS_Selector, EAZY
from galfind.Data import morgan_version_to_dir

def main(
    survey,
    version,
    instrument_names = ["ACS_WFC", "NIRCam"],
    forced_phot_band = ["F277W", "F356W", "F444W"],
    aper_diams = [0.32] * u.arcsec,
    SED_fitter_arr = [EAZY({"templates": "fsps_larson", "lowz_zmax": None})],
    crops = None,
):
    cat = Catalogue.pipeline(
        survey,
        version,
        instrument_names = instrument_names,
        forced_phot_band = forced_phot_band,
        aper_diams = aper_diams,
        version_to_dir_dict = morgan_version_to_dir,
        crops = crops,
    )
    for SED_fitter in SED_fitter_arr:
        SED_fitter(cat, aper_diams[0], update = True)

    for gal in [cat[0]]:
        # plot out PDFs
        fig, ax = plt.subplots()
        # get colour cycler
        colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        PDF_x_arr = {}
        PDF_p_x_arr = {}
        assert len(SED_fitter_arr) == 2
        for i, SED_fitter in enumerate(SED_fitter_arr):
            zbest = gal.aper_phot[aper_diams[0]].SED_results[SED_fitter.label].z
            chi_sq = gal.aper_phot[aper_diams[0]].SED_results[SED_fitter.label].chi_sq
            PDF = gal.aper_phot[aper_diams[0]].SED_results[SED_fitter.label].property_PDFs["z"]
            # get peak index
            peak_idx = np.argmax(PDF.p_x)
            peak_z = PDF.x[peak_idx]
            peak_p_x = PDF.p_x[peak_idx]
            computed_chi_sq = -2.0 * np.log(peak_p_x) #+ 2.0 * np.log(np.trapz(PDF.p_x, PDF.x))
            print(computed_chi_sq, chi_sq, peak_z, zbest)
            breakpoint()

            # if not hasattr(PDF, "input_arr"):
            #     input_arr = PDF.draw_sample(10_000)
            # else:
            #     input_arr = PDF.input_arr

            # if isinstance(input_arr, tuple([u.Quantity, u.Magnitude, u.Dex])):
            #     input_arr = input_arr.value
            # # if log:
            # #     input_arr = np.log10(input_arr)

            # clip input array to max redshift
            z_mask = PDF.x <= 4.0
            PDF_x = PDF.x[z_mask]
            PDF_p_x = PDF.p_x[z_mask]
            # normalize PDF
            PDF_p_x /= np.trapz(PDF_p_x, PDF_x)
            PDF_x_arr[SED_fitter.label] = PDF_x
            PDF_p_x_arr[SED_fitter.label] = PDF_p_x

            # breakpoint()
            # kde = gaussian_kde(input_arr)
            # x = np.linspace(
            #     np.min(input_arr),
            #     np.max(input_arr),
            #     len(input_arr)
            # )
            # y = kde(x)

            # ax.plot(
            #     PDF_x,
            #     PDF_p_x,
            #     color = colours[i],
            #     label = SED_fitter.label,
            #     lw = 3 * i + 1,
            #     zorder = 10 - i,
            #     #**pdf_kwargs
            # )
            #PDF.plot(ax, annotate = False, colour = colours[i], label = SED_fitter.label)
            #ax.plot(PDF.x, PDF.y, label = SED_fitter.label)
            #breakpoint()

        # interpolate PDFs to be on the same x-axis
        for i, SED_fitter in enumerate(SED_fitter_arr):
            PDF_p_x_arr[SED_fitter.label] = np.interp(
                PDF_x_arr[SED_fitter.label],
                PDF_x_arr[SED_fitter_arr[0].label],
                PDF_p_x_arr[SED_fitter.label]
            )
        delta_PDF_p_x = PDF_p_x_arr[SED_fitter_arr[0].label] - PDF_p_x_arr[SED_fitter_arr[1].label]
        if any (delta_PDF_p_x > 1e-4):
            print("Significant difference in PDFs for galaxy", gal.ID)
            breakpoint()
        #print(PDF_x_arr[SED_fitter_arr[0].label] - PDF_x_arr[SED_fitter_arr[1].label])
        #print(delta_PDF_p_x)
        #breakpoint()
        ax.plot(
            PDF_x_arr[SED_fitter_arr[0].label],
            delta_PDF_p_x,
            color = "k",
            label = "Difference"
        )
        ax.set_xlabel(r"Redshift, $z$")
        #ax.set_xlim(0.0, 4.0)
        ax.set_xlim(0.6, 1.2)
        ax.set_ylim(0.0, None)
        ax.legend()
        plt.savefig("test_PDF_zmax.png")
        plt.close(fig)


if __name__ == "__main__":
    
    survey = "CEERSP1"
    version = "v14"
    instrument_names = ["ACS_WFC", "NIRCam"]
    forced_phot_band = ["F277W", "F356W", "F444W"]
    
    aper_diams = [0.32] * u.arcsec
    SED_fitter_arr = [
        EAZY({"templates": "fsps_larson", "lowz_zmax": 4.0}),
        #EAZY({"templates": "fsps_larson", "lowz_zmax": 6.0}),
        EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
    ]
    sample = EPOCHS_Selector(aper_diams[0], SED_fitter_arr[-1], forced_phot_band)

    main(
        survey,
        version,
        instrument_names,
        forced_phot_band,
        crops = sample,
        SED_fitter_arr = SED_fitter_arr,
        aper_diams = aper_diams
    )
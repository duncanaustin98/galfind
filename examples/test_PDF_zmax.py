
from astropy import units as u
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

from galfind import Catalogue, EPOCHS_Selector, EAZY
from galfind.imaging.Data import morgan_version_to_dir

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
        SED_fitter(cat, aper_diams[0], update = True, lowz_zmax_arr = [4.0])
    
    zbest_z4, chi2_best_z4 = eazy_load()
    zbest_arr = {SED_fitter.label: [] for SED_fitter in SED_fitter_arr}
    chi_sq_arr = {SED_fitter.label: [] for SED_fitter in SED_fitter_arr}
    delta_chi_sq = []
    delta_zbest = []
    zfree_zbest_arr = []
    chi_sq_computed_arr = []
    zbest_computed_arr = []
    for gal in cat: #cat[0]]:
        # plot out PDFs
        #fig, ax = plt.subplots()
        # get colour cycler
        colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        #PDF_x_arr = {}
        #PDF_p_x_arr = {}
        #computed_chi_sq_arr = {}
        #cropped_chi_sq_arr = {}
        assert len(SED_fitter_arr) == 2
        for i, SED_fitter in enumerate(SED_fitter_arr):
            zbest = gal.aper_phot[aper_diams[0]].SED_results[SED_fitter.label].z
            chi_sq = gal.aper_phot[aper_diams[0]].SED_results[SED_fitter.label].chi_sq
            zbest_arr[SED_fitter.label].append(zbest)
            chi_sq_arr[SED_fitter.label].append(chi_sq)
            if i == 0:
                idx = gal.ID - 1
                chi_sq_computed_arr.append(chi2_best_z4[idx])
                zbest_computed_arr.append(zbest_z4[idx])
            else:
                zfree_zbest_arr.append(zbest)
            #PDF = gal.aper_phot[aper_diams[0]].SED_results[SED_fitter.label].property_PDFs["z"]
            # # get peak index
            # peak_idx = np.argmax(PDF.p_x)
            # peak_z = PDF.x#[peak_idx]
            # peak_p_x = PDF.p_x#[peak_idx]
            # computed_chi_sq = -2.0 * np.log(peak_p_x) #+ 2.0 * np.log(np.trapz(PDF.p_x, PDF.x))
            # print(computed_chi_sq)
            # breakpoint()
            #computed_chi_sq_arr[SED_fitter.label] = computed_chi_sq
            # print(computed_chi_sq, chi_sq, peak_z, zbest)
            # breakpoint()

            # if not hasattr(PDF, "input_arr"):
            #     input_arr = PDF.draw_sample(10_000)
            # else:
            #     input_arr = PDF.input_arr

            # if isinstance(input_arr, tuple([u.Quantity, u.Magnitude, u.Dex])):
            #     input_arr = input_arr.value
            # # if log:
            # #     input_arr = np.log10(input_arr)

            # # clip input array to max redshift
            # z_mask = PDF.x <= 4.0
            # PDF_x = PDF.x[z_mask]
            # PDF_p_x = PDF.p_x[z_mask]
            # # normalize PDF
            # PDF_p_x /= np.trapz(PDF_p_x, PDF_x)
            # PDF_x_arr[SED_fitter.label] = PDF_x
            # PDF_p_x_arr[SED_fitter.label] = PDF_p_x

            # peak_idx = np.argmax(PDF_p_x)
            # peak_z = PDF_x[peak_idx]
            # peak_p_x = PDF_p_x[peak_idx]
            # new_chi_sq = -2.0 * np.log(peak_p_x) #+ 2.0 * np.log(np.trapz(PDF.p_x, PDF.x))
            # cropped_chi_sq_arr[SED_fitter.label] = new_chi_sq

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
        #delta_chi_sq = chi_sq_arr[SED_fitter_arr[0].label] - chi_sq_arr[SED_fitter_arr[1].label]
        #delta_computed_chi_sq = computed_chi_sq_arr[SED_fitter_arr[0].label] - computed_chi_sq_arr[SED_fitter_arr[1].label]
        #delta_cropped_chi_sq = cropped_chi_sq_arr[SED_fitter_arr[1].label] - computed_chi_sq_arr[SED_fitter_arr[1].label]
        # print("Delta chi_sq:", delta_chi_sq)
        # #print("Delta computed chi_sq:", delta_chi_sq)
        # print("Delta cropped chi_sq:", delta_cropped_chi_sq)
        delta_chi_sq.append(chi_sq_arr[SED_fitter_arr[0].label][-1] - chi_sq_computed_arr[-1])
        delta_zbest.append(zbest_arr[SED_fitter_arr[0].label][-1] - zbest_computed_arr[-1])
        #print(f"{gal.ID=} Chi^2 with zmax=4.0 computed:", chi_sq_computed_arr)
        #print(f"{gal.ID=} Chi^2 with zmax=4.0 run:", chi_sq_arr[SED_fitter_arr[0].label])
        #print("zbest with zmax=4.0:", zbest_arr[SED_fitter_arr[0].label])
        #print("zbest with zmax=None:", zbest_arr[SED_fitter_arr[1].label])
    #breakpoint()
        # # interpolate PDFs to be on the same x-axis
        # for i, SED_fitter in enumerate(SED_fitter_arr):
        #     PDF_p_x_arr[SED_fitter.label] = np.interp(
        #         PDF_x_arr[SED_fitter.label],
        #         PDF_x_arr[SED_fitter_arr[0].label],
        #         PDF_p_x_arr[SED_fitter.label]
        #     )
        # delta_PDF_p_x = PDF_p_x_arr[SED_fitter_arr[0].label] - PDF_p_x_arr[SED_fitter_arr[1].label]
        # if any (delta_PDF_p_x > 1e-4):
        #     print("Significant difference in PDFs for galaxy", gal.ID)
        #     breakpoint()
        # #print(PDF_x_arr[SED_fitter_arr[0].label] - PDF_x_arr[SED_fitter_arr[1].label])
        # #print(delta_PDF_p_x)
        # #breakpoint()
        # ax.plot(
        #     PDF_x_arr[SED_fitter_arr[0].label],
        #     delta_PDF_p_x,
        #     color = "k",
        #     label = "Difference"
        # )
        # ax.set_xlabel(r"Redshift, $z$")
        # #ax.set_xlim(0.0, 4.0)
        # ax.set_xlim(0.6, 1.2)
        # ax.set_ylim(0.0, None)
        # ax.legend()
        # plt.savefig("test_PDF_zmax.png")
        # plt.close(fig)

    # crop out the -1s
    print(len(cat))
    assert len(delta_chi_sq) == len(cat)
    mask = (
        (np.array(chi_sq_computed_arr) > 0.) & \
        (np.array(chi_sq_arr[SED_fitter_arr[0].label]) > 0.) & \
        (np.array(chi_sq_arr[SED_fitter_arr[1].label]) > 0.) & \
        (np.array(zbest_computed_arr) > 0.) & \
        (np.array(zbest_arr[SED_fitter_arr[0].label]) > 0.) & \
        (np.array(zbest_arr[SED_fitter_arr[1].label]) > 0.)
    )
    delta_chi_sq = np.array(delta_chi_sq)[mask]
    delta_zbest = np.array(delta_zbest)[mask]
    zfree_zbest_arr = np.array(zfree_zbest_arr)[mask]
    print(len(delta_chi_sq))

    fig, ax = plt.subplots()
    ax.scatter(delta_zbest, delta_chi_sq, c = zfree_zbest_arr)
    ax.set_xlabel(r"$\Delta z$")
    ax.set_ylabel(r"$\Delta \chi^2$")
    # make colorbar
    cbar = plt.colorbar(ax.collections[0], ax = ax)
    cbar.set_label(r"$z_\mathrm{best, zmax=None}$")
    plt.savefig("test_PDF_zmax_delta_chi_sq.png", bbox_inches = "tight", dpi = 300)
    plt.close(fig)

    # fig, ax = plt.subplots()
    # ax.scatter(zbest_arr[SED_fitter_arr[0].label], zbest_computed_arr, label = f"{SED_fitter_arr[0].label} (zmax=4.0)")
    # ax.set_xlabel(r"$z_\mathrm{best, zmax=4.0}$")
    # ax.set_ylabel(r"$z_\mathrm{best, zmax=4.0, computed}$")
    # ax.legend()
    # plt.savefig("test_PDF_zmax_zbest_comparison.png")
    # plt.close(fig)

def eazy_load(h5_path = "/raid/scratch/work/austind/GALFIND_WORK/EAZY/output/ACS_WFC+NIRCam/v14/CEERSP1/CEERSP1_MASTER_Sel-F277W+F356W+F444W_v14_0.32as_EAZY_fsps_larson_zfree.h5"):
    import h5py
    from eazy import hdf5, photoz
    from astropy.table import Table
    fit = hdf5.initialize_from_hdf5(h5file=h5_path, verbose=True)
    #print(fit.zgrid)
    #breakpoint()
    #lowz_h5_path = "/raid/scratch/work/austind/GALFIND_WORK/EAZY/output/ACS_WFC+NIRCam/v14/CEERSP1/CEERSP1_MASTER_Sel-F277W+F356W+F444W_v14_0.32as_EAZY_fsps_larson_zmax=4.0.h5"
    #lowz_fit = hdf5.initialize_from_hdf5(h5file=lowz_h5_path, verbose=True)
    #breakpoint()
    from copy import deepcopy
    fit_copy = deepcopy(fit)
    # crop to zmax = 4.0
    #breakpoint()
    zgrid_mask = np.array([i for i in range(len(fit_copy.zgrid)) if fit_copy.zgrid[i] <= 4.0])
    fit_copy.chi2_fit = fit_copy.chi2_fit[:, zgrid_mask]
    fit_copy.fit_coeffs = fit_copy.fit_coeffs[:, zgrid_mask, :]
    fit_copy.tef_lnp = fit_copy.tef_lnp[:, zgrid_mask]
    fit_copy.zgrid = fit_copy.zgrid[zgrid_mask]
    fit_copy.trdz = fit_copy.trdz[zgrid_mask]
    #breakpoint()
    fit_copy.lnp = fit_copy.lnp[:, zgrid_mask]
    #breakpoint()
    fit_copy.fit_at_zbest()
    return fit_copy.zbest,fit_copy.chi2_best
    # out = fit.fit_at_zbest(selection = selection)
    #breakpoint()
    # param = hdf5.param_from_hdf5(h5file=h5_path)
    # cat, trans = hdf5.cat_from_hdf5(h5file=h5_path)
    # # trans_ = Table()
    # # breakpoint()
    # param['CATALOG_FILE'] = cat

    # with h5py.File(h5_path, 'r') as f:
    #     pzobj = photoz.PhotoZ(
    #         param_file=None, translate_file=trans,
    #         zeropoint_file=None,
    #         params=param.params,
    #         load_prior=True, 
    #         load_products=False, 
    #         tempfilt_data=f['fit/tempfilt'][:],
    #     )
    
    # # extract chi2 distribution
    # import h5py as h5
    # h5_file = h5.File(h5_path, "r")
    # chi2_fit = h5_file["fit"]["chi2_fit"][:]
    # has_chi2 = (chi2_fit != 0).sum(axis=1) > 0
    # loglike = -chi2_fit[has_chi2,:]/2.

if __name__ == "__main__":
    
    survey = "CEERSP1"
    version = "v14"
    instrument_names = ["ACS_WFC", "NIRCam"]
    forced_phot_band = ["F277W", "F356W", "F444W"]
    
    aper_diams = [0.32] * u.arcsec
    SED_fitter_arr = [
        #EAZY({"templates": "fsps_larson", "lowz_zmax": 4.0}),
        #EAZY({"templates": "fsps_larson", "lowz_zmax": 6.0}),
        EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
    ]
    sample = None #EPOCHS_Selector(aper_diams[0], SED_fitter_arr[-1], forced_phot_band)

    main(
        survey,
        version,
        instrument_names,
        forced_phot_band,
        crops = sample,
        SED_fitter_arr = SED_fitter_arr,
        aper_diams = aper_diams
    )

    #eazy_load()
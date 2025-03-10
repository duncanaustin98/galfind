
import astropy.units as u

from galfind.Data import morgan_version_to_dir
from galfind import PSF_Cutout, Filter, Galfit_Fitter, Catalogue, EPOCHS_Selector, EAZY, Brown_Dwarf_Fitter, Brown_Dwarf_Selector

# Load in a JOF data object
survey = "JOF"
version = "v11"
instrument_names = ["NIRCam"] # "ACS_WFC",
aper_diams = [0.32] * u.arcsec
forced_phot_band = ["F277W", "F356W", "F444W"] #["F814W"]
min_flux_pc_err = 10.

def main():
    SED_fit_params_arr = [{"templates": "fsps_larson", "lowz_zmax": None}]
    
    # import time
    # time.sleep(6 * 60 * 60)
    
    cat = Catalogue.pipeline(
        survey,
        version,
        instrument_names = instrument_names, 
        version_to_dir_dict = morgan_version_to_dir,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        min_flux_pc_err = min_flux_pc_err,
        #crops = "Austin+25_EAZY_fsps_larson_zfree_0.32as", #EPOCHS_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), allow_lowz=True)
    )
    eazy_fitter = EAZY(SED_fit_params_arr[0])
    eazy_fitter(cat, aper_diams[0], load_PDFs = True, load_SEDs = True, update = True)
    bd_fitter = Brown_Dwarf_Fitter()
    bd_fitter(cat, aper_diams[0], update = True)

    cat.load_sextractor_Re()
    cat.load_sextractor_auto_fluxes()
    cat.load_sextractor_kron_radii()

    for band_name in ["F444W"]: #["F200W", "F277W", "F356W", "F444W", "F410M"]: #cat.filterset.band_names:

        bd_selector = Brown_Dwarf_Selector(aper_diams[0], bd_fitter, secondary_SED_fit_label = eazy_fitter, select_band = band_name, morph_cut = False)
        bd_cat = bd_selector(cat, return_copy = True)

        filt = Filter.from_filt_name(band_name)
        psf_path = f"/nvme/scratch/work/westcottl/psf/PSF_Resample_03_{band_name}.fits"
        psf = PSF_Cutout.from_fits(
            fits_path=psf_path,
            filt=filt,
            unit="adu",
            pix_scale = 0.03 * u.arcsec,
            size = 0.96 * u.arcsec,
        )
        galfit_sersic_fitter = Galfit_Fitter(psf, "sersic", fixed_params = ["n"])
        galfit_sersic_fitter(bd_cat)

        bd_selector = Brown_Dwarf_Selector(aper_diams[0], bd_fitter, secondary_SED_fit_label = eazy_fitter, select_band = band_name, morph_cut = True)
        bd_cat_selected = bd_selector(bd_cat, return_copy = True)

        bd_cat_selected.plot_phot_diagnostics(
            aper_diams[0],
            [eazy_fitter, bd_fitter],
            eazy_fitter,
            imshow_kwargs = {},
            norm_kwargs = {},
            aper_kwargs = {},
            kron_kwargs = {},
            n_cutout_rows = 3,
            wav_unit = u.um,
            flux_unit = u.ABmag,
            log_fluxes = False,
            overwrite = True
        )
        IDs = bd_cat_selected.ID
        from galfind import ID_Selector
        id_selector = ID_Selector(IDs, f"brown_dwarf_{band_name}")
        id_selector(cat)

    #breakpoint()

    # galfit_psf_fitter = Galfit_Fitter(psf, "psf")
    # galfit_psf_fitter(bd_cat)

    # from galfind.Morphology import fwhm_nircam
    # from galfind import Re_Selector
    # galfit_re_selector = Re_Selector(galfit_sersic_fitter, "less", 1.2 * fwhm_nircam[band_name])
    # bd_cat = galfit_re_selector(bd_cat, return_copy = True)


if __name__ == "__main__":
    main()
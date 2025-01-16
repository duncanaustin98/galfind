
import astropy.units as u

from galfind.Data import morgan_version_to_dir
from galfind import PSF_Cutout, Filter, Galfit_Fitter, Catalogue, EPOCHS_Selector, EAZY

# Load in a JOF data object
survey = "JOF"
version = "v11"
instrument_names = ["NIRCam"] # "ACS_WFC",
aper_diams = [0.32] * u.arcsec
forced_phot_band = ["F277W", "F356W", "F444W"] #["F814W"]
min_flux_pc_err = 10.

def main():
    SED_fit_params_arr = [{"templates": "fsps_larson", "lowz_zmax": None}]
    cat = Catalogue.pipeline(
        survey,
        version,
        instrument_names = instrument_names, 
        version_to_dir_dict = morgan_version_to_dir,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        min_flux_pc_err = min_flux_pc_err,
        crops = EPOCHS_Selector(aper_diams[0], EAZY(SED_fit_params_arr[-1]), allow_lowz=True)
    )
    
    band_name = "F444W"
    filt = Filter.from_filt_name(band_name)
    psf_path = f'/nvme/scratch/work/westcottl/psf/PSF_Resample_03_{band_name}.fits'
    psf = PSF_Cutout.from_fits(
        fits_path=psf_path,
        filt=filt,
        unit="adu",
        pix_scale=0.03 * u.arcsec,
        size=0.96 * u.arcsec
    )
    Galfit_Fitter(psf, "sersic")(cat)
    breakpoint()


if __name__ == "__main__":
    main()
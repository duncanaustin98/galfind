import astropy.units as u
import numpy as np
import time

from galfind import useful_funcs_austind as funcs
from galfind import NIRCam, Photometry_rest
from galfind.SED import SED_obs
from galfind.Spectrum import Spectral_Catalogue

def load_DJA_spectra(ra_range = None, dec_range = None, UV_SNR_lim = 5.):
    spec_cat = Spectral_Catalogue.from_DJA(ra_range, dec_range, z_cat_range = [5.5, 6.5], version = "v2")
   #spec = 
    #, '1210_13197', '4557_-184']]
    cat_UV_SNR = [spec[0].calc_SNR_cont(1_500. * u.AA, 100. * u.AA) for spec in spec_cat]
    [spec[0].plot_spectrum() for spec, SNR in zip(spec_cat, cat_UV_SNR) if SNR > UV_SNR_lim]
    return [spec[0] for spec, UV_SNR in zip(spec_cat, cat_UV_SNR) if UV_SNR > UV_SNR_lim] #, '1210_13197', '4557_-184']]

def main():
    breakpoint()
    spectra = load_DJA_spectra()
    breakpoint()
    widebands = ["F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F410M", "F444W"]
    JOF_medium_bands = ["F162M", "F182M", "F210M", "F250M", "F300M", "F335M"]
    #NIRCam_8_widebands = NIRCam(excl_bands = [band for band in NIRCam().band_names if band not in widebands])
    NIRCam_JOF = NIRCam(excl_bands = [band for band in NIRCam().band_names if band not in widebands + JOF_medium_bands])
    JOF_depths = [29.80, 29.88, 30.12, 29.89, 30.26, 30.11, 30.07, 29.83, 30.37, 30.09, 30.18, 30.39, 29.78, 30.28]
    mock_phot_JOF = []
    for spec in spectra:
        sed_obs = SED_obs(spec.z, spec.wavs.value, spec.fluxes.value, spec.wavs.unit, spec.fluxes.unit)
        #Halpha_EWobs_template = sed_obs.calc_line_EW("Halpha")
        # mock_phot_widebands = sed_obs.create_mock_phot(NIRCam_8_widebands, [30. for i in range(len(NIRCam_8_widebands))])
        # phot_rest_widebands = Photometry_rest.from_phot(mock_phot_widebands, spec.z)
        # Halpha_widebands, Halpha_widebands_kwargs = phot_rest_widebands.calc_EW_rest_optical(["Halpha"], frame = "rest", single_iter = True)
        _mock_phot_JOF = sed_obs.create_mock_phot(NIRCam_JOF, JOF_depths)
        breakpoint()
        mock_phot_JOF.extend([_mock_phot_JOF])
        phot_rest_JOF = Photometry_rest.from_phot(_mock_phot_JOF, spec.z)
        Halpha_JOF, Halpha_JOF_kwargs = phot_rest_JOF.calc_EW_rest_optical(["Halpha"], frame = "obs", single_iter = True)

        #print(Halpha_widebands, Halpha_widebands_kwargs, Halpha_JOF, Halpha_JOF_kwargs)
        breakpoint()

if __name__ == "__main__":
    main()
    #load_DJA_spectra()


import astropy.units as u
from astropy.table import Table

from galfind import NIRCam, Photometry_rest
from galfind.SED import SED_obs
from galfind.Spectrum import Spectral_Catalogue


def load_DJA_spectra(ra_range=None, dec_range=None, UV_SNR_lim=5.0):
    spec_cat = Spectral_Catalogue.from_DJA(
        ra_range, dec_range, z_cat_range=[5.5, 6.5], version="v2"
    )
    cat_UV_SNR = [
        spec[0].calc_SNR_cont(1_500.0 * u.AA, 100.0 * u.AA)
        for spec in spec_cat
    ]
    # [spec[0].plot_spectrum() for spec, SNR in zip(spec_cat, cat_UV_SNR) if SNR > UV_SNR_lim]
    return [
        spec[0]
        for spec, UV_SNR in zip(spec_cat, cat_UV_SNR)
        if UV_SNR > UV_SNR_lim
    ]  # , '1210_13197', '4557_-184']]


def main():
    spectra = load_DJA_spectra()
    widebands = [
        "F090W",
        "F115W",
        "F150W",
        "F200W",
        "F277W",
        "F356W",
        "F410M",
        "F444W",
    ]
    JOF_medium_bands = ["F162M", "F182M", "F210M", "F250M", "F300M", "F335M"]
    # NIRCam_8_widebands = NIRCam(excl_bands = [band for band in NIRCam().band_names if band not in widebands])
    NIRCam_JOF = NIRCam(
        excl_bands=[
            band
            for band in NIRCam().band_names
            if band not in widebands + JOF_medium_bands
        ]
    )
    JOF_depths = [
        29.80,
        29.88,
        30.12,
        29.89,
        30.26,
        30.11,
        30.07,
        29.83,
        30.37,
        30.09,
        30.18,
        30.39,
        29.78,
        30.28,
    ]
    mock_phot_JOF = []
    Halpha_JOF = []
    beta_JOF = []
    # spectra_Ha = [getattr(spec, ) for spec in spectra]
    for spec in spectra:
        sed_obs = SED_obs(
            spec.z,
            spec.wavs.value,
            spec.fluxes.value,
            spec.wavs.unit,
            spec.fluxes.unit,
        )
        # Halpha_EWobs_template = sed_obs.calc_line_EW("Halpha")
        # mock_phot_widebands = sed_obs.create_mock_phot(NIRCam_8_widebands, [30. for i in range(len(NIRCam_8_widebands))])
        # phot_rest_widebands = Photometry_rest.from_phot(mock_phot_widebands, spec.z)
        # Halpha_widebands, Halpha_widebands_kwargs = phot_rest_widebands.calc_EW_rest_optical(["Halpha"], frame = "rest", single_iter = True)
        _mock_phot_JOF = sed_obs.create_mock_phot(NIRCam_JOF, JOF_depths)
        mock_phot_JOF.extend([_mock_phot_JOF])
        phot_rest_JOF = Photometry_rest.from_phot(_mock_phot_JOF, spec.z)
        _beta_JOF = phot_rest_JOF.calc_beta_phot(
            [1_250.0, 3_000.0] * u.AA, single_iter=True
        )[1][1]
        _Halpha_JOF, Halpha_JOF_kwargs = phot_rest_JOF.calc_EW_rest_optical(
            ["Halpha"], frame="rest", single_iter=True
        )
        # breakpoint()
        beta_JOF.extend([_beta_JOF])
        Halpha_JOF.extend([_Halpha_JOF.to(u.AA).value])
        # print(Halpha_widebands, Halpha_widebands_kwargs, Halpha_JOF, Halpha_JOF_kwargs)

    # phot = {band: mock_phot_JOF for i, band in NIRCam_JOF.band_names}
    tab = Table(
        {
            "ID": [spec.src_name for spec in spectra],
            "z_PRISM": [spec.z for spec in spectra],
            "beta_mock_phot": beta_JOF,
            "Halpha_NII_mock_phot": Halpha_JOF,
            "SNR_UV": [spec.SNR for spec in spectra],
        }
    )
    print(len(tab))
    # breakpoint()
    [spec.plot_spectrum() for spec in spectra]
    tab["Halpha_NII_PRISM_DJA"] = [
        spec.fit_data["eqwidth"]["line Ha+NII"][0] / (1.0 + spec.z)
        for spec in spectra
    ]
    tab.write("DJA_Ha+NII_beta_5.5<z<6.5_cat.fits", overwrite=True)


if __name__ == "__main__":
    main()
    # load_DJA_spectra()

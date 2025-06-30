from __future__ import annotations

from typing import (
    List,
    Optional,
    Union,
)

import astropy.units as u

from galfind import (
    EAZY,
    Bagpipes,
    Band_SNR_Selector,
    Bluewards_Lya_Non_Detect_Selector,
    Catalogue,
    Chi_Sq_Lim_Selector,
    Multiple_Filter,
    Multiple_SED_fit_Selector,
    Redshift_Bin_Selector,
    Redshift_Extractor,
    Redwards_Lya_Detect_Selector,
    Robust_zPDF_Selector,
    SED_code,
    Sextractor_Bands_Radius_Selector,
    Unmasked_Instrument_Selector,
)
from galfind.Data import morgan_version_to_dir

try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self  # python > 3.7 AND python < 3.11


# NEP - ACS_WFC+NIRCam - v14
# JADES GS DR3 (N/E/S/W) - ACS_WFC+NIRCam - v13
# JADES GN DR3 (Deep, Med, Par) - ACS_WFC+NIRCam - v13
# PRIMER - Cosmos/UDS - ACS_WFC+NIRCam - v12
# G165 - NIRCam - v11
# G191 - NIRCam - v11
# NGDEEP2 - NIRCam - v11
# CEERS - ACS_WFC+NIRCam - v9

from joblib import Parallel, delayed



class General_EPOCHS_Selector(Multiple_SED_fit_Selector):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, SED_code],
        allow_lowz: bool = True,
        unmasked_instruments: Union[str, List[str]] = "NIRCam",
        cat_filterset: Optional[Multiple_Filter] = None,
        simulated: bool = False,
        z_min: float = 0.0,
        z_max: float = 25.0,
    ):
        selectors = [
            Bluewards_Lya_Non_Detect_Selector(aper_diam, SED_fit_label, SNR_lim = 2.0),
            Redwards_Lya_Detect_Selector(aper_diam, SED_fit_label, SNR_lims = [8.0, 8.0], widebands_only = True),
            Redwards_Lya_Detect_Selector(aper_diam, SED_fit_label, SNR_lims = 2.0, widebands_only = True),
            Chi_Sq_Lim_Selector(aper_diam, SED_fit_label, chi_sq_lim = 3.0, reduced = True),
            Redshift_Bin_Selector(aper_diam, SED_fit_label, z_bin = [z_min, z_max]),
            Robust_zPDF_Selector(aper_diam, SED_fit_label, integral_lim = 0.6, dz_over_z = 0.1, min_dz=0.15),
        ]
        # add 2σ non-detection in first band if wanted
        if not allow_lowz:
            selectors.extend([Band_SNR_Selector( \
                aper_diam, band = 0, SNR_lim = 2.0, detect_or_non_detect = "non_detect")])

        if not simulated:
            # add unmasked instrument selections
            if isinstance(unmasked_instruments, str):
                unmasked_instruments = unmasked_instruments.split("+")
            selectors.extend([Unmasked_Instrument_Selector(instrument, \
                cat_filterset) for instrument in unmasked_instruments])

            # add hot pixel checks in LW widebands
            selectors.extend([
                Sextractor_Bands_Radius_Selector( \
                band_names = ["F277W", "F356W", "F444W"], \
                gtr_or_less = "gtr", lim = 45. * u.marcsec)
            ])
        lowz_name = "_lowz" if allow_lowz else ""
        unmasked_instr_name = "_" + "+".join(unmasked_instruments)
        selection_name = f"General_EPOCHS{lowz_name}{unmasked_instr_name}"
        super().__init__(aper_diam, SED_fit_label, selectors, selection_name = selection_name)


survey_properties = [
            ['NEP-1', 'v14', ['ACS_WFC', 'NIRCam']],
            ['NEP-2', 'v14', ['ACS_WFC', 'NIRCam']],
            ['NEP-3', 'v14', ['ACS_WFC', 'NIRCam']],
            ['NEP-4', 'v14', ['ACS_WFC', 'NIRCam']],
            ['JADES-DR3-GS-North', 'v13', ['ACS_WFC', 'NIRCam']],
            ['JADES-DR3-GS-South', 'v13', ['ACS_WFC', 'NIRCam']],
            ['JADES-DR3-GS-East', 'v13', ['ACS_WFC', 'NIRCam']],
            ['JADES-DR3-GS-West', 'v13', ['ACS_WFC', 'NIRCam']],
            ['JADES-DR3-GN-Deep', 'v13', ['ACS_WFC', 'NIRCam']],
            ['JADES-DR3-GN-Medium', 'v13', ['ACS_WFC', 'NIRCam']],
            ['JADES-DR3-GN-Parallel', 'v13', ['ACS_WFC', 'NIRCam']],
            ['PRIMER-COSMOS', 'v12', ['ACS_WFC', 'NIRCam']],
            ['PRIMER-UDS', 'v12', ['ACS_WFC', 'NIRCam']],
            ['G165', 'v11', ['NIRCam']],
            ['G191', 'v11', ['NIRCam']],
            ['NGDEEP2', 'v11', ['NIRCam']],
            ['CEERSP1', 'v9', ['ACS_WFC', 'NIRCam']],
            ['CEERSP2', 'v9', ['ACS_WFC', 'NIRCam']],
            ['CEERSP3', 'v9', ['ACS_WFC', 'NIRCam']],
            ['CEERSP4', 'v9', ['ACS_WFC', 'NIRCam']],
            ['CEERSP5', 'v9', ['ACS_WFC', 'NIRCam']],
            ['CEERSP6', 'v9', ['ACS_WFC', 'NIRCam']],
            ['CEERSP7', 'v9', ['ACS_WFC', 'NIRCam']],
            ['CEERSP8', 'v9', ['ACS_WFC', 'NIRCam']],
            ['CEERSP9', 'v9', ['ACS_WFC', 'NIRCam']],
            ['CEERSP10', 'v9', ['ACS_WFC', 'NIRCam']],
]

survey_properties = [
    ['JADES-DR3-GS-North', 'v13', ['ACS_WFC', 'NIRCam']],
    ['JADES-DR3-GS-South', 'v13', ['ACS_WFC', 'NIRCam']],
    ['JADES-DR3-GS-East', 'v13', ['ACS_WFC', 'NIRCam']],
    ['JADES-DR3-GS-West', 'v13', ['ACS_WFC', 'NIRCam']],
    ['JADES-DR3-GN-Deep', 'v13', ['ACS_WFC', 'NIRCam']],
    ['JADES-DR3-GN-Medium', 'v13', ['ACS_WFC', 'NIRCam']],
    ['JADES-DR3-GN-Parallel', 'v13', ['ACS_WFC', 'NIRCam']],
]

# Others - UNCOVER/CANUCS/Technicolor/GLIMPSE/MACS0416 etc
# Where are they?

def main(survey, version, instrument_names):
    print(f"Processing {survey} with version {version} and instruments {instrument_names}")

    aper_diams = [0.32] * u.arcsec
    forced_phot_band = ["F277W", "F356W", "F444W"]
    min_flux_pc_err = 10.

    sed_name = EAZY({"templates": "fsps_larson", "lowz_zmax": None}).label

    # Initialise the catalogue
    cat = Catalogue.pipeline(
        survey,
        version,
        instrument_names = instrument_names,
        version_to_dir_dict = morgan_version_to_dir,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band,
        min_flux_pc_err = min_flux_pc_err,
    )

    print(f'Loaded catalogue for {survey}, {version} with {len(cat)} objects')

    # Load/run EAZY results
    SED_fit_params_arr = [{"templates": "fsps_larson", "lowz_zmax": None}, {"templates": "fsps_larson", "lowz_zmax": 6}]
    for SED_fit_params in SED_fit_params_arr:
        EAZY_fitter = EAZY(SED_fit_params)
        for i in [0, 1]:
            EAZY_fitter(cat, aper_diams[i], load_PDFs = True, load_SEDs = False, update = True)

    # Apply selection
    selector = General_EPOCHS_Selector(aper_diams[0], sed_name, allow_lowz = True, z_min=3, z_max=15)
    cropped_cat = selector(cat, return_copy = True)

    print(f'Selected {len(cropped_cat)} objects from the catalogue.')
    '''
    # Setup Bagpipes config
    pipes_fit_params_arr = [
        {
            "dust": "CF00",
            "dust_prior": "uniform",
            "metallicity_prior": "uniform",
            "sps_model": "BC03",
            "fix_z": sed_name,
            "sfh": "continuity",
            "z_calculator":Redshift_Extractor(aper_diams[0], EAZY_fitter)
            #"z_range": (0.0, 25.0),
        }
    ]

    for pipes_fit_params in pipes_fit_params_arr:
        pipes_fitter = Bagpipes(pipes_fit_params)
        pipes_fitter(cropped_cat, aper_diams[0], save_PDFs = False, load_SEDs = False, update = True)

    '''

if __name__ == "__main__":
    Parallel(n_jobs=6)(delayed(main)(survey, version, instrument_names) for survey, version, instrument_names in survey_properties)
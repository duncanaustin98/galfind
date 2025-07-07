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
    Catalogue_Creator,
    Multiple_SED_fit_Selector,
    Redshift_Bin_Selector,
    Redshift_Extractor,
    Redwards_Lya_Detect_Selector,
    Robust_zPDF_Selector,
    SED_code,
    Multiple_Filter, 
    Filter,
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

from galfind import (
    Multiple_SED_fit_Selector,
    Bluewards_LyLim_Non_Detect_Selector,
    Bluewards_Lya_Non_Detect_Selector,
    Redwards_Lya_Detect_Selector,
    Chi_Sq_Lim_Selector, 
    Chi_Sq_Diff_Selector,
    Robust_zPDF_Selector,
    Sextractor_Bands_Radius_Selector,
    ID_Selector,
    Property_Calculator,
    Combined_Catalogue,
    Min_Instrument_Unmasked_Band_Selector,
    Unmasked_Band_Selector,
    Multiple_Mask_Selector,
)

class Austin25_unmasked_criteria(Multiple_Mask_Selector):

    def __init__(self: Self):
        selectors = [
            Min_Instrument_Unmasked_Band_Selector(min_bands = 2, instrument = "ACS_WFC"),
            Min_Instrument_Unmasked_Band_Selector(min_bands = 6, instrument = "NIRCam"),
        ]
        selectors.extend([Unmasked_Band_Selector(band) for band in ["F090W", "F277W", "F356W", "F410M", "F444W"]])
        selection_name = "Austin+25_unmasked_criteria"
        super().__init__(selectors, selection_name = selection_name)

class Austin25_sample(Multiple_SED_fit_Selector):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, SED_code],
        simulated: bool = False,
    ):
        selectors = [
            Bluewards_LyLim_Non_Detect_Selector(aper_diam, SED_fit_label, SNR_lim = 2.0, ignore_bands = ["F070W", "F850LP"]),
            Bluewards_Lya_Non_Detect_Selector(aper_diam, SED_fit_label, SNR_lim = 3.0, ignore_bands = ["F070W", "F850LP"]),
            Redwards_Lya_Detect_Selector(aper_diam, SED_fit_label, SNR_lims = [8.0, 8.0], widebands_only = True, ignore_bands = ["F070W", "F850LP"]),
            Redwards_Lya_Detect_Selector(aper_diam, SED_fit_label, SNR_lims = 3.0, widebands_only = True, ignore_bands = ["F070W", "F850LP"]),
            Chi_Sq_Lim_Selector(aper_diam, SED_fit_label, chi_sq_lim = 3.0, reduced = True),
            Chi_Sq_Diff_Selector(aper_diam, SED_fit_label, chi_sq_diff = 4.0, dz = 0.5),
            Robust_zPDF_Selector(aper_diam, SED_fit_label, integral_lim = 0.6, dz_over_z = 0.1),
        ]
        assert isinstance(simulated, bool), galfind_logger.critical(f"{type(simulated)=}!=bool")
        if not simulated:
            selectors.extend([Sextractor_Bands_Radius_Selector(band_names = ["F277W", "F356W", "F444W"], gtr_or_less = "gtr", lim = 45. * u.marcsec)])
            # add unmasked instrument selections
            #selectors.extend([Unmasked_Instrument_Selector(instr_name) for instr_name in ["ACS_WFC", "NIRCam"]])
            selectors.extend([Austin25_unmasked_criteria()])
        selection_name = "Austin+25"
        super().__init__(aper_diam, SED_fit_label, selectors, selection_name = selection_name)

    def _assertions(self: Self) -> bool:
        return True


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
    nircam_filt_names = ["F090W", "F115W", "F150W", "F200W", "F277W", "F335M", "F356W", "F410M", "F444W"]
    # Create an instrument object for NIRCam incorporating all the NIRCam bands
    filters = Multiple_Filter([Filter.from_filt_name(filt_name) for filt_name in nircam_filt_names])

    SED_fitter_arr = [
        #EAZY({"templates": "fsps_larson", "lowz_zmax": 6.0}),
        EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
    ]

    print(f"Processing {survey} with version {version} and instruments {instrument_names}")

    aper_diams = [0.32] * u.arcsec
    forced_phot_band = ["F277W", "F356W", "F444W"]
    min_flux_pc_err = 10.
    
    cat_path = f"/mnt/GALFIND_WORK/Catalogues/{version}/{'+'.join(instrument_names)}/{survey}/({aper_diams[0].value})as/{survey}_MASTER_Sel-{'+'.join(forced_phot_band)}_{version}.fits"
   
    cat_creator = Catalogue_Creator(
        survey, 
        version,
        cat_path, 
        filters, 
        aper_diams, 
        crops = Austin25_sample(aper_diams[0], SED_fitter_arr[-1].label)
    )

    # Initialise the catalogue
    cat = cat_creator()

    print(f'Loaded catalogue for {survey}, {version} with {len(cat)} objects')

    pipes_fit_params_arr = [
        {
            "dust": "CF00",
            "dust_prior": "uniform",
            "metallicity_prior": "uniform",
            "sps_model": "BC03",
            "fix_z": False,
            "sfh": "lognorm",
            "z_range": (0.0, 25.0),
        }
    ]

    for pipes_fit_params in pipes_fit_params_arr:
        pipes_fitter = Bagpipes(pipes_fit_params)
        pipes_fitter(cat, aper_diams[0], save_PDFs = False, load_SEDs = False, update = True)


if __name__ == "__main__":
    run = ['JADES-DR3-GS-East', 'v13', ['ACS_WFC', 'NIRCam']]
    main(*run)

    #Parallel(n_jobs=6)(delayed(main)(survey, version, instrument_names) for survey, version, instrument_names in survey_properties)
from galfind import (
    Instrument,
    Photometry,
    Photometry_rest,
    config,
    useful_funcs_austind,
    Euclid, Spitzer, CFHT, Subaru,
    NISP,
    MegaCam,
    HSC,
    VIS,
    IRAC,
    Multiple_Filter,
    SED_obs,
    Catalogue_Creator,
    EAZY,
    EPOCHS_Selector,
    
)
from galfind.selection import Completeness
from galfind.Catalogue import phot_property_from_fits, scattered_depth_labels, scattered_phot_labels_inst
from galfind import useful_funcs_austind as funcs
import itertools
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.table import Table, Row

filterset_mc = Multiple_Filter.from_instrument('MegaCam', excl_bands=['CFHT/MegaCam.g',
                                                                      'CFHT/MegaCam.r', 
                                                                      'CFHT/MegaCam.i', 
                                                                      'CFHT/MegaCam.z', 
                                                                      'CFHT/MegaCam.Y', 
                                                                      'CFHT/MegaCam.J', 
                                                                      'CFHT/MegaCam.gri',
                                                                      'CFHT/MegaCam.H'])    

filterset_hsc = Multiple_Filter.from_instrument('HSC')

filterset = Multiple_Filter.from_instruments(['NISP', 'VIS', 'IRAC'], excl_bands=['Spitzer/IRAC.I3', 'Spitzer/IRAC.I4'])
filterset = filterset + filterset_mc + filterset_hsc       

folder = '/raid/scratch/data/JAGUAR/Full_Spectra'
out_folder = '/nvme/scratch/work/tharvey/catalogs/' 
skip = True 
realization = 1 # realisation
redshift_bins = [(0.2, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 4), (4, 5), (5, 15)]
filenames = [f'{folder}/JADES_SF_mock_r{realization}_v1.2_spec_5A_30um_z_{str(z1).replace(".", "p")}_{str(z2).replace(".", "p")}.fits' for z1, z2 in redshift_bins]

out_filename = f'{out_folder}/JADES_SF_mock_r{realization}_v1.2_{filterset.instrument_name}_phot.fits'
out_colnames = [f'{filter.instrument_name}_{filter.band_name}_fnu' for filter in filterset.filters]
out_colnames=['ID', 'redshift'] + out_colnames


if not skip:
    rows = []
    for file in filenames:
        hdu = fits.open(file)
        wav = hdu[2].data # Angstrom
        table = hdu[3].data
        ids = table['id']
        redshifts = table['redshift']
        
        nrows = hdu[1].shape[0]
    
        for i in tqdm(range(nrows), desc=f'Processing spectra from {file}'):
            flux = hdu[1].section[i, :] / (1+redshifts[i]) # convert from rest-frame to observed-frame
            sed = SED_obs(redshifts[i], wav, flux, wav_units=u.AA, mag_units=u.erg/u.s/u.cm**2/u.AA)
            mock_phot = sed.create_mock_phot(filterset=filterset, min_flux_pc_err=0)
            flux = mock_phot.flux.to(u.nJy)
            row = (ids[i], redshifts[i]) + tuple(flux.value)
            rows.append(row)
            
        output_table = Table(rows = rows, names=out_colnames)
        output_table.write(out_filename, format='fits', overwrite=True)


out_filename = '/nvme/scratch/work/tharvey/catalogs/JADES_SF_mock_r1_v1.2_MegaCam+HSC+VIS+NISP+IRAC_phot_fnu_200x.fits'

survey = 'Euclid_Deep'
version = 'v_sim'
aper_diams = [0.32] * u.arcsec

depths = [26] + [28] * len(filterset.band_names[1:])
depths = np.array(depths) * u.ABmag

def jaguar_phot_labels(
    filterset: Multiple_Filter, 
    aper_diams: u.Quantity, 
    **kwargs
):
    assert "min_flux_pc_err" in kwargs.keys(), "min_flux_pc_err must be provided"
    phot_labels = { aper_diam * aper_diams.unit: [
                f'{filt.instrument_name}_{filt.band_name}_fnu'
                for filt in filterset
            ] for aper_diam in aper_diams.value
    }
    err_labels = {aper_diam * aper_diams.unit: [] for aper_diam in aper_diams.value}
    return phot_labels, err_labels

def load_jaguar_phot(
    cat: Table,
    phot_labels, #Dict[u.Quantity, List[str]],
    err_labels, #Dict[u.Quantity, List[str]],
    **kwargs
):
    assert phot_labels.keys() == err_labels.keys(), f"{phot_labels.keys()=} != {err_labels.keys()=}!"
    assert "ZP" in kwargs.keys(), "ZP not in kwargs!"

    print(phot_labels)
    print(err_labels)
    phot = {aper_diam: funcs.flux_image_to_Jy(_phot, kwargs["ZP"]) for aper_diam, _phot \
        in phot_property_from_fits(cat, phot_labels, **kwargs).items()}
    if "incl_errs" in kwargs.keys():
        if not kwargs["incl_errs"]:
            phot_err = {aper_diam: np.array(list(itertools.repeat(None, len(cat)))) for aper_diam in phot_labels.keys()}
            return phot, phot_err
    phot_err = {aper_diam: funcs.flux_image_to_Jy(_phot_err, kwargs["ZP"]) for aper_diam, _phot_err \
        in phot_property_from_fits(cat, err_labels, **kwargs).items()}
    return phot, phot_err

def load_depth_func(catalog, depth_labels, depths, **kwargs):
    print(depth_labels)
    output = {}
    for key, label in depth_labels.items():
        output[key] = np.repeat(depths, len(catalog)).reshape(len(depths), len(catalog)).T
        
    print(output)
    return output

    

def get_depth_labels(filterset, aper_diams):
    depth_labels = { aper_diam * aper_diams.unit: [
                f'{filt.instrument_name}_{filt.band_name}_fnu'
                for filt in filterset
            ] for aper_diam in aper_diams.value
    }

    return depth_labels

#jaguar_cat_path = f"/nvme/scratch/work/tharvey/catalogs/JADES_SF_mock_r{str(int(realization))}_v1.2_MegaCam+HSC+VIS+NISP+IRAC_phot.fits"
jaguar_cat_path = '/nvme/scratch/work/tharvey/catalogs/JADES_SF_mock_r1_v1.2_MegaCam+HSC+VIS+NISP+IRAC_phot_fnu_200x.fits'
jaguar_cat_creator = Catalogue_Creator(
    survey = f"JAGUAR-{survey}-r{str(int(realization))}_200_fnu",
    version = version,
    cat_path = jaguar_cat_path,
    filterset = filterset,
    aper_diams = aper_diams, # not relevant in this case, but still required
    ID_label = "ID",
    skycoords_labels = {"RA": "RA", "DEC": "DEC"},
    get_phot_labels = jaguar_phot_labels,
    load_phot_func = load_jaguar_phot,
    load_phot_kwargs = {
        "ZP": u.nJy.to(u.ABmag),
        "min_flux_pc_err": 5.0,
        "incl_errs": False,
    },
    load_mask_func = None,
    load_depth_func = load_depth_func,
    get_depth_labels = get_depth_labels,
    load_depth_kwargs = {
        'depths':depths,
    },

    apply_gal_instr_mask = False,
    simulated = True,
)

jaguar_cat = jaguar_cat_creator()

   # construct array of required EAZY SED fitter objects
SED_fitter_arr = [
    #EAZY({"templates": "fsps_larson", "lowz_zmax": 4.0}),
    #EAZY({"templates": "fsps_larson", "lowz_zmax": 6.0}),
    EAZY({"templates": "fsps_larson", "lowz_zmax": 15}),
]

selector = EPOCHS_Selector(
    aper_diam = aper_diams[0],
    SED_fit_label = SED_fitter_arr[-1],
    simulated=True,
)

# make 2D completeness grid from the Jaguar catalogue
completeness = Completeness.from_sim_cat(
    jaguar_cat,
    SED_fitter_arr = SED_fitter_arr,
    sampler = None,
    aper_diam = aper_diams[0],
    mode = "n_nearest",
    depth_region = "all",
    x_calculator = None,
    y_calculator = None,
    x_arr = None,
    y_arr = None,
    sim_filterset = filterset,
    data_filterset = filterset,
    aper_diams = aper_diams,
    depth_labels_func = scattered_depth_labels,
    phot_labels_func = scattered_phot_labels_inst,
    )
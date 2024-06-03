# Catalogue_Base.py

import numpy as np
from astropy.table import Table, join
from copy import deepcopy
import astropy.units as u
from tqdm import tqdm
from astropy.coordinates import SkyCoord
from astropy.io import fits
import time
from . import useful_funcs_austind as useful_funcs
from .Data import Data
from .Galaxy import Galaxy
from . import useful_funcs_austind as funcs
from .Catalogue_Creator import GALFIND_Catalogue_Creator
from . import config, galfind_logger, SED_code
from .EAZY import EAZY
from . import Multiple_Catalogue, Multiple_Data

class Catalogue_Base:
    # later on, the gal_arr should be calculated from the Instrument and sex_cat path, with SED codes already given
    def __init__(self, gals, cat_path, survey, cat_creator, instrument, \
            SED_fit_params_arr = {}, version = '', crops = [], SED_rest_properties = {}):
        self.survey = survey
        self.cat_path = cat_path
        self.cat_creator = cat_creator
        self.instrument = instrument
        self.SED_fit_params_arr = SED_fit_params_arr
        self.gals = np.array(gals)
        if version == "":
            galfind_logger.critical("Version must be specified")
        self.version = version
        
        # keep a record of the crops that have been made to the catalogue
        self.selection_cols = [key.replace("SELECTED_", "") for key in self.open_cat().meta.keys() if "SELECTED_" in key]

        if crops == None:
            crops = []
        self.crops = crops
        self.SED_rest_properties = SED_rest_properties
        
        # concat is commutative for catalogues
        self.__radd__ = self.__add__
        # cross-match is commutative for catalogues - not if they are of different classes
        # self.__rmul__ = self.__mul__

    # %% Overloaded operators

    def __str__(self, print_cls_name = True, print_data = True, print_sel_criteria = True, display_selections = ["EPOCHS", "BROWN_DWARF"]):
        line_sep = "*" * 40 + "\n"
        band_sep = "-" * 10 + "\n"
        output_str = ""
        if print_cls_name:
            output_str += line_sep
            output_str += f"CATALOGUE: {self.survey} {self.version}\n" # could also show median RA/DEC from the array of galaxy sky coords
            output_str += band_sep
        if print_data and "data" in self.__dict__.keys():
            output_str += str(self.data)
        output_str += f"FITS CAT PATH = {self.cat_path}\n"
        # access table header to display what has been run for this catalogue
        cat = self.open_cat()
        output_str += f"N_GALS_TOTAL = {len(cat)}\n"
        # display what other things have previously been calculated for this catalogue, including templates and zmax_lowz
        output_str += "CAT STATUS = SEXTRACTOR, "
        for i, (key, value) in enumerate(cat.meta.items()):
            if key in ["DEPTHS", "MASKED"] + [f"RUN_{subclass.__name__}" for subclass in SED_code.__subclasses__()]:
                output_str += f"{key.split('_')[-1]}, "
        for sel_criteria in display_selections:
            if sel_criteria in cat.colnames:
                output_str += f"{sel_criteria} SELECTION, "
        output_str += "\n"
        # display total number of galaxies that satisfy the selection criteria previously performed
        if print_sel_criteria:
            for sel_criteria in display_selections:
                if sel_criteria in cat.colnames:
                    output_str += f"N_GALS_{sel_criteria} = {len(cat[cat[sel_criteria]])}\n"
        output_str += band_sep
        # display crops that have been performed on this specific object
        if self.crops != []:
            output_str += f"N_GALS_OBJECT = {len(self)}\n"
            output_str += f"CROPS = {' + '.join(self.crops)}\n"
        #breakpoint()
        if hasattr(self, "SED_rest_properties"):
            if len(self.SED_rest_properties) >= 1:
                output_str += band_sep
                for key, properties in self.SED_rest_properties.items():
                    output_str += f"Rest frame SED properties:\n"
                    output_str += f"{key}: {str(properties)}\n"
                    output_str += band_sep
        if print_cls_name:
            output_str += line_sep
        return output_str
    

    def __len__(self):
        return len(self.gals)
    
    def __iter__(self):
        self.iter = 0
        return self
    
    def __next__(self):
        if self.iter > len(self) - 1:
            raise StopIteration
        else:
            gal = self[self.iter]
            self.iter += 1
            return gal
    
    def __getitem__(self, index):
        if type(self.gals) != np.ndarray:
            self.gals = np.array(self.gals)
        return self.gals[index]
    
    def __getattr__(self, name, SED_fit_params = {"code": EAZY(), "templates": "fsps_larson", "lowz_zmax": None}, phot_type = "obs", property_type = "vals"): # only acts on attributes that don't already exist
        if name in self[0].__dict__:
            return np.array([getattr(gal, name) for gal in self])
        elif name.upper() == "RA":
            return np.array([getattr(gal, "sky_coord").ra.degree for gal in self]) * u.deg
        elif name.upper() == "DEC":
            return np.array([getattr(gal, "sky_coord").dec.degree for gal in self]) * u.deg
        elif name in self[0].phot.instrument.__dict__:
            return np.array([getattr(gal.phot.instrument, name) for gal in self])
        elif phot_type == "obs" and name in self[0].phot.__dict__:
            return np.array([getattr(gal.phot, name) for gal in self])
        elif name in self[0].mask_flags.keys():
            return np.array([getattr(gal.mask_flags, name) for gal in self])
        elif name == "full_mask":
            return np.array([getattr(gal, "phot").mask for gal in self])
        elif name in self[0].selection_flags.keys():
            return np.array([getattr(gal, "selection_flags")[name] for gal in self])
        elif name in self[0].phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].__dict__:
            return np.array([getattr(gal.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)], name) for gal in self])
        elif phot_type == "rest" and name in self[0].phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].phot_rest.__dict__:
            return np.array([getattr(gal.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].phot_rest, name) for gal in self])
        elif phot_type == "rest" and property_type == "vals" and name in self[0].phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].phot_rest.properties.keys():
            properties = [getattr(gal.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].phot_rest, "properties")[name] \
                if name in getattr(gal.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].phot_rest, "properties").keys() else np.nan for gal in self]
            return np.array([property.value if type(property) in [u.Quantity, u.Magnitude] else property for property in properties])
        elif phot_type == "rest" and property_type == "errs" and name in self[0].phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].phot_rest.property_errs.keys():
            property_errs_arr = [getattr(gal.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].phot_rest, "property_errs")[name] \
                if name in getattr(gal.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].phot_rest, "property_errs").keys() else [np.nan, np.nan] for gal in self]
            return np.array([property_errs.value if type(property_errs) in [u.Quantity, u.Magnitude] else property_errs for property_errs in property_errs_arr])
        elif phot_type == "rest" and property_type == "PDFs" and name in self[0].phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].phot_rest.property_errs.keys():
            return [getattr(gal.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].phot_rest, "property_PDFs")[name] \
                if name in getattr(gal.phot.SED_results[SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)].phot_rest, "property_PDFs").keys() else None for gal in self]
        else:
            galfind_logger.critical(f"Galaxies do not have attribute = {name}!")
    
    def __setattr__(self, name, value, obj = "cat"):
        if obj == "cat":
            super().__setattr__(name, value)
        elif obj == "gal":
            # set attributes of individual galaxies within the catalogue
            for i, gal in enumerate(self):
                if type(value) in [list, np.array]:
                    setattr(gal, name, value[i])
                else:
                    setattr(gal, name, value)
    
    # not needed!
    def __setitem__(self, index, gal):
        self.gals[index] = gal
    
    def __add__(self, cat, out_survey = None):
        if not cat.__class__.__name__ == "Spectral_Catalogue":
            # concat catalogues
            if out_survey == None:
                out_survey = "+".join([self.survey, cat.survey])
            return Multiple_Catalogue([self, cat], survey = out_survey)
    
    def __mul__(self, other, out_survey=None, max_sep=1.0 * u.arcsec, match_type='compare_within_radius'):
        '''
        'Multiply' two catalogues by performing a cross-match and filtering the best matching galaxies.
        Ensures no duplicate galaxies are present in the output catalogue.

        Parameters:
        - other: Catalogue_Base
            The other catalogue to multiply with.
        - out_survey: str, optional
            The name of the output survey. Default is None.
        - max_sep: Quantity, optional
            The maximum separation allowed for matching galaxies. Default is 1.0 arcsec.
        - match_type: str, optional
            The type of matching to perform. Options are 'nearest' and 'compare_within_radius'.
            Default is 'compare_within_radius'.

        Returns:
        - Multiple_Catalogue
            The resulting MultipleCatalogue after performing the multiplication, with duplicates removed. 

        '''
        # cross-match catalogues
        # update .fits tables with cross-matched version
        # open tables
        self_copy = deepcopy(self)
        other_copy = deepcopy(other)

        # Convert from list of SkyCoord to SkyCoord(array)
        sky_coords_cat = SkyCoord(self_copy.RA, self_copy.DEC, unit=(u.deg, u.deg), frame='icrs')
        other_sky_coords = SkyCoord(other_copy.RA, self_copy.DEC, unit=(u.deg, u.deg), frame='icrs')
        if match_type == 'nearest':
            # This just takes the nearest galaxy as the best match  
            idx, d2d, d3d = sky_coords_cat.match_to_catalog_sky(other_sky_coords)
            # Also check mask - don't keep masked galaxies where there is an unmasked match
            sep_constraint = d2d < max_sep
            # Get indexes of matches
            cat_matches = np.arange(len(sky_coords_cat))[sep_constraint]
            other_cat_matches = idx[sep_constraint]
        elif match_type == 'compare_within_radius':
            # This finds all matches within a certain radius and compares the photometry of the galaxies
            cat_matches = []
            other_cat_matches = []

            for pos, coord in tqdm(enumerate(self_copy.sky_coord), desc="Cross-matching galaxies"):
                # Need to save index of match in other_sky_coords

                d2d = coord.separation(other_sky_coords)

                indexes = np.argwhere(d2d < max_sep)
                indexes = np.ndarray.flatten(indexes)

                if len(indexes) == 0:
                    print('continuing')
                    continue
                elif len(indexes) == 1:
                    cat_matches.append(pos)
                    other_cat_matches.append(indexes[0])
                else:

                    # Compare fluxes and choose the closest match
                    # Get bands that are in both galaxies 

                    coord_gal = self_copy.gals[pos]

                    other_gals = np.ndarray.flatten(other_copy.gals[indexes])
                    # Save indexes of other_gals in other_sky_coords
                    other_gals_indexes = np.arange(len(other_sky_coords))[indexes]

                    bands_gal1 = [band for band in coord_gal.phot.instrument.band_names if band not in coord_gal.mask_flags.keys()]

                    chi_squareds = []
                    for other_gal in other_gals:
                        bands_gal2 = [band for band in other_gal.phot.instrument.band_names if band not in other_gal.mask_flags.keys()]
                        matched_bands = list(set(bands_gal1).union(set(bands_gal2)))

                        if len(matched_bands) == 0:
                            continue
                        # Compare fluxes in matched bands
                        indexes_bands = np.argwhere([band in matched_bands for band in coord_gal.phot.instrument.band_names])
                        indexes_other_bands = np.argwhere([band in matched_bands for band in other_gal.phot.instrument.band_names])
                        coord_gal_fluxes = coord_gal.phot.flux_Jy[indexes_bands]
                        coord_gal_flux_errs = coord_gal.phot.flux_Jy_errs[indexes_bands]
                        other_gal_fluxes = other_gal.phot.flux_Jy[indexes_other_bands]

                        # Chi-squared comparison
                        chi_squared = np.sum((coord_gal_fluxes - other_gal_fluxes)**2 / (coord_gal_flux_errs**2))
                        chi_squareds.append(chi_squared)
                    if len(chi_squareds) == 0:
                        continue
                    best_match_index = int(np.squeeze(np.argmin(chi_squareds)))
                    # pop empty dimensions
                    cat_matches.append(pos)
                    other_cat_matches.append(other_gals_indexes[best_match_index])

        assert len(cat_matches) == len(other_cat_matches), f'{len(cat_matches)} != {len(other_cat_matches)}' # check that the matches are 1-to-1
        print('Getting galaxies')

        cat_matches = np.array(cat_matches)
        gal_matched_cat = self_copy[cat_matches]
        # Use indexes instead
        other_cat_matches = np.array(other_cat_matches)
        gal_matched_other = other_copy[other_cat_matches]
        print('Obtained matched galaxies')
        assert len(gal_matched_cat) == len(gal_matched_other) # check that the matches are 1-to-1

        if self_copy.__class__.__name__ == "Catalogue" and other_copy.__class__.__name__ == "Spectral_Catalogue":
            # update catalogue and galaxies
            self_copy.gals = [deepcopy(gal).add_spectra(spectra) for gal, spectra in \
                tqdm(zip(gal_matched_cat, gal_matched_other), total = len(gal_matched_cat), \
                desc = "Appending spectra to catalogue!")]
            return self_copy
        else:
            for gal1, gal2 in tqdm(zip(gal_matched_cat, gal_matched_other), desc='Filtering best galaxy for matches'):
                # Compare the two galaxies and choose the better one
                bands_gal1 = [band for band in gal1.phot.instrument.band_names if band not in gal1.mask_flags.keys()]
                bands_gal2 = [band for band in gal2.phot.instrument.band_names if band not in gal2.mask_flags.keys()]
                band_names_union = list(set(bands_gal1).union(set(bands_gal2)))

                if len(bands_gal2) > len(bands_gal1):
                    self_copy.remove_gal(id = gal1.ID)
                elif len(bands_gal2) < len(bands_gal1):
                    other_copy.remove_gal(id = gal2.ID)
                else:
                    # If same bands, choose galaxy with deeper depth
                    # Get matching bands between the two galaxies - only use if not masked
                    # logical comparison of depth in each band, keeping the galaxy with the deeper depth in more bands
                    # gal1.phot.depths is just an array. Need to slice by position
                    indexes_gal1 = np.argwhere([band in band_names_union for band in gal1.phot.instrument.band_names])
                    depths_gal1 = gal1.phot.depths[indexes_gal1]
                    indexes_gal2 = np.argwhere([band in band_names_union for band in gal2.phot.instrument.band_names])
                    depths_gal2 = gal2.phot.depths[indexes_gal2]
                    # Compare depths
                    if np.sum(depths_gal1 > depths_gal2) > np.sum(depths_gal1 < depths_gal2):
                        self_copy.remove_gal(id=gal1.ID)
                    elif np.sum(depths_gal1 > depths_gal2) < np.sum(depths_gal1 < depths_gal2):
                        other_copy.remove_gal(id=gal2.ID)
                    else:
                        # Choose first galaxy
                        self_copy.remove_gal(id=gal1.ID)
        return self_copy + other_copy


    # Need to save the cross-match distances
        
    def combine_and_remove_duplicates(self, other, out_survey = None, max_sep = 1.0 * u.arcsec, match_type='nearest'):
        'Alias for self * other'
        return self.__mul__(other, out_survey = None, max_sep = 1.0 * u.arcsec, match_type='nearest')

    def __sub__(self):
        pass
    
    def __repr__(self):
        return str(self.__dict__)
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, deepcopy(value, memo))
        return result
        
    @property
    def cat_dir(self):
        return funcs.split_dir_name(self.cat_path, "dir")
    
    @property
    def cat_name(self):
        return funcs.split_dir_name(self.cat_path, "name")
    
    @property
    def ra_range(self):
        try:
            return self._ra_range
        except:
            self._ra_range = [np.min(self.RA), np.max(self.RA)]
            return self._ra_range
    
    @property
    def dec_range(self):
        try:
            return self._dec_range
        except:
            self._dec_range = [np.min(self.DEC), np.max(self.DEC)]
            return self._dec_range
    
    def remove_gal(self, index = None, id = None):
        if index is not None:
            self.gals = np.delete(self.gals, index)
        elif id is not None:
            self.gals = np.delete(self.gals, np.where(self.ID == id))
        else:
            galfind_logger.critical("No index or ID provided to remove_gal!")

    def crop(self, crop_limits, crop_property): # upper and lower limits on galaxy properties (e.g. ID, redshift, mass, SFR, SkyCoord)
        cat_copy = deepcopy(self)
        if type(crop_limits) in [int, float, bool]:
            cat_copy.gals = cat_copy[getattr(cat_copy, crop_property) == crop_limits]
            if crop_limits == True:
                cat_copy.crops.append(crop_property)
            else:
                cat_copy.crops.append(f"{crop_property}={crop_limits}")
        elif type(crop_limits) in [list, np.array]:
            cat_copy.gals = cat_copy[((getattr(cat_copy, crop_property) >= crop_limits[0]) & (getattr(cat_copy, crop_property) <= crop_limits[1]))]
            cat_copy.crops.append(f"{crop_limits[0]}<{crop_property}<{crop_limits[1]}")
        else:
            galfind_logger.critical(f"crop_limits={crop_limits} with type = {type(crop_limits)} not in [int, float, bool, list, np.array]")
        return cat_copy
    
    def open_cat(self, cropped = False, hdu = None):
        if type(hdu) == type(None):
            fits_cat = Table.read(self.cat_path, character_as_bytes = False, memmap = True)
        elif self.check_hdu_exists(hdu):
            fits_cat = Table.read(self.cat_path, character_as_bytes = False, memmap = True, hdu = hdu)
        else:
            galfind_logger.warning(f"{hdu.upper()=} does not exist in {self.cat_path=}!")
            return None
        if cropped:
            ID_tab = Table({"IDs_temp": self.ID}, dtype = [int])
            combined_tab = join(fits_cat, ID_tab, keys_left = self.cat_creator.ID_label, keys_right = "IDs_temp")
            combined_tab.remove_column("IDs_temp")
            combined_tab.meta = fits_cat.meta
            return combined_tab
        else:
            return fits_cat
    
    def check_hdu_exists(self, hdu):
        # check whether the hdu extension exists
        hdul = fits.open(self.cat_path)
        return any(hdu_.name == hdu.upper() for hdu_ in hdul)
    
    def write_cat(self, tab_arr, tab_names):
        hdu_list = fits.HDUList()
        [hdu_list.append(fits.BinTableHDU(data = tab.as_array(), header = \
            fits.Header(tab.meta), name = name)) for (tab, name) in zip(tab_arr, tab_names)]
        hdu_list.writeto(self.cat_path, overwrite = True)
        galfind_logger.info(f"Writing table to {self.cat_path}")

    def del_hdu(self, hdu):
        galfind_logger.info(f"Deleting {hdu.upper()=} from {self.cat_path=}!")
        assert self.check_hdu_exists(hdu), \
            galfind_logger.critical(f"Cannot delete {hdu=} as it does not exist in {self.cat_path=}")
        tab_arr = [self.open_cat(cropped = False, hdu = hdu_.name) for hdu_ \
            in fits.open(self.cat_path) if hdu_.name != hdu.upper() and hdu_.name != "PRIMARY"]
        tab_names = [hdu_.name for hdu_ in fits.open(self.cat_path) if hdu_.name != hdu.upper() and hdu_.name != "PRIMARY"]
        self.write_cat(tab_arr, tab_names)

    def del_cols_hdrs_from_fits(self, col_names = [], hdr_names = [], hdu = None):
        # open up all fits extensions
        tab_arr = []
        tab_names = []
        for i, hdu_ in enumerate(fits.open(self.cat_path)):
            if hdu_.name != "PRIMARY":
                # open fits extension
                tab = self.open_cat(cropped = False, hdu = hdu_.name)
                # append required fits extension
                if hdu_.name == hdu.upper():
                    # ensure every column/header name is in catalogue
                    assert(all(name in tab.colnames for name in col_names))
                    assert(all(name in tab.meta.keys() for name in hdr_names))
                    [tab.remove_column(name) for name in col_names]
                    tab.meta = {key: value for key, value in dict(tab.meta).items() if key not in hdr_names}
                tab_arr.append(tab)
                tab_names.append(hdu_.name)
        self.write_cat(tab_arr, tab_names)

# Multiple_Catalogue.py
import numpy as np
from astropy.coordinates import SkyCoord

class Multiple_Catalogue:

    def __init__(self, cat_arr, survey):
        self.cat_arr = cat_arr
        self.survey = survey
        # concat is commutative for catalogues
        self.__radd__ = self.__add__
        # cross-match is commutative for catalogues
        #self.__rmul__ = self.__mul__

    @classmethod
    def from_pipeline(cls, survey_list, version, aper_diams, cat_creator, code_names, lowz_zmax, instruments = ['NIRCam', 'ACS_WFC', 'WFC3_IR'], \
                      forced_phot_band = "F444W", excl_bands = [], loc_depth_min_flux_pc_errs = [5, 10], templates_arr = ["fsps_larson"], select_by = None):
        
        cat_arr = [Catalogue.from_pipeline(survey, version, aper_diams, cat_creator, code_names,
                                            lowz_zmax, instruments, forced_phot_band, excl_bands, 
                                            loc_depth_min_flux_pc_errs, templates_arr, select_by) for survey in survey_list]   
            
        return cls(cat_arr)

    def __add__(self, other):
        # Check types to allow adding, Catalogue + Multiple_Catalogue, Multiple_Catalogue + Catalogue, Multiple_Catalogue + Multiple_Catalogue
        pass


    def __and__(self, other):
        pass

    def __len__(self):
        return np.sum([len(cat) for cat in self.cat_arr])

    def calc_UVLF(self):
        pass

    def calc_GSMF(self):
        pass

    def plot(self, x_name, y_name, colour_by, save = False, show = False):
        pass

    def __str__(self):
        # This should be smarter
        return ' '.join([str(cat) for cat in self.cat_arr])

    # Need to be able to save fits


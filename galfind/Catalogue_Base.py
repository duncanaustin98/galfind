from . import useful_funcs_austind as useful_funcs
from .Data import Data
from .Galaxy import Galaxy
from . import useful_funcs_austind as funcs
from .Catalogue_Creator import GALFIND_Catalogue_Creator
from . import SED_code, LePhare, EAZY, Bagpipes
from . import config

class Catalogue_Base:
    # later on, the gal_arr should be calculated from the Instrument and sex_cat path, with SED codes already given
    def __init__(self, gals, cat_path, survey, cat_creator, instrument, codes = [],  version='', ): #, UV_PDF_path):
        self.survey = survey
        self.cat_path = cat_path
        #self.UV_PDF_path = UV_PDF_path
        self.cat_creator = cat_creator
        self.instrument = instrument
        self.codes = codes
        self.gals = gals
        if version == '':
            raise Exception('Version must be specified')
        self.version = version
        
        # concat is commutative for catalogues
        self.__radd__ = self.__add__
        # cross-match is commutative for catalogues
        self.__rmul__ = self.__mul__
        
    @property
    def cat_dir(self):
        return funcs.split_dir_name(self.cat_path, "dir")
    
    @property
    def cat_name(self):
        return funcs.split_dir_name(self.cat_path, "name")
        
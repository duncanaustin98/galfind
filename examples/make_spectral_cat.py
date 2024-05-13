# make_spectral_cat.py

import glob
from msaexp import msa
import astropy.units as u

from galfind import Spectral_Catalogue

def main():
    spec_cat_v1 = Spectral_Catalogue.from_DJA(grade = 0, version = "v1") # ra_range = [53.1, 53.12] * u.deg, 
    #spec_cat_v2 = Spectral_Catalogue.from_DJA(ra_range = [53.1, 53.12] * u.deg, version = "v2")
    #spec_cat = spec_cat_v1 + spec_cat_v2
    pass

if __name__ == "__main__":
    main()
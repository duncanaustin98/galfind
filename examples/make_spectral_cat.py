# make_spectral_cat.py

import astropy.units as u

from galfind import Spectral_Catalogue


def main():
    spec_cat_v1 = Spectral_Catalogue.from_DJA(
        version="v1", ra_range=[3.5, 3.6] * u.deg
    )
    spec_cat_v2 = Spectral_Catalogue.from_DJA(
        version="v2", ra_range=[3.5, 3.6] * u.deg
    )
    spec_cat = spec_cat_v1 + spec_cat_v2
    breakpoint()


if __name__ == "__main__":
    main()

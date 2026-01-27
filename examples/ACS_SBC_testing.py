
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from galfind import Filter, Multiple_Filter, Data, Multiple_Band_Cutout

ACS_SBC_bands = ["F115LP", "F165LP"]

def plot_filters():
    multi_filt = Multiple_Filter([Filter.from_filt_name(fname) for fname in ACS_SBC_bands])
    print(multi_filt)
    fig, ax = plt.subplots()
    multi_filt.plot(ax, save = "ACS_SBC_filters.png")
    plt.close(fig)

def make_data_obj():
    survey = "nltt3330"
    version = "v1"
    instrument_names = ["ACS_SBC"]
    aper_diams = [1.0] * u.arcsec
    forced_phot_band = ["F115LP"] #ACS_SBC_bands
    min_flux_pc_err = 10.0
    # 1
    data = Data.from_survey_version(
        survey,
        version,
        instrument_names = instrument_names,
        aper_diams = aper_diams,
        forced_phot_band = forced_phot_band
    )
    # 2
    data.mask()
    # 3
    data.segment()
    # 4
    data.perform_forced_phot()

    # make catalogue object

    # make cutouts
    tab = Table.read(data.phot_cat_path)
    for ra, dec in zip(tab["ALPHA_J2000"], tab["DELTA_J2000"]):
        cutouts = Multiple_Band_Cutout.from_data_skycoord(
            data,
            sky_coord = SkyCoord(ra = ra, dec = dec, unit = u.deg),
            cutout_size = 1.5 * u.arcsec,
        )
        fig, ax = plt.subplots()
        cutouts.plot(fig = fig)
        plt.close(fig)


if __name__ == "__main__":
    make_data_obj()
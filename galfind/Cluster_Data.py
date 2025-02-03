from __future__ import annotations
from astropy.table import Table
from typing import Union, Tuple, Any, List, Dict, Callable, Optional, NoReturn, TYPE_CHECKING
if TYPE_CHECKING:
    from . import Band_Data_Base, Band_Data
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import Data


class Cluster_Data(Data):

    def __init__(
        self, 
        member_cat_path: str, 
        band_data_arr: List[Type[Band_Data]],
        forced_phot_band: Optional[Union[str, List[str], Type[Band_Data_Base]]] = None
    ):
        self.member_cat_path = member_cat_path
        # make sure this path is real and actually points to something
        # ensure the member catalogue contains everything you need
        super().__init__(band_data_arr, forced_phot_band)

    @classmethod
    def from_data_member_cat(cls, data: Data, member_cat_path: str):
        # wavelet stuff (data -> new_data)
        new_data = data # for now
        return cls.from_data(new_data, member_cat_path)
    
    @classmethod
    def from_data(cls, data: Data, member_cat_path: str):
        return cls(member_cat_path, data.band_data_arr, data.forced_phot_band)
    
    def open_cat(self):
        # open the member catalog
        tab = Table.open(self.member_cat_path)
        return tab


class Lens_Model:

    def __init__(
        self: Self,
        shear_map_path: str,
        convergence_map_path: str,
    ):
        self.shear_map_path = shear_map_path
        self.convergence_map_path = convergence_map_path

    @classmethod
    def from_data_member_cat(cls, data: Data, member_cat_path: str):
        cluster_data = Cluster_Data.from_data_member_cat(data, member_cat_path)
        return cls.from_data(cluster_data, data)

    @classmethod
    def from_data(cls, cluster_data: Cluster_Data, bkg_data: Data):
        # make cluster cat
        # make bkg cat
        return cls.from_catalogues(cluster_cat, bkg_cat)

    @classmethod
    def from_catalogues(cls, cluster_cat, bkg_cat):
        # make shear map
        # make convergence map
        return cls(shear_map_path, convergence_map_path)
    
    def __call__(self: Self):
        pass

    def __str__(self):
        return f"shear_map: {self.shear_map_path}, convergence_map: {self.convergence_map_path}"

    # @property
    # def combined_str(self):
    #     return f"{self.shear_map_path}_{self.convergence_map_path}"

    def make_lens_mag_map_at_z(self: Self, z: float) -> str:
        # TODO:

        # use galfind.astropy_cosmo !!!

        #         def H(z):
        #     """Hubble constant [1/S] at redshift z for a flat universe"""
        #     return h * H100 * np.sqrt(Om * (1 + z)**3 + OL)

        # def Hinv(z):
        #     """Inverse of H(z)"""
        #     return 1.0 / H(z)

        # def DA(z1, z2):
        #     """
        #     Angular diameter distance between two redshifts
        #     """
        #     if z2 < z1:
        #         z1, z2 = z2, z1  # Ensure z2 >= z1 (for consistency)
        #     integral, _ = quad(Hinv, z1, z2)
        #     return c / (1.0 + z2) * integral

        # def Dds_Ds(zl, zs):
        #     """
        #     Ratio of angular diameter distances
        #     """
        #     if zs == float('inf') or zs == np.inf:
        #         return 1.0
        #     else:
        #         Dds = DA(zl, zs)
        #         Ds = DA(0.0, zs)
        #         return Dds / Ds

        # def save_fits(header, data, outfile, overwrite=True):
        #     """
        #     Save data to a FITS file, preserving the header.
        #     """
        #     if os.path.exists(outfile) and overwrite:
        #         os.remove(outfile)
        #     hdu = fits.PrimaryHDU(data=data, header=header)
        #     hdul = fits.HDUList([hdu])
        #     print(f"SAVING {outfile}...")
        #     hdul.writeto(outfile, overwrite=overwrite)

        # #################################

        # def save_magnification_map(inkappa, inshear, zl, outfile, zsout):
        #     """
        #     Compute and save the magnification map based on input kappa and shear maps,
        #     lens redshift, and source redshift.
        #     """
        #     # Distance ratios
        #     Dds_Dsin = 1.0  # Fixed ratio for compatibility with Code 1
        #     Dds_Dsout = Dds_Ds(zl, zsout)

        #     # Open and read kappa map
        #     with fits.open(inkappa) as hdu_kappa:
        #         kappa_header = hdu_kappa[0].header
        #         kappa = hdu_kappa[0].data  # Keep original data type for compatibility

        #     # Open and read shear map
        #     with fits.open(inshear) as hdu_shear:
        #         shear = hdu_shear[0].data

        #     # Scale kappa and shear maps
        #     scaling_factor = Dds_Dsout / Dds_Dsin
        #     kappa_scaled = kappa * scaling_factor
        #     shear_scaled = shear * scaling_factor

        #     # Compute magnification map
        #     denominator = (1.0 - kappa_scaled)**2 - shear_scaled**2
        #     magnification = np.where(denominator != 0, 1.0 / denominator, np.inf)  # Match original behavior
        #     magnification = np.abs(magnification)  # Ensure positive magnification

        #     # Save the magnification map to FITS
        #     save_fits(kappa_header, magnification, outfile)
        return "path/to/mag_map.fits"

    def make_lens_mag_maps(self: Self, z_arr: List[float]):
        if not hasattr(self, "lens_mag_map_paths"):
            self.lens_mag_map_paths = {}
        for z in z_arr:
            self.lens_mag_map_paths[z] = self.make_lens_mag_map_at_z(z)

    def get_multiple_images(self: Self, sky_coord: SkyCoord, z: float):
        # TODO:
        pass

    def get_multiple_images_from_gal(self: Self, gal: Galaxy):
        return self.get_multiple_images(gal.skycoord, gal.z)
    
    def get_mag(self: Self, sky_coord: SkyCoord):
        assert hasattr(self, "lens_mag_map_paths"), \
            galfind_logger.critical("No lens mag maps have been made")
        # get the magnification at the sky_coord
        # interpolate and blah blah
# Spectrum.py

from __future__ import annotations

import numpy as np
import astropy.units as u
from typing import NoReturn, Union, List, TYPE_CHECKING
from numpy.typing import NDArray
from collections.abc import Callable
from abc import abstractmethod, ABC
from astropy.coordinates import SkyCoord
from pathlib import Path
from astropy.io import fits
from tqdm import tqdm
from astropy.table import Table
import msaexp.spectrum
from msaexp import msa

if TYPE_CHECKING:
    from . import Catalogue
from . import config, galfind_logger
from . import useful_funcs_austind as funcs

class Spectral_Grating: # disperser

    def __init__(self,
            name: str
            ) -> NoReturn:
        self.name = name
        self.load_dispersion_curve()
        self.load_resolution_curve()
        self.load_transmission_curve()

    def load_dispersion_curve(self):
        pass

    def get_dispersion(self, wavs):
        pass

    def load_resolution_curve(self):
        self.nominal_resolution = 100. if self.name == "PRISM" \
            else 1_000. if self.name[-1] == "M" else 2_700.

    def get_resolution(self, wavs):
        pass

    def load_transmission_curve(self):
        pass

    def get_transmission(self, wavs):
        pass


class Spectral_Filter:

    def __init__(self,
            name: str
            ) -> NoReturn:
        self.name = name
        self.load_transmission_curve()

    def load_transmission_curve(self):
        pass

    def get_transmission(self, wavs):
        pass


class Spectral_Instrument(ABC):

    def __init__(self,
            grating: Spectral_Grating,
            filter: Spectral_Filter
            ) -> NoReturn:
        pass

    @abstractmethod
    def load_sensitivity(self):
        pass

    @abstractmethod
    def get_sensitivity(self):
        pass

# average_resolution: u.Quantity,
# wavelengths: u.Quantity,
# sensitivity: Callable[..., u.Quantity],

class NIRSpec(Spectral_Instrument):

    available_grating_filters = [
        "G140M/F070LP",
        "G140M/F100LP",
        "G235M/F170LP",
        "G395M/F290LP",
        "G140H/F070LP",
        "G140H/F100LP",
        "G235H/F170LP",
        "G395H/F290LP",
        "PRISM/CLEAR"
    ]

    def __init__(self,
            grating_name: str,
            filter_name: str
            ) -> NoReturn:
        grating_filter_name = f"{grating_name}/{filter_name}"
        assert grating_filter_name in self.available_grating_filters, \
            galfind_logger.critical(f"{grating_filter_name=} not in {self.available_grating_filters=}")
        super().__init__(Spectral_Grating(grating_name), Spectral_Filter(filter_name))

    def load_sensitivity(self):
        # load from pandeia
        pass

    def get_sensitivity(self):
        # determine from self.sensitivity
        pass

instrument_conv_dict = {"NIRSPEC": NIRSpec}

class Spectrum:

    def __init__(self,
            wavs: u.Quantity,
            fluxes: Union[u.Quantity, u.Magnitude],
            fluxe_errs: Union[u.Quantity, u.Magnitude],
            sky_coord: SkyCoord,
            z: float,
            z_method: str,
            instrument: Spectral_Instrument,
            author_years: dict = {}, # {author_year: z}
            meta: dict = {}
            ) -> NoReturn:
        self.wavs = wavs
        self.fluxes = fluxes
        self.flux_errs = fluxe_errs
        self.sky_coord = sky_coord
        self.z = z
        self.z_method = z_method
        self.instrument = instrument
        self.author_years = author_years
        self.meta = meta

    @property
    def PID(self) -> Union[int, None]:
        try:
            return self._PID
        except AttributeError:
            if "PROGRAM" in self.meta.keys():
                self._PID = int(self.meta["PROGRAM"])
            else:
                raise(Exception())
            return self._PID
    
    @property
    def src_ID(self) -> Union[int, None]:
        try:
            return self._src_ID
        except AttributeError:
            if "SOURCEID" in self.meta.keys():
                self._src_ID = int(self.meta["SOURCEID"])
            else:
                raise(Exception())
            return self._src_ID
    
    @property
    def src_name(self):
        return f"{self.PID}_{self.src_ID}"

        #meta = {"PID": int(header["PROGRAM"]), "src_ID": int(header("SOURCEID")), "slit_ID": int(header["SLITID"]), "exp_time": float(header["DURATION"]), "readout_pattern": \
        #    "nod_type": str(header["NOD_TYPE"]).replace(" ", ""), "src_slit_pos": [float(header["SRCXPOS"]), float(header["SRCYPOS"])]}
        #    str(header["READPATT"]).replace(" ", ""), "n_integrations": int(header["NINTS"]), "n_groups": int(header["NGROUPS"]), \

    # def __getattr__(self, name):
    #     if name.upper() == "RA":
    #         return self.sky_coord.ra.deg
    #     elif name.upper() == "DEC":
    #         return self.sky_coord.dec.deg
    #     else:
    #         raise(Exception())#galfind_logger.critical(f"{self.__class__.__name__=} has no attribute = {name}")))

    @classmethod
    def from_DJA(cls, fits_cat: Table, url_path: str, save: bool = True, version: str = "v2") -> Self:
        # open 2D spectrum
        loc_2D_path = url_path.replace(config['Spectra']['DJA_WEB_DIR'], config['Spectra']['DJA_2D_SPECTRA_DIR'])
        if not Path(loc_2D_path).is_file():
            funcs.make_dirs(loc_2D_path)
            img = fits.open(url_path, cache = False)
            if save:
                img.writeto(loc_2D_path)
        else:
            img = fits.open(loc_2D_path)
        # extract info from img header
        header = img["SCI"].header
        sky_coord = SkyCoord(ra = float(header["SRCRA"]) * u.deg, dec = float(header["SRCDEC"]) * u.deg)
        # make Spectral_Instrument object
        grating_name = str(header["GRATING"]).replace(" ", "")
        filter_name = str(header["FILTER"]).replace(" ", "")
        try:
            instrument = instrument_conv_dict[str(header["INSTRUME"]).replace(" ", "")]
        except:
            instrument = NIRSpec
        instrument = instrument(grating_name, filter_name)

        # extract 1D spectrum from 2D fits image using msaexp
        spectrum_1D = msaexp.spectrum.SpectrumSampler(loc_2D_path)
        flux_unit = u.Unit(str(header["BUNIT"].replace(" ", "")))

        # could also extract resolution here
        mask = spectrum_1D.spec["valid"]
        wavs = spectrum_1D.spec["wave"] * u.um
        fluxes = spectrum_1D.spec["flux"] * flux_unit
        if version == "v2":
            flux_errs = None
        else:
            flux_errs = [spectrum_1D.spec["full_err"], spectrum_1D.spec["full_err"]] * flux_unit
        
        z = None
        z_method = None
        return cls(wavs, fluxes, flux_errs, z, z_method, sky_coord, instrument, meta = {name: header[name] for name in header})


# should inherit from Catalogue_Base
class Spectral_Catalogue:

    def __init__(self, 
            spectrum_arr: NDArray[Spectrum]
            ) -> NoReturn:
        # check if any of the sources are the same
        orig_src_names = [spec.src_name for spec in spectrum_arr]
        unique_src_names = np.unique(orig_src_names)
        self.spectrum_arr = np.array([[spec for spec in spectrum_arr \
            if spec.src_name == src_name] for src_name in unique_src_names])
        breakpoint()
        #self.sky_coords = np.array([spec[0].sky_coord for spec in self.spectrum_arr])

    def __len__(self):
        return len(self.spectrum_arr)
    
    def __iter__(self):
        self.iter = 0
        return self
    
    def __next__(self):
        if self.iter > len(self) - 1:
            raise StopIteration
        else:
            gal_spectra = self[self.iter]
            self.iter += 1
            return gal_spectra
    
    def __getitem__(self, index):
        return self.spectrum_arr[index]

    def __getattr__(self, name):
        if hasattr(self[0][0], name):
            return [getattr(gal[0], name) for gal in self]

    def __add__(self, cat):
        return Spectral_Catalogue(np.array(self.spectrum_arr).flatten(), np.array(cat.spectrum_arr).flatten())

    @classmethod
    def from_DJA(cls, ra_range: u.Quantity = None, dec_range: u.Quantity = None, PID: int = None, \
            grating_filter: str = None, grade: int = 3, save: bool = True, version: str = "v2") -> Self:
        if type(grating_filter) != type(None):
            assert grating_filter in NIRSpec.available_grating_filters
        assert version in ["v1", "v2"]
        # open and crop catalogue
        #DJA_cat = utils.read_catalog(config['Spectra']['DJA_CAT_PATH'], format = "ascii.ecsv")
        DJA_cat = Table.read(config['Spectra']['DJA_CAT_PATH'].replace("v2", version))
        if type(ra_range) != type(None):
            assert(len(ra_range) == 2)
            ra_range = sorted(ra_range.to(u.deg).value)
            DJA_cat = DJA_cat[((DJA_cat["ra"] > ra_range[0]) & (DJA_cat["ra"] < ra_range[1]))]
        if type(dec_range) != type(None):
            assert(len(dec_range) == 2)
            dec_range = sorted(ra_range.to(u.deg).value)
            DJA_cat = DJA_cat[((DJA_cat["dec"] > dec_range[0]) & (DJA_cat["dec"] < dec_range[1]))]
        if type(grade) != type(None):
            DJA_cat = DJA_cat[DJA_cat["grade"] == grade]
        if type(grating_filter) != type(None):
            DJA_cat = DJA_cat[DJA_cat["grating"] == grating_filter.split("/")[0]]
        if type(grating_filter) != type(None):
            DJA_cat = DJA_cat[DJA_cat["filter"] == grating_filter.split("/")[1]]
        if type(PID) != type(None):
            DJA_cat = DJA_cat[DJA_cat["PID"] == PID]
        return cls([Spectrum.from_DJA(DJA_cat[i], f"{config['Spectra']['DJA_WEB_DIR']}/{root}/{file}", save = save) \
            for i, (root, file) in tqdm(enumerate(zip(DJA_cat["root"], DJA_cat["file"])), \
            total = len(DJA_cat), desc = f"Saving 2D spectra from {config['Spectra']['DJA_CAT_PATH'].replace('v2', version)}")])

            




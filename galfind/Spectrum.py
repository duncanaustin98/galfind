# Spectrum.py

from __future__ import annotations

import numpy as np
import astropy.units as u
from typing import NoReturn, Union, TYPE_CHECKING
from numpy.typing import NDArray
from astropy.utils.masked import Masked
from abc import abstractmethod, ABC
from astropy.coordinates import SkyCoord
from pathlib import Path
from astropy.io import fits
from tqdm import tqdm
from astropy.table import Table
import os

if TYPE_CHECKING:
    pass
from . import config, galfind_logger
from . import useful_funcs_austind as funcs


class Spectral_Grating:  # disperser
    def __init__(self, name: str) -> NoReturn:
        self.name = name
        self.load_dispersion_curve()
        self.load_resolution_curve()
        self.load_transmission_curve()

    def load_dispersion_curve(self):
        pass

    def get_dispersion(self, wavs):
        pass

    def load_resolution_curve(self):
        self.nominal_resolution = (
            100.0
            if self.name == "PRISM"
            else 1_000.0
            if self.name[-1] == "M"
            else 2_700.0
        )

    def get_resolution(self, wavs):
        pass

    def load_transmission_curve(self):
        pass

    def get_transmission(self, wavs):
        pass


class Spectral_Filter:
    def __init__(self, name: str) -> NoReturn:
        self.name = name
        self.load_transmission_curve()

    def load_transmission_curve(self):
        pass

    def get_transmission(self, wavs):
        pass


class Spectral_Instrument(ABC):
    def __init__(self, grating: Spectral_Grating, filter: Spectral_Filter) -> NoReturn:
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
        "PRISM/CLEAR",
    ]

    def __init__(self, grating_name: str, filter_name: str) -> NoReturn:
        grating_filter_name = f"{grating_name}/{filter_name}"
        self.grating_filter_name = grating_filter_name
        assert (
            grating_filter_name in self.available_grating_filters
        ), galfind_logger.critical(
            f"{grating_filter_name=} not in {self.available_grating_filters=}"
        )
        super().__init__(Spectral_Grating(grating_name), Spectral_Filter(filter_name))

    def load_sensitivity(self):
        # load from pandeia
        pass

    def get_sensitivity(self):
        # determine from self.sensitivity
        pass


instrument_conv_dict = {"NIRSPEC": NIRSpec}


class Spectrum:
    def __init__(
        self,
        wavs: u.Quantity,
        fluxes: Union[u.Quantity, u.Magnitude],
        flux_errs: Union[u.Quantity, u.Magnitude],
        sky_coord: SkyCoord,
        z: float,
        z_method: str,
        instrument: Spectral_Instrument,
        reduction_name: str,
        MSA_metafile_name: str,
        author_years: dict = {},  # {author_year: z}
        meta: dict = {},
    ) -> NoReturn:
        self.wavs = wavs
        self.fluxes = fluxes
        self.flux_errs = flux_errs
        self.sky_coord = sky_coord
        self.z = z
        self.z_method = z_method
        self.instrument = instrument
        self.reduction_name = reduction_name
        self.MSA_metafile_name = MSA_metafile_name
        self.author_years = author_years
        self.meta = meta

    @property
    def PID(self) -> Union[int, None]:
        try:
            return self._PID
        except AttributeError:
            if "PROGRAM" in self.meta.keys():
                self._PID = int(self.meta["PROGRAM"])
            elif "SRCNAM1" in self.meta.keys():
                self._PID = str(self.meta["SRCNAM1"].split("_")[0])
            else:
                raise (Exception())
            return self._PID

    @property
    def src_ID(self) -> Union[int, None]:
        try:
            return self._src_ID
        except AttributeError:
            if "SOURCEID" in self.meta.keys():
                self._src_ID = int(self.meta["SOURCEID"])
            elif "SRCNAM1" in self.meta.keys():
                self._src_ID = int(self.meta["SRCNAM1"].split("_")[1])
            else:
                raise (Exception())
            return self._src_ID

    @property
    def src_name(self):
        return f"{self.PID}_{self.src_ID}"

    @property
    def MSA_ID(self):
        try:
            return self._meta_ID
        except AttributeError:
            if "MSAMETID" in self.meta.keys():
                self._meta_ID = int(self.meta["MSAMETID"])
            else:
                raise (Exception())
            return self._meta_ID

    @property
    def dither_pt(self):
        try:
            return self._dither_pt
        except AttributeError:
            if "PATT_NUM" in self.meta.keys():
                self._dither_pt = int(self.meta["PATT_NUM"])
            else:
                raise (Exception())
            return self._dither_pt

        # meta = {"PID": int(header["PROGRAM"]), "src_ID": int(header("SOURCEID")), "slit_ID": int(header["SLITID"]), "exp_time": float(header["DURATION"]), "readout_pattern": \
        #    "nod_type": str(header["NOD_TYPE"]).replace(" ", ""), "src_slit_pos": [float(header["SRCXPOS"]), float(header["SRCYPOS"])]}
        #    str(header["READPATT"]).replace(" ", ""), "n_integrations": int(header["NINTS"]), "n_groups": int(header["NGROUPS"]), \

    @classmethod
    def from_DJA(
        cls,
        url_path: str,
        save: bool = True,
        version: str = "v2",
        z: Union[float, None] = None,
    ) -> Self:
        import msaexp.spectrum

        # open 2D spectrum
        loc_2D_path = url_path.replace(
            config["Spectra"]["DJA_WEB_DIR"], config["Spectra"]["DJA_2D_SPECTRA_DIR"]
        )
        if not Path(loc_2D_path).is_file():
            funcs.make_dirs(loc_2D_path)
            img = fits.open(url_path, cache=False)
            if save:
                img.writeto(loc_2D_path)
                funcs.change_file_permissions(loc_2D_path)
        else:
            img = fits.open(loc_2D_path)
        # extract info from img header
        header = img["SCI"].header
        sky_coord = SkyCoord(
            ra=float(header["SRCRA"]) * u.deg, dec=float(header["SRCDEC"]) * u.deg
        )
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
        mask = ~spectrum_1D.spec["valid"]
        wavs = spectrum_1D.spec["wave"] * u.um
        fluxes = Masked(spectrum_1D.spec["flux"] * flux_unit, mask=mask)

        if version == "v2":
            msa_metafile = str(header["MSAMETFL"]).replace(" ", "")
            # if int(header["PROGRAM"]) == 2561:
            #     #breakpoint()
            # determine number of exposures
            N_exposures = int(header["NOUTPUTS"]) * int(header["NFRAMES"])
            full_flux_errs = spectrum_1D.spec["full_err"] * (N_exposures**-0.25)
        else:
            msa_metafile = str(header["MSAMET1"]).replace(" ", "")
            full_flux_errs = spectrum_1D.spec["full_err"]
        meta_uri_dir = (
            "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product"
        )
        meta_in_path = f"{meta_uri_dir}/{msa_metafile}"

        try:
            out_dir = config["Spectra"]["DJA_2D_SPECTRA_DIR"].replace(
                "2D", "MSA_metafiles"
            )
            meta_out_path = f"{out_dir}/{msa_metafile}"
            if not Path(meta_out_path).is_file():
                meta = fits.open(meta_in_path, cache=False)
                funcs.make_dirs(meta_out_path)
                meta.writeto(meta_out_path)
                funcs.change_file_permissions(meta_out_path)
            MSA_metafile_name = meta_out_path
        except:
            MSA_metafile_name = None

        flux_errs = Masked(np.array(full_flux_errs) * flux_unit, mask=mask)

        if type(z) != type(None):
            z_method = "cat"
        else:
            z = None
            z_method = None
        reduction_name = f"DJA_{version}"

        spec_obj = cls(
            wavs,
            fluxes,
            flux_errs,
            sky_coord,
            z,
            z_method,
            instrument,
            reduction_name,
            MSA_metafile_name,
            meta={name: header[name] for name in header},
        )
        spec_obj.origin = loc_2D_path
        return spec_obj

    def load_MSA_metafile(self):
        from msaexp import msa

        if not hasattr(self, "MSA_metafile"):
            try:
                self.MSA_metafile = msa.MSAMetafile(self.MSA_metafile_name)
            except:
                self.MSA_metafile = None

    def plot_slitlet(self, ax, colour="black", add_labels=True):
        # mostly copied from msaexp MSAMetafile base code
        self.load_MSA_metafile()
        assert type(self.MSA_metafile) != type(None)
        slits = self.MSA_metafile.regions_from_metafile(
            dither_point_index=self.dither_pt,
            as_string=False,
            with_bars=True,
            msa_metadata_id=self.MSA_ID,
        )
        for s in slits:
            if s.meta["is_source"]:
                kwargs = dict(color=colour, alpha=0.8, zorder=100)
            else:
                kwargs = dict(color="0.7", alpha=0.8, zorder=100)
            ax.plot(*np.vstack([s.xy[0], s.xy[0][:1, :]]).T, **kwargs)

        if add_labels:
            ax.text(
                0.03,
                0.07,
                f"Dither #{self.dither_pt}",
                ha="left",
                va="bottom",
                transform=ax.transAxes,
                color=colour,
                fontsize=8,
            )
            ax.text(
                0.03,
                0.03,
                f"{os.path.basename(self.MSA_metafile.metafile)}",
                ha="left",
                va="bottom",
                transform=ax.transAxes,
                color=colour,
                fontsize=8,
            )
            ax.text(
                0.97,
                0.07,
                f"{self.src_ID}",
                ha="right",
                va="bottom",
                transform=ax.transAxes,
                color=colour,
                fontsize=8,
            )
            # ax.text(
            #     0.97,
            #     0.03,
            #     f"({self.sky_coord.ra.deg:.6f}, {self.sky_coord.dec.deg:.6f})",
            #     ha = "right",
            #     va = "bottom",
            #     transform = ax.transAxes,
            #     color = colour,
            #     fontsize = 8,
            # )

    def calc_SNR_cont(
        self, rest_cont_wav: u.Quantity, delta_wav: u.Quantity = 100 * u.AA
    ):
        rest_wavs = self.wavs / (1.0 + self.z)
        wav_mask = (rest_wavs > rest_cont_wav - delta_wav / 2.0) & (
            rest_wavs < rest_cont_wav + delta_wav / 2.0
        )
        fluxes = self.fluxes[wav_mask]
        flux_errs = self.flux_errs[wav_mask]
        SNR_arr = [flux / err for flux, err in zip(fluxes, flux_errs)]
        mean_SNR = np.mean(SNR_arr)
        return mean_SNR

    def plot_spectrum(self, src="msaexp"):
        if src == "msaexp":
            import msaexp.spectrum

            fig, spec, data = msaexp.spectrum.plot_spectrum(self.origin, z=self.z)
            save_path = f"{config['DEFAULT']['GALFIND_WORK']}/DJA_spec_plots/{self.instrument.grating_filter_name}/{self.src_name}_spec.png"
            funcs.make_dirs(save_path)
            fig.savefig(save_path)


# should inherit from Catalogue_Base
class Spectral_Catalogue:
    def __init__(self, spectrum_arr: NDArray[Spectrum]) -> NoReturn:
        # check if any of the sources are the same
        orig_src_names = [spec.src_name for spec in spectrum_arr]
        unique_src_names = np.unique(orig_src_names)
        self.spectrum_arr = np.array(
            [
                [spec for spec in spectrum_arr if spec.src_name == src_name]
                for src_name in unique_src_names
            ]
        )
        # self.sky_coords = np.array([spec[0].sky_coord for spec in self.spectrum_arr])

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
        assert cat.__class__.__name__ == "Spectral_Catalogue"
        spectra_arr = np.array(
            [spectrum for gal in self for spectrum in gal]
            + [spectrum for gal in cat for spectrum in gal]
        )
        return Spectral_Catalogue(spectra_arr)

    @classmethod
    def from_DJA(
        cls,
        ra_range: Union[list, np.array, u.Quantity] = None,
        dec_range: Union[list, np.array, u.Quantity] = None,
        PID: Union[int, None] = None,
        z_cat_range: Union[list, np.array, None] = None,
        grating_filter: Union[str, None] = None,
        grade: int = 3,
        save: bool = True,
        z_from_cat: bool = False,
        version: str = "v2",
    ):
        if type(grating_filter) != type(None):
            assert grating_filter in NIRSpec.available_grating_filters
        assert version in ["v1", "v2"]
        # open and crop catalogue
        # DJA_cat = utils.read_catalog(config['Spectra']['DJA_CAT_PATH'], format = "ascii.ecsv")
        DJA_cat = Table.read(config["Spectra"]["DJA_CAT_PATH"].replace("v2", version))
        if type(ra_range) != type(None):
            assert len(ra_range) == 2
            if type(ra_range) in [list, np.array]:
                assert ra_range[0].unit == ra_range[1].unit
                ra_range = [ra_range[0].value, ra_range[1].value] * ra_range[0].unit
            ra_range = sorted(ra_range.to(u.deg).value)
            DJA_cat = DJA_cat[
                ((DJA_cat["ra"] > ra_range[0]) & (DJA_cat["ra"] < ra_range[1]))
            ]
        if type(dec_range) != type(None):
            assert len(dec_range) == 2
            if type(dec_range) in [list, np.array]:
                assert dec_range[0].unit == dec_range[1].unit
                dec_range = [dec_range[0].value, dec_range[1].value] * dec_range[0].unit
            dec_range = sorted(dec_range.to(u.deg).value)
            DJA_cat = DJA_cat[
                ((DJA_cat["dec"] > dec_range[0]) & (DJA_cat["dec"] < dec_range[1]))
            ]
        if type(grade) != type(None):
            DJA_cat = DJA_cat[DJA_cat["grade"] == grade]
        if type(grating_filter) != type(None):
            if "grating" in DJA_cat.colnames:
                DJA_cat = DJA_cat[DJA_cat["grating"] == grating_filter.split("/")[0]]
        if type(grating_filter) != type(None):
            if "filter" in DJA_cat.colnames:
                DJA_cat = DJA_cat[DJA_cat["filter"] == grating_filter.split("/")[1]]
        if type(z_cat_range) != type(None):
            DJA_cat = DJA_cat[
                ((DJA_cat["z"] > z_cat_range[0]) & (DJA_cat["z"] < z_cat_range[1]))
            ]
            z_from_cat = True
        if type(PID) != type(None):
            if "PID" in DJA_cat.colnames:
                DJA_cat = DJA_cat[DJA_cat["PID"] == PID]

        if z_from_cat:
            return cls(
                [
                    Spectrum.from_DJA(
                        f"{config['Spectra']['DJA_WEB_DIR']}/{root}/{file}",
                        save=save,
                        version=version,
                        z=z,
                    )
                    for root, file, z in tqdm(
                        zip(DJA_cat["root"], DJA_cat["file"], DJA_cat["z"]),
                        total=len(DJA_cat),
                        desc=f"Loading DJA_{version} catalogue",
                    )
                ]
            )
        else:
            return cls(
                [
                    Spectrum.from_DJA(
                        f"{config['Spectra']['DJA_WEB_DIR']}/{root}/{file}",
                        save=save,
                        version=version,
                    )
                    for root, file in tqdm(
                        zip(DJA_cat["root"], DJA_cat["file"]),
                        total=len(DJA_cat),
                        desc=f"Loading DJA_{version} catalogue",
                    )
                ]
            )

    # def crop_to_grating(self, name = "G395H/F290LP"):

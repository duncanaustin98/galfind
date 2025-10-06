# Spectrum.py

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import NoReturn, Union, Optional
import logging
from copy import deepcopy
import h5py
#from lmfit import Model, Parameters, minimize, fit_report

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.utils.masked import Masked
from numpy.typing import NDArray
from tqdm import tqdm

from . import config, galfind_logger
from . import useful_funcs_austind as funcs
from . import astropy_cosmo as cosmo


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
        self.resolution_curve_path = f"{config['Spectra']['R_CURVE_DIR']}/NIRSpec/jwst_nirspec_prism_disp.fits"

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
    def __init__(
        self,
        grating: Spectral_Grating,
        filter: Spectral_Filter,
    ) -> NoReturn:
        self.grating = grating
        self.filter = filter

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
        super().__init__(
            Spectral_Grating(grating_name),
            Spectral_Filter(filter_name),
        )

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
        **kwargs,
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
        for key, value in kwargs.items():
            setattr(self, key, value)

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
        version: str = "v3",
        z: Union[float, None] = None,
        *args,
        **kwargs,
    ) -> Self:

        # open 2D spectrum
        loc_2d_path = url_path.replace(
            config["Spectra"]["DJA_WEB_DIR"],
            config["Spectra"]["DJA_2D_SPECTRA_DIR"],
        )
        #breakpoint()
        if not Path(loc_2d_path).is_file():
            funcs.make_dirs(loc_2d_path)
            img = fits.open(url_path, cache=False)
            if save:
                img.writeto(loc_2d_path)
                funcs.change_file_permissions(loc_2d_path)
        else:
            img = fits.open(loc_2d_path)
        # extract info from img header
        header = img["SCI"].header
        sky_coord = SkyCoord(
            ra=float(header["SRCRA"]) * u.deg,
            dec=float(header["SRCDEC"]) * u.deg,
        )
        # make Spectral_Instrument object
        grating_name = str(header["GRATING"]).replace(" ", "")
        filter_name = str(header["FILTER"]).replace(" ", "")
        try:
            instrument = instrument_conv_dict[
                str(header["INSTRUME"]).replace(" ", "")
            ]
        except:
            instrument = NIRSpec
        instrument = instrument(grating_name, filter_name)

        # extract 1D spectrum from 2D fits image using msaexp
        loc_1d_path = url_path.replace(
            config["Spectra"]["DJA_WEB_DIR"],
            config["Spectra"]["DJA_1D_SPECTRA_DIR"],
        )
        if not Path(loc_1d_path).is_file():
            import msaexp.spectrum
            spectrum_1D = msaexp.spectrum.SpectrumSampler(loc_2d_path)
            # could also extract resolution here
            mask = ~spectrum_1D.spec["valid"]
            wavs = spectrum_1D.spec["wave"]
            fluxes = spectrum_1D.spec["flux"] #Masked( * flux_unit, mask = mask)
            if version == "v2":
                # determine number of exposures
                N_exposures = int(header["NOUTPUTS"]) * int(header["NFRAMES"])
                flux_errs = spectrum_1D.spec["full_err"] * (
                    N_exposures**-0.25
                )
            elif version in ["v3", "v4_2"]:
                flux_errs = spectrum_1D.spec["full_err"]
            else:
                flux_errs = spectrum_1D.spec["full_err"]
            # save as local .h5 file
            funcs.make_dirs(loc_1d_path)
            hf = h5py.File(loc_1d_path, "w")
            for name, data in zip(
                ["mask", "wavs", "fluxes", "flux_errs"],
                [mask, wavs, fluxes, flux_errs]
            ):
                hf.create_dataset(name, data = data)
            wav_unit = u.um # NOT GENERAL!
            flux_unit = u.Unit(str(header["BUNIT"].replace(" ", "")))
            hf.attrs["wav_unit"] = (u.um).to_string()
            hf.attrs["flux_unit"] = flux_unit.to_string()
            hf.close()
        else:
            hf = h5py.File(loc_1d_path, "r")
            mask = np.array(hf["mask"])
            wavs = np.array(hf["wavs"])
            wav_unit = u.Unit(hf.attrs["wav_unit"])
            flux_unit = u.Unit(hf.attrs["flux_unit"])
            fluxes = np.array(hf["fluxes"])
            flux_errs = np.array(hf["flux_errs"])
        wavs *= wav_unit
        fluxes = Masked(fluxes * flux_unit, mask = mask)
        flux_errs = Masked(np.array(flux_errs) * flux_unit, mask = mask)

        if version == "v2":
            msa_metafile = str(header["MSAMETFL"]).replace(" ", "")
        elif version in ["v3", "v4_2"]:
            msa_metafile = str(header["MSAMETFL"]).replace(" ", "")
        else:
            msa_metafile = str(header["MSAMET1"]).replace(" ", "")
        
        meta_uri_dir = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product"
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

        if z is None:
            z_method = None
        else:
            z_method = "cat"
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
            **kwargs,
        )
        spec_obj.origin = loc_2d_path
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
        assert self.MSA_metafile is not None
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
        self: Self,
        rest_cont_wav: u.Quantity,
        delta_wav: u.Quantity = 100 * u.AA,
    ):
        assert hasattr(self, "z"), galfind_logger.critical(
            f"{repr(self)} does not have a redshift (z) attribute!"
        )
        rest_wavs = self.wavs / (1.0 + self.z)
        wav_mask = (rest_wavs > rest_cont_wav - delta_wav / 2.0) & (
            rest_wavs < rest_cont_wav + delta_wav / 2.0
        )
        fluxes = self.fluxes[wav_mask]
        flux_errs = self.flux_errs[wav_mask]
        SNR_arr = [flux / err for flux, err in zip(fluxes, flux_errs)]
        mean_SNR = np.mean(SNR_arr)
        # HACK: This is not general!
        self.SNR = mean_SNR
        return mean_SNR

    def plot(
        self: Self,
        src: str = "msaexp",
        out_dir: Optional[str] = f"{config['DEFAULT']['GALFIND_WORK']}/DJA_spec_plots/",
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        wav_units: u.Unit = u.um,
        flux_units: u.Unit = u.uJy,
    ) -> NoReturn:
        assert src in ["msaexp", "manual"], galfind_logger.critical(
            f"{src=} not in ['msaexp', 'manual']"
        )
        if src == "msaexp":
            import msaexp.spectrum

            fig, spec, data = msaexp.spectrum.plot_spectrum(
                self.origin, z=self.z
            )
            self.fit_data = data
            if out_dir is None:
                out_dir = ""
            save_path = f"{out_dir}{self.instrument.grating_filter_name}/{self.src_name}_spec.png"
            funcs.make_dirs(save_path)
            fig.savefig(save_path)
        elif src == "manual":
            if fig is None or ax is None:
                fig, ax = plt.subplots()
            # unit conversions
            wavs = funcs.convert_wav_units(self.wavs, wav_units)
            fluxes = funcs.convert_mag_units(self.wavs, self.fluxes, flux_units)
            ax.plot(wavs, fluxes, label=self.src_name)
    
    def make_mock_phot(
        self: Self,
        filterset: Multiple_Filter,
        depths: Optional[Dict[str, float]] = None,
    ):
        from . import SED_obs
        # TODO: Link SED and Spectrum objects
        # make SED object from self
        assert self.z is not None, galfind_logger.critical(
            f"{repr(self)} does not have a redshift (z) attribute!"
        )
        sed_obs = SED_obs(self.z, self.wavs.value, self.fluxes.value, self.wavs.unit, self.fluxes.unit)
        return sed_obs.create_mock_photometry(
            filterset,
            depths = depths
        )

    def fit_MUV(self: Self, wav_range = [1_450.0, 1_550.0] * u.AA, size = 10_000):
        assert hasattr(self, "z"), galfind_logger.critical( 
            f"{repr(self)} does not have a redshift (z) attribute!"
        )
        rest_wavs = funcs.convert_wav_units(self.wavs, u.AA) / (1.0 + self.z)
        wav_range_AA = wav_range.to(u.AA)
        valid = (~self.fluxes.mask & (rest_wavs < wav_range_AA[1]) & (rest_wavs > wav_range_AA[0]))
        rest_wavs = rest_wavs[valid]
        fluxes = self.fluxes.filled(np.nan)[valid]
        flux_errs = self.flux_errs.filled(np.nan)[valid]

        if len(rest_wavs) > 0:
            try:
                #breakpoint()
                flux_errs = funcs.convert_mag_err_units(rest_wavs, fluxes, [flux_errs, flux_errs], u.erg / u.s / u.cm**2 / u.AA)[0] # symmetric in flux space
            except Exception as e:
                print(f"Failed to convert mag err units for {repr(self)}")
                print(e)
                return None
        else:
            print(f"No valid data for {self.src_name}")
            return None
        fluxes = funcs.convert_mag_units(rest_wavs, fluxes, u.erg / u.s / u.cm**2 / u.AA)
        # fluxes *= (1. + z) ** 2
        # flux_errs *= (1. + z) ** 2
        # weighted mean
        weights = flux_errs ** -2.
        flambda_1500 = np.sum(fluxes * weights) / (np.sum(weights) * len(fluxes))
        flambda_1500_err = np.sqrt(1. / np.sum(weights))
        # convert to MUV
        flambda_1500_chains = np.random.normal(flambda_1500.value, flambda_1500_err.value, size) * u.erg / (u.s * (u.cm ** 2) * u.AA)
        self.flambda_1500_chains = flambda_1500_chains
        fnu = funcs.convert_mag_units(1_500. * u.AA, flambda_1500_chains, u.Jy)
        mUV = -2.5 * np.log10(fnu.value) + u.Jy.to(u.ABmag)
        #mUV += 2.5 * np.log10(self.norm_factor)
        MUV_arr = mUV - 5.0 * np.log10(cosmo.luminosity_distance(self.z).to(u.pc).value / 10.0) + 2.5 * np.log10(1. + self.z)
        self.MUV_arr = MUV_arr
        self.MUV = np.nanmedian(MUV_arr)
        self.MUV_l1 = self.MUV - np.nanpercentile(MUV_arr, 16)
        self.MUV_u1 = np.nanpercentile(MUV_arr, 84) - self.MUV
        return self.MUV, [self.MUV_l1, self.MUV_u1]
    
    def fit_Ha(
        self: Self,
        wav_range = [6_200., 6_900.] * u.AA,
        Halpha_wav: u.Quantity = 6562.8 * u.AA,
        frame: str = "rest",
        plot: bool = True,
        size: int = 10_000,
    ):
        rest_wavs = funcs.convert_wav_units(self.wavs, u.AA) / (1. + self.z)
        wav_range_AA = wav_range.to(u.AA)
        valid = (~self.fluxes.mask & (rest_wavs < wav_range_AA[1]) & (rest_wavs > wav_range_AA[0]))
        rest_wavs = rest_wavs[valid]
        fluxes = self.fluxes.filled(np.nan)[valid]
        flux_errs = self.flux_errs.filled(np.nan)[valid]
        if len(rest_wavs) > 0:
            flux_errs = funcs.convert_mag_err_units(rest_wavs, fluxes, [flux_errs, flux_errs], u.erg / u.s / u.cm**2 / u.AA)[0] # symmetric in flux space
        else:
            print(f"No valid data for {spec_filepath.split('/')[-1]}")
            return None
        fluxes = funcs.convert_mag_units(rest_wavs, fluxes, u.erg / u.s / u.cm**2 / u.AA)
    
        # TODO: This doesn't actually constrain the width - something wrong with the fitting here!
        params = Parameters()
        params.add('A', value=np.max(fluxes.value) - np.median(fluxes.value), min=0., max=1e-12)
        params.add('c', value=np.median(fluxes.value), min=0., max=1e-12)
        params.add('sigma', value=15., min=1., max=50.)
        # dmodel = Model(gauss_model)
        # result = dmodel.fit(fluxes.value, params, wavs=wavs.value)
        # print(result.fit_report())

        out = minimize(Halpha_residual, params, args=(rest_wavs.value,), kws={'y': fluxes.value, 'y_err': flux_errs.value})
        print(fit_report(out))

        sigma = out.params["sigma"].value
        cont = out.params["c"].value

        try:
            A_arr = np.random.normal(loc=out.params["A"].value, scale=out.params["A"].stderr, size=size)
            sigma_arr = np.random.normal(loc=params["sigma"].value, scale=out.params["sigma"].stderr, size=size)
            cont_arr = np.random.normal(loc=params["c"].value, scale=out.params["c"].stderr, size=size)
        except:
            breakpoint()
            self.failed_Ha_fit = True
            return

        Halpha_flux_arr =  A_arr * sigma_arr * np.sqrt(2 * np.pi)
        self.Halpha_flux_arr = Halpha_flux_arr * u.erg / u.s / u.cm ** 2
        # Halpha_flux_arr = [self._Halpha_flux(A, sigma, c) * self.norm_factor \
        #     for A, sigma, c in zip(self.A_arr, self.sigma_arr, self.c_arr)]
        if frame == "rest":
            EW_arr = Halpha_flux_arr / cont_arr
        elif frame == "obs":
            EW_arr = Halpha_flux_arr * (1. + self.z) / cont_arr
        else:
            raise ValueError("frame must be 'rest' or 'obs'")
        EW_percentiles = np.percentile(EW_arr, [16, 50, 84])
        if frame == "rest":
            self.Ha_EWrest = {"16": EW_percentiles[0], "50": EW_percentiles[1], "84": EW_percentiles[2]}
            #return EW_percentiles[0], EW_percentiles[1], EW_percentiles[2]
        elif frame == "obs":
            self.Ha_EWobs = {"16": EW_percentiles[0], "50": EW_percentiles[1], "84": EW_percentiles[2]}
            #return EW_percentiles[0], EW_percentiles[1], EW_percentiles[2]
        else:
            raise(Exception("frame must be 'rest' or 'obs'"))

        cont_percentiles = np.percentile(cont_arr, [16, 50, 84])
        self.Ha_cont = {"16": cont_percentiles[0], "50": cont_percentiles[1], "84": cont_percentiles[2]}

        Halpha_flux_percentiles = np.percentile(Halpha_flux_arr, [16, 50, 84])
        self.Ha_flux = {"16": Halpha_flux_percentiles[0], "50": Halpha_flux_percentiles[1], "84": Halpha_flux_percentiles[2]}

        feature_mask = (rest_wavs.value > Halpha_wav.value - 5. * sigma) & (rest_wavs.value < Halpha_wav.value + 5. * sigma)
        SNRs = (fluxes[feature_mask].value - cont) / flux_errs.value[feature_mask]
        integrated_SNR = np.sum(SNRs) / np.sqrt(len(SNRs))
        print(f"Integrated SNR: {integrated_SNR}")
        assert not np.isnan(integrated_SNR)
        self.Ha_SNR = integrated_SNR
    
        fig, ax = plt.subplots()
        if plot:
            ax.plot(rest_wavs.value, fluxes.value, c = "black", label = "NIRSpec/PRISM")
            ax.fill_between(rest_wavs.value, fluxes.value - flux_errs.value, fluxes.value + flux_errs.value, alpha = 0.5, color = "black")
            median_chains = [np.median(Halpha_gauss(wav, A_arr, sigma_arr, cont_arr)) for wav in rest_wavs.value]
            ax.plot(rest_wavs.value, median_chains, c = "red", label = "Halpha model")
            model_l1 = [np.percentile(Halpha_gauss(wav, A_arr, sigma_arr, cont_arr), 16) for wav in rest_wavs.value]
            model_u1 = [np.percentile(Halpha_gauss(wav, A_arr, sigma_arr, cont_arr), 84) for wav in rest_wavs.value]
            ax.fill_between(rest_wavs.value, model_l1, model_u1, alpha = 0.5, color = "red")
            # make rf string containing EW width and errors
            #plt.text(0.05, 0.95, r"EW$_{\mathrm{rest}}$(H$\alpha$)=" + f"{Halpha_EWrest_50:.2f}" + r"$^{+" + f"{Halpha_EWrest_84 - Halpha_EWrest_50:.2f}" + r"}_{-" + f"{Halpha_EWrest_50 - Halpha_EWrest_16:.2f}" + r"}~\mathrm{\AA}$", transform = plt.gca().transAxes)
            # make rf string containing flux and errors
            ax.text(0.05, 0.95, r"$F_{\mathrm{H}\alpha}$=" + f"{Halpha_flux_percentiles[1]:.2e}" + r"$^{+" + f"{Halpha_flux_percentiles[2] - Halpha_flux_percentiles[1]:.2e}" + r"}_{-" + f"{Halpha_flux_percentiles[1] - Halpha_flux_percentiles[0]:.2e}" + r"}~\mathrm{erg/s/cm^2}$", transform = plt.gca().transAxes)
            # make rf string to show the SNR
            ax.text(0.05, 0.9, f"SNR={self.Ha_SNR:.2f}", transform = plt.gca().transAxes)
            out_path = f"../plots/Halpha_spec_fits/manual/{self.file}.png"
            funcs.make_dirs(out_path)
            ax.set_xlabel("Wavelength (AA)")
            ax.set_ylabel("Flux (erg/s/cm^2/AA)")
            ax.legend(loc = "upper right")
            plt.savefig(out_path)
            plt.clf()

    def fit_xi_ion(self: Self, plot: bool = False):
        self.fit_MUV()
        self.fit_Ha(plot = plot)
        #breakpoint()
        LUV_arr = funcs.flux_to_luminosity(self.flambda_1500_chains, 1_500.0 * u.AA, self.z)
        LHa_arr = funcs.flux_to_luminosity(self.Halpha_flux_arr / (1.0 + self.z), 6_562.8 * u.AA, self.z, out_units = u.erg / u.s)
        self.ndot_ion_arr = 7.28e11 * LHa_arr.value
        self.ndot_ion = np.median(self.ndot_ion_arr)
        self.ndot_ion_l1 = np.percentile(self.ndot_ion_arr, 16)
        self.ndot_ion_u1 = np.percentile(self.ndot_ion_arr, 84)
        self.xi_ion_arr = self.ndot_ion_arr / LUV_arr.value
        self.xi_ion = np.median(self.xi_ion_arr)
        self.xi_ion_l1 = np.percentile(self.xi_ion_arr, 16)
        self.xi_ion_u1 = np.percentile(self.xi_ion_arr, 84)

def Halpha_gauss(x, A, sigma, c):
    return A * np.exp(-0.5 * ((x - 6562.8) / sigma)**2) + c

def Halpha_residual(params, x, y, y_err):
    # gaussian plus a constant
    model = Halpha_gauss(x, params['A'], params['sigma'], params['c'])
    return (model - y) / y_err


# should inherit from Catalogue_Base
class Spectral_Catalogue:
    def __init__(self, spectrum_arr: NDArray[Spectrum]) -> NoReturn:
        # check if any of the sources are the same
        orig_src_names = [spec.src_name for spec in spectrum_arr]
        unique_src_names = np.unique(orig_src_names)
        self.spectrum_arr = [
            [spec for spec in spectrum_arr if spec.src_name == src_name]
            for src_name in unique_src_names
        ]
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
    
    def __deepcopy__(self, memo):
        galfind_logger.debug(f"deepcopy({self.__class__.__name__})")
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            try:
                setattr(result, key, deepcopy(value, memo))
            except:
                galfind_logger.critical(
                    f"deepcopy({self.__class__.__name__}) {key}: {value} FAIL!"
                )
                breakpoint()
        return result

    @classmethod
    def from_DJA(
        cls,
        ra_range: Union[list, np.array, u.Quantity] = None,
        dec_range: Union[list, np.array, u.Quantity] = None,
        PID: Union[int, None] = None,
        z_cat_range: Union[list, np.array, None] = None,
        grating_filter: Union[str, None] = None,
        grade: int = 3,
        filename_arr: Optional[List[str]] = None,
        save: bool = True,
        z_from_cat: bool = False,
        version: str = "v2",
    ):
        if grating_filter is not None:
            assert grating_filter in NIRSpec.available_grating_filters
        assert version in ["v1", "v2", "v3", "v4_2"]
        # open and crop catalogue
        # DJA_cat = utils.read_catalog(config['Spectra']['DJA_CAT_PATH'], format = "ascii.ecsv")
        DJA_cat = Table.read(
            config["Spectra"]["DJA_CAT_PATH"].replace("v2", version)
        )
        if filename_arr is None:
            if ra_range is not None:
                assert len(ra_range) == 2
                if type(ra_range) in [list, np.array]:
                    assert ra_range[0].unit == ra_range[1].unit
                    ra_range = [ra_range[0].value, ra_range[1].value] * ra_range[
                        0
                    ].unit
                ra_range = sorted(ra_range.to(u.deg).value)
                DJA_cat = DJA_cat[
                    ((DJA_cat["ra"] > ra_range[0]) & (DJA_cat["ra"] < ra_range[1]))
                ]
            if dec_range is not None:
                assert len(dec_range) == 2
                if type(dec_range) in [list, np.array]:
                    assert dec_range[0].unit == dec_range[1].unit
                    dec_range = [
                        dec_range[0].value,
                        dec_range[1].value,
                    ] * dec_range[0].unit
                dec_range = sorted(dec_range.to(u.deg).value)
                DJA_cat = DJA_cat[
                    (
                        (DJA_cat["dec"] > dec_range[0])
                        & (DJA_cat["dec"] < dec_range[1])
                    )
                ]
            if grade is not None:
                DJA_cat = DJA_cat[DJA_cat["grade"] == grade]
            if grating_filter is not None:
                if "grating" in DJA_cat.colnames:
                    # TODO: Generalize this!
                    if version == "v4_2":
                        if grating_filter == "PRISM/CLEAR":
                            grating_filter = "PRISM_CLEAR"
                    DJA_cat = DJA_cat[
                        DJA_cat["grating"] == grating_filter.split("/")[0]
                    ]
                
                if "filter" in DJA_cat.colnames:
                    DJA_cat = DJA_cat[
                        DJA_cat["filter"] == grating_filter.split("/")[1]
                    ]
            if z_cat_range is not None:
                DJA_cat = DJA_cat[
                    (
                        (DJA_cat["z"] > z_cat_range[0])
                        & (DJA_cat["z"] < z_cat_range[1])
                    )
                ]
                z_from_cat = True
            if PID is not None:
                if "PID" in DJA_cat.colnames:
                    DJA_cat = DJA_cat[DJA_cat["PID"] == PID]
        else:
            mask = np.isin(np.array(DJA_cat["file"]), np.array(filename_arr))
            DJA_cat = DJA_cat[mask]
            # TODO: assertions that these follow the other rules

        if z_from_cat:
            return cls(
                [
                    Spectrum.from_DJA(
                        f"{config['Spectra']['DJA_WEB_DIR']}/{root}/{file}",
                        save = save,
                        version = version,
                        z = z,
                        root = root,
                        file = file,
                    )
                    for root, file, z in tqdm(
                        zip(DJA_cat["root"], DJA_cat["file"], DJA_cat["z"]),
                        total=len(DJA_cat),
                        desc=f"Loading DJA_{version} catalogue",
                        disable=galfind_logger.getEffectiveLevel() > logging.INFO
                    )
                ]
            )
        else:
            return cls(
                [
                    Spectrum.from_DJA(
                        f"{config['Spectra']['DJA_WEB_DIR']}/{root}/{file}",
                        save = save,
                        version = version,
                        root = root,
                        file = file,
                    )
                    for root, file in tqdm(
                        zip(DJA_cat["root"], DJA_cat["file"]),
                        total=len(DJA_cat),
                        desc=f"Loading DJA_{version} catalogue",
                        disable=galfind_logger.getEffectiveLevel() > logging.INFO
                    )
                ]
            )
    
    def plot(
        self: Self,
        src: str = "msaexp"
    ):
        for gal in tqdm(self, desc="Plotting spectra"):
            for spec in gal:
                spec.plot(src = src)

    # def crop_to_grating(self, name = "G395H/F290LP"):

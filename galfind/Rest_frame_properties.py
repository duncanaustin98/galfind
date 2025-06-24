
from __future__ import annotations

import astropy.units as u
import numpy as np
from abc import abstractmethod, ABC
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
import time
from scipy.stats import norm
import logging
from numba import njit
from joblib import Parallel, delayed, parallel_config
import itertools
from typing import TYPE_CHECKING, Dict, Any, List, Union, Tuple, Optional, NoReturn
if TYPE_CHECKING:
    from . import Multiple_Filter
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import galfind_logger, config, all_band_names, astropy_cosmo
from . import useful_funcs_austind as funcs
from .decorators import ignore_warnings
from . import Catalogue, Catalogue_Base, Galaxy, SED_code, Photometry_rest, PDF
from .Emission_lines import line_diagnostics, strong_optical_lines
from .Property_calculator import Property_Calculator
from .Dust_Attenuation import AUV_from_beta, Dust_Law, Calzetti00, M99

# Rest optical line property naming functions

# def get_rest_optical_flux_contam_label(
#     line_names: list, flux_contamination_params: dict
# ):
#     assert all(
#         line_name in line_diagnostics.keys() for line_name in line_names
#     )
#     assert type(flux_contamination_params) == dict
#     flux_cont_keys = flux_contamination_params.keys()
#     if "mu" in flux_cont_keys and "sigma" in flux_cont_keys:
#         return f"{line_names[0]}_cont_G({flux_contamination_params['mu']:.1f},{flux_contamination_params['sigma']:.1f})"  # _{'+'.join(line_names[1:])}"
#     elif "mu" in flux_cont_keys and "sigma" not in flux_cont_keys:
#         return f"{line_names[0]}_cont_{flux_contamination_params['mu']:.1f}"  # _{'+'.join(line_names[1:])}"
#     elif len(flux_contamination_params) == 0:
#         return "+".join(line_names)
#     else:
#         raise NotImplementedError

# def get_rest_optical_flux_contam_scaling(
#     flux_contamination_params: dict, iters: int
# ):
#     assert type(flux_contamination_params) == dict
#     flux_cont_keys = flux_contamination_params.keys()
#     if "mu" in flux_cont_keys and "sigma" in flux_cont_keys:
#         return np.random.normal(
#             1.0 - flux_contamination_params["mu"],
#             flux_contamination_params["sigma"],
#             iters,
#         )
#     elif "mu" in flux_cont_keys and "sigma" not in flux_cont_keys:
#         return 1.0 - flux_contamination_params["mu"]
#     elif len(flux_contamination_params) == 0:
#         return 1.0
#     else:
#         raise NotImplementedError

# def _get_wav_line_precision(self, line_name: str, dz: float):
#     assert line_name in line_diagnostics.keys()
#     wav_rest = line_diagnostics[line_name]["line_wav"]
#     dlambda = dz * wav_rest / (1.0 + self.z)
#     return dlambda

class Rest_Frame_Property_Calculator(Property_Calculator):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        pre_req_properties: List[Rest_Frame_Property_Calculator] = [],
        **global_kwargs
    ) -> None:
        #self.aper_diam = aper_diam
        if isinstance(SED_fit_label, SED_code):
            SED_fit_label = SED_fit_label.label
        self.SED_fit_label = SED_fit_label
        self.pre_req_properties = pre_req_properties
        self.global_kwargs = global_kwargs
        self._kwarg_assertions()
        super().__init__(aper_diam)

    def __call__(
        self: Self,
        object: Union[Type[Catalogue_Base], Galaxy, Photometry_rest],
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        n_jobs: int = 1,
    ) -> Optional[Union[Type[Catalogue_Base], Galaxy, Photometry_rest]]:
        # calculate pre-requisite properties first
        [rest_frame_property(object, n_chains, output = False, \
            overwrite = overwrite, n_jobs = n_jobs) for \
            rest_frame_property in self.pre_req_properties]
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
            obj = self._call_cat(object, n_chains, output, overwrite, n_jobs = n_jobs)
        elif isinstance(object, Galaxy):
            obj = self._call_gal(object, n_chains, output, overwrite)
        elif isinstance(object, Photometry_rest):
            obj = self._call_phot_rest(object, n_chains, output, overwrite)
        else:
            err_message = f"{object=} with {type(object)=} " + \
                f"not in [Catalogue, Galaxy, Photometry_rest]"
            galfind_logger.critical(err_message)
            raise TypeError(err_message)
        if output:
            return obj
    
    def _call_cat(
        self: Self,
        cat: Catalogue,
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        n_jobs: int = 1,
        dtype: np.dtype = np.float32,
    ) -> Optional[Catalogue]:
        assert isinstance(n_jobs, int), \
            galfind_logger.critical(
                f"{n_jobs=} with {type(n_jobs)=} != int"
            )
        try:
            save_dir = f"{config['PhotProperties']['PDF_SAVE_DIR']}/" + \
                f"{cat.version}/{cat.survey}/{cat.filterset.instrument_name}/" + \
                f"{self.aper_diam.to(u.arcsec).value:.2f}as" + \
                f"/{self.SED_fit_label}/{self.name}"
        except:
            breakpoint()
        if n_jobs <= 1:
            # update properties for each galaxy in the catalogue
            [self._call_gal(gal, n_chains = n_chains, output = False, \
                overwrite = overwrite, save_dir = save_dir) \
                for gal in tqdm(cat, total = len(cat), \
                desc = f"Calculating {self.name}",
                disable=galfind_logger.getEffectiveLevel() > logging.INFO)]
        else:
            # TODO: should be set when serializing the object
            for gal in tqdm(cat, total = len(cat), disable=galfind_logger.getEffectiveLevel() > logging.INFO):
                for label in gal.aper_phot[self.aper_diam].SED_results.keys():
                    try:
                        gal.aper_phot[self.aper_diam].flux = \
                            gal.aper_phot[self.aper_diam].flux.unmasked
                    except:
                        pass
                    try:
                        gal.aper_phot[self.aper_diam].flux_errs = \
                            gal.aper_phot[self.aper_diam].flux_errs.unmasked
                    except:
                        pass
                    try:
                        gal.aper_phot[self.aper_diam].SED_results[label].phot_rest.flux = \
                            gal.aper_phot[self.aper_diam].SED_results[label].phot_rest.flux.unmasked
                    except:
                        pass
                    try:
                        gal.aper_phot[self.aper_diam].SED_results[label].phot_rest.flux_errs = \
                            gal.aper_phot[self.aper_diam].SED_results[label].phot_rest.flux_errs.unmasked
                    except:
                        pass
            # multi-process with joblib
            # sort input params
            params_arr = [(self, gal, n_chains, overwrite, save_dir, dtype) for gal in cat]
            # run in parallel
            with funcs.tqdm_joblib(tqdm(desc = f"Calculating {self.name} for " + \
                f"{cat.survey} {cat.version} {cat.filterset.instrument_name}", \
                total = len(cat))) as progress_bar:
                    with parallel_config(backend='loky', n_jobs=n_jobs):
                        gals = Parallel()(delayed( \
                            self._call_gal_multi_process)(params) \
                            for params in params_arr
                        )
            cat.gals = gals
        if cat.cat_creator.crops == []:
            self._update_fits_cat(cat)
        if output:
            return cat
    
    @staticmethod
    def _call_gal_multi_process(params: Dict[str, Any]) -> NoReturn:
        self, gal, n_chains, overwrite, save_dir, dtype = params
        return self._call_gal(gal, n_chains = n_chains, output = True, \
            overwrite = overwrite, save_dir = save_dir, dtype = dtype)
    
    def _update_fits_cat(
        self: Self,
        cat: Catalogue,
    ) -> NoReturn:
        # TODO: generalize this funciton further
        # determine appropriate hdu and name to save properties as
        # TODO: generalize hdu name for non-EAZY SED fitting labels
        property_hdu = "_".join(self.SED_fit_label.split("_")[:-1])
        property_name = f"{self.name}_{self.aper_diam.to(u.arcsec).value:.2f}as"
        # open fits catalogue
        tab = cat.open_cat(hdu=property_hdu)
        if not property_name in tab.colnames:
            assert len(tab) == len(cat)
            # extract median property values from phot_rest.properties
            property_vals = [gal.aper_phot[self.aper_diam].SED_results \
                [self.SED_fit_label].phot_rest.properties[self.name].value if \
                not np.isnan(gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label]. \
                phot_rest.properties[self.name]) else np.nan for gal in cat]
            # calculate errors from PDFs storing chains
            property_l1 = [median_val - gal.aper_phot[self.aper_diam].SED_results \
                [self.SED_fit_label].phot_rest.property_PDFs[self.name]. \
                get_percentile(16.).value if gal.aper_phot[self.aper_diam].SED_results \
                [self.SED_fit_label].phot_rest.property_PDFs[self.name] is not None \
                else np.nan for gal, median_val in zip(cat, property_vals)]
            property_u1 = [gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label] \
                .phot_rest.property_PDFs[self.name].get_percentile(84.).value - median_val \
                if gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label]. \
                phot_rest.property_PDFs[self.name] is not None else np.nan \
                for gal, median_val in zip(cat, property_vals)]
            # update fits catalogue
            tab[property_name] = property_vals
            tab[f"{property_name}_l1"] = property_l1
            tab[f"{property_name}_u1"] = property_u1
            kwarg_names = np.unique([key for gal in cat for key in \
                gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label]. \
                phot_rest.property_kwargs[self.name].keys()])
            for kwarg_name in kwarg_names:
                tab[f"{property_name}_{kwarg_name}"] = [gal.aper_phot[self.aper_diam]. \
                    SED_results[self.SED_fit_label].phot_rest.property_kwargs[self.name] \
                    [kwarg_name] if kwarg_name in gal.aper_phot[self.aper_diam].SED_results \
                    [self.SED_fit_label].phot_rest.property_kwargs[self.name].keys() else np.nan for gal in cat]
            cat.write_hdu(tab, hdu=property_hdu)

    def _call_gal(
        self: Self,
        gal: Galaxy,
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        save_dir: str = "",
        dtype: np.dtype = np.float32,
    ) -> Optional[Galaxy]:
        # update the relevant Photometry_rest object stored in the Galaxy
        assert self.aper_diam in gal.aper_phot.keys(), \
            galfind_logger.critical(
                f"{self.aper_diam=} not in {gal.aper_phot.keys()}"
            )
        assert self.SED_fit_label in gal.aper_phot[self.aper_diam].SED_results.keys(), \
            galfind_logger.critical(
                f"{self.SED_fit_label=} not in " + \
                gal.aper_phot[self.aper_diam].SED_results.keys()
            )
        if save_dir != "":
            save_dir += "/"
        save_path = f"{save_dir}{gal.ID}.npy"
        self._call_phot_rest(gal.aper_phot[self.aper_diam]. \
            SED_results[self.SED_fit_label].phot_rest, n_chains = n_chains, 
            output = False, overwrite = overwrite, save_path = save_path, dtype = dtype)
        
        if output:
            return gal
    
    def _call_phot_rest(
        self: Self,
        phot_rest: Photometry_rest,
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        save_path: Optional[str] = None,
        save_scattered_fluxes: bool = False,
        dtype: np.dtype = np.float32,
    ) -> Optional[Photometry_rest]:
        property_name = self.name
        calculated = False
        # if any pre-requisite properties are NaN, set this property to NaN
        if property_name in phot_rest.properties.keys():
            properties_to_nan_check = [self] + self.pre_req_properties
        else:
            properties_to_nan_check = self.pre_req_properties

        if any(np.isnan(phot_rest.properties[property.name]) \
               for property in properties_to_nan_check):
            phot_rest.properties[property_name] = np.nan
            if n_chains > 1:
                phot_rest.property_errs[property_name] = np.nan
                phot_rest.property_PDFs[property_name] = None
                phot_rest.property_kwargs[property_name] = {}
        else:
            if n_chains <= 1:
                if property_name not in phot_rest.properties.keys() or overwrite:
                    galfind_logger.debug(
                        f"Calculating basic " + \
                        f"{property_name} for {repr(phot_rest)}"
                    )
                    self.obj_kwargs = self._calc_obj_kwargs(phot_rest)
                    if self._fail_criteria(phot_rest):
                        phot_rest.properties[property_name] = np.nan
                    else:
                        #breakpoint()
                        value = self._calculate(np.array([phot_rest.flux[ \
                            self.obj_kwargs["keep_indices"]].value]) \
                            * phot_rest.flux.unit, phot_rest)[0]
                        if value is None:
                            phot_rest.properties[property_name] = np.nan
                        else:
                            phot_rest.properties[property_name] = value
                        calculated = True
                else:
                    galfind_logger.debug(
                        f"Already calculated basic " + \
                        f"{property_name} for {repr(phot_rest)}"
                    )
            else:
                # if PDF does not already exist in the object 
                # but has been run before, load it if not wanting to overwrite
                if save_path is not None and Path(save_path).is_file():
                    if property_name not in phot_rest.property_PDFs.keys() \
                            and not overwrite:
                        PDF_obj = PDF.from_npy(save_path)
                        galfind_logger.debug(
                            f"Loading {len(PDF_obj)=} {property_name}" + \
                            f" PDF in {repr(phot_rest)}"
                        )
                        phot_rest.property_PDFs[property_name] = PDF_obj
                        phot_rest.properties[property_name] = PDF_obj.median
                        phot_rest.property_errs[property_name] = PDF_obj.errs
                        phot_rest.property_kwargs[property_name] = PDF_obj.kwargs
                    else:
                        galfind_logger.debug(
                            f"Already loaded {property_name} PDF in {repr(phot_rest)}"
                        )
                        if output:
                            return phot_rest
                        else:
                            return
                if property_name not in phot_rest.property_PDFs.keys() or overwrite:
                    n_new_chains = n_chains
                    galfind_logger.debug(
                        f"Creating {property_name} PDF in {repr(phot_rest)} from n={n_new_chains} chains"
                    )
                elif len(phot_rest.property_PDFs[property_name]) < n_chains:
                    n_new_chains = n_chains - len(phot_rest.property_PDFs[property_name])
                    galfind_logger.debug(
                        f"Adding n={n_new_chains} {property_name} chains to {repr(phot_rest)}"
                    )
                else: # len(phot_rest.property_PDFs[property_name]) >= n_chains
                    n_new_chains = 0
                    galfind_logger.debug(
                        f"Already calculated n={len(phot_rest.property_PDFs[property_name])}" + \
                        f" {property_name} chains to {repr(phot_rest)}"
                    )
                if n_new_chains > 0:
                    self.obj_kwargs = self._calc_obj_kwargs(phot_rest)
                    if self._fail_criteria(phot_rest):
                        phot_rest.property_PDFs[property_name] = None
                        phot_rest.properties[property_name] = np.nan
                        phot_rest.property_errs[property_name] = np.nan
                        phot_rest.property_kwargs[property_name] = {}
                    else: # phot_rest has not failed
                        galfind_logger.debug("Making PDF")
                        PDF_obj, scattered_fluxes = self._make_PDF(phot_rest, n_new_chains, dtype = dtype)
                        galfind_logger.debug("PDF made")
                        if PDF_obj is None:
                            phot_rest.property_PDFs[property_name] = None
                            phot_rest.properties[property_name] = np.nan
                            phot_rest.property_errs[property_name] = np.nan
                            phot_rest.property_kwargs[property_name] = {}
                        else:
                            if n_new_chains != n_chains:
                                if save_scattered_fluxes:
                                    # load old scattered fluxes
                                    old_scattered_fluxes = 0.
                                    scattered_fluxes = np.concatenate([old_scattered_fluxes, scattered_fluxes])
                                PDF_obj = phot_rest.property_PDFs[property_name] + PDF_obj
                            phot_rest.property_PDFs[property_name] = PDF_obj
                            phot_rest.properties[property_name] = PDF_obj.median
                            phot_rest.property_errs[property_name] = PDF_obj.errs
                            # update saved PDF
                            if save_path is not None:
                                funcs.make_dirs(save_path)
                                if phot_rest.property_PDFs[property_name] is not None and save_path is not None:
                                    if save_scattered_fluxes:
                                        np.save(save_path.replace(".npy", "_scattered_fluxes.npy"), scattered_fluxes.value)
                                        galfind_logger.debug("Scattered fluxes saved")
                                    phot_rest.property_PDFs[property_name].save(save_path)
                                    galfind_logger.debug(f"PDF saved for {save_path.split('/')[-1].replace('.npy', '')}")
                            calculated = True
            if calculated:
                phot_rest.property_kwargs[property_name] = self._get_output_kwargs(phot_rest)
        if output:
            return phot_rest
    
    def _make_PDF(
        self: Self,
        phot_rest: Photometry_rest,
        n_chains: int,
        dtype: np.dtype = np.float32,
    ) -> Tuple[PDF, u.Quantity]:
        # ensure the type is a float
        assert "float" in dtype.__name__, \
            galfind_logger(
                f"{dtype=} is not a float type"
            )
        #try:
        # scatter relevant photometric data points n_chains times
        if "keep_indices" in self.obj_kwargs.keys():
            cropped_phot_rest = phot_rest[self.obj_kwargs["keep_indices"]]
        else:
            cropped_phot_rest = deepcopy(phot_rest)
        scattered_fluxes = cropped_phot_rest.scatter_fluxes(n_chains)
        # calculate chain
        galfind_logger.debug(f"Calculating {self.name} chains")
        vals = self._calculate(scattered_fluxes, phot_rest)

        if vals is not None:
            # increase the floating point precision of saved array if required
            while any(val.value > np.finfo(dtype).max or val.value < np.finfo(dtype).min for val in vals):
                new_dtype_precision = int(dtype.__name__.replace("float", "")) * 2
                dtype = getattr(np, f"float{new_dtype_precision}")
            # update datatype of vals
            vals = vals.astype(dtype)
        galfind_logger.debug(f"{self.name} chains calculated")
        # construct PDF object
        try:
            if vals is None:
                PDF_obj = None
            else:
                PDF_obj = PDF.from_1D_arr(self.name, vals, kwargs = self._get_output_kwargs(phot_rest))
        except:
            breakpoint()
        return PDF_obj, scattered_fluxes

    def extract_vals(
        self: Self, 
        object: Union[Type[Catalogue_Base], Galaxy, Photometry_rest],
    ) -> Union[u.Quantity, u.Magnitude, u.Dex]:
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
            cat_vals = [gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label].phot_rest.properties[self.name] for gal in object]
            cat_vals_no_nans = [val for val in cat_vals if not np.isnan(val)]
            if not all(isinstance(val, float) for val in cat_vals_no_nans):
                assert all(val.unit == cat_vals[0].unit for val in cat_vals_no_nans), \
                    galfind_logger.critical(f"Units of {self.name} in {object} are not consistent")
                cat_vals = np.array([val.value if not np.isnan(val) else val for val in cat_vals]) * cat_vals[0].unit
            else:
                cat_vals = np.array(cat_vals)
            return cat_vals
        elif isinstance(object, Galaxy):
            return object.aper_phot[self.aper_diam].SED_results[self.SED_fit_label].phot_rest.properties[self.name]
        elif isinstance(object, Photometry_rest):
            return object.properties[self.name]
        else:
            err_message = f"{object=} with {type(object)=} " + \
                f"not in [{', '.join(Catalogue_Base.__subclasses__())}, Galaxy, Photometry_rest]"
            galfind_logger.critical(err_message)
            raise TypeError(err_message)

    # TODO: Propagate from parent class
    def extract_errs(
        self: Self, 
        object: Union[Type[Catalogue_Base], Galaxy, Photometry_rest],
    ) -> Union[u.Quantity, u.Magnitude, u.Dex]:
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
            cat_errs = [gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label].phot_rest.property_errs[self.name] for gal in object]
            if all(isinstance(val, tuple([u.Quantity, u.Magnitude, u.Dex])) for val in cat_errs):
                assert all(val.unit == cat_errs[0].unit for val in cat_errs), \
                    galfind_logger.critical(f"Units of {self.name} in {object} are not consistent")
                cat_errs = np.array([val.value for val in cat_errs]) * cat_errs[0].unit
            else:
                cat_errs = np.array(cat_errs)
            return cat_errs 
        elif isinstance(object, Galaxy):
            return object.aper_phot[self.aper_diam].SED_results[self.SED_fit_label].phot_rest.property_errs[self.name]
        elif isinstance(object, Photometry_rest):
            return object.property_errs[self.name]
        else:
            err_message = f"{object=} with {type(object)=} " + \
                f"not in [Catalogue, Galaxy, Photometry_rest]"
            galfind_logger.critical(err_message)
            raise TypeError(err_message)

    def extract_PDFs(
        self: Self,
        object: Union[Type[Catalogue_Base], Galaxy, Photometry_rest],
    ) -> Union[Type[PDF], List[Type[PDF]]]:
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
            return [gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label].phot_rest.property_PDFs[self.name] for gal in object]
        elif isinstance(object, Galaxy):
            return object.aper_phot[self.aper_diam].SED_results[self.SED_fit_label].phot_rest.property_PDFs[self.name]
        elif isinstance(object, Photometry_rest):
            return object.property_PDFs[self.name]
        else:
            err_message = f"{object=} with {type(object)=} " + \
                f"not in [{', '.join(Catalogue_Base.__subclasses__())}, Galaxy, Photometry_rest]"
            galfind_logger.critical(err_message)
            raise TypeError(err_message)

    @abstractmethod
    def _kwarg_assertions(self: Self) -> None:
        pass

    @abstractmethod
    def _calc_obj_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        pass

    @abstractmethod
    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        pass

    @abstractmethod
    def _get_output_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        pass


class beta_fit:
    def __init__(
        self: Self, 
        z: float, 
        filterset: Multiple_Filter
    ) -> NoReturn:
        self.filterset = filterset
        self.wavelength_rest = {}
        self.mid_wav_rest = {}
        self.transmission = {}
        self.norm = {}
        max_length = np.max([len(filt.wav) for filt in filterset])
        for filt in filterset:
            wav_rest = np.array(
                funcs.convert_wav_units(filt.wav, u.AA).value / (1.0 + z)
            )
            trans = np.array(filt.trans)
            length = len(wav_rest)
            if length != max_length:
                wav_rest = np.concatenate([wav_rest, np.full(max_length - length, wav_rest[-1])])
                trans = np.concatenate([trans, np.full(max_length - length, trans[-1])])
            self.mid_wav_rest[filt.band_name] = filt.WavelengthCen.to(u.AA).value
            self.wavelength_rest[filt.band_name] = wav_rest
            self.transmission[filt.band_name] = trans
            self.norm[filt.band_name] = np.trapz(
                self.transmission[filt.band_name],
                x=self.wavelength_rest[filt.band_name],
            )

    def __call__(self, _, A, beta):
        return self.get_fluxes(
            A,
            beta,
            self.wavelength_rest,
            self.transmission,
            self.norm,
            self.filterset.band_names
        )

@njit
def get_fluxes(wav_rest, A, beta, trans, norm):
    return np.array(
        [
            np.trapz(
                (10**A)
                * (wav_rest[i] ** beta)
                * trans[i],
                x=wav_rest[i],
            )
            / norm[i]
            for i in range(len(wav_rest))
        ]
    )

@njit
def fit_beta_gradient_descent(wav_rest, mid_wav_rest, flux, trans, norm, init_A, init_beta, learning_rate=1e-6, max_iter=1000, tol=1e-6):
    """
    Perform gradient descent to minimize the residual sum of squares.
    """
    A = init_A
    beta = init_beta
    for i in range(max_iter):
        model_flux = get_fluxes(wav_rest, A, beta, trans, norm)
        residuals = flux - model_flux
        n = len(residuals)
        # Compute sums for least squares
        sum_res = 0.0
        sum_wav_res = 0.0
        for i in range(n):
            sum_res += residuals[i]
            sum_wav_res += mid_wav_rest[i] * residuals[i]
        #print(residuals)
        #print(sum_res, sum_wav_res)
        # Compute gradients
        grad_A = -2 * sum_res
        grad_beta = -2 * sum_wav_res

        # Update parameters
        A -= learning_rate * grad_A
        beta -= learning_rate * grad_beta
        
        # Check for convergence
        if np.sqrt(grad_A ** 2 + grad_beta ** 2) < tol:
            break
            
    return beta #A, beta, i  # Return optimized parameters and iterations taken

def rest_UV_wavs_name(rest_UV_wav_lims):
    rest_UV_wav_lims = [
        int(
            funcs.convert_wav_units(
                rest_UV_wav_lim * rest_UV_wav_lims.unit, u.AA
            ).value
        )
        for rest_UV_wav_lim in rest_UV_wav_lims.value
    ]
    return f"{str(rest_UV_wav_lims).replace(' ', '')}AA"


class UV_Beta_Calculator(Rest_Frame_Property_Calculator):
    
    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
    ) -> NoReturn:
        global_kwargs = {"rest_UV_wav_lims": rest_UV_wav_lims}
        super().__init__(aper_diam, SED_fit_label, [], **global_kwargs)

    @property
    def name(self: Self) -> str:
        return f"beta_{rest_UV_wavs_name(self.global_kwargs['rest_UV_wav_lims'])}"

    @property
    def plot_name(self: Self) -> str:
        return r"$\beta_{\mathrm{UV}}$"
    
    def _kwarg_assertions(self: Self) -> None:
        assert u.get_physical_type(self.global_kwargs["rest_UV_wav_lims"]) == "length"
        assert len(self.global_kwargs["rest_UV_wav_lims"]) == 2
        assert self.global_kwargs["rest_UV_wav_lims"][0] < \
            self.global_kwargs["rest_UV_wav_lims"][1]
        assert self.global_kwargs["rest_UV_wav_lims"][0] > 1_216.0 * u.AA
        assert self.global_kwargs["rest_UV_wav_lims"][1] < 3_646.0 * u.AA
    
    def _calc_obj_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        # determine bands that fall within rest frame UV wavelength limits
        rest_frame_UV_indices = [
            i for i, filt in enumerate(phot_rest.filterset)
            if filt.WavelengthLower50 > self.global_kwargs["rest_UV_wav_lims"][0] * (1.0 + phot_rest.z.value)
            and filt.WavelengthUpper50 < self.global_kwargs["rest_UV_wav_lims"][1] * (1.0 + phot_rest.z.value)
        ]
        if len(rest_frame_UV_indices) < 2:
            failure = True
        else:
            rest_UV_band_wavs = np.array(
                [
                    funcs.convert_wav_units(filt.WavelengthCen \
                    / (1.0 + phot_rest.z.value), u.AA).value for filt \
                    in phot_rest.filterset[rest_frame_UV_indices]
                ]
            ) * u.AA
            phot_rest_UV = phot_rest[rest_frame_UV_indices]
            rest_UV_SNRs = phot_rest_UV.flux / phot_rest_UV.flux_errs
            if any(np.isnan(SNR) for SNR in rest_UV_SNRs):
                failure = True
            else:
                failure = False
                # determine what percentage of scatters fall into the negative flux region
                negative_flux_pc = (1. - np.prod([1. - norm.cdf(0., loc=mu, scale=std) for mu, std in \
                    zip(phot_rest_UV.flux.value, phot_rest_UV.flux_errs.value)])) * 100.0
        if failure:
            rest_frame_UV_indices = None
            rest_UV_band_wavs = None
            rest_UV_SNRs = None
            negative_flux_pc = None
        
        return {
            "keep_indices": rest_frame_UV_indices,
            "rest_UV_band_wavs": rest_UV_band_wavs,
            "rest_UV_SNRs": rest_UV_SNRs,
            "negative_flux_pc": negative_flux_pc
        }

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        if any(value is None for value in self.obj_kwargs.values()) or \
                self.obj_kwargs["negative_flux_pc"] > 99.0:
            return True
        else:
            return False
    
    @ignore_warnings
    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # appropriately convert flux units
        fluxes_arr = np.log10(funcs.convert_mag_units(
            self.obj_kwargs["rest_UV_band_wavs"],
            fluxes_arr, u.erg / (u.s * u.AA * u.cm**2),
            ).value)
        beta_arr = np.array([funcs.linear_fit(np.log10(self.obj_kwargs \
            ["rest_UV_band_wavs"].value, dtype=np.float64), \
            fluxes)[0] for fluxes in fluxes_arr]) * u.dimensionless_unscaled
        return beta_arr
    
    def _get_output_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {
            "rest_UV_band_names": "+".join(
                np.array(phot_rest.filterset.band_names)
                [self.obj_kwargs["keep_indices"]]),
            "n_UV_bands": len(self.obj_kwargs["keep_indices"]),
            "negative_flux_pc": self.obj_kwargs["negative_flux_pc"]
        }
    
    # save scattered fluxes to retain access to fit amplitudes
    def _call_phot_rest(
        self: Self,
        phot_rest: Photometry_rest,
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        save_path: Optional[str] = None,
        dtype: np.dtype = np.float32,
    ) -> Optional[Photometry_rest]:
        return super()._call_phot_rest(
            phot_rest,
            n_chains,
            output,
            overwrite,
            save_path,
            save_scattered_fluxes = True,
            dtype = dtype,
        )


class UV_Dust_Attenuation_Calculator(Rest_Frame_Property_Calculator):
    
    def __init__(
        self: Self,
        aper_diam: u.Quantity, 
        SED_fit_label: Union[str, Type[SED_code]], 
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        beta_dust_conv: Union[str, Type[AUV_from_beta]] = M99,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        keep_valid: bool = True,
    ) -> NoReturn:
        pre_req_properties = [UV_Beta_Calculator(aper_diam, SED_fit_label, rest_UV_wav_lims)]
        if isinstance(beta_dust_conv, str):
            beta_dust_conv = [beta_dust_conv_cls() for beta_dust_conv_cls \
                in AUV_from_beta.__subclasses__() \
                if beta_dust_conv_cls.__name__ == beta_dust_conv][0]
        elif not isinstance(beta_dust_conv, AUV_from_beta):
            beta_dust_conv = beta_dust_conv()
        global_kwargs = {"ref_wav": ref_wav, "beta_dust_conv": beta_dust_conv, "keep_valid": keep_valid}
        super().__init__(aper_diam, SED_fit_label, pre_req_properties, **global_kwargs)

    @property
    def name(self: Self) -> str:
        label = f"A{self.global_kwargs['ref_wav'].to(u.AA).value:.0f}" + \
            f"_{self.global_kwargs['beta_dust_conv'].__class__.__name__}" + \
            f"_{rest_UV_wavs_name(self.pre_req_properties[0].global_kwargs['rest_UV_wav_lims'])}"
        if self.global_kwargs["keep_valid"]:
            label += "_A>0"
        return label
    
    @property
    def plot_name(self: Self) -> str:
        return r"$A_{{}}$".format(int(self.global_kwargs['ref_wav'].to(u.AA).value))
    
    def _kwarg_assertions(self: Self) -> None:
        assert u.get_physical_type(self.global_kwargs["ref_wav"]) == "length"
        assert self.global_kwargs["ref_wav"] > self.pre_req_properties[0].global_kwargs["rest_UV_wav_lims"][0]
        assert self.global_kwargs["ref_wav"] < self.pre_req_properties[0].global_kwargs["rest_UV_wav_lims"][1]
        assert self.global_kwargs["beta_dust_conv"].__class__ in AUV_from_beta.__subclasses__()
        assert isinstance(self.global_kwargs["keep_valid"], bool)
    
    def _calc_obj_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        # always pass
        return False
    
    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # calculate beta
        if len(fluxes_arr) > 1:
            beta_arr = phot_rest.property_PDFs[self.pre_req_properties[0].name].input_arr
            assert len(fluxes_arr) == len(beta_arr)
        else:
            beta_arr = phot_rest.properties[self.pre_req_properties[0].name]
        # calculate A_UV
        A_UV_arr = self.global_kwargs["beta_dust_conv"](beta_arr)
        # limit to A_UV > 0
        if self.global_kwargs["keep_valid"]:
            A_UV_arr[A_UV_arr < 0.0] = 0.0
        return A_UV_arr
    
    def _get_output_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}
    

class Fesc_From_Beta_Calculator(Rest_Frame_Property_Calculator):
    
    def __init__(
        self: Self, 
        aper_diam: u.Quantity, 
        SED_fit_label: Union[str, Type[SED_code]], 
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        fesc_conv: str = "Chisholm22",
        keep_valid: bool = False,
    ) -> NoReturn:
        pre_req_properties = [UV_Beta_Calculator(aper_diam, SED_fit_label, rest_UV_wav_lims)]
        global_kwargs = {"fesc_conv": fesc_conv, "keep_valid": keep_valid}
        super().__init__(aper_diam, SED_fit_label, pre_req_properties, **global_kwargs)

    @property
    def name(self: Self) -> str:
        #if isinstance(self.global_kwargs["fesc_conv"], str):
        label = f"fesc={self.global_kwargs['fesc_conv']}_" + \
            rest_UV_wavs_name(self.pre_req_properties[0].global_kwargs["rest_UV_wav_lims"])
        #else: # float
        #    label = f"fesc={self.global_kwargs['fesc_conv']:.2f}"
        if self.global_kwargs["keep_valid"]:
            label += "_0<fesc<1"
        return label

    @property
    def plot_name(self: Self) -> str:
        return r"$f_{\mathrm{esc}}$" # type of fesc here too
    
    def _kwarg_assertions(self: Self) -> None:
        #if isinstance(self.global_kwargs["fesc_conv"], str):
        assert self.global_kwargs["fesc_conv"] in funcs.fesc_from_beta_conversions.keys()
        # elif isinstance(self.global_kwargs["fesc_conv"], float):
        #     assert self.global_kwargs["fesc_conv"] >= 0.0
        #     assert self.global_kwargs["fesc_conv"] <= 1.0
        # else:
        #     raise ValueError("fesc_conv must be a string or float")
        assert isinstance(self.global_kwargs["keep_valid"], bool)
    
    def _calc_obj_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        # always pass
        return False
    
    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # calculate beta
        if len(fluxes_arr) > 1:
            beta_arr = phot_rest.property_PDFs[self.pre_req_properties[0].name].input_arr
            assert len(fluxes_arr) == len(beta_arr)
        else:
            beta_arr = phot_rest.properties[self.pre_req_properties[0].name]
        #if isinstance(self.global_kwargs["fesc_conv"], str):
        fesc_arr = funcs.fesc_from_beta_conversions[self.global_kwargs["fesc_conv"]](beta_arr)
        #else:
        #    fesc_arr = np.full_like(beta_arr, self.global_kwargs["fesc_conv"])
        if self.global_kwargs["keep_valid"]:
            fesc_arr[fesc_arr < 0.0] = 0.0
            fesc_arr[fesc_arr > 1.0] = 1.0
        return fesc_arr
    
    def _get_output_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}


class mUV_Calculator(Rest_Frame_Property_Calculator):

    def __init__(
        self: Self, 
        aper_diam: u.Quantity, 
        SED_fit_label: Union[str, Type[SED_code]], 
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        top_hat_width: u.Quantity = 100.0 * u.AA,
        resolution: u.Quantity = 1.0 * u.AA,
        ext_src_corrs: Optional[str] = "UV",
        ext_src_uplim: Optional[Union[int, float]] = 10.0,
    ) -> NoReturn:
        pre_req_properties = [UV_Beta_Calculator(aper_diam, SED_fit_label, rest_UV_wav_lims)]
        global_kwargs = {
            "ref_wav": ref_wav,
            "top_hat_width": top_hat_width,
            "resolution": resolution,
            "ext_src_corrs": ext_src_corrs,
            "ext_src_uplim": ext_src_uplim,
        }
        super().__init__(aper_diam, SED_fit_label, pre_req_properties, **global_kwargs)

    @property
    def name(self: Self) -> str:
        ext_src_label = f"_extsrc_{self.global_kwargs['ext_src_corrs']}" \
            if self.global_kwargs["ext_src_corrs"] is not None else ""
        ext_src_lim_label = f"<{self.global_kwargs['ext_src_uplim']:.0f}" if \
            self.global_kwargs["ext_src_uplim"] is not None and \
            self.global_kwargs["ext_src_corrs"] is not None else ""
        return f"m{self.global_kwargs['ref_wav'].to(u.AA).value:.0f}_" + \
            rest_UV_wavs_name(self.pre_req_properties[0].global_kwargs \
            ["rest_UV_wav_lims"]) + ext_src_label + ext_src_lim_label

    @property
    def plot_name(self: Self) -> str:
        return r"$m_{\mathrm{UV}}$"
    
    def _kwarg_assertions(self: Self) -> None:
        assert all(u.get_physical_type(self.global_kwargs[name]) == "length" \
            for name in ["ref_wav", "top_hat_width", "resolution"])
        assert self.global_kwargs["ref_wav"] > self.pre_req_properties[0].global_kwargs["rest_UV_wav_lims"][0]
        assert self.global_kwargs["ref_wav"] < self.pre_req_properties[0].global_kwargs["rest_UV_wav_lims"][1]
        assert self.global_kwargs["top_hat_width"] > 0.0 * u.AA
        assert self.global_kwargs["resolution"] > 0.0 * u.AA
        if self.global_kwargs["ext_src_corrs"] is not None:
            assert self.global_kwargs["ext_src_corrs"] in ["UV"] + all_band_names
        if self.global_kwargs["ext_src_uplim"] is not None:
            assert isinstance(self.global_kwargs["ext_src_uplim"], (int, float))
            assert self.global_kwargs["ext_src_uplim"] > 0.0
    
    def _calc_obj_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        # always pass
        return False
    
    @ignore_warnings
    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # This doesn't technically require the array of scattered fluxes as input!
        if len(fluxes_arr) > 1:
            save_path = phot_rest.property_PDFs[self.pre_req_properties[0].name]. \
                save_path.replace(".npy", "_scattered_fluxes.npy")
            # load scattered fluxes
            scattered_fluxes = np.load(save_path) * u.Jy
            assert len(fluxes_arr) == len(scattered_fluxes)
            fluxes_arr = scattered_fluxes
            beta_arr = phot_rest.property_PDFs[self.pre_req_properties[0].name].input_arr
        else: # fluxes are unscattered
            beta_arr = phot_rest.properties[self.pre_req_properties[0].name]

        band_wavs = self.pre_req_properties[0]. \
            _calc_obj_kwargs(phot_rest)["rest_UV_band_wavs"]
        # appropriately convert flux units to rest frame
        fluxes_arr = np.log10(funcs.convert_mag_units(band_wavs, \
            fluxes_arr, u.erg / (u.s * u.AA * u.cm ** 2)).value)
        # calculate fit amplitudes
        amplitude_arr = np.array([funcs.linear_fit(np.log10( \
            band_wavs.value, dtype=np.float64), \
            fluxes)[1] for fluxes in fluxes_arr]) \
            * u.erg / (u.s * u.AA * u.cm ** 2)
        assert len(amplitude_arr) == len(beta_arr)

        # re-create linear fit(s) to calculate mUV's
        rest_wavelengths = funcs.convert_wav_units(
            np.linspace(
                self.global_kwargs["ref_wav"] - self.global_kwargs["top_hat_width"] / 2,
                self.global_kwargs["ref_wav"] + self.global_kwargs["top_hat_width"] / 2,
                int(
                    np.round(
                        (self.global_kwargs["top_hat_width"] / self.global_kwargs["resolution"])
                        .to(u.dimensionless_unscaled).value, 0,
                    )
                ),
            ), u.AA)
        mUV_arr = np.median( \
            funcs.convert_mag_units(rest_wavelengths, \
            10 ** (np.full((len(beta_arr), len(rest_wavelengths)), \
            np.log10(rest_wavelengths.value)) * beta_arr[:, np.newaxis].value \
            + amplitude_arr[:, np.newaxis].value) * u.erg / (u.s * u.AA * u.cm**2), \
            u.ABmag), axis = 1)
        # TODO: speed up implementation of extended source corrections
        if self.global_kwargs["ext_src_corrs"] is not None:
            if self.global_kwargs["ext_src_corrs"] == "UV":
                # calculate band nearest to the rest frame UV reference wavelength
                band_wavs = [filt.WavelengthCen.to(u.AA).value \
                    for filt in phot_rest.filterset] * u.AA / (1. + phot_rest.z.value)
                ref_band = phot_rest.filterset.band_names[np.argmin(np.abs( \
                    band_wavs - self.global_kwargs["ref_wav"]))]
                ext_src_corr = phot_rest.ext_src_corrs[ref_band]
            else: # band given
                ext_src_corr = phot_rest.ext_src_corrs[self.global_kwargs["ext_src_corrs"]]
            # apply limit to extended source correction
            if self.global_kwargs["ext_src_uplim"] is not None:
                if ext_src_corr > self.global_kwargs["ext_src_uplim"]:
                    ext_src_corr = self.global_kwargs["ext_src_uplim"]
            if ext_src_corr < 1.0:
                ext_src_corr = 1.0
            # apply extended source corrections
            mUV_arr = (mUV_arr.value + funcs.flux_to_mag_ratio(ext_src_corr)) * u.ABmag
        return mUV_arr
    
    def _get_output_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}

    def _call_cat(
        self: Self,
        cat: Catalogue,
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        n_jobs: int = 1,
    ) -> Optional[Catalogue]:
        if self.global_kwargs["ext_src_corrs"]:
            # load extended source corrections
            cat.load_sextractor_ext_src_corrs()
        return super()._call_cat(cat, n_chains, output, overwrite, n_jobs)
    
    def _call_phot_rest(
        self: Self,
        phot_rest: Photometry_rest,
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        save_path: Optional[str] = None,
        save_scattered_fluxes: bool = False,
        dtype: np.dtype = np.float32,
    ) -> Optional[Photometry_rest]:
        if self.global_kwargs["ext_src_corrs"]:
            # assert that extended source corrections have been loaded
            assert hasattr(phot_rest, "ext_src_corrs"), \
                galfind_logger.critical(
                    "Extended source corrections must be pre-loaded!"
                )
        return super()._call_phot_rest(
            phot_rest, 
            n_chains, 
            output, 
            overwrite, 
            save_path, 
            save_scattered_fluxes,
            dtype,
        )

class MUV_Calculator(Rest_Frame_Property_Calculator):

    def __init__(
        self: Self, 
        aper_diam: u.Quantity, 
        SED_fit_label: Union[str, Type[SED_code]], 
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        top_hat_width: u.Quantity = 100.0 * u.AA,
        resolution: u.Quantity = 1.0 * u.AA,
        ext_src_corrs: Optional[str] = "UV",
        ext_src_uplim: Optional[Union[int, float]] = 10.0,
    ) -> NoReturn:
        mUV_calculator = mUV_Calculator(
            aper_diam, 
            SED_fit_label, 
            rest_UV_wav_lims, 
            ref_wav,
            top_hat_width,
            resolution,
            ext_src_corrs,
            ext_src_uplim,
        )
        super().__init__(aper_diam, SED_fit_label, [mUV_calculator])

    @property
    def name(self: Self) -> str:
        ext_src_label = f"_extsrc_{self.pre_req_properties[0].global_kwargs['ext_src_corrs']}" \
            if self.pre_req_properties[0].global_kwargs["ext_src_corrs"] is not None else ""
        ext_src_lim_label = f"<{self.pre_req_properties[0].global_kwargs['ext_src_uplim']:.0f}" if \
            self.pre_req_properties[0].global_kwargs["ext_src_uplim"] is not None and \
            self.pre_req_properties[0].global_kwargs["ext_src_corrs"] is not None else ""
        return f"M{self.pre_req_properties[0].global_kwargs['ref_wav'].to(u.AA).value:.0f}_" + \
            rest_UV_wavs_name(self.pre_req_properties[0].pre_req_properties[0]. \
            global_kwargs["rest_UV_wav_lims"]) + ext_src_label + ext_src_lim_label
    
    @property
    def plot_name(self: Self) -> str:
        return r"$M_{\mathrm{UV}}$"

    def _kwarg_assertions(self: Self) -> NoReturn:
        pass
    
    def _calc_obj_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        # always pass
        return False
    
    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # load mUVs
        if len(fluxes_arr) > 1:
            mUV_arr = phot_rest.property_PDFs[self.pre_req_properties[0].name].input_arr
        else:
            mUV_arr = phot_rest.properties[self.pre_req_properties[0].name]
        # calculate M_UV from m_UV
        d_L = astropy_cosmo.luminosity_distance(phot_rest.z.value).to(u.pc).value
        return (mUV_arr.value \
            - 5.0 * np.log10(d_L / 10.0) \
            + 2.5 * np.log10(1.0 + phot_rest.z.value)
        ) * u.ABmag
    
    def _get_output_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}
    

class LUV_Calculator(Rest_Frame_Property_Calculator):

    def __init__(
        self: Self, 
        aper_diam: u.Quantity, 
        SED_fit_label: Union[str, Type[SED_code]],
        #frame: str = "obs",
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        beta_dust_conv: Optional[Union[str, Type[AUV_from_beta]]] = M99,
        top_hat_width: u.Quantity = 100.0 * u.AA,
        resolution: u.Quantity = 1.0 * u.AA,
        ext_src_corrs: Optional[str] = "UV",
        ext_src_uplim: Optional[Union[int, float]] = 10.0,
    ) -> NoReturn:
        mUV_calculator = mUV_Calculator(
            aper_diam, 
            SED_fit_label, 
            rest_UV_wav_lims, 
            ref_wav,
            top_hat_width,
            resolution,
            ext_src_corrs,
            ext_src_uplim,
        )
        pre_req_properties = [mUV_calculator]
        if beta_dust_conv is None:
            self.dust_calculator = None
        else:
            if isinstance(beta_dust_conv, str):
                beta_dust_conv = [beta_dust_conv_cls() for beta_dust_conv_cls \
                    in AUV_from_beta.__subclasses__() \
                    if beta_dust_conv_cls.__name__ == beta_dust_conv][0]
            elif not isinstance(beta_dust_conv, AUV_from_beta):
                beta_dust_conv = beta_dust_conv()
            self.dust_calculator = UV_Dust_Attenuation_Calculator( \
                aper_diam, 
                SED_fit_label, 
                rest_UV_wav_lims,
                beta_dust_conv,
                ref_wav,
                keep_valid = True
            )
            pre_req_properties.append(self.dust_calculator)
        global_kwargs = {} #"frame": frame}
        super().__init__(aper_diam, SED_fit_label, pre_req_properties, **global_kwargs)

    @property
    def name(self: Self) -> str:
        if self.dust_calculator is not None:
            dust_label = "_" + "_".join(self.dust_calculator.name.split("_")[1:2]) + "dust"
        else:
            dust_label = ""
        rest_wavs_label = rest_UV_wavs_name(self.pre_req_properties[0]. \
            pre_req_properties[0].global_kwargs["rest_UV_wav_lims"])
        ext_src_label = f"_extsrc_{self.pre_req_properties[0].global_kwargs['ext_src_corrs']}" \
            if self.pre_req_properties[0].global_kwargs["ext_src_corrs"] is not None else ""
        ext_src_lim_label = f"<{self.pre_req_properties[0].global_kwargs['ext_src_uplim']:.0f}" if \
            self.pre_req_properties[0].global_kwargs["ext_src_uplim"] is not None and \
            self.pre_req_properties[0].global_kwargs["ext_src_corrs"] is not None else ""
        # {self.global_kwargs['frame']}
        return f"L{self.pre_req_properties[0].global_kwargs['ref_wav'].to(u.AA).value:.0f}" + \
            f"{dust_label}_{rest_wavs_label}{ext_src_label}{ext_src_lim_label}"
    
    @property
    def plot_name(self: Self) -> str:
        return r"$L_{\mathrm{UV}}$" # frame and units here too

    def _kwarg_assertions(self: Self) -> NoReturn:
        pass
        #assert self.global_kwargs["frame"] in ["rest", "obs"]
    
    def _calc_obj_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        # always pass
        return False
    
    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # load mUVs
        if len(fluxes_arr) > 1:
            mUV_arr = phot_rest.property_PDFs[self.pre_req_properties[0].name].input_arr
        else:
            mUV_arr = phot_rest.properties[self.pre_req_properties[0].name]
        # convert mUVs to LUVs
        # if self.global_kwargs["frame"] == "rest":
        #     z = 0.0
        #     wavs = np.full(len(fluxes_arr), \
        #         self.pre_req_properties[0].global_kwargs["ref_wav"])
        # else: # frame == "obs"
        # use observed frame wavelengths
        wavs = np.full(len(fluxes_arr), self.pre_req_properties \
            [0].global_kwargs["ref_wav"] * (1.0 + phot_rest.z.value))
        LUV_arr = funcs.flux_to_luminosity(mUV_arr, wavs, phot_rest.z.value)
        
        # extract dust chains/value if required
        if self.dust_calculator is not None:
            if len(fluxes_arr) > 1:
                AUV_arr = phot_rest.property_PDFs[self.dust_calculator.name].input_arr
                assert len(fluxes_arr) == len(AUV_arr)
            else:
                AUV_arr = phot_rest.properties[self.dust_calculator.name]
            LUV_arr = funcs.dust_correct(LUV_arr, AUV_arr)
        # output luminosities
        return LUV_arr
    
    def _get_output_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}


class SFR_UV_Calculator(Rest_Frame_Property_Calculator):

    def __init__(
        self: Self, 
        aper_diam: u.Quantity, 
        SED_fit_label: Union[str, Type[SED_code]],
        rest_UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        ref_wav: u.Quantity = 1_500.0 * u.AA,
        beta_dust_conv: Optional[Union[str, Type[AUV_from_beta]]] = M99,
        SFR_conv: str = "MD14",
        top_hat_width: u.Quantity = 100.0 * u.AA,
        resolution: u.Quantity = 1.0 * u.AA,
        ext_src_corrs: Optional[str] = "UV",
        ext_src_uplim: Optional[Union[int, float]] = 10.0,
    ) -> NoReturn:
        LUV_calculator = LUV_Calculator(
            aper_diam, 
            SED_fit_label, 
            # "obs", 
            rest_UV_wav_lims, 
            ref_wav, 
            beta_dust_conv, 
            top_hat_width, 
            resolution,
            ext_src_corrs,
            ext_src_uplim
        )
        pre_req_properties = [LUV_calculator]
        global_kwargs = {"SFR_conv": SFR_conv}
        super().__init__(aper_diam, SED_fit_label, pre_req_properties, **global_kwargs)

    @property
    def name(self: Self) -> str:
        if self.pre_req_properties[0].dust_calculator is not None:
            dust_label = "_" + "_".join(self.pre_req_properties \
                [0].dust_calculator.name.split("_")[1:2]) + "dust"
        else:
            dust_label = ""
        ref_wav_label = f"{self.pre_req_properties[0].pre_req_properties[0].global_kwargs['ref_wav'].to(u.AA).value:.0f}"
        rest_wavs_label = rest_UV_wavs_name(self.pre_req_properties[0]. \
            pre_req_properties[0].pre_req_properties[0].global_kwargs["rest_UV_wav_lims"])
        ext_src_label = f"_extsrc_{self.pre_req_properties[0].pre_req_properties[0].global_kwargs['ext_src_corrs']}" \
            if self.pre_req_properties[0].pre_req_properties[0].global_kwargs["ext_src_corrs"] is not None else ""
        ext_src_lim_label = f"<{self.pre_req_properties[0].pre_req_properties[0].global_kwargs['ext_src_uplim']:.0f}" if \
            self.pre_req_properties[0].pre_req_properties[0].global_kwargs["ext_src_uplim"] is not None and \
            self.pre_req_properties[0].pre_req_properties[0].global_kwargs["ext_src_corrs"] is not None else ""
        return f"SFR{ref_wav_label}{dust_label}_" + \
            f"{rest_wavs_label}_{self.global_kwargs['SFR_conv']}" + \
            ext_src_label + ext_src_lim_label
    
    @property
    def plot_name(self: Self) -> str:
        return r"$\mathrm{SFR}_{\mathrm{UV}}$" # units here too

    def _kwarg_assertions(self: Self) -> NoReturn:
        assert self.global_kwargs["SFR_conv"] in funcs.SFR_conversions.keys()
    
    def _calc_obj_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        # always pass
        return False
    
    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # load LUVs
        if len(fluxes_arr) > 1:
            LUV_arr = phot_rest.property_PDFs[self.pre_req_properties[0].name].input_arr
        else:
            LUV_arr = phot_rest.properties[self.pre_req_properties[0].name]
        # convert LUVs to SFRs
        SFR_arr = funcs.SFR_conversions[self.global_kwargs["SFR_conv"]] * LUV_arr
        return SFR_arr
    
    def _get_output_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}


class Optical_Continuum_Calculator(Rest_Frame_Property_Calculator):
    
    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        strong_line_names: Union[str, list],
        rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
    ) -> None:
        if isinstance(strong_line_names, str):
            strong_line_names = strong_line_names.split("+")
        global_kwargs = {"strong_line_names": strong_line_names, "rest_optical_wavs": rest_optical_wavs}
        super().__init__(aper_diam, SED_fit_label, [], **global_kwargs)

    @property
    def name(self: Self) -> str:
        return f"cont_{'+'.join(self.global_kwargs['strong_line_names'])}"
    
    @property
    def plot_name(self: Self) -> str:
        return f"{'+'.join(self.global_kwargs['strong_line_names'])} continuum / nJy"

    def _kwarg_assertions(self: Self) -> None:
        assert all(
            line_name in strong_optical_lines
            for line_name in self.global_kwargs["strong_line_names"]
        )
    
    def _calc_obj_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        # determine the nearest band to the first line
        wavelength = line_diagnostics[self.global_kwargs \
            ["strong_line_names"][0]]["line_wav"] * (1.0 + phot_rest.z.value)
        if len(phot_rest.filterset) == 0:
            return {
                "emission_band": None,
                "cont_bands": None,
                "keep_indices": None,
            }
        nearest_band = phot_rest.filterset[
            int(np.abs(
                [
                    funcs.convert_wav_units(filt.WavelengthCen, u.AA).value
                    for filt in phot_rest.filterset
                ]
                - funcs.convert_wav_units(wavelength, u.AA).value
            ).argmin())
        ]
        # ensure emission line actually falls within this band
        emission_bands = [
            filt.band_name
            for filt in phot_rest.filterset
            if wavelength > filt.WavelengthLower50
            and wavelength < filt.WavelengthUpper50
        ]
        if nearest_band.band_name not in emission_bands:
            emission_band = None
            cont_bands = None
            cont_band_indices = None
        else:
            emission_band = nearest_band
            cont_bands = []
            cont_band_indices = []
            for i, filt in enumerate(phot_rest.filterset):
                # get continuum bands which are entirely within the 
                # rest frame optical and do not contain any strong optical lines
                if (
                    filt.WavelengthUpper50 < self.global_kwargs["rest_optical_wavs"][1] * (1.0 + phot_rest.z.value)
                    and filt.WavelengthLower50 > self.global_kwargs["rest_optical_wavs"][0] * (1.0 + phot_rest.z.value)
                    and not any(
                        line_diagnostics[line_name]["line_wav"] * (1.0 + phot_rest.z.value)
                        < filt.WavelengthUpper50
                        and line_diagnostics[line_name]["line_wav"]
                        * (1.0 + phot_rest.z.value)
                        > filt.WavelengthLower50
                        for line_name in strong_optical_lines
                    )
                ):
                    cont_bands.extend([filt])
                    cont_band_indices.extend([i])
            if len(cont_bands) == 0 or any(np.isnan(phot_rest.depths[i]) for i in cont_band_indices):
                cont_bands = None
                cont_band_indices = None
        return {
            "emission_band": emission_band,
            "cont_bands": cont_bands,
            "keep_indices": cont_band_indices,
            }

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        if any(value is None for value in self.obj_kwargs.values()):
            return True
        else:
            # ensure all lines lie within this band (defined by 50% throughput boundaries) 
            # and that there are no other strong optical lines in this band
            # and that there is more than 1 relevant continuum band
            if not all(
                (line_diagnostics[line_name]["line_wav"]) * (1.0 + phot_rest.z.value)
                < self.obj_kwargs["emission_band"].WavelengthUpper50
                and (line_diagnostics[line_name]["line_wav"]) * (1.0 + phot_rest.z.value)
                > self.obj_kwargs["emission_band"].WavelengthLower50
                for line_name in self.global_kwargs["strong_line_names"]
            ) or any(
                line_diagnostics[line_name]["line_wav"] * (1.0 + phot_rest.z.value)
                < self.obj_kwargs["emission_band"].WavelengthUpper50
                and line_diagnostics[line_name]["line_wav"] * (1.0 + phot_rest.z.value)
                > self.obj_kwargs["emission_band"].WavelengthLower50
                for line_name in strong_optical_lines
                if line_name not in self.global_kwargs["strong_line_names"]
            ) or len(self.obj_kwargs["cont_bands"]) == 0:
                return True
            else:
                return False
    
    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        #funcs.convert_mag_units(wavs, phot_rest.flux[self.obj_kwargs["cont_band_indices"]], u.nJy)
        # TODO: Generalize in the instance of flux units not convertible to nJy
        flux_unit = fluxes_arr.unit
        cont_fluxes = np.array(fluxes_arr.value, dtype=np.float64)
        
        if len(self.obj_kwargs["cont_bands"]) == 1:
            cont_chains = cont_fluxes[:, 0]
        elif len(self.obj_kwargs["cont_bands"]) >= 2:
            # calculate continuum from interpolation to 
            # middle of the emission band if two continuum bands
            cont_wavs = [
                (band.WavelengthCen.to(u.AA) / (1.0 + phot_rest.z.value)).value
                for band in self.obj_kwargs["cont_bands"]
            ] # in Angstrom
            em_wav = (
                self.obj_kwargs["emission_band"].WavelengthCen.to(u.AA) / (1.0 + phot_rest.z.value)
            ).value # in Angstrom
            cont_chains = np.array([funcs.interpolate_linear_fit( \
                np.array(cont_wavs, dtype=np.float64), cont_fluxes_, em_wav) \
                for cont_fluxes_ in cont_fluxes])
        # set negative fluxes to NaNs
        valid_chains = cont_chains[cont_chains > 0.0]
        if len(fluxes_arr) > 1:
            self.obj_kwargs["negative_flux_pc"] = 100.0 * (1 - len(valid_chains) / len(cont_chains))
            if len(valid_chains) < 50 or self.obj_kwargs["negative_flux_pc"] > 99.0:
                return None
        else:
            if len(valid_chains) < 1:
                return None
        cont_chains[cont_chains < 0.0] = np.nan
        return (cont_chains * flux_unit).to(u.nJy)
    
    def _get_output_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {
            "bands": "+".join([band.band_name for band in self.obj_kwargs["cont_bands"]]),
            "negative_flux_pc": self.obj_kwargs["negative_flux_pc"]
        }
    

class Optical_Line_EW_Calculator(Rest_Frame_Property_Calculator):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        strong_line_names: Union[str, list],
        frame: str = "rest",
        rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
    ) -> None:
        if isinstance(strong_line_names, str):
            strong_line_names = strong_line_names.split("+")
        global_kwargs = {"strong_line_names": strong_line_names, "frame": frame, "rest_optical_wavs": rest_optical_wavs}
        pre_req_properties = [Optical_Continuum_Calculator(aper_diam, SED_fit_label, strong_line_names, rest_optical_wavs)]
        super().__init__(aper_diam, SED_fit_label, pre_req_properties, **global_kwargs)

    @property
    def name(self: Self) -> str:
        return f"EW{self.global_kwargs['frame']}_{'+'.join(self.global_kwargs['strong_line_names'])}"
    
    @property
    def plot_name(self: Self) -> str:
        return f"{'+'.join(self.global_kwargs['strong_line_names'])} EW / " + r"$\mathrm{\AA}$"

    def _kwarg_assertions(self: Self) -> None:
        assert all(
            line_name in strong_optical_lines
            for line_name in self.global_kwargs["strong_line_names"]
        )
        assert self.global_kwargs["frame"] in ["rest", "obs"]
    
    def _calc_obj_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        # determine the nearest band to the first line
        wavelength = line_diagnostics[self.global_kwargs \
            ["strong_line_names"][0]]["line_wav"] * (1.0 + phot_rest.z.value)
        nearest_band = phot_rest.filterset[
            int(np.abs(
                [
                    funcs.convert_wav_units(filt.WavelengthCen, u.AA).value
                    for filt in phot_rest.filterset
                ]
                - funcs.convert_wav_units(wavelength, u.AA).value
            ).argmin())
        ]
        # ensure emission line actually falls within this band
        emission_bands = [
            filt.band_name
            for filt in phot_rest.filterset
            if wavelength > filt.WavelengthLower50
            and wavelength < filt.WavelengthUpper50
        ]

        if nearest_band.band_name not in emission_bands:
            failure = True
        else:
            emission_band = nearest_band
            emission_band_index = int(np.where(np.array( \
                phot_rest.filterset.band_names) == emission_band.band_name)[0][0])
            if np.isnan(phot_rest.depths[emission_band_index]):
                failure = True
            else:
                failure = False
                emission_band_wavelength = emission_band.WavelengthCen
                bandwidth = emission_band.WavelengthUpper50 \
                    - emission_band.WavelengthLower50
                if self.global_kwargs["frame"] == "rest":
                    bandwidth /= (1.0 + phot_rest.z.value)
                bandwidth = bandwidth.to(u.AA)
        if failure:
            emission_band = None
            emission_band_index = None
            emission_band_wavelength = None
            bandwidth = None
        return {
            "emission_band": emission_band,
            "keep_indices": emission_band_index,
            "emission_band_wav": emission_band_wavelength,
            "bandwidth": bandwidth
        }

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        if any(value is None for value in self.obj_kwargs.values()):
            return True
        else:
            # ensure all lines lie within this band (defined by 50% throughput boundaries) 
            # and that there are no other strong optical lines in this band
            if not all(
                (line_diagnostics[line_name]["line_wav"]) * (1.0 + phot_rest.z.value)
                < self.obj_kwargs["emission_band"].WavelengthUpper50
                and (line_diagnostics[line_name]["line_wav"]) * (1.0 + phot_rest.z.value)
                > self.obj_kwargs["emission_band"].WavelengthLower50
                for line_name in self.global_kwargs["strong_line_names"]
            ) or any(
                line_diagnostics[line_name]["line_wav"] * (1.0 + phot_rest.z.value)
                < self.obj_kwargs["emission_band"].WavelengthUpper50
                and line_diagnostics[line_name]["line_wav"] * (1.0 + phot_rest.z.value)
                > self.obj_kwargs["emission_band"].WavelengthLower50
                for line_name in strong_optical_lines
                if line_name not in self.global_kwargs["strong_line_names"]
            ):
                return True
            else:
                return False
    
    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # load previously computed continuum chains
        if len(fluxes_arr) > 1:
            cont_fluxes = phot_rest.property_PDFs[self.pre_req_properties[0].name].input_arr
            assert len(fluxes_arr) == len(cont_fluxes)
        else:
            cont_fluxes = phot_rest.properties[self.pre_req_properties[0].name]
        return ((fluxes_arr.flatten() / cont_fluxes).to(u.dimensionless_unscaled) \
                - 1.0) * self.obj_kwargs["bandwidth"]
    
    def _get_output_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        # calculate potential contaminant lines
        contam_lines = [
            name
            for name, line in line_diagnostics.items()
            if name not in strong_optical_lines
            and line["line_wav"] * (1.0 * phot_rest.z.value) < \
                self.obj_kwargs["emission_band"].WavelengthUpper50
            and line["line_wav"] * (1.0 * phot_rest.z.value) > \
                self.obj_kwargs["emission_band"].WavelengthLower50
        ]
        return {
            "band": self.obj_kwargs["emission_band"].band_name,
            "contam_lines": "+".join(contam_lines)
        }


class Dust_Attenuation_From_UV_Calculator(Rest_Frame_Property_Calculator):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        calc_wav: u.Quantity,
        dust_law: Union[str, Type[Dust_Law]] = Calzetti00,
        beta_dust_conv: Union[str, Type[AUV_from_beta]] = M99,
        UV_ref_wav: u.Quantity = 1_500.0 * u.AA,
        UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        keep_valid: bool = False,
    ) -> NoReturn:
        dust_atten_calculator = \
            UV_Dust_Attenuation_Calculator(
                aper_diam,
                SED_fit_label,
                UV_wav_lims,
                beta_dust_conv,
                UV_ref_wav,
            )
        pre_req_properties = [dust_atten_calculator]
        if isinstance(dust_law, str):
            dust_law = [dust_law_cls() for dust_law_cls \
                in Dust_Law.__subclasses__() \
                if dust_law_cls.__name__ == dust_law][0]
        elif not isinstance(dust_law, Dust_Law):
            dust_law = dust_law()
        global_kwargs = {
            "calc_wav": calc_wav, 
            "dust_law": dust_law,
            "keep_valid": keep_valid
        }
        super().__init__(aper_diam, SED_fit_label, pre_req_properties, **global_kwargs)

    @property
    def name(self: Self) -> str:
        label = f"A{self.global_kwargs['calc_wav'].to(u.AA).value:.0f}" + \
            f"_{self.pre_req_properties[0].global_kwargs['beta_dust_conv'].__class__.__name__}" + \
            f"_{self.global_kwargs['dust_law'].__class__.__name__}"
        if self.global_kwargs["keep_valid"]:
            label += "_A>0"
        return label

    @property
    def plot_name(self: Self) -> str:
        return r"$A_{{}}$".format(int(self.global_kwargs['calc_wav'].to(u.AA).value))
    
    def _kwarg_assertions(self: Self) -> None:
        assert u.get_physical_type(self.global_kwargs["calc_wav"]) == "length"
        assert self.global_kwargs["dust_law"].__class__ in Dust_Law.__subclasses__()
        assert isinstance(self.global_kwargs["keep_valid"], bool)
    
    def _calc_obj_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        if any(value is None for value in self.obj_kwargs.values()):
            return True
        else:
            return False
    
    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # calculate AUV
        if len(fluxes_arr) > 1:
            AUV_arr = phot_rest.property_PDFs[self.pre_req_properties[0].name].input_arr
            assert len(fluxes_arr) == len(AUV_arr)
        else:
            AUV_arr = phot_rest.properties[self.pre_req_properties[0].name]
        AUV_arr = AUV_arr.to(u.ABmag).value
        # calculate A_lambda
        A_lambda = (AUV_arr * self.global_kwargs["dust_law"].k_lambda \
            (self.global_kwargs["calc_wav"].to(u.AA)) / self.global_kwargs["dust_law"] \
            .k_lambda(self.pre_req_properties[0].global_kwargs["ref_wav"].to(u.AA))) * u.ABmag
        if self.global_kwargs["keep_valid"]:
            A_lambda[A_lambda < 0.0] = 0.0
        return A_lambda
    
    def _get_output_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}


class Line_Dust_Attenuation_From_UV_Calculator(Dust_Attenuation_From_UV_Calculator):

    def __init__(
        self: Self, 
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        line_name: str,
        dust_law: Union[str, Type[Dust_Law]] = Calzetti00,
        beta_dust_conv: Union[str, Type[AUV_from_beta]] = M99,
        UV_ref_wav: u.Quantity = 1_500.0 * u.AA,
        UV_wav_lims: u.Quantity = [1_250.0, 3_000.0] * u.AA,
        keep_valid: bool = False
    ) -> NoReturn:
        assert line_name in line_diagnostics.keys(), \
            galfind_logger.critical(
                f"{line_name=} not in {line_diagnostics.keys()}"
            )
        super().__init__(
            aper_diam,
            SED_fit_label,
            line_diagnostics[line_name]["line_wav"].to(u.AA),
            dust_law,
            beta_dust_conv,
            UV_ref_wav,
            UV_wav_lims,
            keep_valid
        )


class Optical_Line_Flux_Calculator(Rest_Frame_Property_Calculator):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        strong_line_names: Union[str, list],
        frame: str = "rest",
        rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
        dust_law: Optional[Union[str, Type[Dust_Law]]] = Calzetti00,
        beta_dust_conv: Optional[Union[str, Type[AUV_from_beta]]] = M99,
        UV_ref_wav: Optional[u.Quantity] = 1_500.0 * u.AA,
        UV_wav_lims: Optional[u.Quantity] = [1_250.0, 3_000.0] * u.AA,
    ) -> NoReturn:
        cont_calculator = \
            Optical_Continuum_Calculator(
                aper_diam,
                SED_fit_label,
                strong_line_names,
                rest_optical_wavs
            )
        EW_calculator = \
            Optical_Line_EW_Calculator(
                aper_diam, 
                SED_fit_label, 
                strong_line_names, 
                frame, 
                rest_optical_wavs
            )
        pre_req_properties = [cont_calculator, EW_calculator]
        if any(dust_arg is None for dust_arg in [dust_law, beta_dust_conv, UV_ref_wav, UV_wav_lims]):
            self.dust_calculator = None
        else:
            if isinstance(strong_line_names, str):
                strong_line_names = strong_line_names.split("+")
            self.dust_calculator = \
                Line_Dust_Attenuation_From_UV_Calculator(
                    aper_diam, 
                    SED_fit_label,
                    strong_line_names[0], 
                    dust_law,
                    beta_dust_conv, 
                    UV_ref_wav, 
                    UV_wav_lims,
                    keep_valid = True
                )
            pre_req_properties.append(self.dust_calculator)
        super().__init__(aper_diam, SED_fit_label, pre_req_properties)

    @property
    def name(self: Self) -> str:
        if self.dust_calculator is not None:
            dust_label = "_" + "_".join(self.dust_calculator.name.split("_")[1:2]) \
                + "_" + self.dust_calculator.name.split("_")[-1]
        else:
            dust_label = ""
        return f"flux_{self.pre_req_properties[1].global_kwargs['frame']}_" + \
            f"{'+'.join(self.pre_req_properties[1].global_kwargs['strong_line_names'])}{dust_label}"
    
    @property
    def plot_name(self: Self) -> str:
        return f"{'+'.join(self.pre_req_properties[1].global_kwargs['strong_line_names'])} flux / " + \
            r"$\mathrm{erg\,s^{-1}\,cm^{-2}}$"

    def _kwarg_assertions(self: Self) -> NoReturn:
        pass
    
    def _calc_obj_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        emission_band = phot_rest.property_kwargs[self.pre_req_properties[1].name]["band"]
        if emission_band in phot_rest.filterset.band_names:
            emission_band_index = int(np.where(np.array(phot_rest.filterset.band_names) == emission_band)[0][0])
            if np.isnan(phot_rest.depths[emission_band_index]):
                band_wav = None
            else:
                band_wav = deepcopy(phot_rest.filterset[emission_band].WavelengthCen)
                if band_wav is not None:
                    if self.pre_req_properties[1].global_kwargs["frame"] == "rest":
                        band_wav /= (1.0 + phot_rest.z.value)
        else:
            band_wav = None
        return {"band_wav": band_wav}

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        if any(value is None for value in self.obj_kwargs.values()):
            return True
        else:
            return False
    
    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # extract continuum and EW chains/value
        if len(fluxes_arr) > 1:
            cont_arr = phot_rest.property_PDFs[self.pre_req_properties[0].name].input_arr
            EW_arr = phot_rest.property_PDFs[self.pre_req_properties[1].name].input_arr
            assert len(fluxes_arr) == len(cont_arr) == len(EW_arr)
        else:
            cont_arr = phot_rest.properties[self.pre_req_properties[0].name]
            EW_arr = phot_rest.properties[self.pre_req_properties[1].name]
        # calculate line fluxes
        line_flux_arr = (EW_arr * funcs.convert_mag_units(self.obj_kwargs["band_wav"], \
            cont_arr, u.erg / (u.s * u.AA * u.cm ** 2))).to(u.erg / (u.s * u.cm ** 2))
        # extract dust chains/value if required
        if self.dust_calculator is not None:
            if len(fluxes_arr) > 1:
                A_arr = phot_rest.property_PDFs[self.dust_calculator.name].input_arr
                assert len(fluxes_arr) == len(A_arr)
            else:
                A_arr = phot_rest.properties[self.dust_calculator.name]
            # correct for dust attenuation
            line_flux_arr = funcs.dust_correct(line_flux_arr, A_arr)
        # output line fluxes in appropriate frame
        return line_flux_arr
    
    def _get_output_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}


class Optical_Line_Luminosity_Calculator(Rest_Frame_Property_Calculator):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        strong_line_names: Union[str, list],
        #frame: str = "rest",
        rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
        dust_law: Optional[Union[str, Type[Dust_Law]]] = Calzetti00,
        beta_dust_conv: Optional[Union[str, Type[AUV_from_beta]]] = M99,
        UV_ref_wav: Optional[u.Quantity] = 1_500.0 * u.AA,
        UV_wav_lims: Optional[u.Quantity] = [1_250.0, 3_000.0] * u.AA,
    ) -> NoReturn:
        pre_req_properties = [
            Optical_Line_Flux_Calculator(
                aper_diam, 
                SED_fit_label, 
                strong_line_names, 
                "obs", 
                rest_optical_wavs, 
                dust_law, 
                beta_dust_conv, 
                UV_ref_wav, 
                UV_wav_lims
            )
        ]
        super().__init__(aper_diam, SED_fit_label, pre_req_properties)

    @property
    def name(self: Self) -> str:
        if self.pre_req_properties[0].dust_calculator is not None:
            dust_label = "_" + "_".join(self.pre_req_properties[0] \
                .dust_calculator.name.split("_")[1:2]) + "_" + \
                self.pre_req_properties[0].dust_calculator.name.split("_")[-1]
        else:
            dust_label = ""
        return f"lum_{'+'.join(self.pre_req_properties[0].pre_req_properties[1].global_kwargs['strong_line_names'])}{dust_label}"
    
    @property
    def plot_name(self: Self) -> str:
        return r"L_{{{}}} / \mathrm{erg\,s^{-1}}".format('+'.join(self.pre_req_properties[0].pre_req_properties[1].global_kwargs['strong_line_names']))

    def _kwarg_assertions(self: Self) -> NoReturn:
        pass
    
    def _calc_obj_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {"lum_distance": astropy_cosmo.luminosity_distance(phot_rest.z.value)}

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        if any(value is None for value in self.obj_kwargs.values()):
            return True
        else:
            return False
    
    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # extract line flux chains/value
        if len(fluxes_arr) > 1:
            line_flux_arr = phot_rest.property_PDFs[self.pre_req_properties[0].name].input_arr
            assert len(fluxes_arr) == len(line_flux_arr)
        else:
            line_flux_arr = phot_rest.properties[self.pre_req_properties[0].name]
        # if len(line_flux_arr[np.isfinite(line_flux_arr)]) == 0:
        #     breakpoint()
        # calculate line luminosities
        line_lum_arr = (4 * np.pi * line_flux_arr * \
            self.obj_kwargs["lum_distance"] ** 2).to(u.erg / u.s)
        # output line luminosities
        return line_lum_arr
    
    def _get_output_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}


class Ndot_Ion_Calculator(Rest_Frame_Property_Calculator):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        #frame: str = "rest",
        rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
        dust_law: Optional[Union[str, Type[Dust_Law]]] = Calzetti00,
        beta_dust_conv: Optional[Union[str, Type[AUV_from_beta]]] = M99,
        UV_wav_lims: Optional[u.Quantity] = [1_250.0, 3_000.0] * u.AA,
        fesc_conv: Optional[Union[str, float]] = None,
        logged: bool = True,
    ) -> NoReturn:
        line_lum_calculator = \
            Optical_Line_Luminosity_Calculator(
                aper_diam, 
                SED_fit_label, 
                "Halpha", 
                rest_optical_wavs, 
                dust_law, 
                beta_dust_conv
            )
        pre_req_properties = [line_lum_calculator]
        if fesc_conv is None:
            self.fesc_calculator = None
        elif isinstance(fesc_conv, str):
            self.fesc_calculator = Fesc_From_Beta_Calculator(
                aper_diam,
                SED_fit_label,
                UV_wav_lims,
                fesc_conv,
                keep_valid = True
            )
            pre_req_properties.append(self.fesc_calculator)
        else: # float
            self.fesc_calculator = fesc_conv
        global_kwargs = {"logged": logged}
        super().__init__(aper_diam, SED_fit_label, pre_req_properties, **global_kwargs)

    @property
    def name(self: Self) -> str:
        if self.pre_req_properties[0].pre_req_properties[0]. \
                dust_calculator is not None:
            dust_label = "_" + "_".join(self.pre_req_properties[0]. \
                pre_req_properties[0].dust_calculator.name.split("_")[1:2]) + "dust"
        else:
            dust_label = ""
        if self.fesc_calculator is None:
            fesc_label = "fesc=0"
        elif isinstance(self.fesc_calculator, Fesc_From_Beta_Calculator):
            fesc_label = self.fesc_calculator.name.split("_")[0]
            if dust_label == "":
                fesc_label += "_".join(fesc_label.split("_")[1:])
        else: # isinstance(fesc_conv, float)
            fesc_label = f"fesc={self.fesc_calculator:.2f}"
        line_label = "+".join(self.pre_req_properties[0]. \
            pre_req_properties[0].pre_req_properties[1]. \
            global_kwargs["strong_line_names"])
        # try:
        #     ext_src_label = f"_extsrc_{self.pre_req_properties[0].pre_req_properties[0].global_kwargs['ext_src_corrs']}" \
        #         if self.pre_req_properties[0].pre_req_properties[0].global_kwargs["ext_src_corrs"] is not None else ""
        #     ext_src_lim_label = f"<{self.pre_req_properties[0].pre_req_properties[0].global_kwargs['ext_src_uplim']:.0f}" if \
        #         self.pre_req_properties[0].pre_req_properties[0].global_kwargs["ext_src_uplim"] is not None and \
        #         self.pre_req_properties[0].pre_req_properties[0].global_kwargs["ext_src_corrs"] is not None else ""
        # except:
        #     breakpoint()
        label = f"ndot_ion_{line_label}{dust_label}_{fesc_label}"#{ext_src_label}{ext_src_lim_label}"
        if self.global_kwargs["logged"]:
            label = f"log_{label}"
        return label

    @property
    def plot_name(self: Self) -> str:
        if self.global_kwargs["logged"]:
            return r"$\log(\dot{n}_{\mathrm{ion}}~/~\mathrm{s}^{-1})$"
        else:
            return r"$\dot{n}_{\mathrm{ion}}~/~\mathrm{s}^{-1}$"
    
    def _kwarg_assertions(self: Self) -> NoReturn:
        if self.fesc_calculator is not None:
            assert isinstance(self.fesc_calculator, (Fesc_From_Beta_Calculator, float))
        if isinstance(self.fesc_calculator, float):
            assert self.fesc_calculator >= 0.0
            assert self.fesc_calculator <= 1.0
    
    def _calc_obj_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        # always pass
        return False
    
    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        if self.fesc_calculator is None or self.fesc_calculator == 0.0 or self.fesc_calculator == 0:
            ndot_0 = True
        else:
            ndot_0 = False
        # extract line and UV luminosity (and fesc is required) chains/value
        if len(fluxes_arr) > 1:
            line_lum_arr = phot_rest.property_PDFs[self.pre_req_properties[0].name].input_arr
            assert len(fluxes_arr) == len(line_lum_arr)
            if self.fesc_calculator is None:
                fesc_arr = np.full(len(fluxes_arr), 0.0)
            elif isinstance(self.fesc_calculator, Fesc_From_Beta_Calculator):
                fesc_arr = phot_rest.property_PDFs[self.fesc_calculator.name].input_arr
            else: # isinstance(fesc_conv, float)
                fesc_arr = np.full(len(fluxes_arr), float(self.fesc_calculator))
            assert len(fluxes_arr) == len(fesc_arr)
        else:
            line_lum_arr = phot_rest.properties[self.pre_req_properties[0].name]
            if self.fesc_calculator is None:
                fesc_arr = 0.0
            elif isinstance(self.fesc_calculator, Fesc_From_Beta_Calculator):
                fesc_arr = phot_rest.properties[self.fesc_calculator.name]
            else: # isinstance(fesc_conv, float)
                fesc_arr = float(self.fesc_calculator)

        # calculate ndot_ion values 
        # under assumption of Case B recombination
        if ndot_0:
            ndot_ion_arr = line_lum_arr / (1.36e-12 * u.erg)
        else:
            ndot_ion_arr = line_lum_arr * fesc_arr / (1.36e-12 * u.erg * (1.0 - fesc_arr))
        ndot_ion_arr = ndot_ion_arr.to(u.Hz)
        ndot_ion_arr[~np.isfinite(ndot_ion_arr)] = np.nan
        if self.global_kwargs["logged"]:
            ndot_ion_arr = np.log10(ndot_ion_arr.value) * u.Unit(f"dex({ndot_ion_arr.unit.to_string()})")
        finite_ndot_ion_arr = ndot_ion_arr[np.isfinite(ndot_ion_arr)]
        if len(fluxes_arr) > 1:
            self.obj_kwargs["negative_ndot_ion_pc"] = 100.0 * (1 - len(finite_ndot_ion_arr) / len(ndot_ion_arr))
            if len(finite_ndot_ion_arr) < 50 or self.obj_kwargs["negative_ndot_ion_pc"] > 99.0:
                return None
        else:
            if len(finite_ndot_ion_arr) < 1:
                return None
        return ndot_ion_arr
    
    def _get_output_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}


# class Ndot_Ion_Fesc_Calculator(Rest_Frame_Property_Calculator):

#     def __init__(
#         self: Self,
#         aper_diam: u.Quantity,
#         SED_fit_label: Union[str, Type[SED_code]],
#         #frame: str = "rest",
#         rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
#         dust_law: Optional[Union[str, Type[Dust_Law]]] = Calzetti00,
#         beta_dust_conv: Optional[Union[str, Type[AUV_from_beta]]] = M99,
#         UV_wav_lims: Optional[u.Quantity] = [1_250.0, 3_000.0] * u.AA,
#         fesc_conv: Union[str, float] = "Chisholm22",
#         logged: bool = True,
#     ) -> NoReturn:
#         ndot_ion_calculator = \
#             Ndot_Ion_Calculator(
#                 aper_diam,
#                 SED_fit_label,
#                 #frame,
#                 rest_optical_wavs,
#                 dust_law,
#                 beta_dust_conv,
#                 UV_wav_lims,
#                 fesc_conv,
#                 logged = False,
#             )
                
#         pre_req_properties = [ndot_ion_calculator]
#         if isinstance(fesc_conv, str):
#             self.fesc_calculator = Fesc_From_Beta_Calculator(
#                 aper_diam,
#                 SED_fit_label,
#                 UV_wav_lims,
#                 fesc_conv,
#                 keep_valid = True
#             )
#             pre_req_properties.append(self.fesc_calculator)
#         else: # float
#             self.fesc_calculator = fesc_conv
#         global_kwargs = {"logged": logged}
#         super().__init__(aper_diam, SED_fit_label, pre_req_properties, **global_kwargs)

#     @property
#     def name(self: Self) -> str:
#         if self.pre_req_properties[0].pre_req_properties[0].pre_req_properties[0]. \
#                 dust_calculator is not None:
#             dust_label = "_" + "_".join(self.pre_req_properties[0]. \
#                 pre_req_properties[0].dust_calculator.name.split("_")[1:2]) + "dust"
#         else:
#             dust_label = ""
#         if isinstance(self.fesc_calculator, Fesc_From_Beta_Calculator):
#             fesc_label = self.fesc_calculator.name.split("_")[0]
#             if dust_label == "":
#                 fesc_label += "_".join(fesc_label.split("_")[1:])
#         else: # isinstance(fesc_conv, float)
#             fesc_label = f"fesc={self.fesc_calculator:.2f}"
#         line_label = "+".join(self.pre_req_properties[0]. \
#             pre_req_properties[0].pre_req_properties[0].pre_req_properties[1]. \
#             global_kwargs["strong_line_names"])
#         # try:
#         #     ext_src_label = f"_extsrc_{self.pre_req_properties[0].pre_req_properties[0].global_kwargs['ext_src_corrs']}" \
#         #         if self.pre_req_properties[0].pre_req_properties[0].global_kwargs["ext_src_corrs"] is not None else ""
#         #     ext_src_lim_label = f"<{self.pre_req_properties[0].pre_req_properties[0].global_kwargs['ext_src_uplim']:.0f}" if \
#         #         self.pre_req_properties[0].pre_req_properties[0].global_kwargs["ext_src_uplim"] is not None and \
#         #         self.pre_req_properties[0].pre_req_properties[0].global_kwargs["ext_src_corrs"] is not None else ""
#         # except:
#         #     breakpoint()
#         label = f"fesc_ndot_ion_{line_label}{dust_label}_{fesc_label}"#{ext_src_label}{ext_src_lim_label}"
#         if self.global_kwargs["logged"]:
#             label = f"log_{label}"
#         return label

#     @property
#     def plot_name(self: Self) -> str:
#         if self.global_kwargs["logged"]:
#             return r"$\log(\dot{n}_{\mathrm{ion}}f_{\mathrm{esc}}~/~\mathrm{s}^{-1})$"
#         else:
#             return r"$\dot{n}_{\mathrm{ion}}f_{\mathrm{esc}}~/~\mathrm{s}^{-1}$"
    
#     def _kwarg_assertions(self: Self) -> NoReturn:
#         if isinstance(self.fesc_calculator, float):
#             assert self.fesc_calculator >= 0.0
#             assert self.fesc_calculator <= 1.0
#         else:
#             assert isinstance(self.fesc_calculator, Fesc_From_Beta_Calculator)
    
#     def _calc_obj_kwargs(
#         self: Self,
#         phot_rest: Photometry_rest
#     ) -> Dict[str, Any]:
#         return {}

#     def _fail_criteria(
#         self: Self,
#         phot_rest: Photometry_rest,
#     ) -> bool:
#         # always pass
#         return False
    
#     def _calculate(
#         self: Self,
#         fluxes_arr: u.Quantity,
#         phot_rest: Photometry_rest,
#     ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
#         # extract line and UV luminosity (and fesc is required) chains/value
#         if len(fluxes_arr) > 1:
#             ndot_ion_arr = phot_rest.property_PDFs[self.pre_req_properties[0].name].input_arr
#             assert len(fluxes_arr) == len(ndot_ion_arr)
#             if self.fesc_calculator is None:
#                 fesc_arr = np.full(len(fluxes_arr), 0.0)
#             elif isinstance(self.fesc_calculator, Fesc_From_Beta_Calculator):
#                 fesc_arr = phot_rest.property_PDFs[self.fesc_calculator.name].input_arr
#             else: # isinstance(fesc_conv, float)
#                 fesc_arr = np.full(len(fluxes_arr), self.fesc_calculator)
#             assert len(fluxes_arr) == len(fesc_arr)
#         else:
#             ndot_ion_arr = phot_rest.properties[self.pre_req_properties[0].name]
#             if self.fesc_calculator is None:
#                 fesc_arr = 0.0
#             elif isinstance(self.fesc_calculator, Fesc_From_Beta_Calculator):
#                 fesc_arr = phot_rest.properties[self.fesc_calculator.name]
#             else: # isinstance(fesc_conv, float)
#                 fesc_arr = self.fesc_calculator
#         # calculate fesc_ndot_ion values
#         fesc_ndot_ion_arr = fesc_arr * ndot_ion_arr
#         return fesc_ndot_ion_arr
    
#     def _get_output_kwargs(
#         self: Self,
#         phot_rest: Photometry_rest
#     ) -> Dict[str, Any]:
#         return {}

class Xi_Ion_Calculator(Rest_Frame_Property_Calculator):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        #frame: str = "rest",
        rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
        dust_law: Optional[Union[str, Type[Dust_Law]]] = Calzetti00,
        beta_dust_conv: Optional[Union[str, Type[AUV_from_beta]]] = M99,
        UV_ref_wav: Optional[u.Quantity] = 1_500.0 * u.AA,
        UV_wav_lims: Optional[u.Quantity] = [1_250.0, 3_000.0] * u.AA,
        top_hat_width: u.Quantity = 100.0 * u.AA,
        resolution: u.Quantity = 1.0 * u.AA,
        fesc_conv: Optional[Union[str, float]] = None,
        logged: bool = True,
        #ext_src_corrs: Optional[str] = "UV",
        #ext_src_uplim: Optional[Union[int, float]] = 10.0,
    ) -> NoReturn:
        line_lum_calculator = \
            Optical_Line_Luminosity_Calculator(
                aper_diam, 
                SED_fit_label, 
                "Halpha", 
                #frame,
                rest_optical_wavs, 
                dust_law, 
                beta_dust_conv, 
                UV_ref_wav, 
                UV_wav_lims
            )
        LUV_calculator = LUV_Calculator(
            aper_diam,
            SED_fit_label,
            #frame,
            UV_wav_lims,
            UV_ref_wav,
            beta_dust_conv,
            top_hat_width,
            resolution,
            ext_src_corrs = None,
            ext_src_uplim = None,
        )
        pre_req_properties = [line_lum_calculator, LUV_calculator]
        if fesc_conv is None:
            self.fesc_calculator = None
        elif isinstance(fesc_conv, str):
            self.fesc_calculator = Fesc_From_Beta_Calculator(
                aper_diam,
                SED_fit_label,
                UV_wav_lims,
                fesc_conv,
                keep_valid = True
            )
            pre_req_properties.append(self.fesc_calculator)
        else: # float
            self.fesc_calculator = fesc_conv
        global_kwargs = {"logged": logged}
        super().__init__(aper_diam, SED_fit_label, pre_req_properties, **global_kwargs)

    @property
    def name(self: Self) -> str:
        if self.pre_req_properties[0].pre_req_properties[0]. \
                dust_calculator is not None:
            dust_label = "_" + "_".join(self.pre_req_properties[0]. \
                pre_req_properties[0].dust_calculator.name.split("_")[1:2]) + "dust"
        else:
            dust_label = ""
        if self.fesc_calculator is None:
            fesc_label = "fesc=0"
        elif isinstance(self.fesc_calculator, Fesc_From_Beta_Calculator):
            fesc_label = self.fesc_calculator.name.split("_")[0]
            if dust_label == "":
                fesc_label += "_".join(fesc_label.split("_")[1:])
        else: # isinstance(fesc_conv, float)
            fesc_label = f"fesc={self.fesc_calculator:.2f}"
        line_label = "+".join(self.pre_req_properties[0]. \
            pre_req_properties[0].pre_req_properties[1]. \
            global_kwargs["strong_line_names"])
        # ext_src_label = "_extsrc" if self.pre_req_properties[1]. \
        #     pre_req_properties[0].global_kwargs["ext_src_corrs"] else ""
        label = f"xi_ion_{line_label}{dust_label}_{fesc_label}" #{ext_src_label}"
        if self.global_kwargs["logged"]:
            label = f"log_{label}"
        return label

    @property
    def plot_name(self: Self) -> str:
        if self.global_kwargs["logged"]:
            return r"$\log(\xi_{\mathrm{ion}}~/~\mathrm{Hz erg}^{-1})$"
        else:
            return r"$\xi_{\mathrm{ion}}~/~\mathrm{Hz erg}^{-1}$"
    
    def _kwarg_assertions(self: Self) -> NoReturn:
        if self.fesc_calculator is not None:
            assert isinstance(self.fesc_calculator, (Fesc_From_Beta_Calculator, float))
        if isinstance(self.fesc_calculator, float):
            assert self.fesc_calculator >= 0.0
            assert self.fesc_calculator <= 1.0
    
    def _calc_obj_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        # always pass
        return False
    
    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # extract line and UV luminosity (and fesc is required) chains/value
        if len(fluxes_arr) > 1:
            line_lum_arr = phot_rest.property_PDFs[self.pre_req_properties[0].name].input_arr
            LUV_arr = phot_rest.property_PDFs[self.pre_req_properties[1].name].input_arr
            assert len(fluxes_arr) == len(line_lum_arr) == len(LUV_arr)
            if self.fesc_calculator is None:
                fesc_arr = np.full(len(fluxes_arr), 0.0)
            elif isinstance(self.fesc_calculator, Fesc_From_Beta_Calculator):
                fesc_arr = phot_rest.property_PDFs[self.fesc_calculator.name].input_arr
            else: # isinstance(fesc_conv, float)
                fesc_arr = np.full(len(fluxes_arr), self.fesc_calculator)
            assert len(fluxes_arr) == len(fesc_arr)
        else:
            line_lum_arr = phot_rest.properties[self.pre_req_properties[0].name]
            LUV_arr = phot_rest.properties[self.pre_req_properties[1].name]
            if self.fesc_calculator is None:
                fesc_arr = 0.0
            elif isinstance(self.fesc_calculator, Fesc_From_Beta_Calculator):
                fesc_arr = phot_rest.properties[self.fesc_calculator.name]
            else: # isinstance(fesc_conv, float)
                fesc_arr = self.fesc_calculator
        # calculate xi_ion values 
        # under assumption of Case B recombination
        xi_ion_arr = (line_lum_arr / (1.36e-12 * u.erg * \
            (1.0 - fesc_arr) * LUV_arr)).to(u.Hz / u.erg)
        xi_ion_arr[~np.isfinite(xi_ion_arr)] = np.nan
        if self.global_kwargs["logged"]:
            xi_ion_arr = np.log10(xi_ion_arr.value) * u.Unit(f"dex({xi_ion_arr.unit.to_string()})")
        finite_xi_ion_arr = xi_ion_arr[np.isfinite(xi_ion_arr)]
        if len(fluxes_arr) > 1:
            self.obj_kwargs["negative_xi_ion_pc"] = 100.0 * (1 - len(finite_xi_ion_arr) / len(xi_ion_arr))
            if len(finite_xi_ion_arr) < 50 or self.obj_kwargs["negative_xi_ion_pc"] > 99.0:
                return None
        else:
            if len(finite_xi_ion_arr) < 1:
                return None
        return xi_ion_arr
    
    def _get_output_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}
    

class SFR_Halpha_Calculator(Rest_Frame_Property_Calculator):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        rest_optical_wavs: u.Quantity = [4_200.0, 10_000.0] * u.AA,
        dust_law: Optional[Union[str, Type[Dust_Law]]] = Calzetti00,
        beta_dust_conv: Optional[Union[str, Type[AUV_from_beta]]] = M99,
        UV_ref_wav: Optional[u.Quantity] = 1_500.0 * u.AA,
        UV_wav_lims: Optional[u.Quantity] = [1_250.0, 3_000.0] * u.AA,
        logged: bool = True,
    ) -> NoReturn:
        line_lum_calculator = \
            Optical_Line_Luminosity_Calculator(
                aper_diam, 
                SED_fit_label, 
                "Halpha",
                rest_optical_wavs, 
                dust_law, 
                beta_dust_conv, 
                UV_ref_wav, 
                UV_wav_lims
            )
        pre_req_properties = [line_lum_calculator]
        global_kwargs = {"logged": logged}
        super().__init__(aper_diam, SED_fit_label, pre_req_properties, **global_kwargs)

    @property
    def name(self: Self) -> str:
        if self.pre_req_properties[0].pre_req_properties[0]. \
                dust_calculator is not None:
            dust_label = "_" + "_".join(self.pre_req_properties[0]. \
                pre_req_properties[0].dust_calculator.name.split("_")[1:2]) + "dust"
        else:
            dust_label = ""
        label = f"SFR_Halpha{dust_label}"
        if self.global_kwargs["logged"]:
            label = f"log_{label}"
        return label

    @property
    def plot_name(self: Self) -> str:
        if self.global_kwargs["logged"]:
            return r"$\log(SFR_{\mathrm{H}\alpha})$"
        else:
            return r"$SFR_{\mathrm{H}\alpha}$"
    
    def _kwarg_assertions(self: Self) -> NoReturn:
        pass
    
    def _calc_obj_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}

    def _fail_criteria(
        self: Self,
        phot_rest: Photometry_rest,
    ) -> bool:
        # always pass
        return False
    
    def _calculate(
        self: Self,
        fluxes_arr: u.Quantity,
        phot_rest: Photometry_rest,
    ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
        # extract line and UV luminosity (and fesc is required) chains/value
        if len(fluxes_arr) > 1:
            line_lum_arr = phot_rest.property_PDFs[self.pre_req_properties[0].name].input_arr
            assert len(fluxes_arr) == len(line_lum_arr)
        else:
            line_lum_arr = phot_rest.properties[self.pre_req_properties[0].name]
        # calculate SFR_Halpha values
        # from Kennicutt 1998 (Salpeter 1955 IMF, 0.1-100 Msun)
        SFR_Halpha_arr = (7.9e-42 * line_lum_arr.to(u.erg / u.s)).value * u.Msun / u.yr
        finite_SFR_Halpha_arr = SFR_Halpha_arr[np.isfinite(SFR_Halpha_arr)]
        if len(fluxes_arr) > 1:
            self.obj_kwargs["negative_SFR_Halpha_pc"] = 100.0 * (1 - len(finite_SFR_Halpha_arr) / len(SFR_Halpha_arr))
            if len(finite_SFR_Halpha_arr) < 50 or self.obj_kwargs["negative_SFR_Halpha_pc"] > 99.0:
                return None
        else:
            if len(finite_SFR_Halpha_arr) < 1:
                return None
        SFR_Halpha_arr[~np.isfinite(SFR_Halpha_arr)] = np.nan
        if self.global_kwargs["logged"]:
            SFR_Halpha_arr = np.log10(SFR_Halpha_arr.value) * u.Unit(f"dex({SFR_Halpha_arr.unit.to_string()})")
        breakpoint()
        return SFR_Halpha_arr
    
    def _get_output_kwargs(
        self: Self,
        phot_rest: Photometry_rest
    ) -> Dict[str, Any]:
        return {}
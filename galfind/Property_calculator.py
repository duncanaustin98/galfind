
from __future__ import annotations

from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import astropy.units as u
from typing import TYPE_CHECKING, Dict, Any, List, Union, Tuple, Optional, NoReturn
if TYPE_CHECKING:
    from . import Catalogue, Galaxy, Photometry_rest, SED_code, SED_obs, PDF, Morphology_Fitter, Morphology_Result, Band_Cutout_Base
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import config, galfind_logger, all_band_names
from . import useful_funcs_austind as funcs
from . import Catalogue, Galaxy, SED_code, SED_result
from . import SED_fit_PDF


class Property_Calculator_Base(ABC):

    @property
    @abstractmethod
    def full_name(self: Self) -> str:
        pass

    @property
    @abstractmethod
    def name(self: Self) -> str:
        pass

    @property
    @abstractmethod
    def plot_name(self: Self) -> str:
        pass

    @abstractmethod
    def __call__(
        self: Self,
        object: Union[Catalogue, Galaxy, Photometry_rest],
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        n_jobs: int = 1,
    ) -> Optional[Union[Catalogue, Galaxy, Photometry_rest]]:
        pass
    
    @abstractmethod
    def _call_cat(
        self: Self,
        cat: Catalogue,
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        n_jobs: int = 1,
    ) -> Optional[Catalogue]:
        pass

    @abstractmethod
    def _call_gal(
        self: Self,
        gal: Galaxy,
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        save_dir: str = ""
    ) -> Optional[Galaxy]:
        pass


class Property_Calculator(Property_Calculator_Base):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
    ) -> None:
        self.aper_diam = aper_diam

    @property
    def full_name(self: Self) -> str:
        return f"{self.name}_{self.aper_diam.to(u.arcsec).value:.2f}as"

    @property
    @abstractmethod
    def name(self: Self) -> str:
        pass

    @property
    @abstractmethod
    def plot_name(self: Self) -> str:
        pass

    @abstractmethod
    def __call__(
        self: Self,
        object: Union[Catalogue, Galaxy, Photometry_rest],
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        n_jobs: int = 1,
    ) -> Optional[Union[Catalogue, Galaxy, Photometry_rest]]:
        pass
    
    @abstractmethod
    def _call_cat(
        self: Self,
        cat: Catalogue,
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        n_jobs: int = 1,
    ) -> Optional[Catalogue]:
        pass

    # # Not sure if this should be abstract or not
    # @abstractmethod
    # @staticmethod
    # def _call_gal_multi_process(params: Dict[str, Any]) -> NoReturn:
    #     self, gal, n_chains, overwrite, save_dir = params
    #     pass
    #     # return self._call_gal(gal, n_chains = n_chains, output = True, \
    #     #     overwrite = overwrite, save_dir = save_dir)
    
    # @abstractmethod
    # def _update_fits_cat(
    #     self: Self,
    #     cat: Catalogue,
    # ) -> NoReturn:
    #     pass

    @abstractmethod
    def _call_gal(
        self: Self,
        gal: Galaxy,
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        save_dir: str = ""
    ) -> Optional[Galaxy]:
        pass

    # @abstractmethod
    # def _calculate(
    #     self: Self,
    #     object: Union[Catalogue, Galaxy, Photometry_rest, SED_result],
    # ) -> Optional[Union[u.Quantity, u.Magnitude, u.Dex]]:
    #     pass

# 
class Property_Extractor:

    def __call__(
        self: Self,
        object: Union[Catalogue, Galaxy, Photometry_rest, SED_obs, SED_result],
        #save: bool = True,
    ) -> Optional[Catalogue]:
        # call the super with n_jobs = 1
        return super().__call__(object)

    def _call_cat(
        self: Self,
        cat: Catalogue,
        #save: bool = True,
    ) -> Optional[Catalogue]:
        # call the super with n_jobs = 1
        return super()._call_cat(cat)
    
    def _call_gal(
        self: Self,
        gal: Galaxy,
    ) -> Optional[Galaxy]:
        # call the super with n_jobs = 1
        return super()._call_gal(gal)
    
    def _call_single(
        self: Self,
        object: Union[Photometry_rest, SED_obs],
    ) -> Optional[Union[Catalogue, Galaxy, Photometry_rest, SED_obs]]:
        return super()._call_single(object)

# Calculates properties from observed frame photometry (e.g. colours, magnitudes)
class Photometry_Property_Calculator(Property_Calculator):
    
    @property
    @abstractmethod
    def name(self: Self) -> str:
        pass

    @property
    @abstractmethod
    def plot_name(self: Self) -> str:
        pass

    @abstractmethod
    def __call__(
        self: Self,
        object: Union[Catalogue, Galaxy, Photometry_rest],
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        n_jobs: int = 1,
    ) -> Optional[Union[Catalogue, Galaxy, Photometry_rest]]:
        pass
    
    @abstractmethod
    def _call_cat(
        self: Self,
        cat: Catalogue,
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        n_jobs: int = 1,
    ) -> Optional[Catalogue]:
        pass

    # # Not sure if this should be abstract or not
    # @abstractmethod
    # @staticmethod
    # def _call_gal_multi_process(params: Dict[str, Any]) -> NoReturn:
    #     self, gal, n_chains, overwrite, save_dir = params
    #     pass
    #     # return self._call_gal(gal, n_chains = n_chains, output = True, \
    #     #     overwrite = overwrite, save_dir = save_dir)
    
    # @abstractmethod
    # def _update_fits_cat(
    #     self: Self,
    #     cat: Catalogue,
    # ) -> NoReturn:
    #     pass

    @abstractmethod
    def _call_gal(
        self: Self,
        gal: Galaxy,
        n_chains: int = 10_000,
        output: bool = False,
        overwrite: bool = False,
        save_dir: str = ""
    ) -> Optional[Galaxy]:
        pass

# # Extracts properties from observed frame photometry (e.g. magnitudes)
# class Photometry_Property_Extractor(Property_Extractor, Photometry_Property_Calculator):
    
#     @property
#     @abstractmethod
#     def name(self: Self) -> str:
#         pass

#     def __call__(
#         self: Self,
#         object: Union[Catalogue, Galaxy, Photometry_rest],
#         n_chains: int = 10_000,
#         output: bool = False,
#         overwrite: bool = False,
#         n_jobs: int = 1,
#     ) -> Optional[Union[Catalogue, Galaxy, Photometry_rest]]:
#         raise NotImplementedError("This method should be implemented in the subclass")
    
#     def _call_cat(
#         self: Self,
#         cat: Catalogue,
#         n_chains: int = 10_000,
#         output: bool = False,
#         overwrite: bool = False,
#         n_jobs: int = 1,
#     ) -> Optional[Catalogue]:
#         raise NotImplementedError("This method should be implemented in the subclass")

#     # # Not sure if this should be abstract or not
#     # @abstractmethod
#     # @staticmethod
#     # def _call_gal_multi_process(params: Dict[str, Any]) -> NoReturn:
#     #     self, gal, n_chains, overwrite, save_dir = params
#     #     pass
#     #     # return self._call_gal(gal, n_chains = n_chains, output = True, \
#     #     #     overwrite = overwrite, save_dir = save_dir)
    
#     # @abstractmethod
#     # def _update_fits_cat(
#     #     self: Self,
#     #     cat: Catalogue,
#     # ) -> NoReturn:
#     #     pass

#     def _call_gal(
#         self: Self,
#         gal: Galaxy,
#         n_chains: int = 10_000,
#         output: bool = False,
#         overwrite: bool = False,
#         save_dir: str = ""
#     ) -> Optional[Galaxy]:
#         raise NotImplementedError("This method should be implemented in the subclass")


class Morphology_Property_Calculator(Property_Calculator_Base):

    def __init__(
        self: Self,
        morph_fitter: Morphology_Fitter
    ) -> None:
        self.morph_fitter = morph_fitter

    @property
    def cutout_label(self: Self) -> str:
        return self.morph_fitter.psf.cutout.band_data.filt_name + \
            f"_{self.morph_fitter.psf.cutout.cutout_size.to(u.arcsec).value:.2f}as"
    
    @property
    def full_name(self: Self) -> str:
        return f"{self.name}_{self.morph_fitter.name}"

    @property
    @abstractmethod
    def name(self: Self) -> str:
        pass

    @property
    @abstractmethod
    def plot_name(self: Self) -> str:
        pass

    def __call__(
        self: Self,
        object: Union[Catalogue, Galaxy, Type[Band_Cutout_Base]],
    ) -> Optional[Union[Catalogue, Galaxy, Type[Band_Cutout_Base]]]:
        from . import Band_Cutout_Base
        if isinstance(object, Catalogue):
            val = self._call_cat(object)
        elif isinstance(object, Galaxy):
            val = self._call_gal(object)
        elif isinstance(object, tuple(Band_Cutout_Base.__subclasses__())):
            val = self._call_sed_result(object)
        else:
            err_message = f"{object=} with {type(object)=} " + \
                f"not in [Catalogue, Galaxy, Band_Cutout_Base]"
            galfind_logger.critical(err_message)
            raise TypeError(err_message)
        return val
    
    def _call_cat(
        self: Self,
        cat: Catalogue,
        #save: bool = True,
    ) -> Optional[Catalogue]:
        pass
        #return np.array([self._call_gal(gal) for gal in cat])
    
    def _call_gal(
        self: Self,
        gal: Galaxy,
    ) -> Optional[Galaxy]:
        pass
        # # update the relevant Photometry_rest object stored in the Galaxy
        # assert self.aper_diam in gal.aper_phot.keys(), \
        #     galfind_logger.critical(
        #         f"{self.aper_diam=} not in {gal.aper_phot.keys()}"
        #     )
        # assert self.SED_fit_label in gal.aper_phot[self.aper_diam].SED_results.keys(), \
        #     galfind_logger.critical(
        #         f"{self.SED_fit_label=} not in " + \
        #         gal.aper_phot[self.aper_diam].SED_results.keys()
        #     )
        # return self._call_cutout(gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label])
    
    def _call_cutout(
        self: Self,
        cutout: Type[Band_Cutout_Base],
    ) -> Optional[Type[Band_Cutout_Base]]:
        pass
        #return getattr(cutout, self.name)
    
    def extract_vals(
        self: Self, 
        object: Union[Catalogue, Galaxy, Type[Band_Cutout_Base]]
    ) -> Union[u.Quantity, u.Magnitude, u.Dex]:
        from . import Band_Cutout_Base
        if isinstance(object, Catalogue):
            cat_vals = [getattr(gal.cutouts[self.cutout_label].morph_fits[self.morph_fitter.name], self.name) for gal in object]
            if not all(isinstance(val, float) for val in cat_vals):
                assert all(val.unit == cat_vals[0].unit for val in cat_vals), \
                    galfind_logger.critical(f"Units of {self.name} in {object} are not consistent")
                cat_vals = np.array([val.value for val in cat_vals]) * cat_vals[0].unit
            else:
                cat_vals = np.array(cat_vals)
            return cat_vals
        elif isinstance(object, Galaxy):
            return getattr(object.cutouts[self.cutout_label].morph_fits[self.morph_fitter.name], self.name)
        elif isinstance(object, tuple(Band_Cutout_Base.__subclasses__())):
            return getattr(object, self.name)
        else:
            err_message = f"{object=} with {type(object)=} " + \
                f"not in [Catalogue, Galaxy, Band_Cutout_Base]"
            galfind_logger.critical(err_message)
            raise TypeError(err_message)
        
    def extract_PDFs(
        self: Self,
        object: Union[Catalogue, Galaxy, Type[Band_Cutout_Base]],
    ) -> Union[Type[PDF], List[Type[PDF]]]:
        from . import Band_Cutout_Base
        if isinstance(object, Catalogue):
            return [gal.cutouts[self.cutout_label].morph_fits[self.morph_fitter.name].property_PDFs[self.name] for gal in object]
        elif isinstance(object, Galaxy):
            return object.cutouts[self.cutout_label].morph_fits[self.morph_fitter.name].property_PDFs[self.name]
        elif isinstance(object, tuple(Band_Cutout_Base.__subclasses__())):
            return object.property_PDFs[self.name]
        else:
            err_message = f"{object=} with {type(object)=} " + \
                f"not in [Catalogue, Galaxy, Band_Cutout_Base]"
            galfind_logger.critical(err_message)
            raise TypeError(err_message)


class Custom_Morphology_Property_Extractor(Property_Extractor, Morphology_Property_Calculator):

    def __init__(
        self: Self,
        name: str,
        plot_name: str,
        morph_fitter: Morphology_Fitter,
    ) -> None:
        self._name = name
        self._plot_name = plot_name
        super().__init__(morph_fitter)
    
    @property
    def name(self: Self) -> str:
        return self._name
    
    @property
    def plot_name(self: Self) -> str:
        return self._plot_name
    

# Calculates additional properties from best fitting rest frame SED
class SED_Property_Calculator(Property_Calculator):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
    ) -> None:
        if isinstance(SED_fit_label, SED_code):
            SED_fit_label = SED_fit_label.label
        self.SED_fit_label = SED_fit_label
        super().__init__(aper_diam)
    
    @property
    def full_name(self: Self) -> str:
        return f"{self.name}_{self.SED_fit_label}" + \
            f"_{self.aper_diam.to(u.arcsec).value:.2f}as"

    @property
    @abstractmethod
    def name(self: Self) -> str:
        pass

    @property
    @abstractmethod
    def plot_name(self: Self) -> str:
        pass

    def __call__(
        self: Self,
        object: Union[Catalogue, Galaxy, Photometry_rest],
    ) -> Optional[Union[Catalogue, Galaxy, Photometry_rest]]:
        # if isinstance(self, tuple(Property_Extractor.__subclasses__())):
        #     # call with n_jobs = 1
        #     n_jobs = 1
        #     n_chains = 1
        #     output = False
        #     overwrite = False
        # below should be in the super class
        # calculate pre-requisite properties first
        # [rest_frame_property(object, n_chains, output = False, \
        #     overwrite = overwrite, n_jobs = n_jobs) for \
        #     rest_frame_property in self.pre_req_properties]
        if isinstance(object, Catalogue):
            val = self._call_cat(object)
        elif isinstance(object, Galaxy):
            val = self._call_gal(object)
        elif isinstance(object, SED_result):
            val = self._call_sed_result(object)
        else:
            err_message = f"{object=} with {type(object)=} " + \
                f"not in [Catalogue, Galaxy, SED_result]"
            galfind_logger.critical(err_message)
            raise TypeError(err_message)
        return val
    
    def _call_cat(
        self: Self,
        cat: Catalogue,
        #save: bool = True,
    ) -> Optional[Catalogue]:
        # assert isinstance(n_jobs, int), \
        #     galfind_logger.critical(
        #         f"{n_jobs=} with {type(n_jobs)=} != int"
        #     )
        # if save:
        # save_dir = f"{config['PhotProperties']['PDF_SAVE_DIR']}/" + \
        #     f"{cat.version}/{cat.survey}/{cat.filterset.instrument_name}/" + \
        #     f"{self.aper_diam.to(u.arcsec).value:.2f}as" + \
        #     f"/{self.SED_fit_label}/{self.name}"
        # if hasattr(self, "n_jobs"):
        #     n_jobs = self.n_jobs
        #if n_jobs <= 1:
            # update properties for each galaxy in the catalogue
        return np.array([self._call_gal(gal) for gal in cat])
        # else:
        #     # TODO: should be set when serializing the object
        #     for gal in tqdm(cat, total = len(cat)):
        #         for label in gal.aper_phot[self.aper_diam].SED_results.keys():
        #             try:
        #                 gal.aper_phot[self.aper_diam].flux = \
        #                     gal.aper_phot[self.aper_diam].flux.unmasked
        #             except:
        #                 pass
        #             try:
        #                 gal.aper_phot[self.aper_diam].flux_errs = \
        #                     gal.aper_phot[self.aper_diam].flux_errs.unmasked
        #             except:
        #                 pass
        #             try:
        #                 gal.aper_phot[self.aper_diam].SED_results[label].phot_rest.flux = \
        #                     gal.aper_phot[self.aper_diam].SED_results[label].phot_rest.flux.unmasked
        #             except:
        #                 pass
        #             try:
        #                 gal.aper_phot[self.aper_diam].SED_results[label].phot_rest.flux_errs = \
        #                     gal.aper_phot[self.aper_diam].SED_results[label].phot_rest.flux_errs.unmasked
        #             except:
        #                 pass
        #     # multi-process with joblib
        #     # sort input params
        #     params_arr = [(self, gal, n_chains, overwrite, save_dir) for gal in cat]
        #     # run in parallel
        #     with funcs.tqdm_joblib(tqdm(desc = f"Calculating {self.name} for " + \
        #         f"{cat.survey} {cat.version} {cat.filterset.instrument_name}", \
        #         total = len(cat))) as progress_bar:
        #             with parallel_config(backend='loky', n_jobs=n_jobs):
        #                 gals = Parallel()(delayed( \
        #                     self._call_gal_multi_process)(params) \
        #                     for params in params_arr
        #                 )
        #     cat.gals = gals
        # if cat.cat_creator.crops == []:
        #     self._update_fits_cat(cat)
        # if output:
        #     return cat

    def _call_gal(
        self: Self,
        gal: Galaxy,
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
        # if save_dir != "":
        #     save_dir += "/"
        # save_path = f"{save_dir}{gal.ID}.npy"
        # self._call_phot_rest(gal.aper_phot[self.aper_diam]. \
        #     SED_results[self.SED_fit_label].phot_rest, n_chains = n_chains, 
        #     output = False, overwrite = overwrite, save_path = save_path)
        return self._call_sed_result(gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label])
        # if output:
        #     return gal
    
    def _call_sed_result(
        self: Self,
        sed_result: SED_result,
    ) -> Optional[SED_result]:
        # assert isinstance(n_chains, int), \
        #     galfind_logger.critical(
        #         f"{n_chains=} with {type(n_chains)=} != int"
        #     )
        # if save_path != "":
        #     save_path += "/"
        # save_path += f"{sed_result.label}.npy"
        # self._call_phot_rest(sed_result.phot_rest, n_chains = n_chains, 
        #     output = False, overwrite = overwrite, save_path = save_path)
        return getattr(sed_result, self.name)
        # if output:
        #     return sed_result
    
    def extract_vals(
        self: Self, 
        object: Union[Catalogue, Galaxy, SED_obs]
    ) -> Union[u.Quantity, u.Magnitude, u.Dex]:
        if isinstance(object, Catalogue):
            cat_vals = [getattr(gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label], self.name) for gal in object]
            if not all(isinstance(val, float) for val in cat_vals):
                assert all(val.unit == cat_vals[0].unit for val in cat_vals), \
                    galfind_logger.critical(f"Units of {self.name} in {object} are not consistent")
                cat_vals = np.array([val.value for val in cat_vals]) * cat_vals[0].unit
            else:
                cat_vals = np.array(cat_vals)
            return cat_vals
        elif isinstance(object, Galaxy):
            return getattr(object.aper_phot[self.aper_diam].SED_results[self.SED_fit_label], self.name)
        elif isinstance(object, SED_obs):
            return getattr(object, self.name)
        else:
            err_message = f"{object=} with {type(object)=} " + \
                f"not in [Catalogue, Galaxy, SED_obs]"
            galfind_logger.critical(err_message)
            raise TypeError(err_message)
    
    # TODO: Propagate from parent class
    def extract_errs(
        self: Self, 
        object: Union[Catalogue, Galaxy, SED_obs]
    ) -> Union[u.Quantity, u.Magnitude, u.Dex]:
        if isinstance(object, Catalogue):
            cat_errs = [gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label].property_errs[self.name] for gal in object]
            # if not all(isinstance(val, float) for val in cat_errs):
            #     assert all(val.unit == cat_errs[0].unit for val in cat_errs), \
            #         galfind_logger.critical(f"Units of {self.name} in {object} are not consistent")
            #     breakpoint()
            #     cat_errs = np.array([val.value for val in cat_errs]) * cat_errs[0].unit
            # else:
            #     cat_errs = np.array(cat_errs)
            return cat_errs
        raise NotImplementedError()
        # elif isinstance(object, Galaxy):
        #     return getattr(object.aper_phot[self.aper_diam].SED_results[self.SED_fit_label], self.name)
        # elif isinstance(object, SED_obs):
        #     return getattr(object, self.name)
        # else:
        #     err_message = f"{object=} with {type(object)=} " + \
        #         f"not in [Catalogue, Galaxy, SED_obs]"
        #     galfind_logger.critical(err_message)
        #     raise TypeError(err_message)
        
    def extract_PDFs(
        self: Self,
        object: Union[Catalogue, Galaxy, SED_obs],
    ) -> Union[Type[PDF], List[Type[PDF]]]:
        if isinstance(object, Catalogue):
            return [gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label].property_PDFs[self.name] for gal in object]
        elif isinstance(object, Galaxy):
            return object.aper_phot[self.aper_diam].SED_results[self.SED_fit_label].property_PDFs[self.name]
        elif isinstance(object, SED_obs):
            return object.property_PDFs[self.name]
        else:
            err_message = f"{object=} with {type(object)=} " + \
                f"not in [Catalogue, Galaxy, SED_obs]"
            galfind_logger.critical(err_message)
            raise TypeError(err_message)
        

class Ext_Src_Property_Calculator(SED_Property_Calculator):

    def __init__(
        self: Self,
        property_name: str,
        plot_name: str,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
        ext_src_corrs: Optional[str] = "UV",
        ext_src_uplim: Optional[Union[int, float]] = 10.0,
        ref_wav: float = 1_500.0 * u.AA,
    ) -> None:
        self.ext_src_corrs = ext_src_corrs
        self.ext_src_uplim = ext_src_uplim
        self.ref_wav = ref_wav
        if self.ext_src_corrs is not None:
            assert self.ext_src_corrs in ["UV"] + all_band_names
        if self.ext_src_uplim is not None:
            assert isinstance(self.ext_src_uplim, (int, float))
            assert self.ext_src_uplim > 0.0
        assert u.get_physical_type(self.ref_wav) == "length"
        self.property_name = property_name
        self._plot_name = plot_name
        super().__init__(aper_diam, SED_fit_label)
    
    @property
    def name(self: Self) -> str:
        ext_src_label = f"_extsrc_{self.ext_src_corrs}" \
            if self.ext_src_corrs is not None else ""
        ext_src_lim_label = f"<{self.ext_src_uplim:.0f}" if \
            self.ext_src_uplim is not None and \
            self.ext_src_corrs is not None else ""
        return self.property_name + ext_src_label + ext_src_lim_label
    
    @property
    def plot_name(self: Self) -> str:
        return self._plot_name

    def _call_cat(
        self: Self,
        cat: Catalogue,
        #save: bool = True,
    ) -> Optional[Catalogue]:
        cat.load_sextractor_ext_src_corrs()
        return np.array([self._call_gal(gal) for gal in cat])

    def _call_gal(
        self: Self,
        gal: Galaxy,
    ) -> Optional[Galaxy]:
        # TODO: Ensure the extended source corrections have already been loaded
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
        if self.ext_src_corrs == "UV":
            # calculate band nearest to the rest frame UV reference wavelength
            band_wavs = [filt.WavelengthCen.to(u.AA).value \
                for filt in gal.aper_phot[self.aper_diam].filterset] * u.AA / (1. + gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label].z)
            ref_band = gal.aper_phot[self.aper_diam].filterset.band_names[np.argmin(np.abs(band_wavs - self.ref_wav))]
            ext_src_corr = gal.aper_phot[self.aper_diam].ext_src_corrs[ref_band]
        else: # band given
            ext_src_corr = gal.aper_phot[self.aper_diam].ext_src_corrs[self.ext_src_corrs]
        # apply limit to extended source correction
        if self.ext_src_uplim is not None:
            if ext_src_corr > self.ext_src_uplim:
                ext_src_corr = self.ext_src_uplim
        if ext_src_corr < 1.0:
            ext_src_corr = 1.0
        return self._call_sed_result(gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label], ext_src_corr)

    def _call_sed_result(
        self: Self,
        sed_result: SED_result,
        ext_src_corr: Optional[Union[int, float]] = None,
    ) -> Optional[SED_result]:
        # extract the relevant PDF
        # load calculated PDF into the SED_result object
        old_pdf = deepcopy(sed_result.property_PDFs[self.property_name])
        if isinstance(old_pdf.input_arr, u.Magnitude):
            assert old_pdf.input_arr.unit.is_equivalent(u.ABmag)
            new_arr = old_pdf.input_arr.value - 2.5 * np.log10(ext_src_corr)
        elif isinstance(old_pdf.input_arr, u.Dex):
            new_arr = old_pdf.input_arr.value + np.log10(ext_src_corr)
        else:
            new_arr = old_pdf.input_arr.value * ext_src_corr
        new_arr = new_arr * old_pdf.input_arr.unit
        new_pdf = SED_fit_PDF.from_1D_arr(
            self.name, 
            new_arr, 
            old_pdf.SED_fit_params
        )
        setattr(sed_result, self.name, new_pdf.median)
        sed_result.property_PDFs[self.name] = new_pdf
        # TODO: save raw data at this point
        return sed_result


class Custom_SED_Property_Extractor(Property_Extractor, SED_Property_Calculator):

    def __init__(
        self: Self,
        name: str,
        plot_name: str,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, Type[SED_code]],
    ) -> None:
        self._name = name
        self._plot_name = plot_name
        super().__init__(aper_diam, SED_fit_label)
    
    @property
    def name(self: Self) -> str:
        return self._name
    
    @property
    def plot_name(self: Self) -> str:
        return self._plot_name


class Redshift_Extractor(Property_Extractor, SED_Property_Calculator):
    
    @property
    def name(self: Self) -> str:
        return "z"
    
    @property
    def plot_name(self: Self) -> str:
        return "Redshift, z"


class Multiple_SED_Property_Calculator(Property_Calculator_Base):

    def __init__(
        self: Self,
        calculators: List[Type[Property_Calculator_Base]],
        name: Optional[str],
        plot_name: Optional[str],
    ) -> None:
        self._name = name
        self._plot_name = plot_name
        self.calculators = calculators

    @property
    @abstractmethod
    def full_name(self: Self) -> str:
        pass

    @property
    @abstractmethod
    def name(self: Self) -> str:
        pass

    @property
    @abstractmethod
    def plot_name(self: Self) -> str:
        pass

    def __call__(
        self: Self,
        object: Union[Catalogue, Galaxy, Photometry_rest, SED_obs, Type[Band_Cutout_Base]],
    ) -> Optional[Union[Catalogue, Galaxy, Photometry_rest, SED_obs, Type[Band_Cutout_Base]]]:
        if isinstance(object, Catalogue):
            val = self._call_cat(object)
        elif isinstance(object, Galaxy):
            val = self._call_gal(object)
        elif isinstance(object, tuple(Photometry_rest, SED_obs) + tuple(Band_Cutout_Base.__subclasses__())):
            val = self._call_single(object)
        else:
            err_message = f"{object=} with {type(object)=} " + \
                f"not in [Catalogue, Galaxy, Photometry_rest, SED_obs, Type[Band_Cutout_Base]]"
            galfind_logger.critical(err_message)
            raise TypeError(err_message)
        return val
    
    def _call_cat(
        self: Self,
        cat: Catalogue,
        #save: bool = True,
    ) -> Optional[Catalogue]:
        return np.array([self._call_gal(gal) for gal in cat])
    
    @abstractmethod
    def _call_gal(
        self: Self,
        gal: Galaxy,
    ) -> Optional[Galaxy]:
        pass

    # @abstractmethod
    # def _call_single(
    #     self: Self,
    #     object: Union[Photometry_rest, SED_obs, Type[Band_Cutout_Base]],
    # ) -> Optional[Union[Photometry_rest, SED_obs, Type[Band_Cutout_Base]]]:
    #     pass


class Property_Multiplier(Multiple_SED_Property_Calculator):

    def __init__(
        self: Self,
        calculators: List[Type[Property_Calculator_Base]],
        name: Optional[str] = None,
        plot_name: Optional[str] = None,
    ) -> None:
        super().__init__(calculators, name, plot_name)

    @property
    def full_name(self: Self) -> str:
        suffixes = [calculator.full_name.replace(calculator.name, "") for calculator in self.calculators]
        if all(suffix == suffixes[0] for suffix in suffixes):
            # add suffix to end of string
            return f"{'_mult_'.join([calculator.name for calculator in self.calculators])}{suffixes[0]}"
        else:
            return "_mult_".join([calculator.full_name for calculator in self.calculators])
    
    @property
    def name(self: Self) -> str:
        pass
        # if self._name is None:
        #     return "*".join([calculator.name for calculator in self.calculators])
        # else:
        #     return self._name
    
    @property
    def plot_name(self: Self) -> str:
        if self._plot_name is None:
            return "".join([calculator.plot_name for calculator in self.calculators])
        else:
            return self._plot_name

    def extract_vals(
        self: Self, 
        object: Union[Catalogue, Galaxy, Photometry_rest, SED_obs, Type[Band_Cutout_Base]]
    ) -> Union[u.Quantity, u.Magnitude, u.Dex]:
        # calculate relevant values using each calculator
        vals = [calculator.extract_vals(object) for calculator in self.calculators]
        breakpoint()

    def extract_PDFs(
        self: Self,
        object: Union[Catalogue, Galaxy, Photometry_rest, SED_obs, Type[Band_Cutout_Base]],
    ) -> Union[Type[PDF], List[Type[PDF]]]:
        # calculate relevant PDFs using each calculator
        PDFs = [calculator.extract_PDFs(object) for calculator in self.calculators]
        breakpoint()

    #@abstractmethod
    def _call_gal(
        self: Self,
        gal: Galaxy,
    ) -> Optional[Galaxy]:
        raise NotImplementedError

class Property_Divider(Multiple_SED_Property_Calculator):

    def __init__(
        self: Self,
        calculators: List[Type[Property_Calculator_Base]],
        name: Optional[str] = None,
        plot_name: Optional[str] = None,
    ) -> None:
        assert len(calculators) == 2
        super().__init__(calculators, name, plot_name)

    @property
    def full_name(self: Self) -> str:
        suffixes = [calculator.full_name.replace(calculator.name, "") for calculator in self.calculators]
        if all(suffix == suffixes[0] for suffix in suffixes):
            # add suffix to end of string
            return f"{'_div_'.join([calculator.name for calculator in self.calculators])}{suffixes[0]}"
        else:
            return "_div_".join([calculator.full_name for calculator in self.calculators])
    
    @property
    def name(self: Self) -> str:
        pass
        # if self._name is None:
        #     return "/".join([calculator.name for calculator in self.calculators])
        # else:
        #     return self._name
    
    @property
    def plot_name(self: Self) -> str:
        # TODO: Improve this so that units appear at the end of the string
        if self._plot_name is None:
            return "/".join([calculator.plot_name for calculator in self.calculators])
        else:
            return self._plot_name

    def extract_vals(
        self: Self, 
        object: Union[Catalogue, Galaxy, Photometry_rest, SED_obs, Type[Band_Cutout_Base]]
    ) -> Union[u.Quantity, u.Magnitude, u.Dex]:
        # calculate relevant values using each calculator
        vals_arr = [calculator.extract_vals(object) for calculator in self.calculators]
        # remove dex units
        vals_arr = [vals.physical if isinstance(vals, u.Dex) else vals for vals in vals_arr]
        return vals_arr[0] / vals_arr[1]

    def extract_PDFs(
        self: Self,
        object: Union[Catalogue, Galaxy, Photometry_rest, SED_obs, Type[Band_Cutout_Base]],
    ) -> Union[Type[PDF], List[Type[PDF]]]:
        from . import PDF
        # calculate relevant PDFs using each calculator
        pdfs_arr = [calculator.extract_PDFs(object) for calculator in self.calculators]
        # remove dex units
        pdf_input_arrs = [[pdf.input_arr.physical if isinstance(pdf.input_arr, u.Dex) else pdf.input_arr for pdf in pdfs] for pdfs in pdfs_arr]
        return PDF.from_1D_arr(
            self.name, 
            pdf_input_arrs[0] / pdf_input_arrs[1], 
            #{**pdfs[0].SED_fit_params, **pdfs[1].SED_fit_params}
        )

    #@abstractmethod
    def _call_gal(
        self: Self,
        gal: Galaxy,
    ) -> Optional[Galaxy]:
        raise NotImplementedError

# class Property_Adder(Multiple_SED_Property_Calculator):

#     def __init__(
#         self: Self,
#         calculators: List[Type[Property_Calculator_Base]],
#     ) -> None:
#         assert len(calculators) == 2
#         super().__init__(calculators)

#     def extract_vals(
#         self: Self, 
#         object: Union[Catalogue, Galaxy, Photometry_rest, SED_obs, Type[Band_Cutout_Base]]
#     ) -> Union[u.Quantity, u.Magnitude, u.Dex]:
#         # calculate relevant values using each calculator
#         vals = [calculator.extract_vals(object) for calculator in self.calculators]
#         breakpoint()

#     def extract_PDFs(
#         self: Self,
#         object: Union[Catalogue, Galaxy, Photometry_rest, SED_obs, Type[Band_Cutout_Base]],
#     ) -> Union[Type[PDF], List[Type[PDF]]]:
#         # calculate relevant PDFs using each calculator
#         PDFs = [calculator.extract_PDFs(object) for calculator in self.calculators]
#         breakpoint()

# class Property_Subtractor(Multiple_SED_Property_Calculator):

#     def __init__(
#         self: Self,
#         calculators: List[Type[Property_Calculator_Base]],
#     ) -> None:
#         assert len(calculators) == 2
#         super().__init__(calculators)

#     def extract_vals(
#         self: Self, 
#         object: Union[Catalogue, Galaxy, Photometry_rest, SED_obs, Type[Band_Cutout_Base]]
#     ) -> Union[u.Quantity, u.Magnitude, u.Dex]:
#         # calculate relevant values using each calculator
#         vals = [calculator.extract_vals(object) for calculator in self.calculators]
#         breakpoint()

#     def extract_PDFs(
#         self: Self,
#         object: Union[Catalogue, Galaxy, Photometry_rest, SED_obs, Type[Band_Cutout_Base]],
#     ) -> Union[Type[PDF], List[Type[PDF]]]:
#         # calculate relevant PDFs using each calculator
#         PDFs = [calculator.extract_PDFs(object) for calculator in self.calculators]
#         breakpoint()


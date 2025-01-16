
from __future__ import annotations

from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
import astropy.units as u
from typing import TYPE_CHECKING, Dict, Any, List, Union, Tuple, Optional, NoReturn
if TYPE_CHECKING:
    from . import Catalogue, Galaxy, Photometry_rest, SED_code, SED_obs, PDF
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import config, galfind_logger
from . import useful_funcs_austind as funcs
from . import Catalogue, Galaxy, SED_code, SED_result

class Property_Calculator(ABC):

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
        return f"{self.name}_{self.SED_fit_label}_{self.aper_diam.to(u.arcsec).value:.2f}as"

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


from __future__ import annotations

from abc import ABC, abstractmethod
import astropy.units as u
import numpy as np
from copy import deepcopy
import json
from astropy.coordinates import SkyCoord
import h5py
from astropy.io import fits
from pathlib import Path
from astropy.utils.masked import Masked
from scipy import ndimage
from tqdm import tqdm
import logging

from typing import TYPE_CHECKING, Any, List, Union, NoReturn, Callable, Optional
if TYPE_CHECKING:
    from . import (
        Multiple_Filter,
        Rest_Frame_Property_Calculator,
        Morphology_Fitter,
        Property_Calculator,
        Data,
    )
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import useful_funcs_austind as funcs
from . import galfind_logger, config, wav_lyman_lim
from . import Galaxy, Catalogue, Catalogue_Base, Instrument, SED_code
from .Instrument import expected_instr_bands
from .Morphology import fwhm_nircam

class Selector(ABC):

    def __init__(
        self: Self,
        aper_diam: Optional[u.Quantity],
        SED_fit_label: Optional[str],
        morph_fitter: Optional[Type[Morphology_Fitter]],
        **kwargs
    ):
        assert (key in kwargs.keys() for key in self._include_kwargs), \
            galfind_logger.critical(
                f"Selection {self.__class__.__name__} given {kwargs=}" + \
                f" missing required keys = {self._include_kwargs}."
            )
        self.aper_diam = aper_diam
        self.SED_fit_label = SED_fit_label
        self.morph_fitter = morph_fitter
        self.kwargs = kwargs
        assert self._assertions()

    def __repr__(self: Self) -> str:
        return f"{self.__class__.__name__}({','.join([repr(kwarg).replace(' ', '') for kwarg in self.kwargs.values()])})"
    
    def __str__(self: Self) -> str:
        output_str = funcs.line_sep
        output_str += f"{self.__class__.__name__}:\n"
        output_str += funcs.line_sep
        for key, val in self.kwargs.items():
            output_str += f"{key}: {val}\n"
        output_str += funcs.line_sep
        return output_str

    @property
    @abstractmethod
    def name(self: Self) -> str:
        pass

    @property
    @abstractmethod
    def _selection_name(self) -> str:
        pass

    @property
    @abstractmethod
    def _include_kwargs(self) -> List[str]:
        pass

    @abstractmethod
    def _assertions(self: Self) -> bool:
        pass

    #@abstractmethod
    def _failure_criteria(
        self: Self,
        gal: Galaxy,
    ) -> bool:
        # always pass by default
        return False

    @abstractmethod
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
    ) -> bool:
        pass
        
    @abstractmethod
    def _check_phot_exists(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> bool:
        pass
    
    @abstractmethod
    def _check_SED_fit_exists(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> bool:
        pass

    @abstractmethod
    def _check_morph_fit_exists(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> bool:
        pass

    def __call__(
        self: Self,
        object: Union[Galaxy, Type[Catalogue_Base]],
        return_copy: bool = True,
        *args,
        **kwargs,
    ) -> Optional[Union[Galaxy, Type[Catalogue_Base]]]:
        if isinstance(object, Galaxy):
            obj = self._call_gal(object, return_copy, *args, **kwargs)
        elif isinstance(object, tuple(Catalogue_Base.__subclasses__())):
            obj = self._call_cat(object, return_copy, *args, **kwargs)
        else:
            raise ValueError(
                f"{object=} must be either a Galaxy or Catalogue object."
            )
        if obj is not None:
            return obj
    
    def _call_gal(
        self: Self, 
        gal: Galaxy,
        return_copy: bool = True,
        *args,
        **kwargs,
    ) -> Union[NoReturn, Galaxy]:
        if return_copy:
            gal_ = deepcopy(gal)
        else:
            gal_ = gal
        selection_name = self.name
        if not selection_name in gal_.selection_flags.keys():
            if self._failure_criteria(gal_, *args, **kwargs) \
                    or not self._check_phot_exists(gal_) \
                    or not self._check_SED_fit_exists(gal_) \
                    or not self._check_morph_fit_exists(gal_):
                gal_.selection_flags[self.name] = False
            else:
                if self._selection_criteria(gal_, *args, **kwargs):
                    gal_.selection_flags[self.name] = True
                else:
                    gal_.selection_flags[self.name] = False
        if return_copy:
            return gal_

    def _call_cat(
        self: Self,
        cat: Catalogue,
        return_copy: bool = True,
        *args,
        **kwargs,
    ) -> Union[NoReturn, Catalogue]:
        if self.SED_fit_label is not None:
            # ensure results have been loaded for 
            # at least 1 galaxy in the catalogue
            assert any(self._check_SED_fit_exists(gal) for gal in cat), \
                galfind_logger.critical(
                    f"SED fitting results for {self.SED_fit_label=} " + \
                    f"not loaded for any galaxy in {repr(cat)}."
                )
        if self.morph_fitter is not None:
            # ensure results have been loaded for 
            # at least 1 galaxy in the catalogue
            assert any(self._check_morph_fit_exists(gal) for gal in cat), \
                galfind_logger.critical(
                    f"Morphology fitting results for {repr(self.morph_fitter)=} " + \
                    f"not loaded for any galaxy in {repr(cat)}."
                )
        [
            self._call_gal(gal, return_copy = False, *args, **kwargs) for gal \
            in tqdm(cat, total = len(cat), desc = f"Selecting {self.name}", \
            disable = galfind_logger.getEffectiveLevel() > logging.INFO)
        ]
        if cat.cat_creator.crops == [] or self.__class__.__name__ != "ID_Selector":
            cat._append_property_to_tab(self.name, "SELECTION")
        if return_copy:
            cat_copy = deepcopy(cat)
            return cat_copy.crop(self)
        else:
            return cat
        

class Data_Selector(Selector, ABC):

    def __init__(self: Self, **kwargs) -> Self:
        super().__init__(aper_diam = None, SED_fit_label = None, morph_fitter = None, **kwargs)

    @property
    def requires_phot(self: Self) -> bool:
        return True

    @property
    def name(self: Self) -> str:
        return self._selection_name
    
    def _check_phot_exists(
        self: Self,
        *args,
        **kwargs,
    ) -> bool:
        return True
    
    def _check_SED_fit_exists(
        self: Self,
        *args,
        **kwargs,
    ) -> bool:
        return True

    def _check_morph_fit_exists(
        self: Self,
        *args,
        **kwargs,
    ) -> bool:
        return True
    
    def __call__(
        self: Self,
        object: Union[Galaxy, Catalogue],
        return_copy: bool = True,
        *args,
        **kwargs
    ) -> Optional[Union[Galaxy, Catalogue]]:
        return Selector.__call__(self, object, return_copy, *args, **kwargs)


class Photometry_Selector(Selector, ABC):

    def __init__(self: Self, aper_diam: u.Quantity, **kwargs) -> Self:
        assert isinstance(aper_diam, u.Quantity)
        assert aper_diam.unit.is_equivalent(u.arcsec)
        assert aper_diam > 0 * u.arcsec
        super().__init__(aper_diam, SED_fit_label = None, morph_fitter = None, **kwargs)

    @property
    def name(self: Self) -> str:
        return self._selection_name + \
            f"_{self.aper_diam.to(u.arcsec).value:.2f}as"

    def _check_phot_exists(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> bool:
        try:
            passed = len(gal.aper_phot[self.aper_diam]) != 0
        except:
            passed = False
        return passed
    
    def _check_SED_fit_exists(
        self: Self,
        *args,
        **kwargs
    ) -> bool:
        return True

    def _check_morph_fit_exists(
        self: Self,
        *args,
        **kwargs,
    ) -> bool:
        return True
    
    def __call__(
        self: Self,
        object: Union[Galaxy, Catalogue],
        return_copy: bool = True,
        *args,
        **kwargs
    ) -> Optional[Union[Galaxy, Catalogue]]:
        return Selector.__call__(self, object, return_copy)
    

class SED_fit_Selector(Selector, ABC):

    def __init__(
        self: Self, 
        aper_diam: u.Quantity, 
        SED_fit_label: Union[str, SED_code], 
        **kwargs
    ) -> Self:
        assert isinstance(aper_diam, u.Quantity)
        assert aper_diam.unit.is_equivalent(u.arcsec)
        assert aper_diam > 0 * u.arcsec
        if isinstance(SED_fit_label, tuple(SED_code.__subclasses__())):
            SED_fit_label = SED_fit_label.label
        else:
            assert isinstance(SED_fit_label, str), \
                galfind_logger.critical(
                    f"{SED_fit_label=} must be a string or SED_code object."
                )

        super().__init__(aper_diam, SED_fit_label, morph_fitter = None, **kwargs)

    @property
    def requires_SED_fit(self: Self) -> bool:
        return True

    @property
    def name(self: Self) -> str:
        return f"{self._selection_name}_{self.SED_fit_label}" + \
            f"_{self.aper_diam.to(u.arcsec).value:.2f}as"

    def _check_phot_exists(
        self: Self,
        gal: Galaxy,
    ) -> bool:
        try:
            passed = len(gal.aper_phot[self.aper_diam]) != 0
        except:
            passed = False
        return passed
    
    def _check_SED_fit_exists(
        self: Self,
        gal: Galaxy,
    ) -> bool:
        try:
            passed = self.SED_fit_label in gal.aper_phot[self.aper_diam].SED_results.keys()
        except:
            passed = False
        return passed
    
    def _check_morph_fit_exists(
        self: Self,
        *args,
        **kwargs,
    ) -> bool:
        return True
    
    def _assert_SED_fit_label(
        self: Self, 
        object: Union[Galaxy, Type[Catalogue_Base]],
    ) -> str:
        if isinstance(object, Galaxy):
            assert self.SED_fit_label in object.aper_phot[self.aper_diam].SED_results.keys(), \
                galfind_logger.critical(
                    f"SED fitting results for {self.SED_fit_label=} " + \
                    f"not loaded for {repr(object)}."
                )
        elif isinstance(object, tuple(Catalogue_Base.__subclasses__())):
            assert any(self.SED_fit_label in gal.aper_phot[self.aper_diam].\
                    SED_results.keys() for gal in object), \
                galfind_logger.critical(
                    f"SED fitting results for {self.SED_fit_label=} " + \
                    f"not loaded for any galaxy in {repr(object)}."
                )
        else:
            raise ValueError(
                f"{object=} with {type(object)=} not in ['Galaxy', 'Catalogue']!"
            )
    
    def __call__(
        self: Self,
        object: Union[Galaxy, Catalogue],
        return_copy: bool = True,
    ) -> Optional[Union[Galaxy, Catalogue]]:
        self._assert_SED_fit_label(object)
        return Selector.__call__(self, object, return_copy)
    

class Morphology_Selector(Selector, ABC):

    def __init__(
        self: Self,
        morph_fitter: Union[str, Morphology_Fitter],
        **kwargs,
    ) -> Self:
        from . import Morphology_Fitter
        assert isinstance(morph_fitter, tuple(Morphology_Fitter.__subclasses__()))
        super().__init__(aper_diam = None, SED_fit_label = None, morph_fitter = morph_fitter, **kwargs)

    @property
    def requires_SED_fit(self: Self) -> bool:
        return False

    @property
    def name(self: Self) -> str:
        morph_name = f"{self._cutout_str}_{self.morph_fitter.name}"
        return f"{self._selection_name}_{self.morph_fitter.name}"

    @property
    def _cutout_str(self: Self) -> str:
        filt_name = self.morph_fitter.psf.cutout.band_data.filt_name
        cutout_size = self.morph_fitter.psf.cutout.cutout_size
        return f"{filt_name}_{cutout_size.to(u.arcsec).value:.2f}as"

    def _check_phot_exists(
        self: Self,
        gal: Galaxy,
    ) -> bool:
        return True

    def _check_SED_fit_exists(
        self: Self,
        *args,
        **kwargs
    ) -> bool:
        return True
    
    def _check_morph_fit_exists(
        self: Self,
        gal: Galaxy,
    ) -> bool:
        try:
            passed = self.morph_fitter.name in gal.cutouts[self._cutout_str].morph_fits.keys()
        except:
            passed = False
        return passed
    
    def _assert_morph_fitter(
        self: Self, 
        object: Union[Galaxy, Type[Catalogue_Base]],
    ) -> str:
        if isinstance(object, Galaxy):
            assert self._check_morph_fit_exists(object), \
                galfind_logger.critical(
                    f"Morphology fitting results for {self.morph_fitter=} " + \
                    f"not loaded for {repr(object)}."
                )
        elif isinstance(object, tuple(Catalogue_Base.__subclasses__())):
            assert any(self._check_morph_fit_exists(gal) for gal in object), \
                galfind_logger.critical(
                    f"Morphology fitting results for {self.morph_fitter=} " + \
                    f"not loaded for any galaxy in {repr(object)}."
                )
        else:
            raise ValueError(
                f"{object=} with {type(object)=} not in ['Galaxy', 'Catalogue']!"
            )
    
    def __call__(
        self: Self,
        object: Union[Galaxy, Catalogue],
        return_copy: bool = True,
    ) -> Optional[Union[Galaxy, Catalogue]]:
        self._assert_morph_fitter(object)
        return Selector.__call__(self, object, return_copy)


class Redshift_Selector(SED_fit_Selector):

    @property
    def requires_SED_fit(self: Self) -> bool:
        return False


class Mask_Selector(Data_Selector):

    @abstractmethod
    def load_mask(
        self: Self,
        data: Data,
        invert: bool = True,
    ) -> u.Quantity:
        pass


class Multiple_Selector(ABC):

    def __init__(
        self: Self,
        selectors: List[Type[Selector]],
        selection_name: Optional[str] = None
    ):
        self.selectors = np.array(selectors, dtype = "object").flatten()
        if selection_name is not None:
            self.selection_name = selection_name
    
    def __len__(self):
        return len(self.selectors)

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter > len(self) - 1:
            raise StopIteration
        else:
            selector = self[self.iter]
            self.iter += 1
            return selector

    def __getitem__(
        self: Self,
        index: Any
    ) -> Union[Selector, List[Selector]]:
        return self.selectors[index]

    @property
    def _selection_name(self: Self) -> str:
        if hasattr(self, "selection_name"):
            return self.selection_name
        else:
            return f"{'+'.join([selector._selection_name for selector in self.selectors])}"

    @property
    def _include_kwargs(self: Self) -> List[str]:
        return []
    
    def _assertions(self: Self) -> bool:
        try:
            return all(
                selector._assertions()
                for selector in self.selectors
            )
        except:
            breakpoint()

    def _failure_criteria(
        self: Self,
        gal: Galaxy,
    ) -> bool:
        return any(
            selector._failure_criteria(gal)
            for selector in self.selectors
        )

    def _selection_criteria(
        self: Self,
        gal: Galaxy,
    ) -> bool:
        return all(
            selector._selection_criteria(gal)
            for selector in self.selectors
        )

class Multiple_Data_Selector(Multiple_Selector, Data_Selector, ABC):
    
    def __init__(
        self: Self,
        selectors: List[Type[Selector]],
        selection_name: Optional[str] = None,
        cat_filterset: Optional[Catalogue] = None,
    ):
        Multiple_Selector.__init__(self, selectors, selection_name)
        Data_Selector.__init__(self)
        if cat_filterset is not None:
            self.crop_to_filterset(cat_filterset)

    def __call__(
        self: Self,
        object: Union[Galaxy, Catalogue],
        return_copy: bool = True,
        *args,
        **kwargs
    ) -> Optional[Union[Galaxy, Type[Catalogue_Base]]]:
        # run selection individually on each selector
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
            self.crop_to_filterset(object.filterset)
        [selector.__call__(object, return_copy = False) for selector in self.selectors]
        return Data_Selector.__call__(self, object, return_copy = return_copy)

    def crop_to_filterset(self: Self, filterset: Multiple_Filter) -> NoReturn:
        # crop each selector to the filterset
        self.selectors = [selector for selector in self.selectors \
            if selector.kwargs["band_name"] in filterset.band_names]
        

class Multiple_Photometry_Selector(Multiple_Selector, Photometry_Selector, ABC):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        selectors: List[Type[Selector]],
        selection_name: Optional[str] = None
    ):
        assert all([selector.aper_diam == aper_diam or selector.aper_diam is None for selector in selectors])
        Multiple_Selector.__init__(self, selectors, selection_name)
        Photometry_Selector.__init__(self, aper_diam)

    def __call__(
        self: Self,
        object: Union[Galaxy, Catalogue],
        return_copy: bool = True,
        *args,
        **kwargs,
    ) -> Optional[Union[Galaxy, Catalogue]]:
        # run selection individually on each selector
        [selector.__call__(object, return_copy = False) for selector in self.selectors]
        return Photometry_Selector.__call__(self, object, return_copy = return_copy)


class Multiple_SED_fit_Selector(Multiple_Selector, SED_fit_Selector, ABC):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, SED_code],
        selectors: List[Type[Selector]],
        selection_name: Optional[str] = None
    ):
        assert all([selector.aper_diam == aper_diam or selector.aper_diam is None for selector in selectors])
        if isinstance(SED_fit_label, tuple(SED_code.__subclasses__())):
            SED_fit_label = SED_fit_label.label
        assert all([selector.SED_fit_label == SED_fit_label or selector.SED_fit_label is None for selector in selectors])
        Multiple_Selector.__init__(self, selectors, selection_name)
        SED_fit_Selector.__init__(self, aper_diam, SED_fit_label)

    def __call__(
        self: Self,
        object: Union[Galaxy, Catalogue],
        return_copy: bool = True,
    ) -> Optional[Union[Galaxy, Catalogue]]:
        # run selection individually on each selector
        [selector.__call__(object, return_copy = False) for selector in self.selectors]
        return SED_fit_Selector.__call__(self, object, return_copy = return_copy)


class Multiple_Mask_Selector(Multiple_Selector, Mask_Selector, ABC):

    def __init__(
        self: Self,
        selectors: List[Type[Mask_Selector]],
        selection_name: Optional[str] = None,
        cat_filterset: Optional[Catalogue] = None,
    ):
        assert all([isinstance(selector, Mask_Selector) for selector in selectors])
        Multiple_Selector.__init__(self, selectors, selection_name)
        Mask_Selector.__init__(self)
        # NOT SURE IF THIS IS REQUIRED?
        if cat_filterset is not None:
            self.crop_to_filterset(cat_filterset)

    def __call__(
        self: Self,
        object: Union[Galaxy, Catalogue],
        return_copy: bool = True,
        *args,
        **kwargs
    ) -> Optional[Union[Galaxy, Type[Catalogue_Base]]]:
        # NOT SURE IF THIS IS REQUIRED?
        # run selection individually on each selector
        if isinstance(object, tuple(Catalogue_Base.__subclasses__())):
            self.crop_to_filterset(object.filterset)
        [selector.__call__(object, return_copy = False) for selector in self.selectors]
        return Mask_Selector.__call__(self, object, return_copy = return_copy)

    # NOT SURE IF THIS IS REQUIRED?
    def crop_to_filterset(self: Self, filterset: Multiple_Filter) -> NoReturn:
        # crop each selector to the filterset
        selectors_arr = []
        for selector in self.selectors:
            append = False
            if hasattr(selector, "band_name"):
                if selector.kwargs["band_name"] in filterset.band_names:
                    append = True
            else:
                append = True
            if append:
                selectors_arr.append(selector)
        self.selectors = selectors_arr
    
    def load_mask(
        self: Self,
        data: Data,
        invert: bool = True,
    ) -> u.Quantity:
        # load mask from each selector
        masks = [selector.load_mask(data, False) for selector in self.selectors]
        # combine masks
        combined_mask = np.logical_and.reduce(masks)
        if invert:
            combined_mask = np.invert(combined_mask)
        galfind_logger.info(
            f"Loaded {repr(self)} mask from {data.survey}!"
        )
        return combined_mask


class ID_Selector(Data_Selector):

    def __init__(
        self: Self,
        IDs: int,
        name: Optional[str] = None,
    ):
        if isinstance(IDs, int):
            IDs = [IDs]
        kwargs = {"IDs": IDs, "name": name}
        super().__init__(**kwargs)

    @property
    def _selection_name(self) -> str:
        if self.kwargs["name"] is None:
            return f"ID_{self.kwargs['IDs']}"
        else:
            return self.kwargs["name"]

    @property
    def _include_kwargs(self) -> List[str]:
        return ["IDs", "name"]

    def _assertions(self: Self) -> bool:
        try:
            assert isinstance(self.kwargs["IDs"], tuple([list, np.ndarray]))
            if self.kwargs["name"] is not None:
                assert(isinstance(self.kwargs["name"], str))
            assert all(isinstance(ID, tuple([int, np.int64])) for ID in self.kwargs["IDs"])
            passed = True
        except:
            passed = False
        return passed
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> bool:
        return gal.ID in self.kwargs["IDs"]


class Region_Selector(Data_Selector):

    def __init__(
        self: Self,
        fail_name: Optional[str] = None,
        **kwargs
    ):
        if fail_name is not None:
            self._fail_name = fail_name
        super().__init__(**kwargs)

    @abstractmethod
    def make_mask(
        self: Self,
        cat: Catalogue,
    ) -> u.Quantity:
        pass

    def load_mask(
        self: Self,
        data: Data,
        invert: bool = True,
    ) -> u.Quantity:
        mask = fits.open(self.get_mask_path(data), mode = "readonly", ignore_missing_simple = True)[self.name].data.astype(bool)
        if invert:
            mask = np.logical_not(mask)
        galfind_logger.info(
            f"Loaded {repr(self)} mask from {data.survey}!"
        )
        return mask

    @property
    def name(self: Self) -> str:
        return self._selection_name

    @property
    def fail_name(self: Self) -> str:
        if hasattr(self, "_fail_name"):
            assert self._fail_name != self.name, \
                galfind_logger.critical(
                    f"{self._fail_name=} cannot be the same as {self.name=}"
                )
            return self._fail_name
        return f"not_{self._selection_name}"
    
    def get_mask_path(
        self: Self,
        data: Data,
    ) -> str:
        return f"{config['Masking']['MASK_DIR']}/{data.survey}/region_masks/{data.version}_{self.name}.fits"

    def _call_cat(
        self: Self,
        cat: Catalogue,
        return_copy: bool = True,
        *args,
        **kwargs,
    ) -> Union[NoReturn, Catalogue]:
        # apply selection
        super()._call_cat(cat, return_copy = False, *args, **kwargs)
        if not hasattr(cat, "regions"):
            cat.regions = []
        cat.region_selector = self
        if hasattr(cat, "data"):
            if not hasattr(cat.data, "regions"):
                cat.data.regions = []
            cat.data.region_selector = self
        for name in [self.name, self.fail_name]:
            if name not in cat.region:
                cat.regions.append(name)
                if hasattr(cat, "data"):
                    cat.data.regions.append(name)

        mask_path = self.get_mask_path(cat.data)
        if not Path(mask_path).is_file():
            galfind_logger.info(
                f"Creating {self.name} mask for {cat.survey}."
            )
            funcs.make_dirs(mask_path)
            mask = self.make_mask(cat)
            # make header
            hdr = cat.data.forced_phot_band.load_wcs().to_header()
            # Save mask
            mask_hdu = fits.ImageHDU(
                mask.astype(np.uint8), header=hdr, name=self.name
            )
            hdulist = fits.HDUList([fits.PrimaryHDU(), mask_hdu])
            hdulist.writeto(mask_path, overwrite=True)

        if return_copy:
            cat_copy = deepcopy(cat)
            cat_ = cat_copy.crop(self)
        else:
            cat_ = cat
        return cat_

    def _call_gal(
        self: Self, 
        gal: Galaxy,
        return_copy: bool = True,
        *args,
        **kwargs,
    ) -> Union[NoReturn, Galaxy]:
        if return_copy:
            gal_ = deepcopy(gal)
        else:
            gal_ = gal
        # apply selection
        super()._call_gal(gal_, return_copy = False, *args, **kwargs)
        if not hasattr(gal_, "region"):
            gal_.region = []
        if gal_.selection_flags[self.name]:
            gal_.region.append(self.name)
        else:
            gal_.region.append(self.fail_name)
        return gal_


class Ds9_Region_Selector(Region_Selector):

    def __init__(
        self: Self,
        region_path: str,
        region_name: Optional[str] = None,
        fail_name: Optional[str] = None,
    ):
        if region_name is None:
            region_name = region_path.split("/")[-1].replace(".reg", "")
        kwargs = {"region_path": region_path, "region_name": region_name}
        super().__init__(fail_name, **kwargs)

    @property
    def _selection_name(self) -> str:
        return self.kwargs["region_name"]

    @property
    def _include_kwargs(self) -> List[str]:
        return ["region_path", "region_name"]

    def make_mask(self: Self):
        raise Exception()

    def _assertions(self: Self) -> bool:
        try:
            assert isinstance(self.kwargs["region_path"], str)
            assert isinstance(self.kwargs["region_name"], str)
            assert self.kwargs["region_path"].endswith(".reg")
            assert Path(self.kwargs["region_path"]).is_file()
            passed = True
        except:
            passed = False
        return passed
    
    def __call__(
        self: Self,
        object: Union[Galaxy, Catalogue],
        return_copy: bool = True,
        *args,
        **kwargs
    ) -> Optional[Union[Galaxy, Catalogue]]:
        breakpoint()
        with open(self.kwargs["region_path"], "r") as f:
            # save .reg file to object
            pass
        return super().__call__(object, return_copy, *args, **kwargs)

    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> bool:
        return False
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> bool:
        raise Exception()


class Depth_Region_Selector(Region_Selector):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        filt_name: Union[str, List[str]],
        region_label: Union[int, str],
        region_name: Optional[str] = None,
        fail_name: Optional[str] = None,
    ):
        assert isinstance(aper_diam, u.Quantity)
        self._aper_diam = aper_diam
        if isinstance(filt_name, list):
            filt_name = "+".join(filt_name)
        if isinstance(region_label, int):
            region_label = str(region_label)
        kwargs = {
            "filt_name": filt_name,
            "region_label": region_label,
            "region_name": region_name,
        }
        super().__init__(fail_name, **kwargs)

    @property
    def _selection_name(self) -> str:
        if self.kwargs["region_name"] is None:
            reg_name = f"{self.kwargs['filt_name']}_{self.kwargs['region_label']}"
        else:
            reg_name = self.kwargs["region_name"]
        return reg_name

    @property
    def _include_kwargs(self) -> List[str]:
        return ["filt_name", "region_label", "region_name"]

    def make_mask(
        self: Self,
        cat: Catalogue,
    ) -> NDArray:
        """
        Fill zeros in a sparse array based on the nearest non-zero value.
        
        Parameters:
        -----------
        cat: Catalogue
            
        Returns:
        --------
        filled_array : 2D numpy array
            Array with zeros replaced by nearest non-zero values
        """
        # Make an array of zeros based on the cat.data filt_name shape
        if isinstance(self.kwargs["filt_name"], str) and "+" not in self.kwargs["filt_name"]:
            data_shape = cat.data[self.kwargs["filt_name"]].data_shape
        else:
            # TODO: the zero index is hard-coded for now, as SExtractor requires all images to be the same shape
            data_shape = cat.data[self.kwargs["filt_name"]][0].data_shape
        mask = np.zeros(data_shape)
        sky_coords = SkyCoord(cat.sky_coord)
        # load wcs of forced photometry band
        wcs = cat.data.forced_phot_band.load_wcs()
        x, y = wcs.all_world2pix(
            sky_coords.ra, sky_coords.dec, 0
        )
        x = x.astype(int)
        y = y.astype(int)
        selection_vals = np.array([2.0 if gal.selection_flags[self.name] else 1.0 for gal in cat])
        assert len(selection_vals) == len(x) == len(y), \
            galfind_logger.critical(
                f"{len(selection_vals)=} != {len(x)=} or != {len(y)=}"
            )
        mask[y, x] = selection_vals
        # Make a copy of the input array
        result = deepcopy(mask)
        # Create a mask for zero values
        blank_mask = (mask == 0)
        distances, indices = ndimage.distance_transform_edt(blank_mask, return_distances=True, return_indices=True)
        # For each zero position, fill with the value of its nearest non-zero neighbor
        for i in tqdm(range(mask.shape[0]), desc = f"Making {self.name} mask", total = mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] == 0:
                    # Get coordinates of nearest non-zero pixel
                    nearest_i, nearest_j = indices[0][i, j], indices[1][i, j]
                    # Get the value at this nearest non-zero pixel
                    nearest_value = mask[nearest_i, nearest_j]
                    # Set the result pixel to this value
                    result[i, j] = nearest_value
        result -= 1.0
        result = result.astype(bool)
        return result

    def _assertions(self: Self) -> bool:
        try:
            assert isinstance(self.kwargs["filt_name"], str)
            assert isinstance(self.kwargs["region_label"], str)
            if self.kwargs["region_name"] is not None:
                assert isinstance(self.kwargs["region_name"], str)
            passed = True
        except:
            passed = False
        return passed

    def _check_phot_exists(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> bool:
        # TODO: Sort out aper_diam set to None as part of Data_Selector
        try:
            passed = len(gal.aper_phot[self._aper_diam]) != 0
        except:
            passed = False
        return passed
    
    def __call__(
        self: Self,
        object: Union[Galaxy, Catalogue],
        return_copy: bool = True,
        *args,
        **kwargs
    ) -> Optional[Union[Galaxy, Catalogue]]:
        # can only call on catalogue objects
        assert isinstance(object, tuple(Catalogue_Base.__subclasses__())), \
            galfind_logger.critical(
                f"{object=} must be a Catalogue object."
            )
        return super().__call__(object, return_copy)

    def _call_cat(
        self: Self,
        cat: Catalogue,
        return_copy: bool = True,
        *args,
        **kwargs,
    ) -> Union[NoReturn, Catalogue]:
        # add array of depth regions to kwargs
        # TODO: Add these as options; hard coded for now!
        if all(self._selection_name in gal.selection_flags.keys() for gal in cat):
            galfind_logger.info(
                f"Depth regions already selected for {repr(cat)}."
            )
        else:
            mode = "n_nearest"
            instr_name = "NIRCam"
            h5_path = f"{config['Depths']['DEPTH_DIR']}/" \
                + f"{instr_name}/{cat.version}/{cat.survey}/" \
                + f"{format(self._aper_diam.value, '.2f')}as/" \
                + f"{mode}/{self.kwargs['filt_name']}.h5"
            assert Path(h5_path).is_file(), \
                galfind_logger.critical(
                    f"{h5_path=} does not exist."
                )
            # open depth.h5 file
            depth_h5 = h5py.File(h5_path, "r")
            # get depth regions
            depth_labels = depth_h5["depth_labels"][:]
            depth_h5.close()
            # TODO: not have this assertion
            assert len(cat) == len(depth_labels), \
                galfind_logger.critical(
                    f"Length of {cat} ({len(cat)}) does not match " + \
                    f"length of depth labels ({len(depth_labels)})"
                )
            reg_dict = {gal.ID: depth_label for gal, depth_label in zip(cat, depth_labels)}
            kwargs["reg_dict"] = reg_dict
        return super()._call_cat(cat, return_copy = return_copy, *args, **kwargs)

    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> bool:
        if "reg_dict" not in kwargs.keys():
            return False
        if np.isnan(kwargs["reg_dict"][gal.ID]):
            return True
        else:
            return False
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> bool:
        if "reg_dict" not in kwargs.keys():
            return True
        depth_label = kwargs["reg_dict"][gal.ID]
        if isinstance(depth_label, float):
            depth_label = str(int(depth_label))
        if depth_label == self.kwargs["region_label"]:
            return True
        else:
            return False
    

class Redshift_Limit_Selector(Redshift_Selector):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, SED_code],
        z_lim: Union[float, int],
        gtr_or_less: str
    ):
        kwargs = {"z_lim": z_lim, "gtr_or_less": gtr_or_less}
        super().__init__(aper_diam, SED_fit_label, **kwargs)

    @property
    def _selection_name(self) -> str:
        if self.kwargs["gtr_or_less"] == "gtr":
            sign = ">"
        else:
            sign = "<"
        return f"z{sign}{self.kwargs['z_lim']:.2f}"

    @property
    def _include_kwargs(self) -> List[str]:
        return ["z_lim", "gtr_or_less"]

    def _assertions(self: Self) -> bool:
        try:
            assert isinstance(self.kwargs["z_lim"], (int, float))
            assert self.kwargs["gtr_or_less"] in ["gtr", "less"]
            passed = True
        except:
            passed = False
        return passed
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> bool:
        if self.kwargs["gtr_or_less"] == "gtr":
            return gal.aper_phot[self.aper_diam].SED_results \
                [self.SED_fit_label].z >= self.kwargs["z_lim"]
        else:
            return gal.aper_phot[self.aper_diam].SED_results \
                [self.SED_fit_label].z <= self.kwargs["z_lim"]
        

class Rest_Frame_Property_Limit_Selector(Redshift_Selector):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, SED_code],
        property_calculator: Type[Property_Calculator],
        property_lim: Union[u.Quantity, u.Magnitude, u.Dex],
        gtr_or_less: str,
    ):
        self.property_calculator = property_calculator
        kwargs = {
            "property_name": property_calculator.name,
            "property_lim": property_lim,
            "gtr_or_less": gtr_or_less
        }
        super().__init__(aper_diam, SED_fit_label, **kwargs)

    @property
    def _selection_name(self) -> str:
        if self.kwargs["gtr_or_less"] == "gtr":
            sign = ">"
        else:
            sign = "<"
        return self.kwargs["property_name"] + \
            f"{sign}{self.kwargs['property_lim'].value:.2f}"

    @property
    def _include_kwargs(self) -> List[str]:
        return ["property_name", "property_lim", "gtr_or_less"]

    def _assertions(self: Self) -> bool:
        try:
            from . import Property_Calculator
            assert isinstance(self.property_calculator, \
                tuple(Property_Calculator.__subclasses__()))
            assert isinstance(self.kwargs["property_lim"], \
                (u.Quantity, u.Magnitude, u.Dex)) #or \
                # all(isinstance(val, (u.Quantity, u.Magnitude, u.Dex)) \
                # for val in self.kwargs["property_lim"])
            assert self.kwargs["gtr_or_less"] in ["gtr", "less"]
            assert self.aper_diam == self.property_calculator.aper_diam
            assert self.SED_fit_label == self.property_calculator.SED_fit_label
            passed = True
        except:
            passed = False
        return passed
    
    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> bool:
        assertions = []
        try:
            assertions.extend([
                self.kwargs["property_name"] in gal.aper_phot[self.aper_diam].SED_results \
                    [self.SED_fit_label].phot_rest.properties.keys()
            ])
            assertions.extend([
                gal.aper_phot[self.aper_diam].SED_results \
                    [self.SED_fit_label].phot_rest.properties \
                    [self.kwargs["property_name"]].unit \
                    .is_equivalent(self.kwargs["property_lim"].unit)
            ])
            failed = not all(assertions)
        except:
            failed = True
        return failed
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> bool:
        if self.kwargs["gtr_or_less"] == "gtr":
            return gal.aper_phot[self.aper_diam].SED_results \
                [self.SED_fit_label].phot_rest.properties \
                [self.kwargs["property_name"]] \
                > self.kwargs["property_lim"]
        else:
            return gal.aper_phot[self.aper_diam].SED_results \
                [self.SED_fit_label].phot_rest.properties \
                [self.kwargs["property_name"]] \
                < self.kwargs["property_lim"]
    
    def __call__(
        self: Self,
        object: Union[Galaxy, Catalogue],
        return_copy: bool = True,
        *args,
        **kwargs
    ) -> Optional[Union[Galaxy, Catalogue]]:
        # calculate property if not already stored
        self.property_calculator(object)
        return SED_fit_Selector.__call__(self, object, return_copy)


class Redshift_Bin_Selector(Multiple_SED_fit_Selector):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, SED_code],
        z_bin: List[Union[int, float]],
    ):
        assert all(isinstance(z_lim, (int, float)) for z_lim in z_bin)
        assert len(z_bin) == 2
        assert z_bin[0] < z_bin[1]
        selection_name = f"{z_bin[0]:.2f}<z<{z_bin[1]:.2f}"
        selectors = [
            Redshift_Limit_Selector(aper_diam, SED_fit_label, z_bin[0], "gtr"),
            Redshift_Limit_Selector(aper_diam, SED_fit_label, z_bin[1], "less"),
        ]
        super().__init__(aper_diam, SED_fit_label, selectors, selection_name)


class Rest_Frame_Property_Bin_Selector(Multiple_SED_fit_Selector):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, SED_code],
        property_calculator: Rest_Frame_Property_Calculator,
        property_bin: List[Union[u.Quantity, u.Magnitude, u.Dex]],
    ):
        assert isinstance(property_bin, (u.Quantity, u.Magnitude, u.Dex))
        assert len(property_bin) == 2
        assert property_bin[0] < property_bin[1]
        from . import Rest_Frame_Property_Calculator
        assert isinstance(property_calculator, \
            tuple(Rest_Frame_Property_Calculator.__subclasses__()))
        selection_name = f"{property_bin[0].value:.2f}<" + \
            f"{property_calculator.name}<{property_bin[1].value:.2f}"
        selectors = [
            Rest_Frame_Property_Limit_Selector(
                aper_diam,
                SED_fit_label,
                property_calculator,
                property_bin[0],
                "gtr"
            ),
            Rest_Frame_Property_Limit_Selector(
                aper_diam,
                SED_fit_label,
                property_calculator,
                property_bin[1],
                "less"
            )
        ]
        super().__init__(aper_diam, SED_fit_label, selectors, selection_name)

class Colour_Selector(Photometry_Selector):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        colour_bands: Union[str, List[str]],
        bluer_or_redder: str,
        colour_val: float,
    ):
        if isinstance(colour_bands, str):
            colour_bands = colour_bands.split("-")
        kwargs = {
            "colour_bands": colour_bands,
            "bluer_or_redder": bluer_or_redder,
            "colour_val": colour_val,
        }
        super().__init__(aper_diam, **kwargs)

    @property
    def _selection_name(self) -> str:
        return f"{'-'.join(self.kwargs['colour_bands'])}" + \
            f"{'<' if self.kwargs['bluer_or_redder'] == 'bluer' else '>'}" + \
            f"{self.kwargs['colour_val']:.2f}"

    @property
    def _include_kwargs(self) -> List[str]:
        return ["colour_bands", "bluer_or_redder", "colour_val"]

    def _assertions(self: Self) -> bool:
        try:
            assert self.kwargs["bluer_or_redder"] in ["bluer", "redder"]
            colour_bands = self.kwargs["colour_bands"]
            assert isinstance(colour_bands, list)
            assert len(colour_bands) == 2
            passed = True
        except:
            passed = False
        return passed
        
    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> bool:
        try:
            assertions = []
            assertions.extend([
                all(
                    colour in gal.aper_phot[self.aper_diam].filterset.band_names
                    for colour in self.kwargs["colour_bands"]
                )
            ])
            # ensure bands are ordered blue -> red
            assertions.extend([
                np.where(np.array(gal.aper_phot[self.aper_diam].filterset.band_names) == self.kwargs["colour_bands"][0])[0][0] \
                    < np.where(np.array(gal.aper_phot[self.aper_diam].filterset.band_names) == self.kwargs["colour_bands"][1])[0][0]
            ])
            failed = not all(assertions)
        except:
            failed = True
        return failed
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> bool:
        band_indices = [
            int(np.where(np.array(gal.aper_phot[self.aper_diam].\
            filterset.band_names) == band_name)[0][0])
            for band_name in self.kwargs["colour_bands"]
        ]
        colour = (
            funcs.convert_mag_units(
                gal.aper_phot[self.aper_diam].filterset[band_indices[0]].WavelengthCen,
                gal.aper_phot[self.aper_diam].flux[band_indices[0]],
                u.ABmag,
            )
            - funcs.convert_mag_units(
                gal.aper_phot[self.aper_diam].filterset[band_indices[1]].WavelengthCen,
                gal.aper_phot[self.aper_diam].flux[band_indices[1]],
                u.ABmag,
            )
        ).value
        return (colour < self.kwargs["colour_val"] and \
            self.kwargs["bluer_or_redder"] == "bluer") or \
            (colour > self.kwargs["colour_val"] and \
            self.kwargs["bluer_or_redder"] == "redder")
    

class Min_Band_Selector(Data_Selector):

    def __init__(
        self: Self,
        min_bands: int,
    ):
        kwargs = {"min_bands": min_bands}
        super().__init__(**kwargs)

    @property
    def _selection_name(self) -> str:
        return f"bands>{self.kwargs['min_bands'] - 1}"

    @property
    def _include_kwargs(self) -> List[str]:
        return ["min_bands"]

    def _assertions(self: Self) -> bool:
        return isinstance(self.kwargs["min_bands"], int)
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> bool:
        return len(gal.aper_phot[list(gal.aper_phot.keys())[0]]) >= self.kwargs["min_bands"]


class Unmasked_Band_Selector(Mask_Selector):

    def __init__(
        self: Self,
        band_name: str,
    ):
        kwargs = {"band_name": band_name}
        super().__init__(**kwargs)

    @property
    def _selection_name(self) -> str:
        return f"unmasked_{self.kwargs['band_name']}"

    @property
    def _include_kwargs(self) -> List[str]:
        return ["band_name"]

    def _assertions(self: Self) -> bool:
        # ensure that each band is a valid band name in galfind
        return self.kwargs["band_name"] in json.loads(config.get("Other", "ALL_BANDS"))
        
    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> bool:
        try:
            failed = self.kwargs["band_name"] not in \
                gal.aper_phot[list(gal.aper_phot.keys())[0]].filterset.band_names
        except:
            failed = True
        return failed
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs,
    ) -> bool:
        band_index = int([i for i, band_name in enumerate( \
            gal.aper_phot[list(gal.aper_phot.keys())[0]].filterset.band_names) \
            if band_name == self.kwargs["band_name"]][0])
        if not isinstance(gal.aper_phot[list(gal.aper_phot.keys())[0]].flux, Masked):
            return True
        else:
            return not gal.aper_phot[list(gal.aper_phot.keys())[0]].flux.mask[band_index]
    
    def _call_cat(
        self: Self,
        cat: Catalogue,
        return_copy: bool = True,
    ) -> Union[NoReturn, Catalogue]:
        assert self.kwargs["band_name"] in cat.filterset.band_names, \
            galfind_logger.critical(
                f"{self.kwargs['band_name']} not in {cat.filterset.band_names}."
            )
        return Data_Selector._call_cat(self, cat, return_copy)

    def load_mask(
        self: Self,
        data: Data,
        invert: bool = True,
    ) -> u.Quantity:
        # load mask from each selector
        if self.kwargs["band_name"] not in data.filterset.band_names:
            data_shapes = [band_data.data_shape for band_data in data]
            assert all(data_shape == data_shapes[0] for data_shape in data_shapes), \
                galfind_logger.critical(
                    f"Data shapes do not match: {data_shapes}"
                )
            mask = np.full(data_shapes[0], False)
            galfind_logger.warning(
                f"{self.kwargs['band_name']} not in {data.filterset.band_names}!"
            )
        else:
            # extract band_data from data
            band_data = data[self.kwargs["band_name"]]
            # load mask from data
            mask = band_data.load_mask("MASK")[0].astype(bool)
        if not invert:
            mask = np.logical_not(mask)
        galfind_logger.info(
            f"Loaded {repr(self)} mask from {data.survey}!"
        )
        return mask


class Min_Unmasked_Band_Selector(Mask_Selector):

    def __init__(
        self: Self,
        min_bands: int,
    ):
        kwargs = {"min_bands": min_bands}
        super().__init__(**kwargs)

    @property
    def _selection_name(self) -> str:
        return f"unmasked_bands>{self.kwargs['min_bands'] - 1}"

    @property
    def _include_kwargs(self) -> List[str]:
        return ["min_bands"]

    def _assertions(self: Self) -> bool:
        return isinstance(self.kwargs["min_bands"], int)
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> bool:
        if isinstance(gal.aper_phot[list(gal.aper_phot.keys())[0]].flux, u.Quantity):
            mask = np.full(len(gal.aper_phot[list(gal.aper_phot.keys())[0]].flux), False)
        else:
            mask = gal.aper_phot[list(gal.aper_phot.keys())[0]].flux.mask
        n_unmasked_bands = len([val for val in mask if not val])
        return n_unmasked_bands >= self.kwargs["min_bands"]

    def load_mask(
        self: Self,
        data: Data,
        invert: bool = True,
    ) -> u.Quantity:
        # add masks
        breakpoint()
        data_shapes = [band_data.data_shape for band_data in data]
        assert all(data_shape == data_shapes[0] for data_shape in data_shapes), \
            galfind_logger.critical(
                f"Data shapes do not match: {data_shapes}"
            )
        n_bands_unmasked = np.zeros(data_shapes[0])
        for band_data in data:
            n_bands_unmasked += band_data.load_mask()
        breakpoint()
        # convert to boolean (True if n_bands_unmasked >= min_bands else False)
        mask = n_bands_unmasked >= self.kwargs["min_bands"]
        mask = mask.astype(bool)
        breakpoint()
        if invert:
            mask = np.logical_not(mask)
        galfind_logger.info(
            f"Loaded {repr(self)} mask from {data.survey}!"
        )
        return mask


class Min_Instrument_Unmasked_Band_Selector(Mask_Selector):

    def __init__(
        self: Self,
        min_bands: int,
        instrument: Union[str, Type[Instrument]],
    ):
        if isinstance(instrument, str):
            assert instrument in expected_instr_bands.keys(), \
                galfind_logger.critical(
                    f"{instrument=} not a valid instrument name."
                )
            instrument = [instr() for instr in Instrument.__subclasses__() \
                if instr.__name__ == instrument][0]
        kwargs = {"min_bands": min_bands, "instrument": instrument}
        super().__init__(**kwargs)

    @property
    def _selection_name(self) -> str:
        return f"unmasked_bands_{self.kwargs['instrument'].__class__.__name__}>{self.kwargs['min_bands'] - 1}"

    @property
    def _include_kwargs(self) -> List[str]:
        return ["min_bands", "instrument"]

    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["min_bands"], int)])
            assertions.extend([isinstance(self.kwargs["instrument"], tuple(Instrument.__subclasses__()))])
            passed = all(assertions)
        except:
            passed = False
        return passed
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> bool:
        phot_obs = gal.aper_phot[list(gal.aper_phot.keys())[0]]
        if isinstance(phot_obs.flux, u.Quantity):
            mask_arr = np.full(len(phot_obs.flux), False)
        else:
            mask_arr = phot_obs.flux.mask
        instr_name_arr = np.array([filt.instrument_name for filt in phot_obs.filterset])
        assert len(mask_arr) == len(instr_name_arr), \
            galfind_logger.critical(
                f"{len(mask_arr)=} != {len(instr_name_arr)=}"
            )
        n_unmasked_bands = len(
            [
                mask for mask, instr_name in zip(mask_arr, instr_name_arr) 
                if not mask and instr_name == self.kwargs["instrument"].__class__.__name__
            ]
        )
        return n_unmasked_bands >= self.kwargs["min_bands"]

    def load_mask(
        self: Self,
        data: Data,
        invert: bool = True,
    ) -> u.Quantity:
        # add masks
        data_shapes = [band_data.data_shape for band_data in data]
        assert all(data_shape == data_shapes[0] for data_shape in data_shapes), \
            galfind_logger.critical(
                f"Data shapes do not match: {data_shapes}"
            )
        n_bands_unmasked = np.zeros(data_shapes[0])
        for band_data in data:
            if band_data.instr_name == self.kwargs["instrument"].__class__.__name__:
                n_bands_unmasked += band_data.load_mask("MASK", True)[0]
        # convert to boolean (True if n_bands_unmasked >= min_bands else False)
        mask = n_bands_unmasked >= self.kwargs["min_bands"]
        mask = mask.astype(bool)
        if invert:
            mask = np.logical_not(mask)
        galfind_logger.info(
            f"Loaded {repr(self)} mask from {data.survey}!"
        )
        return mask


class Bluewards_LyLim_Non_Detect_Selector(Redshift_Selector):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, SED_code],
        SNR_lim: float,
        ignore_bands: Optional[Union[str, List[str]]] = None,
    ):
        if isinstance(ignore_bands, str):
            ignore_bands = [ignore_bands]
        kwargs = {"SNR_lim": SNR_lim, "ignore_bands": ignore_bands}
        super().__init__(aper_diam, SED_fit_label, **kwargs)

    @property
    def _selection_name(self) -> str:
        selection_name = f"blue_LyLim<{self.kwargs['SNR_lim']:.1f}"
        if self.kwargs["ignore_bands"] is not None:
            ignore_str = ",".join(self.kwargs["ignore_bands"])
            selection_name += f"_no_{ignore_str}"
        return selection_name

    @property
    def _include_kwargs(self) -> List[str]:
        return ["SNR_lim", "ignore_bands"]

    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["SNR_lim"], (int, float))])
            if self.kwargs["ignore_bands"] is not None:
                for band in self.kwargs["ignore_bands"]:
                    # ensure this band exists
                    assertions.extend([band in json.loads(config.get("Other", "ALL_BANDS"))])
            passed = all(assertions)
        except:
            passed = False
        return passed
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
    ) -> bool:
        # extract first Lylim non-detect band
        first_Lylim_non_detect_band = gal.aper_phot[self.aper_diam]. \
            SED_results[self.SED_fit_label].phot_rest. \
                get_first_bluewards_band(
                    wav_lyman_lim * u.AA,
                    self.kwargs["ignore_bands"],
                )
        # if no bands bluewards of Lyman alpha,
        # select the galaxy by default
        if first_Lylim_non_detect_band is None:
            return True
        # find index of first Lya non-detect band
        first_Lylim_non_detect_index = np.where(
            np.array(gal.aper_phot[self.aper_diam].filterset.band_names) \
            == first_Lylim_non_detect_band)[0][0]
        SNR_non_detect = gal.aper_phot[self.aper_diam].SNR[: first_Lylim_non_detect_index + 1]
        if isinstance(gal.aper_phot[self.aper_diam].flux, u.Quantity):
            mask_non_detect = np.full(len(SNR_non_detect), False)
        else:
            mask_non_detect = gal.aper_phot[self.aper_diam].flux.mask[: first_Lylim_non_detect_index + 1]
        # require the first Lylim non detect band and all bluewards bands 
        # to be non-detected at < SNR_lim if not masked
        return all(SNR < self.kwargs["SNR_lim"] or mask for mask, SNR in 
            zip(mask_non_detect, SNR_non_detect))


class Bluewards_Lya_Non_Detect_Selector(Redshift_Selector):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, SED_code],
        SNR_lim: float,
        ignore_bands: Optional[Union[str, List[str]]] = None,
    ):
        if isinstance(ignore_bands, str):
            ignore_bands = [ignore_bands]
        kwargs = {"SNR_lim": SNR_lim, "ignore_bands": ignore_bands}
        super().__init__(aper_diam, SED_fit_label, **kwargs)

    @property
    def _selection_name(self) -> str:
        selection_name = f"blue_Lya<{self.kwargs['SNR_lim']:.1f}"
        if self.kwargs["ignore_bands"] is not None:
            ignore_str = ",".join(self.kwargs["ignore_bands"])
            selection_name += f"_no_{ignore_str}"
        return selection_name

    @property
    def _include_kwargs(self) -> List[str]:
        return ["SNR_lim", "ignore_bands"]

    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["SNR_lim"], (int, float))])
            if self.kwargs["ignore_bands"] is not None:
                for band in self.kwargs["ignore_bands"]:
                    # ensure this band exists
                    assertions.extend([band in json.loads(config.get("Other", "ALL_BANDS"))])
            passed = all(assertions)
        except:
            passed = False
        return passed
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
    ) -> bool:
        # extract first Lya non-detect band
        from .Emission_lines import line_diagnostics
        first_Lya_non_detect_band = gal.aper_phot[self.aper_diam]. \
            SED_results[self.SED_fit_label].phot_rest. \
                get_first_bluewards_band(
                    line_diagnostics["Lya"]["line_wav"],
                    self.kwargs["ignore_bands"],
                )
        # if no bands bluewards of Lyman alpha, 
        # select the galaxy by default
        if first_Lya_non_detect_band is None:
            return True
        # find index of first Lya non-detect band
        first_Lya_non_detect_index = np.where(
            np.array(gal.aper_phot[self.aper_diam].filterset.band_names) \
            == first_Lya_non_detect_band)[0][0]
        SNR_non_detect = gal.aper_phot[self.aper_diam].SNR[: first_Lya_non_detect_index + 1]
        if isinstance(gal.aper_phot[self.aper_diam].flux, u.Quantity):
            mask_non_detect = np.full(len(SNR_non_detect), False)
        else:
            mask_non_detect = gal.aper_phot[self.aper_diam].flux.mask[: first_Lya_non_detect_index + 1]
        # require the first Lya non detect band and all bluewards bands 
        # to be non-detected at < SNR_lim if not masked
        return all(SNR < self.kwargs["SNR_lim"] or mask for mask, SNR in 
            zip(mask_non_detect, SNR_non_detect))


class Redwards_Lya_Detect_Selector(Redshift_Selector):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, SED_code],
        SNR_lims: float,
        widebands_only: bool,
        ignore_bands: Optional[Union[str, List[str]]] = None,
    ):
        if isinstance(ignore_bands, str):
            ignore_bands = [ignore_bands]
        kwargs = {
            "SNR_lims": SNR_lims,
            "widebands_only": widebands_only,
            "ignore_bands": ignore_bands,
        }
        super().__init__(aper_diam, SED_fit_label, **kwargs)

    @property
    def _selection_name(self) -> str:
        if isinstance(self.kwargs["SNR_lims"], (int, float)):
            # require all redwards bands to be detected at >SNR_lims
            selection_name = f"ALL_red_Lya>{self.kwargs['SNR_lims']:.1f}"
        else: # isinstance(SNR_lims, (list, np.array)):
            # require the n^th band redwards of Lya 
            # to be detected at >SNR_lims[n]
            SNR_str = ",".join([str(np.round(SNR, 1)) \
                for SNR in self.kwargs["SNR_lims"]])
            selection_name = f"red_Lya>{SNR_str}"
        if self.kwargs["widebands_only"]:
            selection_name += "_wide"
        if self.kwargs["ignore_bands"] is not None:
            ignore_str = ",".join(self.kwargs["ignore_bands"])
            selection_name += f"_no_{ignore_str}"
        return selection_name

    @property
    def _include_kwargs(self) -> List[str]:
        return ["SNR_lims", "widebands_only", "ignore_bands"]

    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            if isinstance(self.kwargs["SNR_lims"], (int, float)):
                pass
            elif isinstance(self.kwargs["SNR_lims"], (list, np.ndarray)):
                assertions.extend([all(isinstance(SNR, (int, float)) \
                    for SNR in self.kwargs["SNR_lims"])])
            else:
                assertions.extend([False])
            assertions.extend([isinstance(self.kwargs["widebands_only"], bool)])
            if self.kwargs["ignore_bands"] is not None:
                for band in self.kwargs["ignore_bands"]:
                    # ensure this band exists
                    assertions.extend([band in json.loads(config.get("Other", "ALL_BANDS"))])
            passed = all(assertions)
        except:
            passed = False
        return passed
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
    ) -> bool:
        if isinstance(self.kwargs["SNR_lims"], (int, float)):
            SNR_lims = np.full(len(gal.aper_phot[self.aper_diam].\
                filterset.band_names), self.kwargs["SNR_lims"])
        else:
            SNR_lims = self.kwargs["SNR_lims"]

        # extract first Lya non-detect band
        from .Emission_lines import line_diagnostics
        first_Lya_detect_band = gal.aper_phot[self.aper_diam]. \
            SED_results[self.SED_fit_label].phot_rest. \
                get_first_redwards_band(
                    line_diagnostics["Lya"]["line_wav"],
                    ignore_bands = self.kwargs["ignore_bands"]
                )
        # if no bands redwards of Lyman alpha, do not select the galaxy
        if first_Lya_detect_band is None:
            return False
        # find index of first Lya non-detect band
        first_Lya_detect_index = np.where(np.array(gal.aper_phot[self.aper_diam]. \
            filterset.band_names) == first_Lya_detect_band)[0][0]
        detect_bands = gal.aper_phot[self.aper_diam]. \
            filterset.band_names[first_Lya_detect_index:]
        SNR_detect = gal.aper_phot[self.aper_diam].SNR[first_Lya_detect_index:]
        if isinstance(gal.aper_phot[self.aper_diam].flux, u.Quantity):
            mask_detect = np.full(len(SNR_detect), False)
        else:
            mask_detect = gal.aper_phot[self.aper_diam].flux.mask[first_Lya_detect_index:]
        # option as to whether to exclude potentially 
        # shallower medium/narrow bands in this calculation
        if self.kwargs["widebands_only"]:
            wide_band_detect_indices = [
                True if "W" in band.upper() or "LP" in band.upper()
                else False for band in detect_bands
            ]
            SNR_detect = list(np.array(SNR_detect)[wide_band_detect_indices])
            mask_detect = list(np.array(mask_detect)[wide_band_detect_indices])
        # selection criteria
        return all(SNR > SNR_lim or mask for mask, SNR, SNR_lim \
            in zip(mask_detect, SNR_detect, SNR_lims))


class Lya_Band_Selector(Redshift_Selector):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, SED_code],
        SNR_lim: Union[int, float],
        detect_or_non_detect: str,
        widebands_only: bool,
    ):
        kwargs = {
            "SNR_lim": SNR_lim,
            "detect_or_non_detect": detect_or_non_detect,
            "widebands_only": widebands_only,
        }
        super().__init__(aper_diam, SED_fit_label, **kwargs)

    @property
    def _selection_name(self) -> str:
        detect_or_non_detect_str = ">" if self.kwargs \
            ["detect_or_non_detect"] == "detect" else "<"
        selection_name = f"Lya_band_SNR{detect_or_non_detect_str}" + \
            f"{self.kwargs['SNR_lim']:.1f}"
        if self.kwargs["widebands_only"]:
            selection_name += "_widebands"
        return selection_name

    @property
    def _include_kwargs(self) -> List[str]:
        return ["SNR_lim", "detect_or_non_detect", "widebands_only"]

    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["SNR_lim"], (int, float))])
            assertions.extend([self.kwargs["detect_or_non_detect"].lower() in ["detect", "non_detect"]])
            assertions.extend([isinstance(self.kwargs["widebands_only"], bool)])
            passed = all(assertions)
        except:
            passed = False
        return passed
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
    ) -> bool:
        bands = np.array(gal.aper_phot[self.aper_diam].filterset.band_names)
        # determine Lya band(s) - usually a single band, 
        # but could be two in the case of medium bands
        first_Lya_detect_band = gal.aper_phot[self.aper_diam]. \
            SED_results[self.SED_fit_label].phot_rest.first_Lya_detect_band
        first_Lya_detect_index = np.where(bands == \
            first_Lya_detect_band)[0][0]
        first_Lya_non_detect_band = gal.aper_phot[self.aper_diam]. \
            SED_results[self.SED_fit_label].phot_rest.first_Lya_non_detect_band
        first_Lya_non_detect_index = np.where(bands == \
            first_Lya_non_detect_band)[0][0]
        # load SNRs, cropping by the relevant bands
        detect_bands = bands[
            first_Lya_detect_index : first_Lya_non_detect_index + 1
        ]
        if len(detect_bands) == 0:
            return False
        SNRs = gal.aper_phot[self.aper_diam].SNR[
            first_Lya_detect_band : first_Lya_non_detect_index + 1
        ]
        if isinstance(gal.aper_phot[self.aper_diam].flux, u.Quantity):
            mask_bands = np.full(SNRs, False)
        else:
            mask_bands = gal.aper_phot[self.aper_diam].flux.mask[
                first_Lya_detect_band : first_Lya_non_detect_index + 1
            ]
        if self.kwargs["widebands_only"]:
            wide_band_detect_indices = [
                True
                if "W" in band.upper() or "LP" in band.upper()
                else False
                for band in detect_bands
            ]
            SNRs = SNRs[wide_band_detect_indices]
        if len(SNRs) == 0:
            return False
        if self.kwargs["detect_or_non_detect"].lower() == "detect":
            return all(
                SNR > self.kwargs["SNR_lim"] or mask
                for SNR, mask in zip(SNRs, mask_bands)
            )
        else:
            return all(
                SNR < self.kwargs["SNR_lim"] or mask
                for SNR, mask in zip(SNRs, mask_bands)
            )


class Band_SNR_Selector(Photometry_Selector):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        band: Union[str, int],
        detect_or_non_detect: str,
        SNR_lim: Union[int, float],
    ):
        kwargs = {
            "band": band,
            "detect_or_non_detect": detect_or_non_detect,
            "SNR_lim": SNR_lim,
        }
        super().__init__(aper_diam, **kwargs)

    @property
    def _selection_name(self) -> str:
        if self.kwargs["detect_or_non_detect"].lower() == "detect":
            sign = ">"
        else: # self.kwargs["detect_or_non_detect"].lower() == "non_detect"
            sign = "<"
        if isinstance(self.kwargs["band"], str):
            selection_name = self.kwargs['band'] + \
                f"_SNR{sign}{self.kwargs['SNR_lim']:.1f}"
        else: # isinstance(self.kwargs["band"], int):
            galfind_logger.debug(
                "Indexing e.g. 2 and -4 when there are 6 bands " + \
                f"results in differing {self.__class__.__name__} selection " + \
                "names even though the same band is referenced!"
            )
            if self.kwargs["band"] == 0:
                selection_name = "bluest_band_SNR" + \
                    f"{sign}{self.kwargs['SNR_lim']:.1f}"
            elif self.kwargs["band"] == -1:
                selection_name = f"reddest_band_SNR" + \
                    f"{sign}{self.kwargs['SNR_lim']:.1f}"
            elif self.kwargs["band"] > 0:
                selection_name = funcs.ordinal(self.kwargs["band"] + 1) + \
                    f"_bluest_band_SNR{sign}{self.kwargs['SNR_lim']:.1f}"
            elif self.kwargs["band"] < -1:
                selection_name = funcs.ordinal(abs(self.kwargs["band"])) + \
                    f"_reddest_band_SNR{sign}{self.kwargs['SNR_lim']:.1f}"
        return selection_name

    @property
    def _include_kwargs(self) -> List[str]:
        return ["band", "SNR_lim", "detect_or_non_detect"]

    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["band"], (int, str))])
            if isinstance(self.kwargs["band"], str):
                assertions.extend([self.kwargs["band"] in json.loads(config.get("Other", "ALL_BANDS"))])
            assertions.extend([isinstance(self.kwargs["SNR_lim"], (int, float))])
            assertions.extend([self.kwargs["detect_or_non_detect"].lower() in ["detect", "non_detect"]])
            passed = all(assertions)
        except:
            passed = False
        return passed
    
    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> bool:
        if isinstance(self.kwargs["band"], str):
            try:
                failed = self.kwargs["band"] not in \
                    gal.aper_phot[self.aper_diam].filterset.band_names
            except:
                failed = True
            return failed
        else:
            return False
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> bool:
        if isinstance(self.kwargs["band"], str):
            band_index = int(np.where(np.array( \
                gal.aper_phot[self.aper_diam].filterset.band_names) \
                == self.kwargs["band"])[0][0])
        else:
            band_index = self.kwargs["band"]
        SNR = gal.aper_phot[self.aper_diam].SNR[band_index]
        if isinstance(gal.aper_phot[self.aper_diam].flux, u.Quantity):
            masked = False
        else:
            masked = gal.aper_phot[self.aper_diam].flux.mask[band_index]
        # fails if masked
        return (
            not masked
            and ((self.kwargs["detect_or_non_detect"].lower() \
            == "detect" and SNR > self.kwargs["SNR_lim"])
            or (self.kwargs["detect_or_non_detect"].lower() \
            == "non_detect" and SNR < self.kwargs["SNR_lim"]))
        )

    def _call_cat(
        self: Self,
        cat: Catalogue,
        return_copy: bool = True,
    ) -> Union[NoReturn, Catalogue]:
        if isinstance(self.kwargs["band"], str):
            assert (self.kwargs["band"] in cat.filterset.band_names), \
                galfind_logger.critical(
                    f"{self.kwargs['band']} not in {cat.filterset.band_names}."
                )
        return Photometry_Selector._call_cat(self, cat, return_copy)


class Band_Mag_Selector(Photometry_Selector):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        band: Union[str, int],
        detect_or_non_detect: str,
        mag_lim: Union[int, float],
    ):
        kwargs = {
            "band": band,
            "detect_or_non_detect": detect_or_non_detect,
            "mag_lim": mag_lim,
        }
        super().__init__(aper_diam, **kwargs)

    @property
    def _selection_name(self) -> str:
        if self.kwargs["detect_or_non_detect"].lower() == "detect":
            sign = "<"
        else: # self.kwargs["detect_or_non_detect"].lower() == "non_detect"
            sign = ">"
        if isinstance(self.kwargs["band"], str):
            selection_name = self.kwargs['band'] + \
                f"_mag{sign}{self.kwargs['mag_lim']:.1f}"
        else: # isinstance(self.kwargs["band"], int):
            galfind_logger.debug(
                "Indexing e.g. 2 and -4 when there are 6 bands " + \
                f"results in differing {self.__class__.__name__} selection " + \
                "names even though the same band is referenced!"
            )
            if self.kwargs["band"] == 0:
                selection_name = "bluest_band_mag" + \
                    f"{sign}{self.kwargs['mag_lim']:.1f}"
            elif self.kwargs["band"] == -1:
                selection_name = f"reddest_band_mag" + \
                    f"{sign}{self.kwargs['mag_lim']:.1f}"
            elif self.kwargs["band"] > 0:
                selection_name = funcs.ordinal(self.kwargs["band"] + 1) + \
                    f"_bluest_band_mag{sign}{self.kwargs['mag_lim']:.1f}"
            elif self.kwargs["band"] < -1:
                selection_name = funcs.ordinal(abs(self.kwargs["band"])) + \
                    f"_reddest_band_mag{sign}{self.kwargs['mag_lim']:.1f}"
        return selection_name

    @property
    def _include_kwargs(self) -> List[str]:
        return ["band", "mag_lim", "detect_or_non_detect"]

    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["band"], (int, str))])
            if isinstance(self.kwargs["band"], str):
                assertions.extend([self.kwargs["band"] in json.loads(config.get("Other", "ALL_BANDS"))])
            assertions.extend([isinstance(self.kwargs["mag_lim"], (int, float))])
            assertions.extend([self.kwargs["detect_or_non_detect"].lower() in ["detect", "non_detect"]])
            passed = all(assertions)
        except:
            passed = False
        return passed
    
    def _failure_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> bool:
        if isinstance(self.kwargs["band"], str):
            try:
                failed = self.kwargs["band"] not in \
                    gal.aper_phot[self.aper_diam].filterset.band_names
            except:
                failed = True
            return failed
        else:
            return False
        
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> bool:
        if isinstance(self.kwargs["band"], str):
            band_index = int(np.where(np.array( \
                gal.aper_phot[self.aper_diam].filterset.band_names) \
                == self.kwargs["band"])[0][0])
        else:
            band_index = self.kwargs["band"]
        mag = funcs.convert_mag_units(
            gal.aper_phot[self.aper_diam].filterset[band_index].WavelengthCen,
            gal.aper_phot[self.aper_diam].flux[band_index],
            u.ABmag,
        ).value
        if isinstance(gal.aper_phot[self.aper_diam].flux, u.Quantity):
            masked = False
        else:
            masked = gal.aper_phot[self.aper_diam].flux.mask[band_index]
        # fails if masked
        return (
            not masked
            and ((self.kwargs["detect_or_non_detect"].lower() \
            == "detect" and mag < self.kwargs["mag_lim"])
            or (self.kwargs["detect_or_non_detect"].lower() \
            == "non_detect" and mag > self.kwargs["mag_lim"]))
        )

    def _call_cat(
        self: Self,
        cat: Catalogue,
        return_copy: bool = True,
    ) -> Union[NoReturn, Catalogue]:
        if isinstance(self.kwargs["band"], str):
            assert (self.kwargs["band"] in cat.filterset.band_names), \
                galfind_logger.critical(
                    f"{self.kwargs['band']} not in {cat.filterset.band_names}."
                )
        return Photometry_Selector._call_cat(self, cat, return_copy)


# TODO: UVJ_Selector should be specific 
# case of a photometric property selector

# class UVJ_Selector(Selector):

#     @property
#     def _selection_name(self) -> str:
#         return f"UVJ_{self.kwargs['quiescent_or_star_forming'].lower()}"

#     @property
#     def _include_kwargs(self) -> List[str]:
#         return ["quiescent_or_star_forming"]

#     def _assertions(self: Self) -> bool:
#         try:
#             failed = self.kwargs["quiescent_or_star_forming"].lower() \
#                 not in ["quiescent", "star_forming"]
#         except:
#             failed = True
#         return failed
    
#     def _failure_criteria(
#         self: Self,
#         gal: Galaxy,
#         aper_diam: u.Quantity,
#         SED_fit_label: str,
#     ) -> bool:
#         try:
#             assertions = [f"{band}_flux" not in \
#                 gal.aper_phot[aper_diam].SED_results \
#                 [SED_fit_label].properties.keys() \
#                 for band in ["U", "V", "J"]]
#             failed = not all(assertions)
#         except:
#             failed = True
#         return failed
        
#     def _selection_criteria(
#         self: Self,
#         gal: Galaxy,
#         aper_diam: u.Quantity,
#         SED_fit_label: str,
#     ) -> bool:
#         # extract UVJ colours
#         U_minus_V = -2.5 * np.log10(
#             (
#                 gal.aper_phot[aper_diam].SED_results[SED_fit_label].properties["U_flux"]
#                 / gal.aper_phot[aper_diam].SED_results[SED_fit_label].properties["V_flux"]
#             )
#             .to(u.dimensionless_unscaled).value
#         )
#         V_minus_J = -2.5 * np.log10(
#             (
#                 gal.aper_phot[aper_diam].SED_results[SED_fit_label].properties["V_flux"]
#                 / gal.aper_phot[aper_diam].SED_results[SED_fit_label].properties["J_flux"]
#             )
#             .to(u.dimensionless_unscaled).value
#         )
#         # selection from Antwi-Danso2022
#         is_quiescent = (
#             U_minus_V > 1.23
#             and V_minus_J < 1.67
#             and U_minus_V > V_minus_J * 0.98 + 0.38
#         )
#         return (self.kwargs["quiescent_or_star_forming"].lower() \
#             == "quiescent" and is_quiescent) or \
#             (self.kwargs["quiescent_or_star_forming"].lower() \
#             == "star_forming" and not is_quiescent)


class Chi_Sq_Lim_Selector(SED_fit_Selector):
 
    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, SED_code],
        chi_sq_lim: Union[int, float],
        reduced: bool,
    ):
        kwargs = {
            "chi_sq_lim": chi_sq_lim,
            "reduced": reduced,
        }
        super().__init__(aper_diam, SED_fit_label, **kwargs)

    @property
    def _selection_name(self) -> str:
        selection_name = f"chi_sq<{self.kwargs['chi_sq_lim']:.1f}"
        if self.kwargs["reduced"]:
            selection_name = f"red_{selection_name}"
        return selection_name

    @property
    def _include_kwargs(self) -> List[str]:
        return ["chi_sq_lim", "reduced"]
    
    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["chi_sq_lim"], (int, float))])
            assertions.extend([isinstance(self.kwargs["reduced"], bool)])
            passed = all(assertions)
        except:
            passed = False
        return passed

    def _selection_criteria(
        self: Self,
        gal: Galaxy,
    ) -> bool:
        chi_sq_lim = self.kwargs["chi_sq_lim"]
        if self.kwargs["reduced"]:
            if isinstance(gal.aper_phot[self.aper_diam].flux, u.Quantity):
                n_bands = len(gal.aper_phot[self.aper_diam].filterset.band_names)
            else:
                n_bands = len(
                    [
                        mask_band
                        for mask_band in gal.aper_phot[self.aper_diam].flux.mask
                        if not mask_band
                    ]
                )
            chi_sq_lim *= n_bands - 1
        chi_sq = gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label].chi_sq
        return chi_sq < chi_sq_lim


class Chi_Sq_Diff_Selector(SED_fit_Selector):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, SED_code],
        chi_sq_diff: Union[int, float],
        dz: Union[int, float],
    ):
        kwargs = {
            "chi_sq_diff": chi_sq_diff,
            "dz": dz,
        }
        super().__init__(aper_diam, SED_fit_label, **kwargs)

    @property
    def _selection_name(self) -> str:
        chi_sq_name = f"chi_sq_diff>{self.kwargs['chi_sq_diff']:.1f}"
        dz_name = f"dz>{self.kwargs['dz']:.1f}"
        return f"{chi_sq_name},{dz_name}"
    
    @property
    def _include_kwargs(self) -> List[str]:
        return ["chi_sq_diff", "dz"]
    
    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["chi_sq_diff"], (int, float))])
            assertions.extend([isinstance(self.kwargs["dz"], (int, float))])
            assertions.extend([self.kwargs["dz"] > 0.0])
            assertions.extend([self.kwargs["chi_sq_diff"] >= 0.0])
            passed = all(assertions)
        except:
            passed = False
        return passed

    def _failure_criteria(
        self: Self,
        gal: Galaxy,
    ) -> bool:
        try:
            assertions = []
            assertions.extend([self.SED_fit_label in gal.aper_phot[self.aper_diam].SED_results.keys()])
            gal_SED_fit_labels = self._get_lowz_SED_fit_labels(gal)
            assertions.extend([len(gal_SED_fit_labels) > 0])
            failed = not all(assertions)
        except:
            failed = True
        return failed
    
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
    ) -> bool:
        # extract redshift + chi_sq of zfree run
        zfree = gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label].z
        chi_sq_zfree = gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label].chi_sq
        # extract redshift and chi_sq of lowz runs
        lowz_SED_fit_labels = [i for i in filter( \
            lambda label: float(label.split("zmax=")[-1][:3]) \
            < zfree - self.kwargs["dz"], \
            self._get_lowz_SED_fit_labels(gal)
        )]
        # if no lowz runs, do not select galaxy
        if len(lowz_SED_fit_labels) == 0:
            return False
        else:
            # sort the lowz_SED_fit_labels to get highest redshift applicable lowz run
            lowz_SED_fit_label = sorted(lowz_SED_fit_labels, reverse = True, \
                key = lambda label: float(label.split("zmax=")[-1][:3]))[0]
        z_lowz = gal.aper_phot[self.aper_diam].SED_results[lowz_SED_fit_label].z
        chi_sq_lowz = gal.aper_phot[self.aper_diam].SED_results[lowz_SED_fit_label].chi_sq
        return (
            (chi_sq_lowz - chi_sq_zfree > self.kwargs["chi_sq_diff"])
            or (chi_sq_lowz == -1.0)
            or (z_lowz < 0.0)
        )
    
    def _get_lowz_SED_fit_labels(
        self: Self,
        gal: Galaxy,
    ) -> List[str]:
        # TODO: Works for EAZY, but not for a general 
        # SED fitting code with different zmax syntax
        return [label for label in \
            gal.aper_phot[self.aper_diam].SED_results.keys() \
            if "zmax=" in label and label.replace( \
            f"_zmax={label.split('zmax=')[-1][:3]}", "") \
            in self.SED_fit_label]
    
    def _call_cat(
        self: Self,
        cat: Catalogue,
        return_copy: bool = True,
    ) -> Union[NoReturn, Catalogue]:
        # ensure a lowz run has been run for at least 1 galaxy in the catalogue
        cat_SED_fit_labels = [self._get_lowz_SED_fit_labels(gal) for gal in cat]
        assert any(len(gal_labels) > 0 for gal_labels in cat_SED_fit_labels), \
            galfind_logger.critical(
                f"{self.SED_fit_label} lowz not run for any galaxy."
            )
        return SED_fit_Selector._call_cat(self, cat, return_copy)


class Chi_Sq_Template_Diff_Selector(SED_fit_Selector):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, SED_code],
        chi_sq_diff: Union[int, float],
        secondary_SED_fit_label: Union[str, SED_code],
        reduced: bool = False,
    ):
        if isinstance(secondary_SED_fit_label, tuple(SED_code.__subclasses__())):
            secondary_SED_fit_label = SED_fit_label.label
        kwargs = {
            "chi_sq_diff": chi_sq_diff,
            "secondary_SED_fit_label": secondary_SED_fit_label,
            "reduced": reduced,
        }
        super().__init__(aper_diam, SED_fit_label, **kwargs)

    @property
    def _selection_name(self) -> str:
        name = f"chi_sq_{self.kwargs['secondary_SED_fit_label']}>{self.kwargs['chi_sq_diff']:.1f}"
        if self.kwargs["reduced"]:
            name = f"red_{name}"
        return name
    
    @property
    def _include_kwargs(self) -> List[str]:
        return ["chi_sq_diff", "secondary_SED_fit_label", "reduced"]
    
    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["chi_sq_diff"], (int, float))])
            assertions.extend([self.kwargs["chi_sq_diff"] >= 0.0])
            assertions.extend([isinstance(self.kwargs["secondary_SED_fit_label"], str)])
            assertions.extend([isinstance(self.kwargs["reduced"], bool)])
            passed = all(assertions)
        except:
            passed = False
        return passed

    def _failure_criteria(
        self: Self,
        gal: Galaxy,
    ) -> bool:
        try:
            failed = not all(
                label in gal.aper_phot[self.aper_diam].SED_results.keys()
                for label in [self.SED_fit_label, self.kwargs["secondary_SED_fit_label"]]
            )
        except:
            failed = True
        return failed
    
    def _selection_criteria(
        self: Self,
        gal: Galaxy,
    ) -> bool:
        # extract chi_sq/red_chi_sq of SED_fitting runs that are to be compared
        chi_sq = np.zeros(2)
        for i, label in enumerate([self.SED_fit_label, self.kwargs["secondary_SED_fit_label"]]):
            if not any(hasattr(gal.aper_phot[self.aper_diam].SED_results[label], chi_sq_label) for chi_sq_label in ["chi_sq", "red_chi_sq"]):
                return False
            if self.kwargs["reduced"]:
                if hasattr(gal.aper_phot[self.aper_diam].SED_results[label], "red_chi_sq"):
                    chi_sq[i] = getattr(gal.aper_phot[self.aper_diam].SED_results[label], "red_chi_sq")
                else:
                    chi_sq_ = getattr(gal.aper_phot[self.aper_diam].SED_results[label], "chi_sq")
                    # work out number of degrees of freedom
                    if isinstance(gal.aper_phot[self.aper_diam].flux, u.Quantity):
                        n_bands = len(gal.aper_phot[self.aper_diam].filterset.band_names)
                    else:
                        n_bands = len(
                            [
                                mask_band
                                for mask_band in gal.aper_phot[self.aper_diam].flux.mask
                                if not mask_band
                            ]
                        )
                    Ndof = n_bands - 1
                    chi_sq[i] = chi_sq_ / Ndof
            else: # absolute chi squared
                if hasattr(gal.aper_phot[self.aper_diam].SED_results[label], "chi_sq"):
                    chi_sq[i] = getattr(gal.aper_phot[self.aper_diam].SED_results[label], "chi_sq")
                else:
                    chi_sq_ = getattr(gal.aper_phot[self.aper_diam].SED_results[label], "red_chi_sq")
                    # work out number of degrees of freedom
                    if isinstance(gal.aper_phot[self.aper_diam].flux, u.Quantity):
                        n_bands = len(gal.aper_phot[self.aper_diam].filterset.band_names)
                    else:
                        n_bands = len(
                            [
                                mask_band
                                for mask_band in gal.aper_phot[self.aper_diam].flux.mask
                                if not mask_band
                            ]
                        )
                    Ndof = n_bands - 1
                    chi_sq[i] = chi_sq_ * Ndof
        return (
            (chi_sq[1] - chi_sq[0] > self.kwargs["chi_sq_diff"])
            or (chi_sq[1] < 0.0)
        )
    
    def _call_cat(
        self: Self,
        cat: Catalogue,
        return_copy: bool = True,
    ) -> Union[NoReturn, Catalogue]:
        # ensure a secondary SED fitting run has been run for at least 1 galaxy in the catalogue
        cat_SED_fit_labels = [gal.aper_phot[self.aper_diam].SED_results.keys() for gal in cat]
        assert any(self.kwargs["secondary_SED_fit_label"] in gal_labels for gal_labels in cat_SED_fit_labels), \
            galfind_logger.critical(
                f"{self.kwargs['secondary_SED_fit_label']} not run for any galaxy."
            )
        return SED_fit_Selector._call_cat(self, cat, return_copy)


class Robust_zPDF_Selector(SED_fit_Selector):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, SED_code],
        integral_lim: float,
        dz_over_z: Union[int, float],
        min_dz: Union[int, float] = 0.0,
    ):
        kwargs = {
            "integral_lim": integral_lim,
            "dz_over_z": dz_over_z,
            "min_dz": min_dz,
        }
        super().__init__(aper_diam, SED_fit_label, **kwargs)

    @property
    def _selection_name(self) -> str:
        name = f"zPDF>{int(self.kwargs['integral_lim'] * 100)}%," + \
            f"|dz|/z<{self.kwargs['dz_over_z']}"
        if self.kwargs["min_dz"] > 0.0:
            name += f",|dz|>{self.kwargs['min_dz']:.1f}"
        return name

    @property
    def _include_kwargs(self) -> List[str]:
        return ["integral_lim", "dz_over_z", "min_dz"]

    def _assertions(self: Self) -> bool:
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["integral_lim"], float)])
            assertions.extend([self.kwargs["integral_lim"] > 0.0])
            assertions.extend([(self.kwargs["integral_lim"] * 100).is_integer()])
            assertions.extend([isinstance(self.kwargs["dz_over_z"], (int, float))])
            assertions.extend([self.kwargs["dz_over_z"] > 0.0])
            assertions.extend([isinstance(self.kwargs["min_dz"], (int, float))])
            assertions.extend([self.kwargs["min_dz"] >= 0.0])

            passed = all(assertions)
        except:
            passed = False
        return passed

    def _failure_criteria(
        self: Self,
        gal: Galaxy,
    ) -> bool:
        try:
            failed = gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label].z < 0.0
        except:
            failed = True
        return failed

    def _selection_criteria(
        self: Self,
        gal: Galaxy,
    ) -> bool:

        # extract best fitting redshift - peak of the redshift PDF
        zbest = gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label].z

        dz_z = self.kwargs["dz_over_z"]
        # adjust dz_z if zbest * dz_z is less than min_dz
        if zbest * dz_z < self.kwargs["min_dz"]:
            dz_z = self.kwargs["min_dz"] / zbest

        integral = gal.aper_phot[self.aper_diam].SED_results[self.SED_fit_label]. \
            property_PDFs["z"].integrate_between_lims(
                float(dz_z), float(zbest)
            )
        return integral > self.kwargs["integral_lim"]


class Sextractor_Band_Radius_Selector(Data_Selector):

    def __init__(
        self: Self,
        band_name: str,
        gtr_or_less: str,
        lim: u.Quantity,
    ):
        kwargs = {
            "band_name": band_name,
            "gtr_or_less": gtr_or_less,
            "lim": lim,
        }
        super().__init__(**kwargs)

    @property
    def _selection_name(self) -> str:
        lim_str = f"{self.kwargs['lim'].to(u.marcsec).value:.1f}mas"
        gtr_or_less_str = ">" if self.kwargs["gtr_or_less"].lower() == "gtr" else "<"
        return f"sex_Re_{self.kwargs['band_name']}{gtr_or_less_str}{lim_str}"

    @property
    def _include_kwargs(self) -> List[str]:
        return ["band_name", "gtr_or_less", "lim"]

    def _assertions(self: Self) -> bool:
        try:
            self.kwargs["lim"].to(u.marcsec)
            assertions = []
            assertions.extend([isinstance(self.kwargs["band_name"], str)])
            assertions.extend([self.kwargs["band_name"] in json.loads(config.get("Other", "ALL_BANDS"))])
            assertions.extend([self.kwargs["gtr_or_less"].lower() in ["gtr", "less"]])
            assertions.extend([isinstance(self.kwargs["lim"], u.Quantity)])
            assertions.extend([self.kwargs["lim"].value > 0.0])
            passed = all(assertions)
        except:
            passed = False
        return passed

    def _selection_criteria(
        self: Self,
        gal: Galaxy,
        *args,
        **kwargs
    ) -> bool:
        if self.kwargs["gtr_or_less"].lower() == "gtr":
            return gal.sex_Re[self.kwargs["band_name"]] > self.kwargs["lim"]
        else: # self.kwargs["gtr_or_less"].lower() == "less"
            return gal.sex_Re[self.kwargs["band_name"]] < self.kwargs["lim"]

    def _call_cat(
        self: Self,
        cat: Catalogue,
        return_copy: bool = True,
    ) -> Union[NoReturn, Catalogue]:
        if isinstance(self.kwargs["band_name"], str):
            assert (self.kwargs["band_name"] in cat.filterset.band_names), \
                galfind_logger.critical(
                    f"{self.kwargs['band_name']} not in" + \
                    f" {cat.filterset.band_names}!"
                )
        # load in effective radii as calculated from SExtractor
        cat.load_sextractor_Re()
        return Data_Selector._call_cat(self, cat, return_copy)


class Re_Selector(Morphology_Selector):

    def __init__(
        self: Self,
        morph_fitter: Type[Morphology_Fitter],
        gtr_or_less: str,
        lim: u.Quantity,
    ):
        kwargs = {
            "gtr_or_less": gtr_or_less,
            "lim": lim,
        }
        super().__init__(morph_fitter, **kwargs)

    @property
    def _selection_name(self) -> str:
        lim_str = f"{self.kwargs['lim'].to(u.marcsec).value:.1f}mas"
        gtr_or_less_str = ">" if self.kwargs["gtr_or_less"].lower() == "gtr" else "<"
        return f"Re{gtr_or_less_str}{lim_str}"

    @property
    def _include_kwargs(self) -> List[str]:
        return ["gtr_or_less", "lim"]

    def _assertions(self):
        try:
            assertions = []
            assertions.extend([self.kwargs["gtr_or_less"].lower() in ["gtr", "less"]])
            assertions.extend([isinstance(self.kwargs["lim"], u.Quantity)])
            assertions.extend([self.kwargs["lim"].value > 0.0])
            passed = all(assertions)
        except:
            passed = False
        return passed
    
    def _selection_criteria(self, gal):
        re_pix = gal.cutouts[self._cutout_str].morph_fits[self.morph_fitter.name].r_e
        if np.isnan(re_pix):
            return False
        pix_scale = self.morph_fitter.psf.cutout.band_data.pix_scale / u.pix
        re_as = re_pix * pix_scale
        if self.kwargs["gtr_or_less"].lower() == "gtr":
            return re_as > self.kwargs["lim"]
        else: # self.kwargs["gtr_or_less"].lower() == "less"
            return re_as < self.kwargs["lim"]
    
class Kokorev24_LRD_red1_Selector(Multiple_Photometry_Selector):

    def __init__(
        self: Self, 
        aper_diam: u.Quantity
    ):
        super().__init__(aper_diam, 
        [
            Colour_Selector(
                aper_diam,
                colour_bands = ["F115W", "F150W"],
                bluer_or_redder = "bluer",
                colour_val = 0.8
            ),
            Colour_Selector(
                aper_diam,
                colour_bands = ["F200W", "F277W"],
                bluer_or_redder = "redder",
                colour_val = 0.7
            ),
            Colour_Selector(
                aper_diam,
                colour_bands = ["F200W", "F356W"],
                bluer_or_redder = "redder",
                colour_val = 1.0
            ),
        ], 
        selection_name = "Kokorev+24_LRD_red1")

class Kokorev24_LRD_red2_Selector(Multiple_Photometry_Selector):

    def __init__(
        self: Self, 
        aper_diam: u.Quantity
    ):
        super().__init__(aper_diam,
        [
            Colour_Selector(
                aper_diam,
                colour_bands = ["F150W", "F200W"],
                bluer_or_redder = "bluer",
                colour_val = 0.8
            ),
            Colour_Selector(
                aper_diam,
                colour_bands = ["F277W", "F356W"],
                bluer_or_redder = "redder",
                colour_val = 0.6
            ),
            Colour_Selector(
                aper_diam,
                colour_bands = ["F277W", "F444W"],
                bluer_or_redder = "redder",
                colour_val = 0.7
            ),
        ], 
        selection_name = "Kokorev+24_LRD_red2")

class Kokorev24_LRD_Selector(Multiple_Photometry_Selector):

    def __init__(
        self: Self,
        aper_diam: u.Quantity
    ):
        super().__init__(aper_diam,
        [
            Kokorev24_LRD_red1_Selector(aper_diam), 
            Kokorev24_LRD_red2_Selector(aper_diam), 
        ], 
        selection_name = "Kokorev+24_LRD")


class Unmasked_Bands_Selector(Multiple_Mask_Selector):

    def __init__(
        self: Self, 
        band_names: Union[str, List[str]]
    ):
        if isinstance(band_names, str):
            band_names = band_names.split("+")
        selectors = [Unmasked_Band_Selector(band_name = name) for name in band_names]
        super().__init__(selectors, f"unmasked_{'+'.join(band_names)}")


class Unmasked_Instrument_Selector(Multiple_Mask_Selector):

    def __init__(
        self: Self, 
        instrument: Union[str, Type[Instrument]],
        cat_filterset: Optional[Multiple_Filter] = None
    ):
        if isinstance(instrument, str):
            assert instrument in expected_instr_bands.keys(), \
                galfind_logger.critical(
                    f"{instrument=} not a valid instrument name."
                )
            instrument = [instr() for instr in Instrument.__subclasses__() \
                if instr.__name__ == instrument][0]
        selectors = [Unmasked_Band_Selector(band_name = name) for name in instrument.filt_names]
        super().__init__(selectors, f"unmasked_{instrument.__class__.__name__}", cat_filterset = cat_filterset)

    def _call_cat(
        self: Self,
        cat: Catalogue,
        return_copy: bool = True,
    ) -> Union[NoReturn, Catalogue]:
        self.crop_to_filterset(cat.filterset)
        return Multiple_Data_Selector._call_cat(self, cat, return_copy)


# class Colour_Colour_Selector(Multiple_Photometry_Selector):
   
#     @property
#     def _selection_name(self: str) -> str:
#        raise NotImplementedError
   
#     @property
#     def _include_kwargs(self: List[str]) -> List[str]:
#         return ["colour_bands_arr", "select_func"]


class Sextractor_Bands_Radius_Selector(Multiple_Data_Selector):

    def __init__(
        self: Self,
        band_names: List[str],
        gtr_or_less: str,
        lim: u.Quantity
    ):
        selectors = [Sextractor_Band_Radius_Selector(
            band_name = band_name, gtr_or_less = gtr_or_less, lim = lim)
            for band_name in band_names]
        lim_str = f"{lim.to(u.marcsec).value:.1f}mas"
        gtr_or_less_str = ">" if gtr_or_less.lower() == "gtr" else "<"
        selection_name = f"sex_Re_{'+'.join(band_names)}{gtr_or_less_str}{lim_str}"
        super().__init__(selectors, selection_name)


class Sextractor_Instrument_Radius_Selector(Multiple_Data_Selector):

    def __init__(
        self: Self,
        instrument: str,
        gtr_or_less: str,
        lim: u.Quantity,
        cat_filterset: Optional[Multiple_Filter] = None
    ):
        if isinstance(instrument, str):
            assert instrument in expected_instr_bands.keys(), \
                galfind_logger.critical(
                    f"{instrument=} not a valid instrument name."
                )
            instrument = [instr() for instr in Instrument.__subclasses__() \
                if instr.__name__ == instrument][0]
        selectors = [Sextractor_Band_Radius_Selector(
            band_name = band_name, gtr_or_less = gtr_or_less, lim = lim) \
            for band_name in instrument.filt_names]
        lim_str = f"{lim.to(u.marcsec).value:.1f}mas"
        gtr_or_less_str = ">" if gtr_or_less.lower() == "gtr" else "<"
        selection_name = f"sex_Re_{instrument.__class__.__name__}{gtr_or_less_str}{lim_str}"
        super().__init__(selectors, selection_name, cat_filterset = cat_filterset)
    
    def _call_cat(
        self: Self,
        cat: Catalogue,
        return_copy: bool = True,
    ) -> Union[NoReturn, Catalogue]:
        self.crop_to_filterset(cat.filterset)
        return Multiple_Data_Selector._call_cat(self, cat, return_copy)


class Sextractor_Instrument_Radius_PSF_FWHM_Selector(Multiple_Data_Selector):

    def __init__(
        self: Self,
        instrument: str,
        gtr_or_less: str,
        scaling: float = 1.0,
        cat_filterset: Optional[Multiple_Filter] = None
    ):
        if isinstance(instrument, str):
            assert instrument in expected_instr_bands.keys(), \
                galfind_logger.critical(
                    f"{instrument=} not a valid instrument name."
                )
            instrument = [instr() for instr in Instrument.__subclasses__() \
                if instr.__name__ == instrument][0]
        assert instrument.__class__.__name__ == "NIRCam", \
            galfind_logger.critical(
                f"{instrument.name=} must be NIRCam."
            )
        selectors = [Sextractor_Band_Radius_Selector(
            band_name = band_name, gtr_or_less = gtr_or_less, lim = fwhm_nircam[band_name] * scaling) \
            for band_name in instrument.filt_names if band_name in fwhm_nircam.keys()]
        gtr_or_less_str = ">" if gtr_or_less.lower() == "gtr" else "<"
        if scaling != 1.0:
            fwhm_name = f"{scaling:.1f}*psf_fwhm"
        else:
            fwhm_name = "psf_fwhm"
        selection_name = f"sex_Re{gtr_or_less_str}{instrument.__class__.__name__}_{fwhm_name}"
        super().__init__(selectors, selection_name, cat_filterset = cat_filterset)
    
    def _call_cat(
        self: Self,
        cat: Catalogue,
        return_copy: bool = True,
    ) -> Union[NoReturn, Catalogue]:
        self.crop_to_filterset(cat.filterset)
        assert all([filt.band_name in fwhm_nircam.keys() for filt in cat.filterset if filt.instrument_name == "NIRCam"]), \
            galfind_logger.critical(
                f"{cat.filterset.instrument_name=} must have all NIRCam bands."
            )
        return Multiple_Data_Selector._call_cat(self, cat, return_copy)

class Brown_Dwarf_Selector(Multiple_SED_fit_Selector):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, SED_code],
        unmasked_instruments: Union[str, List[str]] = "NIRCam",
        red_chi_sq_lim: float = 6.0,
        secondary_SED_fit_label: Optional[Union[str, SED_code]] = None,
        cat_filterset: Optional[Multiple_Filter] = None,
        select_band: str = "F150W",
        morph_cut: bool = True,
        #psf_like_bands: Optional[List[str]] = None,
        #psf_like_SNRs: Optional[List[float]] = None,
    ):
        selectors = [
            Chi_Sq_Lim_Selector(aper_diam, SED_fit_label, chi_sq_lim = red_chi_sq_lim, reduced = True),
            Band_SNR_Selector(aper_diam, band = select_band, SNR_lim = 5.0, detect_or_non_detect = "detect"),
        ]
        if morph_cut:
            from . import Filter, PSF_Cutout, Galfit_Fitter
            filt = Filter.from_filt_name(select_band)
            psf_path = f"/nvme/scratch/work/westcottl/psf/PSF_Resample_03_{select_band}.fits"
            assert Path(psf_path).is_file(), \
                galfind_logger.critical(
                    f"{psf_path=} does not exist."
                )
            psf = PSF_Cutout.from_fits(
                fits_path=psf_path,
                filt=filt,
                unit="adu",
                pix_scale = 0.03 * u.arcsec,
                size = 0.96 * u.arcsec,
            )
            galfit_fitter = Galfit_Fitter(psf, "sersic", fixed_params = ["n"])
            assert select_band in fwhm_nircam, \
                galfind_logger.critical(
                    f"{select_band=} not in fwhm_nircam."
                )
            selectors.extend([Re_Selector(galfit_fitter, gtr_or_less = "less", lim = 1.2 * fwhm_nircam[select_band])])
            morph_name = "_psf"
        else:
            morph_name = ""
        
        # add hot pixel checks in LW widebands
        selectors.extend([
            Sextractor_Bands_Radius_Selector( \
            band_names = ["F277W", "F356W", "F444W"], \
            gtr_or_less = "gtr", lim = 45.0 * u.marcsec)
        ])
        
        chi2_label = f"_red_chi2<{red_chi_sq_lim:.1f}"
        if secondary_SED_fit_label is None:
            gal_chi2_compar_label = chi2_label
        else:
            if isinstance(secondary_SED_fit_label, tuple(SED_code.__subclasses__())):
                secondary_SED_fit_label = secondary_SED_fit_label.label
            selectors.extend([
                Chi_Sq_Template_Diff_Selector(
                    aper_diam,
                    SED_fit_label,
                    chi_sq_diff = 0.0,
                    secondary_SED_fit_label = secondary_SED_fit_label,
                    reduced = True,
                )
            ])
            gal_chi2_compar_label = f"{chi2_label},{secondary_SED_fit_label}"
        
        # add unmasked instrument selections
        if isinstance(unmasked_instruments, str):
            unmasked_instruments = unmasked_instruments.split("+")
        selectors.extend([Unmasked_Instrument_Selector(instrument, \
            cat_filterset) for instrument in unmasked_instruments])
        unmasked_instr_name = f"_{'+'.join(unmasked_instruments)}"

        name = f"bd{gal_chi2_compar_label}{unmasked_instr_name}_{select_band}{morph_name}"
        super().__init__(aper_diam, SED_fit_label, selectors, selection_name = name)

    @property
    def name(self: Self) -> str:
        return f"{self._selection_name}_{self.aper_diam.to(u.arcsec).value:.2f}as"


class Hainline24_TY_Brown_Dwarf_Selector_1(Multiple_Photometry_Selector):
    
    def __init__(
        self: Self,
        aper_diam: u.Quantity,
    ):
        super().__init__(aper_diam,
            [
                Colour_Selector(
                    aper_diam,
                    colour_bands = ["F277W", "F444W"],
                    bluer_or_redder = "redder",
                    colour_val = 1.0
                ),
                Colour_Selector(
                    aper_diam,
                    colour_bands = ["F115W", "F277W"],
                    bluer_or_redder = "bluer",
                    colour_val = 0.0
                ),
                Band_Mag_Selector(
                    aper_diam,
                    band = "F444W",
                    mag_lim = 28.5,
                    detect_or_non_detect = "detect"
                ),
            ],
            selection_name = "Hainline+24_bd_1"
        )


class Hainline24_TY_Brown_Dwarf_Selector_2(Multiple_Photometry_Selector):
    
    def __init__(
        self: Self,
        aper_diam: u.Quantity,
    ):
        super().__init__(aper_diam,
            [
                Colour_Selector(
                    aper_diam,
                    colour_bands = ["F277W", "F444W"],
                    bluer_or_redder = "redder",
                    colour_val = 1.0
                ),
                Colour_Selector(
                    aper_diam,
                    colour_bands = ["F115W", "F150W"],
                    bluer_or_redder = "bluer",
                    colour_val = 0.0
                ),
                Band_Mag_Selector(
                    aper_diam,
                    band = "F444W",
                    mag_lim = 28.5,
                    detect_or_non_detect = "detect"
                ),
            ], 
            selection_name = "Hainline+24_bd_2"
        )


class EPOCHS_Selector(Multiple_SED_fit_Selector):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, SED_code],
        allow_lowz: bool = False,
        unmasked_instruments: Union[str, List[str]] = "NIRCam", # TODO: Update this to allow for custom masking or change to default "EPOCHS" masking
        cat_filterset: Optional[Multiple_Filter] = None,
        simulated: bool = False,
    ):
        selectors = [
            Bluewards_Lya_Non_Detect_Selector(aper_diam, SED_fit_label, SNR_lim = 2.0),
            Redwards_Lya_Detect_Selector(aper_diam, SED_fit_label, SNR_lims = [5.0, 5.0], widebands_only = True),
            Redwards_Lya_Detect_Selector(aper_diam, SED_fit_label, SNR_lims = 2.0, widebands_only = True),
            Chi_Sq_Lim_Selector(aper_diam, SED_fit_label, chi_sq_lim = 3.0, reduced = True),
            Chi_Sq_Diff_Selector(aper_diam, SED_fit_label, chi_sq_diff = 4.0, dz = 0.5),
            Robust_zPDF_Selector(aper_diam, SED_fit_label, integral_lim = 0.6, dz_over_z = 0.1),
        ]
        # add 2σ non-detection in first band if wanted
        if not allow_lowz:
            selectors.extend([Band_SNR_Selector( \
                aper_diam, band = 0, SNR_lim = 2.0, detect_or_non_detect = "non_detect")])

        if not simulated:
            # add unmasked instrument selections
            if isinstance(unmasked_instruments, str):
                unmasked_instruments = unmasked_instruments.split("+")
            selectors.extend([Unmasked_Instrument_Selector(instrument, \
                cat_filterset) for instrument in unmasked_instruments])
            
            # add hot pixel checks in LW widebands
            selectors.extend([
                Sextractor_Bands_Radius_Selector( \
                band_names = ["F277W", "F356W", "F444W"], \
                gtr_or_less = "gtr", lim = 45. * u.marcsec)
            ])
        lowz_name = "_lowz" if allow_lowz else ""
        unmasked_instr_name = "_" + "+".join(unmasked_instruments)
        selection_name = f"EPOCHS{lowz_name}{unmasked_instr_name}"
        super().__init__(aper_diam, SED_fit_label, selectors, selection_name = selection_name)

# Catalogue Level Selection functions

# def select_all_bands(self):
#     return self.select_min_bands(len(self.instrument))

class Rest_Frame_Property_Kwarg_Selector(SED_fit_Selector):

    def __init__(
        self: Self,
        aper_diam: u.Quantity,
        SED_fit_label: Union[str, SED_code],
        property_calculator: Type[Rest_Frame_Property_Calculator],
        kwarg_name: str,
        kwarg_val: Union[int, float],
    ):
        # TODO: Add more assertions here to ensure the 
        # kwarg name is in the photometry_rest object
        kwargs = {
            "property_calculator": property_calculator,
            "kwarg_name": kwarg_name,
            "kwarg_val": kwarg_val,
        }
        super().__init__(aper_diam, SED_fit_label, **kwargs)

    @property
    def _selection_name(self) -> str:
        return self.kwargs["property_calculator"].name + \
            f"_{self.kwargs['kwarg_name']}={self.kwargs['kwarg_val']}"

    @property
    def _include_kwargs(self) -> List[str]:
        return ["property_calculator", "kwarg_name", "kwarg_val"]
    
    def _assertions(self: Self) -> bool:
        from . import Rest_Frame_Property_Calculator
        try:
            assertions = []
            assertions.extend([isinstance(self.kwargs["property_calculator"], \
                tuple(Rest_Frame_Property_Calculator.__subclasses__()))])
            assertions.extend([isinstance(self.kwargs["kwarg_name"], str)])
            passed = all(assertions)
        except:
            passed = False
        return passed
    
    def _failure_criteria(
        self: Self,
        gal: Galaxy,
    ) -> bool:
        if self.kwargs["kwarg_name"] not in gal.aper_phot[self.aper_diam].SED_results \
                [self.SED_fit_label].phot_rest.property_kwargs \
                [self.kwargs["property_calculator"].name].keys():
            return True
        else:
            return False

    def _selection_criteria(
        self: Self,
        gal: Galaxy,
    ) -> bool:
        return gal.aper_phot[self.aper_diam].SED_results \
            [self.SED_fit_label].phot_rest.property_kwargs \
            [self.kwargs["property_calculator"].name] \
            [self.kwargs["kwarg_name"]] == self.kwargs["kwarg_val"]


# Photometric galaxy property selection functions

    # def select_phot_galaxy_property(
    #     self,
    #     property_name,
    #     gtr_or_less,
    #     property_lim,
    #     SED_fit_params,
    #     update=True,
    # ):
    #     key = SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)
    #     assert property_name in self.phot.SED_results[key].properties.keys()
    #     galfind_logger.warning(
    #         "Ideally need to include appropriate units for photometric galaxy property selection"
    #     )
    #     assert type(property_lim) in [int, float]
    #     assert gtr_or_less in ["gtr", "less", ">", "<"]
    #     if gtr_or_less in ["gtr", ">"]:
    #         selection_name = f"{property_name}>{property_lim}"
    #     else:
    #         selection_name = f"{property_name}<{property_lim}"
    #     if selection_name in self.selection_flags.keys():
    #         galfind_logger.debug(
    #             f"{selection_name} already performed for galaxy ID = {self.ID}!"
    #         )
    #     else:
    #         if (
    #             len(self.phot) == 0
    #         ):  # no data at all (not sure why sextractor does this)
    #             if update:
    #                 self.selection_flags[selection_name] = False
    #             return self, selection_name
    #         property_val = self.phot.SED_results[key].properties[property_name]
    #         if (
    #             gtr_or_less in ["gtr", ">"] and property_val > property_lim
    #         ) or (
    #             gtr_or_less in ["less", "<"] and property_val < property_lim
    #         ):
    #             if update:
    #                 self.selection_flags[selection_name] = True
    #         else:
    #             if update:
    #                 self.selection_flags[selection_name] = False
    #     return self, selection_name

    # def select_phot_galaxy_property_bin(
    #     self,
    #     property_name,
    #     property_lims,
    #     SED_fit_params,
    #     update=True,
    # ):
    #     key = SED_fit_params["code"].label_from_SED_fit_params(SED_fit_params)
    #     assert property_name in self.phot.SED_results[key].properties.keys()
    #     galfind_logger.warning(
    #         "Ideally need to include appropriate units for photometric galaxy property selection"
    #     )
    #     assert type(property_lims) in [np.ndarray, list]
    #     assert len(property_lims) == 2
    #     assert property_lims[1] > property_lims[0]
    #     selection_name = (
    #         f"{property_lims[0]}<{property_name}<{property_lims[1]}"
    #     )
    #     if selection_name in self.selection_flags.keys():
    #         galfind_logger.debug(
    #             f"{selection_name} already performed for galaxy ID = {self.ID}!"
    #         )
    #     else:
    #         if (
    #             len(self.phot) == 0
    #         ):  # no data at all (not sure why sextractor does this)
    #             if update:
    #                 self.selection_flags[selection_name] = False
    #             return self, selection_name
    #         property_val = self.phot.SED_results[key].properties[property_name]
    #         if (
    #             property_val > property_lims[0]
    #             and property_val < property_lims[1]
    #         ):
    #             if update:
    #                 self.selection_flags[selection_name] = True
    #         else:
    #             if update:
    #                 self.selection_flags[selection_name] = False
    #     return self, selection_name

# Emission line selection functions

    # def select_rest_UV_line_emitters_dmag(
    #     self,
    #     emission_line_name,
    #     delta_m,
    #     rest_UV_wav_lims=[1_250.0, 3_000.0] * u.AA,
    #     medium_bands_only=True,
    #     SED_fit_params=EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
    #     update=True,
    # ):
    #     assert (
    #         line_diagnostics[emission_line_name]["line_wav"]
    #         > rest_UV_wav_lims[0] * rest_UV_wav_lims.unit
    #         and line_diagnostics[emission_line_name]["line_wav"]
    #         < rest_UV_wav_lims[1] * rest_UV_wav_lims.unit
    #     )
    #     assert type(delta_m) in [int, np.int64, float, np.float64]
    #     assert u.has_physical_type(rest_UV_wav_lims) == "length"
    #     assert type(medium_bands_only) in [bool, np.bool_]
    #     selection_name = f"{emission_line_name},dm{'_med' if medium_bands_only else ''}>{delta_m:.1f},UV_{str(list(np.array(rest_UV_wav_lims.value).astype(int))).replace(' ', '')}AA"
    #     if selection_name in self.selection_flags.keys():
    #         galfind_logger.debug(
    #             f"{selection_name} already performed for galaxy ID = {self.ID}!"
    #         )
    #     else:
    #         if (
    #             len(self.phot) == 0
    #         ):  # no data at all (not sure why sextractor does this)
    #             if update:
    #                 self.selection_flags[selection_name] = False
    #             return self, selection_name
    #         phot_rest = deepcopy(
    #             self.phot.SED_results[
    #                 SED_fit_params["code"].label_from_SED_fit_params(
    #                     SED_fit_params
    #                 )
    #             ].phot_rest
    #         )
    #         # find bands that the emission line lies within
    #         obs_frame_emission_line_wav = line_diagnostics[emission_line_name][
    #             "line_wav"
    #         ] * (1.0 + phot_rest.z)
    #         included_bands = self.phot.instrument.bands_from_wavelength(
    #             obs_frame_emission_line_wav
    #         )
    #         # determine index of the closest band to the emission line
    #         closest_band_index = (
    #             self.phot.instrument.nearest_band_index_to_wavelength(
    #                 obs_frame_emission_line_wav, medium_bands_only
    #             )
    #         )
    #         central_wav = self.phot.instrument[
    #             closest_band_index
    #         ].WavelengthCen
    #         # if there are no included bands or the closest band is masked
    #         if (
    #             len(included_bands) == 0
    #             or self.phot.flux.mask[closest_band_index]
    #         ):
    #             if update:
    #                 self.selection_flags[selection_name] = False
    #             return self, selection_name
    #         # calculate beta excluding the bands that the emission line contaminates
    #         phot_rest.crop_phot(
    #             [
    #                 self.phot.instrument.index_from_band_name(band.band_name)
    #                 for band in included_bands
    #             ]
    #         )
    #         A, beta = phot_rest.calc_beta_phot(rest_UV_wav_lims, iters=1)
    #         # make mock SED to calculate bandpass averaged flux from
    #         mock_SED_obs = Mock_SED_obs.from_Mock_SED_rest(
    #             Mock_SED_rest.power_law_from_beta_m_UV(
    #                 beta,
    #                 funcs.power_law_beta_func(1_500.0, 10**A, beta),
    #                 mag_units=u.Jy,
    #                 wav_lims=[
    #                     self.phot.instrument[
    #                         closest_band_index
    #                     ].WavelengthLower50,
    #                     self.phot.instrument[
    #                         closest_band_index
    #                     ].WavelengthUpper50,
    #                 ],
    #             ),
    #             self.z,
    #             IGM=None,
    #         )
    #         mag_cont = funcs.convert_mag_units(
    #             central_wav,
    #             mock_SED_obs.calc_bandpass_averaged_flux(
    #                 self.phot.instrument[closest_band_index].wav,
    #                 self.phot.instrument[closest_band_index].trans,
    #             )
    #             * u.erg
    #             / (u.s * (u.cm**2) * u.AA),
    #             u.ABmag,
    #         )
    #         # determine observed magnitude
    #         mag_obs = funcs.convert_mag_units(
    #             central_wav, self.phot[closest_band_index], u.ABmag
    #         )
    #         if (mag_cont - mag_obs).value > delta_m:
    #             if update:
    #                 self.selection_flags[selection_name] = True
    #         else:
    #             if update:
    #                 self.selection_flags[selection_name] = False
    #     return self, selection_name

    # def select_rest_UV_line_emitters_sigma(
    #     self,
    #     emission_line_name,
    #     sigma,
    #     rest_UV_wav_lims=[1_250.0, 3_000.0] * u.AA,
    #     medium_bands_only=True,
    #     SED_fit_params=EAZY({"templates": "fsps_larson", "lowz_zmax": None}),
    #     update=True,
    # ) -> tuple[Self, str]:
    #     assert (
    #         line_diagnostics[emission_line_name]["line_wav"]
    #         > rest_UV_wav_lims[0]
    #         and line_diagnostics[emission_line_name]["line_wav"]
    #         < rest_UV_wav_lims[1]
    #     )
    #     assert type(sigma) in [int, np.int64, float, np.float64]
    #     assert u.get_physical_type(rest_UV_wav_lims) == "length"
    #     assert type(medium_bands_only) in [bool, np.bool_]
    #     selection_name = f"{emission_line_name},sigma{'_med' if medium_bands_only else ''}>{sigma:.1f},UV_{str(list(np.array(rest_UV_wav_lims.value).astype(int))).replace(' ', '')}AA"
    #     if selection_name in self.selection_flags.keys():
    #         galfind_logger.debug(
    #             f"{selection_name} already performed for galaxy ID = {self.ID}!"
    #         )
    #     else:
    #         if (
    #             len(self.phot) == 0
    #         ):  # no data at all (not sure why sextractor does this)
    #             if update:
    #                 self.selection_flags[selection_name] = False
    #             return self, selection_name
    #         SED_results_key = SED_fit_params["code"].label_from_SED_fit_params(
    #             SED_fit_params
    #         )
    #         phot_rest = deepcopy(
    #             self.phot.SED_results[
    #                 SED_fit_params["code"].label_from_SED_fit_params(
    #                     SED_fit_params
    #                 )
    #             ].phot_rest
    #         )
    #         # find bands that the emission line lies within
    #         obs_frame_emission_line_wav = line_diagnostics[emission_line_name][
    #             "line_wav"
    #         ] * (1.0 + phot_rest.z)
    #         included_bands = self.phot.instrument.bands_from_wavelength(
    #             obs_frame_emission_line_wav
    #         )
    #         # determine index of the closest band to the emission line
    #         closest_band = self.phot.instrument.nearest_band_to_wavelength(
    #             obs_frame_emission_line_wav, medium_bands_only
    #         )
    #         if type(closest_band) == type(None):
    #             if update:
    #                 self.selection_flags[selection_name] = False
    #             return self, selection_name
    #         closest_band_index = self.phot.instrument.index_from_band_name(
    #             closest_band.band_name
    #         )
    #         central_wav = self.phot.instrument[
    #             closest_band_index
    #         ].WavelengthCen
    #         # if there are no included bands or the closest band is masked
    #         if (
    #             len(included_bands) == 0
    #             or self.phot.flux.mask[closest_band_index]
    #         ):
    #             if update:
    #                 self.selection_flags[selection_name] = False
    #             return self, selection_name
    #         # calculate beta excluding the bands that the emission line contaminates
    #         phot_rest.crop_phot(
    #             [
    #                 self.phot.instrument.index_from_band_name(band.band_name)
    #                 for band in included_bands
    #             ]
    #         )
    #         A, beta = phot_rest.calc_beta_phot(
    #             rest_UV_wav_lims, iters=1, incl_errs=True
    #         )
    #         m_UV = funcs.convert_mag_units(
    #             1_500.0
    #             * (1.0 + self.phot.SED_results[SED_results_key].z)
    #             * u.AA,
    #             funcs.power_law_beta_func(1_500.0, 10**A, beta)
    #             * u.erg
    #             / (u.s * (u.cm**2) * u.AA),
    #             u.ABmag,
    #         )
    #         # make mock SED to calculate bandpass averaged flux from
    #         mock_SED_rest = Mock_SED_rest.power_law_from_beta_m_UV(
    #             beta, m_UV
    #         )  # , wav_range = \
    #         # [funcs.convert_wav_units(self.phot.instrument[closest_band_index].WavelengthLower50, u.AA).value / (1. + self.phot.SED_results[SED_results_key].z), \
    #         # funcs.convert_wav_units(self.phot.instrument[closest_band_index].WavelengthUpper50, u.AA).value / (1. + self.phot.SED_results[SED_results_key].z)] * u.AA)
    #         mock_SED_obs = Mock_SED_obs.from_Mock_SED_rest(
    #             mock_SED_rest,
    #             self.phot.SED_results[SED_results_key].z,
    #             IGM=None,
    #         )
    #         flux_cont = funcs.convert_mag_units(
    #             central_wav,
    #             mock_SED_obs.calc_bandpass_averaged_flux(
    #                 self.phot.instrument[closest_band_index].wav,
    #                 self.phot.instrument[closest_band_index].trans,
    #             )
    #             * u.erg
    #             / (u.s * (u.cm**2) * u.AA),
    #             u.Jy,
    #         )
    #         # determine observed magnitude
    #         flux_obs_err = funcs.convert_mag_err_units(
    #             central_wav,
    #             self.phot.flux[closest_band_index],
    #             [
    #                 self.phot.flux_errs[closest_band_index].value,
    #                 self.phot.flux_errs[closest_band_index].value,
    #             ]
    #             * self.phot.flux_errs.unit,
    #             u.Jy,
    #         )
    #         flux_obs = funcs.convert_mag_units(
    #             central_wav, self.phot.flux[closest_band_index], u.Jy
    #         )
    #         snr_band = abs((flux_obs - flux_cont).value) / np.mean(
    #             flux_obs_err.value
    #         )
    #         mag_cont = funcs.convert_mag_units(
    #             central_wav,
    #             mock_SED_obs.calc_bandpass_averaged_flux(
    #                 self.phot.instrument[closest_band_index].wav,
    #                 self.phot.instrument[closest_band_index].trans,
    #             )
    #             * u.erg
    #             / (u.s * (u.cm**2) * u.AA),
    #             u.ABmag,
    #         )
    #         print(
    #             self.ID,
    #             snr_band,
    #             beta,
    #             mag_cont,
    #             self.phot.SED_results[SED_results_key].z,
    #             closest_band.band_name,
    #         )
    #         if snr_band > sigma:
    #             if update:
    #                 self.selection_flags[selection_name] = True
    #         else:
    #             if update:
    #                 self.selection_flags[selection_name] = False
    #     return self, selection_name

# Depth region selection

# def select_depth_region(self, band, region_ID, update=True):
#     return NotImplementedError